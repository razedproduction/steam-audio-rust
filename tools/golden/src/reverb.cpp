// Golden data for stage 6.2: reverb_effect.{h,cpp}, float8_reverb_effect.cpp, reverb_estimator.{h,cpp},
// hybrid_reverb_estimator.{h,cpp}, hybrid_reverb_effect.{h,cpp} and delay_effect.{h,cpp}.
//
// Inputs come from Lcg draws in a fixed order, one value per statement; crates/steam-audio/tests/reverb.rs replays the
// same order. ReverbEffect picks its delay lengths with the C library's rand() (MSVC: a per-thread LCG), so srand() is
// called with a known seed before every object that contains a ReverbEffect is constructed; the Rust side passes a
// CRand with the same seed. ReverbEffect and the IIR filterers pick their block algorithm from Context::sSIMDLevel when
// they are constructed: everything is built with the AVX level, the path the port follows (risk R8; needs an AVX CPU).
// ReverbEffect's AVX path overruns its buffers unless the frame size is a multiple of 8, so every frame size here is.
//
// Effects record, per produced frame, the output (mono) and a row [state, numTailSamplesRemaining] with
// state 0 = TailRemaining, 1 = TailComplete.
//
//   reverb_<rate>:            ReverbEffect scripts (see kReverbCases);
//   estimate:                 ReverbEstimator::estimate for every field and air model, [fields * models, bands];
//   reverb_scale:             every field after applyReverbScale, concatenated;
//   hybrid_estimator_<rate>:  per HybridReverbEstimator::estimate call, the whole IR afterwards (concatenated), the EQ
//                             coefficients (_eq, [calls, bands]) and the delay (_delay, [calls]);
//   hybrid_effect:            a HybridReverbEffect script.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <air_absorption.h>
#include <context.h>
#include <energy_field.h>
#include <hybrid_reverb_effect.h>
#include <hybrid_reverb_estimator.h>
#include <impulse_response.h>
#include <reverb_effect.h>
#include <reverb_estimator.h>

#include "generators.h"
#include "npy.h"

namespace {

using namespace golden;
using namespace ipl;

void fill(Lcg& lcg, AudioBuffer& buffer)
{
    for (int i = 0; i < buffer.numChannels(); ++i)
    {
        for (int j = 0; j < buffer.numSamples(); ++j)
            buffer[i][j] = lcg.nextSigned();
    }
}

struct Recorder
{
    std::vector<float> outputs;
    std::vector<int32_t> states;
    size_t count = 0;

    void record(const AudioBuffer& out, AudioEffectState state, int tailSamplesRemaining)
    {
        for (int i = 0; i < out.numChannels(); ++i)
            outputs.insert(outputs.end(), out[i], out[i] + out.numSamples());
        states.push_back(state == AudioEffectState::TailRemaining ? 0 : 1);
        states.push_back(tailSamplesRemaining);
        ++count;
    }

    void write(const std::filesystem::path& out, const std::string& name, int frameSize) const
    {
        writeNpy(out / (name + ".npy"), {count, static_cast<size_t>(frameSize)}, outputs.data());
        writeNpy(out / (name + "_states.npy"), {count, 2}, states.data());
    }
};

// Builds an object with the AVX dispatch, after seeding rand(). These classes hold 32-byte-aligned IIR coefficients;
// allocated with std::make_unique (C++17 aligned new) their construction corrupts the heap in this build, so they are
// allocated like upstream does, through ipl::make_unique (64-byte-aligned gMemory).
template <typename T, typename... Args>
ipl::unique_ptr<T> makeAVX(unsigned int seed, Args&&... args)
{
    srand(seed);
    Context::sSIMDLevel = SIMDLevel::AVX;
    auto object = ipl::make_unique<T>(std::forward<Args>(args)...);
    Context::sSIMDLevel = SIMDLevel::SSE2;
    return object;
}

// --------------------------------------------------------------------------------------------------------------------
// ReverbEffect
// --------------------------------------------------------------------------------------------------------------------

// A: draw reverb times, then an input frame, and apply; K: draw an input frame and apply with the same times;
// Z: apply a silent frame with the same times; P: draw an input frame and tailApply; Q: draw short reverb times (tail of
// two frames), then an input frame, and apply; T: tail; R: reset.
struct ReverbCase
{
    int samplingRate;
    int frameSize;
    unsigned int randSeed;
    uint32_t lcgSeed;
    const char* script;
};

const ReverbCase kReverbCases[] = {
    {48000, 256, 1, 701, "AKKKKKKKAKKKKKKKZZZZZZZZPPZZZZZZTTTTTTTTQTTTRAKZZZZZZZZZZZZZZZZZZZZZZZZ"},
    {24000, 128, 12345, 702, "AKKKKKKKKKKKKKKKAKKKKKKKKKKKKKKKZZZZZZZZZZZZZZZZPPPPTTTTTTTTQTTT"},
    {22050, 64, 777, 703,
     "AKKKKKKKKKKKKKKKKKKKAKKKKKKKKKKKKKKKKKKKZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZPPTTTTQTTTRAZZZ"},
};
// Below 16 kHz the shelf filters' 8 kHz cutoff is above Nyquist and the network can diverge (11025 Hz reached 1e36
// within 70 frames), so every rate here is higher.

void generateReverbEffect(const std::filesystem::path& out)
{
    for (const auto& c : kReverbCases)
    {
        Lcg lcg(c.lcgSeed);

        AudioSettings audioSettings{};
        audioSettings.samplingRate = c.samplingRate;
        audioSettings.frameSize = c.frameSize;

        auto effect = makeAVX<ReverbEffect>(c.randSeed, audioSettings);

        Reverb reverb{};
        AudioBuffer in(1, c.frameSize);
        AudioBuffer output(1, c.frameSize);
        Recorder recorder;

        for (const char* op = c.script; *op; ++op)
        {
            AudioEffectState state{};
            switch (*op)
            {
            case 'A':
                for (auto i = 0; i < Bands::kNumBands; ++i)
                {
                    const float u = lcg.nextSigned();
                    reverb.reverbTimes[i] = 0.8f * (u + 1.0f);
                }
                fill(lcg, in);
                break;
            case 'Q':
                for (auto i = 0; i < Bands::kNumBands; ++i)
                {
                    const float u = lcg.nextSigned();
                    reverb.reverbTimes[i] = 0.002f * (u + 1.0f);
                }
                fill(lcg, in);
                break;
            case 'K':
            case 'P':
                fill(lcg, in);
                break;
            case 'Z':
                in.makeSilent();
                break;
            case 'R':
                effect->reset();
                continue;
            }

            if (*op == 'T')
            {
                state = effect->tail(output);
            }
            else if (*op == 'P')
            {
                state = effect->tailApply(in, output);
            }
            else
            {
                ReverbEffectParams params{};
                params.reverb = &reverb;
                state = effect->apply(params, in, output);
            }
            recorder.record(output, state, effect->numTailSamplesRemaining());
        }

        recorder.write(out, "reverb_" + std::to_string(c.samplingRate), c.frameSize);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// ReverbEstimator
// --------------------------------------------------------------------------------------------------------------------

struct FieldSpec
{
    float duration;
    int order;
    int firstBin;
    bool silent;
};

// Energy decays geometrically from the first bin, at a rate drawn per band, with noise; channel 0 is positive, the other
// channels signed; about 10 % of the bins are silent.
void fillField(Lcg& lcg, EnergyField& field, int firstBin, bool silent)
{
    float decay[Bands::kNumBands];
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        const float u = lcg.nextSigned();
        decay[i] = 0.9f + 0.09f * u;
    }

    for (auto i = 0; i < field.numChannels(); ++i)
    {
        for (auto j = 0; j < Bands::kNumBands; ++j)
        {
            float envelope = 0.01f;
            for (auto k = 0; k < field.numBins(); ++k)
            {
                const float u = lcg.nextSigned();
                const float z = lcg.nextSigned();
                float e = 0.0f;
                if (!silent && k >= firstBin && z >= -0.8f)
                    e = (i == 0) ? envelope * (1.0f + 0.5f * u) : envelope * 0.3f * u;
                if (k >= firstBin)
                    envelope *= decay[j];
                field[i][j][k] = e;
            }
        }
    }
}

const FieldSpec kFields[] = {
    {1.0f, 1, 3, false}, {0.5f, 0, 0, false}, {2.0f, 2, 10, false},
    {0.03f, 1, 0, false}, {1.0f, 0, 0, true}, {0.25f, 1, 24, false},
};

void generateReverbEstimator(const std::filesystem::path& out)
{
    Lcg lcg(705);

    const float customCoefficients[Bands::kNumBands] = {0.05f, 0.2f, 0.9f};
    const float zeroCoefficients[Bands::kNumBands] = {0.0f, 0.0f, 0.0f};
    const AirAbsorptionModel airModels[] = {AirAbsorptionModel{},
                                            AirAbsorptionModel(customCoefficients, nullptr, nullptr),
                                            AirAbsorptionModel(zeroCoefficients, nullptr, nullptr)};

    std::vector<std::unique_ptr<EnergyField>> fields;
    for (const auto& spec : kFields)
    {
        fields.push_back(std::make_unique<EnergyField>(spec.duration, spec.order));
        fillField(lcg, *fields.back(), spec.firstBin, spec.silent);
    }

    std::vector<float> estimates;
    for (const auto& field : fields)
    {
        for (const auto& model : airModels)
        {
            Reverb reverb{};
            ReverbEstimator::estimate(*field, model, reverb);
            estimates.insert(estimates.end(), reverb.reverbTimes, reverb.reverbTimes + Bands::kNumBands);
        }
    }
    writeNpy(out / "estimate.npy", {std::size(kFields) * std::size(airModels), static_cast<size_t>(Bands::kNumBands)},
             estimates.data());

    std::vector<float> scaled;
    for (const auto& field : fields)
    {
        float scale[Bands::kNumBands];
        for (auto i = 0; i < Bands::kNumBands; ++i)
        {
            const float u = lcg.nextSigned();
            scale[i] = 0.5f + 0.75f * (u + 1.0f);
        }
        ReverbEstimator::applyReverbScale(scale, *field);
        const auto size = field->numChannels() * Bands::kNumBands * field->numBins();
        scaled.insert(scaled.end(), field->flatData(), field->flatData() + size);
    }
    writeNpy(out / "reverb_scale.npy", scaled);
}

// --------------------------------------------------------------------------------------------------------------------
// HybridReverbEstimator
// --------------------------------------------------------------------------------------------------------------------

struct HybridCall
{
    bool useField;
    float fieldDuration;
    float irDuration;
    int irOrder;
    int order;
    float transitionTime;
    float overlapFraction;
};

struct HybridEstimatorCase
{
    int samplingRate;
    int frameSize;
    float maxDuration;
    unsigned int randSeed;
    uint32_t lcgSeed;
    std::vector<HybridCall> calls;
};

const HybridEstimatorCase kHybridEstimatorCases[] = {
    {24000,
     256,
     0.4f,
     4242,
     711,
     {
         // Ramp from the start; the transition is long enough for the parametric reverb to start.
         {true, 0.3f, 0.2f, 1, 1, 0.18f, 1.0f},
         // EQ from the IR itself.
         {false, 0.0f, 0.25f, 0, 0, 0.24f, 0.9f},
         // A transition time past the IR's end (clamped), and an order below the IR's.
         {true, 0.3f, 0.1f, 1, 0, 0.5f, 0.25f},
         {false, 0.0f, 0.3f, 0, 0, 0.29f, 1.0f},
     }},
    {48000,
     512,
     0.25f,
     99,
     712,
     {
         {false, 0.0f, 0.2f, 0, 0, 0.19f, 0.8f},
         {true, 0.2f, 0.08f, 1, 1, 0.07f, 0.5f},
     }},
};

void generateHybridEstimator(const std::filesystem::path& out)
{
    for (const auto& c : kHybridEstimatorCases)
    {
        Lcg lcg(c.lcgSeed);

        auto estimator = makeAVX<HybridReverbEstimator>(c.randSeed, c.maxDuration, c.samplingRate, c.frameSize);

        std::vector<float> irs;
        std::vector<float> eqs;
        std::vector<int32_t> delays;
        for (const auto& call : c.calls)
        {
            Reverb reverb{};
            for (auto i = 0; i < Bands::kNumBands; ++i)
            {
                const float u = lcg.nextSigned();
                reverb.reverbTimes[i] = 0.3f + 0.6f * (u + 1.0f);
            }

            std::unique_ptr<EnergyField> field;
            if (call.useField)
            {
                field = std::make_unique<EnergyField>(call.fieldDuration, call.irOrder);
                fillField(lcg, *field, 0, false);
            }

            ImpulseResponse ir(call.irDuration, call.irOrder, c.samplingRate);
            for (auto i = 0; i < ir.numChannels(); ++i)
            {
                float envelope = 1.0f;
                for (auto j = 0; j < ir.numSamples(); ++j)
                {
                    const float u = lcg.nextSigned();
                    ir[i][j] = envelope * u;
                    envelope *= 0.9995f;
                }
            }

            float eq[Bands::kNumBands] = {};
            int delay = 0;
            estimator->estimate(field.get(), reverb, ir, call.transitionTime, call.overlapFraction, call.order, eq,
                                delay);

            irs.insert(irs.end(), ir.data(), ir.data() + ir.numChannels() * ir.numSamples());
            eqs.insert(eqs.end(), eq, eq + Bands::kNumBands);
            delays.push_back(delay);
        }

        const auto name = "hybrid_estimator_" + std::to_string(c.samplingRate);
        writeNpy(out / (name + ".npy"), irs);
        writeNpy(out / (name + "_eq.npy"), {c.calls.size(), static_cast<size_t>(Bands::kNumBands)}, eqs.data());
        writeNpy(out / (name + "_delay.npy"), delays);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// HybridReverbEffect
// --------------------------------------------------------------------------------------------------------------------

// A random spectrum of a real signal (real DC and Nyquist bins; see convolution.cpp).
void fill(Lcg& lcg, OverlapSaveFIR& fir)
{
    for (int i = 0; i < fir.numChannels(); ++i)
    {
        for (int j = 0; j < fir.numBlocks(); ++j)
        {
            auto* spectrum = fir[i][j];
            const auto n = fir.numSpectrumSamples();
            for (int k = 0; k < n; ++k)
            {
                const float re = lcg.nextSigned();
                const float im = lcg.nextSigned();
                spectrum[k] = complex_t(re, im);
            }
            spectrum[0].imag(0.0f);
            spectrum[n - 1].imag(0.0f);
        }
    }
}

struct HybridParams
{
    Reverb reverb;
    float eq[Bands::kNumBands];
    int delay;
};

void drawParams(Lcg& lcg, HybridParams& params, int maxDelay)
{
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        const float u = lcg.nextSigned();
        params.reverb.reverbTimes[i] = 0.05f + 0.15f * (u + 1.0f);
    }
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        const float u = lcg.nextSigned();
        params.eq[i] = 0.5f * (u + 1.0f);
    }
    const float u = lcg.nextSigned();
    params.delay = static_cast<int>((u + 1.0f) * 0.5f * (maxDelay - 1));
}

// Parameters are drawn once before the script. U: draw a new FIR, then an input frame, and apply with fftIRUpdated;
// A: draw an input frame and apply; P: draw new parameters, then an input frame, and apply; N: draw an input frame and
// apply without an FIR; D: draw an input frame and apply with a delay past the delay line (for this frame only);
// Q: draw short reverb times (keeping the EQ) and use a delay of 100, then draw an input frame and apply; T: tail;
// R: reset. The effect is mono (R12).
void generateHybridEffect(const std::filesystem::path& out)
{
    constexpr int kSamplingRate = 24000;
    constexpr int kFrameSize = 256;
    constexpr int kIRSize = 1024;
    constexpr const char* kScript = "UAAAPAAAUAAAPAAANNPAAAAAAAAAAAAAAAADAAAQTTTTTTTRUAAT";

    Lcg lcg(721);

    AudioSettings audioSettings{};
    audioSettings.samplingRate = kSamplingRate;
    audioSettings.frameSize = kFrameSize;

    auto effect = makeAVX<HybridReverbEffect>(31337, audioSettings, HybridReverbEffectSettings{1, kIRSize});

    HybridParams params{};
    drawParams(lcg, params, 2 * kIRSize);

    OverlapSaveFIR fir(1, kIRSize, kFrameSize);
    AudioBuffer in(1, kFrameSize);
    AudioBuffer output(1, kFrameSize);
    Recorder recorder;

    for (const char* op = kScript; *op; ++op)
    {
        if (*op == 'R')
        {
            effect->reset();
            continue;
        }

        AudioEffectState state{};
        if (*op == 'T')
        {
            state = effect->tail(output);
        }
        else
        {
            if (*op == 'U')
                fill(lcg, fir);
            if (*op == 'P')
                drawParams(lcg, params, 2 * kIRSize);
            if (*op == 'Q')
            {
                for (auto i = 0; i < Bands::kNumBands; ++i)
                {
                    const float u = lcg.nextSigned();
                    params.reverb.reverbTimes[i] = 0.002f * (u + 1.0f);
                }
                params.delay = 100;
            }
            fill(lcg, in);

            HybridReverbEffectDirectParams direct{};
            direct.fftIR = (*op == 'N') ? nullptr : &fir;
            direct.fftIRUpdated = (*op == 'U');
            direct.reverb = &params.reverb;
            direct.eqCoeffs = params.eq;
            direct.delay = (*op == 'D') ? 2 * kIRSize + 5 : params.delay;
            direct.numChannels = 1;
            direct.numSamples = kIRSize;
            state = effect->apply(direct, in, output);
        }
        recorder.record(output, state, effect->numTailSamplesRemaining());
    }

    recorder.write(out, "hybrid_effect", kFrameSize);
}

void generateReverb(const std::filesystem::path& out)
{
    generateReverbEffect(out);
    generateReverbEstimator(out);
    generateHybridEstimator(out);
    generateHybridEffect(out);
}

Register reverb("reverb", generateReverb);

}
