// Golden data for stage 8a: the C API (phonon.h) for contexts, audio buffers, HRTFs, the audio effects that need no
// simulator, and the direct-sound model functions (api_context, api_geometry, api_audio_buffer, api_hrtf,
// api_*_effect, api_advanced_simulation, phonon_interfaces.h).
//
// Everything is called through the ipl* functions. Inputs come from Lcg draws in a fixed order, one value per
// statement; crates/phonon-capi/tests/capi.rs replays the same order. Effect cases run a script of operations; every
// operation records its whole output buffer (filled with kSentinel first, so channels an effect leaves untouched are
// recorded too) into one flat array, and a row [state, tail size] with state 0 = TailRemaining, 1 = TailComplete.
//
// The API version is a global upstream (Context::sAPIVersion, set by the last context created), so contexts are
// created and used one at a time. Without IPP, context creation leaves Context::sSIMDLevel alone: it is set to AVX
// after each context is created, so that the effects built on IIR filters follow the path the port follows (risk R8).

#include <cstddef>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <context.h>
#include <coordinate_space.h>
#include <phonon.h>

#include "generators.h"
#include "npy.h"

namespace {

using namespace golden;

constexpr int kSamplingRate = 48000;
constexpr int kFrameSize = 97;
constexpr float kSentinel = 9.0f;

constexpr IPLuint32 kVersion = STEAMAUDIO_VERSION;
constexpr IPLuint32 kVersion40 = (4u << 16) | (0u << 8);
constexpr IPLuint32 kVersion41 = (4u << 16) | (1u << 8);

void check(IPLerror status, const char* what)
{
    if (status != IPL_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(status));
}

IPLContext makeContext(IPLuint32 version)
{
    IPLContextSettings settings{};
    settings.version = version;
    IPLContext context = nullptr;
    check(iplContextCreate(&settings, &context), "iplContextCreate");
    ipl::Context::sSIMDLevel = ipl::SIMDLevel::AVX;
    return context;
}

IPLAudioSettings audioSettings()
{
    return IPLAudioSettings{kSamplingRate, kFrameSize};
}

IPLHRTF makeDefaultHRTF(IPLContext context)
{
    auto audio = audioSettings();
    IPLHRTFSettings settings{};
    settings.type = IPL_HRTFTYPE_DEFAULT;
    settings.volume = 1.0f;
    IPLHRTF hrtf = nullptr;
    check(iplHRTFCreate(context, &audio, &settings, &hrtf), "iplHRTFCreate");
    return hrtf;
}

IPLVector3 drawVector(Lcg& lcg)
{
    const float x = lcg.nextSigned();
    const float y = lcg.nextSigned();
    const float z = lcg.nextSigned();
    return IPLVector3{x, y, z};
}

// Uniform in [0, 1).
float drawUnit(Lcg& lcg)
{
    const float u = lcg.nextSigned();
    return 0.5f * (u + 1.0f);
}

// An orientation with a random ahead vector, completed by CoordinateSpace3f's single-vector constructor.
IPLCoordinateSpace3 drawOrientation(Lcg& lcg)
{
    const auto v = drawVector(lcg);
    const auto ahead = ipl::Vector3f::unitVector(ipl::Vector3f(v.x, v.y, v.z));
    const ipl::CoordinateSpace3f space(ahead, ipl::Vector3f::kZero);
    return IPLCoordinateSpace3{
        {space.right.x(), space.right.y(), space.right.z()},
        {space.up.x(), space.up.y(), space.up.z()},
        {space.ahead.x(), space.ahead.y(), space.ahead.z()},
        {space.origin.x(), space.origin.y(), space.origin.z()},
    };
}

// An audio buffer from iplAudioBufferAllocate.
struct Buffer
{
    IPLContext context;
    IPLAudioBuffer buffer{};

    Buffer(IPLContext context, int numChannels, int numSamples)
        : context(context)
    {
        check(iplAudioBufferAllocate(context, numChannels, numSamples, &buffer), "iplAudioBufferAllocate");
    }

    ~Buffer() { iplAudioBufferFree(context, &buffer); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void fill(Lcg& lcg)
    {
        for (int i = 0; i < buffer.numChannels; ++i)
        {
            for (int j = 0; j < buffer.numSamples; ++j)
                buffer.data[i][j] = lcg.nextSigned();
        }
    }

    void fill(float value)
    {
        for (int i = 0; i < buffer.numChannels; ++i)
        {
            for (int j = 0; j < buffer.numSamples; ++j)
                buffer.data[i][j] = value;
        }
    }

    void append(std::vector<float>& out) const
    {
        for (int i = 0; i < buffer.numChannels; ++i)
            out.insert(out.end(), buffer.data[i], buffer.data[i] + buffer.numSamples);
    }
};

struct Recorder
{
    std::vector<float> outputs;
    std::vector<int32_t> states;
    std::vector<float> peaks;
    size_t count = 0;

    void record(const Buffer& out, IPLAudioEffectState state, IPLint32 tailSize)
    {
        out.append(outputs);
        states.push_back(state);
        states.push_back(tailSize);
        ++count;
    }

    void write(const std::filesystem::path& out, const std::string& name) const
    {
        writeNpy(out / (name + ".npy"), outputs);
        writeNpy(out / (name + "_states.npy"), {count, 2}, states.data());
        if (!peaks.empty())
            writeNpy(out / (name + "_peaks.npy"), {peaks.size() / 2, 2}, peaks.data());
    }
};

// Records tail frames until the effect reports that its tail is complete (at most kMaxTails).
constexpr int kMaxTails = 4;

template <typename Effect, typename GetTail, typename GetTailSize>
void recordTails(Recorder& recorder, Effect effect, Buffer& out, GetTail getTail, GetTailSize getTailSize)
{
    for (int i = 0; i < kMaxTails; ++i)
    {
        out.fill(kSentinel);
        const auto state = getTail(effect, &out.buffer);
        recorder.record(out, state, getTailSize(effect));
        if (state == IPL_AUDIOEFFECTSTATE_TAILCOMPLETE)
            break;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Struct layouts
// --------------------------------------------------------------------------------------------------------------------

#define SIZE(T) layout.push_back(static_cast<int32_t>(sizeof(T)))
#define FIELD(T, f) layout.push_back(static_cast<int32_t>(offsetof(T, f)))

void generateLayout(const std::filesystem::path& out)
{
    std::vector<int32_t> layout;

    // Enums are int-sized.
    SIZE(IPLbool); SIZE(IPLerror); SIZE(IPLSIMDLevel); SIZE(IPLContextFlags); SIZE(IPLAudioEffectState);

    SIZE(IPLContextSettings); FIELD(IPLContextSettings, version); FIELD(IPLContextSettings, logCallback);
    FIELD(IPLContextSettings, allocateCallback); FIELD(IPLContextSettings, freeCallback);
    FIELD(IPLContextSettings, simdLevel); FIELD(IPLContextSettings, flags);

    SIZE(IPLVector3); FIELD(IPLVector3, x); FIELD(IPLVector3, y); FIELD(IPLVector3, z);
    SIZE(IPLMatrix4x4); FIELD(IPLMatrix4x4, elements);
    SIZE(IPLBox); FIELD(IPLBox, minCoordinates); FIELD(IPLBox, maxCoordinates);
    SIZE(IPLSphere); FIELD(IPLSphere, center); FIELD(IPLSphere, radius);
    SIZE(IPLCoordinateSpace3); FIELD(IPLCoordinateSpace3, right); FIELD(IPLCoordinateSpace3, up);
    FIELD(IPLCoordinateSpace3, ahead); FIELD(IPLCoordinateSpace3, origin);

    SIZE(IPLSerializedObjectSettings); FIELD(IPLSerializedObjectSettings, data); FIELD(IPLSerializedObjectSettings, size);

    SIZE(IPLSpeakerLayout); FIELD(IPLSpeakerLayout, type); FIELD(IPLSpeakerLayout, numSpeakers);
    FIELD(IPLSpeakerLayout, speakers);
    SIZE(IPLAudioSettings); FIELD(IPLAudioSettings, samplingRate); FIELD(IPLAudioSettings, frameSize);
    SIZE(IPLAudioBuffer); FIELD(IPLAudioBuffer, numChannels); FIELD(IPLAudioBuffer, numSamples);
    FIELD(IPLAudioBuffer, data);

    SIZE(IPLHRTFSettings); FIELD(IPLHRTFSettings, type); FIELD(IPLHRTFSettings, sofaFileName);
    FIELD(IPLHRTFSettings, sofaData); FIELD(IPLHRTFSettings, sofaDataSize); FIELD(IPLHRTFSettings, volume);
    FIELD(IPLHRTFSettings, normType);

    SIZE(IPLPanningEffectSettings); FIELD(IPLPanningEffectSettings, speakerLayout);
    SIZE(IPLPanningEffectParams); FIELD(IPLPanningEffectParams, direction);
    SIZE(IPLBinauralEffectSettings); FIELD(IPLBinauralEffectSettings, hrtf);
    SIZE(IPLBinauralEffectParams); FIELD(IPLBinauralEffectParams, direction);
    FIELD(IPLBinauralEffectParams, interpolation); FIELD(IPLBinauralEffectParams, spatialBlend);
    FIELD(IPLBinauralEffectParams, hrtf); FIELD(IPLBinauralEffectParams, peakDelays);
    SIZE(IPLVirtualSurroundEffectSettings); FIELD(IPLVirtualSurroundEffectSettings, speakerLayout);
    FIELD(IPLVirtualSurroundEffectSettings, hrtf);
    SIZE(IPLVirtualSurroundEffectParams); FIELD(IPLVirtualSurroundEffectParams, hrtf);
    SIZE(IPLAmbisonicsEncodeEffectSettings); FIELD(IPLAmbisonicsEncodeEffectSettings, maxOrder);
    SIZE(IPLAmbisonicsEncodeEffectParams); FIELD(IPLAmbisonicsEncodeEffectParams, direction);
    FIELD(IPLAmbisonicsEncodeEffectParams, order);
    SIZE(IPLAmbisonicsPanningEffectSettings); FIELD(IPLAmbisonicsPanningEffectSettings, speakerLayout);
    FIELD(IPLAmbisonicsPanningEffectSettings, maxOrder);
    SIZE(IPLAmbisonicsPanningEffectParams); FIELD(IPLAmbisonicsPanningEffectParams, order);
    SIZE(IPLAmbisonicsBinauralEffectSettings); FIELD(IPLAmbisonicsBinauralEffectSettings, hrtf);
    FIELD(IPLAmbisonicsBinauralEffectSettings, maxOrder);
    SIZE(IPLAmbisonicsBinauralEffectParams); FIELD(IPLAmbisonicsBinauralEffectParams, hrtf);
    FIELD(IPLAmbisonicsBinauralEffectParams, order);
    SIZE(IPLAmbisonicsRotationEffectSettings); FIELD(IPLAmbisonicsRotationEffectSettings, maxOrder);
    SIZE(IPLAmbisonicsRotationEffectParams); FIELD(IPLAmbisonicsRotationEffectParams, orientation);
    FIELD(IPLAmbisonicsRotationEffectParams, order);
    SIZE(IPLAmbisonicsDecodeEffectSettings); FIELD(IPLAmbisonicsDecodeEffectSettings, speakerLayout);
    FIELD(IPLAmbisonicsDecodeEffectSettings, hrtf); FIELD(IPLAmbisonicsDecodeEffectSettings, maxOrder);
    SIZE(IPLAmbisonicsDecodeEffectParams); FIELD(IPLAmbisonicsDecodeEffectParams, order);
    FIELD(IPLAmbisonicsDecodeEffectParams, hrtf); FIELD(IPLAmbisonicsDecodeEffectParams, orientation);
    FIELD(IPLAmbisonicsDecodeEffectParams, binaural);
    SIZE(IPLDirectEffectSettings); FIELD(IPLDirectEffectSettings, numChannels);
    SIZE(IPLDirectEffectParams); FIELD(IPLDirectEffectParams, flags); FIELD(IPLDirectEffectParams, transmissionType);
    FIELD(IPLDirectEffectParams, distanceAttenuation); FIELD(IPLDirectEffectParams, airAbsorption);
    FIELD(IPLDirectEffectParams, directivity); FIELD(IPLDirectEffectParams, occlusion);
    FIELD(IPLDirectEffectParams, transmission);

    SIZE(IPLDistanceAttenuationModel); FIELD(IPLDistanceAttenuationModel, type);
    FIELD(IPLDistanceAttenuationModel, minDistance); FIELD(IPLDistanceAttenuationModel, callback);
    FIELD(IPLDistanceAttenuationModel, userData); FIELD(IPLDistanceAttenuationModel, dirty);
    SIZE(IPLAirAbsorptionModel); FIELD(IPLAirAbsorptionModel, type); FIELD(IPLAirAbsorptionModel, coefficients);
    FIELD(IPLAirAbsorptionModel, callback); FIELD(IPLAirAbsorptionModel, userData); FIELD(IPLAirAbsorptionModel, dirty);
    SIZE(IPLDirectivity); FIELD(IPLDirectivity, dipoleWeight); FIELD(IPLDirectivity, dipolePower);
    FIELD(IPLDirectivity, callback); FIELD(IPLDirectivity, userData);

    writeNpy(out / "layout.npy", layout);
}

#undef SIZE
#undef FIELD

// --------------------------------------------------------------------------------------------------------------------
// Context and relative direction
// --------------------------------------------------------------------------------------------------------------------

// Caller versions tried by context creation.
const IPLuint32 kCallerVersions[] = {
    kVersion, kVersion40, (4u << 16) | (8u << 8) | 255u, (4u << 16) | (9u << 8), 5u << 16, (3u << 16) | (8u << 8) | 1u,
    (4u << 16) | (7u << 8), 0u,
};

constexpr int kNumRandomDirections = 60;

void generateContext(const std::filesystem::path& out)
{
    std::vector<int32_t> statuses;
    for (auto version : kCallerVersions)
    {
        IPLContextSettings settings{};
        settings.version = version;
        IPLContext context = nullptr;
        statuses.push_back(iplContextCreate(&settings, &context));
        statuses.push_back(context != nullptr ? 1 : 0);
        iplContextRelease(&context);
    }
    {
        IPLContext context = nullptr;
        statuses.push_back(iplContextCreate(nullptr, &context));
        IPLContextSettings settings{};
        settings.version = kVersion;
        statuses.push_back(iplContextCreate(&settings, nullptr));
    }
    writeNpy(out / "context_status.npy", statuses);

    auto context = makeContext(kVersion);

    // Retaining returns the same handle; the context lives until the last release.
    auto retained = iplContextRetain(context);
    if (retained != context)
        throw std::runtime_error("iplContextRetain returned another handle");
    iplContextRelease(&retained);

    Lcg lcg(801);
    std::vector<float> directions;
    auto add = [&](IPLVector3 source, IPLVector3 listener, IPLVector3 ahead, IPLVector3 up) {
        const auto d = iplCalculateRelativeDirection(context, source, listener, ahead, up);
        directions.insert(directions.end(), {d.x, d.y, d.z});
    };

    const IPLVector3 origin{0.0f, 0.0f, 0.0f};
    const IPLVector3 ahead{0.0f, 0.0f, -1.0f};
    const IPLVector3 up{0.0f, 1.0f, 0.0f};
    add(IPLVector3{1.0f, 2.0f, 3.0f}, IPLVector3{1.0f, 2.0f, 3.0f}, ahead, up);
    add(IPLVector3{1.0f, 0.0f, 0.0f}, origin, ahead, up);
    add(IPLVector3{0.0f, 0.0f, -5.0f}, origin, ahead, up);
    add(IPLVector3{0.0f, 0.0f, 4e-6f}, origin, ahead, up);
    for (int i = 0; i < kNumRandomDirections; ++i)
    {
        const auto source = drawVector(lcg);
        const auto listener = drawVector(lcg);
        const auto listenerAhead = drawVector(lcg);
        const auto listenerUp = drawVector(lcg);
        add(source, listener, listenerAhead, listenerUp);
    }
    writeNpy(out / "relative_direction.npy", {directions.size() / 3, 3}, directions.data());

    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Audio buffers
// --------------------------------------------------------------------------------------------------------------------

constexpr int kBufferSamples = 37;

// Channel counts of the Ambisonics conversions: orders 1 and 3 for every pair of formats, order 4 between N3D and SN3D.
constexpr int kConversionChannels[] = {4, 16};

void generateAudioBuffers(const std::filesystem::path& out)
{
    auto context = makeContext(kVersion);
    Lcg lcg(802);

    {
        Buffer src(context, 3, kBufferSamples);
        src.fill(lcg);
        std::vector<float> interleaved(3 * kBufferSamples);
        iplAudioBufferInterleave(context, &src.buffer, interleaved.data());
        writeNpy(out / "interleave.npy", interleaved);
    }
    {
        std::vector<float> interleaved(3 * kBufferSamples);
        for (auto& x : interleaved)
            x = lcg.nextSigned();
        Buffer dst(context, 3, kBufferSamples);
        iplAudioBufferDeinterleave(context, interleaved.data(), &dst.buffer);
        std::vector<float> result;
        dst.append(result);
        writeNpy(out / "deinterleave.npy", result);
    }
    {
        Buffer in(context, 2, kBufferSamples);
        Buffer mix(context, 2, kBufferSamples);
        in.fill(lcg);
        mix.fill(lcg);
        iplAudioBufferMix(context, &in.buffer, &mix.buffer);
        iplAudioBufferMix(context, &mix.buffer, &mix.buffer);
        std::vector<float> result;
        mix.append(result);
        writeNpy(out / "mix.npy", result);
    }
    {
        Buffer in(context, 5, kBufferSamples);
        Buffer mono(context, 1, kBufferSamples);
        in.fill(lcg);
        iplAudioBufferDownmix(context, &in.buffer, &mono.buffer);
        std::vector<float> result;
        mono.append(result);
        writeNpy(out / "downmix.npy", result);
    }

    std::vector<float> converted;
    std::vector<float> convertedInPlace;
    auto convert = [&](IPLAmbisonicsType inType, IPLAmbisonicsType outType, int numChannels) {
        Buffer in(context, numChannels, kBufferSamples);
        Buffer outBuffer(context, numChannels, kBufferSamples);
        in.fill(lcg);
        outBuffer.fill(kSentinel);
        iplAudioBufferConvertAmbisonics(context, inType, outType, &in.buffer, &outBuffer.buffer);
        outBuffer.append(converted);
        iplAudioBufferConvertAmbisonics(context, inType, outType, &in.buffer, &in.buffer);
        in.append(convertedInPlace);
    };
    for (auto numChannels : kConversionChannels)
    {
        for (int inType = 0; inType < 3; ++inType)
        {
            for (int outType = 0; outType < 3; ++outType)
                convert(static_cast<IPLAmbisonicsType>(inType), static_cast<IPLAmbisonicsType>(outType), numChannels);
        }
    }
    convert(IPL_AMBISONICSTYPE_N3D, IPL_AMBISONICSTYPE_SN3D, 25);
    convert(IPL_AMBISONICSTYPE_SN3D, IPL_AMBISONICSTYPE_N3D, 25);
    writeNpy(out / "convert.npy", converted);
    writeNpy(out / "convert_in_place.npy", convertedInPlace);

    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Direct sound models
// --------------------------------------------------------------------------------------------------------------------

constexpr int kNumModelPoints = 50;

float IPLCALL distanceCallback(IPLfloat32 distance, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return 1.0f / (1.0f + k * distance * distance);
}

float IPLCALL airAbsorptionCallback(IPLfloat32 distance, IPLint32 band, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return 1.0f / (1.0f + k * static_cast<float>(band + 1) * distance);
}

float IPLCALL directivityCallback(IPLVector3 direction, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return k + direction.x * direction.z - 0.25f * direction.y;
}

float gK1 = 0.3f;
float gK2 = 0.05f;
float gK3 = 0.5f;

void generateModels(const std::filesystem::path& out)
{
    auto context = makeContext(kVersion);
    Lcg lcg(803);

    std::vector<IPLVector3> sources;
    std::vector<IPLVector3> listeners;
    std::vector<IPLCoordinateSpace3> spaces;
    for (int i = 0; i < kNumModelPoints; ++i)
    {
        const auto source = drawVector(lcg);
        const auto listener = drawVector(lcg);
        const auto scale = 1.0f + 20.0f * drawUnit(lcg);
        sources.push_back(IPLVector3{source.x * scale, source.y * scale, source.z * scale});
        listeners.push_back(listener);
        auto space = drawOrientation(lcg);
        space.origin = sources.back();
        spaces.push_back(space);
    }

    // Distance attenuation: default, inverse distance from 0.5 and 0, a callback, the callback type without a callback,
    // an unknown type, and inverse distance with a callback set (ignored).
    std::vector<IPLDistanceAttenuationModel> distanceModels(7);
    distanceModels[0].type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
    distanceModels[1].type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    distanceModels[1].minDistance = 0.5f;
    distanceModels[2].type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    distanceModels[2].minDistance = 0.0f;
    distanceModels[3].type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
    distanceModels[3].callback = distanceCallback;
    distanceModels[3].userData = &gK1;
    distanceModels[4].type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
    distanceModels[5].type = static_cast<IPLDistanceAttenuationModelType>(9);
    distanceModels[5].minDistance = 3.0f;
    distanceModels[6].type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    distanceModels[6].minDistance = 2.0f;
    distanceModels[6].callback = distanceCallback;
    distanceModels[6].userData = &gK2;

    std::vector<float> distance;
    for (auto& model : distanceModels)
    {
        for (int i = 0; i < kNumModelPoints; ++i)
            distance.push_back(iplDistanceAttenuationCalculate(context, sources[i], listeners[i], &model));
    }
    writeNpy(out / "distance_attenuation.npy", {distanceModels.size(), static_cast<size_t>(kNumModelPoints)},
             distance.data());

    // Air absorption: default, exponential, exponential with a callback set (ignored), a callback, an unknown type.
    std::vector<IPLAirAbsorptionModel> airModels(5);
    airModels[0].type = IPL_AIRABSORPTIONTYPE_DEFAULT;
    airModels[1].type = IPL_AIRABSORPTIONTYPE_EXPONENTIAL;
    airModels[1].coefficients[0] = 0.1f;
    airModels[1].coefficients[1] = 0.2f;
    airModels[1].coefficients[2] = 0.3f;
    airModels[2] = airModels[1];
    airModels[2].callback = airAbsorptionCallback;
    airModels[2].userData = &gK2;
    airModels[3].type = IPL_AIRABSORPTIONTYPE_CALLBACK;
    airModels[3].callback = airAbsorptionCallback;
    airModels[3].userData = &gK2;
    airModels[4].type = static_cast<IPLAirAbsorptionModelType>(9);
    airModels[4].coefficients[0] = 0.5f;

    std::vector<float> air;
    for (auto& model : airModels)
    {
        for (int i = 0; i < kNumModelPoints; ++i)
        {
            float values[IPL_NUM_BANDS];
            iplAirAbsorptionCalculate(context, sources[i], listeners[i], &model, values);
            air.insert(air.end(), std::begin(values), std::end(values));
        }
    }
    writeNpy(out / "air_absorption.npy", {airModels.size(), static_cast<size_t>(kNumModelPoints), 3}, air.data());

    // Directivity: omnidirectional, a cardioid, a sharper dipole, a callback, and a callback with weight 0.
    std::vector<IPLDirectivity> directivityModels(5);
    directivityModels[1].dipoleWeight = 0.5f;
    directivityModels[1].dipolePower = 1.0f;
    directivityModels[2].dipoleWeight = 0.3f;
    directivityModels[2].dipolePower = 2.5f;
    directivityModels[3].dipoleWeight = 0.7f;
    directivityModels[3].callback = directivityCallback;
    directivityModels[3].userData = &gK3;
    directivityModels[4].callback = directivityCallback;
    directivityModels[4].userData = &gK1;

    std::vector<float> directivity;
    for (auto& model : directivityModels)
    {
        for (int i = 0; i < kNumModelPoints; ++i)
            directivity.push_back(iplDirectivityCalculate(context, spaces[i], listeners[i], &model));
    }
    writeNpy(out / "directivity.npy", {directivityModels.size(), static_cast<size_t>(kNumModelPoints)},
             directivity.data());

    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// HRTFs
// --------------------------------------------------------------------------------------------------------------------

std::vector<uint8_t> readFile(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("cannot open ") + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// Statuses of HRTF creation with invalid or unusual settings, in the order below.
void generateHRTFStatus(const std::filesystem::path& out)
{
    std::vector<int32_t> statuses;
    std::vector<uint8_t> garbage(64);
    Lcg lcg(804);
    for (auto& byte : garbage)
        byte = static_cast<uint8_t>(lcg.nextU32() >> 24);

    auto create = [&](IPLContext context, IPLAudioSettings audio, IPLHRTFSettings settings) {
        IPLHRTF hrtf = nullptr;
        statuses.push_back(iplHRTFCreate(context, &audio, &settings, &hrtf));
        statuses.push_back(hrtf != nullptr ? 1 : 0);
        iplHRTFRelease(&hrtf);
    };

    {
        auto context = makeContext(kVersion);
        const auto audio = audioSettings();

        IPLHRTFSettings sofaNothing{};
        sofaNothing.type = IPL_HRTFTYPE_SOFA;
        create(context, audio, sofaNothing);

        IPLHRTFSettings sofaEmptyData{};
        sofaEmptyData.type = IPL_HRTFTYPE_SOFA;
        sofaEmptyData.sofaData = garbage.data();
        create(context, audio, sofaEmptyData);

        IPLHRTFSettings unknownType{};
        unknownType.type = static_cast<IPLHRTFType>(7);
        create(context, audio, unknownType);

        IPLHRTFSettings missingFile{};
        missingFile.type = IPL_HRTFTYPE_SOFA;
        missingFile.sofaFileName = "core/data/hrtf/missing.sofa";
        create(context, audio, missingFile);

        IPLHRTFSettings garbageData{};
        garbageData.type = IPL_HRTFTYPE_SOFA;
        garbageData.sofaData = garbage.data();
        garbageData.sofaDataSize = static_cast<int>(garbage.size());
        create(context, audio, garbageData);

        IPLHRTFSettings defaultSettings{};
        defaultSettings.volume = 1.0f;
        create(context, IPLAudioSettings{0, kFrameSize}, defaultSettings);
        create(context, IPLAudioSettings{kSamplingRate, 0}, defaultSettings);
        create(context, IPLAudioSettings{8000, kFrameSize}, defaultSettings);
        create(context, audio, defaultSettings);

        IPLHRTF hrtf = nullptr;
        auto audioCopy = audio;
        statuses.push_back(iplHRTFCreate(context, &audioCopy, nullptr, &hrtf));
        statuses.push_back(iplHRTFCreate(context, nullptr, &defaultSettings, &hrtf));
        statuses.push_back(iplHRTFCreate(nullptr, &audioCopy, &defaultSettings, &hrtf));

        iplContextRelease(&context);
    }
    {
        // Before 4.2 sofaData is not passed on, although the checks read it: SOFA data alone loads nothing.
        auto context = makeContext(kVersion41);
        IPLHRTFSettings sofaData{};
        sofaData.type = IPL_HRTFTYPE_SOFA;
        sofaData.sofaData = garbage.data();
        sofaData.sofaDataSize = static_cast<int>(garbage.size());
        create(context, audioSettings(), sofaData);
        iplContextRelease(&context);
    }

    writeNpy(out / "hrtf_status.npy", statuses);
}

// Binaural rendering of a few frames with an HRTF, nearest-neighbour, recording outputs and peak delays.
void renderWithHRTF(IPLContext context, IPLHRTF hrtf, Lcg& lcg, Recorder& recorder, int numFrames)
{
    auto audio = audioSettings();
    IPLBinauralEffectSettings settings{};
    settings.hrtf = hrtf;
    IPLBinauralEffect effect = nullptr;
    check(iplBinauralEffectCreate(context, &audio, &settings, &effect), "iplBinauralEffectCreate");

    Buffer in(context, 1, kFrameSize);
    Buffer out(context, 2, kFrameSize);
    for (int i = 0; i < numFrames; ++i)
    {
        in.fill(lcg);
        float peaks[2] = {kSentinel, kSentinel};
        IPLBinauralEffectParams params{};
        params.direction = drawVector(lcg);
        params.interpolation = IPL_HRTFINTERPOLATION_NEAREST;
        params.spatialBlend = 1.0f;
        params.hrtf = hrtf;
        params.peakDelays = peaks;
        out.fill(kSentinel);
        const auto state = iplBinauralEffectApply(effect, &params, &in.buffer, &out.buffer);
        recorder.record(out, state, iplBinauralEffectGetTailSize(effect));
        recorder.peaks.insert(recorder.peaks.end(), std::begin(peaks), std::end(peaks));
    }

    iplBinauralEffectRelease(&effect);
}

constexpr int kHRTFFrames = 3;

void generateHRTFs(const std::filesystem::path& out)
{
    generateHRTFStatus(out);

    Lcg lcg(805);
    auto run = [&](IPLuint32 version, IPLHRTFSettings settings, const std::string& name) {
        auto context = makeContext(version);
        auto audio = audioSettings();
        IPLHRTF hrtf = nullptr;
        check(iplHRTFCreate(context, &audio, &settings, &hrtf), name.c_str());
        Recorder recorder;
        renderWithHRTF(context, hrtf, lcg, recorder, kHRTFFrames);
        recorder.write(out, name);
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
    };

    IPLHRTFSettings file{};
    file.type = IPL_HRTFTYPE_SOFA;
    file.sofaFileName = "core/data/hrtf/cipic_124.sofa";
    file.volume = 1.0f;
    run(kVersion, file, "hrtf_sofa_file");

    const auto data = readFile("core/data/hrtf/sadie_h12.sofa");
    IPLHRTFSettings memory{};
    memory.type = IPL_HRTFTYPE_SOFA;
    memory.sofaData = data.data();
    memory.sofaDataSize = static_cast<int>(data.size());
    memory.volume = 0.5f;
    memory.normType = IPL_HRTFNORMTYPE_RMS;
    run(kVersion, memory, "hrtf_sofa_data");

    IPLHRTFSettings loud{};
    loud.type = IPL_HRTFTYPE_DEFAULT;
    loud.volume = 2.0f;
    loud.normType = IPL_HRTFNORMTYPE_RMS;
    run(kVersion, loud, "hrtf_default_loud");

    // Version 4.1: volume and normalization are ignored. Peak delays are written from 4.1 on.
    run(kVersion41, loud, "hrtf_default_loud_v41");

    // Version 4.0: peak delays are not written either.
    run(kVersion40, loud, "hrtf_default_loud_v40");

    // Both a file name and data: the file wins.
    IPLHRTFSettings both = file;
    both.sofaData = data.data();
    both.sofaDataSize = static_cast<int>(data.size());
    run(kVersion, both, "hrtf_sofa_both");
}

// --------------------------------------------------------------------------------------------------------------------
// Panning and binaural effects, virtual surround
// --------------------------------------------------------------------------------------------------------------------

// A custom layout with an LFE channel (position (0, 0, 0)).
IPLVector3 gCustomSpeakers[] = {
    {-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.3f, -0.5f, 0.8f},
};

IPLSpeakerLayout makeLayout(IPLSpeakerLayoutType type)
{
    IPLSpeakerLayout layout{};
    layout.type = type;
    if (type == IPL_SPEAKERLAYOUTTYPE_CUSTOM)
    {
        layout.numSpeakers = static_cast<IPLint32>(std::size(gCustomSpeakers));
        layout.speakers = gCustomSpeakers;
    }
    return layout;
}

int numSpeakers(IPLSpeakerLayoutType type)
{
    switch (type)
    {
    case IPL_SPEAKERLAYOUTTYPE_MONO: return 1;
    case IPL_SPEAKERLAYOUTTYPE_STEREO: return 2;
    case IPL_SPEAKERLAYOUTTYPE_QUADRAPHONIC: return 4;
    case IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1: return 6;
    case IPL_SPEAKERLAYOUTTYPE_SURROUND_7_1: return 8;
    default: return static_cast<int>(std::size(gCustomSpeakers));
    }
}

struct LayoutCase
{
    const char* name;
    IPLSpeakerLayoutType type;
};

constexpr LayoutCase kPanningLayouts[] = {
    {"panning_stereo", IPL_SPEAKERLAYOUTTYPE_STEREO},
    {"panning_surround51", IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1},
    {"panning_custom", IPL_SPEAKERLAYOUTTYPE_CUSTOM},
};

constexpr int kFrames = 3;

void generatePanning(const std::filesystem::path& out, IPLContext context, Lcg& lcg)
{
    for (const auto& c : kPanningLayouts)
    {
        auto audio = audioSettings();
        IPLPanningEffectSettings settings{};
        settings.speakerLayout = makeLayout(c.type);
        IPLPanningEffect effect = nullptr;
        check(iplPanningEffectCreate(context, &audio, &settings, &effect), "iplPanningEffectCreate");

        Buffer in(context, 1, kFrameSize);
        Buffer outBuffer(context, numSpeakers(c.type), kFrameSize);
        Recorder recorder;
        auto apply = [&]() {
            in.fill(lcg);
            IPLPanningEffectParams params{};
            params.direction = drawVector(lcg);
            outBuffer.fill(kSentinel);
            const auto state = iplPanningEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
            recorder.record(outBuffer, state, iplPanningEffectGetTailSize(effect));
        };
        for (int i = 0; i < kFrames; ++i)
            apply();
        iplPanningEffectReset(effect);
        apply();
        recordTails(recorder, effect, outBuffer, iplPanningEffectGetTail, iplPanningEffectGetTailSize);
        recorder.write(out, c.name);

        iplPanningEffectRelease(&effect);
    }
}

struct BinauralCase
{
    const char* name;
    int numInputChannels;
    IPLHRTFInterpolation interpolation;
    bool drawBlend;
};

constexpr BinauralCase kBinauralCases[] = {
    {"binaural_mono_nearest", 1, IPL_HRTFINTERPOLATION_NEAREST, false},
    {"binaural_mono_bilinear", 1, IPL_HRTFINTERPOLATION_BILINEAR, false},
    {"binaural_stereo_nearest_blend", 2, IPL_HRTFINTERPOLATION_NEAREST, true},
    {"binaural_stereo_bilinear_blend", 2, IPL_HRTFINTERPOLATION_BILINEAR, true},
};

void generateBinaural(const std::filesystem::path& out, IPLContext context, IPLHRTF hrtf, Lcg& lcg)
{
    for (const auto& c : kBinauralCases)
    {
        auto audio = audioSettings();
        IPLBinauralEffectSettings settings{};
        settings.hrtf = hrtf;
        IPLBinauralEffect effect = nullptr;
        check(iplBinauralEffectCreate(context, &audio, &settings, &effect), "iplBinauralEffectCreate");

        Buffer in(context, c.numInputChannels, kFrameSize);
        Buffer outBuffer(context, 2, kFrameSize);
        Recorder recorder;
        auto apply = [&]() {
            in.fill(lcg);
            IPLBinauralEffectParams params{};
            params.direction = drawVector(lcg);
            params.interpolation = c.interpolation;
            params.spatialBlend = c.drawBlend ? drawUnit(lcg) : 1.0f;
            params.hrtf = hrtf;
            float peaks[2] = {kSentinel, kSentinel};
            params.peakDelays = peaks;
            outBuffer.fill(kSentinel);
            const auto state = iplBinauralEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
            recorder.record(outBuffer, state, iplBinauralEffectGetTailSize(effect));
            recorder.peaks.insert(recorder.peaks.end(), std::begin(peaks), std::end(peaks));
        };
        for (int i = 0; i < kFrames; ++i)
            apply();
        iplBinauralEffectReset(effect);
        apply();
        recordTails(recorder, effect, outBuffer, iplBinauralEffectGetTail, iplBinauralEffectGetTailSize);
        recorder.write(out, c.name);

        iplBinauralEffectRelease(&effect);
    }
}

constexpr LayoutCase kVirtualSurroundLayouts[] = {
    {"virtual_surround_surround51", IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1},
    {"virtual_surround_custom", IPL_SPEAKERLAYOUTTYPE_CUSTOM},
};

void generateVirtualSurround(const std::filesystem::path& out, IPLContext context, IPLHRTF hrtf, Lcg& lcg)
{
    for (const auto& c : kVirtualSurroundLayouts)
    {
        auto audio = audioSettings();
        IPLVirtualSurroundEffectSettings settings{};
        settings.speakerLayout = makeLayout(c.type);
        settings.hrtf = hrtf;
        IPLVirtualSurroundEffect effect = nullptr;
        check(iplVirtualSurroundEffectCreate(context, &audio, &settings, &effect), "iplVirtualSurroundEffectCreate");

        Buffer in(context, numSpeakers(c.type), kFrameSize);
        Buffer outBuffer(context, 2, kFrameSize);
        Recorder recorder;
        auto apply = [&]() {
            in.fill(lcg);
            IPLVirtualSurroundEffectParams params{};
            params.hrtf = hrtf;
            outBuffer.fill(kSentinel);
            const auto state = iplVirtualSurroundEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
            recorder.record(outBuffer, state, iplVirtualSurroundEffectGetTailSize(effect));
        };
        for (int i = 0; i < kFrames; ++i)
            apply();
        iplVirtualSurroundEffectReset(effect);
        apply();
        recordTails(recorder, effect, outBuffer, iplVirtualSurroundEffectGetTail, iplVirtualSurroundEffectGetTailSize);
        recorder.write(out, c.name);

        iplVirtualSurroundEffectRelease(&effect);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Ambisonics effects
// --------------------------------------------------------------------------------------------------------------------

int numCoeffs(int order)
{
    return (order + 1) * (order + 1);
}

void generateAmbisonicsEncode(const std::filesystem::path& out, IPLContext context, Lcg& lcg)
{
    auto audio = audioSettings();
    IPLAmbisonicsEncodeEffectSettings settings{};
    settings.maxOrder = 2;
    IPLAmbisonicsEncodeEffect effect = nullptr;
    check(iplAmbisonicsEncodeEffectCreate(context, &audio, &settings, &effect), "iplAmbisonicsEncodeEffectCreate");

    Buffer in(context, 1, kFrameSize);
    Recorder recorder;
    auto apply = [&](int order) {
        Buffer outBuffer(context, numCoeffs(order), kFrameSize);
        in.fill(lcg);
        IPLAmbisonicsEncodeEffectParams params{};
        params.direction = drawVector(lcg);
        params.order = order;
        outBuffer.fill(kSentinel);
        const auto state = iplAmbisonicsEncodeEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
        recorder.record(outBuffer, state, iplAmbisonicsEncodeEffectGetTailSize(effect));
    };
    // Order 3 is above the maximum: only the first 9 channels are written.
    for (int order : {2, 2, 1, 3})
        apply(order);
    iplAmbisonicsEncodeEffectReset(effect);
    apply(2);
    Buffer tailBuffer(context, numCoeffs(2), kFrameSize);
    recordTails(recorder, effect, tailBuffer, iplAmbisonicsEncodeEffectGetTail, iplAmbisonicsEncodeEffectGetTailSize);
    recorder.write(out, "ambisonics_encode");

    iplAmbisonicsEncodeEffectRelease(&effect);
}

void generateAmbisonicsPanning(const std::filesystem::path& out, IPLContext context, Lcg& lcg)
{
    auto audio = audioSettings();
    IPLAmbisonicsPanningEffectSettings settings{};
    settings.speakerLayout = makeLayout(IPL_SPEAKERLAYOUTTYPE_SURROUND_7_1);
    settings.maxOrder = 2;
    IPLAmbisonicsPanningEffect effect = nullptr;
    check(iplAmbisonicsPanningEffectCreate(context, &audio, &settings, &effect), "iplAmbisonicsPanningEffectCreate");

    Buffer outBuffer(context, 8, kFrameSize);
    Recorder recorder;
    auto apply = [&](int order) {
        Buffer in(context, numCoeffs(order), kFrameSize);
        in.fill(lcg);
        IPLAmbisonicsPanningEffectParams params{};
        params.order = order;
        outBuffer.fill(kSentinel);
        const auto state = iplAmbisonicsPanningEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
        recorder.record(outBuffer, state, iplAmbisonicsPanningEffectGetTailSize(effect));
    };
    for (int order : {2, 1, 0})
        apply(order);
    iplAmbisonicsPanningEffectReset(effect);
    apply(2);
    recordTails(recorder, effect, outBuffer, iplAmbisonicsPanningEffectGetTail, iplAmbisonicsPanningEffectGetTailSize);
    recorder.write(out, "ambisonics_panning");

    iplAmbisonicsPanningEffectRelease(&effect);
}

void generateAmbisonicsBinaural(const std::filesystem::path& out, IPLContext context, IPLHRTF hrtf, Lcg& lcg)
{
    auto audio = audioSettings();
    IPLAmbisonicsBinauralEffectSettings settings{};
    settings.hrtf = hrtf;
    settings.maxOrder = 3;
    IPLAmbisonicsBinauralEffect effect = nullptr;
    check(iplAmbisonicsBinauralEffectCreate(context, &audio, &settings, &effect),
          "iplAmbisonicsBinauralEffectCreate");

    Buffer outBuffer(context, 2, kFrameSize);
    Recorder recorder;
    auto apply = [&](int order) {
        Buffer in(context, numCoeffs(order), kFrameSize);
        in.fill(lcg);
        IPLAmbisonicsBinauralEffectParams params{};
        params.hrtf = hrtf;
        params.order = order;
        outBuffer.fill(kSentinel);
        const auto state = iplAmbisonicsBinauralEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
        recorder.record(outBuffer, state, iplAmbisonicsBinauralEffectGetTailSize(effect));
    };
    for (int order : {3, 1, 2})
        apply(order);
    iplAmbisonicsBinauralEffectReset(effect);
    apply(3);
    recordTails(recorder, effect, outBuffer, iplAmbisonicsBinauralEffectGetTail,
                iplAmbisonicsBinauralEffectGetTailSize);
    recorder.write(out, "ambisonics_binaural");

    iplAmbisonicsBinauralEffectRelease(&effect);
}

void generateAmbisonicsRotation(const std::filesystem::path& out, IPLContext context, Lcg& lcg)
{
    auto audio = audioSettings();
    IPLAmbisonicsRotationEffectSettings settings{};
    settings.maxOrder = 2;
    IPLAmbisonicsRotationEffect effect = nullptr;
    check(iplAmbisonicsRotationEffectCreate(context, &audio, &settings, &effect),
          "iplAmbisonicsRotationEffectCreate");

    Recorder recorder;
    auto apply = [&](int order) {
        Buffer in(context, numCoeffs(order), kFrameSize);
        Buffer outBuffer(context, numCoeffs(order), kFrameSize);
        in.fill(lcg);
        IPLAmbisonicsRotationEffectParams params{};
        params.orientation = drawOrientation(lcg);
        params.order = order;
        outBuffer.fill(kSentinel);
        const auto state = iplAmbisonicsRotationEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
        recorder.record(outBuffer, state, iplAmbisonicsRotationEffectGetTailSize(effect));
    };
    // Order 3 is above the maximum: only the first 9 channels are written.
    for (int order : {2, 1, 3})
        apply(order);
    iplAmbisonicsRotationEffectReset(effect);
    apply(2);
    Buffer tailBuffer(context, numCoeffs(2), kFrameSize);
    recordTails(recorder, effect, tailBuffer, iplAmbisonicsRotationEffectGetTail,
                iplAmbisonicsRotationEffectGetTailSize);
    recorder.write(out, "ambisonics_rotation");

    iplAmbisonicsRotationEffectRelease(&effect);
}

struct DecodeOp
{
    int order;
    IPLbool binaural;
    bool withHRTF;
};

void runDecode(const std::filesystem::path& out, IPLContext context, IPLHRTF hrtf, Lcg& lcg, const char* name,
               IPLSpeakerLayoutType layout, int maxOrder, bool createWithHRTF, std::initializer_list<DecodeOp> ops)
{
    auto audio = audioSettings();
    IPLAmbisonicsDecodeEffectSettings settings{};
    settings.speakerLayout = makeLayout(layout);
    settings.hrtf = createWithHRTF ? hrtf : nullptr;
    settings.maxOrder = maxOrder;
    IPLAmbisonicsDecodeEffect effect = nullptr;
    check(iplAmbisonicsDecodeEffectCreate(context, &audio, &settings, &effect), "iplAmbisonicsDecodeEffectCreate");

    Buffer outBuffer(context, numSpeakers(layout), kFrameSize);
    Recorder recorder;
    auto apply = [&](const DecodeOp& op) {
        Buffer in(context, numCoeffs(op.order), kFrameSize);
        in.fill(lcg);
        IPLAmbisonicsDecodeEffectParams params{};
        params.order = op.order;
        params.hrtf = op.withHRTF ? hrtf : nullptr;
        params.orientation = drawOrientation(lcg);
        params.binaural = op.binaural;
        outBuffer.fill(kSentinel);
        const auto state = iplAmbisonicsDecodeEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
        recorder.record(outBuffer, state, iplAmbisonicsDecodeEffectGetTailSize(effect));
    };
    for (const auto& op : ops)
        apply(op);
    iplAmbisonicsDecodeEffectReset(effect);
    apply(*ops.begin());
    recordTails(recorder, effect, outBuffer, iplAmbisonicsDecodeEffectGetTail, iplAmbisonicsDecodeEffectGetTailSize);
    recorder.write(out, name);

    iplAmbisonicsDecodeEffectRelease(&effect);
}

void generateAmbisonicsDecode(const std::filesystem::path& out, IPLContext context, IPLHRTF hrtf, Lcg& lcg)
{
    // Binaural, binaural at a lower order, panning, binaural without an HRTF (no output), binaural again.
    runDecode(out, context, hrtf, lcg, "ambisonics_decode_binaural", IPL_SPEAKERLAYOUTTYPE_STEREO, 2, true,
              {{2, IPL_TRUE, true}, {1, IPL_TRUE, true}, {2, IPL_FALSE, true}, {2, IPL_TRUE, false},
               {2, IPL_TRUE, true}});

    // No HRTF at creation: binaural requests are panned.
    runDecode(out, context, hrtf, lcg, "ambisonics_decode_panning", IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1, 3, false,
              {{3, IPL_FALSE, false}, {2, IPL_FALSE, false}, {3, IPL_TRUE, true}, {1, IPL_FALSE, true}});
}

// --------------------------------------------------------------------------------------------------------------------
// Direct effect
// --------------------------------------------------------------------------------------------------------------------

constexpr int kDirectFrames = 9;
const IPLTransmissionType kTransmissionTypes[] = {
    IPL_TRANSMISSIONTYPE_FREQINDEPENDENT, IPL_TRANSMISSIONTYPE_FREQDEPENDENT, static_cast<IPLTransmissionType>(7)};

void generateDirect(const std::filesystem::path& out, IPLContext context, Lcg& lcg)
{
    auto audio = audioSettings();
    IPLDirectEffectSettings settings{};
    settings.numChannels = 2;
    IPLDirectEffect effect = nullptr;
    check(iplDirectEffectCreate(context, &audio, &settings, &effect), "iplDirectEffectCreate");

    Buffer in(context, 2, kFrameSize);
    Buffer outBuffer(context, 2, kFrameSize);
    Recorder recorder;
    auto apply = [&](int frame, bool inPlace) {
        in.fill(lcg);
        IPLDirectEffectParams params{};
        params.flags = static_cast<IPLDirectEffectFlags>(lcg.nextU32() >> 27);
        params.transmissionType = kTransmissionTypes[frame % 3];
        params.distanceAttenuation = drawUnit(lcg);
        for (auto& x : params.airAbsorption)
            x = drawUnit(lcg);
        params.directivity = drawUnit(lcg);
        params.occlusion = drawUnit(lcg);
        for (auto& x : params.transmission)
            x = drawUnit(lcg);
        if (inPlace)
        {
            const auto state = iplDirectEffectApply(effect, &params, &in.buffer, &in.buffer);
            recorder.record(in, state, iplDirectEffectGetTailSize(effect));
        }
        else
        {
            outBuffer.fill(kSentinel);
            const auto state = iplDirectEffectApply(effect, &params, &in.buffer, &outBuffer.buffer);
            recorder.record(outBuffer, state, iplDirectEffectGetTailSize(effect));
        }
    };
    for (int i = 0; i < kDirectFrames; ++i)
        apply(i, i % 4 == 3);
    iplDirectEffectReset(effect);
    apply(0, false);
    recordTails(recorder, effect, outBuffer, iplDirectEffectGetTail, iplDirectEffectGetTailSize);
    recorder.write(out, "direct");

    iplDirectEffectRelease(&effect);
}

void generateEffects(const std::filesystem::path& out)
{
    auto context = makeContext(kVersion);
    auto hrtf = makeDefaultHRTF(context);
    Lcg lcg(806);

    generatePanning(out, context, lcg);
    generateBinaural(out, context, hrtf, lcg);
    generateVirtualSurround(out, context, hrtf, lcg);
    generateAmbisonicsEncode(out, context, lcg);
    generateAmbisonicsPanning(out, context, lcg);
    generateAmbisonicsBinaural(out, context, hrtf, lcg);
    generateAmbisonicsRotation(out, context, lcg);
    generateAmbisonicsDecode(out, context, hrtf, lcg);
    generateDirect(out, context, lcg);

    iplHRTFRelease(&hrtf);
    iplContextRelease(&context);
}

void generate(const std::filesystem::path& out)
{
    generateLayout(out);
    generateContext(out);
    generateAudioBuffers(out);
    generateModels(out);
    generateHRTFs(out);
    generateEffects(out);
    ipl::Context::sSIMDLevel = ipl::SIMDLevel::SSE2;
}

const Register reg("capi", generate);

}
