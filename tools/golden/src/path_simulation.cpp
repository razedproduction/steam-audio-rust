// Golden data for stage 7b: deviation (DeviationModel), path_simulator (PathSimulator), path_effect (PathEffect) and the
// pathing parts of simulation_manager and simulation_data.
//
// Paths are baked (as in pathing.cpp's bake 0) into the uniform floor probes of simplescene, and simulated in a copy of
// the scene with an instanced smallbox scaled into a wall in its middle, so that some baked paths are blocked and must
// be validated and replaced. Other inputs come from Lcg draws in a fixed order, one value per statement.
// crates/steam-audio/tests/path_simulation.rs replays them. Every thread pool has one thread. Path effects are built
// with the AVX SIMD level, the path the port follows (risk R8; needs an AVX CPU).
//
//   deviation_angles:       the angles the deviation models are evaluated at;
//   deviation_values:       per angle and band, the default (UTD) model, then a callback model;
//   simulator_results:      per findPaths query: the return value and the number of validation ray callbacks;
//   simulator_values:       per query: EQ gains (3), SH coefficients (16, order 3; untouched ones keep a sentinel),
//                           average direction, distance ratio, total deviation;
//   simulator_rays:         every validation ray callback: query, from, to, occluded;
//   effect_<case>_output:   PathEffect outputs, one row per channel and operation (see kEffectCases);
//   effect_<case>_states:   per operation, the returned state and numTailSamplesRemaining;
//   manager_values:         per frame and source (in the order added), the pathing outputs: EQ (3), SH (9), direction,
//                           distance ratio, total deviation;
//   manager_rays:           the validation ray callbacks of frame 1: from, to, occluded.

#include <cstdint>
#include <string>
#include <vector>

#include <context.h>
#include <path_data.h>
#include <path_effect.h>
#include <path_simulator.h>
#include <probe_batch.h>
#include <probe_generator.h>
#include <scene.h>
#include <sh.h>
#include <simulation_data.h>
#include <simulation_manager.h>

#include "generators.h"
#include "npy.h"
#include "scene_inputs.h"

namespace {

using namespace golden;
using namespace ipl;

constexpr float kSentinel = -7.0f;
constexpr int kSamplingRate = 48000;
constexpr int kFrameSize = 100;

void writeMatrix(const std::filesystem::path& path, size_t cols, const std::vector<float>& data)
{
    writeNpy(path, {data.size() / cols, cols}, data.data());
}

void writeMatrix(const std::filesystem::path& path, size_t cols, const std::vector<int32_t>& data)
{
    writeNpy(path, {data.size() / cols, cols}, data.data());
}

void appendVector(std::vector<float>& out, const Vector3f& v)
{
    out.push_back(v.x());
    out.push_back(v.y());
    out.push_back(v.z());
}

Box meshBounds(const ObjMesh& obj)
{
    Box bounds;
    for (const auto& v : obj.vertices)
    {
        for (auto i = 0; i < 3; ++i)
        {
            bounds.minCoordinates[i] = std::min(bounds.minCoordinates[i], v[i]);
            bounds.maxCoordinates[i] = std::max(bounds.maxCoordinates[i], v[i]);
        }
    }
    return bounds;
}

// pathing.cpp's boxTransform.
Matrix4x4f boxTransform(const Box& bounds)
{
    const auto extents = bounds.extents();
    const auto center = bounds.center();

    Matrix4x4f m;
    for (auto i = 0; i < 4; ++i)
        for (auto j = 0; j < 4; ++j)
            m(i, j) = 0.0f;
    for (auto i = 0; i < 3; ++i)
    {
        m(i, i) = extents[i];
        m(i, 3) = center[i];
    }
    m(3, 3) = 1.0f;
    return m;
}

// The smallbox (0.8 m wide) scaled into a wall 2 m thick, 10 m high and 30 m long across the middle of simplescene (x
// from 12.5 to 14.5), leaving gaps of about 9 m at both ends (z from -23.6 to 6.4).
Matrix4x4f wallTransform()
{
    Matrix4x4f m;
    for (auto i = 0; i < 4; ++i)
        for (auto j = 0; j < 4; ++j)
            m(i, j) = 0.0f;
    m(0, 0) = 2.5f;
    m(1, 1) = 12.5f;
    m(2, 2) = 37.5f;
    m(0, 3) = 13.5f;
    m(1, 3) = 0.0f;
    m(2, 3) = -8.6f;
    m(3, 3) = 1.0f;
    return m;
}

// A point above the floor of simplescene (1.5 m below the probes, about 2.2 m above the bottom of `bounds`): anywhere
// horizontally (x within the fraction [x0, x0 + w] of the bounds), 2.5 to 4.5 m above the bottom.
Vector3f drawInterior(Lcg& lcg, const Box& bounds, float x0, float w)
{
    const float x = draw01(lcg);
    const float y = draw01(lcg);
    const float z = draw01(lcg);
    const auto& lo = bounds.minCoordinates;
    const auto& hi = bounds.maxCoordinates;
    return Vector3f(lo.x() + (hi.x() - lo.x()) * (x0 + w * x), lo.y() + 2.5f + 2.0f * y,
                    lo.z() + (hi.z() - lo.z()) * z);
}

// Left of the wall, right of it, or anywhere.
Vector3f drawLeft(Lcg& lcg, const Box& bounds)
{
    return drawInterior(lcg, bounds, 0.0f, 0.45f);
}

Vector3f drawRight(Lcg& lcg, const Box& bounds)
{
    return drawInterior(lcg, bounds, 0.55f, 0.45f);
}

Vector3f drawAnywhere(Lcg& lcg, const Box& bounds)
{
    return drawInterior(lcg, bounds, 0.0f, 1.0f);
}

float callbackDeviation(float angle, int band, void*)
{
    return 1.0f / (1.0f + angle * static_cast<float>(band + 1));
}

float callbackAttenuation(float distance, void*)
{
    return 1.0f / (1.0f + distance);
}

BakedDataIdentifier pathingIdentifier()
{
    BakedDataIdentifier identifier;
    identifier.variation = BakedDataVariation::Dynamic;
    identifier.type = BakedDataType::Pathing;
    return identifier;
}

// Floor probes of the scene (pathing.cpp's bake 0: point-to-point visibility, no range limits), baked and committed.
shared_ptr<ProbeBatch> bakeBatch(const Scene& scene, const Box& bounds)
{
    ProbeArray probes;
    ProbeGenerator::generateProbes(scene, boxTransform(bounds), ProbeGenerationType::UniformFloor, 4.5f, 1.5f, probes);

    auto batch = ipl::make_shared<ProbeBatch>();
    batch->addProbeArray(probes);
    PathBaker::bake(scene, pathingIdentifier(), 1, 0.0f, 0.5f, 1000.0f, 1000.0f, 1000.0f, false,
                    Vector3f(0.0f, -1.0f, 0.0f), false, 1, *batch);
    batch->commit();
    return batch;
}

// Validation rays recorded by the callback, and the query they belong to.
struct RayRecorder
{
    std::vector<float> rays;
    int query = 0;
    int count = 0;
    bool withQuery = true;
};

void IPL_CALLBACK recordRay(Vector3f from, Vector3f to, bool occluded, void* userData)
{
    auto& recorder = *static_cast<RayRecorder*>(userData);
    if (recorder.withQuery)
        recorder.rays.push_back(static_cast<float>(recorder.query));
    appendVector(recorder.rays, from);
    appendVector(recorder.rays, to);
    recorder.rays.push_back(occluded ? 1.0f : 0.0f);
    ++recorder.count;
}

// --------------------------------------------------------------------------------------------------------------------
// DeviationModel
// --------------------------------------------------------------------------------------------------------------------

void generateDeviation(const std::filesystem::path& out)
{
    Lcg lcg(311);

    // 0 makes one cotangent infinite (the isFinite branch); multiples of pi near the shadow boundaries.
    std::vector<float> angles = {0.0f, -0.0f, 1e-8f, 1e-4f, 0.25f, 0.5f, 1.0f, 1.570796f, 2.0f, 3.0f, 3.141592f,
                                 3.5f, 6.283184f, -0.5f, -3.141592f, 10.0f};
    for (auto i = 0; i < 48; ++i)
    {
        const float x = lcg.nextSigned();
        angles.push_back(6.283184f * x);
    }

    const DeviationModel utd;
    const DeviationModel callback(callbackDeviation, nullptr);

    std::vector<float> values;
    for (const auto angle : angles)
    {
        for (auto band = 0; band < Bands::kNumBands; ++band)
            values.push_back(utd.evaluate(angle, band));
        for (auto band = 0; band < Bands::kNumBands; ++band)
            values.push_back(callback.evaluate(angle, band));
    }

    writeNpy(out / "deviation_angles.npy", angles);
    writeMatrix(out / "deviation_values.npy", 2 * Bands::kNumBands, values);
}

// --------------------------------------------------------------------------------------------------------------------
// PathSimulator
// --------------------------------------------------------------------------------------------------------------------

constexpr int kNumQueries = 64;
constexpr int kQueryOrder = 3;

void generateSimulator(const std::filesystem::path& out, const shared_ptr<Scene>& scene, ProbeBatch& batch,
                       const Box& bounds)
{
    Lcg lcg(613);

    const Vector3f down(0.0f, -1.0f, 0.0f);
    PathSimulator pointSimulator(batch, 1, false, down);
    PathSimulator volumetricSimulator(batch, 3, false, down);

    const DistanceAttenuationModel attenuationModels[] = {
        DistanceAttenuationModel{},
        DistanceAttenuationModel(2.0f, nullptr, nullptr),
        DistanceAttenuationModel(1.0f, callbackAttenuation, nullptr),
    };
    const DeviationModel deviationModels[] = {DeviationModel{}, DeviationModel(callbackDeviation, nullptr)};

    ProbeNeighborhood sourceProbes;
    ProbeNeighborhood listenerProbes;
    sourceProbes.resize(ProbeNeighborhood::kMaxProbesPerBatch);
    listenerProbes.resize(ProbeNeighborhood::kMaxProbesPerBatch);

    RayRecorder recorder;
    std::vector<int32_t> results;
    std::vector<float> values;

    for (auto q = 0; q < kNumQueries; ++q)
    {
        // Mostly on opposite sides of the wall.
        const auto anywhere = (q % 4) == 3;
        const auto source = anywhere ? drawAnywhere(lcg, bounds) : drawLeft(lcg, bounds);
        const auto listener = anywhere ? drawAnywhere(lcg, bounds) : drawRight(lcg, bounds);

        const auto volumetric = (q % 2) != 0;
        const auto enableValidation = ((q / 2) % 2) != 0;
        const auto findAlternatePaths = ((q / 4) % 2) != 0;
        const auto simplifyPaths = ((q / 8) % 2) != 0;
        const auto realTimeVis = ((q / 16) % 2) != 0;
        const auto allSourceProbes = (q % 3) == 0;
        const auto forceDirectOcclusion = (q % 5) == 0;
        const auto order = q % 4;
        const auto& attenuation = attenuationModels[q % 3];
        const auto& deviation = deviationModels[(q / 3) % 2];
        const auto radius = volumetric ? 0.6f : 0.0f;

        sourceProbes.reset();
        batch.getInfluencingProbes(source, sourceProbes);
        sourceProbes.checkOcclusion(*scene, source);
        sourceProbes.calcWeights(source);

        listenerProbes.reset();
        batch.getInfluencingProbes(listener, listenerProbes);
        listenerProbes.checkOcclusion(*scene, listener);
        listenerProbes.calcWeights(listener);

        float eq[Bands::kNumBands];
        float coeffs[16];
        std::fill(std::begin(eq), std::end(eq), kSentinel);
        std::fill(std::begin(coeffs), std::end(coeffs), kSentinel);
        Vector3f direction(kSentinel, kSentinel, kSentinel);
        float distanceRatio = kSentinel;
        float totalDeviation = kSentinel;

        recorder.query = q;
        recorder.count = 0;

        PathSimulator::sEnablePathsFromAllSourceProbes = allSourceProbes;
        auto& simulator = volumetric ? volumetricSimulator : pointSimulator;
        const auto found = simulator.findPaths(source, listener, *scene, batch, sourceProbes, listenerProbes, radius,
                                               0.3f, 1000.0f, order, enableValidation, findAlternatePaths,
                                               simplifyPaths, realTimeVis, eq, coeffs, attenuation, deviation,
                                               &direction, &distanceRatio, &totalDeviation, recordRay, &recorder,
                                               forceDirectOcclusion);
        PathSimulator::sEnablePathsFromAllSourceProbes = false;

        results.push_back(found ? 1 : 0);
        results.push_back(recorder.count);

        values.insert(values.end(), std::begin(eq), std::end(eq));
        values.insert(values.end(), std::begin(coeffs), std::end(coeffs));
        appendVector(values, direction);
        values.push_back(distanceRatio);
        values.push_back(totalDeviation);
    }

    writeMatrix(out / "simulator_results.npy", 2, results);
    writeMatrix(out / "simulator_values.npy", 24, values);
    writeMatrix(out / "simulator_rays.npy", 8, recorder.rays);
}

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

struct EffectCase
{
    const char* name;
    int maxOrder;
    bool spatialize;
    SpeakerLayoutType layout;
    // N<o>: apply unspatialized; P<o>: panning; Q<o>: panning with normalized EQ; B<o>: binaural; C<o>: binaural with
    // normalized EQ; T: tail; R: reset.
    const char* script;
};

constexpr EffectCase kEffectCases[] = {
    {"ambisonics", 2, false, SpeakerLayoutType::Mono, "N2N2N1N0TRN2N2"},
    {"stereo", 3, true, SpeakerLayoutType::Stereo, "B3B3C2B1TTP2Q3P1TRB2TP0B3"},
    {"surround", 1, true, SpeakerLayoutType::FivePointOne, "P1Q1P0P1RP1T"},
};

HRTFDatabase makeDatabase()
{
    HRTFSettings settings{};
    settings.type = HRTFMapType::Default;
    settings.hrtfData = nullptr;
    settings.volume = 0.0f;
    settings.normType = HRTFNormType::None;
    return HRTFDatabase(settings, kSamplingRate, kFrameSize);
}

void generateEffect(const std::filesystem::path& out)
{
    Lcg lcg(907);
    auto hrtf = makeDatabase();

    for (const auto& c : kEffectCases)
    {
        const SpeakerLayout layout(c.layout);

        AudioSettings audioSettings{};
        audioSettings.samplingRate = kSamplingRate;
        audioSettings.frameSize = kFrameSize;

        PathEffectSettings settings{};
        settings.maxOrder = c.maxOrder;
        settings.spatialize = c.spatialize;
        settings.speakerLayout = &layout;
        settings.hrtf = &hrtf;

        // EQEffect's IIR filterers pick their block algorithm from Context::sSIMDLevel when they are constructed: the
        // AVX path is the one the port follows (risk R8; needs an AVX CPU).
        Context::sSIMDLevel = SIMDLevel::AVX;
        PathEffect effect(audioSettings, settings);
        Context::sSIMDLevel = SIMDLevel::SSE2;

        const auto numChannels = c.spatialize ? layout.numSpeakers : SphericalHarmonics::numCoeffsForOrder(c.maxOrder);
        AudioBuffer in(1, kFrameSize);
        AudioBuffer output(numChannels, kFrameSize);

        std::vector<float> samples;
        std::vector<int32_t> states;

        for (const char* op = c.script; *op; ++op)
        {
            AudioEffectState state;
            if (*op == 'R')
            {
                effect.reset();
                continue;
            }
            else if (*op == 'T')
            {
                for (auto i = 0; i < numChannels; ++i)
                    std::fill(output[i], output[i] + kFrameSize, kSentinel);
                state = effect.tail(output);
            }
            else
            {
                const auto kind = *op;
                ++op;
                const auto order = *op - '0';

                for (auto i = 0; i < kFrameSize; ++i)
                {
                    const float x = lcg.nextSigned();
                    in[0][i] = x;
                }

                float eq[Bands::kNumBands];
                for (auto& g : eq)
                {
                    const float x = draw01(lcg);
                    g = 0.02f + 0.98f * x;
                }

                float sh[16];
                for (auto i = 0; i < SphericalHarmonics::numCoeffsForOrder(order); ++i)
                {
                    const float x = lcg.nextSigned();
                    sh[i] = x;
                }

                const auto ahead = Vector3f::unitVector(drawVector(lcg));
                const CoordinateSpace3f listener(ahead, Vector3f::kZero);

                PathEffectParams params{};
                params.eqCoeffs = eq;
                params.shCoeffs = sh;
                params.order = order;
                params.binaural = (kind == 'B' || kind == 'C');
                params.hrtf = &hrtf;
                params.listener = &listener;
                params.normalizeEQ = (kind == 'Q' || kind == 'C');

                for (auto i = 0; i < numChannels; ++i)
                    std::fill(output[i], output[i] + kFrameSize, kSentinel);
                state = effect.apply(params, in, output);
            }

            for (auto i = 0; i < numChannels; ++i)
                samples.insert(samples.end(), output[i], output[i] + kFrameSize);
            states.push_back(static_cast<int32_t>(state));
            states.push_back(effect.numTailSamplesRemaining());
        }

        const auto prefix = std::string("effect_") + c.name;
        writeMatrix(out / (prefix + "_output.npy"), kFrameSize, samples);
        writeMatrix(out / (prefix + "_states.npy"), 2, states);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// SimulationManager
// --------------------------------------------------------------------------------------------------------------------

constexpr int kManagerOrder = 2;
constexpr int kNumFrames = 4;

struct ManagerSource
{
    bool enabled;
    int batch; // 0: A, 1: B
    int order;
    float radius;
    bool enableValidation;
    bool findAlternatePaths;
    bool simplifyPaths;
    bool realTimeVis;
    int attenuation; // 0: default, 1: inverse distance from 2 m, 2: callback
    bool callbackDeviation;
};

// In the order added: batches A, B, A, A, then a disabled source, so the listener's probes are looked up again when the
// batch changes. Source 3 starts outside every probe.
const ManagerSource kManagerSources[] = {
    {true, 0, 2, 0.6f, true, true, true, true, 0, false},
    {true, 1, 2, 0.6f, true, false, true, false, 2, false},
    {true, 0, 1, 0.0f, false, false, false, false, 1, true},
    {true, 0, 0, 0.0f, true, true, false, true, 0, false},
    {false, 0, 2, 0.0f, false, false, false, false, 0, false},
};

void generateManager(const std::filesystem::path& out, const shared_ptr<Scene>& scene,
                     const shared_ptr<ProbeBatch>& batchA, const shared_ptr<ProbeBatch>& batchB, const Box& bounds)
{
    Lcg lcg(1032);

    SimulationManager manager(false, false, true, SceneType::Default, IndirectEffectType::Convolution, 1, 1, 1, 1.0f,
                              kManagerOrder, 4, 1, 1, 1, 3, false, Vector3f(0.0f, -1.0f, 0.0f), kSamplingRate,
                              kFrameSize, nullptr, nullptr, nullptr);
    manager.scene() = scene;
    manager.addProbeBatch(batchA);
    manager.addProbeBatch(batchB);

    constexpr int kNumSources = sizeof(kManagerSources) / sizeof(kManagerSources[0]);
    shared_ptr<SimulationData> sources[kNumSources];
    for (auto i = 0; i < kNumSources; ++i)
    {
        const auto& c = kManagerSources[i];
        auto s = ipl::make_shared<SimulationData>(false, true, SceneType::Default, IndirectEffectType::Convolution, 1,
                                                  1.0f, kManagerOrder, kSamplingRate, kFrameSize, nullptr, nullptr);

        // Upstream leaves these uninitialized.
        s->pathingState.direction = Vector3f::kZero;
        s->pathingState.distanceRatio = 0.0f;
        s->pathingState.totalDeviation = 0.0f;
        s->pathingOutputs.direction = Vector3f::kZero;
        s->pathingOutputs.distanceRatio = 0.0f;
        s->pathingOutputs.totalDeviation = 0.0f;

        auto& inputs = s->pathingInputs;
        inputs.enabled = c.enabled;
        inputs.probes = (c.batch == 0) ? batchA : batchB;
        inputs.visRadius = c.radius;
        inputs.visThreshold = 0.3f;
        inputs.visRange = 1000.0f;
        inputs.order = c.order;
        inputs.enableValidation = c.enableValidation;
        inputs.findAlternatePaths = c.findAlternatePaths;
        inputs.simplifyPaths = c.simplifyPaths;
        inputs.realTimeVis = c.realTimeVis;
        if (c.attenuation == 1)
            inputs.distanceAttenuationModel = DistanceAttenuationModel(2.0f, nullptr, nullptr);
        else if (c.attenuation == 2)
            inputs.distanceAttenuationModel = DistanceAttenuationModel(1.0f, callbackAttenuation, nullptr);
        else
            inputs.distanceAttenuationModel = DistanceAttenuationModel{};
        inputs.deviationModel = c.callbackDeviation ? DeviationModel(callbackDeviation, nullptr) : DeviationModel{};

        sources[i] = s;
        manager.addSource(s);
    }
    manager.commit();

    RayRecorder recorder;
    recorder.withQuery = false;
    std::vector<float> values;

    for (auto frame = 0; frame < kNumFrames; ++frame)
    {
        auto listener = drawRight(lcg, bounds);
        if (frame == 3)
            listener = Vector3f(100.0f, 100.0f, 100.0f);

        for (auto i = 0; i < kNumSources; ++i)
        {
            auto origin = drawLeft(lcg, bounds);
            if (i == 3 && frame == 0)
                origin = Vector3f(100.0f, 100.0f, 100.0f);
            sources[i]->pathingInputs.source = CoordinateSpace3f(origin);
        }

        SharedPathingSimulationInputs shared;
        shared.listener = CoordinateSpace3f(listener);
        if (frame == 1)
        {
            shared.visCallback = recordRay;
            shared.userData = &recorder;
        }
        manager.setSharedPathingInputs(shared);

        if (frame == 1)
            PathSimulator::sEnablePathsFromAllSourceProbes = true;

        if (frame == 2)
        {
            for (auto& s : sources)
                manager.simulatePathing(*s);
        }
        else
        {
            manager.simulatePathing();
        }

        PathSimulator::sEnablePathsFromAllSourceProbes = false;

        for (const auto& s : sources)
        {
            const auto& o = s->pathingOutputs;
            values.insert(values.end(), std::begin(o.eq), std::end(o.eq));
            values.insert(values.end(), o.sh.data(), o.sh.data() + o.sh.totalSize());
            appendVector(values, o.direction);
            values.push_back(o.distanceRatio);
            values.push_back(o.totalDeviation);
        }
    }

    writeMatrix(out / "manager_values.npy", 17, values);
    writeMatrix(out / "manager_rays.npy", 7, recorder.rays);
}

void generatePathSimulation(const std::filesystem::path& out)
{
    generateDeviation(out);

    const auto obj = loadObj("simplescene");
    const auto smallbox = loadObj("smallbox");
    const auto bounds = meshBounds(obj);

    Scene bakeScene;
    bakeScene.addStaticMesh(createStaticMesh(bakeScene, obj));
    bakeScene.commit();

    auto boxScene = ipl::make_shared<Scene>();
    boxScene->addStaticMesh(createStaticMesh(*boxScene, smallbox));
    boxScene->commit();

    auto scene = ipl::make_shared<Scene>();
    scene->addStaticMesh(createStaticMesh(*scene, obj));
    scene->addInstancedMesh(scene->createInstancedMesh(boxScene, wallTransform()));
    scene->commit();

    const auto batchA = bakeBatch(bakeScene, bounds);
    const auto batchB = bakeBatch(bakeScene, bounds);

    generateSimulator(out, scene, *batchA, bounds);
    generateEffect(out);
    generateManager(out, scene, batchA, batchB, bounds);
}

Register pathSimulation("path_simulation", generatePathSimulation);

}
