// Golden data for stage 6.4c: reflection_baker, simulation_data and simulation_manager.
//
// The scene is boxroom (core/data/meshes, read with the OBJ loader in scene_inputs.h) with Lcg materials; its 25 floor
// probes (spacing 4, height 1.5) form one probe batch. Other inputs come from Lcg draws in a fixed order, one value
// per statement. crates/steam-audio/tests/simulation.rs replays them.
//
// Upstream seeds the reflection simulators' random number generators with time(NULL). As in reflection_simulator.cpp,
// the generator reseeds them right after construction, reaching private members through explicit template
// instantiation (where access checks do not apply); the Rust simulators take the seed as an argument. Every thread pool
// has one thread, so rays are traced in order on one generator.
//
//   bake_<k>.npy            ReflectionBaker::bake of kBakes[k] into the batch, one after another with one simulator;
//                           the batch serialized after each bake (bytes);
//   bake_<k>_progress.npy   the progress values the bake reported (the last bake cancels itself after 4 of them);
//   manager_<type>_*.npy    a SimulationManager with indirect effect type <type> and six sources (see kSources),
//                           simulated for four frames (see the frame loop); per frame and source, in order:
//     _direct               the direct sound path (distance attenuation, air absorption x3, delay, occlusion,
//                           transmission x3, directivity);
//     _fields               the accumulated energy field (flat data);
//     _outputs              the reverb times, then for hybrid the hybrid EQ and delay, then for convolution and hybrid
//                           the output channel and sample counts;
//     _irs                  for convolution and hybrid, the impulse response (flat data).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <baked_reflection_data.h>
#include <context.h>
#include <probe_batch.h>
#include <probe_generator.h>
#include <reflection_baker.h>
#include <reflection_simulator.h>
#include <scene.h>
#include <serialized_object.h>
#include <simulation_data.h>
#include <simulation_manager.h>

#include "generators.h"
#include "npy.h"
#include "scene_inputs.h"

namespace {

using namespace golden;
using namespace ipl;

// --------------------------------------------------------------------------------------------------------------------
// Access to private members
// --------------------------------------------------------------------------------------------------------------------

std::default_random_engine& engine(RandomNumberGenerator& rng);
void reseed(ReflectionSimulator& simulator, unsigned int seed);
unique_ptr<IReflectionSimulator>& managerSimulator(SimulationManager& manager);

template <typename Member, Member M>
struct EngineAccess
{
    friend std::default_random_engine& engine(RandomNumberGenerator& rng)
    {
        return rng.*M;
    }
};

template <typename Member, Member M>
struct ReseedAccess
{
    friend void reseed(ReflectionSimulator& simulator, unsigned int seed)
    {
        auto& states = simulator.*M;
        for (auto i = 0; i < static_cast<int>(states.size(0)); ++i)
        {
            engine(states[i].rng).seed(seed);
        }
    }
};

template <typename Member, Member M>
struct ManagerAccess
{
    friend unique_ptr<IReflectionSimulator>& managerSimulator(SimulationManager& manager)
    {
        return manager.*M;
    }
};

template struct EngineAccess<decltype(&RandomNumberGenerator::mGenerator), &RandomNumberGenerator::mGenerator>;
template struct ReseedAccess<decltype(&ReflectionSimulator::mThreadState), &ReflectionSimulator::mThreadState>;
template struct ManagerAccess<decltype(&SimulationManager::mReflectionSimulator),
                              &SimulationManager::mReflectionSimulator>;

// --------------------------------------------------------------------------------------------------------------------
// Inputs
// --------------------------------------------------------------------------------------------------------------------

void writeBytes(const std::filesystem::path& path, SerializedObject& object)
{
    const auto* data = reinterpret_cast<const uint8_t*>(object.data());
    writeNpy(path, std::vector<uint8_t>(data, data + object.size()));
}

void writeMatrix(const std::filesystem::path& path, size_t cols, const std::vector<float>& data)
{
    writeNpy(path, {data.size() / cols, cols}, data.data());
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

// The box `bounds` as an oriented box transform (no rotation).
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

// A point in the middle 70% of the box along each axis.
Vector3f drawInterior(Lcg& lcg, const Box& box)
{
    const auto extents = box.extents();
    Vector3f p;
    for (auto i = 0; i < 3; ++i)
    {
        const float t = draw01(lcg) * 0.7f + 0.15f;
        p[i] = box.minCoordinates[i] + extents[i] * t;
    }
    return p;
}

BakedDataIdentifier makeIdentifier(BakedDataVariation variation, const Vector3f& center, float radius)
{
    BakedDataIdentifier identifier;
    identifier.variation = variation;
    identifier.type = BakedDataType::Reflections;
    identifier.endpointInfluence.center = center;
    identifier.endpointInfluence.radius = radius;
    return identifier;
}

// --------------------------------------------------------------------------------------------------------------------
// Baking
// --------------------------------------------------------------------------------------------------------------------

enum class Layer
{
    Reverb,
    Source,
    Listener
};

struct Bake
{
    Layer layer;
    bool convolution;
    bool parametric;
    int numRays;
    int numBounces;
    float simDuration;
    float bakeDuration;
    int order;
    float irradianceMinDistance;
    int cancelAfter; // number of progress reports after which the callback cancels the bake; 0: never
};

// Bake 3 turns convolution on for a parametric layer, with a bake duration above the simulation's; bake 4 bakes the
// static source layer again and cancels itself.
const Bake kBakes[] = {
    {Layer::Reverb, true, true, 256, 4, 0.2f, 0.2f, 1, 1.0f, 0},
    {Layer::Source, true, false, 200, 3, 0.2f, 0.15f, 1, 0.5f, 0},
    {Layer::Listener, false, true, 128, 2, 0.15f, 0.15f, 0, 1.0f, 0},
    {Layer::Listener, true, true, 128, 2, 0.15f, 0.25f, 1, 1.0f, 0},
    {Layer::Source, true, false, 128, 6, 0.2f, 0.2f, 1, 1.0f, 4},
};

struct Progress
{
    std::vector<float> values;
    int cancelAfter;
};

void IPL_CALLBACK onProgress(float progress, void* userData)
{
    auto& p = *static_cast<Progress*>(userData);
    p.values.push_back(progress);
    if (static_cast<int>(p.values.size()) == p.cancelAfter)
        ReflectionBaker::cancel();
}

// --------------------------------------------------------------------------------------------------------------------
// Simulation manager
// --------------------------------------------------------------------------------------------------------------------

constexpr float kMaxDuration = 0.08f;
constexpr int kMaxOrder = 1;
constexpr int kSamplingRate = 24000;
constexpr int kFrameSize = 256;
constexpr int kNumSources = 6;

struct ManagerCase
{
    IndirectEffectType type;
    const char* name;
    unsigned int randSeed;
    unsigned int simulatorSeed;
};

const ManagerCase kManagers[] = {
    {IndirectEffectType::Convolution, "convolution", 11, 1700000301u},
    {IndirectEffectType::Parametric, "parametric", 12, 1700000302u},
    {IndirectEffectType::Hybrid, "hybrid", 13, 1700000303u},
};

struct SourceInputs
{
    CoordinateSpace3f source;
    Directivity directivity;
    float airCoefficients[Bands::kNumBands];
};

void appendOutputs(const SimulationData& s, IndirectEffectType type, std::vector<float>& direct,
                   std::vector<float>& fields, std::vector<float>& outputs, std::vector<float>& irs)
{
    const auto& path = s.directOutputs.directPath;
    direct.push_back(path.distanceAttenuation);
    for (auto i = 0; i < Bands::kNumBands; ++i)
        direct.push_back(path.airAbsorption[i]);
    direct.push_back(path.delay);
    direct.push_back(path.occlusion);
    for (auto i = 0; i < Bands::kNumBands; ++i)
        direct.push_back(path.transmission[i]);
    direct.push_back(path.directivity);

    const auto& field = *s.reflectionState.accumEnergyField;
    const auto fieldSize = static_cast<size_t>(field.numChannels()) * Bands::kNumBands * field.numBins();
    fields.insert(fields.end(), field.flatData(), field.flatData() + fieldSize);

    for (auto i = 0; i < Bands::kNumBands; ++i)
        outputs.push_back(s.reflectionOutputs.reverb.reverbTimes[i]);
    if (type == IndirectEffectType::Hybrid)
    {
        for (auto i = 0; i < Bands::kNumBands; ++i)
            outputs.push_back(s.reflectionOutputs.hybridEQ[i]);
        outputs.push_back(static_cast<float>(s.reflectionOutputs.hybridDelay));
    }
    if (type != IndirectEffectType::Parametric)
    {
        outputs.push_back(static_cast<float>(s.reflectionOutputs.numChannels));
        outputs.push_back(static_cast<float>(s.reflectionOutputs.numSamples));

        const auto& ir = *s.reflectionState.impulseResponse;
        const auto irSize = static_cast<size_t>(ir.numChannels()) * ir.numSamples();
        irs.insert(irs.end(), ir.data(), ir.data() + irSize);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Generator
// --------------------------------------------------------------------------------------------------------------------

void generateSimulation(const std::filesystem::path& out)
{
    Lcg lcg(641);

    auto obj = loadObj("boxroom");
    assignMaterials(lcg, obj);
    const auto bounds = meshBounds(obj);

    auto scene = ipl::make_shared<Scene>();
    scene->addStaticMesh(createStaticMesh(*scene, obj));
    scene->commit();

    ProbeArray probes;
    ProbeGenerator::generateProbes(*scene, boxTransform(bounds), ProbeGenerationType::UniformFloor, 4.0f, 1.5f,
                                   probes);
    if (probes.numProbes() != 25)
        throw std::runtime_error("unexpected number of floor probes");

    auto batch = ipl::make_shared<ProbeBatch>();
    batch->addProbeArray(probes);
    batch->commit();

    const auto reverbId = makeIdentifier(BakedDataVariation::Reverb, Vector3f(0.0f, 0.0f, 0.0f), 0.0f);
    const auto sourcePoint = probes[0].influence.center + Vector3f(1.0f, 0.2f, 1.5f);
    const auto sourceId = makeIdentifier(BakedDataVariation::StaticSource, sourcePoint, 12.0f);
    const auto listenerPoint = probes[6].influence.center + Vector3f(0.5f, 0.0f, 0.5f);
    const auto listenerId = makeIdentifier(BakedDataVariation::StaticListener, listenerPoint, 4.5f);

    // Baking.
    {
        ReflectionSimulator simulator(256, 32, 0.25f, 1, 1, 1);
        reseed(simulator, 1700000201u);

        auto k = 0;
        for (const auto& bake : kBakes)
        {
            const auto& identifier = (bake.layer == Layer::Reverb) ? reverbId
                                   : (bake.layer == Layer::Source) ? sourceId
                                                                    : listenerId;

            Progress progress;
            progress.cancelAfter = bake.cancelAfter;

            ReflectionBaker::bake(*scene, simulator, identifier, bake.convolution, bake.parametric, bake.numRays,
                                  bake.numBounces, bake.simDuration, bake.bakeDuration, bake.order,
                                  bake.irradianceMinDistance, 1, 1, SceneType::Default, nullptr, *batch, onProgress,
                                  &progress);

            SerializedObject serialized;
            batch->serializeAsRoot(serialized);
            writeBytes(out / ("bake_" + std::to_string(k) + ".npy"), serialized);
            writeNpy(out / ("bake_" + std::to_string(k) + "_progress.npy"), progress.values);
            ++k;
        }
    }

    // Simulation inputs: two listeners, six sources (0, 1: real time; 2: baked reverb; 3: baked static listener;
    // 4: baked static source; 5: direct sound only), and where source 0 moves to.
    const auto listener0 = probes[12].influence.center + Vector3f(0.3f, 0.2f, -0.4f);
    const auto listenerAhead = Vector3f::unitVector(drawVector(lcg));
    const auto listener1 = listener0 + Vector3f(1.5f, 0.0f, 0.7f);

    SourceInputs sources[kNumSources];
    for (auto i = 0; i < kNumSources; ++i)
    {
        Vector3f position;
        if (i == 2)
            position = probes[6].influence.center + Vector3f(0.2f, 0.1f, 0.3f);
        else if (i == 3)
            position = listenerPoint + Vector3f(0.1f, 0.0f, -0.2f);
        else
            position = drawInterior(lcg, bounds);

        const auto ahead = Vector3f::unitVector(drawVector(lcg));
        sources[i].source = CoordinateSpace3f(ahead, position);

        const float weight = draw01(lcg);
        const float power = draw01(lcg) * 4.0f;
        sources[i].directivity = Directivity(weight, power);

        for (auto j = 0; j < Bands::kNumBands; ++j)
        {
            const float c = draw01(lcg) * 0.05f;
            sources[i].airCoefficients[j] = c;
        }
    }
    const auto movedSource0 = sources[0].source.origin + Vector3f(0.5f, 0.0f, -0.3f);

    for (const auto& managerCase : kManagers)
    {
        const auto type = managerCase.type;
        const std::string prefix = std::string("manager_") + managerCase.name;

        // The reconstructor's band filters and the hybrid reverb estimator's filters pick their block algorithm when
        // they are built: the AVX path is the one the port follows (risk R8).
        srand(managerCase.randSeed);
        Context::sSIMDLevel = SIMDLevel::AVX;
        SimulationManager manager(true, true, false, SceneType::Default, type, 16, 256, 32, kMaxDuration, kMaxOrder, 4,
                                  1, 1, 16, 1, false, Vector3f(0.0f, -1.0f, 0.0f), kSamplingRate, kFrameSize, nullptr,
                                  nullptr, nullptr);
        Context::sSIMDLevel = SIMDLevel::SSE2;
        reseed(static_cast<ReflectionSimulator&>(*managerSimulator(manager)), managerCase.simulatorSeed);

        manager.scene() = scene;
        manager.addProbeBatch(batch);

        shared_ptr<SimulationData> data[kNumSources];
        for (auto i = 0; i < kNumSources; ++i)
        {
            data[i] = ipl::make_shared<SimulationData>(true, false, SceneType::Default, type, 16, kMaxDuration,
                                                       kMaxOrder, kSamplingRate, kFrameSize, nullptr, nullptr);
            auto& s = *data[i];
            const auto& in = sources[i];

            s.directInputs.flags = static_cast<DirectSimulationFlags>(CalcDistanceAttenuation | CalcAirAbsorption |
                                                                      CalcDelay | CalcOcclusion);
            s.directInputs.source = in.source;
            s.directInputs.directivity = in.directivity;
            s.directInputs.airAbsorptionModel = AirAbsorptionModel(in.airCoefficients, nullptr, nullptr);

            s.reflectionInputs.enabled = (i != 5);
            s.reflectionInputs.source = in.source;
            s.reflectionInputs.directivity = in.directivity;
            s.reflectionInputs.baked = (i >= 2);

            switch (i)
            {
            case 0:
                s.directInputs.flags = static_cast<DirectSimulationFlags>(s.directInputs.flags | CalcDirectivity |
                                                                          CalcTransmission);
                s.directInputs.numTransmissionRays = 3;
                s.reflectionInputs.transitionTime = 0.05f;
                break;
            case 1:
                s.directInputs.flags = static_cast<DirectSimulationFlags>(s.directInputs.flags | CalcDirectivity |
                                                                          CalcTransmission);
                s.directInputs.distanceAttenuationModel = DistanceAttenuationModel(2.0f, nullptr, nullptr);
                s.directInputs.occlusionType = OcclusionType::Volumetric;
                s.directInputs.occlusionRadius = 1.0f;
                s.directInputs.numTransmissionRays = 2;
                s.reflectionInputs.distanceAttenuationModel = DistanceAttenuationModel(2.0f, nullptr, nullptr);
                s.reflectionInputs.airAbsorptionModel = AirAbsorptionModel(in.airCoefficients, nullptr, nullptr);
                s.reflectionInputs.reverbScale[0] = 1.2f;
                s.reflectionInputs.reverbScale[1] = 0.8f;
                s.reflectionInputs.transitionTime = 0.04f;
                s.reflectionInputs.overlapFraction = 0.5f;
                break;
            case 2:
                s.reflectionInputs.bakedDataIdentifier = reverbId;
                break;
            case 3:
                s.reflectionInputs.bakedDataIdentifier = listenerId;
                s.reflectionInputs.reverbScale[0] = 0.9f;
                s.reflectionInputs.reverbScale[1] = 1.1f;
                s.reflectionInputs.reverbScale[2] = 1.3f;
                break;
            case 4:
                s.reflectionInputs.bakedDataIdentifier = sourceId;
                break;
            case 5:
                s.directInputs.flags = static_cast<DirectSimulationFlags>(CalcDistanceAttenuation | CalcDirectivity |
                                                                          CalcDelay);
                break;
            }

            manager.addSource(data[i]);
        }
        manager.commit();

        std::vector<float> direct, fields, outputs, irs;

        // Frame 0 simulates; frame 1 changes nothing (energy fields accumulate); frame 2 moves source 0, changes source
        // 1's distance attenuation model, reconstructs with Gaussians over a shorter duration and removes source 5;
        // frame 3 moves the listener, lowers the order and marks source 1's model dirty.
        for (auto frame = 0; frame < 4; ++frame)
        {
            const auto listenerPosition = (frame == 3) ? listener1 : listener0;
            const CoordinateSpace3f listener(listenerAhead, listenerPosition);

            SharedDirectSimulationInputs sharedDirect;
            sharedDirect.listener = listener;
            manager.setSharedDirectInputs(sharedDirect);

            SharedReflectionSimulationInputs sharedReflection;
            sharedReflection.listener = listener;
            sharedReflection.numRays = 256;
            sharedReflection.numBounces = 4;
            sharedReflection.duration = (frame == 2) ? 0.06f : kMaxDuration;
            sharedReflection.order = (frame == 3) ? 0 : 1;
            sharedReflection.irradianceMinDistance = 1.0f;
            sharedReflection.reconstructionType = (frame == 2) ? ReconstructionType::Gaussian
                                                               : ReconstructionType::Linear;
            manager.setSharedReflectionInputs(sharedReflection);

            if (frame == 2)
            {
                const CoordinateSpace3f moved(sources[0].source.ahead, movedSource0);
                data[0]->directInputs.source = moved;
                data[0]->reflectionInputs.source = moved;
                data[1]->reflectionInputs.distanceAttenuationModel = DistanceAttenuationModel(3.0f, nullptr, nullptr);
                manager.removeSource(data[5]);
                manager.commit();
            }
            if (frame == 3)
            {
                data[1]->reflectionInputs.distanceAttenuationModel.dirty = true;
            }

            manager.simulateDirect();
            manager.simulateIndirect();

            for (auto i = 0; i < kNumSources; ++i)
                appendOutputs(*data[i], type, direct, fields, outputs, irs);
        }

        writeMatrix(out / (prefix + "_direct.npy"), 10, direct);
        writeMatrix(out / (prefix + "_fields.npy"), fields.size() / (4 * kNumSources), fields);
        writeMatrix(out / (prefix + "_outputs.npy"), outputs.size() / (4 * kNumSources), outputs);
        if (type != IndirectEffectType::Parametric)
            writeMatrix(out / (prefix + "_irs.npy"), irs.size() / (4 * kNumSources), irs);
    }
}

Register simulation("simulation", generateSimulation);

}
