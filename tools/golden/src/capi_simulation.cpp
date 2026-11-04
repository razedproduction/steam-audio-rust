// Golden data for stage 8c: the C API (phonon.h) for simulators and sources, the reflection effect and mixer, and the
// path effect (api_simulator, api_indirect_effect, api_path_effect, phonon_interfaces.h).
//
// Everything is called through the ipl* functions. Inputs come from Lcg draws in a fixed order, one value per
// statement; crates/phonon-capi/tests/capi_simulation.rs replays the same order.
//
// The C API seeds its reflection simulators with the clock (risk R2), so reflections are simulated in scenes without
// scattering: every bounce is then specular and the results do not depend on the random numbers. Simulations run on one
// thread. Reverb effects (parametric and hybrid reflection effects, a hybrid simulator's estimator) draw their delay
// lengths from rand(): each case that creates them starts with srand(1), the state of a fresh thread, and the test runs
// it on a new thread.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <context.h>
#include <coordinate_space.h>
#include <phonon.h>

#include "generators.h"
#include "npy.h"
#include "scene_inputs.h"

namespace {

using namespace golden;

constexpr float kSentinel = 9.0f;

void check(IPLerror status, const char* what)
{
    if (status != IPL_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(status));
}

IPLContext makeContext()
{
    IPLContextSettings settings{};
    settings.version = STEAMAUDIO_VERSION;
    IPLContext context = nullptr;
    check(iplContextCreate(&settings, &context), "iplContextCreate");
    // Filters and reverbs follow upstream's AVX path, as the port does (risk R8).
    ipl::Context::sSIMDLevel = ipl::SIMDLevel::AVX;
    return context;
}

IPLVector3 toIpl(const ipl::Vector3f& v)
{
    return IPLVector3{v.x(), v.y(), v.z()};
}

// Row-major, as phonon.h documents it: element [i][j] is m(i, j).
IPLMatrix4x4 toIpl(const ipl::Matrix4x4f& m)
{
    IPLMatrix4x4 out{};
    for (auto i = 0; i < 4; ++i)
        for (auto j = 0; j < 4; ++j)
            out.elements[i][j] = m(i, j);
    return out;
}

IPLCoordinateSpace3 toIpl(const ipl::CoordinateSpace3f& c)
{
    return IPLCoordinateSpace3{toIpl(c.right), toIpl(c.up), toIpl(c.ahead), toIpl(c.origin)};
}

ipl::Box meshBounds(const ObjMesh& obj)
{
    ipl::Box bounds;
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

// The transform that maps the unit cube [-0.5, 0.5]^3 onto a box (for probe generation).
IPLMatrix4x4 boxTransform(const ipl::Box& box)
{
    IPLMatrix4x4 m{};
    for (auto i = 0; i < 3; ++i)
    {
        m.elements[i][i] = box.maxCoordinates[i] - box.minCoordinates[i];
        m.elements[i][3] = 0.5f * (box.minCoordinates[i] + box.maxCoordinates[i]);
    }
    m.elements[3][3] = 1.0f;
    return m;
}

// A point in the middle 40% of a box along each axis.
ipl::Vector3f interiorPoint(Lcg& lcg, const ipl::Box& box)
{
    ipl::Vector3f p;
    for (auto i = 0; i < 3; ++i)
    {
        const float t = 0.3f + 0.4f * draw01(lcg);
        p[i] = box.minCoordinates[i] + (box.maxCoordinates[i] - box.minCoordinates[i]) * t;
    }
    return p;
}

// A coordinate space at `origin`, looking along an Lcg direction.
IPLCoordinateSpace3 drawSpace(Lcg& lcg, const ipl::Vector3f& origin)
{
    const auto ahead = ipl::Vector3f::unitVector(drawVector(lcg));
    return toIpl(ipl::CoordinateSpace3f(ahead, origin));
}

IPLStaticMesh createStaticMesh(IPLScene scene, const ObjMesh& obj)
{
    std::vector<IPLVector3> vertices;
    for (const auto& v : obj.vertices)
        vertices.push_back(toIpl(v));
    std::vector<IPLMaterial> materials;
    for (const auto& m : obj.materials)
    {
        IPLMaterial material{};
        for (auto i = 0; i < 3; ++i)
        {
            material.absorption[i] = m.absorption[i];
            material.transmission[i] = m.transmission[i];
        }
        material.scattering = m.scattering;
        materials.push_back(material);
    }
    auto triangles = obj.triangles;
    auto materialIndices = obj.materialIndices;

    IPLStaticMeshSettings settings{};
    settings.numVertices = static_cast<IPLint32>(vertices.size());
    settings.numTriangles = numTriangles(obj);
    settings.numMaterials = static_cast<IPLint32>(materials.size());
    settings.vertices = vertices.data();
    settings.triangles = reinterpret_cast<IPLTriangle*>(triangles.data());
    settings.materialIndices = materialIndices.data();
    settings.materials = materials.data();

    IPLStaticMesh mesh = nullptr;
    check(iplStaticMeshCreate(scene, &settings, &mesh), "iplStaticMeshCreate");
    return mesh;
}

void makeSpecular(ObjMesh& mesh)
{
    for (auto& m : mesh.materials)
        m.scattering = 0.0f;
}

// A default scene holding `obj` and an instance of a sub-scene holding `instanced` (moved by an Lcg transform).
struct SceneSetup
{
    IPLScene scene = nullptr;
    IPLScene subScene = nullptr;
    IPLStaticMesh mesh = nullptr;
    IPLStaticMesh subMesh = nullptr;
    IPLInstancedMesh instance = nullptr;
    ipl::Box bounds;

    void release()
    {
        iplInstancedMeshRelease(&instance);
        iplStaticMeshRelease(&subMesh);
        iplStaticMeshRelease(&mesh);
        iplSceneRelease(&subScene);
        iplSceneRelease(&scene);
    }
};

SceneSetup makeScene(IPLContext context, Lcg& lcg, const ObjMesh& obj, const ObjMesh& instanced)
{
    SceneSetup s;
    s.bounds = meshBounds(obj);

    IPLSceneSettings sceneSettings{};
    sceneSettings.type = IPL_SCENETYPE_DEFAULT;
    check(iplSceneCreate(context, &sceneSettings, &s.scene), "iplSceneCreate");
    check(iplSceneCreate(context, &sceneSettings, &s.subScene), "iplSceneCreate");

    s.mesh = createStaticMesh(s.scene, obj);
    iplStaticMeshAdd(s.mesh, s.scene);
    s.subMesh = createStaticMesh(s.subScene, instanced);
    iplStaticMeshAdd(s.subMesh, s.subScene);
    iplSceneCommit(s.subScene);

    IPLInstancedMeshSettings instanceSettings{};
    instanceSettings.subScene = s.subScene;
    instanceSettings.transform = toIpl(drawTransform(lcg, s.bounds));
    check(iplInstancedMeshCreate(s.scene, &instanceSettings, &s.instance), "iplInstancedMeshCreate");
    iplInstancedMeshAdd(s.instance, s.scene);
    iplSceneCommit(s.scene);
    return s;
}

// Floor probes over a box, in a committed batch.
IPLProbeBatch floorProbes(IPLContext context, IPLScene scene, const ipl::Box& bounds, float spacingFraction)
{
    IPLProbeArray probes = nullptr;
    check(iplProbeArrayCreate(context, &probes), "iplProbeArrayCreate");
    const auto extents = bounds.extents();
    IPLProbeGenerationParams params{};
    params.type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
    params.spacing = spacingFraction * std::max(extents.x(), extents.z());
    params.height = 1.5f;
    params.transform = boxTransform(bounds);
    iplProbeArrayGenerateProbes(probes, scene, &params);

    IPLProbeBatch batch = nullptr;
    check(iplProbeBatchCreate(context, &batch), "iplProbeBatchCreate");
    iplProbeBatchAddProbeArray(batch, probes);
    iplProbeBatchCommit(batch);
    iplProbeArrayRelease(&probes);
    return batch;
}

// Lcg noise in [-1, 1).
void fillNoise(Lcg& lcg, std::vector<float>& samples)
{
    for (auto& x : samples)
        x = lcg.nextSigned();
}

// A buffer of `channels` x `samples`, prefilled with the sentinel, so that untouched channels are checked too.
struct Buffer
{
    std::vector<float> data;
    std::vector<float*> pointers;
    IPLAudioBuffer buffer{};

    Buffer(int channels, int samples)
        : data(static_cast<size_t>(channels) * samples, kSentinel)
    {
        for (auto i = 0; i < channels; ++i)
            pointers.push_back(&data[static_cast<size_t>(i) * samples]);
        buffer.numChannels = channels;
        buffer.numSamples = samples;
        buffer.data = pointers.data();
    }

    void reset() { std::fill(data.begin(), data.end(), kSentinel); }
    void append(std::vector<float>& out) const { out.insert(out.end(), data.begin(), data.end()); }
};

// --------------------------------------------------------------------------------------------------------------------
// Callbacks
// --------------------------------------------------------------------------------------------------------------------

float IPLCALL distanceCallback(IPLfloat32 distance, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return 1.0f / (1.0f + k * distance);
}

float IPLCALL airCallback(IPLfloat32 distance, IPLint32 band, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return 1.0f / (1.0f + k * distance * static_cast<float>(band + 1));
}

float IPLCALL directivityCallback(IPLVector3 direction, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return k + (1.0f - k) * std::fabs(direction.z);
}

float IPLCALL deviationCallback(IPLfloat32 angle, IPLint32 band, void* userData)
{
    const auto k = *static_cast<const float*>(userData);
    return 1.0f / (1.0f + k * angle * static_cast<float>(band + 1));
}

void IPLCALL visCallback(IPLVector3 from, IPLVector3 to, IPLbool occluded, void* userData)
{
    auto& out = *static_cast<std::vector<float>*>(userData);
    out.insert(out.end(), {from.x, from.y, from.z, to.x, to.y, to.z, static_cast<float>(occluded)});
}

IPLSimulationInputs defaultInputs()
{
    IPLSimulationInputs inputs{};
    inputs.distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
    inputs.airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
    inputs.directivity.dipoleWeight = 0.0f;
    inputs.directivity.dipolePower = 0.0f;
    inputs.occlusionType = IPL_OCCLUSIONTYPE_RAYCAST;
    inputs.numOcclusionSamples = 1;
    inputs.reverbScale[0] = inputs.reverbScale[1] = inputs.reverbScale[2] = 1.0f;
    inputs.hybridReverbTransitionTime = 1.0f;
    inputs.hybridReverbOverlapPercent = 0.25f;
    inputs.numTransmissionRays = 1;
    return inputs;
}

IPLSimulationSettings simulationSettings(IPLSimulationFlags flags, IPLReflectionEffectType reflectionType,
                                         int samplingRate)
{
    IPLSimulationSettings settings{};
    settings.flags = flags;
    settings.sceneType = IPL_SCENETYPE_DEFAULT;
    settings.reflectionType = reflectionType;
    settings.maxNumOcclusionSamples = 32;
    settings.maxNumRays = 512;
    settings.numDiffuseSamples = 8;
    settings.maxDuration = 0.25f;
    settings.maxOrder = 1;
    settings.maxNumSources = 4;
    settings.numThreads = 1;
    settings.rayBatchSize = 1;
    settings.numVisSamples = 4;
    settings.samplingRate = samplingRate;
    settings.frameSize = 240;
    return settings;
}

void appendDirect(std::vector<float>& out, const IPLDirectEffectParams& d)
{
    out.push_back(d.distanceAttenuation);
    out.insert(out.end(), d.airAbsorption, d.airAbsorption + 3);
    out.push_back(d.directivity);
    out.push_back(d.occlusion);
    out.insert(out.end(), d.transmission, d.transmission + 3);
}

// --------------------------------------------------------------------------------------------------------------------
// Struct layouts
// --------------------------------------------------------------------------------------------------------------------

#define SIZE(T) layout.push_back(static_cast<int32_t>(sizeof(T)))
#define FIELD(T, f) layout.push_back(static_cast<int32_t>(offsetof(T, f)))

void generateLayout(const std::filesystem::path& out)
{
    std::vector<int32_t> layout;

    SIZE(IPLReflectionEffectType); SIZE(IPLSimulationFlags); SIZE(IPLDirectSimulationFlags);
    SIZE(IPLDeviationModelType); SIZE(IPLOcclusionType);

    SIZE(IPLReflectionEffectSettings); FIELD(IPLReflectionEffectSettings, type);
    FIELD(IPLReflectionEffectSettings, irSize); FIELD(IPLReflectionEffectSettings, numChannels);
    SIZE(IPLReflectionEffectParams); FIELD(IPLReflectionEffectParams, type); FIELD(IPLReflectionEffectParams, ir);
    FIELD(IPLReflectionEffectParams, reverbTimes); FIELD(IPLReflectionEffectParams, eq);
    FIELD(IPLReflectionEffectParams, delay); FIELD(IPLReflectionEffectParams, numChannels);
    FIELD(IPLReflectionEffectParams, irSize); FIELD(IPLReflectionEffectParams, tanDevice);
    FIELD(IPLReflectionEffectParams, tanSlot);
    SIZE(IPLPathEffectSettings); FIELD(IPLPathEffectSettings, maxOrder); FIELD(IPLPathEffectSettings, spatialize);
    FIELD(IPLPathEffectSettings, speakerLayout); FIELD(IPLPathEffectSettings, hrtf);
    SIZE(IPLPathEffectParams); FIELD(IPLPathEffectParams, eqCoeffs); FIELD(IPLPathEffectParams, shCoeffs);
    FIELD(IPLPathEffectParams, order); FIELD(IPLPathEffectParams, binaural); FIELD(IPLPathEffectParams, hrtf);
    FIELD(IPLPathEffectParams, listener); FIELD(IPLPathEffectParams, normalizeEQ);

    SIZE(IPLDeviationModel); FIELD(IPLDeviationModel, type); FIELD(IPLDeviationModel, callback);
    FIELD(IPLDeviationModel, userData);
    SIZE(IPLSimulationSettings); FIELD(IPLSimulationSettings, flags); FIELD(IPLSimulationSettings, sceneType);
    FIELD(IPLSimulationSettings, reflectionType); FIELD(IPLSimulationSettings, maxNumOcclusionSamples);
    FIELD(IPLSimulationSettings, maxNumRays); FIELD(IPLSimulationSettings, numDiffuseSamples);
    FIELD(IPLSimulationSettings, maxDuration); FIELD(IPLSimulationSettings, maxOrder);
    FIELD(IPLSimulationSettings, maxNumSources); FIELD(IPLSimulationSettings, numThreads);
    FIELD(IPLSimulationSettings, rayBatchSize); FIELD(IPLSimulationSettings, numVisSamples);
    FIELD(IPLSimulationSettings, samplingRate); FIELD(IPLSimulationSettings, frameSize);
    FIELD(IPLSimulationSettings, openCLDevice); FIELD(IPLSimulationSettings, radeonRaysDevice);
    FIELD(IPLSimulationSettings, tanDevice);
    SIZE(IPLSourceSettings); FIELD(IPLSourceSettings, flags);
    SIZE(IPLSimulationInputs); FIELD(IPLSimulationInputs, flags); FIELD(IPLSimulationInputs, directFlags);
    FIELD(IPLSimulationInputs, source); FIELD(IPLSimulationInputs, distanceAttenuationModel);
    FIELD(IPLSimulationInputs, airAbsorptionModel); FIELD(IPLSimulationInputs, directivity);
    FIELD(IPLSimulationInputs, occlusionType); FIELD(IPLSimulationInputs, occlusionRadius);
    FIELD(IPLSimulationInputs, numOcclusionSamples); FIELD(IPLSimulationInputs, reverbScale);
    FIELD(IPLSimulationInputs, hybridReverbTransitionTime); FIELD(IPLSimulationInputs, hybridReverbOverlapPercent);
    FIELD(IPLSimulationInputs, baked); FIELD(IPLSimulationInputs, bakedDataIdentifier);
    FIELD(IPLSimulationInputs, pathingProbes); FIELD(IPLSimulationInputs, visRadius);
    FIELD(IPLSimulationInputs, visThreshold); FIELD(IPLSimulationInputs, visRange);
    FIELD(IPLSimulationInputs, pathingOrder); FIELD(IPLSimulationInputs, enableValidation);
    FIELD(IPLSimulationInputs, findAlternatePaths); FIELD(IPLSimulationInputs, numTransmissionRays);
    FIELD(IPLSimulationInputs, deviationModel);
    SIZE(IPLSimulationSharedInputs); FIELD(IPLSimulationSharedInputs, listener);
    FIELD(IPLSimulationSharedInputs, numRays); FIELD(IPLSimulationSharedInputs, numBounces);
    FIELD(IPLSimulationSharedInputs, duration); FIELD(IPLSimulationSharedInputs, order);
    FIELD(IPLSimulationSharedInputs, irradianceMinDistance); FIELD(IPLSimulationSharedInputs, pathingVisCallback);
    FIELD(IPLSimulationSharedInputs, pathingUserData);
    SIZE(IPLSimulationOutputs); FIELD(IPLSimulationOutputs, direct); FIELD(IPLSimulationOutputs, reflections);
    FIELD(IPLSimulationOutputs, pathing);

    writeNpy(out / "layout.npy", layout);
}

// --------------------------------------------------------------------------------------------------------------------
// Statuses
// --------------------------------------------------------------------------------------------------------------------

void generateStatuses(const std::filesystem::path& out)
{
    auto context = makeContext();
    std::vector<int32_t> statuses;

    auto settings = simulationSettings(IPL_SIMULATIONFLAGS_DIRECT, IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, 24000);
    IPLSimulator simulator = nullptr;
    statuses.push_back(iplSimulatorCreate(context, &settings, &simulator));
    IPLSimulator unused = nullptr;
    statuses.push_back(iplSimulatorCreate(context, nullptr, &unused));
    statuses.push_back(iplSimulatorCreate(nullptr, &settings, &unused));

    IPLSourceSettings sourceSettings{IPL_SIMULATIONFLAGS_DIRECT};
    IPLSource source = nullptr;
    statuses.push_back(iplSourceCreate(simulator, &sourceSettings, &source));
    IPLSource unusedSource = nullptr;
    statuses.push_back(iplSourceCreate(simulator, nullptr, &unusedSource));
    statuses.push_back(iplSourceCreate(nullptr, &sourceSettings, &unusedSource));

    IPLAudioSettings audio{24000, 240};
    IPLAudioSettings badAudio{0, 240};
    IPLReflectionEffectSettings effectSettings{IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, 1000, 4};
    IPLReflectionEffect effect = nullptr;
    statuses.push_back(iplReflectionEffectCreate(context, &audio, &effectSettings, &effect));
    IPLReflectionEffect unusedEffect = nullptr;
    statuses.push_back(iplReflectionEffectCreate(context, &badAudio, &effectSettings, &unusedEffect));
    statuses.push_back(iplReflectionEffectCreate(context, &audio, nullptr, &unusedEffect));

    IPLReflectionMixer mixer = nullptr;
    statuses.push_back(iplReflectionMixerCreate(context, &audio, &effectSettings, &mixer));
    IPLReflectionMixer unusedMixer = nullptr;
    IPLAudioSettings badFrame{24000, 0};
    statuses.push_back(iplReflectionMixerCreate(context, &badFrame, &effectSettings, &unusedMixer));

    IPLPathEffectSettings pathSettings{};
    IPLPathEffect pathEffect = nullptr;
    IPLPathEffect unusedPath = nullptr;
    for (auto maxOrder : {3, 4, -1, 0})
    {
        pathSettings.maxOrder = maxOrder;
        auto status = iplPathEffectCreate(context, &audio, &pathSettings, (maxOrder == 3) ? &pathEffect : &unusedPath);
        statuses.push_back(status);
        if (maxOrder == 0 && status == IPL_STATUS_SUCCESS)
            iplPathEffectRelease(&unusedPath);
    }

    // Null handles: neutral results.
    statuses.push_back(iplReflectionEffectGetTailSize(nullptr));
    statuses.push_back(iplPathEffectGetTailSize(nullptr));

    writeNpy(out / "statuses.npy", statuses);

    iplPathEffectRelease(&pathEffect);
    iplReflectionMixerRelease(&mixer);
    iplReflectionEffectRelease(&effect);
    iplSourceRelease(&source);
    iplSimulatorRelease(&simulator);
    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Direct simulation
// --------------------------------------------------------------------------------------------------------------------

constexpr int kNumDirectSources = 5;

void generateDirect(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(839);

    auto room = loadObj("boxroom");
    assignMaterials(lcg, room);
    auto box = loadObj("smallbox");
    auto setup = makeScene(context, lcg, room, box);

    auto settings = simulationSettings(IPL_SIMULATIONFLAGS_DIRECT, IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, 24000);
    IPLSimulator simulator = nullptr;
    check(iplSimulatorCreate(context, &settings, &simulator), "iplSimulatorCreate");

    float distanceK = 0.3f + draw01(lcg);
    float airK = 0.05f + 0.1f * draw01(lcg);
    float directivityK = 0.2f + 0.5f * draw01(lcg);

    IPLSource sources[kNumDirectSources];
    IPLSimulationInputs inputs[kNumDirectSources];
    for (auto i = 0; i < kNumDirectSources; ++i)
    {
        IPLSourceSettings sourceSettings{IPL_SIMULATIONFLAGS_DIRECT};
        check(iplSourceCreate(simulator, &sourceSettings, &sources[i]), "iplSourceCreate");
        iplSourceAdd(sources[i], simulator);
        inputs[i] = defaultInputs();
    }
    iplSimulatorCommit(simulator);

    // Draws every source's inputs. The model types cycle through the default, explicit and callback models.
    auto drawInputs = [&](int frame)
    {
        for (auto i = 0; i < kNumDirectSources; ++i)
        {
            auto& in = inputs[i];
            const auto flagBits = lcg.nextU32();
            in.directFlags = static_cast<IPLDirectSimulationFlags>((i == 0) ? 31 : (flagBits & 31));
            const auto origin = drawPoint(lcg, setup.bounds);
            in.source = drawSpace(lcg, origin);

            const auto type = (i + frame) % 3;
            in.distanceAttenuationModel.type = static_cast<IPLDistanceAttenuationModelType>(type);
            in.distanceAttenuationModel.minDistance = 0.5f + draw01(lcg);
            in.distanceAttenuationModel.callback = distanceCallback;
            in.distanceAttenuationModel.userData = &distanceK;

            in.airAbsorptionModel.type = static_cast<IPLAirAbsorptionModelType>((i + 2 * frame) % 3);
            for (auto b = 0; b < 3; ++b)
                in.airAbsorptionModel.coefficients[b] = 0.01f + 0.2f * draw01(lcg);
            in.airAbsorptionModel.callback = airCallback;
            in.airAbsorptionModel.userData = &airK;

            in.directivity.dipoleWeight = draw01(lcg);
            in.directivity.dipolePower = 1.0f + 3.0f * draw01(lcg);
            in.directivity.callback = (i == 4) ? directivityCallback : nullptr;
            in.directivity.userData = &directivityK;

            in.occlusionType = static_cast<IPLOcclusionType>((i + frame) % 2);
            in.occlusionRadius = 0.1f + 0.5f * draw01(lcg);
            in.numOcclusionSamples = 1 + static_cast<int>(lcg.nextU32() % 40);
            in.numTransmissionRays = 1 + static_cast<int>(lcg.nextU32() % 6);
        }
    };

    std::vector<float> outputs;
    auto run = [&]()
    {
        IPLSimulationSharedInputs shared{};
        shared.listener = drawSpace(lcg, interiorPoint(lcg, setup.bounds));
        iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_DIRECT, &shared);
        iplSimulatorRunDirect(simulator);
        for (auto source : sources)
        {
            IPLSimulationOutputs o{};
            iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT, &o);
            appendDirect(outputs, o.direct);
        }
    };

    // Frame 0: no scene yet (occlusion and transmission are 1).
    drawInputs(0);
    for (auto i = 0; i < kNumDirectSources; ++i)
        iplSourceSetInputs(sources[i], IPL_SIMULATIONFLAGS_DIRECT, &inputs[i]);
    run();

    iplSimulatorSetScene(simulator, setup.scene);
    iplSimulatorCommit(simulator);
    for (auto frame = 1; frame <= 2; ++frame)
    {
        drawInputs(frame);
        for (auto i = 0; i < kNumDirectSources; ++i)
            iplSourceSetInputs(sources[i], IPL_SIMULATIONFLAGS_DIRECT, &inputs[i]);
        run();
    }

    // Frame 3: unknown model and occlusion types keep the previous ones; source 4 is removed and keeps its outputs;
    // only reflection inputs for source 3 (its direct inputs stay).
    drawInputs(3);
    inputs[0].distanceAttenuationModel.type = static_cast<IPLDistanceAttenuationModelType>(7);
    inputs[0].airAbsorptionModel.type = static_cast<IPLAirAbsorptionModelType>(7);
    inputs[1].occlusionType = static_cast<IPLOcclusionType>(7);
    for (auto i = 0; i < 3; ++i)
        iplSourceSetInputs(sources[i], IPL_SIMULATIONFLAGS_DIRECT, &inputs[i]);
    iplSourceSetInputs(sources[3], IPL_SIMULATIONFLAGS_REFLECTIONS, &inputs[3]);
    iplSourceSetInputs(sources[4], IPL_SIMULATIONFLAGS_DIRECT, &inputs[4]);
    iplSourceRemove(sources[4], simulator);
    iplSimulatorCommit(simulator);
    run();

    // Frame 4: the instance moves (the simulator sees the scene's changes once committed).
    iplInstancedMeshUpdateTransform(setup.instance, setup.scene, toIpl(drawTransform(lcg, setup.bounds)));
    iplSceneCommit(setup.scene);
    run();

    writeNpy(out / "direct.npy", outputs);

    for (auto& source : sources)
        iplSourceRelease(&source);
    iplSimulatorRelease(&simulator);
    setup.release();
    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Reflections
// --------------------------------------------------------------------------------------------------------------------

constexpr int kNumReflectionSources = 4;
constexpr int kReflectionRate = 24000;
constexpr int kFrameSize = 240;

void generateReflections(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(853);

    auto room = loadObj("boxroom");
    assignMaterials(lcg, room);
    makeSpecular(room);
    auto box = loadObj("smallbox");
    makeSpecular(box);
    auto setup = makeScene(context, lcg, room, box);

    // A batch of floor probes with a baked reverb layer (convolution and parametric), for the baked source.
    auto batch = floorProbes(context, setup.scene, setup.bounds, 0.25f);
    IPLBakedDataIdentifier reverbId{IPL_BAKEDDATATYPE_REFLECTIONS, IPL_BAKEDDATAVARIATION_REVERB, {}};
    IPLReflectionsBakeParams bake{};
    bake.scene = setup.scene;
    bake.probeBatch = batch;
    bake.sceneType = IPL_SCENETYPE_DEFAULT;
    bake.identifier = reverbId;
    bake.bakeFlags = static_cast<IPLReflectionsBakeFlags>(IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION |
                                                          IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC);
    bake.numRays = 256;
    bake.numDiffuseSamples = 8;
    bake.numBounces = 4;
    bake.simulatedDuration = 0.25f;
    bake.savedDuration = 0.25f;
    bake.order = 1;
    bake.numThreads = 1;
    bake.rayBatchSize = 1;
    bake.irradianceMinDistance = 1.0f;
    bake.bakeBatchSize = 1;
    iplReflectionsBakerBake(context, &bake, nullptr, nullptr);

    float distanceK = 0.2f + draw01(lcg);
    const IPLReflectionEffectType types[3] = {IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, IPL_REFLECTIONEFFECTTYPE_PARAMETRIC,
                                              IPL_REFLECTIONEFFECTTYPE_HYBRID};
    const char* names[3] = {"convolution", "parametric", "hybrid"};

    srand(1);
    for (auto t = 0; t < 3; ++t)
    {
        const auto type = types[t];
        auto settings = simulationSettings(IPL_SIMULATIONFLAGS_REFLECTIONS, type, kReflectionRate);
        IPLSimulator simulator = nullptr;
        check(iplSimulatorCreate(context, &settings, &simulator), "iplSimulatorCreate");
        iplSimulatorSetScene(simulator, setup.scene);
        iplSimulatorAddProbeBatch(simulator, batch);

        IPLSource sources[kNumReflectionSources];
        IPLSimulationInputs inputs[kNumReflectionSources];
        for (auto i = 0; i < kNumReflectionSources; ++i)
        {
            IPLSourceSettings sourceSettings{IPL_SIMULATIONFLAGS_REFLECTIONS};
            check(iplSourceCreate(simulator, &sourceSettings, &sources[i]), "iplSourceCreate");
            iplSourceAdd(sources[i], simulator);
            inputs[i] = defaultInputs();
            inputs[i].flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
            inputs[i].source = drawSpace(lcg, interiorPoint(lcg, setup.bounds));
        }
        iplSimulatorCommit(simulator);

        // Source 1: inverse distance, exponential air absorption, a dipole, a reverb scale and hybrid settings.
        inputs[1].distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
        inputs[1].distanceAttenuationModel.minDistance = 2.0f;
        inputs[1].airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_EXPONENTIAL;
        for (auto b = 0; b < 3; ++b)
            inputs[1].airAbsorptionModel.coefficients[b] = 0.02f + 0.1f * draw01(lcg);
        inputs[1].directivity.dipoleWeight = 0.5f;
        inputs[1].directivity.dipolePower = 2.0f;
        for (auto b = 0; b < 3; ++b)
            inputs[1].reverbScale[b] = 0.7f + 0.6f * draw01(lcg);
        inputs[1].hybridReverbTransitionTime = 0.1f;
        inputs[1].hybridReverbOverlapPercent = 0.5f;
        // Source 2: baked reverb.
        inputs[2].baked = IPL_TRUE;
        inputs[2].bakedDataIdentifier = reverbId;
        // Source 3: a callback distance model.
        inputs[3].distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        inputs[3].distanceAttenuationModel.callback = distanceCallback;
        inputs[3].distanceAttenuationModel.userData = &distanceK;

        // One effect per source; with convolution, sources 0 and 1 go through a mixer. Convolution and hybrid effects
        // are mono, because the reference build crashes convolving more than one channel (risk R12); they take
        // channel 0 of the simulator's order-1 IRs.
        IPLAudioSettings audio{kReflectionRate, kFrameSize};
        const auto irSize = static_cast<IPLint32>(std::ceil(settings.maxDuration * kReflectionRate));
        IPLReflectionEffectSettings effectSettings{type, irSize, 1};
        const int outChannels = (type == IPL_REFLECTIONEFFECTTYPE_PARAMETRIC) ? 4 : 1;
        IPLReflectionEffect effects[kNumReflectionSources];
        for (auto& effect : effects)
            check(iplReflectionEffectCreate(context, &audio, &effectSettings, &effect), "iplReflectionEffectCreate");
        IPLReflectionMixer mixer = nullptr;
        check(iplReflectionMixerCreate(context, &audio, &effectSettings, &mixer), "iplReflectionMixerCreate");
        const bool useMixer = (type == IPL_REFLECTIONEFFECTTYPE_CONVOLUTION);

        std::vector<float> simOutputs;
        std::vector<float> audioOut;
        std::vector<int32_t> states;
        std::vector<float> input(kFrameSize);
        IPLAudioBuffer inBuffer{1, kFrameSize, nullptr};
        float* inPointer = input.data();
        inBuffer.data = &inPointer;
        Buffer outBuffer(outChannels, kFrameSize);
        Buffer mixBuffer(1, kFrameSize);

        IPLSimulationOutputs outputs[kNumReflectionSources];
        auto listener = drawSpace(lcg, interiorPoint(lcg, setup.bounds));
        for (auto frame = 0; frame < 3; ++frame)
        {
            // Frame 1 changes nothing (the real-time fields are averaged with frame 0's). In frame 2 the listener and
            // source 0 move, source 3's model is marked dirty, and effect 1 is reset.
            if (frame == 2)
            {
                listener = drawSpace(lcg, interiorPoint(lcg, setup.bounds));
                inputs[0].source = drawSpace(lcg, interiorPoint(lcg, setup.bounds));
                inputs[3].distanceAttenuationModel.dirty = IPL_TRUE;
                iplReflectionEffectReset(effects[1]);
            }
            for (auto i = 0; i < kNumReflectionSources; ++i)
                iplSourceSetInputs(sources[i], IPL_SIMULATIONFLAGS_REFLECTIONS, &inputs[i]);

            IPLSimulationSharedInputs shared{};
            shared.listener = listener;
            shared.numRays = 512;
            shared.numBounces = 8;
            shared.duration = 0.25f;
            shared.order = 1;
            shared.irradianceMinDistance = 1.0f;
            iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_REFLECTIONS, &shared);
            iplSimulatorRunReflections(simulator);

            for (auto i = 0; i < kNumReflectionSources; ++i)
            {
                outputs[i] = IPLSimulationOutputs{};
                iplSourceGetOutputs(sources[i], IPL_SIMULATIONFLAGS_REFLECTIONS, &outputs[i]);
                // Upstream leaves the hybrid EQ and delay uninitialized unless the type is hybrid, and the IR size unless
                // it is convolution or hybrid.
                const auto& r = outputs[i].reflections;
                simOutputs.insert(simOutputs.end(), r.reverbTimes, r.reverbTimes + 3);
                if (type == IPL_REFLECTIONEFFECTTYPE_HYBRID)
                {
                    simOutputs.insert(simOutputs.end(), r.eq, r.eq + 3);
                    simOutputs.push_back(static_cast<float>(r.delay));
                }
                if (type != IPL_REFLECTIONEFFECTTYPE_PARAMETRIC)
                {
                    simOutputs.push_back(static_cast<float>(r.numChannels));
                    simOutputs.push_back(static_cast<float>(r.irSize));
                }
            }

            // Two audio frames per simulation; the IR published by the simulation crossfades in during the first.
            for (auto a = 0; a < 2; ++a)
            {
                fillNoise(lcg, input);
                for (auto i = 0; i < kNumReflectionSources; ++i)
                {
                    auto params = outputs[i].reflections;
                    params.type = type;
                    params.numChannels = 1;
                    const bool toMixer = useMixer && i < 2;
                    outBuffer.reset();
                    states.push_back(iplReflectionEffectApply(effects[i], &params, &inBuffer, &outBuffer.buffer,
                                                              toMixer ? mixer : nullptr));
                    if (!toMixer)
                        outBuffer.append(audioOut);
                }
                if (useMixer)
                {
                    auto params = outputs[0].reflections;
                    params.numChannels = 1;
                    mixBuffer.reset();
                    states.push_back(iplReflectionMixerApply(mixer, &params, &mixBuffer.buffer));
                    mixBuffer.append(audioOut);
                }
            }
        }

        // Tails.
        for (auto k = 0; k < 2; ++k)
        {
            for (auto i = 0; i < kNumReflectionSources; ++i)
            {
                const bool toMixer = useMixer && i < 2;
                states.push_back(iplReflectionEffectGetTailSize(effects[i]));
                outBuffer.reset();
                states.push_back(iplReflectionEffectGetTail(effects[i], &outBuffer.buffer, toMixer ? mixer : nullptr));
                if (!toMixer)
                    outBuffer.append(audioOut);
            }
            if (useMixer)
            {
                auto params = outputs[0].reflections;
                params.numChannels = 1;
                mixBuffer.reset();
                states.push_back(iplReflectionMixerApply(mixer, &params, &mixBuffer.buffer));
                mixBuffer.append(audioOut);
            }
        }

        const std::string name = names[t];
        writeNpy(out / ("reflections_" + name + ".npy"), simOutputs);
        writeNpy(out / ("reflections_" + name + "_audio.npy"), audioOut);
        writeNpy(out / ("reflections_" + name + "_states.npy"), states);

        iplReflectionMixerRelease(&mixer);
        for (auto& effect : effects)
            iplReflectionEffectRelease(&effect);
        for (auto& source : sources)
            iplSourceRelease(&source);
        iplSimulatorRelease(&simulator);
    }

    // Effects without an IR: convolution outputs silence, hybrid only the parametric part; a parametric mixer does
    // nothing.
    {
        IPLAudioSettings audio{kReflectionRate, kFrameSize};
        std::vector<float> audioOut;
        std::vector<int32_t> states;
        std::vector<float> input(kFrameSize);
        IPLAudioBuffer inBuffer{1, kFrameSize, nullptr};
        float* inPointer = input.data();
        inBuffer.data = &inPointer;
        Buffer outBuffer(4, kFrameSize);

        IPLReflectionEffect effects[2];
        IPLReflectionEffectSettings convolution{IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, 1000, 4};
        IPLReflectionEffectSettings hybrid{IPL_REFLECTIONEFFECTTYPE_HYBRID, 1000, 4};
        check(iplReflectionEffectCreate(context, &audio, &convolution, &effects[0]), "iplReflectionEffectCreate");
        check(iplReflectionEffectCreate(context, &audio, &hybrid, &effects[1]), "iplReflectionEffectCreate");
        IPLReflectionMixer parametricMixer = nullptr;
        IPLReflectionEffectSettings parametric{IPL_REFLECTIONEFFECTTYPE_PARAMETRIC, 0, 0};
        check(iplReflectionMixerCreate(context, &audio, &parametric, &parametricMixer), "iplReflectionMixerCreate");

        IPLReflectionEffectParams params{};
        for (auto b = 0; b < 3; ++b)
            params.reverbTimes[b] = 0.3f + draw01(lcg);
        for (auto b = 0; b < 3; ++b)
            params.eq[b] = 0.2f + draw01(lcg);
        params.delay = 100;
        params.numChannels = 4;
        params.irSize = 1000;
        for (auto a = 0; a < 3; ++a)
        {
            fillNoise(lcg, input);
            for (auto effect : effects)
            {
                outBuffer.reset();
                states.push_back(iplReflectionEffectApply(effect, &params, &inBuffer, &outBuffer.buffer, nullptr));
                outBuffer.append(audioOut);
            }
            outBuffer.reset();
            states.push_back(iplReflectionMixerApply(parametricMixer, &params, &outBuffer.buffer));
            outBuffer.append(audioOut);
        }
        writeNpy(out / "reflections_no_ir_audio.npy", audioOut);
        writeNpy(out / "reflections_no_ir_states.npy", states);

        iplReflectionMixerRelease(&parametricMixer);
        for (auto& effect : effects)
            iplReflectionEffectRelease(&effect);
    }

    iplProbeBatchRelease(&batch);
    setup.release();
    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Pathing
// --------------------------------------------------------------------------------------------------------------------

constexpr int kNumPathSources = 3;
constexpr int kPathRate = 48000;

void generatePathing(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(857);

    auto room = loadObj("simplescene");
    assignMaterials(lcg, room);
    auto box = loadObj("smallbox");
    auto setup = makeScene(context, lcg, room, box);

    auto batch = floorProbes(context, setup.scene, setup.bounds, 0.12f);
    IPLBakedDataIdentifier pathId{IPL_BAKEDDATATYPE_PATHING, IPL_BAKEDDATAVARIATION_DYNAMIC, {}};
    IPLPathBakeParams pathParams{};
    pathParams.scene = setup.scene;
    pathParams.probeBatch = batch;
    pathParams.identifier = pathId;
    pathParams.numSamples = 1;
    pathParams.radius = 0.5f;
    pathParams.threshold = 0.5f;
    pathParams.visRange = 1000.0f;
    pathParams.pathRange = 1000.0f;
    pathParams.numThreads = 1;
    iplPathBakerBake(context, &pathParams, nullptr, nullptr);

    // After the bake, the instance becomes a wall 2 m thick, 10 m high and 30 m long across the middle of simplescene
    // (as in path_simulation.cpp), which blocks the direct paths between its sides and some baked ones.
    {
        ipl::Matrix4x4f wall;
        for (auto i = 0; i < 4; ++i)
            for (auto j = 0; j < 4; ++j)
                wall(i, j) = 0.0f;
        wall(0, 0) = 2.5f;
        wall(1, 1) = 12.5f;
        wall(2, 2) = 37.5f;
        wall(0, 3) = 13.5f;
        wall(2, 3) = -8.6f;
        wall(3, 3) = 1.0f;
        iplInstancedMeshUpdateTransform(setup.instance, setup.scene, toIpl(wall));
        iplSceneCommit(setup.scene);
    }

    // A point 2.5 to 4.5 m above the bottom of the bounds, within the fraction [x0, x0 + w] of them along x.
    auto drawInterior = [&](float x0, float w)
    {
        const float x = draw01(lcg);
        const float y = draw01(lcg);
        const float z = draw01(lcg);
        const auto& lo = setup.bounds.minCoordinates;
        const auto& hi = setup.bounds.maxCoordinates;
        return ipl::Vector3f(lo.x() + (hi.x() - lo.x()) * (x0 + w * x), lo.y() + 2.5f + 2.0f * y,
                             lo.z() + (hi.z() - lo.z()) * z);
    };

    auto settings = simulationSettings(IPL_SIMULATIONFLAGS_PATHING, IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, kPathRate);
    settings.maxOrder = 2;
    IPLSimulator simulator = nullptr;
    check(iplSimulatorCreate(context, &settings, &simulator), "iplSimulatorCreate");
    iplSimulatorSetScene(simulator, setup.scene);
    iplSimulatorAddProbeBatch(simulator, batch);

    float distanceK = 0.2f + draw01(lcg);
    float deviationK = 0.5f + draw01(lcg);
    IPLDeviationModel deviation{IPL_DEVIATIONTYPE_CALLBACK, deviationCallback, &deviationK};

    IPLSource sources[kNumPathSources];
    IPLSimulationInputs inputs[kNumPathSources];
    for (auto i = 0; i < kNumPathSources; ++i)
    {
        IPLSourceSettings sourceSettings{IPL_SIMULATIONFLAGS_PATHING};
        check(iplSourceCreate(simulator, &sourceSettings, &sources[i]), "iplSourceCreate");
        iplSourceAdd(sources[i], simulator);
        auto& in = inputs[i];
        in = defaultInputs();
        in.flags = IPL_SIMULATIONFLAGS_PATHING;
        in.pathingProbes = batch;
        in.visRadius = 0.5f;
        in.visThreshold = 0.5f;
        in.visRange = 1000.0f;
        in.pathingOrder = (i == 2) ? 0 : i + 1;
        in.enableValidation = (i != 0) ? IPL_TRUE : IPL_FALSE;
        in.findAlternatePaths = (i == 2) ? IPL_TRUE : IPL_FALSE;
    }
    inputs[1].distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    inputs[1].distanceAttenuationModel.minDistance = 1.5f;
    inputs[1].deviationModel = &deviation;
    inputs[2].distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
    inputs[2].distanceAttenuationModel.callback = distanceCallback;
    inputs[2].distanceAttenuationModel.userData = &distanceK;
    iplSimulatorCommit(simulator);

    std::vector<float> visRays;
    std::vector<float> simOutputs;
    std::vector<std::vector<float>> eqs(kNumPathSources);
    std::vector<std::vector<float>> shs(kNumPathSources);
    std::vector<IPLCoordinateSpace3> listeners;

    for (auto frame = 0; frame < 2; ++frame)
    {
        // Sources left of the wall (in frame 1, source 2 anywhere), the listener right of it.
        for (auto i = 0; i < kNumPathSources; ++i)
        {
            const auto origin = (frame == 1 && i == 2) ? drawInterior(0.0f, 1.0f) : drawInterior(0.0f, 0.45f);
            inputs[i].source = drawSpace(lcg, origin);
            iplSourceSetInputs(sources[i], IPL_SIMULATIONFLAGS_PATHING, &inputs[i]);
        }
        IPLSimulationSharedInputs shared{};
        shared.listener = drawSpace(lcg, drawInterior(0.55f, 0.45f));
        shared.pathingVisCallback = visCallback;
        shared.pathingUserData = &visRays;
        listeners.push_back(shared.listener);
        iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_PATHING, &shared);
        iplSimulatorRunPathing(simulator);

        for (auto i = 0; i < kNumPathSources; ++i)
        {
            IPLSimulationOutputs o{};
            iplSourceGetOutputs(sources[i], IPL_SIMULATIONFLAGS_PATHING, &o);
            simOutputs.insert(simOutputs.end(), o.pathing.eqCoeffs, o.pathing.eqCoeffs + 3);
            simOutputs.insert(simOutputs.end(), o.pathing.shCoeffs, o.pathing.shCoeffs + 9);
            eqs[i].insert(eqs[i].end(), o.pathing.eqCoeffs, o.pathing.eqCoeffs + 3);
            shs[i].insert(shs[i].end(), o.pathing.shCoeffs, o.pathing.shCoeffs + 9);
        }
    }
    writeNpy(out / "pathing.npy", simOutputs);
    writeNpy(out / "pathing_vis_rays.npy", visRays);

    // Path effects driven by sources 0 and 1 of each frame: unspatialized order 2, panned to stereo with a normalized
    // EQ, and binaural with the default HRTF.
    IPLAudioSettings audio{kPathRate, kFrameSize};
    IPLHRTFSettings hrtfSettings{};
    hrtfSettings.type = IPL_HRTFTYPE_DEFAULT;
    hrtfSettings.volume = 1.0f;
    IPLHRTF hrtf = nullptr;
    check(iplHRTFCreate(context, &audio, &hrtfSettings, &hrtf), "iplHRTFCreate");

    IPLPathEffect effects[3];
    IPLPathEffectSettings effectSettings{};
    effectSettings.maxOrder = 2;
    check(iplPathEffectCreate(context, &audio, &effectSettings, &effects[0]), "iplPathEffectCreate");
    effectSettings.spatialize = IPL_TRUE;
    effectSettings.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
    effectSettings.hrtf = hrtf;
    check(iplPathEffectCreate(context, &audio, &effectSettings, &effects[1]), "iplPathEffectCreate");
    check(iplPathEffectCreate(context, &audio, &effectSettings, &effects[2]), "iplPathEffectCreate");
    const int outChannels[3] = {9, 2, 2};

    std::vector<float> audioOut;
    std::vector<int32_t> states;
    std::vector<float> input(kFrameSize);
    IPLAudioBuffer inBuffer{1, kFrameSize, nullptr};
    float* inPointer = input.data();
    inBuffer.data = &inPointer;

    for (auto frame = 0; frame < 2; ++frame)
    {
        for (auto a = 0; a < 2; ++a)
        {
            fillNoise(lcg, input);
            for (auto e = 0; e < 3; ++e)
            {
                const auto s = (e == 1) ? 1 : 0;
                IPLPathEffectParams params{};
                std::copy_n(&eqs[s][3 * frame], 3, params.eqCoeffs);
                params.shCoeffs = &shs[s][9 * frame];
                params.order = (e == 0) ? 2 : 1;
                params.binaural = (e == 2) ? IPL_TRUE : IPL_FALSE;
                params.hrtf = hrtf;
                params.listener = listeners[frame];
                params.normalizeEQ = (e == 1) ? IPL_TRUE : IPL_FALSE;
                Buffer outBuffer(outChannels[e], kFrameSize);
                states.push_back(iplPathEffectApply(effects[e], &params, &inBuffer, &outBuffer.buffer));
                outBuffer.append(audioOut);
            }
        }
    }
    for (auto k = 0; k < 2; ++k)
    {
        for (auto e = 0; e < 3; ++e)
        {
            states.push_back(iplPathEffectGetTailSize(effects[e]));
            Buffer outBuffer(outChannels[e], kFrameSize);
            states.push_back(iplPathEffectGetTail(effects[e], &outBuffer.buffer));
            outBuffer.append(audioOut);
        }
    }
    writeNpy(out / "path_effect_audio.npy", audioOut);
    writeNpy(out / "path_effect_states.npy", states);

    for (auto& effect : effects)
        iplPathEffectRelease(&effect);
    iplHRTFRelease(&hrtf);
    for (auto& source : sources)
        iplSourceRelease(&source);
    iplSimulatorRelease(&simulator);
    iplProbeBatchRelease(&batch);
    setup.release();
    iplContextRelease(&context);
}

void generate(const std::filesystem::path& out)
{
    generateLayout(out);
    generateStatuses(out);
    generateDirect(out);
    generateReflections(out);
    generatePathing(out);
    ipl::Context::sSIMDLevel = ipl::SIMDLevel::SSE2;
}

const Register reg("capi_simulation", generate);

}
