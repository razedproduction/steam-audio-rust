// Golden data for stage 8b: the C API (phonon.h) for scenes, static and instanced meshes, custom scenes, energy
// fields, impulse responses, reconstructors, probe arrays and batches, and baking (api_scene, api_energy_field,
// api_impulse_response, api_reconstructor, api_probes, api_baking, phonon_interfaces.h).
//
// Everything is called through the ipl* functions. Inputs come from Lcg draws in a fixed order, one value per
// statement; crates/phonon-capi/tests/capi_baking.rs replays the same order.
//
// The C API seeds the reflection simulators of a bake with the clock (risk R2), so the bakes use materials without
// scattering: every bounce is then specular and the results do not depend on the random numbers. Bakes run on one
// thread.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <context.h>
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
    // The reconstructor's band filters follow upstream's AVX path, as the port does (risk R8).
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
IPLVector3 interiorPoint(Lcg& lcg, const ipl::Box& box)
{
    float p[3];
    for (auto i = 0; i < 3; ++i)
    {
        const float t = 0.3f + 0.4f * draw01(lcg);
        p[i] = box.minCoordinates[i] + (box.maxCoordinates[i] - box.minCoordinates[i]) * t;
    }
    return IPLVector3{p[0], p[1], p[2]};
}

void writeBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes)
{
    writeNpy(path, bytes);
}

std::vector<uint8_t> save(IPLContext context, void (*saveFn)(void*, IPLSerializedObject), void* object)
{
    IPLSerializedObjectSettings settings{};
    IPLSerializedObject serialized = nullptr;
    check(iplSerializedObjectCreate(context, &settings, &serialized), "iplSerializedObjectCreate");
    saveFn(object, serialized);
    const auto* data = iplSerializedObjectGetData(serialized);
    const auto size = iplSerializedObjectGetSize(serialized);
    std::vector<uint8_t> bytes(data, data + size);
    iplSerializedObjectRelease(&serialized);
    return bytes;
}

std::vector<uint8_t> saveScene(IPLContext context, IPLScene scene)
{
    return save(context, [](void* o, IPLSerializedObject s) { iplSceneSave(static_cast<IPLScene>(o), s); }, scene);
}

std::vector<uint8_t> saveStaticMesh(IPLContext context, IPLStaticMesh mesh)
{
    return save(context, [](void* o, IPLSerializedObject s) { iplStaticMeshSave(static_cast<IPLStaticMesh>(o), s); },
                mesh);
}

std::vector<uint8_t> saveProbeBatch(IPLContext context, IPLProbeBatch batch)
{
    return save(context, [](void* o, IPLSerializedObject s) { iplProbeBatchSave(static_cast<IPLProbeBatch>(o), s); },
                batch);
}

std::vector<uint8_t> readText(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot read " + path.string());
    std::vector<uint8_t> bytes;
    for (char c; file.get(c);)
    {
        // fopen(..., "w") writes \r\n on Windows; the port writes \n.
        if (c != '\r')
            bytes.push_back(static_cast<uint8_t>(c));
    }
    return bytes;
}

// A static mesh created through the C API from an OBJ mesh.
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

// The boxroom with three Lcg materials, made specular.
ObjMesh specularRoom(Lcg& lcg)
{
    auto room = loadObj("boxroom");
    assignMaterials(lcg, room);
    makeSpecular(room);
    return room;
}

void appendProbes(std::vector<float>& out, IPLProbeArray probes)
{
    const auto n = iplProbeArrayGetNumProbes(probes);
    out.push_back(static_cast<float>(n));
    for (auto i = 0; i < n; ++i)
    {
        const auto probe = iplProbeArrayGetProbe(probes, i);
        out.push_back(probe.center.x);
        out.push_back(probe.center.y);
        out.push_back(probe.center.z);
        out.push_back(probe.radius);
    }
}

void onProgress(IPLfloat32 progress, void* userData)
{
    static_cast<std::vector<float>*>(userData)->push_back(progress);
}

// --------------------------------------------------------------------------------------------------------------------
// Struct layouts
// --------------------------------------------------------------------------------------------------------------------

#define SIZE(T) layout.push_back(static_cast<int32_t>(sizeof(T)))
#define FIELD(T, f) layout.push_back(static_cast<int32_t>(offsetof(T, f)))

void generateLayout(const std::filesystem::path& out)
{
    std::vector<int32_t> layout;

    SIZE(IPLSceneType); SIZE(IPLProbeGenerationType); SIZE(IPLBakedDataVariation); SIZE(IPLBakedDataType);
    SIZE(IPLReflectionsBakeFlags);

    SIZE(IPLTriangle); FIELD(IPLTriangle, indices);
    SIZE(IPLMaterial); FIELD(IPLMaterial, absorption); FIELD(IPLMaterial, scattering); FIELD(IPLMaterial, transmission);
    SIZE(IPLRay); FIELD(IPLRay, origin); FIELD(IPLRay, direction);
    SIZE(IPLHit); FIELD(IPLHit, distance); FIELD(IPLHit, triangleIndex); FIELD(IPLHit, objectIndex);
    FIELD(IPLHit, materialIndex); FIELD(IPLHit, normal); FIELD(IPLHit, material);
    SIZE(IPLSceneSettings); FIELD(IPLSceneSettings, type); FIELD(IPLSceneSettings, closestHitCallback);
    FIELD(IPLSceneSettings, anyHitCallback); FIELD(IPLSceneSettings, batchedClosestHitCallback);
    FIELD(IPLSceneSettings, batchedAnyHitCallback); FIELD(IPLSceneSettings, userData);
    FIELD(IPLSceneSettings, embreeDevice); FIELD(IPLSceneSettings, radeonRaysDevice);
    SIZE(IPLStaticMeshSettings); FIELD(IPLStaticMeshSettings, numVertices); FIELD(IPLStaticMeshSettings, numTriangles);
    FIELD(IPLStaticMeshSettings, numMaterials); FIELD(IPLStaticMeshSettings, vertices);
    FIELD(IPLStaticMeshSettings, triangles); FIELD(IPLStaticMeshSettings, materialIndices);
    FIELD(IPLStaticMeshSettings, materials);
    SIZE(IPLInstancedMeshSettings); FIELD(IPLInstancedMeshSettings, subScene); FIELD(IPLInstancedMeshSettings, transform);

    SIZE(IPLEnergyFieldSettings); FIELD(IPLEnergyFieldSettings, duration); FIELD(IPLEnergyFieldSettings, order);
    SIZE(IPLImpulseResponseSettings); FIELD(IPLImpulseResponseSettings, duration);
    FIELD(IPLImpulseResponseSettings, order); FIELD(IPLImpulseResponseSettings, samplingRate);
    SIZE(IPLReconstructorSettings); FIELD(IPLReconstructorSettings, maxDuration);
    FIELD(IPLReconstructorSettings, maxOrder); FIELD(IPLReconstructorSettings, samplingRate);
    SIZE(IPLReconstructorInputs); FIELD(IPLReconstructorInputs, energyField);
    SIZE(IPLReconstructorSharedInputs); FIELD(IPLReconstructorSharedInputs, duration);
    FIELD(IPLReconstructorSharedInputs, order);
    SIZE(IPLReconstructorOutputs); FIELD(IPLReconstructorOutputs, impulseResponse);

    SIZE(IPLProbeGenerationParams); FIELD(IPLProbeGenerationParams, type); FIELD(IPLProbeGenerationParams, spacing);
    FIELD(IPLProbeGenerationParams, height); FIELD(IPLProbeGenerationParams, transform);
    SIZE(IPLBakedDataIdentifier); FIELD(IPLBakedDataIdentifier, type); FIELD(IPLBakedDataIdentifier, variation);
    FIELD(IPLBakedDataIdentifier, endpointInfluence);

    SIZE(IPLReflectionsBakeParams); FIELD(IPLReflectionsBakeParams, scene); FIELD(IPLReflectionsBakeParams, probeBatch);
    FIELD(IPLReflectionsBakeParams, sceneType); FIELD(IPLReflectionsBakeParams, identifier);
    FIELD(IPLReflectionsBakeParams, bakeFlags); FIELD(IPLReflectionsBakeParams, numRays);
    FIELD(IPLReflectionsBakeParams, numDiffuseSamples); FIELD(IPLReflectionsBakeParams, numBounces);
    FIELD(IPLReflectionsBakeParams, simulatedDuration); FIELD(IPLReflectionsBakeParams, savedDuration);
    FIELD(IPLReflectionsBakeParams, order); FIELD(IPLReflectionsBakeParams, numThreads);
    FIELD(IPLReflectionsBakeParams, rayBatchSize); FIELD(IPLReflectionsBakeParams, irradianceMinDistance);
    FIELD(IPLReflectionsBakeParams, bakeBatchSize); FIELD(IPLReflectionsBakeParams, openCLDevice);
    FIELD(IPLReflectionsBakeParams, radeonRaysDevice);
    SIZE(IPLPathBakeParams); FIELD(IPLPathBakeParams, scene); FIELD(IPLPathBakeParams, probeBatch);
    FIELD(IPLPathBakeParams, identifier); FIELD(IPLPathBakeParams, numSamples); FIELD(IPLPathBakeParams, radius);
    FIELD(IPLPathBakeParams, threshold); FIELD(IPLPathBakeParams, visRange); FIELD(IPLPathBakeParams, pathRange);
    FIELD(IPLPathBakeParams, numThreads);

    writeNpy(out / "layout.npy", layout);
}

// --------------------------------------------------------------------------------------------------------------------
// Energy fields and impulse responses
// --------------------------------------------------------------------------------------------------------------------

// Runs the same script of operations on three energy fields or three impulse responses. After each operation, every
// object's sizes are recorded as [channels, bins or samples] and its data appended.
template <typename Handle, typename Api>
void generateOperations(const std::filesystem::path& out, const std::string& name, Handle (&objects)[3], Lcg& lcg,
                        const Api& api)
{
    std::vector<float> data;
    std::vector<int32_t> sizes;
    auto record = [&]()
    {
        for (auto object : objects)
        {
            const auto channels = api.numChannels(object);
            const auto length = api.length(object);
            sizes.push_back(channels);
            sizes.push_back(length);
            const auto* values = api.data(object);
            data.insert(data.end(), values, values + static_cast<size_t>(channels) * api.rowsPerChannel * length);
        }
    };

    for (auto object : objects)
    {
        const auto n = static_cast<size_t>(api.numChannels(object)) * api.rowsPerChannel * api.length(object);
        auto* values = api.data(object);
        for (size_t i = 0; i < n; ++i)
            values[i] = lcg.nextSigned();
    }
    record();

    auto [a, b, c] = objects;
    api.scale(a, 0.5f, a); record();
    api.add(a, b, c); record();
    api.add(a, c, a); record();
    api.scaleAccum(b, -1.25f, a); record();
    api.scaleAccum(c, 2.0f, c); record();
    api.copy(a, b); record();
    api.swap(a, c); record();
    api.reset(b); record();
    api.scale(c, 3.0f, b); record();
    api.add(b, b, b); record();

    // Pointer offsets of channels and bands, relative to the data.
    std::vector<int32_t> offsets;
    for (auto object : objects)
    {
        const auto* base = api.data(object);
        for (auto i = 0; i < api.numChannels(object); ++i)
            offsets.push_back(static_cast<int32_t>(api.channel(object, i) - base));
    }

    writeNpy(out / (name + ".npy"), data);
    writeNpy(out / (name + "_sizes.npy"), {sizes.size() / 2, 2}, sizes.data());
    writeNpy(out / (name + "_offsets.npy"), offsets);
}

struct EnergyFieldApi
{
    int rowsPerChannel = 3;
    IPLint32 numChannels(IPLEnergyField f) const { return iplEnergyFieldGetNumChannels(f); }
    IPLint32 length(IPLEnergyField f) const { return iplEnergyFieldGetNumBins(f); }
    float* data(IPLEnergyField f) const { return iplEnergyFieldGetData(f); }
    float* channel(IPLEnergyField f, int i) const { return iplEnergyFieldGetChannel(f, i); }
    void scale(IPLEnergyField in, float s, IPLEnergyField o) const { iplEnergyFieldScale(in, s, o); }
    void scaleAccum(IPLEnergyField in, float s, IPLEnergyField o) const { iplEnergyFieldScaleAccum(in, s, o); }
    void add(IPLEnergyField a, IPLEnergyField b, IPLEnergyField o) const { iplEnergyFieldAdd(a, b, o); }
    void copy(IPLEnergyField src, IPLEnergyField dst) const { iplEnergyFieldCopy(src, dst); }
    void swap(IPLEnergyField a, IPLEnergyField b) const { iplEnergyFieldSwap(a, b); }
    void reset(IPLEnergyField f) const { iplEnergyFieldReset(f); }
};

struct ImpulseResponseApi
{
    int rowsPerChannel = 1;
    IPLint32 numChannels(IPLImpulseResponse r) const { return iplImpulseResponseGetNumChannels(r); }
    IPLint32 length(IPLImpulseResponse r) const { return iplImpulseResponseGetNumSamples(r); }
    float* data(IPLImpulseResponse r) const { return iplImpulseResponseGetData(r); }
    float* channel(IPLImpulseResponse r, int i) const { return iplImpulseResponseGetChannel(r, i); }
    void scale(IPLImpulseResponse in, float s, IPLImpulseResponse o) const { iplImpulseResponseScale(in, s, o); }
    void scaleAccum(IPLImpulseResponse in, float s, IPLImpulseResponse o) const
    {
        iplImpulseResponseScaleAccum(in, s, o);
    }
    void add(IPLImpulseResponse a, IPLImpulseResponse b, IPLImpulseResponse o) const { iplImpulseResponseAdd(a, b, o); }
    void copy(IPLImpulseResponse src, IPLImpulseResponse dst) const { iplImpulseResponseCopy(src, dst); }
    void swap(IPLImpulseResponse a, IPLImpulseResponse b) const { iplImpulseResponseSwap(a, b); }
    void reset(IPLImpulseResponse r) const { iplImpulseResponseReset(r); }
};

void generateFields(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(811);

    // Energy fields of 11 bins and order 1, 7 bins and order 2, 13 bins and order 0.
    const IPLEnergyFieldSettings fieldSettings[3] = {{0.105f, 1}, {0.07f, 2}, {0.13f, 0}};
    IPLEnergyField fields[3];
    for (auto i = 0; i < 3; ++i)
    {
        auto settings = fieldSettings[i];
        check(iplEnergyFieldCreate(context, &settings, &fields[i]), "iplEnergyFieldCreate");
    }

    std::vector<int32_t> bandOffsets;
    for (auto c = 0; c < iplEnergyFieldGetNumChannels(fields[0]); ++c)
        for (auto b = 0; b < 3; ++b)
            bandOffsets.push_back(static_cast<int32_t>(iplEnergyFieldGetBand(fields[0], c, b) -
                                                       iplEnergyFieldGetData(fields[0])));
    writeNpy(out / "energy_field_band_offsets.npy", bandOffsets);

    generateOperations(out, "energy_fields", fields, lcg, EnergyFieldApi{});
    for (auto& field : fields)
        iplEnergyFieldRelease(&field);

    // Impulse responses of 101 samples and order 1, 58 samples and order 0, 38 samples and order 2.
    const IPLImpulseResponseSettings irSettings[3] = {{0.0021f, 1, 48000}, {0.0013f, 0, 44100}, {0.0017f, 2, 22050}};
    IPLImpulseResponse irs[3];
    for (auto i = 0; i < 3; ++i)
    {
        auto settings = irSettings[i];
        check(iplImpulseResponseCreate(context, &settings, &irs[i]), "iplImpulseResponseCreate");
    }
    generateOperations(out, "impulse_responses", irs, lcg, ImpulseResponseApi{});
    for (auto& ir : irs)
        iplImpulseResponseRelease(&ir);

    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Reconstructor
// --------------------------------------------------------------------------------------------------------------------

// Fills an energy field with energies decaying over the bins: positive in channel 0, signed in the others.
void fillEnergies(IPLEnergyField field, Lcg& lcg)
{
    const auto channels = iplEnergyFieldGetNumChannels(field);
    const auto bins = iplEnergyFieldGetNumBins(field);
    for (auto c = 0; c < channels; ++c)
    {
        for (auto b = 0; b < 3; ++b)
        {
            auto* band = iplEnergyFieldGetBand(field, c, b);
            for (auto i = 0; i < bins; ++i)
            {
                const float u = (c == 0) ? draw01(lcg) : 0.25f * lcg.nextSigned();
                band[i] = u * 1e-2f / (1.0f + static_cast<float>(i));
            }
        }
    }
}

void generateReconstructor(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(821);

    IPLReconstructorSettings settings{0.1f, 2, 24000};
    IPLReconstructor reconstructor = nullptr;
    check(iplReconstructorCreate(context, &settings, &reconstructor), "iplReconstructorCreate");

    IPLEnergyFieldSettings fieldSettings[2] = {{0.1f, 1}, {0.05f, 2}};
    IPLEnergyField fields[2];
    for (auto i = 0; i < 2; ++i)
    {
        check(iplEnergyFieldCreate(context, &fieldSettings[i], &fields[i]), "iplEnergyFieldCreate");
        fillEnergies(fields[i], lcg);
    }

    IPLImpulseResponseSettings irSettings[3] = {{0.1f, 1, 24000}, {0.08f, 2, 24000}, {0.03f, 0, 24000}};
    IPLImpulseResponse irs[3];
    for (auto i = 0; i < 3; ++i)
    {
        check(iplImpulseResponseCreate(context, &irSettings[i], &irs[i]), "iplImpulseResponseCreate");
        // Noise, so that samples the reconstruction zeroes are checked too.
        auto* data = iplImpulseResponseGetData(irs[i]);
        const auto n = iplImpulseResponseGetNumChannels(irs[i]) * iplImpulseResponseGetNumSamples(irs[i]);
        for (auto j = 0; j < n; ++j)
            data[j] = lcg.nextSigned();
    }

    std::vector<float> data;
    auto record = [&]()
    {
        for (auto ir : irs)
        {
            const auto* values = iplImpulseResponseGetData(ir);
            data.insert(data.end(), values,
                        values + iplImpulseResponseGetNumChannels(ir) * iplImpulseResponseGetNumSamples(ir));
        }
    };

    {
        IPLReconstructorInputs inputs[2] = {{fields[0]}, {fields[1]}};
        IPLReconstructorOutputs outputs[2] = {{irs[0]}, {irs[1]}};
        IPLReconstructorSharedInputs shared{0.1f, 1};
        iplReconstructorReconstruct(reconstructor, 2, inputs, &shared, outputs);
        record();
    }
    {
        IPLReconstructorInputs inputs[2] = {{fields[1]}, {fields[0]}};
        IPLReconstructorOutputs outputs[2] = {{irs[2]}, {irs[1]}};
        IPLReconstructorSharedInputs shared{0.05f, 2};
        iplReconstructorReconstruct(reconstructor, 2, inputs, &shared, outputs);
        record();
    }
    {
        // An order above the fields' and a duration of 0.
        IPLReconstructorInputs inputs[1] = {{fields[0]}};
        IPLReconstructorOutputs outputs[1] = {{irs[1]}};
        IPLReconstructorSharedInputs shared{0.0f, 3};
        iplReconstructorReconstruct(reconstructor, 1, inputs, &shared, outputs);
        record();
    }

    writeNpy(out / "reconstructed.npy", data);

    for (auto& ir : irs)
        iplImpulseResponseRelease(&ir);
    for (auto& field : fields)
        iplEnergyFieldRelease(&field);
    iplReconstructorRelease(&reconstructor);
    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------------------------------------------------

void generateScenes(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(823);

    auto room = specularRoom(lcg);
    auto box = loadObj("smallbox");
    const auto bounds = meshBounds(room);

    IPLSceneSettings sceneSettings{};
    sceneSettings.type = IPL_SCENETYPE_DEFAULT;
    IPLScene scene = nullptr;
    check(iplSceneCreate(context, &sceneSettings, &scene), "iplSceneCreate");
    IPLScene subScene = nullptr;
    check(iplSceneCreate(context, &sceneSettings, &subScene), "iplSceneCreate");

    auto roomMesh = createStaticMesh(scene, room);
    iplStaticMeshAdd(roomMesh, scene);

    // The sub-scene's mesh is added but not committed: the parent's commit commits it.
    auto boxMesh = createStaticMesh(subScene, box);
    iplStaticMeshAdd(boxMesh, subScene);

    IPLInstancedMeshSettings instanceSettings{};
    instanceSettings.subScene = subScene;
    instanceSettings.transform = toIpl(drawTransform(lcg, bounds));
    IPLInstancedMesh instance = nullptr;
    check(iplInstancedMeshCreate(scene, &instanceSettings, &instance), "iplInstancedMeshCreate");
    iplInstancedMeshAdd(instance, scene);
    iplSceneCommit(scene);

    writeBytes(out / "scene.npy", saveScene(context, scene));
    writeBytes(out / "static_mesh.npy", saveStaticMesh(context, roomMesh));

    const auto objPath = std::filesystem::temp_directory_path() / "steam_audio_capi_baking_scene.obj";
    iplSceneSaveOBJ(scene, objPath.string().c_str());
    writeBytes(out / "scene_obj.npy", readText(objPath));
    writeBytes(out / "scene_mtl.npy", readText(std::filesystem::path(objPath).replace_extension(".mtl")));
    std::filesystem::remove(objPath);
    std::filesystem::remove(std::filesystem::path(objPath).replace_extension(".mtl"));

    // Probes generated in the scene: a centroid probe in a random box, then floor probes over the room, before and
    // after the instance moves.
    std::vector<float> probeData;
    IPLProbeArray probes = nullptr;
    check(iplProbeArrayCreate(context, &probes), "iplProbeArrayCreate");

    IPLProbeGenerationParams params{};
    params.type = IPL_PROBEGENERATIONTYPE_CENTROID;
    params.transform = toIpl(drawTransform(lcg, bounds));
    iplProbeArrayGenerateProbes(probes, scene, &params);
    appendProbes(probeData, probes);

    const auto extents = bounds.extents();
    params.type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
    params.spacing = 0.2f * std::max(extents.x(), extents.z());
    params.height = 1.5f;
    params.transform = boxTransform(bounds);
    iplProbeArrayGenerateProbes(probes, scene, &params);
    appendProbes(probeData, probes);

    iplInstancedMeshUpdateTransform(instance, scene, toIpl(drawTransform(lcg, bounds)));
    iplSceneCommit(scene);
    iplProbeArrayGenerateProbes(probes, scene, &params);
    appendProbes(probeData, probes);
    writeNpy(out / "scene_probes.npy", probeData);

    // A changed material, then the instance removed, then the room removed.
    IPLMaterial material{};
    for (auto i = 0; i < 3; ++i)
        material.absorption[i] = draw01(lcg);
    material.scattering = draw01(lcg);
    for (auto i = 0; i < 3; ++i)
        material.transmission[i] = draw01(lcg);
    iplStaticMeshSetMaterial(roomMesh, scene, &material, 1);
    iplSceneCommit(scene);
    writeBytes(out / "scene_material.npy", saveScene(context, scene));

    iplInstancedMeshRemove(instance, scene);
    iplStaticMeshRemove(roomMesh, scene);
    iplSceneCommit(scene);
    writeBytes(out / "scene_empty.npy", saveScene(context, scene));

    iplProbeArrayRelease(&probes);
    iplInstancedMeshRelease(&instance);
    iplStaticMeshRelease(&boxMesh);
    iplStaticMeshRelease(&roomMesh);
    iplSceneRelease(&subScene);
    iplSceneRelease(&scene);
    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Custom scenes: an analytic axis-aligned room
// --------------------------------------------------------------------------------------------------------------------

struct Room
{
    float lo[3];
    float hi[3];
    IPLMaterial material;
};

// The closest face of the room's box along the ray at a distance in [minDistance, maxDistance], as the axis it is
// perpendicular to (-1: none).
int intersectRoom(const Room& room, const IPLRay& ray, float minDistance, float maxDistance, float& distance)
{
    const float o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float d[3] = {ray.direction.x, ray.direction.y, ray.direction.z};

    float tNear = -std::numeric_limits<float>::infinity();
    float tFar = std::numeric_limits<float>::infinity();
    int nearAxis = -1;
    int farAxis = -1;
    for (auto a = 0; a < 3; ++a)
    {
        if (d[a] == 0.0f)
        {
            if (o[a] < room.lo[a] || o[a] > room.hi[a])
                return -1;
            continue;
        }
        const float t1 = (room.lo[a] - o[a]) / d[a];
        const float t2 = (room.hi[a] - o[a]) / d[a];
        const float tMin = (t1 < t2) ? t1 : t2;
        const float tMax = (t1 < t2) ? t2 : t1;
        if (tMin > tNear)
        {
            tNear = tMin;
            nearAxis = a;
        }
        if (tMax < tFar)
        {
            tFar = tMax;
            farAxis = a;
        }
    }
    if (tNear > tFar)
        return -1;
    if (nearAxis >= 0 && tNear >= minDistance && tNear <= maxDistance)
    {
        distance = tNear;
        return nearAxis;
    }
    if (farAxis >= 0 && tFar >= minDistance && tFar <= maxDistance)
    {
        distance = tFar;
        return farAxis;
    }
    return -1;
}

void IPLCALL roomClosestHit(const IPLRay* ray, IPLfloat32 minDistance, IPLfloat32 maxDistance, IPLHit* hit,
                            void* userData)
{
    auto& room = *static_cast<Room*>(userData);
    hit->distance = std::numeric_limits<float>::infinity();
    hit->triangleIndex = -1;
    hit->objectIndex = -1;
    hit->materialIndex = -1;
    hit->normal = IPLVector3{0.0f, 0.0f, 0.0f};
    hit->material = nullptr;
    if (maxDistance <= minDistance)
        return;

    float distance = 0.0f;
    const auto axis = intersectRoom(room, *ray, minDistance, maxDistance, distance);
    if (axis < 0)
        return;

    const float d[3] = {ray->direction.x, ray->direction.y, ray->direction.z};
    float n[3] = {0.0f, 0.0f, 0.0f};
    n[axis] = (d[axis] > 0.0f) ? -1.0f : 1.0f;
    hit->distance = distance;
    hit->triangleIndex = 2 * axis + ((d[axis] > 0.0f) ? 1 : 0);
    hit->objectIndex = 0;
    hit->materialIndex = 0;
    hit->normal = IPLVector3{n[0], n[1], n[2]};
    hit->material = &room.material;
}

void IPLCALL roomAnyHit(const IPLRay* ray, IPLfloat32 minDistance, IPLfloat32 maxDistance, IPLuint8* occluded,
                        void* userData)
{
    if (maxDistance <= minDistance)
    {
        *occluded = 1;
        return;
    }
    float distance = 0.0f;
    *occluded = (intersectRoom(*static_cast<Room*>(userData), *ray, minDistance, maxDistance, distance) >= 0) ? 1 : 0;
}

void IPLCALL roomBatchedClosestHit(IPLint32 numRays, const IPLRay* rays, const IPLfloat32* minDistances,
                                   const IPLfloat32* maxDistances, IPLHit* hits, void* userData)
{
    for (auto i = 0; i < numRays; ++i)
        roomClosestHit(&rays[i], minDistances[i], maxDistances[i], &hits[i], userData);
}

void IPLCALL roomBatchedAnyHit(IPLint32 numRays, const IPLRay* rays, const IPLfloat32* minDistances,
                               const IPLfloat32* maxDistances, IPLuint8* occluded, void* userData)
{
    for (auto i = 0; i < numRays; ++i)
    {
        if (maxDistances[i] < minDistances[i])
            occluded[i] = 1;
        else
            roomAnyHit(&rays[i], minDistances[i], maxDistances[i], &occluded[i], userData);
    }
}

IPLReflectionsBakeParams bakeParams(IPLScene scene, IPLProbeBatch batch, IPLSceneType sceneType,
                                    const IPLBakedDataIdentifier& identifier, IPLReflectionsBakeFlags flags)
{
    IPLReflectionsBakeParams params{};
    params.scene = scene;
    params.probeBatch = batch;
    params.sceneType = sceneType;
    params.identifier = identifier;
    params.bakeFlags = flags;
    params.numRays = 256;
    params.numDiffuseSamples = 8;
    params.numBounces = 4;
    params.simulatedDuration = 0.3f;
    params.savedDuration = 0.2f;
    params.order = 1;
    params.numThreads = 1;
    params.rayBatchSize = 37;
    params.irradianceMinDistance = 1.0f;
    params.bakeBatchSize = 1;
    return params;
}

// Energy fields (prefilled with the sentinel) and reverb times (likewise) looked up at every probe.
void appendBakedData(std::vector<float>& fields, std::vector<float>& reverbs, IPLContext context, IPLProbeBatch batch,
                     IPLBakedDataIdentifier identifier)
{
    IPLEnergyFieldSettings settings{0.2f, 1};
    IPLEnergyField field = nullptr;
    check(iplEnergyFieldCreate(context, &settings, &field), "iplEnergyFieldCreate");
    const auto size = iplEnergyFieldGetNumChannels(field) * 3 * iplEnergyFieldGetNumBins(field);

    for (auto i = 0; i < iplProbeBatchGetNumProbes(batch); ++i)
    {
        auto* data = iplEnergyFieldGetData(field);
        std::fill(data, data + size, kSentinel);
        iplProbeBatchGetEnergyField(batch, &identifier, i, field);
        fields.insert(fields.end(), data, data + size);

        float reverb[3] = {kSentinel, kSentinel, kSentinel};
        iplProbeBatchGetReverb(batch, &identifier, i, reverb);
        reverbs.insert(reverbs.end(), reverb, reverb + 3);
    }

    iplEnergyFieldRelease(&field);
}

void generateCustomScene(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(827);

    Room room{};
    for (auto i = 0; i < 3; ++i)
    {
        room.lo[i] = -2.0f - 3.0f * draw01(lcg);
        room.hi[i] = 2.0f + 3.0f * draw01(lcg);
    }
    for (auto i = 0; i < 3; ++i)
        room.material.absorption[i] = 0.05f + 0.5f * draw01(lcg);
    room.material.scattering = 0.0f;
    for (auto i = 0; i < 3; ++i)
        room.material.transmission[i] = draw01(lcg);

    IPLSceneSettings sceneSettings{};
    sceneSettings.type = IPL_SCENETYPE_CUSTOM;
    sceneSettings.closestHitCallback = roomClosestHit;
    sceneSettings.anyHitCallback = roomAnyHit;
    sceneSettings.batchedClosestHitCallback = roomBatchedClosestHit;
    sceneSettings.batchedAnyHitCallback = roomBatchedAnyHit;
    sceneSettings.userData = &room;
    IPLScene scene = nullptr;
    check(iplSceneCreate(context, &sceneSettings, &scene), "iplSceneCreate");

    ipl::Box bounds;
    for (auto i = 0; i < 3; ++i)
    {
        bounds.minCoordinates[i] = room.lo[i];
        bounds.maxCoordinates[i] = room.hi[i];
    }

    IPLProbeArray probes = nullptr;
    check(iplProbeArrayCreate(context, &probes), "iplProbeArrayCreate");
    IPLProbeGenerationParams params{};
    params.type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
    params.spacing = 1.7f;
    params.height = 1.2f;
    params.transform = boxTransform(bounds);
    iplProbeArrayGenerateProbes(probes, scene, &params);
    std::vector<float> probeData;
    appendProbes(probeData, probes);
    writeNpy(out / "custom_probes.npy", probeData);

    IPLProbeBatch batch = nullptr;
    check(iplProbeBatchCreate(context, &batch), "iplProbeBatchCreate");
    iplProbeBatchAddProbeArray(batch, probes);
    iplProbeBatchCommit(batch);

    // Reverb, traced by the batched simulator through the batched callbacks, then by the default simulator through the
    // single-ray callbacks.
    std::vector<float> progress;
    std::vector<float> fields;
    std::vector<float> reverbs;
    IPLBakedDataIdentifier reverbId{IPL_BAKEDDATATYPE_REFLECTIONS, IPL_BAKEDDATAVARIATION_REVERB, {}};
    auto bake = bakeParams(scene, batch, IPL_SCENETYPE_CUSTOM, reverbId,
                           static_cast<IPLReflectionsBakeFlags>(IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION |
                                                                IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC));
    iplReflectionsBakerBake(context, &bake, onProgress, &progress);
    appendBakedData(fields, reverbs, context, batch, reverbId);

    bake.sceneType = IPL_SCENETYPE_DEFAULT;
    iplReflectionsBakerBake(context, &bake, onProgress, &progress);
    appendBakedData(fields, reverbs, context, batch, reverbId);

    writeNpy(out / "custom_progress.npy", progress);
    writeNpy(out / "custom_fields.npy", fields);
    writeNpy(out / "custom_reverbs.npy", reverbs);

    iplProbeBatchRelease(&batch);
    iplProbeArrayRelease(&probes);
    iplSceneRelease(&scene);
    iplContextRelease(&context);
}

// --------------------------------------------------------------------------------------------------------------------
// Probe batches and baking
// --------------------------------------------------------------------------------------------------------------------

void generateBaking(const std::filesystem::path& out)
{
    auto context = makeContext();
    Lcg lcg(829);

    auto room = specularRoom(lcg);
    auto box = loadObj("smallbox");
    makeSpecular(box);
    const auto bounds = meshBounds(room);

    IPLSceneSettings sceneSettings{};
    sceneSettings.type = IPL_SCENETYPE_DEFAULT;
    IPLScene scene = nullptr;
    check(iplSceneCreate(context, &sceneSettings, &scene), "iplSceneCreate");
    IPLScene subScene = nullptr;
    check(iplSceneCreate(context, &sceneSettings, &subScene), "iplSceneCreate");

    auto roomMesh = createStaticMesh(scene, room);
    iplStaticMeshAdd(roomMesh, scene);
    auto boxMesh = createStaticMesh(subScene, box);
    iplStaticMeshAdd(boxMesh, subScene);
    iplSceneCommit(subScene);

    IPLInstancedMeshSettings instanceSettings{};
    instanceSettings.subScene = subScene;
    instanceSettings.transform = toIpl(drawTransform(lcg, bounds));
    IPLInstancedMesh instance = nullptr;
    check(iplInstancedMeshCreate(scene, &instanceSettings, &instance), "iplInstancedMeshCreate");
    iplInstancedMeshAdd(instance, scene);
    iplSceneCommit(scene);

    IPLProbeArray probes = nullptr;
    check(iplProbeArrayCreate(context, &probes), "iplProbeArrayCreate");
    const auto extents = bounds.extents();
    IPLProbeGenerationParams params{};
    params.type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
    params.spacing = 0.25f * std::max(extents.x(), extents.z());
    params.height = 1.5f;
    params.transform = boxTransform(bounds);
    iplProbeArrayGenerateProbes(probes, scene, &params);

    // A batch of the floor probes, one more probe, and probe 3 removed.
    IPLProbeBatch batch = nullptr;
    check(iplProbeBatchCreate(context, &batch), "iplProbeBatchCreate");
    iplProbeBatchAddProbeArray(batch, probes);
    IPLSphere extra{};
    extra.center = toIpl(drawPoint(lcg, bounds));
    extra.radius = draw01(lcg) + 0.5f;
    iplProbeBatchAddProbe(batch, extra);
    iplProbeBatchRemoveProbe(batch, 3);
    iplProbeBatchCommit(batch);
    writeBytes(out / "batch_probes.npy", saveProbeBatch(context, batch));

    const auto radius = 0.35f * extents.length();
    IPLBakedDataIdentifier reverbId{IPL_BAKEDDATATYPE_REFLECTIONS, IPL_BAKEDDATAVARIATION_REVERB, {}};
    IPLBakedDataIdentifier sourceId{IPL_BAKEDDATATYPE_REFLECTIONS, IPL_BAKEDDATAVARIATION_STATICSOURCE, {}};
    sourceId.endpointInfluence.center = interiorPoint(lcg, bounds);
    sourceId.endpointInfluence.radius = radius;
    IPLBakedDataIdentifier listenerId{IPL_BAKEDDATATYPE_REFLECTIONS, IPL_BAKEDDATAVARIATION_STATICLISTENER, {}};
    listenerId.endpointInfluence.center = interiorPoint(lcg, bounds);
    listenerId.endpointInfluence.radius = radius;
    IPLBakedDataIdentifier pathId{IPL_BAKEDDATATYPE_PATHING, IPL_BAKEDDATAVARIATION_DYNAMIC, {}};

    std::vector<float> progress;
    std::vector<int32_t> progressCounts;
    std::vector<int32_t> dataSizes;
    auto recordSizes = [&]()
    {
        for (auto id : {reverbId, sourceId, listenerId, pathId})
            dataSizes.push_back(static_cast<int32_t>(iplProbeBatchGetDataSize(batch, &id)));
    };
    auto countProgress = [&](size_t before) { progressCounts.push_back(static_cast<int32_t>(progress.size() - before)); };

    auto both = static_cast<IPLReflectionsBakeFlags>(IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION |
                                                     IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC);
    auto before = progress.size();
    auto bake = bakeParams(scene, batch, IPL_SCENETYPE_DEFAULT, reverbId, both);
    iplReflectionsBakerBake(context, &bake, onProgress, &progress);
    countProgress(before);
    recordSizes();

    before = progress.size();
    bake = bakeParams(scene, batch, IPL_SCENETYPE_DEFAULT, sourceId, IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION);
    iplReflectionsBakerBake(context, &bake, onProgress, &progress);
    countProgress(before);
    recordSizes();

    before = progress.size();
    bake = bakeParams(scene, batch, IPL_SCENETYPE_DEFAULT, listenerId, IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC);
    iplReflectionsBakerBake(context, &bake, onProgress, &progress);
    countProgress(before);
    recordSizes();

    std::vector<float> fields;
    std::vector<float> reverbs;
    for (const auto& id : {reverbId, sourceId, listenerId})
        appendBakedData(fields, reverbs, context, batch, id);
    writeNpy(out / "baked_fields.npy", fields);
    writeNpy(out / "baked_reverbs.npy", reverbs);

    // The path baker reports progress from its jobs as they finish, so how often depends on timing: not recorded.
    std::vector<float> pathProgress;
    IPLPathBakeParams pathParams{};
    pathParams.scene = scene;
    pathParams.probeBatch = batch;
    pathParams.identifier = pathId;
    pathParams.numSamples = 1;
    pathParams.radius = 0.5f;
    pathParams.threshold = 0.5f;
    pathParams.visRange = 1000.0f;
    pathParams.pathRange = 1000.0f;
    pathParams.numThreads = 1;
    iplPathBakerBake(context, &pathParams, onProgress, &pathProgress);
    if (pathProgress.empty() || pathProgress.back() != 1.0f)
        throw std::runtime_error("the path bake did not report completion");
    recordSizes();

    writeBytes(out / "batch_baked.npy", saveProbeBatch(context, batch));

    iplProbeBatchRemoveData(batch, &sourceId);
    recordSizes();

    writeNpy(out / "bake_progress.npy", progress);
    writeNpy(out / "bake_progress_counts.npy", progressCounts);
    writeNpy(out / "baked_data_sizes.npy", {dataSizes.size() / 4, 4}, dataSizes.data());

    iplProbeBatchRelease(&batch);
    iplProbeArrayRelease(&probes);
    iplInstancedMeshRelease(&instance);
    iplStaticMeshRelease(&boxMesh);
    iplStaticMeshRelease(&roomMesh);
    iplSceneRelease(&subScene);
    iplSceneRelease(&scene);
    iplContextRelease(&context);
}

void generate(const std::filesystem::path& out)
{
    generateLayout(out);
    generateFields(out);
    generateReconstructor(out);
    generateScenes(out);
    generateCustomScene(out);
    generateBaking(out);
    ipl::Context::sSIMDLevel = ipl::SIMDLevel::SSE2;
}

const Register reg("capi_baking", generate);

}
