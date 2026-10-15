// Golden data for stage 6.4a: serialized_object.{h,cpp} and the serialize functions of energy_field, mesh,
// static_mesh and scene (FlatBuffers 1.12, the version upstream's build fetches).
//
// Every case is the whole FlatBuffer, as bytes: crates/steam-audio/tests/serialization.rs builds the same objects,
// checks that its serializer writes the same bytes and that its deserializer reads them back into equal objects.
// Meshes are read from core/data/meshes (run from the repository root) with the OBJ loader in scene_inputs.h; other
// inputs come from Lcg draws in a fixed order, one value per statement.
//
//   energy_field_<i>:       an EnergyField of kFieldSizes[i], filled with Lcg values (the last one also with NaN,
//                           infinities and -0), finished as the root of its own buffer; energy_field_empty: a field of
//                           0 bins, whose num_bins equals the default and is not written;
//   energy_field_sizes:     serializedSize() of each field above;
//   static_mesh_<name>:     serializeAsRoot of a StaticMesh of core/data/meshes/<name> (simplescene with Lcg
//                           materials);
//   scene:                  serializeAsRoot of a scene with boxroom, simplescene (Lcg materials) and smallbox added
//                           twice, an instanced mesh (not serialized) and a static mesh added after the last commit
//                           (not serialized either).

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <energy_field.h>
#include <instanced_mesh.h>
#include <scene.h>
#include <serialized_object.h>
#include <static_mesh.h>

#include "generators.h"
#include "npy.h"
#include "scene_inputs.h"

namespace {

using namespace golden;
using namespace ipl;

void writeBytes(const std::filesystem::path& path, SerializedObject& object)
{
    const auto* data = reinterpret_cast<const uint8_t*>(object.data());
    writeNpy(path, std::vector<uint8_t>(data, data + object.size()));
}

struct FieldSize
{
    int numChannels;
    int numBins;
};

const FieldSize kFieldSizes[] = {{1, 7}, {4, 13}, {9, 1}, {16, 30}};

void generateEnergyFields(const std::filesystem::path& out)
{
    Lcg lcg(211);

    std::vector<int32_t> sizes;
    const auto numCases = static_cast<int>(sizeof(kFieldSizes) / sizeof(kFieldSizes[0]));

    for (auto i = 0; i < numCases; ++i)
    {
        const auto order = static_cast<int>(sqrtf(static_cast<float>(kFieldSizes[i].numChannels))) - 1;
        EnergyField field(kFieldSizes[i].numBins * EnergyField::kBinDuration - 1e-4f, order);
        if (field.numChannels() != kFieldSizes[i].numChannels || field.numBins() != kFieldSizes[i].numBins)
            throw std::runtime_error("unexpected energy field size");

        const auto total = field.numChannels() * Bands::kNumBands * field.numBins();
        for (auto j = 0; j < total; ++j)
            field.flatData()[j] = lcg.nextSigned();

        if (i == numCases - 1)
        {
            field.flatData()[0] = std::numeric_limits<float>::quiet_NaN();
            field.flatData()[1] = std::numeric_limits<float>::infinity();
            field.flatData()[2] = -std::numeric_limits<float>::infinity();
            field.flatData()[3] = -0.0f;
        }

        SerializedObject object;
        object.fbb().Finish(field.serialize(object));
        object.commit();
        writeBytes(out / ("energy_field_" + std::to_string(i) + ".npy"), object);

        sizes.push_back(static_cast<int32_t>(field.serializedSize()));
    }

    EnergyField empty(0.0f, 1);
    SerializedObject object;
    object.fbb().Finish(empty.serialize(object));
    object.commit();
    writeBytes(out / "energy_field_empty.npy", object);
    sizes.push_back(static_cast<int32_t>(empty.serializedSize()));

    writeNpy(out / "energy_field_sizes.npy", sizes);
}

void generateStaticMeshes(const std::filesystem::path& out)
{
    Lcg lcg(223);

    for (const auto* name : {"smallbox", "boxroom", "simplescene", "sphere", "teapot"})
    {
        auto obj = loadObj(name);
        if (std::string(name) == "simplescene")
            assignMaterials(lcg, obj);

        Scene scene;
        auto staticMesh = createStaticMesh(scene, obj);

        SerializedObject object;
        static_cast<StaticMesh*>(staticMesh.get())->serializeAsRoot(object);
        writeBytes(out / ("static_mesh_" + std::string(name) + ".npy"), object);
    }
}

void generateScene(const std::filesystem::path& out)
{
    Lcg lcg(227);

    const auto boxroom = loadObj("boxroom");
    auto simplescene = loadObj("simplescene");
    assignMaterials(lcg, simplescene);
    const auto smallbox = loadObj("smallbox");
    const auto sphere = loadObj("sphere");

    auto subScene = ipl::make_shared<Scene>();
    subScene->addStaticMesh(createStaticMesh(*subScene, smallbox));
    subScene->commit();

    Scene scene;
    auto boxroomMesh = createStaticMesh(scene, boxroom);
    auto simplesceneMesh = createStaticMesh(scene, simplescene);
    auto smallboxMesh = createStaticMesh(scene, smallbox);
    scene.addStaticMesh(boxroomMesh);
    scene.addStaticMesh(smallboxMesh);
    scene.addStaticMesh(simplesceneMesh);
    scene.addStaticMesh(smallboxMesh);

    Box bounds;
    bounds.minCoordinates = Vector3f(-1.0f, -1.0f, -1.0f);
    bounds.maxCoordinates = Vector3f(1.0f, 1.0f, 1.0f);
    const auto transform = drawTransform(lcg, bounds);
    scene.addInstancedMesh(scene.createInstancedMesh(subScene, transform));
    scene.commit();

    scene.addStaticMesh(createStaticMesh(scene, sphere));

    SerializedObject object;
    scene.serializeAsRoot(object);
    writeBytes(out / "scene.npy", object);
}

void generateSerialization(const std::filesystem::path& out)
{
    generateEnergyFields(out);
    generateStaticMeshes(out);
    generateScene(out);
}

Register serialization("serialization", generateSerialization);

}
