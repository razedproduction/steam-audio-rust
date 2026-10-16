// Golden data for stage 6.4b: probe_tree, probe_generator, probe_batch, probe_manager (ProbeNeighborhood),
// baked_reflection_data and baked_reflection_simulator.
//
// Meshes are read from core/data/meshes (run from the repository root) with the OBJ loader in scene_inputs.h; other
// inputs come from Lcg draws in a fixed order, one value per statement. crates/steam-audio/tests/probes.rs replays the
// same draws.
//
//   tree_<k>_nodes, tree_<k>_data:   the ProbeTree of probe set k (kTreeSizes[k] probes): per node its box (min, max)
//                                    and split coordinate (0 for leaves), and its packed data;
//   tree_<k>_queries<m>:             getInfluencingProbes of 64 points with at most m probes (-1: none);
//   generator_centroid:              generateCentroidProbe for 3 boxes (center, radius);
//   generator_floor_<case>:          generateUniformFloorProbes in boxroom (axis-aligned box) and simplescene (rotated);
//   neighborhood_*:                  ProbeManager::getInfluencingProbes for 48 points with batches [B0, B1, B0], then
//                                    checkOcclusion in boxroom, calcWeights and findNearest; batches are written as
//                                    their index in the manager's list (-1: none);
//   lookup_*:                        findUniqueProbeBatches (as list indices of the batches' first entries, -1 padded),
//                                    lookupEnergyField and lookupReverb for three identifiers, into outputs prefilled
//                                    with 7 (left alone when no probe is valid);
//   batch0, batch1_<step>:           serializeAsRoot of B0, and of B1 after each step of a sequence of changes;
//   batch_sizes:                     serializedSize of every layer of B0, then of B1 at each step.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <baked_reflection_data.h>
#include <baked_reflection_simulator.h>
#include <probe_generator.h>
#include <probe_manager.h>
#include <probe_tree.h>
#include <quaternion.h>
#include <scene.h>
#include <serialized_object.h>

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

void writeMatrix(const std::filesystem::path& path, size_t cols, const std::vector<float>& data)
{
    writeNpy(path, {data.size() / cols, cols}, data.data());
}

void writeMatrix(const std::filesystem::path& path, size_t cols, const std::vector<int32_t>& data)
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

// The box `bounds`, optionally rotated by an Lcg quaternion about its center.
Matrix4x4f boxTransform(Lcg& lcg, const Box& bounds, bool rotate)
{
    Matrix4x4f rotation;
    for (auto i = 0; i < 4; ++i)
        for (auto j = 0; j < 4; ++j)
            rotation(i, j) = (i == j) ? 1.0f : 0.0f;

    if (rotate)
    {
        Quaternionf q;
        q.x = lcg.nextSigned();
        q.y = lcg.nextSigned();
        q.z = lcg.nextSigned();
        q.w = lcg.nextSigned();
        q.normalize();
        const auto r = q.toRotationMatrix();
        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                rotation(i, j) = r(i, j);
    }

    const auto extents = bounds.extents();
    const auto center = bounds.center();

    Matrix4x4f m;
    for (auto i = 0; i < 4; ++i)
        for (auto j = 0; j < 4; ++j)
            m(i, j) = 0.0f;
    for (auto i = 0; i < 3; ++i)
    {
        for (auto j = 0; j < 3; ++j)
            m(i, j) = rotation(i, j) * extents[j];
        m(i, 3) = center[i];
    }
    m(3, 3) = 1.0f;
    return m;
}

void appendProbes(std::vector<float>& out, const ProbeArray& probes)
{
    for (auto i = 0; i < probes.numProbes(); ++i)
    {
        out.push_back(probes[i].influence.center.x());
        out.push_back(probes[i].influence.center.y());
        out.push_back(probes[i].influence.center.z());
        out.push_back(probes[i].influence.radius);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Probe trees
// ---------------------------------------------------------------------------------------------------------------------

const int kTreeSizes[] = {1, 2, 7, 37, 130};

void generateTrees(const std::filesystem::path& out)
{
    Lcg lcg(233);

    auto k = 0;
    for (const auto numProbes : kTreeSizes)
    {
        std::vector<Probe> probes(numProbes);
        for (auto i = 0; i < numProbes; ++i)
        {
            // Every fifth probe shares the center of an earlier one, so the sorts see ties.
            if (i % 5 == 4)
            {
                probes[i].influence.center = probes[i - 3].influence.center;
            }
            else
            {
                const auto v = drawVector(lcg);
                probes[i].influence.center = v * 10.0f;
            }
            const float r = draw01(lcg);
            probes[i].influence.radius = 0.5f + 3.0f * r;
        }

        ProbeTree tree(numProbes, probes.data());
        const auto* nodes = &tree.getRootNode();
        const auto numNodes = 2 * numProbes - 1;

        std::vector<float> nodeValues;
        std::vector<int32_t> nodeData;
        for (auto i = 0; i < numNodes; ++i)
        {
            const auto& node = nodes[i];
            for (auto j = 0; j < 3; ++j)
                nodeValues.push_back(node.box.minCoordinates[j]);
            for (auto j = 0; j < 3; ++j)
                nodeValues.push_back(node.box.maxCoordinates[j]);
            nodeValues.push_back(node.isLeaf() ? 0.0f : node.getSplitCoordinate());
            nodeData.push_back(node.data());
        }

        const auto prefix = "tree_" + std::to_string(k);
        writeMatrix(out / (prefix + "_nodes.npy"), 7, nodeValues);
        writeNpy(out / (prefix + "_data.npy"), nodeData);

        std::vector<int32_t> queries8;
        std::vector<int32_t> queries3;
        for (auto i = 0; i < 64; ++i)
        {
            const auto v = drawVector(lcg);
            const auto point = v * 12.0f;

            int indices[8];
            tree.getInfluencingProbes(point, probes.data(), 8, indices);
            queries8.insert(queries8.end(), indices, indices + 8);

            tree.getInfluencingProbes(point, probes.data(), 3, indices);
            queries3.insert(queries3.end(), indices, indices + 3);
        }
        writeMatrix(out / (prefix + "_queries8.npy"), 8, queries8);
        writeMatrix(out / (prefix + "_queries3.npy"), 3, queries3);

        ++k;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Probe generation
// ---------------------------------------------------------------------------------------------------------------------

void generateGenerator(const std::filesystem::path& out)
{
    Lcg lcg(239);

    Scene emptyScene;
    emptyScene.commit();

    std::vector<float> centroids;
    Box unitBox;
    unitBox.minCoordinates = Vector3f(-1.0f, -1.0f, -1.0f);
    unitBox.maxCoordinates = Vector3f(1.0f, 1.0f, 1.0f);
    for (auto i = 0; i < 3; ++i)
    {
        const auto transform = drawTransform(lcg, unitBox);
        ProbeArray probes;
        ProbeGenerator::generateProbes(emptyScene, transform, ProbeGenerationType::Centroid, 0.0f, 0.0f, probes);
        appendProbes(centroids, probes);
    }
    writeMatrix(out / "generator_centroid.npy", 4, centroids);

    struct FloorCase
    {
        const char* name;
        const char* mesh;
        bool rotate;
        float spacing;
        float height;
    };

    const FloorCase cases[] = {{"boxroom", "boxroom", false, 1.3f, 1.5f}, {"rotated", "simplescene", true, 2.1f, 0.7f}};
    for (const auto& c : cases)
    {
        const auto obj = loadObj(c.mesh);

        Scene scene;
        scene.addStaticMesh(createStaticMesh(scene, obj));
        scene.commit();

        const auto transform = boxTransform(lcg, meshBounds(obj), c.rotate);
        ProbeArray probes;
        ProbeGenerator::generateProbes(scene, transform, ProbeGenerationType::UniformFloor, c.spacing, c.height, probes);

        std::vector<float> values;
        appendProbes(values, probes);
        writeMatrix(out / ("generator_floor_" + std::string(c.name) + ".npy"), 4, values);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Probe batches, neighbourhoods, baked reflections
// ---------------------------------------------------------------------------------------------------------------------

constexpr int kFieldChannels = 4;
constexpr int kFieldBins = 9;

BakedDataIdentifier makeIdentifier(BakedDataVariation variation, const Vector3f& center, float radius)
{
    BakedDataIdentifier identifier;
    identifier.variation = variation;
    identifier.type = BakedDataType::Reflections;
    identifier.endpointInfluence.center = center;
    identifier.endpointInfluence.radius = radius;
    return identifier;
}

// Adds a layer to `batch`: probe i is left unbaked if i % 4 == 3, gets no energy field if i % 4 == 2, and otherwise an
// Lcg field; every baked probe gets an Lcg reverb.
void addLayer(Lcg& lcg, ProbeBatch& batch, const BakedDataIdentifier& identifier, bool hasConvolution,
              bool hasParametric)
{
    auto data = ipl::make_unique<BakedReflectionsData>(identifier, batch.numProbes(), hasConvolution, hasParametric);

    for (auto i = 0; i < batch.numProbes(); ++i)
    {
        if (i % 4 == 3)
            continue;

        if (hasConvolution)
        {
            if (i % 4 == 2)
            {
                data->set(i, unique_ptr<EnergyField>());
            }
            else
            {
                const auto order = static_cast<int>(sqrtf(static_cast<float>(kFieldChannels))) - 1;
                auto field = ipl::make_unique<EnergyField>(kFieldBins * EnergyField::kBinDuration - 1e-4f, order);
                for (auto j = 0; j < kFieldChannels * Bands::kNumBands * kFieldBins; ++j)
                {
                    const float x = draw01(lcg);
                    field->flatData()[j] = x;
                }
                data->set(i, std::move(field));
            }
        }

        if (hasParametric)
        {
            Reverb reverb;
            for (auto j = 0; j < Bands::kNumBands; ++j)
            {
                const float x = draw01(lcg);
                reverb.reverbTimes[j] = 0.1f + 2.0f * x;
            }
            data->set(i, reverb);
        }
    }

    batch.addData(identifier, std::move(data));
}

int listIndex(const ProbeManager& manager, const ProbeBatch* batch)
{
    auto i = 0;
    for (const auto& b : manager.probeBatches())
    {
        if (b.get() == batch)
            return i;
        ++i;
    }
    return -2;
}

void appendSizes(std::vector<int32_t>& sizes, const ProbeBatch& batch)
{
    for (const auto& layer : batch.getData())
        sizes.push_back(static_cast<int32_t>(layer.second->serializedSize()));
}

void generateBatches(const std::filesystem::path& out)
{
    Lcg lcg(241);

    const auto boxroom = loadObj("boxroom");
    const auto bounds = meshBounds(boxroom);

    Scene scene;
    scene.addStaticMesh(createStaticMesh(scene, boxroom));
    scene.commit();

    // B0: 5 x 5 floor probes in boxroom; B1: as many Lcg probes. Both have 25 probes: a batch evaluating another
    // batch's entries looks their probe indices up in its own data (upstream reads out of bounds past its probes).
    auto b0 = ipl::make_shared<ProbeBatch>();
    {
        const auto transform = boxTransform(lcg, bounds, false);
        ProbeArray probes;
        ProbeGenerator::generateProbes(scene, transform, ProbeGenerationType::UniformFloor, 4.0f, 1.5f, probes);
        if (probes.numProbes() != 25)
            throw std::runtime_error("unexpected number of floor probes");
        b0->addProbeArray(probes);
    }

    auto b1 = ipl::make_shared<ProbeBatch>();
    for (auto i = 0; i < 25; ++i)
    {
        const auto center = drawPoint(lcg, bounds);
        const float r = draw01(lcg);
        b1->addProbe(Sphere(center, 1.0f + 3.0f * r));
    }

    const auto reverbId = makeIdentifier(BakedDataVariation::Reverb, Vector3f(0.0f, 0.0f, 0.0f), 0.0f);
    const auto sourceA = drawPoint(lcg, bounds);
    const auto sourceB = Vector3f(sourceA.x(), sourceA.y() + 0.5f, sourceA.z());
    const auto sourceAId = makeIdentifier(BakedDataVariation::StaticSource, sourceA, 5.0f);
    const auto sourceBId = makeIdentifier(BakedDataVariation::StaticSource, sourceB, 5.0f);
    const auto listenerId = makeIdentifier(BakedDataVariation::StaticListener, Vector3f(1.0f, 2.0f, 3.0f), 4.0f);

    addLayer(lcg, *b0, reverbId, true, true);
    addLayer(lcg, *b0, listenerId, false, true);
    addLayer(lcg, *b1, sourceBId, true, false);
    addLayer(lcg, *b1, reverbId, true, true);
    addLayer(lcg, *b1, sourceAId, true, false);

    b0->commit();
    b1->commit();

    ProbeManager manager;
    manager.addProbeBatch(b0);
    manager.addProbeBatch(b1);
    manager.addProbeBatch(b0);
    manager.commit();

    const BakedDataIdentifier lookupIds[] = {reverbId, sourceAId, listenerId};
    constexpr int kNumPoints = 48;
    constexpr int kOutChannels = 9;
    constexpr int kOutBins = 7;

    std::vector<int32_t> influencingBatches, influencingProbes, visibleBatches, visibleProbes, nearest, unique;
    std::vector<float> weights;
    std::vector<float> fields[3];
    std::vector<float> reverbs[3];

    ProbeNeighborhood neighborhood;
    unordered_set<const ProbeBatch*> uniqueBatches;

    auto appendEntries = [&](std::vector<int32_t>& batches, std::vector<int32_t>& probes) {
        for (auto i = 0; i < neighborhood.numProbes(); ++i)
        {
            batches.push_back(neighborhood.batches[i] ? i / ProbeNeighborhood::kMaxProbesPerBatch : -1);
            probes.push_back(neighborhood.probeIndices[i]);
        }
    };

    for (auto p = 0; p < kNumPoints; ++p)
    {
        const auto point = drawPoint(lcg, bounds);

        manager.getInfluencingProbes(point, neighborhood);
        appendEntries(influencingBatches, influencingProbes);

        neighborhood.checkOcclusion(scene, point);
        appendEntries(visibleBatches, visibleProbes);

        neighborhood.calcWeights(point);
        for (auto i = 0; i < neighborhood.numProbes(); ++i)
            weights.push_back(neighborhood.weights[i]);

        nearest.push_back(neighborhood.findNearest(point));

        BakedReflectionSimulator::findUniqueProbeBatches(neighborhood, uniqueBatches);
        std::vector<int32_t> order;
        for (const auto* batch : uniqueBatches)
            order.push_back(listIndex(manager, batch));
        order.resize(2, -1);
        unique.insert(unique.end(), order.begin(), order.end());

        for (auto k = 0; k < 3; ++k)
        {
            EnergyField field(kOutBins * EnergyField::kBinDuration - 1e-4f, 2);
            for (auto j = 0; j < kOutChannels * Bands::kNumBands * kOutBins; ++j)
                field.flatData()[j] = 7.0f;
            BakedReflectionSimulator::lookupEnergyField(lookupIds[k], neighborhood, uniqueBatches, field);
            fields[k].insert(fields[k].end(), field.flatData(), field.flatData() + kOutChannels * Bands::kNumBands * kOutBins);

            Reverb reverb;
            for (auto j = 0; j < Bands::kNumBands; ++j)
                reverb.reverbTimes[j] = 7.0f;
            BakedReflectionSimulator::lookupReverb(lookupIds[k], neighborhood, uniqueBatches, reverb);
            reverbs[k].insert(reverbs[k].end(), reverb.reverbTimes, reverb.reverbTimes + Bands::kNumBands);
        }
    }

    const size_t numEntries = 3 * ProbeNeighborhood::kMaxProbesPerBatch;
    writeMatrix(out / "neighborhood_batches.npy", numEntries, influencingBatches);
    writeMatrix(out / "neighborhood_probes.npy", numEntries, influencingProbes);
    writeMatrix(out / "neighborhood_visible_batches.npy", numEntries, visibleBatches);
    writeMatrix(out / "neighborhood_visible_probes.npy", numEntries, visibleProbes);
    writeMatrix(out / "neighborhood_weights.npy", numEntries, weights);
    writeNpy(out / "neighborhood_nearest.npy", nearest);
    writeMatrix(out / "lookup_unique.npy", 2, unique);
    for (auto k = 0; k < 3; ++k)
    {
        writeMatrix(out / ("lookup_fields_" + std::to_string(k) + ".npy"), kOutChannels * Bands::kNumBands * kOutBins, fields[k]);
        writeMatrix(out / ("lookup_reverbs_" + std::to_string(k) + ".npy"), Bands::kNumBands, reverbs[k]);
    }

    // Serialization, and changes to B1.
    std::vector<int32_t> sizes;
    {
        SerializedObject object;
        b0->serializeAsRoot(object);
        writeBytes(out / "batch0.npy", object);
        appendSizes(sizes, *b0);
    }

    auto serializeB1 = [&](int step) {
        SerializedObject object;
        b1->serializeAsRoot(object);
        writeBytes(out / ("batch1_" + std::to_string(step) + ".npy"), object);
        appendSizes(sizes, *b1);
    };

    serializeB1(0);

    // 1: move, remove and add probes.
    b1->updateProbePosition(4, drawPoint(lcg, bounds));
    b1->removeProbe(5);
    {
        const auto center = drawPoint(lcg, bounds);
        b1->addProbe(Sphere(center, 2.0f));
    }
    serializeB1(1);

    // 2: shrink source A's influence: probes no longer inside lose their fields but stay baked, so their energy fields
    // are written as null offsets.
    b1->updateEndpoint(sourceAId, Sphere(sourceA, 2.0f));
    serializeB1(2);

    // 3: grow it again: probes back inside need baking.
    b1->updateEndpoint(sourceAId, Sphere(sourceA, 8.0f));
    serializeB1(3);

    // 4: move source A: every probe needs baking.
    b1->updateEndpoint(sourceAId, Sphere(sourceB, 5.0f));
    serializeB1(4);

    // 5: turn the reverb layer's convolution off and parametric off and on again.
    {
        auto& data = static_cast<BakedReflectionsData&>((*b1)[reverbId]);
        data.setHasConvolution(false);
        data.setHasParametric(false);
        data.setHasParametric(true);
    }
    serializeB1(5);

    // 6: remove source B's layer, re-bake source A's probe 0 and 1.
    b1->removeData(sourceBId);
    {
        auto& data = static_cast<BakedReflectionsData&>((*b1)[sourceAId]);
        data.set(0, unique_ptr<EnergyField>());
        auto field = ipl::make_unique<EnergyField>(3 * EnergyField::kBinDuration - 1e-4f, 0);
        for (auto j = 0; j < 1 * Bands::kNumBands * 3; ++j)
        {
            const float x = draw01(lcg);
            field->flatData()[j] = x;
        }
        data.set(1, std::move(field));
    }
    serializeB1(6);

    writeNpy(out / "batch_sizes.npy", sizes);
}

void generateProbes(const std::filesystem::path& out)
{
    generateTrees(out);
    generateGenerator(out);
    generateBatches(out);
}

Register probes("probes", generateProbes);

}
