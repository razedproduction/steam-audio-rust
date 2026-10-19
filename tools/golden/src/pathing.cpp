// Golden data for stage 7a: path_visibility (ProbeVisibilityTester, ProbeVisibilityGraph), path_finder (PathFinder,
// ProbePath) and path_data (SoundPath, BakedPathData, PathBaker).
//
// The scene is simplescene (core/data/meshes, read with the OBJ loader in scene_inputs.h): a floor, four walls and three
// boxes. Each bake uses its uniform floor probes (an axis-aligned grid, so many paths tie in cost). Other inputs come
// from Lcg draws in a fixed order, one value per statement. crates/steam-audio/tests/pathing.rs replays them. Every
// thread pool has one thread.
//
//   bake<k>_probes:          the probes (center, radius);
//   bake<k>_graph_*:         a ProbeVisibilityGraph built with the bake's parameters, before pruning: per row the offset
//                            of its first entry (rows), then each entry's probe index (indices) and cost (costs);
//   bake<k>_batch:           the batch serialized after PathBaker::bake (bytes);
//   bake<k>_size:            serializedSize of the pathing layer;
//   bake<k>_lookups:         for every pair (start, end), lookupShortestPath's firstProbe, lastProbe, probeAfterFirst,
//                            probeBeforeLast, direct and isValid;
//   bake<k>_lookup_values:   for every pair, distanceInternal and deviationInternal, then for valid paths (0 otherwise)
//                            distance and deviation from start to end, toVirtualSource from start to end, and distance
//                            and toVirtualSource from an Lcg source point (one per start) to end;
//   bake<k>_probe_paths:     for every pair, the probe path lookupShortestPath reconstructs: valid, start, end, the
//                            number of nodes, the nodes;
//   astar_paths:             PathFinder::findShortestPath on bake 1's (pruned) graph for kNumQueries pairs, each with
//                            (simplifyPaths, realTimeVis) = (0, 0), (1, 0), (0, 1), (1, 1): the probe path as above, then
//                            the SoundPath built from it (the five integer fields);
//   astar_values:            for each of those, the SoundPath's distanceInternal and deviationInternal.

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <job_graph.h>
#include <path_data.h>
#include <path_finder.h>
#include <path_visibility.h>
#include <probe_batch.h>
#include <probe_generator.h>
#include <scene.h>
#include <serialized_object.h>
#include <thread_pool.h>

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

// The box `bounds` as a transform (probes.cpp's boxTransform without rotation).
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

void appendProbePath(std::vector<int32_t>& out, const ProbePath& path)
{
    out.push_back(path.valid ? 1 : 0);
    out.push_back(path.start);
    out.push_back(path.end);
    out.push_back(static_cast<int32_t>(path.nodes.size()));
    out.insert(out.end(), path.nodes.begin(), path.nodes.end());
}

void appendSoundPath(std::vector<int32_t>& out, const SoundPath& path)
{
    out.push_back(path.firstProbe);
    out.push_back(path.lastProbe);
    out.push_back(path.probeAfterFirst);
    out.push_back(path.probeBeforeLast);
    out.push_back(path.direct ? 1 : 0);
}

void appendVector(std::vector<float>& out, const Vector3f& v)
{
    out.push_back(v.x());
    out.push_back(v.y());
    out.push_back(v.z());
}

struct BakeCase
{
    float spacing;
    float height;
    int numSamples;
    float radius;
    float threshold;
    float visRange;
    float visRangeRealTime;
    float pathRange;
    bool asymmetricVisRange;
    bool pruneVisGraph;
};

// 0: point-to-point visibility, every range unlimited in practice; 1: volumetric visibility, an asymmetric range along
// a tilted "down", a path range that cuts long paths, and a graph pruned to a shorter real-time range.
const BakeCase kBakes[] = {
    {4.5f, 1.5f, 1, 0.0f, 0.5f, 1000.0f, 1000.0f, 1000.0f, false, false},
    {5.5f, 1.0f, 3, 0.8f, 0.3f, 15.0f, 9.0f, 40.0f, true, true},
};

constexpr int kNumQueries = 48;

void generatePathing(const std::filesystem::path& out)
{
    Lcg lcg(257);

    const auto obj = loadObj("simplescene");
    const auto bounds = meshBounds(obj);

    Scene scene;
    scene.addStaticMesh(createStaticMesh(scene, obj));
    scene.commit();

    BakedDataIdentifier identifier;
    identifier.variation = BakedDataVariation::Dynamic;
    identifier.type = BakedDataType::Pathing;

    shared_ptr<ProbeBatch> lastBatch;
    Vector3f lastDown;

    auto k = 0;
    for (const auto& c : kBakes)
    {
        const auto prefix = "bake" + std::to_string(k);

        ProbeArray probes;
        ProbeGenerator::generateProbes(scene, boxTransform(bounds), ProbeGenerationType::UniformFloor, c.spacing,
                                       c.height, probes);

        auto batch = ipl::make_shared<ProbeBatch>();
        batch->addProbeArray(probes);
        const auto n = batch->numProbes();

        std::vector<float> probeValues;
        for (auto i = 0; i < n; ++i)
        {
            appendVector(probeValues, (*batch)[i].influence.center);
            probeValues.push_back((*batch)[i].influence.radius);
        }
        writeMatrix(out / (prefix + "_probes.npy"), 4, probeValues);

        Vector3f down(0.0f, -1.0f, 0.0f);
        if (c.asymmetricVisRange)
        {
            const float x = lcg.nextSigned();
            const float z = lcg.nextSigned();
            down = Vector3f::unitVector(Vector3f(0.3f * x, -1.0f, 0.3f * z));
        }

        // The graph the bake starts from.
        {
            ProbeVisibilityTester tester(c.numSamples, c.asymmetricVisRange, down);
            ThreadPool pool(1);
            JobGraph jobGraph;
            std::atomic<bool> cancel(false);
            ProbeVisibilityGraph graph(scene, *batch, tester, c.radius, c.threshold, c.visRange, 1, jobGraph, cancel);
            pool.process(jobGraph, [](float) {});

            std::vector<int32_t> rows, indices;
            std::vector<float> costs;
            for (auto i = 0; i < n; ++i)
            {
                rows.push_back(static_cast<int32_t>(indices.size()));
                for (const auto& entry : graph.mAdjacent[i])
                {
                    indices.push_back(entry.index);
                    costs.push_back(entry.cost);
                }
            }
            rows.push_back(static_cast<int32_t>(indices.size()));
            writeNpy(out / (prefix + "_graph_rows.npy"), rows);
            writeNpy(out / (prefix + "_graph_indices.npy"), indices);
            writeNpy(out / (prefix + "_graph_costs.npy"), costs);
        }

        PathBaker::bake(scene, identifier, c.numSamples, c.radius, c.threshold, c.visRange, c.visRangeRealTime,
                        c.pathRange, c.asymmetricVisRange, down, c.pruneVisGraph, 1, *batch);

        {
            SerializedObject object;
            batch->serializeAsRoot(object);
            writeBytes(out / (prefix + "_batch.npy"), object);
        }

        const auto& data = static_cast<const BakedPathData&>((*batch)[identifier]);
        writeNpy(out / (prefix + "_size.npy"), std::vector<int32_t>{static_cast<int32_t>(data.serializedSize())});

        std::vector<int32_t> lookups, probePaths;
        std::vector<float> values;
        for (auto s = 0; s < n; ++s)
        {
            const auto source = drawPoint(lcg, bounds);

            for (auto e = 0; e < n; ++e)
            {
                ProbePath probePath;
                const auto path = data.lookupShortestPath(s, e, &probePath);

                appendSoundPath(lookups, path);
                lookups.push_back(path.isValid() ? 1 : 0);
                appendProbePath(probePaths, probePath);

                values.push_back(path.distanceInternal);
                values.push_back(path.deviationInternal);
                if (path.isValid())
                {
                    values.push_back(path.distance(*batch, s, e));
                    values.push_back(path.deviation(*batch, s, e));
                    appendVector(values, path.toVirtualSource(*batch, s, e));
                    values.push_back(path.distance(*batch, source, e));
                    appendVector(values, path.toVirtualSource(*batch, source, e));
                }
                else
                {
                    values.insert(values.end(), 9, 0.0f);
                }
            }
        }
        writeMatrix(out / (prefix + "_lookups.npy"), 6, lookups);
        writeMatrix(out / (prefix + "_lookup_values.npy"), 11, values);
        writeNpy(out / (prefix + "_probe_paths.npy"), probePaths);

        lastBatch = batch;
        lastDown = down;
        ++k;
    }

    // A* on the last bake's pruned graph.
    {
        const auto& c = kBakes[1];
        const auto& batch = *lastBatch;
        const auto n = batch.numProbes();
        const auto& data = static_cast<const BakedPathData&>(batch[identifier]);

        ProbeVisibilityTester tester(c.numSamples, c.asymmetricVisRange, lastDown);
        PathFinder finder(batch, 1);

        std::vector<int32_t> paths;
        std::vector<float> values;
        for (auto q = 0; q < kNumQueries; ++q)
        {
            auto start = 0;
            auto end = 0;
            if (q > 0)
            {
                start = static_cast<int>(lcg.nextU32() % static_cast<uint32_t>(n));
                end = static_cast<int>(lcg.nextU32() % static_cast<uint32_t>(n));
            }

            for (auto flags = 0; flags < 4; ++flags)
            {
                const auto simplify = (flags & 1) != 0;
                const auto realTimeVis = (flags & 2) != 0;
                const auto probePath = finder.findShortestPath(scene, batch, data.visGraph(), tester, start, end,
                                                               c.radius, c.threshold, c.visRangeRealTime, simplify,
                                                               realTimeVis);
                appendProbePath(paths, probePath);

                const SoundPath soundPath(probePath, batch);
                appendSoundPath(paths, soundPath);
                values.push_back(soundPath.distanceInternal);
                values.push_back(soundPath.deviationInternal);
            }
        }
        writeNpy(out / "astar_paths.npy", paths);
        writeMatrix(out / "astar_values.npy", 2, values);
    }
}

Register pathing("pathing", generatePathing);

}
