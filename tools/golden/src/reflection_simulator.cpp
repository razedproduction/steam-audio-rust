// Golden data for stage 6.3: reflection_simulator.{h,cpp}.
//
// Scenes are built as in direct_simulator.cpp (a mesh from core/data/meshes with Lcg materials and an instanced
// smallbox); every other input comes from Lcg draws in a fixed order, one value per statement.
// crates/steam-audio/tests/reflection_simulator.rs replays them.
//
// Upstream seeds every thread's RandomNumberGenerator with time(NULL). To make the goldens reproducible, the generator
// reseeds them right after construction with a fixed seed, reaching the private members through explicit template
// instantiation (where access checks do not apply). The Rust simulators take the seed as an argument.
//
// Every run uses a one-thread pool, so jobs run in order on one random number generator. Each output file holds the
// flat data of every output energy field of a sequence of runs on one simulator, concatenated:
//   <scene>_default.npy        ReflectionSimulator: runs 1, 2, 3 (the generator's state carries over between runs);
//   <scene>_batched_<n>.npy    BatchedReflectionSimulator with a ray batch of n: runs 1, 2, 3;
//   <scene>_small.npy          ReflectionSimulator of order 0, 0.25 s and 5 diffuse samples: run 4;
//   <scene>_specular.npy       ReflectionSimulator on the same scene with every scattering coefficient set to 0
//                              (no random choices): run 1.
// The runs are described in kRuns.

#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <energy_field.h>
#include <instanced_mesh.h>
#include <job_graph.h>
#include <reflection_simulator.h>
#include <sampling.h>
#include <scene.h>
#include <static_mesh.h>
#include <thread_pool.h>

#include "generators.h"
#include "npy.h"
#include "scene_inputs.h"

namespace {

using namespace golden;
using namespace ipl;

// --------------------------------------------------------------------------------------------------------------------
// Access to the private random number generators
// --------------------------------------------------------------------------------------------------------------------

std::default_random_engine& engine(RandomNumberGenerator& rng);
void reseed(ReflectionSimulator& simulator, unsigned int seed);
void reseed(BatchedReflectionSimulator& simulator, unsigned int seed);

template <typename Member, Member M>
struct EngineAccess
{
    friend std::default_random_engine& engine(RandomNumberGenerator& rng)
    {
        return rng.*M;
    }
};

template <typename Simulator, typename Member, Member M>
struct ReseedAccess
{
    friend void reseed(Simulator& simulator, unsigned int seed)
    {
        auto& states = simulator.*M;
        for (auto i = 0; i < static_cast<int>(states.size(0)); ++i)
        {
            engine(states[i].rng).seed(seed);
        }
    }
};

template struct EngineAccess<decltype(&RandomNumberGenerator::mGenerator), &RandomNumberGenerator::mGenerator>;
template struct ReseedAccess<ReflectionSimulator, decltype(&ReflectionSimulator::mThreadState),
                             &ReflectionSimulator::mThreadState>;
template struct ReseedAccess<BatchedReflectionSimulator, decltype(&BatchedReflectionSimulator::mThreadState),
                             &BatchedReflectionSimulator::mThreadState>;

// --------------------------------------------------------------------------------------------------------------------
// Runs
// --------------------------------------------------------------------------------------------------------------------

struct FieldSize
{
    float duration;
    int order;
};

struct Run
{
    int numSources;
    int numRays;
    int numBounces;
    float duration;
    int order;
    float irradianceMinDistance;
    std::vector<FieldSize> fields;
};

// Simulators built with kSettings (3 sources at most, so run 1 ignores its 4th source) get runs 1-3: all the sources,
// duration and order; fewer rays and sources, and a duration that stops rays after about 10 m; a single bounce. The
// small simulator gets run 4: more bounces and a longer duration than its fields.
const Run kRuns[] = {
    {4, 1000, 8, 1.0f, 2, 1.0f, {{1.0f, 2}, {0.5f, 1}, {1.5f, 3}}},
    {2, 517, 3, 0.03f, 1, 0.5f, {{1.0f, 2}, {1.0f, 2}}},
    {3, 333, 1, 1.0f, 2, 2.0f, {{1.0f, 2}, {1.0f, 2}, {1.0f, 2}}},
    {4, 300, 16, 2.0f, 0, 1.0f, {{0.25f, 0}, {0.25f, 0}, {0.25f, 0}, {0.25f, 0}}},
};

struct Settings
{
    int maxNumRays;
    int numDiffuseSamples;
    float maxDuration;
    int maxOrder;
    int maxNumSources;
};

const Settings kSettings = {1000, 77, 1.0f, 2, 3};
const Settings kSmallSettings = {300, 5, 0.25f, 0, 4};
constexpr int kBatchSizes[] = {1, 37, 1024};

struct Inputs
{
    CoordinateSpace3f listener;
    CoordinateSpace3f sources[4];
    Directivity directivities[4];
};

void simulate(IReflectionSimulator& simulator, const IScene& scene, const Inputs& inputs, const Run& run,
              std::vector<float>& out)
{
    std::vector<unique_ptr<EnergyField>> fields;
    std::vector<EnergyField*> pointers;
    for (const auto& size : run.fields)
    {
        fields.push_back(ipl::make_unique<EnergyField>(size.duration, size.order));
        pointers.push_back(fields.back().get());
    }

    JobGraph jobGraph;
    ThreadPool threadPool(1);
    simulator.simulate(scene, run.numSources, inputs.sources, 1, &inputs.listener, inputs.directivities, run.numRays,
                       run.numBounces, run.duration, run.order, run.irradianceMinDistance, pointers.data(), jobGraph);
    threadPool.process(jobGraph);

    for (const auto& field : fields)
    {
        const auto n = static_cast<size_t>(field->numChannels()) * Bands::kNumBands * field->numBins();
        out.insert(out.end(), field->flatData(), field->flatData() + n);
    }
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

struct SceneCase
{
    const char* name;
    uint32_t seed;
    unsigned int simulatorSeed;
};

const SceneCase kSceneCases[] = {
    {"boxroom", 601, 1700000001u},
    {"simplescene", 602, 1700000002u},
};

unique_ptr<Scene> buildScene(const ObjMesh& obj, const ObjMesh& smallbox, const Matrix4x4f& transform)
{
    auto boxScene = ipl::make_shared<Scene>();
    boxScene->addStaticMesh(createStaticMesh(*boxScene, smallbox));
    boxScene->commit();

    auto scene = ipl::make_unique<Scene>();
    scene->addStaticMesh(createStaticMesh(*scene, obj));
    scene->addInstancedMesh(scene->createInstancedMesh(boxScene, transform));
    scene->commit();
    return scene;
}

void generateReflectionSimulator(const std::filesystem::path& out)
{
    for (const auto& sceneCase : kSceneCases)
    {
        const std::string name = sceneCase.name;
        Lcg lcg(sceneCase.seed);

        auto obj = loadObj(name);
        auto smallbox = loadObj("smallbox");
        assignMaterials(lcg, obj);
        assignMaterials(lcg, smallbox);

        Box bounds;
        {
            Scene probe;
            auto mesh = createStaticMesh(probe, obj);
            bounds = static_cast<const StaticMesh&>(*mesh).boundingBox();
        }
        const auto transform = drawTransform(lcg, bounds);
        auto scene = buildScene(obj, smallbox, transform);

        Inputs inputs;
        const auto ahead = Vector3f::unitVector(drawVector(lcg));
        const auto listenerPosition = drawInterior(lcg, bounds);
        inputs.listener = CoordinateSpace3f(ahead, listenerPosition);
        for (auto i = 0; i < 4; ++i)
        {
            const auto sourceAhead = Vector3f::unitVector(drawVector(lcg));
            // The last source sits inside the listener's sphere.
            const auto position = (i < 3) ? drawInterior(lcg, bounds) : listenerPosition + Vector3f(0.05f, 0.0f, 0.0f);
            inputs.sources[i] = CoordinateSpace3f(sourceAhead, position);

            const float weight = draw01(lcg);
            const float power = draw01(lcg) * 4.0f;
            inputs.directivities[i] = Directivity((i == 1) ? 0.0f : weight, power);
        }

        const auto seed = sceneCase.simulatorSeed;
        const auto& s = kSettings;

        {
            std::vector<float> rows;
            ReflectionSimulator simulator(s.maxNumRays, s.numDiffuseSamples, s.maxDuration, s.maxOrder,
                                          s.maxNumSources, 1);
            reseed(simulator, seed);
            for (auto i = 0; i < 3; ++i)
                simulate(simulator, *scene, inputs, kRuns[i], rows);
            writeNpy(out / (name + "_default.npy"), rows);
        }

        for (const auto batchSize : kBatchSizes)
        {
            std::vector<float> rows;
            BatchedReflectionSimulator simulator(s.maxNumRays, s.numDiffuseSamples, s.maxDuration, s.maxOrder,
                                                 s.maxNumSources, 1, batchSize);
            reseed(simulator, seed);
            for (auto i = 0; i < 3; ++i)
                simulate(simulator, *scene, inputs, kRuns[i], rows);
            writeNpy(out / (name + "_batched_" + std::to_string(batchSize) + ".npy"), rows);
        }

        {
            const auto& t = kSmallSettings;
            std::vector<float> rows;
            ReflectionSimulator simulator(t.maxNumRays, t.numDiffuseSamples, t.maxDuration, t.maxOrder,
                                          t.maxNumSources, 1);
            reseed(simulator, seed + 1);
            simulate(simulator, *scene, inputs, kRuns[3], rows);
            writeNpy(out / (name + "_small.npy"), rows);
        }

        {
            auto specularObj = obj;
            auto specularBox = smallbox;
            for (auto& m : specularObj.materials)
                m.scattering = 0.0f;
            for (auto& m : specularBox.materials)
                m.scattering = 0.0f;
            auto specularScene = buildScene(specularObj, specularBox, transform);

            std::vector<float> rows;
            ReflectionSimulator simulator(s.maxNumRays, s.numDiffuseSamples, s.maxDuration, s.maxOrder,
                                          s.maxNumSources, 1);
            reseed(simulator, seed);
            simulate(simulator, *specularScene, inputs, kRuns[0], rows);
            writeNpy(out / (name + "_specular.npy"), rows);
        }
    }
}

Register reflectionSimulator("reflection_simulator", generateReflectionSimulator);

}
