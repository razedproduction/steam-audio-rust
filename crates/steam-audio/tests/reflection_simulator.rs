//! Parity tests for stage 6.3 against golden data from `tools/golden/src/reflection_simulator.cpp`.
//!
//! Each scene (a mesh from `core/data/meshes` with `Lcg` materials and an instanced box) is simulated from one listener
//! to four sources, in the runs listed in the generator, on one thread. Upstream seeds its random number generators
//! with the clock; the generator reseeds them, and the simulators here take the same seed. The inputs replay the
//! generator's `Lcg` draws in the same order.
//!
//! On several threads the random numbers a ray gets depend on which thread traces it, as upstream (risk R2): with
//! specular materials only (no random choices) the result matches the one-thread golden up to summation order, and
//! with scattering it matches statistically.

mod common;

use common::{
    ObjMesh, assign_materials, committed_sub_scene, draw_transform, draw_vector, draw01, load_obj,
    static_mesh,
};
use parity::{Lcg, Tolerance, golden};
use steam_audio::bands::NUM_BANDS;
use steam_audio::directivity::Directivity;
use steam_audio::geometry::Aabb;
use steam_audio::math::{CoordinateSpace3, Matrix4, Vector3};
use steam_audio::scene::{InstancedMesh, Scene};
use steam_audio::simulation::{
    BatchedReflectionSimulator, EnergyField, ReflectionSimulationParams, ReflectionSimulator,
    ReflectionSimulatorSettings,
};
use steam_audio::thread_pool::ThreadPool;

struct Run {
    num_sources: usize,
    num_rays: usize,
    num_bounces: usize,
    duration: f32,
    order: usize,
    irradiance_min_distance: f32,
    fields: &'static [(f32, usize)],
}

const RUNS: [Run; 4] = [
    Run {
        num_sources: 4,
        num_rays: 1000,
        num_bounces: 8,
        duration: 1.0,
        order: 2,
        irradiance_min_distance: 1.0,
        fields: &[(1.0, 2), (0.5, 1), (1.5, 3)],
    },
    Run {
        num_sources: 2,
        num_rays: 517,
        num_bounces: 3,
        duration: 0.03,
        order: 1,
        irradiance_min_distance: 0.5,
        fields: &[(1.0, 2), (1.0, 2)],
    },
    Run {
        num_sources: 3,
        num_rays: 333,
        num_bounces: 1,
        duration: 1.0,
        order: 2,
        irradiance_min_distance: 2.0,
        fields: &[(1.0, 2), (1.0, 2), (1.0, 2)],
    },
    Run {
        num_sources: 4,
        num_rays: 300,
        num_bounces: 16,
        duration: 2.0,
        order: 0,
        irradiance_min_distance: 1.0,
        fields: &[(0.25, 0), (0.25, 0), (0.25, 0), (0.25, 0)],
    },
];

fn settings(num_threads: usize) -> ReflectionSimulatorSettings {
    ReflectionSimulatorSettings {
        max_num_rays: 1000,
        num_diffuse_samples: 77,
        max_duration: 1.0,
        max_order: 2,
        max_num_sources: 3,
        num_threads,
    }
}

const SMALL_SETTINGS: ReflectionSimulatorSettings = ReflectionSimulatorSettings {
    max_num_rays: 300,
    num_diffuse_samples: 5,
    max_duration: 0.25,
    max_order: 0,
    max_num_sources: 4,
    num_threads: 1,
};

const BATCH_SIZES: [usize; 3] = [1, 37, 1024];

struct Inputs {
    listener: CoordinateSpace3,
    sources: [CoordinateSpace3; 4],
    directivities: [Directivity; 4],
}

/// Either simulator.
trait Simulate {
    fn run(
        &mut self,
        scene: &Scene,
        params: &ReflectionSimulationParams<'_>,
        fields: &mut [EnergyField],
        pool: &ThreadPool,
    );
}

impl Simulate for ReflectionSimulator {
    fn run(
        &mut self,
        scene: &Scene,
        params: &ReflectionSimulationParams<'_>,
        fields: &mut [EnergyField],
        pool: &ThreadPool,
    ) {
        self.simulate(scene, params, fields, pool);
    }
}

impl Simulate for BatchedReflectionSimulator {
    fn run(
        &mut self,
        scene: &Scene,
        params: &ReflectionSimulationParams<'_>,
        fields: &mut [EnergyField],
        pool: &ThreadPool,
    ) {
        self.simulate(scene, params, fields, pool);
    }
}

/// Runs `run` and appends every output field's data to `out`.
fn simulate(
    simulator: &mut dyn Simulate,
    scene: &Scene,
    inputs: &Inputs,
    run: &Run,
    pool: &ThreadPool,
    out: &mut Vec<f32>,
) {
    let mut fields: Vec<EnergyField> = run
        .fields
        .iter()
        .map(|&(duration, order)| EnergyField::new(duration, order))
        .collect();
    let params = ReflectionSimulationParams {
        sources: &inputs.sources[..run.num_sources],
        directivities: &inputs.directivities[..run.num_sources],
        listener: inputs.listener,
        num_rays: run.num_rays,
        num_bounces: run.num_bounces,
        duration: run.duration,
        order: run.order,
        irradiance_min_distance: run.irradiance_min_distance,
    };
    simulator.run(scene, &params, &mut fields, pool);
    for field in &fields {
        out.extend_from_slice(field.data());
    }
}

/// A point in the middle 70% of the box along each axis.
fn draw_interior(lcg: &mut Lcg, aabb: &Aabb) -> Vector3 {
    let extents = aabb.extents();
    let mut p = Vector3::ZERO;
    for i in 0..3 {
        let t = draw01(lcg) * 0.7 + 0.15;
        p[i] = aabb.min[i] + extents[i] * t;
    }
    p
}

fn build_scene(obj: &ObjMesh, smallbox: &ObjMesh, transform: Matrix4) -> Scene {
    let box_scene = committed_sub_scene(smallbox);
    let mut scene = Scene::new();
    let mesh = scene.insert_static_mesh(static_mesh(obj));
    scene.add_static_mesh(mesh);
    let instance = scene.insert_instanced_mesh(InstancedMesh::new(box_scene, transform));
    scene.add_instanced_mesh(instance);
    scene.commit();
    scene
}

struct Case {
    name: &'static str,
    scene: Scene,
    specular_scene: Scene,
    inputs: Inputs,
    seed: u32,
}

fn case(name: &'static str, lcg_seed: u32, seed: u32) -> Case {
    let mut lcg = Lcg::new(lcg_seed);

    let mut obj = load_obj(name);
    let mut smallbox = load_obj("smallbox");
    assign_materials(&mut lcg, &mut obj);
    assign_materials(&mut lcg, &mut smallbox);

    let bounds = static_mesh(&obj).bounds();
    let transform = draw_transform(&mut lcg, &bounds);
    let scene = build_scene(&obj, &smallbox, transform);

    let ahead = draw_vector(&mut lcg).normalize_or_zero();
    let listener_position = draw_interior(&mut lcg, &bounds);
    let listener = CoordinateSpace3::from_ahead(ahead, listener_position);

    let mut sources = [listener; 4];
    let mut directivities: [Directivity; 4] = Default::default();
    for i in 0..4 {
        let source_ahead = draw_vector(&mut lcg).normalize_or_zero();
        // The last source sits inside the listener's sphere.
        let position = if i < 3 {
            draw_interior(&mut lcg, &bounds)
        } else {
            listener_position + Vector3::new(0.05, 0.0, 0.0)
        };
        sources[i] = CoordinateSpace3::from_ahead(source_ahead, position);

        let weight = draw01(&mut lcg);
        let power = draw01(&mut lcg) * 4.0;
        directivities[i] = Directivity::Dipole {
            weight: if i == 1 { 0.0 } else { weight },
            power,
        };
    }

    for m in obj.materials.iter_mut().chain(&mut smallbox.materials) {
        m.scattering = 0.0;
    }
    let specular_scene = build_scene(&obj, &smallbox, transform);

    Case {
        name,
        scene,
        specular_scene,
        inputs: Inputs {
            listener,
            sources,
            directivities,
        },
        seed,
    }
}

fn boxroom() -> Case {
    case("boxroom", 601, 1_700_000_001)
}

fn simplescene() -> Case {
    case("simplescene", 602, 1_700_000_002)
}

fn check(actual: &[f32], file: &str) {
    let expected = golden::<f32>(&format!("reflection_simulator/{file}"));
    assert_eq!(expected.shape, [actual.len()], "{file}");
    assert!(
        expected.data.iter().any(|&x| x != 0.0),
        "{file}: silent golden"
    );
    let tol = Tolerance::peak_relative(&expected.data, 1e-5);
    if let Err(m) = parity::compare(actual, &expected.data, tol) {
        panic!("{file}: {m}");
    }
}

fn check_default(c: &Case) {
    let pool = ThreadPool::new(1);
    let mut simulator = ReflectionSimulator::new(&settings(1), c.seed);
    let mut rows = Vec::new();
    for run in &RUNS[..3] {
        simulate(&mut simulator, &c.scene, &c.inputs, run, &pool, &mut rows);
    }
    check(&rows, &format!("{}_default", c.name));
}

fn check_batched(c: &Case) {
    let pool = ThreadPool::new(1);
    for batch_size in BATCH_SIZES {
        let mut simulator = BatchedReflectionSimulator::new(&settings(1), batch_size, c.seed);
        let mut rows = Vec::new();
        for run in &RUNS[..3] {
            simulate(&mut simulator, &c.scene, &c.inputs, run, &pool, &mut rows);
        }
        check(&rows, &format!("{}_batched_{batch_size}", c.name));
    }
}

fn check_small(c: &Case) {
    let pool = ThreadPool::new(1);
    let mut simulator = ReflectionSimulator::new(&SMALL_SETTINGS, c.seed + 1);
    let mut rows = Vec::new();
    simulate(
        &mut simulator,
        &c.scene,
        &c.inputs,
        &RUNS[3],
        &pool,
        &mut rows,
    );
    check(&rows, &format!("{}_small", c.name));
}

/// Without scattering nothing is random, so several threads give the one-thread result up to summation order.
fn check_specular(c: &Case, num_threads: usize) {
    let pool = ThreadPool::new(num_threads);
    let mut simulator = ReflectionSimulator::new(&settings(num_threads), c.seed);
    let mut rows = Vec::new();
    simulate(
        &mut simulator,
        &c.specular_scene,
        &c.inputs,
        &RUNS[0],
        &pool,
        &mut rows,
    );
    check(&rows, &format!("{}_specular", c.name));
}

#[test]
fn reflection_simulator_boxroom() {
    let c = boxroom();
    check_default(&c);
    check_small(&c);
    check_specular(&c, 1);
}

#[test]
fn reflection_simulator_simplescene() {
    let c = simplescene();
    check_default(&c);
    check_small(&c);
    check_specular(&c, 1);
}

#[test]
fn batched_reflection_simulator_boxroom() {
    check_batched(&boxroom());
}

#[test]
fn batched_reflection_simulator_simplescene() {
    check_batched(&simplescene());
}

#[test]
fn specular_on_several_threads() {
    for c in [boxroom(), simplescene()] {
        check_specular(&c, 4);
    }
}

/// Total energy per band of each field's first channel.
fn band_totals(data: &[f32], fields: &[(f32, usize)]) -> Vec<f64> {
    let mut totals = Vec::new();
    let mut offset = 0;
    for &(duration, order) in fields {
        let field = EnergyField::new(duration, order);
        let num_bins = field.num_bins();
        for band in 0..NUM_BANDS {
            let start = offset + band * num_bins;
            totals.push(
                data[start..start + num_bins]
                    .iter()
                    .map(|&x| f64::from(x))
                    .sum(),
            );
        }
        offset += field.data().len();
    }
    totals
}

/// With scattering, several threads draw other random numbers than one thread (whose results are bit-exact against
/// upstream): with enough rays, the energy per band of each field agrees within 0.5 dB (risk R2's statistical check;
/// measured at most 0.2 dB over 10 runs). The golden runs' 1000 rays are too few for that (they differ by up to 1 dB,
/// and 16384 rays by up to 0.38 dB).
#[test]
fn scattering_on_several_threads_agrees_statistically() {
    const NUM_RAYS: usize = 32_768;
    let run = Run {
        num_rays: NUM_RAYS,
        ..RUNS[0]
    };
    for c in [boxroom(), simplescene()] {
        let totals = |num_threads| {
            let pool = ThreadPool::new(num_threads);
            let settings = ReflectionSimulatorSettings {
                max_num_rays: NUM_RAYS,
                ..settings(num_threads)
            };
            let mut simulator = ReflectionSimulator::new(&settings, c.seed);
            let mut rows = Vec::new();
            simulate(&mut simulator, &c.scene, &c.inputs, &run, &pool, &mut rows);
            band_totals(&rows, run.fields)
        };
        let one = totals(1);
        let four = totals(4);
        for (a, e) in four.iter().zip(&one) {
            let db = 10.0 * (a / e).log10();
            assert!(db.abs() < 0.5, "{}: {a} vs {e} ({db:.3} dB)", c.name);
        }
    }
}

/// The same seed on one thread gives the same result; another seed gives another.
#[test]
fn seeded_runs_are_reproducible() {
    let c = boxroom();
    let pool = ThreadPool::new(1);
    let run_with = |seed| {
        let mut simulator = ReflectionSimulator::new(&settings(2), seed);
        let mut rows = Vec::new();
        simulate(
            &mut simulator,
            &c.scene,
            &c.inputs,
            &RUNS[0],
            &pool,
            &mut rows,
        );
        rows
    };
    assert_eq!(run_with(7), run_with(7));
    assert_ne!(run_with(7), run_with(8));
}

#[test]
fn cancelled_pool_leaves_silent_fields() {
    let c = boxroom();
    let pool = ThreadPool::new(1);
    pool.cancel();
    let mut simulator = ReflectionSimulator::new(&settings(1), c.seed);
    let mut rows = Vec::new();
    simulate(
        &mut simulator,
        &c.scene,
        &c.inputs,
        &RUNS[0],
        &pool,
        &mut rows,
    );
    assert!(rows.iter().all(|&x| x == 0.0));
}
