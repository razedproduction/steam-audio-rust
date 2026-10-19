//! Parity tests for stage 6.4c against golden data from `tools/golden/src/simulation.cpp`: the reflection baker and the
//! simulation manager.
//!
//! The scene is `boxroom` with `Lcg` materials and its 25 floor probes. Bakes run one after another into one batch, on
//! one simulator; three managers (convolution, parametric, hybrid) then simulate six sources for four frames, looking
//! baked sources up in that batch. Upstream seeds its reflection simulators with the clock; the generator reseeds them
//! and the simulators here take the same seeds. Inputs replay the generator's `Lcg` draws in the same order.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{ObjMesh, assign_materials, draw_vector, draw01, load_obj, static_mesh};
use parity::{Lcg, Tolerance, golden};
use steam_audio::air_absorption::AirAbsorptionModel;
use steam_audio::baking::{
    BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation, ReflectionBakeParams,
    ReflectionBaker,
};
use steam_audio::bands::NUM_BANDS;
use steam_audio::directivity::Directivity;
use steam_audio::distance_attenuation::DistanceAttenuationModel;
use steam_audio::effects::{OverlapSaveFir, OverlapSavePartitioner};
use steam_audio::geometry::Ray;
use steam_audio::geometry::{Aabb, Sphere};
use steam_audio::math::{CoordinateSpace3, Matrix4, Vector3};
use steam_audio::probes::{Probe, ProbeBatch, ProbeGeneration, generate_probes};
use steam_audio::random::CRand;
use steam_audio::scene::{Hit, RayTracer, Scene};
use steam_audio::simulation::{
    AnyReflectionSimulator, BatchedReflectionSimulator, DirectSimulationFlags, EnergyField,
    IndirectEffectType, OcclusionType, ReconstructionType, ReflectionSimulationParams,
    ReflectionSimulator, ReflectionSimulatorSettings, SceneType, SharedDirectSimulationInputs,
    SharedReflectionSimulationInputs, SharedSimulationData, SimulationData, SimulationManager,
    SimulationScene, SimulationSettings,
};
use steam_audio::thread_pool::ThreadPool;

#[track_caller]
fn check<T: parity::Element + Into<f64> + Copy>(name: &str, actual: &[T], tol: Tolerance) {
    let expected = golden::<T>(&format!("simulation/{name}"));
    if let Err(m) = parity::compare(actual, &expected.data, tol) {
        panic!("{name}: {m}");
    }
}

fn mesh_bounds(obj: &ObjMesh) -> Aabb {
    let mut bounds = Aabb::EMPTY;
    for v in &obj.vertices {
        bounds.min = Vector3::new(
            bounds.min.x.min(v.x),
            bounds.min.y.min(v.y),
            bounds.min.z.min(v.z),
        );
        bounds.max = Vector3::new(
            bounds.max.x.max(v.x),
            bounds.max.y.max(v.y),
            bounds.max.z.max(v.z),
        );
    }
    bounds
}

/// The generator's `boxTransform`: the box `bounds`, not rotated.
fn box_transform(bounds: &Aabb) -> Matrix4 {
    let extents = bounds.extents();
    let center = bounds.center();
    let mut m = Matrix4::ZERO;
    for i in 0..3 {
        m[(i, i)] = extents[i];
        m[(i, 3)] = center[i];
    }
    m[(3, 3)] = 1.0;
    m
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

fn identifier(variation: BakedDataVariation, center: Vector3, radius: f32) -> BakedDataIdentifier {
    BakedDataIdentifier {
        variation,
        data_type: BakedDataType::Reflections,
        endpoint_influence: Sphere::new(center, radius),
    }
}

/// What the generator builds before baking.
struct Setup {
    lcg: Lcg,
    bounds: Aabb,
    scene: Scene,
    probes: Vec<Probe>,
    batch: ProbeBatch,
    reverb_id: BakedDataIdentifier,
    source_id: BakedDataIdentifier,
    listener_id: BakedDataIdentifier,
}

fn setup() -> Setup {
    let mut lcg = Lcg::new(641);

    let mut obj = load_obj("boxroom");
    assign_materials(&mut lcg, &mut obj);
    let bounds = mesh_bounds(&obj);

    let mut scene = Scene::new();
    let mesh = scene.insert_static_mesh(static_mesh(&obj));
    scene.add_static_mesh(mesh);
    scene.commit();

    let mut probes = Vec::new();
    generate_probes(
        &scene,
        &box_transform(&bounds),
        ProbeGeneration::UniformFloor {
            spacing: 4.0,
            height: 1.5,
        },
        &mut probes,
    );
    assert_eq!(probes.len(), 25);

    let mut batch = ProbeBatch::new();
    batch.add_probes(&probes);
    batch.commit();

    let reverb_id = identifier(BakedDataVariation::Reverb, Vector3::ZERO, 0.0);
    let source_point = probes[0].influence.center + Vector3::new(1.0, 0.2, 1.5);
    let source_id = identifier(BakedDataVariation::StaticSource, source_point, 12.0);
    let listener_point = probes[6].influence.center + Vector3::new(0.5, 0.0, 0.5);
    let listener_id = identifier(BakedDataVariation::StaticListener, listener_point, 4.5);

    Setup {
        lcg,
        bounds,
        scene,
        probes,
        batch,
        reverb_id,
        source_id,
        listener_id,
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Baking
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Layer {
    Reverb,
    Source,
    Listener,
}

struct Bake {
    layer: Layer,
    convolution: bool,
    parametric: bool,
    num_rays: usize,
    num_bounces: usize,
    sim_duration: f32,
    bake_duration: f32,
    order: usize,
    irradiance_min_distance: f32,
    /// Number of progress reports after which the bake is cancelled; 0: never.
    cancel_after: usize,
}

#[allow(clippy::too_many_arguments)]
const fn bake(
    layer: Layer,
    convolution: bool,
    parametric: bool,
    num_rays: usize,
    num_bounces: usize,
    sim_duration: f32,
    bake_duration: f32,
    order: usize,
    irradiance_min_distance: f32,
    cancel_after: usize,
) -> Bake {
    Bake {
        layer,
        convolution,
        parametric,
        num_rays,
        num_bounces,
        sim_duration,
        bake_duration,
        order,
        irradiance_min_distance,
        cancel_after,
    }
}

const BAKES: [Bake; 5] = [
    bake(Layer::Reverb, true, true, 256, 4, 0.2, 0.2, 1, 1.0, 0),
    bake(Layer::Source, true, false, 200, 3, 0.2, 0.15, 1, 0.5, 0),
    bake(Layer::Listener, false, true, 128, 2, 0.15, 0.15, 0, 1.0, 0),
    bake(Layer::Listener, true, true, 128, 2, 0.15, 0.25, 1, 1.0, 0),
    bake(Layer::Source, true, false, 128, 6, 0.2, 0.2, 1, 1.0, 4),
];

/// Compares a baked batch with the golden one: byte for byte on Windows; elsewhere libm (`powf`, `sinf`, `cosf`,
/// `logf`) may differ by an ulp, so the layers are deserialized and compared with tolerances: flags `EXACT`, energy
/// fields `peak_relative(1e-5)` per field, reverbs `F32`.
#[track_caller]
fn check_baked_batch(name: &str, batch: &ProbeBatch) {
    let golden_bytes = golden::<u8>(&format!("simulation/{name}")).data;
    if cfg!(windows) {
        check(name, &batch.serialize(), Tolerance::EXACT);
    }

    let expected = ProbeBatch::deserialize(&golden_bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(batch.probes(), expected.probes(), "{name}");
    let layers: Vec<_> = batch.data_layers().collect();
    let expected_layers: Vec<_> = expected.data_layers().collect();
    assert_eq!(layers.len(), expected_layers.len(), "{name}");
    for ((id, data), (expected_id, expected_data)) in layers.into_iter().zip(expected_layers) {
        assert_eq!(id, expected_id, "{name}");
        let (BakedData::Reflections(data), BakedData::Reflections(expected_data)) =
            (data, expected_data)
        else {
            panic!("{name}: not reflections");
        };
        assert_eq!(
            data.has_convolution(),
            expected_data.has_convolution(),
            "{name}"
        );
        assert_eq!(
            data.has_parametric(),
            expected_data.has_parametric(),
            "{name}"
        );
        for i in 0..batch.num_probes() {
            assert_eq!(
                data.needs_update(i),
                expected_data.needs_update(i),
                "{name} probe {i}"
            );
            // Only what is serialized: fields and reverbs of baked probes.
            if data.needs_update(i) {
                continue;
            }
            match (data.energy_field(i), expected_data.energy_field(i)) {
                (Some(a), Some(e)) => {
                    assert_eq!(
                        (a.num_channels(), a.num_bins()),
                        (e.num_channels(), e.num_bins())
                    );
                    if let Err(m) = parity::compare(
                        a.data(),
                        e.data(),
                        Tolerance::peak_relative(e.data(), 1e-5),
                    ) {
                        panic!("{name} probe {i} field: {m}");
                    }
                }
                (a, e) => assert_eq!(a.is_some(), e.is_some(), "{name} probe {i} field"),
            }
            if let (Some(a), Some(e)) = (data.reverb(i), expected_data.reverb(i)) {
                if let Err(m) = parity::compare(&a.reverb_times, &e.reverb_times, Tolerance::F32) {
                    panic!("{name} probe {i} reverb: {m}");
                }
            }
        }
    }
}

/// Runs the generator's bakes into `setup.batch`; `check_goldens` compares each step with the goldens.
fn run_bakes(setup: &mut Setup, check_goldens: bool) {
    let settings = ReflectionSimulatorSettings {
        max_num_rays: 256,
        num_diffuse_samples: 32,
        max_duration: 0.25,
        max_order: 1,
        max_num_sources: 1,
        num_threads: 1,
    };
    let mut simulator =
        AnyReflectionSimulator::from(ReflectionSimulator::new(&settings, 1700000201));
    let pool = ThreadPool::new(1);
    let baker = ReflectionBaker::new();

    for (k, b) in BAKES.iter().enumerate() {
        let identifier = match b.layer {
            Layer::Reverb => setup.reverb_id,
            Layer::Source => setup.source_id,
            Layer::Listener => setup.listener_id,
        };
        let params = ReflectionBakeParams {
            identifier,
            bake_convolution: b.convolution,
            bake_parametric: b.parametric,
            num_rays: b.num_rays,
            num_bounces: b.num_bounces,
            sim_duration: b.sim_duration,
            bake_duration: b.bake_duration,
            order: b.order,
            irradiance_min_distance: b.irradiance_min_distance,
        };

        let mut progress = Vec::new();
        baker.bake(
            &setup.scene,
            &mut simulator,
            &params,
            &pool,
            &mut setup.batch,
            |p| {
                progress.push(p);
                if progress.len() == b.cancel_after {
                    baker.cancel();
                }
            },
        );

        if check_goldens {
            check_baked_batch(&format!("bake_{k}"), &setup.batch);
            check(&format!("bake_{k}_progress"), &progress, Tolerance::EXACT);
        }
    }
}

#[test]
fn reflection_baking() {
    let mut setup = setup();
    run_bakes(&mut setup, true);

    // A cancel while no bake runs is ignored.
    let baker = ReflectionBaker::new();
    baker.cancel();
    let settings = ReflectionSimulatorSettings {
        max_num_rays: 16,
        num_diffuse_samples: 4,
        max_duration: 0.1,
        max_order: 0,
        max_num_sources: 1,
        num_threads: 1,
    };
    let mut simulator = AnyReflectionSimulator::from(ReflectionSimulator::new(&settings, 1));
    let mut last = 0.0;
    baker.bake(
        &setup.scene,
        &mut simulator,
        &ReflectionBakeParams {
            identifier: setup.reverb_id,
            bake_convolution: false,
            bake_parametric: true,
            num_rays: 16,
            num_bounces: 1,
            sim_duration: 0.1,
            bake_duration: 0.1,
            order: 0,
            irradiance_min_distance: 1.0,
        },
        &ThreadPool::new(1),
        &mut setup.batch,
        |p| last = p,
    );
    assert_eq!(last, 1.0);
}

// ---------------------------------------------------------------------------------------------------------------------
// Simulation manager
// ---------------------------------------------------------------------------------------------------------------------

const MAX_DURATION: f32 = 0.08;
const MAX_ORDER: usize = 1;
const SAMPLING_RATE: u32 = 24000;
const FRAME_SIZE: usize = 256;
const NUM_SOURCES: usize = 6;
const NUM_FRAMES: usize = 4;

struct ManagerCase {
    indirect_type: IndirectEffectType,
    name: &'static str,
    rand_seed: u32,
    simulator_seed: u32,
}

const MANAGERS: [ManagerCase; 3] = [
    ManagerCase {
        indirect_type: IndirectEffectType::Convolution,
        name: "convolution",
        rand_seed: 11,
        simulator_seed: 1700000301,
    },
    ManagerCase {
        indirect_type: IndirectEffectType::Parametric,
        name: "parametric",
        rand_seed: 12,
        simulator_seed: 1700000302,
    },
    ManagerCase {
        indirect_type: IndirectEffectType::Hybrid,
        name: "hybrid",
        rand_seed: 13,
        simulator_seed: 1700000303,
    },
];

fn settings(scene_type: SceneType, indirect_type: IndirectEffectType) -> SimulationSettings {
    SimulationSettings {
        enable_direct: true,
        enable_indirect: true,
        enable_pathing: false,
        scene_type,
        indirect_type,
        max_num_occlusion_samples: 16,
        max_num_rays: 256,
        num_diffuse_samples: 32,
        max_duration: MAX_DURATION,
        max_order: MAX_ORDER,
        max_num_sources: 4,
        num_threads: 1,
        ray_batch_size: 16,
        num_vis_samples: 1,
        asymmetric_vis_range: false,
        down: Vector3::new(0.0, -1.0, 0.0),
        sampling_rate: SAMPLING_RATE,
        frame_size: FRAME_SIZE,
    }
}

struct SourceInputs {
    source: CoordinateSpace3,
    directivity: Directivity,
    air_coefficients: [f32; NUM_BANDS],
}

fn shared_inputs(
    listener: CoordinateSpace3,
    frame: usize,
) -> (
    SharedDirectSimulationInputs,
    SharedReflectionSimulationInputs,
) {
    (
        SharedDirectSimulationInputs { listener },
        SharedReflectionSimulationInputs {
            listener,
            num_rays: 256,
            num_bounces: 4,
            duration: if frame == 2 { 0.06 } else { MAX_DURATION },
            order: if frame == 3 { 0 } else { 1 },
            irradiance_min_distance: 1.0,
            reconstruction_type: if frame == 2 {
                ReconstructionType::Gaussian
            } else {
                ReconstructionType::Linear
            },
        },
    )
}

fn lock(source: &SharedSimulationData) -> std::sync::MutexGuard<'_, SimulationData> {
    source.lock().unwrap()
}

#[test]
fn simulation_manager() {
    let mut setup = setup();
    run_bakes(&mut setup, false);
    let batch = Arc::new(setup.batch.clone());
    let probes = &setup.probes;
    let lcg = &mut setup.lcg;

    let listener0 = probes[12].influence.center + Vector3::new(0.3, 0.2, -0.4);
    let listener_ahead = draw_vector(lcg).normalize_or_zero();
    let listener1 = listener0 + Vector3::new(1.5, 0.0, 0.7);

    let listener_point = setup.listener_id.endpoint_influence.center;
    let mut sources = Vec::new();
    for i in 0..NUM_SOURCES {
        let position = match i {
            2 => probes[6].influence.center + Vector3::new(0.2, 0.1, 0.3),
            3 => listener_point + Vector3::new(0.1, 0.0, -0.2),
            _ => draw_interior(lcg, &setup.bounds),
        };
        let ahead = draw_vector(lcg).normalize_or_zero();
        let weight = draw01(lcg);
        let power = draw01(lcg) * 4.0;
        let air_coefficients = std::array::from_fn(|_| draw01(lcg) * 0.05);
        sources.push(SourceInputs {
            source: CoordinateSpace3::from_ahead(ahead, position),
            directivity: Directivity::Dipole { weight, power },
            air_coefficients,
        });
    }
    let moved_source0 = sources[0].source.origin + Vector3::new(0.5, 0.0, -0.3);

    for case in &MANAGERS {
        let ty = case.indirect_type;
        let settings = settings(SceneType::Default, ty);
        let mut manager = SimulationManager::new(
            &settings,
            case.simulator_seed,
            &mut CRand::new(case.rand_seed),
        );
        manager.set_scene(Some(SimulationScene::Default(setup.scene.clone())));
        manager.add_probe_batch(Arc::clone(&batch));

        let data: Vec<SharedSimulationData> = (0..NUM_SOURCES)
            .map(|i| {
                let mut s = SimulationData::new(&settings, true, false);
                let input = &sources[i];
                let air = AirAbsorptionModel::Exponential {
                    coefficients: input.air_coefficients,
                };

                s.direct_inputs.flags = DirectSimulationFlags::DISTANCE_ATTENUATION
                    | DirectSimulationFlags::AIR_ABSORPTION
                    | DirectSimulationFlags::DELAY
                    | DirectSimulationFlags::OCCLUSION;
                s.direct_inputs.source = input.source;
                s.direct_inputs.directivity = input.directivity.clone();
                s.direct_inputs.air_absorption_model = air.clone();

                s.reflection_inputs.enabled = i != 5;
                s.reflection_inputs.source = input.source;
                s.reflection_inputs.directivity = input.directivity.clone();
                s.reflection_inputs.baked = i >= 2;

                let full = s.direct_inputs.flags
                    | DirectSimulationFlags::DIRECTIVITY
                    | DirectSimulationFlags::TRANSMISSION;
                match i {
                    0 => {
                        s.direct_inputs.flags = full;
                        s.direct_inputs.num_transmission_rays = 3;
                        s.reflection_inputs.transition_time = 0.05;
                    }
                    1 => {
                        s.direct_inputs.flags = full;
                        s.direct_inputs.distance_attenuation_model =
                            DistanceAttenuationModel::InverseDistance { min_distance: 2.0 };
                        s.direct_inputs.occlusion_type = OcclusionType::Volumetric;
                        s.direct_inputs.occlusion_radius = 1.0;
                        s.direct_inputs.num_transmission_rays = 2;
                        s.reflection_inputs.distance_attenuation_model =
                            DistanceAttenuationModel::InverseDistance { min_distance: 2.0 };
                        s.reflection_inputs.air_absorption_model = air;
                        s.reflection_inputs.reverb_scale = [1.2, 0.8, 1.0];
                        s.reflection_inputs.transition_time = 0.04;
                        s.reflection_inputs.overlap_fraction = 0.5;
                    }
                    2 => s.reflection_inputs.baked_data_identifier = setup.reverb_id,
                    3 => {
                        s.reflection_inputs.baked_data_identifier = setup.listener_id;
                        s.reflection_inputs.reverb_scale = [0.9, 1.1, 1.3];
                    }
                    4 => s.reflection_inputs.baked_data_identifier = setup.source_id,
                    _ => {
                        s.direct_inputs.flags = DirectSimulationFlags::DISTANCE_ATTENUATION
                            | DirectSimulationFlags::DIRECTIVITY
                            | DirectSimulationFlags::DELAY;
                    }
                }

                Arc::new(Mutex::new(s))
            })
            .collect();
        for s in &data {
            manager.add_source(Arc::clone(s));
        }
        manager.commit();

        // The FIR readers, and a partitioner replaying the manager's partitions (its partial last block keeps stale
        // samples from the previous source, R13).
        let mut readers: Vec<_> = data
            .iter()
            .map(|s| lock(s).reflection_outputs.overlap_save_fir.take())
            .collect();
        let mut partitioner = OverlapSavePartitioner::new(FRAME_SIZE);
        let mut expected_fir = OverlapSaveFir::new(4, 1920, FRAME_SIZE);

        let (mut direct, mut fields, mut outputs, mut irs) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        for frame in 0..NUM_FRAMES {
            let position = if frame == 3 { listener1 } else { listener0 };
            let listener = CoordinateSpace3::from_ahead(listener_ahead, position);
            let (shared_direct, shared_reflection) = shared_inputs(listener, frame);
            manager.set_shared_direct_inputs(shared_direct);
            manager.set_shared_reflection_inputs(shared_reflection);

            if frame == 2 {
                let moved = CoordinateSpace3::from_ahead(sources[0].source.ahead, moved_source0);
                lock(&data[0]).direct_inputs.source = moved;
                lock(&data[0]).reflection_inputs.source = moved;
                lock(&data[1]).reflection_inputs.distance_attenuation_model =
                    DistanceAttenuationModel::InverseDistance { min_distance: 3.0 };
                manager.remove_source(&data[5]);
                manager.commit();
            }
            if frame == 3 {
                lock(&data[1])
                    .reflection_inputs
                    .distance_attenuation_model_dirty = true;
            }

            manager.simulate_direct();
            manager.simulate_indirect();

            let num_channels = if frame == 3 { 1 } else { 4 };
            let num_samples = (shared_reflection.duration * SAMPLING_RATE as f32).ceil() as usize;

            for (i, s) in data.iter().enumerate() {
                let s = lock(s);
                let p = &s.direct_path;
                direct.push(p.distance_attenuation);
                direct.extend_from_slice(&p.air_absorption);
                direct.push(p.delay);
                direct.push(p.occlusion);
                direct.extend_from_slice(&p.transmission);
                direct.push(p.directivity);

                fields.extend_from_slice(s.energy_field().unwrap().data());

                let out = &s.reflection_outputs;
                outputs.extend_from_slice(&out.reverb.reverb_times);
                if ty == IndirectEffectType::Hybrid {
                    outputs.extend_from_slice(&out.hybrid_eq);
                    outputs.push(out.hybrid_delay as f32);
                }
                if ty != IndirectEffectType::Parametric {
                    outputs.push(out.num_channels as f32);
                    outputs.push(out.num_samples as f32);

                    let ir = s.impulse_response().unwrap();
                    irs.extend_from_slice(ir.data());

                    // Every enabled source publishes its partitioned IR each frame.
                    let reader = readers[i].as_mut().unwrap();
                    assert_eq!(
                        reader.update_read_buffer(),
                        i != 5,
                        "{} frame {frame}",
                        case.name
                    );
                    if i != 5 {
                        partitioner.partition(ir, num_channels, num_samples, &mut expected_fir);
                        let fir = reader.read_buffer();
                        for c in 0..4 {
                            for b in 0..fir.num_blocks() {
                                assert_eq!(
                                    fir.block(c, b),
                                    expected_fir.block(c, b),
                                    "{} frame {frame}",
                                    case.name
                                );
                            }
                        }
                    }
                } else {
                    assert!(readers[i].is_none());
                }
            }
        }

        let prefix = format!("manager_{}", case.name);
        check(&format!("{prefix}_direct"), &direct, Tolerance::F32);
        let fields_golden = golden::<f32>(&format!("simulation/{prefix}_fields")).data;
        check(
            &format!("{prefix}_fields"),
            &fields,
            Tolerance::peak_relative(&fields_golden, 1e-5),
        );
        check(&format!("{prefix}_outputs"), &outputs, Tolerance::F32);
        if ty != IndirectEffectType::Parametric {
            let irs_golden = golden::<f32>(&format!("simulation/{prefix}_irs")).data;
            check(
                &format!("{prefix}_irs"),
                &irs,
                Tolerance::peak_relative(&irs_golden, 1e-5),
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Custom scenes
// ---------------------------------------------------------------------------------------------------------------------

/// A user's ray tracer: the crate's scene behind the trait, counting queries.
struct Custom {
    scene: Scene,
    queries: Arc<AtomicUsize>,
}

impl RayTracer for Custom {
    fn closest_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> Hit {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.scene.closest_hit(ray, min_distance, max_distance)
    }

    fn any_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> bool {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.scene.any_hit(ray, min_distance, max_distance)
    }
}

/// With a custom scene the manager traces rays in batches, and the scene always counts as changed, so fields are
/// never averaged over frames: each frame's field is a fresh simulation.
#[test]
fn custom_scene_never_accumulates() {
    let setup = setup();
    let settings = settings(SceneType::Custom, IndirectEffectType::Parametric);
    let queries = Arc::new(AtomicUsize::new(0));
    let mut manager = SimulationManager::new(&settings, 7, &mut CRand::default());
    manager.set_scene(Some(SimulationScene::Custom(Box::new(Custom {
        scene: setup.scene.clone(),
        queries: Arc::clone(&queries),
    }))));

    let source = CoordinateSpace3::canonical_at(setup.probes[7].influence.center);
    let listener = CoordinateSpace3::canonical_at(setup.probes[12].influence.center);
    let mut s = SimulationData::new(&settings, true, false);
    s.reflection_inputs.enabled = true;
    s.reflection_inputs.source = source;
    let s = Arc::new(Mutex::new(s));
    manager.add_source(Arc::clone(&s));
    manager.add_source(Arc::clone(&s));
    manager.commit();
    assert_eq!(manager.sources().len(), 1);

    let (_, shared) = shared_inputs(listener, 0);
    manager.set_shared_reflection_inputs(shared);

    let simulator_settings = ReflectionSimulatorSettings {
        max_num_rays: 256,
        num_diffuse_samples: 32,
        max_duration: MAX_DURATION,
        max_order: MAX_ORDER,
        max_num_sources: 4,
        num_threads: 1,
    };
    let mut simulator = BatchedReflectionSimulator::new(&simulator_settings, 16, 7);
    let pool = ThreadPool::new(1);
    let mut expected = [EnergyField::new(MAX_DURATION, MAX_ORDER)];

    for _ in 0..3 {
        manager.simulate_indirect();
        simulator.simulate(
            &setup.scene,
            &ReflectionSimulationParams {
                sources: &[source],
                directivities: &[Directivity::default()],
                listener,
                num_rays: shared.num_rays,
                num_bounces: shared.num_bounces,
                duration: shared.duration,
                order: shared.order,
                irradiance_min_distance: shared.irradiance_min_distance,
            },
            &mut expected,
            &pool,
        );
        assert_eq!(lock(&s).energy_field().unwrap().data(), expected[0].data());
    }
    assert!(queries.load(Ordering::Relaxed) > 0);
}

/// With probe caching for missing probes, a baked source without a valid probe keeps its previous outputs.
#[test]
fn probe_caching_keeps_outputs_without_probes() {
    let setup = setup();
    let settings = settings(SceneType::Default, IndirectEffectType::Convolution);
    let mut manager = SimulationManager::new(&settings, 1, &mut CRand::default());
    manager.set_scene(Some(SimulationScene::Default(setup.scene.clone())));
    manager.set_enable_probe_caching_for_missing_probes(true);
    manager.add_probe_batch(Arc::new(setup.batch.clone()));

    let mut s = SimulationData::new(&settings, true, false);
    s.reflection_inputs.enabled = true;
    s.reflection_inputs.baked = true;
    s.reflection_inputs.baked_data_identifier = setup.reverb_id;
    let mut reader = s.reflection_outputs.overlap_save_fir.take().unwrap();
    let s = Arc::new(Mutex::new(s));
    manager.add_source(Arc::clone(&s));
    manager.commit();

    // Far outside the room: no probe influences the listener.
    let (_, shared) = shared_inputs(
        CoordinateSpace3::canonical_at(Vector3::new(1e4, 1e4, 1e4)),
        0,
    );
    manager.set_shared_reflection_inputs(shared);
    manager.simulate_indirect();
    assert!(!reader.update_read_buffer());
    // Reconstructed from a silent field (the band filters add 1e-9 to their recurrence, R9).
    assert!(
        lock(&s)
            .impulse_response()
            .unwrap()
            .data()
            .iter()
            .all(|&x| x.abs() < 1e-6)
    );

    // Without caching the source is reconstructed (from its silent field) and published.
    manager.set_enable_probe_caching_for_missing_probes(false);
    manager.simulate_indirect();
    assert!(reader.update_read_buffer());
}
