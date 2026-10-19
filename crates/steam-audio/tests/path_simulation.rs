//! Parity tests for stage 7b against golden data from `tools/golden/src/path_simulation.cpp`: deviation models, path
//! simulation (validation, alternate paths, the sound field), the path effect, and pathing in the simulation manager.

mod common;

use std::sync::{Arc, Mutex};

use common::{committed_sub_scene, draw_vector, draw01, load_obj, mesh_bounds, static_mesh};
use parity::{Lcg, Tolerance, compare, golden};
use steam_audio::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use steam_audio::baking::{BakedDataIdentifier, BakedDataType, BakedDataVariation};
use steam_audio::bands::NUM_BANDS;
use steam_audio::deviation::{DeviationCallback, DeviationModel};
use steam_audio::distance_attenuation::{DistanceAttenuationCallback, DistanceAttenuationModel};
use steam_audio::effects::{PathEffect, PathEffectParams, PathEffectSettings, PathSpatialization};
use steam_audio::geometry::{Aabb, Sphere};
use steam_audio::hrtf::{HrtfDatabase, HrtfNormalization, HrtfSettings, HrtfSource};
use steam_audio::math::{CoordinateSpace3, Matrix4, Vector3};
use steam_audio::pathing::{PathBakeParams, PathBaker, PathSimulationParams, PathSimulator};
use steam_audio::probes::{ProbeBatch, ProbeGeneration, ProbeNeighborhood, generate_probes};
use steam_audio::random::CRand;
use steam_audio::scene::{InstancedMesh, Scene};
use steam_audio::sh;
use steam_audio::simulation::{
    IndirectEffectType, SceneType, SharedPathingSimulationInputs, SimulationData,
    SimulationManager, SimulationScene, SimulationSettings,
};
use steam_audio::speaker_layout::{SpeakerLayout, SpeakerLayoutType};
use steam_audio::thread_pool::ThreadPool;

const SENTINEL: f32 = -7.0;
const SAMPLING_RATE: u32 = 48000;
const FRAME_SIZE: usize = 100;

#[track_caller]
fn check<T: parity::Element + Into<f64> + Copy>(name: &str, actual: &[T], tol: Tolerance) {
    let expected = golden::<T>(&format!("path_simulation/{name}"));
    if let Err(m) = compare(actual, &expected.data, tol) {
        panic!("{name}: {m}");
    }
}

/// The generator's `boxTransform`.
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

/// The generator's `wallTransform`.
fn wall_transform() -> Matrix4 {
    let mut m = Matrix4::ZERO;
    m[(0, 0)] = 2.5;
    m[(1, 1)] = 12.5;
    m[(2, 2)] = 37.5;
    m[(0, 3)] = 13.5;
    m[(1, 3)] = 0.0;
    m[(2, 3)] = -8.6;
    m[(3, 3)] = 1.0;
    m
}

/// The generator's `drawInterior`.
fn draw_interior(lcg: &mut Lcg, bounds: &Aabb, x0: f32, w: f32) -> Vector3 {
    let x = draw01(lcg);
    let y = draw01(lcg);
    let z = draw01(lcg);
    let (lo, hi) = (bounds.min, bounds.max);
    Vector3::new(
        lo.x + (hi.x - lo.x) * (x0 + w * x),
        lo.y + 2.5 + 2.0 * y,
        lo.z + (hi.z - lo.z) * z,
    )
}

fn draw_left(lcg: &mut Lcg, bounds: &Aabb) -> Vector3 {
    draw_interior(lcg, bounds, 0.0, 0.45)
}

fn draw_right(lcg: &mut Lcg, bounds: &Aabb) -> Vector3 {
    draw_interior(lcg, bounds, 0.55, 0.45)
}

fn draw_anywhere(lcg: &mut Lcg, bounds: &Aabb) -> Vector3 {
    draw_interior(lcg, bounds, 0.0, 1.0)
}

fn callback_deviation() -> DeviationModel {
    let callback: DeviationCallback =
        Arc::new(|angle, band| 1.0 / (1.0 + angle * (band + 1) as f32));
    DeviationModel::Callback(callback)
}

fn callback_attenuation() -> DistanceAttenuationModel {
    let callback: DistanceAttenuationCallback = Arc::new(|distance| 1.0 / (1.0 + distance));
    DistanceAttenuationModel::Callback(callback)
}

fn identifier() -> BakedDataIdentifier {
    BakedDataIdentifier {
        variation: BakedDataVariation::Dynamic,
        data_type: BakedDataType::Pathing,
        endpoint_influence: Sphere::default(),
    }
}

/// The generator's `bakeBatch`.
fn bake_batch(scene: &Scene, bounds: &Aabb) -> ProbeBatch {
    let mut probes = Vec::new();
    generate_probes(
        scene,
        &box_transform(bounds),
        ProbeGeneration::UniformFloor {
            spacing: 4.5,
            height: 1.5,
        },
        &mut probes,
    );
    let mut batch = ProbeBatch::new();
    batch.add_probes(&probes);

    let params = PathBakeParams {
        num_samples: 1,
        radius: 0.0,
        threshold: 0.5,
        vis_range: 1000.0,
        vis_range_real_time: 1000.0,
        path_range: 1000.0,
        asymmetric_vis_range: false,
        down: Vector3::new(0.0, -1.0, 0.0),
        prune_vis_graph: false,
    };
    PathBaker::new().bake(
        scene,
        &identifier(),
        &params,
        &ThreadPool::new(1),
        &mut batch,
        |_| {},
    );
    batch.commit();
    batch
}

struct Setup {
    bounds: Aabb,
    /// simplescene with the wall.
    scene: Scene,
    batch: Arc<ProbeBatch>,
    batch_b: Arc<ProbeBatch>,
}

fn setup() -> Setup {
    let obj = load_obj("simplescene");
    let smallbox = load_obj("smallbox");
    let bounds = mesh_bounds(&obj);

    let mut bake_scene = Scene::new();
    let mesh = bake_scene.insert_static_mesh(static_mesh(&obj));
    bake_scene.add_static_mesh(mesh);
    bake_scene.commit();

    let box_scene = committed_sub_scene(&smallbox);
    let mut scene = Scene::new();
    let mesh = scene.insert_static_mesh(static_mesh(&obj));
    scene.add_static_mesh(mesh);
    let wall = scene.insert_instanced_mesh(InstancedMesh::new(box_scene, wall_transform()));
    scene.add_instanced_mesh(wall);
    scene.commit();

    let batch = Arc::new(bake_batch(&bake_scene, &bounds));
    let batch_b = Arc::new(bake_batch(&bake_scene, &bounds));

    Setup {
        bounds,
        scene,
        batch,
        batch_b,
    }
}

/// The probes of `batch` around `point` that it sees, weighted, as the generator finds them.
fn neighborhood(scene: &Scene, batch: &ProbeBatch, point: Vector3) -> ProbeNeighborhood {
    let mut probes = ProbeNeighborhood::default();
    probes.resize(ProbeNeighborhood::MAX_PROBES_PER_BATCH);
    batch.influencing_probes(point, &mut probes, 0, 0);
    probes.check_occlusion(scene, point, &[batch]);
    probes.calc_weights(point, &[batch]);
    probes
}

fn append_vector(out: &mut Vec<f32>, v: Vector3) {
    out.extend([v.x, v.y, v.z]);
}

#[test]
fn deviation_models() {
    let angles = golden::<f32>("path_simulation/deviation_angles").data;
    let utd = DeviationModel::default();
    let callback = callback_deviation();

    let mut values = Vec::new();
    for &angle in &angles {
        values.extend((0..NUM_BANDS).map(|band| utd.evaluate(angle, band)));
        values.extend((0..NUM_BANDS).map(|band| callback.evaluate(angle, band)));
    }
    check("deviation_values", &values, Tolerance::F32);
}

const NUM_QUERIES: usize = 64;

#[test]
fn path_simulator() {
    let Setup {
        bounds,
        scene,
        batch,
        ..
    } = setup();
    let mut lcg = Lcg::new(613);

    let down = Vector3::new(0.0, -1.0, 0.0);
    let point_simulator = PathSimulator::new(&batch, 1, false, down);
    let volumetric_simulator = PathSimulator::new(&batch, 3, false, down);

    let attenuation_models = [
        DistanceAttenuationModel::default(),
        DistanceAttenuationModel::InverseDistance { min_distance: 2.0 },
        callback_attenuation(),
    ];
    let deviation_models = [DeviationModel::default(), callback_deviation()];

    let mut results = Vec::new();
    let mut values = Vec::new();
    let mut rays = Vec::new();

    for q in 0..NUM_QUERIES {
        let anywhere = q % 4 == 3;
        let source = if anywhere {
            draw_anywhere(&mut lcg, &bounds)
        } else {
            draw_left(&mut lcg, &bounds)
        };
        let listener = if anywhere {
            draw_anywhere(&mut lcg, &bounds)
        } else {
            draw_right(&mut lcg, &bounds)
        };

        let volumetric = q % 2 != 0;
        let params = PathSimulationParams {
            source,
            listener,
            radius: if volumetric { 0.6 } else { 0.0 },
            threshold: 0.3,
            vis_range: 1000.0,
            order: q % 4,
            enable_validation: (q / 2) % 2 != 0,
            find_alternate_paths: (q / 4) % 2 != 0,
            simplify_paths: (q / 8) % 2 != 0,
            real_time_vis: (q / 16) % 2 != 0,
            distance_attenuation_model: &attenuation_models[q % 3],
            deviation_model: &deviation_models[(q / 3) % 2],
            paths_from_all_source_probes: q % 3 == 0,
            force_direct_occlusion: q % 5 == 0,
        };

        let source_probes = neighborhood(&scene, &batch, source);
        let listener_probes = neighborhood(&scene, &batch, listener);

        let mut coeffs = [SENTINEL; 16];
        let mut count = 0;
        let mut record = |from: Vector3, to: Vector3, occluded: bool| {
            rays.push(q as f32);
            append_vector(&mut rays, from);
            append_vector(&mut rays, to);
            rays.push(if occluded { 1.0 } else { 0.0 });
            count += 1;
        };

        let simulator = if volumetric {
            &volumetric_simulator
        } else {
            &point_simulator
        };
        let result = simulator.find_paths(
            &scene,
            &[&*batch],
            0,
            &source_probes,
            &listener_probes,
            &params,
            &mut coeffs,
            Some(&mut record),
        );

        results.extend([i32::from(result.is_some()), count]);
        match result {
            Some(r) => {
                values.extend(r.eq_gains);
                values.extend(coeffs);
                append_vector(&mut values, r.direction);
                values.extend([r.distance_ratio, r.total_deviation]);
            }
            None => {
                assert_eq!(coeffs, [SENTINEL; 16], "query {q}: untouched coefficients");
                values.extend([SENTINEL; 24]);
            }
        }
    }

    check("simulator_results", &results, Tolerance::EXACT);
    check("simulator_values", &values, Tolerance::F32);
    check("simulator_rays", &rays, Tolerance::F32);
}

/// The generator's `kEffectCases`.
struct EffectCase {
    name: &'static str,
    max_order: usize,
    spatialize: bool,
    layout: SpeakerLayoutType,
    script: &'static str,
}

const EFFECT_CASES: [EffectCase; 3] = [
    EffectCase {
        name: "ambisonics",
        max_order: 2,
        spatialize: false,
        layout: SpeakerLayoutType::Mono,
        script: "N2N2N1N0TRN2N2",
    },
    EffectCase {
        name: "stereo",
        max_order: 3,
        spatialize: true,
        layout: SpeakerLayoutType::Stereo,
        script: "B3B3C2B1TTP2Q3P1TRB2TP0B3",
    },
    EffectCase {
        name: "surround",
        max_order: 1,
        spatialize: true,
        layout: SpeakerLayoutType::FivePointOne,
        script: "P1Q1P0P1RP1T",
    },
];

fn state_code(state: AudioEffectState) -> i32 {
    match state {
        AudioEffectState::TailRemaining => 0,
        AudioEffectState::TailComplete => 1,
    }
}

#[test]
fn path_effect() {
    let mut lcg = Lcg::new(907);
    let hrtf = HrtfDatabase::new(
        &HrtfSettings {
            source: HrtfSource::BuiltIn,
            volume: 0.0,
            normalization: HrtfNormalization::None,
        },
        SAMPLING_RATE,
        FRAME_SIZE,
    )
    .unwrap();
    let audio = AudioSettings::new(SAMPLING_RATE, FRAME_SIZE);

    for c in &EFFECT_CASES {
        let layout = SpeakerLayout::new(c.layout);
        let mut effect = PathEffect::new(
            &audio,
            &PathEffectSettings {
                max_order: c.max_order,
                spatialization: c.spatialize.then_some(PathSpatialization {
                    speaker_layout: &layout,
                    hrtf: &hrtf,
                }),
            },
        );

        let num_channels = if c.spatialize {
            layout.num_speakers()
        } else {
            sh::num_coeffs_for_order(c.max_order)
        };
        let mut input = AudioBuffer::new(1, FRAME_SIZE);
        let mut output = AudioBuffer::new(num_channels, FRAME_SIZE);

        let mut samples = Vec::new();
        let mut states = Vec::new();

        let mut ops = c.script.bytes();
        while let Some(op) = ops.next() {
            let state = match op {
                b'R' => {
                    effect.reset();
                    continue;
                }
                b'T' => {
                    output.as_mut_slice().fill(SENTINEL);
                    effect.tail(&mut output)
                }
                kind => {
                    let order = usize::from(ops.next().expect("an order") - b'0');

                    for x in input.channel_mut(0) {
                        *x = lcg.next_signed();
                    }

                    let mut eq = [0.0; NUM_BANDS];
                    for g in &mut eq {
                        let x = draw01(&mut lcg);
                        *g = 0.02 + 0.98 * x;
                    }

                    let mut sh_coeffs = [0.0; 16];
                    for s in &mut sh_coeffs[..sh::num_coeffs_for_order(order)] {
                        *s = lcg.next_signed();
                    }

                    let ahead = draw_vector(&mut lcg).normalize_or_zero();
                    let listener = CoordinateSpace3::from_ahead(ahead, Vector3::ZERO);

                    let params = PathEffectParams {
                        eq_coeffs: &eq,
                        sh_coeffs: &sh_coeffs,
                        order,
                        binaural: kind == b'B' || kind == b'C',
                        hrtf: Some(&hrtf),
                        listener,
                        normalize_eq: kind == b'Q' || kind == b'C',
                    };

                    output.as_mut_slice().fill(SENTINEL);
                    effect.apply(&params, &input, &mut output)
                }
            };

            samples.extend_from_slice(output.as_slice());
            states.extend([
                state_code(state),
                effect.num_tail_samples_remaining() as i32,
            ]);
        }

        let prefix = format!("effect_{}", c.name);
        let expected = golden::<f32>(&format!("path_simulation/{prefix}_output")).data;
        // Binaural frames convolve through the FFT (R1).
        let tol = Tolerance::peak_relative(&expected, 1e-5);
        check(&format!("{prefix}_output"), &samples, tol);
        check(&format!("{prefix}_states"), &states, Tolerance::EXACT);
    }
}

/// The generator's `ManagerSource`.
struct ManagerSource {
    enabled: bool,
    batch_b: bool,
    order: usize,
    radius: f32,
    enable_validation: bool,
    find_alternate_paths: bool,
    simplify_paths: bool,
    real_time_vis: bool,
    attenuation: usize,
    callback_deviation: bool,
}

const MANAGER_SOURCES: [ManagerSource; 5] = [
    ManagerSource {
        enabled: true,
        batch_b: false,
        order: 2,
        radius: 0.6,
        enable_validation: true,
        find_alternate_paths: true,
        simplify_paths: true,
        real_time_vis: true,
        attenuation: 0,
        callback_deviation: false,
    },
    ManagerSource {
        enabled: true,
        batch_b: true,
        order: 2,
        radius: 0.6,
        enable_validation: true,
        find_alternate_paths: false,
        simplify_paths: true,
        real_time_vis: false,
        attenuation: 2,
        callback_deviation: false,
    },
    ManagerSource {
        enabled: true,
        batch_b: false,
        order: 1,
        radius: 0.0,
        enable_validation: false,
        find_alternate_paths: false,
        simplify_paths: false,
        real_time_vis: false,
        attenuation: 1,
        callback_deviation: true,
    },
    ManagerSource {
        enabled: true,
        batch_b: false,
        order: 0,
        radius: 0.0,
        enable_validation: true,
        find_alternate_paths: true,
        simplify_paths: false,
        real_time_vis: true,
        attenuation: 0,
        callback_deviation: false,
    },
    ManagerSource {
        enabled: false,
        batch_b: false,
        order: 2,
        radius: 0.0,
        enable_validation: false,
        find_alternate_paths: false,
        simplify_paths: false,
        real_time_vis: false,
        attenuation: 0,
        callback_deviation: false,
    },
];

const MANAGER_ORDER: usize = 2;
const NUM_FRAMES: usize = 4;

#[test]
fn simulation_manager_pathing() {
    let Setup {
        bounds,
        scene,
        batch,
        batch_b,
    } = setup();
    let mut lcg = Lcg::new(1032);

    let settings = SimulationSettings {
        enable_direct: false,
        enable_indirect: false,
        enable_pathing: true,
        scene_type: SceneType::Default,
        indirect_type: IndirectEffectType::Convolution,
        max_num_occlusion_samples: 1,
        max_num_rays: 1,
        num_diffuse_samples: 1,
        max_duration: 1.0,
        max_order: MANAGER_ORDER,
        max_num_sources: 4,
        num_threads: 1,
        ray_batch_size: 1,
        num_vis_samples: 3,
        asymmetric_vis_range: false,
        down: Vector3::new(0.0, -1.0, 0.0),
        sampling_rate: SAMPLING_RATE,
        frame_size: FRAME_SIZE,
    };
    let mut manager = SimulationManager::new(&settings, 0, &mut CRand::new(1));
    manager.set_scene(Some(SimulationScene::Default(scene)));
    manager.add_probe_batch(Arc::clone(&batch));
    manager.add_probe_batch(Arc::clone(&batch_b));

    let sources: Vec<_> = MANAGER_SOURCES
        .iter()
        .map(|c| {
            let mut s = SimulationData::new(&settings, false, true);
            let inputs = &mut s.pathing_inputs;
            inputs.enabled = c.enabled;
            inputs.probes = Some(Arc::clone(if c.batch_b { &batch_b } else { &batch }));
            inputs.vis_radius = c.radius;
            inputs.vis_threshold = 0.3;
            inputs.vis_range = 1000.0;
            inputs.order = c.order;
            inputs.enable_validation = c.enable_validation;
            inputs.find_alternate_paths = c.find_alternate_paths;
            inputs.simplify_paths = c.simplify_paths;
            inputs.real_time_vis = c.real_time_vis;
            inputs.distance_attenuation_model = match c.attenuation {
                1 => DistanceAttenuationModel::InverseDistance { min_distance: 2.0 },
                2 => callback_attenuation(),
                _ => DistanceAttenuationModel::default(),
            };
            inputs.deviation_model = if c.callback_deviation {
                callback_deviation()
            } else {
                DeviationModel::default()
            };
            let s = Arc::new(Mutex::new(s));
            manager.add_source(Arc::clone(&s));
            s
        })
        .collect();
    manager.commit();

    let rays = Arc::new(Mutex::new(Vec::new()));
    let mut values = Vec::new();

    for frame in 0..NUM_FRAMES {
        let mut listener = draw_right(&mut lcg, &bounds);
        if frame == 3 {
            listener = Vector3::new(100.0, 100.0, 100.0);
        }

        for (i, s) in sources.iter().enumerate() {
            let mut origin = draw_left(&mut lcg, &bounds);
            if i == 3 && frame == 0 {
                origin = Vector3::new(100.0, 100.0, 100.0);
            }
            s.lock().unwrap().pathing_inputs.source = CoordinateSpace3 {
                origin,
                ..CoordinateSpace3::default()
            };
        }

        let mut shared = SharedPathingSimulationInputs {
            listener: CoordinateSpace3 {
                origin: listener,
                ..CoordinateSpace3::default()
            },
            vis_callback: None,
        };
        if frame == 1 {
            let rays = Arc::clone(&rays);
            shared.vis_callback = Some(Arc::new(move |from: Vector3, to: Vector3, occluded| {
                let mut rays = rays.lock().unwrap();
                append_vector(&mut rays, from);
                append_vector(&mut rays, to);
                rays.push(if occluded { 1.0 } else { 0.0 });
            }));
        }
        manager.set_shared_pathing_inputs(shared);

        manager.set_enable_paths_from_all_source_probes(frame == 1);

        if frame == 2 {
            for s in &sources {
                manager.simulate_pathing_source(&mut s.lock().unwrap());
            }
        } else {
            manager.simulate_pathing();
        }

        manager.set_enable_paths_from_all_source_probes(false);

        for s in &sources {
            let s = s.lock().unwrap();
            let o = &s.pathing_outputs;
            values.extend(o.eq);
            values.extend(&o.sh);
            append_vector(&mut values, o.direction);
            values.extend([o.distance_ratio, o.total_deviation]);
        }
    }

    check("manager_values", &values, Tolerance::F32);
    check("manager_rays", &rays.lock().unwrap(), Tolerance::F32);
}
