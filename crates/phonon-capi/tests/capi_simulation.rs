//! Parity tests for stage 8c against golden data from `tools/golden/src/capi_simulation.cpp`: upstream's C API for
//! simulators and sources, the reflection effect and mixer, and the path effect, called through the same `ipl*`
//! functions exported by this crate.
//!
//! Inputs are rebuilt from the generator's `Lcg` draws in the same order. The C API seeds its reflection simulators with
//! the clock, so reflections are simulated without scattering, whose results do not depend on the random numbers. The
//! reverb effects draw their delay lengths from the thread's `rand()` state, which the generator resets with `srand(1)`;
//! the cases that create them run on a new thread, whose state starts the same.

#![allow(unsafe_code)] // The tests call the C API.

#[path = "../../steam-audio/tests/common/mod.rs"]
mod common;

use std::ffi::c_void;
use std::mem::offset_of;
use std::ptr;

use common::{
    ObjMesh, assign_materials, draw_point, draw_transform, draw_vector, draw01, load_obj,
    mesh_bounds,
};
use parity::{Element, Lcg, Tolerance, compare, golden};
use phonon::*;
use steam_audio::geometry::Aabb;
use steam_audio::math::{CoordinateSpace3, Matrix4, Vector3};

const SENTINEL: f32 = 9.0;
const FRAME_SIZE: i32 = 240;

/// Signal tolerance relative to the peak.
const SIGNAL: f64 = 1e-5;

/// Reverb times and hybrid EQs off Windows: relative error allowed. They come from lines fitted to the log of simulated
/// energies down to the faint tail, where libm's `powf` (specular lobe) moves them by up to about 6e-4 relative on
/// Linux and macOS (stage 8b).
const REVERB_OFF_WINDOWS: f64 = 1e-3;

/// Parametric and hybrid reverb audio driven by simulated reverb times off Windows, relative to the peak: a reverb time
/// off by 6e-4 relative changes the gains of the delay network's absorption filters by about as much, which accumulates
/// over the reverb's feedback.
const REVERB_AUDIO_OFF_WINDOWS: f64 = 5e-3;

fn expected<T: Element>(name: &str) -> Vec<T> {
    golden::<T>(&format!("capi_simulation/{name}")).data
}

fn peak(values: &[f32]) -> f64 {
    values
        .iter()
        .filter(|&&x| x != SENTINEL)
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()))
}

/// Prints the largest error relative to the peak (for measuring bit-exactness with `--nocapture`).
fn report(name: &str, actual: &[f32], expected: &[f32]) {
    let peak = peak(expected);
    let worst = actual.iter().zip(expected).fold(0.0f64, |m, (&a, &e)| {
        m.max((f64::from(a) - f64::from(e)).abs())
    });
    let identical = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| a.to_bits() == e.to_bits());
    eprintln!(
        "{name}: worst error {:.3e} x peak{}",
        if peak > 0.0 { worst / peak } else { worst },
        if identical { " (bit-exact)" } else { "" }
    );
}

#[track_caller]
fn check_f32(name: &str, actual: &[f32], tol: impl Fn(&[f32]) -> Tolerance) {
    let expected = expected::<f32>(name);
    report(name, actual, &expected);
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    if let Err(m) = compare(actual, &expected, tol(&expected)) {
        panic!("{name}: {m}");
    }
}

#[track_caller]
fn check_signal(name: &str, actual: &[f32], ratio: f64) {
    check_f32(name, actual, |e| Tolerance::new(ratio * peak(e), 0.0));
}

#[track_caller]
fn check_i32(name: &str, actual: &[i32]) {
    assert_eq!(actual, expected::<i32>(name), "{name}");
}

/// Runs a case on a new thread, whose `rand()` state starts as after `srand(1)`.
fn on_new_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::spawn(f).join().expect("the case passed");
}

fn make_context() -> IPLContext {
    let mut settings = IPLContextSettings {
        version: STEAMAUDIO_VERSION,
        logCallback: None,
        allocateCallback: None,
        freeCallback: None,
        simdLevel: IPL_SIMDLEVEL_AVX2,
        flags: 0,
    };
    let mut context = ptr::null_mut();
    // SAFETY: valid pointers.
    assert_eq!(
        unsafe { iplContextCreate(&mut settings, &mut context) },
        IPL_STATUS_SUCCESS
    );
    context
}

fn to_ipl(v: Vector3) -> IPLVector3 {
    IPLVector3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

/// Row-major, as `phonon.h` documents it.
fn to_ipl_matrix(m: &Matrix4) -> IPLMatrix4x4 {
    let mut out = IPLMatrix4x4::default();
    for i in 0..4 {
        for j in 0..4 {
            out.elements[i][j] = m[(i, j)];
        }
    }
    out
}

fn to_ipl_space(c: &CoordinateSpace3) -> IPLCoordinateSpace3 {
    IPLCoordinateSpace3 {
        right: to_ipl(c.right),
        up: to_ipl(c.up),
        ahead: to_ipl(c.ahead),
        origin: to_ipl(c.origin),
    }
}

/// The transform that maps the unit cube `[-0.5, 0.5]³` onto a box.
fn box_transform(aabb: &Aabb) -> IPLMatrix4x4 {
    let mut m = IPLMatrix4x4::default();
    for i in 0..3 {
        m.elements[i][i] = aabb.max[i] - aabb.min[i];
        m.elements[i][3] = 0.5 * (aabb.min[i] + aabb.max[i]);
    }
    m.elements[3][3] = 1.0;
    m
}

/// A point in the middle 40% of a box along each axis.
fn interior_point(lcg: &mut Lcg, aabb: &Aabb) -> Vector3 {
    let mut p = Vector3::ZERO;
    for i in 0..3 {
        let t = 0.3 + 0.4 * draw01(lcg);
        p[i] = aabb.min[i] + (aabb.max[i] - aabb.min[i]) * t;
    }
    p
}

/// A coordinate space at `origin`, looking along an Lcg direction.
fn draw_space(lcg: &mut Lcg, origin: Vector3) -> IPLCoordinateSpace3 {
    let ahead = draw_vector(lcg).normalize_or_zero();
    to_ipl_space(&CoordinateSpace3::from_ahead(ahead, origin))
}

fn create_static_mesh(scene: IPLScene, obj: &ObjMesh) -> IPLStaticMesh {
    let mut vertices: Vec<IPLVector3> = obj.vertices.iter().copied().map(to_ipl).collect();
    let mut triangles: Vec<IPLTriangle> = obj
        .triangles
        .iter()
        .map(|t| IPLTriangle {
            indices: t.indices.map(|i| i as i32),
        })
        .collect();
    let mut material_indices: Vec<i32> = obj.material_indices.iter().map(|&i| i as i32).collect();
    let mut materials: Vec<IPLMaterial> = obj
        .materials
        .iter()
        .map(|m| IPLMaterial {
            absorption: m.absorption,
            scattering: m.scattering,
            transmission: m.transmission,
        })
        .collect();

    let mut settings = IPLStaticMeshSettings {
        numVertices: vertices.len() as i32,
        numTriangles: triangles.len() as i32,
        numMaterials: materials.len() as i32,
        vertices: vertices.as_mut_ptr(),
        triangles: triangles.as_mut_ptr(),
        materialIndices: material_indices.as_mut_ptr(),
        materials: materials.as_mut_ptr(),
    };
    let mut mesh = ptr::null_mut();
    // SAFETY: valid arrays for the counts and a live scene.
    assert_eq!(
        unsafe { iplStaticMeshCreate(scene, &mut settings, &mut mesh) },
        IPL_STATUS_SUCCESS
    );
    mesh
}

fn make_specular(mesh: &mut ObjMesh) {
    for m in &mut mesh.materials {
        m.scattering = 0.0;
    }
}

/// A default scene holding a mesh and an instance of a sub-scene holding another (the generator's `makeScene`).
struct SceneSetup {
    scene: IPLScene,
    sub_scene: IPLScene,
    mesh: IPLStaticMesh,
    sub_mesh: IPLStaticMesh,
    instance: IPLInstancedMesh,
    bounds: Aabb,
}

impl SceneSetup {
    fn new(context: IPLContext, lcg: &mut Lcg, obj: &ObjMesh, instanced: &ObjMesh) -> Self {
        let bounds = mesh_bounds(obj);
        let mut settings = IPLSceneSettings {
            r#type: IPL_SCENETYPE_DEFAULT,
            closestHitCallback: None,
            anyHitCallback: None,
            batchedClosestHitCallback: None,
            batchedAnyHitCallback: None,
            userData: ptr::null_mut(),
            embreeDevice: ptr::null_mut(),
            radeonRaysDevice: ptr::null_mut(),
        };
        let (mut scene, mut sub_scene) = (ptr::null_mut(), ptr::null_mut());
        // SAFETY: valid pointers and live objects throughout.
        unsafe {
            assert_eq!(
                iplSceneCreate(context, &mut settings, &mut scene),
                IPL_STATUS_SUCCESS
            );
            assert_eq!(
                iplSceneCreate(context, &mut settings, &mut sub_scene),
                IPL_STATUS_SUCCESS
            );
            let mesh = create_static_mesh(scene, obj);
            iplStaticMeshAdd(mesh, scene);
            let sub_mesh = create_static_mesh(sub_scene, instanced);
            iplStaticMeshAdd(sub_mesh, sub_scene);
            iplSceneCommit(sub_scene);

            let mut instance_settings = IPLInstancedMeshSettings {
                subScene: sub_scene,
                transform: to_ipl_matrix(&draw_transform(lcg, &bounds)),
            };
            let mut instance = ptr::null_mut();
            assert_eq!(
                iplInstancedMeshCreate(scene, &mut instance_settings, &mut instance),
                IPL_STATUS_SUCCESS
            );
            iplInstancedMeshAdd(instance, scene);
            iplSceneCommit(scene);
            Self {
                scene,
                sub_scene,
                mesh,
                sub_mesh,
                instance,
                bounds,
            }
        }
    }

    fn release(mut self) {
        // SAFETY: live objects, released once.
        unsafe {
            iplInstancedMeshRelease(&mut self.instance);
            iplStaticMeshRelease(&mut self.sub_mesh);
            iplStaticMeshRelease(&mut self.mesh);
            iplSceneRelease(&mut self.sub_scene);
            iplSceneRelease(&mut self.scene);
        }
    }
}

/// Floor probes over a box, in a committed batch.
fn floor_probes(
    context: IPLContext,
    scene: IPLScene,
    bounds: &Aabb,
    spacing_fraction: f32,
) -> IPLProbeBatch {
    let extents = bounds.extents();
    let mut params = IPLProbeGenerationParams {
        r#type: IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR,
        spacing: spacing_fraction * extents.x.max(extents.z),
        height: 1.5,
        transform: box_transform(bounds),
    };
    let (mut probes, mut batch) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: valid pointers and live objects.
    unsafe {
        assert_eq!(
            iplProbeArrayCreate(context, &mut probes),
            IPL_STATUS_SUCCESS
        );
        iplProbeArrayGenerateProbes(probes, scene, &mut params);
        assert_eq!(iplProbeBatchCreate(context, &mut batch), IPL_STATUS_SUCCESS);
        iplProbeBatchAddProbeArray(batch, probes);
        iplProbeBatchCommit(batch);
        iplProbeArrayRelease(&mut probes);
    }
    batch
}

fn fill_noise(lcg: &mut Lcg, samples: &mut [f32]) {
    for x in samples {
        *x = lcg.next_signed();
    }
}

/// A buffer prefilled with the sentinel, so that untouched channels are checked too.
struct Buffer {
    data: Vec<f32>,
    pointers: Vec<*mut f32>,
    channels: i32,
}

impl Buffer {
    fn new(channels: i32) -> Self {
        let mut data = vec![SENTINEL; (channels * FRAME_SIZE) as usize];
        let pointers = (0..channels)
            .map(|i| data[(i * FRAME_SIZE) as usize..].as_mut_ptr())
            .collect();
        Self {
            data,
            pointers,
            channels,
        }
    }

    fn reset(&mut self) {
        self.data.fill(SENTINEL);
    }

    fn ipl(&mut self) -> IPLAudioBuffer {
        IPLAudioBuffer {
            numChannels: self.channels,
            numSamples: FRAME_SIZE,
            data: self.pointers.as_mut_ptr(),
        }
    }
}

/// A mono input frame.
struct Input {
    samples: Vec<f32>,
    pointer: *mut f32,
}

impl Input {
    fn new() -> Self {
        let mut samples = vec![0.0; FRAME_SIZE as usize];
        let pointer = samples.as_mut_ptr();
        Self { samples, pointer }
    }

    fn ipl(&mut self) -> IPLAudioBuffer {
        IPLAudioBuffer {
            numChannels: 1,
            numSamples: FRAME_SIZE,
            data: &mut self.pointer,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------------------------------------------------

unsafe extern "system" fn distance_callback(distance: f32, user_data: *mut c_void) -> f32 {
    // SAFETY: the test passes a pointer to an f32.
    let k = unsafe { *user_data.cast::<f32>() };
    1.0 / (1.0 + k * distance)
}

unsafe extern "system" fn air_callback(distance: f32, band: i32, user_data: *mut c_void) -> f32 {
    // SAFETY: the test passes a pointer to an f32.
    let k = unsafe { *user_data.cast::<f32>() };
    1.0 / (1.0 + k * distance * (band + 1) as f32)
}

unsafe extern "system" fn directivity_callback(
    direction: IPLVector3,
    user_data: *mut c_void,
) -> f32 {
    // SAFETY: the test passes a pointer to an f32.
    let k = unsafe { *user_data.cast::<f32>() };
    k + (1.0 - k) * direction.z.abs()
}

unsafe extern "system" fn deviation_callback(angle: f32, band: i32, user_data: *mut c_void) -> f32 {
    // SAFETY: the test passes a pointer to an f32.
    let k = unsafe { *user_data.cast::<f32>() };
    1.0 / (1.0 + k * angle * (band + 1) as f32)
}

unsafe extern "system" fn vis_callback(
    from: IPLVector3,
    to: IPLVector3,
    occluded: IPLbool,
    user_data: *mut c_void,
) {
    // SAFETY: the test passes a pointer to a `Vec<f32>`.
    let out = unsafe { &mut *user_data.cast::<Vec<f32>>() };
    out.extend_from_slice(&[from.x, from.y, from.z, to.x, to.y, to.z, occluded as f32]);
}

fn default_inputs() -> IPLSimulationInputs {
    IPLSimulationInputs {
        flags: 0,
        directFlags: 0,
        source: IPLCoordinateSpace3::default(),
        distanceAttenuationModel: IPLDistanceAttenuationModel {
            r#type: IPL_DISTANCEATTENUATIONTYPE_DEFAULT,
            minDistance: 0.0,
            callback: None,
            userData: ptr::null_mut(),
            dirty: IPL_FALSE,
        },
        airAbsorptionModel: IPLAirAbsorptionModel {
            r#type: IPL_AIRABSORPTIONTYPE_DEFAULT,
            coefficients: [0.0; 3],
            callback: None,
            userData: ptr::null_mut(),
            dirty: IPL_FALSE,
        },
        directivity: IPLDirectivity {
            dipoleWeight: 0.0,
            dipolePower: 0.0,
            callback: None,
            userData: ptr::null_mut(),
        },
        occlusionType: IPL_OCCLUSIONTYPE_RAYCAST,
        occlusionRadius: 0.0,
        numOcclusionSamples: 1,
        reverbScale: [1.0; 3],
        hybridReverbTransitionTime: 1.0,
        hybridReverbOverlapPercent: 0.25,
        baked: IPL_FALSE,
        bakedDataIdentifier: IPLBakedDataIdentifier::default(),
        pathingProbes: ptr::null_mut(),
        visRadius: 0.0,
        visThreshold: 0.0,
        visRange: 0.0,
        pathingOrder: 0,
        enableValidation: IPL_FALSE,
        findAlternatePaths: IPL_FALSE,
        numTransmissionRays: 1,
        deviationModel: ptr::null_mut(),
    }
}

fn simulation_settings(
    flags: IPLSimulationFlags,
    reflection_type: IPLReflectionEffectType,
    sampling_rate: i32,
) -> IPLSimulationSettings {
    IPLSimulationSettings {
        flags,
        sceneType: IPL_SCENETYPE_DEFAULT,
        reflectionType: reflection_type,
        maxNumOcclusionSamples: 32,
        maxNumRays: 512,
        numDiffuseSamples: 8,
        maxDuration: 0.25,
        maxOrder: 1,
        maxNumSources: 4,
        numThreads: 1,
        rayBatchSize: 1,
        numVisSamples: 4,
        samplingRate: sampling_rate,
        frameSize: FRAME_SIZE,
        openCLDevice: ptr::null_mut(),
        radeonRaysDevice: ptr::null_mut(),
        tanDevice: ptr::null_mut(),
    }
}

fn create_simulator(context: IPLContext, settings: &IPLSimulationSettings) -> IPLSimulator {
    let mut settings = *settings;
    let mut simulator = ptr::null_mut();
    // SAFETY: valid pointers and a live context.
    assert_eq!(
        unsafe { iplSimulatorCreate(context, &mut settings, &mut simulator) },
        IPL_STATUS_SUCCESS
    );
    simulator
}

fn create_source(simulator: IPLSimulator, flags: IPLSimulationFlags) -> IPLSource {
    let mut settings = IPLSourceSettings { flags };
    let mut source = ptr::null_mut();
    // SAFETY: valid pointers and a live simulator.
    assert_eq!(
        unsafe { iplSourceCreate(simulator, &mut settings, &mut source) },
        IPL_STATUS_SUCCESS
    );
    source
}

fn append_direct(out: &mut Vec<f32>, d: &IPLDirectEffectParams) {
    out.push(d.distanceAttenuation);
    out.extend_from_slice(&d.airAbsorption);
    out.push(d.directivity);
    out.push(d.occlusion);
    out.extend_from_slice(&d.transmission);
}

// ---------------------------------------------------------------------------------------------------------------------
// Layouts and statuses
// ---------------------------------------------------------------------------------------------------------------------

macro_rules! size {
    ($layout:ident, $t:ty) => {
        $layout.push(size_of::<$t>() as i32)
    };
}

macro_rules! field {
    ($layout:ident, $t:ty, $f:ident) => {
        $layout.push(offset_of!($t, $f) as i32)
    };
}

#[test]
fn layouts() {
    let mut l = Vec::new();
    size!(l, IPLReflectionEffectType);
    size!(l, IPLSimulationFlags);
    size!(l, IPLDirectSimulationFlags);
    size!(l, IPLDeviationModelType);
    size!(l, IPLOcclusionType);

    size!(l, IPLReflectionEffectSettings);
    field!(l, IPLReflectionEffectSettings, r#type);
    field!(l, IPLReflectionEffectSettings, irSize);
    field!(l, IPLReflectionEffectSettings, numChannels);
    size!(l, IPLReflectionEffectParams);
    field!(l, IPLReflectionEffectParams, r#type);
    field!(l, IPLReflectionEffectParams, ir);
    field!(l, IPLReflectionEffectParams, reverbTimes);
    field!(l, IPLReflectionEffectParams, eq);
    field!(l, IPLReflectionEffectParams, delay);
    field!(l, IPLReflectionEffectParams, numChannels);
    field!(l, IPLReflectionEffectParams, irSize);
    field!(l, IPLReflectionEffectParams, tanDevice);
    field!(l, IPLReflectionEffectParams, tanSlot);
    size!(l, IPLPathEffectSettings);
    field!(l, IPLPathEffectSettings, maxOrder);
    field!(l, IPLPathEffectSettings, spatialize);
    field!(l, IPLPathEffectSettings, speakerLayout);
    field!(l, IPLPathEffectSettings, hrtf);
    size!(l, IPLPathEffectParams);
    field!(l, IPLPathEffectParams, eqCoeffs);
    field!(l, IPLPathEffectParams, shCoeffs);
    field!(l, IPLPathEffectParams, order);
    field!(l, IPLPathEffectParams, binaural);
    field!(l, IPLPathEffectParams, hrtf);
    field!(l, IPLPathEffectParams, listener);
    field!(l, IPLPathEffectParams, normalizeEQ);

    size!(l, IPLDeviationModel);
    field!(l, IPLDeviationModel, r#type);
    field!(l, IPLDeviationModel, callback);
    field!(l, IPLDeviationModel, userData);
    size!(l, IPLSimulationSettings);
    field!(l, IPLSimulationSettings, flags);
    field!(l, IPLSimulationSettings, sceneType);
    field!(l, IPLSimulationSettings, reflectionType);
    field!(l, IPLSimulationSettings, maxNumOcclusionSamples);
    field!(l, IPLSimulationSettings, maxNumRays);
    field!(l, IPLSimulationSettings, numDiffuseSamples);
    field!(l, IPLSimulationSettings, maxDuration);
    field!(l, IPLSimulationSettings, maxOrder);
    field!(l, IPLSimulationSettings, maxNumSources);
    field!(l, IPLSimulationSettings, numThreads);
    field!(l, IPLSimulationSettings, rayBatchSize);
    field!(l, IPLSimulationSettings, numVisSamples);
    field!(l, IPLSimulationSettings, samplingRate);
    field!(l, IPLSimulationSettings, frameSize);
    field!(l, IPLSimulationSettings, openCLDevice);
    field!(l, IPLSimulationSettings, radeonRaysDevice);
    field!(l, IPLSimulationSettings, tanDevice);
    size!(l, IPLSourceSettings);
    field!(l, IPLSourceSettings, flags);
    size!(l, IPLSimulationInputs);
    field!(l, IPLSimulationInputs, flags);
    field!(l, IPLSimulationInputs, directFlags);
    field!(l, IPLSimulationInputs, source);
    field!(l, IPLSimulationInputs, distanceAttenuationModel);
    field!(l, IPLSimulationInputs, airAbsorptionModel);
    field!(l, IPLSimulationInputs, directivity);
    field!(l, IPLSimulationInputs, occlusionType);
    field!(l, IPLSimulationInputs, occlusionRadius);
    field!(l, IPLSimulationInputs, numOcclusionSamples);
    field!(l, IPLSimulationInputs, reverbScale);
    field!(l, IPLSimulationInputs, hybridReverbTransitionTime);
    field!(l, IPLSimulationInputs, hybridReverbOverlapPercent);
    field!(l, IPLSimulationInputs, baked);
    field!(l, IPLSimulationInputs, bakedDataIdentifier);
    field!(l, IPLSimulationInputs, pathingProbes);
    field!(l, IPLSimulationInputs, visRadius);
    field!(l, IPLSimulationInputs, visThreshold);
    field!(l, IPLSimulationInputs, visRange);
    field!(l, IPLSimulationInputs, pathingOrder);
    field!(l, IPLSimulationInputs, enableValidation);
    field!(l, IPLSimulationInputs, findAlternatePaths);
    field!(l, IPLSimulationInputs, numTransmissionRays);
    field!(l, IPLSimulationInputs, deviationModel);
    size!(l, IPLSimulationSharedInputs);
    field!(l, IPLSimulationSharedInputs, listener);
    field!(l, IPLSimulationSharedInputs, numRays);
    field!(l, IPLSimulationSharedInputs, numBounces);
    field!(l, IPLSimulationSharedInputs, duration);
    field!(l, IPLSimulationSharedInputs, order);
    field!(l, IPLSimulationSharedInputs, irradianceMinDistance);
    field!(l, IPLSimulationSharedInputs, pathingVisCallback);
    field!(l, IPLSimulationSharedInputs, pathingUserData);
    size!(l, IPLSimulationOutputs);
    field!(l, IPLSimulationOutputs, direct);
    field!(l, IPLSimulationOutputs, reflections);
    field!(l, IPLSimulationOutputs, pathing);

    if cfg!(target_pointer_width = "64") {
        check_i32("layout", &l);
    }
}

#[test]
fn statuses() {
    let mut context = make_context();
    let mut statuses = Vec::new();

    // SAFETY: valid pointers and live objects throughout; null where the test checks the null case.
    unsafe {
        let mut settings = simulation_settings(
            IPL_SIMULATIONFLAGS_DIRECT,
            IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
            24000,
        );
        let mut simulator = ptr::null_mut();
        let mut unused = ptr::null_mut();
        statuses.push(iplSimulatorCreate(context, &mut settings, &mut simulator));
        statuses.push(iplSimulatorCreate(context, ptr::null_mut(), &mut unused));
        statuses.push(iplSimulatorCreate(
            ptr::null_mut(),
            &mut settings,
            &mut unused,
        ));

        let mut source_settings = IPLSourceSettings {
            flags: IPL_SIMULATIONFLAGS_DIRECT,
        };
        let mut source = ptr::null_mut();
        let mut unused_source = ptr::null_mut();
        statuses.push(iplSourceCreate(
            simulator,
            &mut source_settings,
            &mut source,
        ));
        statuses.push(iplSourceCreate(
            simulator,
            ptr::null_mut(),
            &mut unused_source,
        ));
        statuses.push(iplSourceCreate(
            ptr::null_mut(),
            &mut source_settings,
            &mut unused_source,
        ));

        let mut audio = IPLAudioSettings {
            samplingRate: 24000,
            frameSize: FRAME_SIZE,
        };
        let mut bad_audio = IPLAudioSettings {
            samplingRate: 0,
            frameSize: FRAME_SIZE,
        };
        let mut effect_settings = IPLReflectionEffectSettings {
            r#type: IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
            irSize: 1000,
            numChannels: 4,
        };
        let mut effect = ptr::null_mut();
        let mut unused_effect = ptr::null_mut();
        statuses.push(iplReflectionEffectCreate(
            context,
            &mut audio,
            &mut effect_settings,
            &mut effect,
        ));
        statuses.push(iplReflectionEffectCreate(
            context,
            &mut bad_audio,
            &mut effect_settings,
            &mut unused_effect,
        ));
        statuses.push(iplReflectionEffectCreate(
            context,
            &mut audio,
            ptr::null_mut(),
            &mut unused_effect,
        ));

        let mut mixer = ptr::null_mut();
        let mut unused_mixer = ptr::null_mut();
        let mut bad_frame = IPLAudioSettings {
            samplingRate: 24000,
            frameSize: 0,
        };
        statuses.push(iplReflectionMixerCreate(
            context,
            &mut audio,
            &mut effect_settings,
            &mut mixer,
        ));
        statuses.push(iplReflectionMixerCreate(
            context,
            &mut bad_frame,
            &mut effect_settings,
            &mut unused_mixer,
        ));

        let mut path_settings = IPLPathEffectSettings {
            maxOrder: 0,
            spatialize: IPL_FALSE,
            speakerLayout: IPLSpeakerLayout {
                r#type: IPL_SPEAKERLAYOUTTYPE_MONO,
                numSpeakers: 0,
                speakers: ptr::null_mut(),
            },
            hrtf: ptr::null_mut(),
        };
        let mut path_effect = ptr::null_mut();
        let mut unused_path = ptr::null_mut();
        for max_order in [3, 4, -1, 0] {
            path_settings.maxOrder = max_order;
            let target = if max_order == 3 {
                &mut path_effect
            } else {
                &mut unused_path
            };
            let status = iplPathEffectCreate(context, &mut audio, &mut path_settings, target);
            statuses.push(status);
            if max_order == 0 && status == IPL_STATUS_SUCCESS {
                iplPathEffectRelease(&mut unused_path);
            }
        }

        statuses.push(iplReflectionEffectGetTailSize(ptr::null_mut()));
        statuses.push(iplPathEffectGetTailSize(ptr::null_mut()));

        iplPathEffectRelease(&mut path_effect);
        iplReflectionMixerRelease(&mut mixer);
        iplReflectionEffectRelease(&mut effect);
        iplSourceRelease(&mut source);
        iplSimulatorRelease(&mut simulator);
        iplContextRelease(&mut context);
    }

    check_i32("statuses", &statuses);
}

// ---------------------------------------------------------------------------------------------------------------------
// Direct simulation
// ---------------------------------------------------------------------------------------------------------------------

const NUM_DIRECT_SOURCES: usize = 5;

#[test]
fn direct() {
    let mut context = make_context();
    let mut lcg = Lcg::new(839);

    let mut room = load_obj("boxroom");
    assign_materials(&mut lcg, &mut room);
    let bx = load_obj("smallbox");
    let setup = SceneSetup::new(context, &mut lcg, &room, &bx);
    let bounds = setup.bounds;

    let simulator = create_simulator(
        context,
        &simulation_settings(
            IPL_SIMULATIONFLAGS_DIRECT,
            IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
            24000,
        ),
    );

    let mut distance_k = 0.3 + draw01(&mut lcg);
    let mut air_k = 0.05 + 0.1 * draw01(&mut lcg);
    let mut directivity_k = 0.2 + 0.5 * draw01(&mut lcg);
    let distance_ptr: *mut c_void = (&mut distance_k as *mut f32).cast();
    let air_ptr: *mut c_void = (&mut air_k as *mut f32).cast();
    let directivity_ptr: *mut c_void = (&mut directivity_k as *mut f32).cast();

    let mut sources = [ptr::null_mut(); NUM_DIRECT_SOURCES];
    let mut inputs = [default_inputs(); NUM_DIRECT_SOURCES];
    for source in &mut sources {
        *source = create_source(simulator, IPL_SIMULATIONFLAGS_DIRECT);
        // SAFETY: live objects.
        unsafe { iplSourceAdd(*source, simulator) };
    }
    // SAFETY: a live simulator.
    unsafe { iplSimulatorCommit(simulator) };

    let draw_inputs = |lcg: &mut Lcg, inputs: &mut [IPLSimulationInputs], frame: usize| {
        for (i, input) in inputs.iter_mut().enumerate() {
            let flag_bits = lcg.next_u32();
            input.directFlags = if i == 0 { 31 } else { (flag_bits & 31) as i32 };
            let origin = draw_point(lcg, &bounds);
            input.source = draw_space(lcg, origin);

            input.distanceAttenuationModel.r#type = ((i + frame) % 3) as i32;
            input.distanceAttenuationModel.minDistance = 0.5 + draw01(lcg);
            input.distanceAttenuationModel.callback = Some(distance_callback);
            input.distanceAttenuationModel.userData = distance_ptr;

            input.airAbsorptionModel.r#type = ((i + 2 * frame) % 3) as i32;
            for c in &mut input.airAbsorptionModel.coefficients {
                *c = 0.01 + 0.2 * draw01(lcg);
            }
            input.airAbsorptionModel.callback = Some(air_callback);
            input.airAbsorptionModel.userData = air_ptr;

            input.directivity.dipoleWeight = draw01(lcg);
            input.directivity.dipolePower = 1.0 + 3.0 * draw01(lcg);
            input.directivity.callback = if i == 4 {
                Some(directivity_callback)
            } else {
                None
            };
            input.directivity.userData = directivity_ptr;

            input.occlusionType = ((i + frame) % 2) as i32;
            input.occlusionRadius = 0.1 + 0.5 * draw01(lcg);
            input.numOcclusionSamples = 1 + (lcg.next_u32() % 40) as i32;
            input.numTransmissionRays = 1 + (lcg.next_u32() % 6) as i32;
        }
    };

    let mut outputs = Vec::new();
    let mut run = |lcg: &mut Lcg| {
        let origin = interior_point(lcg, &bounds);
        let mut shared = IPLSimulationSharedInputs {
            listener: draw_space(lcg, origin),
            numRays: 0,
            numBounces: 0,
            duration: 0.0,
            order: 0,
            irradianceMinDistance: 0.0,
            pathingVisCallback: None,
            pathingUserData: ptr::null_mut(),
        };
        // SAFETY: live objects and valid pointers.
        unsafe {
            iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_DIRECT, &mut shared);
            iplSimulatorRunDirect(simulator);
            for &source in &sources {
                let mut o = IPLSimulationOutputs::default();
                iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT, &mut o);
                append_direct(&mut outputs, &o.direct);
            }
        }
    };

    // SAFETY: live objects and valid pointers throughout.
    unsafe {
        draw_inputs(&mut lcg, &mut inputs, 0);
        for (source, input) in sources.iter().zip(&mut inputs) {
            iplSourceSetInputs(*source, IPL_SIMULATIONFLAGS_DIRECT, input);
        }
        run(&mut lcg);

        iplSimulatorSetScene(simulator, setup.scene);
        iplSimulatorCommit(simulator);
        for frame in 1..=2 {
            draw_inputs(&mut lcg, &mut inputs, frame);
            for (source, input) in sources.iter().zip(&mut inputs) {
                iplSourceSetInputs(*source, IPL_SIMULATIONFLAGS_DIRECT, input);
            }
            run(&mut lcg);
        }

        draw_inputs(&mut lcg, &mut inputs, 3);
        inputs[0].distanceAttenuationModel.r#type = 7;
        inputs[0].airAbsorptionModel.r#type = 7;
        inputs[1].occlusionType = 7;
        for i in 0..3 {
            iplSourceSetInputs(sources[i], IPL_SIMULATIONFLAGS_DIRECT, &mut inputs[i]);
        }
        iplSourceSetInputs(sources[3], IPL_SIMULATIONFLAGS_REFLECTIONS, &mut inputs[3]);
        iplSourceSetInputs(sources[4], IPL_SIMULATIONFLAGS_DIRECT, &mut inputs[4]);
        iplSourceRemove(sources[4], simulator);
        iplSimulatorCommit(simulator);
        run(&mut lcg);

        iplInstancedMeshUpdateTransform(
            setup.instance,
            setup.scene,
            to_ipl_matrix(&draw_transform(&mut lcg, &bounds)),
        );
        iplSceneCommit(setup.scene);
        run(&mut lcg);
    }

    check_f32("direct", &outputs, |_| Tolerance::F32);

    let mut simulator = simulator;
    // SAFETY: live objects, released once.
    unsafe {
        for source in &mut sources {
            iplSourceRelease(source);
        }
        iplSimulatorRelease(&mut simulator);
    }
    setup.release();
    // SAFETY: a live context.
    unsafe { iplContextRelease(&mut context) };
}

// ---------------------------------------------------------------------------------------------------------------------
// Reflections
// ---------------------------------------------------------------------------------------------------------------------

const NUM_REFLECTION_SOURCES: usize = 4;
const REFLECTION_RATE: i32 = 24000;

fn reverb_tolerance(_: &[f32]) -> Tolerance {
    if cfg!(windows) {
        Tolerance::F32
    } else {
        Tolerance::new(1e-6, REVERB_OFF_WINDOWS)
    }
}

fn reverb_audio_ratio() -> f64 {
    if cfg!(windows) {
        SIGNAL
    } else {
        REVERB_AUDIO_OFF_WINDOWS
    }
}

#[test]
fn reflections() {
    on_new_thread(reflections_case);
}

fn reflections_case() {
    let mut context = make_context();
    let mut lcg = Lcg::new(853);

    let mut room = load_obj("boxroom");
    assign_materials(&mut lcg, &mut room);
    make_specular(&mut room);
    let mut bx = load_obj("smallbox");
    make_specular(&mut bx);
    let setup = SceneSetup::new(context, &mut lcg, &room, &bx);
    let bounds = setup.bounds;

    let mut batch = floor_probes(context, setup.scene, &bounds, 0.25);
    let reverb_id = IPLBakedDataIdentifier {
        r#type: IPL_BAKEDDATATYPE_REFLECTIONS,
        variation: IPL_BAKEDDATAVARIATION_REVERB,
        endpointInfluence: IPLSphere::default(),
    };
    let mut bake = IPLReflectionsBakeParams {
        scene: setup.scene,
        probeBatch: batch,
        sceneType: IPL_SCENETYPE_DEFAULT,
        identifier: reverb_id,
        bakeFlags: IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION
            | IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC,
        numRays: 256,
        numDiffuseSamples: 8,
        numBounces: 4,
        simulatedDuration: 0.25,
        savedDuration: 0.25,
        order: 1,
        numThreads: 1,
        rayBatchSize: 1,
        irradianceMinDistance: 1.0,
        bakeBatchSize: 1,
        openCLDevice: ptr::null_mut(),
        radeonRaysDevice: ptr::null_mut(),
    };
    // SAFETY: valid pointers and live objects.
    unsafe { iplReflectionsBakerBake(context, &mut bake, None, ptr::null_mut()) };

    let mut distance_k = 0.2 + draw01(&mut lcg);
    let distance_ptr: *mut c_void = (&mut distance_k as *mut f32).cast();
    let types = [
        IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
        IPL_REFLECTIONEFFECTTYPE_PARAMETRIC,
        IPL_REFLECTIONEFFECTTYPE_HYBRID,
    ];
    let names = ["convolution", "parametric", "hybrid"];

    for (&ty, name) in types.iter().zip(names) {
        let settings = simulation_settings(IPL_SIMULATIONFLAGS_REFLECTIONS, ty, REFLECTION_RATE);
        let mut simulator = create_simulator(context, &settings);
        // SAFETY: live objects.
        unsafe {
            iplSimulatorSetScene(simulator, setup.scene);
            iplSimulatorAddProbeBatch(simulator, batch);
        }

        let mut sources = [ptr::null_mut(); NUM_REFLECTION_SOURCES];
        let mut inputs = [default_inputs(); NUM_REFLECTION_SOURCES];
        for (source, input) in sources.iter_mut().zip(&mut inputs) {
            *source = create_source(simulator, IPL_SIMULATIONFLAGS_REFLECTIONS);
            // SAFETY: live objects.
            unsafe { iplSourceAdd(*source, simulator) };
            input.flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
            let origin = interior_point(&mut lcg, &bounds);
            input.source = draw_space(&mut lcg, origin);
        }
        // SAFETY: a live simulator.
        unsafe { iplSimulatorCommit(simulator) };

        inputs[1].distanceAttenuationModel.r#type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
        inputs[1].distanceAttenuationModel.minDistance = 2.0;
        inputs[1].airAbsorptionModel.r#type = IPL_AIRABSORPTIONTYPE_EXPONENTIAL;
        for c in &mut inputs[1].airAbsorptionModel.coefficients {
            *c = 0.02 + 0.1 * draw01(&mut lcg);
        }
        inputs[1].directivity.dipoleWeight = 0.5;
        inputs[1].directivity.dipolePower = 2.0;
        for s in &mut inputs[1].reverbScale {
            *s = 0.7 + 0.6 * draw01(&mut lcg);
        }
        inputs[1].hybridReverbTransitionTime = 0.1;
        inputs[1].hybridReverbOverlapPercent = 0.5;
        inputs[2].baked = IPL_TRUE;
        inputs[2].bakedDataIdentifier = reverb_id;
        inputs[3].distanceAttenuationModel.r#type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        inputs[3].distanceAttenuationModel.callback = Some(distance_callback);
        inputs[3].distanceAttenuationModel.userData = distance_ptr;

        let mut audio = IPLAudioSettings {
            samplingRate: REFLECTION_RATE,
            frameSize: FRAME_SIZE,
        };
        let ir_size = (settings.maxDuration * REFLECTION_RATE as f32).ceil() as i32;
        let mut effect_settings = IPLReflectionEffectSettings {
            r#type: ty,
            irSize: ir_size,
            numChannels: 1,
        };
        let out_channels = if ty == IPL_REFLECTIONEFFECTTYPE_PARAMETRIC {
            4
        } else {
            1
        };
        let mut effects = [ptr::null_mut(); NUM_REFLECTION_SOURCES];
        let mut mixer = ptr::null_mut();
        // SAFETY: valid pointers and a live context.
        unsafe {
            for effect in &mut effects {
                assert_eq!(
                    iplReflectionEffectCreate(context, &mut audio, &mut effect_settings, effect),
                    IPL_STATUS_SUCCESS
                );
            }
            assert_eq!(
                iplReflectionMixerCreate(context, &mut audio, &mut effect_settings, &mut mixer),
                IPL_STATUS_SUCCESS
            );
        }
        let use_mixer = ty == IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;

        let mut sim_outputs = Vec::new();
        let mut audio_out = Vec::new();
        let mut states = Vec::new();
        let mut input = Input::new();
        let mut out_buffer = Buffer::new(out_channels);
        let mut mix_buffer = Buffer::new(1);
        let mut outputs = [IPLSimulationOutputs::default(); NUM_REFLECTION_SOURCES];

        let origin = interior_point(&mut lcg, &bounds);
        let mut listener = draw_space(&mut lcg, origin);
        for frame in 0..3 {
            // SAFETY: live objects and valid pointers throughout.
            unsafe {
                if frame == 2 {
                    let origin = interior_point(&mut lcg, &bounds);
                    listener = draw_space(&mut lcg, origin);
                    let origin = interior_point(&mut lcg, &bounds);
                    inputs[0].source = draw_space(&mut lcg, origin);
                    inputs[3].distanceAttenuationModel.dirty = IPL_TRUE;
                    iplReflectionEffectReset(effects[1]);
                }
                for (source, input) in sources.iter().zip(&mut inputs) {
                    iplSourceSetInputs(*source, IPL_SIMULATIONFLAGS_REFLECTIONS, input);
                }

                let mut shared = IPLSimulationSharedInputs {
                    listener,
                    numRays: 512,
                    numBounces: 8,
                    duration: 0.25,
                    order: 1,
                    irradianceMinDistance: 1.0,
                    pathingVisCallback: None,
                    pathingUserData: ptr::null_mut(),
                };
                iplSimulatorSetSharedInputs(
                    simulator,
                    IPL_SIMULATIONFLAGS_REFLECTIONS,
                    &mut shared,
                );
                iplSimulatorRunReflections(simulator);

                for (source, o) in sources.iter().zip(&mut outputs) {
                    *o = IPLSimulationOutputs::default();
                    iplSourceGetOutputs(*source, IPL_SIMULATIONFLAGS_REFLECTIONS, o);
                    let r = &o.reflections;
                    sim_outputs.extend_from_slice(&r.reverbTimes);
                    if ty == IPL_REFLECTIONEFFECTTYPE_HYBRID {
                        sim_outputs.extend_from_slice(&r.eq);
                        sim_outputs.push(r.delay as f32);
                    }
                    if ty != IPL_REFLECTIONEFFECTTYPE_PARAMETRIC {
                        sim_outputs.push(r.numChannels as f32);
                        sim_outputs.push(r.irSize as f32);
                    }
                }

                for _ in 0..2 {
                    fill_noise(&mut lcg, &mut input.samples);
                    for i in 0..NUM_REFLECTION_SOURCES {
                        let mut params = outputs[i].reflections;
                        params.r#type = ty;
                        params.numChannels = 1;
                        let to_mixer = use_mixer && i < 2;
                        out_buffer.reset();
                        let mut in_ipl = input.ipl();
                        let mut out_ipl = out_buffer.ipl();
                        states.push(iplReflectionEffectApply(
                            effects[i],
                            &mut params,
                            &mut in_ipl,
                            &mut out_ipl,
                            if to_mixer { mixer } else { ptr::null_mut() },
                        ));
                        if !to_mixer {
                            audio_out.extend_from_slice(&out_buffer.data);
                        }
                    }
                    if use_mixer {
                        let mut params = outputs[0].reflections;
                        params.numChannels = 1;
                        mix_buffer.reset();
                        let mut out_ipl = mix_buffer.ipl();
                        states.push(iplReflectionMixerApply(mixer, &mut params, &mut out_ipl));
                        audio_out.extend_from_slice(&mix_buffer.data);
                    }
                }
            }
        }

        // SAFETY: live objects and valid pointers.
        unsafe {
            for _ in 0..2 {
                for (i, &effect) in effects.iter().enumerate() {
                    let to_mixer = use_mixer && i < 2;
                    states.push(iplReflectionEffectGetTailSize(effect));
                    out_buffer.reset();
                    let mut out_ipl = out_buffer.ipl();
                    states.push(iplReflectionEffectGetTail(
                        effect,
                        &mut out_ipl,
                        if to_mixer { mixer } else { ptr::null_mut() },
                    ));
                    if !to_mixer {
                        audio_out.extend_from_slice(&out_buffer.data);
                    }
                }
                if use_mixer {
                    let mut params = outputs[0].reflections;
                    params.numChannels = 1;
                    mix_buffer.reset();
                    let mut out_ipl = mix_buffer.ipl();
                    states.push(iplReflectionMixerApply(mixer, &mut params, &mut out_ipl));
                    audio_out.extend_from_slice(&mix_buffer.data);
                }
            }
        }

        // The IR sizes and delays are integers; the reverb times and EQs come from simulated energies.
        let sim_name = format!("reflections_{name}");
        check_f32(&sim_name, &sim_outputs, reverb_tolerance);
        let ratio = if ty == IPL_REFLECTIONEFFECTTYPE_CONVOLUTION {
            SIGNAL
        } else {
            reverb_audio_ratio()
        };
        check_signal(&format!("{sim_name}_audio"), &audio_out, ratio);
        check_i32(&format!("{sim_name}_states"), &states);

        // SAFETY: live objects, released once.
        unsafe {
            iplReflectionMixerRelease(&mut mixer);
            for effect in &mut effects {
                iplReflectionEffectRelease(effect);
            }
            for source in &mut sources {
                iplSourceRelease(source);
            }
            iplSimulatorRelease(&mut simulator);
        }
    }

    // Effects without an IR.
    {
        let mut audio = IPLAudioSettings {
            samplingRate: REFLECTION_RATE,
            frameSize: FRAME_SIZE,
        };
        let mut audio_out = Vec::new();
        let mut states = Vec::new();
        let mut input = Input::new();
        let mut out_buffer = Buffer::new(4);

        let mut effects = [ptr::null_mut(); 2];
        let mut convolution = IPLReflectionEffectSettings {
            r#type: IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
            irSize: 1000,
            numChannels: 4,
        };
        let mut hybrid = IPLReflectionEffectSettings {
            r#type: IPL_REFLECTIONEFFECTTYPE_HYBRID,
            ..convolution
        };
        let mut parametric = IPLReflectionEffectSettings {
            r#type: IPL_REFLECTIONEFFECTTYPE_PARAMETRIC,
            irSize: 0,
            numChannels: 0,
        };
        let mut parametric_mixer = ptr::null_mut();
        // SAFETY: valid pointers and live objects throughout.
        unsafe {
            assert_eq!(
                iplReflectionEffectCreate(context, &mut audio, &mut convolution, &mut effects[0]),
                IPL_STATUS_SUCCESS
            );
            assert_eq!(
                iplReflectionEffectCreate(context, &mut audio, &mut hybrid, &mut effects[1]),
                IPL_STATUS_SUCCESS
            );
            assert_eq!(
                iplReflectionMixerCreate(
                    context,
                    &mut audio,
                    &mut parametric,
                    &mut parametric_mixer
                ),
                IPL_STATUS_SUCCESS
            );

            let mut params = IPLReflectionEffectParams::default();
            for t in &mut params.reverbTimes {
                *t = 0.3 + draw01(&mut lcg);
            }
            for e in &mut params.eq {
                *e = 0.2 + draw01(&mut lcg);
            }
            params.delay = 100;
            params.numChannels = 4;
            params.irSize = 1000;
            for _ in 0..3 {
                fill_noise(&mut lcg, &mut input.samples);
                for &effect in &effects {
                    out_buffer.reset();
                    let mut in_ipl = input.ipl();
                    let mut out_ipl = out_buffer.ipl();
                    states.push(iplReflectionEffectApply(
                        effect,
                        &mut params,
                        &mut in_ipl,
                        &mut out_ipl,
                        ptr::null_mut(),
                    ));
                    audio_out.extend_from_slice(&out_buffer.data);
                }
                out_buffer.reset();
                let mut out_ipl = out_buffer.ipl();
                states.push(iplReflectionMixerApply(
                    parametric_mixer,
                    &mut params,
                    &mut out_ipl,
                ));
                audio_out.extend_from_slice(&out_buffer.data);
            }

            iplReflectionMixerRelease(&mut parametric_mixer);
            for effect in &mut effects {
                iplReflectionEffectRelease(effect);
            }
        }
        check_signal("reflections_no_ir_audio", &audio_out, SIGNAL);
        check_i32("reflections_no_ir_states", &states);
    }

    // SAFETY: live objects, released once.
    unsafe { iplProbeBatchRelease(&mut batch) };
    setup.release();
    // SAFETY: a live context.
    unsafe { iplContextRelease(&mut context) };
}

// ---------------------------------------------------------------------------------------------------------------------
// Pathing
// ---------------------------------------------------------------------------------------------------------------------

const NUM_PATH_SOURCES: usize = 3;
const PATH_RATE: i32 = 48000;

#[test]
fn pathing() {
    let mut context = make_context();
    let mut lcg = Lcg::new(857);

    let mut room = load_obj("simplescene");
    assign_materials(&mut lcg, &mut room);
    let bx = load_obj("smallbox");
    let setup = SceneSetup::new(context, &mut lcg, &room, &bx);
    let bounds = setup.bounds;

    let mut batch = floor_probes(context, setup.scene, &bounds, 0.12);
    let mut path_params = IPLPathBakeParams {
        scene: setup.scene,
        probeBatch: batch,
        identifier: IPLBakedDataIdentifier {
            r#type: IPL_BAKEDDATATYPE_PATHING,
            variation: IPL_BAKEDDATAVARIATION_DYNAMIC,
            endpointInfluence: IPLSphere::default(),
        },
        numSamples: 1,
        radius: 0.5,
        threshold: 0.5,
        visRange: 1000.0,
        pathRange: 1000.0,
        numThreads: 1,
    };
    // SAFETY: valid pointers and live objects.
    unsafe { iplPathBakerBake(context, &mut path_params, None, ptr::null_mut()) };

    let mut wall = IPLMatrix4x4::default();
    wall.elements[0][0] = 2.5;
    wall.elements[1][1] = 12.5;
    wall.elements[2][2] = 37.5;
    wall.elements[0][3] = 13.5;
    wall.elements[2][3] = -8.6;
    wall.elements[3][3] = 1.0;
    // SAFETY: live objects.
    unsafe {
        iplInstancedMeshUpdateTransform(setup.instance, setup.scene, wall);
        iplSceneCommit(setup.scene);
    }

    let draw_interior = |lcg: &mut Lcg, x0: f32, w: f32| {
        let x = draw01(lcg);
        let y = draw01(lcg);
        let z = draw01(lcg);
        let (lo, hi) = (bounds.min, bounds.max);
        Vector3::new(
            lo.x + (hi.x - lo.x) * (x0 + w * x),
            lo.y + 2.5 + 2.0 * y,
            lo.z + (hi.z - lo.z) * z,
        )
    };

    let mut settings = simulation_settings(
        IPL_SIMULATIONFLAGS_PATHING,
        IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
        PATH_RATE,
    );
    settings.maxOrder = 2;
    let mut simulator = create_simulator(context, &settings);
    // SAFETY: live objects.
    unsafe {
        iplSimulatorSetScene(simulator, setup.scene);
        iplSimulatorAddProbeBatch(simulator, batch);
    }

    let mut distance_k = 0.2 + draw01(&mut lcg);
    let mut deviation_k = 0.5 + draw01(&mut lcg);
    let mut deviation = IPLDeviationModel {
        r#type: IPL_DEVIATIONTYPE_CALLBACK,
        callback: Some(deviation_callback),
        userData: (&mut deviation_k as *mut f32).cast(),
    };

    let mut sources = [ptr::null_mut(); NUM_PATH_SOURCES];
    let mut inputs = [default_inputs(); NUM_PATH_SOURCES];
    for (i, (source, input)) in sources.iter_mut().zip(&mut inputs).enumerate() {
        *source = create_source(simulator, IPL_SIMULATIONFLAGS_PATHING);
        // SAFETY: live objects.
        unsafe { iplSourceAdd(*source, simulator) };
        input.flags = IPL_SIMULATIONFLAGS_PATHING;
        input.pathingProbes = batch;
        input.visRadius = 0.5;
        input.visThreshold = 0.5;
        input.visRange = 1000.0;
        input.pathingOrder = if i == 2 { 0 } else { i as i32 + 1 };
        input.enableValidation = if i != 0 { IPL_TRUE } else { IPL_FALSE };
        input.findAlternatePaths = if i == 2 { IPL_TRUE } else { IPL_FALSE };
    }
    inputs[1].distanceAttenuationModel.r#type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    inputs[1].distanceAttenuationModel.minDistance = 1.5;
    inputs[1].deviationModel = &mut deviation;
    inputs[2].distanceAttenuationModel.r#type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
    inputs[2].distanceAttenuationModel.callback = Some(distance_callback);
    inputs[2].distanceAttenuationModel.userData = (&mut distance_k as *mut f32).cast();
    // SAFETY: a live simulator.
    unsafe { iplSimulatorCommit(simulator) };

    let mut vis_rays: Vec<f32> = Vec::new();
    let mut sim_outputs = Vec::new();
    let mut eqs: Vec<Vec<f32>> = vec![Vec::new(); NUM_PATH_SOURCES];
    let mut shs: Vec<Vec<f32>> = vec![Vec::new(); NUM_PATH_SOURCES];
    let mut listeners = Vec::new();

    for frame in 0..2 {
        // SAFETY: live objects and valid pointers throughout.
        unsafe {
            for (i, (source, input)) in sources.iter().zip(&mut inputs).enumerate() {
                let origin = if frame == 1 && i == 2 {
                    draw_interior(&mut lcg, 0.0, 1.0)
                } else {
                    draw_interior(&mut lcg, 0.0, 0.45)
                };
                input.source = draw_space(&mut lcg, origin);
                iplSourceSetInputs(*source, IPL_SIMULATIONFLAGS_PATHING, input);
            }
            let origin = draw_interior(&mut lcg, 0.55, 0.45);
            let mut shared = IPLSimulationSharedInputs {
                listener: draw_space(&mut lcg, origin),
                numRays: 0,
                numBounces: 0,
                duration: 0.0,
                order: 0,
                irradianceMinDistance: 0.0,
                pathingVisCallback: Some(vis_callback),
                pathingUserData: (&mut vis_rays as *mut Vec<f32>).cast(),
            };
            listeners.push(shared.listener);
            iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_PATHING, &mut shared);
            iplSimulatorRunPathing(simulator);

            for (i, &source) in sources.iter().enumerate() {
                let mut o = IPLSimulationOutputs::default();
                iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_PATHING, &mut o);
                let sh = std::slice::from_raw_parts(o.pathing.shCoeffs, 9);
                sim_outputs.extend_from_slice(&o.pathing.eqCoeffs);
                sim_outputs.extend_from_slice(sh);
                eqs[i].extend_from_slice(&o.pathing.eqCoeffs);
                shs[i].extend_from_slice(sh);
            }
        }
    }
    check_f32("pathing", &sim_outputs, |_| Tolerance::F32);
    check_f32("pathing_vis_rays", &vis_rays, |_| Tolerance::F32);

    let mut audio = IPLAudioSettings {
        samplingRate: PATH_RATE,
        frameSize: FRAME_SIZE,
    };
    let mut hrtf_settings = IPLHRTFSettings {
        r#type: IPL_HRTFTYPE_DEFAULT,
        sofaFileName: ptr::null(),
        sofaData: ptr::null(),
        sofaDataSize: 0,
        volume: 1.0,
        normType: IPL_HRTFNORMTYPE_NONE,
    };
    let mut hrtf = ptr::null_mut();
    let mut effects = [ptr::null_mut(); 3];
    // SAFETY: valid pointers and live objects.
    unsafe {
        assert_eq!(
            iplHRTFCreate(context, &mut audio, &mut hrtf_settings, &mut hrtf),
            IPL_STATUS_SUCCESS
        );
        let mut effect_settings = IPLPathEffectSettings {
            maxOrder: 2,
            spatialize: IPL_FALSE,
            speakerLayout: IPLSpeakerLayout {
                r#type: IPL_SPEAKERLAYOUTTYPE_MONO,
                numSpeakers: 0,
                speakers: ptr::null_mut(),
            },
            hrtf: ptr::null_mut(),
        };
        assert_eq!(
            iplPathEffectCreate(context, &mut audio, &mut effect_settings, &mut effects[0]),
            IPL_STATUS_SUCCESS
        );
        effect_settings.spatialize = IPL_TRUE;
        effect_settings.speakerLayout.r#type = IPL_SPEAKERLAYOUTTYPE_STEREO;
        effect_settings.hrtf = hrtf;
        for effect in &mut effects[1..] {
            assert_eq!(
                iplPathEffectCreate(context, &mut audio, &mut effect_settings, effect),
                IPL_STATUS_SUCCESS
            );
        }
    }
    let out_channels = [9, 2, 2];

    let mut audio_out = Vec::new();
    let mut states = Vec::new();
    let mut input = Input::new();
    // SAFETY: live objects and valid pointers throughout.
    unsafe {
        for (frame, listener) in listeners.iter().enumerate() {
            for _ in 0..2 {
                fill_noise(&mut lcg, &mut input.samples);
                for e in 0..3 {
                    let s = if e == 1 { 1 } else { 0 };
                    let mut params = IPLPathEffectParams {
                        eqCoeffs: eqs[s][3 * frame..3 * frame + 3].try_into().unwrap(),
                        shCoeffs: shs[s][9 * frame..].as_mut_ptr(),
                        order: if e == 0 { 2 } else { 1 },
                        binaural: if e == 2 { IPL_TRUE } else { IPL_FALSE },
                        hrtf,
                        listener: *listener,
                        normalizeEQ: if e == 1 { IPL_TRUE } else { IPL_FALSE },
                    };
                    let mut out_buffer = Buffer::new(out_channels[e]);
                    let mut in_ipl = input.ipl();
                    let mut out_ipl = out_buffer.ipl();
                    states.push(iplPathEffectApply(
                        effects[e],
                        &mut params,
                        &mut in_ipl,
                        &mut out_ipl,
                    ));
                    audio_out.extend_from_slice(&out_buffer.data);
                }
            }
        }
        for _ in 0..2 {
            for (e, &effect) in effects.iter().enumerate() {
                states.push(iplPathEffectGetTailSize(effect));
                let mut out_buffer = Buffer::new(out_channels[e]);
                let mut out_ipl = out_buffer.ipl();
                states.push(iplPathEffectGetTail(effect, &mut out_ipl));
                audio_out.extend_from_slice(&out_buffer.data);
            }
        }
    }
    check_signal("path_effect_audio", &audio_out, SIGNAL);
    check_i32("path_effect_states", &states);

    // SAFETY: live objects, released once.
    unsafe {
        for effect in &mut effects {
            iplPathEffectRelease(effect);
        }
        iplHRTFRelease(&mut hrtf);
        for source in &mut sources {
            iplSourceRelease(source);
        }
        iplSimulatorRelease(&mut simulator);
        iplProbeBatchRelease(&mut batch);
    }
    setup.release();
    // SAFETY: a live context.
    unsafe { iplContextRelease(&mut context) };
}
