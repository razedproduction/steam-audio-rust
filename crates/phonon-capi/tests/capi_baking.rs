//! Parity tests for stage 8b against golden data from `tools/golden/src/capi_baking.cpp`: upstream's C API for scenes,
//! static and instanced meshes, custom scenes, energy fields, impulse responses, reconstructors, probe arrays and
//! batches, and baking, called through the same `ipl*` functions exported by this crate.
//!
//! Inputs are rebuilt from the generator's `Lcg` draws in the same order. The C API seeds a bake's reflection simulator
//! with the clock, so the bakes use materials without scattering, whose results do not depend on the random numbers.

#![allow(unsafe_code)] // The tests call the C API.

#[path = "../../steam-audio/tests/common/mod.rs"]
mod common;

use std::ffi::{CString, c_void};
use std::mem::offset_of;
use std::ptr;

use common::{
    ObjMesh, assign_materials, draw_point, draw_transform, draw01, load_obj, mesh_bounds,
};
use parity::{Element, Lcg, Tolerance, compare, golden};
use phonon::*;
use steam_audio::geometry::Aabb;
use steam_audio::math::{Matrix4, Vector3};

const SENTINEL: f32 = 9.0;

/// Signal tolerance relative to the peak.
const SIGNAL: f64 = 1e-5;

fn expected<T: Element>(name: &str) -> Vec<T> {
    golden::<T>(&format!("capi_baking/{name}")).data
}

/// Prints the largest error relative to the peak (for measuring bit-exactness with `--nocapture`).
fn report(name: &str, actual: &[f32], expected: &[f32]) {
    let peak = expected
        .iter()
        .filter(|&&x| x != SENTINEL)
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
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
    if let Err(m) = compare(actual, &expected, tol(&expected)) {
        panic!("{name}: {m}");
    }
}

#[track_caller]
fn check_signal(name: &str, actual: &[f32]) {
    check_f32(name, actual, |e| {
        let peak = e
            .iter()
            .filter(|&&x| x != SENTINEL)
            .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        Tolerance::new(SIGNAL * peak, 0.0)
    });
}

#[track_caller]
fn check_i32(name: &str, actual: &[i32]) {
    assert_eq!(actual, expected::<i32>(name), "{name}");
}

#[track_caller]
fn check_bytes(name: &str, actual: &[u8]) {
    let expected = expected::<u8>(name);
    assert!(
        actual == expected,
        "{name}: {} bytes, expected {}",
        actual.len(),
        expected.len()
    );
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

fn release_context(mut context: IPLContext) {
    // SAFETY: a live context.
    unsafe { iplContextRelease(&mut context) };
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

fn default_scene_settings() -> IPLSceneSettings {
    IPLSceneSettings {
        r#type: IPL_SCENETYPE_DEFAULT,
        closestHitCallback: None,
        anyHitCallback: None,
        batchedClosestHitCallback: None,
        batchedAnyHitCallback: None,
        userData: ptr::null_mut(),
        embreeDevice: ptr::null_mut(),
        radeonRaysDevice: ptr::null_mut(),
    }
}

fn create_scene(context: IPLContext, settings: &IPLSceneSettings) -> IPLScene {
    let mut settings = *settings;
    let mut scene = ptr::null_mut();
    // SAFETY: valid pointers and a live context.
    assert_eq!(
        unsafe { iplSceneCreate(context, &mut settings, &mut scene) },
        IPL_STATUS_SUCCESS
    );
    scene
}

/// Saves an object through `save` into a new serialized object and returns the bytes.
fn save(context: IPLContext, save: impl FnOnce(IPLSerializedObject)) -> Vec<u8> {
    let mut settings = IPLSerializedObjectSettings {
        data: ptr::null_mut(),
        size: 0,
    };
    let mut serialized = ptr::null_mut();
    // SAFETY: valid pointers and a live context.
    unsafe {
        assert_eq!(
            iplSerializedObjectCreate(context, &mut settings, &mut serialized),
            IPL_STATUS_SUCCESS
        );
    }
    save(serialized);
    // SAFETY: a live serialized object, whose data holds `size` bytes.
    unsafe {
        let data = iplSerializedObjectGetData(serialized);
        let size = iplSerializedObjectGetSize(serialized);
        let bytes = if data.is_null() {
            Vec::new()
        } else {
            std::slice::from_raw_parts(data, size).to_vec()
        };
        iplSerializedObjectRelease(&mut serialized);
        bytes
    }
}

/// A serialized object wrapping `bytes` (which must outlive it).
fn wrap(context: IPLContext, bytes: &mut [u8]) -> IPLSerializedObject {
    let mut settings = IPLSerializedObjectSettings {
        data: bytes.as_mut_ptr(),
        size: bytes.len(),
    };
    let mut serialized = ptr::null_mut();
    // SAFETY: valid pointers and a live context.
    unsafe {
        assert_eq!(
            iplSerializedObjectCreate(context, &mut settings, &mut serialized),
            IPL_STATUS_SUCCESS
        );
    }
    serialized
}

fn save_scene(context: IPLContext, scene: IPLScene) -> Vec<u8> {
    // SAFETY: live objects.
    save(context, |s| unsafe { iplSceneSave(scene, s) })
}

fn save_static_mesh(context: IPLContext, mesh: IPLStaticMesh) -> Vec<u8> {
    // SAFETY: live objects.
    save(context, |s| unsafe { iplStaticMeshSave(mesh, s) })
}

fn save_probe_batch(context: IPLContext, batch: IPLProbeBatch) -> Vec<u8> {
    // SAFETY: live objects.
    save(context, |s| unsafe { iplProbeBatchSave(batch, s) })
}

fn to_ipl_material(m: &steam_audio::scene::Material) -> IPLMaterial {
    IPLMaterial {
        absorption: m.absorption,
        scattering: m.scattering,
        transmission: m.transmission,
    }
}

/// A static mesh created through the C API from an OBJ mesh.
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
    let mut materials: Vec<IPLMaterial> = obj.materials.iter().map(to_ipl_material).collect();

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

fn create_instanced_mesh(
    scene: IPLScene,
    sub_scene: IPLScene,
    transform: IPLMatrix4x4,
) -> IPLInstancedMesh {
    let mut settings = IPLInstancedMeshSettings {
        subScene: sub_scene,
        transform,
    };
    let mut mesh = ptr::null_mut();
    // SAFETY: live scenes and valid pointers.
    assert_eq!(
        unsafe { iplInstancedMeshCreate(scene, &mut settings, &mut mesh) },
        IPL_STATUS_SUCCESS
    );
    mesh
}

/// A point in the middle 40% of a box along each axis (the generator's `interiorPoint`).
fn interior_point(lcg: &mut Lcg, aabb: &Aabb) -> IPLVector3 {
    let mut p = [0.0f32; 3];
    for (i, p) in p.iter_mut().enumerate() {
        let t = 0.3 + 0.4 * draw01(lcg);
        *p = aabb.min[i] + (aabb.max[i] - aabb.min[i]) * t;
    }
    IPLVector3 {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

fn make_specular(mesh: &mut ObjMesh) {
    for m in &mut mesh.materials {
        m.scattering = 0.0;
    }
}

/// The boxroom with three Lcg materials, made specular.
fn specular_room(lcg: &mut Lcg) -> ObjMesh {
    let mut room = load_obj("boxroom");
    assign_materials(lcg, &mut room);
    make_specular(&mut room);
    room
}

fn create_probe_array(context: IPLContext) -> IPLProbeArray {
    let mut probes = ptr::null_mut();
    // SAFETY: a live context and a valid pointer.
    assert_eq!(
        unsafe { iplProbeArrayCreate(context, &mut probes) },
        IPL_STATUS_SUCCESS
    );
    probes
}

fn create_probe_batch(context: IPLContext) -> IPLProbeBatch {
    let mut batch = ptr::null_mut();
    // SAFETY: a live context and a valid pointer.
    assert_eq!(
        unsafe { iplProbeBatchCreate(context, &mut batch) },
        IPL_STATUS_SUCCESS
    );
    batch
}

fn append_probes(out: &mut Vec<f32>, probes: IPLProbeArray) {
    // SAFETY: a live probe array.
    let n = unsafe { iplProbeArrayGetNumProbes(probes) };
    out.push(n as f32);
    for i in 0..n {
        // SAFETY: a live probe array.
        let probe = unsafe { iplProbeArrayGetProbe(probes, i) };
        out.extend([probe.center.x, probe.center.y, probe.center.z, probe.radius]);
    }
}

unsafe extern "system" fn on_progress(progress: f32, user_data: *mut c_void) {
    // SAFETY: the tests pass a `Vec<f32>`.
    unsafe { (*user_data.cast::<Vec<f32>>()).push(progress) };
}

fn create_energy_field(context: IPLContext, duration: f32, order: i32) -> IPLEnergyField {
    let mut settings = IPLEnergyFieldSettings { duration, order };
    let mut field = ptr::null_mut();
    // SAFETY: a live context and valid pointers.
    assert_eq!(
        unsafe { iplEnergyFieldCreate(context, &mut settings, &mut field) },
        IPL_STATUS_SUCCESS
    );
    field
}

fn create_impulse_response(
    context: IPLContext,
    duration: f32,
    order: i32,
    rate: i32,
) -> IPLImpulseResponse {
    let mut settings = IPLImpulseResponseSettings {
        duration,
        order,
        samplingRate: rate,
    };
    let mut ir = ptr::null_mut();
    // SAFETY: a live context and valid pointers.
    assert_eq!(
        unsafe { iplImpulseResponseCreate(context, &mut settings, &mut ir) },
        IPL_STATUS_SUCCESS
    );
    ir
}

// --------------------------------------------------------------------------------------------------------------------
// Struct layouts
// --------------------------------------------------------------------------------------------------------------------

#[test]
#[cfg(target_pointer_width = "64")]
fn struct_layouts_match_phonon_h() {
    let mut layout: Vec<i32> = Vec::new();
    macro_rules! size {
        ($($t:ty),+) => { $(layout.push(size_of::<$t>() as i32);)+ };
    }
    macro_rules! field {
        ($t:ty, $($f:tt),+) => { $(layout.push(offset_of!($t, $f) as i32);)+ };
    }

    size!(
        IPLSceneType,
        IPLProbeGenerationType,
        IPLBakedDataVariation,
        IPLBakedDataType,
        IPLReflectionsBakeFlags
    );

    size!(IPLTriangle);
    field!(IPLTriangle, indices);
    size!(IPLMaterial);
    field!(IPLMaterial, absorption, scattering, transmission);
    size!(IPLRay);
    field!(IPLRay, origin, direction);
    size!(IPLHit);
    field!(
        IPLHit,
        distance,
        triangleIndex,
        objectIndex,
        materialIndex,
        normal,
        material
    );
    size!(IPLSceneSettings);
    field!(
        IPLSceneSettings,
        r#type,
        closestHitCallback,
        anyHitCallback,
        batchedClosestHitCallback,
        batchedAnyHitCallback,
        userData,
        embreeDevice,
        radeonRaysDevice
    );
    size!(IPLStaticMeshSettings);
    field!(
        IPLStaticMeshSettings,
        numVertices,
        numTriangles,
        numMaterials,
        vertices,
        triangles,
        materialIndices,
        materials
    );
    size!(IPLInstancedMeshSettings);
    field!(IPLInstancedMeshSettings, subScene, transform);

    size!(IPLEnergyFieldSettings);
    field!(IPLEnergyFieldSettings, duration, order);
    size!(IPLImpulseResponseSettings);
    field!(IPLImpulseResponseSettings, duration, order, samplingRate);
    size!(IPLReconstructorSettings);
    field!(
        IPLReconstructorSettings,
        maxDuration,
        maxOrder,
        samplingRate
    );
    size!(IPLReconstructorInputs);
    field!(IPLReconstructorInputs, energyField);
    size!(IPLReconstructorSharedInputs);
    field!(IPLReconstructorSharedInputs, duration, order);
    size!(IPLReconstructorOutputs);
    field!(IPLReconstructorOutputs, impulseResponse);

    size!(IPLProbeGenerationParams);
    field!(IPLProbeGenerationParams, r#type, spacing, height, transform);
    size!(IPLBakedDataIdentifier);
    field!(IPLBakedDataIdentifier, r#type, variation, endpointInfluence);

    size!(IPLReflectionsBakeParams);
    field!(
        IPLReflectionsBakeParams,
        scene,
        probeBatch,
        sceneType,
        identifier,
        bakeFlags,
        numRays,
        numDiffuseSamples,
        numBounces,
        simulatedDuration,
        savedDuration,
        order,
        numThreads,
        rayBatchSize,
        irradianceMinDistance,
        bakeBatchSize,
        openCLDevice,
        radeonRaysDevice
    );
    size!(IPLPathBakeParams);
    field!(
        IPLPathBakeParams,
        scene,
        probeBatch,
        identifier,
        numSamples,
        radius,
        threshold,
        visRange,
        pathRange,
        numThreads
    );

    check_i32("layout", &layout);
}

// --------------------------------------------------------------------------------------------------------------------
// Energy fields and impulse responses
// --------------------------------------------------------------------------------------------------------------------

/// The functions of energy fields or impulse responses, so that one script runs on both.
struct Api<H> {
    rows_per_channel: usize,
    num_channels: unsafe extern "system" fn(H) -> i32,
    length: unsafe extern "system" fn(H) -> i32,
    data: unsafe extern "system" fn(H) -> *mut f32,
    channel: unsafe extern "system" fn(H, i32) -> *mut f32,
    scale: unsafe extern "system" fn(H, f32, H),
    scale_accum: unsafe extern "system" fn(H, f32, H),
    add: unsafe extern "system" fn(H, H, H),
    copy: unsafe extern "system" fn(H, H),
    swap: unsafe extern "system" fn(H, H),
    reset: unsafe extern "system" fn(H),
}

/// The generator's `generateOperations`. Every call is on live objects.
#[allow(clippy::undocumented_unsafe_blocks)]
fn run_operations<H: Copy>(name: &str, objects: [H; 3], lcg: &mut Lcg, api: &Api<H>) {
    let len = |o: H| unsafe {
        (api.num_channels)(o) as usize * api.rows_per_channel * (api.length)(o) as usize
    };
    let mut data = Vec::new();
    let mut sizes = Vec::new();
    let mut record = |data: &mut Vec<f32>| {
        for o in objects {
            unsafe {
                sizes.extend([(api.num_channels)(o), (api.length)(o)]);
                data.extend_from_slice(std::slice::from_raw_parts((api.data)(o), len(o)));
            }
        }
    };

    for o in objects {
        let values = unsafe { std::slice::from_raw_parts_mut((api.data)(o), len(o)) };
        for v in values {
            *v = lcg.next_signed();
        }
    }
    record(&mut data);

    let [a, b, c] = objects;
    unsafe {
        (api.scale)(a, 0.5, a);
        record(&mut data);
        (api.add)(a, b, c);
        record(&mut data);
        (api.add)(a, c, a);
        record(&mut data);
        (api.scale_accum)(b, -1.25, a);
        record(&mut data);
        (api.scale_accum)(c, 2.0, c);
        record(&mut data);
        (api.copy)(a, b);
        record(&mut data);
        (api.swap)(a, c);
        record(&mut data);
        (api.reset)(b);
        record(&mut data);
        (api.scale)(c, 3.0, b);
        record(&mut data);
        (api.add)(b, b, b);
        record(&mut data);
    }

    let mut offsets = Vec::new();
    for o in objects {
        unsafe {
            let base = (api.data)(o);
            for i in 0..(api.num_channels)(o) {
                offsets.push((api.channel)(o, i).offset_from(base) as i32);
            }
        }
    }

    check_f32(name, &data, |_| Tolerance::F32);
    check_i32(&format!("{name}_sizes"), &sizes);
    check_i32(&format!("{name}_offsets"), &offsets);
}

#[test]
fn energy_fields_and_impulse_responses_match_upstream() {
    let context = make_context();
    let mut lcg = Lcg::new(811);

    let fields = [
        create_energy_field(context, 0.105, 1),
        create_energy_field(context, 0.07, 2),
        create_energy_field(context, 0.13, 0),
    ];

    // SAFETY: a live energy field.
    let band_offsets: Vec<i32> = unsafe {
        let base = iplEnergyFieldGetData(fields[0]);
        (0..iplEnergyFieldGetNumChannels(fields[0]))
            .flat_map(|c| (0..3).map(move |b| (c, b)))
            .map(|(c, b)| iplEnergyFieldGetBand(fields[0], c, b).offset_from(base) as i32)
            .collect()
    };
    check_i32("energy_field_band_offsets", &band_offsets);

    let api = Api {
        rows_per_channel: 3,
        num_channels: iplEnergyFieldGetNumChannels,
        length: iplEnergyFieldGetNumBins,
        data: iplEnergyFieldGetData,
        channel: iplEnergyFieldGetChannel,
        scale: iplEnergyFieldScale,
        scale_accum: iplEnergyFieldScaleAccum,
        add: iplEnergyFieldAdd,
        copy: iplEnergyFieldCopy,
        swap: iplEnergyFieldSwap,
        reset: iplEnergyFieldReset,
    };
    run_operations("energy_fields", fields, &mut lcg, &api);
    for mut field in fields {
        // SAFETY: a live energy field.
        unsafe { iplEnergyFieldRelease(&mut field) };
    }

    let irs = [
        create_impulse_response(context, 0.0021, 1, 48000),
        create_impulse_response(context, 0.0013, 0, 44100),
        create_impulse_response(context, 0.0017, 2, 22050),
    ];
    let api = Api {
        rows_per_channel: 1,
        num_channels: iplImpulseResponseGetNumChannels,
        length: iplImpulseResponseGetNumSamples,
        data: iplImpulseResponseGetData,
        channel: iplImpulseResponseGetChannel,
        scale: iplImpulseResponseScale,
        scale_accum: iplImpulseResponseScaleAccum,
        add: iplImpulseResponseAdd,
        copy: iplImpulseResponseCopy,
        swap: iplImpulseResponseSwap,
        reset: iplImpulseResponseReset,
    };
    run_operations("impulse_responses", irs, &mut lcg, &api);
    for mut ir in irs {
        // SAFETY: a live impulse response.
        unsafe { iplImpulseResponseRelease(&mut ir) };
    }

    release_context(context);
}

#[test]
fn energy_field_and_impulse_response_edge_cases() {
    let context = make_context();

    // Invalid settings.
    for (duration, order) in [(-1.0, 1), (f32::NAN, 1), (f32::INFINITY, 0), (0.1, -1)] {
        let mut settings = IPLEnergyFieldSettings { duration, order };
        let mut field = ptr::null_mut();
        // SAFETY: valid pointers.
        let status = unsafe { iplEnergyFieldCreate(context, &mut settings, &mut field) };
        assert_eq!(status, IPL_STATUS_FAILURE, "{duration} {order}");
        assert!(field.is_null());
    }
    for (duration, order, rate) in [
        (0.1, 1, 0),
        (0.1, 1, -48000),
        (-0.1, 0, 48000),
        (0.1, -2, 48000),
    ] {
        let mut settings = IPLImpulseResponseSettings {
            duration,
            order,
            samplingRate: rate,
        };
        let mut ir = ptr::null_mut();
        // SAFETY: valid pointers.
        let status = unsafe { iplImpulseResponseCreate(context, &mut settings, &mut ir) };
        assert_eq!(status, IPL_STATUS_FAILURE, "{duration} {order} {rate}");
    }
    let mut huge = IPLEnergyFieldSettings {
        duration: 1e30,
        order: 3,
    };
    let mut field = ptr::null_mut();
    // SAFETY: valid pointers.
    assert_eq!(
        unsafe { iplEnergyFieldCreate(context, &mut huge, &mut field) },
        IPL_STATUS_OUTOFMEMORY
    );

    // SAFETY: null and live handles.
    unsafe {
        assert_eq!(iplEnergyFieldGetNumChannels(ptr::null_mut()), 0);
        assert!(iplEnergyFieldGetData(ptr::null_mut()).is_null());
        assert!(iplImpulseResponseGetChannel(ptr::null_mut(), 0).is_null());
        iplEnergyFieldCopy(ptr::null_mut(), ptr::null_mut());
        iplEnergyFieldAdd(ptr::null_mut(), ptr::null_mut(), ptr::null_mut());

        let field = create_energy_field(context, 0.03, 1);
        assert!(iplEnergyFieldGetChannel(field, 4).is_null());
        assert!(iplEnergyFieldGetChannel(field, -1).is_null());
        assert!(iplEnergyFieldGetBand(field, 0, 3).is_null());
        let empty = create_energy_field(context, 0.0, 0);
        assert_eq!(iplEnergyFieldGetNumBins(empty), 0);
        assert!(iplEnergyFieldGetData(empty).is_null());

        // A swap with itself and a copy into itself change nothing.
        let data = std::slice::from_raw_parts_mut(iplEnergyFieldGetData(field), 4 * 3 * 3);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32;
        }
        iplEnergyFieldSwap(field, field);
        iplEnergyFieldCopy(field, field);
        let data = std::slice::from_raw_parts(iplEnergyFieldGetData(field), 4 * 3 * 3);
        assert!(data.iter().enumerate().all(|(i, &v)| v == i as f32));

        // Swapping exchanges the arrays: the data pointers move with them.
        let field_data = iplEnergyFieldGetData(field);
        iplEnergyFieldSwap(field, empty);
        assert_eq!(iplEnergyFieldGetData(empty), field_data);
        assert!(iplEnergyFieldGetData(field).is_null());

        let mut retained = iplEnergyFieldRetain(empty);
        let mut empty = empty;
        iplEnergyFieldRelease(&mut empty);
        assert!(empty.is_null());
        assert_eq!(iplEnergyFieldGetNumChannels(retained), 4);
        iplEnergyFieldRelease(&mut retained);
        let mut field = field;
        iplEnergyFieldRelease(&mut field);
    }

    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Reconstructor
// --------------------------------------------------------------------------------------------------------------------

/// The generator's `fillEnergies`.
fn fill_energies(field: IPLEnergyField, lcg: &mut Lcg) {
    // SAFETY: a live energy field.
    unsafe {
        let channels = iplEnergyFieldGetNumChannels(field);
        let bins = iplEnergyFieldGetNumBins(field) as usize;
        for c in 0..channels {
            for b in 0..3 {
                let band = std::slice::from_raw_parts_mut(iplEnergyFieldGetBand(field, c, b), bins);
                for (i, v) in band.iter_mut().enumerate() {
                    let u = if c == 0 {
                        draw01(lcg)
                    } else {
                        0.25 * lcg.next_signed()
                    };
                    *v = u * 1e-2 / (1.0 + i as f32);
                }
            }
        }
    }
}

#[test]
fn reconstructor_matches_upstream() {
    let context = make_context();
    let mut lcg = Lcg::new(821);

    let mut settings = IPLReconstructorSettings {
        maxDuration: 0.1,
        maxOrder: 2,
        samplingRate: 24000,
    };
    let mut reconstructor = ptr::null_mut();
    // SAFETY: valid pointers and a live context.
    unsafe {
        assert_eq!(
            iplReconstructorCreate(context, &mut settings, &mut reconstructor),
            IPL_STATUS_SUCCESS
        );
    }

    let fields = [
        create_energy_field(context, 0.1, 1),
        create_energy_field(context, 0.05, 2),
    ];
    for &field in &fields {
        fill_energies(field, &mut lcg);
    }
    let irs = [
        create_impulse_response(context, 0.1, 1, 24000),
        create_impulse_response(context, 0.08, 2, 24000),
        create_impulse_response(context, 0.03, 0, 24000),
    ];
    for &ir in &irs {
        // SAFETY: a live impulse response.
        unsafe {
            let n = (iplImpulseResponseGetNumChannels(ir) * iplImpulseResponseGetNumSamples(ir))
                as usize;
            for v in std::slice::from_raw_parts_mut(iplImpulseResponseGetData(ir), n) {
                *v = lcg.next_signed();
            }
        }
    }

    let mut data = Vec::new();
    let record = |data: &mut Vec<f32>| {
        for &ir in &irs {
            // SAFETY: a live impulse response.
            unsafe {
                let n = (iplImpulseResponseGetNumChannels(ir) * iplImpulseResponseGetNumSamples(ir))
                    as usize;
                data.extend_from_slice(std::slice::from_raw_parts(
                    iplImpulseResponseGetData(ir),
                    n,
                ));
            }
        }
    };
    let reconstruct =
        |inputs: &[(IPLEnergyField, IPLImpulseResponse)], duration: f32, order: i32| {
            let mut ins: Vec<_> = inputs
                .iter()
                .map(|&(f, _)| IPLReconstructorInputs { energyField: f })
                .collect();
            let mut outs: Vec<_> = inputs
                .iter()
                .map(|&(_, ir)| IPLReconstructorOutputs {
                    impulseResponse: ir,
                })
                .collect();
            let mut shared = IPLReconstructorSharedInputs { duration, order };
            // SAFETY: valid arrays of live handles.
            unsafe {
                iplReconstructorReconstruct(
                    reconstructor,
                    inputs.len() as i32,
                    ins.as_mut_ptr(),
                    &mut shared,
                    outs.as_mut_ptr(),
                );
            }
        };

    reconstruct(&[(fields[0], irs[0]), (fields[1], irs[1])], 0.1, 1);
    record(&mut data);
    reconstruct(&[(fields[1], irs[2]), (fields[0], irs[1])], 0.05, 2);
    record(&mut data);
    reconstruct(&[(fields[0], irs[1])], 0.0, 3);
    record(&mut data);
    check_signal("reconstructed", &data);

    // A null handle, a negative order, or an IR longer than the maximum duration: nothing is reconstructed.
    let long = create_impulse_response(context, 0.2, 0, 24000);
    let before = data.clone();
    reconstruct(&[(fields[0], irs[0]), (ptr::null_mut(), irs[1])], 0.1, 1);
    reconstruct(&[(fields[0], irs[0])], 0.1, -1);
    reconstruct(&[(fields[0], irs[0]), (fields[0], long)], 0.2, 0);
    let mut after = Vec::new();
    record(&mut after);
    assert_eq!(after[..], before[before.len() - after.len()..]);

    // SAFETY: live objects.
    unsafe {
        let mut long = long;
        iplImpulseResponseRelease(&mut long);
        for mut ir in irs {
            iplImpulseResponseRelease(&mut ir);
        }
        for mut field in fields {
            iplEnergyFieldRelease(&mut field);
        }
        iplReconstructorRelease(&mut reconstructor);
    }
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn scenes_match_upstream() {
    let context = make_context();
    let mut lcg = Lcg::new(823);

    let room = specular_room(&mut lcg);
    let small_box = load_obj("smallbox");
    let bounds = mesh_bounds(&room);

    let scene = create_scene(context, &default_scene_settings());
    let sub_scene = create_scene(context, &default_scene_settings());

    let room_mesh = create_static_mesh(scene, &room);
    let box_mesh = create_static_mesh(sub_scene, &small_box);
    // SAFETY: live objects.
    unsafe {
        iplStaticMeshAdd(room_mesh, scene);
        // The sub-scene's mesh is added but not committed: the parent's commit commits it.
        iplStaticMeshAdd(box_mesh, sub_scene);
    }

    let instance = create_instanced_mesh(
        scene,
        sub_scene,
        to_ipl_matrix(&draw_transform(&mut lcg, &bounds)),
    );
    // SAFETY: live objects.
    unsafe {
        iplInstancedMeshAdd(instance, scene);
        iplSceneCommit(scene);
    }

    let scene_bytes = save_scene(context, scene);
    check_bytes("scene", &scene_bytes);
    let mesh_bytes = save_static_mesh(context, room_mesh);
    check_bytes("static_mesh", &mesh_bytes);

    let dir = std::env::temp_dir().join(format!("phonon_capi_baking_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The `mtllib` line names the `.mtl` file, so the name is the generator's.
    let obj_path = dir.join("steam_audio_capi_baking_scene.obj");
    let name = CString::new(obj_path.to_str().unwrap()).unwrap();
    // SAFETY: a live scene and a NUL-terminated name.
    unsafe { iplSceneSaveOBJ(scene, name.as_ptr()) };
    check_bytes("scene_obj", &std::fs::read(&obj_path).unwrap());
    check_bytes(
        "scene_mtl",
        &std::fs::read(obj_path.with_extension("mtl")).unwrap(),
    );
    std::fs::remove_dir_all(&dir).unwrap();

    // Probes: a centroid probe in a random box, then floor probes before and after the instance moves.
    let mut probe_data = Vec::new();
    let probes = create_probe_array(context);
    let mut params = IPLProbeGenerationParams {
        r#type: IPL_PROBEGENERATIONTYPE_CENTROID,
        spacing: 0.0,
        height: 0.0,
        transform: to_ipl_matrix(&draw_transform(&mut lcg, &bounds)),
    };
    // SAFETY: live objects and valid pointers.
    unsafe { iplProbeArrayGenerateProbes(probes, scene, &mut params) };
    append_probes(&mut probe_data, probes);

    let extents = bounds.extents();
    params.r#type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
    params.spacing = 0.2 * extents.x.max(extents.z);
    params.height = 1.5;
    params.transform = box_transform(&bounds);
    // SAFETY: live objects and valid pointers.
    unsafe { iplProbeArrayGenerateProbes(probes, scene, &mut params) };
    append_probes(&mut probe_data, probes);

    let moved = to_ipl_matrix(&draw_transform(&mut lcg, &bounds));
    // SAFETY: live objects and valid pointers.
    unsafe {
        iplInstancedMeshUpdateTransform(instance, scene, moved);
        iplSceneCommit(scene);
        iplProbeArrayGenerateProbes(probes, scene, &mut params);
    }
    append_probes(&mut probe_data, probes);
    check_f32("scene_probes", &probe_data, |_| Tolerance::F32);

    let mut material = IPLMaterial::default();
    for a in &mut material.absorption {
        *a = draw01(&mut lcg);
    }
    material.scattering = draw01(&mut lcg);
    for t in &mut material.transmission {
        *t = draw01(&mut lcg);
    }
    // SAFETY: live objects and a valid material.
    unsafe {
        iplStaticMeshSetMaterial(room_mesh, scene, &mut material, 1);
        // Out of range: ignored.
        iplStaticMeshSetMaterial(room_mesh, scene, &mut material, 3);
        iplStaticMeshSetMaterial(room_mesh, scene, &mut material, -1);
        iplSceneCommit(scene);
    }
    check_bytes("scene_material", &save_scene(context, scene));

    // SAFETY: live objects.
    unsafe {
        iplInstancedMeshRemove(instance, scene);
        iplStaticMeshRemove(room_mesh, scene);
        iplSceneCommit(scene);
    }
    check_bytes("scene_empty", &save_scene(context, scene));

    // Loading: the scene and the static mesh come back as saved.
    let mut scene_data = scene_bytes.clone();
    let serialized = wrap(context, &mut scene_data);
    let mut settings = default_scene_settings();
    let mut loaded = ptr::null_mut();
    // SAFETY: live objects and valid pointers.
    unsafe {
        assert_eq!(
            iplSceneLoad(
                context,
                &mut settings,
                serialized,
                None,
                ptr::null_mut(),
                &mut loaded
            ),
            IPL_STATUS_SUCCESS
        );
    }
    assert_eq!(save_scene(context, loaded), scene_bytes);

    let mut mesh_data = mesh_bytes.clone();
    let mesh_serialized = wrap(context, &mut mesh_data);
    let fresh = create_scene(context, &default_scene_settings());
    let mut loaded_mesh = ptr::null_mut();
    // SAFETY: live objects and valid pointers.
    unsafe {
        assert_eq!(
            iplStaticMeshLoad(
                fresh,
                mesh_serialized,
                None,
                ptr::null_mut(),
                &mut loaded_mesh
            ),
            IPL_STATUS_SUCCESS
        );
        iplStaticMeshAdd(loaded_mesh, fresh);
        iplSceneCommit(fresh);
    }
    assert_eq!(save_static_mesh(context, loaded_mesh), mesh_bytes);
    assert_eq!(save_scene(context, fresh), scene_bytes);

    // SAFETY: live objects.
    unsafe {
        for mut s in [serialized, mesh_serialized] {
            iplSerializedObjectRelease(&mut s);
        }
        let mut probes = probes;
        iplProbeArrayRelease(&mut probes);
        let mut instance = instance;
        iplInstancedMeshRelease(&mut instance);
        for mut m in [room_mesh, box_mesh, loaded_mesh] {
            iplStaticMeshRelease(&mut m);
        }
        for mut s in [scene, sub_scene, loaded, fresh] {
            iplSceneRelease(&mut s);
        }
    }
    release_context(context);
}

#[test]
fn scene_edge_cases() {
    let context = make_context();
    let room = load_obj("smallbox");

    // SAFETY: valid pointers, null handles and live objects throughout.
    unsafe {
        for (scene_type, status) in [
            (IPL_SCENETYPE_EMBREE, IPL_STATUS_INITIALIZATION),
            (IPL_SCENETYPE_RADEONRAYS, IPL_STATUS_INITIALIZATION),
            (7, IPL_STATUS_INITIALIZATION),
        ] {
            let mut settings = default_scene_settings();
            settings.r#type = scene_type;
            let mut scene = ptr::null_mut();
            assert_eq!(iplSceneCreate(context, &mut settings, &mut scene), status);
            assert!(scene.is_null());
        }
        let mut settings = default_scene_settings();
        let mut scene = ptr::null_mut();
        assert_eq!(
            iplSceneCreate(ptr::null_mut(), &mut settings, &mut scene),
            IPL_STATUS_FAILURE
        );
        assert_eq!(
            iplSceneCreate(context, ptr::null_mut(), &mut scene),
            IPL_STATUS_FAILURE
        );

        let scene = create_scene(context, &default_scene_settings());
        let other = create_scene(context, &default_scene_settings());

        // Invalid static mesh settings.
        let mut vertices: Vec<IPLVector3> = room.vertices.iter().copied().map(to_ipl).collect();
        let mut triangles = vec![IPLTriangle { indices: [0, 1, 2] }];
        let mut material_indices = vec![0];
        let mut materials = vec![IPLMaterial::default()];
        let mut settings = IPLStaticMeshSettings {
            numVertices: vertices.len() as i32,
            numTriangles: 1,
            numMaterials: 1,
            vertices: vertices.as_mut_ptr(),
            triangles: triangles.as_mut_ptr(),
            materialIndices: material_indices.as_mut_ptr(),
            materials: materials.as_mut_ptr(),
        };
        let mut mesh = ptr::null_mut();
        assert_eq!(
            iplStaticMeshCreate(scene, &mut settings, &mut mesh),
            IPL_STATUS_SUCCESS
        );
        let mut bad = settings;
        bad.numTriangles = 0;
        let mut ignored = ptr::null_mut();
        assert_eq!(
            iplStaticMeshCreate(scene, &mut bad, &mut ignored),
            IPL_STATUS_FAILURE
        );
        triangles[0].indices[2] = vertices.len() as i32;
        settings.triangles = triangles.as_mut_ptr();
        assert_eq!(
            iplStaticMeshCreate(scene, &mut settings, &mut ignored),
            IPL_STATUS_FAILURE
        );
        triangles[0].indices[2] = 2;
        material_indices[0] = 1;
        settings.triangles = triangles.as_mut_ptr();
        settings.materialIndices = material_indices.as_mut_ptr();
        assert_eq!(
            iplStaticMeshCreate(scene, &mut settings, &mut ignored),
            IPL_STATUS_FAILURE
        );
        material_indices[0] = 0;
        settings.materialIndices = material_indices.as_mut_ptr();
        bad = settings;
        bad.materials = ptr::null_mut();
        assert_eq!(
            iplStaticMeshCreate(scene, &mut bad, &mut ignored),
            IPL_STATUS_FAILURE
        );
        assert!(ignored.is_null());

        // A mesh is only added to the scene it was created in.
        iplStaticMeshAdd(mesh, other);
        iplSceneCommit(other);
        let empty = save_scene(context, other);
        iplStaticMeshAdd(mesh, scene);
        iplSceneCommit(scene);
        assert_ne!(save_scene(context, scene), empty);

        // Releasing the last handle keeps the mesh in the scene, as upstream's lists keep it alive.
        let with_mesh = save_scene(context, scene);
        iplStaticMeshRelease(&mut mesh);
        assert!(mesh.is_null());
        iplSceneCommit(scene);
        assert_eq!(save_scene(context, scene), with_mesh);

        // A custom sub-scene cannot be instanced in a default scene; null sub-scenes are rejected.
        let mut custom_settings = default_scene_settings();
        custom_settings.r#type = IPL_SCENETYPE_CUSTOM;
        let custom = create_scene(context, &custom_settings);
        let mut instance_settings = IPLInstancedMeshSettings {
            subScene: custom,
            transform: IPLMatrix4x4::default(),
        };
        let mut instance = ptr::null_mut();
        assert_eq!(
            iplInstancedMeshCreate(scene, &mut instance_settings, &mut instance),
            IPL_STATUS_FAILURE
        );
        instance_settings.subScene = ptr::null_mut();
        assert_eq!(
            iplInstancedMeshCreate(scene, &mut instance_settings, &mut instance),
            IPL_STATUS_FAILURE
        );

        // In a custom scene, meshes are empty objects that every function ignores; saving does nothing.
        let mut custom_mesh = ptr::null_mut();
        assert_eq!(
            iplStaticMeshCreate(custom, &mut settings, &mut custom_mesh),
            IPL_STATUS_SUCCESS
        );
        assert!(!custom_mesh.is_null());
        iplStaticMeshAdd(custom_mesh, custom);
        iplSceneCommit(custom);
        assert!(save_scene(context, custom).is_empty());
        assert!(save_static_mesh(context, custom_mesh).is_empty());
        iplStaticMeshRelease(&mut custom_mesh);

        // Loading: other types, missing data and garbage.
        let mut garbage = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let serialized = wrap(context, &mut garbage);
        let mut loaded = ptr::null_mut();
        let mut load_settings = default_scene_settings();
        assert_eq!(
            iplSceneLoad(
                context,
                &mut load_settings,
                serialized,
                None,
                ptr::null_mut(),
                &mut loaded
            ),
            IPL_STATUS_FAILURE
        );
        load_settings.r#type = IPL_SCENETYPE_CUSTOM;
        assert_eq!(
            iplSceneLoad(
                context,
                &mut load_settings,
                serialized,
                None,
                ptr::null_mut(),
                &mut loaded
            ),
            IPL_STATUS_INITIALIZATION
        );
        let mut loaded_mesh = ptr::null_mut();
        assert_eq!(
            iplStaticMeshLoad(scene, serialized, None, ptr::null_mut(), &mut loaded_mesh),
            IPL_STATUS_FAILURE
        );
        assert!(loaded.is_null() && loaded_mesh.is_null());

        // Saving into a serialized object made from the caller's data replaces its contents.
        iplSceneSave(scene, serialized);
        assert_eq!(iplSerializedObjectGetSize(serialized), with_mesh.len());
        assert_eq!(garbage, [1, 2, 3, 4, 5, 6, 7, 8]);

        // Null handles.
        iplSceneCommit(ptr::null_mut());
        iplSceneSave(ptr::null_mut(), serialized);
        iplStaticMeshAdd(ptr::null_mut(), scene);
        iplInstancedMeshUpdateTransform(ptr::null_mut(), scene, IPLMatrix4x4::default());
        assert!(iplSceneRetain(ptr::null_mut()).is_null());

        let mut serialized = serialized;
        iplSerializedObjectRelease(&mut serialized);
        for mut s in [scene, other, custom] {
            iplSceneRelease(&mut s);
        }
    }
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Custom scenes: an analytic axis-aligned room
// --------------------------------------------------------------------------------------------------------------------

struct Room {
    lo: [f32; 3],
    hi: [f32; 3],
    material: IPLMaterial,
}

/// The generator's `intersectRoom`: the closest face along the ray at a distance in `[min, max]`, as `(distance,
/// axis)`.
fn intersect_room(
    room: &Room,
    ray: &IPLRay,
    min_distance: f32,
    max_distance: f32,
) -> Option<(f32, usize)> {
    let o = [ray.origin.x, ray.origin.y, ray.origin.z];
    let d = [ray.direction.x, ray.direction.y, ray.direction.z];

    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    let mut near_axis = None;
    let mut far_axis = None;
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < room.lo[a] || o[a] > room.hi[a] {
                return None;
            }
            continue;
        }
        let t1 = (room.lo[a] - o[a]) / d[a];
        let t2 = (room.hi[a] - o[a]) / d[a];
        let t_min = if t1 < t2 { t1 } else { t2 };
        let t_max = if t1 < t2 { t2 } else { t1 };
        if t_min > t_near {
            t_near = t_min;
            near_axis = Some(a);
        }
        if t_max < t_far {
            t_far = t_max;
            far_axis = Some(a);
        }
    }
    if t_near > t_far {
        return None;
    }
    if let Some(a) = near_axis {
        if t_near >= min_distance && t_near <= max_distance {
            return Some((t_near, a));
        }
    }
    if let Some(a) = far_axis {
        if t_far >= min_distance && t_far <= max_distance {
            return Some((t_far, a));
        }
    }
    None
}

unsafe extern "system" fn room_closest_hit(
    ray: *const IPLRay,
    min_distance: f32,
    max_distance: f32,
    hit: *mut IPLHit,
    user_data: *mut c_void,
) {
    // SAFETY: the scene passes a `Room` and valid pointers.
    let (room, ray, hit) = unsafe { (&*user_data.cast::<Room>(), &*ray, &mut *hit) };
    *hit = IPLHit {
        distance: f32::INFINITY,
        triangleIndex: -1,
        objectIndex: -1,
        materialIndex: -1,
        normal: IPLVector3::default(),
        material: ptr::null_mut(),
    };
    if max_distance <= min_distance {
        return;
    }
    let Some((distance, axis)) = intersect_room(room, ray, min_distance, max_distance) else {
        return;
    };
    let d = [ray.direction.x, ray.direction.y, ray.direction.z];
    let mut n = [0.0f32; 3];
    n[axis] = if d[axis] > 0.0 { -1.0 } else { 1.0 };
    *hit = IPLHit {
        distance,
        triangleIndex: 2 * axis as i32 + i32::from(d[axis] > 0.0),
        objectIndex: 0,
        materialIndex: 0,
        normal: IPLVector3 {
            x: n[0],
            y: n[1],
            z: n[2],
        },
        material: ptr::from_ref(&room.material).cast_mut(),
    };
}

unsafe extern "system" fn room_any_hit(
    ray: *const IPLRay,
    min_distance: f32,
    max_distance: f32,
    occluded: *mut u8,
    user_data: *mut c_void,
) {
    // SAFETY: the scene passes a `Room` and valid pointers.
    let (room, ray, occluded) = unsafe { (&*user_data.cast::<Room>(), &*ray, &mut *occluded) };
    *occluded = if max_distance <= min_distance {
        1
    } else {
        u8::from(intersect_room(room, ray, min_distance, max_distance).is_some())
    };
}

unsafe extern "system" fn room_batched_closest_hit(
    num_rays: i32,
    rays: *const IPLRay,
    min_distances: *const f32,
    max_distances: *const f32,
    hits: *mut IPLHit,
    user_data: *mut c_void,
) {
    for i in 0..num_rays as usize {
        // SAFETY: the scene passes arrays of `num_rays` elements.
        unsafe {
            room_closest_hit(
                rays.add(i),
                *min_distances.add(i),
                *max_distances.add(i),
                hits.add(i),
                user_data,
            );
        }
    }
}

unsafe extern "system" fn room_batched_any_hit(
    num_rays: i32,
    rays: *const IPLRay,
    min_distances: *const f32,
    max_distances: *const f32,
    occluded: *mut u8,
    user_data: *mut c_void,
) {
    for i in 0..num_rays as usize {
        // SAFETY: the scene passes arrays of `num_rays` elements.
        unsafe {
            if *max_distances.add(i) < *min_distances.add(i) {
                *occluded.add(i) = 1;
            } else {
                room_any_hit(
                    rays.add(i),
                    *min_distances.add(i),
                    *max_distances.add(i),
                    occluded.add(i),
                    user_data,
                );
            }
        }
    }
}

fn bake_params(
    scene: IPLScene,
    batch: IPLProbeBatch,
    scene_type: IPLSceneType,
    identifier: IPLBakedDataIdentifier,
    flags: IPLReflectionsBakeFlags,
) -> IPLReflectionsBakeParams {
    IPLReflectionsBakeParams {
        scene,
        probeBatch: batch,
        sceneType: scene_type,
        identifier,
        bakeFlags: flags,
        numRays: 256,
        numDiffuseSamples: 8,
        numBounces: 4,
        simulatedDuration: 0.3,
        savedDuration: 0.2,
        order: 1,
        numThreads: 1,
        rayBatchSize: 37,
        irradianceMinDistance: 1.0,
        bakeBatchSize: 1,
        openCLDevice: ptr::null_mut(),
        radeonRaysDevice: ptr::null_mut(),
    }
}

fn bake(context: IPLContext, params: IPLReflectionsBakeParams, progress: &mut Vec<f32>) {
    let mut params = params;
    // SAFETY: live objects, and a callback that takes a `Vec<f32>`.
    unsafe {
        iplReflectionsBakerBake(
            context,
            &mut params,
            Some(on_progress),
            ptr::from_mut(progress).cast(),
        );
    }
}

/// The generator's `appendBakedData`.
fn append_baked_data(
    fields: &mut Vec<f32>,
    reverbs: &mut Vec<f32>,
    context: IPLContext,
    batch: IPLProbeBatch,
    identifier: IPLBakedDataIdentifier,
) {
    let mut field = create_energy_field(context, 0.2, 1);
    let mut identifier = identifier;
    // SAFETY: live objects and valid pointers.
    unsafe {
        let size =
            (iplEnergyFieldGetNumChannels(field) * 3 * iplEnergyFieldGetNumBins(field)) as usize;
        for i in 0..iplProbeBatchGetNumProbes(batch) {
            let data = std::slice::from_raw_parts_mut(iplEnergyFieldGetData(field), size);
            data.fill(SENTINEL);
            iplProbeBatchGetEnergyField(batch, &mut identifier, i, field);
            fields.extend_from_slice(std::slice::from_raw_parts(
                iplEnergyFieldGetData(field),
                size,
            ));

            let mut reverb = [SENTINEL; 3];
            iplProbeBatchGetReverb(batch, &mut identifier, i, reverb.as_mut_ptr());
            reverbs.extend(reverb);
        }
        iplEnergyFieldRelease(&mut field);
    }
}

const REVERB_ID: IPLBakedDataIdentifier = IPLBakedDataIdentifier {
    r#type: IPL_BAKEDDATATYPE_REFLECTIONS,
    variation: IPL_BAKEDDATAVARIATION_REVERB,
    endpointInfluence: IPLSphere {
        center: IPLVector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        radius: 0.0,
    },
};

#[test]
fn custom_scenes_match_upstream() {
    let context = make_context();
    let mut lcg = Lcg::new(827);

    let mut room = Room {
        lo: [0.0; 3],
        hi: [0.0; 3],
        material: IPLMaterial::default(),
    };
    for i in 0..3 {
        room.lo[i] = -2.0 - 3.0 * draw01(&mut lcg);
        room.hi[i] = 2.0 + 3.0 * draw01(&mut lcg);
    }
    for a in &mut room.material.absorption {
        *a = 0.05 + 0.5 * draw01(&mut lcg);
    }
    room.material.scattering = 0.0;
    for t in &mut room.material.transmission {
        *t = draw01(&mut lcg);
    }

    let bounds = Aabb {
        min: Vector3::new(room.lo[0], room.lo[1], room.lo[2]),
        max: Vector3::new(room.hi[0], room.hi[1], room.hi[2]),
    };

    let mut settings = default_scene_settings();
    settings.r#type = IPL_SCENETYPE_CUSTOM;
    settings.closestHitCallback = Some(room_closest_hit);
    settings.anyHitCallback = Some(room_any_hit);
    settings.batchedClosestHitCallback = Some(room_batched_closest_hit);
    settings.batchedAnyHitCallback = Some(room_batched_any_hit);
    settings.userData = ptr::from_ref(&room).cast_mut().cast();
    let scene = create_scene(context, &settings);
    let probes = create_probe_array(context);
    let mut params = IPLProbeGenerationParams {
        r#type: IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR,
        spacing: 1.7,
        height: 1.2,
        transform: box_transform(&bounds),
    };
    // SAFETY: live objects and valid pointers.
    unsafe { iplProbeArrayGenerateProbes(probes, scene, &mut params) };
    let mut probe_data = Vec::new();
    append_probes(&mut probe_data, probes);
    check_f32("custom_probes", &probe_data, |_| Tolerance::F32);

    let batch = create_probe_batch(context);
    // SAFETY: live objects.
    unsafe {
        iplProbeBatchAddProbeArray(batch, probes);
        iplProbeBatchCommit(batch);
    }

    let mut progress = Vec::new();
    let mut fields = Vec::new();
    let mut reverbs = Vec::new();
    let both = IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION | IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC;
    let mut params = bake_params(scene, batch, IPL_SCENETYPE_CUSTOM, REVERB_ID, both);
    bake(context, params, &mut progress);
    append_baked_data(&mut fields, &mut reverbs, context, batch, REVERB_ID);
    params.sceneType = IPL_SCENETYPE_DEFAULT;
    bake(context, params, &mut progress);
    append_baked_data(&mut fields, &mut reverbs, context, batch, REVERB_ID);

    check_f32("custom_progress", &progress, |_| Tolerance::EXACT);
    check_signal("custom_fields", &fields);
    check_f32("custom_reverbs", &reverbs, |_| Tolerance::F32);

    // SAFETY: live objects.
    unsafe {
        let mut batch = batch;
        iplProbeBatchRelease(&mut batch);
        let mut probes = probes;
        iplProbeArrayRelease(&mut probes);
        let mut scene = scene;
        iplSceneRelease(&mut scene);
    }
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Probe batches and baking
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn baking_matches_upstream() {
    let context = make_context();
    let mut lcg = Lcg::new(829);

    let room = specular_room(&mut lcg);
    let mut small_box = load_obj("smallbox");
    make_specular(&mut small_box);
    let bounds = mesh_bounds(&room);

    let scene = create_scene(context, &default_scene_settings());
    let sub_scene = create_scene(context, &default_scene_settings());
    let room_mesh = create_static_mesh(scene, &room);
    let box_mesh = create_static_mesh(sub_scene, &small_box);
    // SAFETY: live objects.
    unsafe {
        iplStaticMeshAdd(room_mesh, scene);
        iplStaticMeshAdd(box_mesh, sub_scene);
        iplSceneCommit(sub_scene);
    }
    let instance = create_instanced_mesh(
        scene,
        sub_scene,
        to_ipl_matrix(&draw_transform(&mut lcg, &bounds)),
    );
    // SAFETY: live objects.
    unsafe {
        iplInstancedMeshAdd(instance, scene);
        iplSceneCommit(scene);
    }

    let probes = create_probe_array(context);
    let extents = bounds.extents();
    let mut params = IPLProbeGenerationParams {
        r#type: IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR,
        spacing: 0.25 * extents.x.max(extents.z),
        height: 1.5,
        transform: box_transform(&bounds),
    };
    // SAFETY: live objects and valid pointers.
    unsafe { iplProbeArrayGenerateProbes(probes, scene, &mut params) };

    let batch = create_probe_batch(context);
    let extra = IPLSphere {
        center: to_ipl(draw_point(&mut lcg, &bounds)),
        radius: draw01(&mut lcg) + 0.5,
    };
    // SAFETY: live objects.
    unsafe {
        iplProbeBatchAddProbeArray(batch, probes);
        iplProbeBatchAddProbe(batch, extra);
        iplProbeBatchRemoveProbe(batch, 3);
        // Out of range: ignored.
        iplProbeBatchRemoveProbe(batch, iplProbeBatchGetNumProbes(batch));
        iplProbeBatchRemoveProbe(batch, -1);
        iplProbeBatchCommit(batch);
    }
    check_bytes("batch_probes", &save_probe_batch(context, batch));

    let radius = 0.35 * extents.length();
    let reverb_id = REVERB_ID;
    let mut source_id = REVERB_ID;
    source_id.variation = IPL_BAKEDDATAVARIATION_STATICSOURCE;
    source_id.endpointInfluence.center = interior_point(&mut lcg, &bounds);
    source_id.endpointInfluence.radius = radius;
    let mut listener_id = REVERB_ID;
    listener_id.variation = IPL_BAKEDDATAVARIATION_STATICLISTENER;
    listener_id.endpointInfluence.center = interior_point(&mut lcg, &bounds);
    listener_id.endpointInfluence.radius = radius;
    let path_id = IPLBakedDataIdentifier {
        r#type: IPL_BAKEDDATATYPE_PATHING,
        variation: IPL_BAKEDDATAVARIATION_DYNAMIC,
        ..REVERB_ID
    };

    let mut progress = Vec::new();
    let mut progress_counts = Vec::new();
    let mut data_sizes = Vec::new();
    let record_sizes = |data_sizes: &mut Vec<i32>| {
        for mut id in [reverb_id, source_id, listener_id, path_id] {
            // SAFETY: a live batch and a valid identifier.
            data_sizes.push(unsafe { iplProbeBatchGetDataSize(batch, &mut id) } as i32);
        }
    };

    let both = IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION | IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC;
    for (id, flags) in [
        (reverb_id, both),
        (source_id, IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION),
        (listener_id, IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC),
    ] {
        let before = progress.len();
        bake(
            context,
            bake_params(scene, batch, IPL_SCENETYPE_DEFAULT, id, flags),
            &mut progress,
        );
        progress_counts.push((progress.len() - before) as i32);
        record_sizes(&mut data_sizes);
    }

    let mut fields = Vec::new();
    let mut reverbs = Vec::new();
    for id in [reverb_id, source_id, listener_id] {
        append_baked_data(&mut fields, &mut reverbs, context, batch, id);
    }
    check_signal("baked_fields", &fields);
    // A reverb time comes from a line fitted to the log of the energies, down to the faint tail, so off Windows the
    // ulp differences of libm's `powf` in the fields (8e-9 x peak) move it by up to 6.2e-4 relative (measured on
    // Linux and macOS CI); on Windows it is bit-exact.
    let reverb_tolerance = if cfg!(windows) {
        Tolerance::F32
    } else {
        Tolerance::new(1e-6, 1e-3)
    };
    check_f32("baked_reverbs", &reverbs, |_| reverb_tolerance);

    let mut path_progress: Vec<f32> = Vec::new();
    let mut path_params = IPLPathBakeParams {
        scene,
        probeBatch: batch,
        identifier: path_id,
        numSamples: 1,
        radius: 0.5,
        threshold: 0.5,
        visRange: 1000.0,
        pathRange: 1000.0,
        numThreads: 1,
    };
    // SAFETY: live objects, and a callback that takes a `Vec<f32>`.
    unsafe {
        iplPathBakerBake(
            context,
            &mut path_params,
            Some(on_progress),
            ptr::from_mut(&mut path_progress).cast(),
        );
    }
    assert_eq!(path_progress.last(), Some(&1.0));
    record_sizes(&mut data_sizes);

    let baked = save_probe_batch(context, batch);
    // Off Windows, libm's `powf` (specular lobe) and `atan2f` (path deviations) may differ by an ulp; the energy
    // fields and reverbs above are checked with tolerances everywhere.
    if cfg!(windows) {
        check_bytes("batch_baked", &baked);
    }

    let mut source = source_id;
    // SAFETY: a live batch and a valid identifier.
    unsafe { iplProbeBatchRemoveData(batch, &mut source) };
    record_sizes(&mut data_sizes);

    check_f32("bake_progress", &progress, |_| Tolerance::EXACT);
    check_i32("bake_progress_counts", &progress_counts);
    check_i32("baked_data_sizes", &data_sizes);

    // A loaded batch saves the same bytes.
    let mut baked_data = baked.clone();
    let serialized = wrap(context, &mut baked_data);
    let mut loaded = ptr::null_mut();
    // SAFETY: live objects and valid pointers.
    unsafe {
        assert_eq!(
            iplProbeBatchLoad(context, serialized, &mut loaded),
            IPL_STATUS_SUCCESS
        );
    }
    assert_eq!(save_probe_batch(context, loaded), baked);

    // SAFETY: live objects.
    unsafe {
        let mut serialized = serialized;
        iplSerializedObjectRelease(&mut serialized);
        for mut b in [batch, loaded] {
            iplProbeBatchRelease(&mut b);
        }
        let mut probes = probes;
        iplProbeArrayRelease(&mut probes);
        let mut instance = instance;
        iplInstancedMeshRelease(&mut instance);
        for mut m in [room_mesh, box_mesh] {
            iplStaticMeshRelease(&mut m);
        }
        for mut s in [scene, sub_scene] {
            iplSceneRelease(&mut s);
        }
    }
    release_context(context);
}

#[test]
fn probe_and_baking_edge_cases() {
    let context = make_context();
    let scene = create_scene(context, &default_scene_settings());
    let probes = create_probe_array(context);
    let batch = create_probe_batch(context);

    // SAFETY: valid pointers, null handles and live objects throughout.
    unsafe {
        // An unknown generation type does nothing; a centroid probe replaces the array.
        let mut params = IPLProbeGenerationParams {
            r#type: 2,
            spacing: 1.0,
            height: 1.0,
            transform: box_transform(&Aabb {
                min: Vector3::new(-1.0, -2.0, -3.0),
                max: Vector3::new(1.0, 2.0, 3.0),
            }),
        };
        iplProbeArrayGenerateProbes(probes, scene, &mut params);
        assert_eq!(iplProbeArrayGetNumProbes(probes), 0);
        params.r#type = IPL_PROBEGENERATIONTYPE_CENTROID;
        iplProbeArrayGenerateProbes(probes, scene, &mut params);
        assert_eq!(iplProbeArrayGetNumProbes(probes), 1);
        let probe = iplProbeArrayGetProbe(probes, 0);
        assert_eq!((probe.center, probe.radius), (IPLVector3::default(), 1.0));
        assert_eq!(iplProbeArrayGetProbe(probes, 1), IPLSphere::default());
        assert_eq!(iplProbeArrayGetProbe(probes, -1), IPLSphere::default());
        assert_eq!(iplProbeArrayGetNumProbes(ptr::null_mut()), 0);

        // Baking into an empty batch, with invalid parameters or with the wrong layer type does nothing.
        let mut progress: Vec<f32> = Vec::new();
        let both =
            IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION | IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC;
        bake(
            context,
            bake_params(scene, batch, IPL_SCENETYPE_DEFAULT, REVERB_ID, both),
            &mut progress,
        );
        iplProbeBatchAddProbeArray(batch, probes);
        iplProbeBatchCommit(batch);
        let mut invalid = [
            bake_params(scene, batch, IPL_SCENETYPE_EMBREE, REVERB_ID, both),
            bake_params(scene, batch, IPL_SCENETYPE_DEFAULT, REVERB_ID, 0),
            IPLReflectionsBakeParams {
                numThreads: 0,
                ..bake_params(scene, batch, IPL_SCENETYPE_DEFAULT, REVERB_ID, both)
            },
            IPLReflectionsBakeParams {
                simulatedDuration: f32::NAN,
                ..bake_params(scene, batch, IPL_SCENETYPE_DEFAULT, REVERB_ID, both)
            },
            IPLReflectionsBakeParams {
                rayBatchSize: 0,
                ..bake_params(scene, batch, IPL_SCENETYPE_CUSTOM, REVERB_ID, both)
            },
            bake_params(
                scene,
                batch,
                IPL_SCENETYPE_DEFAULT,
                IPLBakedDataIdentifier {
                    r#type: IPL_BAKEDDATATYPE_PATHING,
                    ..REVERB_ID
                },
                both,
            ),
            bake_params(
                ptr::null_mut(),
                batch,
                IPL_SCENETYPE_DEFAULT,
                REVERB_ID,
                both,
            ),
        ];
        for params in &mut invalid {
            iplReflectionsBakerBake(
                context,
                params,
                Some(on_progress),
                ptr::from_mut(&mut progress).cast(),
            );
        }
        assert!(progress.is_empty());
        let mut id = REVERB_ID;
        assert_eq!(iplProbeBatchGetDataSize(batch, &mut id), 0);
        let mut path_params = IPLPathBakeParams {
            scene,
            probeBatch: batch,
            identifier: REVERB_ID,
            numSamples: 1,
            radius: 0.5,
            threshold: 0.5,
            visRange: 10.0,
            pathRange: 10.0,
            numThreads: 1,
        };
        iplPathBakerBake(context, &mut path_params, None, ptr::null_mut());
        assert_eq!(iplProbeBatchGetDataSize(batch, &mut id), 0);

        // A bake in an empty scene: one probe, nothing hit, so a zero field and minimum reverb times.
        let mut progress = Vec::new();
        bake(
            context,
            bake_params(scene, batch, IPL_SCENETYPE_DEFAULT, REVERB_ID, both),
            &mut progress,
        );
        assert_eq!(progress, [1.0]);
        assert!(iplProbeBatchGetDataSize(batch, &mut id) > 0);
        let mut reverb = [SENTINEL; 3];
        iplProbeBatchGetReverb(batch, &mut id, 0, reverb.as_mut_ptr());
        assert_eq!(reverb, [0.1; 3]);
        let mut unknown = IPLBakedDataIdentifier { variation: 9, ..id };
        assert_eq!(iplProbeBatchGetDataSize(batch, &mut unknown), 0);
        iplProbeBatchRemoveData(batch, &mut unknown);
        iplProbeBatchRemoveData(batch, ptr::null_mut());
        assert!(iplProbeBatchGetDataSize(batch, &mut id) > 0);

        // Cancelling with no bake running does nothing.
        iplReflectionsBakerCancelBake(context);
        iplPathBakerCancelBake(context);
        iplReflectionsBakerCancelBake(ptr::null_mut());

        // Garbage does not load.
        let mut garbage = vec![0u8; 16];
        let mut serialized = wrap(context, &mut garbage);
        let mut loaded = ptr::null_mut();
        assert_eq!(
            iplProbeBatchLoad(context, serialized, &mut loaded),
            IPL_STATUS_FAILURE
        );
        assert!(loaded.is_null());
        iplSerializedObjectRelease(&mut serialized);

        let mut batch = batch;
        iplProbeBatchRelease(&mut batch);
        let mut probes = probes;
        iplProbeArrayRelease(&mut probes);
        let mut scene = scene;
        iplSceneRelease(&mut scene);
    }
    release_context(context);
}
