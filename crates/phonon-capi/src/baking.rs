//! Baking reflections and paths into probe batches (upstream `api_baking.cpp`).
//!
//! The scene (read) and the probe batch are locked for the whole bake, so a progress callback must not call functions
//! that change them; [`iplReflectionsBakerCancelBake`] and [`iplPathBakerCancelBake`] can be called from anywhere.

use std::ffi::c_void;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use steam_audio::baking::{BakedDataType, BakedDataVariation, ReflectionBakeParams};
use steam_audio::math::Vector3;
use steam_audio::pathing::PathBakeParams;
use steam_audio::simulation::{
    AnyReflectionSimulator, BatchedReflectionSimulator, ReflectionSimulator,
    ReflectionSimulatorSettings,
};
use steam_audio::thread_pool::ThreadPool;

use crate::context::{context_arc, context_validator};
use crate::guard;
use crate::handle::lock;
use crate::probes::{from_ipl_identifier, probe_batch_ref};
use crate::scene::scene_arc;
use crate::types::*;
use crate::validation::{Validator, baked_data_identifier};

/// `VALIDATE_IPLReflectionsBakeParams(params)`.
///
/// # Safety
///
/// `params` is null or valid.
unsafe fn validate_reflections_bake_params(v: Validator, params: *const IPLReflectionsBakeParams) {
    if !v.pointer("params", params) {
        return;
    }
    // SAFETY: not null; valid as guaranteed by the caller.
    let p = unsafe { &*params };
    v.pointer("params->scene", p.scene);
    v.pointer("params->probeBatch", p.probeBatch);
    v.range(
        "IPLSceneType",
        "params->sceneType",
        p.sceneType,
        IPL_SCENETYPE_DEFAULT,
        IPL_SCENETYPE_CUSTOM,
    );
    baked_data_identifier!(v, p.identifier, "params->identifier");
    v.flags(
        "IPLReflectionsBakeFlags",
        "params->bakeFlags",
        p.bakeFlags,
        IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION | IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC,
    );
    v.int("params->numRays", p.numRays, p.numRays > 0);
    v.int(
        "params->numDiffuseSamples",
        p.numDiffuseSamples,
        p.numDiffuseSamples > 0,
    );
    v.int("params->numBounces", p.numBounces, p.numBounces > 0);
    v.float_if(
        "params->simulatedDuration",
        p.simulatedDuration,
        p.simulatedDuration > 0.0,
    );
    v.float_if(
        "params->savedDuration",
        p.savedDuration,
        0.0 < p.savedDuration && p.savedDuration <= p.simulatedDuration,
    );
    v.int("params->order", p.order, p.order > 0);
    v.int("params->numThreads", p.numThreads, p.numThreads > 0);
    if p.sceneType == IPL_SCENETYPE_CUSTOM {
        v.int("params->rayBatchSize", p.rayBatchSize, p.rayBatchSize > 0);
    }
    v.float_if(
        "params->irradianceMinDistance",
        p.irradianceMinDistance,
        p.irradianceMinDistance > 0.0,
    );
    if p.sceneType == IPL_SCENETYPE_RADEONRAYS {
        v.int(
            "params->bakeBatchSize",
            p.bakeBatchSize,
            p.bakeBatchSize > 0,
        );
        v.pointer("params->openCLDevice", p.openCLDevice);
        v.pointer("params->radeonRaysDevice", p.radeonRaysDevice);
    }
}

/// `VALIDATE_IPLPathBakeParams(params)`.
///
/// # Safety
///
/// `params` is null or valid.
unsafe fn validate_path_bake_params(v: Validator, params: *const IPLPathBakeParams) {
    if !v.pointer("params", params) {
        return;
    }
    // SAFETY: not null; valid as guaranteed by the caller.
    let p = unsafe { &*params };
    v.pointer("params->scene", p.scene);
    v.pointer("params->probeBatch", p.probeBatch);
    baked_data_identifier!(v, p.identifier, "params->identifier");
    v.int("params->numSamples", p.numSamples, p.numSamples > 0);
    v.float_if("params->radius", p.radius, p.radius > 0.0);
    v.float_if("params->threshold", p.threshold, p.threshold > 0.0);
    v.float_if("params->visRange", p.visRange, p.visRange > 0.0);
    v.float_if("params->pathRange", p.pathRange, p.pathRange > 0.0);
    v.int("params->numThreads", p.numThreads, p.numThreads > 0);
}

/// The seed of the reflection simulator's random number generators: the time in seconds, as upstream's
/// `time(NULL)` (risk R2).
fn clock_seed() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as u32)
}

/// A count from the API, if it is not negative.
fn count(value: IPLint32) -> Option<usize> {
    usize::try_from(value).ok()
}

/// A duration from the API, if it is finite and not negative.
fn duration(value: f32) -> Option<f32> {
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// Reports progress to the caller's callback, if any.
fn progress_reporter(callback: IPLProgressCallback, user_data: *mut c_void) -> impl FnMut(f32) {
    move |progress| {
        if let Some(callback) = callback {
            // SAFETY: the callback's contract is the caller's.
            unsafe { callback(progress, user_data) };
        }
    }
}

/// Bakes a layer of reflections (reverb, static source or static listener) into a probe batch, simulating each probe
/// with a reflection simulator built for `params.sceneType` (`IPL_SCENETYPE_CUSTOM`: batches of `rayBatchSize` rays) on
/// `numThreads` threads, whose random number generators are seeded with the clock, as upstream.
///
/// Does nothing for a null pointer or handle, a batch without probes (as upstream), an Embree or Radeon Rays scene type
/// (upstream, built without them, throws through C), a layer that is not reflections of a static variation, no data
/// to bake, or invalid counts and durations (negative, not finite, or no threads).
///
/// # Safety
///
/// `context` is null or a live context, `params` null or valid with null or live handles, and the callback (if any)
/// can be called with `userData`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionsBakerBake(
    context: IPLContext,
    params: *mut IPLReflectionsBakeParams,
    progressCallback: IPLProgressCallback,
    userData: *mut c_void,
) {
    // SAFETY: guaranteed by the caller.
    if let Some(v) = unsafe { context_validator(context, "bakeReflections") } {
        // SAFETY: guaranteed by the caller.
        unsafe { validate_reflections_bake_params(v, params) };
    }
    // SAFETY: guaranteed by the caller.
    unsafe { bake_reflections(context, params, progressCallback, userData) }
}

/// [`iplReflectionsBakerBake`] without validation.
///
/// # Safety
///
/// As [`iplReflectionsBakerBake`].
unsafe fn bake_reflections(
    context: IPLContext,
    params: *mut IPLReflectionsBakeParams,
    progressCallback: IPLProgressCallback,
    userData: *mut c_void,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(context), Some(params)) =
            (unsafe { context_arc(context) }, unsafe { params.as_ref() })
        else {
            return;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(scene), Some(batch)) = (unsafe { scene_arc(params.scene) }, unsafe {
            probe_batch_ref(params.probeBatch)
        }) else {
            return;
        };
        let Some(identifier) = from_ipl_identifier(&params.identifier) else {
            return;
        };
        if identifier.data_type != BakedDataType::Reflections
            || identifier.variation == BakedDataVariation::Dynamic
        {
            return;
        }

        let bake_convolution = params.bakeFlags & IPL_REFLECTIONSBAKEFLAGS_BAKECONVOLUTION != 0;
        let bake_parametric = params.bakeFlags & IPL_REFLECTIONSBAKEFLAGS_BAKEPARAMETRIC != 0;
        // Upstream sizes its simulator for a batch of sources only when baking static listeners.
        let max_num_sources = if identifier.variation == BakedDataVariation::StaticListener {
            count(params.bakeBatchSize)
        } else {
            Some(1)
        };
        let (
            Some(num_rays),
            Some(num_diffuse_samples),
            Some(num_bounces),
            Some(sim_duration),
            Some(bake_duration),
            Some(order),
            Some(num_threads),
            Some(max_num_sources),
        ) = (
            count(params.numRays),
            count(params.numDiffuseSamples),
            count(params.numBounces),
            duration(params.simulatedDuration),
            duration(params.savedDuration),
            count(params.order),
            count(params.numThreads).filter(|&n| n > 0),
            max_num_sources,
        )
        else {
            return;
        };
        if !(bake_convolution || bake_parametric) || lock(batch).num_probes() == 0 {
            return;
        }

        let settings = ReflectionSimulatorSettings {
            max_num_rays: num_rays,
            num_diffuse_samples,
            max_duration: sim_duration,
            max_order: order,
            max_num_sources,
            num_threads,
        };
        let mut simulator: AnyReflectionSimulator = match params.sceneType {
            IPL_SCENETYPE_DEFAULT => ReflectionSimulator::new(&settings, clock_seed()).into(),
            IPL_SCENETYPE_CUSTOM => {
                let Some(ray_batch_size) = count(params.rayBatchSize).filter(|&n| n > 0) else {
                    return;
                };
                BatchedReflectionSimulator::new(&settings, ray_batch_size, clock_seed()).into()
            }
            _ => return,
        };

        let bake_params = ReflectionBakeParams {
            identifier,
            bake_convolution,
            bake_parametric,
            num_rays,
            num_bounces,
            sim_duration,
            bake_duration,
            order,
            irradiance_min_distance: params.irradianceMinDistance,
        };
        let pool = ThreadPool::new(num_threads);

        let mut batch = lock(batch);
        if batch.num_probes() == 0 {
            return;
        }
        let batch = Arc::make_mut(&mut batch);
        scene.with_ray_tracer(|tracer| {
            context.reflection_baker.bake(
                tracer,
                &mut simulator,
                &bake_params,
                &pool,
                batch,
                progress_reporter(progressCallback, userData),
            );
        });
    })
}

/// Cancels reflection bakes running with this context, after their current probe.
///
/// # Safety
///
/// `context` is null or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionsBakerCancelBake(context: IPLContext) {
    // SAFETY: guaranteed by the caller.
    if let Some(context) = unsafe { context_arc(context) } {
        context.reflection_baker.cancel();
    }
}

/// Bakes paths between the probes of a batch into the layer `params.identifier` (replacing it), with visibility ranges
/// measured perpendicular to −y, as upstream.
///
/// Does nothing for a null pointer or handle, a batch without probes (as upstream), a layer that is not pathing data of
/// the dynamic variation, a negative sample count, or no threads.
///
/// # Safety
///
/// `context` is null or a live context, `params` null or valid with null or live handles, and the callback (if any)
/// can be called with `userData`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplPathBakerBake(
    context: IPLContext,
    params: *mut IPLPathBakeParams,
    progressCallback: IPLProgressCallback,
    userData: *mut c_void,
) {
    // SAFETY: guaranteed by the caller.
    if let Some(v) = unsafe { context_validator(context, "bakePaths") } {
        // SAFETY: guaranteed by the caller.
        unsafe { validate_path_bake_params(v, params) };
    }
    // SAFETY: guaranteed by the caller.
    unsafe { bake_paths(context, params, progressCallback, userData) }
}

/// [`iplPathBakerBake`] without validation.
///
/// # Safety
///
/// As [`iplPathBakerBake`].
unsafe fn bake_paths(
    context: IPLContext,
    params: *mut IPLPathBakeParams,
    progressCallback: IPLProgressCallback,
    userData: *mut c_void,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(context), Some(params)) =
            (unsafe { context_arc(context) }, unsafe { params.as_ref() })
        else {
            return;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(scene), Some(batch)) = (unsafe { scene_arc(params.scene) }, unsafe {
            probe_batch_ref(params.probeBatch)
        }) else {
            return;
        };
        let Some(identifier) = from_ipl_identifier(&params.identifier) else {
            return;
        };
        if identifier.data_type != BakedDataType::Pathing
            || identifier.variation != BakedDataVariation::Dynamic
        {
            return;
        }
        let (Some(num_samples), Some(num_threads)) = (
            count(params.numSamples),
            count(params.numThreads).filter(|&n| n > 0),
        ) else {
            return;
        };

        let bake_params = PathBakeParams {
            num_samples,
            radius: params.radius,
            threshold: params.threshold,
            vis_range: params.visRange,
            vis_range_real_time: params.visRange,
            path_range: params.pathRange,
            asymmetric_vis_range: true,
            down: Vector3::new(0.0, -1.0, 0.0),
            prune_vis_graph: false,
        };
        let pool = ThreadPool::new(num_threads);

        let mut batch = lock(batch);
        if batch.num_probes() == 0 {
            return;
        }
        let batch = Arc::make_mut(&mut batch);
        scene.with_ray_tracer(|tracer| {
            context.path_baker.bake(
                tracer,
                &identifier,
                &bake_params,
                &pool,
                batch,
                progress_reporter(progressCallback, userData),
            );
        });
    })
}

/// Cancels path bakes running with this context.
///
/// # Safety
///
/// `context` is null or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplPathBakerCancelBake(context: IPLContext) {
    // SAFETY: guaranteed by the caller.
    if let Some(context) = unsafe { context_arc(context) } {
        context.path_baker.cancel();
    }
}
