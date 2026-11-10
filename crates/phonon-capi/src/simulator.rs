//! Simulators and sources (upstream `api_simulator.cpp`).
//!
//! A simulator is the core's [`SimulationManager`] behind a mutex, so its functions (including the `Run*` ones) take
//! turns; a source is the core's [`SimulationData`], shared with the simulators it is added to. Upstream lets one
//! thread run direct simulation while another runs reflections on the same simulator; here the second waits for the
//! first (`DEVIATIONS.md`).
//!
//! The reflection simulators are seeded with the clock, as upstream's (risk R2), and the reverb effects of a hybrid
//! simulator's estimator draw their delay lengths from the calling thread's `rand()` state ([`crate::rand`]).
//!
//! A source's impulse response is handed to reflection effects through its triple buffer: the reading side is taken out
//! of the source when it is created and kept apart ([`ReflectionIr`]), so that audio threads never wait for a running
//! simulation. `IPLReflectionEffectIR` points at it; like upstream's pointer into the source, it is valid while the
//! source is.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use steam_audio::baking::BakedDataIdentifier;
use steam_audio::deviation::DeviationModel;
use steam_audio::distance_attenuation::DistanceAttenuationModel;
use steam_audio::effects::OverlapSaveFir;
use steam_audio::math::Vector3;
use steam_audio::probes::ProbeBatch;
use steam_audio::simulation::{
    DirectSimulationFlags, IndirectEffectType, OcclusionType, ReconstructionType, SceneType,
    SharedDirectSimulationInputs, SharedPathingSimulationInputs, SharedReflectionSimulationInputs,
    SharedSimulationData, SimulationData, SimulationManager, SimulationSettings,
};
use steam_audio::triple_buffer::TripleBufferReader;

use crate::context::{Context, context_arc};
use crate::handle::{Handle, lock, object, retain_release, validator};
use crate::models::{UserData, air_absorption_model, directivity};
use crate::probes::{ProbeBatchObject, from_ipl_identifier};
use crate::rand::with_rand;
use crate::scene::scene_arc;
use crate::types::*;
use crate::validation::{
    Validator, air_absorption_model as validate_air_absorption_model, baked_data_identifier,
    coordinate_space3, direct_effect_params, directivity as validate_directivity,
    distance_attenuation_model as validate_distance_attenuation_model, path_effect_params,
    reflection_effect_params,
};
use crate::{from_ipl_coordinate_space, guard, to_ipl_vector};

// --------------------------------------------------------------------------------------------------------------------
// Objects
// --------------------------------------------------------------------------------------------------------------------

/// The probe batches added to a simulator, as the batch objects and the copies the simulator holds.
///
/// A batch object is changed copy-on-write ([`crate::probes`]); the simulator keeps the batch as it was when first
/// added, and sources that name the batch for pathing are given that same copy, which the core looks up by identity.
type BatchRegistry = Mutex<Vec<(Arc<ProbeBatchObject>, Arc<ProbeBatch>)>>;

pub(crate) struct SimulatorObject {
    manager: Mutex<SimulationManager>,
    settings: SimulationSettings,
    batches: Arc<BatchRegistry>,
    /// Whether a scene has been set (simulating reflections or paths needs one).
    has_scene: Mutex<bool>,
}

/// The reading side of a source's impulse response triple buffer (what `IPLReflectionEffectIR` points at).
pub(crate) struct ReflectionIr(pub(crate) Mutex<TripleBufferReader<OverlapSaveFir>>);

/// The last distance attenuation callback a source was given, so that the same callback and user data give the same
/// model (the core compares callbacks by identity, upstream by function and user data).
type CallbackCache = Option<(usize, usize, DistanceAttenuationModel)>;

pub(crate) struct SourceObject {
    data: SharedSimulationData,
    ir: Option<Arc<ReflectionIr>>,
    batches: Arc<BatchRegistry>,
    has_reflections: bool,
    has_pathing: bool,
    max_order: usize,
    distance_callback: Mutex<CallbackCache>,
}

/// # Safety
///
/// `simulator` is null or a live simulator.
unsafe fn simulator_ref<'a>(simulator: IPLSimulator) -> Option<&'a Handle<SimulatorObject>> {
    // SAFETY: guaranteed by the caller.
    unsafe { Handle::<SimulatorObject>::from_raw(simulator) }
}

/// # Safety
///
/// `source` is null or a live source.
unsafe fn source_ref<'a>(source: IPLSource) -> Option<&'a Handle<SourceObject>> {
    // SAFETY: guaranteed by the caller.
    unsafe { Handle::<SourceObject>::from_raw(source) }
}

const ALL_SIMULATION_FLAGS: IPLSimulationFlags =
    IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS | IPL_SIMULATIONFLAGS_PATHING;

/// `VALIDATE_IPLSimulationFlags(<name>)`.
fn validate_flags(v: Validator, name: &str, flags: IPLSimulationFlags) {
    v.flags("IPLSimulationFlags", name, flags, ALL_SIMULATION_FLAGS);
}

/// `VALIDATE_IPLSimulationSettings(settings)`.
///
/// # Safety
///
/// `settings` is null or valid.
unsafe fn validate_simulation_settings(v: Validator, settings: *const IPLSimulationSettings) {
    if !v.pointer("settings", settings) {
        return;
    }
    // SAFETY: not null; valid as guaranteed by the caller.
    let s = unsafe { &*settings };
    validate_flags(v, "settings->flags", s.flags);
    v.range(
        "IPLSceneType",
        "settings->sceneType",
        s.sceneType,
        IPL_SCENETYPE_DEFAULT,
        IPL_SCENETYPE_CUSTOM,
    );
    v.range(
        "IPLReflectionEffectType",
        "settings->reflectionType",
        s.reflectionType,
        IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
        IPL_REFLECTIONEFFECTTYPE_TAN,
    );
    v.int(
        "settings->maxNumOcclusionSamples",
        s.maxNumOcclusionSamples,
        s.maxNumOcclusionSamples > 0,
    );
    v.int("settings->maxNumRays", s.maxNumRays, s.maxNumRays > 0);
    v.int(
        "settings->numDiffuseSamples",
        s.numDiffuseSamples,
        s.numDiffuseSamples > 0,
    );
    v.float_if("settings->maxDuration", s.maxDuration, s.maxDuration > 0.0);
    v.int("settings->maxOrder", s.maxOrder, s.maxOrder >= 0);
    v.int(
        "settings->maxNumSources",
        s.maxNumSources,
        s.maxNumSources > 0,
    );
    v.int("settings->numThreads", s.numThreads, s.numThreads > 0);
    if s.sceneType == IPL_SCENETYPE_CUSTOM {
        v.int("settings->rayBatchSize", s.rayBatchSize, s.rayBatchSize > 0);
    }
    v.int(
        "settings->numVisSamples",
        s.numVisSamples,
        s.numVisSamples > 0,
    );
    v.int("settings->samplingRate", s.samplingRate, s.samplingRate > 0);
    v.int("settings->frameSize", s.frameSize, s.frameSize > 0);
    let radeon_rays = s.sceneType == IPL_SCENETYPE_RADEONRAYS;
    let tan = s.reflectionType == IPL_REFLECTIONEFFECTTYPE_TAN;
    if radeon_rays || tan {
        v.pointer("settings->openCLDevice", s.openCLDevice);
        if radeon_rays {
            v.pointer("settings->radeonRaysDevice", s.radeonRaysDevice);
        }
        if tan {
            v.pointer("settings->tanDevice", s.tanDevice);
        }
    }
}

/// `VALIDATE_IPLSimulationInputs(inputs)`. `deviationModel` is read for callers of API version 4.7 or later, which
/// have the field (upstream reads it whatever the caller's version).
///
/// # Safety
///
/// `inputs` is null or valid, with a `deviationModel` that is null or valid.
unsafe fn validate_simulation_inputs(
    v: Validator,
    inputs: *const IPLSimulationInputs,
    has_deviation_model: bool,
) {
    if !v.pointer("inputs", inputs) {
        return;
    }
    // SAFETY: not null; valid as guaranteed by the caller.
    let i = unsafe { &*inputs };
    validate_flags(v, "inputs->flags", i.flags);
    coordinate_space3!(v, i.source, "inputs->source");
    if has_flag(i.flags, IPL_SIMULATIONFLAGS_DIRECT) {
        v.flags(
            "IPLDirectSimulationFlags",
            "inputs->directFlags",
            i.directFlags,
            IPL_DIRECTSIMULATIONFLAGS_DISTANCEATTENUATION
                | IPL_DIRECTSIMULATIONFLAGS_AIRABSORPTION
                | IPL_DIRECTSIMULATIONFLAGS_DIRECTIVITY
                | IPL_DIRECTSIMULATIONFLAGS_OCCLUSION
                | IPL_DIRECTSIMULATIONFLAGS_TRANSMISSION,
        );
        validate_distance_attenuation_model!(
            v,
            &i.distanceAttenuationModel,
            "(&inputs->distanceAttenuationModel)"
        );
        validate_air_absorption_model!(v, &i.airAbsorptionModel, "(&inputs->airAbsorptionModel)");
        validate_directivity!(v, &i.directivity, "(&inputs->directivity)");
        let deviation = if has_deviation_model {
            i.deviationModel.cast_const()
        } else {
            std::ptr::null()
        };
        // SAFETY: guaranteed by the caller.
        if let Some(model) = unsafe { deviation.as_ref() } {
            v.range(
                "IPLDeviationModelType",
                "(inputs->deviationModel)->type",
                model.r#type,
                IPL_DEVIATIONTYPE_DEFAULT,
                IPL_DEVIATIONTYPE_CALLBACK,
            );
            // Upstream compares with IPL_AIRABSORPTIONTYPE_CALLBACK (2), so a callback deviation model is never
            // checked for its callback.
            if model.r#type == IPL_AIRABSORPTIONTYPE_CALLBACK {
                v.pointer(
                    "(inputs->deviationModel)->callback",
                    model.callback.map_or(std::ptr::null(), |f| f as *const ()),
                );
            }
        }
        v.range(
            "IPLOcclusionType",
            "inputs->occlusionType",
            i.occlusionType,
            IPL_OCCLUSIONTYPE_RAYCAST,
            IPL_OCCLUSIONTYPE_VOLUMETRIC,
        );
        v.float_if(
            "inputs->occlusionRadius",
            i.occlusionRadius,
            i.occlusionRadius > 0.0,
        );
        v.int(
            "inputs->numOcclusionSamples",
            i.numOcclusionSamples,
            i.numOcclusionSamples > 0,
        );
        v.int(
            "inputs->numTransmissionRays",
            i.numTransmissionRays,
            i.numTransmissionRays > 0,
        );
    }
    if i.flags & (IPL_SIMULATIONFLAGS_REFLECTIONS | IPL_SIMULATIONFLAGS_PATHING) != 0 {
        v.bool("inputs->baked", i.baked);
        if i.baked != IPL_FALSE {
            baked_data_identifier!(v, i.bakedDataIdentifier, "inputs->bakedDataIdentifier");
        }
        if has_flag(i.flags, IPL_SIMULATIONFLAGS_REFLECTIONS) {
            for &x in &i.reverbScale {
                v.float_if("inputs->reverbScale[iBand]", x, x > 0.0);
            }
            v.float_if(
                "inputs->hybridReverbTransitionTime",
                i.hybridReverbTransitionTime,
                i.hybridReverbTransitionTime > 0.0,
            );
            v.float_if(
                "inputs->hybridReverbOverlapPercent",
                i.hybridReverbOverlapPercent,
                0.0 < i.hybridReverbOverlapPercent && i.hybridReverbOverlapPercent <= 1.0,
            );
        }
        if has_flag(i.flags, IPL_SIMULATIONFLAGS_PATHING) {
            v.pointer("inputs->pathingProbes", i.pathingProbes);
            v.float_if("inputs->visRadius", i.visRadius, i.visRadius >= 0.0);
            v.float_if(
                "inputs->visThreshold",
                i.visThreshold,
                i.visThreshold >= 0.0,
            );
            v.float_if("inputs->visRange", i.visRange, i.visRange >= 0.0);
            v.int("inputs->pathingOrder", i.pathingOrder, i.pathingOrder >= 0);
            v.bool("inputs->enableValidation", i.enableValidation);
            v.bool("inputs->findAlternatePaths", i.findAlternatePaths);
        }
    }
}

fn has_flag(flags: IPLSimulationFlags, flag: IPLSimulationFlags) -> bool {
    flags & flag != 0
}

/// A count from the API (negative counts are invalid).
fn count(value: IPLint32) -> Option<usize> {
    usize::try_from(value).ok()
}

/// A count from the API, with negative values read as 0.
fn count_or_zero(value: IPLint32) -> usize {
    usize::try_from(value).unwrap_or(0)
}

/// The settings a caller describes, or `None` if the manager cannot be built from them (upstream would fail later or
/// misbehave).
fn simulation_settings(s: &IPLSimulationSettings) -> Option<SimulationSettings> {
    let enable_direct = has_flag(s.flags, IPL_SIMULATIONFLAGS_DIRECT);
    let enable_indirect = has_flag(s.flags, IPL_SIMULATIONFLAGS_REFLECTIONS);
    let enable_pathing = has_flag(s.flags, IPL_SIMULATIONFLAGS_PATHING);

    // Embree, Radeon Rays and TrueAudio Next are out of scope; they matter only for reflections.
    let scene_type = match s.sceneType {
        IPL_SCENETYPE_DEFAULT => SceneType::Default,
        IPL_SCENETYPE_CUSTOM => SceneType::Custom,
        _ if !enable_indirect => SceneType::Default,
        _ => return None,
    };
    let indirect_type = match s.reflectionType {
        IPL_REFLECTIONEFFECTTYPE_CONVOLUTION => IndirectEffectType::Convolution,
        IPL_REFLECTIONEFFECTTYPE_PARAMETRIC => IndirectEffectType::Parametric,
        IPL_REFLECTIONEFFECTTYPE_HYBRID => IndirectEffectType::Hybrid,
        _ if !enable_indirect => IndirectEffectType::Convolution,
        _ => return None,
    };

    let settings = SimulationSettings {
        enable_direct,
        enable_indirect,
        enable_pathing,
        scene_type,
        indirect_type,
        max_num_occlusion_samples: count(s.maxNumOcclusionSamples)?,
        max_num_rays: count(s.maxNumRays)?,
        num_diffuse_samples: count(s.numDiffuseSamples)?,
        max_duration: s.maxDuration,
        max_order: count(s.maxOrder)?,
        max_num_sources: count(s.maxNumSources)?,
        num_threads: count(s.numThreads)?,
        ray_batch_size: count(s.rayBatchSize)?,
        num_vis_samples: count(s.numVisSamples)?,
        asymmetric_vis_range: true,
        down: Vector3::new(0.0, -1.0, 0.0),
        sampling_rate: u32::try_from(s.samplingRate).ok()?,
        frame_size: count(s.frameSize)?,
    };

    // A source made by this simulator sizes its energy fields and impulse responses from these.
    let sizes_valid = settings.max_duration.is_finite()
        && settings.max_duration >= 0.0
        && settings.sampling_rate > 0
        && settings.frame_size > 0;
    if enable_indirect
        && (!sizes_valid
            || settings.num_threads == 0
            || settings.num_diffuse_samples == 0
            || (scene_type == SceneType::Custom && settings.ray_batch_size == 0))
    {
        return None;
    }
    Some(settings)
}

/// Upstream seeds its reflection simulators with `time(NULL)` (risk R2).
fn clock_seed() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as u32)
}

// --------------------------------------------------------------------------------------------------------------------
// Simulator
// --------------------------------------------------------------------------------------------------------------------

/// Creates a simulator. Settings it cannot be built from (negative counts; with reflections, no threads, no diffuse
/// samples, a sampling rate or frame size of 0, an invalid duration, a custom scene with a ray batch size of 0, or an
/// Embree, Radeon Rays or TrueAudio Next type) give `IPL_STATUS_FAILURE`.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorCreate(
    context: IPLContext,
    settings: *mut IPLSimulationSettings,
    simulator: *mut IPLSimulator,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        if let Some(v) = context.validator("createSimulator") {
            // SAFETY: guaranteed by the caller.
            unsafe { validate_simulation_settings(v, settings) };
            v.pointer("simulator", simulator);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) =
            (unsafe { settings.as_ref() }, unsafe { simulator.as_mut() })
        else {
            return IPL_STATUS_FAILURE;
        };
        let Some(settings) = simulation_settings(settings) else {
            return IPL_STATUS_FAILURE;
        };

        let manager = with_rand(|rand| SimulationManager::new(&settings, clock_seed(), rand));
        let value = SimulatorObject {
            manager: Mutex::new(manager),
            settings,
            batches: Arc::default(),
            has_scene: Mutex::new(false),
        };
        *out = Handle::into_raw(value, Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLSimulator,
    SimulatorObject,
    iplSimulatorRetain,
    iplSimulatorRelease
);

/// Sets the scene to simulate in (a null scene is ignored, as upstream). The simulator shares the scene: later changes
/// to it show once committed. A scene of another type than the simulator's works, as upstream's.
///
/// # Safety
///
/// `simulator` and `scene` are null or live objects.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorSetScene(simulator: IPLSimulator, scene: IPLScene) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SimulatorObject, _>(simulator, "setScene") } {
            v.pointer("scene", scene);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(simulator), Some(scene)) = (unsafe { simulator_ref(simulator) }, unsafe {
            scene_arc(scene)
        }) else {
            return;
        };
        let simulator = simulator.get();
        lock(&simulator.manager).set_scene(Some(scene.simulation_scene()));
        *lock(&simulator.has_scene) = true;
    })
}

/// Adds a probe batch at the next commit. The simulator uses the batch as it was when it was first added; adding it
/// again adds that same batch once more (as upstream adds the same object twice).
///
/// # Safety
///
/// `simulator` and `probeBatch` are null or live objects.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorAddProbeBatch(
    simulator: IPLSimulator,
    probeBatch: IPLProbeBatch,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SimulatorObject, _>(simulator, "addProbeBatch") } {
            v.pointer("probeBatch", probeBatch);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(simulator), Some(object)) = (unsafe { simulator_ref(simulator) }, unsafe {
            object::<ProbeBatchObject, _>(probeBatch)
        }) else {
            return;
        };
        let simulator = simulator.get();
        let batch = {
            let mut batches = lock(&simulator.batches);
            match batches.iter().find(|(o, _)| Arc::ptr_eq(o, object)) {
                Some((_, batch)) => Arc::clone(batch),
                None => {
                    let batch = Arc::clone(&lock(object));
                    batches.push((Arc::clone(object), Arc::clone(&batch)));
                    batch
                }
            }
        };
        lock(&simulator.manager).add_probe_batch(batch);
    })
}

/// Removes a probe batch at the next commit.
///
/// # Safety
///
/// `simulator` and `probeBatch` are null or live objects.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorRemoveProbeBatch(
    simulator: IPLSimulator,
    probeBatch: IPLProbeBatch,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SimulatorObject, _>(simulator, "removeProbeBatch") } {
            v.pointer("probeBatch", probeBatch);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(simulator), Some(object)) = (unsafe { simulator_ref(simulator) }, unsafe {
            object::<ProbeBatchObject, _>(probeBatch)
        }) else {
            return;
        };
        let simulator = simulator.get();
        let batch = {
            let mut batches = lock(&simulator.batches);
            let Some(i) = batches.iter().position(|(o, _)| Arc::ptr_eq(o, object)) else {
                return;
            };
            batches.remove(i).1
        };
        lock(&simulator.manager).remove_probe_batch(&batch);
    })
}

/// Called with the ends of each validation ray and whether it was occluded.
struct VisCallback {
    callback: unsafe extern "system" fn(IPLVector3, IPLVector3, IPLbool, *mut c_void),
    user_data: UserData,
}

/// Sets the inputs shared by every source, for the types of simulation in `flags`. Reflection inputs beyond the
/// simulator's maxima (rays, duration, order) are clamped to them, and negative counts read as 0 (upstream would
/// overrun its buffers). The pathing visualization callback is read for callers of API version 4.3 or later.
///
/// # Safety
///
/// `simulator` is null or a live simulator, and `sharedInputs` is null or valid, with a callback that is safe to call
/// from the thread that runs pathing.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorSetSharedInputs(
    simulator: IPLSimulator,
    flags: IPLSimulationFlags,
    sharedInputs: *mut IPLSimulationSharedInputs,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SimulatorObject, _>(simulator, "setSharedInputs") } {
            validate_flags(v, "flags", flags);
            if v.pointer("sharedInputs", sharedInputs) {
                // SAFETY: not null; valid as guaranteed by the caller.
                let shared = unsafe { &*sharedInputs };
                coordinate_space3!(v, shared.listener, "sharedInputs->listener");
                if has_flag(flags, IPL_SIMULATIONFLAGS_REFLECTIONS) {
                    v.int("sharedInputs->numRays", shared.numRays, shared.numRays > 0);
                    v.int(
                        "sharedInputs->numBounces",
                        shared.numBounces,
                        shared.numBounces > 0,
                    );
                    v.float_if(
                        "sharedInputs->duration",
                        shared.duration,
                        shared.duration > 0.0,
                    );
                    v.int("sharedInputs->order", shared.order, shared.order >= 0);
                    v.float_if(
                        "sharedInputs->irradianceMinDistance",
                        shared.irradianceMinDistance,
                        shared.irradianceMinDistance > 0.0,
                    );
                }
            }
        }
        // SAFETY: guaranteed by the caller.
        let (Some(handle), Some(shared)) = (unsafe { simulator_ref(simulator) }, unsafe {
            sharedInputs.as_ref()
        }) else {
            return;
        };
        let simulator = handle.get();
        let listener = from_ipl_coordinate_space(&shared.listener);
        let mut manager = lock(&simulator.manager);

        if has_flag(flags, IPL_SIMULATIONFLAGS_DIRECT) {
            manager.set_shared_direct_inputs(SharedDirectSimulationInputs { listener });
        }

        if has_flag(flags, IPL_SIMULATIONFLAGS_REFLECTIONS) {
            let s = &simulator.settings;
            manager.set_shared_reflection_inputs(SharedReflectionSimulationInputs {
                listener,
                num_rays: count_or_zero(shared.numRays).min(s.max_num_rays),
                num_bounces: count_or_zero(shared.numBounces),
                duration: shared.duration.clamp(0.0, s.max_duration),
                order: count_or_zero(shared.order).min(s.max_order),
                irradiance_min_distance: shared.irradianceMinDistance,
                reconstruction_type: ReconstructionType::Linear,
            });
        }

        if has_flag(flags, IPL_SIMULATIONFLAGS_PATHING) {
            let version_4_3 = handle
                .context()
                .is_some_and(|c| c.is_caller_api_version_at_least(4, 3));
            let vis_callback = shared
                .pathingVisCallback
                .filter(|_| version_4_3)
                .map(|callback| {
                    let vis = VisCallback {
                        callback,
                        user_data: UserData(shared.pathingUserData),
                    };
                    Arc::new(move |from: Vector3, to: Vector3, occluded: bool| {
                        // Captures the whole struct (not its raw pointer field alone), which is `Send` and `Sync`.
                        let vis = &vis;
                        let occluded = if occluded { IPL_TRUE } else { IPL_FALSE };
                        // SAFETY: the caller's callback, called as `phonon.h` specifies.
                        unsafe {
                            (vis.callback)(
                                to_ipl_vector(from),
                                to_ipl_vector(to),
                                occluded,
                                vis.user_data.0,
                            )
                        }
                    }) as Arc<dyn Fn(Vector3, Vector3, bool) + Send + Sync>
                });
            manager.set_shared_pathing_inputs(SharedPathingSimulationInputs {
                listener,
                vis_callback,
            });
        }
    })
}

/// Applies the probe batches and sources added and removed since the last commit.
///
/// # Safety
///
/// `simulator` is null or a live simulator.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorCommit(simulator: IPLSimulator) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(simulator) = unsafe { simulator_ref(simulator) } {
            lock(&simulator.get().manager).commit();
        }
    })
}

/// Simulates the direct sound of every committed source. Does nothing for a simulator without direct simulation
/// (upstream dereferences a null simulator).
///
/// # Safety
///
/// `simulator` is null or a live simulator, and the sources' callbacks are safe to call from this thread.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorRunDirect(simulator: IPLSimulator) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(simulator) = (unsafe { simulator_ref(simulator) }) else {
            return;
        };
        let simulator = simulator.get();
        if simulator.settings.enable_direct {
            lock(&simulator.manager).simulate_direct();
        }
    })
}

/// Simulates the reflections of every committed source with reflections enabled. Does nothing for a simulator without
/// reflections or without a scene (upstream dereferences null pointers).
///
/// # Safety
///
/// `simulator` is null or a live simulator, and the scene's and sources' callbacks are safe to call from the
/// simulation threads.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorRunReflections(simulator: IPLSimulator) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(simulator) = (unsafe { simulator_ref(simulator) }) else {
            return;
        };
        let simulator = simulator.get();
        if simulator.settings.enable_indirect && *lock(&simulator.has_scene) {
            lock(&simulator.manager).simulate_indirect();
        }
    })
}

/// Simulates the paths of every committed source with pathing enabled. Does nothing without a scene; sources whose
/// probe batch was not added to the simulator (with pathing enabled) are skipped (upstream dereferences null
/// pointers).
///
/// # Safety
///
/// `simulator` is null or a live simulator, and the callbacks are safe to call from this thread.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSimulatorRunPathing(simulator: IPLSimulator) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(simulator) = (unsafe { simulator_ref(simulator) }) else {
            return;
        };
        let simulator = simulator.get();
        if *lock(&simulator.has_scene) {
            lock(&simulator.manager).simulate_pathing();
        }
    })
}

// --------------------------------------------------------------------------------------------------------------------
// Sources
// --------------------------------------------------------------------------------------------------------------------

/// Creates a source for a simulator, with reflections and pathing if `settings.flags` asks for them.
///
/// # Safety
///
/// `simulator` is null or a live simulator, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSourceCreate(
    simulator: IPLSimulator,
    settings: *mut IPLSourceSettings,
    source: *mut IPLSource,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SimulatorObject, _>(simulator, "createSource") } {
            if v.pointer("settings", settings) {
                // SAFETY: not null; valid as guaranteed by the caller.
                validate_flags(v, "settings->flags", unsafe { (*settings).flags });
            }
            v.pointer("source", source);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(handle), Some(settings), Some(out)) = (
            unsafe { simulator_ref(simulator) },
            unsafe { settings.as_ref() },
            unsafe { source.as_mut() },
        ) else {
            return IPL_STATUS_FAILURE;
        };
        let Some(context) = handle.context() else {
            return IPL_STATUS_FAILURE;
        };
        let simulator = handle.get();
        let has_reflections = has_flag(settings.flags, IPL_SIMULATIONFLAGS_REFLECTIONS);
        let has_pathing = has_flag(settings.flags, IPL_SIMULATIONFLAGS_PATHING);
        if has_reflections && !simulator.settings.enable_indirect {
            // Without reflections the simulator's sizes were not checked.
            let s = &simulator.settings;
            if !(s.max_duration.is_finite()
                && s.max_duration >= 0.0
                && s.sampling_rate > 0
                && s.frame_size > 0)
            {
                return IPL_STATUS_FAILURE;
            }
        }

        let mut data = SimulationData::new(&simulator.settings, has_reflections, has_pathing);
        let ir = data
            .reflection_outputs
            .overlap_save_fir
            .take()
            .map(|reader| Arc::new(ReflectionIr(Mutex::new(reader))));

        let value = SourceObject {
            data: Arc::new(Mutex::new(data)),
            ir,
            batches: Arc::clone(&simulator.batches),
            has_reflections,
            has_pathing,
            max_order: simulator.settings.max_order,
            distance_callback: Mutex::new(None),
        };
        *out = Handle::into_raw(value, Some(Arc::clone(context)));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(IPLSource, SourceObject, iplSourceRetain, iplSourceRelease);

/// Adds a source to a simulator at its next commit.
///
/// # Safety
///
/// `source` and `simulator` are null or live objects.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSourceAdd(source: IPLSource, simulator: IPLSimulator) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SourceObject, _>(source, "add") } {
            v.pointer("simulator", simulator);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(source), Some(simulator)) = (unsafe { source_ref(source) }, unsafe {
            simulator_ref(simulator)
        }) else {
            return;
        };
        lock(&simulator.get().manager).add_source(Arc::clone(&source.get().data));
    })
}

/// Removes a source from a simulator at its next commit.
///
/// # Safety
///
/// `source` and `simulator` are null or live objects.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSourceRemove(source: IPLSource, simulator: IPLSimulator) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SourceObject, _>(source, "remove") } {
            v.pointer("simulator", simulator);
        }
        // SAFETY: guaranteed by the caller.
        let (Some(source), Some(simulator)) = (unsafe { source_ref(source) }, unsafe {
            simulator_ref(simulator)
        }) else {
            return;
        };
        lock(&simulator.get().manager).remove_source(&source.get().data);
    })
}

impl SourceObject {
    /// The model a caller describes, or `None` for an unknown type (upstream's switch then keeps the previous model).
    /// A callback model is built once for each callback and user data.
    fn distance_attenuation_model(
        &self,
        model: &IPLDistanceAttenuationModel,
    ) -> Option<DistanceAttenuationModel> {
        match model.r#type {
            IPL_DISTANCEATTENUATIONTYPE_DEFAULT => Some(DistanceAttenuationModel::default()),
            IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE => {
                Some(DistanceAttenuationModel::InverseDistance {
                    min_distance: model.minDistance,
                })
            }
            IPL_DISTANCEATTENUATIONTYPE_CALLBACK => {
                // Upstream's callback model has a minimum distance of 1, so without a callback it is the default.
                let Some(callback) = model.callback else {
                    return Some(DistanceAttenuationModel::default());
                };
                let key = (callback as usize, model.userData as usize);
                let mut cache = lock(&self.distance_callback);
                if let Some((f, u, cached)) = cache.as_ref() {
                    if (*f, *u) == key {
                        return Some(cached.clone());
                    }
                }
                let built = crate::models::distance_attenuation_model(model);
                *cache = Some((key.0, key.1, built.clone()));
                Some(built)
            }
            _ => None,
        }
    }

    /// The batch a source names for pathing: the copy its simulator holds, or (for a batch not added to it) the batch as
    /// it is now, which the simulator then skips.
    ///
    /// # Safety
    ///
    /// `batch` is null or a live probe batch.
    unsafe fn pathing_batch(&self, batch: IPLProbeBatch) -> Option<Arc<ProbeBatch>> {
        // SAFETY: guaranteed by the caller.
        let object = unsafe { object::<ProbeBatchObject, _>(batch) }?;
        let batches = lock(&self.batches);
        Some(match batches.iter().find(|(o, _)| Arc::ptr_eq(o, object)) {
            Some((_, batch)) => Arc::clone(batch),
            None => Arc::clone(&lock(object)),
        })
    }
}

fn air_absorption_known(model: &IPLAirAbsorptionModel) -> bool {
    matches!(
        model.r#type,
        IPL_AIRABSORPTIONTYPE_DEFAULT
            | IPL_AIRABSORPTIONTYPE_EXPONENTIAL
            | IPL_AIRABSORPTIONTYPE_CALLBACK
    )
}

/// Sets a source's inputs for the types of simulation in `flags`.
///
/// As upstream, a distance attenuation or air absorption model of an unknown type keeps the previous model, and an
/// unknown occlusion type gives an occlusion of 0; an unknown baked data identifier keeps the previous one here
/// (upstream stores it, and finds no data for it). Reflections and pathing stay disabled for a source
/// created without them, the pathing order is clamped to the simulator's maximum, and negative counts read as 0
/// (upstream would overrun buffers). `numTransmissionRays` is read for callers of API version 4.3 or later, and
/// `deviationModel` for 4.7 or later.
///
/// # Safety
///
/// `source` is null or a live source, and `inputs` is null or valid, with callbacks that are safe to call from the
/// simulation threads and a `pathingProbes` and `deviationModel` that are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSourceSetInputs(
    source: IPLSource,
    flags: IPLSimulationFlags,
    inputs: *mut IPLSimulationInputs,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(v) = unsafe { validator::<SourceObject, _>(source, "setInputs") } {
            // SAFETY: guaranteed by the caller.
            let at_least_4_7 = unsafe { source_ref(source) }
                .and_then(Handle::context)
                .is_some_and(|c| c.is_caller_api_version_at_least(4, 7));
            validate_flags(v, "flags", flags);
            // SAFETY: guaranteed by the caller.
            unsafe { validate_simulation_inputs(v, inputs, at_least_4_7) };
        }
        // SAFETY: guaranteed by the caller.
        let (Some(handle), Some(inputs)) =
            (unsafe { source_ref(source) }, unsafe { inputs.as_ref() })
        else {
            return;
        };
        let object = handle.get();
        let context: Option<&Arc<Context>> = handle.context();
        let at_least = |minor| context.is_some_and(|c| c.is_caller_api_version_at_least(4, minor));
        let coordinates = from_ipl_coordinate_space(&inputs.source);

        // Built before locking the source, so that a callback model never waits for a simulation.
        let distance_model = object.distance_attenuation_model(&inputs.distanceAttenuationModel);
        let air_model = air_absorption_known(&inputs.airAbsorptionModel)
            .then(|| air_absorption_model(&inputs.airAbsorptionModel));
        let directivity = directivity(&inputs.directivity);
        // SAFETY: guaranteed by the caller.
        let probes = unsafe { object.pathing_batch(inputs.pathingProbes) };

        let mut data = lock(&object.data);

        if has_flag(flags, IPL_SIMULATIONFLAGS_DIRECT) {
            let direct = &mut data.direct_inputs;
            direct.flags = DirectSimulationFlags::from_bits(inputs.directFlags as u32);
            direct.source = coordinates;
            if let Some(model) = &distance_model {
                direct.distance_attenuation_model = model.clone();
            }
            if let Some(model) = &air_model {
                direct.air_absorption_model = model.clone();
            }
            direct.directivity = directivity.clone();
            direct.occlusion_type = match inputs.occlusionType {
                IPL_OCCLUSIONTYPE_RAYCAST => OcclusionType::Raycast,
                IPL_OCCLUSIONTYPE_VOLUMETRIC => OcclusionType::Volumetric,
                _ => OcclusionType::Unknown,
            };
            direct.occlusion_radius = inputs.occlusionRadius;
            direct.num_occlusion_samples = count_or_zero(inputs.numOcclusionSamples);
            if at_least(3) {
                direct.num_transmission_rays = count_or_zero(inputs.numTransmissionRays);
            }
        }

        if has_flag(flags, IPL_SIMULATIONFLAGS_REFLECTIONS) {
            let reflection = &mut data.reflection_inputs;
            reflection.enabled =
                object.has_reflections && has_flag(inputs.flags, IPL_SIMULATIONFLAGS_REFLECTIONS);
            reflection.source = coordinates;
            if let Some(model) = &distance_model {
                reflection.distance_attenuation_model = model.clone();
            }
            reflection.distance_attenuation_model_dirty =
                inputs.distanceAttenuationModel.dirty == IPL_TRUE;
            if let Some(model) = &air_model {
                reflection.air_absorption_model = model.clone();
            }
            reflection.directivity = directivity.clone();
            reflection.reverb_scale = inputs.reverbScale;
            reflection.transition_time = inputs.hybridReverbTransitionTime;
            reflection.overlap_fraction = inputs.hybridReverbOverlapPercent;
            reflection.baked = inputs.baked == IPL_TRUE;
            let identifier: Option<BakedDataIdentifier> =
                from_ipl_identifier(&inputs.bakedDataIdentifier);
            if let Some(identifier) = identifier {
                reflection.baked_data_identifier = identifier;
            }
        }

        if has_flag(flags, IPL_SIMULATIONFLAGS_PATHING) {
            let pathing = &mut data.pathing_inputs;
            pathing.enabled =
                object.has_pathing && has_flag(inputs.flags, IPL_SIMULATIONFLAGS_PATHING);
            pathing.source = coordinates;
            pathing.probes = probes;
            pathing.vis_radius = inputs.visRadius;
            pathing.vis_threshold = inputs.visThreshold;
            pathing.vis_range = inputs.visRange;
            pathing.order = count_or_zero(inputs.pathingOrder).min(object.max_order);
            pathing.enable_validation = inputs.enableValidation == IPL_TRUE;
            pathing.find_alternate_paths = inputs.findAlternatePaths == IPL_TRUE;
            pathing.simplify_paths = true;
            pathing.real_time_vis = true;
            if let Some(model) = distance_model {
                pathing.distance_attenuation_model = model;
            }

            pathing.deviation_model = DeviationModel::default();
            if at_least(7) {
                // SAFETY: guaranteed by the caller.
                if let Some(model) = unsafe { inputs.deviationModel.as_ref() } {
                    if model.r#type == IPL_DEVIATIONTYPE_CALLBACK {
                        if let Some(callback) = model.callback {
                            let user_data = UserData(model.userData);
                            pathing.deviation_model =
                                DeviationModel::Callback(Arc::new(move |angle, band| {
                                    let user_data = user_data;
                                    // SAFETY: the caller's callback, called as `phonon.h` specifies.
                                    unsafe { callback(angle, band as i32, user_data.0) }
                                }));
                        }
                    }
                }
            }
        }
    })
}

/// Reads a source's outputs for the types of simulation in `flags`. The reflection IR is null for a source without one
/// (parametric reverb, or no reflections), and the TrueAudio Next slot is -1. The SH coefficients point into the source
/// (null without pathing) and are valid while it is.
///
/// # Safety
///
/// `source` is null or a live source, and `outputs` is null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSourceGetOutputs(
    source: IPLSource,
    flags: IPLSimulationFlags,
    outputs: *mut IPLSimulationOutputs,
) {
    // SAFETY: guaranteed by the caller.
    let v = unsafe { validator::<SourceObject, _>(source, "getOutputs") };
    if let Some(v) = v {
        validate_flags(v, "flags", flags);
        v.pointer("outputs", outputs);
    }
    // SAFETY: guaranteed by the caller.
    unsafe { get_outputs(source, flags, outputs) };
    if let Some(v) = v {
        // SAFETY: guaranteed by the caller.
        let (Some(object), Some(out)) =
            (unsafe { source_ref(source) }, unsafe { outputs.as_ref() })
        else {
            v.pointer("outputs", outputs);
            return;
        };
        if has_flag(flags, IPL_SIMULATIONFLAGS_DIRECT) {
            direct_effect_params!(v, &out.direct, "(&outputs->direct)");
        }
        if has_flag(flags, IPL_SIMULATIONFLAGS_REFLECTIONS) {
            reflection_effect_params!(v, &out.reflections, "(&outputs->reflections)");
        }
        if has_flag(flags, IPL_SIMULATIONFLAGS_PATHING) {
            // The caller's order picks how many coefficients upstream reads from the source's array; it is limited to
            // the array here.
            let mut pathing = out.pathing;
            let max_order = i32::try_from(object.get().max_order).unwrap_or(i32::MAX);
            pathing.order = pathing.order.min(max_order);
            path_effect_params!(v, &pathing, "(&outputs->pathing)");
        }
    }
}

/// [`iplSourceGetOutputs`] without validation.
///
/// # Safety
///
/// As [`iplSourceGetOutputs`].
unsafe fn get_outputs(
    source: IPLSource,
    flags: IPLSimulationFlags,
    outputs: *mut IPLSimulationOutputs,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(handle), Some(outputs)) =
            (unsafe { source_ref(source) }, unsafe { outputs.as_mut() })
        else {
            return;
        };
        let object = handle.get();
        let mut data = lock(&object.data);

        if has_flag(flags, IPL_SIMULATIONFLAGS_DIRECT) {
            let path = &data.direct_path;
            let direct = &mut outputs.direct;
            direct.distanceAttenuation = path.distance_attenuation;
            direct.airAbsorption = path.air_absorption;
            direct.directivity = path.directivity;
            direct.occlusion = path.occlusion;
            direct.transmission = path.transmission;
        }

        if has_flag(flags, IPL_SIMULATIONFLAGS_REFLECTIONS) {
            let reflections = &mut outputs.reflections;
            let r = &data.reflection_outputs;
            reflections.ir = object
                .ir
                .as_ref()
                .map_or(std::ptr::null_mut(), |ir| Arc::as_ptr(ir).cast_mut().cast());
            reflections.numChannels = i32::try_from(r.num_channels).unwrap_or(i32::MAX);
            reflections.irSize = i32::try_from(r.num_samples).unwrap_or(i32::MAX);
            reflections.reverbTimes = r.reverb.reverb_times;
            reflections.eq = r.hybrid_eq;
            reflections.delay = i32::try_from(r.hybrid_delay).unwrap_or(i32::MAX);
            reflections.tanSlot = -1;
        }

        if has_flag(flags, IPL_SIMULATIONFLAGS_PATHING) {
            let pathing = &mut outputs.pathing;
            pathing.eqCoeffs = data.pathing_outputs.eq;
            let sh = &mut data.pathing_outputs.sh;
            pathing.shCoeffs = if sh.is_empty() {
                std::ptr::null_mut()
            } else {
                sh.as_mut_ptr()
            };
        }
    })
}
