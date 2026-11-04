//! Probe arrays and probe batches (upstream `api_probes.cpp`).
//!
//! A batch is kept as an `Arc<ProbeBatch>` and changed copy-on-write, so that a simulator (stage 8c) can hold the
//! batch as it was committed while the caller goes on changing it.

use std::sync::{Arc, Mutex};

use steam_audio::baking::{BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation};
use steam_audio::probes::{Probe, ProbeBatch, ProbeGeneration, generate_probes};

use crate::context::context_arc;
use crate::energy_field::energy_field_ref;
use crate::handle::{Handle, lock, object, retain_release};
use crate::scene::scene_arc;
use crate::serialized_object::{store_serialized_bytes, with_serialized_bytes};
use crate::types::*;
use crate::{from_ipl_matrix, from_ipl_sphere, guard, to_ipl_sphere};

pub(crate) type ProbeArrayObject = Mutex<Vec<Probe>>;
pub(crate) type ProbeBatchObject = Mutex<Arc<ProbeBatch>>;

/// An identifier from the API; `None` for an unknown type or variation (upstream stores any value).
pub(crate) fn from_ipl_identifier(
    identifier: &IPLBakedDataIdentifier,
) -> Option<BakedDataIdentifier> {
    let data_type = match identifier.r#type {
        IPL_BAKEDDATATYPE_REFLECTIONS => BakedDataType::Reflections,
        IPL_BAKEDDATATYPE_PATHING => BakedDataType::Pathing,
        _ => return None,
    };
    let variation = match identifier.variation {
        IPL_BAKEDDATAVARIATION_REVERB => BakedDataVariation::Reverb,
        IPL_BAKEDDATAVARIATION_STATICSOURCE => BakedDataVariation::StaticSource,
        IPL_BAKEDDATAVARIATION_STATICLISTENER => BakedDataVariation::StaticListener,
        IPL_BAKEDDATAVARIATION_DYNAMIC => BakedDataVariation::Dynamic,
        _ => return None,
    };
    Some(BakedDataIdentifier {
        variation,
        data_type,
        endpoint_influence: from_ipl_sphere(identifier.endpointInfluence),
    })
}

/// A probe index from the API, if it is in range.
fn probe_index(index: IPLint32, num_probes: usize) -> Option<usize> {
    usize::try_from(index).ok().filter(|&i| i < num_probes)
}

// --------------------------------------------------------------------------------------------------------------------
// Probe arrays
// --------------------------------------------------------------------------------------------------------------------

/// Creates an empty probe array.
///
/// # Safety
///
/// `context` is null or a live context, and `probeArray` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeArrayCreate(
    context: IPLContext,
    probeArray: *mut IPLProbeArray,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let (Some(context), Some(out)) = (unsafe { context_arc(context) }, unsafe {
            probeArray.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        *out = Handle::into_raw(Mutex::new(Vec::<Probe>::new()), Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLProbeArray,
    ProbeArrayObject,
    iplProbeArrayRetain,
    iplProbeArrayRelease
);

/// Replaces the array's probes with probes generated in the box `params.transform` maps the unit cube onto (centroid
/// or uniform floor). Does nothing for an unknown generation type (upstream throws through C).
///
/// # Safety
///
/// `probeArray` is null or a live probe array, `scene` null or a live scene, and `params` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeArrayGenerateProbes(
    probeArray: IPLProbeArray,
    scene: IPLScene,
    params: *mut IPLProbeGenerationParams,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(array), Some(scene), Some(params)) = (
            unsafe { object::<ProbeArrayObject, _>(probeArray) },
            unsafe { scene_arc(scene) },
            unsafe { params.as_ref() },
        ) else {
            return;
        };
        let generation = match params.r#type {
            IPL_PROBEGENERATIONTYPE_CENTROID => ProbeGeneration::Centroid,
            IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR => ProbeGeneration::UniformFloor {
                spacing: params.spacing,
                height: params.height,
            },
            _ => return,
        };
        let transform = from_ipl_matrix(&params.transform);
        let mut probes = lock(array);
        scene
            .with_ray_tracer(|tracer| generate_probes(tracer, &transform, generation, &mut probes));
    })
}

/// Number of probes (0 for a null handle).
///
/// # Safety
///
/// `probeArray` is null or a live probe array.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeArrayGetNumProbes(probeArray: IPLProbeArray) -> IPLint32 {
    // SAFETY: guaranteed by the caller.
    unsafe { object::<ProbeArrayObject, _>(probeArray) }.map_or(0, |a| lock(a).len() as i32)
}

/// The probe at `index` (a zero sphere for a null handle or an index out of range).
///
/// # Safety
///
/// `probeArray` is null or a live probe array.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeArrayGetProbe(
    probeArray: IPLProbeArray,
    index: IPLint32,
) -> IPLSphere {
    // SAFETY: guaranteed by the caller.
    let Some(array) = (unsafe { object::<ProbeArrayObject, _>(probeArray) }) else {
        return IPLSphere::default();
    };
    let probes = lock(array);
    probe_index(index, probes.len())
        .map_or_else(IPLSphere::default, |i| to_ipl_sphere(probes[i].influence))
}

// --------------------------------------------------------------------------------------------------------------------
// Probe batches
// --------------------------------------------------------------------------------------------------------------------

/// Creates an empty probe batch.
///
/// # Safety
///
/// `context` is null or a live context, and `probeBatch` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchCreate(
    context: IPLContext,
    probeBatch: *mut IPLProbeBatch,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let (Some(context), Some(out)) = (unsafe { context_arc(context) }, unsafe {
            probeBatch.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        *out = Handle::into_raw(Mutex::new(Arc::new(ProbeBatch::new())), Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLProbeBatch,
    ProbeBatchObject,
    iplProbeBatchRetain,
    iplProbeBatchRelease
);

/// Loads a probe batch from a serialized object (not committed). Returns `IPL_STATUS_FAILURE` for data that is not a
/// serialized batch (upstream does not check it).
///
/// # Safety
///
/// `context` is null or a live context, `serializedObject` null or a live serialized object, and `probeBatch` null or
/// valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchLoad(
    context: IPLContext,
    serializedObject: IPLSerializedObject,
    probeBatch: *mut IPLProbeBatch,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let (Some(context), Some(out)) = (unsafe { context_arc(context) }, unsafe {
            probeBatch.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        match unsafe { with_serialized_bytes(serializedObject, ProbeBatch::deserialize) } {
            Some(Ok(batch)) => {
                *out = Handle::into_raw(Mutex::new(Arc::new(batch)), Some(context));
                IPL_STATUS_SUCCESS
            }
            _ => IPL_STATUS_FAILURE,
        }
    })
}

/// The batch behind a handle.
///
/// # Safety
///
/// `batch` is null or a live probe batch.
pub(crate) unsafe fn probe_batch_ref<'a>(batch: IPLProbeBatch) -> Option<&'a ProbeBatchObject> {
    // SAFETY: guaranteed by the caller.
    unsafe { object::<ProbeBatchObject, _>(batch) }.map(|arc| &**arc)
}

/// Runs `f` on a batch to change it (copying it first if a simulator holds it).
///
/// # Safety
///
/// `batch` is null or a live probe batch.
unsafe fn update(batch: IPLProbeBatch, f: impl FnOnce(&mut ProbeBatch)) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(batch) = unsafe { probe_batch_ref(batch) } {
            f(Arc::make_mut(&mut lock(batch)));
        }
    })
}

/// Saves a probe batch into a serialized object.
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch, and `serializedObject` null or a live serialized object.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchSave(
    probeBatch: IPLProbeBatch,
    serializedObject: IPLSerializedObject,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(batch) = unsafe { probe_batch_ref(probeBatch) } {
            let bytes = lock(batch).serialize();
            // SAFETY: guaranteed by the caller.
            unsafe { store_serialized_bytes(serializedObject, bytes) };
        }
    })
}

/// Number of probes (0 for a null handle).
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchGetNumProbes(probeBatch: IPLProbeBatch) -> IPLint32 {
    // SAFETY: guaranteed by the caller.
    unsafe { probe_batch_ref(probeBatch) }.map_or(0, |b| lock(b).num_probes() as i32)
}

/// Adds a probe at the end of the batch; it needs baking in every layer.
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchAddProbe(probeBatch: IPLProbeBatch, probe: IPLSphere) {
    // SAFETY: guaranteed by the caller.
    unsafe { update(probeBatch, |batch| batch.add_probe(from_ipl_sphere(probe))) };
}

/// Adds every probe of an array, in order, at the end of the batch.
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch, and `probeArray` null or a live probe array.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchAddProbeArray(
    probeBatch: IPLProbeBatch,
    probeArray: IPLProbeArray,
) {
    // SAFETY: guaranteed by the caller.
    let Some(array) = (unsafe { object::<ProbeArrayObject, _>(probeArray) }) else {
        return;
    };
    let probes = lock(array).clone();
    // SAFETY: guaranteed by the caller.
    unsafe { update(probeBatch, |batch| batch.add_probes(&probes)) };
}

/// Removes probe `index` and its baked data (nothing for an index out of range).
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchRemoveProbe(probeBatch: IPLProbeBatch, index: IPLint32) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        update(probeBatch, |batch| {
            if let Some(i) = probe_index(index, batch.num_probes()) {
                batch.remove_probe(i);
            }
        });
    }
}

/// Builds the batch's probe tree from its probes.
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchCommit(probeBatch: IPLProbeBatch) {
    // SAFETY: guaranteed by the caller.
    unsafe { update(probeBatch, ProbeBatch::commit) };
}

/// Deletes a layer of baked data (nothing for a null identifier or an unknown type or variation, which no layer can
/// have).
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch, and `identifier` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchRemoveData(
    probeBatch: IPLProbeBatch,
    identifier: *mut IPLBakedDataIdentifier,
) {
    // SAFETY: guaranteed by the caller.
    let Some(identifier) = unsafe { identifier.as_ref() }.and_then(from_ipl_identifier) else {
        return;
    };
    // SAFETY: guaranteed by the caller.
    let has_data =
        unsafe { probe_batch_ref(probeBatch) }.is_some_and(|b| lock(b).has_data(&identifier));
    if has_data {
        // SAFETY: guaranteed by the caller.
        unsafe { update(probeBatch, |batch| batch.remove_data(&identifier)) };
    }
}

/// Estimated size in bytes of a layer's data (upstream `serializedSize`; 0 if the batch has no such layer).
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch, and `identifier` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchGetDataSize(
    probeBatch: IPLProbeBatch,
    identifier: *mut IPLBakedDataIdentifier,
) -> IPLsize {
    guard(0, || {
        // SAFETY: guaranteed by the caller.
        let (Some(batch), Some(identifier)) = (
            unsafe { probe_batch_ref(probeBatch) },
            unsafe { identifier.as_ref() }.and_then(from_ipl_identifier),
        ) else {
            return 0;
        };
        lock(batch).data(&identifier).map_or(0, |data| {
            usize::try_from(data.serialized_size()).unwrap_or(usize::MAX)
        })
    })
}

/// Zeroes `energyField` and copies the energy field baked at a probe into it (the channels and bins both have). Does
/// nothing if the batch has no such reflections layer, the probe is out of range, or it has no energy field.
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch, `identifier` null or valid, and `energyField` null or a live energy
/// field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchGetEnergyField(
    probeBatch: IPLProbeBatch,
    identifier: *mut IPLBakedDataIdentifier,
    probeIndex: IPLint32,
    energyField: IPLEnergyField,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(batch), Some(identifier), Some(field)) = (
            unsafe { probe_batch_ref(probeBatch) },
            unsafe { identifier.as_ref() }.and_then(from_ipl_identifier),
            unsafe { energy_field_ref(energyField) },
        ) else {
            return;
        };
        let batch = lock(batch);
        let Some(BakedData::Reflections(data)) = batch.data(&identifier) else {
            return;
        };
        let Some(source) =
            probe_index(probeIndex, batch.num_probes()).and_then(|i| data.energy_field(i))
        else {
            return;
        };
        let mut field = lock(field);
        field.reset();
        field.copy_from(source);
    })
}

/// Copies the 3 reverb times baked at a probe into `reverbTimes`. Does nothing if the batch has no such reflections
/// layer with parametric data or the probe is out of range.
///
/// # Safety
///
/// `probeBatch` is null or a live probe batch, `identifier` null or valid, and `reverbTimes` null or valid for writes
/// of 3 floats.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplProbeBatchGetReverb(
    probeBatch: IPLProbeBatch,
    identifier: *mut IPLBakedDataIdentifier,
    probeIndex: IPLint32,
    reverbTimes: *mut IPLfloat32,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(batch), Some(identifier)) = (
            unsafe { probe_batch_ref(probeBatch) },
            unsafe { identifier.as_ref() }.and_then(from_ipl_identifier),
        ) else {
            return;
        };
        if reverbTimes.is_null() {
            return;
        }
        let batch = lock(batch);
        let Some(BakedData::Reflections(data)) = batch.data(&identifier) else {
            return;
        };
        if let Some(reverb) =
            probe_index(probeIndex, batch.num_probes()).and_then(|i| data.reverb(i))
        {
            // SAFETY: guaranteed by the caller.
            unsafe { std::ptr::copy_nonoverlapping(reverb.reverb_times.as_ptr(), reverbTimes, 3) };
        }
    })
}
