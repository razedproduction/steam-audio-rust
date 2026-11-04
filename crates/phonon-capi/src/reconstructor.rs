//! Reconstructors (upstream `api_reconstructor.cpp`).

use std::sync::Mutex;

use steam_audio::air_absorption::AirAbsorptionModel;
use steam_audio::simulation::{ReconstructionParams, ReconstructionType, Reconstructor};

use crate::context::context_arc;
use crate::energy_field::{array_len, energy_field_ref};
use crate::guard;
use crate::handle::{Handle, lock, object, retain_release};
use crate::impulse_response::impulse_response_ref;
use crate::types::*;

pub(crate) struct ReconstructorState {
    reconstructor: Reconstructor,
    sampling_rate: u32,
    /// Samples of white noise the reconstructor holds: the longest IR it can reconstruct.
    max_samples: usize,
}

pub(crate) type ReconstructorObject = Mutex<ReconstructorState>;

/// Creates a reconstructor for IRs of up to `maxDuration` seconds at `samplingRate` (`maxOrder` is not used, as
/// upstream).
///
/// Returns `IPL_STATUS_FAILURE` for a duration that is negative or not finite or a sampling rate that is not positive,
/// and `IPL_STATUS_OUTOFMEMORY` for a duration too long to allocate.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReconstructorCreate(
    context: IPLContext,
    settings: *mut IPLReconstructorSettings,
    reconstructor: *mut IPLReconstructor,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe {
            reconstructor.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        let duration = settings.maxDuration;
        let Some(rate) = u32::try_from(settings.samplingRate).ok().filter(|&r| r > 0) else {
            return IPL_STATUS_FAILURE;
        };
        if !(duration.is_finite() && duration >= 0.0) {
            return IPL_STATUS_FAILURE;
        }
        let max_samples = f64::from(duration * rate as f32).ceil();
        // The noise and the three band IRs.
        if max_samples > f64::from(i32::MAX) || array_len(max_samples as usize, 4, 1).is_none() {
            return IPL_STATUS_OUTOFMEMORY;
        }

        let state = ReconstructorState {
            reconstructor: Reconstructor::new(duration, rate),
            sampling_rate: rate,
            max_samples: (duration * rate as f32).ceil() as usize,
        };
        *out = Handle::into_raw(Mutex::new(state), Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLReconstructor,
    ReconstructorObject,
    iplReconstructorRetain,
    iplReconstructorRelease
);

/// Reconstructs `numInputs` impulse responses from energy fields, by linear reconstruction, with the default air
/// absorption model and no distance attenuation correction (as upstream).
///
/// Does nothing if a pointer or handle is null (upstream checks the handles before reconstructing any IR), the order
/// is negative, or an IR would be longer than the reconstructor's maximum duration (upstream writes past its buffers).
///
/// # Safety
///
/// `reconstructor` is null or a live reconstructor, `inputs` and `outputs` are null or valid for `numInputs` elements
/// holding null or live handles, and `sharedInputs` is null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReconstructorReconstruct(
    reconstructor: IPLReconstructor,
    numInputs: IPLint32,
    inputs: *mut IPLReconstructorInputs,
    sharedInputs: *mut IPLReconstructorSharedInputs,
    outputs: *mut IPLReconstructorOutputs,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(reconstructor) = (unsafe { object::<ReconstructorObject, _>(reconstructor) })
        else {
            return;
        };
        // SAFETY: guaranteed by the caller.
        let Some(shared) = (unsafe { sharedInputs.as_ref() }) else {
            return;
        };
        if inputs.is_null() || outputs.is_null() {
            return;
        }
        let Ok(order) = usize::try_from(shared.order) else {
            return;
        };
        let n = usize::try_from(numInputs).unwrap_or(0);

        let mut pairs = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: guaranteed by the caller.
            let (input, output) = unsafe { (&*inputs.add(i), &*outputs.add(i)) };
            // SAFETY: guaranteed by the caller.
            let (Some(field), Some(ir)) =
                (unsafe { energy_field_ref(input.energyField) }, unsafe {
                    impulse_response_ref(output.impulseResponse)
                })
            else {
                return;
            };
            pairs.push((field, ir));
        }

        let mut state = lock(reconstructor);
        let requested = (shared.duration * state.sampling_rate as f32).ceil() as usize;
        let fits = pairs
            .iter()
            .all(|(_, ir)| lock(ir).num_samples().min(requested) <= state.max_samples);
        if !fits {
            return;
        }

        let air_absorption = AirAbsorptionModel::default();
        let params = ReconstructionParams {
            kind: ReconstructionType::Linear,
            duration: shared.duration,
            order,
            air_absorption: &air_absorption,
            distance_attenuation_correction: None,
        };
        for (field, ir) in pairs {
            let field = lock(field);
            let mut ir = lock(ir);
            state.reconstructor.reconstruct(&field, &mut ir, &params);
        }
    })
}
