//! Impulse responses (upstream `api_impulse_response.cpp`).
//!
//! As for energy fields, the data pointers stay valid until the object is released.

use std::sync::Mutex;

use steam_audio::simulation::ImpulseResponse;

use crate::context::context_arc;
use crate::energy_field::{array_len, combine, duration_and_order};
use crate::guard;
use crate::handle::{Handle, lock, lock_pair, object, retain_release};
use crate::types::*;

pub(crate) type ImpulseResponseObject = Mutex<ImpulseResponse>;

/// Creates a silent impulse response of `ceil(duration × samplingRate)` samples for an Ambisonics order.
///
/// Returns `IPL_STATUS_FAILURE` for a negative order, a duration that is negative or not finite, or a sampling rate
/// that is not positive, and `IPL_STATUS_OUTOFMEMORY` for an IR too large to allocate.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseCreate(
    context: IPLContext,
    settings: *mut IPLImpulseResponseSettings,
    impulseResponse: *mut IPLImpulseResponse,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe {
            impulseResponse.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        let Some((duration, order)) = duration_and_order(settings.duration, settings.order) else {
            return IPL_STATUS_FAILURE;
        };
        let Some(rate) = u32::try_from(settings.samplingRate).ok().filter(|&r| r > 0) else {
            return IPL_STATUS_FAILURE;
        };

        let num_samples = f64::from(duration * rate as f32).ceil();
        let num_channels = (order as u64 + 1).saturating_mul(order as u64 + 1);
        let fits = num_samples <= f64::from(i32::MAX)
            && num_channels <= i32::MAX as u64
            && array_len(num_channels as usize, num_samples as usize, 1).is_some();
        if !fits {
            return IPL_STATUS_OUTOFMEMORY;
        }

        let ir = ImpulseResponse::new(duration, order, rate);
        *out = Handle::into_raw(Mutex::new(ir), Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLImpulseResponse,
    ImpulseResponseObject,
    iplImpulseResponseRetain,
    iplImpulseResponseRelease
);

/// The impulse response behind a handle.
///
/// # Safety
///
/// `ir` is null or a live impulse response.
pub(crate) unsafe fn impulse_response_ref<'a>(
    ir: IPLImpulseResponse,
) -> Option<&'a ImpulseResponseObject> {
    // SAFETY: guaranteed by the caller.
    unsafe { object::<ImpulseResponseObject, _>(ir) }.map(|arc| &**arc)
}

/// Number of channels (0 for a null handle).
///
/// # Safety
///
/// `impulseResponse` is null or a live impulse response.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseGetNumChannels(
    impulseResponse: IPLImpulseResponse,
) -> IPLint32 {
    // SAFETY: guaranteed by the caller.
    unsafe { impulse_response_ref(impulseResponse) }.map_or(0, |ir| lock(ir).num_channels() as i32)
}

/// Number of samples per channel (0 for a null handle).
///
/// # Safety
///
/// `impulseResponse` is null or a live impulse response.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseGetNumSamples(
    impulseResponse: IPLImpulseResponse,
) -> IPLint32 {
    // SAFETY: guaranteed by the caller.
    unsafe { impulse_response_ref(impulseResponse) }.map_or(0, |ir| lock(ir).num_samples() as i32)
}

/// The samples from channel `channel` on; null for a null handle, an empty IR or a channel out of range.
///
/// # Safety
///
/// `ir` is null or a live impulse response.
unsafe fn channel_data(ir: IPLImpulseResponse, channel: IPLint32) -> *mut f32 {
    // SAFETY: guaranteed by the caller.
    let Some(ir) = (unsafe { impulse_response_ref(ir) }) else {
        return std::ptr::null_mut();
    };
    let mut ir = lock(ir);
    let n = ir.num_samples();
    match usize::try_from(channel) {
        Ok(c) if c < ir.num_channels() && n > 0 => ir.data_mut()[c * n..].as_mut_ptr(),
        _ => std::ptr::null_mut(),
    }
}

/// All samples, channel after channel (null for a null handle or an empty IR).
///
/// # Safety
///
/// `impulseResponse` is null or a live impulse response.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseGetData(
    impulseResponse: IPLImpulseResponse,
) -> *mut IPLfloat32 {
    // SAFETY: guaranteed by the caller.
    unsafe { channel_data(impulseResponse, 0) }
}

/// The samples of one channel (null for a null handle or a channel out of range).
///
/// # Safety
///
/// `impulseResponse` is null or a live impulse response.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseGetChannel(
    impulseResponse: IPLImpulseResponse,
    channelIndex: IPLint32,
) -> *mut IPLfloat32 {
    // SAFETY: guaranteed by the caller.
    unsafe { channel_data(impulseResponse, channelIndex) }
}

/// Zeroes every sample.
///
/// # Safety
///
/// `impulseResponse` is null or a live impulse response.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseReset(impulseResponse: IPLImpulseResponse) {
    // SAFETY: guaranteed by the caller.
    if let Some(ir) = unsafe { impulse_response_ref(impulseResponse) } {
        lock(ir).reset();
    }
}

/// Copies the channels and samples both IRs have from `src` into `dst`.
///
/// # Safety
///
/// Both handles are null or live impulse responses.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseCopy(
    src: IPLImpulseResponse,
    dst: IPLImpulseResponse,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(src), Some(dst)) = (unsafe { impulse_response_ref(src) }, unsafe {
            impulse_response_ref(dst)
        }) else {
            return;
        };
        combine([src], dst, |[src], dst| dst.copy_from(src));
    })
}

/// Exchanges the contents (sizes and arrays) of two IRs.
///
/// # Safety
///
/// Both handles are null or live impulse responses.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseSwap(
    ir1: IPLImpulseResponse,
    ir2: IPLImpulseResponse,
) {
    // SAFETY: guaranteed by the caller.
    let (Some(a), Some(b)) = (unsafe { impulse_response_ref(ir1) }, unsafe {
        impulse_response_ref(ir2)
    }) else {
        return;
    };
    if let (mut a, Some(mut b)) = lock_pair(a, b) {
        std::mem::swap(&mut *a, &mut *b);
    }
}

/// `out = in1 + in2` over the channels and samples all three have; `out` may be either input.
///
/// # Safety
///
/// The handles are null or live impulse responses.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseAdd(
    in1: IPLImpulseResponse,
    in2: IPLImpulseResponse,
    out: IPLImpulseResponse,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(in1), Some(in2), Some(out)) = (
            unsafe { impulse_response_ref(in1) },
            unsafe { impulse_response_ref(in2) },
            unsafe { impulse_response_ref(out) },
        ) else {
            return;
        };
        combine([in1, in2], out, |[in1, in2], out| {
            ImpulseResponse::add(in1, in2, out)
        });
    })
}

/// `out = in × scalar` over the channels and samples both have; `out` may be `in`.
///
/// # Safety
///
/// The handles are null or live impulse responses.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseScale(
    r#in: IPLImpulseResponse,
    scalar: IPLfloat32,
    out: IPLImpulseResponse,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(out)) = (unsafe { impulse_response_ref(r#in) }, unsafe {
            impulse_response_ref(out)
        }) else {
            return;
        };
        combine([input], out, |[input], out| {
            ImpulseResponse::scale(input, scalar, out)
        });
    })
}

/// `out += in × scalar` over the channels and samples both have; `out` may be `in`.
///
/// # Safety
///
/// The handles are null or live impulse responses.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplImpulseResponseScaleAccum(
    r#in: IPLImpulseResponse,
    scalar: IPLfloat32,
    out: IPLImpulseResponse,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(out)) = (unsafe { impulse_response_ref(r#in) }, unsafe {
            impulse_response_ref(out)
        }) else {
            return;
        };
        combine([input], out, |[input], out| {
            ImpulseResponse::scale_accumulate(input, scalar, out);
        });
    })
}
