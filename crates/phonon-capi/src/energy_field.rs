//! Energy fields (upstream `api_energy_field.cpp`).
//!
//! The data pointers these functions return stay valid until the object is released (the arrays never move, also not
//! when [`iplEnergyFieldSwap`] exchanges two fields' arrays); as upstream, the caller must not write through them while
//! another call uses the object.

use std::sync::Mutex;

use steam_audio::bands::NUM_BANDS;
use steam_audio::simulation::EnergyField;

use crate::context::context_arc;
use crate::guard;
use crate::handle::{Handle, lock, lock_pair, object, retain_release};
use crate::types::*;

pub(crate) type EnergyFieldObject = Mutex<EnergyField>;

/// The number of elements of a `a × b × c` array, if it can be allocated (its size in bytes fits an `isize`).
pub(crate) fn array_len(a: usize, b: usize, c: usize) -> Option<usize> {
    let len = a.checked_mul(b)?.checked_mul(c)?;
    (len.checked_mul(size_of::<f32>())? <= isize::MAX as usize).then_some(len)
}

/// A duration and an Ambisonics order from the API as `(seconds, order)`: the duration finite and not negative, the
/// order not negative. Upstream turns other values into negative array sizes (a crash).
pub(crate) fn duration_and_order(duration: f32, order: IPLint32) -> Option<(f32, usize)> {
    let order = usize::try_from(order).ok()?;
    (duration.is_finite() && duration >= 0.0).then_some((duration, order))
}

/// Runs `f` on the inputs and the output of an operation, locking each object once, in address order (so that two
/// threads cannot deadlock). An input that is the output is passed as a copy of the output taken before `f` runs,
/// since `phonon.h` allows in-place operations.
pub(crate) fn combine<T: Clone, const N: usize>(
    inputs: [&Mutex<T>; N],
    out: &Mutex<T>,
    f: impl FnOnce([&T; N], &mut T),
) {
    let mut objects: Vec<&Mutex<T>> = inputs.iter().copied().chain([out]).collect();
    objects.sort_by_key(|&m| m as *const Mutex<T>);
    objects.dedup_by_key(|m| *m as *const Mutex<T>);

    let mut guards: Vec<_> = objects.iter().map(|&m| lock(m)).collect();
    let position = |m: &Mutex<T>| {
        objects
            .iter()
            .position(|&o| std::ptr::eq(o, m))
            .expect("every object is locked")
    };
    let out_index = position(out);

    let out_copy = inputs
        .iter()
        .any(|&m| std::ptr::eq(m, out))
        .then(|| (*guards[out_index]).clone());

    let (left, rest) = guards.split_at_mut(out_index);
    let (out_guard, right) = rest.split_first_mut().expect("the output is locked");
    let refs = inputs.map(|m| {
        let i = position(m);
        if i == out_index {
            out_copy.as_ref().expect("copied above")
        } else if i < out_index {
            &*left[i]
        } else {
            &*right[i - out_index - 1]
        }
    });
    f(refs, out_guard);
}

/// Creates a silent energy field of `ceil(duration / 0.01)` bins for an Ambisonics order.
///
/// Returns `IPL_STATUS_FAILURE` for a negative order or a duration that is negative or not finite, and
/// `IPL_STATUS_OUTOFMEMORY` for a field too large to allocate.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldCreate(
    context: IPLContext,
    settings: *mut IPLEnergyFieldSettings,
    energyField: *mut IPLEnergyField,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe {
            energyField.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        let Some((duration, order)) = duration_and_order(settings.duration, settings.order) else {
            return IPL_STATUS_FAILURE;
        };

        let num_bins = f64::from(duration / EnergyField::BIN_DURATION).ceil();
        let num_channels = (order as u64 + 1).saturating_mul(order as u64 + 1);
        let fits = num_bins <= f64::from(i32::MAX)
            && num_channels <= i32::MAX as u64
            && array_len(num_channels as usize, NUM_BANDS, num_bins as usize).is_some();
        if !fits {
            return IPL_STATUS_OUTOFMEMORY;
        }

        let field = EnergyField::new(duration, order);
        *out = Handle::into_raw(Mutex::new(field), Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLEnergyField,
    EnergyFieldObject,
    iplEnergyFieldRetain,
    iplEnergyFieldRelease
);

/// The energy field behind a handle.
///
/// # Safety
///
/// `field` is null or a live energy field.
pub(crate) unsafe fn energy_field_ref<'a>(field: IPLEnergyField) -> Option<&'a EnergyFieldObject> {
    // SAFETY: guaranteed by the caller.
    unsafe { object::<EnergyFieldObject, _>(field) }.map(|arc| &**arc)
}

/// Number of channels (0 for a null handle).
///
/// # Safety
///
/// `energyField` is null or a live energy field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldGetNumChannels(
    energyField: IPLEnergyField,
) -> IPLint32 {
    // SAFETY: guaranteed by the caller.
    unsafe { energy_field_ref(energyField) }.map_or(0, |f| lock(f).num_channels() as i32)
}

/// Number of bins (0 for a null handle).
///
/// # Safety
///
/// `energyField` is null or a live energy field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldGetNumBins(energyField: IPLEnergyField) -> IPLint32 {
    // SAFETY: guaranteed by the caller.
    unsafe { energy_field_ref(energyField) }.map_or(0, |f| lock(f).num_bins() as i32)
}

/// The `channels × bands × bins` energies, or the energies from `offset` on; null for a null handle, an empty field
/// or an offset past the end.
///
/// # Safety
///
/// `field` is null or a live energy field.
unsafe fn data_at(
    field: IPLEnergyField,
    offset: impl FnOnce(&EnergyField) -> Option<usize>,
) -> *mut f32 {
    // SAFETY: guaranteed by the caller.
    let Some(field) = (unsafe { energy_field_ref(field) }) else {
        return std::ptr::null_mut();
    };
    let mut field = lock(field);
    match offset(&field) {
        Some(offset) if offset < field.data().len() => field.data_mut()[offset..].as_mut_ptr(),
        _ => std::ptr::null_mut(),
    }
}

/// All energies, channel-major, then band, then bin (null for a null handle or an empty field).
///
/// # Safety
///
/// `energyField` is null or a live energy field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldGetData(
    energyField: IPLEnergyField,
) -> *mut IPLfloat32 {
    // SAFETY: guaranteed by the caller.
    unsafe { data_at(energyField, |_| Some(0)) }
}

/// The energies of one channel, band after band (null for a null handle or a channel out of range).
///
/// # Safety
///
/// `energyField` is null or a live energy field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldGetChannel(
    energyField: IPLEnergyField,
    channelIndex: IPLint32,
) -> *mut IPLfloat32 {
    // SAFETY: guaranteed by the caller.
    unsafe {
        data_at(energyField, |f| {
            let channel = usize::try_from(channelIndex)
                .ok()
                .filter(|&c| c < f.num_channels())?;
            Some(channel * NUM_BANDS * f.num_bins())
        })
    }
}

/// The bins of one band of one channel (null for a null handle or a channel or band out of range).
///
/// # Safety
///
/// `energyField` is null or a live energy field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldGetBand(
    energyField: IPLEnergyField,
    channelIndex: IPLint32,
    bandIndex: IPLint32,
) -> *mut IPLfloat32 {
    // SAFETY: guaranteed by the caller.
    unsafe {
        data_at(energyField, |f| {
            let channel = usize::try_from(channelIndex)
                .ok()
                .filter(|&c| c < f.num_channels())?;
            let band = usize::try_from(bandIndex).ok().filter(|&b| b < NUM_BANDS)?;
            Some((channel * NUM_BANDS + band) * f.num_bins())
        })
    }
}

/// Zeroes every energy.
///
/// # Safety
///
/// `energyField` is null or a live energy field.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldReset(energyField: IPLEnergyField) {
    // SAFETY: guaranteed by the caller.
    if let Some(field) = unsafe { energy_field_ref(energyField) } {
        lock(field).reset();
    }
}

/// Copies the channels and bins both fields have from `src` into `dst`.
///
/// # Safety
///
/// Both handles are null or live energy fields.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldCopy(src: IPLEnergyField, dst: IPLEnergyField) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(src), Some(dst)) = (unsafe { energy_field_ref(src) }, unsafe {
            energy_field_ref(dst)
        }) else {
            return;
        };
        combine([src], dst, |[src], dst| dst.copy_from(src));
    })
}

/// Exchanges the contents (sizes and arrays) of two fields.
///
/// # Safety
///
/// Both handles are null or live energy fields.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldSwap(a: IPLEnergyField, b: IPLEnergyField) {
    // SAFETY: guaranteed by the caller.
    let (Some(a), Some(b)) = (unsafe { energy_field_ref(a) }, unsafe {
        energy_field_ref(b)
    }) else {
        return;
    };
    if let (mut a, Some(mut b)) = lock_pair(a, b) {
        std::mem::swap(&mut *a, &mut *b);
    }
}

/// `out = in1 + in2` over the channels and bins all three have; `out` may be either input.
///
/// # Safety
///
/// The handles are null or live energy fields.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldAdd(
    in1: IPLEnergyField,
    in2: IPLEnergyField,
    out: IPLEnergyField,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(in1), Some(in2), Some(out)) = (
            unsafe { energy_field_ref(in1) },
            unsafe { energy_field_ref(in2) },
            unsafe { energy_field_ref(out) },
        ) else {
            return;
        };
        combine([in1, in2], out, |[in1, in2], out| {
            EnergyField::add(in1, in2, out)
        });
    })
}

/// `out = in × scalar` over the channels and bins both have; `out` may be `in`.
///
/// # Safety
///
/// The handles are null or live energy fields.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldScale(
    r#in: IPLEnergyField,
    scalar: IPLfloat32,
    out: IPLEnergyField,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(out)) = (unsafe { energy_field_ref(r#in) }, unsafe {
            energy_field_ref(out)
        }) else {
            return;
        };
        combine([input], out, |[input], out| {
            EnergyField::scale(input, scalar, out)
        });
    })
}

/// `out += in × scalar` over the channels and bins both have; `out` may be `in`.
///
/// # Safety
///
/// The handles are null or live energy fields.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEnergyFieldScaleAccum(
    r#in: IPLEnergyField,
    scalar: IPLfloat32,
    out: IPLEnergyField,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(out)) = (unsafe { energy_field_ref(r#in) }, unsafe {
            energy_field_ref(out)
        }) else {
            return;
        };
        combine([input], out, |[input], out| {
            EnergyField::scale_accumulate(input, scalar, out);
        });
    })
}
