//! Serialized objects (upstream `api_serialized_object.cpp`, `serialized_object.cpp`).

use std::sync::Mutex;

use crate::context::context_arc;
use crate::guard;
use crate::handle::{Handle, lock, retain_release};
use crate::types::*;

/// The bytes of a serialized object (upstream `SerializedObject`).
#[derive(Debug)]
pub(crate) enum SerializedData {
    /// Nothing written yet: size 0 and a null data pointer.
    Empty,
    /// The caller's buffer, which upstream wraps without copying; the caller keeps it alive.
    External { data: *mut u8, size: usize },
    /// Written by a `save` function (stage 8b).
    #[allow(dead_code)]
    Owned(Vec<u8>),
}

// SAFETY: `External` only holds the address of the caller's buffer; the API contract makes the caller responsible for
// its lifetime and for not writing it while the library reads it, as upstream does.
unsafe impl Send for SerializedData {}

pub(crate) type SerializedObject = Mutex<SerializedData>;

/// Creates a serialized object: empty, or wrapping `settings.data` (not copied) when it is non-null and `size > 0`.
///
/// # Safety
///
/// `context` is null or a live context, and `settings` and `serializedObject` are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSerializedObjectCreate(
    context: IPLContext,
    settings: *mut IPLSerializedObjectSettings,
    serializedObject: *mut IPLSerializedObject,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe {
            serializedObject.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };

        let data = if !settings.data.is_null() && settings.size > 0 {
            SerializedData::External {
                data: settings.data,
                size: settings.size,
            }
        } else {
            SerializedData::Empty
        };
        *out = Handle::into_raw(Mutex::new(data), Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLSerializedObject,
    SerializedObject,
    iplSerializedObjectRetain,
    iplSerializedObjectRelease
);

/// Size of the serialized data in bytes (0 for a null handle).
///
/// # Safety
///
/// `serializedObject` is null or a live serialized object.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSerializedObjectGetSize(
    serializedObject: IPLSerializedObject,
) -> IPLsize {
    // SAFETY: guaranteed by the caller.
    let Some(handle) = (unsafe { Handle::<SerializedObject>::from_raw(serializedObject) }) else {
        return 0;
    };
    match &*lock(handle.get()) {
        SerializedData::Empty => 0,
        SerializedData::External { size, .. } => *size,
        SerializedData::Owned(bytes) => bytes.len(),
    }
}

/// The serialized data (null for a null handle or an object with nothing written).
///
/// # Safety
///
/// `serializedObject` is null or a live serialized object.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSerializedObjectGetData(
    serializedObject: IPLSerializedObject,
) -> *mut IPLbyte {
    // SAFETY: guaranteed by the caller.
    let Some(handle) = (unsafe { Handle::<SerializedObject>::from_raw(serializedObject) }) else {
        return std::ptr::null_mut();
    };
    match &mut *lock(handle.get()) {
        SerializedData::Empty => std::ptr::null_mut(),
        SerializedData::External { data, .. } => *data,
        SerializedData::Owned(bytes) => bytes.as_mut_ptr(),
    }
}
