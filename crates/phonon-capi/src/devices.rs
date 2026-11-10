//! Embree, OpenCL, Radeon Rays and TrueAudio Next devices (upstream `api_embree_device.cpp`, `api_opencl_device.cpp`,
//! `api_radeonrays_device.cpp`, `api_tan_device.cpp`): out of scope, so the functions behave as upstream built without
//! `IPL_USES_EMBREE`, `IPL_USES_OPENCL`, `IPL_USES_RADEONRAYS` and `IPL_USES_TRUEAUDIONEXT`. Creating a device fails,
//! a device list has no devices, and since no device can exist, retaining one returns null and releasing one only sets
//! the handle to null.
//!
//! A validating context still validates the creation arguments. Upstream's validated factories then construct an
//! empty device and report success; the port reports failure, as without validation (`DEVIATIONS.md`).

use std::ffi::c_void;

use crate::context::context_arc;
use crate::guard;
use crate::types::*;

/// Defines `ipl<Name>Retain` and `ipl<Name>Release` for a device handle that can never be live.
macro_rules! no_device_retain_release {
    ($handle:ty, $retain:ident, $release:ident) => {
        /// Returns null: devices of this type cannot be created, so there is nothing to retain.
        #[unsafe(no_mangle)]
        pub extern "system" fn $retain(_device: $handle) -> $handle {
            std::ptr::null_mut()
        }

        /// Sets the handle to null: devices of this type cannot be created, so there is nothing to release.
        ///
        /// # Safety
        ///
        /// `device` is null or valid for writes.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $release(device: *mut $handle) {
            // SAFETY: guaranteed by the caller.
            if let Some(device) = unsafe { device.as_mut() } {
                *device = std::ptr::null_mut();
            }
        }
    };
}

// --------------------------------------------------------------------------------------------------------------------
// Embree
// --------------------------------------------------------------------------------------------------------------------

/// Fails: Embree is not supported.
///
/// # Safety
///
/// `context` is null or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplEmbreeDeviceCreate(
    context: IPLContext,
    settings: *mut IPLEmbreeDeviceSettings,
    device: *mut IPLEmbreeDevice,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        if let Some(v) = context.validator("createEmbreeDevice") {
            v.pointer("settings", settings);
            v.pointer("device", device);
        }
        IPL_STATUS_FAILURE
    })
}

no_device_retain_release!(
    IPLEmbreeDevice,
    iplEmbreeDeviceRetain,
    iplEmbreeDeviceRelease
);

// --------------------------------------------------------------------------------------------------------------------
// OpenCL
// --------------------------------------------------------------------------------------------------------------------

/// Fails: OpenCL is not supported.
///
/// # Safety
///
/// `context` is null or a live context, and `settings` is null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplOpenCLDeviceListCreate(
    context: IPLContext,
    settings: *mut IPLOpenCLDeviceSettings,
    deviceList: *mut IPLOpenCLDeviceList,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        if let Some(v) = context.validator("createOpenCLDeviceList") {
            if v.pointer("settings", settings) {
                // SAFETY: not null; valid as guaranteed by the caller.
                let settings = unsafe { &*settings };
                v.range(
                    "IPLOpenCLDeviceType",
                    "settings->type",
                    settings.r#type,
                    IPL_OPENCLDEVICETYPE_ANY,
                    IPL_OPENCLDEVICETYPE_GPU,
                );
                v.int(
                    "settings->numCUsToReserve",
                    settings.numCUsToReserve,
                    settings.numCUsToReserve >= 0,
                );
                v.float_if(
                    "settings->fractionCUsForIRUpdate",
                    settings.fractionCUsForIRUpdate,
                    (0.0..=1.0).contains(&settings.fractionCUsForIRUpdate),
                );
                v.bool("settings->requiresTAN", settings.requiresTAN);
            }
            v.pointer("deviceList", deviceList);
        }
        IPL_STATUS_FAILURE
    })
}

no_device_retain_release!(
    IPLOpenCLDeviceList,
    iplOpenCLDeviceListRetain,
    iplOpenCLDeviceListRelease
);

/// 0: no device list can be created.
#[unsafe(no_mangle)]
pub extern "system" fn iplOpenCLDeviceListGetNumDevices(
    _deviceList: IPLOpenCLDeviceList,
) -> IPLint32 {
    0
}

/// Does nothing: no device list can be created.
#[unsafe(no_mangle)]
pub extern "system" fn iplOpenCLDeviceListGetDeviceDesc(
    _deviceList: IPLOpenCLDeviceList,
    _index: IPLint32,
    _deviceDesc: *mut IPLOpenCLDeviceDesc,
) {
}

/// Fails: OpenCL is not supported.
///
/// # Safety
///
/// `context` is null or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplOpenCLDeviceCreate(
    context: IPLContext,
    deviceList: IPLOpenCLDeviceList,
    index: IPLint32,
    device: *mut IPLOpenCLDevice,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        if let Some(v) = context.validator("createOpenCLDevice") {
            // Upstream compares the index with the list's device count, dereferencing a null list; a list has no
            // devices, so any index is out of range.
            if v.pointer("deviceList", deviceList) {
                v.int("index", index, false);
            }
            v.pointer("device", device);
        }
        IPL_STATUS_FAILURE
    })
}

/// Fails: OpenCL is not supported.
///
/// # Safety
///
/// `context` is null or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplOpenCLDeviceCreateFromExisting(
    context: IPLContext,
    convolutionQueue: *mut c_void,
    irUpdateQueue: *mut c_void,
    device: *mut IPLOpenCLDevice,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        if let Some(v) = context.validator("createOpenCLDeviceFromExisting") {
            v.pointer("convolutionQueue", convolutionQueue);
            v.pointer("irUpdateQueue", irUpdateQueue);
            v.pointer("device", device);
        }
        IPL_STATUS_FAILURE
    })
}

no_device_retain_release!(
    IPLOpenCLDevice,
    iplOpenCLDeviceRetain,
    iplOpenCLDeviceRelease
);

// --------------------------------------------------------------------------------------------------------------------
// Radeon Rays, TrueAudio Next
// --------------------------------------------------------------------------------------------------------------------

/// Fails: Radeon Rays is not supported (and no OpenCL device can exist).
#[unsafe(no_mangle)]
pub extern "system" fn iplRadeonRaysDeviceCreate(
    _openCLDevice: IPLOpenCLDevice,
    _settings: *mut IPLRadeonRaysDeviceSettings,
    _rrDevice: *mut IPLRadeonRaysDevice,
) -> IPLerror {
    IPL_STATUS_FAILURE
}

no_device_retain_release!(
    IPLRadeonRaysDevice,
    iplRadeonRaysDeviceRetain,
    iplRadeonRaysDeviceRelease
);

/// Fails: TrueAudio Next is not supported (and no OpenCL device can exist).
#[unsafe(no_mangle)]
pub extern "system" fn iplTrueAudioNextDeviceCreate(
    _openCLDevice: IPLOpenCLDevice,
    _settings: *mut IPLTrueAudioNextDeviceSettings,
    _tanDevice: *mut IPLTrueAudioNextDevice,
) -> IPLerror {
    IPL_STATUS_FAILURE
}

no_device_retain_release!(
    IPLTrueAudioNextDevice,
    iplTrueAudioNextDeviceRetain,
    iplTrueAudioNextDeviceRelease
);
