//! The context (upstream `api_context.cpp`, `context.cpp`, the context part of `api_validation_layer.cpp`) and
//! `iplCalculateRelativeDirection` (`api_geometry.cpp`).

use steam_audio::baking::ReflectionBaker;
use steam_audio::math::Vector3;
use steam_audio::pathing::PathBaker;

use crate::handle::{Handle, retain_release};
use crate::types::*;
use crate::validation::{Validator, vector3};
use crate::{from_ipl_vector, guard, to_ipl_vector};

/// What a context holds. Upstream keeps all of it in globals (`Context::sAPIVersion`, `sLog`, `sMemory`,
/// `sSIMDLevel`), set by the last context created; here every object reads the version of its own context.
#[derive(Debug)]
pub(crate) struct Context {
    /// The API version the caller was built against (`IPLContextSettings::version`).
    version: u32,
    /// Where warnings go: upstream's global `sLog` takes the callback of the last context created; here each context
    /// logs to its own.
    log_callback: IPLLogFunction,
    /// Whether the context and its objects validate their calls (`IPL_CONTEXTFLAGS_VALIDATION`, API 4.5 and later).
    validation: bool,
    /// Upstream's bakers are static functions with global cancel flags; here each context has its own.
    pub(crate) reflection_baker: ReflectionBaker,
    pub(crate) path_baker: PathBaker,
}

impl Context {
    /// Whether the caller's API version is at least `major.minor` (upstream `isCallerAPIVersionAtLeast`), which decides
    /// which fields of later API versions are read.
    pub(crate) fn is_caller_api_version_at_least(&self, major: u32, minor: u32) -> bool {
        let caller_major = (self.version >> 16) & 0xff;
        let caller_minor = (self.version >> 8) & 0xff;
        caller_major > major || (caller_major == major && caller_minor >= minor)
    }

    /// A validator for a call to upstream's method `method` on an object of this context, if the context validates.
    pub(crate) fn validator(&self, method: &'static str) -> Option<Validator> {
        self.validation
            .then(|| Validator::new(self.log_callback, method))
    }
}

/// Whether this library can serve a caller built against `version`: the same major version and a minor version no
/// newer than ours (upstream `isVersionCompatible`).
fn is_version_compatible(version: u32) -> bool {
    let major = (version >> 16) & 0xff;
    let minor = (version >> 8) & 0xff;
    major == STEAMAUDIO_VERSION_MAJOR && minor <= STEAMAUDIO_VERSION_MINOR
}

/// Creates a context.
///
/// The allocation callbacks are accepted but not used (Rust allocates through its global allocator), and the SIMD
/// level is ignored (`DEVIATIONS.md`). With `IPL_CONTEXTFLAGS_VALIDATION` (callers of API 4.5 or later), the settings
/// are validated and the context and every object created from it validate their calls (`validation.rs`).
///
/// # Safety
///
/// `settings` and `context` are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplContextCreate(
    settings: *mut IPLContextSettings,
    context: *mut IPLContext,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(context)) =
            (unsafe { settings.as_ref() }, unsafe { context.as_mut() })
        else {
            return IPL_STATUS_FAILURE;
        };
        if !is_version_compatible(settings.version) {
            return IPL_STATUS_FAILURE;
        }

        let caller = Context {
            version: settings.version,
            log_callback: settings.logCallback,
            validation: false,
            reflection_baker: ReflectionBaker::new(),
            path_baker: PathBaker::new(),
        };
        let validation = caller.is_caller_api_version_at_least(4, 5)
            && settings.flags & IPL_CONTEXTFLAGS_VALIDATION != 0;
        if validation {
            // Upstream logs these through the global log before the new context sets its callback, so they reach the
            // previous context's callback; here they reach the new one's.
            let v = Validator::new(settings.logCallback, "createContext");
            if settings.allocateCallback.is_some() {
                v.pointer(
                    "settings->freeCallback",
                    settings
                        .freeCallback
                        .map_or(std::ptr::null(), |f| f as *const ()),
                );
            }
            v.range(
                "IPLSIMDLevel",
                "settings->simdLevel",
                settings.simdLevel,
                IPL_SIMDLEVEL_SSE2,
                IPL_SIMDLEVEL_AVX512,
            );
        }

        let value = Context {
            validation,
            version: settings.version,
            log_callback: settings.logCallback,
            reflection_baker: ReflectionBaker::new(),
            path_baker: PathBaker::new(),
        };
        *context = Handle::into_raw(value, None);
        IPL_STATUS_SUCCESS
    })
}

retain_release!(IPLContext, Context, iplContextRetain, iplContextRelease);

/// The context object behind a handle.
///
/// # Safety
///
/// `context` is null or a live context.
pub(crate) unsafe fn context_arc(context: IPLContext) -> Option<std::sync::Arc<Context>> {
    // SAFETY: guaranteed by the caller.
    unsafe { Handle::<Context>::from_raw(context) }.map(|handle| handle.get().clone())
}

/// A validator for a call to upstream's method `method` on a context, if it is not null and validates.
///
/// # Safety
///
/// `context` is null or a live context.
pub(crate) unsafe fn context_validator(
    context: IPLContext,
    method: &'static str,
) -> Option<Validator> {
    // SAFETY: guaranteed by the caller.
    unsafe { Handle::<Context>::from_raw(context) }
        .and_then(|handle| handle.get().validator(method))
}

/// The direction from the listener to the source in the listener's coordinate system: x right, y up, -z ahead.
/// Returns +y (up) when the source is within 1e-5 of the listener, and a zero vector for a null context.
///
/// # Safety
///
/// `context` is null or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplCalculateRelativeDirection(
    context: IPLContext,
    sourcePosition: IPLVector3,
    listenerPosition: IPLVector3,
    listenerAhead: IPLVector3,
    listenerUp: IPLVector3,
) -> IPLVector3 {
    // SAFETY: guaranteed by the caller.
    let Some(context) = (unsafe { context_arc(context) }) else {
        return IPLVector3::default();
    };
    let v = context.validator("calculateRelativeDirection");
    if let Some(v) = v {
        vector3!(v, sourcePosition, "sourcePosition");
        vector3!(v, listenerPosition, "listenerPosition");
        vector3!(v, listenerAhead, "listenerAhead");
        vector3!(v, listenerUp, "listenerUp");
    }

    let result = relative_direction(sourcePosition, listenerPosition, listenerAhead, listenerUp);
    if let Some(v) = v {
        vector3!(v, result, "result");
    }
    result
}

fn relative_direction(
    sourcePosition: IPLVector3,
    listenerPosition: IPLVector3,
    listenerAhead: IPLVector3,
    listenerUp: IPLVector3,
) -> IPLVector3 {
    let source = from_ipl_vector(sourcePosition);
    let listener = from_ipl_vector(listenerPosition);
    let ahead = from_ipl_vector(listenerAhead);
    let up = from_ipl_vector(listenerUp);
    let right = ahead.cross(up).normalize_or_zero();

    let mut listener_to_source = source - listener;
    let mut direction = Vector3::Y_AXIS;

    if listener_to_source.length() > 1e-5 {
        listener_to_source = listener_to_source.normalize_or_zero();
        direction.x = listener_to_source.dot(right);
        direction.y = listener_to_source.dot(up);
        direction.z = -listener_to_source.dot(ahead);
    }

    to_ipl_vector(direction)
}
