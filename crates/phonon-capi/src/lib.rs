//! The C API of the Steam Audio port (upstream `phonon.h`), built as the `phonon` library so that it can replace
//! upstream's without changes to calling code.
//!
//! Each function follows the matching `api_*.cpp` file and `phonon_interfaces.h`: the same null-handle checks, the
//! same status codes, and the same API-version-dependent fields. Where upstream would read out of bounds or assert on
//! invalid input (a buffer with the wrong channel count, a missing HRTF), the port checks first and returns
//! `IPL_STATUS_FAILURE` or `IPL_AUDIOEFFECTSTATE_TAILCOMPLETE` without touching the output; as a last line of defence,
//! every function catches panics, which must not unwind into C. `DEVIATIONS.md` lists the differences.
//!
//! Stage 8a covers the context, serialized objects, audio buffers, HRTFs, the audio effects that do not need a
//! simulator, and the direct-sound model functions; 8b scenes, energy fields, impulse responses, reconstructors, probes
//! and baking; 8c the simulator and its sources, and the reflection and path effects.

// The C ABI needs raw pointers and exported symbols; every unsafe block states why it is sound.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
// Names follow `phonon.h`.
#![allow(non_snake_case)]

mod audio_buffer;
mod baking;
mod buffer;
mod context;
mod effects;
mod energy_field;
mod handle;
mod hrtf;
mod impulse_response;
mod indirect_effect;
mod models;
mod probes;
mod rand;
mod reconstructor;
mod scene;
mod serialized_object;
mod simulator;
pub mod types;

use std::ffi::CStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use steam_audio::geometry::Sphere;
use steam_audio::math::{CoordinateSpace3, Matrix4, Vector3};

pub use audio_buffer::*;
pub use baking::*;
pub use context::{
    iplCalculateRelativeDirection, iplContextCreate, iplContextRelease, iplContextRetain,
};
pub use effects::*;
pub use energy_field::*;
pub use hrtf::*;
pub use impulse_response::*;
pub use indirect_effect::*;
pub use models::*;
pub use probes::*;
pub use reconstructor::*;
pub use scene::*;
pub use serialized_object::*;
pub use simulator::*;
pub use types::*;

/// Runs `f`, returning `default` if it panics, so that no panic unwinds into C.
fn guard<R>(default: R, f: impl FnOnce() -> R) -> R {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(default)
}

fn from_ipl_vector(v: IPLVector3) -> Vector3 {
    Vector3::new(v.x, v.y, v.z)
}

fn to_ipl_vector(v: Vector3) -> IPLVector3 {
    IPLVector3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn from_ipl_coordinate_space(c: &IPLCoordinateSpace3) -> CoordinateSpace3 {
    CoordinateSpace3 {
        right: from_ipl_vector(c.right),
        up: from_ipl_vector(c.up),
        ahead: from_ipl_vector(c.ahead),
        origin: from_ipl_vector(c.origin),
    }
}

/// An `IPLMatrix4x4` (row-major) as a matrix. Upstream reinterprets the elements as its own column-major
/// `Matrix4x4f` and then transposes it, which amounts to reading them row by row.
fn from_ipl_matrix(m: &IPLMatrix4x4) -> Matrix4 {
    Matrix4::from_rows(m.elements)
}

fn from_ipl_sphere(s: IPLSphere) -> Sphere {
    Sphere::new(from_ipl_vector(s.center), s.radius)
}

fn to_ipl_sphere(s: Sphere) -> IPLSphere {
    IPLSphere {
        center: to_ipl_vector(s.center),
        radius: s.radius,
    }
}

/// A file name from the API as a path: bytes on Unix, UTF-8 elsewhere (upstream passes the `char*` to `fopen`, or
/// converts it from UTF-8 to UTF-16 for `_wfopen` on Windows).
fn c_path(name: &CStr) -> Option<PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(std::ffi::OsStr::from_bytes(name.to_bytes()).into())
    }
    #[cfg(not(unix))]
    {
        name.to_str().ok().map(Into::into)
    }
}
