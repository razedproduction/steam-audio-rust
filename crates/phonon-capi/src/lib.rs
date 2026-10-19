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
//! simulator, and the direct-sound model functions; scenes, probes, baking and simulation follow in 8b.

// The C ABI needs raw pointers and exported symbols; every unsafe block states why it is sound.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
// Names follow `phonon.h`.
#![allow(non_snake_case)]

mod audio_buffer;
mod buffer;
mod context;
mod effects;
mod handle;
mod hrtf;
mod models;
mod serialized_object;
pub mod types;

use std::panic::{AssertUnwindSafe, catch_unwind};

use steam_audio::math::{CoordinateSpace3, Vector3};

pub use audio_buffer::*;
pub use context::{
    iplCalculateRelativeDirection, iplContextCreate, iplContextRelease, iplContextRetain,
};
pub use effects::*;
pub use hrtf::*;
pub use models::*;
pub use serialized_object::*;
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
