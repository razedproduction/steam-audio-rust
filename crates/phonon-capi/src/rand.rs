//! The `rand()` state of each thread (risk R2).
//!
//! Upstream's reverb effects pick their delay lengths with `rand()`, whose state MSVC's runtime keeps per thread,
//! starting at 1. The core takes that state as a [`CRand`]; the C API keeps one per thread, so that objects created in
//! turn on a thread get the delays upstream's get when created in turn on one thread.

use std::cell::RefCell;

use steam_audio::random::CRand;

thread_local! {
    static RAND: RefCell<CRand> = RefCell::new(CRand::default());
}

/// Runs `f` with this thread's `rand()` state.
pub(crate) fn with_rand<R>(f: impl FnOnce(&mut CRand) -> R) -> R {
    RAND.with(|rand| f(&mut rand.borrow_mut()))
}
