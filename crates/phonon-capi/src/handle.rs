//! API object handles (upstream `api::Handle` and the `C*` wrapper classes).
//!
//! A handle is a boxed [`Handle`] cast to the opaque pointer type of `phonon.h`. Like upstream, it counts the
//! references the caller holds; the last `release` frees it. The object itself sits in an `Arc`, so that objects that
//! use one another (the simulator and its sources, stage 8b) can share it past the caller's last reference. Every object
//! but the context also keeps its context alive, as upstream's handles hold a `shared_ptr<Context>`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::context::Context;

#[derive(Debug)]
pub(crate) struct Handle<T> {
    refs: AtomicUsize,
    context: Option<Arc<Context>>,
    value: Arc<T>,
}

impl<T> Handle<T> {
    /// Allocates a handle with one reference and returns it as the opaque pointer type `O`.
    pub(crate) fn into_raw<O>(value: T, context: Option<Arc<Context>>) -> *mut O {
        let handle = Box::new(Self {
            refs: AtomicUsize::new(1),
            context,
            value: Arc::new(value),
        });
        Box::into_raw(handle).cast()
    }

    /// The handle behind an opaque pointer, or `None` for null.
    ///
    /// # Safety
    ///
    /// `ptr` is null or a live handle of this type returned by [`into_raw`](Self::into_raw).
    pub(crate) unsafe fn from_raw<'a, O>(ptr: *mut O) -> Option<&'a Self> {
        // SAFETY: guaranteed by the caller.
        unsafe { ptr.cast::<Self>().as_ref() }
    }

    /// Adds a reference and returns the same pointer (null stays null).
    ///
    /// # Safety
    ///
    /// As [`from_raw`](Self::from_raw).
    pub(crate) unsafe fn retain<O>(ptr: *mut O) -> *mut O {
        // SAFETY: guaranteed by the caller.
        if let Some(handle) = unsafe { Self::from_raw(ptr) } {
            handle.refs.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    /// Drops a reference, frees the handle with the last one, and sets `*ptr` to null.
    ///
    /// # Safety
    ///
    /// `ptr` is null or points to a pointer that [`from_raw`](Self::from_raw) accepts, and the caller gives up that
    /// reference.
    pub(crate) unsafe fn release<O>(ptr: *mut *mut O) {
        // SAFETY: guaranteed by the caller.
        let Some(slot) = (unsafe { ptr.as_mut() }) else {
            return;
        };
        // SAFETY: guaranteed by the caller.
        if let Some(handle) = unsafe { Self::from_raw(*slot) } {
            if handle.refs.fetch_sub(1, Ordering::AcqRel) == 1 {
                // SAFETY: this was the last reference to a handle allocated by `into_raw`.
                drop(unsafe { Box::from_raw(slot.cast::<Self>()) });
            }
        }
        *slot = std::ptr::null_mut();
    }

    pub(crate) fn get(&self) -> &Arc<T> {
        &self.value
    }

    /// The context the object was created in (`None` for the context itself).
    pub(crate) fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }
}

/// Locks an object, ignoring poisoning: a panic is caught at the API boundary ([`crate::guard`]), and the object is
/// left as upstream would leave it after an exception (in whatever state it reached).
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Defines `ipl<Name>Retain` and `ipl<Name>Release` for a handle type holding `$inner`.
macro_rules! retain_release {
    ($handle:ty, $inner:ty, $retain:ident, $release:ident) => {
        /// Adds a reference to an object and returns it.
        ///
        /// # Safety
        ///
        /// The handle is null or a live object of this type.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $retain(handle: $handle) -> $handle {
            // SAFETY: guaranteed by the caller.
            unsafe { $crate::handle::Handle::<$inner>::retain(handle) }
        }

        /// Releases a reference to an object (freeing it with the last one) and sets the handle to null.
        ///
        /// # Safety
        ///
        /// `handle` is null or points to a handle that is null or a live object of this type.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $release(handle: *mut $handle) {
            // SAFETY: guaranteed by the caller.
            unsafe { $crate::handle::Handle::<$inner>::release(handle) }
        }
    };
}

pub(crate) use retain_release;
