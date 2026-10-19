//! Bridging `IPLAudioBuffer` (one pointer per channel) to the core's contiguous planar buffers (risk R7).
//!
//! When a buffer's channels lie back to back (`data[i + 1] == data[i] + numSamples`, always true for
//! `iplAudioBufferAllocate`), the core works on the caller's memory directly. Otherwise, and whenever an input overlaps
//! the output (so that a shared and a mutable slice of the same memory never coexist), the samples go through scratch
//! arrays kept per thread: they grow to the largest buffer a thread has seen and are reused, so after the first frame
//! nothing is allocated. Outputs that go through scratch are copied in first, because some effects leave channels
//! untouched (risk R15), and copied back afterwards.

use std::cell::RefCell;
use std::slice;

use steam_audio::audio_buffer::{AudioBuffer, AudioBufferMut, AudioBufferRef};

use crate::types::IPLAudioBuffer;

/// A validated `IPLAudioBuffer`: at least one channel and one sample, and a non-null pointer for every channel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawBuffer {
    num_channels: usize,
    num_samples: usize,
    data: *const *mut f32,
}

impl RawBuffer {
    /// Validates `buffer`; `None` if it is null, empty, or has a null channel pointer.
    ///
    /// # Safety
    ///
    /// `buffer` is null or points to an `IPLAudioBuffer` whose `data` (if not null) points to `numChannels` pointers.
    pub(crate) unsafe fn new(buffer: *const IPLAudioBuffer) -> Option<Self> {
        // SAFETY: guaranteed by the caller.
        let buffer = unsafe { buffer.as_ref() }?;
        let num_channels = usize::try_from(buffer.numChannels)
            .ok()
            .filter(|&n| n > 0)?;
        let num_samples = usize::try_from(buffer.numSamples).ok().filter(|&n| n > 0)?;
        if buffer.data.is_null() {
            return None;
        }
        let raw = Self {
            num_channels,
            num_samples,
            data: buffer.data.cast_const(),
        };
        (0..num_channels)
            .all(|i| !raw.channel(i).is_null())
            .then_some(raw)
    }

    pub(crate) fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub(crate) fn num_samples(&self) -> usize {
        self.num_samples
    }

    pub(crate) fn len(&self) -> usize {
        self.num_channels * self.num_samples
    }

    fn channel(&self, i: usize) -> *mut f32 {
        debug_assert!(i < self.num_channels);
        // SAFETY: `new` checked that `data` points to `num_channels` pointers.
        unsafe { *self.data.add(i) }
    }

    /// Whether the channels lie back to back, so the whole buffer is one planar array.
    pub(crate) fn is_contiguous(&self) -> bool {
        (1..self.num_channels)
            .all(|i| self.channel(i) == self.channel(i - 1).wrapping_add(self.num_samples))
    }

    /// Address ranges of the channels.
    fn ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.num_channels).map(|i| {
            let start = self.channel(i) as usize;
            (start, start + self.num_samples * size_of::<f32>())
        })
    }

    /// Whether any channel overlaps the address range `[start, end)`.
    pub(crate) fn overlaps_range(&self, start: usize, end: usize) -> bool {
        self.ranges().any(|(s, e)| s < end && start < e)
    }

    /// Whether any channel of `self` overlaps any channel of `other`.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        other.ranges().any(|(s, e)| self.overlaps_range(s, e))
    }

    /// Whether `other` has exactly the same shape and channel pointers.
    pub(crate) fn is_same(&self, other: &Self) -> bool {
        self.num_channels == other.num_channels
            && self.num_samples == other.num_samples
            && (0..self.num_channels).all(|i| self.channel(i) == other.channel(i))
    }

    /// The samples as one planar slice.
    ///
    /// # Safety
    ///
    /// The buffer is contiguous, its memory is valid for reads for `'a`, and nothing writes it meanwhile.
    unsafe fn as_slice<'a>(&self) -> &'a [f32] {
        debug_assert!(self.is_contiguous());
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts(self.channel(0), self.len()) }
    }

    /// The samples as one mutable planar slice.
    ///
    /// # Safety
    ///
    /// The buffer is contiguous, its memory is valid for reads and writes for `'a`, and nothing else accesses it
    /// meanwhile.
    unsafe fn as_mut_slice<'a>(&self) -> &'a mut [f32] {
        debug_assert!(self.is_contiguous());
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts_mut(self.channel(0), self.len()) }
    }

    /// Copies the samples, channel by channel, into `out` (resized to fit).
    ///
    /// # Safety
    ///
    /// Every channel is valid for reads of `num_samples` floats.
    unsafe fn copy_to(&self, out: &mut Vec<f32>) {
        out.clear();
        for i in 0..self.num_channels {
            // SAFETY: guaranteed by the caller.
            out.extend_from_slice(unsafe {
                slice::from_raw_parts(self.channel(i), self.num_samples)
            });
        }
    }

    /// Copies planar samples back into the channels.
    ///
    /// # Safety
    ///
    /// Every channel is valid for writes of `num_samples` floats, and `src` lies outside them.
    unsafe fn copy_from(&self, src: &[f32]) {
        for (i, chunk) in src.chunks_exact(self.num_samples).enumerate() {
            // SAFETY: guaranteed by the caller.
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), self.channel(i), self.num_samples)
            };
        }
    }
}

#[derive(Debug, Default)]
struct Scratch {
    input: Vec<f32>,
    output: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::default();
}

/// Runs `f` with this thread's scratch, or with fresh scratch if it is in use (a callback re-entering the API).
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    let mut f = Some(f);
    let result = SCRATCH.try_with(|cell| {
        cell.try_borrow_mut()
            .ok()
            .map(|mut scratch| (f.take().expect("called once"))(&mut scratch))
    });
    match result {
        Ok(Some(r)) => r,
        _ => (f.take().expect("called once"))(&mut Scratch::default()),
    }
}

/// A view of `input`: borrowed if `borrow` (the caller checked that the buffer is contiguous and overlaps nothing
/// written meanwhile), copied into `storage` otherwise.
///
/// # Safety
///
/// Every channel of `input` is valid for reads during `'a`; if `borrow`, as [`RawBuffer::as_slice`].
unsafe fn input_view<'a>(
    input: &RawBuffer,
    borrow: bool,
    storage: &'a mut Vec<f32>,
) -> AudioBufferRef<'a> {
    let data = if borrow {
        // SAFETY: guaranteed by the caller.
        unsafe { input.as_slice() }
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { input.copy_to(storage) };
        &storage[..]
    };
    AudioBuffer::from_planar(input.num_channels, input.num_samples, data)
}

/// Runs `f` on a writable view of `output`: borrowed if it is contiguous, else through `storage` (filled with the
/// current samples first and copied back afterwards).
///
/// # Safety
///
/// Every channel of `output` is valid for reads and writes, and nothing else accesses them during the call.
unsafe fn with_output_view<R>(
    output: &RawBuffer,
    storage: &mut Vec<f32>,
    f: impl FnOnce(&mut AudioBufferMut<'_>) -> R,
) -> R {
    if output.is_contiguous() {
        // SAFETY: guaranteed by the caller.
        let data = unsafe { output.as_mut_slice() };
        f(&mut AudioBuffer::from_planar(
            output.num_channels,
            output.num_samples,
            data,
        ))
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { output.copy_to(storage) };
        let result = f(&mut AudioBuffer::from_planar(
            output.num_channels,
            output.num_samples,
            &mut storage[..],
        ));
        // SAFETY: guaranteed by the caller; `storage` is ours.
        unsafe { output.copy_from(storage) };
        result
    }
}

/// Runs `f` with a view of `input` and a writable view of `output`. An input that overlaps the output is copied first.
///
/// # Safety
///
/// Every channel of `input` is valid for reads and every channel of `output` for reads and writes, and nothing else
/// accesses them during the call.
pub(crate) unsafe fn with_input_output<R>(
    input: &RawBuffer,
    output: &RawBuffer,
    f: impl FnOnce(&AudioBufferRef<'_>, &mut AudioBufferMut<'_>) -> R,
) -> R {
    with_scratch(|scratch| {
        let borrow = input.is_contiguous() && !input.overlaps(output);
        // SAFETY: guaranteed by the caller; a borrowed input does not overlap the output.
        let input = unsafe { input_view(input, borrow, &mut scratch.input) };
        // SAFETY: guaranteed by the caller.
        unsafe { with_output_view(output, &mut scratch.output, |out| f(&input, out)) }
    })
}

/// Runs `f` with a writable view of `output`.
///
/// # Safety
///
/// As [`with_input_output`] for `output`.
pub(crate) unsafe fn with_output<R>(
    output: &RawBuffer,
    f: impl FnOnce(&mut AudioBufferMut<'_>) -> R,
) -> R {
    // SAFETY: guaranteed by the caller.
    with_scratch(|scratch| unsafe { with_output_view(output, &mut scratch.output, f) })
}

/// Runs `f` with a view of `input` and the `len` floats at `out`. An input that overlaps them is copied first.
///
/// # Safety
///
/// Every channel of `input` is valid for reads, `out` is valid for reads and writes of `len` floats, and nothing else
/// accesses them during the call.
pub(crate) unsafe fn with_input_slice<R>(
    input: &RawBuffer,
    out: *mut f32,
    len: usize,
    f: impl FnOnce(&AudioBufferRef<'_>, &mut [f32]) -> R,
) -> R {
    with_scratch(|scratch| {
        let start = out as usize;
        let borrow =
            input.is_contiguous() && !input.overlaps_range(start, start + len * size_of::<f32>());
        // SAFETY: guaranteed by the caller; a borrowed input does not overlap `out`.
        let input = unsafe { input_view(input, borrow, &mut scratch.input) };
        // SAFETY: guaranteed by the caller.
        f(&input, unsafe { slice::from_raw_parts_mut(out, len) })
    })
}

/// Runs `f` with the `len` floats at `input` and a writable view of `output`. An input that overlaps the output is
/// copied first.
///
/// # Safety
///
/// `input` is valid for reads of `len` floats, every channel of `output` for reads and writes, and nothing else
/// accesses them during the call.
pub(crate) unsafe fn with_slice_output<R>(
    input: *const f32,
    len: usize,
    output: &RawBuffer,
    f: impl FnOnce(&[f32], &mut AudioBufferMut<'_>) -> R,
) -> R {
    with_scratch(|scratch| {
        let start = input as usize;
        let data = if output.overlaps_range(start, start + len * size_of::<f32>()) {
            scratch.input.clear();
            // SAFETY: guaranteed by the caller.
            scratch
                .input
                .extend_from_slice(unsafe { slice::from_raw_parts(input, len) });
            &scratch.input[..]
        } else {
            // SAFETY: guaranteed by the caller; the output does not overlap the input.
            unsafe { slice::from_raw_parts(input, len) }
        };
        // SAFETY: guaranteed by the caller.
        unsafe { with_output_view(output, &mut scratch.output, |out| f(data, out)) }
    })
}
