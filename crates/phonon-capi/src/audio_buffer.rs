//! Audio buffer functions (upstream `api_audio_buffer.cpp`, `audio_buffer.cpp`).
//!
//! Upstream asserts matching shapes only in debug builds; here a call with mismatched or invalid buffers does nothing.

use std::alloc::{self, Layout};

use steam_audio::audio_buffer::{self, AmbisonicsType, convert_ambisonics_in_place};

use crate::buffer::{
    RawBuffer, with_input_output, with_input_slice, with_output, with_slice_output,
};
use crate::guard;
use crate::types::*;

/// Bytes before the channel pointer table: the size of the whole block, for `iplAudioBufferFree`.
const HEADER: usize = size_of::<*mut f32>();

/// Layout of a block of `size` bytes holding a header, a pointer table and samples.
fn block_layout(size: usize) -> Option<Layout> {
    Layout::from_size_align(size, align_of::<*mut f32>()).ok()
}

/// Allocates a buffer of `numChannels × numSamples` silent samples. Like upstream, it is one block: the channel pointer
/// table followed by the samples, channel after channel (so the channels are contiguous); here a header holding the
/// block size precedes the table.
///
/// # Safety
///
/// `context` is null or a live context, and `audioBuffer` is null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferAllocate(
    context: IPLContext,
    numChannels: IPLint32,
    numSamples: IPLint32,
    audioBuffer: *mut IPLAudioBuffer,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        if context.is_null() || numChannels <= 0 || numSamples <= 0 {
            return IPL_STATUS_FAILURE;
        }
        // SAFETY: guaranteed by the caller.
        let Some(buffer) = (unsafe { audioBuffer.as_mut() }) else {
            return IPL_STATUS_FAILURE;
        };

        let channels = numChannels as usize;
        let samples = numSamples as usize;
        let size = channels
            .checked_mul(samples)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .and_then(|n| n.checked_add(HEADER + channels * size_of::<*mut f32>()));
        let Some(layout) = size.and_then(block_layout) else {
            return IPL_STATUS_OUTOFMEMORY;
        };

        // SAFETY: the layout has a non-zero size.
        let block = unsafe { alloc::alloc_zeroed(layout) };
        if block.is_null() {
            return IPL_STATUS_OUTOFMEMORY;
        }

        // SAFETY: the block holds the header, `channels` pointers and `channels × samples` floats, each part aligned
        // (the pointer-sized header and table keep the samples 4-byte aligned).
        unsafe {
            block.cast::<usize>().write(layout.size());
            let table = block.add(HEADER).cast::<*mut f32>();
            let data = table.add(channels).cast::<f32>();
            for i in 0..channels {
                table.add(i).write(data.add(i * samples));
            }
            buffer.data = table;
        }
        buffer.numChannels = numChannels;
        buffer.numSamples = numSamples;
        IPL_STATUS_SUCCESS
    })
}

/// Frees a buffer allocated by [`iplAudioBufferAllocate`] and sets its `data` to null.
///
/// # Safety
///
/// `context` is null or a live context, and `audioBuffer` is null or a buffer from [`iplAudioBufferAllocate`] (or with
/// a null `data`), not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferFree(
    context: IPLContext,
    audioBuffer: *mut IPLAudioBuffer,
) {
    if context.is_null() {
        return;
    }
    // SAFETY: guaranteed by the caller.
    let Some(buffer) = (unsafe { audioBuffer.as_mut() }) else {
        return;
    };
    if !buffer.data.is_null() {
        // SAFETY: `data` follows the header of a block allocated by `iplAudioBufferAllocate` with this layout.
        unsafe {
            let block = buffer.data.cast::<u8>().sub(HEADER);
            let size = block.cast::<usize>().read();
            if let Some(layout) = block_layout(size) {
                alloc::dealloc(block, layout);
            }
        }
    }
    buffer.data = std::ptr::null_mut();
}

/// Writes the buffer's samples, interleaved, to `dst` (`numChannels × numSamples` floats).
///
/// # Safety
///
/// `context` is null or a live context, `src` is null or a valid buffer, and `dst` is null or valid for writes of
/// `numChannels × numSamples` floats.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferInterleave(
    context: IPLContext,
    src: *mut IPLAudioBuffer,
    dst: *mut IPLfloat32,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(src) = (unsafe { RawBuffer::new(src) }) else {
            return;
        };
        if context.is_null() || dst.is_null() {
            return;
        }
        // SAFETY: guaranteed by the caller.
        unsafe { with_input_slice(&src, dst, src.len(), |input, out| input.interleave(out)) }
    })
}

/// Reads interleaved samples from `src` (`numChannels × numSamples` floats) into the buffer.
///
/// # Safety
///
/// `context` is null or a live context, `src` is null or valid for reads of `numChannels × numSamples` floats, and
/// `dst` is null or a valid buffer.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferDeinterleave(
    context: IPLContext,
    src: *mut IPLfloat32,
    dst: *mut IPLAudioBuffer,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(dst) = (unsafe { RawBuffer::new(dst) }) else {
            return;
        };
        if context.is_null() || src.is_null() {
            return;
        }
        // SAFETY: guaranteed by the caller.
        unsafe { with_slice_output(src, dst.len(), &dst, |input, out| out.deinterleave(input)) }
    })
}

/// Adds `in` to `mix`. Both must have the same shape.
///
/// # Safety
///
/// `context` is null or a live context, and `in` and `mix` are null or valid buffers.
// The closure is needed: the function item is not general over the views' lifetimes.
#[allow(clippy::redundant_closure)]
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferMix(
    context: IPLContext,
    in_: *mut IPLAudioBuffer,
    mix: *mut IPLAudioBuffer,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(mix)) = (unsafe { RawBuffer::new(in_) }, unsafe {
            RawBuffer::new(mix)
        }) else {
            return;
        };
        if context.is_null()
            || input.num_channels() != mix.num_channels()
            || input.num_samples() != mix.num_samples()
        {
            return;
        }
        // SAFETY: guaranteed by the caller.
        unsafe { with_input_output(&input, &mix, |input, out| audio_buffer::mix(input, out)) }
    })
}

/// Averages the channels of `in` into the mono buffer `out`, which must have as many samples.
///
/// # Safety
///
/// `context` is null or a live context, and `in` and `out` are null or valid buffers.
#[allow(clippy::redundant_closure)]
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferDownmix(
    context: IPLContext,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(output)) = (unsafe { RawBuffer::new(in_) }, unsafe {
            RawBuffer::new(out)
        }) else {
            return;
        };
        if context.is_null()
            || output.num_channels() != 1
            || input.num_samples() != output.num_samples()
        {
            return;
        }
        // SAFETY: guaranteed by the caller.
        unsafe {
            with_input_output(&input, &output, |input, out| {
                audio_buffer::downmix(input, out)
            })
        }
    })
}

fn ambisonics_type(value: IPLAmbisonicsType) -> Option<AmbisonicsType> {
    match value {
        IPL_AMBISONICSTYPE_N3D => Some(AmbisonicsType::N3d),
        IPL_AMBISONICSTYPE_SN3D => Some(AmbisonicsType::Sn3d),
        IPL_AMBISONICSTYPE_FUMA => Some(AmbisonicsType::FuMa),
        _ => None,
    }
}

/// Converts `in` from one Ambisonics format to another into `out`, which may be the same buffer. Conversions to or
/// from FuMa support up to 16 channels (order 3).
///
/// # Safety
///
/// `context` is null or a live context, and `in` and `out` are null or valid buffers.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAudioBufferConvertAmbisonics(
    context: IPLContext,
    inType: IPLAmbisonicsType,
    outType: IPLAmbisonicsType,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some(input), Some(output)) = (unsafe { RawBuffer::new(in_) }, unsafe {
            RawBuffer::new(out)
        }) else {
            return;
        };
        let (Some(in_type), Some(out_type)) = (ambisonics_type(inType), ambisonics_type(outType))
        else {
            return;
        };
        let fuma = in_type == AmbisonicsType::FuMa || out_type == AmbisonicsType::FuMa;
        if context.is_null()
            || input.num_channels() != output.num_channels()
            || input.num_samples() != output.num_samples()
            || (fuma && input.num_channels() > 16)
        {
            return;
        }

        if input.is_same(&output) {
            // SAFETY: guaranteed by the caller.
            unsafe {
                with_output(&output, |buffer| {
                    convert_ambisonics_in_place(in_type, out_type, buffer)
                })
            }
        } else {
            // SAFETY: guaranteed by the caller.
            unsafe {
                with_input_output(&input, &output, |input, out| {
                    audio_buffer::convert_ambisonics(in_type, out_type, input, out)
                })
            }
        }
    })
}
