//! Audio effects (upstream `api_panning_effect.cpp`, `api_binaural_effect.cpp`, `api_virtual_surround_effect.cpp`,
//! `api_ambisonics_*_effect.cpp`, `api_direct_effect.cpp`).
//!
//! Every effect checks its buffers against what the core needs (channel counts, frame size) before processing; a call
//! that does not fit returns `IPL_AUDIOEFFECTSTATE_TAILCOMPLETE` and leaves the output alone, where upstream would
//! assert in a debug build and read or write out of bounds in a release build.

use std::sync::{Arc, Mutex};

use steam_audio::audio_buffer::{AudioBufferMut, AudioEffectState, AudioSettings};
use steam_audio::effects::{
    AmbisonicsBinauralEffect, AmbisonicsDecodeEffect, AmbisonicsDecodeParams,
    AmbisonicsEncodeEffect, AmbisonicsPanningEffect, AmbisonicsRotateEffect, BinauralEffect,
    BinauralParams, DirectEffect, DirectEffectFlags, DirectEffectParams, DirectSoundPath,
    PanningEffect, TransmissionType, VirtualSurroundEffect,
};
use steam_audio::hrtf::{HrtfInterpolation, HrtfPhaseType, MAX_AMBISONICS_ORDER, NUM_EARS};
use steam_audio::sh::num_coeffs_for_order;
use steam_audio::speaker_layout::{SpeakerLayout, SpeakerLayoutType};

use crate::buffer::{RawBuffer, with_input_output, with_output};
use crate::context::{Context, context_arc};
use crate::handle::{Handle, lock, retain_release};
use crate::hrtf::hrtf_ref;
use crate::types::*;
use crate::{from_ipl_coordinate_space, from_ipl_vector, guard};

// --------------------------------------------------------------------------------------------------------------------
// Shared plumbing
// --------------------------------------------------------------------------------------------------------------------

/// What every effect handle supports besides creation and `apply`.
trait ApiEffect {
    fn reset(&mut self);
    fn tail_size(&self) -> usize;
    /// Whether [`tail`](Self::tail) can write `out`.
    fn tail_fits(&self, out: &RawBuffer) -> bool;
    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState;
}

fn effect_state(state: AudioEffectState) -> IPLAudioEffectState {
    match state {
        AudioEffectState::TailRemaining => IPL_AUDIOEFFECTSTATE_TAILREMAINING,
        AudioEffectState::TailComplete => IPL_AUDIOEFFECTSTATE_TAILCOMPLETE,
    }
}

/// Whether `buffer` has `channels` channels of `samples` samples.
fn has_shape(buffer: &RawBuffer, channels: usize, samples: usize) -> bool {
    buffer.num_channels() == channels && buffer.num_samples() == samples
}

/// An Ambisonics order from the API (negative orders are invalid).
fn order(value: IPLint32) -> Option<usize> {
    usize::try_from(value).ok()
}

/// A speaker layout from the API: a standard type, or a custom one with at least one speaker.
///
/// # Safety
///
/// For a custom layout, `speakers` is null or valid for reads of `numSpeakers` vectors.
unsafe fn speaker_layout(layout: &IPLSpeakerLayout) -> Option<SpeakerLayout> {
    let ty = match layout.r#type {
        IPL_SPEAKERLAYOUTTYPE_MONO => SpeakerLayoutType::Mono,
        IPL_SPEAKERLAYOUTTYPE_STEREO => SpeakerLayoutType::Stereo,
        IPL_SPEAKERLAYOUTTYPE_QUADRAPHONIC => SpeakerLayoutType::Quadraphonic,
        IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1 => SpeakerLayoutType::FivePointOne,
        IPL_SPEAKERLAYOUTTYPE_SURROUND_7_1 => SpeakerLayoutType::SevenPointOne,
        IPL_SPEAKERLAYOUTTYPE_CUSTOM => {
            let n = usize::try_from(layout.numSpeakers)
                .ok()
                .filter(|&n| n > 0)?;
            if layout.speakers.is_null() {
                return None;
            }
            // SAFETY: guaranteed by the caller.
            let speakers = unsafe { std::slice::from_raw_parts(layout.speakers, n) };
            return Some(SpeakerLayout::custom(
                speakers.iter().copied().map(from_ipl_vector).collect(),
            ));
        }
        _ => return None,
    };
    Some(SpeakerLayout::new(ty))
}

/// Creates an effect: checks the context and settings like upstream's `CContext::create*Effect`, then lets `build`
/// make the effect (returning an error status if the settings are invalid).
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
unsafe fn create<O, S, T>(
    context: IPLContext,
    audio_settings: *mut IPLAudioSettings,
    effect_settings: *mut S,
    effect: *mut *mut O,
    build: impl FnOnce(AudioSettings, &S, &Arc<Context>) -> Result<T, IPLerror>,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(audio), Some(settings), Some(out)) = (
            unsafe { audio_settings.as_ref() },
            unsafe { effect_settings.as_ref() },
            unsafe { effect.as_mut() },
        ) else {
            return IPL_STATUS_FAILURE;
        };
        if audio.samplingRate <= 0 || audio.frameSize <= 0 {
            return IPL_STATUS_FAILURE;
        }

        let audio = AudioSettings::new(audio.samplingRate as u32, audio.frameSize as usize);
        match build(audio, settings, &context) {
            Ok(value) => {
                *out = Handle::into_raw(Mutex::new(value), Some(context));
                IPL_STATUS_SUCCESS
            }
            Err(status) => status,
        }
    })
}

/// The effect behind a handle and its context.
///
/// # Safety
///
/// `effect` is null or a live effect holding a `Mutex<T>`.
unsafe fn effect_ref<'a, O, T>(effect: *mut O) -> Option<(&'a Mutex<T>, &'a Arc<Context>)> {
    // SAFETY: guaranteed by the caller.
    let handle = unsafe { Handle::<Mutex<T>>::from_raw(effect) }?;
    Some((&**handle.get(), handle.context()?))
}

/// Applies an effect: `f` receives the effect, the parameters, the validated buffers and the context, and returns
/// `None` if the call does not fit.
///
/// # Safety
///
/// `effect` is null or a live effect holding a `Mutex<T>`, and the other pointers are null or valid.
unsafe fn apply<O, T, P>(
    effect: *mut O,
    params: *mut P,
    input: *mut IPLAudioBuffer,
    output: *mut IPLAudioBuffer,
    f: impl FnOnce(&mut T, &P, &RawBuffer, &RawBuffer, &Context) -> Option<AudioEffectState>,
) -> IPLAudioEffectState {
    guard(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, || {
        // SAFETY: guaranteed by the caller.
        let Some((effect, context)) = (unsafe { effect_ref::<O, T>(effect) }) else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(params), Some(input), Some(output)) = (
            unsafe { params.as_ref() },
            unsafe { RawBuffer::new(input) },
            unsafe { RawBuffer::new(output) },
        ) else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        f(&mut lock(effect), params, &input, &output, context)
            .map_or(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, effect_state)
    })
}

/// `ipl*EffectReset`.
///
/// # Safety
///
/// `effect` is null or a live effect holding a `Mutex<T>`.
unsafe fn reset<O, T: ApiEffect>(effect: *mut O) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some((effect, _)) = unsafe { effect_ref::<O, T>(effect) } {
            lock(effect).reset();
        }
    })
}

/// `ipl*EffectGetTailSize`.
///
/// # Safety
///
/// `effect` is null or a live effect holding a `Mutex<T>`.
unsafe fn tail_size<O, T: ApiEffect>(effect: *mut O) -> IPLint32 {
    guard(0, || {
        // SAFETY: guaranteed by the caller.
        unsafe { effect_ref::<O, T>(effect) }.map_or(0, |(effect, _)| {
            i32::try_from(lock(effect).tail_size()).unwrap_or(i32::MAX)
        })
    })
}

/// `ipl*EffectGetTail`.
///
/// # Safety
///
/// `effect` is null or a live effect holding a `Mutex<T>`, and `output` is null or a valid buffer.
unsafe fn tail<O, T: ApiEffect>(
    effect: *mut O,
    output: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    guard(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, || {
        // SAFETY: guaranteed by the caller.
        let (Some((effect, _)), Some(output)) = (unsafe { effect_ref::<O, T>(effect) }, unsafe {
            RawBuffer::new(output)
        }) else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        let mut effect = lock(effect);
        if !effect.tail_fits(&output) {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        }
        // SAFETY: guaranteed by the caller.
        effect_state(unsafe { with_output(&output, |out| effect.tail(out)) })
    })
}

/// Defines the retain, release, reset, tail size and tail functions of an effect.
macro_rules! effect_functions {
    ($handle:ty, $inner:ty, $retain:ident, $release:ident, $reset:ident, $tail_size:ident, $tail:ident) => {
        retain_release!($handle, Mutex<$inner>, $retain, $release);

        /// Resets the effect's internal state.
        ///
        /// # Safety
        ///
        /// The handle is null or a live effect of this type.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $reset(effect: $handle) {
            // SAFETY: guaranteed by the caller.
            unsafe { reset::<_, $inner>(effect) }
        }

        /// Number of tail samples left in the effect (0 for a null handle).
        ///
        /// # Safety
        ///
        /// The handle is null or a live effect of this type.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $tail_size(effect: $handle) -> IPLint32 {
            // SAFETY: guaranteed by the caller.
            unsafe { tail_size::<_, $inner>(effect) }
        }

        /// Writes the next frame of the effect's tail.
        ///
        /// # Safety
        ///
        /// The handle is null or a live effect of this type, and `out` is null or a valid buffer.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $tail(
            effect: $handle,
            out: *mut IPLAudioBuffer,
        ) -> IPLAudioEffectState {
            // SAFETY: guaranteed by the caller.
            unsafe { tail::<_, $inner>(effect, out) }
        }
    };
}

// --------------------------------------------------------------------------------------------------------------------
// Panning
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct Panning {
    effect: PanningEffect,
    num_speakers: usize,
}

impl ApiEffect for Panning {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, _: &RawBuffer) -> bool {
        true
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates a panning effect for a speaker layout.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid (with `numSpeakers` speakers for a
/// custom layout).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplPanningEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLPanningEffectSettings,
    effect: *mut IPLPanningEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |_, settings, _| {
                let layout = speaker_layout(&settings.speakerLayout).ok_or(IPL_STATUS_FAILURE)?;
                Ok(Panning {
                    effect: PanningEffect::new(&layout),
                    num_speakers: layout.num_speakers(),
                })
            },
        )
    }
}

/// Pans a mono buffer to the speaker layout. Not in place.
///
/// # Safety
///
/// The handle is null or a live panning effect, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplPanningEffectApply(
    effect: IPLPanningEffect,
    params: *mut IPLPanningEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut Panning, params, input, output, _| {
                let n = input.num_samples();
                if !has_shape(input, 1, n) || !has_shape(output, effect.num_speakers, n) {
                    return None;
                }
                let direction = from_ipl_vector(params.direction);
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(direction, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLPanningEffect,
    Panning,
    iplPanningEffectRetain,
    iplPanningEffectRelease,
    iplPanningEffectReset,
    iplPanningEffectGetTailSize,
    iplPanningEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Binaural
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct Binaural {
    effect: BinauralEffect,
    frame_size: usize,
}

impl ApiEffect for Binaural {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, out: &RawBuffer) -> bool {
        has_shape(out, NUM_EARS, self.frame_size)
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates a binaural effect sized for an HRTF.
///
/// # Safety
///
/// `context` is null or a live context, the other pointers are null or valid, and `hrtf` is null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplBinauralEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLBinauralEffectSettings,
    effect: *mut IPLBinauralEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let hrtf = hrtf_ref(settings.hrtf).ok_or(IPL_STATUS_FAILURE)?;
                Ok(Binaural {
                    effect: BinauralEffect::new(&audio, &lock(hrtf)),
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// Spatializes a mono or stereo buffer with an HRTF into a stereo buffer. Not in place. Peak delays are written for
/// callers of API version 4.1 or later.
///
/// # Safety
///
/// The handle is null or a live binaural effect, the other pointers are null or valid, `params.hrtf` is null or a
/// live HRTF, and `params.peakDelays` is null or valid for writes of 2 floats.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplBinauralEffectApply(
    effect: IPLBinauralEffect,
    params: *mut IPLBinauralEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut Binaural, params, input, output, context| {
                let hrtf = hrtf_ref(params.hrtf)?;
                let interpolation = match params.interpolation {
                    IPL_HRTFINTERPOLATION_NEAREST => HrtfInterpolation::NearestNeighbor,
                    IPL_HRTFINTERPOLATION_BILINEAR => HrtfInterpolation::Bilinear,
                    _ => return None,
                };
                let f = effect.frame_size;
                if !(has_shape(input, 1, f) || has_shape(input, 2, f))
                    || !has_shape(output, NUM_EARS, f)
                {
                    return None;
                }
                let peak_delays = if context.is_caller_api_version_at_least(4, 1) {
                    params.peakDelays.cast::<[f32; NUM_EARS]>().as_mut()
                } else {
                    None
                };

                let mut hrtf = lock(hrtf);
                let params = BinauralParams {
                    direction: from_ipl_vector(params.direction),
                    interpolation,
                    spatial_blend: params.spatialBlend,
                    phase_type: HrtfPhaseType::None,
                    hrtf: &mut hrtf,
                    peak_delays,
                };
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(params, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLBinauralEffect,
    Binaural,
    iplBinauralEffectRetain,
    iplBinauralEffectRelease,
    iplBinauralEffectReset,
    iplBinauralEffectGetTailSize,
    iplBinauralEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Virtual surround
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct VirtualSurround {
    effect: VirtualSurroundEffect,
    num_speakers: usize,
    frame_size: usize,
}

impl ApiEffect for VirtualSurround {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, out: &RawBuffer) -> bool {
        has_shape(out, NUM_EARS, self.frame_size)
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates a virtual surround effect for a speaker layout, sized for an HRTF.
///
/// # Safety
///
/// `context` is null or a live context, the other pointers are null or valid (with `numSpeakers` speakers for a
/// custom layout), and `hrtf` is null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplVirtualSurroundEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLVirtualSurroundEffectSettings,
    effect: *mut IPLVirtualSurroundEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let layout = speaker_layout(&settings.speakerLayout).ok_or(IPL_STATUS_FAILURE)?;
                let hrtf = hrtf_ref(settings.hrtf).ok_or(IPL_STATUS_FAILURE)?;
                Ok(VirtualSurround {
                    effect: VirtualSurroundEffect::new(&audio, &layout, &lock(hrtf)),
                    num_speakers: layout.num_speakers(),
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// Renders a buffer with one channel per speaker binaurally into a stereo buffer. Not in place.
///
/// # Safety
///
/// The handle is null or a live virtual surround effect, the other pointers are null or valid, and `params.hrtf` is
/// null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplVirtualSurroundEffectApply(
    effect: IPLVirtualSurroundEffect,
    params: *mut IPLVirtualSurroundEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut VirtualSurround, params, input, output, _| {
                let hrtf = hrtf_ref(params.hrtf)?;
                let f = effect.frame_size;
                if !has_shape(input, effect.num_speakers, f) || !has_shape(output, NUM_EARS, f) {
                    return None;
                }
                let mut hrtf = lock(hrtf);
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(&mut hrtf, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLVirtualSurroundEffect,
    VirtualSurround,
    iplVirtualSurroundEffectRetain,
    iplVirtualSurroundEffectRelease,
    iplVirtualSurroundEffectReset,
    iplVirtualSurroundEffectGetTailSize,
    iplVirtualSurroundEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Ambisonics encode
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct AmbisonicsEncode {
    effect: AmbisonicsEncodeEffect,
}

impl ApiEffect for AmbisonicsEncode {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, _: &RawBuffer) -> bool {
        true
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates an Ambisonics encode effect.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsEncodeEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLAmbisonicsEncodeEffectSettings,
    effect: *mut IPLAmbisonicsEncodeEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |_, settings, _| {
                let max_order = order(settings.maxOrder).ok_or(IPL_STATUS_FAILURE)?;
                Ok(AmbisonicsEncode {
                    effect: AmbisonicsEncodeEffect::new(max_order),
                })
            },
        )
    }
}

/// Encodes a mono buffer into Ambisonics of `params.order`. Only channels up to the maximum order are written. Not in
/// place.
///
/// # Safety
///
/// The handle is null or a live Ambisonics encode effect, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsEncodeEffectApply(
    effect: IPLAmbisonicsEncodeEffect,
    params: *mut IPLAmbisonicsEncodeEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut AmbisonicsEncode, params, input, output, _| {
                let order = order(params.order)?;
                let n = input.num_samples();
                if !has_shape(input, 1, n) || !has_shape(output, num_coeffs_for_order(order), n) {
                    return None;
                }
                let direction = from_ipl_vector(params.direction);
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(direction, order, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLAmbisonicsEncodeEffect,
    AmbisonicsEncode,
    iplAmbisonicsEncodeEffectRetain,
    iplAmbisonicsEncodeEffectRelease,
    iplAmbisonicsEncodeEffectReset,
    iplAmbisonicsEncodeEffectGetTailSize,
    iplAmbisonicsEncodeEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Ambisonics panning
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct AmbisonicsPanning {
    effect: AmbisonicsPanningEffect,
    max_order: usize,
    num_speakers: usize,
    frame_size: usize,
}

impl ApiEffect for AmbisonicsPanning {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, _: &RawBuffer) -> bool {
        true
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates an Ambisonics panning effect for a speaker layout.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid (with `numSpeakers` speakers for a
/// custom layout).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsPanningEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLAmbisonicsPanningEffectSettings,
    effect: *mut IPLAmbisonicsPanningEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let layout = speaker_layout(&settings.speakerLayout).ok_or(IPL_STATUS_FAILURE)?;
                let max_order = order(settings.maxOrder).ok_or(IPL_STATUS_FAILURE)?;
                Ok(AmbisonicsPanning {
                    effect: AmbisonicsPanningEffect::new(&audio, &layout, max_order),
                    max_order,
                    num_speakers: layout.num_speakers(),
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// Decodes Ambisonics of `params.order` (at most the maximum order) to the speaker layout. Not in place.
///
/// # Safety
///
/// The handle is null or a live Ambisonics panning effect, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsPanningEffectApply(
    effect: IPLAmbisonicsPanningEffect,
    params: *mut IPLAmbisonicsPanningEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut AmbisonicsPanning, params, input, output, _| {
                let order = order(params.order).filter(|&order| order <= effect.max_order)?;
                let f = effect.frame_size;
                if !has_shape(input, num_coeffs_for_order(order), f)
                    || !has_shape(output, effect.num_speakers, f)
                {
                    return None;
                }
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(order, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLAmbisonicsPanningEffect,
    AmbisonicsPanning,
    iplAmbisonicsPanningEffectRetain,
    iplAmbisonicsPanningEffectRelease,
    iplAmbisonicsPanningEffectReset,
    iplAmbisonicsPanningEffectGetTailSize,
    iplAmbisonicsPanningEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Ambisonics binaural
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct AmbisonicsBinaural {
    effect: AmbisonicsBinauralEffect,
    max_order: usize,
    frame_size: usize,
}

impl ApiEffect for AmbisonicsBinaural {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, out: &RawBuffer) -> bool {
        has_shape(out, NUM_EARS, self.frame_size)
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates an Ambisonics binaural effect sized for an HRTF. The maximum order is at most 3 (the order of the
/// Ambisonics HRTFs).
///
/// # Safety
///
/// `context` is null or a live context, the other pointers are null or valid, and `hrtf` is null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsBinauralEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLAmbisonicsBinauralEffectSettings,
    effect: *mut IPLAmbisonicsBinauralEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let hrtf = hrtf_ref(settings.hrtf).ok_or(IPL_STATUS_FAILURE)?;
                let max_order = order(settings.maxOrder)
                    .filter(|&order| order <= MAX_AMBISONICS_ORDER)
                    .ok_or(IPL_STATUS_FAILURE)?;
                Ok(AmbisonicsBinaural {
                    effect: AmbisonicsBinauralEffect::new(&audio, max_order, &lock(hrtf)),
                    max_order,
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// Renders Ambisonics of `params.order` (at most the maximum order) binaurally into a stereo buffer. Not in place.
///
/// # Safety
///
/// The handle is null or a live Ambisonics binaural effect, the other pointers are null or valid, and `params.hrtf` is
/// null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsBinauralEffectApply(
    effect: IPLAmbisonicsBinauralEffect,
    params: *mut IPLAmbisonicsBinauralEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut AmbisonicsBinaural, params, input, output, _| {
                let hrtf = hrtf_ref(params.hrtf)?;
                let order = order(params.order).filter(|&order| order <= effect.max_order)?;
                let f = effect.frame_size;
                if !has_shape(input, num_coeffs_for_order(order), f)
                    || !has_shape(output, NUM_EARS, f)
                {
                    return None;
                }
                let hrtf = lock(hrtf);
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(&hrtf, order, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLAmbisonicsBinauralEffect,
    AmbisonicsBinaural,
    iplAmbisonicsBinauralEffectRetain,
    iplAmbisonicsBinauralEffectRelease,
    iplAmbisonicsBinauralEffectReset,
    iplAmbisonicsBinauralEffectGetTailSize,
    iplAmbisonicsBinauralEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Ambisonics rotation
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct AmbisonicsRotation {
    effect: AmbisonicsRotateEffect,
    frame_size: usize,
}

impl ApiEffect for AmbisonicsRotation {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, _: &RawBuffer) -> bool {
        true
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates an Ambisonics rotation effect.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsRotationEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLAmbisonicsRotationEffectSettings,
    effect: *mut IPLAmbisonicsRotationEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let max_order = order(settings.maxOrder).ok_or(IPL_STATUS_FAILURE)?;
                Ok(AmbisonicsRotation {
                    effect: AmbisonicsRotateEffect::new(&audio, max_order),
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// Rotates Ambisonics of `params.order` into the listener's orientation. Only channels up to the maximum order are
/// written. Not in place.
///
/// # Safety
///
/// The handle is null or a live Ambisonics rotation effect, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsRotationEffectApply(
    effect: IPLAmbisonicsRotationEffect,
    params: *mut IPLAmbisonicsRotationEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut AmbisonicsRotation, params, input, output, _| {
                let order = order(params.order)?;
                let (coeffs, f) = (num_coeffs_for_order(order), effect.frame_size);
                if !has_shape(input, coeffs, f) || !has_shape(output, coeffs, f) {
                    return None;
                }
                let orientation = from_ipl_coordinate_space(&params.orientation);
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(&orientation, order, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLAmbisonicsRotationEffect,
    AmbisonicsRotation,
    iplAmbisonicsRotationEffectRetain,
    iplAmbisonicsRotationEffectRelease,
    iplAmbisonicsRotationEffectReset,
    iplAmbisonicsRotationEffectGetTailSize,
    iplAmbisonicsRotationEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Ambisonics decode
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct AmbisonicsDecode {
    effect: AmbisonicsDecodeEffect,
    max_order: usize,
    num_speakers: usize,
    frame_size: usize,
}

impl ApiEffect for AmbisonicsDecode {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    /// The tail of binaural rendering needs a stereo frame; that of panning writes anything. Both fit a buffer shaped
    /// like `apply`'s output.
    fn tail_fits(&self, out: &RawBuffer) -> bool {
        has_shape(out, self.num_speakers, self.frame_size)
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates an Ambisonics decode effect for a speaker layout; binaural rendering is available if an HRTF is given, with
/// a maximum order of at most 3.
///
/// # Safety
///
/// `context` is null or a live context, the other pointers are null or valid (with `numSpeakers` speakers for a
/// custom layout), and `hrtf` is null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsDecodeEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLAmbisonicsDecodeEffectSettings,
    effect: *mut IPLAmbisonicsDecodeEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let layout = speaker_layout(&settings.speakerLayout).ok_or(IPL_STATUS_FAILURE)?;
                let hrtf = hrtf_ref(settings.hrtf);
                let max_order = order(settings.maxOrder)
                    .filter(|&order| hrtf.is_none() || order <= MAX_AMBISONICS_ORDER)
                    .ok_or(IPL_STATUS_FAILURE)?;
                let hrtf = hrtf.map(lock);
                Ok(AmbisonicsDecode {
                    effect: AmbisonicsDecodeEffect::new(
                        &audio,
                        &layout,
                        max_order,
                        hrtf.as_deref(),
                    ),
                    max_order,
                    num_speakers: layout.num_speakers(),
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// Rotates Ambisonics of `params.order` (at most the maximum order) into the listener's orientation and renders it
/// binaurally or to the speaker layout. Asking for binaural rendering without an HRTF outputs nothing, like upstream.
/// Not in place.
///
/// # Safety
///
/// The handle is null or a live Ambisonics decode effect, the other pointers are null or valid, and `params.hrtf` is
/// null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAmbisonicsDecodeEffectApply(
    effect: IPLAmbisonicsDecodeEffect,
    params: *mut IPLAmbisonicsDecodeEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut AmbisonicsDecode, params, input, output, _| {
                // Upstream tests any non-zero value here, but only `IPL_TRUE` selects binaural rendering.
                let hrtf = hrtf_ref(params.hrtf);
                if params.binaural != IPL_FALSE && hrtf.is_none() {
                    return None;
                }
                let binaural = params.binaural == IPL_TRUE;
                let order = order(params.order).filter(|&order| order <= effect.max_order)?;
                let f = effect.frame_size;
                if !has_shape(input, num_coeffs_for_order(order), f)
                    || !has_shape(output, effect.num_speakers, f)
                {
                    return None;
                }
                let hrtf = hrtf.map(lock);
                let params = AmbisonicsDecodeParams {
                    order,
                    orientation: from_ipl_coordinate_space(&params.orientation),
                    binaural,
                    hrtf: hrtf.as_deref(),
                };
                Some(with_input_output(input, output, |input, out| {
                    effect.effect.apply(params, input, out)
                }))
            },
        )
    }
}

effect_functions!(
    IPLAmbisonicsDecodeEffect,
    AmbisonicsDecode,
    iplAmbisonicsDecodeEffectRetain,
    iplAmbisonicsDecodeEffectRelease,
    iplAmbisonicsDecodeEffectReset,
    iplAmbisonicsDecodeEffectGetTailSize,
    iplAmbisonicsDecodeEffectGetTail
);

// --------------------------------------------------------------------------------------------------------------------
// Direct
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct Direct {
    effect: DirectEffect,
    frame_size: usize,
}

impl ApiEffect for Direct {
    fn reset(&mut self) {
        self.effect.reset();
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, _: &RawBuffer) -> bool {
        true
    }

    fn tail(&mut self, out: &mut AudioBufferMut<'_>) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates a direct effect for buffers of `numChannels` channels.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplDirectEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLDirectEffectSettings,
    effect: *mut IPLDirectEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings, _| {
                let num_channels =
                    usize::try_from(settings.numChannels).map_err(|_| IPL_STATUS_FAILURE)?;
                Ok(Direct {
                    effect: DirectEffect::new(&audio, num_channels),
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

/// The core's parameters for the API's. Upstream compares the transmission type with both values, so an unknown one
/// applies neither occlusion nor transmission when transmission is on; clearing both flags does the same.
fn direct_params(params: &IPLDirectEffectParams) -> DirectEffectParams {
    let mut flags = DirectEffectFlags::from_bits(params.flags as u32);
    let transmission_type = match params.transmissionType {
        IPL_TRANSMISSIONTYPE_FREQINDEPENDENT => TransmissionType::FreqIndependent,
        IPL_TRANSMISSIONTYPE_FREQDEPENDENT => TransmissionType::FreqDependent,
        _ => {
            if flags.contains(DirectEffectFlags::TRANSMISSION) {
                let cleared =
                    DirectEffectFlags::OCCLUSION.bits() | DirectEffectFlags::TRANSMISSION.bits();
                flags = DirectEffectFlags::from_bits(flags.bits() & !cleared);
            }
            TransmissionType::FreqIndependent
        }
    };
    DirectEffectParams {
        direct_path: DirectSoundPath {
            distance_attenuation: params.distanceAttenuation,
            air_absorption: params.airAbsorption,
            delay: 0.0,
            occlusion: params.occlusion,
            transmission: params.transmission,
            directivity: params.directivity,
        },
        flags,
        transmission_type,
    }
}

/// Applies distance attenuation, air absorption, directivity, occlusion and transmission to every channel. Can be
/// applied in place (`in == out`).
///
/// # Safety
///
/// The handle is null or a live direct effect, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplDirectEffectApply(
    effect: IPLDirectEffect,
    params: *mut IPLDirectEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    // SAFETY: guaranteed by the caller.
    unsafe {
        apply(
            effect,
            params,
            in_,
            out,
            |effect: &mut Direct, params, input, output, _| {
                let (c, f) = (effect.effect.num_channels(), effect.frame_size);
                if !has_shape(input, c, f) || !has_shape(output, c, f) {
                    return None;
                }
                let params = direct_params(params);
                Some(if input.is_same(output) {
                    with_output(output, |buffer| {
                        effect.effect.apply_in_place(&params, buffer)
                    })
                } else {
                    with_input_output(input, output, |input, out| {
                        effect.effect.apply(&params, input, out)
                    })
                })
            },
        )
    }
}

effect_functions!(
    IPLDirectEffect,
    Direct,
    iplDirectEffectRetain,
    iplDirectEffectRelease,
    iplDirectEffectReset,
    iplDirectEffectGetTailSize,
    iplDirectEffectGetTail
);
