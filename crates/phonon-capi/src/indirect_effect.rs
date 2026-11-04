//! The reflection effect and mixer (upstream `api_indirect_effect.cpp`, `indirect_effect.cpp`) and the path effect
//! (`api_path_effect.cpp`).
//!
//! As the other effects ([`crate::effects`]), a call whose buffers or parameters do not fit returns
//! `IPL_AUDIOEFFECTSTATE_TAILCOMPLETE` and leaves the output alone.

use std::sync::Mutex;

use steam_audio::audio_buffer::{AudioEffectState, AudioSettings};
use steam_audio::bands::NUM_BANDS;
use steam_audio::effects::{
    HybridReverbEffect, HybridReverbIr, HybridReverbParams, OverlapSaveConvolutionEffect,
    OverlapSaveConvolutionMixer, OverlapSaveConvolutionSettings, OverlapSaveFir, PathEffect,
    PathEffectParams, PathEffectSettings, PathSpatialization, ReverbEffect,
};
use steam_audio::fft::RealFft;
use steam_audio::hrtf::NUM_EARS;
use steam_audio::math::CoordinateSpace3;
use steam_audio::sh::num_coeffs_for_order;
use steam_audio::simulation::Reverb;

use crate::buffer::{RawBuffer, with_input, with_input_output, with_output};
use crate::effects::{
    ApiEffect, apply, create, effect_functions, effect_ref, effect_state, has_shape, order,
    speaker_layout,
};
use crate::handle::{lock, retain_release};
use crate::hrtf::hrtf_ref;
use crate::rand::with_rand;
use crate::simulator::ReflectionIr;
use crate::types::*;
use crate::{from_ipl_coordinate_space, guard};

// --------------------------------------------------------------------------------------------------------------------
// Reflection effect
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
enum Reflection {
    Convolution(Box<OverlapSaveConvolutionEffect>),
    Parametric(Box<ReverbEffect>),
    Hybrid(Box<HybridReverbEffect>),
    /// TrueAudio Next (out of scope) or an unknown type: upstream's effect then does nothing.
    None,
}

/// Convolution and hybrid: the IR's channels and blocks, and the length of a block's spectrum.
#[derive(Debug, Clone, Copy)]
struct IrShape {
    num_channels: usize,
    num_blocks: usize,
    num_spectrum_samples: usize,
}

#[derive(Debug)]
pub(crate) struct ReflectionEffect {
    effect: Reflection,
    shape: IrShape,
    num_channels: usize,
    frame_size: usize,
}

impl IrShape {
    /// Whether a new IR can replace the effect's: it keeps it for the frames that follow, which need all of the
    /// effect's channels and blocks.
    fn accepts(&self, fir: &OverlapSaveFir) -> bool {
        fir.num_channels() >= self.num_channels
            && fir.num_blocks() >= self.num_blocks
            && fir.num_spectrum_samples() == self.num_spectrum_samples
    }
}

#[derive(Debug)]
pub(crate) struct ReflectionMixer {
    /// Convolution only (upstream's mixer of another type holds nothing and does nothing).
    mixer: Option<OverlapSaveConvolutionMixer>,
    num_channels: usize,
    frame_size: usize,
}

/// The IR behind an `IPLReflectionEffectIR` (null for none).
///
/// # Safety
///
/// `ir` is null or was returned by `iplSourceGetOutputs` for a live source.
unsafe fn reflection_ir<'a>(ir: IPLReflectionEffectIR) -> Option<&'a ReflectionIr> {
    // SAFETY: guaranteed by the caller.
    unsafe { ir.cast::<ReflectionIr>().as_ref() }
}

/// Creates a reflection effect. Convolution and hybrid effects need a positive IR size and channel count
/// (`IPL_STATUS_FAILURE` otherwise); a TrueAudio Next or unknown type gives an effect that does nothing. A parametric or
/// hybrid effect draws its reverb's delay lengths from this thread's `rand()` state.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLReflectionEffectSettings,
    effect: *mut IPLReflectionEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings: &IPLReflectionEffectSettings, _| {
                let sizes = || -> Result<(usize, usize), IPLerror> {
                    let channels = usize::try_from(settings.numChannels)
                        .ok()
                        .filter(|&n| n > 0)
                        .ok_or(IPL_STATUS_FAILURE)?;
                    let ir_size = usize::try_from(settings.irSize)
                        .ok()
                        .filter(|&n| n > 0)
                        .ok_or(IPL_STATUS_FAILURE)?;
                    Ok((channels, ir_size))
                };
                let convolution_settings =
                    |(num_channels, ir_size)| OverlapSaveConvolutionSettings {
                        num_channels,
                        ir_size,
                    };
                let (effect, sizes) = match settings.r#type {
                    IPL_REFLECTIONEFFECTTYPE_CONVOLUTION => {
                        let sizes = sizes()?;
                        let effect =
                            OverlapSaveConvolutionEffect::new(&audio, &convolution_settings(sizes));
                        (Reflection::Convolution(Box::new(effect)), Some(sizes))
                    }
                    IPL_REFLECTIONEFFECTTYPE_PARAMETRIC => (
                        Reflection::Parametric(Box::new(with_rand(|rand| {
                            ReverbEffect::new(&audio, rand)
                        }))),
                        None,
                    ),
                    IPL_REFLECTIONEFFECTTYPE_HYBRID => {
                        let sizes = sizes()?;
                        let effect = with_rand(|rand| {
                            HybridReverbEffect::new(&audio, &convolution_settings(sizes), rand)
                        });
                        (Reflection::Hybrid(Box::new(effect)), Some(sizes))
                    }
                    _ => (Reflection::None, None),
                };
                let (num_channels, ir_size) = sizes.unwrap_or((0, 0));
                Ok(ReflectionEffect {
                    effect,
                    shape: IrShape {
                        num_channels,
                        num_blocks: OverlapSaveConvolutionEffect::num_blocks(
                            audio.frame_size,
                            ir_size,
                        ),
                        num_spectrum_samples: RealFft::new(2 * audio.frame_size)
                            .num_complex_samples(),
                    },
                    num_channels,
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

retain_release!(
    IPLReflectionEffect,
    Mutex<ReflectionEffect>,
    iplReflectionEffectRetain,
    iplReflectionEffectRelease
);

/// Resets the effect's internal state.
///
/// # Safety
///
/// The handle is null or a live reflection effect.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionEffectReset(effect: IPLReflectionEffect) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some((effect, _)) = unsafe { effect_ref::<_, ReflectionEffect>(effect) } {
            match &mut lock(effect).effect {
                Reflection::Convolution(e) => e.reset(),
                Reflection::Parametric(e) => e.reset(),
                Reflection::Hybrid(e) => e.reset(),
                Reflection::None => {}
            }
        }
    })
}

/// The mixer behind a handle (null for none).
///
/// # Safety
///
/// `mixer` is null or a live reflection mixer.
unsafe fn mixer_ref<'a>(mixer: IPLReflectionMixer) -> Option<&'a Mutex<ReflectionMixer>> {
    // SAFETY: guaranteed by the caller.
    unsafe { effect_ref::<_, ReflectionMixer>(mixer) }.map(|(mixer, _)| mixer)
}

/// The IR a convolution or hybrid frame uses: takes the source's latest IR, if it published one since the last frame,
/// and runs `f` with it (`Some` if it is new). `None` if the new IR does not fit the effect.
fn with_ir<R>(
    shape: IrShape,
    ir: &ReflectionIr,
    f: impl FnOnce(Option<&mut OverlapSaveFir>) -> R,
) -> Option<R> {
    let mut reader = lock(&ir.0);
    let updated = reader.update_read_buffer();
    let mut fir = reader.read_buffer();
    if !updated {
        return Some(f(None));
    }
    shape.accepts(&fir).then(|| f(Some(&mut fir)))
}

/// Applies a reflection effect to a mono frame: convolution with the IR, parametric reverb, or both (hybrid). With a
/// mixer, a convolution effect adds its result to the mixer instead (`out` is then not used and may be null), and the
/// other types do nothing, as upstream.
///
/// A convolution effect without an IR outputs silence. An IR the source published since the last frame replaces the
/// effect's (crossfading over the frame); one with fewer channels or blocks than the effect's, or of another frame
/// size, is refused.
///
/// # Safety
///
/// The handles are null or live objects, `params.ir` is null or an IR of a live source, and the other pointers are null
/// or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionEffectApply(
    effect: IPLReflectionEffect,
    params: *mut IPLReflectionEffectParams,
    in_: *mut IPLAudioBuffer,
    out: *mut IPLAudioBuffer,
    mixer: IPLReflectionMixer,
) -> IPLAudioEffectState {
    guard(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, || {
        // SAFETY: guaranteed by the caller.
        let (Some((effect, _)), Some(params), Some(input)) = (
            unsafe { effect_ref::<_, ReflectionEffect>(effect) },
            unsafe { params.as_ref() },
            unsafe { RawBuffer::new(in_) },
        ) else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        // SAFETY: guaranteed by the caller.
        let mixer = unsafe { mixer_ref(mixer) };
        // SAFETY: guaranteed by the caller.
        let output = unsafe { RawBuffer::new(out) };
        // SAFETY: guaranteed by the caller.
        let ir = unsafe { reflection_ir(params.ir) };

        let mut effect = lock(effect);
        let f = effect.frame_size;
        if !has_shape(&input, 1, f) {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        }
        let Some(num_channels) = usize::try_from(params.numChannels)
            .ok()
            .filter(|&n| n <= effect.num_channels)
            .or_else(|| matches!(effect.effect, Reflection::Parametric(_)).then_some(0))
        else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        let effect_channels = effect.num_channels;

        let state = match (&effect.effect, mixer) {
            (Reflection::Convolution(_), Some(mixer)) => {
                let Some(ir) = ir else {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                let mut mixer = lock(mixer);
                let fits = mixer.frame_size == f && mixer.num_channels <= effect_channels;
                let (Some(m), true) = (mixer.mixer.as_mut(), fits) else {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                let shape = effect.shape;
                let Reflection::Convolution(e) = &mut effect.effect else {
                    unreachable!("matched above")
                };
                with_ir(shape, ir, |fir| {
                    // SAFETY: guaranteed by the caller.
                    unsafe {
                        with_input(&input, |input| {
                            e.apply_to_mixer(fir, num_channels, input, m)
                        })
                    }
                })
            }
            (Reflection::Convolution(_), None) => {
                let Some(output) = output else {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                let Some(ir) = ir else {
                    // SAFETY: guaranteed by the caller.
                    unsafe { with_output(&output, |out| out.make_silent()) };
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                if !has_shape(&output, effect_channels, f) {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                }
                let shape = effect.shape;
                let Reflection::Convolution(e) = &mut effect.effect else {
                    unreachable!("matched above")
                };
                with_ir(shape, ir, |fir| {
                    // SAFETY: guaranteed by the caller.
                    unsafe {
                        with_input_output(&input, &output, |input, out| {
                            e.apply(fir, num_channels, input, out)
                        })
                    }
                })
            }
            (Reflection::Parametric(_), None) => {
                let Some(output) = output.filter(|o| o.num_samples() == f) else {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                let Reflection::Parametric(e) = &mut effect.effect else {
                    unreachable!("matched above")
                };
                let reverb = Reverb {
                    reverb_times: params.reverbTimes,
                };
                // SAFETY: guaranteed by the caller.
                Some(unsafe {
                    with_input_output(&input, &output, |input, out| e.apply(&reverb, input, out))
                })
            }
            (Reflection::Hybrid(_), None) => {
                let (Some(output), Ok(delay)) = (output, usize::try_from(params.delay)) else {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                if !has_shape(&output, effect_channels, f) {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                }
                let reverb = Reverb {
                    reverb_times: params.reverbTimes,
                };
                let eq_coeffs: [f32; NUM_BANDS] = params.eq;
                let shape = effect.shape;
                let Reflection::Hybrid(e) = &mut effect.effect else {
                    unreachable!("matched above")
                };
                let run = |e: &mut HybridReverbEffect, ir: HybridReverbIr<'_>| {
                    let params = HybridReverbParams {
                        ir,
                        num_channels,
                        reverb: &reverb,
                        eq_coeffs: &eq_coeffs,
                        delay,
                    };
                    // SAFETY: guaranteed by the caller.
                    unsafe {
                        with_input_output(&input, &output, |input, out| e.apply(params, input, out))
                    }
                };
                match ir {
                    None => Some(run(e, HybridReverbIr::None)),
                    Some(ir) => with_ir(shape, ir, |fir| match fir {
                        Some(fir) => run(e, HybridReverbIr::Updated(fir)),
                        None => run(e, HybridReverbIr::Current),
                    }),
                }
            }
            _ => return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE,
        };
        state.map_or(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, effect_state)
    })
}

/// Number of tail samples left in the effect (0 for a null handle).
///
/// # Safety
///
/// The handle is null or a live reflection effect.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionEffectGetTailSize(
    effect: IPLReflectionEffect,
) -> IPLint32 {
    guard(0, || {
        // SAFETY: guaranteed by the caller.
        let Some((effect, _)) = (unsafe { effect_ref::<_, ReflectionEffect>(effect) }) else {
            return 0;
        };
        let size = match &lock(effect).effect {
            Reflection::Convolution(e) => e.num_tail_samples_remaining(),
            Reflection::Parametric(e) => e.num_tail_samples_remaining(),
            Reflection::Hybrid(e) => e.num_tail_samples_remaining(),
            Reflection::None => 0,
        };
        i32::try_from(size).unwrap_or(i32::MAX)
    })
}

/// Writes the next frame of the effect's tail, into the mixer for a convolution effect given one (the other types then
/// do nothing, as upstream).
///
/// # Safety
///
/// The handles are null or live objects, and `out` is null or a valid buffer.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionEffectGetTail(
    effect: IPLReflectionEffect,
    out: *mut IPLAudioBuffer,
    mixer: IPLReflectionMixer,
) -> IPLAudioEffectState {
    guard(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, || {
        // SAFETY: guaranteed by the caller.
        let Some((effect, _)) = (unsafe { effect_ref::<_, ReflectionEffect>(effect) }) else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        // SAFETY: guaranteed by the caller.
        let mixer = unsafe { mixer_ref(mixer) };
        // SAFETY: guaranteed by the caller.
        let output = unsafe { RawBuffer::new(out) };

        let mut effect = lock(effect);
        let f = effect.frame_size;
        let channels = effect.num_channels;

        let state = match (&mut effect.effect, mixer) {
            (Reflection::Convolution(e), Some(mixer)) => {
                let mut mixer = lock(mixer);
                let fits = mixer.frame_size == f && mixer.num_channels <= channels;
                match (mixer.mixer.as_mut(), fits) {
                    (Some(m), true) => e.tail_to_mixer(m),
                    _ => return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE,
                }
            }
            (_, Some(_)) | (Reflection::None, None) => return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE,
            (e, None) => {
                let Some(output) = output.filter(|o| o.num_samples() == f) else {
                    return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
                };
                let n = output.num_channels();
                match e {
                    Reflection::Convolution(e) if n <= channels => {
                        // SAFETY: guaranteed by the caller.
                        unsafe { with_output(&output, |out| e.tail(out)) }
                    }
                    Reflection::Parametric(e) => {
                        // SAFETY: guaranteed by the caller.
                        unsafe { with_output(&output, |out| e.tail(out)) }
                    }
                    Reflection::Hybrid(e) if n <= channels => {
                        // SAFETY: guaranteed by the caller.
                        unsafe { with_output(&output, |out| e.tail(out)) }
                    }
                    _ => return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE,
                }
            }
        };
        effect_state(state)
    })
}

// --------------------------------------------------------------------------------------------------------------------
// Reflection mixer
// --------------------------------------------------------------------------------------------------------------------

/// Creates a reflection mixer. A convolution mixer needs a positive channel count (`IPL_STATUS_FAILURE` otherwise);
/// a mixer of another type does nothing, as upstream's.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionMixerCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLReflectionEffectSettings,
    mixer: *mut IPLReflectionMixer,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            mixer,
            |audio: AudioSettings, settings: &IPLReflectionEffectSettings, _| {
                if settings.r#type != IPL_REFLECTIONEFFECTTYPE_CONVOLUTION {
                    return Ok(ReflectionMixer {
                        mixer: None,
                        num_channels: 0,
                        frame_size: audio.frame_size,
                    });
                }
                let num_channels = usize::try_from(settings.numChannels)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or(IPL_STATUS_FAILURE)?;
                let mixer = OverlapSaveConvolutionMixer::new(
                    &audio,
                    &OverlapSaveConvolutionSettings {
                        num_channels,
                        ir_size: 0,
                    },
                );
                Ok(ReflectionMixer {
                    mixer: Some(mixer),
                    num_channels,
                    frame_size: audio.frame_size,
                })
            },
        )
    }
}

retain_release!(
    IPLReflectionMixer,
    Mutex<ReflectionMixer>,
    iplReflectionMixerRetain,
    iplReflectionMixerRelease
);

/// Clears what has been mixed.
///
/// # Safety
///
/// The handle is null or a live reflection mixer.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionMixerReset(mixer: IPLReflectionMixer) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(mixer) = unsafe { mixer_ref(mixer) } {
            if let Some(m) = lock(mixer).mixer.as_mut() {
                m.reset();
            }
        }
    })
}

/// Writes the first `params.numChannels` channels of what has been mixed into `out` and clears the mixer. Always
/// returns `IPL_AUDIOEFFECTSTATE_TAILCOMPLETE`, as upstream.
///
/// # Safety
///
/// The handle is null or a live reflection mixer, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplReflectionMixerApply(
    mixer: IPLReflectionMixer,
    params: *mut IPLReflectionEffectParams,
    out: *mut IPLAudioBuffer,
) -> IPLAudioEffectState {
    guard(IPL_AUDIOEFFECTSTATE_TAILCOMPLETE, || {
        // SAFETY: guaranteed by the caller.
        let (Some(mixer), Some(params), Some(output)) = (
            unsafe { mixer_ref(mixer) },
            unsafe { params.as_ref() },
            unsafe { RawBuffer::new(out) },
        ) else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        let mut mixer = lock(mixer);
        let (f, channels) = (mixer.frame_size, mixer.num_channels);
        let Some(m) = mixer.mixer.as_mut() else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        let Some(num_channels) = usize::try_from(params.numChannels)
            .ok()
            .filter(|&n| n <= channels && n <= output.num_channels())
        else {
            return IPL_AUDIOEFFECTSTATE_TAILCOMPLETE;
        };
        if output.num_samples() == f {
            // SAFETY: guaranteed by the caller.
            unsafe { with_output(&output, |out| m.apply(num_channels, out)) };
        }
        IPL_AUDIOEFFECTSTATE_TAILCOMPLETE
    })
}

// --------------------------------------------------------------------------------------------------------------------
// Path effect
// --------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct Path {
    effect: PathEffect,
    max_order: usize,
    frame_size: usize,
    /// Spatializing: the speakers panned to, and the HRTF length binaural frames need.
    spatialization: Option<(usize, usize)>,
    /// Whether the last frame was rendered binaurally (its tail then has two channels).
    prev_binaural: bool,
}

impl ApiEffect for Path {
    fn reset(&mut self) {
        self.effect.reset();
        self.prev_binaural = false;
    }

    fn tail_size(&self) -> usize {
        self.effect.num_tail_samples_remaining()
    }

    fn tail_fits(&self, out: &RawBuffer) -> bool {
        out.num_samples() == self.frame_size
            && (!self.prev_binaural || out.num_channels() == NUM_EARS)
    }

    fn tail(
        &mut self,
        out: &mut steam_audio::audio_buffer::AudioBufferMut<'_>,
    ) -> AudioEffectState {
        self.effect.tail(out)
    }
}

/// Creates a path effect of an Ambisonics order from 0 to 3. Callers of API version 4.4 or later can ask it to
/// spatialize, which needs a valid speaker layout and an HRTF (`IPL_STATUS_FAILURE` otherwise).
///
/// # Safety
///
/// `context` is null or a live context, the other pointers are null or valid, and the settings' HRTF is null or live.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplPathEffectCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    effectSettings: *mut IPLPathEffectSettings,
    effect: *mut IPLPathEffect,
) -> IPLerror {
    // SAFETY: guaranteed by the caller.
    unsafe {
        create(
            context,
            audioSettings,
            effectSettings,
            effect,
            |audio, settings: &IPLPathEffectSettings, context| {
                let max_order = order(settings.maxOrder)
                    .filter(|&o| o <= 3)
                    .ok_or(IPL_STATUS_FAILURE)?;
                let spatialize =
                    context.is_caller_api_version_at_least(4, 4) && settings.spatialize == IPL_TRUE;

                let (effect, spatialization) = if spatialize {
                    let layout =
                        speaker_layout(&settings.speakerLayout).ok_or(IPL_STATUS_FAILURE)?;
                    let hrtf = hrtf_ref(settings.hrtf).ok_or(IPL_STATUS_FAILURE)?;
                    let hrtf = lock(hrtf);
                    let effect = PathEffect::new(
                        &audio,
                        &PathEffectSettings {
                            max_order,
                            spatialization: Some(PathSpatialization {
                                speaker_layout: &layout,
                                hrtf: &hrtf,
                            }),
                        },
                    );
                    (effect, Some((layout.num_speakers(), hrtf.num_samples())))
                } else {
                    let effect = PathEffect::new(
                        &audio,
                        &PathEffectSettings {
                            max_order,
                            spatialization: None,
                        },
                    );
                    (effect, None)
                };
                Ok(Path {
                    effect,
                    max_order,
                    frame_size: audio.frame_size,
                    spatialization,
                    prev_binaural: false,
                })
            },
        )
    }
}

/// Renders a mono frame along the simulated paths: Ambisonics of the maximum order (channels above `params.order`
/// silent), or, when spatializing, panned to the speakers or (for callers of API version 4.4 or later asking for it)
/// rendered binaurally with `params.hrtf`, which must have the length of the effect's. Not in place.
///
/// # Safety
///
/// The handle is null or a live path effect, the other pointers are null or valid, `params.shCoeffs` is null or
/// valid for reads of `(order + 1)²` floats, and `params.hrtf` is null or a live HRTF.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplPathEffectApply(
    effect: IPLPathEffect,
    params: *mut IPLPathEffectParams,
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
            |effect: &mut Path, params: &IPLPathEffectParams, input, output, context| {
                let f = effect.frame_size;
                let order = order(params.order).filter(|&o| o <= effect.max_order)?;
                if !has_shape(input, 1, f) || params.shCoeffs.is_null() {
                    return None;
                }
                let sh_coeffs =
                    std::slice::from_raw_parts(params.shCoeffs, num_coeffs_for_order(order));

                let version_4_4 = context.is_caller_api_version_at_least(4, 4);
                let binaural = version_4_4 && params.binaural == IPL_TRUE;
                let listener = if version_4_4 {
                    from_ipl_coordinate_space(&params.listener)
                } else {
                    CoordinateSpace3::default()
                };
                let hrtf = if version_4_4 {
                    hrtf_ref(params.hrtf)
                } else {
                    None
                };

                match effect.spatialization {
                    None => {
                        if !has_shape(output, num_coeffs_for_order(effect.max_order), f) {
                            return None;
                        }
                    }
                    Some((_, hrtf_length)) if binaural => {
                        let fits = hrtf.is_some_and(|h| lock(h).num_samples() == hrtf_length);
                        if !fits || !has_shape(output, NUM_EARS, f) {
                            return None;
                        }
                    }
                    Some((num_speakers, _)) => {
                        if !has_shape(output, num_speakers, f) {
                            return None;
                        }
                    }
                }

                let hrtf_guard = match (&effect.spatialization, binaural) {
                    (Some(_), true) => hrtf.map(lock),
                    _ => None,
                };
                let params = PathEffectParams {
                    eq_coeffs: &params.eqCoeffs,
                    sh_coeffs,
                    order,
                    binaural,
                    hrtf: hrtf_guard.as_deref(),
                    listener,
                    normalize_eq: params.normalizeEQ == IPL_TRUE,
                };
                let state = with_input_output(input, output, |input, out| {
                    effect.effect.apply(&params, input, out)
                });
                effect.prev_binaural = effect.spatialization.is_some() && binaural;
                Some(state)
            },
        )
    }
}

effect_functions!(
    IPLPathEffect,
    Path,
    iplPathEffectRetain,
    iplPathEffectRelease,
    iplPathEffectReset,
    iplPathEffectGetTailSize,
    iplPathEffectGetTail
);
