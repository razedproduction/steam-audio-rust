//! Parity tests for stage 8a against golden data from `tools/golden/src/capi.cpp`: upstream's C API (contexts, audio
//! buffers, HRTFs, the audio effects that need no simulator, and the direct-sound model functions), called through the
//! same `ipl*` functions exported by this crate.
//!
//! Inputs are rebuilt from the generator's `Lcg` draws in the same order. The generator uses buffers from
//! `iplAudioBufferAllocate`, whose channels are contiguous; every audio case runs here twice, with contiguous channels
//! and with channels scattered in memory (reversed, with gaps), which the library bridges through scratch buffers. Both
//! must match the same goldens.

#![allow(unsafe_code)] // The tests call the C API.
// The HRTF tests need SOFA support.
#![cfg_attr(not(feature = "sofa"), allow(dead_code, unused_imports))]

use std::ffi::{CString, c_void};
use std::mem::offset_of;
use std::ptr;

use parity::{Element, Lcg, Tolerance, compare, data_path, golden};
use phonon::*;
use steam_audio::math::{CoordinateSpace3, Vector3};

const SAMPLING_RATE: i32 = 48000;
const FRAME_SIZE: usize = 97;
const SENTINEL: f32 = 9.0;

const VERSION: u32 = STEAMAUDIO_VERSION;
const VERSION_40: u32 = 4 << 16;
const VERSION_41: u32 = (4 << 16) | (1 << 8);

/// Signal tolerance relative to the peak.
const SIGNAL: f64 = 1e-5;

/// Signal tolerance relative to the peak with synthesized HRTFs (bilinear interpolation, spatial blend), as in stage 3.3:
/// their phase is interpolated unwrapped from FFT spectra that already differ (R1, R10).
const SYNTHESIZED: f64 = 5e-5;

/// Prints the largest error relative to the peak (for measuring bit-exactness with `--nocapture`).
fn report(name: &str, actual: &[f32], expected: &[f32]) {
    let peak = expected
        .iter()
        .filter(|&&x| x != SENTINEL)
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    let worst = actual.iter().zip(expected).fold(0.0f64, |m, (&a, &e)| {
        m.max((f64::from(a) - f64::from(e)).abs())
    });
    let identical = actual
        .iter()
        .zip(expected)
        .all(|(a, e)| a.to_bits() == e.to_bits());
    eprintln!(
        "{name}: worst error {:.3e} x peak{}",
        if peak > 0.0 { worst / peak } else { worst },
        if identical { " (bit-exact)" } else { "" }
    );
}

#[track_caller]
fn check<T: Element + Into<f64>>(name: &str, actual: &[T], tol: impl Fn(&[T]) -> Tolerance) {
    let expected = golden::<T>(&format!("capi/{name}"));
    if let Err(m) = compare(actual, &expected.data, tol(&expected.data)) {
        panic!("{name}: {m}");
    }
}

#[track_caller]
fn check_exact_i32(name: &str, actual: &[i32]) {
    assert_eq!(
        actual,
        golden::<i32>(&format!("capi/{name}")).data,
        "{name}"
    );
}

/// Peak-relative tolerance, with the peak taken over values other than the sentinel.
fn signal(expected: &[f32], ratio: f64) -> Tolerance {
    let peak = expected
        .iter()
        .filter(|&&x| x != SENTINEL)
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    Tolerance::new(ratio * peak, 0.0)
}

#[track_caller]
fn check_signal(name: &str, actual: &[f32], ratio: f64) {
    let expected = golden::<f32>(&format!("capi/{name}"));
    report(name, actual, &expected.data);
    if let Err(m) = compare(actual, &expected.data, signal(&expected.data, ratio)) {
        panic!("{name}: {m}");
    }
}

fn make_context(version: u32) -> IPLContext {
    let mut settings = IPLContextSettings {
        version,
        logCallback: None,
        allocateCallback: None,
        freeCallback: None,
        simdLevel: IPL_SIMDLEVEL_AVX2,
        flags: 0,
    };
    let mut context = ptr::null_mut();
    // SAFETY: valid pointers.
    let status = unsafe { iplContextCreate(&mut settings, &mut context) };
    assert_eq!(status, IPL_STATUS_SUCCESS);
    context
}

fn release_context(mut context: IPLContext) {
    // SAFETY: a live context.
    unsafe { iplContextRelease(&mut context) };
    assert!(context.is_null());
}

fn audio_settings() -> IPLAudioSettings {
    IPLAudioSettings {
        samplingRate: SAMPLING_RATE,
        frameSize: FRAME_SIZE as i32,
    }
}

fn default_hrtf_settings() -> IPLHRTFSettings {
    IPLHRTFSettings {
        r#type: IPL_HRTFTYPE_DEFAULT,
        sofaFileName: ptr::null(),
        sofaData: ptr::null(),
        sofaDataSize: 0,
        volume: 1.0,
        normType: IPL_HRTFNORMTYPE_NONE,
    }
}

fn make_hrtf(context: IPLContext, settings: IPLHRTFSettings) -> IPLHRTF {
    let mut audio = audio_settings();
    let mut settings = settings;
    let mut hrtf = ptr::null_mut();
    // SAFETY: valid pointers and a live context.
    let status = unsafe { iplHRTFCreate(context, &mut audio, &mut settings, &mut hrtf) };
    assert_eq!(status, IPL_STATUS_SUCCESS);
    hrtf
}

fn draw_vector(lcg: &mut Lcg) -> IPLVector3 {
    let x = lcg.next_signed();
    let y = lcg.next_signed();
    let z = lcg.next_signed();
    IPLVector3 { x, y, z }
}

fn draw_unit(lcg: &mut Lcg) -> f32 {
    let u = lcg.next_signed();
    0.5 * (u + 1.0)
}

fn ipl(v: Vector3) -> IPLVector3 {
    IPLVector3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

/// An orientation with a random ahead vector, as the generator's `drawOrientation`.
fn draw_orientation(lcg: &mut Lcg) -> IPLCoordinateSpace3 {
    let v = draw_vector(lcg);
    let ahead = Vector3::new(v.x, v.y, v.z).normalize_or_zero();
    let space = CoordinateSpace3::from_ahead(ahead, Vector3::ZERO);
    IPLCoordinateSpace3 {
        right: ipl(space.right),
        up: ipl(space.up),
        ahead: ipl(space.ahead),
        origin: ipl(space.origin),
    }
}

/// An audio buffer owned by the test: channels back to back, or reversed with gaps between them.
struct Buffer {
    channels: usize,
    samples: usize,
    data: Vec<f32>,
    offsets: Vec<usize>,
    pointers: Vec<*mut f32>,
    raw: IPLAudioBuffer,
}

impl Buffer {
    fn new(channels: usize, samples: usize, scattered: bool) -> Self {
        let (len, offsets) = if scattered {
            let stride = samples + 3;
            (
                channels * stride,
                (0..channels)
                    .map(|i| (channels - 1 - i) * stride + 1)
                    .collect(),
            )
        } else {
            (
                channels * samples,
                (0..channels).map(|i| i * samples).collect(),
            )
        };
        Self {
            channels,
            samples,
            data: vec![0.0; len],
            offsets,
            pointers: Vec::new(),
            raw: IPLAudioBuffer {
                numChannels: 0,
                numSamples: 0,
                data: ptr::null_mut(),
            },
        }
    }

    /// The buffer for the C API (pointers are refreshed, so they stay derived from the latest borrow of the data).
    fn ipl(&mut self) -> *mut IPLAudioBuffer {
        let base = self.data.as_mut_ptr();
        self.pointers = self.offsets.iter().map(|&o| base.wrapping_add(o)).collect();
        self.raw = IPLAudioBuffer {
            numChannels: self.channels as i32,
            numSamples: self.samples as i32,
            data: self.pointers.as_mut_ptr(),
        };
        &mut self.raw
    }

    fn channel(&self, i: usize) -> &[f32] {
        &self.data[self.offsets[i]..self.offsets[i] + self.samples]
    }

    fn channel_mut(&mut self, i: usize) -> &mut [f32] {
        let o = self.offsets[i];
        &mut self.data[o..o + self.samples]
    }

    fn fill(&mut self, lcg: &mut Lcg) {
        for i in 0..self.channels {
            for x in self.channel_mut(i) {
                *x = lcg.next_signed();
            }
        }
    }

    fn fill_value(&mut self, value: f32) {
        for i in 0..self.channels {
            self.channel_mut(i).fill(value);
        }
    }

    fn append(&self, out: &mut Vec<f32>) {
        for i in 0..self.channels {
            out.extend_from_slice(self.channel(i));
        }
    }
}

#[derive(Default)]
struct Recorder {
    outputs: Vec<f32>,
    states: Vec<i32>,
    peaks: Vec<f32>,
}

impl Recorder {
    fn record(&mut self, out: &Buffer, state: IPLAudioEffectState, tail_size: i32) {
        out.append(&mut self.outputs);
        self.states.extend([state, tail_size]);
    }

    #[track_caller]
    fn check(&self, name: &str) {
        self.check_within(name, SIGNAL);
    }

    #[track_caller]
    fn check_within(&self, name: &str, ratio: f64) {
        check_signal(name, &self.outputs, ratio);
        check_exact_i32(&format!("{name}_states"), &self.states);
        if !self.peaks.is_empty() {
            check(&format!("{name}_peaks"), &self.peaks, |_| Tolerance::F32);
        }
    }
}

const MAX_TAILS: usize = 4;

/// Records tail frames until the effect reports a complete tail (at most `MAX_TAILS`).
fn record_tails(
    recorder: &mut Recorder,
    out: &mut Buffer,
    mut tail: impl FnMut(*mut IPLAudioBuffer) -> IPLAudioEffectState,
    tail_size: impl Fn() -> i32,
) {
    for _ in 0..MAX_TAILS {
        out.fill_value(SENTINEL);
        let state = tail(out.ipl());
        recorder.record(out, state, tail_size());
        if state == IPL_AUDIOEFFECTSTATE_TAILCOMPLETE {
            break;
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Struct layouts
// --------------------------------------------------------------------------------------------------------------------

#[test]
#[cfg(target_pointer_width = "64")]
fn struct_layouts_match_phonon_h() {
    let mut layout: Vec<i32> = Vec::new();
    macro_rules! size {
        ($($t:ty),+) => { $(layout.push(size_of::<$t>() as i32);)+ };
    }
    macro_rules! field {
        ($t:ty, $($f:tt),+) => { $(layout.push(offset_of!($t, $f) as i32);)+ };
    }

    size!(
        IPLbool,
        IPLerror,
        IPLSIMDLevel,
        IPLContextFlags,
        IPLAudioEffectState
    );

    size!(IPLContextSettings);
    field!(
        IPLContextSettings,
        version,
        logCallback,
        allocateCallback,
        freeCallback,
        simdLevel,
        flags
    );
    size!(IPLVector3);
    field!(IPLVector3, x, y, z);
    size!(IPLMatrix4x4);
    field!(IPLMatrix4x4, elements);
    size!(IPLBox);
    field!(IPLBox, minCoordinates, maxCoordinates);
    size!(IPLSphere);
    field!(IPLSphere, center, radius);
    size!(IPLCoordinateSpace3);
    field!(IPLCoordinateSpace3, right, up, ahead, origin);
    size!(IPLSerializedObjectSettings);
    field!(IPLSerializedObjectSettings, data, size);
    size!(IPLSpeakerLayout);
    field!(IPLSpeakerLayout, r#type, numSpeakers, speakers);
    size!(IPLAudioSettings);
    field!(IPLAudioSettings, samplingRate, frameSize);
    size!(IPLAudioBuffer);
    field!(IPLAudioBuffer, numChannels, numSamples, data);
    size!(IPLHRTFSettings);
    field!(
        IPLHRTFSettings,
        r#type,
        sofaFileName,
        sofaData,
        sofaDataSize,
        volume,
        normType
    );
    size!(IPLPanningEffectSettings);
    field!(IPLPanningEffectSettings, speakerLayout);
    size!(IPLPanningEffectParams);
    field!(IPLPanningEffectParams, direction);
    size!(IPLBinauralEffectSettings);
    field!(IPLBinauralEffectSettings, hrtf);
    size!(IPLBinauralEffectParams);
    field!(
        IPLBinauralEffectParams,
        direction,
        interpolation,
        spatialBlend,
        hrtf,
        peakDelays
    );
    size!(IPLVirtualSurroundEffectSettings);
    field!(IPLVirtualSurroundEffectSettings, speakerLayout, hrtf);
    size!(IPLVirtualSurroundEffectParams);
    field!(IPLVirtualSurroundEffectParams, hrtf);
    size!(IPLAmbisonicsEncodeEffectSettings);
    field!(IPLAmbisonicsEncodeEffectSettings, maxOrder);
    size!(IPLAmbisonicsEncodeEffectParams);
    field!(IPLAmbisonicsEncodeEffectParams, direction, order);
    size!(IPLAmbisonicsPanningEffectSettings);
    field!(IPLAmbisonicsPanningEffectSettings, speakerLayout, maxOrder);
    size!(IPLAmbisonicsPanningEffectParams);
    field!(IPLAmbisonicsPanningEffectParams, order);
    size!(IPLAmbisonicsBinauralEffectSettings);
    field!(IPLAmbisonicsBinauralEffectSettings, hrtf, maxOrder);
    size!(IPLAmbisonicsBinauralEffectParams);
    field!(IPLAmbisonicsBinauralEffectParams, hrtf, order);
    size!(IPLAmbisonicsRotationEffectSettings);
    field!(IPLAmbisonicsRotationEffectSettings, maxOrder);
    size!(IPLAmbisonicsRotationEffectParams);
    field!(IPLAmbisonicsRotationEffectParams, orientation, order);
    size!(IPLAmbisonicsDecodeEffectSettings);
    field!(
        IPLAmbisonicsDecodeEffectSettings,
        speakerLayout,
        hrtf,
        maxOrder
    );
    size!(IPLAmbisonicsDecodeEffectParams);
    field!(
        IPLAmbisonicsDecodeEffectParams,
        order,
        hrtf,
        orientation,
        binaural
    );
    size!(IPLDirectEffectSettings);
    field!(IPLDirectEffectSettings, numChannels);
    size!(IPLDirectEffectParams);
    field!(
        IPLDirectEffectParams,
        flags,
        transmissionType,
        distanceAttenuation,
        airAbsorption,
        directivity,
        occlusion,
        transmission
    );
    size!(IPLDistanceAttenuationModel);
    field!(
        IPLDistanceAttenuationModel,
        r#type,
        minDistance,
        callback,
        userData,
        dirty
    );
    size!(IPLAirAbsorptionModel);
    field!(
        IPLAirAbsorptionModel,
        r#type,
        coefficients,
        callback,
        userData,
        dirty
    );
    size!(IPLDirectivity);
    field!(
        IPLDirectivity,
        dipoleWeight,
        dipolePower,
        callback,
        userData
    );

    check_exact_i32("layout", &layout);
}

// --------------------------------------------------------------------------------------------------------------------
// Context and relative direction
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn context_creation_checks_the_version() {
    let versions = [
        VERSION,
        VERSION_40,
        (4 << 16) | (8 << 8) | 255,
        (4 << 16) | (9 << 8),
        5 << 16,
        (3 << 16) | (8 << 8) | 1,
        (4 << 16) | (7 << 8),
        0,
    ];

    let mut statuses = Vec::new();
    for version in versions {
        let mut settings = IPLContextSettings {
            version,
            logCallback: None,
            allocateCallback: None,
            freeCallback: None,
            simdLevel: IPL_SIMDLEVEL_SSE2,
            flags: 0,
        };
        let mut context = ptr::null_mut();
        // SAFETY: valid pointers.
        statuses.push(unsafe { iplContextCreate(&mut settings, &mut context) });
        statuses.push(i32::from(!context.is_null()));
        // SAFETY: null or a live context.
        unsafe { iplContextRelease(&mut context) };
    }

    let mut context = ptr::null_mut();
    let mut settings = IPLContextSettings {
        version: VERSION,
        logCallback: None,
        allocateCallback: None,
        freeCallback: None,
        simdLevel: IPL_SIMDLEVEL_SSE2,
        flags: 0,
    };
    // SAFETY: null pointers are allowed.
    unsafe {
        statuses.push(iplContextCreate(ptr::null_mut(), &mut context));
        statuses.push(iplContextCreate(&mut settings, ptr::null_mut()));
    }
    check_exact_i32("context_status", &statuses);
}

#[test]
fn context_retain_and_release() {
    let context = make_context(VERSION);
    // SAFETY: a live context; every retained reference is released.
    unsafe {
        let mut retained = iplContextRetain(context);
        assert_eq!(retained, context);
        iplContextRelease(&mut retained);
        assert!(retained.is_null());

        // Objects keep their context alive.
        let mut hrtf = make_hrtf(context, default_hrtf_settings());
        release_context(context);
        let mut more = iplHRTFRetain(hrtf);
        iplHRTFRelease(&mut more);
        iplHRTFRelease(&mut hrtf);
        assert!(hrtf.is_null());

        // Null handles are ignored.
        assert!(iplContextRetain(ptr::null_mut()).is_null());
        iplContextRelease(ptr::null_mut());
        let mut null: IPLContext = ptr::null_mut();
        iplContextRelease(&mut null);
    }
}

#[test]
fn relative_direction() {
    let context = make_context(VERSION);
    let mut lcg = Lcg::new(801);
    let mut directions = Vec::new();
    let mut add = |source, listener, ahead, up| {
        // SAFETY: a live context.
        let d = unsafe { iplCalculateRelativeDirection(context, source, listener, ahead, up) };
        directions.extend([d.x, d.y, d.z]);
    };

    let v = |x, y, z| IPLVector3 { x, y, z };
    let (origin, ahead, up) = (v(0.0, 0.0, 0.0), v(0.0, 0.0, -1.0), v(0.0, 1.0, 0.0));
    add(v(1.0, 2.0, 3.0), v(1.0, 2.0, 3.0), ahead, up);
    add(v(1.0, 0.0, 0.0), origin, ahead, up);
    add(v(0.0, 0.0, -5.0), origin, ahead, up);
    add(v(0.0, 0.0, 4e-6), origin, ahead, up);
    for _ in 0..60 {
        let source = draw_vector(&mut lcg);
        let listener = draw_vector(&mut lcg);
        let listener_ahead = draw_vector(&mut lcg);
        let listener_up = draw_vector(&mut lcg);
        add(source, listener, listener_ahead, listener_up);
    }
    report(
        "relative_direction",
        &directions,
        &golden::<f32>("capi/relative_direction").data,
    );
    check("relative_direction", &directions, |_| Tolerance::F32);

    // SAFETY: a null context gives the zero vector.
    let d = unsafe { iplCalculateRelativeDirection(ptr::null_mut(), origin, origin, ahead, up) };
    assert_eq!((d.x, d.y, d.z), (0.0, 0.0, 0.0));
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Serialized objects
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn serialized_objects_wrap_or_start_empty() {
    let context = make_context(VERSION);
    let mut bytes = vec![1u8, 2, 3, 4, 5];
    // SAFETY: valid pointers; `bytes` outlives the object.
    unsafe {
        let mut settings = IPLSerializedObjectSettings {
            data: bytes.as_mut_ptr(),
            size: bytes.len(),
        };
        let mut object = ptr::null_mut();
        assert_eq!(
            iplSerializedObjectCreate(context, &mut settings, &mut object),
            IPL_STATUS_SUCCESS
        );
        assert_eq!(iplSerializedObjectGetSize(object), 5);
        assert_eq!(iplSerializedObjectGetData(object), bytes.as_mut_ptr());
        iplSerializedObjectRelease(&mut object);

        settings.size = 0;
        assert_eq!(
            iplSerializedObjectCreate(context, &mut settings, &mut object),
            IPL_STATUS_SUCCESS
        );
        assert_eq!(iplSerializedObjectGetSize(object), 0);
        assert!(iplSerializedObjectGetData(object).is_null());
        iplSerializedObjectRelease(&mut object);

        assert_eq!(
            iplSerializedObjectCreate(ptr::null_mut(), &mut settings, &mut object),
            IPL_STATUS_FAILURE
        );
        assert_eq!(
            iplSerializedObjectCreate(context, ptr::null_mut(), &mut object),
            IPL_STATUS_FAILURE
        );
        assert_eq!(iplSerializedObjectGetSize(ptr::null_mut()), 0);
        assert!(iplSerializedObjectGetData(ptr::null_mut()).is_null());
    }
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Audio buffers
// --------------------------------------------------------------------------------------------------------------------

const BUFFER_SAMPLES: usize = 37;

#[test]
fn audio_buffer_allocate_and_free() {
    let context = make_context(VERSION);
    // SAFETY: valid pointers; the buffer is used within its allocation and freed once.
    unsafe {
        let mut buffer = IPLAudioBuffer {
            numChannels: 0,
            numSamples: 0,
            data: ptr::null_mut(),
        };
        assert_eq!(
            iplAudioBufferAllocate(context, 3, 37, &mut buffer),
            IPL_STATUS_SUCCESS
        );
        assert_eq!((buffer.numChannels, buffer.numSamples), (3, 37));
        let channels: Vec<*mut f32> = (0..3).map(|i| *buffer.data.add(i)).collect();
        for i in 0..3 {
            assert_eq!(
                channels[i],
                channels[0].add(i * 37),
                "channels are contiguous"
            );
            let channel = std::slice::from_raw_parts_mut(channels[i], 37);
            assert!(channel.iter().all(|&x| x == 0.0));
            channel.fill(i as f32);
        }
        iplAudioBufferFree(context, &mut buffer);
        assert!(buffer.data.is_null());
        iplAudioBufferFree(context, &mut buffer);

        for (channels, samples) in [(0, 37), (3, 0), (-1, 37)] {
            assert_eq!(
                iplAudioBufferAllocate(context, channels, samples, &mut buffer),
                IPL_STATUS_FAILURE
            );
        }
        assert_eq!(
            iplAudioBufferAllocate(ptr::null_mut(), 1, 1, &mut buffer),
            IPL_STATUS_FAILURE
        );
        assert_eq!(
            iplAudioBufferAllocate(context, 1, 1, ptr::null_mut()),
            IPL_STATUS_FAILURE
        );
    }
    release_context(context);
}

fn audio_buffers(scattered: bool) {
    let context = make_context(VERSION);
    let mut lcg = Lcg::new(802);
    let n = BUFFER_SAMPLES;

    // SAFETY: a live context and valid buffers throughout.
    unsafe {
        let mut src = Buffer::new(3, n, scattered);
        src.fill(&mut lcg);
        let mut interleaved = vec![0.0f32; 3 * n];
        iplAudioBufferInterleave(context, src.ipl(), interleaved.as_mut_ptr());
        check("interleave", &interleaved, |_| Tolerance::EXACT);

        let mut interleaved: Vec<f32> = (0..3 * n).map(|_| lcg.next_signed()).collect();
        let mut dst = Buffer::new(3, n, scattered);
        iplAudioBufferDeinterleave(context, interleaved.as_mut_ptr(), dst.ipl());
        let mut result = Vec::new();
        dst.append(&mut result);
        check("deinterleave", &result, |_| Tolerance::EXACT);

        let mut input = Buffer::new(2, n, scattered);
        let mut mix = Buffer::new(2, n, scattered);
        input.fill(&mut lcg);
        mix.fill(&mut lcg);
        iplAudioBufferMix(context, input.ipl(), mix.ipl());
        let mix_ptr = mix.ipl();
        iplAudioBufferMix(context, mix_ptr, mix_ptr);
        let mut result = Vec::new();
        mix.append(&mut result);
        check("mix", &result, |_| Tolerance::EXACT);

        let mut input = Buffer::new(5, n, scattered);
        let mut mono = Buffer::new(1, n, scattered);
        input.fill(&mut lcg);
        iplAudioBufferDownmix(context, input.ipl(), mono.ipl());
        let mut result = Vec::new();
        mono.append(&mut result);
        check("downmix", &result, |_| Tolerance::F32);

        let mut converted = Vec::new();
        let mut converted_in_place = Vec::new();
        let mut convert = |in_type, out_type, channels| {
            let mut input = Buffer::new(channels, n, scattered);
            let mut out = Buffer::new(channels, n, scattered);
            input.fill(&mut lcg);
            out.fill_value(SENTINEL);
            iplAudioBufferConvertAmbisonics(context, in_type, out_type, input.ipl(), out.ipl());
            out.append(&mut converted);
            let input_ptr = input.ipl();
            iplAudioBufferConvertAmbisonics(context, in_type, out_type, input_ptr, input_ptr);
            input.append(&mut converted_in_place);
        };
        for channels in [4, 16] {
            for in_type in 0..3 {
                for out_type in 0..3 {
                    convert(in_type, out_type, channels);
                }
            }
        }
        convert(IPL_AMBISONICSTYPE_N3D, IPL_AMBISONICSTYPE_SN3D, 25);
        convert(IPL_AMBISONICSTYPE_SN3D, IPL_AMBISONICSTYPE_N3D, 25);
        report("convert", &converted, &golden::<f32>("capi/convert").data);
        check("convert", &converted, |_| Tolerance::F32);
        check("convert_in_place", &converted_in_place, |_| Tolerance::F32);
    }

    release_context(context);
}

#[test]
fn audio_buffers_contiguous() {
    audio_buffers(false);
}

#[test]
fn audio_buffers_scattered() {
    audio_buffers(true);
}

#[test]
fn audio_buffer_functions_ignore_invalid_buffers() {
    let context = make_context(VERSION);
    let mut a = Buffer::new(2, 8, false);
    let mut b = Buffer::new(3, 8, false);
    let mut fuma = Buffer::new(25, 8, false);
    let mut fuma_out = Buffer::new(25, 8, false);
    b.fill_value(1.0);
    fuma_out.fill_value(1.0);
    // SAFETY: valid or null pointers.
    unsafe {
        iplAudioBufferMix(context, a.ipl(), b.ipl());
        iplAudioBufferDownmix(context, a.ipl(), b.ipl());
        iplAudioBufferConvertAmbisonics(context, 0, 2, fuma.ipl(), fuma_out.ipl());
        iplAudioBufferConvertAmbisonics(context, 0, 7, a.ipl(), a.ipl());
        iplAudioBufferMix(context, ptr::null_mut(), b.ipl());
        iplAudioBufferInterleave(context, a.ipl(), ptr::null_mut());
        iplAudioBufferDeinterleave(context, ptr::null_mut(), a.ipl());
        let mut empty = IPLAudioBuffer {
            numChannels: 2,
            numSamples: 8,
            data: ptr::null_mut(),
        };
        iplAudioBufferMix(context, &mut empty, b.ipl());
    }
    assert!(b.data.iter().all(|&x| x == 1.0));
    assert!(fuma_out.data.iter().all(|&x| x == 1.0));
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// Direct sound models
// --------------------------------------------------------------------------------------------------------------------

unsafe extern "system" fn distance_callback(distance: f32, user_data: *mut c_void) -> f32 {
    // SAFETY: the tests pass a pointer to an `f32`.
    let k = unsafe { *user_data.cast::<f32>() };
    1.0 / (1.0 + k * distance * distance)
}

unsafe extern "system" fn air_absorption_callback(
    distance: f32,
    band: i32,
    user_data: *mut c_void,
) -> f32 {
    // SAFETY: the tests pass a pointer to an `f32`.
    let k = unsafe { *user_data.cast::<f32>() };
    1.0 / (1.0 + k * (band + 1) as f32 * distance)
}

unsafe extern "system" fn directivity_callback(
    direction: IPLVector3,
    user_data: *mut c_void,
) -> f32 {
    // SAFETY: the tests pass a pointer to an `f32`.
    let k = unsafe { *user_data.cast::<f32>() };
    k + direction.x * direction.z - 0.25 * direction.y
}

static K1: f32 = 0.3;
static K2: f32 = 0.05;
static K3: f32 = 0.5;

fn user_data(k: &'static f32) -> *mut c_void {
    ptr::from_ref(k).cast_mut().cast()
}

const NUM_MODEL_POINTS: usize = 50;

#[test]
fn direct_sound_models() {
    let context = make_context(VERSION);
    let mut lcg = Lcg::new(803);

    let mut sources = Vec::new();
    let mut listeners = Vec::new();
    let mut spaces = Vec::new();
    for _ in 0..NUM_MODEL_POINTS {
        let source = draw_vector(&mut lcg);
        let listener = draw_vector(&mut lcg);
        let scale = 1.0 + 20.0 * draw_unit(&mut lcg);
        sources.push(IPLVector3 {
            x: source.x * scale,
            y: source.y * scale,
            z: source.z * scale,
        });
        listeners.push(listener);
        let mut space = draw_orientation(&mut lcg);
        space.origin = *sources.last().unwrap();
        spaces.push(space);
    }

    let distance_model = |r#type, min_distance, callback: IPLDistanceAttenuationCallback, data| {
        IPLDistanceAttenuationModel {
            r#type,
            minDistance: min_distance,
            callback,
            userData: data,
            dirty: IPL_FALSE,
        }
    };
    let null = ptr::null_mut();
    let mut distance_models = [
        distance_model(IPL_DISTANCEATTENUATIONTYPE_DEFAULT, 0.0, None, null),
        distance_model(IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE, 0.5, None, null),
        distance_model(IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE, 0.0, None, null),
        distance_model(
            IPL_DISTANCEATTENUATIONTYPE_CALLBACK,
            0.0,
            Some(distance_callback),
            user_data(&K1),
        ),
        distance_model(IPL_DISTANCEATTENUATIONTYPE_CALLBACK, 0.0, None, null),
        distance_model(9, 3.0, None, null),
        distance_model(
            IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE,
            2.0,
            Some(distance_callback),
            user_data(&K2),
        ),
    ];
    let mut values = Vec::new();
    for model in &mut distance_models {
        for i in 0..NUM_MODEL_POINTS {
            // SAFETY: a live context and a valid model.
            values.push(unsafe {
                iplDistanceAttenuationCalculate(context, sources[i], listeners[i], model)
            });
        }
    }
    report(
        "distance_attenuation",
        &values,
        &golden::<f32>("capi/distance_attenuation").data,
    );
    check("distance_attenuation", &values, |_| Tolerance::F32);

    let air_model =
        |r#type, coefficients, callback: IPLAirAbsorptionCallback, data| IPLAirAbsorptionModel {
            r#type,
            coefficients,
            callback,
            userData: data,
            dirty: IPL_FALSE,
        };
    let exponential = [0.1, 0.2, 0.3];
    let mut air_models = [
        air_model(IPL_AIRABSORPTIONTYPE_DEFAULT, [0.0; 3], None, null),
        air_model(IPL_AIRABSORPTIONTYPE_EXPONENTIAL, exponential, None, null),
        air_model(
            IPL_AIRABSORPTIONTYPE_EXPONENTIAL,
            exponential,
            Some(air_absorption_callback),
            user_data(&K2),
        ),
        air_model(
            IPL_AIRABSORPTIONTYPE_CALLBACK,
            [0.0; 3],
            Some(air_absorption_callback),
            user_data(&K2),
        ),
        air_model(9, [0.5, 0.0, 0.0], None, null),
    ];
    let mut values = Vec::new();
    for model in &mut air_models {
        for i in 0..NUM_MODEL_POINTS {
            let mut bands = [0.0f32; 3];
            // SAFETY: a live context, a valid model and room for 3 bands.
            unsafe {
                iplAirAbsorptionCalculate(
                    context,
                    sources[i],
                    listeners[i],
                    model,
                    bands.as_mut_ptr(),
                )
            };
            values.extend(bands);
        }
    }
    report(
        "air_absorption",
        &values,
        &golden::<f32>("capi/air_absorption").data,
    );
    check("air_absorption", &values, |_| Tolerance::F32);

    let directivity = |weight, power, callback: IPLDirectivityCallback, data| IPLDirectivity {
        dipoleWeight: weight,
        dipolePower: power,
        callback,
        userData: data,
    };
    let mut directivity_models = [
        directivity(0.0, 0.0, None, null),
        directivity(0.5, 1.0, None, null),
        directivity(0.3, 2.5, None, null),
        directivity(0.7, 0.0, Some(directivity_callback), user_data(&K3)),
        directivity(0.0, 0.0, Some(directivity_callback), user_data(&K1)),
    ];
    let mut values = Vec::new();
    for model in &mut directivity_models {
        for i in 0..NUM_MODEL_POINTS {
            // SAFETY: a live context and a valid model.
            values
                .push(unsafe { iplDirectivityCalculate(context, spaces[i], listeners[i], model) });
        }
    }
    report(
        "directivity",
        &values,
        &golden::<f32>("capi/directivity").data,
    );
    check("directivity", &values, |_| Tolerance::F32);

    // SAFETY: null models and contexts give neutral values.
    unsafe {
        let (s, l) = (sources[0], listeners[0]);
        assert_eq!(
            iplDistanceAttenuationCalculate(context, s, l, ptr::null_mut()),
            1.0
        );
        assert_eq!(
            iplDirectivityCalculate(context, spaces[0], l, ptr::null_mut()),
            1.0
        );
        assert_eq!(
            iplDistanceAttenuationCalculate(ptr::null_mut(), s, l, &mut distance_models[1]),
            1.0
        );
        let mut bands = [SENTINEL; 3];
        iplAirAbsorptionCalculate(context, s, l, ptr::null_mut(), bands.as_mut_ptr());
        assert_eq!(bands, [SENTINEL; 3]);
    }
    release_context(context);
}

// --------------------------------------------------------------------------------------------------------------------
// HRTFs
// --------------------------------------------------------------------------------------------------------------------

// Without SOFA support, SOFA settings fail with other statuses.
#[test]
#[cfg(feature = "sofa")]
fn hrtf_creation_statuses() {
    let mut lcg = Lcg::new(804);
    let garbage: Vec<u8> = (0..64).map(|_| (lcg.next_u32() >> 24) as u8).collect();
    let missing = CString::new(data_path("hrtf/missing.sofa").to_str().unwrap()).unwrap();

    let mut statuses = Vec::new();
    let create = |context, mut audio: IPLAudioSettings, mut settings: IPLHRTFSettings| {
        let mut hrtf = ptr::null_mut();
        // SAFETY: valid pointers.
        unsafe {
            let status = iplHRTFCreate(context, &mut audio, &mut settings, &mut hrtf);
            let created = i32::from(!hrtf.is_null());
            iplHRTFRelease(&mut hrtf);
            [status, created]
        }
    };

    let context = make_context(VERSION);
    let audio = audio_settings();
    let sofa = IPLHRTFSettings {
        r#type: IPL_HRTFTYPE_SOFA,
        volume: 0.0,
        ..default_hrtf_settings()
    };
    statuses.extend(create(context, audio, sofa));
    statuses.extend(create(
        context,
        audio,
        IPLHRTFSettings {
            sofaData: garbage.as_ptr(),
            ..sofa
        },
    ));
    statuses.extend(create(
        context,
        audio,
        IPLHRTFSettings {
            r#type: 7,
            volume: 0.0,
            ..default_hrtf_settings()
        },
    ));
    statuses.extend(create(
        context,
        audio,
        IPLHRTFSettings {
            sofaFileName: missing.as_ptr(),
            ..sofa
        },
    ));
    statuses.extend(create(
        context,
        audio,
        IPLHRTFSettings {
            sofaData: garbage.as_ptr(),
            sofaDataSize: garbage.len() as i32,
            ..sofa
        },
    ));
    let default = default_hrtf_settings();
    statuses.extend(create(
        context,
        IPLAudioSettings {
            samplingRate: 0,
            ..audio
        },
        default,
    ));
    statuses.extend(create(
        context,
        IPLAudioSettings {
            frameSize: 0,
            ..audio
        },
        default,
    ));
    statuses.extend(create(
        context,
        IPLAudioSettings {
            samplingRate: 8000,
            ..audio
        },
        default,
    ));
    statuses.extend(create(context, audio, default));

    let mut hrtf = ptr::null_mut();
    let (mut audio_copy, mut settings_copy) = (audio, default);
    // SAFETY: null pointers are allowed.
    unsafe {
        statuses.push(iplHRTFCreate(
            context,
            &mut audio_copy,
            ptr::null_mut(),
            &mut hrtf,
        ));
        statuses.push(iplHRTFCreate(
            context,
            ptr::null_mut(),
            &mut settings_copy,
            &mut hrtf,
        ));
        statuses.push(iplHRTFCreate(
            ptr::null_mut(),
            &mut audio_copy,
            &mut settings_copy,
            &mut hrtf,
        ));
    }
    release_context(context);

    let context = make_context(VERSION_41);
    statuses.extend(create(
        context,
        audio,
        IPLHRTFSettings {
            sofaData: garbage.as_ptr(),
            sofaDataSize: garbage.len() as i32,
            ..sofa
        },
    ));
    release_context(context);

    check_exact_i32("hrtf_status", &statuses);
}

/// Nearest-neighbour binaural rendering of a few frames, recording outputs and peak delays.
#[cfg(feature = "sofa")]
fn render_with_hrtf(
    context: IPLContext,
    hrtf: IPLHRTF,
    lcg: &mut Lcg,
    recorder: &mut Recorder,
    scattered: bool,
) {
    let mut audio = audio_settings();
    let mut settings = IPLBinauralEffectSettings { hrtf };
    let mut effect = ptr::null_mut();
    // SAFETY: valid pointers and live objects.
    unsafe {
        assert_eq!(
            iplBinauralEffectCreate(context, &mut audio, &mut settings, &mut effect),
            IPL_STATUS_SUCCESS
        );
        let mut input = Buffer::new(1, FRAME_SIZE, scattered);
        let mut out = Buffer::new(2, FRAME_SIZE, scattered);
        for _ in 0..3 {
            input.fill(lcg);
            let mut peaks = [SENTINEL; 2];
            let mut params = IPLBinauralEffectParams {
                direction: draw_vector(lcg),
                interpolation: IPL_HRTFINTERPOLATION_NEAREST,
                spatialBlend: 1.0,
                hrtf,
                peakDelays: peaks.as_mut_ptr(),
            };
            out.fill_value(SENTINEL);
            let state = iplBinauralEffectApply(effect, &mut params, input.ipl(), out.ipl());
            recorder.record(&out, state, iplBinauralEffectGetTailSize(effect));
            recorder.peaks.extend(peaks);
        }
        iplBinauralEffectRelease(&mut effect);
    }
}

#[cfg(feature = "sofa")]
fn hrtfs(scattered: bool) {
    let mut lcg = Lcg::new(805);
    let mut run = |version, settings: IPLHRTFSettings, name: &str| {
        let context = make_context(version);
        let mut hrtf = make_hrtf(context, settings);
        let mut recorder = Recorder::default();
        render_with_hrtf(context, hrtf, &mut lcg, &mut recorder, scattered);
        recorder.check(name);
        // SAFETY: a live HRTF.
        unsafe { iplHRTFRelease(&mut hrtf) };
        release_context(context);
    };

    let file_name = CString::new(data_path("hrtf/cipic_124.sofa").to_str().unwrap()).unwrap();
    let file = IPLHRTFSettings {
        r#type: IPL_HRTFTYPE_SOFA,
        sofaFileName: file_name.as_ptr(),
        ..default_hrtf_settings()
    };
    run(VERSION, file, "hrtf_sofa_file");

    let data = std::fs::read(data_path("hrtf/sadie_h12.sofa")).unwrap();
    let memory = IPLHRTFSettings {
        r#type: IPL_HRTFTYPE_SOFA,
        sofaData: data.as_ptr(),
        sofaDataSize: data.len() as i32,
        volume: 0.5,
        normType: IPL_HRTFNORMTYPE_RMS,
        ..default_hrtf_settings()
    };
    run(VERSION, memory, "hrtf_sofa_data");

    let loud = IPLHRTFSettings {
        volume: 2.0,
        normType: IPL_HRTFNORMTYPE_RMS,
        ..default_hrtf_settings()
    };
    run(VERSION, loud, "hrtf_default_loud");
    run(VERSION_41, loud, "hrtf_default_loud_v41");
    run(VERSION_40, loud, "hrtf_default_loud_v40");

    let both = IPLHRTFSettings {
        sofaData: data.as_ptr(),
        sofaDataSize: data.len() as i32,
        ..file
    };
    run(VERSION, both, "hrtf_sofa_both");
}

#[test]
#[cfg(feature = "sofa")]
fn hrtfs_contiguous() {
    hrtfs(false);
}

#[test]
#[cfg(feature = "sofa")]
fn hrtfs_scattered() {
    hrtfs(true);
}

// --------------------------------------------------------------------------------------------------------------------
// Effects
// --------------------------------------------------------------------------------------------------------------------

static mut CUSTOM_SPEAKERS: [IPLVector3; 5] = [
    IPLVector3 {
        x: -1.0,
        y: 0.0,
        z: -1.0,
    },
    IPLVector3 {
        x: 1.0,
        y: 0.0,
        z: -1.0,
    },
    IPLVector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    },
    IPLVector3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    },
    IPLVector3 {
        x: 0.3,
        y: -0.5,
        z: 0.8,
    },
];

fn make_layout(r#type: IPLSpeakerLayoutType) -> IPLSpeakerLayout {
    if r#type == IPL_SPEAKERLAYOUTTYPE_CUSTOM {
        IPLSpeakerLayout {
            r#type,
            numSpeakers: 5,
            speakers: (&raw mut CUSTOM_SPEAKERS).cast(),
        }
    } else {
        IPLSpeakerLayout {
            r#type,
            numSpeakers: 0,
            speakers: ptr::null_mut(),
        }
    }
}

fn num_speakers(r#type: IPLSpeakerLayoutType) -> usize {
    match r#type {
        IPL_SPEAKERLAYOUTTYPE_MONO => 1,
        IPL_SPEAKERLAYOUTTYPE_STEREO => 2,
        IPL_SPEAKERLAYOUTTYPE_QUADRAPHONIC => 4,
        IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1 => 6,
        IPL_SPEAKERLAYOUTTYPE_SURROUND_7_1 => 8,
        _ => 5,
    }
}

fn num_coeffs(order: i32) -> usize {
    ((order + 1) * (order + 1)) as usize
}

const FRAMES: usize = 3;

/// The effect cases of the generator, in its order, sharing one `Lcg`.
struct Effects {
    context: IPLContext,
    hrtf: IPLHRTF,
    lcg: Lcg,
    scattered: bool,
}

impl Effects {
    fn buffer(&self, channels: usize) -> Buffer {
        Buffer::new(channels, FRAME_SIZE, self.scattered)
    }

    /// # Safety
    ///
    /// The context and HRTF are live.
    unsafe fn panning(&mut self) {
        let cases = [
            ("panning_stereo", IPL_SPEAKERLAYOUTTYPE_STEREO),
            ("panning_surround51", IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1),
            ("panning_custom", IPL_SPEAKERLAYOUTTYPE_CUSTOM),
        ];
        for (name, layout) in cases {
            let mut audio = audio_settings();
            let mut settings = IPLPanningEffectSettings {
                speakerLayout: make_layout(layout),
            };
            let mut effect = ptr::null_mut();
            // SAFETY: valid pointers and live objects.
            unsafe {
                assert_eq!(
                    iplPanningEffectCreate(self.context, &mut audio, &mut settings, &mut effect),
                    IPL_STATUS_SUCCESS
                );
                let mut input = self.buffer(1);
                let mut out = self.buffer(num_speakers(layout));
                let mut recorder = Recorder::default();
                let mut apply = |lcg: &mut Lcg, recorder: &mut Recorder| {
                    input.fill(lcg);
                    let mut params = IPLPanningEffectParams {
                        direction: draw_vector(lcg),
                    };
                    out.fill_value(SENTINEL);
                    let state = iplPanningEffectApply(effect, &mut params, input.ipl(), out.ipl());
                    recorder.record(&out, state, iplPanningEffectGetTailSize(effect));
                };
                for _ in 0..FRAMES {
                    apply(&mut self.lcg, &mut recorder);
                }
                iplPanningEffectReset(effect);
                apply(&mut self.lcg, &mut recorder);
                record_tails(
                    &mut recorder,
                    &mut out,
                    |out| iplPanningEffectGetTail(effect, out),
                    || iplPanningEffectGetTailSize(effect),
                );
                recorder.check(name);
                iplPanningEffectRelease(&mut effect);
            }
        }
    }

    /// # Safety
    ///
    /// The context and HRTF are live.
    unsafe fn binaural(&mut self) {
        let cases = [
            (
                "binaural_mono_nearest",
                1,
                IPL_HRTFINTERPOLATION_NEAREST,
                false,
            ),
            (
                "binaural_mono_bilinear",
                1,
                IPL_HRTFINTERPOLATION_BILINEAR,
                false,
            ),
            (
                "binaural_stereo_nearest_blend",
                2,
                IPL_HRTFINTERPOLATION_NEAREST,
                true,
            ),
            (
                "binaural_stereo_bilinear_blend",
                2,
                IPL_HRTFINTERPOLATION_BILINEAR,
                true,
            ),
        ];
        let hrtf = self.hrtf;
        for (name, channels, interpolation, draw_blend) in cases {
            let mut audio = audio_settings();
            let mut settings = IPLBinauralEffectSettings { hrtf };
            let mut effect = ptr::null_mut();
            // SAFETY: valid pointers and live objects.
            unsafe {
                assert_eq!(
                    iplBinauralEffectCreate(self.context, &mut audio, &mut settings, &mut effect),
                    IPL_STATUS_SUCCESS
                );
                let mut input = self.buffer(channels);
                let mut out = self.buffer(2);
                let mut recorder = Recorder::default();
                let mut apply = |lcg: &mut Lcg, recorder: &mut Recorder| {
                    input.fill(lcg);
                    let direction = draw_vector(lcg);
                    let blend = if draw_blend { draw_unit(lcg) } else { 1.0 };
                    let mut peaks = [SENTINEL; 2];
                    let mut params = IPLBinauralEffectParams {
                        direction,
                        interpolation,
                        spatialBlend: blend,
                        hrtf,
                        peakDelays: peaks.as_mut_ptr(),
                    };
                    out.fill_value(SENTINEL);
                    let state = iplBinauralEffectApply(effect, &mut params, input.ipl(), out.ipl());
                    recorder.record(&out, state, iplBinauralEffectGetTailSize(effect));
                    recorder.peaks.extend(peaks);
                };
                for _ in 0..FRAMES {
                    apply(&mut self.lcg, &mut recorder);
                }
                iplBinauralEffectReset(effect);
                apply(&mut self.lcg, &mut recorder);
                record_tails(
                    &mut recorder,
                    &mut out,
                    |out| iplBinauralEffectGetTail(effect, out),
                    || iplBinauralEffectGetTailSize(effect),
                );
                let synthesized = interpolation == IPL_HRTFINTERPOLATION_BILINEAR || draw_blend;
                recorder.check_within(name, if synthesized { SYNTHESIZED } else { SIGNAL });
                iplBinauralEffectRelease(&mut effect);
            }
        }
    }

    /// # Safety
    ///
    /// The context and HRTF are live.
    unsafe fn virtual_surround(&mut self) {
        let cases = [
            (
                "virtual_surround_surround51",
                IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1,
            ),
            ("virtual_surround_custom", IPL_SPEAKERLAYOUTTYPE_CUSTOM),
        ];
        let hrtf = self.hrtf;
        for (name, layout) in cases {
            let mut audio = audio_settings();
            let mut settings = IPLVirtualSurroundEffectSettings {
                speakerLayout: make_layout(layout),
                hrtf,
            };
            let mut effect = ptr::null_mut();
            // SAFETY: valid pointers and live objects.
            unsafe {
                assert_eq!(
                    iplVirtualSurroundEffectCreate(
                        self.context,
                        &mut audio,
                        &mut settings,
                        &mut effect
                    ),
                    IPL_STATUS_SUCCESS
                );
                let mut input = self.buffer(num_speakers(layout));
                let mut out = self.buffer(2);
                let mut recorder = Recorder::default();
                let mut apply = |lcg: &mut Lcg, recorder: &mut Recorder| {
                    input.fill(lcg);
                    let mut params = IPLVirtualSurroundEffectParams { hrtf };
                    out.fill_value(SENTINEL);
                    let state =
                        iplVirtualSurroundEffectApply(effect, &mut params, input.ipl(), out.ipl());
                    recorder.record(&out, state, iplVirtualSurroundEffectGetTailSize(effect));
                };
                for _ in 0..FRAMES {
                    apply(&mut self.lcg, &mut recorder);
                }
                iplVirtualSurroundEffectReset(effect);
                apply(&mut self.lcg, &mut recorder);
                record_tails(
                    &mut recorder,
                    &mut out,
                    |out| iplVirtualSurroundEffectGetTail(effect, out),
                    || iplVirtualSurroundEffectGetTailSize(effect),
                );
                recorder.check(name);
                iplVirtualSurroundEffectRelease(&mut effect);
            }
        }
    }

    /// # Safety
    ///
    /// The context is live.
    unsafe fn ambisonics_encode(&mut self) {
        let mut audio = audio_settings();
        let mut settings = IPLAmbisonicsEncodeEffectSettings { maxOrder: 2 };
        let mut effect = ptr::null_mut();
        let scattered = self.scattered;
        // SAFETY: valid pointers and live objects.
        unsafe {
            assert_eq!(
                iplAmbisonicsEncodeEffectCreate(
                    self.context,
                    &mut audio,
                    &mut settings,
                    &mut effect
                ),
                IPL_STATUS_SUCCESS
            );
            let mut input = self.buffer(1);
            let mut recorder = Recorder::default();
            let mut apply = |lcg: &mut Lcg, recorder: &mut Recorder, order: i32| {
                let mut out = Buffer::new(num_coeffs(order), FRAME_SIZE, scattered);
                input.fill(lcg);
                let mut params = IPLAmbisonicsEncodeEffectParams {
                    direction: draw_vector(lcg),
                    order,
                };
                out.fill_value(SENTINEL);
                let state =
                    iplAmbisonicsEncodeEffectApply(effect, &mut params, input.ipl(), out.ipl());
                recorder.record(&out, state, iplAmbisonicsEncodeEffectGetTailSize(effect));
            };
            for order in [2, 2, 1, 3] {
                apply(&mut self.lcg, &mut recorder, order);
            }
            iplAmbisonicsEncodeEffectReset(effect);
            apply(&mut self.lcg, &mut recorder, 2);
            let mut tail = self.buffer(num_coeffs(2));
            record_tails(
                &mut recorder,
                &mut tail,
                |out| iplAmbisonicsEncodeEffectGetTail(effect, out),
                || iplAmbisonicsEncodeEffectGetTailSize(effect),
            );
            recorder.check("ambisonics_encode");
            iplAmbisonicsEncodeEffectRelease(&mut effect);
        }
    }

    /// # Safety
    ///
    /// The context is live.
    unsafe fn ambisonics_panning(&mut self) {
        let mut audio = audio_settings();
        let mut settings = IPLAmbisonicsPanningEffectSettings {
            speakerLayout: make_layout(IPL_SPEAKERLAYOUTTYPE_SURROUND_7_1),
            maxOrder: 2,
        };
        let mut effect = ptr::null_mut();
        let scattered = self.scattered;
        // SAFETY: valid pointers and live objects.
        unsafe {
            assert_eq!(
                iplAmbisonicsPanningEffectCreate(
                    self.context,
                    &mut audio,
                    &mut settings,
                    &mut effect
                ),
                IPL_STATUS_SUCCESS
            );
            let mut out = self.buffer(8);
            let mut recorder = Recorder::default();
            let mut apply = |lcg: &mut Lcg, recorder: &mut Recorder, order: i32| {
                let mut input = Buffer::new(num_coeffs(order), FRAME_SIZE, scattered);
                input.fill(lcg);
                let mut params = IPLAmbisonicsPanningEffectParams { order };
                out.fill_value(SENTINEL);
                let state =
                    iplAmbisonicsPanningEffectApply(effect, &mut params, input.ipl(), out.ipl());
                recorder.record(&out, state, iplAmbisonicsPanningEffectGetTailSize(effect));
            };
            for order in [2, 1, 0] {
                apply(&mut self.lcg, &mut recorder, order);
            }
            iplAmbisonicsPanningEffectReset(effect);
            apply(&mut self.lcg, &mut recorder, 2);
            record_tails(
                &mut recorder,
                &mut out,
                |out| iplAmbisonicsPanningEffectGetTail(effect, out),
                || iplAmbisonicsPanningEffectGetTailSize(effect),
            );
            recorder.check("ambisonics_panning");
            iplAmbisonicsPanningEffectRelease(&mut effect);
        }
    }

    /// # Safety
    ///
    /// The context and HRTF are live.
    unsafe fn ambisonics_binaural(&mut self) {
        let hrtf = self.hrtf;
        let mut audio = audio_settings();
        let mut settings = IPLAmbisonicsBinauralEffectSettings { hrtf, maxOrder: 3 };
        let mut effect = ptr::null_mut();
        let scattered = self.scattered;
        // SAFETY: valid pointers and live objects.
        unsafe {
            assert_eq!(
                iplAmbisonicsBinauralEffectCreate(
                    self.context,
                    &mut audio,
                    &mut settings,
                    &mut effect
                ),
                IPL_STATUS_SUCCESS
            );
            let mut out = self.buffer(2);
            let mut recorder = Recorder::default();
            let mut apply = |lcg: &mut Lcg, recorder: &mut Recorder, order: i32| {
                let mut input = Buffer::new(num_coeffs(order), FRAME_SIZE, scattered);
                input.fill(lcg);
                let mut params = IPLAmbisonicsBinauralEffectParams { hrtf, order };
                out.fill_value(SENTINEL);
                let state =
                    iplAmbisonicsBinauralEffectApply(effect, &mut params, input.ipl(), out.ipl());
                recorder.record(&out, state, iplAmbisonicsBinauralEffectGetTailSize(effect));
            };
            for order in [3, 1, 2] {
                apply(&mut self.lcg, &mut recorder, order);
            }
            iplAmbisonicsBinauralEffectReset(effect);
            apply(&mut self.lcg, &mut recorder, 3);
            record_tails(
                &mut recorder,
                &mut out,
                |out| iplAmbisonicsBinauralEffectGetTail(effect, out),
                || iplAmbisonicsBinauralEffectGetTailSize(effect),
            );
            recorder.check("ambisonics_binaural");
            iplAmbisonicsBinauralEffectRelease(&mut effect);
        }
    }

    /// # Safety
    ///
    /// The context is live.
    unsafe fn ambisonics_rotation(&mut self) {
        let mut audio = audio_settings();
        let mut settings = IPLAmbisonicsRotationEffectSettings { maxOrder: 2 };
        let mut effect = ptr::null_mut();
        let scattered = self.scattered;
        // SAFETY: valid pointers and live objects.
        unsafe {
            assert_eq!(
                iplAmbisonicsRotationEffectCreate(
                    self.context,
                    &mut audio,
                    &mut settings,
                    &mut effect
                ),
                IPL_STATUS_SUCCESS
            );
            let mut recorder = Recorder::default();
            let apply = |lcg: &mut Lcg, recorder: &mut Recorder, order: i32| {
                let mut input = Buffer::new(num_coeffs(order), FRAME_SIZE, scattered);
                let mut out = Buffer::new(num_coeffs(order), FRAME_SIZE, scattered);
                input.fill(lcg);
                let mut params = IPLAmbisonicsRotationEffectParams {
                    orientation: draw_orientation(lcg),
                    order,
                };
                out.fill_value(SENTINEL);
                let state =
                    iplAmbisonicsRotationEffectApply(effect, &mut params, input.ipl(), out.ipl());
                recorder.record(&out, state, iplAmbisonicsRotationEffectGetTailSize(effect));
            };
            for order in [2, 1, 3] {
                apply(&mut self.lcg, &mut recorder, order);
            }
            iplAmbisonicsRotationEffectReset(effect);
            apply(&mut self.lcg, &mut recorder, 2);
            let mut tail = self.buffer(num_coeffs(2));
            record_tails(
                &mut recorder,
                &mut tail,
                |out| iplAmbisonicsRotationEffectGetTail(effect, out),
                || iplAmbisonicsRotationEffectGetTailSize(effect),
            );
            recorder.check("ambisonics_rotation");
            iplAmbisonicsRotationEffectRelease(&mut effect);
        }
    }

    /// Runs a decode case: `ops` are (order, binaural, pass the HRTF); the first is repeated after a reset.
    ///
    /// # Safety
    ///
    /// The context and HRTF are live.
    unsafe fn decode(
        &mut self,
        name: &str,
        layout: IPLSpeakerLayoutType,
        max_order: i32,
        create_with_hrtf: bool,
        ops: &[(i32, IPLbool, bool)],
    ) {
        let hrtf = self.hrtf;
        let mut audio = audio_settings();
        let mut settings = IPLAmbisonicsDecodeEffectSettings {
            speakerLayout: make_layout(layout),
            hrtf: if create_with_hrtf {
                hrtf
            } else {
                ptr::null_mut()
            },
            maxOrder: max_order,
        };
        let mut effect = ptr::null_mut();
        let scattered = self.scattered;
        // SAFETY: valid pointers and live objects.
        unsafe {
            assert_eq!(
                iplAmbisonicsDecodeEffectCreate(
                    self.context,
                    &mut audio,
                    &mut settings,
                    &mut effect
                ),
                IPL_STATUS_SUCCESS
            );
            let mut out = self.buffer(num_speakers(layout));
            let mut recorder = Recorder::default();
            let mut apply =
                |lcg: &mut Lcg,
                 recorder: &mut Recorder,
                 (order, binaural, with_hrtf): (i32, IPLbool, bool)| {
                    let mut input = Buffer::new(num_coeffs(order), FRAME_SIZE, scattered);
                    input.fill(lcg);
                    let mut params = IPLAmbisonicsDecodeEffectParams {
                        order,
                        hrtf: if with_hrtf { hrtf } else { ptr::null_mut() },
                        orientation: draw_orientation(lcg),
                        binaural,
                    };
                    out.fill_value(SENTINEL);
                    let state =
                        iplAmbisonicsDecodeEffectApply(effect, &mut params, input.ipl(), out.ipl());
                    recorder.record(&out, state, iplAmbisonicsDecodeEffectGetTailSize(effect));
                };
            for &op in ops {
                apply(&mut self.lcg, &mut recorder, op);
            }
            iplAmbisonicsDecodeEffectReset(effect);
            apply(&mut self.lcg, &mut recorder, ops[0]);
            record_tails(
                &mut recorder,
                &mut out,
                |out| iplAmbisonicsDecodeEffectGetTail(effect, out),
                || iplAmbisonicsDecodeEffectGetTailSize(effect),
            );
            recorder.check(name);
            iplAmbisonicsDecodeEffectRelease(&mut effect);
        }
    }

    /// # Safety
    ///
    /// The context is live.
    unsafe fn direct(&mut self) {
        let mut audio = audio_settings();
        let mut settings = IPLDirectEffectSettings { numChannels: 2 };
        let mut effect = ptr::null_mut();
        let transmission_types = [
            IPL_TRANSMISSIONTYPE_FREQINDEPENDENT,
            IPL_TRANSMISSIONTYPE_FREQDEPENDENT,
            7,
        ];
        // SAFETY: valid pointers and live objects.
        unsafe {
            assert_eq!(
                iplDirectEffectCreate(self.context, &mut audio, &mut settings, &mut effect),
                IPL_STATUS_SUCCESS
            );
            let mut input = self.buffer(2);
            let mut out = self.buffer(2);
            let mut recorder = Recorder::default();
            let mut apply =
                |lcg: &mut Lcg, recorder: &mut Recorder, frame: usize, in_place: bool| {
                    input.fill(lcg);
                    let flags = (lcg.next_u32() >> 27) as i32;
                    let distance_attenuation = draw_unit(lcg);
                    let air_absorption = [draw_unit(lcg), draw_unit(lcg), draw_unit(lcg)];
                    let directivity = draw_unit(lcg);
                    let occlusion = draw_unit(lcg);
                    let transmission = [draw_unit(lcg), draw_unit(lcg), draw_unit(lcg)];
                    let mut params = IPLDirectEffectParams {
                        flags,
                        transmissionType: transmission_types[frame % 3],
                        distanceAttenuation: distance_attenuation,
                        airAbsorption: air_absorption,
                        directivity,
                        occlusion,
                        transmission,
                    };
                    if in_place {
                        let p = input.ipl();
                        let state = iplDirectEffectApply(effect, &mut params, p, p);
                        recorder.record(&input, state, iplDirectEffectGetTailSize(effect));
                    } else {
                        out.fill_value(SENTINEL);
                        let state =
                            iplDirectEffectApply(effect, &mut params, input.ipl(), out.ipl());
                        recorder.record(&out, state, iplDirectEffectGetTailSize(effect));
                    }
                };
            for i in 0..9 {
                apply(&mut self.lcg, &mut recorder, i, i % 4 == 3);
            }
            iplDirectEffectReset(effect);
            apply(&mut self.lcg, &mut recorder, 0, false);
            record_tails(
                &mut recorder,
                &mut out,
                |out| iplDirectEffectGetTail(effect, out),
                || iplDirectEffectGetTailSize(effect),
            );
            recorder.check("direct");
            iplDirectEffectRelease(&mut effect);
        }
    }
}

fn effects(scattered: bool) {
    let context = make_context(VERSION);
    let mut hrtf = make_hrtf(context, default_hrtf_settings());
    let mut effects = Effects {
        context,
        hrtf,
        lcg: Lcg::new(806),
        scattered,
    };
    // SAFETY: the context and HRTF stay live until the end.
    unsafe {
        effects.panning();
        effects.binaural();
        effects.virtual_surround();
        effects.ambisonics_encode();
        effects.ambisonics_panning();
        effects.ambisonics_binaural();
        effects.ambisonics_rotation();
        effects.decode(
            "ambisonics_decode_binaural",
            IPL_SPEAKERLAYOUTTYPE_STEREO,
            2,
            true,
            &[
                (2, IPL_TRUE, true),
                (1, IPL_TRUE, true),
                (2, IPL_FALSE, true),
                (2, IPL_TRUE, false),
                (2, IPL_TRUE, true),
            ],
        );
        effects.decode(
            "ambisonics_decode_panning",
            IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1,
            3,
            false,
            &[
                (3, IPL_FALSE, false),
                (2, IPL_FALSE, false),
                (3, IPL_TRUE, true),
                (1, IPL_FALSE, true),
            ],
        );
        effects.direct();
        iplHRTFRelease(&mut hrtf);
    }
    release_context(context);
}

#[test]
fn effects_contiguous() {
    effects(false);
}

#[test]
fn effects_scattered() {
    effects(true);
}

// --------------------------------------------------------------------------------------------------------------------
// Invalid calls
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn effects_reject_invalid_settings_and_buffers() {
    let context = make_context(VERSION);
    let mut hrtf = make_hrtf(context, default_hrtf_settings());
    let mut audio = audio_settings();
    // SAFETY: valid or null pointers and live objects.
    unsafe {
        // Creation.
        let mut panning = ptr::null_mut();
        let mut bad_layout = IPLPanningEffectSettings {
            speakerLayout: IPLSpeakerLayout {
                r#type: IPL_SPEAKERLAYOUTTYPE_CUSTOM,
                numSpeakers: 3,
                speakers: ptr::null_mut(),
            },
        };
        assert_eq!(
            iplPanningEffectCreate(context, &mut audio, &mut bad_layout, &mut panning),
            IPL_STATUS_FAILURE
        );
        bad_layout.speakerLayout.r#type = 42;
        assert_eq!(
            iplPanningEffectCreate(context, &mut audio, &mut bad_layout, &mut panning),
            IPL_STATUS_FAILURE
        );
        let mut binaural = ptr::null_mut();
        let mut no_hrtf = IPLBinauralEffectSettings {
            hrtf: ptr::null_mut(),
        };
        assert_eq!(
            iplBinauralEffectCreate(context, &mut audio, &mut no_hrtf, &mut binaural),
            IPL_STATUS_FAILURE
        );
        let mut ambisonics_binaural = ptr::null_mut();
        let mut order_4 = IPLAmbisonicsBinauralEffectSettings { hrtf, maxOrder: 4 };
        assert_eq!(
            iplAmbisonicsBinauralEffectCreate(
                context,
                &mut audio,
                &mut order_4,
                &mut ambisonics_binaural
            ),
            IPL_STATUS_FAILURE
        );
        let mut encode = ptr::null_mut();
        let mut negative = IPLAmbisonicsEncodeEffectSettings { maxOrder: -1 };
        assert_eq!(
            iplAmbisonicsEncodeEffectCreate(context, &mut audio, &mut negative, &mut encode),
            IPL_STATUS_FAILURE
        );
        let mut zero_rate = IPLAudioSettings {
            samplingRate: 0,
            frameSize: 64,
        };
        let mut direct = ptr::null_mut();
        let mut direct_settings = IPLDirectEffectSettings { numChannels: 1 };
        assert_eq!(
            iplDirectEffectCreate(context, &mut zero_rate, &mut direct_settings, &mut direct),
            IPL_STATUS_FAILURE
        );
        assert_eq!(
            iplDirectEffectCreate(
                ptr::null_mut(),
                &mut audio,
                &mut direct_settings,
                &mut direct
            ),
            IPL_STATUS_FAILURE
        );

        // Application with buffers of the wrong shape leaves the output alone.
        let mut binaural_settings = IPLBinauralEffectSettings { hrtf };
        assert_eq!(
            iplBinauralEffectCreate(context, &mut audio, &mut binaural_settings, &mut binaural),
            IPL_STATUS_SUCCESS
        );
        let mut params = IPLBinauralEffectParams {
            direction: IPLVector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            interpolation: IPL_HRTFINTERPOLATION_NEAREST,
            spatialBlend: 1.0,
            hrtf,
            peakDelays: ptr::null_mut(),
        };
        let mut input = Buffer::new(3, FRAME_SIZE, false);
        let mut out = Buffer::new(2, FRAME_SIZE, false);
        out.fill_value(SENTINEL);
        let state = iplBinauralEffectApply(binaural, &mut params, input.ipl(), out.ipl());
        assert_eq!(state, IPL_AUDIOEFFECTSTATE_TAILCOMPLETE);
        let mut short = Buffer::new(1, FRAME_SIZE - 1, false);
        iplBinauralEffectApply(binaural, &mut params, short.ipl(), out.ipl());
        params.interpolation = 5;
        iplBinauralEffectApply(binaural, &mut params, input.ipl(), out.ipl());
        params.interpolation = IPL_HRTFINTERPOLATION_NEAREST;
        params.hrtf = ptr::null_mut();
        let mut mono = Buffer::new(1, FRAME_SIZE, false);
        iplBinauralEffectApply(binaural, &mut params, mono.ipl(), out.ipl());
        let mut tail_out = Buffer::new(1, FRAME_SIZE, false);
        tail_out.fill_value(SENTINEL);
        iplBinauralEffectGetTail(binaural, tail_out.ipl());
        assert!(
            out.data
                .iter()
                .chain(&tail_out.data)
                .all(|&x| x == SENTINEL)
        );

        assert_eq!(
            iplBinauralEffectApply(ptr::null_mut(), &mut params, mono.ipl(), out.ipl()),
            IPL_AUDIOEFFECTSTATE_TAILCOMPLETE
        );
        assert_eq!(iplBinauralEffectGetTailSize(ptr::null_mut()), 0);
        iplBinauralEffectReset(ptr::null_mut());
        iplBinauralEffectRelease(&mut binaural);
        assert!(binaural.is_null());
        iplHRTFRelease(&mut hrtf);
    }
    release_context(context);
}
