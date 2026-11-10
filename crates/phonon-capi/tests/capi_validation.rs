//! Parity tests for stage 8d against golden data from `tools/golden/src/capi_validation.cpp`: upstream's validation
//! layer (`api_validation_layer.cpp`), whose warnings a validating context sends to its log callback, and the device
//! functions (Embree, OpenCL, Radeon Rays, TrueAudio Next), which fail.
//!
//! Each section makes the generator's calls in the same order with the same hand-picked invalid values, records what
//! the log callback receives (each message prefixed with its level as a digit), and compares the text with the golden
//! bytes, `EXACT`. The log callback runs on the calling thread, so each test records into a thread-local string.

#![allow(unsafe_code)] // The tests call the C API.

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};
use std::ptr;

use parity::golden;
use phonon::*;

thread_local! {
    static LOG: RefCell<String> = const { RefCell::new(String::new()) };
}

unsafe extern "system" fn record(level: IPLLogLevel, message: *const c_char) {
    // SAFETY: the library passes a NUL-terminated string.
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    LOG.with(|log| {
        let mut log = log.borrow_mut();
        log.push(char::from(b'0' + level as u8));
        log.push_str(&message);
    });
}

fn take_log() -> String {
    LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

/// Compares what was logged since the last check with `capi_validation/<section>`, line by line.
#[track_caller]
fn check_log(section: &str) {
    let actual = take_log();
    let expected = golden::<u8>(&format!("capi_validation/{section}")).data;
    let expected = String::from_utf8(expected).expect("golden log is UTF-8");
    let (a, e): (Vec<_>, Vec<_>) = (actual.lines().collect(), expected.lines().collect());
    for (i, (a, e)) in a.iter().zip(&e).enumerate() {
        assert_eq!(a, e, "{section}: message {i} differs");
    }
    assert_eq!(
        a.len(),
        e.len(),
        "{section}: {} messages, expected {}; first extra: {:?}",
        a.len(),
        e.len(),
        a.get(e.len()).or(e.get(a.len()))
    );
    assert_eq!(actual, expected, "{section}: line endings differ");
}

/// A C-style zero-initialized struct (`T x{};`).
fn zeroed<T>() -> T {
    // SAFETY: only used for `phonon.h` structs, which are valid when all zero (null pointers, `None` callbacks).
    unsafe { std::mem::zeroed() }
}

const NAN: f32 = f32::from_bits(0x7fc0_0000);
const NEG_NAN: f32 = f32::from_bits(0xffc0_0000);
const NEG_NAN_PAYLOAD: f32 = f32::from_bits(0xffc0_0001);
const INF: f32 = f32::INFINITY;

fn v3(x: f32, y: f32, z: f32) -> IPLVector3 {
    IPLVector3 { x, y, z }
}

fn make_context(version: u32, flags: IPLContextFlags) -> IPLContext {
    let mut settings = IPLContextSettings {
        version,
        logCallback: Some(record),
        allocateCallback: None,
        freeCallback: None,
        simdLevel: IPL_SIMDLEVEL_SSE2,
        flags,
    };
    let mut context = ptr::null_mut();
    // SAFETY: valid pointers.
    assert_eq!(
        unsafe { iplContextCreate(&mut settings, &mut context) },
        IPL_STATUS_SUCCESS
    );
    context
}

fn validated() -> IPLContext {
    make_context(STEAMAUDIO_VERSION, IPL_CONTEXTFLAGS_VALIDATION)
}

fn release_context(mut context: IPLContext) {
    // SAFETY: a live context.
    unsafe { iplContextRelease(&mut context) };
}

#[track_caller]
fn ok(status: IPLerror) {
    assert_eq!(status, IPL_STATUS_SUCCESS);
}

fn identity() -> IPLCoordinateSpace3 {
    IPLCoordinateSpace3 {
        right: v3(1.0, 0.0, 0.0),
        up: v3(0.0, 1.0, 0.0),
        ahead: v3(0.0, 0.0, -1.0),
        origin: v3(0.0, 0.0, 0.0),
    }
}

fn identity_matrix() -> IPLMatrix4x4 {
    let mut m: IPLMatrix4x4 = zeroed();
    for i in 0..4 {
        m.elements[i][i] = 1.0;
    }
    m
}

unsafe extern "system" fn nan_distance(_: f32, _: *mut c_void) -> f32 {
    NAN
}

unsafe extern "system" fn nan_band1(_: f32, band: i32, _: *mut c_void) -> f32 {
    if band == 1 { NAN } else { 0.5 }
}

unsafe extern "system" fn aligned_allocate(_: IPLsize, _: IPLsize) -> *mut c_void {
    unreachable!("the port never calls the allocation callback")
}

/// A buffer of `channels × samples` samples of 0.25.
struct Buffer {
    samples: Vec<f32>,
    channels: Vec<*mut f32>,
    buffer: IPLAudioBuffer,
}

impl Buffer {
    fn new(channels: usize, samples: usize) -> Self {
        let mut b = Self {
            samples: vec![0.25; channels * samples],
            channels: Vec::new(),
            buffer: zeroed(),
        };
        let base = b.samples.as_mut_ptr();
        // SAFETY: within the sample vector.
        b.channels = (0..channels)
            .map(|i| unsafe { base.add(i * samples) })
            .collect();
        b.buffer = IPLAudioBuffer {
            numChannels: channels as i32,
            numSamples: samples as i32,
            data: b.channels.as_mut_ptr(),
        };
        b
    }

    fn ipl(&mut self) -> *mut IPLAudioBuffer {
        &mut self.buffer
    }

    fn set(&mut self, channel: usize, sample: usize, value: f32) {
        let samples = self.buffer.numSamples as usize;
        self.samples[channel * samples + sample] = value;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Context, model functions
// --------------------------------------------------------------------------------------------------------------------

/// The generator's `contextCases`.
fn context_cases(v: IPLContext) {
    // SAFETY: a live context and valid (or null) pointers throughout.
    unsafe {
        iplCalculateRelativeDirection(
            v,
            v3(NAN, NEG_NAN, NEG_NAN_PAYLOAD),
            v3(INF, -INF, 0.0),
            v3(0.0, 0.0, -1.0),
            v3(0.0, 1.0, 0.0),
        );

        for w in [1.5, -1e-9, 3e38, 1.000_000_1, -0.0, 123_456.79, 0.5, -7.0] {
            let mut model: IPLDirectivity = zeroed();
            model.dipoleWeight = w;
            model.dipolePower = 1.0;
            iplDirectivityCalculate(v, identity(), v3(0.0, 0.0, -1.0), &mut model);
        }
        for p in [-123.456_79, -2.5e-7, 2.0] {
            let mut model: IPLDirectivity = zeroed();
            model.dipoleWeight = 0.5;
            model.dipolePower = p;
            iplDirectivityCalculate(v, identity(), v3(0.0, 0.0, -1.0), &mut model);
        }
        iplDirectivityCalculate(v, identity(), v3(NAN, 0.0, 0.0), ptr::null_mut());

        let origin = v3(0.0, 0.0, 0.0);
        iplDistanceAttenuationCalculate(v, v3(INF, 0.0, 0.0), origin, ptr::null_mut());
        let mut model: IPLDistanceAttenuationModel = zeroed();
        model.r#type = 7;
        iplDistanceAttenuationCalculate(v, v3(1.0, 0.0, 0.0), origin, &mut model);
        let mut model: IPLDistanceAttenuationModel = zeroed();
        model.r#type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
        model.minDistance = -1.0;
        iplDistanceAttenuationCalculate(v, v3(3.0, 0.0, 0.0), origin, &mut model);
        let mut model: IPLDistanceAttenuationModel = zeroed();
        model.r#type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        model.dirty = 5;
        iplDistanceAttenuationCalculate(v, v3(2.0, 0.0, 0.0), origin, &mut model);
        let mut model: IPLDistanceAttenuationModel = zeroed();
        model.r#type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        model.callback = Some(nan_distance);
        iplDistanceAttenuationCalculate(v, v3(2.0, 0.0, 0.0), origin, &mut model);

        let source = v3(1.0, 0.0, 0.0);
        iplAirAbsorptionCalculate(v, source, origin, ptr::null_mut(), ptr::null_mut());
        let mut out = [0.0f32; 3];
        let mut model: IPLAirAbsorptionModel = zeroed();
        model.r#type = IPL_AIRABSORPTIONTYPE_EXPONENTIAL;
        model.coefficients = [1.0, INF, 2.0];
        iplAirAbsorptionCalculate(v, source, origin, &mut model, out.as_mut_ptr());
        let mut model: IPLAirAbsorptionModel = zeroed();
        model.r#type = 9;
        model.dirty = -1;
        iplAirAbsorptionCalculate(v, source, origin, &mut model, out.as_mut_ptr());
        let mut model: IPLAirAbsorptionModel = zeroed();
        model.r#type = IPL_AIRABSORPTIONTYPE_CALLBACK;
        model.callback = Some(nan_band1);
        iplAirAbsorptionCalculate(v, source, origin, &mut model, out.as_mut_ptr());
    }
}

#[test]
fn context_and_models() {
    let plain = make_context(STEAMAUDIO_VERSION, 0);

    let mut settings = IPLContextSettings {
        version: STEAMAUDIO_VERSION,
        logCallback: Some(record),
        allocateCallback: Some(aligned_allocate),
        freeCallback: None,
        simdLevel: 7,
        flags: IPL_CONTEXTFLAGS_VALIDATION,
    };
    let mut bad = ptr::null_mut();
    // SAFETY: valid pointers.
    ok(unsafe { iplContextCreate(&mut settings, &mut bad) });
    release_context(bad);

    // A caller of API 4.4 cannot ask for validation.
    let old = make_context(
        (STEAMAUDIO_VERSION_MAJOR << 16) | (4 << 8),
        IPL_CONTEXTFLAGS_VALIDATION,
    );
    // SAFETY: a live context.
    unsafe {
        iplCalculateRelativeDirection(
            old,
            v3(NAN, 0.0, 0.0),
            v3(0.0, 0.0, 0.0),
            v3(0.0, 0.0, -1.0),
            v3(0.0, 1.0, 0.0),
        )
    };
    release_context(old);

    let v = validated();
    context_cases(v);
    check_log("context");
    release_context(v);

    // Without validation nothing is logged.
    context_cases(plain);
    assert_eq!(take_log(), "");
    release_context(plain);
}

// --------------------------------------------------------------------------------------------------------------------
// Audio buffers
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn audio_buffers() {
    let v = validated();
    // SAFETY: a live context and valid (or null) pointers throughout.
    unsafe {
        let mut allocated: IPLAudioBuffer = zeroed();
        iplAudioBufferAllocate(v, 0, -3, &mut allocated);
        iplAudioBufferAllocate(v, 2, 5, ptr::null_mut());
        ok(iplAudioBufferAllocate(v, 2, 37, &mut allocated));
        iplAudioBufferFree(v, ptr::null_mut());
        iplAudioBufferFree(v, &mut allocated);

        let mut a = Buffer::new(2, 37);
        a.set(1, 36, NAN);
        let mut interleaved = vec![0.0f32; 2 * 37];
        iplAudioBufferInterleave(v, a.ipl(), interleaved.as_mut_ptr());
        iplAudioBufferInterleave(v, ptr::null_mut(), ptr::null_mut());

        let mut b = Buffer::new(2, 37);
        interleaved = vec![0.5; 2 * 37];
        interleaved[10] = INF;
        iplAudioBufferDeinterleave(v, interleaved.as_mut_ptr(), b.ipl());
        iplAudioBufferDeinterleave(v, ptr::null_mut(), b.ipl());

        let mut c = Buffer::new(2, 37);
        let mut d = Buffer::new(2, 37);
        let mut one_channel = IPLAudioBuffer {
            numChannels: 1,
            numSamples: 37,
            data: c.channels.as_mut_ptr(),
        };
        iplAudioBufferMix(v, &mut one_channel, d.ipl());
        let mut shorter = IPLAudioBuffer {
            numChannels: 2,
            numSamples: 30,
            data: c.channels.as_mut_ptr(),
        };
        iplAudioBufferMix(v, &mut shorter, d.ipl());
        iplAudioBufferMix(v, c.ipl(), d.ipl());

        let mut mono = Buffer::new(1, 37);
        iplAudioBufferDownmix(v, a.ipl(), mono.ipl());
        iplAudioBufferDownmix(v, c.ipl(), d.ipl());
        iplAudioBufferDownmix(v, &mut shorter, mono.ipl());

        let mut e = Buffer::new(4, 37);
        let mut f = Buffer::new(4, 37);
        let mut g = Buffer::new(9, 37);
        iplAudioBufferConvertAmbisonics(v, 5, -1, e.ipl(), f.ipl());
        iplAudioBufferConvertAmbisonics(
            v,
            IPL_AMBISONICSTYPE_N3D,
            IPL_AMBISONICSTYPE_SN3D,
            e.ipl(),
            g.ipl(),
        );
        iplAudioBufferConvertAmbisonics(
            v,
            IPL_AMBISONICSTYPE_N3D,
            IPL_AMBISONICSTYPE_FUMA,
            e.ipl(),
            f.ipl(),
        );
    }
    check_log("audio_buffer");
    release_context(v);
}

// --------------------------------------------------------------------------------------------------------------------
// Devices
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn devices() {
    let v = validated();
    // SAFETY: a live context and valid (or null) pointers; the dangling device list handle is never dereferenced.
    unsafe {
        let mut embree = ptr::null_mut();
        iplEmbreeDeviceCreate(v, ptr::null_mut(), &mut embree);
        iplEmbreeDeviceRelease(&mut embree);

        let mut settings = IPLOpenCLDeviceSettings {
            r#type: 4,
            numCUsToReserve: -1,
            fractionCUsForIRUpdate: 1.5,
            requiresTAN: 3,
        };
        let mut list = ptr::null_mut();
        iplOpenCLDeviceListCreate(v, &mut settings, &mut list);
        assert!(list.is_null());

        // Upstream's validated factory made an empty list; no list can exist here, so pass a dangling handle.
        let dangling = ptr::NonNull::<_IPLOpenCLDeviceList_t>::dangling().as_ptr();
        let mut device = ptr::null_mut();
        iplOpenCLDeviceCreate(v, dangling, 2, &mut device);
        iplOpenCLDeviceRelease(&mut device);
        iplOpenCLDeviceListRelease(&mut list);

        iplOpenCLDeviceCreateFromExisting(v, ptr::null_mut(), ptr::null_mut(), &mut device);
        iplOpenCLDeviceRelease(&mut device);
    }
    check_log("devices");
    release_context(v);

    // Without validation, every device function fails.
    let plain = make_context(STEAMAUDIO_VERSION, 0);
    let mut statuses = Vec::new();
    // SAFETY: a live context and valid (or null) pointers; the dangling handle is never dereferenced.
    unsafe {
        let mut embree = ptr::null_mut();
        let mut embree_settings = IPLEmbreeDeviceSettings::default();
        statuses.push(iplEmbreeDeviceCreate(
            plain,
            &mut embree_settings,
            &mut embree,
        ));
        statuses.push(embree.is_null() as i32);

        let mut settings = IPLOpenCLDeviceSettings::default();
        let mut list = ptr::null_mut();
        statuses.push(iplOpenCLDeviceListCreate(plain, &mut settings, &mut list));
        statuses.push(list.is_null() as i32);
        statuses.push(iplOpenCLDeviceListGetNumDevices(ptr::null_mut()));

        let mut device = ptr::null_mut();
        statuses.push(iplOpenCLDeviceCreate(
            plain,
            ptr::null_mut(),
            0,
            &mut device,
        ));
        statuses.push(iplOpenCLDeviceCreateFromExisting(
            plain,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut device,
        ));
        statuses.push(device.is_null() as i32);

        let mut rr = ptr::null_mut();
        let mut rr_settings = IPLRadeonRaysDeviceSettings::default();
        statuses.push(iplRadeonRaysDeviceCreate(
            ptr::null_mut(),
            &mut rr_settings,
            &mut rr,
        ));
        let mut tan = ptr::null_mut();
        let mut tan_settings = IPLTrueAudioNextDeviceSettings::default();
        statuses.push(iplTrueAudioNextDeviceCreate(
            ptr::null_mut(),
            &mut tan_settings,
            &mut tan,
        ));
        statuses.push((rr.is_null() && tan.is_null()) as i32);

        // Retaining a device that cannot exist gives null; releasing one only clears the handle.
        let mut fake = ptr::NonNull::<_IPLEmbreeDevice_t>::dangling().as_ptr();
        assert!(iplEmbreeDeviceRetain(fake).is_null());
        iplEmbreeDeviceRelease(&mut fake);
        assert!(fake.is_null());
        iplOpenCLDeviceListGetDeviceDesc(ptr::null_mut(), 0, ptr::null_mut());
    }
    assert_eq!(
        statuses,
        golden::<i32>("capi_validation/device_statuses").data
    );
    assert_eq!(take_log(), "");
    release_context(plain);
}

// --------------------------------------------------------------------------------------------------------------------
// HRTFs and effects
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn effects() {
    let v = validated();
    let mut audio = IPLAudioSettings {
        samplingRate: 48000,
        frameSize: 256,
    };
    let f = 256;
    // SAFETY: a live context, live effects and valid (or null) pointers throughout.
    unsafe {
        let mut hrtf_settings: IPLHRTFSettings = zeroed();
        hrtf_settings.r#type = IPL_HRTFTYPE_DEFAULT;
        hrtf_settings.volume = -1.0;
        hrtf_settings.normType = 5;
        let mut bad_hrtf = ptr::null_mut();
        iplHRTFCreate(v, &mut audio, &mut hrtf_settings, &mut bad_hrtf);
        iplHRTFRelease(&mut bad_hrtf);

        hrtf_settings.volume = 1.0;
        hrtf_settings.normType = IPL_HRTFNORMTYPE_NONE;
        let mut hrtf = ptr::null_mut();
        ok(iplHRTFCreate(v, &mut audio, &mut hrtf_settings, &mut hrtf));

        let mut mono = Buffer::new(1, f);
        let mut stereo = Buffer::new(2, f);
        let mut ambi = Buffer::new(4, f);
        let mut surround = Buffer::new(6, f);

        {
            let mut settings: IPLPanningEffectSettings = zeroed();
            settings.speakerLayout.r#type = IPL_SPEAKERLAYOUTTYPE_STEREO;
            let mut effect = ptr::null_mut();
            ok(iplPanningEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLPanningEffectParams {
                direction: v3(1.0, 0.0, 0.0),
            };
            iplPanningEffectApply(effect, &mut params, mono.ipl(), stereo.ipl());
            iplPanningEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLBinauralEffectSettings { hrtf };
            let mut effect = ptr::null_mut();
            ok(iplBinauralEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params: IPLBinauralEffectParams = zeroed();
            params.direction = v3(1.0, 0.0, 0.0);
            params.interpolation = IPL_HRTFINTERPOLATION_NEAREST;
            params.hrtf = hrtf;
            params.spatialBlend = 1.5;
            iplBinauralEffectApply(effect, &mut params, mono.ipl(), stereo.ipl());
            iplBinauralEffectRelease(&mut effect);
        }
        {
            let mut settings: IPLVirtualSurroundEffectSettings = zeroed();
            settings.speakerLayout.r#type = IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1;
            settings.hrtf = hrtf;
            let mut effect = ptr::null_mut();
            ok(iplVirtualSurroundEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLVirtualSurroundEffectParams { hrtf };
            iplVirtualSurroundEffectApply(effect, &mut params, surround.ipl(), stereo.ipl());
            iplVirtualSurroundEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLAmbisonicsEncodeEffectSettings { maxOrder: 1 };
            let mut effect = ptr::null_mut();
            ok(iplAmbisonicsEncodeEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLAmbisonicsEncodeEffectParams {
                direction: v3(0.0, 1.0, 0.0),
                order: -1,
            };
            iplAmbisonicsEncodeEffectApply(effect, &mut params, mono.ipl(), ambi.ipl());
            iplAmbisonicsEncodeEffectRelease(&mut effect);
        }
        {
            let mut settings: IPLAmbisonicsPanningEffectSettings = zeroed();
            settings.speakerLayout.r#type = IPL_SPEAKERLAYOUTTYPE_STEREO;
            settings.maxOrder = 1;
            let mut effect = ptr::null_mut();
            ok(iplAmbisonicsPanningEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLAmbisonicsPanningEffectParams { order: -2 };
            iplAmbisonicsPanningEffectApply(effect, &mut params, ambi.ipl(), stereo.ipl());
            iplAmbisonicsPanningEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLAmbisonicsBinauralEffectSettings { hrtf, maxOrder: 1 };
            let mut effect = ptr::null_mut();
            ok(iplAmbisonicsBinauralEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLAmbisonicsBinauralEffectParams { hrtf, order: -1 };
            iplAmbisonicsBinauralEffectApply(effect, &mut params, ambi.ipl(), stereo.ipl());
            iplAmbisonicsBinauralEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLAmbisonicsRotationEffectSettings { maxOrder: 1 };
            let mut effect = ptr::null_mut();
            ok(iplAmbisonicsRotationEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLAmbisonicsRotationEffectParams {
                orientation: identity(),
                order: -1,
            };
            let mut rotated = Buffer::new(4, f);
            iplAmbisonicsRotationEffectApply(effect, &mut params, ambi.ipl(), rotated.ipl());
            iplAmbisonicsRotationEffectRelease(&mut effect);
        }
        {
            let mut settings: IPLAmbisonicsDecodeEffectSettings = zeroed();
            settings.speakerLayout.r#type = IPL_SPEAKERLAYOUTTYPE_STEREO;
            settings.hrtf = hrtf;
            settings.maxOrder = 1;
            let mut effect = ptr::null_mut();
            ok(iplAmbisonicsDecodeEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLAmbisonicsDecodeEffectParams {
                order: 1,
                hrtf: ptr::null_mut(),
                orientation: identity(),
                binaural: IPL_FALSE,
            };
            iplAmbisonicsDecodeEffectApply(effect, &mut params, ambi.ipl(), stereo.ipl());
            iplAmbisonicsDecodeEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLDirectEffectSettings { numChannels: 1 };
            let mut effect = ptr::null_mut();
            ok(iplDirectEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params = IPLDirectEffectParams {
                flags: 64 | IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION,
                transmissionType: 5,
                distanceAttenuation: 1.5,
                airAbsorption: [-0.1, 0.5, 2.0],
                directivity: -1.0,
                occlusion: 2.0,
                transmission: [0.5, 1.25, -3.0],
            };
            let mut out = Buffer::new(1, f);
            iplDirectEffectApply(effect, &mut params, mono.ipl(), out.ipl());
            iplDirectEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLReflectionEffectSettings {
                r#type: IPL_REFLECTIONEFFECTTYPE_PARAMETRIC,
                irSize: 0,
                numChannels: 0,
            };
            let mut effect = ptr::null_mut();
            ok(iplReflectionEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut params: IPLReflectionEffectParams = zeroed();
            params.r#type = IPL_REFLECTIONEFFECTTYPE_HYBRID;
            params.reverbTimes = [1.0; 3];
            params.eq = [0.0, 0.5, 1.5];
            params.delay = 0;
            let mut out = Buffer::new(1, f);
            let none = ptr::null_mut();
            iplReflectionEffectApply(effect, &mut params, mono.ipl(), out.ipl(), none);
            params.r#type = IPL_REFLECTIONEFFECTTYPE_TAN;
            params.tanDevice = ptr::null_mut();
            params.tanSlot = -1;
            iplReflectionEffectApply(effect, &mut params, mono.ipl(), out.ipl(), none);
            iplReflectionEffectRelease(&mut effect);
        }
        {
            let mut settings = IPLReflectionEffectSettings {
                r#type: IPL_REFLECTIONEFFECTTYPE_PARAMETRIC,
                irSize: 0,
                numChannels: -1,
            };
            let mut mixer = ptr::null_mut();
            ok(iplReflectionMixerCreate(
                v,
                &mut audio,
                &mut settings,
                &mut mixer,
            ));
            let mut params: IPLReflectionEffectParams = zeroed();
            params.r#type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
            iplReflectionMixerApply(mixer, &mut params, ambi.ipl());
            iplReflectionMixerRelease(&mut mixer);
        }
        {
            let mut settings: IPLPathEffectSettings = zeroed();
            settings.maxOrder = 1;
            settings.spatialize = IPL_FALSE;
            settings.hrtf = hrtf;
            let mut effect = ptr::null_mut();
            ok(iplPathEffectCreate(
                v,
                &mut audio,
                &mut settings,
                &mut effect,
            ));
            let mut sh = [0.5f32, 0.1, 0.2, 0.3];
            let mut params: IPLPathEffectParams = zeroed();
            params.eqCoeffs = [0.0, 0.5, 1.5];
            params.shCoeffs = sh.as_mut_ptr();
            params.order = 1;
            params.binaural = IPL_FALSE;
            params.hrtf = hrtf;
            params.listener = identity();
            let mut out = Buffer::new(4, f);
            iplPathEffectApply(effect, &mut params, mono.ipl(), out.ipl());
            iplPathEffectRelease(&mut effect);
        }

        iplHRTFRelease(&mut hrtf);
    }
    check_log("effects");
    release_context(v);
}

// --------------------------------------------------------------------------------------------------------------------
// Scenes and meshes
// --------------------------------------------------------------------------------------------------------------------

/// The generator's `makeQuad`.
fn make_quad(scene: IPLScene, material: IPLMaterial) -> IPLStaticMesh {
    let mut vertices = [
        v3(-1.0, 0.0, -1.0),
        v3(1.0, 0.0, -1.0),
        v3(1.0, 0.0, 1.0),
        v3(-1.0, 0.0, 1.0),
    ];
    let mut triangles = [
        IPLTriangle { indices: [0, 2, 1] },
        IPLTriangle { indices: [0, 3, 2] },
    ];
    let mut material_indices = [0i32, 0];
    let mut materials = [material];
    let mut settings = IPLStaticMeshSettings {
        numVertices: 4,
        numTriangles: 2,
        numMaterials: 1,
        vertices: vertices.as_mut_ptr(),
        triangles: triangles.as_mut_ptr(),
        materialIndices: material_indices.as_mut_ptr(),
        materials: materials.as_mut_ptr(),
    };
    let mut mesh = ptr::null_mut();
    // SAFETY: a live scene and valid arrays.
    ok(unsafe { iplStaticMeshCreate(scene, &mut settings, &mut mesh) });
    mesh
}

fn good_material() -> IPLMaterial {
    IPLMaterial {
        absorption: [0.1, 0.2, 0.3],
        scattering: 0.05,
        transmission: [0.1, 0.05, 0.03],
    }
}

fn default_scene(v: IPLContext) -> IPLScene {
    let mut settings: IPLSceneSettings = zeroed();
    settings.r#type = IPL_SCENETYPE_DEFAULT;
    let mut scene = ptr::null_mut();
    // SAFETY: a live context and valid pointers.
    ok(unsafe { iplSceneCreate(v, &mut settings, &mut scene) });
    scene
}

#[test]
fn scenes() {
    let v = validated();
    // SAFETY: a live context, live objects and valid (or null) pointers throughout.
    unsafe {
        let mut settings: IPLSceneSettings = zeroed();
        settings.r#type = IPL_SCENETYPE_CUSTOM;
        let mut custom = ptr::null_mut();
        iplSceneCreate(v, &mut settings, &mut custom);
        iplSceneRelease(&mut custom);
        settings.r#type = 7;
        let mut unknown = ptr::null_mut();
        iplSceneCreate(v, &mut settings, &mut unknown);
        iplSceneRelease(&mut unknown);

        let mut scene = default_scene(v);
        let bad = IPLMaterial {
            absorption: [0.1, 1.5, 0.2],
            scattering: -0.5,
            transmission: [0.1, 0.2, 2.0],
        };
        let mut mesh = make_quad(scene, bad);
        iplStaticMeshAdd(mesh, ptr::null_mut());
        iplStaticMeshRemove(mesh, ptr::null_mut());
        iplStaticMeshSave(mesh, ptr::null_mut());
        iplStaticMeshAdd(mesh, scene);
        iplSceneCommit(scene);
        iplSceneSave(scene, ptr::null_mut());
        iplSceneSaveOBJ(scene, ptr::null());

        let mut sub = default_scene(v);
        let mut sub_mesh = make_quad(sub, good_material());
        iplStaticMeshAdd(sub_mesh, sub);
        iplSceneCommit(sub);

        let mut instance_settings = IPLInstancedMeshSettings {
            subScene: sub,
            transform: identity_matrix(),
        };
        instance_settings.transform.elements[1][2] = NAN;
        let mut instance = ptr::null_mut();
        ok(iplInstancedMeshCreate(
            scene,
            &mut instance_settings,
            &mut instance,
        ));
        let mut transform = identity_matrix();
        transform.elements[3][0] = -INF;
        iplInstancedMeshUpdateTransform(instance, ptr::null_mut(), transform);
        iplInstancedMeshUpdateTransform(instance, scene, identity_matrix());
        iplInstancedMeshAdd(instance, ptr::null_mut());
        iplInstancedMeshRemove(instance, ptr::null_mut());

        let mut serialized_settings: IPLSerializedObjectSettings = zeroed();
        let mut serialized = ptr::null_mut();
        ok(iplSerializedObjectCreate(
            v,
            &mut serialized_settings,
            &mut serialized,
        ));
        iplSceneSave(scene, serialized);
        let mut scene_settings: IPLSceneSettings = zeroed();
        scene_settings.r#type = IPL_SCENETYPE_DEFAULT;
        let mut loaded = ptr::null_mut();
        ok(iplSceneLoad(
            v,
            &mut scene_settings,
            serialized,
            None,
            ptr::null_mut(),
            &mut loaded,
        ));
        iplSceneRelease(&mut loaded);
        iplSerializedObjectRelease(&mut serialized);

        serialized_settings.data = ptr::null_mut();
        serialized_settings.size = 10;
        iplSerializedObjectCreate(v, &mut serialized_settings, &mut serialized);
        iplSerializedObjectRelease(&mut serialized);

        iplInstancedMeshRelease(&mut instance);
        iplStaticMeshRelease(&mut sub_mesh);
        iplStaticMeshRelease(&mut mesh);
        iplSceneRelease(&mut sub);
        iplSceneRelease(&mut scene);
    }
    check_log("scene");
    release_context(v);
}

// --------------------------------------------------------------------------------------------------------------------
// Probes and baking
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn probes_and_baking() {
    let v = validated();
    // SAFETY: a live context, live objects and valid (or null) pointers throughout.
    unsafe {
        let mut scene = default_scene(v);
        let mut floor = make_quad(scene, good_material());
        iplStaticMeshAdd(floor, scene);
        iplSceneCommit(scene);

        let mut array = ptr::null_mut();
        ok(iplProbeArrayCreate(v, &mut array));
        let mut params = IPLProbeGenerationParams {
            r#type: 5,
            spacing: -1.0,
            height: 0.0,
            transform: identity_matrix(),
        };
        params.transform.elements[0][3] = NAN;
        iplProbeArrayGenerateProbes(array, ptr::null_mut(), &mut params);

        params.r#type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
        params.spacing = 0.5;
        params.height = 0.0;
        params.transform = identity_matrix();
        for i in 0..3 {
            params.transform.elements[i][i] = 2.0;
        }
        iplProbeArrayGenerateProbes(array, scene, &mut params);
        iplProbeArrayGetProbe(array, -1);
        iplProbeArrayGetProbe(array, 1000);
        iplProbeArrayGetProbe(array, 0);

        let mut batch = ptr::null_mut();
        ok(iplProbeBatchCreate(v, &mut batch));
        iplProbeBatchAddProbeArray(batch, ptr::null_mut());
        iplProbeBatchRemoveProbe(batch, 5);
        iplProbeBatchAddProbeArray(batch, array);
        iplProbeBatchCommit(batch);
        iplProbeBatchRemoveProbe(batch, -2);

        let mut unknown = IPLBakedDataIdentifier {
            r#type: 5,
            variation: 9,
            endpointInfluence: IPLSphere {
                center: v3(0.0, INF, 0.0),
                radius: -2.0,
            },
        };
        iplProbeBatchRemoveData(batch, &mut unknown);
        iplProbeBatchGetDataSize(batch, &mut unknown);

        let mut reverb: IPLBakedDataIdentifier = zeroed();
        reverb.r#type = IPL_BAKEDDATATYPE_REFLECTIONS;
        reverb.variation = IPL_BAKEDDATAVARIATION_REVERB;
        iplProbeBatchGetEnergyField(batch, &mut reverb, 1000, ptr::null_mut());
        iplProbeBatchGetReverb(batch, &mut reverb, -1, ptr::null_mut());
        iplProbeBatchSave(batch, ptr::null_mut());

        let mut serialized_settings: IPLSerializedObjectSettings = zeroed();
        let mut serialized = ptr::null_mut();
        ok(iplSerializedObjectCreate(
            v,
            &mut serialized_settings,
            &mut serialized,
        ));
        iplProbeBatchSave(batch, serialized);
        let mut loaded = ptr::null_mut();
        ok(iplProbeBatchLoad(v, serialized, &mut loaded));
        iplProbeBatchRelease(&mut loaded);
        iplSerializedObjectRelease(&mut serialized);

        let mut other = ptr::null_mut();
        ok(iplProbeBatchCreate(v, &mut other));
        let probe = IPLSphere {
            center: v3(NAN, 0.0, 0.0),
            radius: -1.0,
        };
        iplProbeBatchAddProbe(other, probe);
        iplProbeBatchRelease(&mut other);

        let mut bake: IPLReflectionsBakeParams = zeroed();
        bake.sceneType = 9;
        bake.identifier.r#type = 7;
        bake.identifier.variation = 5;
        bake.identifier.endpointInfluence.center = v3(INF, 0.0, 0.0);
        bake.bakeFlags = 4;
        bake.numRays = 0;
        bake.numDiffuseSamples = -1;
        bake.numBounces = 0;
        bake.simulatedDuration = 0.0;
        bake.savedDuration = 1.0;
        bake.order = 0;
        bake.numThreads = 0;
        bake.irradianceMinDistance = 0.0;
        iplReflectionsBakerBake(v, &mut bake, None, ptr::null_mut());
        bake.sceneType = IPL_SCENETYPE_CUSTOM;
        bake.rayBatchSize = 0;
        iplReflectionsBakerBake(v, &mut bake, None, ptr::null_mut());
        bake.sceneType = IPL_SCENETYPE_RADEONRAYS;
        bake.bakeBatchSize = 0;
        iplReflectionsBakerBake(v, &mut bake, None, ptr::null_mut());

        let mut bake: IPLPathBakeParams = zeroed();
        bake.identifier.r#type = IPL_BAKEDDATATYPE_PATHING;
        bake.identifier.variation = IPL_BAKEDDATAVARIATION_DYNAMIC;
        bake.numSamples = 0;
        bake.radius = -1.0;
        bake.threshold = 0.0;
        bake.visRange = 0.0;
        bake.pathRange = -5.0;
        bake.numThreads = -1;
        iplPathBakerBake(v, &mut bake, None, ptr::null_mut());

        iplProbeBatchRelease(&mut batch);
        iplProbeArrayRelease(&mut array);
        iplStaticMeshRelease(&mut floor);
        iplSceneRelease(&mut scene);
    }
    check_log("probes");
    release_context(v);
}

// --------------------------------------------------------------------------------------------------------------------
// Simulator and sources
// --------------------------------------------------------------------------------------------------------------------

fn direct_settings() -> IPLSimulationSettings {
    let mut s: IPLSimulationSettings = zeroed();
    s.flags = IPL_SIMULATIONFLAGS_DIRECT;
    s.sceneType = IPL_SCENETYPE_DEFAULT;
    s.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
    s.maxNumOcclusionSamples = 4;
    s.maxNumRays = 1;
    s.numDiffuseSamples = 1;
    s.maxDuration = 1.0;
    s.maxOrder = 0;
    s.maxNumSources = 2;
    s.numThreads = 1;
    s.rayBatchSize = 1;
    s.numVisSamples = 1;
    s.samplingRate = 48000;
    s.frameSize = 256;
    s
}

#[test]
fn simulation() {
    let v = validated();
    // SAFETY: a live context, live objects and valid (or null) pointers throughout.
    unsafe {
        let mut settings = direct_settings();
        settings.flags = IPL_SIMULATIONFLAGS_DIRECT | 16;
        settings.sceneType = IPL_SCENETYPE_CUSTOM;
        settings.reflectionType = IPL_REFLECTIONEFFECTTYPE_TAN;
        settings.maxNumRays = 0;
        settings.numDiffuseSamples = 0;
        settings.maxDuration = 0.0;
        settings.rayBatchSize = 0;
        settings.numVisSamples = 0;
        let mut bad = ptr::null_mut();
        iplSimulatorCreate(v, &mut settings, &mut bad);
        iplSimulatorRelease(&mut bad);
        settings.flags = IPL_SIMULATIONFLAGS_DIRECT;
        settings.sceneType = IPL_SCENETYPE_RADEONRAYS;
        settings.reflectionType = -1;
        iplSimulatorCreate(v, &mut settings, &mut bad);
        iplSimulatorRelease(&mut bad);

        let mut settings = direct_settings();
        let mut simulator = ptr::null_mut();
        ok(iplSimulatorCreate(v, &mut settings, &mut simulator));
        iplSimulatorSetScene(simulator, ptr::null_mut());
        iplSimulatorAddProbeBatch(simulator, ptr::null_mut());
        iplSimulatorRemoveProbeBatch(simulator, ptr::null_mut());

        let mut shared: IPLSimulationSharedInputs = zeroed();
        shared.listener = identity();
        shared.listener.up.y = NAN;
        shared.numRays = 0;
        shared.numBounces = 0;
        shared.duration = 0.0;
        shared.order = -1;
        shared.irradianceMinDistance = 0.0;
        iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_REFLECTIONS | 32, &mut shared);
        iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_DIRECT, ptr::null_mut());
        shared.listener = identity();
        iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_DIRECT, &mut shared);

        let mut source_settings = IPLSourceSettings { flags: 9 };
        let mut source = ptr::null_mut();
        ok(iplSourceCreate(
            simulator,
            &mut source_settings,
            &mut source,
        ));
        iplSourceAdd(source, ptr::null_mut());
        iplSourceRemove(source, ptr::null_mut());

        let mut deviation: IPLDeviationModel = zeroed();
        deviation.r#type = 2;
        let mut inputs: IPLSimulationInputs = zeroed();
        inputs.flags = IPL_SIMULATIONFLAGS_DIRECT
            | IPL_SIMULATIONFLAGS_REFLECTIONS
            | IPL_SIMULATIONFLAGS_PATHING;
        inputs.directFlags = 32;
        inputs.source = identity();
        inputs.source.origin.z = NAN;
        inputs.distanceAttenuationModel.r#type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        inputs.distanceAttenuationModel.dirty = 2;
        inputs.airAbsorptionModel.r#type = 4;
        inputs.directivity.dipoleWeight = 2.0;
        inputs.directivity.dipolePower = -1.0;
        inputs.occlusionType = 3;
        inputs.occlusionRadius = 0.0;
        inputs.numOcclusionSamples = 0;
        inputs.reverbScale = [0.0, 1.0, -1.0];
        inputs.hybridReverbTransitionTime = 0.0;
        inputs.hybridReverbOverlapPercent = 0.0;
        inputs.baked = IPL_TRUE;
        inputs.bakedDataIdentifier.r#type = 3;
        inputs.bakedDataIdentifier.variation = IPL_BAKEDDATAVARIATION_DYNAMIC;
        inputs.bakedDataIdentifier.endpointInfluence.radius = -1.0;
        inputs.pathingProbes = ptr::null_mut();
        inputs.visRadius = -1.0;
        inputs.visThreshold = -0.5;
        inputs.visRange = -2.0;
        inputs.pathingOrder = -1;
        inputs.enableValidation = 2;
        inputs.findAlternatePaths = 3;
        inputs.numTransmissionRays = 0;
        inputs.deviationModel = &mut deviation;
        iplSourceSetInputs(source, 64, &mut inputs);
        iplSourceSetInputs(source, IPL_SIMULATIONFLAGS_DIRECT, ptr::null_mut());

        let mut outputs: IPLSimulationOutputs = zeroed();
        iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT, &mut outputs);
        iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT | 8, ptr::null_mut());
        outputs.direct.flags = 128;
        outputs.direct.transmissionType = -1;
        iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT, &mut outputs);

        iplSourceRelease(&mut source);
        iplSimulatorRelease(&mut simulator);
    }
    check_log("simulation");
    release_context(v);
}

// --------------------------------------------------------------------------------------------------------------------
// Energy fields, impulse responses, reconstructors
// --------------------------------------------------------------------------------------------------------------------

#[test]
fn reconstructor() {
    let v = validated();
    // SAFETY: a live context, live objects and valid (or null) pointers throughout.
    unsafe {
        let mut settings = IPLEnergyFieldSettings {
            duration: 0.0,
            order: 0,
        };
        let mut field = ptr::null_mut();
        iplEnergyFieldCreate(v, &mut settings, &mut field);
        iplEnergyFieldRelease(&mut field);
        let mut ir_settings = IPLImpulseResponseSettings {
            duration: 0.0,
            order: 0,
            samplingRate: -1,
        };
        let mut ir = ptr::null_mut();
        iplImpulseResponseCreate(v, &mut ir_settings, &mut ir);
        iplImpulseResponseRelease(&mut ir);

        let mut settings = IPLReconstructorSettings {
            maxDuration: 1.0,
            maxOrder: 1,
            samplingRate: 24000,
        };
        let mut reconstructor = ptr::null_mut();
        ok(iplReconstructorCreate(v, &mut settings, &mut reconstructor));

        let mut shared = IPLReconstructorSharedInputs {
            duration: 0.0,
            order: -1,
        };
        let none = ptr::null_mut();
        iplReconstructorReconstruct(reconstructor, -1, none, &mut shared, ptr::null_mut());
        let mut inputs: [IPLReconstructorInputs; 2] = zeroed();
        let mut outputs: [IPLReconstructorOutputs; 2] = zeroed();
        shared = IPLReconstructorSharedInputs {
            duration: 0.5,
            order: 1,
        };
        let (i, o) = (inputs.as_mut_ptr(), outputs.as_mut_ptr());
        iplReconstructorReconstruct(reconstructor, 2, i, &mut shared, o);
        iplReconstructorReconstruct(reconstructor, 1, i, ptr::null_mut(), o);

        iplReconstructorRelease(&mut reconstructor);
    }
    check_log("reconstructor");
    release_context(v);
}
