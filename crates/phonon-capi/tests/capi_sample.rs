//! Stage 8d: upstream's `sample_binaural` (`core/src/samples/sample_binaural.cpp`), a program written against
//! `phonon.h` alone. `tools/golden/golden.cmake` builds the unchanged source against upstream's library and against
//! this crate's `phonon` library, and `tools/golden/run_sample.py` runs either on `capi_sample/input.npy`; upstream's
//! output is `capi_sample/output.npy`.
//!
//! `sample_matches_upstream` makes the sample's calls in the same order through this crate on every platform.
//! `port_built_sample_matches_upstream` checks the output of the sample built against the port (Windows, reference
//! build): set `SAMPLE_BINAURAL_PORT_OUTPUT` to the `.npy` that `run_sample.py` wrote and run it with `--ignored`.
//! The binaural effect convolves through the FFT (risk R1), so the audio is compared within 1e-5 of the peak.

#![allow(unsafe_code)] // The test calls the C API.

use std::ptr;

use parity::{Tolerance, compare, golden, read_npy};
use phonon::*;

/// Signal tolerance relative to the peak.
const SIGNAL: f64 = 1e-5;

/// Prints the largest error relative to the peak (for measuring bit-exactness with `--nocapture`).
fn report(name: &str, actual: &[f32], expected: &[f32]) {
    let peak = expected
        .iter()
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    let worst = actual.iter().zip(expected).fold(0.0f64, |m, (&a, &e)| {
        m.max((f64::from(a) - f64::from(e)).abs())
    });
    let identical = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| a.to_bits() == e.to_bits());
    eprintln!(
        "{name}: worst error {:.3e} x peak{}",
        worst / peak,
        if identical { " (bit-exact)" } else { "" }
    );
}

#[track_caller]
fn check_output(name: &str, actual: &[f32]) {
    let expected = golden::<f32>("capi_sample/output").data;
    report(name, actual, &expected);
    if let Err(m) = compare(
        actual,
        &expected,
        Tolerance::peak_relative(&expected, SIGNAL),
    ) {
        panic!("{name}: {m}");
    }
}

/// `sample_binaural.cpp`'s `main`, with the input and output in memory instead of `inputaudio.raw` and
/// `outputaudio.raw`.
fn sample_binaural(input: &mut [f32]) -> Vec<f32> {
    // SAFETY: the calls of the sample, with live objects and valid pointers.
    unsafe {
        let mut context_settings = IPLContextSettings {
            version: STEAMAUDIO_VERSION,
            logCallback: None,
            allocateCallback: None,
            freeCallback: None,
            simdLevel: IPL_SIMDLEVEL_SSE2,
            flags: 0,
        };
        let mut context = ptr::null_mut();
        iplContextCreate(&mut context_settings, &mut context);

        let sampling_rate = 44100;
        let frame_size = 1024;
        let mut audio_settings = IPLAudioSettings {
            samplingRate: sampling_rate,
            frameSize: frame_size,
        };

        let mut hrtf_settings = IPLHRTFSettings {
            r#type: IPL_HRTFTYPE_DEFAULT,
            sofaFileName: ptr::null(),
            sofaData: ptr::null(),
            sofaDataSize: 0,
            volume: 1.0,
            normType: IPL_HRTFNORMTYPE_NONE,
        };
        let mut hrtf = ptr::null_mut();
        iplHRTFCreate(context, &mut audio_settings, &mut hrtf_settings, &mut hrtf);

        let mut effect_settings = IPLBinauralEffectSettings { hrtf };
        let mut effect = ptr::null_mut();
        iplBinauralEffectCreate(
            context,
            &mut audio_settings,
            &mut effect_settings,
            &mut effect,
        );

        let frame = frame_size as usize;
        let mut output_frame = vec![0.0f32; 2 * frame];
        let mut output = Vec::new();

        let num_frames = input.len() / frame;
        let mut in_data = [input.as_mut_ptr()];
        let mut in_buffer = IPLAudioBuffer {
            numChannels: 1,
            numSamples: frame_size,
            data: in_data.as_mut_ptr(),
        };
        let mut out_buffer = IPLAudioBuffer {
            numChannels: 0,
            numSamples: 0,
            data: ptr::null_mut(),
        };
        iplAudioBufferAllocate(context, 2, frame_size, &mut out_buffer);

        for _ in 0..num_frames {
            let mut params = IPLBinauralEffectParams {
                direction: IPLVector3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
                interpolation: IPL_HRTFINTERPOLATION_NEAREST,
                spatialBlend: 1.0,
                hrtf,
                peakDelays: ptr::null_mut(),
            };
            iplBinauralEffectApply(effect, &mut params, &mut in_buffer, &mut out_buffer);
            iplAudioBufferInterleave(context, &mut out_buffer, output_frame.as_mut_ptr());
            output.extend_from_slice(&output_frame);
            in_data[0] = in_data[0].add(frame);
        }

        iplAudioBufferFree(context, &mut out_buffer);
        iplBinauralEffectRelease(&mut effect);
        iplHRTFRelease(&mut hrtf);
        iplContextRelease(&mut context);
        output
    }
}

#[test]
fn sample_matches_upstream() {
    let mut input = golden::<f32>("capi_sample/input").data;
    let output = sample_binaural(&mut input);
    check_output("sample_binaural", &output);
}

#[test]
#[ignore = "needs the output of sample_binaural built against the port (tools/golden/run_sample.py)"]
fn port_built_sample_matches_upstream() {
    let path = std::env::var_os("SAMPLE_BINAURAL_PORT_OUTPUT")
        .expect("set SAMPLE_BINAURAL_PORT_OUTPUT to the .npy written by run_sample.py");
    let output = read_npy::<f32>(path.as_ref())
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.to_string_lossy()))
        .data;
    check_output("sample_binaural (built against the port)", &output);
}
