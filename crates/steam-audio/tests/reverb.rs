//! Parity tests for stage 6.2 against golden data from `tools/golden/src/reverb.cpp`. Upstream has no Catch tests for
//! these modules.
//!
//! Inputs are rebuilt from the generator's `Lcg` draws in the same order, and every object holding a `ReverbEffect` is
//! built from a `CRand` with the seed the generator passes to `srand()`. Outputs record, per frame, the audio and
//! `[state, numTailSamplesRemaining]` (upstream's count goes negative after the tail; the port's is clamped at 0).

use parity::{Element, Lcg, Tolerance, compare, golden};
use steam_audio::Complex;
use steam_audio::air_absorption::AirAbsorptionModel;
use steam_audio::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use steam_audio::bands::NUM_BANDS;
use steam_audio::effects::{
    HybridReverbEffect, HybridReverbIr, HybridReverbParams, OverlapSaveConvolutionSettings,
    OverlapSaveFir, ReverbEffect,
};
use steam_audio::random::CRand;
use steam_audio::simulation::{
    EnergyField, HybridReverbEstimator, ImpulseResponse, Reverb, reverb_estimator,
};

#[track_caller]
fn check<T: Element + Into<f64>>(name: &str, actual: &[T], tol: impl Fn(&[T]) -> Tolerance) {
    let expected = golden::<T>(name);
    assert_eq!(actual.len(), expected.data.len(), "{name}: length");
    if let Err(m) = compare(actual, &expected.data, tol(&expected.data)) {
        panic!("{name}: {m}");
    }
}

fn fill(lcg: &mut Lcg, buffer: &mut AudioBuffer) {
    for x in buffer.as_mut_slice() {
        *x = lcg.next_signed();
    }
}

#[derive(Default)]
struct Recorder {
    outputs: Vec<f32>,
    states: Vec<i32>,
}

impl Recorder {
    fn record(&mut self, out: &AudioBuffer, state: AudioEffectState, tail_samples: usize) {
        self.outputs.extend_from_slice(out.as_slice());
        self.states.push(match state {
            AudioEffectState::TailRemaining => 0,
            AudioEffectState::TailComplete => 1,
        });
        self.states.push(tail_samples as i32);
    }

    #[track_caller]
    fn check(&self, name: &str) {
        check(&format!("reverb/{name}"), &self.outputs, |e| {
            Tolerance::peak_relative(e, 1e-5)
        });

        let mut expected = golden::<i32>(&format!("reverb/{name}_states")).data;
        for row in expected.chunks_exact_mut(2) {
            row[1] = row[1].max(0);
        }
        assert_eq!(self.states, expected, "{name}: states");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// ReverbEffect
// ---------------------------------------------------------------------------------------------------------------------

struct ReverbCase {
    sampling_rate: u32,
    frame_size: usize,
    rand_seed: u32,
    lcg_seed: u32,
    script: &'static str,
}

const REVERB_CASES: [ReverbCase; 3] = [
    ReverbCase {
        sampling_rate: 48000,
        frame_size: 256,
        rand_seed: 1,
        lcg_seed: 701,
        script: "AKKKKKKKAKKKKKKKZZZZZZZZPPZZZZZZTTTTTTTTQTTTRAKZZZZZZZZZZZZZZZZZZZZZZZZ",
    },
    ReverbCase {
        sampling_rate: 24000,
        frame_size: 128,
        rand_seed: 12345,
        lcg_seed: 702,
        script: "AKKKKKKKKKKKKKKKAKKKKKKKKKKKKKKKZZZZZZZZZZZZZZZZPPPPTTTTTTTTQTTT",
    },
    ReverbCase {
        sampling_rate: 22050,
        frame_size: 64,
        rand_seed: 777,
        lcg_seed: 703,
        script: "AKKKKKKKKKKKKKKKKKKKAKKKKKKKKKKKKKKKKKKKZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZPPTTTTQTTTRAZZZ",
    },
];

fn run_reverb_case(c: &ReverbCase) {
    let mut lcg = Lcg::new(c.lcg_seed);
    let mut effect = ReverbEffect::new(
        &AudioSettings::new(c.sampling_rate, c.frame_size),
        &mut CRand::new(c.rand_seed),
    );

    let mut reverb = Reverb::default();
    let mut input = AudioBuffer::new(1, c.frame_size);
    let mut out = AudioBuffer::new(1, c.frame_size);
    let mut recorder = Recorder::default();

    for op in c.script.chars() {
        match op {
            'A' => {
                reverb.reverb_times = std::array::from_fn(|_| 0.8 * (lcg.next_signed() + 1.0));
                fill(&mut lcg, &mut input);
            }
            'Q' => {
                reverb.reverb_times = std::array::from_fn(|_| 0.002 * (lcg.next_signed() + 1.0));
                fill(&mut lcg, &mut input);
            }
            'K' | 'P' => fill(&mut lcg, &mut input),
            'Z' => input.make_silent(),
            'R' => {
                effect.reset();
                continue;
            }
            'T' => {}
            _ => unreachable!(),
        }

        let state = match op {
            'T' => effect.tail(&mut out),
            'P' => effect.tail_apply(&input, &mut out),
            _ => effect.apply(&reverb, &input, &mut out),
        };
        recorder.record(&out, state, effect.num_tail_samples_remaining());
    }

    recorder.check(&format!("reverb_{}", c.sampling_rate));
}

#[test]
fn reverb_48000() {
    run_reverb_case(&REVERB_CASES[0]);
}

#[test]
fn reverb_24000() {
    run_reverb_case(&REVERB_CASES[1]);
}

#[test]
fn reverb_22050() {
    run_reverb_case(&REVERB_CASES[2]);
}

// ---------------------------------------------------------------------------------------------------------------------
// ReverbEstimator
// ---------------------------------------------------------------------------------------------------------------------

/// Geometric decay from `first_bin` with noise; channel 0 positive, other channels signed; about 10 % of the bins
/// silent.
fn fill_field(lcg: &mut Lcg, field: &mut EnergyField, first_bin: usize, silent: bool) {
    let decay: [f32; NUM_BANDS] = std::array::from_fn(|_| 0.9 + 0.09 * lcg.next_signed());

    for i in 0..field.num_channels() {
        for (j, &d) in decay.iter().enumerate() {
            let mut envelope = 0.01f32;
            for k in 0..field.num_bins() {
                let u = lcg.next_signed();
                let z = lcg.next_signed();
                let mut e = 0.0;
                if !silent && k >= first_bin && z >= -0.8 {
                    e = if i == 0 {
                        envelope * (1.0 + 0.5 * u)
                    } else {
                        envelope * 0.3 * u
                    };
                }
                if k >= first_bin {
                    envelope *= d;
                }
                field.band_mut(i, j)[k] = e;
            }
        }
    }
}

/// `(duration, order, first bin, silent)`.
const FIELDS: [(f32, usize, usize, bool); 6] = [
    (1.0, 1, 3, false),
    (0.5, 0, 0, false),
    (2.0, 2, 10, false),
    (0.03, 1, 0, false),
    (1.0, 0, 0, true),
    (0.25, 1, 24, false),
];

#[test]
fn reverb_estimator() {
    let mut lcg = Lcg::new(705);

    let air_models = [
        AirAbsorptionModel::default(),
        AirAbsorptionModel::Exponential {
            coefficients: [0.05, 0.2, 0.9],
        },
        AirAbsorptionModel::Exponential {
            coefficients: [0.0; NUM_BANDS],
        },
    ];

    let mut fields: Vec<EnergyField> = FIELDS
        .iter()
        .map(|&(duration, order, first_bin, silent)| {
            let mut field = EnergyField::new(duration, order);
            fill_field(&mut lcg, &mut field, first_bin, silent);
            field
        })
        .collect();

    let mut estimates = Vec::new();
    for field in &fields {
        for model in &air_models {
            estimates.extend(reverb_estimator::estimate(field, model).reverb_times);
        }
    }
    check("reverb/estimate", &estimates, |_| Tolerance::F32);

    let mut scaled = Vec::new();
    for field in &mut fields {
        let scale: [f32; NUM_BANDS] =
            std::array::from_fn(|_| 0.5 + 0.75 * (lcg.next_signed() + 1.0));
        reverb_estimator::apply_reverb_scale(&scale, field);
        scaled.extend_from_slice(field.data());
    }
    // Energies are small (up to 0.015), so the absolute part of `F32` would hide errors; relative only.
    check("reverb/reverb_scale", &scaled, |_| {
        Tolerance::new(0.0, 1e-5)
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// HybridReverbEstimator
// ---------------------------------------------------------------------------------------------------------------------

struct HybridCall {
    use_field: bool,
    field_duration: f32,
    ir_duration: f32,
    ir_order: usize,
    order: usize,
    transition_time: f32,
    overlap_fraction: f32,
}

const fn call(
    use_field: bool,
    field_duration: f32,
    ir_duration: f32,
    ir_order: usize,
    order: usize,
    transition_time: f32,
    overlap_fraction: f32,
) -> HybridCall {
    HybridCall {
        use_field,
        field_duration,
        ir_duration,
        ir_order,
        order,
        transition_time,
        overlap_fraction,
    }
}

fn run_hybrid_estimator(
    sampling_rate: u32,
    frame_size: usize,
    max_duration: f32,
    rand_seed: u32,
    lcg_seed: u32,
    calls: &[HybridCall],
) {
    let mut lcg = Lcg::new(lcg_seed);
    let mut estimator = HybridReverbEstimator::new(
        max_duration,
        sampling_rate,
        frame_size,
        &mut CRand::new(rand_seed),
    );

    let mut irs = Vec::new();
    let mut eqs = Vec::new();
    let mut delays = Vec::new();
    for c in calls {
        let reverb = Reverb {
            reverb_times: std::array::from_fn(|_| 0.3 + 0.6 * (lcg.next_signed() + 1.0)),
        };

        let field = c.use_field.then(|| {
            let mut field = EnergyField::new(c.field_duration, c.ir_order);
            fill_field(&mut lcg, &mut field, 0, false);
            field
        });

        let mut ir = ImpulseResponse::new(c.ir_duration, c.ir_order, sampling_rate);
        for i in 0..ir.num_channels() {
            let mut envelope = 1.0f32;
            for x in ir.channel_mut(i) {
                let u = lcg.next_signed();
                *x = envelope * u;
                envelope *= 0.9995;
            }
        }

        let estimate = estimator.estimate(
            field.as_ref(),
            &reverb,
            &mut ir,
            c.transition_time,
            c.overlap_fraction,
            c.order,
        );

        irs.extend_from_slice(ir.data());
        eqs.extend(estimate.eq_coeffs);
        delays.push(estimate.delay as i32);
    }

    let name = format!("reverb/hybrid_estimator_{sampling_rate}");
    check(&name, &irs, |e| Tolerance::peak_relative(e, 1e-5));
    check(&format!("{name}_eq"), &eqs, |_| Tolerance::F32);
    assert_eq!(delays, golden::<i32>(&format!("{name}_delay")).data);
}

#[test]
fn hybrid_estimator_24000() {
    run_hybrid_estimator(
        24000,
        256,
        0.4,
        4242,
        711,
        &[
            call(true, 0.3, 0.2, 1, 1, 0.18, 1.0),
            call(false, 0.0, 0.25, 0, 0, 0.24, 0.9),
            call(true, 0.3, 0.1, 1, 0, 0.5, 0.25),
            call(false, 0.0, 0.3, 0, 0, 0.29, 1.0),
        ],
    );
}

#[test]
fn hybrid_estimator_48000() {
    run_hybrid_estimator(
        48000,
        512,
        0.25,
        99,
        712,
        &[
            call(false, 0.0, 0.2, 0, 0, 0.19, 0.8),
            call(true, 0.2, 0.08, 1, 1, 0.07, 0.5),
        ],
    );
}

// ---------------------------------------------------------------------------------------------------------------------
// HybridReverbEffect
// ---------------------------------------------------------------------------------------------------------------------

/// A random spectrum of a real signal (real DC and Nyquist bins), as in `tests/convolution.rs`.
fn fill_fir(lcg: &mut Lcg, fir: &mut OverlapSaveFir) {
    for i in 0..fir.num_channels() {
        for j in 0..fir.num_blocks() {
            let block = fir.block_mut(i, j);
            for x in block.iter_mut() {
                let re = lcg.next_signed();
                let im = lcg.next_signed();
                *x = Complex::new(re, im);
            }
            let n = block.len();
            block[0].im = 0.0;
            block[n - 1].im = 0.0;
        }
    }
}

struct HybridParams {
    reverb: Reverb,
    eq: [f32; NUM_BANDS],
    delay: usize,
}

fn draw_params(lcg: &mut Lcg, params: &mut HybridParams, max_delay: usize) {
    params.reverb.reverb_times = std::array::from_fn(|_| 0.05 + 0.15 * (lcg.next_signed() + 1.0));
    params.eq = std::array::from_fn(|_| 0.5 * (lcg.next_signed() + 1.0));
    let u = lcg.next_signed();
    params.delay = ((u + 1.0) * 0.5 * (max_delay - 1) as f32) as usize;
}

#[test]
fn hybrid_effect() {
    const SAMPLING_RATE: u32 = 24000;
    const FRAME_SIZE: usize = 256;
    const IR_SIZE: usize = 1024;
    const SCRIPT: &str = "UAAAPAAAUAAAPAAANNPAAAAAAAAAAAAAAAADAAAQTTTTTTTRUAAT";

    let mut lcg = Lcg::new(721);
    let mut effect = HybridReverbEffect::new(
        &AudioSettings::new(SAMPLING_RATE, FRAME_SIZE),
        &OverlapSaveConvolutionSettings {
            num_channels: 1,
            ir_size: IR_SIZE,
        },
        &mut CRand::new(31337),
    );

    let mut params = HybridParams {
        reverb: Reverb::default(),
        eq: [0.0; NUM_BANDS],
        delay: 0,
    };
    draw_params(&mut lcg, &mut params, 2 * IR_SIZE);

    let mut fir = OverlapSaveFir::new(1, IR_SIZE, FRAME_SIZE);
    let mut input = AudioBuffer::new(1, FRAME_SIZE);
    let mut out = AudioBuffer::new(1, FRAME_SIZE);
    let mut recorder = Recorder::default();

    for op in SCRIPT.chars() {
        if op == 'R' {
            effect.reset();
            continue;
        }

        let state = if op == 'T' {
            effect.tail(&mut out)
        } else {
            match op {
                'U' => fill_fir(&mut lcg, &mut fir),
                'P' => draw_params(&mut lcg, &mut params, 2 * IR_SIZE),
                'Q' => {
                    params.reverb.reverb_times =
                        std::array::from_fn(|_| 0.002 * (lcg.next_signed() + 1.0));
                    params.delay = 100;
                }
                _ => {}
            }
            fill(&mut lcg, &mut input);

            let ir = match op {
                'N' => HybridReverbIr::None,
                'U' => HybridReverbIr::Updated(&mut fir),
                _ => HybridReverbIr::Current,
            };
            let hybrid_params = HybridReverbParams {
                ir,
                num_channels: 1,
                reverb: &params.reverb,
                eq_coeffs: &params.eq,
                delay: if op == 'D' {
                    2 * IR_SIZE + 5
                } else {
                    params.delay
                },
            };
            effect.apply(hybrid_params, &input, &mut out)
        };
        recorder.record(&out, state, effect.num_tail_samples_remaining());
    }

    recorder.check("hybrid_effect");
}
