//! The inputs, outputs and state of one source in a [`SimulationManager`](super::SimulationManager) (upstream
//! `simulation_data.{h,cpp}`, `SimulationData`). Pathing inputs and outputs follow in stage 7.

use crate::air_absorption::AirAbsorptionModel;
use crate::baking::{BakedDataIdentifier, BakedDataType, BakedDataVariation};
use crate::bands::NUM_BANDS;
use crate::directivity::Directivity;
use crate::distance_attenuation::DistanceAttenuationModel;
use crate::effects::OverlapSaveFir;
use crate::geometry::Sphere;
use crate::math::{CoordinateSpace3, Vector3};
use crate::sh;
use crate::triple_buffer::{TripleBufferReader, TripleBufferWriter, triple_buffer};

use super::{
    DirectSimulationFlags, DirectSoundPath, EnergyField, ImpulseResponse, OcclusionType, Reverb,
    SimulationSettings,
};

/// What indirect sound is rendered with, and so what the simulation produces for it (upstream
/// `IndirectEffectType`). TrueAudioNext is out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndirectEffectType {
    /// An impulse response, partitioned for convolution.
    Convolution,
    /// A reverb time per band.
    Parametric,
    /// Convolution for the early part, parametric reverb for the rest.
    Hybrid,
}

/// Inputs of a source's direct sound simulation (upstream `DirectSimulationInputs`).
#[derive(Debug, Clone)]
pub struct DirectSimulationInputs {
    pub flags: DirectSimulationFlags,
    pub source: CoordinateSpace3,
    pub distance_attenuation_model: DistanceAttenuationModel,
    pub air_absorption_model: AirAbsorptionModel,
    pub directivity: Directivity,
    pub occlusion_type: OcclusionType,
    pub occlusion_radius: f32,
    pub num_occlusion_samples: usize,
    pub num_transmission_rays: usize,
}

/// Inputs of a source's reflection simulation (upstream `ReflectionSimulationInputs`).
#[derive(Debug, Clone)]
pub struct ReflectionSimulationInputs {
    pub enabled: bool,
    pub source: CoordinateSpace3,
    /// The model the reflections are reconstructed for. Reflections are simulated for the default model; any other
    /// model gets a correction curve, made again whenever the model changes or is marked dirty.
    pub distance_attenuation_model: DistanceAttenuationModel,
    /// The model changed although it compares equal (a callback that now behaves differently): upstream
    /// `distanceAttenuationModel.dirty`. Cleared when the correction curve is made.
    pub distance_attenuation_model_dirty: bool,
    pub air_absorption_model: AirAbsorptionModel,
    pub directivity: Directivity,
    /// Factors applied to the reverb times (and the energy decay) per band.
    pub reverb_scale: [f32; NUM_BANDS],
    /// Hybrid reverb: seconds of the impulse response used before the parametric part takes over.
    pub transition_time: f32,
    /// Hybrid reverb: fraction of the transition time over which the two parts are crossfaded.
    pub overlap_fraction: f32,
    /// Look the reflections up in the baked layer `baked_data_identifier` instead of simulating them.
    pub baked: bool,
    pub baked_data_identifier: BakedDataIdentifier,
}

/// Results of a source's reflection simulation (upstream `ReflectionSimulationOutputs`).
#[derive(Debug)]
pub struct ReflectionSimulationOutputs {
    /// The reading side of the triple buffer the partitioned impulse response is published to, for
    /// [`IndirectEffectType::Convolution`] and [`Hybrid`](IndirectEffectType::Hybrid); `take` it to hand it to
    /// the audio thread.
    pub overlap_save_fir: Option<TripleBufferReader<OverlapSaveFir>>,
    pub reverb: Reverb,
    pub hybrid_eq: [f32; NUM_BANDS],
    /// Hybrid reverb: delay of the parametric part, in samples.
    pub hybrid_delay: usize,
    /// Channels and samples of the last published impulse response.
    pub num_channels: usize,
    pub num_samples: usize,
}

/// What a source keeps between reflection simulations (upstream `ReflectionSimulationState`).
#[derive(Debug)]
pub(crate) struct ReflectionSimulationState {
    pub(crate) prev_source: CoordinateSpace3,
    pub(crate) prev_distance_attenuation_model: DistanceAttenuationModel,
    pub(crate) prev_directivity: Directivity,
    /// The latest simulation, when it is accumulated into `accum_energy_field`.
    pub(crate) energy_field: EnergyField,
    /// The energy field reconstructed from: averaged over the frames since the source, listener or scene last
    /// changed, or looked up from baked data.
    pub(crate) accum_energy_field: EnergyField,
    pub(crate) num_frames_accumulated: u32,
    /// Empty for [`IndirectEffectType::Parametric`].
    pub(crate) distance_attenuation_correction_curve: Vec<f32>,
    pub(crate) apply_distance_attenuation_correction_curve: bool,
    /// `None` for [`IndirectEffectType::Parametric`].
    pub(crate) impulse_response: Option<ImpulseResponse>,
    /// `None` unless the impulse response is partitioned.
    pub(crate) overlap_save_fir: Option<TripleBufferWriter<OverlapSaveFir>>,
    pub(crate) valid_simulation_data: bool,
}

/// A source: its simulation inputs, outputs and state (upstream `SimulationData`).
///
/// A [`SimulationManager`](super::SimulationManager) simulates the sources added to it, reading `*_inputs` and
/// writing `*_outputs`.
#[derive(Debug)]
pub struct SimulationData {
    pub direct_inputs: DirectSimulationInputs,
    pub reflection_inputs: ReflectionSimulationInputs,
    /// Upstream `directOutputs.directPath`.
    pub direct_path: DirectSoundPath,
    pub reflection_outputs: ReflectionSimulationOutputs,
    /// `None` when the source was created without indirect simulation.
    pub(crate) reflection_state: Option<ReflectionSimulationState>,
}

impl SimulationData {
    /// A source for a manager with `settings`, with reflections if `enable_indirect`. Its inputs start with nothing
    /// to simulate: no direct flags, reflections disabled.
    pub fn new(settings: &SimulationSettings, enable_indirect: bool) -> Self {
        let direct_inputs = DirectSimulationInputs {
            flags: DirectSimulationFlags::NONE,
            source: CoordinateSpace3::default(),
            distance_attenuation_model: DistanceAttenuationModel::default(),
            air_absorption_model: AirAbsorptionModel::default(),
            directivity: Directivity::default(),
            occlusion_type: OcclusionType::Raycast,
            occlusion_radius: 0.0,
            num_occlusion_samples: settings.max_num_occlusion_samples,
            num_transmission_rays: 1,
        };

        // Upstream leaves these uninitialized when the source has no indirect simulation.
        let mut reflection_inputs = ReflectionSimulationInputs {
            enabled: false,
            source: CoordinateSpace3::default(),
            distance_attenuation_model: DistanceAttenuationModel::default(),
            distance_attenuation_model_dirty: false,
            air_absorption_model: AirAbsorptionModel::default(),
            directivity: Directivity::default(),
            reverb_scale: [1.0; NUM_BANDS],
            transition_time: 1.0,
            overlap_fraction: 0.25,
            baked: false,
            baked_data_identifier: BakedDataIdentifier {
                variation: BakedDataVariation::Reverb,
                data_type: BakedDataType::Reflections,
                endpoint_influence: Sphere::new(Vector3::ZERO, 0.0),
            },
        };

        let mut reflection_outputs = ReflectionSimulationOutputs {
            overlap_save_fir: None,
            reverb: Reverb::default(),
            hybrid_eq: [1.0; NUM_BANDS],
            hybrid_delay: 0,
            num_channels: 0,
            num_samples: 0,
        };

        let direct_path = DirectSoundPath {
            distance_attenuation: 1.0,
            air_absorption: [1.0; NUM_BANDS],
            delay: 0.0,
            occlusion: 1.0,
            transmission: [1.0; NUM_BANDS],
            directivity: 1.0,
        };

        let reflection_state = enable_indirect.then(|| {
            reflection_inputs.transition_time = 1.0;
            reflection_inputs.overlap_fraction = 0.25;
            reflection_inputs.reverb_scale = [1.0; NUM_BANDS];

            let indirect_type = settings.indirect_type;
            let mut state = ReflectionSimulationState {
                prev_source: CoordinateSpace3::default(),
                prev_distance_attenuation_model: DistanceAttenuationModel::default(),
                prev_directivity: Directivity::default(),
                energy_field: EnergyField::new(settings.max_duration, settings.max_order),
                accum_energy_field: EnergyField::new(settings.max_duration, settings.max_order),
                num_frames_accumulated: 0,
                distance_attenuation_correction_curve: Vec::new(),
                apply_distance_attenuation_correction_curve: false,
                impulse_response: None,
                overlap_save_fir: None,
                valid_simulation_data: false,
            };

            if indirect_type != IndirectEffectType::Parametric {
                state.impulse_response = Some(ImpulseResponse::new(
                    settings.max_duration,
                    settings.max_order,
                    settings.sampling_rate,
                ));

                let num_channels = sh::num_coeffs_for_order(settings.max_order);
                let ir_size =
                    (settings.max_duration * settings.sampling_rate as f32).ceil() as usize;

                state.distance_attenuation_correction_curve = vec![0.0; ir_size];

                let (writer, reader) = triple_buffer(|| {
                    OverlapSaveFir::new(num_channels, ir_size, settings.frame_size)
                });
                state.overlap_save_fir = Some(writer);
                reflection_outputs.overlap_save_fir = Some(reader);

                reflection_outputs.num_channels = num_channels;
                reflection_outputs.num_samples = ir_size;
            }

            if indirect_type == IndirectEffectType::Hybrid {
                let delay = ((1.0 - reflection_inputs.overlap_fraction)
                    * reflection_inputs.transition_time
                    * settings.sampling_rate as f32)
                    .ceil();
                reflection_outputs.hybrid_delay = delay as usize;
            }

            state
        });

        Self {
            direct_inputs,
            reflection_inputs,
            direct_path,
            reflection_outputs,
            reflection_state,
        }
    }

    /// The energy field reconstructed from in the last simulation (upstream `reflectionState.accumEnergyField`), if
    /// the source has indirect simulation.
    pub fn energy_field(&self) -> Option<&EnergyField> {
        Some(&self.reflection_state.as_ref()?.accum_energy_field)
    }

    /// The impulse response of the last simulation, before partitioning (upstream
    /// `reflectionState.impulseResponse`), if the source has indirect simulation and it is not parametric.
    pub fn impulse_response(&self) -> Option<&ImpulseResponse> {
        self.reflection_state.as_ref()?.impulse_response.as_ref()
    }

    /// Whether the source or its directivity moved on since the last real-time reflection simulation (upstream
    /// `hasSourceChanged`). A callback directivity counts as a dipole of weight and power 0.
    pub(crate) fn has_source_changed(&self) -> bool {
        let state = self
            .reflection_state
            .as_ref()
            .expect("a source with indirect simulation");
        let (source, prev) = (&self.reflection_inputs.source, &state.prev_source);
        let (weight, power) = dipole(&self.reflection_inputs.directivity);
        let (prev_weight, prev_power) = dipole(&state.prev_directivity);

        let mut changed = (source.origin - prev.origin).length() > 1e-4;
        changed = changed || (source.ahead - prev.ahead).length() > 1e-4;
        changed = changed || (source.up - prev.up).length() > 1e-4;
        changed = changed || (weight - prev_weight).abs() > 1e-4;
        changed = changed || (power - prev_power).abs() > 1e-4;
        changed
    }
}

fn dipole(directivity: &Directivity) -> (f32, f32) {
    match directivity {
        Directivity::Dipole { weight, power } => (*weight, *power),
        Directivity::Callback(_) => (0.0, 0.0),
    }
}
