//! Simulation of sound propagation through a scene. So far: the direct path (upstream `direct_simulator`), the energy
//! fields and impulse responses of reflections with their reconstruction (upstream `energy_field`,
//! `impulse_response`, `reconstructor`), reverb estimation (upstream `reverb_estimator`,
//! `hybrid_reverb_estimator`), the reflection simulators that produce energy fields (upstream
//! `reflection_simulator`), and lookups of baked reflections at probes (upstream `baked_reflection_simulator`).

pub mod baked_reflection;
pub mod direct;
pub mod energy_field;
pub mod hybrid_reverb_estimator;
pub mod impulse_response;
pub mod reconstructor;
pub mod reflection;
pub mod reverb_estimator;

pub use direct::{
    DirectSimulationFlags, DirectSimulationParams, DirectSimulator, DirectSoundPath, OcclusionType,
};
pub use energy_field::EnergyField;
pub use hybrid_reverb_estimator::{HybridReverbEstimate, HybridReverbEstimator};
pub use impulse_response::ImpulseResponse;
pub use reconstructor::{ReconstructionParams, ReconstructionType, Reconstructor};
pub use reflection::{
    BatchedReflectionSimulator, ReflectionSimulationParams, ReflectionSimulator,
    ReflectionSimulatorSettings,
};
pub use reverb_estimator::Reverb;
