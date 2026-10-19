//! Distance attenuation, air absorption and directivity models (upstream `api_advanced_simulation.cpp`).

use std::ffi::c_void;
use std::sync::Arc;

use steam_audio::air_absorption::AirAbsorptionModel;
use steam_audio::bands::NUM_BANDS;
use steam_audio::directivity::Directivity;
use steam_audio::distance_attenuation::DistanceAttenuationModel;

use crate::types::*;
use crate::{from_ipl_coordinate_space, from_ipl_vector, guard, to_ipl_vector};

/// A caller's `userData` pointer, passed back to its callback.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UserData(pub(crate) *mut c_void);

// SAFETY: the pointer is only handed back to the caller's callback; `phonon.h` makes the caller responsible for
// callbacks being safe to call from the threads that use the model, as upstream.
unsafe impl Send for UserData {}
// SAFETY: as above.
unsafe impl Sync for UserData {}

/// The model a caller describes (upstream builds a `DistanceAttenuationModel` from the type: the default model, inverse
/// distance from `minDistance`, or the callback, which with a null callback is the default model).
pub(crate) fn distance_attenuation_model(
    model: &IPLDistanceAttenuationModel,
) -> DistanceAttenuationModel {
    match model.r#type {
        IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE => DistanceAttenuationModel::InverseDistance {
            min_distance: model.minDistance,
        },
        IPL_DISTANCEATTENUATIONTYPE_CALLBACK => match model.callback {
            Some(callback) => {
                let user_data = UserData(model.userData);
                DistanceAttenuationModel::Callback(Arc::new(move |distance| {
                    let user_data = user_data;
                    // SAFETY: the caller's callback, called as `phonon.h` specifies.
                    unsafe { callback(distance, user_data.0) }
                }))
            }
            None => DistanceAttenuationModel::default(),
        },
        _ => DistanceAttenuationModel::default(),
    }
}

/// The model a caller describes: the default coefficients, the given ones, or the callback. Upstream leaves the
/// coefficients uninitialized for a callback model, so a null callback reads garbage there; here it is the default
/// model.
pub(crate) fn air_absorption_model(model: &IPLAirAbsorptionModel) -> AirAbsorptionModel {
    match model.r#type {
        IPL_AIRABSORPTIONTYPE_EXPONENTIAL => AirAbsorptionModel::Exponential {
            coefficients: model.coefficients,
        },
        IPL_AIRABSORPTIONTYPE_CALLBACK => match model.callback {
            Some(callback) => {
                let user_data = UserData(model.userData);
                AirAbsorptionModel::Callback(Arc::new(move |distance, band| {
                    let user_data = user_data;
                    // SAFETY: the caller's callback, called as `phonon.h` specifies; bands are below `NUM_BANDS`.
                    unsafe { callback(distance, band as i32, user_data.0) }
                }))
            }
            None => AirAbsorptionModel::default(),
        },
        _ => AirAbsorptionModel::default(),
    }
}

/// The pattern a caller describes: the callback if set, otherwise the weighted dipole.
pub(crate) fn directivity(model: &IPLDirectivity) -> Directivity {
    match model.callback {
        Some(callback) => {
            let user_data = UserData(model.userData);
            Directivity::Callback(Arc::new(move |direction| {
                let user_data = user_data;
                // SAFETY: the caller's callback, called as `phonon.h` specifies.
                unsafe { callback(to_ipl_vector(direction), user_data.0) }
            }))
        }
        None => Directivity::Dipole {
            weight: model.dipoleWeight,
            power: model.dipolePower,
        },
    }
}

/// Distance attenuation between a source and the listener (1 for a null model).
///
/// # Safety
///
/// `context` is null or a live context, and `model` is null or valid, with a callback that is safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplDistanceAttenuationCalculate(
    context: IPLContext,
    source: IPLVector3,
    listener: IPLVector3,
    model: *mut IPLDistanceAttenuationModel,
) -> IPLfloat32 {
    if context.is_null() {
        return 1.0;
    }
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        return 1.0;
    };
    guard(1.0, || {
        let model = distance_attenuation_model(model);
        model.evaluate((from_ipl_vector(source) - from_ipl_vector(listener)).length())
    })
}

/// Air absorption between a source and the listener, written to `airAbsorption` (3 floats). Does nothing for a null
/// model or output.
///
/// # Safety
///
/// `context` is null or a live context, `model` is null or valid (with a callback that is safe to call), and
/// `airAbsorption` is null or valid for writes of 3 floats.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplAirAbsorptionCalculate(
    context: IPLContext,
    source: IPLVector3,
    listener: IPLVector3,
    model: *mut IPLAirAbsorptionModel,
    airAbsorption: *mut IPLfloat32,
) {
    if context.is_null() || airAbsorption.is_null() {
        return;
    }
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        return;
    };
    // SAFETY: guaranteed by the caller.
    let out = unsafe { std::slice::from_raw_parts_mut(airAbsorption, NUM_BANDS) };
    guard((), || {
        let model = air_absorption_model(model);
        let distance = (from_ipl_vector(source) - from_ipl_vector(listener)).length();
        for (band, value) in out.iter_mut().enumerate() {
            *value = model.evaluate(distance, band);
        }
    })
}

/// Gain of a source's directivity pattern towards the listener (1 for a null model).
///
/// # Safety
///
/// `context` is null or a live context, and `model` is null or valid, with a callback that is safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplDirectivityCalculate(
    context: IPLContext,
    source: IPLCoordinateSpace3,
    listener: IPLVector3,
    model: *mut IPLDirectivity,
) -> IPLfloat32 {
    if context.is_null() {
        return 1.0;
    }
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        return 1.0;
    };
    guard(1.0, || {
        directivity(model).evaluate_at(
            from_ipl_vector(listener),
            &from_ipl_coordinate_space(&source),
        )
    })
}
