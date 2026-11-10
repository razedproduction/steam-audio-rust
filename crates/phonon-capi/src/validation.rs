//! The validation layer (upstream `api_validation_layer.cpp`): with `IPL_CONTEXTFLAGS_VALIDATION`, every object of the
//! context checks the arguments and results of its calls and logs a warning for each invalid value, then does what it
//! would do without validation.
//!
//! A message reads `Warning: <method>: invalid <type>: <expression> = <value>\n`, as upstream's `VALIDATE` macro writes
//! it: `<method>` is the name of upstream's C++ method (`__func__`), `<expression>` the argument expression the macro was
//! given (so loop variables appear by name: `params->eqCoeffs[iBand]`), and `<value>` is formatted as `std::to_string`
//! and MSVC's `%p` format it. Expressions are `&'static str`s built with `concat!`, so a call that passes validation
//! allocates nothing.
//!
//! The checks follow upstream's macros one for one, in the same order. Where upstream would dereference a null pointer
//! or read out of bounds while validating (a null input buffer whose channel count it compares, a channel pointer it
//! reads), the port logs what upstream logs up to that point and skips the check that would crash (`DEVIATIONS.md`).

use std::ffi::{CString, c_void};

use crate::types::*;

/// `IPL_NUM_BANDS`.
pub(crate) const NUM_BANDS: usize = 3;

/// A value as upstream's `to_string` overloads format it in a message.
pub(crate) trait CValue {
    fn c_format(&self) -> String;
}

impl CValue for i32 {
    fn c_format(&self) -> String {
        self.to_string()
    }
}

impl CValue for f32 {
    fn c_format(&self) -> String {
        c_float(*self)
    }
}

impl<T> CValue for *const T {
    fn c_format(&self) -> String {
        c_pointer(self.cast())
    }
}

impl<T> CValue for *mut T {
    fn c_format(&self) -> String {
        c_pointer(self.cast_const().cast())
    }
}

/// A float as MSVC's `printf("%f")` writes it (`std::to_string(float)` promotes it to `double`): six decimals,
/// `inf`, and for NaNs (always quiet after the promotion) `nan`, `-nan(ind)` for the negative default NaN and `-nan`
/// for other negative ones.
pub(crate) fn c_float(x: f32) -> String {
    let x = f64::from(x);
    if x.is_nan() {
        let bits = x.to_bits();
        let negative = bits >> 63 != 0;
        let payload = bits & ((1 << 51) - 1);
        match (negative, payload) {
            (false, _) => "nan".to_string(),
            (true, 0) => "-nan(ind)".to_string(),
            (true, _) => "-nan".to_string(),
        }
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{x:.6}")
    }
}

/// A pointer as MSVC's `printf("%p")` writes it on a 64-bit target: 16 upper-case hexadecimal digits.
pub(crate) fn c_pointer(p: *const c_void) -> String {
    format!("{:0width$X}", p as usize, width = 2 * size_of::<usize>())
}

/// Logs the warnings of one call to a method of a validating object.
#[derive(Clone, Copy)]
pub(crate) struct Validator {
    log: IPLLogFunction,
    /// Upstream's method name (`__func__`).
    method: &'static str,
}

impl Validator {
    pub(crate) fn new(log: IPLLogFunction, method: &'static str) -> Self {
        Self { log, method }
    }

    /// Logs `Warning: <method>: invalid <ty>: <name> = <value>` unless `ok`.
    pub(crate) fn check(&self, ty: &str, name: &str, value: impl CValue, ok: bool) {
        if !ok {
            let message = format!(
                "Warning: {}: invalid {ty}: {name} = {}\n",
                self.method,
                value.c_format()
            );
            log(self.log, IPL_LOGLEVEL_WARNING, &message);
        }
    }

    /// `VALIDATE_IPLfloat32`: finite.
    pub(crate) fn float(&self, name: &str, value: f32) {
        self.check("IPLfloat32", name, value, value.is_finite());
    }

    /// `VALIDATE_POINTER`: not null. Returns whether it is not null.
    pub(crate) fn pointer<T>(&self, name: &str, value: *const T) -> bool {
        self.check("void*", name, value, !value.is_null());
        !value.is_null()
    }

    /// `VALIDATE(IPLint32, ...)`.
    pub(crate) fn int(&self, name: &str, value: i32, ok: bool) {
        self.check("IPLint32", name, value, ok);
    }

    /// `VALIDATE(IPLfloat32, ...)` with a condition other than finiteness.
    pub(crate) fn float_if(&self, name: &str, value: f32, ok: bool) {
        self.check("IPLfloat32", name, value, ok);
    }

    /// An enum `ty` whose valid values are `low..=high`.
    pub(crate) fn range(&self, ty: &str, name: &str, value: i32, low: i32, high: i32) {
        self.check(ty, name, value, low <= value && value <= high);
    }

    /// A flag set `ty` with no bits outside `allowed`.
    pub(crate) fn flags(&self, ty: &str, name: &str, value: i32, allowed: i32) {
        self.check(ty, name, value, value & !allowed == 0);
    }

    /// `VALIDATE_IPLbool`.
    pub(crate) fn bool(&self, name: &str, value: IPLbool) {
        self.range("IPLbool", name, value, IPL_FALSE, IPL_TRUE);
    }

    /// `VALIDATE_ARRAY_IPLfloat32`: a non-null array of `size` floats, the first non-finite one logged. `name` is the
    /// array expression followed by `[iArray]`.
    ///
    /// # Safety
    ///
    /// `values` is null or valid for reads of `size` floats.
    pub(crate) unsafe fn float_array(
        &self,
        pointer_name: &str,
        name: &str,
        values: *const f32,
        size: i32,
    ) {
        if self.pointer(pointer_name, values) {
            for i in 0..usize::try_from(size).unwrap_or(0) {
                // SAFETY: guaranteed by the caller.
                let value = unsafe { *values.add(i) };
                self.float(name, value);
                if !value.is_finite() {
                    break;
                }
            }
        }
    }

    /// `VALIDATE_IPLAudioBuffer`: a non-null buffer with channels, samples and channel pointers; with `data`, the last
    /// sample of each channel finite. Returns the buffer when it is not null.
    ///
    /// # Safety
    ///
    /// `buffer` is null or valid, and so are its channel pointers that are not null, for `numSamples` floats.
    pub(crate) unsafe fn audio_buffer<'a>(
        &self,
        names: &AudioBufferNames,
        buffer: *const IPLAudioBuffer,
        data: bool,
    ) -> Option<&'a IPLAudioBuffer> {
        self.pointer(names.buffer, buffer);
        // SAFETY: guaranteed by the caller.
        let buffer = unsafe { buffer.as_ref() }?;
        self.int(
            names.num_channels,
            buffer.numChannels,
            buffer.numChannels > 0,
        );
        self.int(names.num_samples, buffer.numSamples, buffer.numSamples > 0);
        self.pointer(names.data, buffer.data);
        if data && !buffer.data.is_null() && buffer.numSamples > 0 {
            let last = buffer.numSamples as usize - 1;
            for i in 0..usize::try_from(buffer.numChannels).unwrap_or(0) {
                // SAFETY: guaranteed by the caller.
                let channel = unsafe { *buffer.data.add(i) };
                // Upstream reads through a null channel pointer; the port skips it.
                if !channel.is_null() {
                    // SAFETY: guaranteed by the caller.
                    self.float(names.last_sample, unsafe { *channel.add(last) });
                }
            }
        }
        Some(buffer)
    }
}

/// The expressions `VALIDATE_IPLAudioBuffer` logs for one buffer argument.
pub(crate) struct AudioBufferNames {
    pub(crate) buffer: &'static str,
    pub(crate) num_channels: &'static str,
    pub(crate) num_samples: &'static str,
    pub(crate) data: &'static str,
    pub(crate) last_sample: &'static str,
}

/// The [`AudioBufferNames`] of an argument named `$name`.
macro_rules! buffer_names {
    ($name:literal) => {
        &$crate::validation::AudioBufferNames {
            buffer: $name,
            num_channels: concat!($name, "->numChannels"),
            num_samples: concat!($name, "->numSamples"),
            data: concat!($name, "->data"),
            last_sample: concat!($name, "->data[iChannel][", $name, "->numSamples - 1]"),
        }
    };
}
pub(crate) use buffer_names;

/// Sends `message` to the log callback, or to standard output without one (upstream also sends it to the debugger on
/// Windows).
pub(crate) fn log(callback: IPLLogFunction, level: IPLLogLevel, message: &str) {
    match callback {
        Some(callback) => {
            let message = CString::new(message.replace('\0', "")).unwrap_or_default();
            // SAFETY: the caller registered this callback for log messages; the string outlives the call.
            unsafe { callback(level, message.as_ptr()) };
        }
        None => print!("{message}"),
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Struct validators (upstream's VALIDATE_IPL* macros). Each takes the validator, the value and the expression upstream
// passes to its macro, as a string literal.
// --------------------------------------------------------------------------------------------------------------------

/// `VALIDATE_IPLVector3`.
macro_rules! vector3 {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: $crate::types::IPLVector3 = $value;
        $v.float(concat!($name, ".x"), value.x);
        $v.float(concat!($name, ".y"), value.y);
        $v.float(concat!($name, ".z"), value.z);
    }};
}
pub(crate) use vector3;

/// `VALIDATE_IPLMatrix4x4`.
macro_rules! matrix4x4 {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: &$crate::types::IPLMatrix4x4 = &$value;
        for row in &value.elements {
            for &element in row {
                $v.float(concat!($name, ".elements[iRow][iColumn]"), element);
            }
        }
    }};
}
pub(crate) use matrix4x4;

/// `VALIDATE_IPLCoordinateSpace3`.
macro_rules! coordinate_space3 {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: &$crate::types::IPLCoordinateSpace3 = &$value;
        $crate::validation::vector3!($v, value.origin, concat!($name, ".origin"));
        $crate::validation::vector3!($v, value.right, concat!($name, ".right"));
        $crate::validation::vector3!($v, value.up, concat!($name, ".up"));
        $crate::validation::vector3!($v, value.ahead, concat!($name, ".ahead"));
    }};
}
pub(crate) use coordinate_space3;

/// `VALIDATE_IPLSphere`.
macro_rules! sphere {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: $crate::types::IPLSphere = $value;
        $crate::validation::vector3!($v, value.center, concat!($name, ".center"));
        $v.float_if(concat!($name, ".radius"), value.radius, value.radius >= 0.0);
    }};
}
pub(crate) use sphere;

/// `VALIDATE_IPLSpeakerLayout`.
///
/// Safety: `speakers` is null or valid for `numSpeakers` vectors when the type is custom.
macro_rules! speaker_layout {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: &$crate::types::IPLSpeakerLayout = &$value;
        $v.range(
            "IPLSpeakerLayoutType",
            concat!($name, ".type"),
            value.r#type,
            $crate::types::IPL_SPEAKERLAYOUTTYPE_MONO,
            $crate::types::IPL_SPEAKERLAYOUTTYPE_CUSTOM,
        );
        if value.r#type == $crate::types::IPL_SPEAKERLAYOUTTYPE_CUSTOM {
            $v.int(
                concat!($name, ".numSpeakers"),
                value.numSpeakers,
                value.numSpeakers > 0,
            );
            if $v.pointer(concat!($name, ".speakers"), value.speakers) {
                for i in 0..usize::try_from(value.numSpeakers).unwrap_or(0) {
                    // SAFETY: guaranteed by the caller of the API function.
                    let speaker = unsafe { *value.speakers.add(i) };
                    $crate::validation::vector3!(
                        $v,
                        speaker,
                        concat!($name, ".speakers[iSpeaker]")
                    );
                }
            }
        }
    }};
}

/// `VALIDATE_IPLAudioSettings`.
macro_rules! audio_settings {
    ($v:expr, $value:expr, $name:literal) => {{
        let value: *const $crate::types::IPLAudioSettings = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            $v.int(
                concat!($name, "->samplingRate"),
                value.samplingRate,
                value.samplingRate > 0,
            );
            $v.int(
                concat!($name, "->frameSize"),
                value.frameSize,
                value.frameSize > 0,
            );
        }
    }};
}
pub(crate) use audio_settings;

/// `VALIDATE_IPLBakedDataIdentifier`.
macro_rules! baked_data_identifier {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: &$crate::types::IPLBakedDataIdentifier = &$value;
        $v.range(
            "IPLBakedDataType",
            concat!($name, ".type"),
            value.r#type,
            $crate::types::IPL_BAKEDDATATYPE_REFLECTIONS,
            $crate::types::IPL_BAKEDDATATYPE_PATHING,
        );
        $v.range(
            "IPLBakedDataVariation",
            concat!($name, ".variation"),
            value.variation,
            $crate::types::IPL_BAKEDDATAVARIATION_REVERB,
            $crate::types::IPL_BAKEDDATAVARIATION_DYNAMIC,
        );
        $crate::validation::sphere!(
            $v,
            value.endpointInfluence,
            concat!($name, ".endpointInfluence")
        );
    }};
}
pub(crate) use baked_data_identifier;

/// `VALIDATE_IPLDirectEffectParams`.
macro_rules! direct_effect_params {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: *const $crate::types::IPLDirectEffectParams = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            $v.flags(
                "IPLDirectEffectFlags",
                concat!($name, "->flags"),
                value.flags,
                $crate::types::IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION
                    | $crate::types::IPL_DIRECTEFFECTFLAGS_APPLYAIRABSORPTION
                    | $crate::types::IPL_DIRECTEFFECTFLAGS_APPLYDIRECTIVITY
                    | $crate::types::IPL_DIRECTEFFECTFLAGS_APPLYOCCLUSION
                    | $crate::types::IPL_DIRECTEFFECTFLAGS_APPLYTRANSMISSION,
            );
            $v.range(
                "IPLTransmissionType",
                concat!($name, "->transmissionType"),
                value.transmissionType,
                $crate::types::IPL_TRANSMISSIONTYPE_FREQINDEPENDENT,
                $crate::types::IPL_TRANSMISSIONTYPE_FREQDEPENDENT,
            );
            let unit = |x: f32| (0.0..=1.0).contains(&x);
            $v.float_if(
                concat!($name, "->distanceAttenuation"),
                value.distanceAttenuation,
                unit(value.distanceAttenuation),
            );
            for &x in &value.airAbsorption {
                $v.float_if(concat!($name, "->airAbsorption[iBand]"), x, unit(x));
            }
            $v.float_if(
                concat!($name, "->directivity"),
                value.directivity,
                unit(value.directivity),
            );
            $v.float_if(
                concat!($name, "->occlusion"),
                value.occlusion,
                unit(value.occlusion),
            );
            for &x in &value.transmission {
                $v.float_if(concat!($name, "->transmission[iBand]"), x, unit(x));
            }
        }
    }};
}
pub(crate) use direct_effect_params;

/// `VALIDATE_IPLReflectionEffectParams`.
macro_rules! reflection_effect_params {
    ($v:expr, $value:expr, $name:expr) => {{
        use $crate::types::*;
        let value: *const IPLReflectionEffectParams = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            $v.range(
                "IPLReflectionEffectType",
                concat!($name, "->type"),
                value.r#type,
                IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
                IPL_REFLECTIONEFFECTTYPE_TAN,
            );
            let t = value.r#type;
            if t == IPL_REFLECTIONEFFECTTYPE_CONVOLUTION || t == IPL_REFLECTIONEFFECTTYPE_HYBRID {
                $v.pointer(concat!($name, "->ir"), value.ir);
                $v.int(
                    concat!($name, "->numChannels"),
                    value.numChannels,
                    value.numChannels > 0,
                );
                $v.int(concat!($name, "->irSize"), value.irSize, value.irSize > 0);
            }
            if t == IPL_REFLECTIONEFFECTTYPE_PARAMETRIC || t == IPL_REFLECTIONEFFECTTYPE_HYBRID {
                for &x in &value.reverbTimes {
                    $v.float_if(concat!($name, "->reverbTimes[iBand]"), x, x > 0.0);
                }
            }
            if t == IPL_REFLECTIONEFFECTTYPE_HYBRID {
                for &x in &value.eq {
                    $v.float_if(concat!($name, "->eq[iBand]"), x, 0.0 < x && x <= 1.0);
                }
                $v.int(concat!($name, "->delay"), value.delay, value.delay > 0);
            }
            if t == IPL_REFLECTIONEFFECTTYPE_TAN {
                $v.pointer(concat!($name, "->tanDevice"), value.tanDevice);
                $v.int(
                    concat!($name, "->tanSlot"),
                    value.tanSlot,
                    value.tanSlot >= 0,
                );
            }
        }
    }};
}
pub(crate) use reflection_effect_params;

/// `VALIDATE_IPLPathEffectParams`.
///
/// Safety: `shCoeffs` is valid for `(order + 1)²` floats.
macro_rules! path_effect_params {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: *const $crate::types::IPLPathEffectParams = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            for &x in &value.eqCoeffs {
                $v.float_if(concat!($name, "->eqCoeffs[iBand]"), x, 0.0 < x && x <= 1.0);
            }
            $v.int(concat!($name, "->order"), value.order, value.order >= 0);
            let num_coeffs = (i64::from(value.order) + 1).pow(2);
            if !value.shCoeffs.is_null() {
                for i in 0..usize::try_from(num_coeffs).unwrap_or(0) {
                    // SAFETY: guaranteed by the caller of the API function.
                    $v.float(concat!($name, "->shCoeffs[iCoeff]"), unsafe {
                        *value.shCoeffs.add(i)
                    });
                }
            }
            $v.bool(concat!($name, "->binaural"), value.binaural);
            if value.binaural != $crate::types::IPL_FALSE {
                $v.pointer(concat!($name, "->hrtf"), value.hrtf);
                $crate::validation::coordinate_space3!(
                    $v,
                    value.listener,
                    concat!($name, "->listener")
                );
            }
        }
    }};
}
pub(crate) use path_effect_params;

/// `VALIDATE_IPLDistanceAttenuationModel`.
macro_rules! distance_attenuation_model {
    ($v:expr, $value:expr, $name:expr) => {{
        use $crate::types::*;
        let value: *const IPLDistanceAttenuationModel = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            $v.range(
                "IPLDistanceAttenuationModelType",
                concat!($name, "->type"),
                value.r#type,
                IPL_DISTANCEATTENUATIONTYPE_DEFAULT,
                IPL_DISTANCEATTENUATIONTYPE_CALLBACK,
            );
            if value.r#type == IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE {
                $v.float_if(
                    concat!($name, "->minDistance"),
                    value.minDistance,
                    value.minDistance > 0.0,
                );
            } else if value.r#type == IPL_DISTANCEATTENUATIONTYPE_CALLBACK {
                $v.pointer(
                    concat!($name, "->callback"),
                    value.callback.map_or(std::ptr::null(), |f| f as *const ()),
                );
            }
            $v.bool(concat!($name, "->dirty"), value.dirty);
        }
    }};
}
pub(crate) use distance_attenuation_model;

/// `VALIDATE_IPLAirAbsorptionModel`.
macro_rules! air_absorption_model {
    ($v:expr, $value:expr, $name:expr) => {{
        use $crate::types::*;
        let value: *const IPLAirAbsorptionModel = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            $v.range(
                "IPLAirAbsorptionModelType",
                concat!($name, "->type"),
                value.r#type,
                IPL_AIRABSORPTIONTYPE_DEFAULT,
                IPL_AIRABSORPTIONTYPE_CALLBACK,
            );
            if value.r#type == IPL_AIRABSORPTIONTYPE_EXPONENTIAL {
                // SAFETY: an array of IPL_NUM_BANDS floats inside the model.
                unsafe {
                    $v.float_array(
                        concat!($name, "->coefficients"),
                        concat!($name, "->coefficients[iArray]"),
                        value.coefficients.as_ptr(),
                        $crate::validation::NUM_BANDS as i32,
                    );
                }
            } else if value.r#type == IPL_AIRABSORPTIONTYPE_CALLBACK {
                $v.pointer(
                    concat!($name, "->callback"),
                    value.callback.map_or(std::ptr::null(), |f| f as *const ()),
                );
            }
            $v.bool(concat!($name, "->dirty"), value.dirty);
        }
    }};
}
pub(crate) use air_absorption_model;

/// `VALIDATE_IPLDirectivity`.
macro_rules! directivity {
    ($v:expr, $value:expr, $name:expr) => {{
        let value: *const $crate::types::IPLDirectivity = $value;
        if $v.pointer($name, value) {
            // SAFETY: not null; valid as guaranteed by the caller of the API function.
            let value = unsafe { &*value };
            $v.float_if(
                concat!($name, "->dipoleWeight"),
                value.dipoleWeight,
                (0.0..=1.0).contains(&value.dipoleWeight),
            );
            $v.float_if(
                concat!($name, "->dipolePower"),
                value.dipolePower,
                value.dipolePower >= 0.0,
            );
        }
    }};
}
pub(crate) use directivity;

// --------------------------------------------------------------------------------------------------------------------
// Effect settings and parameters
// --------------------------------------------------------------------------------------------------------------------

/// Effect settings that `CValidatedContext::create*` validates.
pub(crate) trait ValidateSettings {
    /// Upstream's factory method (`createPanningEffect`, …).
    const METHOD: &'static str;

    /// Validates the settings argument, named `effectSettings`.
    ///
    /// # Safety
    ///
    /// `value` is null or valid, with the arrays it points to.
    unsafe fn validate(v: Validator, value: *const Self);
}

/// Effect parameters that the validated effects' `apply` validates.
pub(crate) trait ValidateParams {
    /// Validates the parameters argument, named `params`.
    ///
    /// # Safety
    ///
    /// `value` is null or valid, with the arrays it points to.
    unsafe fn validate(v: Validator, value: *const Self);
}

/// Implements [`ValidateSettings`] or [`ValidateParams`]: `$body` sees the non-null value as `$value`.
macro_rules! validate_struct {
    (settings $ty:ty, $method:literal, |$v:ident, $value:ident| $body:block) => {
        impl ValidateSettings for $ty {
            const METHOD: &'static str = $method;

            unsafe fn validate($v: Validator, value: *const Self) {
                if $v.pointer("effectSettings", value) {
                    // SAFETY: not null; valid as guaranteed by the caller.
                    let $value = unsafe { &*value };
                    $body
                }
            }
        }
    };
    (params $ty:ty, |$v:ident, $value:ident| $body:block) => {
        impl ValidateParams for $ty {
            unsafe fn validate($v: Validator, value: *const Self) {
                if $v.pointer("params", value) {
                    // SAFETY: not null; valid as guaranteed by the caller.
                    let $value = unsafe { &*value };
                    $body
                }
            }
        }
    };
}

validate_struct!(settings IPLPanningEffectSettings, "createPanningEffect", |v, s| {
    speaker_layout!(v, s.speakerLayout, "effectSettings->speakerLayout");
});

validate_struct!(params IPLPanningEffectParams, |v, p| {
    vector3!(v, p.direction, "params->direction");
});

validate_struct!(settings IPLBinauralEffectSettings, "createBinauralEffect", |v, s| {
    v.pointer("effectSettings->hrtf", s.hrtf);
});

validate_struct!(params IPLBinauralEffectParams, |v, p| {
    vector3!(v, p.direction, "params->direction");
    v.range(
        "IPLHRTFInterpolation",
        "params->interpolation",
        p.interpolation,
        IPL_HRTFINTERPOLATION_NEAREST,
        IPL_HRTFINTERPOLATION_BILINEAR,
    );
    v.float_if(
        "params->spatialBlend",
        p.spatialBlend,
        (0.0..=1.0).contains(&p.spatialBlend),
    );
    v.pointer("params->hrtf", p.hrtf);
});

validate_struct!(settings IPLVirtualSurroundEffectSettings, "createVirtualSurroundEffect", |v, s| {
    speaker_layout!(v, s.speakerLayout, "effectSettings->speakerLayout");
    v.pointer("effectSettings->hrtf", s.hrtf);
});

validate_struct!(params IPLVirtualSurroundEffectParams, |v, p| {
    v.pointer("params->hrtf", p.hrtf);
});

validate_struct!(settings IPLAmbisonicsEncodeEffectSettings, "createAmbisonicsEncodeEffect", |v, s| {
    v.int("effectSettings->maxOrder", s.maxOrder, s.maxOrder >= 0);
});

validate_struct!(params IPLAmbisonicsEncodeEffectParams, |v, p| {
    vector3!(v, p.direction, "params->direction");
    v.int("params->order", p.order, p.order >= 0);
});

validate_struct!(settings IPLAmbisonicsPanningEffectSettings, "createAmbisonicsPanningEffect", |v, s| {
    speaker_layout!(v, s.speakerLayout, "effectSettings->speakerLayout");
    v.int("effectSettings->maxOrder", s.maxOrder, s.maxOrder >= 0);
});

validate_struct!(params IPLAmbisonicsPanningEffectParams, |v, p| {
    v.int("params->order", p.order, p.order >= 0);
});

validate_struct!(settings IPLAmbisonicsBinauralEffectSettings, "createAmbisonicsBinauralEffect", |v, s| {
    v.pointer("effectSettings->hrtf", s.hrtf);
    v.int("effectSettings->maxOrder", s.maxOrder, s.maxOrder >= 0);
});

validate_struct!(params IPLAmbisonicsBinauralEffectParams, |v, p| {
    v.pointer("params->hrtf", p.hrtf);
    v.int("params->order", p.order, p.order >= 0);
});

validate_struct!(settings IPLAmbisonicsRotationEffectSettings, "createAmbisonicsRotationEffect", |v, s| {
    v.int("effectSettings->maxOrder", s.maxOrder, s.maxOrder >= 0);
});

validate_struct!(params IPLAmbisonicsRotationEffectParams, |v, p| {
    coordinate_space3!(v, p.orientation, "params->orientation");
    v.int("params->order", p.order, p.order >= 0);
});

validate_struct!(settings IPLAmbisonicsDecodeEffectSettings, "createAmbisonicsDecodeEffect", |v, s| {
    speaker_layout!(v, s.speakerLayout, "effectSettings->speakerLayout");
    v.int("effectSettings->maxOrder", s.maxOrder, s.maxOrder >= 0);
});

validate_struct!(params IPLAmbisonicsDecodeEffectParams, |v, p| {
    v.int("params->order", p.order, p.order >= 0);
    // Upstream checks the HRTF even when not decoding binaurally.
    v.pointer("params->hrtf", p.hrtf);
    coordinate_space3!(v, p.orientation, "params->orientation");
    v.bool("params->binaural", p.binaural);
});

validate_struct!(settings IPLDirectEffectSettings, "createDirectEffect", |v, s| {
    v.int("effectSettings->numChannels", s.numChannels, s.numChannels > 0);
});

impl ValidateParams for IPLDirectEffectParams {
    unsafe fn validate(v: Validator, value: *const Self) {
        direct_effect_params!(v, value, "params");
    }
}

/// `VALIDATE_IPLReflectionEffectSettings`, for the reflection effect and the mixer.
///
/// # Safety
///
/// `value` is null or valid.
pub(crate) unsafe fn reflection_effect_settings(
    v: Validator,
    value: *const IPLReflectionEffectSettings,
) {
    if v.pointer("effectSettings", value) {
        // SAFETY: not null; valid as guaranteed by the caller.
        let s = unsafe { &*value };
        v.range(
            "IPLReflectionEffectType",
            "effectSettings->type",
            s.r#type,
            IPL_REFLECTIONEFFECTTYPE_CONVOLUTION,
            IPL_REFLECTIONEFFECTTYPE_TAN,
        );
        v.int("effectSettings->irSize", s.irSize, s.irSize > 0);
        v.int(
            "effectSettings->numChannels",
            s.numChannels,
            s.numChannels > 0,
        );
    }
}

validate_struct!(settings IPLPathEffectSettings, "createPathEffect", |v, s| {
    v.int("effectSettings->maxOrder", s.maxOrder, s.maxOrder >= 0);
    v.bool("effectSettings->spatialize", s.spatialize);
    if s.spatialize != IPL_FALSE {
        speaker_layout!(v, s.speakerLayout, "effectSettings->speakerLayout");
    }
});

impl ValidateParams for IPLPathEffectParams {
    unsafe fn validate(v: Validator, value: *const Self) {
        path_effect_params!(v, value, "params");
    }
}

/// The checks of a validated effect's `apply` around the call: the parameters and buffers before, the result and the
/// output after.
///
/// # Safety
///
/// The pointers are null or valid, with the arrays they point to.
pub(crate) unsafe fn validate_apply<P: ValidateParams>(
    v: Option<Validator>,
    params: *const P,
    input: *const IPLAudioBuffer,
    output: *const IPLAudioBuffer,
    apply: impl FnOnce() -> IPLAudioEffectState,
) -> IPLAudioEffectState {
    if let Some(v) = v {
        // SAFETY: guaranteed by the caller.
        unsafe {
            P::validate(v, params);
            v.audio_buffer(buffer_names!("in"), input, true);
            v.audio_buffer(buffer_names!("out"), output, false);
        }
    }
    let result = apply();
    if let Some(v) = v {
        audio_effect_state(v, result);
        // SAFETY: guaranteed by the caller.
        unsafe { v.audio_buffer(buffer_names!("out"), output, true) };
    }
    result
}

/// `VALIDATE_IPLAudioEffectState(result)`.
pub(crate) fn audio_effect_state(v: Validator, result: IPLAudioEffectState) {
    v.range(
        "IPLAudioEffectState",
        "result",
        result,
        IPL_AUDIOEFFECTSTATE_TAILREMAINING,
        IPL_AUDIOEFFECTSTATE_TAILCOMPLETE,
    );
}
