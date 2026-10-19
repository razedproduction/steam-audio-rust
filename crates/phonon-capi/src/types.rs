//! The types of `phonon.h`, with the same names and layouts.
//!
//! C enums are `int`-sized integers with named constants rather than Rust enums, because a caller may pass any value.
//! Opaque handles are pointers to zero-sized marker structs, like `DECLARE_OPAQUE_HANDLE`.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_char, c_void};

// --------------------------------------------------------------------------------------------------------------------
// Data types
// --------------------------------------------------------------------------------------------------------------------

pub type IPLint8 = i8;
pub type IPLuint8 = u8;
pub type IPLint16 = i16;
pub type IPLuint16 = u16;
pub type IPLint32 = i32;
pub type IPLuint32 = u32;
pub type IPLint64 = i64;
pub type IPLuint64 = u64;
pub type IPLfloat32 = f32;
pub type IPLfloat64 = f64;
pub type IPLbyte = u8;
pub type IPLsize = usize;
pub type IPLstring = *const c_char;

/// `IPL_NUM_BANDS` (upstream's default build, without `IPL_ENABLE_OCTAVE_BANDS`).
pub const IPL_NUM_BANDS: usize = 3;

pub type IPLbool = i32;
pub const IPL_FALSE: IPLbool = 0;
pub const IPL_TRUE: IPLbool = 1;

pub type IPLerror = i32;
pub const IPL_STATUS_SUCCESS: IPLerror = 0;
pub const IPL_STATUS_FAILURE: IPLerror = 1;
pub const IPL_STATUS_OUTOFMEMORY: IPLerror = 2;
pub const IPL_STATUS_INITIALIZATION: IPLerror = 3;

pub type IPLProgressCallback =
    Option<unsafe extern "system" fn(progress: IPLfloat32, user_data: *mut c_void)>;

// --------------------------------------------------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------------------------------------------------

/// The API version this library implements (`STEAMAUDIO_VERSION_*`, from `project(Phonon VERSION 4.8.1)` in
/// `core/CMakeLists.txt`).
pub const STEAMAUDIO_VERSION_MAJOR: u32 = 4;
pub const STEAMAUDIO_VERSION_MINOR: u32 = 8;
pub const STEAMAUDIO_VERSION_PATCH: u32 = 1;
pub const STEAMAUDIO_VERSION: u32 =
    (STEAMAUDIO_VERSION_MAJOR << 16) | (STEAMAUDIO_VERSION_MINOR << 8) | STEAMAUDIO_VERSION_PATCH;

pub type IPLLogLevel = i32;
pub const IPL_LOGLEVEL_INFO: IPLLogLevel = 0;
pub const IPL_LOGLEVEL_WARNING: IPLLogLevel = 1;
pub const IPL_LOGLEVEL_ERROR: IPLLogLevel = 2;
pub const IPL_LOGLEVEL_DEBUG: IPLLogLevel = 3;

pub type IPLSIMDLevel = i32;
pub const IPL_SIMDLEVEL_SSE2: IPLSIMDLevel = 0;
pub const IPL_SIMDLEVEL_SSE4: IPLSIMDLevel = 1;
pub const IPL_SIMDLEVEL_AVX: IPLSIMDLevel = 2;
pub const IPL_SIMDLEVEL_AVX2: IPLSIMDLevel = 3;
pub const IPL_SIMDLEVEL_AVX512: IPLSIMDLevel = 4;
pub const IPL_SIMDLEVEL_NEON: IPLSIMDLevel = IPL_SIMDLEVEL_SSE2;

pub type IPLContextFlags = i32;
pub const IPL_CONTEXTFLAGS_VALIDATION: IPLContextFlags = 1 << 0;
pub const IPL_CONTEXTFLAGS_FORCE_32BIT: IPLContextFlags = 0x7fff_ffff;

pub type IPLLogFunction =
    Option<unsafe extern "system" fn(level: IPLLogLevel, message: *const c_char)>;
pub type IPLAllocateFunction =
    Option<unsafe extern "system" fn(size: IPLsize, alignment: IPLsize) -> *mut c_void>;
pub type IPLFreeFunction = Option<unsafe extern "system" fn(memory_block: *mut c_void)>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLContextSettings {
    pub version: IPLuint32,
    pub logCallback: IPLLogFunction,
    pub allocateCallback: IPLAllocateFunction,
    pub freeCallback: IPLFreeFunction,
    pub simdLevel: IPLSIMDLevel,
    pub flags: IPLContextFlags,
}

// --------------------------------------------------------------------------------------------------------------------
// Geometry
// --------------------------------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IPLVector3 {
    pub x: IPLfloat32,
    pub y: IPLfloat32,
    pub z: IPLfloat32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IPLMatrix4x4 {
    /// Row-major.
    pub elements: [[IPLfloat32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IPLBox {
    pub minCoordinates: IPLVector3,
    pub maxCoordinates: IPLVector3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IPLSphere {
    pub center: IPLVector3,
    pub radius: IPLfloat32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IPLCoordinateSpace3 {
    pub right: IPLVector3,
    pub up: IPLVector3,
    pub ahead: IPLVector3,
    pub origin: IPLVector3,
}

// --------------------------------------------------------------------------------------------------------------------
// Serialization
// --------------------------------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLSerializedObjectSettings {
    pub data: *mut IPLbyte,
    pub size: IPLsize,
}

// --------------------------------------------------------------------------------------------------------------------
// Audio buffers
// --------------------------------------------------------------------------------------------------------------------

pub type IPLSpeakerLayoutType = i32;
pub const IPL_SPEAKERLAYOUTTYPE_MONO: IPLSpeakerLayoutType = 0;
pub const IPL_SPEAKERLAYOUTTYPE_STEREO: IPLSpeakerLayoutType = 1;
pub const IPL_SPEAKERLAYOUTTYPE_QUADRAPHONIC: IPLSpeakerLayoutType = 2;
pub const IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1: IPLSpeakerLayoutType = 3;
pub const IPL_SPEAKERLAYOUTTYPE_SURROUND_7_1: IPLSpeakerLayoutType = 4;
pub const IPL_SPEAKERLAYOUTTYPE_CUSTOM: IPLSpeakerLayoutType = 5;

pub type IPLAmbisonicsType = i32;
pub const IPL_AMBISONICSTYPE_N3D: IPLAmbisonicsType = 0;
pub const IPL_AMBISONICSTYPE_SN3D: IPLAmbisonicsType = 1;
pub const IPL_AMBISONICSTYPE_FUMA: IPLAmbisonicsType = 2;

pub type IPLAudioEffectState = i32;
pub const IPL_AUDIOEFFECTSTATE_TAILREMAINING: IPLAudioEffectState = 0;
pub const IPL_AUDIOEFFECTSTATE_TAILCOMPLETE: IPLAudioEffectState = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLSpeakerLayout {
    pub r#type: IPLSpeakerLayoutType,
    pub numSpeakers: IPLint32,
    pub speakers: *mut IPLVector3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IPLAudioSettings {
    pub samplingRate: IPLint32,
    pub frameSize: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAudioBuffer {
    pub numChannels: IPLint32,
    pub numSamples: IPLint32,
    /// One pointer per channel.
    pub data: *mut *mut IPLfloat32,
}

// --------------------------------------------------------------------------------------------------------------------
// HRTFs
// --------------------------------------------------------------------------------------------------------------------

pub type IPLHRTFType = i32;
pub const IPL_HRTFTYPE_DEFAULT: IPLHRTFType = 0;
pub const IPL_HRTFTYPE_SOFA: IPLHRTFType = 1;

pub type IPLHRTFNormType = i32;
pub const IPL_HRTFNORMTYPE_NONE: IPLHRTFNormType = 0;
pub const IPL_HRTFNORMTYPE_RMS: IPLHRTFNormType = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLHRTFSettings {
    pub r#type: IPLHRTFType,
    pub sofaFileName: *const c_char,
    pub sofaData: *const IPLuint8,
    pub sofaDataSize: i32,
    pub volume: f32,
    pub normType: IPLHRTFNormType,
}

// --------------------------------------------------------------------------------------------------------------------
// Effects
// --------------------------------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLPanningEffectSettings {
    pub speakerLayout: IPLSpeakerLayout,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLPanningEffectParams {
    pub direction: IPLVector3,
}

pub type IPLHRTFInterpolation = i32;
pub const IPL_HRTFINTERPOLATION_NEAREST: IPLHRTFInterpolation = 0;
pub const IPL_HRTFINTERPOLATION_BILINEAR: IPLHRTFInterpolation = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLBinauralEffectSettings {
    pub hrtf: IPLHRTF,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLBinauralEffectParams {
    pub direction: IPLVector3,
    pub interpolation: IPLHRTFInterpolation,
    pub spatialBlend: IPLfloat32,
    pub hrtf: IPLHRTF,
    /// Two floats (left, right), or null.
    pub peakDelays: *mut IPLfloat32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLVirtualSurroundEffectSettings {
    pub speakerLayout: IPLSpeakerLayout,
    pub hrtf: IPLHRTF,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLVirtualSurroundEffectParams {
    pub hrtf: IPLHRTF,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLAmbisonicsEncodeEffectSettings {
    pub maxOrder: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLAmbisonicsEncodeEffectParams {
    pub direction: IPLVector3,
    pub order: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAmbisonicsPanningEffectSettings {
    pub speakerLayout: IPLSpeakerLayout,
    pub maxOrder: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLAmbisonicsPanningEffectParams {
    pub order: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAmbisonicsBinauralEffectSettings {
    pub hrtf: IPLHRTF,
    pub maxOrder: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAmbisonicsBinauralEffectParams {
    pub hrtf: IPLHRTF,
    pub order: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLAmbisonicsRotationEffectSettings {
    pub maxOrder: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLAmbisonicsRotationEffectParams {
    pub orientation: IPLCoordinateSpace3,
    pub order: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAmbisonicsDecodeEffectSettings {
    pub speakerLayout: IPLSpeakerLayout,
    pub hrtf: IPLHRTF,
    pub maxOrder: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAmbisonicsDecodeEffectParams {
    pub order: IPLint32,
    pub hrtf: IPLHRTF,
    pub orientation: IPLCoordinateSpace3,
    pub binaural: IPLbool,
}

pub type IPLDirectEffectFlags = i32;
pub const IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION: IPLDirectEffectFlags = 1 << 0;
pub const IPL_DIRECTEFFECTFLAGS_APPLYAIRABSORPTION: IPLDirectEffectFlags = 1 << 1;
pub const IPL_DIRECTEFFECTFLAGS_APPLYDIRECTIVITY: IPLDirectEffectFlags = 1 << 2;
pub const IPL_DIRECTEFFECTFLAGS_APPLYOCCLUSION: IPLDirectEffectFlags = 1 << 3;
pub const IPL_DIRECTEFFECTFLAGS_APPLYTRANSMISSION: IPLDirectEffectFlags = 1 << 4;

pub type IPLTransmissionType = i32;
pub const IPL_TRANSMISSIONTYPE_FREQINDEPENDENT: IPLTransmissionType = 0;
pub const IPL_TRANSMISSIONTYPE_FREQDEPENDENT: IPLTransmissionType = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLDirectEffectSettings {
    pub numChannels: IPLint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IPLDirectEffectParams {
    pub flags: IPLDirectEffectFlags,
    pub transmissionType: IPLTransmissionType,
    pub distanceAttenuation: IPLfloat32,
    pub airAbsorption: [IPLfloat32; IPL_NUM_BANDS],
    pub directivity: IPLfloat32,
    pub occlusion: IPLfloat32,
    pub transmission: [IPLfloat32; IPL_NUM_BANDS],
}

// --------------------------------------------------------------------------------------------------------------------
// Simulation models
// --------------------------------------------------------------------------------------------------------------------

pub type IPLDistanceAttenuationModelType = i32;
pub const IPL_DISTANCEATTENUATIONTYPE_DEFAULT: IPLDistanceAttenuationModelType = 0;
pub const IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE: IPLDistanceAttenuationModelType = 1;
pub const IPL_DISTANCEATTENUATIONTYPE_CALLBACK: IPLDistanceAttenuationModelType = 2;

pub type IPLAirAbsorptionModelType = i32;
pub const IPL_AIRABSORPTIONTYPE_DEFAULT: IPLAirAbsorptionModelType = 0;
pub const IPL_AIRABSORPTIONTYPE_EXPONENTIAL: IPLAirAbsorptionModelType = 1;
pub const IPL_AIRABSORPTIONTYPE_CALLBACK: IPLAirAbsorptionModelType = 2;

pub type IPLDistanceAttenuationCallback =
    Option<unsafe extern "system" fn(distance: IPLfloat32, user_data: *mut c_void) -> f32>;
pub type IPLAirAbsorptionCallback = Option<
    unsafe extern "system" fn(distance: IPLfloat32, band: IPLint32, user_data: *mut c_void) -> f32,
>;
pub type IPLDirectivityCallback =
    Option<unsafe extern "system" fn(direction: IPLVector3, user_data: *mut c_void) -> f32>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLDistanceAttenuationModel {
    pub r#type: IPLDistanceAttenuationModelType,
    pub minDistance: IPLfloat32,
    pub callback: IPLDistanceAttenuationCallback,
    pub userData: *mut c_void,
    pub dirty: IPLbool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLAirAbsorptionModel {
    pub r#type: IPLAirAbsorptionModelType,
    pub coefficients: [IPLfloat32; IPL_NUM_BANDS],
    pub callback: IPLAirAbsorptionCallback,
    pub userData: *mut c_void,
    pub dirty: IPLbool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IPLDirectivity {
    pub dipoleWeight: IPLfloat32,
    pub dipolePower: IPLfloat32,
    pub callback: IPLDirectivityCallback,
    pub userData: *mut c_void,
}

// --------------------------------------------------------------------------------------------------------------------
// Opaque handles
// --------------------------------------------------------------------------------------------------------------------

macro_rules! opaque_handles {
    ($($handle:ident => $target:ident;)+) => {
        $(
            #[repr(C)]
            #[derive(Debug)]
            pub struct $target {
                _private: [u8; 0],
            }

            pub type $handle = *mut $target;
        )+
    };
}

opaque_handles! {
    IPLContext => _IPLContext_t;
    IPLSerializedObject => _IPLSerializedObject_t;
    IPLHRTF => _IPLHRTF_t;
    IPLPanningEffect => _IPLPanningEffect_t;
    IPLBinauralEffect => _IPLBinauralEffect_t;
    IPLVirtualSurroundEffect => _IPLVirtualSurroundEffect_t;
    IPLAmbisonicsEncodeEffect => _IPLAmbisonicsEncodeEffect_t;
    IPLAmbisonicsPanningEffect => _IPLAmbisonicsPanningEffect_t;
    IPLAmbisonicsBinauralEffect => _IPLAmbisonicsBinauralEffect_t;
    IPLAmbisonicsRotationEffect => _IPLAmbisonicsRotationEffect_t;
    IPLAmbisonicsDecodeEffect => _IPLAmbisonicsDecodeEffect_t;
    IPLDirectEffect => _IPLDirectEffect_t;
}
