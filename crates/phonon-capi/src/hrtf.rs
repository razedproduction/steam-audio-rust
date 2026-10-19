//! HRTFs (upstream `api_hrtf.cpp`).

use std::sync::Mutex;

#[cfg(feature = "sofa")]
use steam_audio::Error;
use steam_audio::hrtf::{HrtfDatabase, HrtfNormalization, HrtfSettings, HrtfSource, loudness};

use crate::context::context_arc;
use crate::guard;
use crate::handle::{Handle, retain_release};
use crate::types::*;

pub(crate) type Hrtf = Mutex<HrtfDatabase>;

/// Creates an HRTF: the built-in one, or one loaded from a SOFA file or from SOFA data in memory (the file name wins
/// when both are given). `sofaData`, `sofaDataSize` and `volume` are read for callers of API version 4.2 or later,
/// `normType` for 4.3 or later.
///
/// Returns `IPL_STATUS_FAILURE` for invalid settings and for a SOFA file that cannot be loaded or has delays, and
/// `IPL_STATUS_INITIALIZATION` for SOFA data that cannot be loaded or an unknown HRTF type.
///
/// # Safety
///
/// `context` is null or a live context; the settings pointers are null or valid, and so are the file name (a
/// NUL-terminated string) and the SOFA data (`sofaDataSize` bytes) when used.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplHRTFCreate(
    context: IPLContext,
    audioSettings: *mut IPLAudioSettings,
    hrtfSettings: *mut IPLHRTFSettings,
    hrtf: *mut IPLHRTF,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(audio), Some(settings), Some(out)) = (
            unsafe { audioSettings.as_ref() },
            unsafe { hrtfSettings.as_ref() },
            unsafe { hrtf.as_mut() },
        ) else {
            return IPL_STATUS_FAILURE;
        };
        if audio.samplingRate <= 0 || audio.frameSize <= 0 {
            return IPL_STATUS_FAILURE;
        }

        // Like upstream, these checks read `sofaData` and `sofaDataSize` whatever the caller's version, but the
        // loader only gets them (and `volume`, `normType`) from the versions that added them.
        if settings.r#type == IPL_HRTFTYPE_SOFA {
            if settings.sofaFileName.is_null() && settings.sofaData.is_null() {
                return IPL_STATUS_FAILURE;
            }
            if !settings.sofaData.is_null() && settings.sofaDataSize == 0 {
                return IPL_STATUS_FAILURE;
            }
        }

        let at_least_4_2 = context.is_caller_api_version_at_least(4, 2);
        let at_least_4_3 = context.is_caller_api_version_at_least(4, 3);
        let sofa_data = if at_least_4_2 {
            settings.sofaData
        } else {
            std::ptr::null()
        };
        let sofa_data_size = if at_least_4_2 {
            settings.sofaDataSize
        } else {
            0
        };

        let volume = if at_least_4_2 {
            loudness::gain_to_db(settings.volume)
        } else {
            0.0
        };
        let normalization = if at_least_4_3 && settings.normType == IPL_HRTFNORMTYPE_RMS {
            HrtfNormalization::Rms
        } else {
            HrtfNormalization::None
        };

        // SAFETY: guaranteed by the caller.
        let result = unsafe {
            load(
                settings,
                sofa_data,
                sofa_data_size,
                volume,
                normalization,
                audio,
            )
        };
        match result {
            Ok(database) => {
                *out = Handle::into_raw(Mutex::new(database), Some(context));
                IPL_STATUS_SUCCESS
            }
            Err(status) => status,
        }
    })
}

/// Builds the database, mapping errors to upstream's status codes.
///
/// # Safety
///
/// As [`iplHRTFCreate`].
#[cfg_attr(not(feature = "sofa"), allow(unused_variables))]
unsafe fn load(
    settings: &IPLHRTFSettings,
    sofa_data: *const u8,
    sofa_data_size: i32,
    volume: f32,
    normalization: HrtfNormalization,
    audio: &IPLAudioSettings,
) -> Result<HrtfDatabase, IPLerror> {
    let sampling_rate = audio.samplingRate as u32;
    let frame_size = audio.frameSize as usize;
    let database = |source| {
        let settings = HrtfSettings {
            source,
            volume,
            normalization,
        };
        HrtfDatabase::new(&settings, sampling_rate, frame_size)
    };

    match settings.r#type {
        IPL_HRTFTYPE_DEFAULT => {
            database(HrtfSource::BuiltIn).map_err(|_| IPL_STATUS_INITIALIZATION)
        }
        #[cfg(feature = "sofa")]
        IPL_HRTFTYPE_SOFA => {
            if !settings.sofaFileName.is_null() {
                // SAFETY: guaranteed by the caller.
                let name = unsafe { std::ffi::CStr::from_ptr(settings.sofaFileName) };
                let path = sofa_path(name).ok_or(IPL_STATUS_FAILURE)?;
                database(HrtfSource::SofaFile(&path)).map_err(|error| match error {
                    Error::Sofa(_) | Error::UnsupportedSofaDelay => IPL_STATUS_FAILURE,
                    _ => IPL_STATUS_INITIALIZATION,
                })
            } else if sofa_data.is_null() {
                // Neither a file name nor data (data is not read before API version 4.2).
                Err(IPL_STATUS_INITIALIZATION)
            } else {
                // A negative size is passed to libmysofa as an `int` upstream, which fails to load it.
                let size =
                    usize::try_from(sofa_data_size).map_err(|_| IPL_STATUS_INITIALIZATION)?;
                // SAFETY: guaranteed by the caller.
                let data = unsafe { std::slice::from_raw_parts(sofa_data, size) };
                database(HrtfSource::SofaData(data)).map_err(|error| match error {
                    Error::UnsupportedSofaDelay => IPL_STATUS_FAILURE,
                    _ => IPL_STATUS_INITIALIZATION,
                })
            }
        }
        _ => Err(IPL_STATUS_INITIALIZATION),
    }
}

/// The file name as a path: bytes on Unix, UTF-8 elsewhere (upstream passes the `char*` to `fopen`).
#[cfg(feature = "sofa")]
fn sofa_path(name: &std::ffi::CStr) -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(std::ffi::OsStr::from_bytes(name.to_bytes()).into())
    }
    #[cfg(not(unix))]
    {
        name.to_str().ok().map(Into::into)
    }
}

retain_release!(IPLHRTF, Hrtf, iplHRTFRetain, iplHRTFRelease);

/// The HRTF behind a handle.
///
/// # Safety
///
/// `hrtf` is null or a live HRTF.
pub(crate) unsafe fn hrtf_ref<'a>(hrtf: IPLHRTF) -> Option<&'a Hrtf> {
    // SAFETY: guaranteed by the caller.
    unsafe { Handle::<Hrtf>::from_raw(hrtf) }.map(|handle| &**handle.get())
}
