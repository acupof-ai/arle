//! Shared argv-parsing helpers for `crates/train/src/bin/*` binaries.
//!
//! Each binary carries its own `CliError` enum because their error surfaces
//! differ (some wrap `Qwen3Error`, some don't; most but not all have a
//! `Custom(String)` variant). What's truly shared is the mechanical work of
//! pulling the next value from the argv iterator and parsing it via
//! `FromStr`. This module owns those helpers plus a narrow `ArgError` that
//! each binary folds into its own `CliError` via
//! `#[error(transparent)] Arg(#[from] ArgError)`.
//!
//! Error message strings here match the pre-extraction wording verbatim so
//! the refactor is strictly behavior-preserving.

use std::{
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use autograd::{Backend, CpuBackend, Device, optim::AdamW};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArgError {
    #[error("unknown flag {0}")]
    UnknownFlag(String),
    #[error("missing value for flag {0}")]
    MissingValue(String),
    #[error("invalid value for {flag}: {value}")]
    InvalidValue { flag: String, value: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Cpu,
    Metal,
    Cuda,
}

impl BackendChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn build_backend_or_cpu(
        self,
        job: &str,
    ) -> Result<Arc<dyn Backend>, autograd::AutogradError> {
        match self {
            Self::Cpu => Ok(Arc::new(CpuBackend)),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
            #[cfg(not(feature = "metal"))]
            Self::Metal => {
                eprintln!(
                    "[{job}] warning: metal backend requested without --features metal; falling back to cpu"
                );
                Ok(Arc::new(CpuBackend))
            }
            #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
            Self::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
            #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
            Self::Cuda => {
                eprintln!(
                    "[{job}] warning: cuda backend requested without --features cuda; falling back to cpu"
                );
                Ok(Arc::new(CpuBackend))
            }
        }
    }

    pub fn build_backend_or_arg_error(self, flag: &str) -> Result<Arc<dyn Backend>, ArgError> {
        match self {
            Self::Cpu => Ok(Arc::new(CpuBackend)),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
            #[cfg(not(feature = "metal"))]
            Self::Metal => Err(ArgError::InvalidValue {
                flag: flag.into(),
                value: "metal (build with --features metal)".into(),
            }),
            #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
            Self::Cuda => Ok(Arc::new(
                autograd::backend_cuda::CudaBackend::new(0).map_err(|_| {
                    ArgError::InvalidValue {
                        flag: flag.into(),
                        value: "cuda backend init failed".into(),
                    }
                })?,
            )),
            #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
            Self::Cuda => Err(ArgError::InvalidValue {
                flag: flag.into(),
                value: "cuda (build with --features cuda and no no-cuda)".into(),
            }),
        }
    }
}

impl FromStr for BackendChoice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unknown backend: {value}")),
        }
    }
}

/// Use the device-backed AdamW path only on Metal, where `Backend::adamw_step`
/// is overridden to stay device-resident. CPU keeps the host path, and CUDA
/// also stays host-backed until it has a real device-side AdamW override.
pub fn adamw_for_backend(
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    wd: f32,
    backend: Arc<dyn Backend>,
) -> AdamW {
    match backend.device() {
        Device::Metal => AdamW::new_with_device(lr, betas, eps, wd, backend),
        Device::Cpu | Device::Cuda => AdamW::new(lr, betas, eps, wd),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveDtype {
    F32,
    Bf16,
}

impl SaveDtype {
    pub fn torch_dtype(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::Bf16 => "bfloat16",
        }
    }
}

impl FromStr for SaveDtype {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            _ => Err(format!("unknown save dtype: {value}")),
        }
    }
}

pub fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, ArgError> {
    iter.next()
        .ok_or_else(|| ArgError::MissingValue(flag.to_string()))
}

pub fn parse_value<T: FromStr>(flag: &str, value: String) -> Result<T, ArgError> {
    value.parse::<T>().map_err(|_| ArgError::InvalidValue {
        flag: flag.to_string(),
        value,
    })
}

pub fn canonicalize_path(flag: &str, path: &Path) -> Result<PathBuf, ArgError> {
    path.canonicalize().map_err(|_| ArgError::InvalidValue {
        flag: flag.to_string(),
        value: path.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_value_returns_next_token() {
        let mut iter = vec!["42".to_string()].into_iter();
        assert_eq!(next_value(&mut iter, "--n").unwrap(), "42");
    }

    #[test]
    fn next_value_missing_reports_flag() {
        let mut iter = std::iter::empty::<String>();
        let err = next_value(&mut iter, "--missing").unwrap_err();
        assert!(
            matches!(&err, ArgError::MissingValue(f) if f == "--missing"),
            "got {err:?}"
        );
        assert_eq!(err.to_string(), "missing value for flag --missing");
    }

    #[test]
    fn parse_value_ok() {
        let n: usize = parse_value("--steps", "7".to_string()).unwrap();
        assert_eq!(n, 7);
    }

    #[test]
    fn parse_value_invalid_reports_flag_and_value() {
        let err: ArgError = parse_value::<usize>("--steps", "nope".to_string()).unwrap_err();
        match &err {
            ArgError::InvalidValue { flag, value } => {
                assert_eq!(flag, "--steps");
                assert_eq!(value, "nope");
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
        assert_eq!(err.to_string(), "invalid value for --steps: nope");
    }

    #[test]
    fn unknown_flag_display_matches_legacy_wording() {
        let err = ArgError::UnknownFlag("--bogus".to_string());
        assert_eq!(err.to_string(), "unknown flag --bogus");
    }

    #[test]
    fn backend_choice_parser_accepts_supported_values() {
        assert_eq!("cpu".parse::<BackendChoice>().unwrap(), BackendChoice::Cpu);
        assert_eq!(
            "metal".parse::<BackendChoice>().unwrap(),
            BackendChoice::Metal
        );
        assert_eq!(
            "cuda".parse::<BackendChoice>().unwrap(),
            BackendChoice::Cuda
        );
    }

    #[test]
    fn save_dtype_parser_accepts_supported_values() {
        assert_eq!("f32".parse::<SaveDtype>().unwrap(), SaveDtype::F32);
        assert_eq!("bf16".parse::<SaveDtype>().unwrap(), SaveDtype::Bf16);
    }

    #[test]
    fn adamw_for_backend_keeps_cpu_host_backed() {
        let optim = adamw_for_backend(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0, Arc::new(CpuBackend));
        assert!(!optim.is_device_backed(), "cpu helper must keep host AdamW");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn adamw_for_backend_uses_device_path_on_metal() {
        let optim = adamw_for_backend(
            1.0e-3,
            (0.9, 0.999),
            1.0e-8,
            0.0,
            Arc::new(autograd::backend_metal::MetalBackend),
        );
        assert!(
            optim.is_device_backed(),
            "metal helper must enable device-backed AdamW"
        );
    }
}
