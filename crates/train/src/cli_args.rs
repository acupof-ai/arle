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

use std::str::FromStr;

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
}
