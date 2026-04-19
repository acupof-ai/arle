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
