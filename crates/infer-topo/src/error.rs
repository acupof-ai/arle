//! std-only error type for `infer-topo`.
//!
//! The legacy `infer/src/tensor_parallel.rs` used `anyhow::Result` + `bail!`.
//! This foundation crate is std-only, so we carry a tiny owned-message error
//! that implements [`std::error::Error`]. The error *messages* are ported
//! verbatim from the legacy `bail!` strings so callers (and tests) that match
//! on substrings behave identically.

use std::fmt;

/// Crate-local result alias mirroring the legacy `anyhow::Result` signature.
pub type Result<T> = std::result::Result<T, TopoError>;

/// A topology/sharding configuration error.
///
/// Carries an owned message string. This is intentionally simple — the only
/// error paths in this crate are config validation and out-of-range ranks,
/// none of which need structured fields or a source chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopoError(pub(crate) String);

impl TopoError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

    /// The error message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TopoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TopoError {}

/// `bail!`-equivalent: build a [`TopoError`] from a format string and return early.
macro_rules! bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err($crate::error::TopoError::new(format!($($arg)*)))
    };
}

pub(crate) use bail;
