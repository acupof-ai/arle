//! std-only error type for `infer-moe`.
//!
//! The legacy DSv4 router (`crates/deepseek-spec/src/v4.rs`) used a
//! `DeepSeekConfigError::InvalidForwardBatch(String)` variant; the Qwen3.6
//! router relied on `anyhow::ensure!`. This foundation crate is std-only, so we
//! carry a tiny owned-message error that implements [`std::error::Error`],
//! mirroring `infer-topo`'s `TopoError`.

use std::fmt;

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, MoeError>;

/// A MoE config / routing-input validation error.
///
/// Carries an owned message string. The only fallible paths in this crate are
/// config validation and routing-input shape/finiteness checks, none of which
/// need structured fields or a source chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeError(pub(crate) String);

impl MoeError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

    /// The error message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MoeError {}

/// `bail!`-equivalent: build a [`MoeError`] from a format string and return early.
macro_rules! bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err($crate::error::MoeError::new(format!($($arg)*)))
    };
}

pub(crate) use bail;
