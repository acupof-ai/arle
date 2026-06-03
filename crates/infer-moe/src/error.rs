//! std-only error type for `infer-moe`, mirroring `infer-topo`'s `TopoError`.

use std::fmt;

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, MoeError>;

/// A MoE config / routing-input validation error (an owned message string).
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
