//! std-only error type for `infer-moe`, mirroring `infer-topo`'s `TopoError`.

use std::fmt;

pub type Result<T> = std::result::Result<T, MoeError>;

/// MoE config / routing-input validation error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeError(pub(crate) String);

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MoeError {}

macro_rules! bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err($crate::error::MoeError(format!($($arg)*)))
    };
}

pub(crate) use bail;
