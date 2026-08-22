//! std-only error type for `infer-topo`. Messages are ported verbatim from the
//! legacy `bail!` strings so substring-matching callers behave identically.

use std::fmt;

pub type Result<T> = std::result::Result<T, TopoError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopoError(pub(crate) String);

impl TopoError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

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

macro_rules! bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err($crate::error::TopoError::new(format!($($arg)*)))
    };
}

pub(crate) use bail;
