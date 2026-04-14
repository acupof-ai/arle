//! Convenience re-exports of the CUDA backend types most commonly imported
//! by `model/` and `ops/`.
//!
//! Importing `crate::backend::cuda::prelude::*` (or a named subset of it) is
//! the preferred single entry point for model code that talks to the CUDA
//! backend. It keeps the consumer's import block stable when the underlying
//! module layout changes.

pub(crate) use super::flashinfer::FlashInferDecodeMetadata;
pub(crate) use super::paged_kv::{PagedKVPool, TokenKVPool};
pub(crate) use super::tensor::{
    DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, RawDevicePtr,
};
