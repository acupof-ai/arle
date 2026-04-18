#[path = "ops.rs"]
pub mod ops;
#[path = "tape.rs"]
pub mod tape;
#[path = "tensor.rs"]
pub mod tensor;

pub use tape::{BackwardOp, SavedContext, Tape, TapeEntry};
pub use tensor::{GpuTensor, TensorId, TensorStore};

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AutogradError {
    #[error("tensor data length {len} does not match shape {shape:?} (size {size})")]
    DataLengthMismatch {
        len: usize,
        shape: Vec<usize>,
        size: usize,
    },
    #[error("invalid tensor id {0}")]
    InvalidTensorId(TensorId),
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("gradient shape mismatch for tensor {tensor_id}: expected {expected:?}, got {got:?}")]
    GradientShapeMismatch {
        tensor_id: TensorId,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("missing gradient for tensor {0}")]
    MissingGradient(TensorId),
    #[error("{0}")]
    TapeInvariant(&'static str),
}

pub type Result<T> = std::result::Result<T, AutogradError>;
