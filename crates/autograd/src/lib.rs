#[path = "adamw_state.rs"]
pub mod adamw_state;
#[path = "backend.rs"]
pub mod backend;
#[cfg(feature = "cuda")]
#[path = "backend_cuda.rs"]
pub mod backend_cuda;
#[cfg(feature = "metal")]
#[path = "backend_metal.rs"]
pub mod backend_metal;
#[path = "lr_schedule.rs"]
pub mod lr_schedule;
#[path = "module.rs"]
pub mod module;
#[path = "ops.rs"]
pub mod ops;
#[path = "optim.rs"]
pub mod optim;
#[cfg(feature = "safetensors")]
#[path = "safetensors_io.rs"]
pub mod safetensors_io;
#[path = "tape.rs"]
pub mod tape;
#[path = "tensor.rs"]
pub mod tensor;

#[cfg(feature = "metal")]
pub use backend::MlxHandle;
pub use backend::{Backend, CpuBackend, Device, DeviceHandle};
pub use lr_schedule::{ConstantLr, CosineWithWarmup, LinearWarmup, LrSchedule, parse_lr_schedule};
pub use optim::{AdamW, Optimizer};
#[cfg(feature = "safetensors")]
pub use safetensors_io::SafetensorsRegistry;
pub use tape::{BackwardOp, SavedContext, Tape, TapeEntry};
pub use tensor::{Tensor, TensorId, TensorStore};

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
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("gradient shape mismatch for tensor {tensor_id}: expected {expected:?}, got {got:?}")]
    GradientShapeMismatch {
        tensor_id: TensorId,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("missing gradient for tensor {0}")]
    MissingGradient(TensorId),
    #[error("axis {axis} is out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("invalid rank {got}, expected {expected}")]
    InvalidRank { expected: &'static str, got: usize },
    #[error("index {index} is out of bounds for upper bound {upper}")]
    IndexOutOfBounds { index: usize, upper: usize },
    #[error("invalid indices length: expected {expected}, got {got}")]
    InvalidIndicesLen { expected: usize, got: usize },
    #[error("{0}")]
    TapeInvariant(&'static str),
}

pub type Result<T> = std::result::Result<T, AutogradError>;
