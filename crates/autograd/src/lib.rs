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
#[path = "grad_clip.rs"]
pub mod grad_clip;
#[path = "lr_schedule.rs"]
pub mod lr_schedule;
#[path = "ops.rs"]
pub mod ops;
#[path = "optim.rs"]
pub mod optim;
// CLI-driven runtime toggles; no env reads (env-conformance).
#[path = "runtime_flags.rs"]
mod runtime_flags;
pub use runtime_flags::{AutogradRuntimeFlags, TapePrecision, apply_runtime_flags};
#[cfg(feature = "safetensors")]
#[path = "safetensors_io.rs"]
pub mod safetensors_io;
#[path = "tape.rs"]
pub mod tape;
#[path = "tensor.rs"]
pub mod tensor;

#[cfg(feature = "metal")]
pub use backend::MlxHandle;
pub use backend::{Backend, CommAxis, CpuBackend, Device, DeviceHandle};
pub use lr_schedule::{CosineWithWarmup, LrSchedule};
pub use optim::{AdamW, Optimizer};
#[cfg(feature = "safetensors")]
pub use safetensors_io::SafetensorsRegistry;
pub use tape::{BackwardOp, BackwardOpProfile, BackwardProfile, SavedContext, Tape, TapeEntry};
pub use tensor::{TapeDtype, Tensor, TensorId, TensorStore};

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
    #[error("cuda allocation failed in {op}: shape {shape:?}, bytes {bytes}")]
    CudaAllocFailed {
        op: &'static str,
        shape: Vec<usize>,
        bytes: usize,
    },
    #[error("{0}")]
    TapeInvariant(&'static str),
}

pub type Result<T> = std::result::Result<T, AutogradError>;
