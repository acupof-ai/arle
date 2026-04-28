use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub type Half = u16;

#[path = "ffi/attention.rs"]
pub mod attention;
#[path = "ffi/elementwise.rs"]
pub mod elementwise;
#[path = "ffi/embedding.rs"]
pub mod embedding;
#[path = "ffi/gemm.rs"]
pub mod gemm;
#[path = "ffi/kv.rs"]
pub mod kv;
#[path = "ffi/misc.rs"]
pub mod misc;
#[cfg(feature = "nccl")]
#[path = "ffi/nccl.rs"]
pub mod nccl;
#[path = "ffi/norm.rs"]
pub mod norm;
#[path = "ffi/quant.rs"]
pub mod quant;
#[path = "ffi/recurrent.rs"]
pub mod recurrent;
#[path = "ffi/sampling.rs"]
pub mod sampling;

pub use attention::*;
pub use elementwise::*;
pub use embedding::*;
pub use gemm::*;
pub use kv::*;
pub use misc::*;
pub use norm::*;
pub use quant::*;
pub use recurrent::*;
pub use sampling::*;
