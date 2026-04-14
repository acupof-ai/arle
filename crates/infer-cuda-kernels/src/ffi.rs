use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub(crate) type Half = u16;

#[path = "ffi/attention.rs"]
pub(crate) mod attention;
#[path = "ffi/elementwise.rs"]
pub(crate) mod elementwise;
#[path = "ffi/embedding.rs"]
pub(crate) mod embedding;
#[path = "ffi/gemm.rs"]
pub(crate) mod gemm;
#[path = "ffi/kv.rs"]
pub(crate) mod kv;
#[path = "ffi/misc.rs"]
pub(crate) mod misc;
#[path = "ffi/norm.rs"]
pub(crate) mod norm;
#[path = "ffi/quant.rs"]
pub(crate) mod quant;
#[path = "ffi/recurrent.rs"]
pub(crate) mod recurrent;
#[path = "ffi/sampling.rs"]
pub(crate) mod sampling;

pub(crate) use attention::*;
pub(crate) use elementwise::*;
pub(crate) use embedding::*;
pub(crate) use gemm::*;
pub(crate) use kv::*;
pub(crate) use misc::*;
pub(crate) use norm::*;
pub(crate) use quant::*;
pub(crate) use recurrent::*;
pub(crate) use sampling::*;
