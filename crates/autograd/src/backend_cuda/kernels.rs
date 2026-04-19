use crate::{AutogradError, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchArgs, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(not(feature = "no-cuda"))]
const ELEMENTWISE_CU: &str = include_str!("kernels/elementwise.cu");
#[cfg(not(feature = "no-cuda"))]
const SOFTMAX_CU: &str = include_str!("kernels/softmax.cu");
#[cfg(not(feature = "no-cuda"))]
const SILU_CU: &str = include_str!("kernels/silu.cu");
#[cfg(not(feature = "no-cuda"))]
const RMS_NORM_CU: &str = include_str!("kernels/rms_norm.cu");
#[cfg(not(feature = "no-cuda"))]
const EMBEDDING_CU: &str = include_str!("kernels/embedding.cu");
#[cfg(not(feature = "no-cuda"))]
const REDUCE_CU: &str = include_str!("kernels/reduce.cu");
#[cfg(not(feature = "no-cuda"))]
const ROPE_CU: &str = include_str!("kernels/rope.cu");
#[cfg(not(feature = "no-cuda"))]
const GATHER_CU: &str = include_str!("kernels/gather.cu");

#[cfg(not(feature = "no-cuda"))]
const FUNCTION_NAMES: &[&str] = &[
    "add_f32",
    "mul_f32",
    "mul_scalar_f32",
    "gelu_f32",
    "exp_f32",
    "neg_f32",
    "softmax_last_axis_f32",
    "log_softmax_last_axis_f32",
    "silu_f32",
    "rms_norm_f32",
    "embedding_f32",
    "sum_last_axis_f32",
    "mean_last_axis_f32",
    "rope_f32",
    "gather_last_dim_f32",
];

#[derive(Debug)]
pub(super) struct KernelCache {
    _module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl KernelCache {
    pub(super) fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = ctx;
            todo!("GPU required: cuda kernel compilation is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let ptx = compile_ptx(concat_sources()).map_err(|_| {
                AutogradError::TapeInvariant("nvrtc compile_ptx failed for autograd kernels")
            })?;
            let module = ctx.load_module(ptx).map_err(|_| {
                AutogradError::TapeInvariant("cuda load_module failed for autograd kernels")
            })?;
            let functions = FUNCTION_NAMES
                .iter()
                .map(|&name| {
                    module
                        .load_function(name)
                        .map(|function| (name, function))
                        .map_err(|_| {
                            AutogradError::TapeInvariant(
                                "cuda load_function failed for autograd kernel",
                            )
                        })
                })
                .collect::<Result<HashMap<_, _>>>()?;
            Ok(Self {
                _module: module,
                functions,
            })
        }
    }

    pub(super) fn function(&self, name: &'static str) -> Result<&CudaFunction> {
        self.functions.get(name).ok_or(AutogradError::TapeInvariant(
            "autograd cuda kernel not found in cache",
        ))
    }
}

pub(super) fn launch_rows<F>(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    rows: usize,
    block: u32,
    shared_bytes: u32,
    build_args: F,
) -> Result<()>
where
    F: FnOnce(&mut LaunchArgs<'_>),
{
    #[cfg(feature = "no-cuda")]
    {
        let _ = (stream, func, rows, block, shared_bytes, build_args);
        todo!("GPU required: cuda kernel launch is unavailable under feature no-cuda")
    }

    #[cfg(not(feature = "no-cuda"))]
    {
        if rows == 0 {
            return Ok(());
        }
        let grid_x = u32::try_from(rows)
            .map_err(|_| AutogradError::TapeInvariant("cuda launch rows exceeds u32"))?;
        let mut launch_args = stream.launch_builder(func);
        build_args(&mut launch_args);
        // Safety: caller controls the kernel symbol + argument order, and all
        // device buffers outlive the asynchronous launch.
        unsafe {
            launch_args
                .launch(LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: shared_bytes,
                })
                .map_err(|_| AutogradError::TapeInvariant("cuda kernel launch failed"))?;
        }
        Ok(())
    }
}

pub(super) fn launch_1d<F>(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    n: usize,
    build_args: F,
) -> Result<()>
where
    F: FnOnce(&mut LaunchArgs<'_>),
{
    #[cfg(feature = "no-cuda")]
    {
        let _ = (stream, func, n, build_args);
        todo!("GPU required: cuda kernel launch is unavailable under feature no-cuda")
    }

    #[cfg(not(feature = "no-cuda"))]
    {
        if n == 0 {
            return Ok(());
        }

        let grid_x = u32::try_from(n.div_ceil(256))
            .map_err(|_| AutogradError::TapeInvariant("cuda launch grid exceeds u32"))?;
        let mut launch_args = stream.launch_builder(func);
        build_args(&mut launch_args);
        // Safety: caller controls the kernel symbol + argument order, and all
        // device buffers outlive the asynchronous launch.
        unsafe {
            launch_args
                .launch(LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|_| AutogradError::TapeInvariant("cuda kernel launch failed"))?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "no-cuda"))]
fn concat_sources() -> String {
    let mut src = String::new();
    for chunk in [
        ELEMENTWISE_CU,
        SOFTMAX_CU,
        SILU_CU,
        RMS_NORM_CU,
        EMBEDDING_CU,
        REDUCE_CU,
        ROPE_CU,
        GATHER_CU,
    ] {
        src.push_str(chunk);
        src.push('\n');
    }
    src
}
