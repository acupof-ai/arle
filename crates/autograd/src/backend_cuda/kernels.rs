use crate::{AutogradError, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchArgs, LaunchConfig, sys,
};
use cudarc::nvrtc::{Ptx, result as nvrtc_result, sys as nvrtc_sys};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
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
const SCATTER_ADD_CU: &str = include_str!("kernels/scatter_add.cu");
#[cfg(not(feature = "no-cuda"))]
const ADD_BROADCAST_CU: &str = include_str!("kernels/add_broadcast.cu");
#[cfg(not(feature = "no-cuda"))]
const LAYOUT_CU: &str = include_str!("kernels/layout.cu");
#[cfg(not(feature = "no-cuda"))]
const ADAMW_CU: &str = include_str!("kernels/adamw.cu");
#[cfg(not(feature = "no-cuda"))]
const LOG_SOFTMAX_BACKWARD_CU: &str = include_str!("kernels/log_softmax_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const GATHER_BACKWARD_CU: &str = include_str!("kernels/gather_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const ADD_INTO_CU: &str = include_str!("kernels/add_into.cu");
#[cfg(not(feature = "no-cuda"))]
const MEAN_BACKWARD_CU: &str = include_str!("kernels/mean_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const MUL_SCALAR_BACKWARD_CU: &str = include_str!("kernels/mul_scalar_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const EMBEDDING_BACKWARD_CU: &str = include_str!("kernels/embedding_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const ADD_BROADCAST_BACKWARD_CU: &str = include_str!("kernels/add_broadcast_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const ACTIVATION_BACKWARD_CU: &str = include_str!("kernels/activation_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const MUL_BACKWARD_CU: &str = include_str!("kernels/mul_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const RMS_NORM_BACKWARD_CU: &str = include_str!("kernels/rms_norm_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const ROPE_BACKWARD_CU: &str = include_str!("kernels/rope_backward.cu");
#[cfg(not(feature = "no-cuda"))]
const ROLLOUT_CU: &str = include_str!("kernels/rollout.cu");
#[cfg(not(feature = "no-cuda"))]
const ATTENTION_CU: &str = include_str!("kernels/attention.cu");
#[cfg(not(feature = "no-cuda"))]
const BRIDGE_CU: &str = include_str!("kernels/bridge.cu");

#[cfg(not(feature = "no-cuda"))]
const FUNCTION_NAMES: &[&str] = &[
    "add_f32",
    "mul_f32",
    "mul_scalar_f32",
    "sigmoid_f32",
    "gelu_f32",
    "exp_f32",
    "neg_f32",
    "softmax_last_axis_f32",
    "log_softmax_last_axis_f32",
    "softmax_last_axis_backward_f32",
    "silu_f32",
    "rms_norm_f32",
    "embedding_f32",
    "embedding_bf16_to_f32",
    "sum_squares_partial_f32",
    "grad_clip_sumsq_f32",
    "grad_clip_scale_f32",
    "sum_last_axis_f32",
    "mean_last_axis_f32",
    "rope_f32",
    "gather_last_dim_f32",
    "scatter_add_rows_f32",
    "add_broadcast_f32",
    "transpose_axes_swap_f32",
    "slice_f32",
    "concat_axis2_f32",
    "kv_cache_write_axis2_f32",
    "slice_backward_f32",
    "adamw_step_f32",
    "log_softmax_last_axis_backward_f32",
    "gather_last_dim_backward_f32",
    "add_into_f32",
    "mean_backward_f32",
    "mul_scalar_backward_f32",
    "embedding_backward_f32",
    "add_broadcast_backward_f32",
    "silu_backward_f32",
    "gelu_backward_f32",
    "sigmoid_backward_f32",
    "exp_backward_f32",
    "mul_backward_lhs_f32",
    "mul_backward_rhs_f32",
    "rms_norm_inv_rms_f32",
    "rms_norm_backward_x_f32",
    "rms_norm_backward_w_f32",
    "rope_backward_f32",
    "argmax_last_dim_f32",
    "embedding_f32_ids_f32",
    "embedding_bf16_ids_f32",
    "write_scalar_at_f32",
    "causal_sdpa_decode_gqa_f32",
    "causal_sdpa_decode_gqa_cache_f32",
    "qwen_decode_prepare_q_f32",
    "qwen_decode_prepare_q_gated_f32",
    "qwen_decode_prepare_kv_f32",
    "bf16_bits_to_f32",
    "f32_to_bf16_bits",
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
            let (image, arch) = compile_cubin_for_current_device(ctx)?;
            let module = ctx.load_module(image).map_err(|err| {
                cuda_kernel_error(format!(
                    "cuda load_module failed for autograd kernels arch={arch}: {err:?}"
                ))
            })?;
            let functions = FUNCTION_NAMES
                .iter()
                .map(|&name| {
                    module
                        .load_function(name)
                        .map(|function| (name, function))
                        .map_err(|err| {
                            cuda_kernel_error(format!(
                                "cuda load_function failed for autograd kernel {name}: {err:?}"
                            ))
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

#[cfg(not(feature = "no-cuda"))]
fn cuda_kernel_error(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}

#[cfg(not(feature = "no-cuda"))]
fn compile_cubin_for_current_device(ctx: &Arc<CudaContext>) -> Result<(Ptx, &'static str)> {
    let arch = current_sm_arch(ctx)?;
    // Emit SASS cubin for the exact device instead of PTX. On V100 the
    // deployment driver supports CUDA 12.2 while the available NVRTC is 12.4;
    // PTX 8.4 would fail driver JIT with CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
    // but an sm_70 cubin loads cleanly and keeps the kernel code uniform.
    let image = compile_cubin(&concat_sources(), arch).map_err(|err| {
        cuda_kernel_error(format!(
            "nvrtc compile cubin failed for autograd kernels arch={arch}: {err}"
        ))
    })?;
    Ok((image, arch))
}

#[cfg(not(feature = "no-cuda"))]
fn current_sm_arch(ctx: &Arc<CudaContext>) -> Result<&'static str> {
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|err| {
            cuda_kernel_error(format!(
                "cuda device attribute COMPUTE_CAPABILITY_MAJOR failed: {err:?}"
            ))
        })?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|err| {
            cuda_kernel_error(format!(
                "cuda device attribute COMPUTE_CAPABILITY_MINOR failed: {err:?}"
            ))
        })?;
    sm_arch(major, minor)
}

#[cfg(not(feature = "no-cuda"))]
fn sm_arch(major: i32, minor: i32) -> Result<&'static str> {
    match (major, minor) {
        (7, 0) => Ok("sm_70"),
        (7, 5) => Ok("sm_75"),
        (8, 0) => Ok("sm_80"),
        (8, 6) => Ok("sm_86"),
        (8, 7) => Ok("sm_87"),
        (8, 9) => Ok("sm_89"),
        (9, 0) => Ok("sm_90"),
        (10, 0) => Ok("sm_100"),
        (10, 1) => Ok("sm_101"),
        (12, 0) => Ok("sm_120"),
        _ => Err(cuda_kernel_error(format!(
            "unsupported cuda compute capability for autograd kernels: sm_{major}{minor}"
        ))),
    }
}

#[cfg(not(feature = "no-cuda"))]
fn compile_cubin(src: &str, arch: &'static str) -> Result<Ptx> {
    let program = NvrtcProgram::create(src, "arle_autograd_kernels.cu")?;
    let options = [format!("--gpu-architecture={arch}")];
    unsafe { nvrtc_result::compile_program(program.raw(), &options) }.map_err(|err| {
        cuda_kernel_error(format!(
            "nvrtc compile_program failed arch={arch} err={err:?} log={}",
            program.log()
        ))
    })?;
    let cubin = get_cubin(program.raw()).map_err(|err| {
        cuda_kernel_error(format!(
            "nvrtc get cubin failed arch={arch} err={err:?} log={}",
            program.log()
        ))
    })?;
    Ok(Ptx::from_binary(cubin))
}

#[cfg(not(feature = "no-cuda"))]
struct NvrtcProgram {
    prog: nvrtc_sys::nvrtcProgram,
    _src: CString,
    _name: CString,
}

#[cfg(not(feature = "no-cuda"))]
impl NvrtcProgram {
    fn create(src: &str, name: &str) -> Result<Self> {
        let src = CString::new(src.as_bytes())
            .map_err(|_| cuda_kernel_error("autograd cuda source contains NUL".to_string()))?;
        let name = CString::new(name.as_bytes()).map_err(|_| {
            cuda_kernel_error("autograd cuda program name contains NUL".to_string())
        })?;
        let prog = nvrtc_result::create_program(&src, Some(&name)).map_err(|err| {
            cuda_kernel_error(format!(
                "nvrtc create_program failed for autograd kernels: {err:?}"
            ))
        })?;
        Ok(Self {
            prog,
            _src: src,
            _name: name,
        })
    }

    fn raw(&self) -> nvrtc_sys::nvrtcProgram {
        self.prog
    }

    fn log(&self) -> String {
        unsafe { nvrtc_result::get_program_log(self.prog) }
            .ok()
            .and_then(|raw| unsafe {
                CStr::from_ptr(raw.as_ptr())
                    .to_str()
                    .ok()
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "<no nvrtc log>".to_string())
    }
}

#[cfg(not(feature = "no-cuda"))]
impl Drop for NvrtcProgram {
    fn drop(&mut self) {
        if !self.prog.is_null() {
            unsafe {
                let _ = nvrtc_result::destroy_program(self.prog);
            }
        }
    }
}

#[cfg(not(feature = "no-cuda"))]
fn get_cubin(
    prog: nvrtc_sys::nvrtcProgram,
) -> std::result::Result<Vec<u8>, nvrtc_result::NvrtcError> {
    let mut size = 0usize;
    unsafe {
        nvrtc_sys::nvrtcGetCUBINSize(prog, &mut size as *mut _).result()?;
    }

    let mut cubin = vec![0u8; size];
    unsafe {
        nvrtc_sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()).result()?;
    }
    Ok(cubin)
}

pub(super) fn launch_rows<'a, F>(
    stream: &'a Arc<CudaStream>,
    func: &'a CudaFunction,
    rows: usize,
    block: u32,
    shared_bytes: u32,
    build_args: F,
) -> Result<()>
where
    F: FnOnce(LaunchArgs<'a>) -> LaunchArgs<'a>,
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
        let mut launch_args = build_args(stream.launch_builder(func));
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

pub(super) fn launch_1d<'a, F>(
    stream: &'a Arc<CudaStream>,
    func: &'a CudaFunction,
    n: usize,
    build_args: F,
) -> Result<()>
where
    F: FnOnce(LaunchArgs<'a>) -> LaunchArgs<'a>,
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
        let mut launch_args = build_args(stream.launch_builder(func));
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
        SCATTER_ADD_CU,
        ADD_BROADCAST_CU,
        LAYOUT_CU,
        ADAMW_CU,
        LOG_SOFTMAX_BACKWARD_CU,
        GATHER_BACKWARD_CU,
        ADD_INTO_CU,
        MEAN_BACKWARD_CU,
        MUL_SCALAR_BACKWARD_CU,
        EMBEDDING_BACKWARD_CU,
        ADD_BROADCAST_BACKWARD_CU,
        ACTIVATION_BACKWARD_CU,
        MUL_BACKWARD_CU,
        RMS_NORM_BACKWARD_CU,
        ROPE_BACKWARD_CU,
        ROLLOUT_CU,
        ATTENTION_CU,
        BRIDGE_CU,
    ] {
        src.push_str(chunk);
        src.push('\n');
    }
    src
}
