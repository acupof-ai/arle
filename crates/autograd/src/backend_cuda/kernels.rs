use super::NvrtcIdentity;
use crate::{AutogradError, Result, TapeDtype};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchArgs, LaunchConfig, sys,
};
use cudarc::nvrtc::{Ptx, result as nvrtc_result, sys as nvrtc_sys};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::Arc;
#[cfg(not(feature = "no-cuda"))]
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFamily {
    Forward,
    Backward,
    /// Fused SDPA fwd+bwd plus the KV-cache/slice layout kernels feeding it.
    Attention,
    /// Rollout decode helpers and quantized-weight dequant for rollout.
    Rollout,
    Optimizer,
    /// bf16<->f32 bit casts bridging tape and serving tensors.
    Bridge,
    /// Pending numerical-contract proof against the serving launcher.
    Uncertain,
}

/// Order is load-bearing: it fixes both the concatenated compile unit and the
/// identity hash.
#[cfg(not(feature = "no-cuda"))]
const KERNEL_SOURCES: &[(&str, KernelFamily, &str)] = &[
    (
        "elementwise.cu",
        KernelFamily::Forward,
        include_str!("kernels/elementwise.cu"),
    ),
    (
        "softmax.cu",
        KernelFamily::Forward,
        include_str!("kernels/softmax.cu"),
    ),
    (
        "silu.cu",
        KernelFamily::Forward,
        include_str!("kernels/silu.cu"),
    ),
    (
        "rms_norm.cu",
        KernelFamily::Forward,
        include_str!("kernels/rms_norm.cu"),
    ),
    (
        "embedding.cu",
        KernelFamily::Forward,
        include_str!("kernels/embedding.cu"),
    ),
    (
        "reduce.cu",
        KernelFamily::Forward,
        include_str!("kernels/reduce.cu"),
    ),
    (
        "rope.cu",
        KernelFamily::Forward,
        include_str!("kernels/rope.cu"),
    ),
    (
        "gather.cu",
        KernelFamily::Forward,
        include_str!("kernels/gather.cu"),
    ),
    (
        "scatter_add.cu",
        KernelFamily::Backward,
        include_str!("kernels/scatter_add.cu"),
    ),
    (
        "add_broadcast.cu",
        KernelFamily::Forward,
        include_str!("kernels/add_broadcast.cu"),
    ),
    (
        "layout.cu",
        KernelFamily::Attention,
        include_str!("kernels/layout.cu"),
    ),
    (
        "adamw.cu",
        KernelFamily::Optimizer,
        include_str!("kernels/adamw.cu"),
    ),
    (
        "log_softmax_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/log_softmax_backward.cu"),
    ),
    (
        "gather_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/gather_backward.cu"),
    ),
    (
        "add_into.cu",
        KernelFamily::Backward,
        include_str!("kernels/add_into.cu"),
    ),
    (
        "mean_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/mean_backward.cu"),
    ),
    (
        "mul_scalar_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/mul_scalar_backward.cu"),
    ),
    (
        "embedding_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/embedding_backward.cu"),
    ),
    (
        "add_broadcast_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/add_broadcast_backward.cu"),
    ),
    (
        "activation_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/activation_backward.cu"),
    ),
    (
        "mul_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/mul_backward.cu"),
    ),
    (
        "rms_norm_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/rms_norm_backward.cu"),
    ),
    (
        "rope_backward.cu",
        KernelFamily::Backward,
        include_str!("kernels/rope_backward.cu"),
    ),
    (
        "rollout.cu",
        KernelFamily::Rollout,
        include_str!("kernels/rollout.cu"),
    ),
    (
        "attention.cu",
        KernelFamily::Attention,
        include_str!("kernels/attention.cu"),
    ),
    (
        "attention_decode_online.cu",
        KernelFamily::Uncertain,
        include_str!("kernels/attention_decode_online.cu"),
    ),
    (
        "bridge.cu",
        KernelFamily::Bridge,
        include_str!("kernels/bridge.cu"),
    ),
    (
        "linear_attention.cu",
        KernelFamily::Uncertain,
        include_str!("kernels/linear_attention.cu"),
    ),
    (
        "fp8_block_scaled.cu",
        KernelFamily::Rollout,
        include_str!("kernels/fp8_block_scaled.cu"),
    ),
];

#[cfg(not(feature = "no-cuda"))]
const FUNCTION_NAMES: &[&str] = &[
    "add_f32",
    "mul_f32",
    "mul_scalar_f32",
    "sigmoid_f32",
    "exp_f32",
    "abs_f32",
    "neg_f32",
    "softmax_last_axis_f32",
    "log_softmax_last_axis_f32",
    "softmax_last_axis_backward_f32",
    "silu_f32",
    "rms_norm_f32",
    "embedding_f32",
    "embedding_bf16_to_f32",
    "sum_squares_partial_f32",
    "sum_partial_f32",
    "grad_clip_sumsq_f32",
    "grad_clip_scale_f32",
    "sum_last_axis_f32",
    "mean_last_axis_f32",
    "rope_f32",
    "gather_last_dim_f32",
    "scatter_add_rows_f32",
    "add_broadcast_f32",
    "broadcast_copy_f32",
    "transpose_axes_swap_f32",
    "slice_f32",
    "concat_axis2_f32",
    "kv_cache_write_axis2_f32",
    "slice_backward_f32",
    "slice_backward_accum_f32",
    "adamw_step_f32",
    "log_softmax_last_axis_backward_f32",
    "gather_last_dim_backward_f32",
    "add_into_f32",
    "accumulate_into_f32",
    "mean_backward_f32",
    "mul_scalar_backward_f32",
    "embedding_backward_f32",
    "add_broadcast_backward_f32",
    "silu_backward_f32",
    "sigmoid_backward_f32",
    "abs_backward_f32",
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
    "causal_sdpa_decode_gqa_cache_f32",
    "causal_sdpa_recompute_backward_f32",
    "causal_sdpa_decode_gqa_cache_online_f32_hd256",
    "qwen_decode_prepare_q_f32",
    "qwen_decode_prepare_q_gated_f32",
    "qwen_decode_prepare_kv_f32",
    "bf16_bits_to_f32",
    "f32_to_bf16_bits",
    "linear_attention_conv1d_silu_forward_f32_to_bf16",
    "linear_attention_conv1d_silu_boundary_f32_to_bf16",
    "linear_attention_copy_f32",
    "linear_attention_rms_gated_forward_f32_from_bf16",
    "linear_attention_rms_gated_backward_f32_to_bf16",
    "linear_attention_gdr_prepare_backward_f32",
    "linear_attention_chunked_scan_backward_f32",
    "linear_attention_chunk_transfer_f32",
    "linear_attention_chunk_carry_f32",
    "linear_attention_chunk_grad_f32",
    "linear_attention_conv1d_silu_backward_f32",
    "linear_attention_scan_backward_f32",
    "fp8_block_scaled_to_bf16",
    "fp4_e2m1_group_to_bf16",
    "marlin_fp4_to_bf16",
];

#[derive(Debug)]
pub(super) struct KernelCache {
    _module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
    #[cfg(not(feature = "no-cuda"))]
    ctx: Arc<CudaContext>,
    #[cfg(not(feature = "no-cuda"))]
    bf16: Mutex<Option<DtypeModule>>,
}

#[cfg(not(feature = "no-cuda"))]
#[derive(Debug)]
struct DtypeModule {
    module: Arc<CudaModule>,
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
            let (image, arch) = compile_cubin_for_current_device(ctx, TapeDtype::F32)?;
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
                ctx: ctx.clone(),
                bf16: Mutex::new(None),
            })
        }
    }

    pub(super) fn function(&self, name: &'static str) -> Result<&CudaFunction> {
        self.functions.get(name).ok_or(AutogradError::TapeInvariant(
            "autograd cuda kernel not found in cache",
        ))
    }

    pub(super) fn function_for(
        &self,
        name: &'static str,
        dtype: TapeDtype,
    ) -> Result<CudaFunction> {
        match dtype {
            TapeDtype::F32 => self.function(name).cloned(),
            TapeDtype::Bf16 => {
                #[cfg(feature = "no-cuda")]
                {
                    let _ = name;
                    todo!(
                        "GPU required: cuda kernel compilation is unavailable under feature no-cuda"
                    )
                }
                #[cfg(not(feature = "no-cuda"))]
                {
                    let mut guard = self.ensure_bf16_module()?;
                    let entry = guard.as_mut().ok_or(AutogradError::TapeInvariant(
                        "autograd cuda bf16 module missing after compile",
                    ))?;
                    if !entry.functions.contains_key(name) {
                        let function = entry.module.load_function(name).map_err(|err| {
                            cuda_kernel_error(format!(
                                "cuda load_function failed for autograd bf16 kernel {name}: {err:?}"
                            ))
                        })?;
                        entry.functions.insert(name, function);
                    }
                    Ok(entry.functions[name].clone())
                }
            }
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn ensure_bf16_module(&self) -> Result<std::sync::MutexGuard<'_, Option<DtypeModule>>> {
        let mut guard = self.bf16.lock().map_err(|_| {
            AutogradError::TapeInvariant("autograd cuda bf16 kernel cache poisoned")
        })?;
        if guard.is_none() {
            let (image, arch) = compile_cubin_for_current_device(&self.ctx, TapeDtype::Bf16)?;
            let module = self.ctx.load_module(image).map_err(|err| {
                cuda_kernel_error(format!(
                    "cuda load_module failed for autograd bf16 kernels arch={arch}: {err:?}"
                ))
            })?;
            *guard = Some(DtypeModule {
                module,
                functions: HashMap::new(),
            });
        }
        Ok(guard)
    }

    /// F32 always compiles at construction; Bf16 compiles here when the tape
    /// dtype is declared, instead of on the first hot-path `function_for` call.
    pub(super) fn warm_dtype(&self, dtype: TapeDtype) -> Result<()> {
        match dtype {
            TapeDtype::F32 => Ok(()),
            TapeDtype::Bf16 => {
                #[cfg(feature = "no-cuda")]
                {
                    todo!(
                        "GPU required: cuda kernel compilation is unavailable under feature no-cuda"
                    )
                }
                #[cfg(not(feature = "no-cuda"))]
                {
                    self.ensure_bf16_module().map(|_| ())
                }
            }
        }
    }

    pub(super) fn nvrtc_identity(&self, dtype: TapeDtype) -> Result<NvrtcIdentity> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = dtype;
            todo!("GPU required: cuda kernel identity is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let arch = current_sm_arch(&self.ctx)?;
            let (mut major, mut minor) = (0i32, 0i32);
            // SAFETY: both out-pointers are valid for the duration of the call.
            unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) }
                .result()
                .map_err(|err| cuda_kernel_error(format!("nvrtcVersion failed: {err:?}")))?;
            let mut driver = 0i32;
            // SAFETY: the out-pointer is valid for the duration of the call.
            unsafe { sys::cuDriverGetVersion(&mut driver) }
                .result()
                .map_err(|err| cuda_kernel_error(format!("cuDriverGetVersion failed: {err:?}")))?;
            Ok(NvrtcIdentity {
                source_hash: format!("{:016x}", fnv1a64(concat_sources(dtype).as_bytes())),
                compile_flags: format!("--gpu-architecture={arch}"),
                sm_arch: arch,
                tape_dtype: dtype,
                nvrtc_version: (major, minor),
                cuda_driver_version: driver,
            })
        }
    }
}

#[cfg(not(feature = "no-cuda"))]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_kernel_error(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}

#[cfg(not(feature = "no-cuda"))]
fn compile_cubin_for_current_device(
    ctx: &Arc<CudaContext>,
    dtype: TapeDtype,
) -> Result<(Ptx, &'static str)> {
    let arch = current_sm_arch(ctx)?;
    // Emit SASS cubin for the exact device instead of PTX. On V100 the
    // deployment driver supports CUDA 12.2 while the available NVRTC is 12.4;
    // PTX 8.4 would fail driver JIT with CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
    // but an sm_70 cubin loads cleanly and keeps the kernel code uniform.
    let image = compile_cubin(&concat_sources(dtype), arch).map_err(|err| {
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
    // SAFETY: `program.raw()` is a live nvrtcProgram owned by `program`, destroyed only in Drop.
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
        // SAFETY: self.prog is live while &self exists — only Drop destroys it.
        unsafe { nvrtc_result::get_program_log(self.prog) }
            .ok()
            // SAFETY: NVRTC logs are NUL-terminated; the CStr borrow ends before `raw` drops.
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
            // SAFETY: a non-null prog came from create_program and is destroyed exactly once, here.
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
    // SAFETY: the caller passes a live compiled program; &mut size is a valid out-pointer.
    unsafe {
        nvrtc_sys::nvrtcGetCUBINSize(prog, &mut size as *mut _).result()?;
    }

    let mut cubin = vec![0u8; size];
    // SAFETY: cubin is exactly the `size` bytes nvrtcGetCUBINSize reported for this prog.
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
fn concat_sources(dtype: TapeDtype) -> String {
    let mut src = dtype.nvrtc_prelude().to_string();
    for (_, _, source) in KERNEL_SOURCES {
        src.push('\n');
        src.push_str(source);
    }
    src.push('\n');
    src
}
