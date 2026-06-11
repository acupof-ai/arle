//! Vulkan kernel build + raw-buffer launch layer for the AIPC lane.
//!
//! The borrowed operator corpus is adapted from ggml-org/llama.cpp
//! `vulkan-shaders` @ d2462f8f (MIT). `build.rs` compiles selected shaders
//! with `glslc` when the `vulkan` feature is on; missing `glslc` or
//! unresolved macro variants leave a typecheck-only crate whose launchers
//! fail loud with [`KernelError::ShaderMissing`].

pub const QK_K: usize = 256;
pub const QK8_1: usize = 32;

pub const BLOCK_IQ2_XXS_BYTES: usize = 66;
pub const BLOCK_Q2_K_BYTES: usize = 84;
pub const BLOCK_Q4_K_BYTES: usize = 144;
pub const BLOCK_Q5_K_BYTES: usize = 176;
pub const BLOCK_Q6_K_BYTES: usize = 210;
pub const BLOCK_Q8_1_BYTES: usize = 36;

pub const fn iq2_xxs_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_IQ2_XXS_BYTES)
}

pub const fn q2_k_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_Q2_K_BYTES)
}

pub const fn q4_k_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_Q4_K_BYTES)
}

pub const fn q5_k_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_Q5_K_BYTES)
}

pub const fn q6_k_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_Q6_K_BYTES)
}

pub const fn q8_1_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK8_1) {
        return None;
    }
    Some(ncols / QK8_1 * BLOCK_Q8_1_BYTES)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    MmvqIq2Xxs,
    MmvqQ2K,
    GemvQ4K,
    GemvQ5K,
    GemvQ6K,
    QuantizeQ8_1,
    RmsNorm,
    RopeNeox,
    RopeNorm,
    Silu,
    Gelu,
    GeGlu,
    SwiGlu,
    Add,
    GetRows,
    SoftMax,
    ArgMax,
    FlashAttn,
    Dsv4PrepareQk,
    Dsv4CompressorUpdate,
    Dsv4CsaSelect,
    Dsv4HybridAttention,
    Dsv4SwaAttention,
    Dsv4Mhc,
    Dsv4OutputInverseRope,
    SwigluClamped,
    Qwen35SsmConv,
    Qwen35GatedDeltaNet,
}

const SPEC_WORKGROUP_32: &[(u32, u32)] = &[(0, 32)];
const SPEC_FLASH_ATTN_WORKGROUP: &[(u32, u32)] = &[(0, 128)];
const SPEC_MMVQ_IQ2_XXS: &[(u32, u32)] = &[(0, 32), (1, 4), (2, 1)];
const SPEC_MMVQ_Q2_K: &[(u32, u32)] = &[(0, 32), (1, 2), (2, 1)];
const SPEC_GEMV_K_Q8_1: &[(u32, u32)] = &[(0, 32), (1, 1), (2, 1)];
const SPEC_RMS_NORM_MUL: &[(u32, u32)] = &[(1, 1)];

impl Kernel {
    pub const ALL: &'static [Self] = &[
        Self::MmvqIq2Xxs,
        Self::MmvqQ2K,
        Self::GemvQ4K,
        Self::GemvQ5K,
        Self::GemvQ6K,
        Self::QuantizeQ8_1,
        Self::RmsNorm,
        Self::RopeNeox,
        Self::RopeNorm,
        Self::Silu,
        Self::Gelu,
        Self::GeGlu,
        Self::SwiGlu,
        Self::Add,
        Self::GetRows,
        Self::SoftMax,
        Self::ArgMax,
        Self::FlashAttn,
        Self::Dsv4PrepareQk,
        Self::Dsv4CompressorUpdate,
        Self::Dsv4CsaSelect,
        Self::Dsv4HybridAttention,
        Self::Dsv4SwaAttention,
        Self::Dsv4Mhc,
        Self::Dsv4OutputInverseRope,
        Self::SwigluClamped,
        Self::Qwen35SsmConv,
        Self::Qwen35GatedDeltaNet,
    ];

    pub const fn shader_name(self) -> &'static str {
        match self {
            Kernel::MmvqIq2Xxs => "mul_mat_vec_iq2_xxs",
            Kernel::MmvqQ2K => "mul_mat_vec_q2_k",
            Kernel::GemvQ4K => "mul_mat_vecq_q4_k",
            Kernel::GemvQ5K => "mul_mat_vecq_q5_k",
            Kernel::GemvQ6K => "mul_mat_vecq_q6_k",
            Kernel::QuantizeQ8_1 => "q8_1_quantize",
            Kernel::RmsNorm => "rms_norm",
            Kernel::RopeNeox => "rope_neox",
            Kernel::RopeNorm => "rope_norm",
            Kernel::Silu => "silu",
            Kernel::Gelu => "gelu",
            Kernel::GeGlu => "geglu",
            Kernel::SwiGlu => "swiglu",
            Kernel::Add => "add",
            Kernel::GetRows => "get_rows",
            Kernel::SoftMax => "soft_max",
            Kernel::ArgMax => "argmax",
            Kernel::FlashAttn => "flash_attn",
            Kernel::Dsv4PrepareQk => "dsv4_prepare_qk",
            Kernel::Dsv4CompressorUpdate => "dsv4_compressor_update",
            Kernel::Dsv4CsaSelect => "dsv4_csa_select",
            Kernel::Dsv4HybridAttention => "dsv4_hybrid_attention",
            Kernel::Dsv4SwaAttention => "dsv4_swa_attention",
            Kernel::Dsv4Mhc => "dsv4_mhc",
            Kernel::Dsv4OutputInverseRope => "dsv4_output_inverse_rope",
            Kernel::SwigluClamped => "swiglu_clamped",
            Kernel::Qwen35SsmConv => "qwen35_ssm_conv",
            Kernel::Qwen35GatedDeltaNet => "qwen35_gated_delta_net",
        }
    }

    pub const fn specialization_u32(self) -> &'static [(u32, u32)] {
        match self {
            Kernel::MmvqIq2Xxs => SPEC_MMVQ_IQ2_XXS,
            Kernel::MmvqQ2K => SPEC_MMVQ_Q2_K,
            Kernel::GemvQ4K | Kernel::GemvQ5K | Kernel::GemvQ6K => SPEC_GEMV_K_Q8_1,
            Kernel::RmsNorm => SPEC_RMS_NORM_MUL,
            Kernel::SoftMax | Kernel::ArgMax => SPEC_WORKGROUP_32,
            Kernel::FlashAttn => SPEC_FLASH_ATTN_WORKGROUP,
            Kernel::QuantizeQ8_1
            | Kernel::RopeNeox
            | Kernel::RopeNorm
            | Kernel::Silu
            | Kernel::Gelu
            | Kernel::GeGlu
            | Kernel::SwiGlu
            | Kernel::Add
            | Kernel::GetRows
            | Kernel::Dsv4PrepareQk
            | Kernel::Dsv4CompressorUpdate
            | Kernel::Dsv4CsaSelect
            | Kernel::Dsv4HybridAttention
            | Kernel::Dsv4SwaAttention
            | Kernel::Dsv4Mhc
            | Kernel::Dsv4OutputInverseRope
            | Kernel::SwigluClamped
            | Kernel::Qwen35SsmConv
            | Kernel::Qwen35GatedDeltaNet => &[],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dispatch {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dispatch {
    pub const fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    NotCompiled,
    ShaderMissing(&'static str),
    InvalidDispatch,
    InvalidPushConstants,
    Runtime(String),
}

pub const NOT_COMPILED: KernelError = KernelError::NotCompiled;

pub type Result<T> = std::result::Result<T, KernelError>;

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::NotCompiled => {
                write!(
                    f,
                    "Vulkan kernels not compiled (build with --features vulkan)"
                )
            }
            KernelError::ShaderMissing(name) => write!(f, "Vulkan shader {name}.spv missing"),
            KernelError::InvalidDispatch => {
                write!(f, "Vulkan dispatch dimensions must be non-zero")
            }
            KernelError::InvalidPushConstants => {
                write!(f, "Vulkan push constants must be 4-byte aligned")
            }
            KernelError::Runtime(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for KernelError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelParams {
    words: Vec<u32>,
}

impl KernelParams {
    pub fn empty() -> Self {
        Self { words: Vec::new() }
    }

    pub fn from_words(words: impl Into<Vec<u32>>) -> Self {
        Self {
            words: words.into(),
        }
    }

    pub fn push_u32(&mut self, value: u32) {
        self.words.push(value);
    }

    pub fn push_i32(&mut self, value: i32) {
        self.words.push(value as u32);
    }

    pub fn push_f32(&mut self, value: f32) {
        self.words.push(value.to_bits());
    }

    pub fn len_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u32>()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
}

macro_rules! launcher_fns {
    ($call:path) => {
        pub fn mmvq_iq2_xxs(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::MmvqIq2Xxs, ctx, buffers, dispatch)
        }

        pub fn mmvq_q2_k(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::MmvqQ2K, ctx, buffers, dispatch)
        }

        pub fn q4_k_gemv(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::GemvQ4K, ctx, buffers, dispatch)
        }

        pub fn q5_k_gemv(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::GemvQ5K, ctx, buffers, dispatch)
        }

        pub fn q6_k_gemv(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::GemvQ6K, ctx, buffers, dispatch)
        }

        pub fn q8_1_quantize(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::QuantizeQ8_1, ctx, buffers, dispatch)
        }

        pub fn rms_norm(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::RmsNorm, ctx, buffers, dispatch)
        }

        pub fn rope_neox(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::RopeNeox, ctx, buffers, dispatch)
        }

        pub fn rope_norm(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::RopeNorm, ctx, buffers, dispatch)
        }

        pub fn silu(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::Silu, ctx, buffers, dispatch)
        }

        pub fn gelu(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::Gelu, ctx, buffers, dispatch)
        }

        pub fn geglu(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::GeGlu, ctx, buffers, dispatch)
        }

        pub fn swiglu(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::SwiGlu, ctx, buffers, dispatch)
        }

        pub fn add(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::Add, ctx, buffers, dispatch)
        }

        pub fn embedding_get_rows(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::GetRows, ctx, buffers, dispatch)
        }

        pub fn soft_max(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::SoftMax, ctx, buffers, dispatch)
        }

        pub fn argmax(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::ArgMax, ctx, buffers, dispatch)
        }

        pub fn flash_attn(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::FlashAttn, ctx, buffers, dispatch)
        }
    };
}

macro_rules! fused_launcher_fns {
    ($call:path) => {
        pub fn dsv4_prepare_qk(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Dsv4PrepareQk, ctx, buffers, dispatch, params)
        }

        pub fn dsv4_compressor_update(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Dsv4CompressorUpdate, ctx, buffers, dispatch, params)
        }

        pub fn dsv4_csa_select(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Dsv4CsaSelect, ctx, buffers, dispatch, params)
        }

        pub fn dsv4_hybrid_attention(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Dsv4HybridAttention, ctx, buffers, dispatch, params)
        }

        pub fn dsv4_swa_attention(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Dsv4SwaAttention, ctx, buffers, dispatch, params)
        }

        pub fn dsv4_mhc(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Dsv4Mhc, ctx, buffers, dispatch, params)
        }

        pub fn dsv4_output_inverse_rope(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(
                Kernel::Dsv4OutputInverseRope,
                ctx,
                buffers,
                dispatch,
                params,
            )
        }

        pub fn swiglu_clamped(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::SwigluClamped, ctx, buffers, dispatch, params)
        }

        pub fn qwen35_ssm_conv(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Qwen35SsmConv, ctx, buffers, dispatch, params)
        }

        pub fn qwen35_gated_delta_net(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call(Kernel::Qwen35GatedDeltaNet, ctx, buffers, dispatch, params)
        }
    };
}

#[cfg(feature = "vulkan")]
mod real {
    use super::{Dispatch, Kernel, KernelError, Result};
    use ash::vk;
    use std::path::PathBuf;

    pub fn launch(
        kernel: Kernel,
        ctx: &vulkan_sys::VulkanContext,
        buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        dispatch: Dispatch,
    ) -> Result<()> {
        launch_with_params(
            kernel,
            ctx,
            buffers,
            dispatch,
            &super::KernelParams::empty(),
        )
    }

    pub fn launch_with_params(
        kernel: Kernel,
        ctx: &vulkan_sys::VulkanContext,
        buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        dispatch: Dispatch,
        params: &super::KernelParams,
    ) -> Result<()> {
        if dispatch.x == 0 || dispatch.y == 0 || dispatch.z == 0 {
            return Err(KernelError::InvalidDispatch);
        }
        let push_bytes = params.to_le_bytes();
        if !push_bytes.len().is_multiple_of(4) {
            return Err(KernelError::InvalidPushConstants);
        }
        let Some(path) = shader_path(kernel) else {
            return Err(KernelError::ShaderMissing(kernel.shader_name()));
        };
        let bytes =
            std::fs::read(&path).map_err(|_| KernelError::ShaderMissing(kernel.shader_name()))?;
        let shader = vulkan_sys::ShaderModule::from_spirv_bytes(ctx, &bytes)
            .map_err(|e| KernelError::Runtime(e.to_string()))?;
        let layout = vulkan_sys::DescriptorSetLayout::storage_buffers(ctx, buffers.len())
            .map_err(|e| KernelError::Runtime(e.to_string()))?;
        let set = vulkan_sys::DescriptorSet::storage_buffers(ctx, &layout, buffers)
            .map_err(|e| KernelError::Runtime(e.to_string()))?;
        let pipeline = vulkan_sys::ComputePipeline::create_with_push_constants_and_specialization(
            ctx,
            &shader,
            &[&layout],
            push_bytes.len() as u32,
            kernel.specialization_u32(),
        )
        .map_err(|e| KernelError::Runtime(e.to_string()))?;
        let commands = vulkan_sys::CommandPool::create(ctx)
            .map_err(|e| KernelError::Runtime(e.to_string()))?;
        commands
            .one_shot_submit(|cmd| {
                let device = ctx.raw_device();
                unsafe {
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.raw());
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.layout(),
                        0,
                        &[set.raw()],
                        &[],
                    );
                    if !push_bytes.is_empty() {
                        device.cmd_push_constants(
                            cmd,
                            pipeline.layout(),
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &push_bytes,
                        );
                    }
                    device.cmd_dispatch(cmd, dispatch.x, dispatch.y, dispatch.z);
                }
                Ok(())
            })
            .map_err(|e| KernelError::Runtime(e.to_string()))
    }

    fn shader_path(kernel: Kernel) -> Option<PathBuf> {
        let dir = option_env!("ARLE_VULKAN_SPV_DIR")?;
        let path = PathBuf::from(dir).join(format!("{}.spv", kernel.shader_name()));
        path.exists().then_some(path)
    }
}

#[cfg(feature = "vulkan")]
launcher_fns!(real::launch);
#[cfg(feature = "vulkan")]
fused_launcher_fns!(real::launch_with_params);

#[cfg(not(feature = "vulkan"))]
mod stub {
    use super::{Kernel, KernelError, Result};

    pub fn launch(
        _kernel: Kernel,
        _ctx: &vulkan_sys::VulkanContext,
        _buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        _dispatch: super::Dispatch,
    ) -> Result<()> {
        Err(KernelError::NotCompiled)
    }

    pub fn launch_with_params(
        _kernel: Kernel,
        _ctx: &vulkan_sys::VulkanContext,
        _buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        _dispatch: super::Dispatch,
        _params: &super::KernelParams,
    ) -> Result<()> {
        Err(KernelError::NotCompiled)
    }
}

#[cfg(not(feature = "vulkan"))]
launcher_fns!(stub::launch);
#[cfg(not(feature = "vulkan"))]
fused_launcher_fns!(stub::launch_with_params);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_llama_layouts() {
        assert_eq!(iq2_xxs_row_bytes(256), Some(66));
        assert_eq!(q2_k_row_bytes(256), Some(84));
        assert_eq!(q4_k_row_bytes(256), Some(144));
        assert_eq!(q5_k_row_bytes(256), Some(176));
        assert_eq!(q6_k_row_bytes(256), Some(210));
        assert_eq!(q8_1_row_bytes(32), Some(36));
        assert_eq!(q4_k_row_bytes(255), None);
        assert_eq!(q8_1_row_bytes(31), None);
    }

    #[test]
    fn shader_names_cover_generic_op_set() {
        let names = [
            Kernel::MmvqIq2Xxs.shader_name(),
            Kernel::MmvqQ2K.shader_name(),
            Kernel::GemvQ4K.shader_name(),
            Kernel::GemvQ5K.shader_name(),
            Kernel::GemvQ6K.shader_name(),
            Kernel::QuantizeQ8_1.shader_name(),
            Kernel::RmsNorm.shader_name(),
            Kernel::RopeNeox.shader_name(),
            Kernel::RopeNorm.shader_name(),
            Kernel::Silu.shader_name(),
            Kernel::Gelu.shader_name(),
            Kernel::GeGlu.shader_name(),
            Kernel::SwiGlu.shader_name(),
            Kernel::Add.shader_name(),
            Kernel::GetRows.shader_name(),
            Kernel::SoftMax.shader_name(),
            Kernel::ArgMax.shader_name(),
            Kernel::FlashAttn.shader_name(),
        ];
        assert_eq!(names.len(), 18);
        assert!(names.iter().all(|name| !name.is_empty()));
    }

    #[test]
    fn shader_specializations_cover_runtime_specialization_constants() {
        assert_eq!(
            Kernel::MmvqIq2Xxs.specialization_u32(),
            &[(0, 32), (1, 4), (2, 1)]
        );
        assert_eq!(
            Kernel::MmvqQ2K.specialization_u32(),
            &[(0, 32), (1, 2), (2, 1)]
        );
        assert_eq!(
            Kernel::GemvQ4K.specialization_u32(),
            &[(0, 32), (1, 1), (2, 1)]
        );
        assert_eq!(
            Kernel::GemvQ5K.specialization_u32(),
            Kernel::GemvQ4K.specialization_u32()
        );
        assert_eq!(
            Kernel::GemvQ6K.specialization_u32(),
            Kernel::GemvQ4K.specialization_u32()
        );
        assert_eq!(Kernel::SoftMax.specialization_u32(), &[(0, 32)]);
        assert_eq!(Kernel::ArgMax.specialization_u32(), &[(0, 32)]);
        assert_eq!(Kernel::FlashAttn.specialization_u32(), &[(0, 128)]);
        assert_eq!(Kernel::RmsNorm.specialization_u32(), &[(1, 1)]);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_feature_build_produces_every_registered_spv() {
        if option_env!("ARLE_VULKAN_GLSLC_PRESENT") != Some("1") {
            eprintln!("vulkan-kernels: glslc unavailable, skipping SPIR-V existence check");
            return;
        }
        let dir = option_env!("ARLE_VULKAN_SPV_DIR").expect("build.rs sets ARLE_VULKAN_SPV_DIR");
        let manifest_path = std::path::Path::new(dir).join("registered-shaders.txt");
        let manifest = std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|_| panic!("missing shader manifest {}", manifest_path.display()));
        let registered: std::collections::BTreeSet<_> =
            manifest.lines().filter(|name| !name.is_empty()).collect();
        let exposed: std::collections::BTreeSet<_> = Kernel::ALL
            .iter()
            .map(|kernel| kernel.shader_name())
            .collect();
        assert_eq!(
            registered, exposed,
            "build.rs registry and Kernel::ALL drifted"
        );
        for name in registered {
            let path = std::path::Path::new(dir).join(format!("{name}.spv"));
            let len = std::fs::metadata(&path)
                .unwrap_or_else(|_| panic!("missing SPIR-V for {name}"))
                .len();
            assert!(len > 0, "empty SPIR-V for {name}");
        }
    }

    #[test]
    fn shader_names_cover_dsv4_fused_op_set() {
        let names = [
            Kernel::Dsv4PrepareQk.shader_name(),
            Kernel::Dsv4CompressorUpdate.shader_name(),
            Kernel::Dsv4CsaSelect.shader_name(),
            Kernel::Dsv4HybridAttention.shader_name(),
            Kernel::Dsv4SwaAttention.shader_name(),
            Kernel::Dsv4Mhc.shader_name(),
            Kernel::Dsv4OutputInverseRope.shader_name(),
            Kernel::SwigluClamped.shader_name(),
        ];
        assert_eq!(names.len(), 8);
        assert!(names.iter().all(|name| !name.is_empty()));
    }

    #[test]
    fn shader_names_cover_qwen35_hybrid_fused_op_set() {
        let names = [
            Kernel::Qwen35SsmConv.shader_name(),
            Kernel::Qwen35GatedDeltaNet.shader_name(),
            Kernel::FlashAttn.shader_name(),
        ];
        assert_eq!(names.len(), 3);
        assert!(names.iter().all(|name| !name.is_empty()));
    }

    #[test]
    fn kernel_params_are_le_words() {
        let mut params = KernelParams::empty();
        params.push_u32(0x1122_3344);
        params.push_i32(-1);
        params.push_f32(1.0);
        assert_eq!(params.len_bytes(), 12);
        assert_eq!(
            params.to_le_bytes(),
            [
                0x44, 0x33, 0x22, 0x11, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x80, 0x3f
            ]
        );
    }

    #[cfg(not(feature = "vulkan"))]
    #[test]
    fn stub_error_is_clear() {
        assert_eq!(
            NOT_COMPILED.to_string(),
            "Vulkan kernels not compiled (build with --features vulkan)"
        );
    }
}
