//! Vulkan kernel build + raw-buffer launch layer for the AIPC lane.
//!
//! The borrowed operator corpus is adapted from ggml-org/llama.cpp
//! `vulkan-shaders` @ d2462f8f (MIT). `build.rs` compiles selected shaders
//! with `glslc` when the `vulkan` feature is on; missing `glslc` or
//! unresolved macro variants leave a typecheck-only crate whose launchers
//! fail loud with [`KernelError::ShaderMissing`].

mod cache;

pub use cache::{KernelCache, launch_cached, record_dispatch};

pub const QK_K: usize = 256;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

pub const BLOCK_IQ2_XXS_BYTES: usize = 66;
pub const BLOCK_Q2_K_BYTES: usize = 84;
pub const BLOCK_Q4_K_BYTES: usize = 144;
pub const BLOCK_Q5_K_BYTES: usize = 176;
pub const BLOCK_Q6_K_BYTES: usize = 210;
pub const BLOCK_Q8_0_BYTES: usize = 34;
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

pub const fn q8_0_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK8_0) {
        return None;
    }
    Some(ncols / QK8_0 * BLOCK_Q8_0_BYTES)
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
    GemvQ8_0,
    /// Fused MoE expert GEMV (`mul_mat_vec_id`): one dispatch runs a token
    /// through ALL its top-k routed experts. Same `mul_mat_vecq` body + q8_1
    /// activation as the plain GEMV, but the `MUL_MAT_ID` push tail + a 6th
    /// expert-id binding (see [`gemv_id_params`]).
    GemvIdQ4K,
    GemvIdQ5K,
    GemvIdQ6K,
    GemvIdQ8_0,
    QuantizeQ8_1,
    RmsNorm,
    RopeNeox,
    RopeNorm,
    Silu,
    Gelu,
    GeGlu,
    SwiGlu,
    Add,
    ScaledAdd,
    SigmoidMul,
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
    /// Pack one f32 head row into the f16 KV cache (`out[i] = float16_t(in[i])`).
    /// Lets the full-attention block write this token's roped K / raw V into the
    /// device-resident f16 cache without a host readback, so the projection /
    /// rope / pack / flash / gate record into ONE submit.
    F16KvPack,
    Qwen35SsmConv,
    Qwen35GatedDeltaNet,
    /// Qwen3.6 MoE router top-k: n_expert F32 logits → top_k expert ids (i32) +
    /// top_k weights (f32). Single-thread softmax-over-all → top-k by prob →
    /// renorm, replacing the host `qwen36_topk_routes` so routing stays on-device.
    Qwen36RouterTopk,
    /// Qwen3.6 MoE router / shared-gate F32 GEMV: `y[e] = Σ_c W[e,c]·x[c]` for an
    /// F32 weight `[n_out, hidden]`, optional sigmoid (shared-expert gate). Reads
    /// the router from device bandwidth instead of host write-combined read-back.
    Qwen36RouterGemv,
    /// Qwen3.6 MoE device-weighted accumulate: `acc += Σ_e weights[e]·src[e]`
    /// using DEVICE weights (from `Qwen36RouterTopk`), replacing the host-constant
    /// per-expert scaled-add loop so routing stays on-device.
    Qwen36MoeWeightedAccum,
}

const SPEC_WORKGROUP_32: &[(u32, u32)] = &[(0, 32)];
const SPEC_FLASH_ATTN_F32_F16_HD256: &[(u32, u32)] = &[
    (0, 128),
    (1, 1),
    (2, 64),
    (3, 256),
    (4, 256),
    (5, 1),
    (6, 8),
    (7, 1),
    (8, 32),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 1),
    (13, 1),
    (14, 2),
    (15, 2),
];
const SPEC_MMVQ_IQ2_XXS: &[(u32, u32)] = &[(0, 32), (1, 4), (2, 1)];
const SPEC_MMVQ_Q2_K: &[(u32, u32)] = &[(0, 32), (1, 2), (2, 1)];
// BLOCK_SIZE=64 matches the 8060S wave width (subgroup_size=64): a 32-wide
// workgroup would leave half of every wave idle. `mul_mat_vecq.comp` derives its
// per-thread iteration count from `ncols/(K_PER_ITER*BLOCK_SIZE)`, so a wider
// BLOCK_SIZE is purely a lane-occupancy change — the dot-product math and the
// 13-uint/5-buffer ABI are untouched. NUM_ROWS=1 (constant_id 1) matches
// llama.cpp's `rm_kq_int`/`rm_stdq_int` for AMD non-GCN. The pipeline pins
// requiredSubgroupSize=64 (see `Kernel::required_subgroup_size`) so the wave is
// exactly one full subgroup.
const SPEC_GEMV_K_Q8_1: &[(u32, u32)] = &[(0, 64), (1, 1), (2, 1)];
const SPEC_RMS_NORM_MUL: &[(u32, u32)] = &[(1, 1)];

impl Kernel {
    pub const ALL: &'static [Self] = &[
        Self::MmvqIq2Xxs,
        Self::MmvqQ2K,
        Self::GemvQ4K,
        Self::GemvQ5K,
        Self::GemvQ6K,
        Self::GemvQ8_0,
        Self::GemvIdQ4K,
        Self::GemvIdQ5K,
        Self::GemvIdQ6K,
        Self::GemvIdQ8_0,
        Self::QuantizeQ8_1,
        Self::RmsNorm,
        Self::RopeNeox,
        Self::RopeNorm,
        Self::Silu,
        Self::Gelu,
        Self::GeGlu,
        Self::SwiGlu,
        Self::Add,
        Self::ScaledAdd,
        Self::SigmoidMul,
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
        Self::F16KvPack,
        Self::Qwen36RouterTopk,
        Self::Qwen36RouterGemv,
        Self::Qwen36MoeWeightedAccum,
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
            Kernel::GemvQ8_0 => "mul_mat_vecq_q8_0",
            Kernel::GemvIdQ4K => "mul_mat_vec_id_q4_k",
            Kernel::GemvIdQ5K => "mul_mat_vec_id_q5_k",
            Kernel::GemvIdQ6K => "mul_mat_vec_id_q6_k",
            Kernel::GemvIdQ8_0 => "mul_mat_vec_id_q8_0",
            Kernel::QuantizeQ8_1 => "q8_1_quantize",
            Kernel::RmsNorm => "rms_norm",
            Kernel::RopeNeox => "rope_neox",
            Kernel::RopeNorm => "rope_norm",
            Kernel::Silu => "silu",
            Kernel::Gelu => "gelu",
            Kernel::GeGlu => "geglu",
            Kernel::SwiGlu => "swiglu",
            Kernel::Add => "add",
            Kernel::ScaledAdd => "scaled_add",
            Kernel::SigmoidMul => "sigmoid_mul",
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
            Kernel::F16KvPack => "f16_kv_pack",
            Kernel::Qwen35SsmConv => "qwen35_ssm_conv",
            Kernel::Qwen35GatedDeltaNet => "qwen35_gated_delta_net",
            Kernel::Qwen36RouterTopk => "qwen36_router_topk",
            Kernel::Qwen36RouterGemv => "qwen36_router_gemv",
            Kernel::Qwen36MoeWeightedAccum => "qwen36_moe_weighted_accum",
        }
    }

    pub const fn specialization_u32(self) -> &'static [(u32, u32)] {
        match self {
            Kernel::MmvqIq2Xxs => SPEC_MMVQ_IQ2_XXS,
            Kernel::MmvqQ2K => SPEC_MMVQ_Q2_K,
            Kernel::GemvQ4K
            | Kernel::GemvQ5K
            | Kernel::GemvQ6K
            | Kernel::GemvQ8_0
            | Kernel::GemvIdQ4K
            | Kernel::GemvIdQ5K
            | Kernel::GemvIdQ6K
            | Kernel::GemvIdQ8_0 => SPEC_GEMV_K_Q8_1,
            Kernel::QuantizeQ8_1 => SPEC_WORKGROUP_32,
            Kernel::RmsNorm => SPEC_RMS_NORM_MUL,
            Kernel::SoftMax | Kernel::ArgMax => SPEC_WORKGROUP_32,
            Kernel::FlashAttn => SPEC_FLASH_ATTN_F32_F16_HD256,
            Kernel::RopeNeox
            | Kernel::RopeNorm
            | Kernel::Silu
            | Kernel::Gelu
            | Kernel::GeGlu
            | Kernel::SwiGlu
            | Kernel::Add
            | Kernel::ScaledAdd
            | Kernel::SigmoidMul
            | Kernel::GetRows
            | Kernel::Dsv4PrepareQk
            | Kernel::Dsv4CompressorUpdate
            | Kernel::Dsv4CsaSelect
            | Kernel::Dsv4HybridAttention
            | Kernel::Dsv4SwaAttention
            | Kernel::Dsv4Mhc
            | Kernel::Dsv4OutputInverseRope
            | Kernel::SwigluClamped
            | Kernel::F16KvPack
            | Kernel::Qwen35SsmConv
            | Kernel::Qwen35GatedDeltaNet
            | Kernel::Qwen36RouterTopk
            | Kernel::Qwen36RouterGemv
            | Kernel::Qwen36MoeWeightedAccum => &[],
        }
    }

    /// The subgroup size this kernel's pipeline must be created with, if any.
    /// `flash_attn` is specialized for `SubGroupSize=32` and reduces with
    /// subgroup shuffles + `num_subgroups = WorkGroupSize/32`, so on a wave64
    /// device (the 8060S defaults to 64) its pipeline MUST pin a 32-wide
    /// subgroup via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`.
    ///
    /// The `mul_mat_vecq` GEMVs run a `BLOCK_SIZE`-wide workgroup (constant_id 0
    /// = 64) and pin requiredSubgroupSize=64 so the workgroup is exactly one
    /// full 64-wide wave — every lane busy, and (when `USE_SUBGROUP_ADD` is
    /// compiled in) the cross-lane reduction is a single-subgroup `subgroupAdd`.
    /// This mirrors llama.cpp's `subgroup_size_int = device->subgroup_size` for
    /// the q8_1 decode pipelines on AMD non-GCN.
    ///
    /// Every other kernel is subgroup-size-agnostic (`None` = driver default).
    pub const fn required_subgroup_size(self) -> Option<u32> {
        match self {
            Kernel::FlashAttn => Some(32),
            Kernel::GemvQ4K
            | Kernel::GemvQ5K
            | Kernel::GemvQ6K
            | Kernel::GemvQ8_0
            | Kernel::GemvIdQ4K
            | Kernel::GemvIdQ5K
            | Kernel::GemvIdQ6K
            | Kernel::GemvIdQ8_0 => Some(64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttentionSpec {
    specialization_u32: [(u32, u32); 16],
}

impl FlashAttentionSpec {
    pub const fn f32_f16(head_dim: u32) -> Self {
        Self::f32_f16_dims(head_dim, head_dim)
    }

    pub const fn f32_f16_dims(hsk: u32, hsv: u32) -> Self {
        Self {
            specialization_u32: [
                (0, 128),
                (1, 1),
                (2, 64),
                (3, hsk),
                (4, hsv),
                (5, 1),
                (6, 8),
                (7, 1),
                (8, 32),
                (9, 0),
                (10, 0),
                (11, 0),
                (12, 1),
                (13, 1),
                (14, 2),
                (15, 2),
            ],
        }
    }

    pub const fn specialization_u32(&self) -> &[(u32, u32)] {
        &self.specialization_u32
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

    /// The raw `u32` words (for tests / introspection of a push-constant block).
    pub fn words(&self) -> &[u32] {
        &self.words
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

/// Push-constant layout for the `mul_mat_vecq` GEMV (binding interface in
/// `mul_mat_vec_base.glsl`, non-`MUL_MAT_ID` branch). 13 `uint`s, in order:
/// `ncols, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b,
/// batch_stride_d, fusion_flags, base_work_group_y, ne02, ne12, broadcast2,
/// broadcast3`.
///
/// `stride_d` is the shader's row-count guard (number of output rows), and
/// `ncols` is the per-row width in elements (a multiple of 256 for the K-quant
/// shaders, 32 for Q8_0). For a single, unbatched `[nrows, ncols]` weight ×
/// `[ncols]` activation matvec with no bias/scale fusion, the remaining fields
/// reduce to: strides set to their natural row widths, every batch stride 0,
/// `fusion_flags = 0`, and `ne02 = ne12 = broadcast2 = broadcast3 = 1`.
pub fn gemv_params(ncols: u32, nrows: u32) -> KernelParams {
    KernelParams::from_words(vec![
        ncols,      // ncols: per-row width in elements
        ncols,      // stride_a: weight row stride (elements); unused for batch 0
        ncols / 32, // stride_b: activation row stride in q8_1 blocks; unused for batch 0
        nrows,      // stride_d: ROW COUNT guard the shader checks first_row against
        ncols,      // batch_stride_a (elements); only used via /QUANT_K when batched
        0,          // batch_stride_b: single batch
        0,          // batch_stride_d: single batch
        0,          // fusion_flags: no bias/scale fusion (bindings 3/4 unread)
        0,          // base_work_group_y: batch offset
        1,          // ne02
        1,          // ne12
        1,          // broadcast2
        1,          // broadcast3
    ])
}

/// Dispatch grid for [`gemv_params`]: one workgroup per output row (NUM_ROWS=1),
/// single batch. `main()` derives `first_row` from `gl_WorkGroupID.x`.
pub fn gemv_dispatch(nrows: u32) -> Dispatch {
    Dispatch::x(nrows.max(1))
}

/// Push-constant layout for the FUSED MoE expert GEMV (`mul_mat_vecq.comp` built
/// with `MUL_MAT_ID`). The `MUL_MAT_ID` branch of `mul_mat_vec_base.glsl` replaces
/// the trailing 5 batch fields of [`gemv_params`] with 4 expert-id fields, so the
/// block is 12 `uint`s in order:
/// `ncols, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b,
/// batch_stride_d, fusion_flags, nei0, ne11, expert_i1, nbi1`.
///
/// Mapped from `ggml_vk_mul_mat_vec_id_q_f16` (ggml-vulkan.cpp:8454) for ONE
/// decode token routed to `n_experts` (= top-k) experts, with the activation
/// shared by every expert (`ne11 = 1`, single `expert_i1 = 0` batch). Field map:
///
/// - `ncols` = `ne00` = the expert in-dim (per-row width, elements).
/// - `stride_a` = `ne10` = `ncols` (weight row stride; the real expert offset is
///   `batch_stride_a`).
/// - `stride_b` = `ne10` = `ncols` (activation row stride, elements).
/// - `stride_d` = `ne01` = `nrows` = per-expert OUTPUT row count. BOTH the
///   row-count guard `main()` checks `first_row` against AND the per-expert dst
///   stride (`d_offset = expert_i0 * stride_d`): expert `i` writes rows
///   `[i*nrows .. +nrows)`.
/// - `batch_stride_a` = `ncols * nrows` = the full expert matrix in ELEMENTS; the
///   shader does `expert_id * (batch_stride_a / QUANT_K)` to land on expert
///   `data_ids[expert_i0]`'s slice.
/// - `batch_stride_b` = `ncols` (one token's activation; unused at expert_i1=0).
/// - `batch_stride_d` = `nrows * n_experts` (full dst; unused at expert_i1=0).
/// - `fusion_flags` = 0 (no bias/scale fusion; bindings 3/4 unread).
/// - `nei0` = `n_experts` = experts selected for this token (the y dispatch
///   dimension; `expert_i0 = gl_WorkGroupID.y`).
/// - `ne11` = 1 (one token → every expert reads activation offset 0).
/// - `expert_i1` = 0 (single-token batch row).
/// - `nbi1` = `n_experts` = the id-buffer row stride (`data_ids[expert_i0 +
///   expert_i1*nbi1]`; irrelevant at expert_i1=0, set to the natural `nei0`).
///
/// Binding order is the plain GEMV's 5 buffers + a 6th: `[A weights (stacked
/// experts), B q8_1_x4 activation, D f32 dst (n_experts*nrows rows), Fuse0, Fuse1,
/// IDS (i32 expert-id list)]`.
pub fn gemv_id_params(ncols: u32, nrows: u32, n_experts: u32) -> KernelParams {
    KernelParams::from_words(vec![
        ncols,             // ncols: per-row width in elements
        ncols,             // stride_a: weight row stride (elements)
        ncols,             // stride_b: activation row stride (elements)
        nrows,             // stride_d: per-expert row count guard + dst stride
        ncols * nrows,     // batch_stride_a: full expert matrix (elements) -> /QUANT_K
        ncols,             // batch_stride_b: one token's activation
        nrows * n_experts, // batch_stride_d: full dst
        0,                 // fusion_flags: no bias/scale fusion
        n_experts,         // nei0: experts selected for this token
        1,                 // ne11: single token
        0,                 // expert_i1: single-token batch row
        n_experts,         // nbi1: id-buffer row stride
    ])
}

/// Dispatch grid for [`gemv_id_params`]: `x` = output rows per expert (each
/// workgroup computes `NUM_ROWS=1` rows of one expert), `y` = `n_experts` (the
/// expert slot; `mul_mat_vec_base.glsl` reads `expert_i0 = gl_WorkGroupID.y`).
pub fn gemv_id_dispatch(nrows: u32, n_experts: u32) -> Dispatch {
    Dispatch {
        x: nrows.max(1),
        y: n_experts.max(1),
        z: 1,
    }
}

pub const Q8_1_X4_VALUES_PER_GROUP: u32 = 128;

pub fn q8_1_quantize_params(ne: u32) -> KernelParams {
    KernelParams::from_words(vec![ne, ne.div_ceil(Q8_1_X4_VALUES_PER_GROUP)])
}

pub fn q8_1_quantize_dispatch(ne: u32) -> Dispatch {
    Dispatch::x(ne.div_ceil(Q8_1_X4_VALUES_PER_GROUP).max(1))
}

// RoPE (NeoX) push-constant contract (full-attention on-device). The
// `rope_neox.comp` (+ `rope_head.glsl`, `rope_funcs.glsl`) rotates pairs
// `(x[d], x[d + n_dims/2])` by `angle = pos * theta_scale^d` for `d < n_dims/2`,
// passing through dims `>= n_dims`. The host f32 reference it replaces is
// `forward.rs::rope_neox`, where `theta_scale = rope_theta^(-2/rotary_dim)` so
// `theta_scale^d = rope_theta^(-2d/rotary_dim) = inv_freq(d)`. We apply RoPE to
// one (or several batched) head vectors of `head_dim` elements at a fixed
// position, with no YaRN (ext_factor=0), no freq-factor table (has_ff=0), no
// set_rows (set_rows_stride=0), and no mscale (attn_factor=1, freq_scale=1).

/// Push-constant block for `rope_neox.comp` = the `rope_params` struct in
/// `rope_params.glsl`, laid out as 29 std430 4-byte words in declared order:
/// `rope_mode, nrows, n_dims, freq_scale(f32), freq_base(f32), ext_factor(f32),
/// attn_factor(f32), corr_dims[0](f32), corr_dims[1](f32), theta_scale(f32),
/// has_ff, sections[0..4](i32), is_imrope, is_back, set_rows_stride, ne00, ne01,
/// ne02, nb01, nb02, nb03, nb11, nb12, nb13, a_offset, d_offset`.
///
/// For `nrows` independent head vectors, each `head_dim` f32 wide and laid out
/// contiguously (row stride = `head_dim`), rotated at a single absolute
/// `pos` with rotary width `rotary_dim` (= n_dims):
/// - `rope_mode = 2` (GGML_ROPE_TYPE_NEOX), unused by `rope_neox()` itself but
///   set for clarity.
/// - `nrows` = the `main()` row guard (`row >= pc.nrows` returns).
/// - `n_dims = rotary_dim` — the shader rotates `d < n_dims/2` and passes the
///   rest through. With `rotary_dim == head_dim`, every dim rotates.
/// - `freq_scale = 1, ext_factor = 0, attn_factor = 1` → `rope_yarn` reduces to
///   `cos_theta = cos(theta_base)`, `sin_theta = sin(theta_base)` (no YaRN,
///   no mscale) — matching the host's bare `(sin, cos)`.
/// - `corr_dims = [0, 0]` (unused when `ext_factor == 0`).
/// - `theta_scale = rope_theta^(-2/rotary_dim)` so `theta_scale^d = inv_freq(d)`.
/// - `has_ff = 0` (binding 2 freq table unread; bind a dummy).
/// - `sections = [0;4], is_imrope = 0, is_back = 0` (NeoX, not mRoPE).
/// - `set_rows_stride = 0` (binding 4 indices unread; bind a dummy).
/// - `ne00 = n_dims` — the `i0 >= p.ne00` early-out guard. `rope_head.glsl`
///   dispatches `i0 = 2*gl_GlobalInvocationID.y`, so y must cover `n_dims/2`
///   pairs; `ne00 = n_dims` lets the last pair (i0 = n_dims-2) run and rejects
///   any padded invocation.
/// - `ne01 = nrows, ne02 = 1` — used by `main()` to decompose `row` into
///   `(i1, i2, i3)`; with `ne02 = 1` every row is `i1 = row, i2 = i3 = 0`, so a
///   single shared `pos` is read from `rope_data_pos[i2] = pos[0]`.
/// - `nb01 = head_dim` (input row stride, elements), `nb02 = nb03 = 0` (single
///   channel/sample). The input coord is `a_offset + i1*nb01 + (i0/2)`.
/// - `nb11 = head_dim, nb12 = nb13 = 0` (output row stride). The NeoX dst coord
///   is `i0/2 + i1*nb11`, pair halves at `+0` and `+n_dims/2`.
/// - `a_offset = d_offset = 0` (caller binds the head sub-range directly).
///
/// Binding order: `[0 = X input f32, 1 = Y pos int[], 2 = Z freq f32 (dummy),
/// 3 = D output, 4 = I uvec2 indices (dummy)]`.
pub fn rope_neox_params(
    head_dim: u32,
    rotary_dim: u32,
    nrows: u32,
    rope_theta: f32,
) -> KernelParams {
    let n_dims = rotary_dim;
    let theta_scale = rope_theta.powf(-2.0 / n_dims as f32);
    KernelParams::from_words(vec![
        2,                     // rope_mode = GGML_ROPE_TYPE_NEOX
        nrows,                 // nrows (row guard)
        n_dims,                // n_dims = rotary_dim
        1.0f32.to_bits(),      // freq_scale
        rope_theta.to_bits(),  // freq_base (unused by rope_neox path beyond theta_scale)
        0.0f32.to_bits(),      // ext_factor = 0 (no YaRN)
        1.0f32.to_bits(),      // attn_factor = 1 (no mscale)
        0.0f32.to_bits(),      // corr_dims[0]
        0.0f32.to_bits(),      // corr_dims[1]
        theta_scale.to_bits(), // theta_scale = rope_theta^(-2/n_dims)
        0,                     // has_ff = 0 (freq table unread)
        0,                     // sections[0]
        0,                     // sections[1]
        0,                     // sections[2]
        0,                     // sections[3]
        0,                     // is_imrope = 0
        0,                     // is_back = 0
        0,                     // set_rows_stride = 0 (indices unread)
        n_dims,                // ne00 = n_dims (i0 >= ne00 early-out)
        nrows,                 // ne01 = nrows (row decomposition)
        1,                     // ne02 = 1 (single channel => pos[i2]=pos[0])
        head_dim,              // nb01 = input row stride (elements)
        0,                     // nb02
        0,                     // nb03
        head_dim,              // nb11 = output row stride (elements)
        0,                     // nb12
        0,                     // nb13
        0,                     // a_offset
        0,                     // d_offset
    ])
}

/// Dispatch grid for [`rope_neox_params`]. `rope_head.glsl` has
/// `local_size = (1, 256, 1)` and indexes `i0 = 2*gl_GlobalInvocationID.y`,
/// `row = gl_GlobalInvocationID.x + 32768*z`. So x covers the rows and y covers
/// the `n_dims/2` rotation pairs (each thread does one pair). With local_y=256,
/// `y_groups = ceil(rotary_dim/2 / 256)`; one workgroup per row in x.
pub fn rope_neox_dispatch(rotary_dim: u32, nrows: u32) -> Dispatch {
    let pairs = (rotary_dim / 2).max(1);
    Dispatch {
        x: nrows.max(1),
        y: pairs.div_ceil(256).max(1),
        z: 1,
    }
}

// Elementwise / norm push-constant contracts. These move
// the per-layer RMSNorm / SwiGLU / residual-Add off the host (where each forced
// a device→host→device hop around a GEMV) onto the already-compiled device
// kernels, reverse-engineered from their `.comp` push-constant interfaces.

/// `rms_norm.comp` (+ `generic_binary_head.glsl`, all-f32, no rope fusion),
/// applying the PLAIN weight `out[i] = x[i] * inv_rms * w[i]` for ONE row of
/// `ncols` elements. Dispatch is a single workgroup (`gl_NumWorkGroups.x = 1`,
/// `BLOCK_SIZE = 512` threads reducing the row). Bindings: `0 = A` input (read),
/// `1 = B` weight (read), `2 = D` output (write). The `do_multiply` spec
/// constant (id 1 = 1) selects the weighted branch; `eps` is `param1`.
///
/// The push block is the 29-uint `generic_binary_head.glsl` `parameter`:
/// `ne, ne00..ne03, nb00..nb03, ne10..ne13, nb10..nb13, ne20..ne23, nb20..nb23,
/// misalign_offsets, param1(f32 eps), param2(f32), param3(i32)`. For a single
/// `[ncols]` row: `ne00 = ne10 = ne20 = ncols`, all higher dims 1, every stride
/// the natural row width (so all per-row offsets resolve to 0 and the weight is
/// indexed plainly by column since `ncols <= ne10`).
pub fn rms_norm_params(ncols: u32, eps: f32) -> KernelParams {
    let n = ncols;
    KernelParams::from_words(vec![
        n, // ne (total elements)
        n,
        1,
        1,
        1, // ne00..ne03
        1,
        n,
        n,
        n, // nb00..nb03 (nb00=1 element stride; row strides = n)
        n,
        1,
        1,
        1, // ne10..ne13 (weight: ncols <= ne10 => plain column index)
        1,
        n,
        n,
        n, // nb10..nb13
        n,
        1,
        1,
        1, // ne20..ne23
        1,
        n,
        n,
        n,             // nb20..nb23
        0,             // misalign_offsets
        eps.to_bits(), // param1 (f32 eps)
        0,             // param2 (f32, unused)
        0,             // param3 (i32, unused by rms_norm)
    ])
}

/// One workgroup reduces the whole row in `rms_norm.comp`, so the grid is a
/// single workgroup regardless of `ncols` (the 512-thread block strides the row).
pub fn rms_norm_dispatch() -> Dispatch {
    Dispatch::x(1)
}

/// `swiglu.comp` (+ `glu_head.glsl` / `glu_main.glsl`) in SPLIT mode (`mode=2`):
/// `out[i] = silu(a[i]) * b[i]` over `n` elements with `a` = gate (binding 0),
/// `b` = up (binding 1), `d` = out (binding 2). The 16-uint push block is
/// `N, ne00, ne20, mode, alpha(f32), limit(f32), nb01, nb02, nb03, ne01, ne02,
/// nb11, nb12, nb13, ne11, ne12`. For a flat `[n]` row in split mode: `N = n`,
/// `ne20 = n` (so every element has row 0), strides natural, `ne00` unused
/// (split path ignores the `ne00/2` half-offset). `local_size_x = 512`.
pub fn swiglu_params(n: u32) -> KernelParams {
    KernelParams::from_words(vec![
        n,     // N (element guard)
        2 * n, // ne00 (gate||up width; unused in split mode, set for completeness)
        n,     // ne20 (dst row width => row = i/ne20 = 0)
        2,     // mode = 2 (Split: op(a[i], b[i]))
        0,     // alpha (f32, unused by swiglu op)
        0,     // limit (f32, unused)
        n,
        n,
        n, // nb01, nb02, nb03 (src row strides)
        1,
        1, // ne01, ne02
        n,
        n,
        n, // nb11, nb12, nb13 (dst row strides)
        1,
        1, // ne11, ne12
    ])
}

/// SwiGLU grid: `glu_main.glsl` derives `i` from a 512-wide x dimension, so one
/// workgroup per 512 elements.
pub fn swiglu_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(512).max(1))
}

/// `add.comp` (+ `generic_binary_head.glsl`, `ADD_RMS=0`): `out[i] = a[i] + b[i]`
/// over `n` elements. Same 29-uint generic-binary push block as `rms_norm`, with
/// `param3 = 0` (skip the RMS-fused reduction). Bindings: `0=A, 1=B, 2=D` — the
/// shader's optional `3=PartialBuf` is dead-code-eliminated by `glslc -O` when
/// built with `ADD_RMS=0`, leaving 3 bindings.
pub fn add_params(n: u32) -> KernelParams {
    KernelParams::from_words(vec![
        n, // ne
        n, 1, 1, 1, // ne00..ne03
        1, n, n, n, // nb00..nb03
        n, 1, 1, 1, // ne10..ne13
        1, n, n, n, // nb10..nb13
        n, 1, 1, 1, // ne20..ne23
        1, n, n, n, // nb20..nb23
        0, // misalign_offsets
        0, // param1 (f32, unused)
        0, // param2 (f32, unused)
        0, // param3 (i32) = 0 => no RMS reduction (binding 3 untouched)
    ])
}

/// Add grid. `add.comp` uses `local_size_x = 256` and `num_iter = 2`, with each
/// thread `t` of workgroup `wg` handling global-thread `wg*256 + t` (iter 0) and
/// `+256` (iter 1) via `get_idx()`. The coverage of `G` workgroups is therefore
/// `[0, G*256 + 256)`, so `G = ceil(n / 256)` (which gives `G*256 >= n` and thus
/// `G*256 + 256 > n`) covers every element. Dispatching `ceil(n/512)` instead
/// would leave the top `~n/2 mod 512` elements unwritten — the oracle test caught
/// exactly that (e.g. n=5120 left `[2816, 5120)` at 0).
pub fn add_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(256).max(1))
}

/// `scaled_add.comp` (ARLE-local): `out[i] = a[i] + scale * b[i]` over `n`
/// elements. Bindings `0=A` (accumulator, read), `1=B` (addend, read),
/// `2=D` (out, write) — same 3-binding layout as `add`, so it shares the
/// decode `ring3`. The 2-field push is `[n (u32), scale (f32 bits)]`.
///
/// Folds the MoE router weight into the per-expert accumulate
/// (`acc += w_e * y_e`) so the whole accumulate stays device-resident — no host
/// readback of the expert output to scale + add.
pub fn scaled_add_params(n: u32, scale: f32) -> KernelParams {
    KernelParams::from_words(vec![n, scale.to_bits()])
}

/// `scaled_add.comp` grid: one thread per element, `local_size_x = 256`, so
/// `ceil(n / 256)` workgroups cover the row.
pub fn scaled_add_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(256).max(1))
}

/// `sigmoid_mul.comp` (ARLE-local): `out[i] = sigmoid(a[i]) * b[i]` over `n`
/// elements. Bindings `0=A` (gate, read), `1=B` (value, read), `2=D` (out,
/// write) — same 3-binding layout as `add`/`scaled_add`, so it shares the
/// decode `ring3`. The single push field is `[n (u32)]`. Applies the
/// full-attention per-head sigmoid gate device-resident.
pub fn sigmoid_mul_params(n: u32) -> KernelParams {
    KernelParams::from_words(vec![n])
}

/// `sigmoid_mul.comp` grid: one thread per element, `local_size_x = 256`, so
/// `ceil(n / 256)` workgroups cover the row.
pub fn sigmoid_mul_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(256).max(1))
}

/// `f16_kv_pack.comp` (ARLE-local): pack `n` f32 values into f16
/// (`dst[i] = float16_t(src[i])`). Bindings `0=A` (f32 src, read), `1=D` (f16
/// dst, write) — a 2-binding layout, so it shares the decode `ring2` with the
/// q8_1 quantize. The single push field is `[n (u32)]`. Writes one full-attention
/// head row (`head_dim` f16) into the device KV cache plane bound at the
/// `(layer, kv_head, pos)` byte offset, removing the host K/V readback+convert.
pub fn f16_kv_pack_params(n: u32) -> KernelParams {
    KernelParams::from_words(vec![n])
}

/// `f16_kv_pack.comp` grid: one thread per element, `local_size_x = 256`, so
/// `ceil(n / 256)` workgroups cover the head row.
pub fn f16_kv_pack_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(256).max(1))
}

// Qwen3.5 gated-delta linear-attention push-constant contracts (linear-attention
// on-device port). The two model-specific shaders (`qwen35_ssm_conv.comp`,
// `qwen35_gated_delta_net.comp`) replace the host depthwise-conv1d + recurrent
// gated-delta routines in `infer-vulkan`'s `linear_attention`. Both are now
// throughput kernels filling a full workgroup: conv runs one thread per channel
// (`local_size_x = 256`, `ceil(channels/256)` workgroups); gated-delta runs one
// workgroup per value head with `local_size_x = 128` threads over the `val_dim`
// state columns. Still oracle-gated byte-for-byte against the host f32 routine
// (the per-head scalar reductions keep the host's serial order).

/// Push-constant block for `qwen35_ssm_conv.comp` = `{num_channels, seq_len,
/// kernel_size}` (3 `uint`s). Depthwise causal conv1d over all `qkv` channels:
/// taps `[ring | x]` (ring = the previous `kernel-1` inputs per channel), then
/// `silu(round_to_bf16(sum))`, then advances the per-channel ring in place.
/// Bindings (in order): `0 = XSeq [seq_len*num_channels] f32` (row-major, token
/// `t` at `t*num_channels`), `1 = ConvWeight [num_channels*kernel_size] f32`
/// (row-major, channel `c` at `c*kernel_size`), `2 = ConvState
/// [num_channels*(kernel-1)] f32` (the ring, read+written), `3 = OutSeq
/// [seq_len*num_channels] f32`.
pub fn qwen35_ssm_conv_params(num_channels: u32, seq_len: u32, kernel_size: u32) -> KernelParams {
    KernelParams::from_words(vec![num_channels, seq_len, kernel_size])
}

/// Dispatch grid for `qwen35_ssm_conv.comp`: `local_size_x = 256`, one invocation
/// per channel (`gl_GlobalInvocationID.x = c`), so `ceil(num_channels/256)`
/// workgroups cover all channels (tail lanes self-mask on `c >= num_channels`).
pub fn qwen35_ssm_conv_dispatch(num_channels: u32) -> Dispatch {
    Dispatch::x(num_channels.div_ceil(256).max(1))
}

/// Push-constant block for `qwen35_gated_delta_net.comp` = `{num_key_heads,
/// num_value_heads, key_dim, val_dim, seq_len}` (5 `uint`s). The recurrent
/// gated-delta state update for one (or more) tokens: per value head it
/// l2-normalizes q/k over `key_dim` (eps 1e-12), scales q by `1/sqrt(key_dim)`,
/// decays the `[key_dim, val_dim]` state by `exp_g`, does the two-pass rank-1
/// update, and writes `S^T q`. Bindings (in order): `0 = Qkv
/// [seq_len*(2*nk*kd + nv*vd)] f32` (post-conv `[q|k|v]`, token-major),
/// `1 = BProj [seq_len*nv] f32`, `2 = AProj [seq_len*nv] f32`, `3 = DtBias [nv]
/// f32`, `4 = ALog [nv] f32` (the GGUF `ssm_a`, already `= -exp(A_log)`),
/// `5 = State [nv*kd*vd] f32` (val contiguous, read+written), `6 = Output
/// [seq_len*nv*vd] f32`.
pub fn qwen35_gated_delta_net_params(
    num_key_heads: u32,
    num_value_heads: u32,
    key_dim: u32,
    val_dim: u32,
    seq_len: u32,
) -> KernelParams {
    KernelParams::from_words(vec![
        num_key_heads,
        num_value_heads,
        key_dim,
        val_dim,
        seq_len,
    ])
}

/// Dispatch grid for `qwen35_gated_delta_net.comp`: ONE workgroup per value head
/// (`gl_WorkGroupID.x = v_head`, `x = num_value_heads`) with `local_size_x = 128`
/// threads mapping the `val_dim` state columns. Each thread owns its columns for
/// the whole sequence, so the recurrence needs no shared memory or barriers.
pub fn qwen35_gated_delta_net_dispatch(num_value_heads: u32) -> Dispatch {
    Dispatch::x(num_value_heads.max(1))
}

/// Push-constant block for `qwen36_router_topk.comp` = `{n_expert, top_k,
/// norm_topk}` (3 `uint`s). Bindings (in order): `0 = Logits [n_expert] f32`
/// (read), `1 = Ids [top_k] i32` (write), `2 = Weights [top_k] f32` (write).
/// `norm_topk` is 1 to renormalize the kept weights to sum 1 (clamp F16-min).
pub fn qwen36_router_topk_params(n_expert: u32, top_k: u32, norm_topk: bool) -> KernelParams {
    KernelParams::from_words(vec![n_expert, top_k, u32::from(norm_topk)])
}

/// Dispatch grid for `qwen36_router_topk.comp`: ONE workgroup of 256 threads
/// (one expert per lane); parallel max/Σexp reductions + `top_k` parallel
/// argmax-and-mask rounds. (The earlier single-thread `local_size_x=1` version
/// was ~428µs/call — ~46% of MoE decode — wasting the whole wave.)
pub fn qwen36_router_topk_dispatch() -> Dispatch {
    Dispatch { x: 1, y: 1, z: 1 }
}

/// Push-constant block for `qwen36_router_gemv.comp` = `{n_out, hidden,
/// apply_sigmoid}` (3 `uint`s). Bindings (in order): `0 = Input [hidden] f32`
/// (read), `1 = Weight [n_out*hidden] f32` (read, GGUF row-major [out,in]),
/// `2 = Output [n_out] f32` (write). `apply_sigmoid` = 1 folds the shared-expert
/// sigmoid gate.
pub fn qwen36_router_gemv_params(n_out: u32, hidden: u32, apply_sigmoid: bool) -> KernelParams {
    KernelParams::from_words(vec![n_out, hidden, u32::from(apply_sigmoid)])
}

/// Dispatch grid for `qwen36_router_gemv.comp`: ONE workgroup per output row
/// (`local_size_x = 64` lanes cooperate on the row's dot with coalesced reads),
/// so `n_out` workgroups.
pub fn qwen36_router_gemv_dispatch(n_out: u32) -> Dispatch {
    Dispatch::x(n_out.max(1))
}

/// Push-constant block for `qwen36_moe_weighted_accum.comp` = `{hidden, count,
/// init}` (3 `uint`s). Bindings (in order): `0 = Src [count*hidden] f32` (read,
/// expert-major), `1 = Weights [count] f32` (read), `2 = Acc [hidden] f32`
/// (read+write). `init` = 1 starts the accumulate from 0; 0 adds into `acc`.
pub fn qwen36_moe_weighted_accum_params(hidden: u32, count: u32, init: bool) -> KernelParams {
    KernelParams::from_words(vec![hidden, count, u32::from(init)])
}

/// Dispatch grid for `qwen36_moe_weighted_accum.comp`: one thread per hidden
/// element (`local_size_x = 256`), `ceil(hidden/256)` workgroups.
pub fn qwen36_moe_weighted_accum_dispatch(hidden: u32) -> Dispatch {
    Dispatch::x(hidden.div_ceil(256).max(1))
}

// Flash-attention push-constant contract (full-attention on-device).
// `flash_attn.comp` (+ `flash_attn_base.glsl`) is the scalar/subgroup-shuffle
// FA path llama.cpp uses for N==1 decode on RDNA (coopmat is NV/prefill). The
// registered [`FlashAttentionSpec::f32_f16`] specializes it to:
//   WorkGroupSize=128, Br=1, Bc=64, HSK=HSV=head_dim, Clamp=1 (KV not a multiple
//   of Bc), D_split=8, row_split=1, SubGroupSize=32, SHMEM_STAGING=0, Flags=0
//   (no mask / no softcap), FaTypeK=FaTypeV=F16 (1), FaBlockBytesK/V=2.
// So Q is f32, K/V are f16 (read directly, no dequant), output is f32. The host
// reference it replaces is the `full_attention` causal-SDPA inner loop (scale
// 1/sqrt(head_dim), online softmax) — computed over the SAME f16-rounded K/V.
//
// This contract drives ONE query head against `[kv_len, head_dim]` f16 K and V
// (single KV head; the forward dispatches once per query head with the head's
// own K/V sub-range, so gqa_ratio stays 1 and the GQA mapping is host-side).

/// Push-constant block for `flash_attn.comp` = the 33-field `parameter` struct
/// in `flash_attn_base.glsl`, in declared order:
/// `N, KV, ne1, ne2, ne3, neq2, neq3, nek2, nek3, nev2, nev3, nem1, nem2, nem3,
/// nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, scale(f32),
/// max_bias(f32), logit_softcap(f32), mask_n_head_log2, m0(f32), m1(f32),
/// gqa_ratio, split_kv, k_num`.
///
/// For ONE decode query (`N=1`) of `hsk`-wide Q against `kv_len` cached K/V rows
/// of `hsk`/`hsv` width, single head (gqa_ratio=1), no split-k (k_num=1), no mask
/// (Flags has no MASK bit), no ALiBi (max_bias=0):
/// - `N=1` (one query row), `KV=kv_len` (cached length, the `j*Bc+c < KV` guard).
/// - `ne1=1` (query rows → output row stride), `ne2=1` (q heads in this dispatch
///   → output head stride), `ne3=1`.
/// - `neq2=neq3=nek2=nek3=nev2=nev3=1` (single head/batch → broadcast ratios
///   `rk2=rk3=rv2=rv3=1`, so `ik2=ik3=iv2=iv3=0`).
/// - `nem1=nem2=nem3=1` (mask dims; mask unread).
/// - `nb01=hsk` (Q row stride, elements; `q_stride=nb01` when gqa_ratio=1),
///   `nb02=nb03=hsk` (unused single-head).
/// - `nb11=hsk` (K row stride in ELEMENTS — f16 `data_kv4` is indexed
///   `k_offset/4 + kvrow*(nb11/4) + d`, so a `hsk`-wide row strides `hsk`
///   elements = `hsk/4` vec4s), `nb12=nb13=0` (single KV head; `k_offset=0`).
/// - `nb21=hsv` (V row stride, elements), `nb22=nb23=0`.
/// - `scale` = `1/sqrt(head_dim)` (applied to Q when staged into `Qf`).
/// - `max_bias=0` (slope=1, no ALiBi), `logit_softcap=0`.
/// - `mask_n_head_log2=0` (no sink bit, no ALiBi head split), `m0=m1=0`.
/// - `gqa_ratio=1` (one query head per workgroup; the forward maps GQA on host
///   by passing each query head its KV head's cache), `split_kv=KV` (so
///   `end_j = ceil(KV/Bc)` covers every cached position), `k_num=1` (no split-k
///   reduce pass; the shader divides by L and writes the final O directly).
///
/// Binding order: `[0 = Q f32, 1 = K f16, 2 = V f16, 3 = M mask f16 (dummy),
/// 4 = S sinks f32 (dummy), 5 = O f32 output, 6 = MO mask_opt uint (dummy)]`.
pub fn flash_attn_params(hsk: u32, hsv: u32, kv_len: u32, scale: f32) -> KernelParams {
    KernelParams::from_words(vec![
        1,                // N (query rows)
        kv_len,           // KV (cached length)
        1,                // ne1 (query rows -> O row stride)
        1,                // ne2 (q heads -> O head stride)
        1,                // ne3
        1,                // neq2
        1,                // neq3
        1,                // nek2
        1,                // nek3
        1,                // nev2
        1,                // nev3
        1,                // nem1
        1,                // nem2
        1,                // nem3
        hsk,              // nb01 (Q row stride, elements)
        hsk,              // nb02
        hsk,              // nb03
        hsk,              // nb11 (K row stride, elements)
        0,                // nb12 (single KV head)
        0,                // nb13
        hsv,              // nb21 (V row stride, elements)
        0,                // nb22
        0,                // nb23
        scale.to_bits(),  // scale = 1/sqrt(head_dim)
        0.0f32.to_bits(), // max_bias = 0 (no ALiBi)
        0.0f32.to_bits(), // logit_softcap = 0
        0,                // mask_n_head_log2 (no sink, no ALiBi split)
        0.0f32.to_bits(), // m0
        0.0f32.to_bits(), // m1
        1,                // gqa_ratio = 1
        kv_len,           // split_kv = KV (end_j covers all positions)
        1,                // k_num = 1 (no split-k reduce)
    ])
}

/// Dispatch grid for [`flash_attn_params`]: ONE workgroup. `init_indices` reads
/// `i = gl_WorkGroupID.x` (query tile), `iq2 = gl_WorkGroupID.y * gqa_ratio`
/// (head), `iq3 = gl_WorkGroupID.z` (batch). For a single query head / single
/// batch decode token, all three are 0, so `(1, 1, 1)`.
pub fn flash_attn_dispatch() -> Dispatch {
    Dispatch { x: 1, y: 1, z: 1 }
}

macro_rules! launcher_fns {
    ($call:path, $call_params:path) => {
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

        // The `mul_mat_vecq` GEMV requires the 13-uint push-constant block from
        // `mul_mat_vec_base.glsl` (ncols/strides/row-count). The no-push
        // launchers above are insufficient on their own — use these
        // `*_with_params` variants (see [`gemv_params`]) for a working matvec.
        // Buffer order: [A weights, B q8_1_x4 activations, D f32 dst,
        // Fuse0, Fuse1] — bindings 3/4 are declared by the shader but only read
        // when `fusion_flags != 0`; bind small dummies to satisfy the layout.
        pub fn q4_k_gemv_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvQ4K, ctx, buffers, dispatch, params)
        }

        pub fn q5_k_gemv_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvQ5K, ctx, buffers, dispatch, params)
        }

        pub fn q6_k_gemv_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvQ6K, ctx, buffers, dispatch, params)
        }

        pub fn q8_0_gemv(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
        ) -> Result<()> {
            $call(Kernel::GemvQ8_0, ctx, buffers, dispatch)
        }

        pub fn q8_0_gemv_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvQ8_0, ctx, buffers, dispatch, params)
        }

        // Fused MoE expert GEMV (`mul_mat_vec_id`). Buffer order is the plain
        // GEMV's 5 + a 6th IDS binding: [A stacked-expert weights, B q8_1_x4
        // activation, D f32 dst (n_experts*nrows rows), Fuse0, Fuse1, IDS (i32
        // expert ids)]. Push = [`gemv_id_params`]; dispatch = [`gemv_id_dispatch`].
        pub fn q4_k_gemv_id_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvIdQ4K, ctx, buffers, dispatch, params)
        }

        pub fn q5_k_gemv_id_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvIdQ5K, ctx, buffers, dispatch, params)
        }

        pub fn q6_k_gemv_id_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvIdQ6K, ctx, buffers, dispatch, params)
        }

        pub fn q8_0_gemv_id_with_params(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::GemvIdQ8_0, ctx, buffers, dispatch, params)
        }

        pub fn q8_1_quantize(
            ctx: &vulkan_sys::VulkanContext,
            buffers: &[&vulkan_sys::DeviceBuffer<'_>],
            dispatch: Dispatch,
            params: &KernelParams,
        ) -> Result<()> {
            $call_params(Kernel::QuantizeQ8_1, ctx, buffers, dispatch, params)
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
    use super::{Dispatch, FlashAttentionSpec, Kernel, Result};

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
        launch_with_params_and_specialization(
            kernel,
            ctx,
            buffers,
            dispatch,
            params,
            kernel.specialization_u32(),
        )
    }

    /// Single-dispatch launcher routed through a transient [`KernelCache`].
    ///
    /// The per-dispatch object graph (`fs::read(.spv)` → `ShaderModule` →
    /// `DescriptorSetLayout` → `ComputePipeline`) now lives in the cache's
    /// cache-miss builder (`cache.rs`), and the bind+push+dispatch body is
    /// `record_dispatch` into a `CommandRecorder` — no NULL-fence
    /// `one_shot_submit` drain on this path. A persistent cache + a batch-record
    /// `CommandRecorder` is the real decode path; this keeps
    /// the proven single-shot launchers/tests working through the same builder.
    pub fn launch_with_params_and_specialization(
        kernel: Kernel,
        ctx: &vulkan_sys::VulkanContext,
        buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        dispatch: Dispatch,
        params: &super::KernelParams,
        specialization_u32: &[(u32, u32)],
    ) -> Result<()> {
        let push_bytes = params.to_le_bytes();
        let mut cache = crate::KernelCache::new();
        crate::launch_cached(
            &mut cache,
            ctx,
            kernel,
            buffers,
            dispatch,
            &push_bytes,
            specialization_u32,
        )
    }

    pub fn flash_attn_with_params_and_spec(
        ctx: &vulkan_sys::VulkanContext,
        buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        dispatch: Dispatch,
        params: &super::KernelParams,
        spec: &FlashAttentionSpec,
    ) -> Result<()> {
        launch_with_params_and_specialization(
            Kernel::FlashAttn,
            ctx,
            buffers,
            dispatch,
            params,
            spec.specialization_u32(),
        )
    }
}

#[cfg(feature = "vulkan")]
launcher_fns!(real::launch, real::launch_with_params);
#[cfg(feature = "vulkan")]
fused_launcher_fns!(real::launch_with_params);

#[cfg(not(feature = "vulkan"))]
mod stub {
    use super::{FlashAttentionSpec, Kernel, KernelError, Result};

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

    pub fn flash_attn_with_params_and_spec(
        _ctx: &vulkan_sys::VulkanContext,
        _buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        _dispatch: super::Dispatch,
        _params: &super::KernelParams,
        _spec: &FlashAttentionSpec,
    ) -> Result<()> {
        Err(KernelError::NotCompiled)
    }
}

#[cfg(not(feature = "vulkan"))]
launcher_fns!(stub::launch, stub::launch_with_params);
#[cfg(not(feature = "vulkan"))]
fused_launcher_fns!(stub::launch_with_params);

#[cfg(feature = "vulkan")]
pub use real::flash_attn_with_params_and_spec;
#[cfg(not(feature = "vulkan"))]
pub use stub::flash_attn_with_params_and_spec;
