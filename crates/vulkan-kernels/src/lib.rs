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
const SPEC_GEMV_K_Q8_1: &[(u32, u32)] = &[(0, 32), (1, 1), (2, 1)];
const SPEC_RMS_NORM_MUL: &[(u32, u32)] = &[(1, 1)];

impl Kernel {
    pub const ALL: &'static [Self] = &[
        Self::MmvqIq2Xxs,
        Self::MmvqQ2K,
        Self::GemvQ4K,
        Self::GemvQ5K,
        Self::GemvQ6K,
        Self::GemvQ8_0,
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
            Kernel::GemvQ8_0 => "mul_mat_vecq_q8_0",
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
            Kernel::GemvQ4K | Kernel::GemvQ5K | Kernel::GemvQ6K | Kernel::GemvQ8_0 => {
                SPEC_GEMV_K_Q8_1
            }
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

pub const Q8_1_X4_VALUES_PER_GROUP: u32 = 128;

pub fn q8_1_quantize_params(ne: u32) -> KernelParams {
    KernelParams::from_words(vec![ne, ne.div_ceil(Q8_1_X4_VALUES_PER_GROUP)])
}

pub fn q8_1_quantize_dispatch(ne: u32) -> Dispatch {
    Dispatch::x(ne.div_ceil(Q8_1_X4_VALUES_PER_GROUP).max(1))
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise / norm push-constant contracts (perf-parity Step 5b). These move
// the per-layer RMSNorm / SwiGLU / residual-Add off the host (where each forced
// a device→host→device hop around a GEMV) onto the already-compiled device
// kernels, reverse-engineered from their `.comp` push-constant interfaces.
// ─────────────────────────────────────────────────────────────────────────────

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
    /// `record_dispatch` into a Step-1 `CommandRecorder` — no NULL-fence
    /// `one_shot_submit` drain on this path. A persistent cache + a batch-record
    /// `CommandRecorder` is the real decode path (wired in Step 4); this keeps
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
        assert_eq!(q8_0_row_bytes(32), Some(34));
        assert_eq!(q8_0_row_bytes(256), Some(8 * 34));
        assert_eq!(q8_0_row_bytes(31), None);
    }

    #[test]
    fn gemv_params_match_mul_mat_vec_base_layout() {
        // 13 uints: ncols, stride_a, stride_b, stride_d, batch_stride_a,
        // batch_stride_b, batch_stride_d, fusion_flags, base_work_group_y,
        // ne02, ne12, broadcast2, broadcast3.
        let params = gemv_params(256, 4);
        assert_eq!(params.len_bytes(), 13 * 4);
        assert_eq!(
            params,
            KernelParams::from_words(vec![256, 256, 8, 4, 256, 0, 0, 0, 0, 1, 1, 1, 1])
        );
        // stride_d (index 3) is the row-count guard the shader checks.
        assert_eq!(gemv_dispatch(4), Dispatch::x(4));
        assert_eq!(gemv_dispatch(0), Dispatch::x(1));
    }

    #[test]
    fn shader_names_cover_generic_op_set() {
        let names = [
            Kernel::MmvqIq2Xxs.shader_name(),
            Kernel::MmvqQ2K.shader_name(),
            Kernel::GemvQ4K.shader_name(),
            Kernel::GemvQ5K.shader_name(),
            Kernel::GemvQ6K.shader_name(),
            Kernel::GemvQ8_0.shader_name(),
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
        assert_eq!(names.len(), 19);
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
        assert_eq!(
            Kernel::GemvQ8_0.specialization_u32(),
            Kernel::GemvQ4K.specialization_u32()
        );
        assert_eq!(Kernel::SoftMax.specialization_u32(), &[(0, 32)]);
        assert_eq!(Kernel::ArgMax.specialization_u32(), &[(0, 32)]);
        assert_eq!(Kernel::QuantizeQ8_1.specialization_u32(), &[(0, 32)]);
        assert_eq!(
            Kernel::FlashAttn.specialization_u32(),
            FlashAttentionSpec::f32_f16(256).specialization_u32()
        );
        assert_eq!(Kernel::FlashAttn.specialization_u32().len(), 16);
        assert_eq!(Kernel::FlashAttn.specialization_u32()[3], (3, 256));
        assert_eq!(Kernel::FlashAttn.specialization_u32()[4], (4, 256));
        assert_eq!(Kernel::FlashAttn.specialization_u32()[5], (5, 1));
        assert_eq!(Kernel::FlashAttn.specialization_u32()[12], (12, 1));
        assert_eq!(Kernel::FlashAttn.specialization_u32()[13], (13, 1));
        assert_eq!(Kernel::FlashAttn.specialization_u32()[14], (14, 2));
        assert_eq!(Kernel::FlashAttn.specialization_u32()[15], (15, 2));
        assert_eq!(Kernel::RmsNorm.specialization_u32(), &[(1, 1)]);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_feature_build_produces_every_registered_spv() {
        if option_env!("ARLE_VULKAN_GLSLC_PRESENT") != Some("1") {
            eprintln!("vulkan-kernels: glslc unavailable, skipping SPIR-V existence check");
            return;
        }
        let Some(dir) = option_env!("ARLE_VULKAN_SPV_DIR") else {
            eprintln!("vulkan-kernels: ARLE_VULKAN_SPV_DIR unset, skipping SPIR-V existence check");
            return;
        };
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

    #[test]
    fn rms_norm_params_match_generic_binary_head_layout() {
        // 29 uints: ne, ne00..ne03, nb00..nb03, ne10..ne13, nb10..nb13,
        // ne20..ne23, nb20..nb23, misalign, param1(f32 eps), param2, param3.
        let p = rms_norm_params(5120, 1e-6);
        assert_eq!(p.len_bytes(), 29 * 4);
        let w = p.words().to_vec();
        assert_eq!(w.len(), 29);
        assert_eq!(w[0], 5120); // ne
        assert_eq!(&w[1..5], &[5120, 1, 1, 1]); // ne00..ne03
        assert_eq!(&w[5..9], &[1, 5120, 5120, 5120]); // nb00..nb03
        assert_eq!(&w[9..13], &[5120, 1, 1, 1]); // ne10..ne13 (weight)
        assert_eq!(w[25], 0); // misalign_offsets
        assert_eq!(f32::from_bits(w[26]), 1e-6); // param1 = eps
        assert_eq!(w[28], 0); // param3
        assert_eq!(rms_norm_dispatch(), Dispatch::x(1));
    }

    #[test]
    fn swiglu_params_match_glu_split_layout() {
        // 16 uints: N, ne00, ne20, mode, alpha(f32), limit(f32), nb01..nb03,
        // ne01, ne02, nb11..nb13, ne11, ne12.
        let p = swiglu_params(17408);
        assert_eq!(p.len_bytes(), 16 * 4);
        let w = p.words().to_vec();
        assert_eq!(w[0], 17408); // N
        assert_eq!(w[1], 2 * 17408); // ne00
        assert_eq!(w[2], 17408); // ne20 (=> row 0)
        assert_eq!(w[3], 2); // mode = Split
        assert_eq!(swiglu_dispatch(17408), Dispatch::x(17408u32.div_ceil(512)));
        assert_eq!(swiglu_dispatch(0), Dispatch::x(1));
    }

    #[test]
    fn add_params_skip_rms_reduction() {
        // Same 29-uint generic-binary push block; param3 must be 0.
        let p = add_params(5120);
        assert_eq!(p.len_bytes(), 29 * 4);
        let w = p.words().to_vec();
        assert_eq!(w[0], 5120); // ne
        assert_eq!(&w[1..5], &[5120, 1, 1, 1]); // ne00..ne03
        assert_eq!(w[28], 0); // param3 = 0 => no RMS reduction
        // ceil(n/256) workgroups: each covers [wg*256, wg*256+512), so G*256 >= n
        // guarantees full coverage (ceil(n/512) would leave the tail unwritten).
        assert_eq!(add_dispatch(5120), Dispatch::x(5120u32.div_ceil(256)));
        assert_eq!(add_dispatch(0), Dispatch::x(1));
    }

    #[test]
    fn q8_1_quantize_params_match_x4_shader_contract() {
        let params = q8_1_quantize_params(257);
        assert_eq!(params.len_bytes(), 8);
        assert_eq!(params.to_le_bytes(), [1, 1, 0, 0, 3, 0, 0, 0]);
        assert_eq!(q8_1_quantize_dispatch(0), Dispatch::x(1));
        assert_eq!(q8_1_quantize_dispatch(128), Dispatch::x(1));
        assert_eq!(q8_1_quantize_dispatch(129), Dispatch::x(2));
        assert_eq!(q8_1_quantize_dispatch(257), Dispatch::x(3));
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
