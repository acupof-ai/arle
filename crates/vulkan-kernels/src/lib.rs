//! Vulkan kernel build + raw-buffer launch layer for the AIPC lane.
//!
//! The borrowed operator corpus is adapted from ggml-org/llama.cpp
//! `vulkan-shaders` @ d2462f8f (MIT). `build.rs` compiles selected shaders
//! with `glslc` when the `vulkan` feature is on; missing `glslc` or
//! unresolved macro variants leave a typecheck-only crate whose launchers
//! fail loud with [`KernelError::ShaderMissing`].

mod cache;

pub use cache::{KernelCache, launch_cached, record_dispatch};
/// Re-exported so callers can size a [`MmSpec`] without depending on
/// `vulkan-sys` directly. Resolves to the stub definition when the `vulkan`
/// feature is off, so the typecheck-only lane builds unchanged.
pub use vulkan_sys::CoopmatShape;

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

/// `block_nvfp4` values per block (ggml-common.h:211 `QK_NVFP4`).
pub const QK_NVFP4: usize = 64;
/// Values per UE4M3 sub-block scale (ggml-common.h:212 `QK_NVFP4_SUB`). Matches
/// the checkpoint's `group_size: 16` (hf_quant_config.json).
pub const QK_NVFP4_SUB: usize = 16;
/// `sizeof(block_nvfp4)`: `uint8 d[4]` UE4M3 sub-block scales + `uint8 qs[32]`
/// packed E2M1 nibbles (ggml-common.h:213-217).
pub const BLOCK_NVFP4_BYTES: usize = 36;

/// Device bytes one NVFP4 row of `ncols` values occupies once it is in
/// `block_nvfp4` form (see [`repack_nvfp4_planes`]).
///
/// `None` unless `ncols` is a whole number of 64-value blocks: `mul_mat_vec.comp`
/// indexes A as `ib = (row*ncols + col) / QUANT_K`, so a row width that is not a
/// block multiple silently makes row `r > 0` read the previous row's tail.
pub const fn nvfp4_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_NVFP4) {
        return None;
    }
    Some(ncols / QK_NVFP4 * BLOCK_NVFP4_BYTES)
}

/// Rewrite one NVFP4 weight matrix `[nrows, ncols]` from the checkpoint's
/// separate planes into the contiguous ggml `block_nvfp4` stream the vendored
/// `DATA_A_NVFP4` shaders read.
///
/// Inputs are two of the four tensors ModelOpt writes per quantized matrix:
/// `...weight` (`qs_plane`, U8, `ncols/2` bytes per row) and `...weight_scale`
/// (`scale_plane`, FP8 E4M3, `ncols/16` bytes per row). The other two are F32
/// scalars and do NOT belong in the blocks: `weight_scale_2` is per-tensor and
/// rides out-of-band (see [`gemv_id_params_fused`]); `input_scale` describes the
/// W4A4 *activation* quantizer, which this f32-activation lane does not use.
///
/// ## Why the bytes are BUILT and not copied
///
/// Two independent layout gaps, both measured against the on-box checkpoint
/// (`layer-00000-experts-0000-0127.safetensors`, layer 0):
///
/// 1. **The scales are interleaved with the data.** `block_nvfp4` is
///    `d[4] || qs[32]` per 64 values, so each row's scale bytes have to be
///    sliced into the qs stream 4 at a time. Total bytes are unchanged
///    (`1280 + 160 == 40 * 36` for a 2560-wide row), so this costs no VRAM.
///
/// 2. **The nibble order differs.** ModelOpt packs consecutive pairs — value
///    `2i` in the low nibble of byte `i`, value `2i+1` in the high nibble.
///    ggml packs split halves *within each 16-value sub-block* — the low
///    nibbles of the 8 bytes are elements 0..7 and the high nibbles are
///    elements 8..15 (`ggml-quants.c:531`, and the consumer at
///    `vulkan-shaders/dequant_funcs.glsl:493`). A straight `d || qs` copy
///    therefore permutes every group of 16 weights against its activations and
///    still produces plausible-looking finite output.
///
///    This is not taken on faith from the producer's docs; it is measured, and
///    `tests/device_nvfp4_gemv.rs` re-measures it. Under each hypothesis the
///    routed experts' within-group normalized-magnitude profile (mean
///    `|w| / group_amax` per input channel, 4 experts x 640 rows) is correlated
///    against the same profile of the layer's own BF16
///    `shared_expert.gate_proj`, which reads the same residual stream and is
///    NOT quantized: consecutive-pairs scores +0.54 (up_proj +0.50),
///    split-halves +0.039 — noise. Unchanged at 32 experts (+0.51 / +0.035),
///    and the layer-0-vs-MTP-layer control is -0.03, so the signal is
///    layer-specific structure and not an artifact of the statistic.
///
/// ## Why host-side, and not a shader that reads the planes directly
///
/// A plane-reading variant would avoid this transform, but:
/// - the transform is not an extra pass. The weights must be copied host->device
///   once regardless; this rewrites the bytes *in* that copy, so the marginal
///   cost is CPU byte-shuffling overlapped with an I/O-bound upload, not a
///   second traversal of the checkpoint.
/// - device footprint is identical either way (36 B per 64 values both ways).
/// - `mul_mat_vec_base.glsl` computes ONE A-side offset,
///   `expert_id * (batch_stride_a / QUANT_K)`, in block units. A second plane
///   needs a second, differently-scaled offset and a 7th binding, i.e. forking
///   the shared binding/offset contract that every registered GEMV uses —
///   new untested shader code on the MoE hot path, versus a host loop whose
///   output is checked against an independent CPU dequantizer.
///
/// # Errors
/// `ncols` not a positive multiple of [`QK_NVFP4`], or any of the three slices
/// not exactly the length that `nrows`/`ncols` implies.
pub fn repack_nvfp4_planes(
    qs_plane: &[u8],
    scale_plane: &[u8],
    nrows: usize,
    ncols: usize,
    out: &mut [u8],
) -> Result<()> {
    let Some(row_bytes) = nvfp4_row_bytes(ncols) else {
        return Err(KernelError::Runtime(format!(
            "nvfp4 repack: ncols {ncols} is not a positive multiple of {QK_NVFP4}"
        )));
    };
    let qs_row = ncols / 2;
    let scale_row = ncols / QK_NVFP4_SUB;
    for (label, got, want) in [
        ("weight plane", qs_plane.len(), nrows * qs_row),
        ("weight_scale plane", scale_plane.len(), nrows * scale_row),
        ("destination", out.len(), nrows * row_bytes),
    ] {
        if got != want {
            return Err(KernelError::Runtime(format!(
                "nvfp4 repack: {label} is {got} B, expected {want} B for [{nrows}, {ncols}]"
            )));
        }
    }

    const SUBS: usize = QK_NVFP4 / QK_NVFP4_SUB; // 4 scales per block
    const SUB_BYTES: usize = QK_NVFP4_SUB / 2; // 8 qs bytes per sub-block
    const SUB_HALF: usize = QK_NVFP4_SUB / 4; // src byte holding element 8

    for r in 0..nrows {
        let src_qs = &qs_plane[r * qs_row..][..qs_row];
        let src_sc = &scale_plane[r * scale_row..][..scale_row];
        let dst_row = &mut out[r * row_bytes..][..row_bytes];

        for (b, block) in dst_row.chunks_exact_mut(BLOCK_NVFP4_BYTES).enumerate() {
            let (d, qs) = block.split_at_mut(SUBS);
            // Verbatim: every scale byte in this checkpoint is already a legal
            // ggml UE4M3 code. Measured over all 39,321,600 layer-0 scale bytes
            // the range is 0x48..=0x7E — no sign bit (which ggml's 128-entry LUT
            // has no room for) and no 0x7F (which ggml decodes as 0, not NaN).
            d.copy_from_slice(&src_sc[b * SUBS..][..SUBS]);

            let src_block = &src_qs[b * (QK_NVFP4 / 2)..][..QK_NVFP4 / 2];
            for s in 0..SUBS {
                let src_sub = &src_block[s * SUB_BYTES..][..SUB_BYTES];
                let dst_sub = &mut qs[s * SUB_BYTES..][..SUB_BYTES];
                for (j, dst_byte) in dst_sub.iter_mut().enumerate() {
                    // dst byte j holds elements (j, j+8); in the source those
                    // live at bytes (j/2, 4 + j/2), both in nibble j&1.
                    let shift = 4 * (j & 1) as u32;
                    let lo = (src_sub[j / 2] >> shift) & 0xF;
                    let hi = (src_sub[SUB_HALF + j / 2] >> shift) & 0xF;
                    *dst_byte = lo | (hi << 4);
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    MmvqIq2Xxs,
    MmvqQ2K,
    GemvQ4K,
    GemvQ5K,
    GemvQ6K,
    GemvQ8_0,
    /// MXFP4 (E8M0 exponent + 16 packed E2M1 nibbles per 32 values). Carries
    /// the routed experts and most attention projections of Unsloth's
    /// "UD-Q*_XL" dynamic quants — 90% of a Qwen3.5-122B-A10B's elements.
    GemvMxfp4,
    /// Fused MoE expert GEMV (`mul_mat_vec_id`): one dispatch runs a token
    /// through ALL its top-k routed experts. Same `mul_mat_vecq` body + q8_1
    /// activation as the plain GEMV, but the `MUL_MAT_ID` push tail + a 6th
    /// expert-id binding (see [`gemv_id_params`]).
    GemvIdQ4K,
    GemvIdQ5K,
    GemvIdQ6K,
    GemvIdQ8_0,
    /// The variant the 122B-A10B MoE hot path actually dispatches: every
    /// `ffn_gate_exps`/`ffn_up_exps` and most `ffn_down_exps` are MXFP4.
    GemvIdMxfp4,
    /// NVFP4 (four UE4M3 sub-block scales + 32 packed E2M1 nibble bytes per 64
    /// values, 36 B/block). Carries every routed expert of
    /// Qwen3.8-Flash-Next — and nothing else in that checkpoint.
    ///
    /// UNLIKE every other `Gemv*` here this is `mul_mat_vec.comp`, not
    /// `mul_mat_vecq.comp`: **B is a plain f32 activation vector, not
    /// `block_q8_1_x4`** (see `build.rs`'s `mul_mat_vec_nvfp4` comment).
    /// Push constants are [`gemv_params_f32_b`], not [`gemv_params`].
    ///
    /// The weight bytes must be in ggml `block_nvfp4` form, which the
    /// checkpoint's four separate planes are NOT — see [`repack_nvfp4_planes`].
    GemvNvfp4,
    /// The fused MoE expert GEMV for NVFP4 routed experts: `mul_mat_vec.comp`
    /// built with `MUL_MAT_ID`, so one dispatch runs a token through all top-k
    /// experts. Same f32-B caveat as [`Kernel::GemvNvfp4`]; push constants are
    /// [`gemv_id_params`] (whose `stride_b` is already in elements).
    ///
    /// The per-tensor `weight_scale_2` has no slot in `block_nvfp4` and rides
    /// out-of-band through the shader's SCALE0/SCALE1 output fusion — see
    /// [`gemv_id_params_fused`].
    GemvIdNvfp4,
    /// Batched prefill GEMM (`mul_mmq`): `D[n, m] = A[m, k] · Bᵀ[n, k]` with a
    /// quantized `A` and `block_q8_1_x4` `B` — the SAME activation format the
    /// decode GEMVs consume, so one `QuantizeQ8_1` dispatch feeds both. Tile
    /// geometry is NOT baked into the variant: pass an [`MmqSpec`], which the
    /// pipeline cache keys on (see [`mmq_params`] / [`mmq_dispatch`]).
    MmqQ4K,
    MmqQ5K,
    MmqQ6K,
    MmqQ8_0,
    /// The same batched prefill GEMM on the MATRIX CORES (`mul_mm.comp` built
    /// with `COOPMAT`): `D[n, m] = A[m, k] · Bᵀ[n, k]`, quantized `A`
    /// dequantized into shared memory, `B` a plain **f16** `[n, k]` row-major
    /// block, `D` f32. Note the operand change versus [`Kernel::MmqQ4K`] — the
    /// activation side is an [`f16_kv_pack_params`] convert, NOT a
    /// [`q8_1_quantize_params`]; feeding it q8_1_x4 produces silent garbage.
    ///
    /// Only usable when [`vulkan_sys::VulkanContext::coopmat`] is `Some`; tile
    /// geometry comes from [`MmSpec::choose`], which keys on the device's
    /// advertised shape. `mul_mmq` remains the fallback everywhere else.
    MmCmQ4K,
    MmCmQ5K,
    MmCmQ6K,
    MmCmQ8_0,
    QuantizeQ8_1,
    RmsNorm,
    RopeNeox,
    RopeNorm,
    Silu,
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
    /// Qwen3.8-Flash-Next PLE gate, fused: the two grouped RMSNorms, the
    /// per-stream dot, the signed-sqrt gate + sigmoid, the broadcast multiply
    /// and `norm_conv` — eight generic dispatches in one. Emits BOTH the
    /// un-normed gated value (residual branch) and its `norm_conv` output (conv
    /// branch); see [`qwen4_ple_gate_params`].
    Qwen4PleGate,
    /// Qwen3.8-Flash-Next PLE short conv, fused: dilated depthwise conv
    /// (kernel 4, dilation 3) over a 9-step ring, SiLU, the ring roll and the
    /// residual add. See [`qwen4_ple_conv_params`].
    Qwen4PleConv,
    /// Qwen3.8-Flash-Next hyper-connection MIX, fused: the `W_up` GEMV, its
    /// sigmoid, the `* hn` and the cross-stream mean, with the bottleneck's
    /// `silu(u / hc_count)` folded into the read of binding 2. One workgroup per
    /// HIDDEN channel, so the `[hc_count*hidden]` mixing gate is never
    /// materialized and no second pass is needed for the mean. See
    /// [`qwen4_hc_mix_params`].
    Qwen4HcMix,
    /// Qwen3.8-Flash-Next hyper-connection COMBINE, fused: the `W_inject` dots
    /// and `2*sigmoid(dot/hc_count)`, then the in-place
    /// `h[s] += inj[s] * y` scatter across all streams. See
    /// [`qwen4_hc_combine_params`].
    Qwen4HcCombine,
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
/// `mul_mat_vec.comp`'s `BLOCK_SIZE, NUM_ROWS, NUM_COLS`. Numerically equal to
/// [`SPEC_GEMV_K_Q8_1`] but deliberately a separate constant: the two shaders
/// only share the spec-constant *ids*, and tying the NVFP4 lane to a name that
/// says `Q8_1` would hide the fact that a future retune of the q8_1 decode
/// geometry has nothing to do with this one.
///
/// `BLOCK_SIZE = 64` is the 8060S wave width. `mul_mat_vec.comp` gives thread
/// `tid` the columns `{512*i + 8*tid}` (`K_PER_ITER = 8`), so with a 64-wide
/// workgroup the coverage is exactly every 8-aligned column once — which is why
/// `ncols` must be a multiple of 8, and (for the A-side block index to stay
/// row-aligned) of `QK_NVFP4`. Both hold for this model: hidden 2560 = 40x64,
/// `moe_intermediate_size` 640 = 10x64.
const SPEC_GEMV_NVFP4: &[(u32, u32)] = &[(0, 64), (1, 1), (2, 1)];
const SPEC_RMS_NORM_MUL: &[(u32, u32)] = &[(1, 1)];

impl Kernel {
    pub const ALL: &'static [Self] = &[
        Self::MmvqIq2Xxs,
        Self::MmvqQ2K,
        Self::GemvQ4K,
        Self::GemvQ5K,
        Self::GemvQ6K,
        Self::GemvQ8_0,
        Self::GemvMxfp4,
        Self::GemvIdQ4K,
        Self::GemvIdQ5K,
        Self::GemvIdQ6K,
        Self::GemvIdQ8_0,
        Self::GemvIdMxfp4,
        Self::GemvNvfp4,
        Self::GemvIdNvfp4,
        Self::MmqQ4K,
        Self::MmqQ5K,
        Self::MmqQ6K,
        Self::MmqQ8_0,
        Self::MmCmQ4K,
        Self::MmCmQ5K,
        Self::MmCmQ6K,
        Self::MmCmQ8_0,
        Self::QuantizeQ8_1,
        Self::RmsNorm,
        Self::RopeNeox,
        Self::RopeNorm,
        Self::Silu,
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
        Self::Qwen4PleGate,
        Self::Qwen4PleConv,
        Self::Qwen4HcMix,
        Self::Qwen4HcCombine,
    ];

    pub const fn shader_name(self) -> &'static str {
        match self {
            Kernel::MmvqIq2Xxs => "mul_mat_vec_iq2_xxs",
            Kernel::MmvqQ2K => "mul_mat_vec_q2_k",
            Kernel::GemvQ4K => "mul_mat_vecq_q4_k",
            Kernel::GemvQ5K => "mul_mat_vecq_q5_k",
            Kernel::GemvQ6K => "mul_mat_vecq_q6_k",
            Kernel::GemvQ8_0 => "mul_mat_vecq_q8_0",
            Kernel::GemvMxfp4 => "mul_mat_vecq_mxfp4",
            Kernel::GemvIdQ4K => "mul_mat_vec_id_q4_k",
            Kernel::GemvIdQ5K => "mul_mat_vec_id_q5_k",
            Kernel::GemvIdQ6K => "mul_mat_vec_id_q6_k",
            Kernel::GemvIdQ8_0 => "mul_mat_vec_id_q8_0",
            Kernel::GemvIdMxfp4 => "mul_mat_vec_id_mxfp4",
            Kernel::GemvNvfp4 => "mul_mat_vec_nvfp4",
            Kernel::GemvIdNvfp4 => "mul_mat_vec_id_nvfp4",
            Kernel::MmqQ4K => "mul_mmq_q4_k",
            Kernel::MmqQ5K => "mul_mmq_q5_k",
            Kernel::MmqQ6K => "mul_mmq_q6_k",
            Kernel::MmqQ8_0 => "mul_mmq_q8_0",
            Kernel::MmCmQ4K => "mul_mm_cm_q4_k",
            Kernel::MmCmQ5K => "mul_mm_cm_q5_k",
            Kernel::MmCmQ6K => "mul_mm_cm_q6_k",
            Kernel::MmCmQ8_0 => "mul_mm_cm_q8_0",
            Kernel::QuantizeQ8_1 => "q8_1_quantize",
            Kernel::RmsNorm => "rms_norm",
            Kernel::RopeNeox => "rope_neox",
            Kernel::RopeNorm => "rope_norm",
            Kernel::Silu => "silu",
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
            Kernel::Qwen4PleGate => "qwen4_ple_gate",
            Kernel::Qwen4PleConv => "qwen4_ple_conv",
            Kernel::Qwen4HcMix => "qwen4_hc_mix",
            Kernel::Qwen4HcCombine => "qwen4_hc_combine",
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
            | Kernel::GemvIdQ8_0
            | Kernel::GemvMxfp4
            | Kernel::GemvIdMxfp4 => SPEC_GEMV_K_Q8_1,
            Kernel::GemvNvfp4 | Kernel::GemvIdNvfp4 => SPEC_GEMV_NVFP4,
            Kernel::QuantizeQ8_1 => SPEC_WORKGROUP_32,
            Kernel::RmsNorm => SPEC_RMS_NORM_MUL,
            Kernel::SoftMax | Kernel::ArgMax => SPEC_WORKGROUP_32,
            Kernel::FlashAttn => SPEC_FLASH_ATTN_F32_F16_HD256,
            Kernel::RopeNeox
            | Kernel::RopeNorm
            | Kernel::Silu
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
            | Kernel::Qwen36MoeWeightedAccum
            | Kernel::Qwen4PleGate
            | Kernel::Qwen4PleConv
            | Kernel::Qwen4HcMix
            | Kernel::Qwen4HcCombine => &[],
            // `mul_mmq`'s tile geometry is chosen per call from the matmul shape
            // (see [`MmqSpec::choose`]); there is no single default, and running
            // it with the shader's built-in defaults would silently pick a tile
            // whose shared-memory footprint the caller never checked.
            Kernel::MmqQ4K | Kernel::MmqQ5K | Kernel::MmqQ6K | Kernel::MmqQ8_0 => &[],
            // Same reasoning for the coopmat GEMM, plus its tile additionally
            // depends on the device's advertised matrix shape ([`MmSpec`]).
            Kernel::MmCmQ4K | Kernel::MmCmQ5K | Kernel::MmCmQ6K | Kernel::MmCmQ8_0 => &[],
        }
    }

    /// The `mul_mm.comp` COOPMAT sibling of a `mul_mmq` variant, i.e. the same
    /// weight quant run on the matrix cores. `None` for every other kernel.
    pub const fn coopmat_variant(self) -> Option<Kernel> {
        match self {
            Kernel::MmqQ4K => Some(Kernel::MmCmQ4K),
            Kernel::MmqQ5K => Some(Kernel::MmCmQ5K),
            Kernel::MmqQ6K => Some(Kernel::MmCmQ6K),
            Kernel::MmqQ8_0 => Some(Kernel::MmCmQ8_0),
            _ => None,
        }
    }

    /// Bytes one `block_a_cache` entry occupies in shared memory for this
    /// `mul_mmq` variant (`mul_mmq_shmem_types.glsl`, std430). Q4_K packs two
    /// 4-bit quants per byte (`QUANT_R_MMQ = 2`) so it needs half the `qs`
    /// words of Q5_K/Q6_K/Q8_0. Returns `None` for non-`mul_mmq` kernels.
    pub const fn mmq_a_cache_bytes(self) -> Option<u32> {
        match self {
            // { uint32_t qs[4]; f16vec2 dm; }
            Kernel::MmqQ4K => Some(4 * 4 + 4),
            // { int32_t qs[8]; f16vec2 dm | d_scales; }
            Kernel::MmqQ5K | Kernel::MmqQ6K => Some(8 * 4 + 4),
            // { int32_t qs[8]; float16_t dm; } — padded to the 4-byte struct align.
            Kernel::MmqQ8_0 => Some(8 * 4 + 4),
            _ => None,
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
    /// The COOPMAT `mul_mm` pipelines take the width their OWN tile was built
    /// for — spec constant [`MM_CM_WARP_SPEC_ID`], which is why this takes the
    /// specialization list rather than a device property. `WARP` partitions the
    /// workgroup into `BLOCK_SIZE / WARP` subgroups addressed by
    /// `gl_SubgroupID`, and the cooperative-matrix tile itself is
    /// `gl_ScopeSubgroup`, so a pinned size that disagrees with `WARP` either
    /// idles subgroups or double-books them. Reading it back out of the spec
    /// list makes the two impossible to desynchronize, and lets a tuning sweep
    /// try 32- and 64-wide tiles on the same device (the 8060S allows both:
    /// `subgroupSize` control range is 32..64).
    ///
    /// Every other kernel is subgroup-size-agnostic (`None` = driver default).
    /// That includes `mul_mmq`, whose `WARP = 32` warptile constant tempts a
    /// `Some(32)` pin — measured on the 8060S it changes nothing (5.0 vs 5.5
    /// GB/s, inside run-to-run throttle noise), so the driver default stands.
    /// (`mul_mmq`'s body is scalar integer-dot; unlike COOPMAT it never touches
    /// `gl_SubgroupID`, so `WARP` there is only a tiling constant.)
    pub fn required_subgroup_size(self, specialization_u32: &[(u32, u32)]) -> Option<u32> {
        match self {
            Kernel::FlashAttn => Some(32),
            Kernel::MmCmQ4K | Kernel::MmCmQ5K | Kernel::MmCmQ6K | Kernel::MmCmQ8_0 => {
                specialization_u32
                    .iter()
                    .find(|&&(id, _)| id == MM_CM_WARP_SPEC_ID)
                    .map(|&(_, warp)| warp)
            }
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
        Self::with_flags(hsk, hsv, 0)
    }

    /// [`Self::f32_f16_dims`] with `Flags = 2` (`MASK_ENABLE`), which makes the
    /// kernel read binding 3 as an `f16` additive mask of shape `[N][KV]`
    /// (`0` = attend, `-inf` = blocked) — required for a batched prefill tile,
    /// where the `N > 1` query rows each see a different causal prefix.
    ///
    /// `USE_MASK_OPT` (bit 1) stays OFF, so binding 6 remains an unread dummy
    /// and the shader loads every mask block rather than consulting a precomputed
    /// all-zero / all-neg-inf summary. It still skips a block whose loaded mask is
    /// entirely `-inf`, so a causal chunk pays only for the blocks it needs.
    pub const fn f32_f16_masked(hsk: u32, hsv: u32) -> Self {
        Self::with_flags(hsk, hsv, 2)
    }

    const fn with_flags(hsk: u32, hsv: u32, flags: u32) -> Self {
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
                (10, flags),
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

/// `mul_mmq` shared-memory `block_b_cache` size (`{ int32_t qs[8]; f16vec2 ds; }`).
/// Independent of the weight quant — B is always `block_q8_1_x4`.
const MMQ_B_CACHE_BYTES: u32 = 8 * 4 + 4;

/// `mul_mmq.comp`'s `BK_STEP` for the non-`MUL_MAT_ID` build: four 32-value
/// K-slices are staged per shared-memory round trip.
const MMQ_BK_STEP: u32 = 4;

/// Warptile geometry for one `mul_mmq` pipeline — spec constants
/// `0 BLOCK_SIZE, 1 BM, 2 BN, 4 WM, 5 WN, 6 WMITER, 7 TM, 8 TN, 9 TK, 10 WARP`.
/// (`constant_id = 3` (BK) is commented out in the shader, which `#define`s
/// `BK 32` instead, so it is deliberately absent from the map.)
///
/// The values are llama.cpp's `s_warptile_mmq_int{,_k}` families
/// (`ggml-vulkan.cpp:3081`) at `subgroup_size = 32`. The per-chip overrides
/// below them are all gated on `driver_id != eAmdProprietary`, so the AMD
/// Windows lane this crate targets uses the base tiles unchanged. K-quants get
/// `WMITER = 1` and legacy quants `WMITER = 2`, matching the `_k` split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmqSpec {
    specialization_u32: [(u32, u32); 10],
    bm: u32,
    bn: u32,
}

impl MmqSpec {
    const fn new(
        block_size: u32,
        bm: u32,
        bn: u32,
        wm: u32,
        wn: u32,
        wmiter: u32,
        tm: u32,
        tn: u32,
        warp: u32,
    ) -> Self {
        Self {
            specialization_u32: [
                (0, block_size),
                (1, bm),
                (2, bn),
                (4, wm),
                (5, wn),
                (6, wmiter),
                (7, tm),
                (8, tn),
                (9, 1), // TK: coopmat-only, unused by the scalar body
                (10, warp),
            ],
            bm,
            bn,
        }
    }

    pub const fn k_quant_large() -> Self {
        Self::new(128, 128, 128, 64, 64, 1, 4, 4, 32)
    }

    pub const fn k_quant_medium() -> Self {
        Self::new(128, 64, 64, 32, 32, 1, 2, 2, 32)
    }

    pub const fn k_quant_small() -> Self {
        Self::new(32, 32, 32, 32, 32, 1, 2, 1, 32)
    }

    pub const fn legacy_large() -> Self {
        Self::new(128, 128, 128, 64, 64, 2, 4, 4, 32)
    }

    pub const fn legacy_medium() -> Self {
        Self::new(128, 64, 64, 32, 32, 2, 2, 2, 32)
    }

    pub const fn legacy_small() -> Self {
        Self::new(32, 32, 32, 32, 32, 2, 2, 1, 32)
    }

    pub const fn bm(&self) -> u32 {
        self.bm
    }

    pub const fn bn(&self) -> u32 {
        self.bn
    }

    pub const fn specialization_u32(&self) -> &[(u32, u32)] {
        &self.specialization_u32
    }

    /// Shared-memory bytes this tile needs for `kernel`:
    /// `BK_STEP * (BM * sizeof(block_a_cache) + BN * sizeof(block_b_cache))`.
    /// Must be compared against `maxComputeSharedMemorySize` (32 KB on the
    /// 8060S) before the pipeline is built — a tile that overflows fails at
    /// pipeline creation, not at dispatch.
    pub const fn shared_bytes(&self, kernel: Kernel) -> Option<u32> {
        let Some(a_bytes) = kernel.mmq_a_cache_bytes() else {
            return None;
        };
        Some(MMQ_BK_STEP * (self.bm * a_bytes + self.bn * MMQ_B_CACHE_BYTES))
    }

    /// Pick the largest tile that both suits the `[m, n]` output shape and fits
    /// `max_shared_bytes`. Mirrors `ggml_vk_guess_matmul_pipeline`
    /// (`ggml-vulkan.cpp:6801`): small when either dimension is ≤ 32, medium
    /// when either is ≤ 64, large otherwise — then falls back a size at a time
    /// when the shared-memory budget says no. Only the Q4_K large tile fits
    /// 32 KB (its `block_a_cache` is half the width of the other three), so in
    /// practice Q5_K/Q6_K/Q8_0 land on medium.
    pub fn choose(kernel: Kernel, m: u32, n: u32, max_shared_bytes: u32) -> Option<Self> {
        let k_quant = matches!(kernel, Kernel::MmqQ4K | Kernel::MmqQ5K | Kernel::MmqQ6K);
        let (large, medium, small) = if k_quant {
            (
                Self::k_quant_large(),
                Self::k_quant_medium(),
                Self::k_quant_small(),
            )
        } else {
            (
                Self::legacy_large(),
                Self::legacy_medium(),
                Self::legacy_small(),
            )
        };
        let preferred = if m <= 32 || n <= 32 {
            small
        } else if m <= 64 || n <= 64 {
            medium
        } else {
            large
        };
        [preferred, medium, small].into_iter().find(
            |tile| matches!(tile.shared_bytes(kernel), Some(bytes) if bytes <= max_shared_bytes),
        )
    }
}

/// `mul_mm.comp`'s COOPMAT `SHMEM_STRIDE` in `FLOAT_TYPEV2` units:
/// `BK / 2 + 4` (the scalar build uses `+ 1`; coopmat pads harder to keep
/// `coopMatLoad` off shared-memory bank conflicts).
const MM_CM_SHMEM_STRIDE_PAD: u32 = 4;

/// Every COOPMAT tile below stages `BK = 32` reduction elements per round.
/// `mul_mm.comp` exposes BK as `constant_id = 3` with a default of **16** and
/// only a comment ("Assumed to be 32 if working with a quant") to say so — the
/// quant `load_a_to_shmem` bodies hardcode 32-value sub-blocks, so leaving the
/// default in place is silently wrong. [`MmSpec`] always sets it.
const MM_CM_BK: u32 = 32;

/// `mul_mm.comp`'s `layout (constant_id = 10) const uint WARP`. Named because
/// [`Kernel::required_subgroup_size`] reads it back out of a built
/// specialization list to pin the pipeline to the tile's own warp width.
pub const MM_CM_WARP_SPEC_ID: u32 = 10;

/// Per-subgroup warp-tile edge (`WM` = `WN`) shared by every [`MmSpec`] below.
/// At the device's 16x16x16 matrix shape this is `(32/16) * (32/16) = 4` live
/// `coopmat` accumulators — the measured occupancy sweet spot; see the
/// [`MmSpec`] type doc for why it is a constant rather than a tuning knob.
const MM_CM_W: u32 = 32;

/// Ceiling on a derived `BLOCK_SIZE`, standing in for the unexposed
/// `maxComputeWorkGroupInvocations`. Every real Vulkan 1.1 device reports 1024
/// (the spec floor is 128, but no compute-capable GPU ships that low); the
/// widest tile here needs 512. Hardcoded rather than plumbed through
/// `vulkan-sys` because it only ever rejects a hand-written sweep candidate —
/// none of the shipped tiles come close.
const MM_CM_MAX_BLOCK_SIZE: u32 = 1024;

/// Warptile geometry for one COOPMAT `mul_mm` pipeline — spec constants
/// `0 BLOCK_SIZE, 1 BM, 2 BN, 3 BK, 4 WM, 5 WN, 6 WMITER, 7 TM, 8 TN, 9 TK,
/// 10 WARP`. `TM/TN/TK` are NOT free here: they are the device's advertised
/// cooperative-matrix `M/N/K`, because the shader declares
/// `coopmat<..., TM, TK, gl_MatrixUseA>` directly.
///
/// ## Why these are NOT llama.cpp's `{s,m,l}_warptile_mmq`
///
/// The obvious move is to copy `ggml-vulkan.cpp:3076`'s tiles verbatim. That
/// was tried and **measured**: `l_warptile_mmq` (`BM=BN=128`, `WM=subgroup*2`,
/// `WN=64`) is the single *worst* tile of eleven candidates on the 8060S —
/// **0.57x geomean vs `mul_mmq`**, and as bad as 0.20x at `n = 32`. Since the
/// old `choose` picked it for every `n > 64`, i.e. every prefill chunk, the
/// whole coopmat route ran at 0.75x the scalar integer-dot kernel it was meant
/// to replace.
///
/// The mechanism is per-subgroup accumulator count, not warp width.
/// `mul_mm.comp:179` declares `sums[(WM/TM) * (WN/TN)]` live across the entire
/// K loop, each `coopmat` costing `TM*TN/WARP` VGPRs per lane. `l_warptile_mmq`
/// at `warp = 64` gives `8 * 4 = 32` accumulators ≈ **128 VGPRs/lane** before
/// operands — deep into occupancy collapse on RDNA 3.5. Capping `WM = WN = 32`
/// (4 accumulators, 16 VGPRs) and buying back the tile area with *more
/// subgroups per workgroup* instead wins by 2-3x. Every tile below therefore
/// holds `WM = WN = 32` and varies only `BM`/`BN`. Sweep geomeans vs `mul_mmq`,
/// `crates/vulkan-kernels/tests/device_mm_coopmat_bench.rs`:
///
/// ```text
///            tile     n=32     n=64    n=128    n=192    n=256    all n
///  l 128x128 w128x64  0.32x    0.25x    0.89x    0.97x    0.89x    0.57x   <- llama.cpp
///     128x32  w32x32  1.80x    0.85x    2.14x    2.42x    1.90x    1.72x   <- narrow
///      64x64  w32x32  1.31x    1.17x    2.93x    2.82x    2.40x    1.98x   <- medium
///     128x64  w32x32  1.53x    1.16x    2.55x    3.12x    2.60x    2.06x   <- wide
///      32x32  w32x32  1.34x    0.84x    1.71x    1.78x    1.47x    1.38x   <- tiny
/// ```
///
/// ## Dead ends, recorded so they are not re-walked
///
/// Two earlier diagnoses of the same 0.75x regression were wrong and both cost
/// a full build+measure cycle:
/// - *"the f16 B-operand pack kernel dominates"* — ruled out by the per-op GPU
///   profile: 19.63 ms of 12448 ms.
/// - *"`WARP` is hardcoded to 32 on a wave64 device"* — plausible (llama.cpp
///   does derive every tile from `max(subgroup_size, 8)`), fully implemented,
///   and a **measured non-effect**: 0.75x before, 0.75x after.
///
/// `WARP` is still parameterized — it must equal the pipeline's
/// `requiredSubgroupSize` ([`Kernel::required_subgroup_size`], which reads it
/// back out of the spec list) — it just was not the bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmSpec {
    specialization_u32: [(u32, u32); 11],
    bm: u32,
    bn: u32,
}

impl MmSpec {
    /// `BLOCK_SIZE` is derived, not chosen: the shader hands warp tile
    /// `gl_SubgroupID` to each subgroup and never loops, so the workgroup must
    /// hold exactly `(BM/WM) * (BN/WN)` subgroups. Passing it separately only
    /// creates a way to get it wrong.
    ///
    /// Callers must pre-check `wm`/`wn` divide `bm`/`bn` (see
    /// [`MmSpec::is_valid`]); a non-dividing pair yields a truncated — and thus
    /// invalid — `BLOCK_SIZE` rather than a wrong answer.
    const fn new(bm: u32, bn: u32, wm: u32, wn: u32, warp: u32, shape: CoopmatShape) -> Self {
        Self {
            specialization_u32: [
                (0, (bm / wm) * (bn / wn) * warp),
                (1, bm),
                (2, bn),
                (3, MM_CM_BK),
                (4, wm),
                (5, wn),
                // WMITER is dead in the COOPMAT body (it only feeds the scalar
                // `WNITER`/`WSUBN` derivation, which the driver folds away);
                // llama.cpp passes 2 here regardless, so match it.
                (6, 2),
                (7, shape.m),
                (8, shape.n),
                (9, shape.k),
                (10, warp),
            ],
            bm,
            bn,
        }
    }

    /// `n > 64`: 8 subgroups over a 128x64 block. The prefill workhorse —
    /// 2.55x/3.12x/2.60x at n = 128/192/256.
    pub const fn wide(warp: u32, shape: CoopmatShape) -> Self {
        Self::new(128, 64, MM_CM_W, MM_CM_W, warp, shape)
    }

    /// `n <= 64`: 4 subgroups over a square 64x64 block. The only tile that
    /// beats `mul_mmq` at the awkward n = 64 width (1.17x).
    pub const fn medium(warp: u32, shape: CoopmatShape) -> Self {
        Self::new(64, 64, MM_CM_W, MM_CM_W, warp, shape)
    }

    /// `n <= 32`: 4 subgroups stacked down M, since there is only one N tile to
    /// win. 1.80x at n = 32, where every wider tile wastes half its columns.
    pub const fn narrow(warp: u32, shape: CoopmatShape) -> Self {
        Self::new(128, 32, MM_CM_W, MM_CM_W, warp, shape)
    }

    /// Single-subgroup 32x32 block, 6 KiB of shared memory. Not the fastest at
    /// any width (1.38x geomean) — it exists as the last fallback for a device
    /// too shared-memory-poor for the tiles above.
    pub const fn tiny(warp: u32, shape: CoopmatShape) -> Self {
        Self::new(32, 32, MM_CM_W, MM_CM_W, warp, shape)
    }

    pub const fn bm(&self) -> u32 {
        self.bm
    }

    pub const fn bn(&self) -> u32 {
        self.bn
    }

    pub const fn specialization_u32(&self) -> &[(u32, u32)] {
        &self.specialization_u32
    }

    /// Shared-memory bytes: the two `FLOAT_TYPEV2` (f16vec2, 4 B) staging
    /// buffers `buf_a[BM * SHMEM_STRIDE]` / `buf_b[BN * SHMEM_STRIDE]`, plus the
    /// `coopmat_stage[TM * TN * NUM_WARPS]` f32 spill used by the unaligned
    /// store paths. Unlike `mul_mmq` this does NOT depend on the weight quant —
    /// A is dequantized to f16 on the way in.
    pub const fn shared_bytes(&self) -> u32 {
        let stride = MM_CM_BK / 2 + MM_CM_SHMEM_STRIDE_PAD;
        let staging = (self.bm + self.bn) * stride * 4;
        let num_warps = self.specialization_u32[0].1 / self.specialization_u32[10].1;
        let stage = self.specialization_u32[7].1 * self.specialization_u32[8].1 * num_warps * 4;
        staging + stage
    }

    /// An arbitrary warptile, for benches and tuning sweeps. `None` unless the
    /// geometry is self-consistent under [`MmSpec::is_valid`] — so a sweep can
    /// enumerate candidates freely and let this filter the unrunnable ones,
    /// rather than each caller re-deriving the divisibility rules.
    #[allow(clippy::too_many_arguments)]
    pub fn tile(
        bm: u32,
        bn: u32,
        wm: u32,
        wn: u32,
        warp: u32,
        shape: CoopmatShape,
        max_shared_bytes: u32,
    ) -> Option<Self> {
        // `new` divides by `wm`/`wn`/`warp` to derive BLOCK_SIZE, so the zero
        // check cannot wait for `is_valid`.
        if wm == 0 || wn == 0 || warp == 0 {
            return None;
        }
        let spec = Self::new(bm, bn, wm, wn, warp, shape);
        spec.is_valid(shape, warp, max_shared_bytes).then_some(spec)
    }

    /// Can this tile actually run on `shape` at `warp` within
    /// `max_shared_bytes`?
    ///
    /// `WM`/`WN` must be whole multiples of the matrix `M`/`N` (the shader
    /// iterates `cms_per_row = WM / TM` sub-matrices with no remainder
    /// handling), and `BK` a multiple of `K`. The 8060S advertises 16x16x16,
    /// which divides every tile below; a device advertising, say, 8x8x32 fails
    /// here and falls back to `mul_mmq` rather than computing a wrong answer
    /// from a truncated tile count.
    ///
    /// `BM`/`BN` must likewise be whole multiples of `WM`/`WN`, or the
    /// `BLOCK_SIZE` [`MmSpec::new`] derived from them was truncated and the
    /// workgroup would be short the subgroups needed to cover the block.
    pub fn is_valid(&self, shape: CoopmatShape, warp: u32, max_shared_bytes: u32) -> bool {
        if shape.m == 0 || shape.n == 0 || shape.k == 0 || !MM_CM_BK.is_multiple_of(shape.k) {
            return false;
        }
        if warp == 0 || !warp.is_power_of_two() {
            return false;
        }
        let block_size = self.specialization_u32[0].1;
        let wm = self.specialization_u32[4].1;
        let wn = self.specialization_u32[5].1;
        wm.is_multiple_of(shape.m)
            && wn.is_multiple_of(shape.n)
            && self.bm.is_multiple_of(wm)
            && self.bn.is_multiple_of(wn)
            && block_size <= MM_CM_MAX_BLOCK_SIZE
            && self.shared_bytes() <= max_shared_bytes
    }

    /// Pick the tile that suits the `[m, n]` output shape and fits
    /// `max_shared_bytes`, or `None` when this device's cooperative-matrix
    /// shape cannot tile it — in which case the caller must stay on `mul_mmq`.
    ///
    /// Selection keys on `n` — the batch/token width — alone. `m` is the
    /// weight's output-feature count, in the thousands for every matmul on the
    /// prefill path, so it never constrains the tile; the old `m <= 32 || n <=
    /// 32` form let a wide-`n` chunk fall to the narrow tile whenever `m`
    /// happened to be small, which no measurement supports. `m` is therefore
    /// not a parameter — over-covering it costs at most one partial tile, which
    /// [`mm_dispatch`] already rounds up for.
    ///
    /// `warp` MUST be the value the pipeline will actually run at, i.e.
    /// `VulkanContext::subgroup_size().0`; see the type doc.
    pub fn choose(shape: CoopmatShape, warp: u32, n: u32, max_shared_bytes: u32) -> Option<Self> {
        let preferred = if n <= 32 {
            Self::narrow(warp, shape)
        } else if n <= 64 {
            Self::medium(warp, shape)
        } else {
            Self::wide(warp, shape)
        };
        // Fallbacks in descending shared-memory order, so a device tighter than
        // the 32 KiB this box reports still lands on *some* runnable tile.
        [
            preferred,
            Self::medium(warp, shape),
            Self::tiny(warp, shape),
        ]
        .into_iter()
        .find(|tile| tile.is_valid(shape, warp, max_shared_bytes))
    }
}

/// Dispatch grid for a COOPMAT [`MmSpec`]. Identical in form to
/// [`mmq_dispatch`] — same push-constant contract, same `ir = x % blocks_m`
/// recovery — but keyed on the coopmat tile's `BM`/`BN`.
pub fn mm_dispatch(m: u32, n: u32, spec: &MmSpec) -> Dispatch {
    Dispatch {
        x: m.div_ceil(spec.bm()).max(1),
        y: n.div_ceil(spec.bn()).max(1),
        z: 1,
    }
}

/// Push-constant block for `mul_mmq.comp` (non-`MUL_MAT_ID`): 16 `uint`s, in
/// declared order `M, N, K, stride_a, stride_b, stride_d, batch_stride_a,
/// batch_stride_b, batch_stride_d, base_work_group_z, num_batches, k_split,
/// ne02, ne12, broadcast2, broadcast3`.
///
/// Mapped from `ggml_vk_matmul` (`ggml-vulkan.cpp:6855`) for ONE unbatched,
/// unsplit matmul:
/// - `A` is the `[m, k]` quantized weight, row-major, `stride_a = k` elements.
///   `k` must be a multiple of the weight's `QUANT_K` (256 for the K-quants,
///   32 for Q8_0) — the shader indexes A in 32-value sub-blocks.
/// - `B` is `[n, k]` `block_q8_1_x4` activations, `stride_b = k` elements. Row
///   starts must land on an x4 group boundary, so `k` must be a multiple of 128.
/// - `D` is `[n, m]` f32 with `stride_d = m`: the shader writes
///   `data_d[col * stride_d + row]`, i.e. one contiguous `m`-wide row per
///   B row. For prefill that is exactly "one output vector per token".
/// - `k_split = k` and `num_batches = 1` collapse the split-K grid to `ik = 0`,
///   so `gl_WorkGroupID.x` is purely the M tile.
///
/// Binding order: `[0 = A quantized weight, 1 = B q8_1_x4, 2 = D f32]`.
pub fn mmq_params(m: u32, n: u32, k: u32) -> KernelParams {
    KernelParams::from_words(vec![
        m,     // M: output rows (weight rows)
        n,     // N: output cols (tokens in the chunk)
        k,     // K: reduction width in elements
        k,     // stride_a: weight row stride (elements)
        k,     // stride_b: activation row stride (elements)
        m,     // stride_d: dst row stride (elements) = M
        m * k, // batch_stride_a: unused at batch 0, set to the natural stride
        n * k, // batch_stride_b: same
        m * n, // batch_stride_d: same
        0,     // base_work_group_z: single batch
        1,     // num_batches
        k,     // k_split: no split-K
        1,     // ne02
        1,     // ne12
        1,     // broadcast2
        1,     // broadcast3
    ])
}

/// Dispatch grid for [`mmq_params`]: `x` = M tiles, `y` = N tiles, `z` = 1
/// (single batch). `mul_mmq.comp` recovers `ir = x % blocks_m` and, with
/// `k_split = K`, `ik = x / blocks_m = 0`.
pub fn mmq_dispatch(m: u32, n: u32, spec: &MmqSpec) -> Dispatch {
    Dispatch {
        x: m.div_ceil(spec.bm()).max(1),
        y: n.div_ceil(spec.bn()).max(1),
        z: 1,
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

    pub fn len_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u32>()
    }

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

/// [`gemv_params`] for the `mul_mat_vec.comp` GEMVs, whose B operand is a plain
/// f32 vector rather than `block_q8_1_x4` ([`Kernel::GemvNvfp4`]).
///
/// The push block is byte-identical in shape; the one field that changes meaning
/// is `stride_b`, which is a row stride in ELEMENTS here (`mul_mat_vecq.comp`
/// divides `b_offset` by `QUANT_K_Q8_1` in `compute_outputs`, `mul_mat_vec.comp`
/// does not). At `batch_idx == 0` the non-`MUL_MAT_ID` branch never reads
/// `stride_b`, so passing [`gemv_params`] here happens to work today — which is
/// exactly why the wrong value should not be left sitting in the block.
pub fn gemv_params_f32_b(ncols: u32, nrows: u32) -> KernelParams {
    KernelParams::from_words(vec![
        ncols, // ncols: per-row width in elements
        ncols, // stride_a: weight row stride (elements)
        ncols, // stride_b: activation row stride in ELEMENTS (f32 B)
        nrows, // stride_d: ROW COUNT guard the shader checks first_row against
        ncols, // batch_stride_a (elements); only used via /QUANT_K when batched
        0,     // batch_stride_b: single batch
        0,     // batch_stride_d: single batch
        0,     // fusion_flags: no bias/scale fusion (bindings 3/4 unread)
        0,     // base_work_group_y: batch offset
        1,     // ne02
        1,     // ne12
        1,     // broadcast2
        1,     // broadcast3
    ])
}

/// `MAT_VEC_FUSION_FLAGS_*` from `mul_mat_vec_iface.glsl`, for the
/// `fusion_flags` word of [`gemv_id_params_fused`].
///
/// Under `MUL_MAT_ID` the two SCALE flags multiply each output row by
/// `data_fuse{0,1}[gl_GlobalInvocationID.y]` — indexed by the expert SLOT, not
/// the expert id — so binding 3 / binding 4 are per-selected-expert f32 vectors
/// of length `n_experts`. That is the slot the NVFP4 per-tensor
/// `weight_scale_2` folds into: `y_true = weight_scale_2 * y_shader`, because
/// `block_nvfp4` has nowhere to keep it and it cannot be folded into the block
/// scale (measured on layer-0 expert-0 `gate_proj`: re-encoding
/// `weight_scale * weight_scale_2` into E4M3 puts 99.98% of groups below E4M3's
/// smallest normal, 2^-6). With two flags the routing weight can ride in the
/// other one and the MoE weighted-accumulate loses a multiply.
///
/// The BIAS flags are the non-`MUL_MAT_ID` half of the same enum and are listed
/// for completeness; note BIAS0 under `MUL_MAT_ID` indexes by expert *id*.
pub const MAT_VEC_FUSION_BIAS0: u32 = 0x1;
/// See [`MAT_VEC_FUSION_BIAS0`].
pub const MAT_VEC_FUSION_BIAS1: u32 = 0x2;
/// See [`MAT_VEC_FUSION_BIAS0`].
pub const MAT_VEC_FUSION_SCALE0: u32 = 0x4;
/// See [`MAT_VEC_FUSION_BIAS0`].
pub const MAT_VEC_FUSION_SCALE1: u32 = 0x8;

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
    gemv_id_params_fused(ncols, nrows, n_experts, 0)
}

/// [`gemv_id_params`] with the shader's output fusion turned on. `fusion_flags`
/// is an OR of [`MAT_VEC_FUSION_SCALE0`] / [`MAT_VEC_FUSION_SCALE1`] /
/// [`MAT_VEC_FUSION_BIAS0`]; each enabled SCALE flag makes binding 3 (resp. 4)
/// a live `f32[n_experts]` read at the expert SLOT, so a dummy 4-byte buffer is
/// no longer sufficient there.
///
/// This is how the NVFP4 per-tensor `weight_scale_2` reaches the product
/// without a slot in `block_nvfp4` and without an extra dispatch — see
/// [`MAT_VEC_FUSION_BIAS0`] for the arithmetic.
pub fn gemv_id_params_fused(
    ncols: u32,
    nrows: u32,
    n_experts: u32,
    fusion_flags: u32,
) -> KernelParams {
    KernelParams::from_words(vec![
        ncols,             // ncols: per-row width in elements
        ncols,             // stride_a: weight row stride (elements)
        ncols,             // stride_b: activation row stride (elements)
        nrows,             // stride_d: per-expert row count guard + dst stride
        ncols * nrows,     // batch_stride_a: full expert matrix (elements) -> /QUANT_K
        ncols,             // batch_stride_b: one token's activation
        nrows * n_experts, // batch_stride_d: full dst
        fusion_flags,      // fusion_flags: bindings 3/4 are read only when set
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

/// Batched [`rope_neox_params`]: rotates a whole prefill chunk laid out as
/// `[token][head][head_dim]` in ONE dispatch, with a per-token position.
///
/// `rope_head.glsl` decomposes `row` as `i3 = row/(ne01*ne02)`,
/// `i2 = (row - i3*ne01*ne02)/ne01`, `i1 = row - i3*ne01*ne02 - i2*ne01`, reads
/// `theta_base = rope_data_pos[i2] * theta_scale^(i0/2)`, and addresses
/// `i3*nb03 + i2*nb02 + i1*nb01 + i0/2`. Setting `ne01 = heads`, `ne02 = tokens`,
/// `nb01 = nb11 = head_dim`, `nb02 = nb12 = heads*head_dim` therefore makes `i1`
/// the head and `i2` the token — i.e. row-major `[token][head][head_dim]` with
/// `pos[t]` applied to token `t`. Binding 1 must hold a `tokens`-element `int32`
/// position buffer (`start_pos .. start_pos+tokens`), not the single-element
/// buffer the decode path binds.
///
/// The decode contract is the `tokens = 1` reduction of this (`ne02 = 1` makes
/// every row read `pos[0]`), so the two share the shader and the pipeline.
pub fn rope_neox_params_batched(
    head_dim: u32,
    rotary_dim: u32,
    heads: u32,
    tokens: u32,
    rope_theta: f32,
) -> KernelParams {
    let n_dims = rotary_dim;
    let theta_scale = rope_theta.powf(-2.0 / n_dims as f32);
    let plane = heads * head_dim;
    KernelParams::from_words(vec![
        2,                     // rope_mode = GGML_ROPE_TYPE_NEOX
        tokens * heads,        // nrows (row guard)
        n_dims,                // n_dims = rotary_dim
        1.0f32.to_bits(),      // freq_scale
        rope_theta.to_bits(),  // freq_base
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
        heads,                 // ne01 = heads   => i1 = head
        tokens,                // ne02 = tokens  => i2 = token => pos[token]
        head_dim,              // nb01 = head stride (elements)
        plane,                 // nb02 = token stride (elements)
        0,                     // nb03
        head_dim,              // nb11 = head stride (output)
        plane,                 // nb12 = token stride (output)
        0,                     // nb13
        0,                     // a_offset
        0,                     // d_offset
    ])
}

/// Grid for [`rope_neox_params_batched`]: x covers `tokens*heads` rows, y covers
/// the `rotary_dim/2` rotation pairs at `local_size_y = 256`.
pub fn rope_neox_dispatch_batched(rotary_dim: u32, heads: u32, tokens: u32) -> Dispatch {
    rope_neox_dispatch(rotary_dim, tokens * heads)
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
    rms_norm_params_rows(ncols, 1, ncols, eps)
}

/// Multi-row [`rms_norm_params`]: normalizes `nrows` independent rows of
/// `ncols` elements in ONE dispatch, reading row `r` at `r*src_row_stride` and
/// writing it PACKED at `r*ncols`, with the SAME weight vector applied to every
/// row.
///
/// `rms_norm.comp` takes `nrows` from `gl_NumWorkGroups.x` and `row` from
/// `gl_WorkGroupID.x`, so the grid ([`rms_norm_dispatch_rows`]) is what selects
/// the row count; `nb01 = src_row_stride` is what lets the source rows be
/// strided. `d_offset = ((samp*nchannels + channel)*nrows + row)*ncols` with a
/// 1-deep y/z grid is exactly `row*ncols`, i.e. the destination is always packed.
/// `ne11 = 1` makes `src1_idx`'s `fastmod(row, ne11)` zero, broadcasting the
/// weight across rows; `ne10 = ncols` selects the plain `data_b[col]` branch.
///
/// This collapses the per-head q/k-norm and per-value-head `ssm_norm` loops of a
/// prefill chunk into one dispatch each: the batched projections lay a chunk out
/// as `[token][head][head_dim]`, which is `T*heads` rows of `head_dim`. When the
/// source is the interleaved `[query|gate]` q-projection, `src_row_stride =
/// 2*head_dim` also extracts the query half for free.
///
/// In-place (`src == dst`) stays safe only when `src_row_stride == ncols`: each
/// thread reads and writes only its own column, after the reduction barrier.
pub fn rms_norm_params_rows(ncols: u32, nrows: u32, src_row_stride: u32, eps: f32) -> KernelParams {
    let n = ncols;
    let src_plane = nrows * src_row_stride;
    let dst_plane = nrows * n;
    KernelParams::from_words(vec![
        dst_plane, // ne (total elements)
        n,
        nrows,
        1,
        1, // ne00..ne03
        1,
        src_row_stride,
        src_plane,
        src_plane, // nb00..nb03 (nb00=1 element stride)
        n,
        1,
        1,
        1, // ne10..ne13 (weight: ne11=1 => broadcast; ncols<=ne10 => plain col)
        1,
        n,
        n,
        n, // nb10..nb13
        n,
        nrows,
        1,
        1, // ne20..ne23
        1,
        n,
        dst_plane,
        dst_plane,     // nb20..nb23
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

/// Multi-row grid for [`rms_norm_params_rows`]: `nrows = gl_NumWorkGroups.x`, so
/// one workgroup per row (and y = z = 1 keeps `d_offset` packed at `row*ncols`).
pub fn rms_norm_dispatch_rows(nrows: u32) -> Dispatch {
    Dispatch::x(nrows.max(1))
}

/// GROUPED RMSNorm: `ngroups` independent norms of `group_size` elements over
/// one contiguous `[ngroups * group_size]` vector, each group scaled by its OWN
/// slice of an equally-wide weight vector. Grid is
/// [`rms_norm_dispatch_rows`]`(ngroups)`; source and destination are both packed.
///
/// This is the shape `Qwen4ExpTextRMSNorm(group_size=hidden_size)` needs —
/// `hc_norm` over a `hc_count * hidden` hyper-connection residual, and the PLE's
/// key/query norms. It is NOT what [`rms_norm_params_rows`] does: that one sets
/// `ne11 = 1`, whose `fastmod(row, ne11)` is always 0, i.e. it BROADCASTS one
/// weight row over every row. The only difference here is `ne11 = ngroups` with
/// `nb11 = group_size`, so `src1_idx(0, row, 0, 0) = row * group_size` picks
/// group `row`'s own gains. Getting this wrong is silent: with a freshly seeded
/// hyper state all `hc_count` streams are byte-identical copies, so a broadcast
/// weight and a grouped weight agree exactly until the streams diverge.
///
/// ## Caller contract: the weight must arrive PRE-BIASED
///
/// `Qwen4ExpTextRMSNorm` applies `x * inv_rms * (1.0 + weight)`, but
/// `rms_norm.comp` applies the plain `x * inv_rms * weight` and it is vendored
/// (`vendor/llama.cpp/vulkan-shaders/`) — this crate's build script never edits
/// that directory. So the `+ 1` is folded ONCE at load: store `1.0 + w` in the
/// device buffer, exactly as the GGUF converters pre-fold norm scales. The
/// parameter is zero-initialised in this model, so the fold's own rounding is
/// bounded by 2^-24 on a gain of ~1.
pub fn rms_norm_params_grouped(group_size: u32, ngroups: u32, eps: f32) -> KernelParams {
    let n = group_size;
    let total = ngroups * n;
    KernelParams::from_words(vec![
        total, // ne (total elements)
        n,
        ngroups,
        1,
        1, // ne00..ne03
        1,
        n,
        total,
        total, // nb00..nb03 (nb00 = 1 element stride)
        n,
        ngroups,
        1,
        1, // ne10..ne13 — ne11 = ngroups is THE grouped-vs-broadcast bit
        1,
        n,
        total,
        total, // nb10..nb13 (nb11 = n: group `row`'s gains start at row*n)
        n,
        ngroups,
        1,
        1, // ne20..ne23
        1,
        n,
        total,
        total,         // nb20..nb23
        0,             // misalign_offsets
        eps.to_bits(), // param1 (f32 eps)
        0,             // param2 (f32, unused)
        0,             // param3 (i32, unused by rms_norm)
    ])
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

pub fn scaled_add_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(256).max(1))
}

/// `sigmoid_mul.comp` (ARLE-local): `out[i] = sigmoid(a[i]) * b[i]` over `n`
/// elements. Bindings `0=A` (gate, read), `1=B` (value, read), `2=D` (out,
/// write) — same 3-binding layout as `add`/`scaled_add`, so it shares the
/// decode `ring3`. The single push field is `[n (u32)]`. Applies the
/// full-attention per-head sigmoid gate device-resident.
pub fn sigmoid_mul_params(n: u32) -> KernelParams {
    sigmoid_mul_params_strided(n, n.max(1), n.max(1), 0)
}

/// Strided-gate [`sigmoid_mul_params`]: value element `i` is gated by
/// `gate[(i / inner) * gate_stride + gate_off + (i % inner)]`.
///
/// The batched-prefill full-attention block gates a packed
/// `[tokens][heads][head_dim]` flash output against the odd half of the
/// interleaved `[tokens][heads][2*head_dim]` q-projection, i.e.
/// `inner = head_dim, gate_stride = 2*head_dim, gate_off = head_dim` — one
/// dispatch for the whole chunk instead of one per (token, head).
pub fn sigmoid_mul_params_strided(
    n: u32,
    inner: u32,
    gate_stride: u32,
    gate_off: u32,
) -> KernelParams {
    KernelParams::from_words(vec![n, inner.max(1), gate_stride, gate_off])
}

pub fn sigmoid_mul_dispatch(n: u32) -> Dispatch {
    Dispatch::x(n.div_ceil(256).max(1))
}

/// `f16_kv_pack.comp` (ARLE-local): pack `n` f32 values into f16
/// (`dst[i] = float16_t(src[i])`). Bindings `0=A` (f32 src, read), `1=D` (f16
/// dst, write) — a 2-binding layout, so it shares the decode `ring2` with the
/// q8_1 quantize. Writes one full-attention head row (`head_dim` f16) into the
/// device KV cache plane bound at the `(layer, kv_head, pos)` byte offset,
/// removing the host K/V readback+convert.
pub fn f16_kv_pack_params(n: u32) -> KernelParams {
    f16_kv_pack_params_rows(n, 1, n, n)
}

/// Strided multi-row [`f16_kv_pack_params`]: packs `rows` rows of `n` values,
/// reading row `r` at `r*src_stride` (f32 elements) and writing it at
/// `r*dst_stride` (f16 elements).
///
/// A prefill chunk's post-rope K/V is `[token][kv_head][head_dim]`, so one kv
/// head's `T` rows are strided by `n_kv_heads*head_dim` in the arena while they
/// land CONTIGUOUSLY (`dst_stride = head_dim`) in the cache plane at
/// `[pos .. pos+T]`. One dispatch per (layer, kv_head) instead of one per
/// (layer, kv_head, token).
pub fn f16_kv_pack_params_rows(
    n: u32,
    rows: u32,
    src_stride: u32,
    dst_stride: u32,
) -> KernelParams {
    KernelParams::from_words(vec![n, rows, src_stride, dst_stride])
}

pub fn f16_kv_pack_dispatch(n: u32) -> Dispatch {
    f16_kv_pack_dispatch_rows(n, 1)
}

/// Multi-row grid for [`f16_kv_pack_params_rows`]: the guard is
/// `idx >= n*rows`, so `ceil(n*rows / 256)` workgroups cover the block.
pub fn f16_kv_pack_dispatch_rows(n: u32, rows: u32) -> Dispatch {
    Dispatch::x((n * rows).div_ceil(256).max(1))
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

/// Dispatch grid for `qwen36_router_topk.comp`: ONE workgroup of 256 threads,
/// each lane walking a STRIDE of experts (so 512-expert tables work); parallel
/// max/Σexp reductions + `top_k` parallel argmax-and-mask rounds. (The earlier single-thread `local_size_x=1` version
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

pub fn qwen36_moe_weighted_accum_dispatch(hidden: u32) -> Dispatch {
    Dispatch::x(hidden.div_ceil(256).max(1))
}

// Qwen3.8-Flash-Next (`qwen4_exp`) PLE push-constant contracts. The host oracle
// both kernels are diffed against is `infer_vulkan::qwen4_ple` (`PleLayer`),
// itself transcribed from `Qwen4ExpTextPLELayer`. Neither half could be a
// define of a vendored shader: `Qwen4ExpTextRMSNorm` scales by `1.0 + weight`
// (the parameter is zero-initialised) and normalises in `hidden`-wide GROUPS of
// a `hc_count * hidden`-wide row, and the conv is dilated. They are fused to
// this model's sequence rather than written as primitives because decode on
// this box is dispatch-bound: the naive composition is ~12 dispatches per PLE
// layer, this pair is 2.
//
// The two tensors the pair passes between them are NOT interchangeable. The
// gate writes `gated` (un-normed, for the residual) and `gated_normed`
// (`norm_conv`'s output, for the conv); the conv reads both, taps only the
// normed one and adds only the un-normed one.

/// Push-constant block for `qwen4_ple_gate.comp` = `{hidden, hc_count, seq_len,
/// eps(f32 bits)}` (4 words). Runs `key_normed = grouped_rmsnorm(key_proj_out)`,
/// `query_normed = grouped_rmsnorm(hidden_states)`, the per-stream
/// `dot / sqrt(hidden)`, `sign(g) * sqrt(max(|g|, 1e-6))`, `sigmoid`, the
/// broadcast multiply against `value_proj_out`, and `norm_conv` — one dispatch.
///
/// Bindings (in order): `0 = KeyProj [seq*hc_count*hidden] f32`,
/// `1 = HiddenIn [seq*hc_count*hidden] f32` (the hyper residual),
/// `2 = ValueProj [seq*hidden] f32` (no stream axis — one value row gates into
/// every stream), `3 = NormKeyW [hc_count*hidden] f32`,
/// `4 = NormQueryW [hc_count*hidden] f32`, `5 = NormConvW [hc_count*hidden]
/// f32`, `6 = Gated [seq*hc_count*hidden] f32` (write, UN-normed — the residual
/// branch's tensor), `7 = GatedNormed [seq*hc_count*hidden] f32` (write,
/// `norm_conv`'s output — the conv branch's tensor).
///
/// `hidden` is the RMSNorm group width, so a group is one stream's slice of the
/// wide row and the norm weights are indexed `stream * hidden + i`.
pub fn qwen4_ple_gate_params(hidden: u32, hc_count: u32, seq_len: u32, eps: f32) -> KernelParams {
    KernelParams::from_words(vec![hidden, hc_count, seq_len, eps.to_bits()])
}

/// Dispatch grid for `qwen4_ple_gate.comp`: ONE workgroup of 256 threads per
/// (stream, token) — `x = hc_count`, `y = seq_len`. A stream owns exactly one
/// `hidden`-wide RMSNorm group of every tensor involved, so all four reductions
/// (two sums of squares, the dot, and the gated sum of squares `norm_conv`
/// needs) are workgroup-local; lanes stride the group, so `hidden` need not be
/// a multiple of 256.
pub fn qwen4_ple_gate_dispatch(hc_count: u32, seq_len: u32) -> Dispatch {
    Dispatch {
        x: hc_count.max(1),
        y: seq_len.max(1),
        z: 1,
    }
}

/// Push-constant block for `qwen4_ple_conv.comp` = `{channels, seq_len,
/// kernel_size, dilation, ring_pos}` (5 `uint`s). Depthwise dilated conv +
/// SiLU + ring roll + residual add, one dispatch. With `kernel_size = 4` and
/// `dilation = 3` the taps are `t-9, t-6, t-3, t`, weighted by
/// `conv1d[c][0..4]` in that order.
///
/// Bindings (in order): `0 = XNormed [seq*channels] f32` (`gated_normed` — the
/// ONLY tensor the taps and the ring see), `1 = ConvW [channels*kernel_size]
/// f32` (row-major, channel `c` at `c*kernel_size`), `2 = Ring
/// [state_len*channels] f32` (read+written; slot-major, `slot*channels + c`),
/// `3 = Residual [seq*channels] f32` (`gated`, UN-normed), `4 = OutBuf
/// [seq*channels] f32` = `residual + silu(conv)`.
///
/// `ring_pos` names the slot the next row is written into, which is also the
/// oldest row held, so lag `L` reads `(ring_pos + state_len - L) % state_len`.
/// A zero-filled ring at `ring_pos = 0` is the reference's zero left-pad at
/// sequence start. Advance it with [`qwen4_ple_conv_ring_advance`].
pub fn qwen4_ple_conv_params(
    channels: u32,
    seq_len: u32,
    kernel_size: u32,
    dilation: u32,
    ring_pos: u32,
) -> KernelParams {
    KernelParams::from_words(vec![channels, seq_len, kernel_size, dilation, ring_pos])
}

/// `state_len` for `qwen4_ple_conv.comp` = `(kernel_size - 1) * dilation`, i.e.
/// how many rows the ring holds (9 for this checkpoint). Exposed so a caller
/// sizes the ring buffer from the same expression the shader indexes it with.
pub const fn qwen4_ple_conv_state_len(kernel_size: u32, dilation: u32) -> u32 {
    kernel_size.saturating_sub(1) * dilation
}

/// The `ring_pos` to pass NEXT, after a dispatch that consumed `seq_len` rows.
/// The shader writes one row per step and advances by one, so this is a plain
/// `+ seq_len` modulo the state length — kept here rather than at each call
/// site because an off-by-one silently reads the wrong lag rather than failing.
pub const fn qwen4_ple_conv_ring_advance(
    ring_pos: u32,
    seq_len: u32,
    kernel_size: u32,
    dilation: u32,
) -> u32 {
    let state_len = qwen4_ple_conv_state_len(kernel_size, dilation);
    if state_len == 0 {
        return 0;
    }
    (ring_pos + seq_len % state_len) % state_len
}

/// Dispatch grid for `qwen4_ple_conv.comp`: `local_size_x = 256`, one
/// invocation per channel (`gl_GlobalInvocationID.x = c`), so
/// `ceil(channels/256)` workgroups — 40 for the 10240-wide PLE row. The conv is
/// fully per-channel, and each lane walks the sequence itself, so the ring
/// update needs no shared memory or barriers. Tail lanes self-mask on
/// `c >= channels`.
pub fn qwen4_ple_conv_dispatch(channels: u32) -> Dispatch {
    Dispatch::x(channels.div_ceil(256).max(1))
}

// Qwen3.8-Flash-Next HYPER-CONNECTIONS. One `Qwen4ExpTextGatedResidual` is FOUR
// dispatches, and the count is the design:
//
//   1. `Kernel::RmsNorm`          + [`rms_norm_params_grouped`]        -> hn
//   2. `Kernel::Qwen36RouterGemv` + [`qwen36_router_gemv_params`]      -> u_raw
//   3. `Kernel::Qwen4HcMix`       + [`qwen4_hc_mix_params`]            -> x
//   4. `Kernel::Qwen4HcCombine`   + [`qwen4_hc_combine_params`]        -> h
//
// with a `CommandRecorder::barrier()` between each (every step reads the
// previous step's writes) and ONE submit for the whole site. The naive
// composition of the same math is ten dispatches; at 97 sites per token and
// ~3.4us of host recording each, that difference is ~2ms/token of host time on
// a decode path that is already dispatch-bound. Steps 1 and 2 need no new
// shader — the grouped norm is the vendored `rms_norm` reached through push
// constants alone, and the down-projection is the generic f32
// `qwen36_router_gemv`.
//
// The oracle all four are diffed against is `infer_vulkan::qwen4_hc`
// (`gated_residual` + `inject_block_output`); `tests/device_qwen4_hc.rs` runs
// the whole sequence against it.

/// Push-constant block for `qwen4_hc_mix.comp` = `{hidden, hc_count,
/// hc_lowrank}` (3 `uint`s). Bindings (in order): `0 = hn`
/// [hc_count*hidden] f32 (read), `1 = W_up` [hc_count*hidden, hc_lowrank] f32
/// row-major (read), `2 = u_raw` [hc_lowrank] f32 (read), `3 = x` [hidden] f32
/// (write).
///
/// **Binding 2 is the RAW `W_down @ hn` dot, not `silu(dot / hc_count)`.** The
/// activation is applied inside this kernel. Step 2 of the sequence is the
/// generic `qwen36_router_gemv`, whose only epilogue is a plain sigmoid, and
/// that shader belongs to the Qwen3.6 MoE lane; rather than spend a fifth
/// dispatch on a `silu` pass, the scale-then-silu rides along on the read of
/// `u_raw`, which every workgroup makes anyway. Feeding this kernel an already
/// activated bottleneck applies silu twice and is not an error the driver can
/// catch.
///
/// `hc_count` must be <= 8 (`HC_MAX` in the shader, which sizes the register
/// accumulators).
pub fn qwen4_hc_mix_params(hidden: u32, hc_count: u32, hc_lowrank: u32) -> KernelParams {
    KernelParams::from_words(vec![hidden, hc_count, hc_lowrank])
}

/// Dispatch grid for `qwen4_hc_mix.comp`: ONE workgroup (64 lanes = one wave64)
/// per HIDDEN channel, NOT per output row of `W_up`. A row-per-workgroup grid
/// would be `hc_count * hidden` workgroups each producing one element of the
/// mixing gate, which then needs a second dispatch and a [hc_count*hidden]
/// scratch buffer for the cross-stream mean; this grid lets workgroup `i`
/// compute all `hc_count` gates for channel `i` and reduce them on the spot.
pub fn qwen4_hc_mix_dispatch(hidden: u32) -> Dispatch {
    Dispatch::x(hidden.max(1))
}

/// Push-constant block for `qwen4_hc_combine.comp` = `{hidden, hc_count}`
/// (2 `uint`s). Bindings (in order): `0 = hn` [hc_count*hidden] f32 (read),
/// `1 = W_inject` [hc_count, hc_count*hidden] f32 row-major (read), `2 = h`
/// [hc_count*hidden] f32 (**read and written, in place**), `3 = y` [hidden] f32
/// (read — the sublayer's block output).
///
/// `hn` and `h` must be DIFFERENT buffers: `hn` is the grouped-RMSNorm output
/// the injection gate is computed from, `h` is the raw residual the gated block
/// output accumulates onto (the reference adds onto `hyper_input`, not onto the
/// norm). Aliasing them lets one workgroup's phase-2 write corrupt another
/// workgroup's phase-1 read.
///
/// `hc_count` must be <= 8 (`HC_MAX` in the shader).
pub fn qwen4_hc_combine_params(hidden: u32, hc_count: u32) -> KernelParams {
    KernelParams::from_words(vec![hidden, hc_count])
}

/// Dispatch grid for `qwen4_hc_combine.comp`: `local_size_x = 256`, one
/// workgroup per 256 HIDDEN channels (each covering that slice in all
/// `hc_count` streams) — 10 workgroups at hidden = 2560.
///
/// Deliberately small. There is no cross-workgroup sync inside a dispatch, so
/// every workgroup recomputes the whole phase-1 reduction over `W_inject`
/// (~160 KB at these shapes) before it may write its slice; the per-CU byte
/// count is `160 KB + hidden/G` and is flat past a handful of workgroups, while
/// L2 traffic grows linearly in `G`. A one-element-per-thread grid (40
/// workgroups) would quadruple that traffic to shave nothing.
pub fn qwen4_hc_combine_dispatch(hidden: u32) -> Dispatch {
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

/// Batched, masked [`flash_attn_params`]: ALL query heads of a whole prefill
/// chunk in ONE dispatch, against the layer's full KV cache region.
///
/// Must be paired with [`FlashAttentionSpec::f32_f16_masked`] (Br is still 1, so
/// the query-tile grid dimension is `n` itself).
///
/// - Q is `[token][q_head][hsk]` f32. The shader reads
///   `data_qv4[q_offset/4 + (i*Br+r)*q_stride/4 + d]` with
///   `q_offset = iq2*nb02/4` and `q_stride = nb01` (gqa_ratio == 1), i.e. element
///   `iq2*(nb02/4) + i*nb01 + dim`. So `nb01 = nq*hsk` (token stride, ELEMENTS)
///   and `nb02 = 4*hsk` (head stride, BYTES).
/// - K/V are the layer's cache region bound at kv head 0. `k_offset =
///   ik2*nb12/2` (f16 elements) with `ik2 = iq2 / (neq2/nek2)`, so `neq2 = nq`,
///   `nek2 = nev2 = nkv` makes `ik2` the GQA-mapped kv head and
///   `nb12 = nb22 = plane_bytes` walks to it. Row strides `nb11 = hsk`,
///   `nb21 = hsv` (elements) index `[pos][head_dim]` inside a plane.
/// - The mask is one `[n][kv_len]` f16 block shared by every head
///   (`nem1 = n, nem2 = nem3 = 1` ⇒ `m_offset = 0`, `m_stride = KV`), with
///   `mask[r][c] = 0` iff `c <= start_pos + r`. `nem1 % Br == 0` disables the
///   shader's `nem1_bounds_check`.
/// - Output is `data_ov4[(iq2*hsv + (i*Br+row)*ne1*hsv)/4 + ...]`, so `ne1 = nq`
///   lays it out as `[token][q_head][hsv]` — exactly the o-projection's mmq input.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_params_batched(
    hsk: u32,
    hsv: u32,
    n: u32,
    kv_len: u32,
    nq: u32,
    nkv: u32,
    k_plane_bytes: u32,
    v_plane_bytes: u32,
    scale: f32,
) -> KernelParams {
    KernelParams::from_words(vec![
        n,                // N (query rows in this chunk)
        kv_len,           // KV (cached length incl. this chunk)
        nq,               // ne1 (query heads -> O token stride)
        nq,               // ne2
        1,                // ne3
        nq,               // neq2 (query heads)
        1,                // neq3
        nkv,              // nek2 (kv heads => rk2 = nq/nkv is the GQA ratio)
        1,                // nek3
        nkv,              // nev2
        1,                // nev3
        n,                // nem1 (mask rows)
        1,                // nem2 (=> m_offset = 0, mask shared by all heads)
        1,                // nem3
        nq * hsk,         // nb01 = Q token stride (elements)
        4 * hsk,          // nb02 = Q head stride (BYTES)
        0,                // nb03
        hsk,              // nb11 = K row stride (elements)
        k_plane_bytes,    // nb12 = K kv-head plane stride (BYTES)
        0,                // nb13
        hsv,              // nb21 = V row stride (elements)
        v_plane_bytes,    // nb22 = V kv-head plane stride (BYTES)
        0,                // nb23
        scale.to_bits(),  // scale = 1/sqrt(head_dim)
        0.0f32.to_bits(), // max_bias = 0 (no ALiBi)
        0.0f32.to_bits(), // logit_softcap = 0
        0,                // mask_n_head_log2 (no sink, no ALiBi split)
        0.0f32.to_bits(), // m0
        0.0f32.to_bits(), // m1
        1,                // gqa_ratio = 1 (one query head per workgroup)
        kv_len,           // split_kv = KV (end_j covers all positions)
        1,                // k_num = 1 (no split-k reduce)
    ])
}

/// Grid for [`flash_attn_params_batched`]: `i = gl_WorkGroupID.x` is the query
/// tile (`Tr = ceil(N/Br)` and `Br = 1`, so one workgroup per token) and
/// `iq2 = gl_WorkGroupID.y` is the query head.
pub fn flash_attn_dispatch_batched(n: u32, nq: u32) -> Dispatch {
    Dispatch {
        x: n.max(1),
        y: nq.max(1),
        z: 1,
    }
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

    /// Batched prefill GEMM. `kernel` must be one of `Kernel::Mmq*`; buffers are
    /// `[A quantized weight, B q8_1_x4 activations, D f32 dst]`, push is
    /// [`mmq_params`], dispatch is [`mmq_dispatch`].
    pub fn mmq_with_params_and_spec(
        kernel: Kernel,
        ctx: &vulkan_sys::VulkanContext,
        buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        dispatch: Dispatch,
        params: &super::KernelParams,
        spec: &super::MmqSpec,
    ) -> Result<()> {
        launch_with_params_and_specialization(
            kernel,
            ctx,
            buffers,
            dispatch,
            params,
            spec.specialization_u32(),
        )
    }

    /// Cooperative-matrix batched GEMM. `kernel` must be one of
    /// `Kernel::MmCm*`; buffers are `[A quantized weight, B **f16**
    /// activations, D f32 dst]`, push is [`mmq_params`] (byte-identical
    /// block), dispatch is [`mm_dispatch`].
    ///
    /// The `B` operand is the one real difference from
    /// [`mmq_with_params_and_spec`]: this shader reads plain row-major f16,
    /// not `block_q8_1_x4`. Handing it q8_1 bytes is not an error the driver
    /// can catch — it just computes the wrong answer.
    pub fn mm_with_params_and_spec(
        kernel: Kernel,
        ctx: &vulkan_sys::VulkanContext,
        buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        dispatch: Dispatch,
        params: &super::KernelParams,
        spec: &super::MmSpec,
    ) -> Result<()> {
        launch_with_params_and_specialization(
            kernel,
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

    pub fn mmq_with_params_and_spec(
        _kernel: Kernel,
        _ctx: &vulkan_sys::VulkanContext,
        _buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        _dispatch: super::Dispatch,
        _params: &super::KernelParams,
        _spec: &super::MmqSpec,
    ) -> Result<()> {
        Err(KernelError::NotCompiled)
    }

    pub fn mm_with_params_and_spec(
        _kernel: Kernel,
        _ctx: &vulkan_sys::VulkanContext,
        _buffers: &[&vulkan_sys::DeviceBuffer<'_>],
        _dispatch: super::Dispatch,
        _params: &super::KernelParams,
        _spec: &super::MmSpec,
    ) -> Result<()> {
        Err(KernelError::NotCompiled)
    }
}

#[cfg(not(feature = "vulkan"))]
launcher_fns!(stub::launch, stub::launch_with_params);
#[cfg(not(feature = "vulkan"))]
fused_launcher_fns!(stub::launch_with_params);

#[cfg(feature = "vulkan")]
pub use real::{
    flash_attn_with_params_and_spec, mm_with_params_and_spec, mmq_with_params_and_spec,
};
#[cfg(not(feature = "vulkan"))]
pub use stub::{
    flash_attn_with_params_and_spec, mm_with_params_and_spec, mmq_with_params_and_spec,
};

#[cfg(test)]
mod nvfp4_repack_tests {
    use super::{BLOCK_NVFP4_BYTES, QK_NVFP4, QK_NVFP4_SUB, nvfp4_row_bytes, repack_nvfp4_planes};

    /// UE4M3 codes with exact power-of-two values, one per sub-block, so the
    /// expected weights below are written down rather than recomputed:
    /// `0x30 -> 2^-1`, `0x38 -> 2^0`, `0x40 -> 2^1`, `0x48 -> 2^2`
    /// (bits `x eeee mmm`, bias 7; `ue4m3_to_f32` in `infer_gguf::dequant`).
    const SCALES: [u8; 4] = [0x38, 0x40, 0x30, 0x48];
    const SCALE_VALUES: [f32; 4] = [1.0, 2.0, 0.5, 4.0];

    /// E2M1 nibble -> value, `s eem`: the OCP FP4 table ggml spells as a
    /// doubled int8 `kvalues_mxfp4` (types.glsl:1767).
    const E2M1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    #[test]
    fn row_bytes_is_36_per_64_values() {
        assert_eq!(nvfp4_row_bytes(QK_NVFP4), Some(BLOCK_NVFP4_BYTES));
        // The two widths this model actually uses: hidden and moe_intermediate.
        assert_eq!(nvfp4_row_bytes(2560), Some(1440));
        assert_eq!(nvfp4_row_bytes(640), Some(360));
        // A row that is not a whole number of blocks would make row r>0 read
        // the previous row's tail, so it must be refused, not rounded.
        assert_eq!(nvfp4_row_bytes(32), None);
        assert_eq!(nvfp4_row_bytes(2560 + 16), None);
        assert_eq!(nvfp4_row_bytes(0), None);
    }

    /// Nibble for GLOBAL element index `g = r * ncols + k`:
    /// `(5*g + g/16) % 16`. Within one sub-block this is a full permutation of
    /// the 16 E2M1 codes (x5 is a bijection mod 16), and the `g/16` term
    /// shifts the permutation by one per sub-block — so no two sub-blocks,
    /// blocks, or rows carry the same byte pattern. The first version of this
    /// fixture used `k % 16`, which is periodic with the sub-block: dropping
    /// the sub-block index, dropping the block index, and pinning the source
    /// row to 0 were three mutations that all passed. Aperiodicity is the
    /// property the tests below actually rely on; keep it.
    fn nibble_at(g: usize) -> u8 {
        ((5 * g + g / QK_NVFP4_SUB) % 16) as u8
    }

    /// Scale code for global sub-block index `g_sub`, rotating the 4-code
    /// cycle by one per block so block 0's scales fit no other block.
    fn scale_index_at(g_sub: usize) -> usize {
        (g_sub + g_sub / 4) % SCALES.len()
    }

    fn source_planes(nrows: usize, ncols: usize) -> (Vec<u8>, Vec<u8>) {
        let qs = (0..nrows * ncols / 2)
            .map(|i| nibble_at(2 * i) | (nibble_at(2 * i + 1) << 4))
            .collect();
        let sc = (0..nrows * ncols / QK_NVFP4_SUB)
            .map(|g| SCALES[scale_index_at(g)])
            .collect();
        (qs, sc)
    }

    fn expected_row(r: usize, ncols: usize) -> Vec<f32> {
        (0..ncols)
            .map(|k| {
                let g = r * ncols + k;
                E2M1[nibble_at(g) as usize] * SCALE_VALUES[scale_index_at(g / QK_NVFP4_SUB)]
            })
            .collect()
    }

    /// Repack, then dequantize the result with `infer_gguf`'s independent
    /// `block_nvfp4` reader, and require the hand-written value sequence back.
    /// The fixture is aperiodic (see [`nibble_at`]), so a wrong sub-block
    /// index, a wrong block index, and a wrong source row each produce a
    /// value mismatch here rather than a coincidental pass.
    #[test]
    fn repacked_blocks_dequantize_to_the_source_elements() {
        const NROWS: usize = 2;
        const NCOLS: usize = 3 * QK_NVFP4;
        let (qs, sc) = source_planes(NROWS, NCOLS);
        let row_bytes = nvfp4_row_bytes(NCOLS).expect("row bytes");
        let mut packed = vec![0u8; NROWS * row_bytes];
        repack_nvfp4_planes(&qs, &sc, NROWS, NCOLS, &mut packed).expect("repack");

        for r in 0..NROWS {
            let got = infer_gguf::dequant::dequantize_row_nvfp4(
                &packed[r * row_bytes..(r + 1) * row_bytes],
                NCOLS,
            )
            .expect("dequantize repacked row");
            assert_eq!(got, expected_row(r, NCOLS), "row {r}");
        }
    }

    /// Pin the permutation at the byte level too, so the contract survives even
    /// if `infer_gguf`'s dequantizer is ever changed in the same direction as a
    /// repack bug. One sub-block: source bytes hold elements (2i, 2i+1); the
    /// destination byte `j` must hold elements (j, j+8).
    #[test]
    fn one_sub_block_permutes_pairs_into_halves() {
        // Elements 0..15 carry nibbles 0..15, i.e. element e has nibble e.
        let qs: Vec<u8> = (0..QK_NVFP4 / 2)
            .map(|i| {
                let lo = (2 * i % QK_NVFP4_SUB) as u8;
                let hi = ((2 * i + 1) % QK_NVFP4_SUB) as u8;
                lo | (hi << 4)
            })
            .collect();
        let sc = vec![SCALES[0]; QK_NVFP4 / QK_NVFP4_SUB];
        let mut packed = vec![0u8; BLOCK_NVFP4_BYTES];
        repack_nvfp4_planes(&qs, &sc, 1, QK_NVFP4, &mut packed).expect("repack");

        assert_eq!(&packed[..4], &[SCALES[0]; 4], "scales copied verbatim");
        // dst byte j = element j | element (j+8) << 4  =>  j | (j+8) << 4.
        let want: Vec<u8> = (0..8u8).map(|j| j | ((j + 8) << 4)).collect();
        assert_eq!(&packed[4..12], want.as_slice(), "sub-block 0 nibbles");
    }

    #[test]
    fn mis_sized_planes_are_refused_not_truncated() {
        const NCOLS: usize = QK_NVFP4;
        let (qs, sc) = source_planes(1, NCOLS);
        let mut packed = vec![0u8; BLOCK_NVFP4_BYTES];

        assert!(repack_nvfp4_planes(&qs[..qs.len() - 1], &sc, 1, NCOLS, &mut packed).is_err());
        assert!(repack_nvfp4_planes(&qs, &sc[..sc.len() - 1], 1, NCOLS, &mut packed).is_err());
        assert!(
            repack_nvfp4_planes(&qs, &sc, 1, NCOLS, &mut packed[..BLOCK_NVFP4_BYTES - 1]).is_err()
        );
        // ncols must be a whole number of blocks.
        assert!(repack_nvfp4_planes(&qs, &sc, 1, NCOLS + 16, &mut packed).is_err());
    }
}
