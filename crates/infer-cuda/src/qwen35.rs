//! Qwen3.5 / Qwen3.6 HYBRID model: gated-delta linear attention + periodic
//! full attention, BF16 MoE / dense MLP.
//!
//! Qwen3.5/3.6 production checkpoints interleave two attention kinds per layer
//! (`config.layer_types`):
//!   - `LinearAttention` (the majority): a gated-delta-rule recurrent linear
//!     attention with a depthwise conv1d front end and a gated output RMSNorm.
//!     This carries a per-slot recurrent state (`[V*K*V_head]` f32) + conv ring
//!     (`[qkv_dim*(kernel-1)]` bf16) ACROSS prefill and decode steps.
//!   - `FullAttention` (periodic): standard GQA with a per-head sigmoid gate on
//!     the q_proj output (`q_proj` rows = `heads*head_dim*2`), head_dim 128/256,
//!     on a contiguous per-slot K/V cache.
//!
//! Like [`crate::dsv4`], this model OWNS its KV state (no [`PagedKVPool`]) and
//! runs the uncached full-prefix correctness path: full-attn layers recompute
//! attention over the contiguous cache each step, linear-attn layers advance the
//! recurrent state in place. The continuous-batching paged + packed-batch path
//! (legacy `infer/src/model/qwen35`) is a perf follow-up.
//!
//! Gated-delta uses the RECURRENT kernel, never chunkwise: the chunkwise
//! TileLang WGMMA short-seq path HANGS on sm_90
//! (`errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md`).
//!
//! Precision: BF16 (the shared `moe::moe_forward` grouped GEMM). The two MoE swap
//! points for FP8 / 4-bit (Qwen3.6-4bit q4k) are inside
//! [`crate::moe::moe_forward`]'s two `moe_bf16_grouped_gemm_*` calls — a follow-up.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::ffi;
use cuda_kernels::kv_quant;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tensor::{
    HostMatrixSnapshot, WeightFormat, cache_ptr, offload_raw_slice, reload_raw_slice,
};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, PinnedHostSlice};
use half::bf16;
use infer_plan::SamplingParams;
use infer_topo::TpConfig;
use qwen35_spec::{Qwen35AttentionTensorNames, Qwen35Config};
use safetensors::tensor::Dtype;

use crate::loader::SafetensorLoader;
use crate::moe::{
    DEEPGEMM_CONTIG_ALIGN, MoeForwardScratch, QWEN35_DEEPGEMM_MIN_ROUTES, deepgemm_contig_rows_cap,
    moe_forward_into,
};
use crate::moe_config::ExpertSplit;
use crate::ops::{
    add_batch, argmax_into, argmax_row_into, copy_row_to_vec, embedding_batch, gemm_batch, gemv,
    silu_mul_fused, split_qkv, split2, upload_i32, warm_fp8_deepgemm_dense,
};
use crate::workspace::{HiddenSlot, PinnedSlot, SliceSlot, VecSlot};

#[path = "qwen35/dspark.rs"]
pub(crate) mod dspark;

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;
const QWEN35_BATCHED_DECODE_KV_SPLITS: usize = 4;

/// Longest query the FA3 paged path takes. FA3 zeroes the page stride when
/// `seqused_k` is set, so a ragged batch needs one launch per request; that is
/// free for decode/verify shapes and ruinous for prefill chunks, where the
/// TileLang kernel does the whole batch in one launch (measured 2026-07-28:
/// c=8 TTFT p50 12.07 → 18.23 s with prefill routed here).
const FA3_MAX_QLEN: usize = 64;

/// Floor for the derived FA3 decode split ceiling — the value shipped as a
/// constant before the ceiling was derived, and the measured optimum from
/// batch 4 up (2026-08-04: batch 8 is +0.36% at a ceiling of 20).
const FA3_DECODE_SPLITS_FLOOR: usize = 8;

/// Route full-attention prefill chunks (`seq_len > 1`) through the vendored
/// FA3 hopper fwd shim instead of the in-tree `nonpaged_prefill_attention`
/// kernel (42.1% of prefill GPU time at 3k).
/// Default ON (licensed 2026-06-11: 3k prefill −36%, multi-shape verified —
/// see `wins/2026-06-11-qwen35-fa3-prefill-licensed.md`);
/// `--qwen35-fa3 false` is the same-binary fallback arm. A build without an
/// sm_90 target links the stub, whose marker is 0, and the gate keeps the
/// in-tree kernel.
/// The marker is process-wide, but capability is checked on the bound context
/// so mixed-device workers cannot inherit another device's decision.
fn qwen35_fa3_enabled(ctx: &DeviceContext) -> bool {
    if !crate::runtime_flags::qwen35_fa3() {
        return false;
    }
    // SAFETY: pure host query exported by both the real shim and the stub.
    let real = unsafe { ffi::arle_fa3_real_kernel_marker_cuda() } == 1;
    if !real {
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            log::warn!(
                "FA3 stub build (no sm_90 target) — full-attention prefill \
                 stays on the in-tree kernel"
            );
        });
        return false;
    }
    ctx.compute_capability() == (9, 0)
}

/// The two conditions the AOT dispatch wrapper itself switches on: the flashqla
/// rows were compiled into this build, and the device is sm_90 (the only SM they
/// are emitted for). Probing by launching `seq_len = 0` instead would issue a
/// zero-block grid, which is an illegal launch that compute-sanitizer reports on
/// every run.
fn fq_kernels_available(ctx: &DeviceContext) -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let ok = cuda_kernels::KERNEL_CAPABILITIES
            .split(',')
            .any(|c| c == "flashqla")
            && ctx.compute_capability() == (9, 0);
        if !ok {
            log::warn!("FlashQLA chunked GDR unavailable (stub build or non-sm90); using the recurrent scan");
        }
        ok
    })
}

/// sm_70 (V100) has no FA3 (sm_80+ CUTLASS-3.x) and no BF16 compute — the
/// hand-written `arle_fa2_sm70_attention_cuda` (FA2, FP16 half2 math, BF16 I/O)
/// is the SOTA option. `major < 8` covers sm_70 and sm_75.
fn qwen35_fa2_sm70_enabled(ctx: &DeviceContext) -> bool {
    ctx.compute_capability().0 < 8
}

#[path = "qwen35_lora.rs"]
mod qwen35_lora;
pub use qwen35_lora::*;

pub(crate) struct Qwen35Layer {
    input_layernorm: DeviceVec,
    attn: Qwen35Attn,
    post_attention_layernorm: DeviceVec,
    /// Exactly one of `mlp` / `moe` is `Some`.
    mlp: Option<DenseMlp>,
    moe: Option<crate::loader::MoeLayerWeights>,
}

pub(crate) struct Qwen35MtpHead {
    pre_fc_norm_embedding: DeviceVec,
    pre_fc_norm_hidden: DeviceVec,
    /// Concat projection `[hidden, 2*hidden]` mapping the two normed inputs to
    /// the transformer-block input.
    fc: DeviceMatrix,
    /// The head's single transformer block (Full attention + dense MLP).
    layer: Qwen35Layer,
    /// Final RMSNorm before the shared lm_head.
    norm: DeviceVec,
}

pub(crate) struct Qwen35Model {
    pub(crate) ctx: DeviceContext,
    pub(crate) config: Qwen35Config,
    embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    layers: Vec<Qwen35Layer>,
    norm: DeviceVec,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    moe_config: Option<infer_moe::MoeConfig>,
    /// Tensor-parallel runtime. `world_size == 1` uses the no-op communicator,
    /// so every `all_reduce_sum` returns immediately (mirrors the dense
    /// [`crate::model::CudaModel`]).
    pub(crate) tp: crate::tp::TpRuntime,
    /// This rank's per-shard head counts (= global config on a single GPU).
    /// The forward sizes its buffers, slot state, and kernel launches from
    /// these, not the config.
    local_q_heads: usize,
    local_kv_heads: usize,
    local_linear_k_heads: usize,
    local_linear_v_heads: usize,
    expert_split: ExpertSplit,
    max_seq_len: usize,
    /// NextN-MTP draft head for speculative decode. `Some` (and
    /// `spec_draft_tokens > 0`) only when the constructor was asked to load it;
    /// the default decode path never reads it, so the baseline stays
    /// byte-identical when spec-decode is off.
    #[allow(dead_code)] // read by the spec-decode draft/verify path (next increment)
    mtp: Option<Qwen35MtpHead>,
    #[allow(dead_code)] // read by the spec-decode orchestrator (next increment)
    spec_draft_tokens: usize,
    /// Host-resident weight snapshot while the engine is offloaded for the OPD
    /// teacher time-share. `Some` iff [`Qwen35Model::offload_engine_weights`] ran
    /// without a matching [`Qwen35Model::reload_engine_weights`]; the device
    /// weight buffers are 1-element placeholders in that state and must NOT be
    /// forwarded through until reloaded.
    offloaded: Option<Box<OffloadedWeights>>,
    /// FP8 qweight/scales are kept in each `DeviceMatrix` after FP8→BF16
    /// promotion (see `promote_lora_target_to_bf16`): the share-frozen-base
    /// student aliases these device pointers, and the per-step LoRA merge
    /// dequantizes them on the fly to recover the pristine BF16 base — no
    /// separate BF16 base cache, saving ~2×base bytes during the 27B sync.
    /// Latched by [`Qwen35Model::frozen_base_fp8_pointers`]: resident FP8 base
    /// pointers have been exported for train-infer weight sharing.
    frozen_base_ptrs_exported: AtomicBool,
    /// Reusable device scratch for the `B·A` delta (sized to the largest dense
    /// merged matrix seen). Avoids a per-projection device alloc each step.
    lora_delta_scratch: Option<DeviceVec>,
    /// Projections whose resident device matrix currently includes a non-zero
    /// LoRA delta. Lets all-zero adapter steps skip weight uploads unless they
    /// need to restore a previously merged projection back to base.
    lora_dirty: HashSet<LoraBaseKey>,
    /// Pristine BF16 base for projections whose resident matrix is NOT FP8
    /// (e.g. linear-attn `in_proj_ba`). FP8 projections recover their base by
    /// dequantizing the kept-alive FP8 qweight/scales, so they are not cached
    /// here. Only small BF16 projections use this cache — no OOM risk.
    lora_base_dev: HashMap<LoraBaseKey, DeviceVec>,
    /// Cheap model/weights-version tag (hash of the checkpoint's safetensors
    /// file names + lengths + mtimes). Stamps the durable KV-recall manifest so
    /// a restart after an OPD weight update (which rewrites the checkpoint, so
    /// the mtimes/lengths shift) discards the now-stale recalled KV.
    #[allow(dead_code)] // WIP: durable KV-recall manifest weight-version stamp, not yet wired
    weights_epoch: String,
}

/// Host-resident snapshot of one transformer block's device weight buffers,
/// captured by [`Qwen35Model::offload_engine_weights`]. Reload restores in the
/// same field order.
struct OffloadedBlock {
    input_layernorm: Vec<bf16>,
    post_attention_layernorm: Vec<bf16>,
    attn: OffloadedAttn,
    /// Dense MLP snapshot — `Some` iff this layer is dense (mutually exclusive
    /// with `moe`, mirroring [`Qwen35Layer`]).
    mlp: Option<OffloadedDenseMlp>,
    moe: Option<crate::loader::MoeLayerHostSnapshot>,
}

struct OffloadedDenseMlp {
    gate_up_proj: HostMatrixSnapshot,
    down_proj: HostMatrixSnapshot,
}

struct OffloadedFullAttn {
    qkv_proj: HostMatrixSnapshot,
    o_proj: HostMatrixSnapshot,
    q_norm: Vec<bf16>,
    k_norm: Vec<bf16>,
}

struct OffloadedLinearAttn {
    in_proj_qkvz: HostMatrixSnapshot,
    in_proj_ba: HostMatrixSnapshot,
    conv1d_weight: Vec<bf16>,
    dt_bias: Vec<bf16>,
    a_log: Vec<f32>,
    norm_weight: Vec<f32>,
    out_proj: HostMatrixSnapshot,
}

enum OffloadedAttn {
    // Boxed: mirrors the device-side `Qwen35Attn` (large, size-skewed variants);
    // boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<OffloadedFullAttn>),
    Linear(Box<OffloadedLinearAttn>),
}

/// Full host-resident snapshot of all model device weights while offloaded.
/// `embed_tokens` doubles as the (tied) lm_head when `lm_head` is `None`.
struct OffloadedWeights {
    embed_tokens: HostMatrixSnapshot,
    lm_head: Option<HostMatrixSnapshot>,
    norm: Vec<bf16>,
    blocks: Vec<OffloadedBlock>,
}

impl std::fmt::Debug for Qwen35Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35Model")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("kv_heads", &self.config.num_key_value_heads)
            .field("head_dim", &self.config.head_dim)
            .field("experts", &self.config.num_experts)
            .finish()
    }
}

fn validate_qwen35_cuda_config(m: &Qwen35Config) -> Result<()> {
    ensure!(
        matches!(m.head_dim, 128 | 256),
        "clean CUDA Qwen3.5 hybrid path supports full-attention head_dim 128/256, got {}",
        m.head_dim
    );
    ensure!(
        m.num_attention_heads.is_multiple_of(m.num_key_value_heads),
        "Qwen3.5 num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
        m.num_attention_heads,
        m.num_key_value_heads
    );
    ensure!(
        m.linear_key_head_dim == 128 && m.linear_value_head_dim == 128,
        "clean CUDA Qwen3.5 gated-delta path supports linear key/value dim 128/128, got {}/{}",
        m.linear_key_head_dim,
        m.linear_value_head_dim
    );
    Ok(())
}

#[path = "qwen35_state.rs"]
mod state;
pub(crate) use state::*;

#[path = "qwen35_spec_state.rs"]
mod spec_state;
pub(crate) use spec_state::*;

#[path = "qwen35_workspace.rs"]
mod workspace;
pub(crate) use workspace::*;

#[path = "qwen35_load.rs"]
mod load;

#[path = "qwen35_forward.rs"]
mod forward;

#[path = "qwen35_attention.rs"]
mod attention;
pub(crate) use attention::*;

#[path = "qwen35_mlp.rs"]
mod mlp;
pub(crate) use mlp::*;

#[path = "qwen35_decode.rs"]
mod decode;

#[path = "qwen35_spec.rs"]
mod spec;
pub(crate) use spec::*;

/// Offset RMSNorm: Qwen3.5 norms store `weight - 1`, so apply `1 + weight`.
pub(crate) fn rms_norm_offset(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: x/weight/out valid on ctx.stream; out matches x shape.
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Offset RMSNorm (1+weight) over a single vector (the final norm before lm_head).
pub(crate) fn rms_norm_offset_vec(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: x/weight/out valid on ctx.stream; out matches x len.
    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_spec::LayerType;

    /// G3 whole-slot spill: `from_bytes(to_bytes(img))` must reproduce EVERY
    /// field byte-for-byte (the new correctness surface — the device buffers and
    /// the round-trip are independently proven). Ragged per-vec lengths +
    /// non-trivial bytes exercise the length-prefixed layout and the f32/bf16
    /// LE round-trips; a truncated buffer must error, never restore garbage.
    #[test]
    fn slot_image_byte_inverse() {
        let img = Qwen35SlotImage {
            full_attn_pages: (0u8..=233).cycle().take(777).collect(),
            full_attn_page_count: 3,
            gdr_host: vec![
                vec![1.5_f32, -2.0, f32::from_bits(0x4048_f5c3)],
                vec![],
                vec![0.0, -0.0, 123.456],
            ],
            conv_host: vec![
                vec![bf16::from_f32(1.0), bf16::from_f32(-3.5)],
                vec![bf16::from_f32(0.25)],
                vec![],
            ],
            seq_len: 1234,
        };
        let bytes = img.to_bytes();
        let back = Qwen35SlotImage::from_bytes(&bytes).expect("round-trip");
        assert_eq!(back.full_attn_pages, img.full_attn_pages);
        assert_eq!(back.full_attn_page_count, img.full_attn_page_count);
        assert_eq!(back.seq_len, img.seq_len);
        // f32 / bf16 compared on bit patterns so a -0.0 or NaN can't false-pass.
        assert_eq!(back.gdr_host.len(), img.gdr_host.len());
        for (b, a) in back.gdr_host.iter().zip(&img.gdr_host) {
            let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
            let ab: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
            assert_eq!(bb, ab);
        }
        assert_eq!(back.conv_host.len(), img.conv_host.len());
        for (b, a) in back.conv_host.iter().zip(&img.conv_host) {
            let bb: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
            let ab: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
            assert_eq!(bb, ab);
        }
        // A truncated payload errors rather than restoring a partial image.
        assert!(Qwen35SlotImage::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        // A foreign trailing byte is rejected (exact-length invariant).
        let mut extra = bytes.clone();
        extra.push(0xAB);
        assert!(Qwen35SlotImage::from_bytes(&extra).is_err());
    }

    fn qwen35_dense_4b_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 2,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            stop_token_ids: vec![151_645],
            bos_token_id: None,
            eos_token_id: 151_645,
            tie_word_embeddings: true,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 128,
            rope_cache_len_hint: Some(32_768),
            layer_types: vec![LayerType::FullAttention, LayerType::LinearAttention],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: vec![],
            full_attn_gated: true,
        }
    }

    #[test]
    fn cuda_qwen35_guard_accepts_dense_hybrid_4b_shape() {
        let cfg = qwen35_dense_4b_config();
        cfg.validate().unwrap();
        validate_qwen35_cuda_config(&cfg).unwrap();
    }

    #[test]
    fn cuda_qwen35_guard_rejects_unknown_full_attention_head_dim() {
        let mut cfg = qwen35_dense_4b_config();
        cfg.head_dim = 64;
        cfg.rotary_dim = 64;
        assert!(validate_qwen35_cuda_config(&cfg).is_err());
    }

    /// Reference host merge: `W[r,c] = base[r,c] + scale·Σ_k B[r,k]·A[k,c]`.
    fn host_merge_reference(
        base: &[bf16],
        a: &[f32],
        b: &[f32],
        rows: usize,
        cols: usize,
        rank: usize,
        scale: f32,
    ) -> Vec<bf16> {
        let mut merged = vec![bf16::ZERO; rows * cols];
        for row in 0..rows {
            let b_row = &b[row * rank..row * rank + rank];
            for col in 0..cols {
                let mut delta = 0.0f32;
                for (k, &b_rk) in b_row.iter().enumerate() {
                    delta += b_rk * a[k * cols + col];
                }
                let idx = row * cols + col;
                merged[idx] = bf16::from_f32(base[idx].to_f32() + scale * delta);
            }
        }
        merged
    }

    /// GPU gate: the on-device dense LoRA merge (`lora_device_gemm` +
    /// `lora_scaled_add_into`, including the host-side A transpose used by
    /// `merge_lora_proj_device`) must match the host triple-loop reference
    /// within BF16 tolerance. Requires a real CUDA device (run on GPU7).
    #[test]
    fn device_lora_merge_matches_host_reference() {
        let Ok(ctx) = DeviceContext::new() else {
            eprintln!("[device_lora_merge] no CUDA device; skipping");
            return;
        };

        // A representative dense projection shape (qkv-ish): rows=out, cols=in,
        // rank=32. Deterministic pseudo-random A/B/base.
        let rows = 512usize;
        let cols = 384usize;
        let rank = 32usize;
        let scale = 0.5f32 / rank as f32; // alpha=0.5 / rank

        let prng = |seed: usize| -> f32 {
            // Cheap deterministic LCG-ish value in [-0.5, 0.5).
            let x = (seed as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9_7F4A_7C15;
            ((x >> 33) as f32 / u32::MAX as f32) - 0.5
        };

        let base: Vec<bf16> = (0..rows * cols)
            .map(|i| bf16::from_f32(prng(i + 1) * 2.0))
            .collect();
        let a: Vec<f32> = (0..rank * cols).map(|i| prng(i + 1_000_003)).collect();
        let b: Vec<f32> = (0..rows * rank).map(|i| prng(i + 7_000_019)).collect();

        let reference = host_merge_reference(&base, &a, &b, rows, cols, rank, scale);

        // A transposed to [cols, rank] row-major: a_t[c*rank+k] = a[k*cols+c].
        let mut a_t = vec![bf16::ZERO; cols * rank];
        for k in 0..rank {
            for c in 0..cols {
                a_t[c * rank + k] = bf16::from_f32(a[k * cols + c]);
            }
        }
        let b_host: Vec<bf16> = b.iter().map(|&v| bf16::from_f32(v)).collect();

        let a_t_dev = DeviceVec::from_host(&ctx, &a_t).unwrap();
        let b_dev = DeviceVec::from_host(&ctx, &b_host).unwrap();
        let base_dev = DeviceVec::from_host(&ctx, &base).unwrap();
        let mut delta = DeviceVec::zeros(&ctx, rows * cols).unwrap();
        let mut out = DeviceMatrix::from_host(&ctx, &base, rows, cols).unwrap();

        crate::ops::lora_device_gemm(
            &ctx,
            &a_t_dev.data,
            &b_dev.data,
            &mut delta.data,
            rows,
            cols,
            rank,
        )
        .unwrap();
        crate::ops::lora_scaled_add_into(
            &ctx,
            &base_dev.data,
            &delta.data,
            &mut out.data,
            rows * cols,
            scale,
        )
        .unwrap();
        ctx.sync().unwrap();

        let device_merged = ctx.stream.clone_dtoh(&out.data).unwrap();
        ctx.sync().unwrap();
        assert_eq!(device_merged.len(), reference.len());

        let mut dot = 0.0f64;
        let mut nr = 0.0f64;
        let mut nd = 0.0f64;
        let mut max_abs = 0.0f32;
        for (r, d) in reference.iter().zip(device_merged.iter()) {
            let rf = r.to_f32();
            let df = d.to_f32();
            dot += (rf as f64) * (df as f64);
            nr += (rf as f64) * (rf as f64);
            nd += (df as f64) * (df as f64);
            max_abs = max_abs.max((rf - df).abs());
        }
        let cosine = dot / (nr.sqrt() * nd.sqrt());
        eprintln!(
            "[device_lora_merge] rows={rows} cols={cols} rank={rank} cosine={cosine:.8} max_abs_err={max_abs:e}"
        );
        assert!(
            cosine >= 0.9999,
            "device merge cosine {cosine:.8} < 0.9999 (max_abs_err {max_abs:e})"
        );
        // BF16 mantissa is 8 bits; a rank-32 reduction lands well within ~1e-2.
        assert!(
            max_abs <= 2.0e-2,
            "device merge max-abs-err {max_abs:e} exceeds BF16 tolerance"
        );
    }
}
