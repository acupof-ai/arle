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
use cudarc::driver::{
    CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, PinnedHostSlice, sys::CUevent_flags,
};
use half::bf16;
use infer_plan::SamplingParams;
use infer_topo::TpConfig;
use qwen35_spec::{Qwen35AttentionTensorNames, Qwen35Config};
use safetensors::tensor::Dtype;

use crate::executor::sample_cuda_token_scratched;
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

#[cfg(test)]
#[path = "qwen35_probe.rs"]
mod probe;
#[cfg(test)]
pub(crate) use probe::*;
fn qwen35_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_QWEN35_PROFILE").is_some()
            || std::env::var_os("ARLE_QWEN35_MOE_PROFILE").is_some()
    })
}

fn qwen35_startup_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_CUDA_STARTUP_PROFILE").is_some())
}

fn qwen35_startup_log(phase: &str, start: Instant, extra: std::fmt::Arguments<'_>) {
    if qwen35_startup_profile_enabled() {
        log::info!(
            target: "infer_cuda::startup",
            "cuda_startup phase=qwen35.{phase} elapsed_ms={:.1} {extra}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn qwen35_profile<T>(
    ctx: &DeviceContext,
    label: &'static str,
    layer_idx: Option<usize>,
    seq_len: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !qwen35_profile_enabled() {
        return f();
    }
    let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&ctx.stream)?;
    let host_t0 = Instant::now();
    let result = f();
    let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
    stop.record(&ctx.stream)?;
    stop.synchronize()?;
    let cuda_ms = start.elapsed_ms(&stop)? as f64;
    if std::env::var("INFER_TP_RANK")
        .map(|rank| rank == "0")
        .unwrap_or(true)
    {
        match layer_idx {
            Some(layer_idx) => eprintln!(
                "[qwen-layer-profile] {label} layer={layer_idx} seq={seq_len} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
            ),
            None => eprintln!(
                "[qwen-layer-profile] {label} layer=na seq={seq_len} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
            ),
        }
    }
    result
}

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

fn qwen35_fa3_decode_splits() -> usize {
    crate::runtime_flags::qwen35_fa3_decode_splits()
}

/// `--qwen35-gdr-chunked true`: route GDN prefill chunks (`seq_len > 1`)
/// through the FlashQLA chunked kernels (TileLang AOT, sm_90a) instead of
/// the serial `gated_delta_rule_prefill_recurrent` kernel (28.0% of prefill
/// GPU time pre-FA3).
/// Default ON (licensed 2026-08-02). The call site shape-guards the head
/// geometry, and builds without an sm_90 target link NOT_SUPPORTED stubs that
/// the probe below detects. Decode (`seq_len == 1`) stays on the recurrent
/// kernel.
fn qwen35_gdr_chunked_enabled() -> bool {
    crate::runtime_flags::qwen35_gdr_chunked()
}

/// One-shot probe: stub builds and non-sm90 devices return
/// CUDA_ERROR_NOT_SUPPORTED from the dispatch wrapper before touching any
/// pointer; a real kernel rejects the seq_len=0 launch with a different code.
fn fq_kernels_available(ctx: &DeviceContext, cumsum: ffi::FqCumsumFn) -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // The dispatch wrapper resolves SM via the calling thread's DRIVER
        // context — bind first or the probe is a per-thread lottery.
        if ctx.ctx.bind_to_thread().is_err() {
            return false;
        }
        // SAFETY: seq_len=0 → zero grid; no pointer is dereferenced.
        let r = unsafe { cumsum(std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null_mut()) };
        let ok = r != cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_SUPPORTED;
        if !ok {
            log::warn!("FlashQLA chunked GDR unavailable (stub build or non-sm90); using the recurrent scan");
        }
        ok
    })
}

/// sm_70 (V100) has no FA3 (sm_80+ CUTLASS-3.x) and no BF16 compute — the
/// hand-written `arle_fa2_sm70_attention_cuda` (FA2, FP16 half2 math, BF16 I/O)
/// is the SOTA option. `major < 8` covers sm_70 and sm_75. Latched once.
/// `ARLE_QWEN35_FA2_SM70=0` forces the naive FP32 SIMT fallback (A/B use).
fn qwen35_fa2_sm70_enabled(ctx: &DeviceContext) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (major, _minor) = ctx.compute_capability();
        major < 8
            && !matches!(
                std::env::var("ARLE_QWEN35_FA2_SM70").as_deref(),
                Ok("0") | Ok("false") | Ok("off")
            )
    })
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
    /// RMSNorm over the candidate token embedding, pre-`fc`.
    pre_fc_norm_embedding: DeviceVec,
    /// RMSNorm over the previous-step trunk hidden, pre-`fc`.
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
    /// This rank's EP expert ownership (every expert local on a single GPU).
    expert_split: ExpertSplit,
    /// Per-slot cache capacity (full-attn contiguous cache rows).
    max_seq_len: usize,
    /// NextN-MTP draft head for speculative decode. `Some` (and
    /// `spec_draft_tokens > 0`) only when the constructor was asked to load it;
    /// the default decode path never reads it, so the baseline stays
    /// byte-identical when spec-decode is off.
    #[allow(dead_code)] // read by the spec-decode draft/verify path (next increment)
    mtp: Option<Qwen35MtpHead>,
    /// Requested MTP draft depth (`0` = spec-decode off).
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

/// Host snapshot of a full-attention block (mirrors [`FullAttn`]).
struct OffloadedFullAttn {
    qkv_proj: HostMatrixSnapshot,
    o_proj: HostMatrixSnapshot,
    q_norm: Vec<bf16>,
    k_norm: Vec<bf16>,
}

/// Host snapshot of a gated-delta linear-attention block (mirrors [`LinearAttn`]).
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

    /// First index where two token streams differ (or the shorter length).
    fn first_divergence(a: &[u32], b: &[u32]) -> usize {
        a.iter()
            .zip(b.iter())
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()))
    }

    /// On-GPU correctness gate + perf A/B for the depth-1 NextN-MTP spec decode.
    ///
    /// Gated by `INFER_MTP_BENCH=1` (a no-op otherwise, so CI stays green). Run on
    /// a CUDA box with the 27B-FP8 checkpoint:
    /// ```text
    /// INFER_MTP_BENCH=1 INFER_MTP_MODEL=/data01/models/Qwen3.6-27B-FP8 \
    ///   INFER_MTP_PROMPT_IDS="<comma-separated real prompt token ids>" \
    ///   cargo test -p infer-cuda --release --features cuda \
    ///   mtp_spec_decode_gate_and_bench -- --ignored --nocapture
    /// ```
    /// Correctness (MoE → token-exact is confounded by atomic-scatter
    /// non-determinism, per the KV parity gate): run greedy no-spec TWICE to
    /// measure the self-consistency floor, then assert spec greedy diverges from
    /// the no-spec reference NO EARLIER than that floor. Perf: same binary, same
    /// prompt, same token budget — clean in-process A/B (no HTTP/scheduler noise).
    #[test]
    #[ignore = "GPU + 27B-FP8 checkpoint; opt in via INFER_MTP_BENCH=1"]
    fn mtp_spec_decode_gate_and_bench() {
        use std::time::Instant;
        if std::env::var("INFER_MTP_BENCH").is_err() {
            return;
        }
        let path = std::env::var("INFER_MTP_MODEL")
            .unwrap_or_else(|_| "/data01/models/Qwen3.6-27B-FP8".to_string());
        let prompt: Vec<u32> = std::env::var("INFER_MTP_PROMPT_IDS")
            .expect("INFER_MTP_PROMPT_IDS (comma-separated token ids) required")
            .split(',')
            .map(|s| s.trim().parse::<u32>().expect("token id"))
            .collect();
        let n_decode: usize = std::env::var("INFER_MTP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
        let depth: usize = std::env::var("INFER_MTP_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1);
        let max_seq = (prompt.len() + n_decode + 8).next_power_of_two().max(2048);
        let greedy = SamplingParams {
            temperature: 0.0,
            top_k: -1,
            ..Default::default()
        };

        // Load with the MTP head sized for `depth` draft tokens (the head KV
        // needs depth+1 slots; new_spec_slot_state derives this from the param).
        let model = Qwen35Model::from_safetensors(Path::new(&path), max_seq, Some(depth))
            .expect("load Qwen3.6-27B-FP8 with MTP head");

        let (num_linear, gdr_len, conv_len) = model.recurrent_dims();
        let run_nospec = || -> (Vec<u32>, f64) {
            let mut slot = model.new_slot_state();
            let mut acq_pool = Vec::new();
            slot.acquire_recurrent(&model.ctx, num_linear, gdr_len, conv_len, &mut acq_pool)
                .unwrap();
            let mut ws = Qwen35Workspace::new();
            let mut out = Vec::with_capacity(n_decode);
            let mut tok = model
                .forward_tokens(&mut slot, &mut ws, &prompt, 0, &greedy, 0)
                .unwrap()
                .0;
            out.push(tok);
            model.ctx.sync().unwrap();
            let t0 = Instant::now();
            for step in 1..n_decode {
                let sp = slot.seq_len();
                tok = model
                    .forward_tokens(&mut slot, &mut ws, &[tok], sp, &greedy, step as u64)
                    .unwrap()
                    .0;
                out.push(tok);
            }
            model.ctx.sync().unwrap();
            (out, t0.elapsed().as_secs_f64())
        };

        let run_spec = || -> (Vec<u32>, f64, usize, usize) {
            let mut slot = model.new_slot_state();
            let mut acq_pool = Vec::new();
            slot.acquire_recurrent(&model.ctx, num_linear, gdr_len, conv_len, &mut acq_pool)
                .unwrap();
            let mut spec = model.new_spec_slot_state().unwrap();
            let mut ws = Qwen35Workspace::new();
            // Prefill returns the next token's logits + the producing hidden.
            let (logits, dims, mut hidden) = model
                .forward_tokens_with_hidden(&mut slot, &mut ws, &prompt, 0, None)
                .unwrap();
            let vocab = dims[1];
            // The prefill's last-row argmax IS the first decoded token = the seed
            // `pending`; `hidden` is its producing trunk hidden (last prompt row).
            // On-device argmax (same path as the live decode), no full-logits D2H.
            let mut pending = argmax_row_into(
                &model.ctx,
                &logits,
                dims[0] - 1,
                vocab,
                &mut spec.argmax_scratch,
            )
            .unwrap();
            let mut out = vec![pending];
            let mut accepts = 0usize;
            let mut steps = 0usize;
            model.ctx.sync().unwrap();
            let t0 = Instant::now();
            while out.len() < n_decode {
                let sp = slot.seq_len();
                let (emitted, next_pending, next_hidden) = model
                    .spec_step(
                        &mut slot, &mut spec, &mut ws, pending, &hidden, sp, depth, &greedy, None,
                    )
                    .unwrap();
                // Accepted drafts this step = emitted.len() - 1 (the bonus is the
                // trunk's own token, not a draft). accept_rate = accepts/(steps*depth).
                accepts += emitted.len().saturating_sub(1);
                steps += 1;
                out.extend(emitted.iter().map(|&(t, _)| t));
                pending = next_pending;
                hidden = next_hidden;
            }
            model.ctx.sync().unwrap();
            (out, t0.elapsed().as_secs_f64(), accepts, steps)
        };

        let (ref1, _) = run_nospec();
        let (ref2, t_nospec) = run_nospec();
        let floor = first_divergence(&ref1, &ref2);

        let (spec_out, t_spec, accepts, steps) = run_spec();
        let spec_div = first_divergence(&ref1, &spec_out);

        let n = n_decode as f64;
        let drafted = (steps * depth).max(1);
        eprintln!(
            "[mtp-bench] depth={} | no-spec {:.1} tok/s ({:.3}s/{} tok) | spec {:.1} tok/s ({:.3}s) | speedup {:.2}x | accept {}/{} drafts = {:.1}% | {:.2} tok/step",
            depth,
            n / t_nospec,
            t_nospec,
            n_decode,
            n / t_spec,
            t_spec,
            t_nospec / t_spec,
            accepts,
            drafted,
            100.0 * accepts as f64 / drafted as f64,
            spec_out.len() as f64 / steps.max(1) as f64,
        );
        eprintln!(
            "[mtp-gate] MoE self-consistency floor: no-spec runs diverge @{floor}/{} | spec-vs-ref diverge @{spec_div}/{}",
            ref1.len(),
            ref1.len(),
        );
        // Spec is correct iff it tracks the no-spec reference at least as far as
        // two no-spec runs track each other (the MoE non-determinism floor).
        assert!(
            spec_div >= floor,
            "MTP spec greedy diverged from no-spec @{spec_div} BEFORE the MoE floor @{floor} — real spec bug, not non-determinism"
        );
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

        // --- device path (replicates merge_lora_proj_device) ---
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

    /// Microbench (ignored by default): replay the full per-step all-linear LoRA
    /// merge at Qwen3.5-4B dense shapes and time host triple-loop vs the
    /// on-device path. Run with `--ignored --nocapture` on GPU7.
    #[test]
    #[ignore = "GPU microbench; run explicitly with --ignored"]
    fn bench_lora_remerge_host_vs_device() {
        let Ok(ctx) = DeviceContext::new() else {
            eprintln!("[lora_remerge_bench] no CUDA device; skipping");
            return;
        };
        let hidden = 2560usize;
        let rank = 32usize;
        let scale = 16.0f32 / rank as f32;
        // Qwen3.5-4B all-linear per-layer projection [out, in] shapes
        // (full-attn QKVO + linear-attn in/out + dense MLP gate/up/down).
        let proj_shapes: &[(usize, usize)] = &[
            (4096, hidden), // q_proj  (32 heads × 128)
            (1024, hidden), // k_proj  (8  × 128)
            (1024, hidden), // v_proj
            (hidden, 4096), // o_proj
            (hidden, 9216), // mlp.down_proj
            (9216, hidden), // mlp.gate_proj
            (9216, hidden), // mlp.up_proj
        ];
        let num_layers = 40usize;

        let prng = |seed: usize| -> f32 {
            let x = (seed as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9_7F4A_7C15;
            ((x >> 33) as f32 / u32::MAX as f32) - 0.5
        };

        // Pre-stage device base + resident matrices once (the per-step cost is
        // the merge, not the one-time base capture).
        struct Proj {
            rows: usize,
            cols: usize,
            base: Vec<bf16>,
            a: Vec<f32>,
            b: Vec<f32>,
            base_dev: DeviceVec,
            matrix: DeviceMatrix,
        }
        let mut projs: Vec<Proj> = Vec::new();
        for layer in 0..num_layers {
            for (pi, &(rows, cols)) in proj_shapes.iter().enumerate() {
                let s = layer * 97 + pi * 13 + 1;
                let base: Vec<bf16> = (0..rows * cols)
                    .map(|i| bf16::from_f32(prng(i + s) * 2.0))
                    .collect();
                let a: Vec<f32> = (0..rank * cols).map(|i| prng(i + s + 11)).collect();
                let b: Vec<f32> = (0..rows * rank).map(|i| prng(i + s + 23)).collect();
                let base_dev = DeviceVec::from_host(&ctx, &base).unwrap();
                let matrix = DeviceMatrix::from_host(&ctx, &base, rows, cols).unwrap();
                projs.push(Proj {
                    rows,
                    cols,
                    base,
                    a,
                    b,
                    base_dev,
                    matrix,
                });
            }
        }
        let max_n = proj_shapes.iter().map(|(r, c)| r * c).max().unwrap();
        let mut delta = DeviceVec::zeros(&ctx, max_n).unwrap();

        // --- HOST triple-loop path (the old merge) ---
        let t0 = std::time::Instant::now();
        for p in &projs {
            let mut merged = vec![bf16::ZERO; p.rows * p.cols];
            for row in 0..p.rows {
                let b_row = &p.b[row * rank..row * rank + rank];
                for col in 0..p.cols {
                    let mut d = 0.0f32;
                    for (k, &b_rk) in b_row.iter().enumerate() {
                        d += b_rk * p.a[k * p.cols + col];
                    }
                    let idx = row * p.cols + col;
                    merged[idx] = bf16::from_f32(p.base[idx].to_f32() + scale * d);
                }
            }
            std::hint::black_box(&merged);
        }
        let host_ms = t0.elapsed().as_secs_f64() * 1e3;

        // --- DEVICE path (the new merge) ---
        ctx.sync().unwrap();
        let t1 = std::time::Instant::now();
        for p in &mut projs {
            let mut a_t = vec![bf16::ZERO; p.cols * rank];
            for k in 0..rank {
                for c in 0..p.cols {
                    a_t[c * rank + k] = bf16::from_f32(p.a[k * p.cols + c]);
                }
            }
            let b_host: Vec<bf16> = p.b.iter().map(|&v| bf16::from_f32(v)).collect();
            let a_t_dev = DeviceVec::from_host(&ctx, &a_t).unwrap();
            let b_dev = DeviceVec::from_host(&ctx, &b_host).unwrap();
            let n = p.rows * p.cols;
            crate::ops::lora_device_gemm(
                &ctx,
                &a_t_dev.data,
                &b_dev.data,
                &mut delta.data,
                p.rows,
                p.cols,
                rank,
            )
            .unwrap();
            let delta_view = delta.data.slice(0..n);
            crate::ops::lora_scaled_add_into(
                &ctx,
                &p.base_dev.data,
                &delta_view,
                &mut p.matrix.data,
                n,
                scale,
            )
            .unwrap();
        }
        ctx.sync().unwrap();
        let dev_ms = t1.elapsed().as_secs_f64() * 1e3;

        eprintln!(
            "[lora_remerge_bench] {} projections × {num_layers} layers ({} total)\n\
             [lora_remerge_bench]   HOST triple-loop: {host_ms:.1} ms\n\
             [lora_remerge_bench]   DEVICE merge:     {dev_ms:.1} ms\n\
             [lora_remerge_bench]   speedup:          {:.1}×",
            proj_shapes.len(),
            projs.len(),
            host_ms / dev_ms
        );
    }
}
