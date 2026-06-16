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
//! points for FP8 (`dsv4_fp8_grouped_gemm`) / 4-bit (Qwen3.6-4bit q4k) are inside
//! [`crate::moe::moe_forward`]'s two `moe_bf16_grouped_gemm_*` calls — a follow-up.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::{HostMatrixSnapshot, cache_ptr, offload_raw_slice, reload_raw_slice};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, sys::CUevent_flags};
use half::bf16;
use infer_plan::SamplingParams;
use infer_topo::TpConfig;
use qwen35_spec::{LayerType, Qwen35AttentionTensorNames, Qwen35Config};
use safetensors::tensor::Dtype;

use crate::executor::sample_cuda_token_scratched;
use crate::loader::SafetensorLoader;
use crate::moe::{
    DEEPGEMM_CONTIG_ALIGN, MoeForwardScratch, QWEN35_DEEPGEMM_MIN_ROUTES, deepgemm_contig_rows_cap,
    moe_forward_into,
};
use crate::moe_config::ExpertSplit;
use crate::ops::{
    add_batch, copy_row_to_vec, embedding_batch, gemm_batch, gemv, silu_mul,
    warm_fp8_deepgemm_dense,
};
use crate::workspace::{HiddenSlot, SliceSlot, VecSlot};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;

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
/// kernel (42.1% of prefill GPU time at 3k —
/// `docs/reviews/2026-06-11-qwen35-post-license-reprofile-rerank.md`).
/// Default ON (licensed 2026-06-11: 3k prefill −36%, multi-shape verified —
/// see `wins/2026-06-11-qwen35-fa3-prefill-licensed.md`);
/// `ARLE_QWEN35_FA3=0` is the same-binary fallback arm. On stub builds
/// (no `ARLE_CUDA_ENABLE_FA3`) the link marker is 0 and the gate silently
/// keeps the in-tree kernel, so the default is safe across build flavors.
/// Read once — prefill is never graph-captured, so a process-lifetime latch
/// is safe.
fn qwen35_fa3_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = !matches!(
            std::env::var("ARLE_QWEN35_FA3").as_deref(),
            Ok("0" | "false" | "FALSE" | "no" | "off" | "OFF")
        );
        if !on {
            return false;
        }
        // SAFETY: pure host query exported by both the real shim and the stub.
        let real = unsafe { ffi::arle_fa3_real_kernel_marker_cuda() } == 1;
        if !real {
            log::info!(
                "FA3 stub build (no ARLE_CUDA_ENABLE_FA3) — full-attention \
                 prefill stays on the in-tree kernel"
            );
        }
        real
    })
}

/// `ARLE_QWEN35_FA3_DECODE=1`: route single-token full-attention decode through
/// the vendored FA3 split-KV + PackGQA path. Default OFF until the 4K/c=1
/// needle + ITL gate licenses it. This path uses a host `seqlen_k` launch
/// parameter, so keep it out of the whole-step decode graph for now.
fn qwen35_fa3_decode_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = matches!(
            std::env::var("ARLE_QWEN35_FA3_DECODE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        );
        if !on {
            return false;
        }
        let graph_on = matches!(
            std::env::var("ARLE_QWEN35_DECODE_GRAPH").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        );
        if graph_on {
            log::info!(
                "ARLE_QWEN35_FA3_DECODE ignored while ARLE_QWEN35_DECODE_GRAPH=1; \
                 FA3 split decode uses host seqlen_k and is not graph-replay safe"
            );
            return false;
        }
        qwen35_fa3_enabled()
    })
}

fn qwen35_fa3_decode_splits() -> usize {
    static SPLITS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SPLITS.get_or_init(|| {
        std::env::var("ARLE_QWEN35_FA3_DECODE_SPLITS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(2, 256)
    })
}

/// `ARLE_QWEN35_GDR_CHUNKED=1`: route GDN prefill chunks (`seq_len > 1`)
/// through the FlashQLA chunked kernels (TileLang AOT, sm_90a) instead of
/// the serial `gated_delta_rule_prefill_recurrent` kernel (28.0% of prefill
/// GPU time pre-FA3 —
/// `docs/reviews/2026-06-11-qwen35-post-license-reprofile-rerank.md`).
/// Default OFF (candidate arm). Only valid on the baked Qwen3.6 single-GPU
/// shard (H=32/Hg=16/128/128 — the call site additionally shape-guards);
/// builds without an sm_90 target link NOT_SUPPORTED stubs, so keep the gate
/// off there. Decode (`seq_len == 1`) always stays on the recurrent kernel.
fn qwen35_gdr_chunked_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ARLE_QWEN35_GDR_CHUNKED").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    })
}

/// Raw (un-scaled) LoRA A/B matrices for one full-attention q/v projection,
/// pushed from the train crate's OPD student loop for the per-step re-merge.
///
/// `a` is row-major `[rank, in_features]`, `b` is row-major
/// `[out_features, rank]` — the PEFT on-disk convention (mirrors the legacy
/// `infer/src/model/qwen35/lora.rs::StudentLoraMatrices`). Values are raw;
/// `scale = alpha / rank` is applied exactly once at merge time.
#[derive(Debug, Clone)]
pub struct StudentLoraMatrices {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
}

/// One full-attention layer's optional q/v adapter for the in-memory re-merge
/// sync. `layer_idx` is the absolute model-layer index (must name a
/// full-attention layer).
#[derive(Debug, Clone)]
pub struct StudentLoraLayer {
    pub layer_idx: usize,
    pub q_proj: Option<StudentLoraMatrices>,
    pub v_proj: Option<StudentLoraMatrices>,
}

/// A full LoRA update pushed from the train crate into the infer student
/// engine. Carries raw A/B per full-attention layer plus `rank`/`alpha`; the
/// merge path applies `scale = alpha / rank` once.
#[derive(Debug, Clone)]
pub struct StudentLoraUpdate {
    pub layers: Vec<StudentLoraLayer>,
    pub rank: usize,
    pub alpha: f32,
}

/// Per-slot full-attention K/V cache (one contiguous bf16 cache per full-attn
/// layer) + per-slot gated-delta recurrent state (state + conv ring per
/// linear-attn layer). Carried across prefill/decode for one request.
///
/// All dims are this rank's LOCAL shard sizes (= the global config dims on a
/// single GPU): each TP rank caches only its own kv heads, v-head recurrent
/// slabs, and qkv conv channels.
pub(crate) struct Qwen35SlotState {
    /// `[num_full_layers]` contiguous K caches, each `max_seq_len*kv_dim` bf16.
    k_caches: Vec<DeviceVec>,
    v_caches: Vec<DeviceVec>,
    /// `[num_linear_layers]` gated-delta recurrent states (`Vh*Kd*Vd` f32).
    gdr_states: Vec<CudaSlice<f32>>,
    /// `[num_linear_layers]` conv1d rings (`qkv_dim*(kernel-1)` bf16).
    conv_states: Vec<DeviceVec>,
    /// Tokens materialized into the caches so far (full-attn kv_len).
    seq_len: usize,
}

impl Qwen35SlotState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_full: usize,
        num_linear: usize,
        max_seq_len: usize,
        kv_dim: usize,
        gdr_state_len: usize,
        conv_len: usize,
    ) -> Result<Self> {
        let mut k_caches = Vec::with_capacity(num_full);
        let mut v_caches = Vec::with_capacity(num_full);
        for _ in 0..num_full {
            k_caches.push(DeviceVec::zeros(ctx, max_seq_len * kv_dim)?);
            v_caches.push(DeviceVec::zeros(ctx, max_seq_len * kv_dim)?);
        }
        let mut gdr_states = Vec::with_capacity(num_linear);
        let mut conv_states = Vec::with_capacity(num_linear);
        for _ in 0..num_linear {
            gdr_states.push(
                ctx.stream
                    .alloc_zeros::<f32>(gdr_state_len)
                    .map_err(|e| anyhow!("alloc gated-delta state failed: {e}"))?,
            );
            conv_states.push(DeviceVec::zeros(ctx, conv_len)?);
        }
        Ok(Self {
            k_caches,
            v_caches,
            gdr_states,
            conv_states,
            seq_len: 0,
        })
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance the materialized length by `n` tokens. The captured decode
    /// graph's caller uses this: the graph body
    /// ([`Qwen35Model::forward_decode_step_captured`]) is host-state-free —
    /// replay re-launches only GPU work — so the host-side length advance
    /// happens exactly once per step at the call site, never inside the
    /// captured closure.
    pub(crate) fn advance_seq_len(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Reset for a fresh generation in this slot (zeros recurrent + conv state,
    /// rewinds the full-attn cache cursor; stale cache rows are overwritten on
    /// the next prefill).
    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.seq_len = 0;
        for s in &mut self.gdr_states {
            ctx.stream
                .memset_zeros(s)
                .map_err(|e| anyhow!("memset gated-delta state failed: {e}"))?;
        }
        for c in &mut self.conv_states {
            ctx.stream
                .memset_zeros(&mut c.data)
                .map_err(|e| anyhow!("memset conv state failed: {e}"))?;
        }
        Ok(())
    }
}

/// Persistent device workspace for the Qwen3.5/3.6 hybrid forward.
///
/// The forward used to perform ~425 fresh device allocations PER CALL
/// (3 `HiddenStates::zeros` per layer in the residual stream, ~6 per
/// attention block, ~10 per MoE layer, plus the sampling tail) — all fully
/// overwritten before any read, per decode token. This workspace gives every
/// such buffer a persistent exact-shape slot (see [`crate::workspace`] for the
/// reuse/zeroing/free-safety contract and why the cache is exact-shape rather
/// than capacity-sized: `all_reduce_sum` derives the TP collective message
/// length — and the one-shot-vs-NCCL choice — from `data.len()`, so oversized
/// buffers would change the reduction on TP>=2).
///
/// One workspace per executor (forwards are strictly serial); passed `&mut`
/// from the executor like the DSv4 decode scratch pools. Numerics are
/// byte-identical to the per-call path: same buffer sizes, same kernel
/// arguments, same launch order — only the allocation lifetime changes.
///
/// Write-before-read proof per slot (kernels verified in
/// `csrc/misc/elementwise_basic.cu`, `csrc/misc/norm.cu`,
/// `csrc/misc/conv1d.cu`, `csrc/misc/gated_delta_rule.cu`,
/// `csrc/attention/prefill_attention_hd256.cu`,
/// `csrc/attention/nonpaged_prefill_attention.cu`, `csrc/gemm/gemv.cu`;
/// MoE slots carry their own table on [`MoeForwardScratch`]):
///
/// | slot          | shape           | proof of full overwrite before read |
/// |---------------|-----------------|--------------------------------------|
/// | `token_ids`   | `[S]` i32       | H2D upload overwrites every call |
/// | `start_pos`   | `[1]` i32       | H2D upload overwrites every call |
/// | `hidden`      | `[H, S]`        | `embedding_batched` writes all `H*S`; per layer, the residual `add_cuda` rewrites all `H*S` before the next read |
/// | `normed`      | `[H, S]`        | `rms_norm_batched_offset` writes all rows (grid = S blocks × full row) |
/// | `hidden_mid`  | `[H, S]`        | `add_cuda` writes all `H*S` |
/// | `attn_out`    | `[H, S]`        | o/out-proj `gemm_cuda` is beta=0 (writes all `M*N`) |
/// | `mlp_out`     | `[H, S]`        | dense: down `gemm_cuda` beta=0; MoE: combine kernel writes all, then gated-add RMWs |
/// | `full.q_full` | `[2*Hq*D, S]`   | `gemm_cuda` beta=0 |
/// | `full.k_batch`/`v_batch` | `[Hkv*D, S]` | `gemm_cuda` beta=0 |
/// | `full.q_prepped` | `[Hq*D, S]`  | full-attn prep writes every (token, head, d): RoPE pair covers `[0, rotary)`, the `d >= rotary` branch covers the rest |
/// | `full.attn_heads`| `[Hq*D, S]`  | nonpaged attention writes every (token, head, d) from register accumulators; the gate kernel then RMWs in place |
/// | `linear.qkv`/`z`/`b_proj`/`a_proj` | `[*, S]` | `gemm_cuda` beta=0 |
/// | `linear.qkv_conv` | `[QKV, S]`  | conv1d prefill writes every (channel, t) |
/// | `linear.gdr_out`  | `[Vh*Vd, S]`| gated-delta kernels write every (token, v_head, d) (one block per v_head, j_slice 0 writes the reduced output) |
/// | `linear.normed_out`| `[Vh*Vd, S]`| `rms_norm_gated` writes every (head, d) over `Vh*S` blocks |
/// | `dense.gate`/`up` | `[I, S]`    | `gemm_cuda` beta=0 |
/// | `dense.act`   | `[I, S]`        | `silu_mul` writes all `I*S` |
/// | `last_hidden` | `[H]`           | `memcpy_dtod` overwrites the full vec |
/// | `last_normed` | `[H]`           | `rms_norm_offset` writes all `H` |
/// | `logits`      | `[V]`           | `gemv` writes every output row |
/// | `argmax_out`  | `[1]` i32       | argmax kernel writes it before the D2H read (sampling tail, never captured) |
///
/// The full-logits OPD tail (`[V, S]`) is RETURNED to the caller and stays a
/// per-call allocation by design.
#[derive(Default)]
pub(crate) struct Qwen35Workspace {
    token_ids: SliceSlot<i32>,
    /// GPU-resident `start_pos` for the full-attn prep kernel — uploaded once per
    /// forward (the value is identical for every full-attn layer in the call;
    /// the old path uploaded one identical buffer per layer). The decode graph
    /// also reads it from the devpos attention kernel, so it is the single
    /// per-step position scalar staged pre-replay.
    start_pos: SliceSlot<i32>,
    hidden: HiddenSlot,
    normed: HiddenSlot,
    hidden_mid: HiddenSlot,
    attn_out: HiddenSlot,
    mlp_out: HiddenSlot,
    full: FullAttnScratch,
    linear: LinearAttnScratch,
    dense: DenseMlpScratch,
    moe: MoeForwardScratch,
    last_hidden: VecSlot,
    last_normed: VecSlot,
    logits: VecSlot,
    /// Persistent argmax output (one i32) for the greedy sampling tail —
    /// removes the last steady-state per-token device allocation
    /// (`ops::argmax`'s `alloc_zeros(1)`).
    argmax_out: SliceSlot<i32>,
    /// Buffer-address generation. Bumped whenever cached buffers are dropped
    /// wholesale ([`Self::release`]) — i.e. whenever previously-cached device
    /// ADDRESSES may change on the next `get`. The captured decode graph bakes
    /// buffer addresses, so it records this at capture and recaptures on
    /// mismatch instead of replaying against freed memory.
    epoch: u64,
}

#[derive(Default)]
pub(crate) struct FullAttnScratch {
    q_full: HiddenSlot,
    k_batch: HiddenSlot,
    v_batch: HiddenSlot,
    q_prepped: HiddenSlot,
    attn_heads: HiddenSlot,
    /// FA3 prefill scratch (`ARLE_QWEN35_FA3`): fp32 softmax LSE
    /// `[local_q_heads * seq_len]` (write-only output of the fwd kernel) and
    /// the persistent-scheduler semaphore (1 i32, zeroed by the shim per
    /// launch).
    fa3_lse: SliceSlot<f32>,
    /// FA3 split-decode scratch (`ARLE_QWEN35_FA3_DECODE`): fp32 partial
    /// outputs `[splits, b=1, local_q_heads, seq_len, head_dim]`.
    fa3_oaccum: SliceSlot<f32>,
    /// FA3 split-decode scratch: fp32 partial LSE
    /// `[splits, b=1, local_q_heads, seq_len]`.
    fa3_lseaccum: SliceSlot<f32>,
    fa3_semaphore: SliceSlot<i32>,
}

#[derive(Default)]
pub(crate) struct LinearAttnScratch {
    qkv: HiddenSlot,
    z: HiddenSlot,
    b_proj: HiddenSlot,
    a_proj: HiddenSlot,
    qkv_conv: HiddenSlot,
    gdr_out: HiddenSlot,
    normed_out: HiddenSlot,
    /// FlashQLA chunked-prefill scratch (`ARLE_QWEN35_GDR_CHUNKED`), all
    /// token-major dense: q/k `[S, Hg, 128]` bf16 l2norm'd, v `[S, H, 128]`
    /// bf16, a_inv `[S, H, 64]` bf16, g/g_cumsum/beta `[S, H]` f32.
    fq_q: HiddenSlot,
    fq_k: HiddenSlot,
    fq_v: HiddenSlot,
    fq_a: HiddenSlot,
    fq_g: SliceSlot<f32>,
    fq_g_cumsum: SliceSlot<f32>,
    fq_beta: SliceSlot<f32>,
}

#[derive(Default)]
pub(crate) struct DenseMlpScratch {
    gate: HiddenSlot,
    up: HiddenSlot,
    act: HiddenSlot,
}

impl Qwen35Workspace {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Buffer-address generation (see the `epoch` field).
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Drop every cached buffer (frees the VRAM). Called by the executor after
    /// the OPD weight offload so the workspace does not hold prefill-shaped
    /// scratch while the student backward needs the headroom. The caller must
    /// have quiesced the device first (`offload_engine_weights` syncs).
    /// Bumps the address epoch: any captured decode graph over these buffers
    /// is stale after this.
    pub(crate) fn release(&mut self) {
        self.epoch += 1;
        let Self {
            token_ids,
            start_pos,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            last_hidden,
            last_normed,
            logits,
            argmax_out,
            epoch: _,
        } = self;
        token_ids.release();
        start_pos.release();
        hidden.release();
        normed.release();
        hidden_mid.release();
        attn_out.release();
        mlp_out.release();
        full.q_full.release();
        full.k_batch.release();
        full.v_batch.release();
        full.q_prepped.release();
        full.attn_heads.release();
        linear.qkv.release();
        linear.z.release();
        linear.b_proj.release();
        linear.a_proj.release();
        linear.qkv_conv.release();
        linear.gdr_out.release();
        linear.normed_out.release();
        dense.gate.release();
        dense.up.release();
        dense.act.release();
        moe.release();
        last_hidden.release();
        last_normed.release();
        logits.release();
        argmax_out.release();
    }
}

/// Persistent device state for the rows>1 BATCHED DECODE path (stage 1:
/// contiguous per-slot KV kept, no paged migration). Re-port of the deleted
/// monolith's proven design (`e81b98fb~1` `infer/src/model/qwen35/batch_decode.rs`,
/// `BatchDecodeBuffers35` + per-layer pointer tables, lines 87-89/780-815)
/// onto the rewrite's workspace slots, using DSv4's batched-decode executor
/// shape (`dsv4.rs` Step-A per-row attention) as the template.
///
/// Owns a DEDICATED forward workspace: it only ever sees `[*, B]` decode
/// shapes, so the main (prefill-reshaping) workspace never thrashes, and —
/// critically — every buffer that feeds `TpRuntime::all_reduce_sum` is an
/// EXACT-shape `[dim, B]` allocation. `all_reduce_sum` derives the collective
/// message length (and the one-shot-vs-NCCL choice) from `data.len()`
/// (see `workspace.rs`), so exact-shape buffers make the reduced message
/// exactly B valid columns BY CONSTRUCTION — no capacity tail of stale
/// columns can ever enter a reduction. (Deviation from the monolith's
/// capacity-sized `set_batch_size` buffers, deliberate: the monolith had a
/// length-honest collective API; the rewrite's does not.)
///
/// Pointer tables are capacity-sized (`[num_slots]` u64 per linear layer per
/// kind) and uploaded with B valid entries; the batch kernels read entries
/// `[0, gridDim.y)` = `[0, B)` only, so the dead tail is never dereferenced.
/// Tables are a pure function of `(slot_indices, layer)`: the per-slot conv
/// ring / GDR state `CudaSlice`s are allocated once at executor construction
/// and never re-allocated (`Qwen35SlotState::reset` memsets in place; the OPD
/// weight offload leaves slot state untouched), so restaging is needed only
/// when the row→slot mapping changes (monolith `TileLangDecodeMetadata.update`
/// pattern).
pub(crate) struct Qwen35BatchDecodeState {
    /// Dedicated `[*, B]`-shaped forward workspace (see struct docs).
    ws: Qwen35Workspace,
    /// Per-row absolute start positions (`[B]` i32), staged once per step;
    /// the per-row full-attention kernel launches read `positions + r`.
    positions: SliceSlot<i32>,
    /// Per-LINEAR-layer `[num_slots]` u64 device tables of conv-ring pointers
    /// (`Half** [B] -> [C, K-1]` as `conv1d_decode_batch_cuda` consumes them).
    conv_state_ptrs: Vec<CudaSlice<u64>>,
    /// Per-LINEAR-layer `[num_slots]` u64 device tables of GDR-state pointers
    /// (`float** [B] -> [Vh, Kd, Vd]` as `gdr_decode_batch_cuda` consumes them).
    gdr_state_ptrs: Vec<CudaSlice<u64>>,
    /// Host staging vecs for the table uploads (monolith pattern: one
    /// `memcpy_htod` per layer per table, no per-row H2D).
    conv_host: Vec<u64>,
    gdr_host: Vec<u64>,
    /// Row→slot mapping the tables were last staged for (empty = never).
    staged_slot_indices: Vec<usize>,
    /// Batched logits `[vocab, B]` (final norm + lm_head GEMM over all rows).
    logits_batch: HiddenSlot,
    /// Batched greedy argmax outputs `[B]` i32.
    argmax: SliceSlot<i32>,
}

impl Qwen35BatchDecodeState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_linear_layers: usize,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!(
            max_batch > 0,
            "Qwen3.5 batched decode requires max_batch > 0"
        );
        let mut conv_state_ptrs = Vec::with_capacity(num_linear_layers);
        let mut gdr_state_ptrs = Vec::with_capacity(num_linear_layers);
        for layer_idx in 0..num_linear_layers {
            conv_state_ptrs.push(ctx.stream.alloc_zeros::<u64>(max_batch).map_err(|e| {
                anyhow!("alloc qwen35 batch conv_state_ptrs layer {layer_idx}: {e}")
            })?);
            gdr_state_ptrs.push(ctx.stream.alloc_zeros::<u64>(max_batch).map_err(|e| {
                anyhow!("alloc qwen35 batch gdr_state_ptrs layer {layer_idx}: {e}")
            })?);
        }
        Ok(Self {
            ws: Qwen35Workspace::new(),
            positions: SliceSlot::default(),
            conv_state_ptrs,
            gdr_state_ptrs,
            conv_host: vec![0u64; max_batch],
            gdr_host: vec![0u64; max_batch],
            staged_slot_indices: Vec::new(),
            logits_batch: HiddenSlot::default(),
            argmax: SliceSlot::default(),
        })
    }

    /// Re-upload the per-layer state pointer tables iff the row→slot mapping
    /// changed (tables are a pure function of `(slot_indices, layer)`; see
    /// struct docs for why the slot-state addresses themselves are stable).
    fn stage_pointer_tables(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
    ) -> Result<()> {
        if self.staged_slot_indices == slot_indices {
            return Ok(());
        }
        let b = slot_indices.len();
        ensure!(
            b <= self.conv_host.len(),
            "Qwen3.5 batched decode batch {} exceeds table capacity {}",
            b,
            self.conv_host.len()
        );
        for layer_idx in 0..self.conv_state_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.conv_states.len() && layer_idx < slot.gdr_states.len(),
                    "Qwen3.5 batched decode linear layer {layer_idx} outside slot state \
                     (conv={}, gdr={})",
                    slot.conv_states.len(),
                    slot.gdr_states.len()
                );
                let (conv_ptr, _gc) = slot.conv_states[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (gdr_ptr, _gg) = slot.gdr_states[layer_idx].device_ptr_mut(&ctx.stream);
                self.conv_host[r] = conv_ptr;
                self.gdr_host[r] = gdr_ptr;
            }
            ctx.stream
                .memcpy_htod(&self.conv_host[..b], &mut self.conv_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 conv_state_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(&self.gdr_host[..b], &mut self.gdr_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 gdr_state_ptrs layer {layer_idx}: {e}"))?;
        }
        self.staged_slot_indices = slot_indices.to_vec();
        Ok(())
    }

    /// Drop the workspace VRAM (OPD weight time-share hook). The pointer
    /// TABLES and the staged mapping stay: the per-slot state addresses they
    /// hold are executor-owned and untouched by the weight offload, so they
    /// remain valid across an offload/reload cycle (and they are ~KB-scale).
    pub(crate) fn release(&mut self) {
        self.ws.release();
        self.positions.release();
        self.logits_batch.release();
        self.argmax.release();
    }
}

pub(crate) enum Qwen35Attn {
    // Boxed: `FullAttn`/`LinearAttn` are large (multiple DeviceMatrix) and
    // size-skewed; boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<FullAttn>),
    Linear(Box<LinearAttn>),
}

/// Gated full attention (q_proj carries the per-head sigmoid gate → rows =
/// `heads*head_dim*2`).
pub(crate) struct FullAttn {
    q_proj: DeviceMatrix,
    k_proj: DeviceMatrix,
    v_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
}

/// Gated-delta-rule linear attention.
pub(crate) struct LinearAttn {
    in_proj_qkv: DeviceMatrix,
    in_proj_z: DeviceMatrix,
    in_proj_b: DeviceMatrix,
    in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight `[qkv_dim*kernel]` bf16.
    conv1d_weight: DeviceVec,
    dt_bias: DeviceVec,
    /// `A_log` `[num_value_heads]` f32 (this rank's v-head shard under TP).
    a_log: CudaSlice<f32>,
    /// Gated output RMSNorm scale `[value_head_dim]` f32, broadcast across
    /// heads by rms_norm_gated (norm.cu `weight[tid]`) — replicated under TP.
    norm_weight: CudaSlice<f32>,
    out_proj: DeviceMatrix,
}

pub(crate) struct Qwen35Layer {
    input_layernorm: DeviceVec,
    attn: Qwen35Attn,
    post_attention_layernorm: DeviceVec,
    /// Exactly one of `mlp` / `moe` is `Some`.
    mlp: Option<DenseMlp>,
    moe: Option<crate::loader::MoeLayerWeights>,
}

pub(crate) struct DenseMlp {
    gate_proj: DeviceMatrix,
    up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
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
    /// Host-resident weight snapshot while the engine is offloaded for the OPD
    /// teacher time-share. `Some` iff [`Qwen35Model::offload_engine_weights`] ran
    /// without a matching [`Qwen35Model::reload_engine_weights`]; the device
    /// weight buffers are 1-element placeholders in that state and must NOT be
    /// forwarded through until reloaded.
    offloaded: Option<Box<OffloadedWeights>>,
    /// Pristine base-weight cache for the per-step student LoRA re-merge, keyed
    /// by absolute (full-attention) layer index. Populated lazily on the first
    /// [`Qwen35Model::remerge_student_lora`] call (before any merge mutates the
    /// resident weights) so every re-merge recomputes `W = base + scale·B·A`
    /// from the *original* checkpoint weight, never from an already-merged one.
    /// Each entry is `(q_proj base bf16, v_proj base bf16)` row-major host copies.
    lora_base: HashMap<usize, LoraBaseWeights>,
}

/// Pristine host copies of one full-attention layer's q/v projection weights,
/// captured on the first re-merge so subsequent merges start from the original
/// checkpoint values, not the previously-merged ones.
#[derive(Debug, Clone, Default)]
struct LoraBaseWeights {
    q_proj: Option<Vec<bf16>>,
    v_proj: Option<Vec<bf16>>,
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
}

struct OffloadedDenseMlp {
    gate_proj: HostMatrixSnapshot,
    up_proj: HostMatrixSnapshot,
    down_proj: HostMatrixSnapshot,
}

/// Host snapshot of a full-attention block (mirrors [`FullAttn`]).
struct OffloadedFullAttn {
    q_proj: HostMatrixSnapshot,
    k_proj: HostMatrixSnapshot,
    v_proj: HostMatrixSnapshot,
    o_proj: HostMatrixSnapshot,
    q_norm: Vec<bf16>,
    k_norm: Vec<bf16>,
}

/// Host snapshot of a gated-delta linear-attention block (mirrors [`LinearAttn`]).
struct OffloadedLinearAttn {
    in_proj_qkv: HostMatrixSnapshot,
    in_proj_z: HostMatrixSnapshot,
    in_proj_b: HostMatrixSnapshot,
    in_proj_a: HostMatrixSnapshot,
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

impl Qwen35Model {
    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    /// This rank's full-attention query width (`local_q_heads * head_dim`).
    fn local_full_attn_q_dim(&self) -> usize {
        self.local_q_heads * self.config.head_dim
    }

    /// This rank's GATED q_proj output width: the projection interleaves
    /// `[query; gate]` per head, so each local head contributes `2*head_dim` rows.
    fn local_full_attn_q_proj_dim(&self) -> usize {
        self.local_q_heads * self.config.head_dim * 2
    }

    /// This rank's full-attention K/V width (`local_kv_heads * head_dim`).
    fn local_full_attn_kv_dim(&self) -> usize {
        self.local_kv_heads * self.config.head_dim
    }

    /// This rank's fused gated-delta `[q | k | v]` width.
    fn local_linear_qkv_dim(&self) -> usize {
        let qk = 2 * self.local_linear_k_heads * self.config.linear_key_head_dim;
        qk + self.local_linear_v_heads * self.config.linear_value_head_dim
    }

    /// This rank's gated-delta output / z-gate width.
    fn local_linear_z_dim(&self) -> usize {
        self.local_linear_v_heads * self.config.linear_value_head_dim
    }

    pub(crate) fn new_slot_state(&self) -> Result<Qwen35SlotState> {
        let c = &self.config;
        let num_full = c.num_full_attention_layers();
        let num_linear = c.num_hidden_layers - num_full;
        // Slot state is sized from this rank's LOCAL shard widths: each rank
        // caches only its own kv heads / v-head recurrent slabs / qkv channels.
        Qwen35SlotState::new(
            &self.ctx,
            num_full,
            num_linear,
            self.max_seq_len,
            self.local_full_attn_kv_dim(),
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim,
            self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1),
        )
    }

    /// Warm the Qwen FP8 dense DeepGEMM JIT for the CUDA default prefill chunk.
    ///
    /// The routed-expert port already uses DSv4's grouped path. The SLO 4096/256
    /// regression was the divergent dense-projection lane: first request paid
    /// one DeepGEMM JIT compile per `(M,N,K)` dense FP8 projection shape. Warming
    /// unique `(rows, cols)` at `M=2048` mirrors the scheduler's CUDA Qwen chunk
    /// floor and keeps the compile cost out of request TTFT without mutating KV
    /// or recurrent state.
    pub(crate) fn warm_fp8_deepgemm_dense_prefill(&self) -> Result<(usize, usize)> {
        let warm_m = self.max_seq_len.min(2048);
        if warm_m < 1024 {
            return Ok((0, warm_m));
        }
        let mut seen = HashSet::new();
        let mut warmed = 0usize;
        let mut warm = |weight: &DeviceMatrix| -> Result<()> {
            if seen.insert((weight.rows, weight.cols))
                && warm_fp8_deepgemm_dense(&self.ctx, weight, warm_m)?
            {
                warmed += 1;
            }
            Ok(())
        };

        for layer in &self.layers {
            match &layer.attn {
                Qwen35Attn::Full(full) => {
                    warm(&full.q_proj)?;
                    warm(&full.k_proj)?;
                    warm(&full.v_proj)?;
                    warm(&full.o_proj)?;
                }
                Qwen35Attn::Linear(linear) => {
                    warm(&linear.in_proj_qkv)?;
                    warm(&linear.in_proj_z)?;
                    warm(&linear.in_proj_b)?;
                    warm(&linear.in_proj_a)?;
                    warm(&linear.out_proj)?;
                }
            }
            if let Some(mlp) = &layer.mlp {
                warm(&mlp.gate_proj)?;
                warm(&mlp.up_proj)?;
                warm(&mlp.down_proj)?;
            }
            if let Some(moe) = &layer.moe {
                warm(&moe.router_gate)?;
                warm(&moe.shared_gate)?;
                warm(&moe.shared_up)?;
                warm(&moe.shared_down)?;
                warm(&moe.shared_gate_router)?;
            }
        }
        if warmed > 0 {
            self.ctx.sync()?;
        }
        Ok((warmed, warm_m))
    }

    /// Warm the Qwen FP8 routed-MoE grouped DeepGEMM JIT for the CUDA default
    /// prefill chunk. The helper only launches the two JIT-backed grouped GEMMs;
    /// pack/requant kernels are static CUDA kernels and do not need JIT warmup.
    pub(crate) fn warm_fp8_deepgemm_grouped_prefill(&self) -> Result<(usize, usize, usize, usize)> {
        let warm_tokens = self.max_seq_len.min(2048);
        let topk = self.config.num_experts_per_tok;
        let mut seen = HashSet::new();
        let mut warmed = 0usize;
        let mut min_rows = usize::MAX;
        let mut max_rows = 0usize;
        for warm_tokens in [warm_tokens, warm_tokens.saturating_sub(16)] {
            let warm_routes = warm_tokens.saturating_mul(topk);
            if warm_routes < QWEN35_DEEPGEMM_MIN_ROUTES {
                continue;
            }
            for layer in &self.layers {
                let Some(moe) = &layer.moe else {
                    continue;
                };
                let (Some(w13), Some(down)) = (&moe.w13_fp8_grouped, &moe.down_fp8_grouped) else {
                    continue;
                };
                let rows = deepgemm_contig_rows_cap(warm_routes, w13.groups, DEEPGEMM_CONTIG_ALIGN);
                let key = (w13.groups, w13.rows, w13.cols, down.rows, down.cols, rows);
                if seen.insert(key) {
                    Self::warm_fp8_deepgemm_grouped_pair(&self.ctx, w13, down, rows)?;
                    min_rows = min_rows.min(rows);
                    max_rows = max_rows.max(rows);
                    warmed += 2;
                }
            }
        }
        if warmed > 0 {
            self.ctx.sync()?;
        }
        if warmed == 0 {
            min_rows = 0;
        }
        Ok((warmed, self.max_seq_len.min(2048), min_rows, max_rows))
    }

    fn warm_fp8_deepgemm_grouped_pair(
        ctx: &DeviceContext,
        w13: &crate::loader::MoeFp8ExpertGroup,
        down: &crate::loader::MoeFp8ExpertGroup,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            w13.groups == down.groups && w13.cols == down.rows && w13.rows == 2 * down.cols,
            "Qwen FP8 grouped DeepGEMM warm shape mismatch: w13={}x{} g={} down={}x{} g={}",
            w13.rows,
            w13.cols,
            w13.groups,
            down.rows,
            down.cols,
            down.groups
        );
        ensure!(
            rows.is_multiple_of(DEEPGEMM_CONTIG_ALIGN),
            "Qwen FP8 grouped DeepGEMM warm rows {rows} not aligned to {DEEPGEMM_CONTIG_ALIGN}"
        );
        cuda_moe::dsv4_deepgemm_native_preflight()?;

        let hidden = w13.cols;
        let intermediate = down.cols;
        let scale_stride_m = rows.div_ceil(4) * 4;
        let hidden_scale_cols = hidden.div_ceil(128);
        let inter_scale_cols = intermediate.div_ceil(128);
        let input_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(rows * hidden)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm input alloc failed: {e}"))?;
        let input_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * hidden_scale_cols)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm input scale alloc failed: {e}"))?;
        let w13_out = ctx
            .stream
            .alloc_zeros::<bf16>(rows * w13.rows)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm w13 output alloc failed: {e}"))?;
        let act_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(rows * intermediate)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm act alloc failed: {e}"))?;
        let act_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * inter_scale_cols)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm act scale alloc failed: {e}"))?;
        let out = ctx
            .stream
            .alloc_zeros::<bf16>(rows * hidden)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm output alloc failed: {e}"))?;
        let m_indices = ctx
            .stream
            .alloc_zeros::<i32>(rows)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm m_indices alloc failed: {e}"))?;
        let stream = ctx.stream.cu_stream();

        unsafe {
            cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                cache_ptr(&input_fp8, ctx),
                cache_ptr(&input_scales, ctx),
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                cache_ptr(&w13_out, ctx),
                cache_ptr(&m_indices, ctx),
                w13.groups,
                rows,
                w13.rows,
                hidden,
                scale_stride_m,
                DEEPGEMM_CONTIG_ALIGN,
                stream,
            )?;
            cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                cache_ptr(&act_fp8, ctx),
                cache_ptr(&act_scales, ctx),
                cache_ptr(&down.weight, ctx),
                cache_ptr(&down.scales, ctx),
                cache_ptr(&out, ctx),
                cache_ptr(&m_indices, ctx),
                down.groups,
                rows,
                hidden,
                intermediate,
                scale_stride_m,
                DEEPGEMM_CONTIG_ALIGN,
                stream,
            )?;
        }
        Ok(())
    }

    /// Per-slot device-memory cost (bytes) at this rank's local shard widths:
    /// K+V contiguous full-attn caches + gated-delta recurrent state + conv
    /// rings. Mirrors exactly what [`Self::new_slot_state`] allocates per slot.
    fn per_slot_kv_bytes(&self) -> (usize, usize, usize, usize) {
        let c = &self.config;
        let num_full = c.num_full_attention_layers();
        let num_linear = c.num_hidden_layers - num_full;
        let bf16 = std::mem::size_of::<half::bf16>();
        let f32sz = std::mem::size_of::<f32>();
        // K and V contiguous caches: num_full × (max_seq_len × kv_dim) bf16, ×2.
        let kv_bytes = num_full
            .saturating_mul(self.max_seq_len)
            .saturating_mul(self.local_full_attn_kv_dim())
            .saturating_mul(2)
            .saturating_mul(bf16);
        // Gated-delta recurrent state: num_linear × (Vh × Kd × Vd) f32.
        let gdr_len = self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let gdr_bytes = num_linear.saturating_mul(gdr_len).saturating_mul(f32sz);
        // Conv1d rings: num_linear × (qkv_dim × (kernel-1)) bf16.
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        let conv_bytes = num_linear.saturating_mul(conv_len).saturating_mul(bf16);
        let per_slot = kv_bytes
            .saturating_add(gdr_bytes)
            .saturating_add(conv_bytes);
        (per_slot, kv_bytes, gdr_bytes, conv_bytes)
    }

    /// Clamp `requested` slots to what post-weights free VRAM affords — the
    /// dynamic KV-memory budget Qwen3.5/3.6 previously lacked (requested slots
    /// were admitted as-is → OOM at large max_seq_len, the #60 failure class).
    /// Unified with DSv4 through the infer-seam budget kernel; the affordable
    /// count is NCCL min-reduced for TP-consistent slot counts. Call AFTER
    /// weights load so `mem_get_info().free` already excludes them.
    pub(crate) fn kv_budget_num_slots(&self, requested: usize) -> Result<usize> {
        const MEM_FRACTION: f64 = 0.9;
        let (per_slot, kv_bytes, gdr_bytes, conv_bytes) = self.per_slot_kv_bytes();
        let affordable_local: i32 = match cudarc::driver::result::mem_get_info() {
            Ok((free, _total)) => {
                // Same neutral kernel as DSv4: floor(free × fraction) / per_slot.
                let budget = infer_seam::SlotBudget::from_free(free, MEM_FRACTION, 0, per_slot);
                log::info!(
                    "Qwen3.5 KV budget: free {}MB, per_slot {}MB (K+V {}MB + gdr {}MB + conv {}MB)",
                    free >> 20,
                    per_slot >> 20,
                    kv_bytes >> 20,
                    gdr_bytes >> 20,
                    conv_bytes >> 20,
                );
                budget
                    .affordable()
                    .map_or(i32::MAX, |n| i32::try_from(n).unwrap_or(i32::MAX))
            }
            // Can't query (no active context / driver error) → don't bind the
            // min; the other ranks' budgets still apply.
            Err(_) => i32::MAX,
        };
        let affordable =
            self.tp
                .all_reduce_min_scalar_i32(&self.ctx, affordable_local)? as usize;
        // Reject-below-fixed guard (parity with Metal's fits_fixed + DSv4): a
        // cross-rank-min affordable of 0 means post-weights free VRAM cannot
        // hold even one slot at this max_seq_len. Fail closed uniformly
        // (lockstep-safe — same reduced scalar on every rank) instead of the
        // former `max(1)` admitting one slot and OOMing at slot allocation.
        anyhow::ensure!(
            affordable > 0,
            "Qwen3.5 KV budget rejected startup: post-weights free VRAM affords 0 slots at \
             max_seq_len {} (per_slot ~{}MB exceeds {MEM_FRACTION} of free). Lower --total-pages \
             or free VRAM.",
            self.max_seq_len,
            per_slot >> 20,
        );
        let (planned, clamped) = infer_seam::clamp_to_affordable(requested, affordable);
        if clamped {
            log::warn!(
                "Qwen3.5 KV budget: requested {requested} slots × ~{}MB/slot exceeds the \
                 cross-rank-min affordable {affordable} (local affordable {affordable_local}, \
                 {MEM_FRACTION} of post-weights free); clamping num_slots to {affordable}. \
                 Lower --total-pages (max_seq_len {}) to raise concurrency.",
                per_slot >> 20,
                self.max_seq_len,
            );
        }
        Ok(planned)
    }

    /// Load a BF16 Qwen3.5/3.6 hybrid dense-or-MoE checkpoint, resolving the TP
    /// runtime from the environment (single-GPU when no TP env is set).
    ///
    /// `max_seq_len` sizes the per-slot full-attn contiguous K/V cache.
    pub(crate) fn from_safetensors(model_path: &Path, max_seq_len: usize) -> Result<Self> {
        let tp = crate::loader::build_tp_runtime()?;
        Self::from_safetensors_with_tp(model_path, max_seq_len, tp)
    }

    /// Load with an explicit [`crate::tp::TpRuntime`] (tests inject a single-GPU
    /// runtime — mirrors the dense loader's `from_safetensors_with_tp`).
    pub(crate) fn from_safetensors_with_tp(
        model_path: &Path,
        max_seq_len: usize,
        tp: crate::tp::TpRuntime,
    ) -> Result<Self> {
        let total_t0 = Instant::now();
        let config_t0 = Instant::now();
        let m = Qwen35Config::from_model_dir(model_path)
            .map_err(|e| anyhow!("load Qwen3.5 config from {}: {e}", model_path.display()))?;
        validate_qwen35_cuda_config(&m)?;
        qwen35_startup_log(
            "config",
            config_t0,
            format_args!(
                "layers={} hidden={} moe={} model_path={}",
                m.num_hidden_layers,
                m.hidden_size,
                m.is_moe(),
                model_path.display()
            ),
        );
        // Full attention here is the GATED q_proj variant (Qwen3.5/3.6); the
        // prep+gate kernels assume it. Vanilla un-gated Qwen3 would need
        // the dense path, not this loader.
        ensure!(
            m.full_attn_gated,
            "clean CUDA Qwen3.5 hybrid path expects the gated full-attention q_proj \
             (Qwen3.5/3.6); un-gated Qwen3 uses from_qwen3_bf16_safetensors"
        );
        ensure!(
            m.rope_scaling.is_none(),
            "Qwen3.5 rope_scaling is set but the YaRN bridge is not wired into the \
             clean hybrid RoPE precompute; refusing to silently drop it (pod follow-up)"
        );
        ensure!(
            max_seq_len > 0,
            "Qwen3.5 hybrid model requires a non-zero KV cache budget"
        );

        let tp_cfg = *tp.config();
        let world = tp_cfg.world_size;
        // Per-rank GQA head counts. `head_shard` requires both counts divide the
        // world size (kv_heads=2 caps Qwen3.6-35B at TP∈{1,2}), keeping every
        // rank's attention shape uniform — the full-attn kernels and the all-reduce
        // both rely on it.
        let (local_q_heads, local_kv_heads) = if tp_cfg.is_single() {
            (m.num_attention_heads, m.num_key_value_heads)
        } else {
            infer_topo::head_shard(m.num_attention_heads, m.num_key_value_heads, &tp_cfg)
                .map_err(|e| anyhow!("Qwen3.5 TP full-attention head shard failed: {e}"))?
        };
        // Gated-delta head counts. Both must divide the world size so a
        // contiguous head-major block shard preserves the v-per-k grouping
        // (gated_delta_rule.cu maps k_head = v_head * Kh / Vh): each rank's
        // v-head range then reads exactly its own k-head range.
        ensure!(
            m.linear_num_key_heads.is_multiple_of(world),
            "Qwen3.5 TP: linear_num_key_heads ({}) not divisible by world_size ({world})",
            m.linear_num_key_heads
        );
        ensure!(
            m.linear_num_value_heads.is_multiple_of(world),
            "Qwen3.5 TP: linear_num_value_heads ({}) not divisible by world_size ({world})",
            m.linear_num_value_heads
        );
        let local_linear_k_heads = m.linear_num_key_heads / world;
        let local_linear_v_heads = m.linear_num_value_heads / world;
        // Shared expert is column/row-sharded like a dense MLP (its partial
        // joins the routed partial in one post-MoE all-reduce).
        ensure!(
            m.shared_expert_intermediate_size.is_multiple_of(world),
            "Qwen3.5 TP: shared_expert_intermediate_size ({}) not divisible by world_size ({world})",
            m.shared_expert_intermediate_size
        );
        // Dense-MLP layers (mlp_only_layers / sparse-step gaps) shard their
        // intermediate dim; only constrain it when such a layer exists.
        if (0..m.num_hidden_layers).any(|i| !m.is_moe_layer(i)) {
            ensure!(
                m.intermediate_size.is_multiple_of(world),
                "Qwen3.5 TP: dense intermediate_size ({}) not divisible by world_size ({world})",
                m.intermediate_size
            );
        }

        let moe_config = if m.is_moe() {
            Some(crate::moe_config::moe_config_from_qwen35(&m)?)
        } else {
            None
        };
        // EP mirrors TP for MoE: each rank owns `num_experts / world` whole
        // experts (`ExpertSplit::new` rejects an indivisible expert count
        // loudly). Dense Qwen3.5 has no expert-owned buffers; keep an inert
        // split so the struct layout stays uniform.
        let split = if !m.is_moe() {
            ExpertSplit::single(0)
        } else if tp_cfg.is_single() {
            ExpertSplit::single(m.num_experts)
        } else {
            ExpertSplit::new(m.num_experts, world, tp_cfg.rank)
                .map_err(|e| anyhow!("Qwen3.5 TP expert split: {e}"))?
        };

        let loader_t0 = Instant::now();
        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;
        qwen35_startup_log("ctx_loader", loader_t0, format_args!(""));

        let embed_t0 = Instant::now();
        let embed_tokens = loader.load_matrix(&ctx, m.embed_tokens_tensor_name())?;
        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, m.lm_head_tensor_name())?)
        };
        qwen35_startup_log(
            "embeddings",
            embed_t0,
            format_args!("tie_word_embeddings={}", m.tie_word_embeddings),
        );

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for layer_idx in 0..m.num_hidden_layers {
            let layer_t0 = Instant::now();
            let names = m.layer_tensor_names(layer_idx);
            let attn_t0 = Instant::now();
            let attn = match &names.attention {
                // Single GPU: full tensors, byte-identical to the pre-TP path.
                Qwen35AttentionTensorNames::Full(full) if tp_cfg.is_single() => {
                    Qwen35Attn::Full(Box::new(FullAttn {
                        q_proj: loader.load_matrix_quant_aware(&ctx, &full.q_proj)?,
                        k_proj: loader.load_matrix_quant_aware(&ctx, &full.k_proj)?,
                        v_proj: loader.load_matrix_quant_aware(&ctx, &full.v_proj)?,
                        o_proj: loader.load_matrix_quant_aware(&ctx, &full.o_proj)?,
                        q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                        k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                    }))
                }
                Qwen35AttentionTensorNames::Full(full) => Qwen35Attn::Full(Box::new(FullAttn {
                    // The GATED q_proj interleaves [query(HD); gate(HD)] PER
                    // HEAD (prefill_attention_hd256.cu reads q at
                    // `head*2*HD + d`, the gate kernel at `head*2*HD + HD + d`),
                    // so a whole-head slice with per-head row block 2*head_dim
                    // carries each head's query rows AND its matching gate rows.
                    q_proj: loader.load_qkv_head_sharded_quant_aware(
                        &ctx,
                        &full.q_proj,
                        local_q_heads,
                        m.head_dim * 2,
                        &tp_cfg,
                    )?,
                    k_proj: loader.load_qkv_head_sharded_quant_aware(
                        &ctx,
                        &full.k_proj,
                        local_kv_heads,
                        m.head_dim,
                        &tp_cfg,
                    )?,
                    v_proj: loader.load_qkv_head_sharded_quant_aware(
                        &ctx,
                        &full.v_proj,
                        local_kv_heads,
                        m.head_dim,
                        &tp_cfg,
                    )?,
                    // Row-parallel: each rank holds the o_proj input columns of
                    // its own heads; the forward all-reduces the partial sums.
                    o_proj: loader.load_matrix_sharded_quant_aware(
                        &ctx,
                        &full.o_proj,
                        infer_topo::ParallelLinearKind::Row,
                        &tp_cfg,
                    )?,
                    // q/k_norm are `[head_dim]`, broadcast across heads by the
                    // full-attention prep kernel — replicated.
                    q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                })),
                Qwen35AttentionTensorNames::Linear(lin) if tp_cfg.is_single() => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        in_proj_qkv: loader.load_matrix_quant_aware(&ctx, &lin.in_proj_qkv)?,
                        in_proj_z: loader.load_matrix_quant_aware(&ctx, &lin.in_proj_z)?,
                        in_proj_b: loader.load_matrix(&ctx, &lin.in_proj_b)?,
                        in_proj_a: loader.load_matrix(&ctx, &lin.in_proj_a)?,
                        conv1d_weight: loader.load_conv1d_vec(&ctx, &lin.conv1d_weight)?,
                        dt_bias: loader.load_vec_any(&ctx, &lin.dt_bias)?,
                        a_log: loader.load_f32_vec(&ctx, &lin.a_log)?,
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        out_proj: loader.load_matrix_quant_aware(&ctx, &lin.out_proj)?,
                    }))
                }
                Qwen35AttentionTensorNames::Linear(lin) => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        // Fused [q | k | v] blocks: shard EACH block on whole-head
                        // boundaries and re-stack this rank's three slices (a flat
                        // column shard would cut across the block boundaries).
                        in_proj_qkv: load_linear_qkv_sharded(
                            &loader,
                            &ctx,
                            &lin.in_proj_qkv,
                            &m,
                            &tp_cfg,
                        )?,
                        // z gate is v-head-major `[Vh*Vd]` (rms_norm_gated reads
                        // the gate at `head*Vd + d`).
                        in_proj_z: loader.load_qkv_head_sharded_quant_aware(
                            &ctx,
                            &lin.in_proj_z,
                            local_linear_v_heads,
                            m.linear_value_head_dim,
                            &tp_cfg,
                        )?,
                        // b/a are ONE SCALAR PER V HEAD (gated_delta_rule.cu reads
                        // `b_proj[token*Vh + v_head]`) → per-head row count 1.
                        in_proj_b: loader.load_qkv_head_sharded(
                            &ctx,
                            &lin.in_proj_b,
                            local_linear_v_heads,
                            1,
                            &tp_cfg,
                        )?,
                        in_proj_a: loader.load_qkv_head_sharded(
                            &ctx,
                            &lin.in_proj_a,
                            local_linear_v_heads,
                            1,
                            &tp_cfg,
                        )?,
                        // Depthwise conv: channel rows mirror the qkv block shard.
                        conv1d_weight: load_conv1d_sharded(
                            &loader,
                            &ctx,
                            &lin.conv1d_weight,
                            &m,
                            &tp_cfg,
                        )?,
                        // Per-v-head vectors `[Vh]`.
                        dt_bias: load_v_head_vec_sharded(
                            &loader,
                            &ctx,
                            &lin.dt_bias,
                            m.linear_num_value_heads,
                            &tp_cfg,
                        )?,
                        a_log: load_v_head_f32_sharded(
                            &loader,
                            &ctx,
                            &lin.a_log,
                            m.linear_num_value_heads,
                            &tp_cfg,
                        )?,
                        // Gated-norm scale is `[Vd]`, broadcast across heads by
                        // rms_norm_gated (norm.cu `weight[tid]`) — replicated,
                        // matching the qwen35-spec Shard contract.
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        // Row-parallel: input columns follow this rank's v heads;
                        // the forward all-reduces the partial sums.
                        out_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &lin.out_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }))
                }
            };
            qwen35_startup_log(
                "layer.attn",
                attn_t0,
                format_args!("layer={layer_idx} type={:?}", m.layer_types[layer_idx]),
            );

            let ffn_t0 = Instant::now();
            let (mlp, moe) = if m.is_moe_layer(layer_idx) {
                let moe = loader.load_moe_layer_experts(
                    &ctx,
                    &names.common.moe_tensor_names(),
                    &split,
                    &tp_cfg,
                    m.moe_intermediate_size,
                    m.hidden_size,
                )?;
                (None, Some(moe))
            } else if tp_cfg.is_single() {
                (
                    Some(DenseMlp {
                        gate_proj: loader
                            .load_matrix_quant_aware(&ctx, &names.common.mlp_gate_proj)?,
                        up_proj: loader.load_matrix_quant_aware(&ctx, &names.common.mlp_up_proj)?,
                        down_proj: loader
                            .load_matrix_quant_aware(&ctx, &names.common.mlp_down_proj)?,
                    }),
                    None,
                )
            } else {
                (
                    Some(DenseMlp {
                        gate_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &names.common.mlp_gate_proj,
                            infer_topo::ParallelLinearKind::Column,
                            &tp_cfg,
                        )?,
                        up_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &names.common.mlp_up_proj,
                            infer_topo::ParallelLinearKind::Column,
                            &tp_cfg,
                        )?,
                        down_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &names.common.mlp_down_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }),
                    None,
                )
            };
            qwen35_startup_log(
                "layer.ffn",
                ffn_t0,
                format_args!("layer={layer_idx} moe={}", m.is_moe_layer(layer_idx)),
            );

            layers.push(Qwen35Layer {
                input_layernorm: loader.load_vec(&ctx, &names.common.input_layernorm)?,
                attn,
                post_attention_layernorm: loader
                    .load_vec(&ctx, &names.common.post_attention_layernorm)?,
                mlp,
                moe,
            });
            qwen35_startup_log(
                "layer.total",
                layer_t0,
                format_args!("layer={layer_idx} moe={}", m.is_moe_layer(layer_idx)),
            );
        }
        let tail_t0 = Instant::now();
        let norm = loader.load_vec(&ctx, m.norm_tensor_name())?;

        let rope_len = m
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        ensure!(
            max_seq_len <= rope_len,
            "Qwen3.5 max_seq_len ({max_seq_len}) exceeds the RoPE cache length ({rope_len}); \
             positions beyond the table would read out of bounds"
        );
        // PARTIAL RoPE: the table must be built over `rotary_dim` (= head_dim ×
        // partial_rotary_factor, 64 on Qwen3.6), not head_dim — the prep
        // kernel indexes `cos_cache[pos * rotary_dim + d]` and expects inv_freq
        // computed over rotary_dim dims (`precompute_rope` is generic over its
        // dim arg and emits the half-duplicated stride-dim layout it reads).
        let (cos_cache, sin_cache) =
            crate::ops::precompute_rope(&ctx, m.rotary_dim, rope_len, m.rope_theta, None)?;
        ctx.sync()?;
        qwen35_startup_log(
            "tail_norm_rope_sync",
            tail_t0,
            format_args!("rope_len={rope_len} max_seq_len={max_seq_len}"),
        );
        qwen35_startup_log(
            "total",
            total_t0,
            format_args!("layers={} max_seq_len={max_seq_len}", m.num_hidden_layers),
        );

        Ok(Self {
            ctx,
            config: m,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            moe_config,
            tp,
            local_q_heads,
            local_kv_heads,
            local_linear_k_heads,
            local_linear_v_heads,
            expert_split: split,
            max_seq_len,
            offloaded: None,
            lora_base: HashMap::new(),
        })
    }

    /// Move every device weight buffer to host RAM and free the VRAM (OPD teacher
    /// time-share). Idempotent: a no-op (returns 0) if already offloaded.
    ///
    /// Returns the total device VRAM (bytes) freed. After this returns the model
    /// must NOT be forwarded through until [`Qwen35Model::reload_engine_weights`].
    /// The freed weight blocks return to the device's async memory pool, which the
    /// co-resident OPD student/autograd allocator reuses for the backward pass —
    /// the VRAM headroom the time-share buys. Per-slot KV / recurrent state
    /// ([`Qwen35SlotState`]) is owned by the executor and untouched here.
    ///
    /// MoE expert weight offload is NOT supported (OPD is dense-only): a MoE
    /// layer bails so the caller does not silently keep ~19 GB of expert VRAM
    /// resident while believing the engine is offloaded.
    pub(crate) fn offload_engine_weights(&mut self) -> Result<usize> {
        if self.offloaded.is_some() {
            return Ok(0);
        }
        // PREFLIGHT (before ANY mutation): MoE weight offload is unsupported.
        // Bailing mid-loop after embed/lm_head/norm are already swapped to
        // placeholders — with `self.offloaded` still unset — would make reload a
        // no-op and the next forward run on corrupted placeholder weights.
        for layer in &self.layers {
            ensure!(
                layer.moe.is_none(),
                "Qwen3.6 MoE weight offload is not supported (OPD teacher time-share is dense-only)"
            );
        }
        let ctx = self.ctx.clone();
        // Drain ALL in-flight GPU work before snapshotting. The OPD step has
        // co-resident allocators (infer-teacher + train autograd) sharing one
        // device/pool on separate streams; a full synchronize quiesces every
        // stream so the D2H snapshot and the subsequent block frees do not race
        // other-stream allocations from the shared async pool.
        ctx.sync()?;
        let mut freed = 0usize;

        let embed_tokens = self.embed_tokens.offload_to_host(&ctx)?;
        freed += embed_tokens.freed_bytes();
        let lm_head = match self.lm_head.as_mut() {
            Some(head) => {
                let snap = head.offload_to_host(&ctx)?;
                freed += snap.freed_bytes();
                Some(snap)
            }
            None => None,
        };
        let (norm, norm_n) = self.norm.offload_to_host(&ctx)?;
        freed += norm_n;

        let mut blocks = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            // (MoE guard hoisted to the preflight above — no mid-loop bail.)
            let (input_layernorm, in_ln_n) = layer.input_layernorm.offload_to_host(&ctx)?;
            let (post_attention_layernorm, post_ln_n) =
                layer.post_attention_layernorm.offload_to_host(&ctx)?;
            freed += in_ln_n + post_ln_n;

            let mlp = match layer.mlp.as_mut() {
                Some(dense) => {
                    let gate_proj = dense.gate_proj.offload_to_host(&ctx)?;
                    let up_proj = dense.up_proj.offload_to_host(&ctx)?;
                    let down_proj = dense.down_proj.offload_to_host(&ctx)?;
                    freed +=
                        gate_proj.freed_bytes() + up_proj.freed_bytes() + down_proj.freed_bytes();
                    Some(OffloadedDenseMlp {
                        gate_proj,
                        up_proj,
                        down_proj,
                    })
                }
                None => None,
            };

            let attn = match &mut layer.attn {
                Qwen35Attn::Full(full) => {
                    let q_proj = full.q_proj.offload_to_host(&ctx)?;
                    let k_proj = full.k_proj.offload_to_host(&ctx)?;
                    let v_proj = full.v_proj.offload_to_host(&ctx)?;
                    let o_proj = full.o_proj.offload_to_host(&ctx)?;
                    let (q_norm, qn) = full.q_norm.offload_to_host(&ctx)?;
                    let (k_norm, kn) = full.k_norm.offload_to_host(&ctx)?;
                    freed += q_proj.freed_bytes()
                        + k_proj.freed_bytes()
                        + v_proj.freed_bytes()
                        + o_proj.freed_bytes()
                        + qn
                        + kn;
                    OffloadedAttn::Full(Box::new(OffloadedFullAttn {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    }))
                }
                Qwen35Attn::Linear(lin) => {
                    let in_proj_qkv = lin.in_proj_qkv.offload_to_host(&ctx)?;
                    let in_proj_z = lin.in_proj_z.offload_to_host(&ctx)?;
                    let in_proj_b = lin.in_proj_b.offload_to_host(&ctx)?;
                    let in_proj_a = lin.in_proj_a.offload_to_host(&ctx)?;
                    let (conv1d_weight, conv_n) = lin.conv1d_weight.offload_to_host(&ctx)?;
                    let (dt_bias, dt_n) = lin.dt_bias.offload_to_host(&ctx)?;
                    let (a_log, al) = offload_raw_slice(&ctx, &mut lin.a_log)?;
                    let (norm_weight, nw) = offload_raw_slice(&ctx, &mut lin.norm_weight)?;
                    let out_proj = lin.out_proj.offload_to_host(&ctx)?;
                    freed += in_proj_qkv.freed_bytes()
                        + in_proj_z.freed_bytes()
                        + in_proj_b.freed_bytes()
                        + in_proj_a.freed_bytes()
                        + out_proj.freed_bytes()
                        + conv_n
                        + dt_n
                        + al
                        + nw;
                    OffloadedAttn::Linear(Box::new(OffloadedLinearAttn {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm_weight,
                        out_proj,
                    }))
                }
            };

            blocks.push(OffloadedBlock {
                input_layernorm,
                post_attention_layernorm,
                attn,
                mlp,
            });
        }

        // Quiesce again after the block frees so reload (or a co-resident
        // backward) sees a settled pool. We deliberately do NOT trim the pool to
        // the OS: the freed blocks must stay in the shared async pool for the
        // student backward to reuse.
        ctx.sync()?;

        self.offloaded = Some(Box::new(OffloadedWeights {
            embed_tokens,
            lm_head,
            norm,
            blocks,
        }));
        Ok(freed)
    }

    /// Restore every device weight buffer from the host snapshot, re-allocating
    /// VRAM (OPD teacher time-share). Idempotent: a no-op if not offloaded.
    pub(crate) fn reload_engine_weights(&mut self) -> Result<()> {
        let Some(snapshot) = self.offloaded.take() else {
            return Ok(());
        };
        let ctx = self.ctx.clone();
        // Quiesce the whole device before re-allocating weight VRAM so the H2D
        // restores do not race the train/optimizer allocations still draining
        // from the shared async pool (see offload note).
        ctx.sync()?;
        let OffloadedWeights {
            embed_tokens,
            lm_head,
            norm,
            blocks,
        } = *snapshot;

        self.embed_tokens.reload_from_host(&ctx, &embed_tokens)?;
        match (self.lm_head.as_mut(), &lm_head) {
            (Some(head), Some(snap)) => head.reload_from_host(&ctx, snap)?,
            (None, None) => {}
            _ => anyhow::bail!("offload/reload lm_head presence mismatch"),
        }
        self.norm.reload_from_host(&ctx, &norm)?;

        ensure!(
            blocks.len() == self.layers.len(),
            "offload/reload layer count mismatch: snapshot {} vs model {}",
            blocks.len(),
            self.layers.len()
        );
        for (layer, block) in self.layers.iter_mut().zip(blocks) {
            layer
                .input_layernorm
                .reload_from_host(&ctx, &block.input_layernorm)?;
            layer
                .post_attention_layernorm
                .reload_from_host(&ctx, &block.post_attention_layernorm)?;

            match (layer.mlp.as_mut(), block.mlp) {
                (Some(dense), Some(snap)) => {
                    dense.gate_proj.reload_from_host(&ctx, &snap.gate_proj)?;
                    dense.up_proj.reload_from_host(&ctx, &snap.up_proj)?;
                    dense.down_proj.reload_from_host(&ctx, &snap.down_proj)?;
                }
                (None, None) => {
                    anyhow::bail!("Qwen3.6 MoE weight reload is not supported (OPD is dense-only)")
                }
                _ => anyhow::bail!("offload/reload dense-MLP presence mismatch"),
            }

            match (&mut layer.attn, block.attn) {
                (Qwen35Attn::Full(full), OffloadedAttn::Full(snap)) => {
                    let OffloadedFullAttn {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    } = *snap;
                    full.q_proj.reload_from_host(&ctx, &q_proj)?;
                    full.k_proj.reload_from_host(&ctx, &k_proj)?;
                    full.v_proj.reload_from_host(&ctx, &v_proj)?;
                    full.o_proj.reload_from_host(&ctx, &o_proj)?;
                    full.q_norm.reload_from_host(&ctx, &q_norm)?;
                    full.k_norm.reload_from_host(&ctx, &k_norm)?;
                }
                (Qwen35Attn::Linear(lin), OffloadedAttn::Linear(snap)) => {
                    let OffloadedLinearAttn {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm_weight,
                        out_proj,
                    } = *snap;
                    lin.in_proj_qkv.reload_from_host(&ctx, &in_proj_qkv)?;
                    lin.in_proj_z.reload_from_host(&ctx, &in_proj_z)?;
                    lin.in_proj_b.reload_from_host(&ctx, &in_proj_b)?;
                    lin.in_proj_a.reload_from_host(&ctx, &in_proj_a)?;
                    lin.conv1d_weight.reload_from_host(&ctx, &conv1d_weight)?;
                    lin.dt_bias.reload_from_host(&ctx, &dt_bias)?;
                    reload_raw_slice(&ctx, &mut lin.a_log, &a_log)?;
                    reload_raw_slice(&ctx, &mut lin.norm_weight, &norm_weight)?;
                    lin.out_proj.reload_from_host(&ctx, &out_proj)?;
                }
                _ => anyhow::bail!("offload/reload attention-kind mismatch"),
            }
        }
        ctx.sync()?;
        Ok(())
    }

    /// Shared layer stack for [`Self::forward_tokens`] /
    /// [`Self::forward_token_logits_full`]: stages the per-step host inputs
    /// (token ids + start_pos), runs [`Self::forward_hidden_staged`], and
    /// advances `slot.seq_len`. Leaves the final hidden states in `ws.hidden`
    /// (`[hidden, seq_len]`).
    fn forward_hidden(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<()> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3.5 hybrid forward requires at least one token"
        );
        ensure!(
            slot.seq_len == start_pos,
            "Qwen3.5 hybrid slot seq_len {} != start_pos {start_pos} (uncached full-prefix \
             path requires contiguous appends)",
            slot.seq_len
        );
        let seq_len = tokens.len();
        ensure!(
            start_pos + seq_len <= self.max_seq_len,
            "Qwen3.5 hybrid sequence {} exceeds KV cache budget {}",
            start_pos + seq_len,
            self.max_seq_len
        );
        self.stage_step_inputs(ws, tokens, start_pos)?;
        self.forward_hidden_staged(slot, ws, seq_len, start_pos)?;
        slot.seq_len += seq_len;
        Ok(())
    }

    /// Stage the per-step HOST inputs (token ids, start_pos) into their
    /// persistent device slots. This is the ONLY H2D traffic of a decode step;
    /// the captured decode graph runs it OUTSIDE the capture/replay (the dense
    /// `stage1_write` pattern), so the graph body below is a pure GPU kernel
    /// sequence. Returns the `(token_ids, start_pos)` device addresses for the
    /// graph-bake fingerprint (length-matched slot reuse keeps them stable; a
    /// change means the captured graph reads stale memory and must recapture).
    pub(crate) fn stage_step_inputs(
        &self,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(u64, u64)> {
        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = ws.token_ids.upload(&self.ctx, &token_ids_host)?;
        let (token_ids_ptr, _g0) = token_ids.device_ptr(&self.ctx.stream);
        let start_pos_dev = ws.start_pos.upload(&self.ctx, &[start_pos as i32])?;
        let (start_pos_ptr, _g1) = start_pos_dev.device_ptr(&self.ctx.stream);
        Ok((token_ids_ptr, start_pos_ptr))
    }

    /// The pure-GPU layer stack over already-staged inputs: embeds the staged
    /// token ids, runs every layer over the workspace buffers, and advances
    /// the recurrent/conv/KV device state in place. Does NOT advance
    /// `slot.seq_len` and performs NO H2D/D2H/sync — at `seq_len == 1` this is
    /// the CUDA-graph-capturable decode body (every per-step scalar is read
    /// from the staged device buffers; see
    /// [`Self::forward_decode_step_captured`] for the capture-safety table).
    ///
    /// `start_pos` (host) is consumed only by the `seq_len > 1` prefill
    /// attention launch; the `seq_len == 1` path reads the position from the
    /// staged device buffer.
    fn forward_hidden_staged(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;

        // Destructure once: each binding borrows its own workspace field, so
        // the residual-stream buffers, the per-block scratch, and the MoE
        // scratch stay simultaneously borrowable across the layer loop.
        let Qwen35Workspace {
            token_ids,
            start_pos: start_pos_slot,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            ..
        } = ws;

        // Shape-matched re-gets return the SAME buffers `stage_step_inputs`
        // just wrote (no H2D here — a mismatch would mean staging was skipped,
        // which the two call paths above make impossible).
        let token_ids = &*token_ids.get(&self.ctx, seq_len)?;
        let start_pos_dev = &*start_pos_slot.get(&self.ctx, 1)?;

        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        qwen35_profile(&self.ctx, "qwen/embedding", None, seq_len, || {
            embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)
        })?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, seq_len)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, seq_len)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, seq_len)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            qwen35_profile(
                &self.ctx,
                "qwen/input_norm",
                Some(layer_idx),
                seq_len,
                || rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed),
            )?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    qwen35_profile(
                        &self.ctx,
                        "qwen/full_attention",
                        Some(layer_idx),
                        seq_len,
                        || {
                            self.full_attention(
                                full_attn,
                                normed,
                                slot,
                                full_idx,
                                start_pos,
                                start_pos_dev,
                                full,
                                attn_out,
                            )
                        },
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    qwen35_profile(
                        &self.ctx,
                        "qwen/linear_attention",
                        Some(layer_idx),
                        seq_len,
                        || self.linear_attention(lin, normed, slot, linear_idx, linear, attn_out),
                    )?;
                    linear_idx += 1;
                }
            }

            // Post-attn residual add + post_attention_layernorm via the
            // `add_batch` + `rms_norm_offset` pair (`hidden_mid`/`normed`).
            qwen35_profile(
                &self.ctx,
                "qwen/post_attn_norm",
                Some(layer_idx),
                seq_len,
                || {
                    add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
                    rms_norm_offset(
                        &self.ctx,
                        hidden_mid,
                        &layer.post_attention_layernorm,
                        eps,
                        normed,
                    )
                },
            )?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                qwen35_profile(&self.ctx, "qwen/moe_ffn", Some(layer_idx), seq_len, || {
                    moe_forward_into(
                        &self.ctx,
                        moe_weights,
                        mlp_in,
                        cfg,
                        &self.expert_split,
                        moe,
                        mlp_out,
                    )
                })?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                qwen35_profile(
                    &self.ctx,
                    "qwen/dense_ffn",
                    Some(layer_idx),
                    seq_len,
                    || self.dense_mlp(mlp, mlp_in, dense, mlp_out),
                )?;
            }
            // ONE all-reduce covers the whole FFN partial: the MoE buffer already
            // sums this rank's routed experts (non-local routes contribute zero)
            // + the column/row-sharded shared expert; the dense branch is a
            // row-parallel down_proj partial. No-op on a single GPU.
            qwen35_profile(
                &self.ctx,
                "qwen/ffn_allreduce",
                Some(layer_idx),
                seq_len,
                || self.tp.all_reduce_sum(&self.ctx, mlp_out),
            )?;

            // MLP residual add producing the next layer's residual stream.
            // The post-attn sum lives in `hidden_mid`; add_batch reads
            // hidden_mid/mlp_out and writes `hidden` (whose previous value is
            // dead).
            qwen35_profile(
                &self.ctx,
                "qwen/ffn_residual",
                Some(layer_idx),
                seq_len,
                || add_batch(&self.ctx, hidden_mid, mlp_out, hidden),
            )?;
        }

        Ok(())
    }

    /// Run prefill or decode for one row. `start_pos` is the absolute position of
    /// the first token; `tokens` are the new tokens (whole prompt on prefill, one
    /// token on decode). Advances `slot.seq_len` and the recurrent state. Returns
    /// the next sampled token. `ws` is the executor's persistent forward
    /// workspace (serial forwards share one).
    pub(crate) fn forward_tokens(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        qwen35_profile(&self.ctx, "qwen/forward_hidden", None, tokens.len(), || {
            self.forward_hidden(slot, ws, tokens, start_pos)
        })?;
        qwen35_profile(&self.ctx, "qwen/lm_head", None, tokens.len(), || {
            self.lm_head_logits(ws, tokens.len())
        })?;
        qwen35_profile(&self.ctx, "qwen/sample", None, tokens.len(), || {
            self.sample_workspace_logits(ws, params, position)
        })
    }

    /// Final norm (offset) + LM head on the last token only, into `ws.logits`.
    /// Last stage of the captured decode graph (the capture ends at the logits
    /// buffer, dense-style); sampling stays outside.
    ///
    /// TP invariant: embed/lm_head are replicated and `hidden` is
    /// post-all-reduce (every row-parallel output above was summed), so the
    /// logits — and therefore the sampled token — are identical on every
    /// rank. No rank ever needs to broadcast its sample.
    fn lm_head_logits(&self, ws: &mut Qwen35Workspace, seq_len: usize) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        let Qwen35Workspace {
            hidden,
            last_hidden,
            last_normed,
            logits,
            ..
        } = ws;
        // Shape-matched re-get returns the SAME buffer forward_hidden filled.
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let last_hidden = last_hidden.get(&self.ctx, hidden_size)?;
        copy_row_to_vec(&self.ctx, hidden, seq_len - 1, last_hidden)?;
        let last_normed = last_normed.get(&self.ctx, hidden_size)?;
        rms_norm_offset_vec(&self.ctx, last_hidden, &self.norm, eps, last_normed)?;
        let logits = logits.get(&self.ctx, self.output_projection().rows)?;
        gemv(&self.ctx, self.output_projection(), last_normed, logits)?;
        Ok(())
    }

    /// Sample the next token from `ws.logits` (written by
    /// [`Self::lm_head_logits`] — eagerly or by a decode-graph replay). Greedy
    /// uses the persistent `argmax_out` slot (zero per-token allocations);
    /// non-greedy reads the logits to host. Always OUTSIDE any capture (syncs).
    pub(crate) fn sample_workspace_logits(
        &self,
        ws: &mut Qwen35Workspace,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        let Qwen35Workspace {
            logits, argmax_out, ..
        } = ws;
        let logits = logits.get(&self.ctx, self.output_projection().rows)?;
        let argmax_out = argmax_out.get(&self.ctx, 1)?;
        sample_cuda_token_scratched(&self.ctx, logits, params, position, argmax_out)
    }

    /// Device address of the workspace logits buffer (allocating it at vocab
    /// size if absent) — the decode-graph bake fingerprint's output anchor.
    pub(crate) fn workspace_logits_ptr(&self, ws: &mut Qwen35Workspace) -> Result<u64> {
        let logits = ws.logits.get(&self.ctx, self.output_projection().rows)?;
        let (ptr, _g) = logits.data.device_ptr(&self.ctx.stream);
        Ok(ptr)
    }

    /// Per-slot KV-cache capacity (tokens).
    pub(crate) fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Whole-step captured decode body: embedding → every layer (norms,
    /// full/linear attention, dense/MoE FFN) → final norm → lm_head GEMV,
    /// ending at `ws.logits`. One `seq_len == 1` pass over already-staged
    /// inputs; sampling stays outside (dense pattern). The host-side
    /// `slot.seq_len` advance is the CALLER's job (replay never re-runs host
    /// code), and the per-step scalars (token id, position) live in the staged
    /// device buffers, so ONE capture replays across positions.
    ///
    /// Why this is the big lever (formula, vs the DSv4 whole-step WASH):
    /// Qwen3.5/3.6 B=1 decode measured 24.5 ms/token (40.8 tok/s) against a
    /// ~1.7 ms HBM active-weight floor — ~94% orchestration: ~1,074 kernel
    /// launches per token, each paying serialized host issue (~3-8 us) plus
    /// inter-launch gaps. predicted = 24.5 ms − (host issue + gap reclaim by
    /// graph scheduling) → ~14-19 ms/token (+30-75% tok/s) at TP=1. DSv4's
    /// whole-step graph was wall-neutral because ITS decode is GPU-bound;
    /// this one is host-bound, the opposite regime. License threshold:
    /// ≥ +10% tok/s with needle-gate pass AND replay-reuse evidence (the
    /// capture/replay counters in the executor), per the bench spec.
    ///
    /// Captured-kernel enumeration (capture-safety proof per kernel; the
    /// captured-bodies-allocate-nothing rule). All workspace slots are at
    /// steady decode shapes (allocated by the warm eager run `CudaGraphState`
    /// forces before first capture), so every `get`/`upload_const` inside is a
    /// pure cache hit; the only stream ops recorded are kernels, D2D memcpys,
    /// and device memsets — no H2D, no host callback, no alloc (the
    /// `audit_capturing_graph` host-memcpy census enforces this at capture).
    ///
    /// | # | kernel (per occurrence) | capture-safety justification |
    /// |---|-------------------------|------------------------------|
    /// | 1 | `embedding_batched_cuda` | reads staged `token_ids` device buffer; writes ws.hidden |
    /// | 2 | `rms_norm_batched_offset_cuda` ×2/layer | stateless; fixed ws buffers |
    /// | 3 | `gemm_cuda` (q/k/v/o, qkv/z/b/a/out, gate/up/down, router, shared ×4) | cuBLASLt with load-time workspace + algo cache warmed by the eager warm run (heuristic query happens outside capture; autotune self-suppresses during capture) |
    /// | 4 | `prefill_attention_hd256_prep_cuda` | position read from staged `start_pos` DEVICE buffer (already a device-pointer arg); writes K/V cache rows + q_prepped in place |
    /// | 5 | `nonpaged_prefill_attention_devpos_cuda` | NEW devpos entry: kv walk bounded by `*start_pos_dev` read on device; grid `(heads, 1)` shape-constant |
    /// | 6 | `attention_gate_batch_hd256_cuda` | stateless gate RMW on ws buffers |
    /// | 7 | `conv1d_prefill_cuda` (+ its internal `conv1d_state_update_kernel`) | depthwise conv + ring shift are content-based in-place device writes; each replay advances the ring by one token exactly like eager |
    /// | 8 | `gated_delta_rule_decode_cuda` | recurrent-state advance is a content-based in-place device write; no position arg |
    /// | 9 | `rms_norm_gated_cuda` | stateless; fixed ws buffers |
    /// | 10 | `dsv4_route` + `qwen36_renorm_topk_weights` | device router (gate requires `qwen35_decode_moe_graph_capturable`): all-zero bias table is `upload_const` (warm, no H2D node); writes route indices/weights slots unconditionally |
    /// | 11 | memset(counts), memset(cursors) | `get_zeroed` device memsets — legal graph memset nodes, re-executed per replay (atomicAdd accumulators NEED the per-replay re-zero) |
    /// | 12 | `dsv4_count_local_experts`, `dsv4_exclusive_scan_i32`, `dsv4_pack_local_experts_with_slots` | device counts/offsets/pack; grids sized by `total_routes = top_k` (shape constant, 8); counts live on device |
    /// | 13 | `moe_bf16_grouped_gemm_pair_batch` + `moe_bf16_grouped_gemm_batch` | HAND grouped kernels (hybrid dispatch: R=8 < `QWEN35_DEEPGEMM_MIN_ROUTES`, DeepGEMM JIT never runs at decode); weight-ptr tables are load-time device buffers |
    /// | 14 | `silu_mul_cuda` ×2 (routed + shared) | stateless |
    /// | 15 | `dsv4_scatter_all_route_slots` + `dsv4_combine_route_slot_outputs` | single-GPU (graph gated `tp.is_single()`): scatter writes ALL `top_k` slots (no EP sentinel/zeroed-tail path taken) |
    /// | 16 | `qwen36_add_shared_expert_gated` | stateless RMW over fully-written `mlp_out` |
    /// | 17 | `add_cuda` ×2/layer | stateless residual adds |
    /// | 18 | `memcpy_dtod` (last row), `rms_norm_offset_cuda`, `gemv_cuda` (lm_head) | D2D memcpy node + stateless kernels; capture ends at `ws.logits` |
    ///
    /// NOT in the captured region (stays eager): `SliceSlot::upload` staging
    /// H2D (pre-replay), argmax + D2H + `ctx.sync` sampling tail, the MoE
    /// host-route fallback (gated off), NCCL all-reduce (`tp.is_single()`
    /// only — `TpComm::Single` is a literal `Ok(())`).
    pub(crate) fn forward_decode_step_captured(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        start_pos: usize,
    ) -> Result<()> {
        self.forward_hidden_staged(slot, ws, 1, start_pos)?;
        self.lm_head_logits(ws, 1)
    }

    /// Whether every layer of this model can run the captured decode body —
    /// i.e. each MoE layer's decode step is a pure device-kernel sequence.
    /// Dense-MLP layers are always capturable; MoE layers need the device
    /// router + hand grouped kernels
    /// ([`crate::moe::qwen35_decode_moe_graph_capturable`]).
    pub(crate) fn decode_graph_unsupported_reason(&self) -> Option<&'static str> {
        let has_moe = self.layers.iter().any(|l| l.moe.is_some());
        if !has_moe {
            return None;
        }
        let Some(cfg) = self.moe_config.as_ref() else {
            return Some("MoE layers present but no moe_config");
        };
        if !crate::moe::qwen35_decode_moe_graph_capturable(cfg) {
            return Some(
                "MoE decode is not device-routable (host router fallback active — \
                 ARLE_QWEN35_GPU_ROUTER=0 or non-greedy/grouped routing)",
            );
        }
        None
    }

    /// Run the full forward over `tokens` and return the FULL `[seq_len, vocab]`
    /// logits (every row, not just the last) WITHOUT sampling. Mirrors
    /// [`Self::forward_tokens`]'s layer stack but, instead of slicing the last
    /// row + a single `gemv`, applies the final offset-RMSNorm over the whole
    /// batch and a batched lm-head GEMM, returning the device logits buffer plus
    /// its `[seq_len, vocab]` shape.
    ///
    /// This is the OPD teacher raw-logits path: the distillation loss needs the
    /// teacher's per-position logit distribution over the prompt, so it cannot
    /// use the sampling tail. `slot` carries the per-slot KV + recurrent state
    /// exactly like [`Self::forward_tokens`].
    pub(crate) fn forward_token_logits_full(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(DeviceVec, [usize; 2])> {
        self.forward_hidden(slot, ws, tokens, start_pos)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;

        // Final norm (offset) over the WHOLE batch, then the batched lm-head GEMM
        // produces every row's logits — no last-row slice, no sampling.
        // (TP invariant as in `forward_tokens`: replicated lm_head over
        // post-all-reduce hidden ⇒ identical logits on every rank.)
        let Qwen35Workspace { hidden, normed, .. } = ws;
        // Shape-matched re-gets return the SAME buffers forward_hidden used;
        // rms_norm fully overwrites `normed` before the lm-head GEMM reads it.
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let vocab = self.output_projection().rows;
        // The logits buffer is RETURNED to the OPD caller (ownership leaves the
        // forward), so it stays a per-call allocation — not a workspace slot.
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)?;
        self.ctx.sync()?;

        // `HiddenStates` is a `[vocab, seq_len]` column-batch over a flat device
        // buffer; reinterpret it as a `[seq_len, vocab]` row-major `DeviceVec`
        // (the train OPD bridge reads it as `[1, seq_len, vocab]`).
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_token_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab]))
    }

    /// Per-step student LoRA re-merge (OPD P2).
    ///
    /// Folds a fresh [`StudentLoraUpdate`] into the resident full-attention
    /// q/v projection weights in place. On the first call the pristine base
    /// weight of every touched projection is cached host-side; each call then
    /// recomputes `W = base + (alpha/rank)·(B·A)` from that pristine base — so
    /// re-merging never compounds onto an already-merged weight.
    ///
    /// `A` is `[rank, in]`, `B` is `[out, rank]`, matching the legacy disk
    /// loader / `infer/src/model/qwen35/lora.rs` contract. The forward path
    /// recomputes attention from these resident matrices every step, so the
    /// merged weight is picked up by the next `forward_tokens` automatically.
    pub(crate) fn remerge_student_lora(&mut self, update: StudentLoraUpdate) -> Result<()> {
        ensure!(update.rank > 0, "student LoRA update has rank=0");
        let scale = update.alpha / update.rank as f32;
        let num_layers = self.config.num_hidden_layers;

        for layer in &update.layers {
            let layer_idx = layer.layer_idx;
            ensure!(
                layer_idx < num_layers,
                "student LoRA references layer {layer_idx} but model has {num_layers} layers"
            );
            ensure!(
                self.config.layer_types[layer_idx] == LayerType::FullAttention,
                "student LoRA layer {layer_idx} is not a full-attention layer; \
                 the OPD q/v adapter merge only targets gated full-attention projections"
            );

            // Cache the pristine base weights for this layer once, before the
            // first merge mutates them. Split-borrow safe: we read the resident
            // q/v `DeviceMatrix` (immutable) into host, then later overwrite.
            self.ensure_lora_base_cached(layer_idx, layer)?;

            if let Some(q) = &layer.q_proj {
                self.merge_lora_proj(layer_idx, q, scale, /* is_q = */ true)?;
            }
            if let Some(v) = &layer.v_proj {
                self.merge_lora_proj(layer_idx, v, scale, /* is_q = */ false)?;
            }
        }
        self.ctx.sync()?;
        Ok(())
    }

    /// Capture pristine host copies of this layer's q/v base weights on first
    /// touch (only for the projections the update actually carries).
    fn ensure_lora_base_cached(
        &mut self,
        layer_idx: usize,
        layer: &StudentLoraLayer,
    ) -> Result<()> {
        let attn = match &self.layers[layer_idx].attn {
            Qwen35Attn::Full(full) => full,
            Qwen35Attn::Linear(_) => {
                // Guarded by the caller's FullAttention check; defensive only.
                return Err(anyhow!(
                    "student LoRA layer {layer_idx} resolved to a linear-attention block"
                ));
            }
        };
        let entry = self.lora_base.entry(layer_idx).or_default();
        if layer.q_proj.is_some() && entry.q_proj.is_none() {
            entry.q_proj = Some(clone_matrix_to_host(
                &self.ctx,
                &attn.q_proj,
                layer_idx,
                "q_proj",
            )?);
        }
        if layer.v_proj.is_some() && entry.v_proj.is_none() {
            entry.v_proj = Some(clone_matrix_to_host(
                &self.ctx,
                &attn.v_proj,
                layer_idx,
                "v_proj",
            )?);
        }
        Ok(())
    }

    /// Recompute `W = base + scale·(B·A)` for one projection and upload it into
    /// the resident `DeviceMatrix`. `base` is the pristine cached host copy.
    fn merge_lora_proj(
        &mut self,
        layer_idx: usize,
        adapter: &StudentLoraMatrices,
        scale: f32,
        is_q: bool,
    ) -> Result<()> {
        let label = if is_q { "q_proj" } else { "v_proj" };
        // Pull the pristine base + the resident matrix shape under split borrows.
        let base = {
            let cached = self
                .lora_base
                .get(&layer_idx)
                .ok_or_else(|| anyhow!("layer {layer_idx} {label}: base weight not cached"))?;
            let base = if is_q { &cached.q_proj } else { &cached.v_proj };
            base.clone()
                .ok_or_else(|| anyhow!("layer {layer_idx} {label}: base weight missing"))?
        };

        let attn = match &self.layers[layer_idx].attn {
            Qwen35Attn::Full(full) => full,
            Qwen35Attn::Linear(_) => {
                return Err(anyhow!(
                    "student LoRA layer {layer_idx} resolved to a linear-attention block"
                ));
            }
        };
        let matrix = if is_q { &attn.q_proj } else { &attn.v_proj };
        ensure!(
            matrix.is_dense_bf16(),
            "layer {layer_idx} {label}: LoRA re-merge requires dense BF16 base weights; got {:?}",
            matrix.weight_format()
        );
        let rows = matrix.rows;
        let cols = matrix.cols;
        ensure!(
            base.len() == rows * cols,
            "layer {layer_idx} {label}: cached base len {} != rows*cols {}",
            base.len(),
            rows * cols
        );
        ensure!(
            adapter.in_features == cols,
            "layer {layer_idx} {label}: lora_A in_features {} != base cols {cols}",
            adapter.in_features
        );
        ensure!(
            adapter.out_features == rows,
            "layer {layer_idx} {label}: lora_B out_features {} != base rows {rows}",
            adapter.out_features
        );
        ensure!(
            adapter.a.len() == adapter.rank * cols,
            "layer {layer_idx} {label}: lora_A len {} != rank*in {}",
            adapter.a.len(),
            adapter.rank * cols
        );
        ensure!(
            adapter.b.len() == rows * adapter.rank,
            "layer {layer_idx} {label}: lora_B len {} != out*rank {}",
            adapter.b.len(),
            rows * adapter.rank
        );

        // W[r, c] = base[r, c] + scale · Σ_k B[r, k] · A[k, c].
        let rank = adapter.rank;
        let mut merged = vec![bf16::ZERO; rows * cols];
        for row in 0..rows {
            let b_row = &adapter.b[row * rank..row * rank + rank];
            for col in 0..cols {
                let mut delta = 0.0f32;
                for (k, &b_rk) in b_row.iter().enumerate() {
                    delta += b_rk * adapter.a[k * cols + col];
                }
                let idx = row * cols + col;
                merged[idx] = bf16::from_f32(base[idx].to_f32() + scale * delta);
            }
        }

        let uploaded = DeviceMatrix::from_host(&self.ctx, &merged, rows, cols)
            .map_err(|e| anyhow!("layer {layer_idx} {label}: upload merged weight failed: {e}"))?;
        match &mut self.layers[layer_idx].attn {
            Qwen35Attn::Full(full) => {
                if is_q {
                    full.q_proj = uploaded;
                } else {
                    full.v_proj = uploaded;
                }
            }
            Qwen35Attn::Linear(_) => unreachable!("FullAttention checked above"),
        }
        Ok(())
    }

    /// Dense SwiGLU MLP into `out` (`[hidden, seq]`). Every stage fully
    /// overwrites its scratch buffer (beta=0 GEMMs + full-range silu_mul).
    fn dense_mlp(
        &self,
        mlp: &DenseMlp,
        normed: &HiddenStates,
        dw: &mut DenseMlpScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let inter = mlp.gate_proj.rows;
        let seq_len = normed.seq_len;
        let gate = dw.gate.get(&self.ctx, inter, seq_len)?;
        let up = dw.up.get(&self.ctx, inter, seq_len)?;
        gemm_batch(&self.ctx, &mlp.gate_proj, normed, gate)?;
        gemm_batch(&self.ctx, &mlp.up_proj, normed, up)?;
        let act = dw.act.get(&self.ctx, inter, seq_len)?;
        silu_mul(&self.ctx, gate, up, act)?;
        gemm_batch(&self.ctx, &mlp.down_proj, act, out)?;
        Ok(())
    }

    /// Gated full attention over the contiguous per-slot K/V cache (uncached
    /// recompute over `[0, start_pos+seq_len)` each call) into `out`
    /// (`[hidden, seq]`, beta=0 o_proj GEMM). The prep kernel fuses q/k RMSNorm +
    /// RoPE + cache write; the gate kernel applies the per-head sigmoid gate
    /// carried in `q_full`. `start_pos_dev` is the forward-level GPU-resident
    /// `start_pos` (identical for every layer of one call).
    ///
    /// Prefill chunks (`seq_len > 1`) route through the vendored FA3 hopper
    /// fwd when [`qwen35_fa3_enabled`]; decode keeps the devpos kernel
    /// (graph-captured) untouched.
    #[allow(clippy::too_many_arguments)]
    fn full_attention(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        full_idx: usize,
        start_pos: usize,
        start_pos_dev: &CudaSlice<i32>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        // LOCAL per-rank widths (= global config on a single GPU): the sharded
        // q/k/v GEMM outputs, the per-slot caches, and the kernel launches must
        // all agree on this rank's head shard.
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();

        let FullAttnScratch {
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse,
            fa3_oaccum,
            fa3_lseaccum,
            fa3_semaphore,
        } = fw;
        let q_full = q_full.get(&self.ctx, q_proj_dim, seq_len)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, seq_len)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, seq_len)?;
        qwen35_profile(
            &self.ctx,
            "qwen/full/qkv_gemm",
            Some(full_idx),
            seq_len,
            || {
                gemm_batch(&self.ctx, &attn.q_proj, normed, q_full)?;
                gemm_batch(&self.ctx, &attn.k_proj, normed, k_batch)?;
                gemm_batch(&self.ctx, &attn.v_proj, normed, v_batch)?;
                Ok(())
            },
        )?;

        let q_prepped = q_prepped.get(&self.ctx, q_dim, seq_len)?;
        let attn_out = attn_heads.get(&self.ctx, q_dim, seq_len)?;
        let k_cache = &mut slot.k_caches[full_idx];
        let v_cache = &mut slot.v_caches[full_idx];

        let max_seq_len = k_cache.len / kv_dim;
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let kv_len = start_pos + seq_len;

        // ── 1. Prep: q/k RMSNorm + RoPE + write K/V into the contiguous cache. ──
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (kc_ptr, _g8) = k_cache.data.device_ptr_mut(&self.ctx.stream);
            let (vc_ptr, _g9) = v_cache.data.device_ptr_mut(&self.ctx.stream);
            let (sp_ptr, _g10) = start_pos_dev.device_ptr(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream; cache sized max_seq_len*kv_dim.
            qwen35_profile(&self.ctx, "qwen/full/prep", Some(full_idx), seq_len, || {
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qf_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        qp_ptr as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        seq_len as i32,
                        sp_ptr as *const i32,
                        c.rotary_dim as i32,
                        c.rms_norm_eps,
                        max_seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        // ── 2. Attention over the contiguous cache (causal; decode = qlen 1). ──
        // Decode (`seq_len == 1`) takes the devpos entry: the kv length is read
        // from the staged `start_pos` DEVICE buffer inside the kernel (same
        // math — kv_len = start_pos + token + 1 either way), so the launch is
        // CUDA-graph capture-safe and ONE captured graph replays across
        // positions. Eager decode uses the same entry, keeping the graph lane
        // kernel-for-kernel identical to its eager warm runs. Prefill keeps
        // the host-scalar entry (multi-token, never captured).
        {
            let (q_ptr, _g0) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (kc_ptr, _g1) = k_cache.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _g2) = v_cache.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_prepped/caches/out valid on ctx.stream for the shapes
            // above; `start_pos_dev` is the forward-level staged position (one
            // i32, value == start_pos).
            qwen35_profile(
                &self.ctx,
                "qwen/full/attention",
                Some(full_idx),
                seq_len,
                || {
                    unsafe {
                        if seq_len == 1 && c.head_dim == 256 && qwen35_fa3_decode_enabled() {
                            // FA3 split-KV decode mirrors SGLang/FlashInfer's
                            // flash-decoding shape: split the 4K KV sweep into
                            // multiple KV ranges, then combine partial softmax
                            // states. Default stays on the devpos kernel; this
                            // opt-in path is not decode-graph-safe because
                            // FA3 takes host seqlen_k as a launch parameter.
                            let splits = qwen35_fa3_decode_splits();
                            let lse = fa3_lse.get(&self.ctx, self.local_q_heads * seq_len)?;
                            let oaccum = fa3_oaccum.get(
                                &self.ctx,
                                splits * self.local_q_heads * seq_len * c.head_dim,
                            )?;
                            let lseaccum = fa3_lseaccum
                                .get(&self.ctx, splits * self.local_q_heads * seq_len)?;
                            let sem = fa3_semaphore.get(&self.ctx, 1)?;
                            let (lse_ptr, _g4) = lse.device_ptr_mut(&self.ctx.stream);
                            let (oaccum_ptr, _g5) = oaccum.device_ptr_mut(&self.ctx.stream);
                            let (lseaccum_ptr, _g6) = lseaccum.device_ptr_mut(&self.ctx.stream);
                            let (sem_ptr, _g7) = sem.device_ptr_mut(&self.ctx.stream);
                            let head_dim = c.head_dim as i64;
                            let args = ffi::ArleFa3FwdHd256Args {
                                q: q_ptr as *const ffi::Half,
                                k: kc_ptr as *const ffi::Half,
                                v: vc_ptr as *const ffi::Half,
                                o: o_ptr as *mut ffi::Half,
                                softmax_lse: lse_ptr as *mut f32,
                                out_accum: oaccum_ptr as *mut f32,
                                softmax_lse_accum: lseaccum_ptr as *mut f32,
                                tile_count_semaphore: sem_ptr as *mut i32,
                                seqlen_q: seq_len as i32,
                                seqlen_k: kv_len as i32,
                                num_heads: self.local_q_heads as i32,
                                num_heads_k: self.local_kv_heads as i32,
                                head_dim: c.head_dim as i32,
                                q_row_stride: q_dim as i64,
                                k_row_stride: head_dim,
                                v_row_stride: head_dim,
                                o_row_stride: q_dim as i64,
                                q_head_stride: head_dim,
                                k_head_stride: max_seq_len as i64 * head_dim,
                                v_head_stride: max_seq_len as i64 * head_dim,
                                o_head_stride: head_dim,
                                softmax_scale: sm_scale,
                                is_causal: 0,
                                num_splits: splits as i32,
                            };
                            ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, self.ctx.stream.cu_stream())
                                .result()?;
                        } else if seq_len == 1 {
                            let (sp_ptr, _g4) = start_pos_dev.device_ptr(&self.ctx.stream);
                            ffi::nonpaged_prefill_attention_devpos_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                sp_ptr as *const i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else if c.head_dim == 256 && qwen35_fa3_enabled() {
                            // FA3 fwd over the SAME buffers the in-tree kernel uses:
                            // q/out token-major [S, h, 256] (HD256 prep layout),
                            // cache head-major [h_k, max_seq, 256]. Passing the
                            // exact `kv_len` as seqlen_k keeps the shim on the
                            // non-varlen path; causal is bottom-right aligned =
                            // chunked-prefill semantics. Gate + o_proj follow
                            // unchanged.
                            let lse = fa3_lse.get(&self.ctx, self.local_q_heads * seq_len)?;
                            let sem = fa3_semaphore.get(&self.ctx, 1)?;
                            let (lse_ptr, _g4) = lse.device_ptr_mut(&self.ctx.stream);
                            let (sem_ptr, _g5) = sem.device_ptr_mut(&self.ctx.stream);
                            let head_dim = c.head_dim as i64;
                            let args = ffi::ArleFa3FwdHd256Args {
                                q: q_ptr as *const ffi::Half,
                                k: kc_ptr as *const ffi::Half,
                                v: vc_ptr as *const ffi::Half,
                                o: o_ptr as *mut ffi::Half,
                                softmax_lse: lse_ptr as *mut f32,
                                out_accum: std::ptr::null_mut(),
                                softmax_lse_accum: std::ptr::null_mut(),
                                tile_count_semaphore: sem_ptr as *mut i32,
                                seqlen_q: seq_len as i32,
                                seqlen_k: kv_len as i32,
                                num_heads: self.local_q_heads as i32,
                                num_heads_k: self.local_kv_heads as i32,
                                head_dim: c.head_dim as i32,
                                q_row_stride: q_dim as i64,
                                k_row_stride: head_dim,
                                v_row_stride: head_dim,
                                o_row_stride: q_dim as i64,
                                q_head_stride: head_dim,
                                k_head_stride: max_seq_len as i64 * head_dim,
                                v_head_stride: max_seq_len as i64 * head_dim,
                                o_head_stride: head_dim,
                                softmax_scale: sm_scale,
                                is_causal: 1,
                                num_splits: 1,
                            };
                            ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, self.ctx.stream.cu_stream())
                                .result()?;
                        } else {
                            ffi::nonpaged_prefill_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                    }
                    Ok(())
                },
            )?;
        }

        // ── 3. Per-head sigmoid gate from q_full's gate half. ──
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_full/attn_out valid on ctx.stream; gate layout per full-attn prep.
            qwen35_profile(&self.ctx, "qwen/full/gate", Some(full_idx), seq_len, || {
                unsafe {
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        c.head_dim as i32,
                        seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        qwen35_profile(
            &self.ctx,
            "qwen/full/o_proj",
            Some(full_idx),
            seq_len,
            || gemm_batch(&self.ctx, &attn.o_proj, attn_out, out),
        )?;
        // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
        qwen35_profile(
            &self.ctx,
            "qwen/full/allreduce",
            Some(full_idx),
            seq_len,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// Gated-delta-rule linear attention into `out` (`[hidden, seq]`, beta=0
    /// out-proj GEMM): in-proj → depthwise conv1d → RECURRENT gated-delta
    /// (advances the per-slot state in place) → gated output RMSNorm →
    /// out-proj. The conv ring + recurrent state carry across prefill/decode.
    fn linear_attention(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        linear_idx: usize,
        lw: &mut LinearAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        // LOCAL per-rank widths (= global config on a single GPU): the fused
        // [q|k|v] shard, conv channels, recurrent state, and kernel launches all
        // follow this rank's linear k/v head shard. b/a widths come off the
        // sharded projection rows directly (`[local_Vh, hidden]`).
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let b_dim = attn.in_proj_b.rows;
        let a_dim = attn.in_proj_a.rows;

        let LinearAttnScratch {
            qkv,
            z,
            b_proj,
            a_proj,
            qkv_conv,
            gdr_out,
            normed_out,
            fq_q,
            fq_k,
            fq_v,
            fq_a,
            fq_g,
            fq_g_cumsum,
            fq_beta,
        } = lw;
        let qkv = qkv.get(&self.ctx, qkv_dim, seq_len)?;
        let z = z.get(&self.ctx, z_dim, seq_len)?;
        let b_proj = b_proj.get(&self.ctx, b_dim, seq_len)?;
        let a_proj = a_proj.get(&self.ctx, a_dim, seq_len)?;
        qwen35_profile(
            &self.ctx,
            "qwen/linear/in_proj",
            Some(linear_idx),
            seq_len,
            || {
                gemm_batch(&self.ctx, &attn.in_proj_qkv, normed, qkv)?;
                gemm_batch(&self.ctx, &attn.in_proj_z, normed, z)?;
                gemm_batch(&self.ctx, &attn.in_proj_b, normed, b_proj)?;
                gemm_batch(&self.ctx, &attn.in_proj_a, normed, a_proj)?;
                Ok(())
            },
        )?;

        // ── conv1d (advances the per-slot conv ring). ──
        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, seq_len)?;
        let conv_state = &mut slot.conv_states[linear_idx];
        ensure!(
            conv_state.len == qkv_dim * (c.linear_conv_kernel_dim - 1),
            "Qwen3.5 conv state len {} != qkv_dim*(kernel-1) {}",
            conv_state.len,
            qkv_dim * (c.linear_conv_kernel_dim - 1)
        );
        {
            let (x_ptr, _g0) = qkv.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
            let (s_ptr, _g2) = conv_state.data.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g3) = qkv_conv.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: qkv/weight/state/out valid on ctx.stream; weight len checked
            // by the kernel against num_channels*kernel.
            qwen35_profile(
                &self.ctx,
                "qwen/linear/conv1d",
                Some(linear_idx),
                seq_len,
                || {
                    unsafe {
                        ffi::conv1d_prefill_cuda(
                            x_ptr as *const ffi::Half,
                            w_ptr as *const ffi::Half,
                            s_ptr as *mut ffi::Half,
                            o_ptr as *mut ffi::Half,
                            qkv_dim as i32,
                            seq_len as i32,
                            c.linear_conv_kernel_dim as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }

        // ── gated-delta rule. Decode (seq_len==1) is always the recurrent
        //    kernel. Prefill chunks default to the recurrent kernel; the
        //    FlashQLA chunked path (ARLE_QWEN35_GDR_CHUNKED) replaces the
        //    serial token scan with chunk-parallel TileLang kernels on the
        //    baked Qwen3.6 single-GPU shard. The legacy in-tree chunkwise
        //    TileLang path stays dead (sm_90 hang was in ITS kernels). ──
        let gdr_out = gdr_out.get(&self.ctx, z_dim, seq_len)?;
        let use_fq_chunked = seq_len > 1
            && qwen35_gdr_chunked_enabled()
            && self.local_linear_k_heads == 16
            && self.local_linear_v_heads == 32
            && c.linear_key_head_dim == 128
            && c.linear_value_head_dim == 128;
        if use_fq_chunked {
            let hg_dim = self.local_linear_k_heads * c.linear_key_head_dim;
            let fq_q = fq_q.get(&self.ctx, hg_dim, seq_len)?;
            let fq_k = fq_k.get(&self.ctx, hg_dim, seq_len)?;
            let fq_v = fq_v.get(&self.ctx, z_dim, seq_len)?;
            let fq_a = fq_a.get(&self.ctx, self.local_linear_v_heads * 64, seq_len)?;
            let g_len = self.local_linear_v_heads * seq_len;
            let fq_g = fq_g.get(&self.ctx, g_len)?;
            let fq_g_cumsum = fq_g_cumsum.get(&self.ctx, g_len)?;
            let fq_beta = fq_beta.get(&self.ctx, g_len)?;
            let gdr_state = &mut slot.gdr_states[linear_idx];

            let (qkv_ptr, _g0) = qkv_conv.data.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_proj.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_proj.data.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let (q_ptr, _g5) = fq_q.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _g6) = fq_k.data.device_ptr_mut(&self.ctx.stream);
            let (v_ptr, _g7) = fq_v.data.device_ptr_mut(&self.ctx.stream);
            let (a_inv_ptr, _g8) = fq_a.data.device_ptr_mut(&self.ctx.stream);
            let (g_ptr, _g9) = fq_g.device_ptr_mut(&self.ctx.stream);
            let (gc_ptr, _g10) = fq_g_cumsum.device_ptr_mut(&self.ctx.stream);
            let (beta_ptr, _g11) = fq_beta.device_ptr_mut(&self.ctx.stream);
            let (s_ptr, _g12) = gdr_state.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g13) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream, shapes per the slot
            // `.get` calls above. The slot state pointer is passed as BOTH
            // h0 and ht (in-place chunk chaining): each fwd CTA reads its h0
            // slice fully before writing the same ht slice.
            qwen35_profile(
                &self.ctx,
                "qwen/linear/gdr_fq",
                Some(linear_idx),
                seq_len,
                || {
                    unsafe {
                        ffi::gdr_fq_prep_cuda(
                            qkv_ptr as *const ffi::Half,
                            b_ptr as *const ffi::Half,
                            a_ptr as *const ffi::Half,
                            dt_ptr as *const ffi::Half,
                            alog_ptr as *const f32,
                            q_ptr as *mut ffi::Half,
                            k_ptr as *mut ffi::Half,
                            v_ptr as *mut ffi::Half,
                            g_ptr as *mut f32,
                            beta_ptr as *mut f32,
                            self.local_linear_k_heads as i32,
                            self.local_linear_v_heads as i32,
                            c.linear_key_head_dim as i32,
                            c.linear_value_head_dim as i32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        ffi::gdr_fq_cumsum_cuda(
                            g_ptr as *const f32,
                            gc_ptr as *mut f32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        ffi::gdr_fq_kkt_cuda(
                            k_ptr as *const ffi::Half,
                            beta_ptr as *const f32,
                            a_inv_ptr as *mut ffi::Half,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        ffi::gdr_fq_fwd_cuda(
                            q_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            v_ptr as *const ffi::Half,
                            a_inv_ptr as *const ffi::Half,
                            gc_ptr as *const f32,
                            beta_ptr as *const f32,
                            s_ptr as *const f32,
                            o_ptr as *mut ffi::Half,
                            s_ptr as *mut f32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }
        if !use_fq_chunked {
            let gdr_state = &mut slot.gdr_states[linear_idx];
            let (qkv_ptr, _g0) = qkv_conv.data.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_proj.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_proj.data.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let (s_ptr, _g5) = gdr_state.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g6) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream; head dims from config.
            qwen35_profile(
                &self.ctx,
                "qwen/linear/gdr_recurrent",
                Some(linear_idx),
                seq_len,
                || {
                    unsafe {
                        if seq_len == 1 {
                            ffi::gated_delta_rule_decode_cuda(
                                qkv_ptr as *const ffi::Half,
                                b_ptr as *const ffi::Half,
                                a_ptr as *const ffi::Half,
                                dt_ptr as *const ffi::Half,
                                alog_ptr as *const f32,
                                s_ptr as *mut f32,
                                o_ptr as *mut ffi::Half,
                                self.local_linear_k_heads as i32,
                                self.local_linear_v_heads as i32,
                                c.linear_key_head_dim as i32,
                                c.linear_value_head_dim as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else {
                            ffi::gated_delta_rule_prefill_recurrent_cuda(
                                qkv_ptr as *const ffi::Half,
                                b_ptr as *const ffi::Half,
                                a_ptr as *const ffi::Half,
                                dt_ptr as *const ffi::Half,
                                alog_ptr as *const f32,
                                s_ptr as *mut f32,
                                o_ptr as *mut ffi::Half,
                                self.local_linear_k_heads as i32,
                                self.local_linear_v_heads as i32,
                                c.linear_key_head_dim as i32,
                                c.linear_value_head_dim as i32,
                                seq_len as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                    }
                    Ok(())
                },
            )?;
        }

        // ── gated output RMSNorm (per value head; gate = z). ──
        let normed_out = normed_out.get(&self.ctx, z_dim, seq_len)?;
        {
            let (x_ptr, _g0) = gdr_out.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.norm_weight.device_ptr(&self.ctx.stream);
            let (gate_ptr, _g2) = z.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = normed_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: gdr_out/norm/z/out valid on ctx.stream; per-head layout from config.
            // The kernel launches exactly `num_heads` blocks, each normalizing one
            // flat `[val_dim]` slice at `blockIdx.x * val_dim` — gdr_out/z are
            // `[seq_len, Vh*Vd]` row-major, so the grid must cover all
            // seq_len*Vh (token, head) slices, not just token 0. `weight[tid]`
            // is a per-[Vd] broadcast (no blockIdx dependence), so the
            // extension is exact (the monolith's `rms_norm_gated_batch_into`
            // passed `seq_len * num_heads` identically).
            qwen35_profile(
                &self.ctx,
                "qwen/linear/norm",
                Some(linear_idx),
                seq_len,
                || {
                    unsafe {
                        ffi::rms_norm_gated_cuda(
                            x_ptr as *const ffi::Half,
                            w_ptr as *const f32,
                            gate_ptr as *const ffi::Half,
                            o_ptr as *mut ffi::Half,
                            (self.local_linear_v_heads * seq_len) as i32,
                            c.linear_value_head_dim as i32,
                            c.rms_norm_eps,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }

        qwen35_profile(
            &self.ctx,
            "qwen/linear/out_proj",
            Some(linear_idx),
            seq_len,
            || gemm_batch(&self.ctx, &attn.out_proj, normed_out, out),
        )?;
        // Row-parallel out_proj: sum the per-rank partials (no-op single-GPU).
        qwen35_profile(
            &self.ctx,
            "qwen/linear/allreduce",
            Some(linear_idx),
            seq_len,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// BATCHED DECODE (stage 1, contiguous KV): execute a whole multi-row
    /// pure-decode plan in ONE forward — B rows packed as `seq_len == B`.
    ///
    /// Token-parallel ops run batched exactly like a B-token prefill chunk
    /// (they already support `seq_len > 1`): embedding, every GEMM (q/k/v/o,
    /// qkv/z/b/a/out, router, experts, lm_head), `rms_norm_batched_offset`,
    /// residual adds, MoE (`moe_forward_into` with `num_tokens = B` → `R = 8B`
    /// routes — stays on the hand grouped kernels below
    /// `QWEN35_DEEPGEMM_MIN_ROUTES = 1024`), final norm + batched argmax.
    /// Per-row handling ONLY where state is per-slot:
    ///   - full attention (per-row loop over the existing single-row kernels
    ///     against each row's contiguous cache — DSv4 Step-A pattern, but with
    ///     column-offset pointers instead of copy-in/out:
    ///     [`Self::full_attention_batch_rows`]);
    ///   - conv1d + GDR (one batched kernel each via per-layer device pointer
    ///     tables: [`Self::linear_attention_batch`]);
    ///   - sampling (batched greedy argmax; non-greedy rows host-sample).
    ///
    /// FORMULA (the c=4 prediction this path is licensed against): at B=4 the
    /// token-parallel ops amortize ~4x (GEMM/MoE/norm launch count AND time
    /// per generated token /4), while per-row ops (full attention on 10
    /// layers x 3 kernels, sampling) stay linear in B → predicted aggregate
    /// ~2.3-3.2x single-stream tok/s at B=4 (vs ~1.0x today, because a
    /// rows>1 plan previously hard-errored the executor). License: pod
    /// c-sweep 1/2/4, aggregate tok/s + per-stream ITL + needle PASS at c=2.
    ///
    /// TP: safe under TP=2 by the same lockstep argument as single-row decode
    /// — every rank executes the identical plan (slot order fixed by the
    /// deterministic scheduler), the layer loop and the per-row attention
    /// loop have a fixed order, and every collective is an `all_reduce_sum`
    /// over an exact-shape `[hidden, B]` buffer (identical message length and
    /// call sequence on every rank; the per-row loop itself contains NO
    /// collectives). Complete all-reduce enumeration for one batched step:
    ///   1. full-attn o_proj partial   — `attn_out` `[hidden, B]` x num_full
    ///   2. linear-attn out_proj partial — `attn_out` `[hidden, B]` x num_linear
    ///   3. FFN (MoE routed+shared / dense down_proj) partial — `mlp_out`
    ///      `[hidden, B]` x num_layers
    ///
    /// All three are exact-shape `HiddenSlot` buffers at `seq_len == B`, so
    /// `all_reduce_sum`'s `data.len()`-derived message is exactly B columns.
    ///
    /// Advances every row's slot state (KV rows, conv ring, GDR state,
    /// `seq_len`) by one token, exactly like B sequential single-row decodes.
    /// Returns the B sampled tokens in row order.
    pub(crate) fn forward_decode_batch(
        &self,
        slots: &mut [Qwen35SlotState],
        bd: &mut Qwen35BatchDecodeState,
        slot_indices: &[usize],
        tokens: &[u32],
        params: &[SamplingParams],
        sample_positions: &[u64],
    ) -> Result<Vec<u32>> {
        let b = tokens.len();
        ensure!(b >= 1, "Qwen3.5 batched decode requires at least one row");
        ensure!(
            slot_indices.len() == b && params.len() == b && sample_positions.len() == b,
            "Qwen3.5 batched decode surface length mismatch: slots={} tokens={} params={} positions={}",
            slot_indices.len(),
            b,
            params.len(),
            sample_positions.len()
        );
        // Pre-mutation validation: every row in bounds and in budget BEFORE
        // any device state is touched.
        for &si in slot_indices {
            ensure!(
                si < slots.len(),
                "Qwen3.5 batched decode slot {si} outside executor slots {}",
                slots.len()
            );
            ensure!(
                slots[si].seq_len() < self.max_seq_len,
                "Qwen3.5 batched decode sequence {} exceeds KV cache budget {}",
                slots[si].seq_len() + 1,
                self.max_seq_len
            );
        }

        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;
        let vocab = self.output_projection().rows;

        // ── Stage pointer tables (no-op when the row→slot mapping is unchanged). ──
        bd.stage_pointer_tables(&self.ctx, slots, slot_indices)?;

        // ── Stage per-step inputs: token ids + per-row absolute positions. ──
        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let positions_host: Vec<i32> = slot_indices
            .iter()
            .map(|&si| slots[si].seq_len() as i32)
            .collect();

        let Qwen35BatchDecodeState {
            ws,
            positions,
            conv_state_ptrs,
            gdr_state_ptrs,
            logits_batch,
            argmax,
            ..
        } = bd;
        let Qwen35Workspace {
            token_ids,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            logits: row_logits,
            ..
        } = ws;
        let token_ids = token_ids.upload(&self.ctx, &token_ids_host)?;
        let positions_dev = positions.upload(&self.ctx, &positions_host)?;

        // ── Forward body: identical layer stack to `forward_hidden_staged`
        //    at seq_len == B, with the two per-slot dispatch differences. ──
        let hidden = hidden.get(&self.ctx, hidden_size, b)?;
        embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)?;
        let normed = normed.get(&self.ctx, hidden_size, b)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, b)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, b)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, b)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for layer in &self.layers {
            rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed)?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    self.full_attention_batch_rows(
                        full_attn,
                        normed,
                        slots,
                        slot_indices,
                        full_idx,
                        positions_dev,
                        full,
                        attn_out,
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    ensure!(
                        linear_idx < conv_state_ptrs.len(),
                        "Qwen3.5 batched decode linear layer {linear_idx} outside pointer tables {}",
                        conv_state_ptrs.len()
                    );
                    self.linear_attention_batch(
                        lin,
                        normed,
                        &conv_state_ptrs[linear_idx],
                        &gdr_state_ptrs[linear_idx],
                        linear,
                        attn_out,
                    )?;
                    linear_idx += 1;
                }
            }

            // Post-attn residual add + post_attention_layernorm via the
            // `add_batch` + `rms_norm_offset` pair (`hidden_mid`/`normed`).
            add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
            rms_norm_offset(
                &self.ctx,
                hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                normed,
            )?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                moe_forward_into(
                    &self.ctx,
                    moe_weights,
                    mlp_in,
                    cfg,
                    &self.expert_split,
                    moe,
                    mlp_out,
                )?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                self.dense_mlp(mlp, mlp_in, dense, mlp_out)?;
            }
            // ONE all-reduce covers the whole FFN partial (see the per-layer
            // enumeration in the method docs); exact `[hidden, B]` message.
            self.tp.all_reduce_sum(&self.ctx, mlp_out)?;

            // MLP residual add: the post-attn sum lives in `hidden_mid`;
            // add_batch reads hidden_mid/mlp_out and writes `hidden`.
            add_batch(&self.ctx, hidden_mid, mlp_out, hidden)?;
        }

        // ── Final norm over ALL rows + batched lm_head GEMM. ──
        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let logits_buf = logits_batch.get(&self.ctx, vocab, b)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, logits_buf)?;

        // Host seq_len advance: the device state (KV rows, conv rings, GDR
        // states) advanced in-stream above, so the host counters advance here
        // regardless of how sampling below fares — host and device stay
        // consistent (mirrors `forward_hidden`).
        for &si in slot_indices {
            slots[si].advance_seq_len(1);
        }

        // ── Sampling: ONE batched argmax over `[B, vocab]` (greedy fast
        //    path); any non-greedy row falls back to per-row host sampling. ──
        let argmax_buf = argmax.get(&self.ctx, b)?;
        {
            let (l_ptr, _gl) = logits_buf.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _ga) = argmax_buf.device_ptr_mut(&self.ctx.stream);
            // SAFETY: logits is a live `[B, vocab]` bf16 buffer and argmax a
            // live `[B]` i32 buffer on ctx.stream.
            unsafe {
                ffi::argmax_batch_cuda(
                    l_ptr as *const ffi::Half,
                    a_ptr as *mut i32,
                    b as i32,
                    vocab as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        self.ctx.sync()?;
        let greedy_ids = self
            .ctx
            .stream
            .clone_dtoh(argmax_buf)
            .map_err(|e| anyhow!("D2H qwen35 batched argmax failed: {e}"))?;
        let mut out = Vec::with_capacity(b);
        for (r, p) in params.iter().enumerate() {
            if p.is_greedy() {
                out.push(greedy_ids[r] as u32);
            } else {
                let row_vec = row_logits.get(&self.ctx, vocab)?;
                copy_row_to_vec(&self.ctx, logits_buf, r, row_vec)?;
                let host = row_vec.to_host(&self.ctx)?;
                out.push(infer_plan::sample_token(&host, p, sample_positions[r]));
            }
        }
        Ok(out)
    }

    /// Batched-decode full attention: batched q/k/v projections over all B
    /// rows (the GEMM amortizes), then a PER-ROW loop over the existing
    /// single-row prep + devpos-attention + gate kernels against each row's
    /// contiguous per-slot cache (DSv4 `dsv4.rs` Step-A pattern). Unlike
    /// DSv4's copy-in/copy-out, the full-attn kernels index strictly
    /// `token * dim + ...` from their base pointers with the position read
    /// from a device scalar, so at `seq_len == 1` a column-offset pointer
    /// (`base + r*dim`) and a per-row position pointer (`positions + r`)
    /// address row r exactly — zero extra copies. Stream-ordered (all
    /// launches on `ctx.stream`; rows touch disjoint caches and disjoint
    /// scratch columns), NO host sync between rows.
    #[allow(clippy::too_many_arguments)]
    fn full_attention_batch_rows(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
        full_idx: usize,
        positions_dev: &CudaSlice<i32>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let b = normed.seq_len;
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();

        let FullAttnScratch {
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            // Batched decode stays on the devpos kernel; FA3 scratch unused.
            fa3_lse: _,
            fa3_oaccum: _,
            fa3_lseaccum: _,
            fa3_semaphore: _,
        } = fw;
        let q_full = q_full.get(&self.ctx, q_proj_dim, b)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, b)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, b)?;
        gemm_batch(&self.ctx, &attn.q_proj, normed, q_full)?;
        gemm_batch(&self.ctx, &attn.k_proj, normed, k_batch)?;
        gemm_batch(&self.ctx, &attn.v_proj, normed, v_batch)?;

        let q_prepped = q_prepped.get(&self.ctx, q_dim, b)?;
        let attn_heads = attn_heads.get(&self.ctx, q_dim, b)?;

        {
            let (qf_base, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_base, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_base, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_base, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (ao_base, _g8) = attn_heads.data.device_ptr_mut(&self.ctx.stream);
            let (pos_base, _g9) = positions_dev.device_ptr(&self.ctx.stream);

            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                let k_cache = &mut slot.k_caches[full_idx];
                let v_cache = &mut slot.v_caches[full_idx];
                let max_seq_len = k_cache.len / kv_dim;
                let (kc_ptr, _gk) = k_cache.data.device_ptr_mut(&self.ctx.stream);
                let (vc_ptr, _gv) = v_cache.data.device_ptr_mut(&self.ctx.stream);
                // Column-offset device addresses: token-major storage makes
                // row r's block contiguous at element offset r*dim (bf16 = 2
                // bytes; i32 = 4 bytes).
                let qf_r = qf_base + (r * q_proj_dim * 2) as u64;
                let k_r = k_base + (r * kv_dim * 2) as u64;
                let v_r = v_base + (r * kv_dim * 2) as u64;
                let qp_r = qp_base + (r * q_dim * 2) as u64;
                let ao_r = ao_base + (r * q_dim * 2) as u64;
                let pos_r = pos_base + (r * 4) as u64;
                // SAFETY: every pointer is a live device allocation on
                // ctx.stream; the offsets stay inside the `[*, B]` buffers for
                // r < B; each kernel runs at seq_len == 1 so it touches only
                // row r's block + slot r's caches; `pos_r` points at this
                // row's staged i32 position.
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qf_r as *const ffi::Half,
                        k_r as *const ffi::Half,
                        v_r as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        qp_r as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        1, // seq_len: one new token per row
                        pos_r as *const i32,
                        c.rotary_dim as i32,
                        c.rms_norm_eps,
                        max_seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                    ffi::nonpaged_prefill_attention_devpos_cuda(
                        qp_r as *const ffi::Half,
                        kc_ptr as *const ffi::Half,
                        vc_ptr as *const ffi::Half,
                        ao_r as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        1, // seq_len
                        pos_r as *const i32,
                        max_seq_len as i32,
                        sm_scale,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_r as *const ffi::Half,
                        ao_r as *mut ffi::Half,
                        self.local_q_heads as i32,
                        c.head_dim as i32,
                        1, // seq_len
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }

        gemm_batch(&self.ctx, &attn.o_proj, attn_heads, out)?;
        // Row-parallel o_proj: ONE all-reduce over the exact `[hidden, B]`
        // buffer — message length is B valid columns by construction (no-op
        // single-GPU).
        self.tp.all_reduce_sum(&self.ctx, out)?;
        Ok(())
    }

    /// Batched-decode gated-delta linear attention: batched in-projections
    /// over all B rows, then the BATCHED conv1d + GDR kernels
    /// (`conv1d_decode_batch_cuda` / `gdr_decode_batch_cuda`) advancing every
    /// row's per-slot conv ring + recurrent state through the pre-staged
    /// per-layer device pointer tables — one launch each for all B rows
    /// (monolith `decode_batch_linear_attn_layer_graphable` re-port). Layout
    /// contracts verified against the single-row kernels:
    ///   - `x_batch [B, C]`: the in_proj GEMM output is token-major, so row
    ///     r's channels are contiguous at r*C — exactly the batch kernel's
    ///     `x_batch[b * num_channels + c]`;
    ///   - conv state `[C, K-1]` per slot: identical layout in `conv1d.cu`
    ///     (`conv_state[c * state_width + i]`) and the batch kernel
    ///     (`conv_state_ptrs[b] + c * sw`);
    ///   - GDR state `[Vh, Kd, Vd]` f32 per slot: identical in both kernels;
    ///   - `rms_norm_gated` over `[B, Vh*Vd]` row-major gdr_out/z: pass
    ///     `num_heads = Vh*B`, the same per-(token, head) grid extension the
    ///     single-row path already uses for `seq_len > 1`.
    fn linear_attention_batch(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        conv_table: &CudaSlice<u64>,
        gdr_table: &CudaSlice<u64>,
        lw: &mut LinearAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let b = normed.seq_len;
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let b_dim = attn.in_proj_b.rows;
        let a_dim = attn.in_proj_a.rows;

        let LinearAttnScratch {
            qkv,
            z,
            b_proj,
            a_proj,
            qkv_conv,
            gdr_out,
            normed_out,
            // Batched decode is per-token recurrent; FlashQLA scratch unused.
            fq_q: _,
            fq_k: _,
            fq_v: _,
            fq_a: _,
            fq_g: _,
            fq_g_cumsum: _,
            fq_beta: _,
        } = lw;
        let qkv = qkv.get(&self.ctx, qkv_dim, b)?;
        let z = z.get(&self.ctx, z_dim, b)?;
        let b_proj = b_proj.get(&self.ctx, b_dim, b)?;
        let a_proj = a_proj.get(&self.ctx, a_dim, b)?;
        gemm_batch(&self.ctx, &attn.in_proj_qkv, normed, qkv)?;
        gemm_batch(&self.ctx, &attn.in_proj_z, normed, z)?;
        gemm_batch(&self.ctx, &attn.in_proj_b, normed, b_proj)?;
        gemm_batch(&self.ctx, &attn.in_proj_a, normed, a_proj)?;

        // ── Batched conv1d (advances every row's conv ring in place). ──
        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, b)?;
        {
            let (x_ptr, _g0) = qkv.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
            let (tbl_ptr, _g2) = conv_table.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = qkv_conv.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: x/weight/out are live `[B, C]`/`[C*K]` buffers on
            // ctx.stream; `tbl_ptr` is the staged `[>=B]` u64 table whose
            // first B entries point at live `[C, K-1]` conv rings.
            unsafe {
                ffi::conv1d_decode_batch_cuda(
                    x_ptr as *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    tbl_ptr as *mut *mut ffi::Half,
                    o_ptr as *mut ffi::Half,
                    qkv_dim as i32,
                    c.linear_conv_kernel_dim as i32,
                    b as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // ── Batched gated-delta recurrent (advances every row's state). ──
        let gdr_out = gdr_out.get(&self.ctx, z_dim, b)?;
        {
            let (qkv_ptr, _g0) = qkv_conv.data.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_proj.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_proj.data.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let (tbl_ptr, _g5) = gdr_table.device_ptr(&self.ctx.stream);
            let (o_ptr, _g6) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: all buffers live on ctx.stream; `tbl_ptr` is the staged
            // `[>=B]` u64 table whose first B entries point at live
            // `[Vh, Kd, Vd]` f32 states; head dims from this rank's shard.
            unsafe {
                ffi::gdr_decode_batch_cuda(
                    qkv_ptr as *const ffi::Half,
                    b_ptr as *const ffi::Half,
                    a_ptr as *const ffi::Half,
                    dt_ptr as *const ffi::Half,
                    alog_ptr as *const f32,
                    tbl_ptr as *mut *mut f32,
                    o_ptr as *mut ffi::Half,
                    self.local_linear_k_heads as i32,
                    self.local_linear_v_heads as i32,
                    c.linear_key_head_dim as i32,
                    c.linear_value_head_dim as i32,
                    b as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // ── Gated output RMSNorm over all (token, head) slices. ──
        let normed_out = normed_out.get(&self.ctx, z_dim, b)?;
        {
            let (x_ptr, _g0) = gdr_out.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.norm_weight.device_ptr(&self.ctx.stream);
            let (gate_ptr, _g2) = z.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = normed_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: same per-(token, head) grid extension as the single-row
            // path's `seq_len > 1` case — `num_heads = Vh*B` covers all B*Vh
            // `[Vd]` slices of the row-major `[B, Vh*Vd]` buffers; `weight`
            // is a per-`[Vd]` broadcast with no blockIdx dependence.
            unsafe {
                ffi::rms_norm_gated_cuda(
                    x_ptr as *const ffi::Half,
                    w_ptr as *const f32,
                    gate_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    (self.local_linear_v_heads * b) as i32,
                    c.linear_value_head_dim as i32,
                    c.rms_norm_eps,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        gemm_batch(&self.ctx, &attn.out_proj, normed_out, out)?;
        // Row-parallel out_proj: ONE all-reduce over the exact `[hidden, B]`
        // buffer — message length is B valid columns by construction (no-op
        // single-GPU).
        self.tp.all_reduce_sum(&self.ctx, out)?;
        Ok(())
    }
}

/// The gated-delta fused `in_proj_qkv` row blocks `[q(Kh×Kd); k(Kh×Kd); v(Vh×Vd)]`
/// (gated_delta_rule.cu reads q at `k_head*Kd + d`, k at `q_dim + k_head*Kd + d`,
/// v at `q_dim + k_dim + v_head*Vd + d`). Shared by the qkv weight slice and the
/// depthwise conv1d slice — conv channel `c` filters in_proj_qkv output row `c`,
/// so both must shard the SAME row ranges.
fn linear_qkv_head_blocks(m: &Qwen35Config) -> [crate::shard_slice::HeadBlock; 3] {
    let k_block = crate::shard_slice::HeadBlock {
        heads: m.linear_num_key_heads,
        head_rows: m.linear_key_head_dim,
    };
    let v_block = crate::shard_slice::HeadBlock {
        heads: m.linear_num_value_heads,
        head_rows: m.linear_value_head_dim,
    };
    [k_block, k_block, v_block]
}

/// Load the fused gated-delta `in_proj_qkv` (`[2*Kh*Kd + Vh*Vd, hidden]` BF16)
/// sharded for this TP rank: each of the `[q; k; v]` blocks is head-sharded
/// independently and re-stacked, preserving the k↔v head grouping (a flat
/// column shard would cut across the block boundaries).
fn load_linear_qkv_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    m: &Qwen35Config,
    tp: &TpConfig,
) -> Result<DeviceMatrix> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "{name}: expected BF16 fused qkv projection, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape.len() == 2 && tensor.shape[0] == m.linear_attn_qkv_dim(),
        "{name}: expected [{}, hidden] fused qkv projection, got shape {:?}",
        m.linear_attn_qkv_dim(),
        tensor.shape
    );
    let sharded = crate::shard_slice::shard_head_blocks_column_parallel(
        &tensor.bytes,
        tensor.shape[1],
        2,
        &linear_qkv_head_blocks(m),
        tp,
    )?;
    DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
        .map_err(|e| anyhow!("upload sharded fused qkv {name}: {e}"))
}

/// Load the depthwise conv1d weight (`[qkv_dim, 1, kernel]` BF16) sharded for
/// this TP rank as a flat `[local_qkv_dim*kernel]` [`DeviceVec`]: the channel
/// rows mirror [`load_linear_qkv_sharded`]'s `[q; k; v]` block shard exactly
/// (conv1d.cu reads `conv_weight[c*kernel + k]` for channel `c`).
fn load_conv1d_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    m: &Qwen35Config,
    tp: &TpConfig,
) -> Result<DeviceVec> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "{name}: expected BF16 conv1d weight, got {:?}",
        tensor.dtype
    );
    let channels = tensor.shape.first().copied().unwrap_or(0);
    ensure!(
        channels == m.linear_attn_qkv_dim(),
        "{name}: conv1d channels {channels} != qkv_dim {} (shape {:?})",
        m.linear_attn_qkv_dim(),
        tensor.shape
    );
    // `[channels, 1, kernel]` (HF) or `[channels, kernel]`: the singleton middle
    // dim is squeezed by treating each channel's row as `kernel` elements.
    let kernel: usize = tensor.shape[1..].iter().product();
    ensure!(
        kernel == m.linear_conv_kernel_dim,
        "{name}: conv1d kernel {kernel} != linear_conv_kernel_dim {} (shape {:?})",
        m.linear_conv_kernel_dim,
        tensor.shape
    );
    let sharded = crate::shard_slice::shard_head_blocks_column_parallel(
        &tensor.bytes,
        kernel,
        2,
        &linear_qkv_head_blocks(m),
        tp,
    )?;
    DeviceVec::from_safetensors(ctx, &sharded.bytes)
        .map_err(|e| anyhow!("upload sharded conv1d {name}: {e}"))
}

/// Load a per-v-head 1D vector (`[Vh]`, BF16 or F32 normalized to BF16 exactly
/// like the single-GPU `load_vec_any`) sliced to this rank's contiguous v-head
/// range (gated_delta_rule.cu indexes `dt_bias[v_head]`).
fn load_v_head_vec_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    total_v_heads: usize,
    tp: &TpConfig,
) -> Result<DeviceVec> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.shape.len() == 1 && tensor.shape[0] == total_v_heads,
        "{name}: expected 1D [{total_v_heads}] per-v-head vector, got shape {:?}",
        tensor.shape
    );
    let bf16_bytes = SafetensorLoader::dsv4_bytes_to_bf16(name, &tensor)?;
    let (start, len) = v_head_shard_range(name, total_v_heads, tp)?;
    DeviceVec::from_safetensors(ctx, &bf16_bytes[start * 2..(start + len) * 2])
        .map_err(|e| anyhow!("upload sharded per-v-head vec {name}: {e}"))
}

/// Load a per-v-head 1D F32 tensor (`A_log`; F32 passthrough or BF16 widened,
/// matching the single-GPU `load_f32_vec`) sliced to this rank's v-head range,
/// uploaded as a device `f32` slice.
fn load_v_head_f32_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    total_v_heads: usize,
    tp: &TpConfig,
) -> Result<CudaSlice<f32>> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.shape.len() == 1 && tensor.shape[0] == total_v_heads,
        "{name}: expected 1D [{total_v_heads}] per-v-head tensor, got shape {:?}",
        tensor.shape
    );
    let host: Vec<f32> = match tensor.dtype {
        Dtype::F32 => tensor
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::BF16 => tensor
            .bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => anyhow::bail!("{name}: expected F32/BF16 1D tensor, got {other:?}"),
    };
    let (start, len) = v_head_shard_range(name, total_v_heads, tp)?;
    ctx.stream
        .clone_htod(&host[start..start + len])
        .map_err(|e| anyhow!("upload sharded per-v-head f32 {name}: {e}"))
}

/// This rank's contiguous v-head range `(start, len)` within `[0, total_v_heads)`.
fn v_head_shard_range(name: &str, total_v_heads: usize, tp: &TpConfig) -> Result<(usize, usize)> {
    ensure!(
        total_v_heads.is_multiple_of(tp.world_size),
        "{name}: {total_v_heads} v heads not divisible by world_size {}",
        tp.world_size
    );
    let local = total_v_heads / tp.world_size;
    Ok((tp.rank * local, local))
}

/// Copy a dense BF16 `DeviceMatrix`'s `data` buffer to a row-major host `Vec`.
/// Used to snapshot the pristine LoRA base weight before the first re-merge.
fn clone_matrix_to_host(
    ctx: &DeviceContext,
    matrix: &DeviceMatrix,
    layer_idx: usize,
    label: &str,
) -> Result<Vec<bf16>> {
    ensure!(
        matrix.is_dense_bf16(),
        "layer {layer_idx} {label}: LoRA base snapshot requires dense BF16; got {:?}",
        matrix.weight_format()
    );
    let host = ctx
        .stream
        .clone_dtoh(&matrix.data)
        .map_err(|e| anyhow!("layer {layer_idx} {label}: D2H base weight copy failed: {e}"))?;
    ctx.sync()?;
    ensure!(
        host.len() == matrix.rows * matrix.cols,
        "layer {layer_idx} {label}: base copy len {} != rows*cols {}",
        host.len(),
        matrix.rows * matrix.cols
    );
    Ok(host)
}

/// Offset RMSNorm (1+weight) over a batch — Qwen3.5 norms store `weight - 1`.
fn rms_norm_offset(
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
fn rms_norm_offset_vec(
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
}
