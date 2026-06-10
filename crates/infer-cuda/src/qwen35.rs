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
//!     the q_proj output (`q_proj` rows = `heads*head_dim*2`), HD256 / kv2, on a
//!     contiguous per-slot K/V cache.
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

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::{HostMatrixSnapshot, offload_raw_slice, reload_raw_slice};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use infer_plan::SamplingParams;
use infer_topo::TpConfig;
use qwen35_spec::{LayerType, Qwen35AttentionTensorNames, Qwen35Config};
use safetensors::tensor::Dtype;

use crate::executor::sample_cuda_token;
use crate::loader::SafetensorLoader;
use crate::moe::moe_forward;
use crate::moe_config::ExpertSplit;
use crate::ops::{
    add_batch, copy_row_to_vec, embedding_batch, gemm_batch, gemv, silu_mul, upload_i32,
};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;

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

    /// Load a BF16 Qwen3.5/3.6 HYBRID MoE checkpoint, resolving the TP runtime
    /// from the environment (single-GPU when no TP env is set).
    ///
    /// `max_seq_len` sizes the per-slot full-attn contiguous K/V cache.
    pub(crate) fn from_qwen35_moe_safetensors(
        model_path: &Path,
        max_seq_len: usize,
    ) -> Result<Self> {
        let tp = crate::loader::build_tp_runtime()?;
        Self::from_qwen35_moe_safetensors_with_tp(model_path, max_seq_len, tp)
    }

    /// Load with an explicit [`crate::tp::TpRuntime`] (tests inject a single-GPU
    /// runtime — mirrors the dense loader's `from_safetensors_with_tp`).
    pub(crate) fn from_qwen35_moe_safetensors_with_tp(
        model_path: &Path,
        max_seq_len: usize,
        tp: crate::tp::TpRuntime,
    ) -> Result<Self> {
        let m = Qwen35Config::from_model_dir(model_path)
            .map_err(|e| anyhow!("load Qwen3.5 config from {}: {e}", model_path.display()))?;
        ensure!(
            m.is_moe(),
            "from_qwen35_moe_safetensors requires a MoE checkpoint (num_experts > 0)"
        );
        ensure!(
            m.head_dim == 256 && m.num_attention_heads == 16 && m.num_key_value_heads == 2,
            "clean CUDA Qwen3.5 hybrid path only wires the HD256 q16/kv2 TileLang \
             full-attention kernels (the only HD256 SUPPORTED_HEADS that cover \
             Qwen3.6-35B-A3B); got heads={} kv_heads={} head_dim={}",
            m.num_attention_heads,
            m.num_key_value_heads,
            m.head_dim
        );
        // Full attention here is the GATED q_proj variant (Qwen3.5/3.6); the
        // HD256 prep+gate kernels assume it. Vanilla un-gated Qwen3 would need
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
        // rank's attention shape uniform — the HD256 kernels and the all-reduce
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

        let moe_config = crate::moe_config::moe_config_from_qwen35(&m)?;
        // EP mirrors TP: each rank owns `num_experts / world` whole experts
        // (`ExpertSplit::new` rejects an indivisible expert count loudly).
        let split = if tp_cfg.is_single() {
            ExpertSplit::single(m.num_experts)
        } else {
            ExpertSplit::new(m.num_experts, world, tp_cfg.rank)
                .map_err(|e| anyhow!("Qwen3.5 TP expert split: {e}"))?
        };

        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;

        let embed_tokens = loader.load_matrix(&ctx, m.embed_tokens_tensor_name())?;
        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, m.lm_head_tensor_name())?)
        };

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for layer_idx in 0..m.num_hidden_layers {
            let names = m.layer_tensor_names(layer_idx);
            let attn = match &names.attention {
                // Single GPU: full tensors, byte-identical to the pre-TP path.
                Qwen35AttentionTensorNames::Full(full) if tp_cfg.is_single() => {
                    Qwen35Attn::Full(Box::new(FullAttn {
                        q_proj: loader.load_matrix(&ctx, &full.q_proj)?,
                        k_proj: loader.load_matrix(&ctx, &full.k_proj)?,
                        v_proj: loader.load_matrix(&ctx, &full.v_proj)?,
                        o_proj: loader.load_matrix(&ctx, &full.o_proj)?,
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
                    q_proj: loader.load_qkv_head_sharded(
                        &ctx,
                        &full.q_proj,
                        local_q_heads,
                        m.head_dim * 2,
                        &tp_cfg,
                    )?,
                    k_proj: loader.load_qkv_head_sharded(
                        &ctx,
                        &full.k_proj,
                        local_kv_heads,
                        m.head_dim,
                        &tp_cfg,
                    )?,
                    v_proj: loader.load_qkv_head_sharded(
                        &ctx,
                        &full.v_proj,
                        local_kv_heads,
                        m.head_dim,
                        &tp_cfg,
                    )?,
                    // Row-parallel: each rank holds the o_proj input columns of
                    // its own heads; the forward all-reduces the partial sums.
                    o_proj: loader.load_matrix_sharded(
                        &ctx,
                        &full.o_proj,
                        infer_topo::ParallelLinearKind::Row,
                        &tp_cfg,
                    )?,
                    // q/k_norm are `[head_dim]`, broadcast across heads by the
                    // HD256 prep kernel — replicated.
                    q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                })),
                Qwen35AttentionTensorNames::Linear(lin) if tp_cfg.is_single() => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        in_proj_qkv: loader.load_matrix(&ctx, &lin.in_proj_qkv)?,
                        in_proj_z: loader.load_matrix(&ctx, &lin.in_proj_z)?,
                        in_proj_b: loader.load_matrix(&ctx, &lin.in_proj_b)?,
                        in_proj_a: loader.load_matrix(&ctx, &lin.in_proj_a)?,
                        conv1d_weight: loader.load_conv1d_vec(&ctx, &lin.conv1d_weight)?,
                        dt_bias: loader.load_vec_any(&ctx, &lin.dt_bias)?,
                        a_log: loader.load_f32_vec(&ctx, &lin.a_log)?,
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        out_proj: loader.load_matrix(&ctx, &lin.out_proj)?,
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
                        in_proj_z: loader.load_qkv_head_sharded(
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
                        out_proj: loader.load_matrix_sharded(
                            &ctx,
                            &lin.out_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }))
                }
            };

            let (mlp, moe) = if m.is_moe_layer(layer_idx) {
                let moe = loader.load_moe_layer_experts(
                    &ctx,
                    &names.common.layer_prefix,
                    &split,
                    &tp_cfg,
                )?;
                (None, Some(moe))
            } else if tp_cfg.is_single() {
                (
                    Some(DenseMlp {
                        gate_proj: loader.load_matrix(&ctx, &names.common.mlp_gate_proj)?,
                        up_proj: loader.load_matrix(&ctx, &names.common.mlp_up_proj)?,
                        down_proj: loader.load_matrix(&ctx, &names.common.mlp_down_proj)?,
                    }),
                    None,
                )
            } else {
                (
                    Some(DenseMlp {
                        gate_proj: loader.load_matrix_sharded(
                            &ctx,
                            &names.common.mlp_gate_proj,
                            infer_topo::ParallelLinearKind::Column,
                            &tp_cfg,
                        )?,
                        up_proj: loader.load_matrix_sharded(
                            &ctx,
                            &names.common.mlp_up_proj,
                            infer_topo::ParallelLinearKind::Column,
                            &tp_cfg,
                        )?,
                        down_proj: loader.load_matrix_sharded(
                            &ctx,
                            &names.common.mlp_down_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }),
                    None,
                )
            };

            layers.push(Qwen35Layer {
                input_layernorm: loader.load_vec(&ctx, &names.common.input_layernorm)?,
                attn,
                post_attention_layernorm: loader
                    .load_vec(&ctx, &names.common.post_attention_layernorm)?,
                mlp,
                moe,
            });
        }
        let norm = loader.load_vec(&ctx, m.norm_tensor_name())?;

        let rope_len = m
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        let (cos_cache, sin_cache) =
            crate::ops::precompute_rope(&ctx, m.head_dim, rope_len, m.rope_theta, None)?;
        ctx.sync()?;

        Ok(Self {
            ctx,
            config: m,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            moe_config: Some(moe_config),
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

    /// Run prefill or decode for one row. `start_pos` is the absolute position of
    /// the first token; `tokens` are the new tokens (whole prompt on prefill, one
    /// token on decode). Advances `slot.seq_len` and the recurrent state. Returns
    /// the next sampled token.
    pub(crate) fn forward_tokens(
        &self,
        slot: &mut Qwen35SlotState,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "forward_tokens requires at least one token"
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
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for layer in &self.layers {
            let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            rms_norm_offset(&self.ctx, &hidden, &layer.input_layernorm, eps, &mut normed)?;

            let attn_out = match &layer.attn {
                Qwen35Attn::Full(full) => {
                    let out = self.full_attention(full, &normed, slot, full_idx, start_pos)?;
                    full_idx += 1;
                    out
                }
                Qwen35Attn::Linear(lin) => {
                    let out = self.linear_attention(lin, &normed, slot, linear_idx)?;
                    linear_idx += 1;
                    out
                }
            };

            let mut hidden_mid = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            add_batch(&self.ctx, &hidden, &attn_out, &mut hidden_mid)?;

            rms_norm_offset(
                &self.ctx,
                &hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut normed,
            )?;
            let mut mlp_out = if let Some(moe) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                moe_forward(&self.ctx, moe, &normed, cfg, &self.expert_split)?
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                self.dense_mlp(mlp, &normed)?
            };
            // ONE all-reduce covers the whole FFN partial: the MoE buffer already
            // sums this rank's routed experts (non-local routes contribute zero)
            // + the column/row-sharded shared expert; the dense branch is a
            // row-parallel down_proj partial. No-op on a single GPU.
            self.tp.all_reduce_sum(&self.ctx, &mut mlp_out)?;

            let mut hidden_next = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            add_batch(&self.ctx, &hidden_mid, &mlp_out, &mut hidden_next)?;
            hidden = hidden_next;
        }

        slot.seq_len += seq_len;

        // Final norm (offset) + LM head on the last token only.
        //
        // TP invariant: embed/lm_head are replicated and `hidden` is
        // post-all-reduce (every row-parallel output above was summed), so the
        // logits — and therefore the sampled token — are identical on every
        // rank. No rank ever needs to broadcast its sample.
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        copy_row_to_vec(&self.ctx, &hidden, seq_len - 1, &mut last_hidden)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        rms_norm_offset_vec(&self.ctx, &last_hidden, &self.norm, eps, &mut last_normed)?;
        let mut logits = DeviceVec::zeros(&self.ctx, self.output_projection().rows)?;
        gemv(
            &self.ctx,
            self.output_projection(),
            &last_normed,
            &mut logits,
        )?;
        sample_cuda_token(&self.ctx, &logits, params, position)
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
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(DeviceVec, [usize; 2])> {
        ensure!(
            !tokens.is_empty(),
            "forward_token_logits_full requires at least one token"
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
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for layer in &self.layers {
            let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            rms_norm_offset(&self.ctx, &hidden, &layer.input_layernorm, eps, &mut normed)?;

            let attn_out = match &layer.attn {
                Qwen35Attn::Full(full) => {
                    let out = self.full_attention(full, &normed, slot, full_idx, start_pos)?;
                    full_idx += 1;
                    out
                }
                Qwen35Attn::Linear(lin) => {
                    let out = self.linear_attention(lin, &normed, slot, linear_idx)?;
                    linear_idx += 1;
                    out
                }
            };

            let mut hidden_mid = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            add_batch(&self.ctx, &hidden, &attn_out, &mut hidden_mid)?;

            rms_norm_offset(
                &self.ctx,
                &hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut normed,
            )?;
            let mut mlp_out = if let Some(moe) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                moe_forward(&self.ctx, moe, &normed, cfg, &self.expert_split)?
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                self.dense_mlp(mlp, &normed)?
            };
            // ONE all-reduce covers the whole FFN partial: the MoE buffer already
            // sums this rank's routed experts (non-local routes contribute zero)
            // + the column/row-sharded shared expert; the dense branch is a
            // row-parallel down_proj partial. No-op on a single GPU.
            self.tp.all_reduce_sum(&self.ctx, &mut mlp_out)?;

            let mut hidden_next = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            add_batch(&self.ctx, &hidden_mid, &mlp_out, &mut hidden_next)?;
            hidden = hidden_next;
        }

        slot.seq_len += seq_len;

        // Final norm (offset) over the WHOLE batch, then the batched lm-head GEMM
        // produces every row's logits — no last-row slice, no sampling.
        // (TP invariant as in `forward_tokens`: replicated lm_head over
        // post-all-reduce hidden ⇒ identical logits on every rank.)
        let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        rms_norm_offset(&self.ctx, &hidden, &self.norm, eps, &mut normed)?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        gemm_batch(&self.ctx, self.output_projection(), &normed, &mut logits)?;
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

    fn dense_mlp(&self, mlp: &DenseMlp, normed: &HiddenStates) -> Result<HiddenStates> {
        let inter = mlp.gate_proj.rows;
        let seq_len = normed.seq_len;
        let mut gate = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut up = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        gemm_batch(&self.ctx, &mlp.gate_proj, normed, &mut gate)?;
        gemm_batch(&self.ctx, &mlp.up_proj, normed, &mut up)?;
        let mut act = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        silu_mul(&self.ctx, &gate, &up, &mut act)?;
        let mut out = HiddenStates::zeros(&self.ctx, self.config.hidden_size, seq_len)?;
        gemm_batch(&self.ctx, &mlp.down_proj, &act, &mut out)?;
        Ok(out)
    }

    /// Gated full attention over the contiguous per-slot K/V cache (uncached
    /// recompute over `[0, start_pos+seq_len)` each call). HD256 prep fuses
    /// q/k RMSNorm + RoPE + cache write; the gate kernel applies the per-head
    /// sigmoid gate carried in `q_full`.
    fn full_attention(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        full_idx: usize,
        start_pos: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        // LOCAL per-rank widths (= global config on a single GPU): the sharded
        // q/k/v GEMM outputs, the per-slot caches, and the kernel launches must
        // all agree on this rank's head shard.
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();

        let mut q_full = HiddenStates::zeros(&self.ctx, q_proj_dim, seq_len)?;
        let mut k_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut v_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        gemm_batch(&self.ctx, &attn.q_proj, normed, &mut q_full)?;
        gemm_batch(&self.ctx, &attn.k_proj, normed, &mut k_batch)?;
        gemm_batch(&self.ctx, &attn.v_proj, normed, &mut v_batch)?;

        let mut q_prepped = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut attn_out = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let k_cache = &mut slot.k_caches[full_idx];
        let v_cache = &mut slot.v_caches[full_idx];

        let start_pos_buf = upload_i32(&self.ctx, &[start_pos as i32])?;
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
            let (sp_ptr, _g10) = start_pos_buf.device_ptr(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream; cache sized max_seq_len*kv_dim.
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
                    seq_len as i32,
                    sp_ptr as *const i32,
                    c.rotary_dim as i32,
                    c.rms_norm_eps,
                    max_seq_len as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // ── 2. Attention over the contiguous cache (causal; decode = qlen 1). ──
        {
            let (q_ptr, _g0) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (kc_ptr, _g1) = k_cache.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _g2) = v_cache.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_prepped/caches/out valid on ctx.stream for the shapes above.
            unsafe {
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

        // ── 3. Per-head sigmoid gate from q_full's gate half. ──
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_full/attn_out valid on ctx.stream; gate layout per HD256 prep.
            unsafe {
                ffi::attention_gate_batch_hd256_cuda(
                    qf_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    self.local_q_heads as i32,
                    seq_len as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        let mut out = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        gemm_batch(&self.ctx, &attn.o_proj, &attn_out, &mut out)?;
        // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
        self.tp.all_reduce_sum(&self.ctx, &mut out)?;
        Ok(out)
    }

    /// Gated-delta-rule linear attention: in-proj → depthwise conv1d → RECURRENT
    /// gated-delta (advances the per-slot state in place) → gated output RMSNorm
    /// → out-proj. The conv ring + recurrent state carry across prefill/decode.
    fn linear_attention(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        linear_idx: usize,
    ) -> Result<HiddenStates> {
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

        let mut qkv = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        let mut z = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        let mut b_proj = HiddenStates::zeros(&self.ctx, b_dim, seq_len)?;
        let mut a_proj = HiddenStates::zeros(&self.ctx, a_dim, seq_len)?;
        gemm_batch(&self.ctx, &attn.in_proj_qkv, normed, &mut qkv)?;
        gemm_batch(&self.ctx, &attn.in_proj_z, normed, &mut z)?;
        gemm_batch(&self.ctx, &attn.in_proj_b, normed, &mut b_proj)?;
        gemm_batch(&self.ctx, &attn.in_proj_a, normed, &mut a_proj)?;

        // ── conv1d (advances the per-slot conv ring). ──
        let mut qkv_conv = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
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
        }

        // ── gated-delta RECURRENT (never chunkwise: WGMMA short-seq path hangs
        //    on sm_90; the recurrent kernel handles seq_len==1 decode too). ──
        let mut gdr_out = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        let gdr_state = &mut slot.gdr_states[linear_idx];
        {
            let (qkv_ptr, _g0) = qkv_conv.data.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_proj.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_proj.data.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let (s_ptr, _g5) = gdr_state.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g6) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream; head dims from config.
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
        }

        // ── gated output RMSNorm (per value head; gate = z). ──
        let mut normed_out = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        {
            let (x_ptr, _g0) = gdr_out.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.norm_weight.device_ptr(&self.ctx.stream);
            let (gate_ptr, _g2) = z.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = normed_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: gdr_out/norm/z/out valid on ctx.stream; per-head layout from config.
            unsafe {
                ffi::rms_norm_gated_cuda(
                    x_ptr as *const ffi::Half,
                    w_ptr as *const f32,
                    gate_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    self.local_linear_v_heads as i32,
                    c.linear_value_head_dim as i32,
                    c.rms_norm_eps,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        let mut out = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        gemm_batch(&self.ctx, &attn.out_proj, &normed_out, &mut out)?;
        // Row-parallel out_proj: sum the per-rank partials (no-op single-GPU).
        self.tp.all_reduce_sum(&self.ctx, &mut out)?;
        Ok(out)
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
