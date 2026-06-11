//! DeepSeek-V4-Flash FP8 model: weight structs, MLA KV arena, EP-aware loader.
//!
//! The DSv4 port lives here: it loads the FP8 block-scaled weights (reusing the
//! shared `cuda-kernels` DSv4 tensors), stands up the MLA latent KV arena, and
//! drives the full forward — SlidingWindow / CompressedSparse / HybridCompressed
//! MLA attention (`attention.rs`), hyper-connections (`hc_mult > 1`, this file),
//! hash- and bias-routed FP8 DeepGEMM MoE (`moe.rs`). DSv4 is multi-GPU only
//! (256 FP8 experts + MLA sharding don't fit one GPU); `ExpertSplit` carries the
//! per-rank EP ownership, `ExpertSplit::single` is the dev/typecheck fallback.
//!
//! The `RealCudaExecutor` Dsv4 branch (`executor.rs`) constructs + runs this
//! model; the lead wires the multi-process TP=8/EP=8 launcher + bench entry.

use std::path::Path;

use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::{CudaPipelineStreamKind, Dsv4Fp8DeepGemmWeightCache};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config, DeepSeekV4MoeRoutingKind};
use infer_moe::MoeConfig;
use infer_plan::SamplingParams;

use crate::loader::SafetensorLoader;
use crate::moe_config::ExpertSplit;

/// MLA latent KV arena descriptor (kv_heads = 1).
///
/// Unlike the per-head BF16 [`cuda_kernels::prelude::PagedKVPool`], MLA caches a
/// single compressed latent per token in the flat FP8 block layout FlashMLA's
/// sparse-decode consumes: `[NoPE | RoPE]` packed to `bytes_per_token` bytes
/// (`cuda-kernels/src/attention.rs` `dsv4_fp8_kv_pack`, 584 B/token for the
/// canonical NoPE=448 / RoPE=64 / head_dim=512 shape). The device arena itself
/// is allocated by Piece 2 once the FlashMLA decode launch lands; Piece 1 only
/// pins the shape contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dsv4MlaKvArena {
    /// RoPE-carrying dims (`qk_rope_head_dim`, 64 for DSv4-Flash).
    pub rope_dim: usize,
    /// NoPE latent dims (`head_dim - qk_rope_head_dim`, 448 for DSv4-Flash).
    pub nope_dim: usize,
    /// FlashMLA paged block size (`page_block_size`, 64 for DSv4-Flash MODEL1).
    pub page_block_size: usize,
    /// Packed bytes per token in the FP8 arena (NoPE FP8 + RoPE bf16 + e8m0).
    pub bytes_per_token: usize,
    pub num_layers: usize,
}

/// Packed bytes per token the FlashMLA sparse-FP8 decode reads for the canonical
/// NoPE=448 / RoPE=64 shape (`dsv4_fp8_kv_pack` doc).
const DSV4_FLASH_KV_BYTES_PER_TOKEN: usize = 584;
const DSV4_FLASH_PAGE_BLOCK_SIZE: usize = 64;

impl Dsv4MlaKvArena {
    fn from_config(config: &DeepSeekV4Config) -> Result<Self> {
        let rope_dim = config.qk_rope_head_dim;
        let nope_dim = config
            .head_dim
            .checked_sub(rope_dim)
            .filter(|&d| d > 0)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DSv4 head_dim {} must exceed qk_rope_head_dim {rope_dim}",
                    config.head_dim
                )
            })?;
        // The shared pack kernel is fixed to the MODEL1 NoPE=448/RoPE=64 layout;
        // a different shape needs a new pack kernel, not a param tweak.
        ensure!(
            nope_dim == 448 && rope_dim == 64,
            "DSv4 MLA KV arena only wires the FlashMLA MODEL1 NoPE=448/RoPE=64 \
             pack (584 B/token), got NoPE={nope_dim} RoPE={rope_dim}"
        );
        Ok(Self {
            rope_dim,
            nope_dim,
            page_block_size: DSV4_FLASH_PAGE_BLOCK_SIZE,
            bytes_per_token: DSV4_FLASH_KV_BYTES_PER_TOKEN,
            num_layers: config.num_hidden_layers,
        })
    }
}

/// Compressor sub-block for CSA/HCA layers (`compress_ratio > 0`): projects the
/// wide hidden into the compressed-key latent stream the sparse attention reads.
/// `wkv`/`wgate`/`ape` may be FP8/FP4 block-scaled or bf16 (`dsv4_linear`
/// dispatches on `weight_format`); `norm` is bf16.
pub(crate) struct Dsv4Compressor {
    pub wkv: DeviceMatrix,
    pub wgate: DeviceMatrix,
    pub ape: DeviceMatrix,
    pub norm: DeviceVec,
}

/// Sparse indexer sub-block (CompressedSparse mode only): a second compressor
/// over `index_head_dim` keys + `wq_b`/`weights_proj` projections that feed the
/// `dsv4_csa_select_cuda` top-k block selector.
pub(crate) struct Dsv4Indexer {
    pub wq_b: DeviceMatrix,
    pub weights_proj: DeviceMatrix,
    pub compressor: Dsv4Compressor,
    /// DeepGEMM repack of `wq_b` for the prefill index-query projection (the #1
    /// remaining projection after wq_b/wo: 135ms / 67% of linear at M=1024). Built
    /// when the prefill DeepGEMM scratch is enabled; `None` falls back to scalar.
    pub wq_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
}

/// One hyper-connection mixing block (`hc_attn` / `hc_ffn` per layer, `hc_head`
/// at the head). `mix_fn` projects the wide stream into the `(2+hc_mult)*hc_mult`
/// mixing weights; `base`/`scale` are the learned bias + scale read by the
/// sinkhorn `dsv4_mhc_params_cuda`.
pub(crate) struct Dsv4HyperConnection {
    pub base: DeviceVec,
    pub mix_fn: DeviceMatrix,
    pub scale: DeviceVec,
}

/// One DSv4 MLA attention block's weights.
///
/// Q-LoRA: `wq_a` (down) → `q_norm` → `wq_b` (up to per-head Q). KV is the
/// compressed latent: `wkv` → `kv_norm`. Output is also low-rank: `wo_a` (per
/// o-group) → `wo_b` (back to hidden). `attn_sink` is the per-head sink logit.
/// `compressor`/`indexer` are present on CSA/HCA layers (`compress_ratio > 0`):
/// the compressor on both CSA and HCA, the indexer on CSA only.
pub(crate) struct Dsv4Attention {
    pub wq_a: DeviceMatrix,
    pub wqkv_a_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub q_norm: DeviceVec,
    pub wq_b: DeviceMatrix,
    /// DeepGEMM-layout FP8 cache of `wq_b` for the decode projection (M=1) — lets
    /// the residual scalar `dsv4_fp8_gemv_batch` GEMV (nsys #1, 3.62ms) route
    /// through tensor-core DeepGEMM like the fused wq_a|wkv path. `None` unless
    /// the fused-wqkv decode alloc gate is on.
    pub wq_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub wkv: DeviceMatrix,
    pub kv_norm: DeviceVec,
    pub wo_a: DeviceMatrix,
    pub wo_b: DeviceMatrix,
    /// DeepGEMM-layout FP8 caches of the output projection (`wo_a`/`wo_b`) for the
    /// decode path (lever #1b), companion to [`Self::wq_b_deepgemm`]. `local_width
    /// == hidden_size` on DSv4-Flash, so the M=1 quantize reuses the fused-wqkv FP8
    /// scratch. `None` unless the fused-wqkv decode alloc gate is on.
    pub wo_a_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub wo_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub attn_sink: DeviceVec,
    pub attn_sink_f32: CudaSlice<f32>,
    pub compressor: Option<Dsv4Compressor>,
    pub indexer: Option<Dsv4Indexer>,
}

/// One DSv4 routed-MoE block: prebuilt group-major FP8 DeepGEMM caches for
/// w1/w3 (gate/up) and w2 (down), the router gate, and the dense shared expert.
/// Only this rank's `ExpertSplit` slice is resident.
///
/// Routing kind is per-layer: bias-routed layers carry `gate_bias` (the
/// `noaux_tc` correction); hash-routed layers (`layer_idx < num_hash_layers`)
/// carry `hash_tid2eid` (a host `[vocab_size * topk]` table mapping token id →
/// experts directly) and ignore the learned router gate. Exactly one is `Some`.
pub(crate) struct Dsv4MoeLayer {
    /// Contiguous per-rank group-major fused gate+up FP8 cache (w1 over w3,
    /// row-stacked) and down cache (w2). Built once by the loader; the masked
    /// grouped GEMM reads these directly every step.
    pub w13_grouped: crate::moe::GroupedCache,
    pub w2_grouped: crate::moe::GroupedCache,
    pub num_groups: usize,
    pub hidden_dim: usize,
    pub intermediate: usize,
    /// Router gate `[n_routed_experts, hidden]` (BF16 — the small router GEMM is
    /// not FP8). Read by bias-routed layers; hash layers still load it (harmless).
    pub gate: DeviceMatrix,
    /// Bias-routed layers only: per-expert `noaux_tc` correction `[n_routed]`.
    pub gate_bias: Option<DeviceVec>,
    /// Hash-routed layers only: host `tid2eid` table (`vocab_size * topk` i64),
    /// sliced per token to pick experts without the learned router.
    pub hash_tid2eid: Option<Vec<i64>>,
    /// Hash-routed layers only: device `tid2eid` table used by the on-device
    /// router. Kept alongside the host table so the existing host route remains
    /// available as an A/B oracle.
    pub hash_tid2eid_device: Option<CudaSlice<i64>>,
    pub routing_kind: DeepSeekV4MoeRoutingKind,
    /// Dense shared expert FP8 caches (always-on, n_shared_experts == 1).
    pub shared_w13: Dsv4Fp8DeepGemmWeightCache,
    pub shared_w2: Dsv4Fp8DeepGemmWeightCache,
}

/// One DSv4 transformer layer: hyper-connection mixers (`hc_attn`/`hc_ffn`),
/// pre-attn / pre-ffn norms, attention, and MoE. `mode` records the attention
/// variant (SW / CSA / HCA) the forward dispatches on.
pub(crate) struct Dsv4Layer {
    pub hc_attn: Dsv4HyperConnection,
    pub hc_ffn: Dsv4HyperConnection,
    pub attn_norm: DeviceVec,
    pub ffn_norm: DeviceVec,
    pub attention: Dsv4Attention,
    pub moe: Dsv4MoeLayer,
    pub mode: DeepSeekV4AttentionMode,
    pub compress_ratio: usize,
}

/// One shipped DSv4 MTP draft head (`mtp.0.*`): a full transformer layer plus
/// the DeepSeek MTP input-combine and output-head tensors. Loaded only under
/// `ARLE_DSV4_SPEC_DECODE=1`; the verify loop and KV rollback are Phase 2.
pub(crate) struct Dsv4MtpLayer {
    pub layer: Dsv4Layer,
    pub head_hc: Dsv4HyperConnection,
    pub enorm: DeviceVec,
    pub hnorm: DeviceVec,
    pub e_proj: DeviceMatrix,
    pub h_proj: DeviceMatrix,
    pub norm: DeviceVec,
}

/// Loaded DSv4-Flash model for one TP/EP rank.
pub(crate) struct Dsv4Model {
    pub ctx: DeviceContext,
    pub config: DeepSeekV4Config,
    pub moe_config: MoeConfig,
    pub split: ExpertSplit,
    pub kv_arena: Dsv4MlaKvArena,
    pub embed_tokens: DeviceMatrix,
    pub lm_head: DeviceMatrix,
    pub layers: Vec<Dsv4Layer>,
    pub norm: DeviceVec,
    /// Head hyper-connection: folds the wide residual stream back to one hidden
    /// row before the final RMSNorm + lm_head projection.
    pub head_hc: Dsv4HyperConnection,
    pub mtp: Option<Dsv4MtpLayer>,
    pub tp: crate::tp::TpRuntime,
    #[cfg(feature = "deepep")]
    pub deepep: Option<crate::deepep::DeepEpTransport>,
}

pub(crate) struct Dsv4SlotState {
    attention: Vec<crate::attention::Dsv4LayerAttentionState>,
    spec_rollback: Option<Vec<crate::attention::Dsv4LayerAttentionSnapshot>>,
    moe_decode_scratch: Vec<crate::moe::Dsv4MoeDecodeScratch>,
    start_pos_device: CudaSlice<i32>,
    decode_graph: Option<Dsv4DecodeGraphScratch>,
    /// Pre-allocated NVSHMEM low-latency MoE scratch, reused across all layers +
    /// decode steps (overwritten in place each `dsv4_moe_forward_deepep_ll`
    /// call). `Some` only when the `deepep_ll` transport is booted. Held once per
    /// slot — layers run sequentially so a single scratch suffices.
    #[cfg(feature = "deepep")]
    deepep_ll_scratch: Option<crate::deepep::DeepEpLlScratch>,
    seq_len: usize,
    max_seq_len: usize,
}

pub(crate) struct Dsv4DecodeGraphScratch {
    token_ids: CudaSlice<i32>,
    token_ids_u32: CudaSlice<u32>,
    embeddings: HiddenStates,
    initial_stream: HiddenStates,
    layers: Vec<Dsv4DecodeLayerGraphScratch>,
    tail_graph: crate::graph::CudaGraphState,
    whole_graph: crate::graph::CudaGraphState,
    last_hidden: DeviceVec,
    last_normed: DeviceVec,
    logits_batch: HiddenStates,
    logits: DeviceVec,
}

impl Dsv4DecodeGraphScratch {
    /// Request-boundary hook: re-arm one eager warm run on every graph state
    /// (whole-step + per-portion + tail) without dropping the captures. See
    /// [`crate::graph::CudaGraphState::rearm_warm`].
    fn rearm_for_new_request(&mut self) {
        self.whole_graph.rearm_warm(1);
        self.tail_graph.rearm_warm(1);
        for layer in &mut self.layers {
            layer.attn_graph.rearm_warm(1);
            layer.moe_graph.rearm_warm(1);
        }
    }
}

struct Dsv4DecodeLayerGraphScratch {
    attn_graph: crate::graph::CudaGraphState,
    moe_graph: crate::graph::CudaGraphState,
    attn_mhc: crate::hc::MhcDecodeScratch,
    ffn_mhc: crate::hc::MhcDecodeScratch,
    attn_in: HiddenStates,
    attn_normed: HiddenStates,
    attn_out: HiddenStates,
    attn_stream: HiddenStates,
    ffn_in: HiddenStates,
    ffn_normed: HiddenStates,
    moe_out: HiddenStates,
    shared: HiddenStates,
    moe_with_shared: HiddenStates,
    ffn_stream: HiddenStates,
}

impl Dsv4DecodeGraphScratch {
    fn new(model: &Dsv4Model) -> Result<Self> {
        let hidden_size = model.config.hidden_size;
        let stream_dim = hidden_size * model.config.hc_mult;
        let token_ids = model
            .ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow!("DSv4 decode graph token-id scratch alloc failed: {e}"))?;
        let token_ids_u32 = model
            .ctx
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow!("DSv4 decode graph u32 token-id scratch alloc failed: {e}"))?;
        let embeddings = unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? };
        let initial_stream = unsafe { HiddenStates::uninit(&model.ctx, stream_dim, 1)? };
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(Dsv4DecodeLayerGraphScratch::new(model, layer)?);
        }
        let last_hidden = DeviceVec::zeros(&model.ctx, hidden_size)?;
        let last_normed = DeviceVec::zeros(&model.ctx, hidden_size)?;
        let logits_batch = unsafe { HiddenStates::uninit(&model.ctx, model.lm_head.rows, 1)? };
        let logits = DeviceVec::zeros(&model.ctx, model.lm_head.rows)?;
        Ok(Self {
            token_ids,
            token_ids_u32,
            embeddings,
            initial_stream,
            layers,
            tail_graph: crate::graph::CudaGraphState::new(model.ctx.stream.clone()),
            whole_graph: crate::graph::CudaGraphState::new(model.ctx.stream.clone()),
            last_hidden,
            last_normed,
            logits_batch,
            logits,
        })
    }
}

impl Dsv4DecodeLayerGraphScratch {
    fn new(model: &Dsv4Model, layer: &Dsv4Layer) -> Result<Self> {
        let hidden_size = model.config.hidden_size;
        let stream_dim = hidden_size * model.config.hc_mult;
        Ok(Self {
            attn_graph: crate::graph::CudaGraphState::new(model.ctx.stream.clone()),
            moe_graph: crate::graph::CudaGraphState::new(model.ctx.stream.clone()),
            attn_mhc: crate::hc::MhcDecodeScratch::new(&model.ctx, &model.config, &layer.hc_attn)?,
            ffn_mhc: crate::hc::MhcDecodeScratch::new(&model.ctx, &model.config, &layer.hc_ffn)?,
            attn_in: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            attn_normed: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            attn_out: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            attn_stream: unsafe { HiddenStates::uninit(&model.ctx, stream_dim, 1)? },
            ffn_in: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            ffn_normed: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            moe_out: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            shared: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            moe_with_shared: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            ffn_stream: unsafe { HiddenStates::uninit(&model.ctx, stream_dim, 1)? },
        })
    }
}

/// Explicit lifetime owner for DSv4 eager forward temporaries.
///
/// `DeviceContext` disables cudarc's implicit event tracking so CUDA graph and
/// copy/comm stream dependencies stay explicit. DSv4's eager decode launches a
/// long chain of kernels over per-call buffers; without an owner that lives until
/// the final host-sync sample, Rust can drop/reuse those allocations while the
/// stream still has in-flight work that reads them.
///
/// #29's decode scratch pool now owns the small DeepGEMM/MoE temporaries that
/// originally needed this bridge. Leave the deep-copy keepalive as an explicit
/// diagnostic fallback only: `CudaSlice::clone()` is a device-to-device copy,
/// not a cheap handle clone, so enabling it in production reintroduces tens of
/// thousands of D2D API calls per decode window.
pub(crate) struct Dsv4ForwardKeepalive {
    active: bool,
    bf16: Vec<CudaSlice<half::bf16>>,
    f32: Vec<CudaSlice<f32>>,
    i32: Vec<CudaSlice<i32>>,
    #[cfg(feature = "deepep")]
    i64: Vec<CudaSlice<i64>>,
    u32: Vec<CudaSlice<u32>>,
    u8: Vec<CudaSlice<u8>>,
}

impl Dsv4ForwardKeepalive {
    fn new(active: bool) -> Self {
        let active = active && std::env::var_os("ARLE_DSV4_DEEP_COPY_KEEPALIVE").is_some();
        Self {
            active,
            bf16: Vec::with_capacity(512),
            f32: Vec::with_capacity(256),
            i32: Vec::with_capacity(128),
            #[cfg(feature = "deepep")]
            i64: Vec::with_capacity(32),
            u32: Vec::with_capacity(16),
            u8: Vec::with_capacity(128),
        }
    }

    pub(crate) fn keep_hidden(&mut self, value: &HiddenStates) {
        if !self.active {
            return;
        }
        self.bf16.push(value.data.clone());
    }

    pub(crate) fn keep_vec(&mut self, value: &DeviceVec) {
        if !self.active {
            return;
        }
        self.bf16.push(value.data.clone());
    }

    pub(crate) fn keep_f32(&mut self, value: &CudaSlice<f32>) {
        if !self.active {
            return;
        }
        self.f32.push(value.clone());
    }

    pub(crate) fn keep_i32(&mut self, value: &CudaSlice<i32>) {
        if !self.active {
            return;
        }
        self.i32.push(value.clone());
    }

    #[cfg(feature = "deepep")]
    pub(crate) fn keep_i64(&mut self, value: &CudaSlice<i64>) {
        if !self.active {
            return;
        }
        self.i64.push(value.clone());
    }

    pub(crate) fn keep_u8(&mut self, value: &CudaSlice<u8>) {
        if !self.active {
            return;
        }
        self.u8.push(value.clone());
    }

    /// Legacy deep-copy fallback for small device-router buffers. Default-off:
    /// #29's persistent scratch owns the decode buffers, and `CudaSlice::clone`
    /// would otherwise add a D2D copy for each retained slice.
    pub(crate) fn keep_route_hidden(&mut self, value: &HiddenStates) {
        if !self.active {
            return;
        }
        self.bf16.push(value.data.clone());
    }

    pub(crate) fn keep_route_f32(&mut self, value: &CudaSlice<f32>) {
        if !self.active {
            return;
        }
        self.f32.push(value.clone());
    }

    pub(crate) fn keep_route_i32(&mut self, value: &CudaSlice<i32>) {
        if !self.active {
            return;
        }
        self.i32.push(value.clone());
    }

    #[cfg(feature = "deepep")]
    pub(crate) fn keep_route_i64(&mut self, value: &CudaSlice<i64>) {
        if !self.active {
            return;
        }
        self.i64.push(value.clone());
    }

    pub(crate) fn keep_route_u32(&mut self, value: &CudaSlice<u32>) {
        if !self.active {
            return;
        }
        self.u32.push(value.clone());
    }

    fn len(&self) -> usize {
        let len =
            self.bf16.len() + self.f32.len() + self.i32.len() + self.u32.len() + self.u8.len();
        #[cfg(feature = "deepep")]
        let len = len + self.i64.len();
        len
    }
}

impl Dsv4SlotState {
    fn new(
        model: &Dsv4Model,
        max_seq_len: usize,
        slot_idx: usize,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
    ) -> Result<Self> {
        ensure!(max_seq_len > 0, "DSv4 slot max_seq_len must be positive");
        let mut attention = Vec::with_capacity(model.layers.len());
        let mut moe_decode_scratch = Vec::with_capacity(model.layers.len());
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let local_width = layer.attention.wq_b.rows;
            ensure!(
                local_width.is_multiple_of(model.config.head_dim),
                "DSv4 slot attention local width {local_width} is not a multiple of head_dim {}",
                model.config.head_dim
            );
            let pool = kv_adapter.layer(layer_idx)?;
            attention.push(crate::attention::Dsv4LayerAttentionState::new(
                &model.ctx,
                &model.config,
                layer.mode,
                layer.compress_ratio,
                max_seq_len,
                &model.kv_arena,
                local_width / model.config.head_dim,
                model.tp.config().world_size,
                slot_idx,
                pool,
            )?);
            moe_decode_scratch.push(crate::moe::Dsv4MoeDecodeScratch::new(
                &model.ctx,
                &model.moe_config,
                &model.split,
                &layer.moe,
            )?);
        }
        let spec_rollback = if dsv4_spec_decode_enabled() {
            let mut snapshots = Vec::with_capacity(attention.len());
            for state in &attention {
                snapshots.push(state.rollback_snapshot(
                    &model.ctx,
                    &model.config,
                    &model.kv_arena,
                )?);
            }
            Some(snapshots)
        } else {
            None
        };
        let start_pos_device = model
            .ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow!("DSv4 slot start_pos device scalar alloc failed: {e}"))?;
        // Pre-allocate the LL MoE scratch ONCE when the deepep_ll transport is
        // booted. `intermediate` is uniform across layers (all share the MoE
        // config), so layer 0's value sizes the SwiGLU/w2 stages.
        #[cfg(feature = "deepep")]
        let deepep_ll_scratch = match model.deepep.as_ref() {
            Some(transport) if transport.is_low_latency() => {
                let intermediate = model
                    .layers
                    .first()
                    .map(|layer| layer.moe.intermediate)
                    .ok_or_else(|| anyhow!("DSv4 deepep_ll: model has no layers"))?;
                Some(transport.alloc_ll_scratch(&model.ctx, intermediate)?)
            }
            _ => None,
        };
        Ok(Self {
            attention,
            spec_rollback,
            moe_decode_scratch,
            start_pos_device,
            decode_graph: None,
            #[cfg(feature = "deepep")]
            deepep_ll_scratch,
            seq_len: 0,
            max_seq_len,
        })
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub(crate) fn reset(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
    ) -> Result<()> {
        self.seq_len = 0;
        ctx.stream
            .memset_zeros(&mut self.start_pos_device)
            .map_err(|e| anyhow!("DSv4 slot start_pos reset failed: {e}"))?;
        for (layer_idx, layer) in self.attention.iter_mut().enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            layer.reset(ctx, pool)?;
        }
        // Re-arm one eager warm step on every captured decode graph: the new
        // request's first decode runs the host-side per-request work (SW ring
        // bootstrap, compressed bulk pack) eagerly, then replay resumes with
        // the SAME captured graph (capture cost paid once per slot lifetime).
        if let Some(graph) = self.decode_graph.as_mut() {
            graph.rearm_for_new_request();
        }
        Ok(())
    }

    pub(crate) fn capture_spec_rollback(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        draft_abs_pos: usize,
    ) -> Result<()> {
        let snapshots = self
            .spec_rollback
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 spec rollback snapshot not allocated"))?;
        ensure!(
            self.attention.len() == snapshots.len(),
            "DSv4 rollback snapshot layer count {} != attention states {}",
            snapshots.len(),
            self.attention.len()
        );
        for (layer_idx, (state, snapshot)) in self.attention.iter().zip(snapshots).enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            state.capture_rollback_snapshot(ctx, pool, snapshot, draft_abs_pos)?;
        }
        Ok(())
    }

    pub(crate) fn restore_spec_rollback(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        draft_abs_pos: usize,
    ) -> Result<()> {
        let snapshots = self
            .spec_rollback
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 spec rollback snapshot not allocated"))?;
        ensure!(
            self.attention.len() == snapshots.len(),
            "DSv4 rollback restore layer count {} != attention states {}",
            snapshots.len(),
            self.attention.len()
        );
        for (layer_idx, (state, snapshot)) in self.attention.iter_mut().zip(snapshots).enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            state.restore_rollback_snapshot(ctx, pool, snapshot, draft_abs_pos)?;
        }
        Ok(())
    }

    pub(crate) fn truncate(&mut self, layers: &[Dsv4Layer], new_len: usize) -> Result<()> {
        ensure!(
            new_len <= self.seq_len,
            "DSv4 slot truncate cannot grow from {} to {new_len}",
            self.seq_len
        );
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 slot truncate layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        self.seq_len = new_len;
        for (layer, state) in layers.iter().zip(&mut self.attention) {
            state.truncate_decode_len(layer.mode, layer.compress_ratio, new_len);
        }
        Ok(())
    }
}

impl std::fmt::Debug for Dsv4Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4Model")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("experts", &self.config.n_routed_experts)
            .field("experts_per_rank", &self.split.experts_per_rank)
            .field("kv_bytes_per_token", &self.kv_arena.bytes_per_token)
            .field("mtp_loaded", &self.mtp.is_some())
            .finish()
    }
}

impl Dsv4Model {
    /// Load a DSv4-Flash FP8 checkpoint for this TP/EP rank.
    ///
    /// EP mirrors TP (the plan's TP=8/EP=8 layout): `ep_size = world_size`,
    /// `ep_rank = rank`, so each rank owns `256 / world_size` experts. Single-GPU
    /// keeps all experts local (dev/typecheck). Weight FP8/FP4 + E8M0 scales load
    /// through the shared `cuda-kernels` DSv4 tensors; per-expert DeepGEMM caches
    /// are built at load. The forward (MLA, FP8 MoE) is Pieces 2/3.
    pub(crate) fn from_dsv4_fp8_safetensors(model_path: &Path) -> Result<Self> {
        let tp = build_dsv4_tp_runtime()?;
        Self::from_dsv4_fp8_safetensors_with_tp(model_path, tp)
    }

    pub(crate) fn from_dsv4_fp8_safetensors_with_tp(
        model_path: &Path,
        #[cfg_attr(not(feature = "nccl"), allow(unused_mut))] mut tp: crate::tp::TpRuntime,
    ) -> Result<Self> {
        let config = DeepSeekV4Config::from_json_file(model_path.join("config.json"))
            .map_err(|e| anyhow!("load DSv4 config from {}: {e}", model_path.display()))?;
        ensure_loadable(&config)?;

        let moe_config = Self::moe_config_from_config(&config)?;
        let tp_cfg = *tp.config();
        let split = if tp_cfg.is_single() {
            ExpertSplit::single(config.n_routed_experts)
        } else {
            ExpertSplit::new(config.n_routed_experts, tp_cfg.world_size, tp_cfg.rank)
                .map_err(|e| anyhow!("DSv4 EP split: {e}"))?
        };
        let kv_arena = Dsv4MlaKvArena::from_config(&config)?;

        let ctx = DeviceContext::new()?;
        // One-shot small-message collectives (default-on, loud auto-degrade).
        // COLLECTIVE boot — identical construction point on every rank, BEFORE
        // the DeepEP boot so the collective sequences line up across ranks.
        #[cfg(feature = "nccl")]
        tp.init_oneshot_comm(&ctx);
        #[cfg(feature = "deepep")]
        let deepep = crate::deepep::DeepEpTransport::maybe_boot(
            &ctx,
            &tp,
            config.hidden_size,
            config.n_routed_experts,
        )?;
        let loader = SafetensorLoader::new(model_path)?;
        let names = config.tensor_names();

        let embed_tokens = loader.load_dsv4_global_matrix(&ctx, names.embed_tokens())?;
        let lm_head = loader.load_dsv4_global_matrix(&ctx, names.lm_head())?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let plan = config
                .attention_layer_plan(layer_idx)
                .ok_or_else(|| anyhow!("DSv4 layer {layer_idx} has no attention plan"))?;
            let lnames = config.layer_tensor_names(layer_idx);
            let attention = loader.load_dsv4_attention(&ctx, &config, &lnames.attn, &tp_cfg)?;
            let moe = loader.load_dsv4_moe_layer(
                &ctx,
                &lnames.ffn,
                &split,
                config.moe_routing_kind(layer_idx),
            )?;
            layers.push(Dsv4Layer {
                hc_attn: loader.load_dsv4_hyper_connection(&ctx, &lnames.hc_attn)?,
                hc_ffn: loader.load_dsv4_hyper_connection(&ctx, &lnames.hc_ffn)?,
                attn_norm: loader.load_dsv4_vec(&ctx, &lnames.attn_norm)?,
                ffn_norm: loader.load_dsv4_vec(&ctx, &lnames.ffn_norm)?,
                attention,
                moe,
                mode: plan.mode,
                compress_ratio: plan.compress_ratio,
            });
        }
        let norm = loader.load_dsv4_vec(&ctx, names.norm())?;
        let head_hc = loader.load_dsv4_hyper_connection(&ctx, &names.head_hc())?;
        let mtp = if dsv4_spec_decode_enabled() && config.num_nextn_predict_layers > 0 {
            ensure!(
                config.num_nextn_predict_layers == 1,
                "DSv4 Phase-1 MTP loader supports exactly one nextn layer, got {}",
                config.num_nextn_predict_layers
            );
            let mtp_names = config.mtp_tensor_names(0);
            let attention = loader.load_dsv4_attention(&ctx, &config, &mtp_names.attn, &tp_cfg)?;
            let moe = loader.load_dsv4_moe_layer(
                &ctx,
                &mtp_names.ffn,
                &split,
                DeepSeekV4MoeRoutingKind::LearnedBias,
            )?;
            let compress_ratio = 0;
            Some(Dsv4MtpLayer {
                layer: Dsv4Layer {
                    hc_attn: loader.load_dsv4_hyper_connection(&ctx, &mtp_names.hc_attn)?,
                    hc_ffn: loader.load_dsv4_hyper_connection(&ctx, &mtp_names.hc_ffn)?,
                    attn_norm: loader.load_dsv4_vec(&ctx, &mtp_names.attn_norm)?,
                    ffn_norm: loader.load_dsv4_vec(&ctx, &mtp_names.ffn_norm)?,
                    attention,
                    moe,
                    mode: config.attention_mode_for_compress_ratio(compress_ratio),
                    compress_ratio,
                },
                head_hc: loader.load_dsv4_hyper_connection(&ctx, &mtp_names.hc_head)?,
                enorm: loader.load_dsv4_vec(&ctx, &mtp_names.enorm)?,
                hnorm: loader.load_dsv4_vec(&ctx, &mtp_names.hnorm)?,
                e_proj: loader.load_dsv4_global_matrix(&ctx, &mtp_names.e_proj)?,
                h_proj: loader.load_dsv4_global_matrix(&ctx, &mtp_names.h_proj)?,
                norm: loader.load_dsv4_vec(&ctx, &mtp_names.norm)?,
            })
        } else {
            None
        };
        ctx.sync()?;

        Ok(Self {
            ctx,
            config,
            moe_config,
            split,
            kv_arena,
            embed_tokens,
            lm_head,
            layers,
            norm,
            head_hc,
            mtp,
            tp,
            #[cfg(feature = "deepep")]
            deepep,
        })
    }

    pub(crate) fn new_kv_adapter(
        &self,
        max_seq_len: usize,
        num_slots: usize,
    ) -> Result<crate::attention::Dsv4KvAdapter> {
        let mut specs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let local_width = layer.attention.wq_b.rows;
            ensure!(
                local_width.is_multiple_of(self.config.head_dim),
                "DSv4 attention pool local width {local_width} is not a multiple of head_dim {}",
                self.config.head_dim
            );
            specs.push((
                layer.mode,
                layer.compress_ratio,
                local_width / self.config.head_dim,
            ));
        }
        crate::attention::Dsv4KvAdapter::new(
            &self.ctx,
            &self.config,
            &specs,
            max_seq_len,
            &self.kv_arena,
            self.tp.config().world_size,
            num_slots,
        )
    }

    pub(crate) fn new_slot_state(
        &self,
        max_seq_len: usize,
        slot_idx: usize,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
    ) -> Result<Dsv4SlotState> {
        Dsv4SlotState::new(self, max_seq_len, slot_idx, kv_adapter)
    }

    /// Clamp `requested` decode slots to what the KV budget affords, from
    /// `cudaMemGetInfo() × MEM_FRACTION ÷ per-slot KV bytes`. This is the dynamic-mem-budget
    /// fix for the c=32 OOM CRASH (root cause: a fixed `num_slots` whose arena alloc OOMs at
    /// high concurrency × long `max_seq_len`). The per-slot cost is an itemized
    /// ledger: the EXACT FP8 arena term (`max_seq_len × bytes_per_token ×
    /// num_layers`) scaled ×2 to cover compressor/SW per-slot buffers + forward
    /// activations, PLUS the official-DSA indexer scratch (one
    /// `Dsv4DsaOfficialState` per CSA layer per slot — its `logits` tile scales
    /// with `max_seq/cr` and dwarfs the arena at long context; un-budgeted it
    /// OOMs engine build at 256K, issue #67).
    ///
    /// Cross-rank consistency: per-rank `mem_get_info` is NOT guaranteed identical
    /// (allocator state differs per rank), and the clamped count feeds the
    /// scheduler's slot gate — any per-rank divergence in scheduler-visible
    /// capacity diverges the deterministic planner and deadlocks NCCL. The local
    /// affordable count is therefore NCCL min-reduced; every rank calls this at
    /// the same construction point (collective). A rank that cannot query its
    /// memory contributes `i32::MAX` (does not bind) instead of skipping the
    /// collective.
    pub(crate) fn kv_budget_num_slots(
        &self,
        requested: usize,
        max_seq_len: usize,
    ) -> Result<usize> {
        const MEM_FRACTION: f64 = 0.9;
        const PER_SLOT_OVERHEAD_X: usize = 2;
        let arena_per_slot = max_seq_len
            .saturating_mul(self.kv_arena.bytes_per_token)
            .saturating_mul(self.kv_arena.num_layers);
        // Official-DSA selector memory splits into the ONE model-wide shared
        // scratch (a fixed subtraction from the budget) and the per-(slot,
        // CSA-layer) stateful rotated_keys mirrors (a per-slot term). #67.
        let official_on = crate::attention::dsv4_dsa_official_enabled().unwrap_or(true);
        let csa_cr = self
            .layers
            .iter()
            .find(|layer| matches!(layer.mode, DeepSeekV4AttentionMode::CompressedSparse))
            .map(|layer| layer.compress_ratio);
        let dsa_shared_bytes: usize = match (official_on, csa_cr) {
            (true, Some(cr)) => {
                crate::attention::dsv4_dsa_shared_scratch_bytes(&self.config, cr, max_seq_len)
            }
            _ => 0,
        };
        // Per-slot stateful selector/compressor caches, itemized per layer:
        // rotated_keys + DSA key-cache band (CSA only), compressor compressed
        // cache (every cr>0 layer, head_dim wide) + indexer compressed cache
        // (CSA, index_head_dim wide). All scale with max_seq/cr.
        let mut dsa_rotated_per_slot: usize = 0;
        let mut state_caches_per_slot: usize = 0;
        for layer in &self.layers {
            if layer.compress_ratio == 0 {
                continue;
            }
            let cc = max_seq_len.div_ceil(layer.compress_ratio).max(1);
            state_caches_per_slot = state_caches_per_slot
                .saturating_add(cc.saturating_mul(self.config.head_dim).saturating_mul(2));
            if matches!(layer.mode, DeepSeekV4AttentionMode::CompressedSparse) {
                state_caches_per_slot = state_caches_per_slot.saturating_add(
                    cc.saturating_mul(self.config.index_head_dim)
                        .saturating_mul(2),
                );
                if official_on {
                    dsa_rotated_per_slot = dsa_rotated_per_slot.saturating_add(
                        crate::attention::dsv4_dsa_rotated_keys_bytes(
                            &self.config,
                            layer.compress_ratio,
                            max_seq_len,
                        ),
                    );
                    state_caches_per_slot = state_caches_per_slot.saturating_add(
                        crate::attention::dsv4_dsa_key_cache_bytes(
                            &self.config,
                            layer.compress_ratio,
                            max_seq_len,
                        )
                        .unwrap_or(0),
                    );
                }
            }
        }
        let per_slot = arena_per_slot
            .saturating_mul(PER_SLOT_OVERHEAD_X)
            .saturating_add(dsa_rotated_per_slot)
            .saturating_add(state_caches_per_slot);
        let affordable_local: i32 = match cudarc::driver::result::mem_get_info() {
            Ok((free, _total)) => {
                let budget =
                    ((free as f64 * MEM_FRACTION) as usize).saturating_sub(dsa_shared_bytes);
                log::info!(
                    "DSv4 KV budget: free {}MB, per_slot {}MB (arena×2 {}MB + rotated {}MB + \
                     state caches {}MB), shared DSA {}MB",
                    free >> 20,
                    per_slot >> 20,
                    arena_per_slot.saturating_mul(PER_SLOT_OVERHEAD_X) >> 20,
                    dsa_rotated_per_slot >> 20,
                    state_caches_per_slot >> 20,
                    dsa_shared_bytes >> 20,
                );
                budget
                    .checked_div(per_slot)
                    .map_or(i32::MAX, |n| i32::try_from(n.max(1)).unwrap_or(i32::MAX))
            }
            // Can't query (no active context / driver error) → don't bind
            // the min; the other ranks' budgets still apply.
            Err(_) => i32::MAX,
        };
        let affordable =
            self.tp
                .all_reduce_min_scalar_i32(&self.ctx, affordable_local)? as usize;
        if requested > affordable {
            log::warn!(
                "DSv4 KV budget: requested {requested} slots × ~{}MB/slot (arena×2 ~{}MB + \
                 per-slot selector/compressor caches ~{}MB) + shared DSA scratch ~{}MB exceeds the \
                 cross-rank-min affordable {affordable} (local affordable {affordable_local}, \
                 {MEM_FRACTION} of post-weights free); clamping num_slots to {affordable}. \
                 Lower --max-seq-len ({max_seq_len}) to raise concurrency.",
                per_slot >> 20,
                arena_per_slot.saturating_mul(PER_SLOT_OVERHEAD_X) >> 20,
                dsa_rotated_per_slot.saturating_add(state_caches_per_slot) >> 20,
                dsa_shared_bytes >> 20,
            );
        }
        Ok(requested.min(affordable))
    }

    pub(crate) fn truncate_slot(&self, slot: &mut Dsv4SlotState, new_len: usize) -> Result<()> {
        slot.truncate(&self.layers, new_len)
    }

    pub(crate) fn dump_mtp_rollback_state(
        &self,
        slot: &Dsv4SlotState,
        label: &str,
        abs_len: usize,
    ) -> Result<()> {
        let layer_idx = std::env::var("ARLE_DSV4_MTP_ROLLBACK_DUMP_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        if layer_idx >= slot.attention.len() {
            return Ok(());
        }
        slot.attention[layer_idx].dump_mtp_rollback_state(&self.ctx, layer_idx, label, abs_len)
    }

    /// Forward one prefill/decode step over `tokens` starting at `start_pos`,
    /// returning the next greedy/sampled token.
    ///
    /// The residual is the `hidden_size * hc_mult`-wide hyper-connection STREAM,
    /// not a plain hidden vector. Per layer the flow is:
    ///   `gen_mhc(hc_attn) → hc_pre → attn_norm → mla_attention → hc_post`
    ///   (+TP all-reduce of the O-LoRA partials) then the same wrap around
    ///   `ffn_norm → dsv4_moe_forward` via `hc_ffn`. The head HC then folds the
    ///   wide stream to one hidden row before the final RMSNorm + lm_head + sample.
    pub(crate) fn forward_tokens(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        self.forward_tokens_impl(slot, kv_adapter, tokens, start_pos, params, position, None)
    }

    pub(crate) fn forward_tokens_with_hidden(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<(u32, DeviceVec)> {
        let mut last_hidden =
            DeviceVec::zeros(&self.ctx, self.config.hidden_size * self.config.hc_mult)?;
        let token = self.forward_tokens_impl(
            slot,
            kv_adapter,
            tokens,
            start_pos,
            params,
            position,
            Some(&mut last_hidden),
        )?;
        Ok((token, last_hidden))
    }

    fn forward_tokens_impl(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        mut last_hidden_out: Option<&mut DeviceVec>,
    ) -> Result<u32> {
        let seq_len = tokens.len();
        let use_deepep_transport = dsv4_use_deepep_transport()?;
        // FlashMLA-decode captures cleanly (capture-safety fixes e95e11b6).
        // The graph routes on-device into fixed scratch and runs the masked
        // MoE tail — no gpu_router/pooled dependency.
        if dsv4_decode_graph_enabled()
            && last_hidden_out.is_none()
            && seq_len == 1
            && !use_deepep_transport
        {
            return self.forward_tokens_decode_graph(
                slot, kv_adapter, tokens[0], start_pos, params, position,
            );
        }

        let (stream, mut keepalive) =
            self.forward_tokens_stream_impl(slot, kv_adapter, tokens, start_pos, false)?;
        if let Some(out) = last_hidden_out.as_deref_mut() {
            self.capture_mtp_stream_hidden(&stream, seq_len - 1, out, &mut keepalive)?;
        }
        let token = self.forward_stream_last_token(
            &stream,
            seq_len,
            params,
            position,
            None,
            &mut keepalive,
        )?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(token)
    }

    pub(crate) fn forward_tokens_verify(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        position: u64,
    ) -> Result<(Vec<u32>, Vec<DeviceVec>)> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 verify forward requires at least one token"
        );
        let _nvtx = crate::nvtx::range("dsv4/lm_head_verify");
        let params = SamplingParams::default();

        // Batched verify (opt-in, default off; KNOWN col1 bug — see selftest probe).
        // ONE 2-token forward amortizes the weight read (proven +63%). Capture the
        // rollback only when the spec snapshot is allocated (skipped for the
        // selftest probe, which doesn't reject).
        if dsv4_mtp_batched_verify_enabled() && tokens.len() == 2 {
            let stream_dim = self.config.hidden_size * self.config.hc_mult;
            if dsv4_spec_decode_enabled() {
                slot.capture_spec_rollback(&self.ctx, kv_adapter, start_pos + 1)?;
            }
            let (stream, mut keepalive) =
                self.forward_tokens_stream_impl(slot, kv_adapter, tokens, start_pos, true)?;
            let mut h0 = DeviceVec::zeros(&self.ctx, stream_dim)?;
            let mut h1 = DeviceVec::zeros(&self.ctx, stream_dim)?;
            self.capture_mtp_stream_hidden(&stream, 0, &mut h0, &mut keepalive)?;
            self.capture_mtp_stream_hidden(&stream, 1, &mut h1, &mut keepalive)?;
            let base_next = self.forward_stream_last_token(
                &stream,
                1,
                &params,
                position,
                None,
                &mut keepalive,
            )?;
            let bonus = self.forward_stream_last_token(
                &stream,
                2,
                &params,
                position + 1,
                None,
                &mut keepalive,
            )?;
            std::hint::black_box(keepalive.len());
            drop(keepalive);
            return Ok((vec![base_next, bonus], vec![h0, h1]));
        }

        let mut argmax_tokens = Vec::with_capacity(tokens.len());
        let mut hiddens = Vec::with_capacity(tokens.len());
        for (row, &token_id) in tokens.iter().enumerate() {
            let row_start = start_pos + row;
            let row_position = position + row as u64;
            let (stream, mut keepalive) =
                self.forward_tokens_stream_impl(slot, kv_adapter, &[token_id], row_start, false)?;
            let mut row_hidden =
                DeviceVec::zeros(&self.ctx, self.config.hidden_size * self.config.hc_mult)?;
            self.capture_mtp_stream_hidden(&stream, 0, &mut row_hidden, &mut keepalive)?;
            let token = self.forward_stream_last_token(
                &stream,
                1,
                &params,
                row_position,
                None,
                &mut keepalive,
            )?;
            argmax_tokens.push(token);
            hiddens.push(row_hidden);
            std::hint::black_box(keepalive.len());
            drop(keepalive);
            if tokens.len() == 2 && row == 0 {
                let draft_abs_pos = row_start + 1;
                self.dump_mtp_rollback_state(
                    slot,
                    "spec_after_pending_before_draft",
                    draft_abs_pos,
                )?;
                slot.capture_spec_rollback(&self.ctx, kv_adapter, draft_abs_pos)?;
            }
        }
        Ok((argmax_tokens, hiddens))
    }

    /// Layer-major batched decode over N independent slots (one decode token each).
    ///
    /// Each row `r` decodes slot `slot_ids[r]` at `start_positions[r]`. The
    /// point-wise pipeline (embed / HC wrap / rms_norm / MoE / shared expert /
    /// all-reduce) runs over the whole `seq_len = N` batch exactly as the prefill
    /// path does — those ops are token-independent, so stacking N rows is
    /// math-identical to N separate single-row forwards. Attention is the only
    /// per-row-dependent step (each row attends to its own slot's KV history);
    /// Step A loops it per row (copy row r in/out of a `[hidden,1]` scratch and
    /// call the existing single-row [`mla_attention`]) so this restructure changes
    /// the *order* of work (row-major → layer-major) but not the math. Later
    /// phases replace the per-row attention loop with batched DSA / FlashMLA and
    /// the per-row MoE with grouped MoE. Returns one sampled token per row.
    pub(crate) fn forward_decode_batch(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        tokens: &[u32],
        start_positions: &[usize],
        positions: &[u64],
        params: &[SamplingParams],
    ) -> Result<Vec<u32>> {
        let n = slot_ids.len();
        ensure!(n > 0, "DSv4 batched decode requires at least one row");
        ensure!(
            tokens.len() == n
                && start_positions.len() == n
                && positions.len() == n
                && params.len() == n,
            "DSv4 batched decode surface length mismatch (slots {n}, tokens {}, starts {}, positions {}, params {})",
            tokens.len(),
            start_positions.len(),
            positions.len(),
            params.len()
        );
        let (stream, mut keepalive) = self.forward_decode_batch_stream_impl(
            slots,
            kv_adapter,
            slot_ids,
            tokens,
            start_positions,
        )?;
        let _nvtx = crate::nvtx::range("dsv4/lm_head_sample_batched");
        let mut out_tokens = Vec::with_capacity(n);
        for r in 0..n {
            // `forward_stream_last_token` folds stream row `seq_len - 1`; passing
            // `seq_len = r + 1` samples row r of the batched stream.
            let token = self.forward_stream_last_token(
                &stream,
                r + 1,
                &params[r],
                positions[r],
                None,
                &mut keepalive,
            )?;
            out_tokens.push(token);
        }
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(out_tokens)
    }

    fn forward_decode_batch_stream_impl(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        tokens: &[u32],
        start_positions: &[usize],
    ) -> Result<(HiddenStates, Dsv4ForwardKeepalive)> {
        let n = slot_ids.len();
        for r in 0..n {
            let slot = &slots[slot_ids[r]];
            ensure!(
                slot.seq_len == start_positions[r],
                "DSv4 batched decode slot {} seq_len {} != start_pos {}; decode requires contiguous appends",
                slot_ids[r],
                slot.seq_len,
                start_positions[r]
            );
            ensure!(
                start_positions[r] + 1 <= slot.max_seq_len,
                "DSv4 batched decode slot {} sequence {} exceeds max_seq_len {}",
                slot_ids[r],
                start_positions[r] + 1,
                slot.max_seq_len
            );
        }

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = n; // batch dimension: N independent decode rows
        let eps = self.config.rms_norm_eps;
        let use_deepep_transport = dsv4_use_deepep_transport()?;
        // N>1: mirror the prefill keepalive discipline (the per-token decode
        // scratch / comm-overlap fast paths are seq_len==1 only).
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let ctx = &self.ctx;

        // Per-slot decode position scalars (each row's attention reads its own).
        for r in 0..n {
            let start_pos_i32 = i32::try_from(start_positions[r])
                .map_err(|_| anyhow!("DSv4 start_pos {} overflows i32", start_positions[r]))?;
            let slot = &mut slots[slot_ids[r]];
            ctx.stream
                .memcpy_htod(&[start_pos_i32], &mut slot.start_pos_device)
                .map_err(|e| anyhow!("DSv4 batched start_pos H2D failed: {e}"))?;
        }

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let nvtx_embed = crate::nvtx::range("dsv4/embed");
        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids_host)?;
        keepalive.keep_i32(&token_ids);
        // SAFETY: embedding_batch writes the full [seq_len, hidden_size] buffer.
        let mut embeddings = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
        crate::ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut embeddings)?;
        keepalive.keep_hidden(&embeddings);
        // SAFETY: initial_stream_from_embeddings writes the full stream buffer.
        let mut stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
        crate::hc::initial_stream_from_embeddings(
            &self.ctx,
            &embeddings,
            hidden_size,
            hc_mult,
            &mut stream,
        )?;
        keepalive.keep_hidden(&stream);
        drop(nvtx_embed);

        // Reusable [hidden,1] scratch for the per-row attention copy-in/out.
        // Declared once (lives to function return → stream-ordered free, like the
        // single-row intermediates); reuse across rows/layers is safe because all
        // ops run on `ctx.stream` (WAR/RAW resolved by stream ordering).
        // SAFETY: fully written by the copy-in / mla_attention each row before read.
        let mut normed_row = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
        let mut attn_out_row = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
        keepalive.keep_hidden(&normed_row);
        keepalive.keep_hidden(&attn_out_row);
        // MoE/shared are now grouped over [N] (Phase 6a) — no per-row MoE scratch.

        // Localization probe: all rows have identical inputs, so every intermediate
        // must be row-identical. Print the first (layer, half) where rows diverge.
        let probe = std::env::var_os("INFER_DSV4_BATCH_PROBE").is_some()
            && self.tp.config().rank == 0
            && n >= 2;
        let probe_rows = |label: &str, hs: &HiddenStates, layer_idx: usize| -> Result<()> {
            if !probe {
                return Ok(());
            }
            // FULL-VECTOR max-abs-diff of each row vs row0. All rows share an
            // identical prompt, so at decode step 1 every element must be
            // bit-identical. elem0-only is blind: hc_pre mixes hc_mult lanes and
            // rms_norm reduces over the whole hidden vector, so a divergence at
            // ANY element propagates. This finds the true first (layer, stage,
            // element) of divergence.
            let dim = hs.hidden_dim;
            let host: Vec<half::bf16> = self
                .ctx
                .stream
                .clone_dtoh(&hs.data)
                .map_err(|e| anyhow!("DSv4 batch probe D2H failed: {e}"))?;
            let mut global_max = 0.0f32;
            let mut worst_r = 0usize;
            let mut worst_i = 0usize;
            let per_row: Vec<String> = (1..n)
                .map(|r| {
                    let mut m = 0.0f32;
                    for i in 0..dim {
                        let d = (host[r * dim + i].to_f32() - host[i].to_f32()).abs();
                        if d > m {
                            m = d;
                        }
                        if d > global_max {
                            global_max = d;
                            worst_r = r;
                            worst_i = i;
                        }
                    }
                    format!("{m:.6}")
                })
                .collect();
            // Report the first 3 layers always (to confirm clean baseline), plus
            // any later stage that diverges above bf16 round-trip noise.
            if (layer_idx < 3 && start_positions[0] == 5) || global_max > 1e-4 {
                eprintln!(
                    "[batch-vdiff] start_pos={} L{layer_idx} {label}: maxdiff_vs_row0=[{}] worst(r={worst_r},i={worst_i})={global_max:.6}",
                    start_positions[0],
                    per_row.join(", "),
                );
            }
            Ok(())
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
            // ── Attention half: HC params + pre + norm over the whole [N] batch.
            let mhc = crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_attn, &stream)?;
            keepalive.keep_f32(&mhc.pre);
            keepalive.keep_f32(&mhc.post);
            keepalive.keep_f32(&mhc.comb);
            // SAFETY: fused hc_pre+rms_norm writes the full [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::hc::mhc_pre_rms_norm(
                &self.ctx,
                &stream,
                &mhc.pre,
                &layer.attn_norm,
                eps,
                hidden_size,
                hc_mult,
                &mut normed,
            )?;
            keepalive.keep_hidden(&normed);
            probe_rows("norm_in", &normed, layer_idx)?;

            // ── Attention: per-row independent single-token MLA into row r.
            // SAFETY: every [r*hidden, (r+1)*hidden) span of attn_out is written by
            // the copy-out below before attn_out is read by hc_post.
            let mut attn_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            {
                let _nvtx = crate::nvtx::range("dsv4/mla_attn_batched");
                for r in 0..n {
                    let src = normed.data.slice(r * hidden_size..(r + 1) * hidden_size);
                    ctx.stream
                        .memcpy_dtod(&src, &mut normed_row.data)
                        .map_err(|e| anyhow!("DSv4 batched attn copy-in failed: {e}"))?;
                    let (layer_pool, dsa_shared) =
                        kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                    let slot = &mut slots[slot_ids[r]];
                    crate::attention::mla_attention(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer.mode,
                        layer.compress_ratio,
                        layer_idx,
                        &normed_row,
                        &mut slot.attention[layer_idx],
                        layer_pool,
                        dsa_shared,
                        start_positions[r],
                        Some(&slot.start_pos_device),
                        &self.tp,
                        &mut attn_out_row,
                        &mut keepalive,
                    )?;
                    let mut dst = attn_out
                        .data
                        .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                    ctx.stream
                        .memcpy_dtod(&attn_out_row.data, &mut dst)
                        .map_err(|e| anyhow!("DSv4 batched attn copy-out failed: {e}"))?;
                    // No per-row host sync: every op (memcpy, mla_attention's
                    // FlashMLA/compressor/indexer FFI, copy-out) runs on ctx.stream,
                    // so stream ordering already serializes row r's reads of the
                    // shared {normed,attn_out}_row scratch before row r+1's writes
                    // (WAR resolved by stream order). The sync was debug isolation.
                }
            }
            keepalive.keep_hidden(&attn_out);
            // Probe the RAW per-row attention output (pre all-reduce, pre hc_post):
            // this is a per-row single-column kernel, so identical inputs MUST give
            // bit-identical rows. Divergence here = real logic bug, not numerics.
            probe_rows("attn_raw", &attn_out, layer_idx)?;
            // Row-parallel O-LoRA: one all-reduce over [N, hidden]. NOT bit-identical
            // to N per-row all-reduces: NCCL tiles a [hidden,N] message differently
            // than N×[hidden,1], so identical-input rows pick up ~1 bf16 ULP of
            // per-row drift here. This is the legitimate batched-numerics seed.
            {
                let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?;
            }
            // Probe POST-all-reduce (pre hc_post): isolates the all-reduce as the
            // divergence seed vs the per-token hc_post (which cannot cross rows).
            probe_rows("attn_ar", &attn_out, layer_idx)?;
            // SAFETY: hc_post writes the full stream buffer.
            let mut attn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            crate::hc::hc_post(
                &self.ctx,
                &attn_out,
                &stream,
                &mhc.post,
                &mhc.comb,
                hidden_size,
                hc_mult,
                &mut attn_stream,
            )?;
            keepalive.keep_hidden(&attn_stream);
            stream = attn_stream;
            probe_rows("attn", &stream, layer_idx)?;

            // ── MoE half: HC-wrap the grouped FP8 DeepGEMM MoE over the [N] batch.
            let mhc = crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)?;
            keepalive.keep_f32(&mhc.pre);
            keepalive.keep_f32(&mhc.post);
            keepalive.keep_f32(&mhc.comb);
            // SAFETY: fused hc_pre+rms_norm writes the full [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::hc::mhc_pre_rms_norm(
                &self.ctx,
                &stream,
                &mhc.pre,
                &layer.ffn_norm,
                eps,
                hidden_size,
                hc_mult,
                &mut normed,
            )?;
            keepalive.keep_hidden(&normed);
            // Routed MoE over the whole [N] batch. allreduce transport: Phase 6a
            // grouped path (one router gemm + one DeepGEMM grouped expert GEMM
            // over N×topk routes, decode_scratch=None) + one EP all-reduce.
            // deepep transport: the token-owned LL / intranode pipelines, which
            // are natively [N]-batched — same owned-slice structure as the
            // single-row forward (see its collective-participation invariants).
            // Bit-identity vs per-row is NOT expected (grouped GEMM / LL combine
            // tile over N differently); gated on needle retrieval, not byte-parity.
            let mut moe_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if use_deepep_transport {
                #[cfg(feature = "deepep")]
                {
                    let transport = self.deepep.as_ref().ok_or_else(|| {
                        anyhow!("ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted")
                    })?;
                    if transport.is_low_latency() {
                        // Token-owned LL path: this rank owns the contiguous
                        // token cols [start..end) of the replicated `normed`.
                        // Every rank participates in the dispatch/combine
                        // COLLECTIVES even with owned_n == 0 (seq_len < world).
                        let world = self.tp.config().world_size;
                        let rank = self.tp.config().rank;
                        let per = seq_len.div_ceil(world);
                        let start = (rank * per).min(seq_len);
                        let end = ((rank + 1) * per).min(seq_len);
                        let owned_n = end - start;
                        // Zero the full output; each rank scatters its owned
                        // cols, then an all-reduce gathers them (replacing the
                        // moe all-reduce — DeepEP combine already EP-reduced).
                        self.ctx
                            .stream
                            .memset_zeros(&mut moe_out.data)
                            .map_err(|e| anyhow!("deepep_ll batched moe_out zero failed: {e}"))?;
                        let mut owned_in =
                            HiddenStates::zeros(&self.ctx, hidden_size, owned_n.max(1))?;
                        owned_in.seq_len = owned_n;
                        keepalive.keep_hidden(&owned_in);
                        if owned_n > 0 {
                            self.ctx
                                .stream
                                .memcpy_dtod(
                                    &normed.data.slice(start * hidden_size..end * hidden_size),
                                    &mut owned_in.data.slice_mut(0..owned_n * hidden_size),
                                )
                                .map_err(|e| {
                                    anyhow!("deepep_ll batched owned-slice copy failed: {e}")
                                })?;
                        }
                        let mut owned_out =
                            HiddenStates::zeros(&self.ctx, hidden_size, owned_n.max(1))?;
                        owned_out.seq_len = owned_n;
                        keepalive.keep_hidden(&owned_out);
                        // The LL scratch is whole-forward scratch (fully
                        // overwritten per call); it is parked per-slot for the
                        // single-row path, so the batch borrows row 0's.
                        let scratch =
                            slots[slot_ids[0]]
                                .deepep_ll_scratch
                                .as_mut()
                                .ok_or_else(|| {
                                    anyhow!("deepep_ll selected but slot LL scratch not allocated")
                                })?;
                        crate::moe::dsv4_moe_forward_deepep_ll(
                            self,
                            transport,
                            scratch,
                            &layer.moe,
                            &tokens[start..end],
                            tokens.len(),
                            &owned_in,
                            &mut owned_out,
                            &mut keepalive,
                        )?;
                        if owned_n > 0 {
                            self.ctx
                                .stream
                                .memcpy_dtod(
                                    &owned_out.data.slice(0..owned_n * hidden_size),
                                    &mut moe_out
                                        .data
                                        .slice_mut(start * hidden_size..end * hidden_size),
                                )
                                .map_err(|e| {
                                    anyhow!("deepep_ll batched owned scatter failed: {e}")
                                })?;
                        }
                        // All-gather via all-reduce: only owned cols are nonzero.
                        self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
                    } else {
                        // Intranode normal-mode DeepEP: already [N]-shaped; its
                        // combine reduces across EP, no moe all-reduce needed.
                        crate::moe::dsv4_moe_forward_deepep(
                            self,
                            transport,
                            &layer.moe,
                            tokens,
                            &normed,
                            &mut moe_out,
                            &mut keepalive,
                        )?;
                    }
                }
                #[cfg(not(feature = "deepep"))]
                bail!("ARLE_DSV4_MOE_TRANSPORT=deepep requires infer-cuda feature deepep");
            } else {
                crate::moe::dsv4_moe_forward(
                    self,
                    &layer.moe,
                    tokens,
                    &normed,
                    &mut moe_out,
                    None,
                    &mut keepalive,
                )?;
                keepalive.keep_hidden(&moe_out);
                // Probe the RAW per-row MoE output (pre all-reduce): like attn_raw,
                // the per-row single-token MoE must be bit-identical across
                // identical rows. (allreduce arm only — the deepep arm's pre-gather
                // buffer is owned-cols-sparse by construction.)
                probe_rows("moe_raw", &moe_out, layer_idx)?;
                // Routed experts are EP-sharded → sum, then add the replicated
                // shared expert once per rank. One all-reduce over [N, hidden].
                let _nvtx = crate::nvtx::range("dsv4/moe_allreduce");
                self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
            }
            keepalive.keep_hidden(&moe_out);
            probe_rows("moe_ar", &moe_out, layer_idx)?;
            // Phase 6a: grouped shared expert over [N] (dense FFN, prefill path,
            // decode_scratch=None) — one batched SwiGLU GEMM pair, replacing the
            // per-row loop + N host syncs.
            let mut shared = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::moe::dsv4_shared_expert_forward(
                &self.ctx,
                &self.ctx.stream,
                &layer.moe,
                &normed,
                &mut shared,
                self.config.swiglu_limit,
                None,
                &mut keepalive,
            )?;
            keepalive.keep_hidden(&shared);
            // SAFETY: add_batch writes the full [seq_len, hidden_size] buffer.
            let mut moe_with_shared =
                unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::ops::add_batch(&self.ctx, &moe_out, &shared, &mut moe_with_shared)?;
            keepalive.keep_hidden(&moe_with_shared);
            // SAFETY: hc_post writes the full stream buffer.
            let mut ffn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            crate::hc::hc_post(
                &self.ctx,
                &moe_with_shared,
                &stream,
                &mhc.post,
                &mhc.comb,
                hidden_size,
                hc_mult,
                &mut ffn_stream,
            )?;
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
            probe_rows("moe", &stream, layer_idx)?;
        }

        for r in 0..n {
            slots[slot_ids[r]].seq_len += 1;
        }
        Ok((stream, keepalive))
    }

    fn capture_mtp_stream_hidden(
        &self,
        stream: &HiddenStates,
        row: usize,
        out: &mut DeviceVec,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let stream_dim = self.config.hidden_size * self.config.hc_mult;
        ensure!(
            stream.hidden_dim == stream_dim,
            "DSv4 MTP hidden source stream dim {} != hidden_size {} * hc_mult {}",
            stream.hidden_dim,
            self.config.hidden_size,
            self.config.hc_mult
        );
        ensure!(
            out.len == stream_dim,
            "DSv4 MTP hidden capture len {} != stream_dim {stream_dim}",
            out.len
        );
        crate::ops::copy_row_to_vec(&self.ctx, stream, row, out)?;
        keepalive.keep_vec(out);
        Ok(())
    }

    fn forward_stream_last_token(
        &self,
        stream: &HiddenStates,
        seq_len: usize,
        params: &SamplingParams,
        position: u64,
        last_hidden_out: Option<&mut DeviceVec>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<u32> {
        ensure!(seq_len > 0, "DSv4 head stage requires seq_len > 0");
        let hidden_size = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let ctx = &self.ctx;

        let _nvtx = crate::nvtx::range("dsv4/lm_head_sample");
        // ── Head HC: fold the last token's wide stream row → one hidden vector.
        let mut last_hidden = DeviceVec::zeros(ctx, hidden_size)?;
        crate::stage_profile::profile(ctx, "dsv4/stage/head_hc", || {
            crate::hc::head_hidden_from_stream(
                ctx,
                &self.config,
                &self.head_hc,
                stream,
                seq_len - 1,
                &mut last_hidden,
            )
        })?;
        keepalive.keep_hidden(stream);
        keepalive.keep_vec(&last_hidden);
        if let Some(out) = last_hidden_out {
            ensure!(
                out.len == hidden_size,
                "DSv4 hidden capture len {} != hidden_size {hidden_size}",
                out.len
            );
            ctx.stream
                .memcpy_dtod(&last_hidden.data, &mut out.data)
                .map_err(|e| anyhow!("DSv4 hidden capture D2D failed: {e}"))?;
            keepalive.keep_vec(out);
        }

        // ── Final norm + lm_head projection + sample (last token row).
        let mut last_normed = DeviceVec::zeros(ctx, hidden_size)?;
        crate::stage_profile::profile(ctx, "dsv4/stage/head_norm", || {
            crate::ops::rms_norm_vec(ctx, &last_hidden, &self.norm, eps, &mut last_normed)
        })?;
        keepalive.keep_vec(&last_normed);
        let mut logits = DeviceVec::zeros(ctx, self.lm_head.rows)?;
        crate::stage_profile::profile(ctx, "dsv4/stage/lm_head_project", || {
            self.lm_head_project(&last_normed, &mut logits)
        })?;
        keepalive.keep_vec(&logits);
        crate::stage_profile::profile(ctx, "dsv4/stage/sample", || {
            crate::executor::sample_cuda_token(ctx, &logits, params, position)
        })
    }

    // col1-bug pinpoint (2026-06-08, ARLE_DSV4_TAIL_DUMP): for the capital selftest
    // (start_pos=5), the batched verify token-1 (sp=5 seq=2) vs the per-token reference
    // (sp=6 seq=1) are BIT-IDENTICAL at init_stream + attn_in_L0, then DIVERGE at
    // attn_out_L0 (L2 144.86 vs 145.18, concentrated in the first/RoPE dims). So the
    // chunked-verify col1 bug is in the SWA attention for the chunk's 2nd token — NOT
    // embed/HC/MoE. Inputs (token_a key k_new[0] vs ring[5], history, query, sink, the
    // abs_pos inverse-RoPE) all appear to match on read; next dump is k_new[0] vs
    // ring[5] for token_a to confirm whether the prepare/store path differs.
    /// Debug: dump the TAIL row (seq_len-1) of `h` — L2 + first4 — gated by
    /// ARLE_DSV4_TAIL_DUMP, rank 0 only. The tail row is the LAST token, so it is
    /// directly comparable between a batched [a,b] forward (row 1) and a per-token
    /// [b]@start_pos+1 reference (row 0) — used to localize the batched-verify col1
    /// bug to the first diverging stage. Syncs (debug only; not on the hot path).
    fn dump_tail_row(&self, label: &str, h: &HiddenStates, start_pos: usize) {
        if self.tp.config().rank != 0 || std::env::var_os("ARLE_DSV4_TAIL_DUMP").is_none() {
            return;
        }
        if self.ctx.sync().is_err() {
            return;
        }
        let host: Vec<half::bf16> = match self.ctx.stream.clone_dtoh(&h.data) {
            Ok(v) => v,
            Err(_) => return,
        };
        let row = h.seq_len.saturating_sub(1);
        let base = row * h.hidden_dim;
        let mut l2 = 0.0f32;
        for i in 0..h.hidden_dim {
            let x = host[base + i].to_f32();
            l2 += x * x;
        }
        let first4: Vec<f32> = (0..4.min(h.hidden_dim))
            .map(|i| host[base + i].to_f32())
            .collect();
        eprintln!(
            "[tail-dump] {label} sp={start_pos} seq={} dim={} tailrow={row} l2={:.5} first4={first4:?}",
            h.seq_len,
            h.hidden_dim,
            l2.sqrt()
        );
    }

    fn forward_tokens_stream_impl(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        // When true (MTP batched verify only), run attention PER TOKEN on the
        // seq_len==1 decode path (device start_pos → FlashMLA/DSA decode), in
        // order so token r+1 attends to token r's just-written KV — point-wise/
        // MoE stay batched for the weight-read amortization. The batched
        // (host-start_pos) compressed/DSA path is incorrect for a small chunk at
        // a fully-populated mid-sequence position; prefill keeps it (false).
        per_token_attn: bool,
    ) -> Result<(HiddenStates, Dsv4ForwardKeepalive)> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 forward requires at least one token"
        );
        ensure!(
            slot.seq_len == start_pos,
            "DSv4 slot seq_len {} != start_pos {start_pos}; decode requires contiguous appends",
            slot.seq_len
        );
        ensure!(
            start_pos + tokens.len() <= slot.max_seq_len,
            "DSv4 sequence {} exceeds slot max_seq_len {}",
            start_pos + tokens.len(),
            slot.max_seq_len
        );

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let use_gpu_router = std::env::var_os("ARLE_DSV4_GPU_ROUTER").is_some();
        let use_deepep_transport = dsv4_use_deepep_transport()?;
        let use_moe_decode_scratch = use_gpu_router && seq_len == 1 && !use_deepep_transport;
        let mut keepalive = Dsv4ForwardKeepalive::new(seq_len == 1);
        let ctx = &self.ctx;
        let start_pos_device = if seq_len == 1 {
            let start_pos_i32 = i32::try_from(start_pos)
                .map_err(|_| anyhow!("DSv4 start_pos {start_pos} overflows i32"))?;
            self.ctx
                .stream
                .memcpy_htod(&[start_pos_i32], &mut slot.start_pos_device)
                .map_err(|e| anyhow!("DSv4 start_pos H2D failed: {e}"))?;
            Some(&slot.start_pos_device)
        } else {
            None
        };

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let nvtx_embed = crate::nvtx::range("dsv4/embed");
        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids_host)?;
        keepalive.keep_i32(&token_ids);
        // SAFETY: embedding_batch writes the full [seq_len, hidden_size] buffer.
        let mut embeddings = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
        crate::stage_profile::profile(ctx, "dsv4/stage/embed", || {
            crate::ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut embeddings)
        })?;
        keepalive.keep_hidden(&embeddings);

        // Wide HC residual stream from the token embeddings.
        // SAFETY: initial_stream_from_embeddings writes the full stream buffer.
        let mut stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
        crate::stage_profile::profile(ctx, "dsv4/stage/embed_hc_expand", || {
            crate::hc::initial_stream_from_embeddings(
                &self.ctx,
                &embeddings,
                hidden_size,
                hc_mult,
                &mut stream,
            )
        })?;
        keepalive.keep_hidden(&stream);
        drop(nvtx_embed);
        self.dump_tail_row("init_stream", &stream, start_pos);
        // Per-token verify attention scratch (one [hidden,1] row), reused across
        // layers. Allocated only on the verify path; keepalive'd against the
        // disabled-event-tracking premature-free hazard.
        // Separate device start_pos buffer (the function-level `start_pos_device`
        // immutably borrows `slot.start_pos_device` for the whole fn, so the loop
        // needs its own).
        let (mut normed_row, mut attn_out_row, mut verify_pos_dev) = if per_token_attn {
            let nr = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
            let ar = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
            let pos = self
                .ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow!("DSv4 verify pos buffer alloc failed: {e}"))?;
            keepalive.keep_hidden(&nr);
            keepalive.keep_hidden(&ar);
            (Some(nr), Some(ar), Some(pos))
        } else {
            (None, None, None)
        };
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
            // ── Attention half: HC-wrap MLA attention.
            let mhc = crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_params", || {
                crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_attn, &stream)
            })?;
            keepalive.keep_f32(&mhc.pre);
            keepalive.keep_f32(&mhc.post);
            keepalive.keep_f32(&mhc.comb);
            // SAFETY: hc_pre writes the full [seq_len, hidden_size] buffer.
            // SAFETY: fused hc_pre+rms_norm writes the full [seq_len, hidden_size]
            // buffer. (The old attn_in_L0 tail dump went with the intermediate.)
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_pre_norm", || {
                crate::hc::mhc_pre_rms_norm(
                    &self.ctx,
                    &stream,
                    &mhc.pre,
                    &layer.attn_norm,
                    eps,
                    hidden_size,
                    hc_mult,
                    &mut normed,
                )
            })?;
            keepalive.keep_hidden(&normed);
            // SAFETY: mla_attention writes the full [seq_len, hidden_size] buffer.
            let mut attn_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if per_token_attn && seq_len > 1 {
                // MTP verify: attention PER TOKEN on the seq_len==1 decode path
                // (device start_pos), in order so token r+1 attends to token r's
                // just-written KV. Mirrors forward_decode_batch's Step-A loop and
                // the (correct) non-batched verify; avoids the broken batched
                // host-start_pos compressed/DSA path at a mid-sequence chunk.
                let _nvtx = crate::nvtx::range("dsv4/mla_attn_per_token");
                let normed_row = normed_row.as_mut().expect("per-token verify scratch");
                let attn_out_row = attn_out_row.as_mut().expect("per-token verify scratch");
                let verify_pos_dev = verify_pos_dev.as_mut().expect("per-token verify scratch");
                let (layer_pool, mut dsa_shared) =
                    kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                for r in 0..seq_len {
                    let pos_r = start_pos + r;
                    let pos_r_i32 = i32::try_from(pos_r)
                        .map_err(|_| anyhow!("DSv4 verify pos {pos_r} overflows i32"))?;
                    self.ctx
                        .stream
                        .memcpy_htod(&[pos_r_i32], verify_pos_dev)
                        .map_err(|e| anyhow!("DSv4 verify start_pos H2D failed: {e}"))?;
                    let src = normed.data.slice(r * hidden_size..(r + 1) * hidden_size);
                    self.ctx
                        .stream
                        .memcpy_dtod(&src, &mut normed_row.data)
                        .map_err(|e| anyhow!("DSv4 verify attn copy-in failed: {e}"))?;
                    crate::attention::mla_attention(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer.mode,
                        layer.compress_ratio,
                        layer_idx,
                        &*normed_row,
                        &mut slot.attention[layer_idx],
                        layer_pool,
                        dsa_shared.as_deref_mut(),
                        pos_r,
                        Some(&*verify_pos_dev),
                        &self.tp,
                        attn_out_row,
                        &mut keepalive,
                    )?;
                    let mut dst = attn_out
                        .data
                        .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                    self.ctx
                        .stream
                        .memcpy_dtod(&attn_out_row.data, &mut dst)
                        .map_err(|e| anyhow!("DSv4 verify attn copy-out failed: {e}"))?;
                }
            } else {
                let _nvtx = crate::nvtx::range("dsv4/mla_attn");
                crate::stage_profile::profile(ctx, "dsv4/stage/mla_attn", || {
                    let (layer_pool, dsa_shared) =
                        kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                    crate::attention::mla_attention(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer.mode,
                        layer.compress_ratio,
                        layer_idx,
                        &normed,
                        &mut slot.attention[layer_idx],
                        layer_pool,
                        dsa_shared,
                        start_pos,
                        start_pos_device,
                        &self.tp,
                        &mut attn_out,
                        &mut keepalive,
                    )
                })?;
            }
            keepalive.keep_hidden(&attn_out);
            if layer_idx == 0 {
                self.dump_tail_row("attn_out_pre_ar_L0", &attn_out, start_pos);
            }
            // Row-parallel O-LoRA: sum the per-rank partials (no-op single-GPU).
            {
                let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                crate::stage_profile::profile(ctx, "dsv4/stage/attn_allreduce", || {
                    self.tp.all_reduce_sum(&self.ctx, &mut attn_out)
                })?;
            }
            if layer_idx == 0 {
                self.dump_tail_row("attn_out_L0", &attn_out, start_pos);
            }
            // SAFETY: hc_post writes the full stream buffer.
            let mut attn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_post", || {
                crate::hc::hc_post(
                    &self.ctx,
                    &attn_out,
                    &stream,
                    &mhc.post,
                    &mhc.comb,
                    hidden_size,
                    hc_mult,
                    &mut attn_stream,
                )
            })?;
            keepalive.keep_hidden(&attn_stream);
            stream = attn_stream;

            // ── MoE half: HC-wrap the FP8 DeepGEMM MoE block.
            let mhc = crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_params", || {
                crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)
            })?;
            keepalive.keep_f32(&mhc.pre);
            keepalive.keep_f32(&mhc.post);
            keepalive.keep_f32(&mhc.comb);
            // SAFETY: hc_pre writes the full [seq_len, hidden_size] buffer.
            // SAFETY: fused hc_pre+rms_norm writes the full [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_pre_norm", || {
                crate::hc::mhc_pre_rms_norm(
                    &self.ctx,
                    &stream,
                    &mhc.pre,
                    &layer.ffn_norm,
                    eps,
                    hidden_size,
                    hc_mult,
                    &mut normed,
                )
            })?;
            keepalive.keep_hidden(&normed);
            let use_comm_overlap =
                dsv4_comm_overlap_enabled() && seq_len == 1 && !use_deepep_transport;
            let normed_ready = if use_comm_overlap {
                let fence = ctx.record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
                ctx.wait_on_pipeline_fence(&fence, CudaPipelineStreamKind::Comm)?;
                Some(fence)
            } else {
                None
            };
            // SAFETY: the MoE forward writes the full routed output buffer.
            let mut moe_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            // DeepEP combine already reduces the EP-sharded routed output; the
            // non-deepep path needs the explicit TP all-reduce below.
            let needs_moe_allreduce = !use_deepep_transport;
            if use_deepep_transport {
                #[cfg(feature = "deepep")]
                {
                    let transport = self.deepep.as_ref().ok_or_else(|| {
                        anyhow!("ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted")
                    })?;
                    if transport.is_low_latency() {
                        // ── deepep_ll token-owned path ──────────────────────
                        // TP8 replicates `normed` [hidden, n]; HiddenStates is
                        // token-major (token i at i*hidden), so this rank's owned
                        // shard cols [start..end] is a CONTIGUOUS byte range.
                        let world = self.tp.config().world_size;
                        let rank = self.tp.config().rank;
                        let per = seq_len.div_ceil(world);
                        let start = (rank * per).min(seq_len);
                        let end = ((rank + 1) * per).min(seq_len);
                        let owned_n = end - start;
                        // Zero the full output; every rank scatters its owned cols
                        // then an all-reduce gathers them (replaces the moe AR).
                        self.ctx
                            .stream
                            .memset_zeros(&mut moe_out.data)
                            .map_err(|e| anyhow!("deepep_ll moe_out zero failed: {e}"))?;
                        // The LL dispatch + combine are COLLECTIVES: EVERY rank must
                        // participate every step or the symmetric protocol deadlocks
                        // (DeepEP "timeout for dispatch receive"). A rank that owns 0
                        // tokens (when seq_len < world) still dispatches 0 tokens and
                        // runs the masked GEMMs over the tokens routed to ITS local
                        // experts. So always call the LL forward — it internally
                        // handles owned_n == 0 — and only scatter when owned_n > 0.
                        let scratch = slot.deepep_ll_scratch.as_mut().ok_or_else(|| {
                            anyhow!("deepep_ll selected but slot LL scratch not allocated")
                        })?;
                        // Copy this rank's owned contiguous columns of `normed`
                        // into a compact `[hidden, owned_n]` buffer (the LL dispatch
                        // needs a standalone `[owned_n, hidden]` input; `.slice()`
                        // yields a borrowed CudaView, not an owned CudaSlice, so we
                        // materialize it — same one-copy pattern as the per-row
                        // attention slab path above). owned_n may be 0.
                        let mut owned_in =
                            HiddenStates::zeros(&self.ctx, hidden_size, owned_n.max(1))?;
                        owned_in.seq_len = owned_n;
                        keepalive.keep_hidden(&owned_in);
                        if owned_n > 0 {
                            self.ctx
                                .stream
                                .memcpy_dtod(
                                    &normed.data.slice(start * hidden_size..end * hidden_size),
                                    &mut owned_in.data.slice_mut(0..owned_n * hidden_size),
                                )
                                .map_err(|e| anyhow!("deepep_ll owned-slice copy failed: {e}"))?;
                        }
                        let mut owned_out =
                            HiddenStates::zeros(&self.ctx, hidden_size, owned_n.max(1))?;
                        owned_out.seq_len = owned_n;
                        keepalive.keep_hidden(&owned_out);
                        crate::moe::dsv4_moe_forward_deepep_ll(
                            self,
                            transport,
                            scratch,
                            &layer.moe,
                            &tokens[start..end],
                            tokens.len(),
                            &owned_in,
                            &mut owned_out,
                            &mut keepalive,
                        )?;
                        if owned_n > 0 {
                            // Scatter owned_out into moe_out's owned cols (contiguous).
                            self.ctx
                                .stream
                                .memcpy_dtod(
                                    &owned_out.data.slice(0..owned_n * hidden_size),
                                    &mut moe_out
                                        .data
                                        .slice_mut(start * hidden_size..end * hidden_size),
                                )
                                .map_err(|e| {
                                    anyhow!("deepep_ll owned scatter into moe_out failed: {e}")
                                })?;
                        }
                        // All-gather via all-reduce: each rank contributed only its
                        // owned cols (rest zero), so the sum is the full gather.
                        // This REPLACES the moe all-reduce (needs_moe_allreduce=false).
                        self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
                    } else {
                        crate::moe::dsv4_moe_forward_deepep(
                            self,
                            transport,
                            &layer.moe,
                            tokens,
                            &normed,
                            &mut moe_out,
                            &mut keepalive,
                        )?;
                    }
                }
                #[cfg(not(feature = "deepep"))]
                {
                    bail!("ARLE_DSV4_MOE_TRANSPORT=deepep requires infer-cuda feature deepep");
                }
            } else {
                let decode_scratch = if use_moe_decode_scratch {
                    Some(&mut slot.moe_decode_scratch[layer_idx])
                } else {
                    None
                };
                crate::moe::dsv4_moe_forward(
                    self,
                    &layer.moe,
                    tokens,
                    &normed,
                    &mut moe_out,
                    decode_scratch,
                    &mut keepalive,
                )?;
            }
            keepalive.keep_hidden(&moe_out);
            let nvtx_shared_hc = crate::nvtx::range("dsv4/shared_hc");
            let mut shared_opt = None;
            let shared_ready = if use_comm_overlap {
                let len = hidden_size * seq_len;
                let data = unsafe {
                    self.ctx
                        .comm_stream
                        .alloc::<half::bf16>(len)
                        .map_err(|e| anyhow!("DSv4 shared expert comm-stream alloc failed: {e}"))?
                };
                let mut shared = HiddenStates {
                    data,
                    hidden_dim: hidden_size,
                    seq_len,
                };
                let _normed_ready = normed_ready
                    .as_ref()
                    .expect("comm-overlap path records normed fence");
                crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                    crate::moe::dsv4_shared_expert_forward(
                        &self.ctx,
                        &self.ctx.comm_stream,
                        &layer.moe,
                        &normed,
                        &mut shared,
                        self.config.swiglu_limit,
                        if use_moe_decode_scratch {
                            Some(&mut slot.moe_decode_scratch[layer_idx])
                        } else {
                            None
                        },
                        &mut keepalive,
                    )
                })?;
                shared_opt = Some(shared);
                Some(ctx.record_pipeline_fence(CudaPipelineStreamKind::Comm)?)
            } else {
                None
            };
            // Routed experts are EP-sharded; sum them first, then add the replicated
            // shared expert exactly once per rank. In the comm-overlap path, shared
            // expert depends on `normed`, while this compute-stream collective depends
            // on the routed `moe_out`; the two can run concurrently.
            if needs_moe_allreduce {
                let _nvtx = crate::nvtx::range("dsv4/moe_allreduce");
                crate::stage_profile::profile(ctx, "dsv4/stage/moe_allreduce", || {
                    self.tp.all_reduce_sum(&self.ctx, &mut moe_out)
                })?;
            }
            let shared = if let Some(shared) = shared_opt {
                shared
            } else {
                // Keep the default masked path order byte-for-byte with main:
                // routed MoE → all-reduce → allocate/run shared expert. Moving
                // shared scratch allocation before all-reduce can reuse in-flight
                // helper buffers under disabled cudarc event tracking.
                // SAFETY: dsv4_shared_expert_forward writes the full shared output.
                let mut shared = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                    crate::moe::dsv4_shared_expert_forward(
                        &self.ctx,
                        &self.ctx.stream,
                        &layer.moe,
                        &normed,
                        &mut shared,
                        self.config.swiglu_limit,
                        if use_moe_decode_scratch {
                            Some(&mut slot.moe_decode_scratch[layer_idx])
                        } else {
                            None
                        },
                        &mut keepalive,
                    )
                })?;
                shared
            };
            keepalive.keep_hidden(&shared);
            if let Some(fence) = shared_ready.as_ref() {
                ctx.wait_on_pipeline_fence(fence, CudaPipelineStreamKind::Compute)?;
            }
            // SAFETY: add_batch writes the full [seq_len, hidden_size] buffer.
            let mut moe_with_shared =
                unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::stage_profile::profile(ctx, "dsv4/stage/shared_add", || {
                crate::ops::add_batch(&self.ctx, &moe_out, &shared, &mut moe_with_shared)
            })?;
            keepalive.keep_hidden(&moe_with_shared);
            // SAFETY: hc_post writes the full stream buffer.
            let mut ffn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_post", || {
                crate::hc::hc_post(
                    &self.ctx,
                    &moe_with_shared,
                    &stream,
                    &mhc.post,
                    &mhc.comb,
                    hidden_size,
                    hc_mult,
                    &mut ffn_stream,
                )
            })?;
            drop(nvtx_shared_hc);
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
        }

        slot.seq_len += seq_len;
        Ok((stream, keepalive))
    }

    pub(crate) fn mtp_forward(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        h_prev: &DeviceVec,
        next_token: u32,
        position: u64,
    ) -> Result<(u32, DeviceVec)> {
        ensure!(
            dsv4_spec_decode_enabled(),
            "DSv4 MTP forward called while ARLE_DSV4_SPEC_DECODE is disabled"
        );
        ensure!(
            !dsv4_use_deepep_transport()?,
            "DSv4 MTP Phase 1 supports allreduce transport only"
        );
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow!("ARLE_DSV4_SPEC_DECODE=1 but DSv4 MTP head is not loaded"))?;
        let stream_dim = self.config.hidden_size * self.config.hc_mult;
        ensure!(
            h_prev.len == stream_dim,
            "DSv4 MTP h_prev len {} != hidden_size {} * hc_mult {}",
            h_prev.len,
            self.config.hidden_size,
            self.config.hc_mult
        );

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let eps = self.config.rms_norm_eps;
        let ctx = &self.ctx;

        let token_ids = crate::ops::upload_i32(ctx, &[next_token as i32])?;
        // SAFETY: embedding_batch writes the full [1, hidden_size] row.
        let mut emb = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
        crate::ops::embedding_batch(ctx, &self.embed_tokens, &token_ids, &mut emb)?;

        // h' = e_proj(enorm(emb)) + h_proj(hnorm(h_prev)).
        let mut emb_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
        crate::ops::rms_norm_batch(ctx, &emb, &mtp.enorm, eps, &mut emb_normed)?;
        let mut e_proj = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
        crate::attention::dsv4_linear(ctx, &mtp.e_proj, &emb_normed, &mut e_proj)?;

        let mut h_prev_batch = unsafe { HiddenStates::uninit(ctx, hidden_size, hc_mult)? };
        ctx.stream
            .memcpy_dtod(&h_prev.data, &mut h_prev_batch.data)
            .map_err(|e| anyhow!("DSv4 MTP h_prev D2D batch copy failed: {e}"))?;
        let mut h_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, hc_mult)? };
        crate::ops::rms_norm_batch(ctx, &h_prev_batch, &mtp.hnorm, eps, &mut h_normed)?;
        let mut h_proj = unsafe { HiddenStates::uninit(ctx, hidden_size, hc_mult)? };
        crate::attention::dsv4_linear(ctx, &mtp.h_proj, &h_normed, &mut h_proj)?;

        let h_proj_stream = HiddenStates {
            data: h_proj.data,
            hidden_dim: hidden_size * hc_mult,
            seq_len: 1,
        };
        let mut stream = unsafe { HiddenStates::uninit(ctx, hidden_size * hc_mult, 1)? };
        {
            let (e_ptr, _ge) = e_proj.data.device_ptr(&ctx.stream);
            let (h_ptr, _gh) = h_proj_stream.data.device_ptr(&ctx.stream);
            let (out_ptr, _go) = stream.data.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_mtp_add_eproj_hproj_cuda(
                    e_ptr as *const ffi::Half,
                    h_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    hidden_size as i32,
                    hc_mult as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        let target_layer_idx = self.mtp_frozen_target_layer_idx(mtp)?;
        let slot_attention_len = slot.attention.len();
        let target_attention_state = slot.attention.get_mut(target_layer_idx).ok_or_else(|| {
            anyhow!(
                "DSv4 MTP frozen-KV target layer {target_layer_idx} outside slot attention len {}",
                slot_attention_len
            )
        })?;
        let (target_attention_pool, target_dsa_shared) =
            kv_adapter.layer_and_dsa_shared_mut(target_layer_idx)?;
        let stream = self.run_mtp_transformer_layer(
            mtp,
            stream,
            next_token,
            position as usize,
            target_layer_idx,
            target_attention_state,
            target_attention_pool,
            target_dsa_shared,
        )?;

        let mut last_hidden = DeviceVec::zeros(ctx, hidden_size)?;
        crate::hc::head_hidden_from_stream(
            ctx,
            &self.config,
            &mtp.head_hc,
            &stream,
            0,
            &mut last_hidden,
        )?;
        let mut last_normed = DeviceVec::zeros(ctx, hidden_size)?;
        crate::ops::rms_norm_vec(ctx, &last_hidden, &mtp.norm, eps, &mut last_normed)?;
        let mut logits = DeviceVec::zeros(ctx, self.lm_head.rows)?;
        self.lm_head_project(&last_normed, &mut logits)?;
        let token =
            crate::executor::sample_cuda_token(ctx, &logits, &SamplingParams::default(), position)?;
        // Return the wide MTP stream (stream_dim) so a depth-2 draft can chain from it.
        let stream_len = stream.hidden_dim * stream.seq_len;
        Ok((
            token,
            DeviceVec {
                data: stream.data,
                len: stream_len,
                label: "mtp_stream",
            },
        ))
    }

    fn mtp_frozen_target_layer_idx(&self, mtp: &Dsv4MtpLayer) -> Result<usize> {
        // SGLang's frozen-KV MTP harness maps assistant logical layer 0 to
        // target physical layer 0. DSv4's shipped MTP layer is forced to
        // compress_ratio=0 (SW-only), so the draft reads that committed target
        // SW ring instead of a fresh one-token attention state.
        let idx = if let Some(raw) = std::env::var_os("ARLE_DSV4_MTP_FROZEN_LAYER") {
            raw.to_string_lossy().parse::<usize>().map_err(|err| {
                anyhow!(
                    "ARLE_DSV4_MTP_FROZEN_LAYER={} is not a usize: {err}",
                    raw.to_string_lossy()
                )
            })?
        } else {
            0
        };
        let layer = self.layers.get(idx).ok_or_else(|| {
            anyhow!(
                "DSv4 MTP frozen-KV target layer {idx} outside base layer count {}",
                self.layers.len()
            )
        })?;
        ensure!(
            mtp.layer.mode == DeepSeekV4AttentionMode::SlidingWindow,
            "DSv4 MTP frozen-KV path expects the MTP layer to be SlidingWindow, got {:?}",
            mtp.layer.mode
        );
        ensure!(
            layer.mode == DeepSeekV4AttentionMode::SlidingWindow,
            "DSv4 MTP frozen-KV target layer {idx} must be SlidingWindow for the current MTP layer, got {:?}",
            layer.mode
        );
        Ok(idx)
    }

    fn run_mtp_transformer_layer(
        &self,
        mtp: &Dsv4MtpLayer,
        stream: HiddenStates,
        token: u32,
        start_pos: usize,
        target_layer_idx: usize,
        target_attention_state: &mut crate::attention::Dsv4LayerAttentionState,
        target_attention_pool: &mut crate::attention::Dsv4LayerKvLayout,
        target_dsa_shared: Option<&mut crate::attention::Dsv4DsaSharedScratch>,
    ) -> Result<HiddenStates> {
        let layer = &mtp.layer;
        let ctx = &self.ctx;
        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let eps = self.config.rms_norm_eps;
        let seq_len = 1;
        let start_pos_device = ctx
            .stream
            .clone_htod(&[start_pos as i32])
            .map_err(|e| anyhow!("DSv4 MTP start_pos H2D failed: {e}"))?;
        let local_width = layer.attention.wq_b.rows;
        ensure!(
            local_width.is_multiple_of(self.config.head_dim),
            "DSv4 MTP attention local width {local_width} is not a multiple of head_dim {}",
            self.config.head_dim
        );
        let mut moe_scratch =
            crate::moe::Dsv4MoeDecodeScratch::new(ctx, &self.moe_config, &self.split, &layer.moe)?;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let tokens = [token];

        let attn_mhc = crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_attn, &stream)?;
        let mut attn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, seq_len)? };
        crate::hc::mhc_pre_rms_norm(
            ctx,
            &stream,
            &attn_mhc.pre,
            &layer.attn_norm,
            eps,
            hidden_size,
            hc_mult,
            &mut attn_normed,
        )?;
        let mut attn_out = unsafe { HiddenStates::uninit(ctx, hidden_size, seq_len)? };
        crate::attention::mla_attention(
            ctx,
            &self.config,
            &layer.attention,
            layer.mode,
            layer.compress_ratio,
            target_layer_idx,
            &attn_normed,
            target_attention_state,
            target_attention_pool,
            target_dsa_shared,
            start_pos,
            Some(&start_pos_device),
            &self.tp,
            &mut attn_out,
            &mut keepalive,
        )?;
        self.tp.all_reduce_sum(ctx, &mut attn_out)?;
        let mut attn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, seq_len)? };
        crate::hc::hc_post(
            ctx,
            &attn_out,
            &stream,
            &attn_mhc.post,
            &attn_mhc.comb,
            hidden_size,
            hc_mult,
            &mut attn_stream,
        )?;

        let ffn_mhc = crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_ffn, &attn_stream)?;
        let mut ffn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, seq_len)? };
        crate::hc::mhc_pre_rms_norm(
            ctx,
            &attn_stream,
            &ffn_mhc.pre,
            &layer.ffn_norm,
            eps,
            hidden_size,
            hc_mult,
            &mut ffn_normed,
        )?;
        let mut moe_out = unsafe { HiddenStates::uninit(ctx, hidden_size, seq_len)? };
        crate::moe::dsv4_moe_forward(
            self,
            &layer.moe,
            &tokens,
            &ffn_normed,
            &mut moe_out,
            Some(&mut moe_scratch),
            &mut keepalive,
        )?;
        self.tp.all_reduce_sum(ctx, &mut moe_out)?;
        let mut shared = unsafe { HiddenStates::uninit(ctx, hidden_size, seq_len)? };
        crate::moe::dsv4_shared_expert_forward(
            ctx,
            &ctx.stream,
            &layer.moe,
            &ffn_normed,
            &mut shared,
            self.config.swiglu_limit,
            Some(&mut moe_scratch),
            &mut keepalive,
        )?;
        let mut moe_with_shared = unsafe { HiddenStates::uninit(ctx, hidden_size, seq_len)? };
        crate::ops::add_batch(ctx, &moe_out, &shared, &mut moe_with_shared)?;
        let mut ffn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, seq_len)? };
        crate::hc::hc_post(
            ctx,
            &moe_with_shared,
            &attn_stream,
            &ffn_mhc.post,
            &ffn_mhc.comb,
            hidden_size,
            hc_mult,
            &mut ffn_stream,
        )?;
        ctx.sync()?;
        Ok(ffn_stream)
    }

    fn forward_tokens_decode_graph(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(
            slot.seq_len == start_pos,
            "DSv4 graph decode slot seq_len {} != start_pos {start_pos}",
            slot.seq_len
        );
        ensure!(
            start_pos + 1 <= slot.max_seq_len,
            "DSv4 graph decode sequence {} exceeds slot max_seq_len {}",
            start_pos + 1,
            slot.max_seq_len
        );
        ensure!(
            !dsv4_use_deepep_transport()?,
            "DSv4 decode graph v1 supports the allreduce transport only"
        );

        if slot.decode_graph.is_none() {
            slot.decode_graph = Some(Dsv4DecodeGraphScratch::new(self)?);
        }
        let graph = slot
            .decode_graph
            .as_mut()
            .expect("decode graph scratch initialized");
        let start_pos_i32 = i32::try_from(start_pos)
            .map_err(|_| anyhow!("DSv4 start_pos {start_pos} overflows i32"))?;
        self.ctx
            .stream
            .memcpy_htod(&[start_pos_i32], &mut slot.start_pos_device)
            .map_err(|e| anyhow!("DSv4 graph start_pos H2D failed: {e}"))?;
        self.ctx
            .stream
            .memcpy_htod(&[token as i32], &mut graph.token_ids)
            .map_err(|e| anyhow!("DSv4 graph token-id H2D failed: {e}"))?;
        self.ctx
            .stream
            .memcpy_htod(&[token], &mut graph.token_ids_u32)
            .map_err(|e| anyhow!("DSv4 graph u32 token-id H2D failed: {e}"))?;

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let eps = self.config.rms_norm_eps;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);

        let whole_step = dsv4_whole_step_graph_enabled();
        if whole_step {
            for s in graph.layers.iter_mut() {
                s.attn_graph.set_bypass(true);
                s.moe_graph.set_bypass(true);
            }
            graph.tail_graph.set_bypass(true);
        }
        // The whole per-token forward (loop + tail) as one closure: when whole_step,
        // captured as ONE graph (graph.whole_graph); else run eagerly with the existing
        // per-portion captures. advance_decode_len is host bookkeeping that must run
        // every replay, so it is hoisted OUT of the closure (post-capture) for whole_step.
        let mut body = || -> Result<()> {
            for layer_idx in 0..self.layers.len() {
                let layer = &self.layers[layer_idx];
                if layer_idx == 0 {
                    let current = &mut graph.layers[0];
                    let Dsv4DecodeLayerGraphScratch {
                        attn_graph,
                        attn_mhc,
                        attn_in,
                        attn_normed,
                        attn_out,
                        ..
                    } = current;
                    let attn_state = &mut slot.attention[layer_idx];
                    let attn_pool = kv_adapter.layer_mut(layer_idx)?;
                    attn_graph.run_or_capture(|| {
                        crate::ops::embedding_batch(
                            &self.ctx,
                            &self.embed_tokens,
                            &graph.token_ids,
                            &mut graph.embeddings,
                        )?;
                        crate::hc::initial_stream_from_embeddings(
                            &self.ctx,
                            &graph.embeddings,
                            hidden_size,
                            hc_mult,
                            &mut graph.initial_stream,
                        )?;
                        let mhc = crate::hc::gen_mhc_params_into(
                            &self.ctx,
                            &self.config,
                            &layer.hc_attn,
                            &graph.initial_stream,
                            attn_mhc,
                        )?;
                        crate::hc::mhc_pre_rms_norm(
                            &self.ctx,
                            &graph.initial_stream,
                            mhc.pre,
                            &layer.attn_norm,
                            eps,
                            hidden_size,
                            hc_mult,
                            attn_normed,
                        )?;
                        crate::attention::mla_attention(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            attn_normed,
                            attn_state,
                            attn_pool,
                            None,
                            start_pos,
                            Some(&slot.start_pos_device),
                            &self.tp,
                            attn_out,
                            &mut keepalive,
                        )
                    })?;
                } else {
                    let (prev_layers, current_layers) = graph.layers.split_at_mut(layer_idx);
                    let prev = &mut prev_layers[layer_idx - 1];
                    let current = &mut current_layers[0];
                    let Dsv4DecodeLayerGraphScratch {
                        attn_graph,
                        attn_mhc,
                        attn_in,
                        attn_normed,
                        attn_out,
                        ..
                    } = current;
                    let attn_state = &mut slot.attention[layer_idx];
                    let attn_pool = kv_adapter.layer_mut(layer_idx)?;
                    attn_graph.run_or_capture(|| {
                        crate::ops::add_batch(
                            &self.ctx,
                            &prev.moe_out,
                            &prev.shared,
                            &mut prev.moe_with_shared,
                        )?;
                        crate::hc::hc_post(
                            &self.ctx,
                            &prev.moe_with_shared,
                            &prev.attn_stream,
                            &prev.ffn_mhc.post,
                            &prev.ffn_mhc.comb,
                            hidden_size,
                            hc_mult,
                            &mut prev.ffn_stream,
                        )?;
                        let mhc = crate::hc::gen_mhc_params_into(
                            &self.ctx,
                            &self.config,
                            &layer.hc_attn,
                            &prev.ffn_stream,
                            attn_mhc,
                        )?;
                        crate::hc::mhc_pre_rms_norm(
                            &self.ctx,
                            &prev.ffn_stream,
                            mhc.pre,
                            &layer.attn_norm,
                            eps,
                            hidden_size,
                            hc_mult,
                            attn_normed,
                        )?;
                        crate::attention::mla_attention(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            attn_normed,
                            attn_state,
                            attn_pool,
                            None,
                            start_pos,
                            Some(&slot.start_pos_device),
                            &self.tp,
                            attn_out,
                            &mut keepalive,
                        )?;
                        Ok(())
                    })?;
                }
                if !whole_step {
                    slot.attention[layer_idx].advance_decode_len(
                        layer.mode,
                        layer.compress_ratio,
                        start_pos + 1,
                    );
                }
                self.tp
                    .all_reduce_sum(&self.ctx, &mut graph.layers[layer_idx].attn_out)?;

                let (stream_in, layer_scratch) = if layer_idx == 0 {
                    (&graph.initial_stream, &mut graph.layers[0])
                } else {
                    let (prev_layers, current_layers) = graph.layers.split_at_mut(layer_idx);
                    (
                        &prev_layers[layer_idx - 1].ffn_stream,
                        &mut current_layers[0],
                    )
                };
                let Dsv4DecodeLayerGraphScratch {
                    moe_graph,
                    attn_mhc,
                    ffn_mhc,
                    attn_out,
                    attn_stream,
                    ffn_in,
                    ffn_normed,
                    moe_out,
                    shared,
                    ..
                } = layer_scratch;
                let moe_scratch = &mut slot.moe_decode_scratch[layer_idx];
                moe_graph.run_or_capture(|| {
                    crate::hc::hc_post(
                        &self.ctx,
                        attn_out,
                        stream_in,
                        &attn_mhc.post,
                        &attn_mhc.comb,
                        hidden_size,
                        hc_mult,
                        attn_stream,
                    )?;
                    let mhc = crate::hc::gen_mhc_params_into(
                        &self.ctx,
                        &self.config,
                        &layer.hc_ffn,
                        attn_stream,
                        ffn_mhc,
                    )?;
                    crate::hc::mhc_pre_rms_norm(
                        &self.ctx,
                        attn_stream,
                        mhc.pre,
                        &layer.ffn_norm,
                        eps,
                        hidden_size,
                        hc_mult,
                        ffn_normed,
                    )?;
                    crate::moe::dsv4_moe_forward_decode_graph(
                        self,
                        &layer.moe,
                        &graph.token_ids_u32,
                        ffn_normed,
                        moe_out,
                        moe_scratch,
                    )?;
                    crate::moe::dsv4_shared_expert_forward(
                        &self.ctx,
                        &self.ctx.stream,
                        &layer.moe,
                        ffn_normed,
                        shared,
                        self.config.swiglu_limit,
                        Some(moe_scratch),
                        &mut keepalive,
                    )
                })?;
                self.tp
                    .all_reduce_sum(&self.ctx, &mut graph.layers[layer_idx].moe_out)?;
            }

            let final_idx = self.layers.len() - 1;
            {
                // Field-access (no destructure-move) so the outer `body` closure can borrow
                // `graph` by disjoint fields. Nested tail capture (bypassed under whole_step).
                let final_scratch = &mut graph.layers[final_idx];
                let last_hidden_ref = &mut graph.last_hidden;
                let last_normed_ref = &mut graph.last_normed;
                let logits_batch_ref = &mut graph.logits_batch;
                let logits_ref = &mut graph.logits;
                graph.tail_graph.run_or_capture(|| {
                    crate::ops::add_batch(
                        &self.ctx,
                        &final_scratch.moe_out,
                        &final_scratch.shared,
                        &mut final_scratch.moe_with_shared,
                    )?;
                    crate::hc::hc_post(
                        &self.ctx,
                        &final_scratch.moe_with_shared,
                        &final_scratch.attn_stream,
                        &final_scratch.ffn_mhc.post,
                        &final_scratch.ffn_mhc.comb,
                        hidden_size,
                        hc_mult,
                        &mut final_scratch.ffn_stream,
                    )?;
                    crate::hc::head_hidden_from_stream(
                        &self.ctx,
                        &self.config,
                        &self.head_hc,
                        &final_scratch.ffn_stream,
                        0,
                        last_hidden_ref,
                    )?;
                    crate::ops::rms_norm_vec(
                        &self.ctx,
                        last_hidden_ref,
                        &self.norm,
                        eps,
                        last_normed_ref,
                    )?;
                    self.lm_head_project_decode_graph(
                        last_normed_ref,
                        logits_batch_ref,
                        logits_ref,
                    )?;
                    Ok(())
                })?;
                Ok(())
            }
        }; // end `body` closure

        if whole_step {
            graph.whole_graph.run_or_capture(body)?;
            // advance_decode_len is host bookkeeping — run every step (replay skips body).
            for layer_idx in 0..self.layers.len() {
                let layer = &self.layers[layer_idx];
                slot.attention[layer_idx].advance_decode_len(
                    layer.mode,
                    layer.compress_ratio,
                    start_pos + 1,
                );
            }
        } else {
            body()?;
        }

        slot.seq_len += 1;
        crate::executor::sample_cuda_token(&self.ctx, &graph.logits, params, position)
    }

    /// Project the final hidden vector through the LM head into `logits`. The
    /// head can be plain bf16 or DSv4 FP8/FP4 block-scaled, so dispatch the
    /// matching single-vector kernel (`seq_len == 1`).
    fn lm_head_project(&self, x: &DeviceVec, logits: &mut DeviceVec) -> Result<()> {
        use cuda_kernels::tensor::WeightFormat;
        ensure!(
            self.lm_head.cols == x.len && self.lm_head.rows == logits.len,
            "DSv4 lm_head shape mismatch: [{}x{}] x.len {} logits.len {}",
            self.lm_head.rows,
            self.lm_head.cols,
            x.len,
            logits.len
        );
        match self.lm_head.weight_format {
            WeightFormat::DenseBf16 => crate::ops::gemv(&self.ctx, &self.lm_head, x, logits),
            // FP8/FP4 block-scaled: run the batched GEMV path at batch=1, then
            // copy the one-token output row into the caller's logits vec.
            WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
                let x_batch = HiddenStates {
                    data: x.data.clone(),
                    hidden_dim: x.len,
                    seq_len: 1,
                };
                // SAFETY: mla_linear writes the full one-token logits batch.
                let mut out_batch = unsafe { HiddenStates::uninit(&self.ctx, logits.len, 1)? };
                crate::attention::mla_linear(&self.ctx, &self.lm_head, &x_batch, &mut out_batch)?;
                self.ctx
                    .stream
                    .memcpy_dtod(&out_batch.data, &mut logits.data)
                    .map_err(|e| anyhow!("DSv4 lm_head logits copy-back failed: {e}"))?;
                Ok(())
            }
            other => anyhow::bail!("DSv4 lm_head unsupported weight format {other:?}"),
        }
    }

    fn lm_head_project_decode_graph(
        &self,
        x: &DeviceVec,
        logits_batch: &mut HiddenStates,
        logits: &mut DeviceVec,
    ) -> Result<()> {
        use cuda_kernels::tensor::WeightFormat;
        ensure!(
            self.lm_head.cols == x.len && self.lm_head.rows == logits.len,
            "DSv4 lm_head shape mismatch: [{}x{}] x.len {} logits.len {}",
            self.lm_head.rows,
            self.lm_head.cols,
            x.len,
            logits.len
        );
        match self.lm_head.weight_format {
            WeightFormat::DenseBf16 => crate::ops::gemv(&self.ctx, &self.lm_head, x, logits),
            WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
                ensure!(
                    logits_batch.hidden_dim == logits.len && logits_batch.seq_len == 1,
                    "DSv4 decode graph lm_head batch shape {}x{} != {}x1",
                    logits_batch.hidden_dim,
                    logits_batch.seq_len,
                    logits.len
                );
                crate::attention::mla_linear_vec(&self.ctx, &self.lm_head, x, logits_batch)?;
                self.ctx
                    .stream
                    .memcpy_dtod(&logits_batch.data, &mut logits.data)
                    .map_err(|e| {
                        anyhow!("DSv4 decode graph lm_head logits copy-back failed: {e}")
                    })?;
                Ok(())
            }
            other => anyhow::bail!("DSv4 lm_head unsupported weight format {other:?}"),
        }
    }

    /// MoE config built from the DSv4 router fields (sqrtsoftplus + noaux_tc).
    pub(crate) fn moe_config_from_config(config: &DeepSeekV4Config) -> Result<MoeConfig> {
        let moe = MoeConfig::dsv4(
            config.n_routed_experts,
            config.n_shared_experts,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
            config.hidden_size,
        );
        moe.validate()
            .map_err(|e| anyhow::anyhow!("DSv4 MoE config invalid: {e}"))?;
        Ok(moe)
    }
}

fn dsv4_use_deepep_transport() -> Result<bool> {
    let value = std::env::var("ARLE_DSV4_MOE_TRANSPORT")
        .or_else(|_| std::env::var("ARLE_DSV4_MOE_BACKEND"))
        .unwrap_or_else(|_| "allreduce".to_string());
    match value.as_str() {
        "allreduce" | "all_reduce" | "native" | "scalar" | "static" | "deepgemm" | "" => Ok(false),
        "deepep" | "native-deepep" | "native_deepep" | "deepep_ll" | "deepep-ll"
        | "deepep_low_latency" | "native_deepep_ll" => Ok(true),
        other => bail!(
            "unsupported ARLE_DSV4_MOE_TRANSPORT/ARLE_DSV4_MOE_BACKEND `{other}` \
             (expected allreduce, deepep, or deepep_ll)"
        ),
    }
}

fn dsv4_decode_graph_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

/// Whole-step decode CUDA graph: capture the ENTIRE per-token forward (all layers
/// + the 86 all-reduces + tail) as ONE graph, replayed with ~0 host orchestration.
/// Requires ARLE_DSV4_DECODE_GRAPH=1 (reuses its pre-allocated scratch).
///
/// STATUS (2026-06-08): VALIDATED-CORRECT but WALL-NEUTRAL — kept as gated infra, not
/// default. The 86-NCCL-all-reduces-in-one-graph capture WORKS and is byte-identical,
/// but the A/B is wall-neutral (eager 30.38 / per-portion 30.13 / whole-step 30.32 tok/s):
/// removing ALL host orchestration moves the wall 0%, which is the DEFINITIVE proof that
/// B=1 decode is GPU/critical-path-bound, NOT host-bound (the earlier "94% GPU-idle" was a
/// harness-window artifact). Retained as the re-runnable host-vs-GPU diagnostic and the only
/// concrete capturable-whole-decode reference for future tree-EAGLE/chain-fusion work.
/// See docs/experience/wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md.
fn dsv4_whole_step_graph_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_WHOLE_STEP_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

fn dsv4_comm_overlap_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_COMM_OVERLAP").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

pub(crate) fn dsv4_spec_decode_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_SPEC_DECODE").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

/// Batch the MTP verify over [pending, draft] in one 2-token forward (amortizes the
/// 149GB weight read). Default ON: validated 2026-06-08 on the TP=8 pod — with the
/// executor batched-reject, the FULL spec decode is BYTE-IDENTICAL to non-spec on
/// needle AND capital while running +61/+70% (needle 39.9→64.2, capital 38.2→65.0
/// tok/s; decode ~27→~16ms). The earlier col1 "divergence" was a selftest artifact
/// (the bonus on a forced-reject draft, which real decode discards) + the per-token
/// reject mis-applied to the batched path; both resolved. Only matters when
/// ARLE_DSV4_SPEC_DECODE is on. Opt out with ARLE_DSV4_MTP_BATCHED_VERIFY=0.
pub(crate) fn dsv4_mtp_batched_verify_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_DSV4_MTP_BATCHED_VERIFY").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    )
}

/// TP runtime for DSv4 load — multi-rank `nccl` builds resolve the NCCL
/// `unique_id` like the dense path; otherwise the no-op single runtime.
fn build_dsv4_tp_runtime() -> Result<crate::tp::TpRuntime> {
    #[cfg(feature = "nccl")]
    {
        let cfg = crate::tp::resolve_tp_config_from_env().map_err(|e| anyhow!("{e}"))?;
        if !cfg.is_single() {
            let ordinal = cuda_kernels::tensor::parse_device_ordinal_from_env()?;
            cudarc::runtime::result::device::set(ordinal as i32)
                .map_err(|e| anyhow!("cudaSetDevice({ordinal}) before NCCL init failed: {e}"))?;
            let unique_id = crate::loader::nccl_unique_id_from_env()?;
            return crate::tp::TpRuntime::from_env_with_nccl(unique_id);
        }
    }
    crate::tp::TpRuntime::from_env().map_err(|e| anyhow!("{e}"))
}

/// Refuse the genuinely-unported variants up front so the loader never
/// half-loads a shape the forward can't run. CSA/HCA attention, hyper-connections
/// (`hc_mult > 1`), and hash-routed MoE layers are all wired now. MTP
/// (speculative-draft) layers are tolerated but **not loaded**: the base forward
/// loops `0..num_hidden_layers` (see [`Dsv4Model::from_fp8_safetensors`]) and the
/// MTP predictor head is a separate path with no consumer in the base decode
/// loop, so we run the production config (`num_nextn_predict_layers=1`) directly
/// rather than forcing a hand-trimmed base-only config view. Called by
/// [`crate::loader`] before any device I/O.
pub(crate) fn ensure_loadable(config: &DeepSeekV4Config) -> Result<()> {
    ensure!(
        config.num_key_value_heads == 1,
        "DSv4 MLA expects num_key_value_heads=1, got {}",
        config.num_key_value_heads
    );
    if config.num_nextn_predict_layers > 0 {
        if dsv4_spec_decode_enabled() {
            eprintln!(
                "[dsv4] num_nextn_predict_layers={} present; ARLE_DSV4_SPEC_DECODE=1, \
                 loading base layers plus mtp.0 draft head.",
                config.num_nextn_predict_layers
            );
        } else {
            eprintln!(
                "[dsv4] num_nextn_predict_layers={} present; loading the {} base layers \
                 only (MTP draft head deferred — separate speculative-decode path).",
                config.num_nextn_predict_layers, config.num_hidden_layers
            );
        }
    }
    ensure!(
        config.hc_mult >= 1,
        "DSv4 hc_mult must be >= 1, got {}",
        config.hc_mult
    );
    Ok(())
}
