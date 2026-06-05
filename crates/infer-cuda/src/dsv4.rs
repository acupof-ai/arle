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
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
use cudarc::driver::CudaSlice;
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

    /// Allocate the flat device FP8 KV arena the FlashMLA sparse-FP8 decode reads.
    pub(crate) fn alloc_fp8_arena(
        &self,
        ctx: &DeviceContext,
        num_blocks: usize,
    ) -> Result<CudaSlice<u8>> {
        ensure!(num_blocks > 0, "DSv4 FP8 KV arena needs at least one block");
        let rows = num_blocks
            .checked_mul(self.page_block_size)
            .ok_or_else(|| anyhow!("DSv4 FP8 KV arena row count overflow"))?;
        let bytes = rows
            .checked_mul(self.bytes_per_token)
            .ok_or_else(|| anyhow!("DSv4 FP8 KV arena byte size overflow"))?;
        ctx.stream
            .alloc_zeros::<u8>(bytes)
            .map_err(|e| anyhow!("DSv4 FP8 KV arena alloc failed ({bytes} bytes): {e}"))
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
    pub q_norm: DeviceVec,
    pub wq_b: DeviceMatrix,
    pub wkv: DeviceMatrix,
    pub kv_norm: DeviceVec,
    pub wo_a: DeviceMatrix,
    pub wo_b: DeviceMatrix,
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
    pub tp: crate::tp::TpRuntime,
    #[cfg(feature = "deepep")]
    pub deepep: Option<crate::deepep::DeepEpTransport>,
}

pub(crate) struct Dsv4SlotState {
    attention: Vec<crate::attention::Dsv4LayerAttentionState>,
    moe_decode_scratch: Vec<crate::moe::Dsv4MoeDecodeScratch>,
    start_pos_device: CudaSlice<i32>,
    decode_graph: Option<Dsv4DecodeGraphScratch>,
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
    last_hidden: DeviceVec,
    last_normed: DeviceVec,
    logits_batch: HiddenStates,
    logits: DeviceVec,
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
    fn new(model: &Dsv4Model, max_seq_len: usize) -> Result<Self> {
        ensure!(max_seq_len > 0, "DSv4 slot max_seq_len must be positive");
        let mut attention = Vec::with_capacity(model.layers.len());
        let mut moe_decode_scratch = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let local_width = layer.attention.wq_b.rows;
            ensure!(
                local_width.is_multiple_of(model.config.head_dim),
                "DSv4 slot attention local width {local_width} is not a multiple of head_dim {}",
                model.config.head_dim
            );
            attention.push(crate::attention::Dsv4LayerAttentionState::new(
                &model.ctx,
                &model.config,
                layer.mode,
                layer.compress_ratio,
                max_seq_len,
                &model.kv_arena,
                local_width / model.config.head_dim,
                model.tp.config().world_size,
            )?);
            moe_decode_scratch.push(crate::moe::Dsv4MoeDecodeScratch::new(
                &model.ctx,
                &model.moe_config,
                &model.split,
                &layer.moe,
            )?);
        }
        let start_pos_device = model
            .ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow!("DSv4 slot start_pos device scalar alloc failed: {e}"))?;
        Ok(Self {
            attention,
            moe_decode_scratch,
            start_pos_device,
            decode_graph: None,
            seq_len: 0,
            max_seq_len,
        })
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.seq_len = 0;
        ctx.stream
            .memset_zeros(&mut self.start_pos_device)
            .map_err(|e| anyhow!("DSv4 slot start_pos reset failed: {e}"))?;
        for layer in &mut self.attention {
            layer.reset(ctx)?;
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
        tp: crate::tp::TpRuntime,
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
        #[cfg(feature = "deepep")]
        let deepep = crate::deepep::DeepEpTransport::maybe_boot(&ctx, &tp)?;
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
            tp,
            #[cfg(feature = "deepep")]
            deepep,
        })
    }

    pub(crate) fn new_slot_state(&self, max_seq_len: usize) -> Result<Dsv4SlotState> {
        Dsv4SlotState::new(self, max_seq_len)
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
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
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
        if dsv4_decode_graph_enabled()
            && !crate::attention::dsv4_flashmla_decode_enabled()?
            && seq_len == 1
            && use_gpu_router
            && !use_deepep_transport
        {
            return self.forward_tokens_decode_graph(slot, tokens[0], start_pos, params, position);
        }
        let use_moe_decode_scratch = use_gpu_router && seq_len == 1 && !use_deepep_transport;
        let mut keepalive = Dsv4ForwardKeepalive::new(seq_len == 1);
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
        crate::ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut embeddings)?;
        keepalive.keep_hidden(&embeddings);

        // Wide HC residual stream from the token embeddings.
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
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
            // ── Attention half: HC-wrap MLA attention.
            let mhc = crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_attn, &stream)?;
            keepalive.keep_f32(&mhc.pre);
            keepalive.keep_f32(&mhc.post);
            keepalive.keep_f32(&mhc.comb);
            // SAFETY: hc_pre writes the full [seq_len, hidden_size] buffer.
            let mut attn_in = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::hc::hc_pre(
                &self.ctx,
                &stream,
                &mhc.pre,
                hidden_size,
                hc_mult,
                &mut attn_in,
            )?;
            keepalive.keep_hidden(&attn_in);
            // SAFETY: rms_norm_batch writes the full [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::ops::rms_norm_batch(&self.ctx, &attn_in, &layer.attn_norm, eps, &mut normed)?;
            keepalive.keep_hidden(&normed);
            // SAFETY: mla_attention writes the full [seq_len, hidden_size] buffer.
            let mut attn_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            {
                let _nvtx = crate::nvtx::range("dsv4/mla_attn");
                crate::attention::mla_attention(
                    &self.ctx,
                    &self.config,
                    &layer.attention,
                    layer.mode,
                    layer.compress_ratio,
                    layer_idx,
                    &normed,
                    &mut slot.attention[layer_idx],
                    start_pos,
                    start_pos_device,
                    &self.tp,
                    &mut attn_out,
                    &mut keepalive,
                )?;
            }
            keepalive.keep_hidden(&attn_out);
            // Row-parallel O-LoRA: sum the per-rank partials (no-op single-GPU).
            {
                let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?;
            }
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

            // ── MoE half: HC-wrap the FP8 DeepGEMM MoE block.
            let mhc = crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)?;
            keepalive.keep_f32(&mhc.pre);
            keepalive.keep_f32(&mhc.post);
            keepalive.keep_f32(&mhc.comb);
            // SAFETY: hc_pre writes the full [seq_len, hidden_size] buffer.
            let mut ffn_in = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::hc::hc_pre(
                &self.ctx,
                &stream,
                &mhc.pre,
                hidden_size,
                hc_mult,
                &mut ffn_in,
            )?;
            keepalive.keep_hidden(&ffn_in);
            // SAFETY: rms_norm_batch writes the full [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::ops::rms_norm_batch(&self.ctx, &ffn_in, &layer.ffn_norm, eps, &mut normed)?;
            keepalive.keep_hidden(&normed);
            // SAFETY: the MoE forward writes the full routed output buffer.
            let mut moe_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if use_deepep_transport {
                #[cfg(feature = "deepep")]
                {
                    let transport = self.deepep.as_ref().ok_or_else(|| {
                        anyhow!("ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted")
                    })?;
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
                // Routed experts are EP-sharded; sum them first, then add the replicated
                // shared expert exactly once per rank.
                {
                    let _nvtx = crate::nvtx::range("dsv4/moe_allreduce");
                    self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
                }
            }
            keepalive.keep_hidden(&moe_out);
            let nvtx_shared_hc = crate::nvtx::range("dsv4/shared_hc");
            // SAFETY: dsv4_shared_expert_forward writes the full shared output.
            let mut shared = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            crate::moe::dsv4_shared_expert_forward(
                &self.ctx,
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
            drop(nvtx_shared_hc);
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
        }

        slot.seq_len += seq_len;

        let token = {
            let _nvtx = crate::nvtx::range("dsv4/lm_head_sample");
            // ── Head HC: fold the last token's wide stream row → one hidden vector.
            let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
            crate::hc::head_hidden_from_stream(
                &self.ctx,
                &self.config,
                &self.head_hc,
                &stream,
                seq_len - 1,
                &mut last_hidden,
            )?;
            keepalive.keep_hidden(&stream);
            keepalive.keep_vec(&last_hidden);

            // ── Final norm + lm_head projection + sample (last token row).
            let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
            crate::ops::rms_norm_vec(&self.ctx, &last_hidden, &self.norm, eps, &mut last_normed)?;
            keepalive.keep_vec(&last_normed);
            let mut logits = DeviceVec::zeros(&self.ctx, self.lm_head.rows)?;
            self.lm_head_project(&last_normed, &mut logits)?;
            keepalive.keep_vec(&logits);
            crate::executor::sample_cuda_token(&self.ctx, &logits, params, position)?
        };
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(token)
    }

    fn forward_tokens_decode_graph(
        &self,
        slot: &mut Dsv4SlotState,
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
                    crate::hc::hc_pre(
                        &self.ctx,
                        &graph.initial_stream,
                        mhc.pre,
                        hidden_size,
                        hc_mult,
                        attn_in,
                    )?;
                    crate::ops::rms_norm_batch(
                        &self.ctx,
                        attn_in,
                        &layer.attn_norm,
                        eps,
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
                    crate::hc::hc_pre(
                        &self.ctx,
                        &prev.ffn_stream,
                        mhc.pre,
                        hidden_size,
                        hc_mult,
                        attn_in,
                    )?;
                    crate::ops::rms_norm_batch(
                        &self.ctx,
                        attn_in,
                        &layer.attn_norm,
                        eps,
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
                        start_pos,
                        Some(&slot.start_pos_device),
                        &self.tp,
                        attn_out,
                        &mut keepalive,
                    )?;
                    Ok(())
                })?;
            }
            slot.attention[layer_idx].advance_decode_len(
                layer.mode,
                layer.compress_ratio,
                start_pos + 1,
            );
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
                crate::hc::hc_pre(
                    &self.ctx,
                    attn_stream,
                    mhc.pre,
                    hidden_size,
                    hc_mult,
                    ffn_in,
                )?;
                crate::ops::rms_norm_batch(&self.ctx, ffn_in, &layer.ffn_norm, eps, ffn_normed)?;
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
        let Dsv4DecodeGraphScratch {
            layers,
            tail_graph,
            last_hidden,
            last_normed,
            logits_batch,
            logits,
            ..
        } = graph;
        let final_scratch = &mut layers[final_idx];
        tail_graph.run_or_capture(|| {
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
                last_hidden,
            )?;
            crate::ops::rms_norm_vec(&self.ctx, last_hidden, &self.norm, eps, last_normed)?;
            self.lm_head_project_decode_graph(last_normed, logits_batch, logits)?;
            Ok(())
        })?;

        slot.seq_len += 1;
        crate::executor::sample_cuda_token(&self.ctx, logits, params, position)
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
        "deepep" | "native-deepep" | "native_deepep" => Ok(true),
        other => bail!(
            "unsupported ARLE_DSV4_MOE_TRANSPORT/ARLE_DSV4_MOE_BACKEND `{other}` \
             (expected allreduce or deepep)"
        ),
    }
}

fn dsv4_decode_graph_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
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
        eprintln!(
            "[dsv4] num_nextn_predict_layers={} present; loading the {} base layers \
             only (MTP draft head deferred — separate speculative-decode path).",
            config.num_nextn_predict_layers, config.num_hidden_layers
        );
    }
    ensure!(
        config.hc_mult >= 1,
        "DSv4 hc_mult must be >= 1, got {}",
        config.hc_mult
    );
    Ok(())
}
