//! DeepSeek-V4-Flash FP8 model: weight structs, MLA KV arena, EP-aware loader.
//!
//! Piece 1 of the DSv4 port: loads the FP8 block-scaled weights (reusing the
//! shared `cuda-kernels` DSv4 tensors) and stands up the MLA latent KV arena.
//! The MLA attention forward (Piece 2) and the FP8 DeepGEMM MoE forward
//! (Piece 3) are gated [`todo!`]/`ensure!` here so the loader compiles and the
//! state shape is the contract those pieces build on. DSv4 is multi-GPU only
//! (256 FP8 experts + MLA sharding don't fit one GPU); `ExpertSplit` carries the
//! per-rank EP ownership, `ExpertSplit::single` is the dev/typecheck fallback.
//!
//! Loader-only milestone: the model state + forward seam exist but no executor
//! consumes them yet (the `RealCudaExecutor` enum branch lands with the Piece 2/3
//! forward + MLA KV arena allocation). The `allow(dead_code)` marks pending-
//! consumer infra, not cruft — see `feedback_necessity_not_callers`.
#![allow(dead_code)]

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
use cudarc::driver::CudaSlice;
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use half::bf16;
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

    /// Allocate the flat device FP8 KV arena (`num_blocks * page_block_size *
    /// bytes_per_token` bytes) the FlashMLA sparse-FP8 decode reads.
    ///
    /// GATED to Piece 2b: only the FlashMLA decode launch (`dsv4_fp8_kv_pack` →
    /// `dsv4_flashmla_decode_build_indices_*` → `*_sched_meta` →
    /// `arle_flashmla_sm90_sparse_decode_fwd` → `arle_dsv4_output_inverse_rope_*`)
    /// consumes this arena. The correctness-complete SlidingWindow MLA core
    /// ([`crate::attention::mla_attention`]) uses the bf16 SW ring cache
    /// ([`Dsv4Model::alloc_sw_window_caches`]) instead, so this stays unwired —
    /// allocating it now would reserve device bytes nothing reads yet.
    #[allow(dead_code)]
    pub(crate) fn alloc_fp8_arena(
        &self,
        _ctx: &DeviceContext,
        _num_blocks: usize,
    ) -> Result<CudaSlice<u8>> {
        anyhow::bail!(
            "DSv4 MLA FP8 KV arena alloc (FlashMLA sparse-FP8 decode) is Piece 2b; the \
             SlidingWindow MLA core uses the bf16 SW ring cache (alloc_sw_window_caches)"
        )
    }
}

/// One DSv4 MLA attention block's FP8 weights (compress_ratio == 0 / SW mode).
///
/// Q-LoRA: `wq_a` (down) → `q_norm` → `wq_b` (up to per-head Q). KV is the
/// compressed latent: `wkv` → `kv_norm`. Output is also low-rank: `wo_a` (per
/// o-group) → `wo_b` (back to hidden). `attn_sink` is the per-head sink logit.
/// CSA/HCA compressor + indexer weights are Piece 2 — see [`Dsv4Layer::mode`].
pub(crate) struct Dsv4Attention {
    pub wq_a: DeviceMatrix,
    pub q_norm: DeviceVec,
    pub wq_b: DeviceMatrix,
    pub wkv: DeviceMatrix,
    pub kv_norm: DeviceVec,
    pub wo_a: DeviceMatrix,
    pub wo_b: DeviceMatrix,
    pub attn_sink: DeviceVec,
}

/// One DSv4 routed-MoE block: per-(local)-expert FP8 DeepGEMM caches for w1/w3
/// (gate/up) and w2 (down), the router gate + its `noaux_tc` correction bias,
/// and the dense shared expert. Only this rank's `ExpertSplit` slice is resident.
pub(crate) struct Dsv4MoeLayer {
    /// Per-local-expert fused gate+up FP8 cache (w1 over w3, row-stacked) and the
    /// down cache (w2). Piece 3's masked-grouped GEMM reads these.
    pub w13: Vec<Dsv4Fp8DeepGemmWeightCache>,
    pub w2: Vec<Dsv4Fp8DeepGemmWeightCache>,
    /// Router gate `[n_routed_experts, hidden]` (BF16 — the small router GEMM is
    /// not FP8) and the per-expert correction bias `[n_routed_experts]`.
    pub gate: DeviceMatrix,
    pub gate_bias: DeviceVec,
    /// Dense shared expert FP8 caches (always-on, n_shared_experts == 1).
    pub shared_w13: Dsv4Fp8DeepGemmWeightCache,
    pub shared_w2: Dsv4Fp8DeepGemmWeightCache,
}

/// One DSv4 transformer layer (pre-attn / pre-ffn norms + attention + MoE).
/// Hyper-connection (`hc_mult > 1`) mixing weights are Piece 2; `mode` records
/// the attention variant so the forward can dispatch / refuse unsupported modes.
pub(crate) struct Dsv4Layer {
    pub attn_norm: DeviceVec,
    pub ffn_norm: DeviceVec,
    pub attention: Dsv4Attention,
    pub moe: Dsv4MoeLayer,
    pub mode: DeepSeekV4AttentionMode,
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
    pub tp: crate::tp::TpRuntime,
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
        let loader = SafetensorLoader::new(model_path)?;
        let names = config.tensor_names();

        let embed_tokens = loader.load_dsv4_global_matrix(&ctx, names.embed_tokens())?;
        let lm_head = loader.load_dsv4_global_matrix(&ctx, names.lm_head())?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let mode = Self::supported_attention_mode(&config, layer_idx)?;
            let lnames = config.layer_tensor_names(layer_idx);
            let attention = loader.load_dsv4_attention(&ctx, &lnames.attn)?;
            let moe = loader.load_dsv4_moe_layer(&ctx, &lnames.ffn, &split)?;
            layers.push(Dsv4Layer {
                attn_norm: loader.load_dsv4_vec(&ctx, &lnames.attn_norm)?,
                ffn_norm: loader.load_dsv4_vec(&ctx, &lnames.ffn_norm)?,
                attention,
                moe,
                mode,
            });
        }
        let norm = loader.load_dsv4_vec(&ctx, names.norm())?;
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
            tp,
        })
    }

    /// Allocate this rank's per-layer bf16 sliding-window ring caches.
    ///
    /// Each SlidingWindow MLA layer holds `sliding_window * head_dim` bf16
    /// elements that [`crate::attention::mla_attention`] reads and updates in
    /// place. Zero-initialized = an empty window (a fresh prefill from
    /// `start_pos == 0`); a continuous-batching executor (Piece 4) will instead
    /// retain these across decode steps per slot.
    pub(crate) fn alloc_sw_window_caches(&self) -> Result<Vec<CudaSlice<bf16>>> {
        let cache_len = self.config.sliding_window * self.config.head_dim;
        ensure!(
            cache_len > 0,
            "DSv4 SW window cache len is zero (sliding_window={} head_dim={})",
            self.config.sliding_window,
            self.config.head_dim
        );
        self.layers
            .iter()
            .map(|_| {
                self.ctx
                    .stream
                    .alloc_zeros::<bf16>(cache_len)
                    .map_err(|e| anyhow!("DSv4 SW window cache alloc failed: {e}"))
            })
            .collect()
    }

    /// Forward one prefill/decode step over `tokens` starting at `start_pos`,
    /// returning the next greedy/sampled token.
    ///
    /// Per-layer: `attn_norm → mla_attention (SW-mode) → residual (+TP all-reduce)
    /// → ffn_norm → dsv4_moe_forward (Piece 3) → residual (+TP all-reduce)`, then
    /// the final RMSNorm + lm_head projection + sample. The MLA core is Piece 2a
    /// (this file); the FP8 DeepGEMM MoE block is Piece 3's
    /// [`crate::moe::dsv4_moe_forward`] (concurrent — its body may not be in this
    /// worktree yet; a typecheck shim covers the missing symbol).
    pub(crate) fn forward_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 forward requires at least one token"
        );

        let hidden_size = self.config.hidden_size;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        crate::ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)?;

        let mut sw_caches = self.alloc_sw_window_caches()?;
        let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut attn_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut hidden_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut moe_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // ── Attention block.
            crate::ops::rms_norm_batch(&self.ctx, &hidden, &layer.attn_norm, eps, &mut normed)?;
            // Hyper-connection pre-mix (hc_mult > 1) is Piece 2b: the loader
            // already `ensure!`s hc_mult == 1, so the dense (hc_mult == 1) stream
            // is the identity wrap — no hc_pre/hc_post mixing on this path.
            crate::attention::mla_attention(
                &self.ctx,
                &self.config,
                &layer.attention,
                layer.mode,
                layer_idx,
                &normed,
                &mut sw_caches[layer_idx],
                start_pos,
                &mut attn_out,
            )?;
            // Row-parallel O-LoRA: sum the per-rank partials (no-op single-GPU).
            self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?;
            crate::ops::add_batch(&self.ctx, &hidden, &attn_out, &mut hidden_out)?;
            std::mem::swap(&mut hidden, &mut hidden_out);

            // ── MoE block (Piece 3).
            crate::ops::rms_norm_batch(&self.ctx, &hidden, &layer.ffn_norm, eps, &mut normed)?;
            crate::moe::dsv4_moe_forward(self, &layer.moe, &normed, &mut moe_out)?;
            // Row-parallel down/combine: sum the per-rank partials.
            self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
            crate::ops::add_batch(&self.ctx, &hidden, &moe_out, &mut hidden_out)?;
            std::mem::swap(&mut hidden, &mut hidden_out);
        }

        // ── Final norm + lm_head projection + sample (last token row).
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        crate::ops::copy_row_to_vec(&self.ctx, &hidden, seq_len - 1, &mut last_hidden)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        crate::ops::rms_norm_vec(&self.ctx, &last_hidden, &self.norm, eps, &mut last_normed)?;
        let mut logits = DeviceVec::zeros(&self.ctx, self.lm_head.rows)?;
        self.lm_head_project(&last_normed, &mut logits)?;
        crate::executor::sample_cuda_token(&self.ctx, &logits, params, position)
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
                let mut out_batch = HiddenStates::zeros(&self.ctx, logits.len, 1)?;
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

    /// Pin the attention mode for a layer, refusing the modes whose compressor /
    /// indexer / hyper-connection weights Piece 2 still owns. Keeps Piece 1 from
    /// silently mis-loading a CSA/HCA layer as plain sliding-window.
    pub(crate) fn supported_attention_mode(
        config: &DeepSeekV4Config,
        layer_idx: usize,
    ) -> Result<DeepSeekV4AttentionMode> {
        let plan = config
            .attention_layer_plan(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("DSv4 layer {layer_idx} has no attention plan"))?;
        ensure!(
            plan.mode == DeepSeekV4AttentionMode::SlidingWindow,
            "DSv4 layer {layer_idx} is {:?}; CSA/HCA compressor+indexer attention is Piece 2 \
             (not yet ported)",
            plan.mode
        );
        ensure!(
            config.hc_mult == 1,
            "DSv4 hyper-connections (hc_mult={}) are Piece 2 (not yet ported)",
            config.hc_mult
        );
        Ok(plan.mode)
    }
}

/// TP runtime for DSv4 load — multi-rank `nccl` builds resolve the NCCL
/// `unique_id` like the dense path; otherwise the no-op single runtime.
fn build_dsv4_tp_runtime() -> Result<crate::tp::TpRuntime> {
    #[cfg(feature = "nccl")]
    {
        let cfg = crate::tp::resolve_tp_config_from_env().map_err(|e| anyhow!("{e}"))?;
        if !cfg.is_single() {
            let unique_id = crate::loader::nccl_unique_id_from_env()?;
            return crate::tp::TpRuntime::from_env_with_nccl(unique_id);
        }
    }
    crate::tp::TpRuntime::from_env().map_err(|e| anyhow!("{e}"))
}

/// Refuse the not-yet-ported variants up front so the loader never half-loads a
/// shape Piece 2/3 can't run. Called by [`crate::loader`] before any device I/O.
pub(crate) fn ensure_loadable(config: &DeepSeekV4Config) -> Result<()> {
    ensure!(
        config.num_key_value_heads == 1,
        "DSv4 MLA expects num_key_value_heads=1, got {}",
        config.num_key_value_heads
    );
    ensure!(
        config.num_nextn_predict_layers == 0,
        "DSv4 MTP layers (num_nextn_predict_layers={}) are deferred (Piece 2 follow-up)",
        config.num_nextn_predict_layers
    );
    ensure!(
        config.num_hash_layers == 0,
        "DSv4 hash-routed MoE layers (num_hash_layers={}) are Piece 3 (not yet ported)",
        config.num_hash_layers
    );
    Ok(())
}
