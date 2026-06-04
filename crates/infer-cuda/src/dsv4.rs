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

use anyhow::{Result, anyhow, ensure};
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
    ///
    /// GATED (perf path, not correctness): only the FlashMLA decode launch
    /// (`dsv4_fp8_kv_pack` → `*_decode_sched_meta` →
    /// `arle_flashmla_sm90_sparse_decode_fwd` → `arle_dsv4_output_inverse_rope_*`)
    /// consumes this arena. The correctness-complete bf16 MLA core
    /// ([`crate::attention::mla_attention`]) attends over the bf16 SW ring cache +
    /// bf16 compressed pool (`dsv4_hybrid_attention_cuda`) instead, so this stays
    /// unwired — allocating it now reserves device bytes nothing reads yet. The
    /// lead can flip on the FP8 decode path once the bf16 forward parity-matches
    /// the oracle.
    #[allow(dead_code)]
    pub(crate) fn alloc_fp8_arena(
        &self,
        _ctx: &DeviceContext,
        _num_blocks: usize,
    ) -> Result<CudaSlice<u8>> {
        anyhow::bail!(
            "DSv4 MLA FP8 KV arena alloc (FlashMLA sparse-FP8 decode) is a perf path; the \
             bf16 MLA core attends over the SW ring + bf16 compressed pool"
        )
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
    pub compressor: Option<Dsv4Compressor>,
    pub indexer: Option<Dsv4Indexer>,
}

/// One DSv4 routed-MoE block: per-(local)-expert FP8 DeepGEMM caches for w1/w3
/// (gate/up) and w2 (down), the router gate, and the dense shared expert. Only
/// this rank's `ExpertSplit` slice is resident.
///
/// Routing kind is per-layer: bias-routed layers carry `gate_bias` (the
/// `noaux_tc` correction); hash-routed layers (`layer_idx < num_hash_layers`)
/// carry `hash_tid2eid` (a host `[vocab_size * topk]` table mapping token id →
/// experts directly) and ignore the learned router gate. Exactly one is `Some`.
pub(crate) struct Dsv4MoeLayer {
    /// Per-local-expert fused gate+up FP8 cache (w1 over w3, row-stacked) and the
    /// down cache (w2). The masked-grouped GEMM reads these.
    pub w13: Vec<Dsv4Fp8DeepGemmWeightCache>,
    pub w2: Vec<Dsv4Fp8DeepGemmWeightCache>,
    /// Router gate `[n_routed_experts, hidden]` (BF16 — the small router GEMM is
    /// not FP8). Read by bias-routed layers; hash layers still load it (harmless).
    pub gate: DeviceMatrix,
    /// Bias-routed layers only: per-expert `noaux_tc` correction `[n_routed]`.
    pub gate_bias: Option<DeviceVec>,
    /// Hash-routed layers only: host `tid2eid` table (`vocab_size * topk` i64),
    /// sliced per token to pick experts without the learned router.
    pub hash_tid2eid: Option<Vec<i64>>,
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
}

pub(crate) struct Dsv4SlotState {
    attention: Vec<crate::attention::Dsv4LayerAttentionState>,
    seq_len: usize,
    max_seq_len: usize,
}

impl Dsv4SlotState {
    fn new(model: &Dsv4Model, max_seq_len: usize) -> Result<Self> {
        ensure!(max_seq_len > 0, "DSv4 slot max_seq_len must be positive");
        let mut attention = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            attention.push(crate::attention::Dsv4LayerAttentionState::new(
                &model.ctx,
                &model.config,
                layer.mode,
                layer.compress_ratio,
                max_seq_len,
            )?);
        }
        Ok(Self {
            attention,
            seq_len: 0,
            max_seq_len,
        })
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.seq_len = 0;
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
            let attention = loader.load_dsv4_attention(&ctx, &lnames.attn)?;
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

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids_host)?;
        let mut embeddings = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        crate::ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut embeddings)?;

        // Wide HC residual stream from the token embeddings.
        let mut stream = HiddenStates::zeros(&self.ctx, stream_dim, seq_len)?;
        crate::hc::initial_stream_from_embeddings(
            &self.ctx,
            &embeddings,
            hidden_size,
            hc_mult,
            &mut stream,
        )?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // ── Attention half: HC-wrap MLA attention.
            let mhc = crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_attn, &stream)?;
            let mut attn_in = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            crate::hc::hc_pre(
                &self.ctx,
                &stream,
                &mhc.pre,
                hidden_size,
                hc_mult,
                &mut attn_in,
            )?;
            let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            crate::ops::rms_norm_batch(&self.ctx, &attn_in, &layer.attn_norm, eps, &mut normed)?;
            let mut attn_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
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
                self.tp.config().rank,
                &mut attn_out,
            )?;
            // Row-parallel O-LoRA: sum the per-rank partials (no-op single-GPU).
            self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?;
            let mut attn_stream = HiddenStates::zeros(&self.ctx, stream_dim, seq_len)?;
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
            stream = attn_stream;

            // ── MoE half: HC-wrap the FP8 DeepGEMM MoE block.
            let mhc = crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)?;
            let mut ffn_in = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            crate::hc::hc_pre(
                &self.ctx,
                &stream,
                &mhc.pre,
                hidden_size,
                hc_mult,
                &mut ffn_in,
            )?;
            let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            crate::ops::rms_norm_batch(&self.ctx, &ffn_in, &layer.ffn_norm, eps, &mut normed)?;
            let mut moe_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            crate::moe::dsv4_moe_forward(self, &layer.moe, tokens, &normed, &mut moe_out)?;
            // Routed experts are EP-sharded; sum them first, then add the replicated
            // shared expert exactly once per rank.
            self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
            let shared = crate::moe::dsv4_shared_expert_forward(
                &self.ctx,
                &layer.moe,
                &normed,
                self.config.swiglu_limit,
            )?;
            let mut moe_with_shared = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            crate::ops::add_batch(&self.ctx, &moe_out, &shared, &mut moe_with_shared)?;
            let mut ffn_stream = HiddenStates::zeros(&self.ctx, stream_dim, seq_len)?;
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
            stream = ffn_stream;
        }

        slot.seq_len += seq_len;

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

        // ── Final norm + lm_head projection + sample (last token row).
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
