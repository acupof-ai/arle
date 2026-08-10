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

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::{CudaPipelineStreamKind, Dsv4Fp8DeepGemmWeightCache};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config, DeepSeekV4MoeRoutingKind};
use infer_moe::MoeConfig;
use infer_plan::SamplingParams;

use crate::moe_config::ExpertSplit;

#[path = "dsv4/budget.rs"]
mod budget;
#[path = "dsv4/dspark.rs"]
pub(crate) mod dspark;
#[path = "dsv4/load.rs"]
mod load;

pub(crate) use budget::Dsv4MlaKvArena;
pub(crate) use load::load_dspark_draft;

/// Compressor sub-block for CSA/HCA layers (`compress_ratio > 0`): projects the
/// wide hidden into the compressed-key latent stream the sparse attention reads.
/// `wkv`/`wgate` may be FP8/FP4 block-scaled or bf16 (`dsv4_linear` dispatches on
/// `weight_format`); the main-value probe retains BF16 input projections and F32
/// APE for the FP32 compressor forward. The indexer and default path stay
/// unchanged.
pub(crate) struct Dsv4Compressor {
    pub wkv: DeviceMatrix,
    pub wgate: DeviceMatrix,
    pub ape: DeviceMatrix,
    pub fp32_probe: Dsv4CompressorFp32Probe,
    pub norm: DeviceVec,
    /// DeepGEMM repacks of the FP8 `wkv`/`wgate` projections for the batched (m=N)
    /// decode pre-pass — tensor-core `sm90_fp8_gemm_1d2d` instead of the scalar
    /// `dsv4_fp8_gemv_batch`. `None` ⇒ scalar (bf16 GLM dialect, or gate off).
    pub wkv_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub wgate_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
}

pub(crate) struct Dsv4CompressorFp32Probe {
    pub wkv: DeviceMatrix,
    pub wgate: DeviceMatrix,
    pub ape: CudaSlice<f32>,
}

/// Sparse indexer sub-block. DSv4 CompressedSparse: a key compressor over
/// `index_head_dim` keys + `wq_b`/`weights_proj`. GLM DSA (`SparseIndexed`): the
/// indexer runs over the full (uncompressed) latent KV — no compressor; instead a
/// plain key projection `wk` + key RMSNorm `k_norm`. Both feed the top-k block
/// selector.
pub(crate) struct Dsv4Indexer {
    pub wq_b: DeviceMatrix,
    pub weights_proj: DeviceMatrix,
    /// DSv4 CSA key compressor. `None` ⇒ GLM SparseIndexed (uses `wk`/`k_norm`).
    pub compressor: Option<Dsv4Compressor>,
    /// GLM SparseIndexed only: indexer key projection `[index_n_heads*index_head_dim,
    /// hidden]`. `None` ⇒ DSv4 (keys come through the compressor). Consumed by the
    /// SparseIndexed index-key build (`sparse_indexed_index_key_forward`).
    pub wk: Option<DeviceMatrix>,
    /// GLM SparseIndexed only: indexer key RMSNorm weight + bias. `None` ⇒ DSv4.
    /// Consumed by the SparseIndexed index-key build.
    pub k_norm: Option<DeviceVec>,
    #[allow(dead_code)]
    pub k_norm_bias: Option<DeviceVec>,
    /// DeepGEMM repack of `wq_b` for the prefill index-query projection (the #1
    /// remaining projection after wq_b/wo: 135ms / 67% of linear at M=1024). Built
    /// when the prefill DeepGEMM scratch is enabled; `None` falls back to scalar.
    pub wq_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    /// DeepGEMM repack of `weights_proj` for the batched (m=N) decode indexer-query
    /// pre-pass. Same gate as `wq_b_deepgemm`; `None` ⇒ scalar.
    pub weights_proj_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
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

impl Dsv4HyperConnection {
    /// Zero placeholder for GLM (`hc_mult == 1`, no hyper-connections): the mixers
    /// are the identity (`stream_dim == hidden_size`), so the forward never reads
    /// these weights. GLM checkpoints ship no `hc_*` tensors, so the loader builds
    /// this instead of loading.
    fn identity_placeholder(ctx: &DeviceContext) -> Result<Self> {
        Ok(Self {
            base: DeviceVec::zeros(ctx, 1)?,
            mix_fn: DeviceMatrix::from_safetensors(ctx, &[0u8; 2], 1, 1)?,
            scale: DeviceVec::zeros(ctx, 1)?,
        })
    }
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
    /// #150: dense-bf16 dequant copies of `wq_a`/`wq_b`/`wkv`, built at load only
    /// under `ARLE_DSV4_MLA_PROJ_BF16=1` (the checkpoint ships these F8_E4M3-only).
    /// Presence routes BOTH decode lanes (n=1 fused and n≥2 batched) through bf16
    /// cublasLt so every decode batch size shares one arithmetic; prefill keeps
    /// the FP8 originals (DeepGEMM prefill licensed −47%). `None` ⇒ no VRAM cost.
    pub mla_proj_bf16: Option<Dsv4MlaProjBf16>,
    /// DSv4 low-rank output down-projection. `None` ⇔ GLM (`config.plain_o_proj`,
    /// which uses `o_proj` instead).
    pub wo_a: Option<DeviceMatrix>,
    /// Per-output-group pointer table for `wo_a`. DSv4 output projection groups
    /// attention heads before the low-rank down projection; one TP rank may own
    /// more than one full output group (TP1/2/4 on DSv4-Flash), so `wo_a` is
    /// launched as group-routed `[token, group]` rows instead of pretending the
    /// rank-local attention row is one flat K. `None` ⇔ GLM plain-o.
    pub wo_a_groups: Option<Dsv4WoAGroupTables>,
    /// DSv4 low-rank output up-projection. `None` ⇔ GLM plain-o.
    pub wo_b: Option<DeviceMatrix>,
    /// DeepGEMM-layout FP8 caches of the output projection (`wo_a`/`wo_b`) for the
    /// decode path (lever #1b), companion to [`Self::wq_b_deepgemm`]. The flat
    /// `wo_a` cache is present only when this rank owns exactly one output group.
    /// Multi-group ranks use `wo_a_group_deepgemm` below so TP4 does not fall back
    /// to the scalar route GEMV path.
    pub wo_a_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    /// Per-output-group DeepGEMM caches for `wo_a` on ranks that own multiple
    /// whole output groups (TP1/2/4). Each cache is `[o_lora_rank, group_width]`.
    pub wo_a_group_deepgemm: Option<Vec<Dsv4Fp8DeepGemmWeightCache>>,
    pub wo_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    /// Per-head attention sink logit. `None` ⇔ GLM (no `attn_sink` tensor).
    pub attn_sink: Option<DeviceVec>,
    pub attn_sink_f32: Option<CudaSlice<f32>>,
    pub compressor: Option<Dsv4Compressor>,
    pub indexer: Option<Dsv4Indexer>,
    /// GLM plain output projection `[hidden, num_heads*v_head_dim]` (replaces the
    /// DSv4 `wo_a`/`wo_b` low-rank). `Some` ⇔ `config.plain_o_proj`; DSv4 leaves
    /// `None` and uses `wo_a`/`wo_b`/`wo_a_groups`. Consumed by the Tranche-D GLM
    /// forward (not yet wired).
    #[allow(dead_code)]
    pub o_proj: Option<DeviceMatrix>,
    /// GLM `kv_b` absorption split — the up-projection of the compressed KV latent,
    /// folded at runtime around the V32 FlashMLA call (SGLang `forward_absorb_core`
    /// `deepseek_v2.py`). `w_kc[heads, qk_nope_head_dim, kv_lora_rank]` lifts q_nope
    /// into the 512-latent; `w_vc[heads, kv_lora_rank, v_head_dim]` projects the
    /// latent output back to v. `Some` ⇔ GLM (`config.kv_lora_rank > 0`); DSv4's
    /// pre-absorbed checkpoint leaves both `None`. Consumed by the Tranche-D forward.
    #[allow(dead_code)]
    pub w_kc: Option<DeviceMatrix>,
    #[allow(dead_code)]
    pub w_vc: Option<DeviceMatrix>,
}

/// #150 bf16 dequant copies of the MLA Q/KV LoRA projections (see
/// [`Dsv4Attention::mla_proj_bf16`]).
pub(crate) struct Dsv4MlaProjBf16 {
    pub wq_a: DeviceMatrix,
    pub wq_b: DeviceMatrix,
    pub wkv: DeviceMatrix,
}

impl Dsv4Attention {
    /// #150: decode-lane `wq_a`/`wq_b`/`wkv` — the bf16 dequant copies when
    /// `ARLE_DSV4_MLA_PROJ_BF16` built them at load, else the FP8 originals.
    pub(crate) fn decode_proj_weights(&self) -> (&DeviceMatrix, &DeviceMatrix, &DeviceMatrix) {
        match &self.mla_proj_bf16 {
            Some(p) => (&p.wq_a, &p.wq_b, &p.wkv),
            None => (&self.wq_a, &self.wq_b, &self.wkv),
        }
    }
}

pub(crate) struct Dsv4WoAGroupTables {
    pub weight_ptrs: CudaSlice<u64>,
    pub scale_ptrs: CudaSlice<u64>,
    pub groups: usize,
    pub rows_per_group: usize,
    pub cols_per_group: usize,
    pub scale_rows_per_group: usize,
    pub scale_cols: usize,
}

/// One DSv4 routed-MoE block: prebuilt group-major FP8 DeepGEMM caches for
/// w1/w3 (gate/up) and w2 (down), the router gate, and the dense shared expert.
/// Only this rank's `ExpertSplit` slice is resident.
///
/// Routing kind is per-layer: bias-routed layers carry `gate_bias` (the
/// `noaux_tc` correction); hash-routed layers (`layer_idx < num_hash_layers`)
/// carry a device `hash_tid2eid` table mapping token id to experts directly and
/// ignore the learned router gate.
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
    /// Hash-routed layers only: device `tid2eid` table used by the on-device
    /// router.
    pub hash_tid2eid_device: Option<CudaSlice<i64>>,
    pub routing_kind: DeepSeekV4MoeRoutingKind,
    /// Dense shared expert FP8 caches (always-on, n_shared_experts == 1).
    /// Both DSv4 and GLM run the FP8 DeepGEMM shared path; GLM ships F32
    /// `weight_scale_inv` block scales which the 1D2D FP8 GEMM consumes directly.
    pub shared_w13: Dsv4Fp8DeepGemmWeightCache,
    pub shared_w2: Dsv4Fp8DeepGemmWeightCache,
    /// Decode-band grouped-GEMV lane tables (per-expert weight/scale pointer
    /// tables + UE8M0-re-encoded scales), built lazily on first decode-band
    /// MoE forward. `Some(None)` = build attempted and failed the lossless
    /// UE8M0 check — the contiguous DeepGEMM lane stays the fallback.
    pub gemv_tables: std::sync::OnceLock<Option<crate::moe::Dsv4GemvTables>>,
}

/// GLM dense-MLP layer (the first `first_k_dense_replace` layers): a plain SwiGLU
/// FFN at `intermediate_size` (12288) replacing the routed-expert stack. GLM ships
/// FP8 + F32 `weight_scale_inv`; the loader dequantizes to dense bf16 (the FP8
/// DeepGEMM caches need the E8M0 `dsv4_scales` layout GLM doesn't carry). Present
/// only on GLM dense layers; DSv4 (all-MoE) never builds this.
///
/// GPU-UNVERIFIABLE (pod): keeping the dense FFN in bf16 (vs FP8) trades VRAM for
/// a simpler bring-up; the GLM dense forward consumes `gate`/`up`/`down` directly.
///
/// All fields are consumed by the GLM dense forward `dsv4_dense_mlp_forward`
/// (wired via `Dsv4Layer::dense_mlp`). The `allow(dead_code)` stays because the
/// DSv4-only build (every layer MoE) never constructs this — `dense_mlp` is
/// `None` there — so the GLM-only fields read as dead under that config.
#[allow(dead_code)]
pub(crate) struct Dsv4DenseMlp {
    /// Gate projection `[intermediate, hidden]` (bf16).
    pub gate: DeviceMatrix,
    /// Up projection `[intermediate, hidden]` (bf16).
    pub up: DeviceMatrix,
    /// Down projection `[hidden, intermediate]` (bf16).
    pub down: DeviceMatrix,
    pub hidden_dim: usize,
    pub intermediate: usize,
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
    /// Routed MoE block. `None` ⇔ a GLM dense layer (uses `dense_mlp`). DSv4 always
    /// `Some` (every layer is MoE).
    pub moe: Option<Dsv4MoeLayer>,
    pub mode: DeepSeekV4AttentionMode,
    pub compress_ratio: usize,
    /// GLM dense layers only (`config.per_layer_dense_mlp[i]`): a plain FFN that
    /// replaces the routed-expert forward. `Some` ⇒ the forward runs `dense_mlp`
    /// instead of `moe`. DSv4 layers leave this `None`. Consumed by the GLM dense
    /// forward `dsv4_dense_mlp_forward`; `allow(dead_code)` stays for the
    /// DSv4-only build, where every layer is MoE and this is always `None`.
    #[allow(dead_code)]
    pub dense_mlp: Option<Dsv4DenseMlp>,
}

/// One shipped DSv4 MTP draft head (`mtp.0.*`): a full transformer layer plus
/// the DeepSeek MTP input-combine and output-head tensors. Loaded when
/// `--spec-type mtp`/`--mtp-draft-tokens` requests MTP.
pub(crate) struct Dsv4MtpLayer {
    pub layer: Dsv4Layer,
    pub head_hc: Dsv4HyperConnection,
    pub enorm: DeviceVec,
    pub hnorm: DeviceVec,
    pub e_proj: DeviceMatrix,
    pub h_proj: DeviceMatrix,
    pub norm: DeviceVec,
}

/// One stage of the DSpark spec-decode draft — a full DSv4 transformer block
/// (`layer`: MLA attn + FP8 MoE, same body as `Dsv4MtpLayer`) plus the
/// position-dependent extras. The draft stacks 3 such stages (`mtp.0` → `mtp.1`
/// → `mtp.2`):
///
/// - `mtp.0` (entry): `main_proj` (fp8-block, F8_E4M3+F8_E8M0 like the attn
///   `wq_a`/`wkv` projections) + `main_norm` for the 3-tap context fusion.
/// - `mtp.{n-1}` (exit): `hc_head`, final `norm`, the low-rank Markov
///   token-transition head (`markov_w1`/`markov_w2`), and the scalar
///   `confidence_proj`. Output logits decode via the base model's separate
///   `head.weight` output head (`tie_word_embeddings=false`; our loader exposes
///   it as `model.lm_head`), NOT a draft-local or embed-tied head — the draft
///   checkpoint ships that same `head.weight`. (codex T4.3 flagged "tie to embed"
///   — a false positive: DSv4 has a distinct `head.weight`, verified in the index.)
/// - middle stages: bare block, all extras `None`.
// Fields are read only by the DSpark forward tranche (T4); inert until then.
#[allow(dead_code)]
pub(crate) struct Dsv4DsparkStage {
    pub layer: Dsv4Layer,
    pub main_proj: Option<DeviceMatrix>,
    pub main_norm: Option<DeviceVec>,
    pub hc_head: Option<Dsv4HyperConnection>,
    pub norm: Option<DeviceVec>,
    pub markov_w1: Option<DeviceMatrix>,
    pub markov_w2: Option<DeviceMatrix>,
    pub confidence_proj: Option<DeviceMatrix>,
}

/// The full 3-stage DSpark draft (`stages[0]` = `mtp.0` … `stages[n-1]` =
/// `mtp.{n-1}`), loaded from a DSpark checkpoint by [`load_dspark_draft`].
#[allow(dead_code)]
pub(crate) struct Dsv4DsparkDraft {
    pub stages: Vec<Dsv4DsparkStage>,
}

/// Vocab-shard geometry for the opt-in `ARLE_DSV4_LM_HEAD_SHARD=1` lever
/// (#99 H3). When set,
/// `lm_head` holds `[rows_per_rank, hidden]` = this rank's contiguous vocab
/// rows `[rank*rows_per_rank, rank*rows_per_rank + local_rows)` plus zero-pad
/// rows up to the uniform 128-aligned `rows_per_rank`. Padding exists only
/// past the global vocab end, so a rank-order logits gather is vocab-ordered
/// with the pad tail strictly after `vocab`.
// The fields are read by the nccl-gated cross-rank sampler
// (`executor::sample_cuda_token_vocab_sharded`); non-nccl builds bail before use.
#[cfg_attr(not(feature = "nccl"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dsv4LmHeadShard {
    /// Global vocab rows (`config.vocab_size`).
    pub vocab: usize,
    /// Uniform padded rows per rank (128-aligned; the local `lm_head.rows`).
    pub rows_per_rank: usize,
    /// This rank's real (unpadded) row count; `<= rows_per_rank`.
    pub local_rows: usize,
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
    /// `Some` iff `ARLE_DSV4_LM_HEAD_SHARD=1` at TP>1: `lm_head` then holds
    /// this rank's vocab slice and decode sampling merges across ranks.
    pub lm_head_shard: Option<Dsv4LmHeadShard>,
    pub layers: Vec<Dsv4Layer>,
    pub norm: DeviceVec,
    /// Head hyper-connection: folds the wide residual stream back to one hidden
    /// row before the final RMSNorm + lm_head projection.
    pub head_hc: Dsv4HyperConnection,
    pub mtp: Option<Dsv4MtpLayer>,
    /// Resolved at construction: spec decode is on when the serve config
    /// requested `--spec-type mtp` / `--spec-type dspark`. Per-slot construction
    /// (`Dsv4SlotState::new`) reads this so the MTP-head load and the per-slot
    /// rollback snapshots agree on one decision.
    pub spec_decode_on: bool,
    pub tp: crate::tp::TpRuntime,
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub(crate) mega_moe: Option<Dsv4MegaMoeTransport>,
    #[cfg(feature = "deepep")]
    pub deepep: Option<crate::deepep::DeepEpTransport>,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub(crate) struct Dsv4MegaMoeTransport {
    pub(crate) workspace: crate::tp::SymmetricIpcBuffer,
    pub(crate) owned_out: CudaSlice<half::bf16>,
    pub(crate) layout: cuda_kernels::moe::Sm90MegaMoeWorkspaceLayout,
    pub(crate) shape: cuda_kernels::moe::Sm90MegaMoeShape,
    pub(crate) fast_math: bool,
    pub(crate) enable_pdl: bool,
    epoch: std::sync::Mutex<(u64, Option<(u64, usize)>)>,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl Dsv4MegaMoeTransport {
    fn assert_collective_values(&self, model: &Dsv4Model, values: &[u64]) -> Result<()> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let gathered = model.tp.all_gather_bytes(&model.ctx, &bytes, bytes.len())?;
        ensure!(
            gathered.chunks_exact(bytes.len()).all(|peer| peer == bytes),
            "DSv4 MegaMoE specialization differs across TP ranks"
        );
        Ok(())
    }

    fn assert_static_spec(&self, model: &Dsv4Model, activation_clamp: f32) -> Result<()> {
        fn env_i32(name: &str, fallback: i32) -> i32 {
            std::env::var(name)
                .ok()
                .filter(|value| !value.is_empty())
                .and_then(|value| value.parse().ok())
                .unwrap_or(fallback)
        }

        let values = [
            u64::try_from(self.shape.num_ranks)?,
            u64::try_from(self.shape.num_experts)?,
            u64::try_from(self.shape.requested_max_tokens_per_rank)?,
            u64::try_from(self.shape.num_topk)?,
            u64::try_from(self.shape.hidden)?,
            u64::try_from(self.shape.intermediate_hidden)?,
            u64::from(activation_clamp.to_bits()),
            u64::from(self.fast_math),
            u64::from(self.enable_pdl),
            env_i32("DG_NUM_SMS", i32::try_from(model.ctx.sm_count())?) as u32 as u64,
            u64::from(env_i32("DG_SM90_FP8_SWAP_AB", 1) != 0),
        ];
        self.assert_collective_values(model, &values)
    }

    fn begin_forward(&self, model: &Dsv4Model, num_tokens: usize) -> Result<u64> {
        self.assert_collective_values(model, &[u64::try_from(num_tokens)?])?;
        let mut epoch = self
            .epoch
            .lock()
            .map_err(|_| anyhow!("DSv4 MegaMoE forward epoch lock poisoned"))?;
        epoch.0 = epoch.0.wrapping_add(1);
        let id = epoch.0;
        epoch.1 = Some((id, num_tokens));
        Ok(id)
    }

    pub(crate) fn assert_forward_epoch(&self, epoch_id: u64, num_tokens: usize) -> Result<()> {
        let epoch = self
            .epoch
            .lock()
            .map_err(|_| anyhow!("DSv4 MegaMoE forward epoch lock poisoned"))?;
        ensure!(
            epoch.1 == Some((epoch_id, num_tokens)),
            "DSv4 MegaMoE layer has no matching top-level forward epoch"
        );
        Ok(())
    }
}

/// Max depth for the per-slot spec-ring snapshot. `topk` widens candidate
/// matching only; verifier rows remain chain-shaped. `spec_depth` clamps
/// runtime requests to this ceiling before capture.
pub(crate) const MAX_SPEC_DRAFT_DEPTH: usize = 8;
/// Bounded chain verifier rows per slot. Current MTP uses `depth + 1` rows;
/// `topk` does not increase this count.
pub(crate) const MAX_SPEC_VERIFY_ROWS: usize = 64;
pub(crate) const DEFAULT_SPEC_DRAFT_DEPTH: usize = 2;

pub(crate) const DEFAULT_SPEC_DRAFT_TOPK: usize = 1;

/// Row schedule for one speculative verify forward. Sparse FlashMLA verifies
/// the scheduled rows in one target pass. `ancestors` is the prefix metadata
/// used by the batched FlashMLA sparse verify lane: row `r` attends committed
/// KV plus the listed earlier chunk rows and self.
pub(crate) struct SpecVerifySchedule {
    /// Per row: absolute position (`start_pos + node depth`).
    pub(crate) positions: Vec<usize>,
    /// Per row: chunk-row ancestors, shallow to deep, self excluded.
    pub(crate) ancestors: Vec<Vec<usize>>,
}

impl SpecVerifySchedule {
    pub(crate) fn validate_sparse_at(&self, start_pos: usize) -> Result<()> {
        ensure!(
            !self.positions.is_empty() && self.positions.len() == self.ancestors.len(),
            "DSv4 sparse verify schedule shape mismatch: positions={} ancestors={}",
            self.positions.len(),
            self.ancestors.len()
        );
        ensure!(
            self.positions.len() <= MAX_SPEC_VERIFY_ROWS,
            "DSv4 sparse verify rows {} exceed fold cache rows {MAX_SPEC_VERIFY_ROWS}",
            self.positions.len()
        );
        for (row, &pos) in self.positions.iter().enumerate() {
            ensure!(
                pos >= start_pos,
                "DSv4 sparse verify row {row} position {pos} precedes start_pos {start_pos}"
            );
            for &ancestor in &self.ancestors[row] {
                ensure!(
                    ancestor < row,
                    "DSv4 sparse verify row {row} has non-causal ancestor row {ancestor}"
                );
                ensure!(
                    self.positions[ancestor] < pos,
                    "DSv4 sparse verify row {row} position {pos} ancestor {ancestor} position {} is not earlier",
                    self.positions[ancestor]
                );
            }
        }
        Ok(())
    }
}

/// Target verify output for an MTP verify chunk: the full `[rows, vocab]` logits
/// matrix plus the greedy top-1 view and per-row MTP stream hiddens.
pub(crate) struct SpecVerifyResult {
    pub(crate) logits: HiddenStates,
    pub(crate) argmax: Vec<u32>,
    pub(crate) hiddens: Vec<DeviceVec>,
}

/// One draft row to expand in an MTP head pass ([`Dsv4Model::mtp_forward_level`]).
pub(crate) struct MtpDraftRow {
    pub token: u32,
}

pub(crate) struct Dsv4SlotState {
    attention: Vec<crate::attention::Dsv4LayerAttentionState>,
    /// Per-attention-layer one-slot snapshot of speculative-verify ring writes.
    /// `Some` only when `model.spec_decode_on`; pre-allocated once and
    /// index-aligned with `attention`.
    spec_rings: Option<Vec<crate::attention::Dsv4SpecRingSnapshot>>,
    /// P2 commit-fold scratch: per-layer attn-normed verify rows
    /// (`[hidden, MAX_SPEC_VERIFY_ROWS]`), persisted by the verify lane so
    /// the commit can re-ingest the accepted prefix (compressor/indexer + ring
    /// K) without a second full forward. `Some` only when
    /// `model.spec_decode_on`.
    spec_normed: Option<Vec<HiddenStates>>,
    /// Scheduled MTP verify forward workspace. The verifier runs a tiny
    /// `depth + 1` row forward; keep the row-major layer temporaries resident so
    /// it does not fall back to the generic prefill allocator on every layer.
    spec_verify: Option<Dsv4SpecVerifyScratch>,
    start_pos_device: CudaSlice<i32>,
    decode_graph: Option<Dsv4DecodeGraphScratch>,
    /// Pre-allocated NVSHMEM low-latency MoE scratch, reused across all layers +
    /// decode steps (overwritten in place each `dsv4_moe_forward_deepep_ll`
    /// call). `Some` only when the `deepep_ll` transport is booted. Held once per
    /// slot — layers run sequentially so a single scratch suffices.
    #[cfg(feature = "deepep")]
    deepep_ll_scratch: Option<crate::deepep::DeepEpLlScratch>,
    /// DSpark T3 tap buffers: the wide HC residual stream captured at each
    /// `config.dspark_target_layer_ids` layer OUTPUT (`hidden_states[layer+1]`),
    /// one `DeviceVec` sized for a verify block, index-aligned with the id list.
    /// Decode writes row 0; verify writes its full batch. Empty unless DSpark.
    dspark_taps: Vec<DeviceVec>,
    /// Transient multi-row prompt taps captured during a PREFILL forward
    /// (`seq_len > 1` under `is_dspark()`): the full wide HC stream
    /// `[stream_dim, seq_len]` per target layer, consumed once by the DSpark
    /// prefix seed ([`crate::dsv4::dspark`]) and then dropped. Prefill-scoped and
    /// bounded by the chunked-prefill chunk size — deliberately EXCLUDED from the
    /// per-slot VRAM ledger (like `decode_graph`, it is not resident at build).
    dspark_prompt_taps: Option<Dsv4DsparkPromptTaps>,
    seq_len: usize,
    max_seq_len: usize,
}

/// Host image of one whole DSv4 slot for capacity spill. Captures the per-layer
/// attention carry (SW ring, compressor/indexer, FlashMLA metadata, DSA) plus
/// the slot's FP8 KV band pages from each layer's shared pool.
pub(crate) struct Dsv4SlotImage {
    pub(crate) seq_len: usize,
    pub(crate) layers: Vec<crate::attention::Dsv4LayerAttentionImage>,
    pub(crate) kv_pages: Vec<Vec<u8>>,
}

impl Dsv4SlotImage {
    pub(crate) fn dram_bytes(&self) -> usize {
        let mut bytes = 8 + 8;
        for layer in &self.layers {
            bytes += layer.sw_window_cache.len() * 2;
            if let Some(c) = &layer.compressor {
                bytes += c.pending_kv.len() * 2
                    + c.pending_score.len() * 2
                    + c.prev_overlap_kv.len() * 2
                    + c.prev_overlap_score.len() * 2
                    + c.compressed.len() * 2
                    + c.fp32_pending_kv.len() * 4
                    + c.fp32_pending_score.len() * 4
                    + c.fp32_prev_kv.len() * 4
                    + c.fp32_prev_score.len() * 4
                    + 8 * 4;
            }
            if let Some(c) = &layer.indexer {
                bytes += c.pending_kv.len() * 2
                    + c.pending_score.len() * 2
                    + c.prev_overlap_kv.len() * 2
                    + c.prev_overlap_score.len() * 2
                    + c.compressed.len() * 2
                    + c.fp32_pending_kv.len() * 4
                    + c.fp32_pending_score.len() * 4
                    + c.fp32_prev_kv.len() * 4
                    + c.fp32_prev_score.len() * 4
                    + 8 * 4;
            }
            if let Some(f) = &layer.flashmla {
                bytes += f.topk_length.len() * 4
                    + f.sched_meta.len() * 4
                    + f.num_splits.len() * 4
                    + f.device_page_table.len() * 4
                    + 8 * 8;
            }
            if let Some(d) = &layer.dsa_official {
                bytes += d.rotated_keys.len() * 2 + 8 * 2;
            }
        }
        for pages in &self.kv_pages {
            bytes += pages.len();
        }
        bytes
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.dram_bytes() + 256);
        buf.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.layers.len() as u64).to_le_bytes());
        for layer in &self.layers {
            write_bf16_vec(&mut buf, &layer.sw_window_cache);
            write_compressor(&mut buf, layer.compressor.as_ref());
            write_compressor(&mut buf, layer.indexer.as_ref());
            write_flashmla(&mut buf, layer.flashmla.as_ref());
            write_dsa(&mut buf, layer.dsa_official.as_ref());
        }
        buf.extend_from_slice(&(self.kv_pages.len() as u64).to_le_bytes());
        for pages in &self.kv_pages {
            buf.extend_from_slice(&(pages.len() as u64).to_le_bytes());
            buf.extend_from_slice(pages);
        }
        buf
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let read_u64 = |pos: &mut usize| -> Result<u64> {
            let end = pos
                .checked_add(8)
                .ok_or_else(|| anyhow!("dsv4 slot image u64 overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("dsv4 slot image truncated u64 at {pos}"))?;
            *pos = end;
            Ok(u64::from_le_bytes(slice.try_into().unwrap()))
        };
        let read_u8 = |pos: &mut usize| -> Result<u8> {
            let end = pos
                .checked_add(1)
                .ok_or_else(|| anyhow!("dsv4 slot image u8 overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("dsv4 slot image truncated u8 at {pos}"))?;
            *pos = end;
            Ok(slice[0])
        };
        let read_bytes = |pos: &mut usize| -> Result<Vec<u8>> {
            let len = read_u64(pos)? as usize;
            let end = pos
                .checked_add(len)
                .ok_or_else(|| anyhow!("dsv4 slot image bytes overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("dsv4 slot image truncated bytes at {pos}"))?;
            *pos = end;
            Ok(slice.to_vec())
        };
        let read_bf16_vec = |pos: &mut usize| -> Result<Vec<half::bf16>> {
            let raw = read_bytes(pos)?;
            ensure!(
                raw.len() % 2 == 0,
                "dsv4 slot image bf16 vec odd length {}",
                raw.len()
            );
            Ok(raw
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect())
        };
        let read_f32_vec = |pos: &mut usize| -> Result<Vec<f32>> {
            let raw = read_bytes(pos)?;
            ensure!(
                raw.len() % 4 == 0,
                "dsv4 slot image f32 vec odd length {}",
                raw.len()
            );
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let read_i32_vec = |pos: &mut usize| -> Result<Vec<i32>> {
            let raw = read_bytes(pos)?;
            ensure!(
                raw.len() % 4 == 0,
                "dsv4 slot image i32 vec odd length {}",
                raw.len()
            );
            Ok(raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };

        let seq_len = read_u64(&mut pos)? as usize;
        let num_layers = read_u64(&mut pos)? as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let sw_window_cache = read_bf16_vec(&mut pos)?;
            let compressor = read_compressor(&mut pos, &read_u8, &read_bf16_vec, &read_f32_vec)?;
            let indexer = read_compressor(&mut pos, &read_u8, &read_bf16_vec, &read_f32_vec)?;
            let flashmla = read_flashmla(&mut pos, &read_u8, &read_i32_vec)?;
            let dsa_official = read_dsa(&mut pos, &read_u8, &read_bf16_vec)?;
            layers.push(crate::attention::Dsv4LayerAttentionImage {
                sw_window_cache,
                compressor,
                indexer,
                flashmla,
                dsa_official,
            });
        }
        let num_kv = read_u64(&mut pos)? as usize;
        let mut kv_pages = Vec::with_capacity(num_kv);
        for _ in 0..num_kv {
            kv_pages.push(read_bytes(&mut pos)?);
        }
        Ok(Dsv4SlotImage {
            seq_len,
            layers,
            kv_pages,
        })
    }
}

fn write_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.extend_from_slice(data);
}

fn write_bf16_vec(buf: &mut Vec<u8>, v: &[half::bf16]) {
    let bytes: Vec<u8> = v.iter().flat_map(|b| b.to_le_bytes()).collect();
    write_bytes(buf, &bytes);
}

fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    write_bytes(buf, &bytes);
}

fn write_i32_vec(buf: &mut Vec<u8>, v: &[i32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|i| i.to_le_bytes()).collect();
    write_bytes(buf, &bytes);
}

fn write_compressor(buf: &mut Vec<u8>, c: Option<&crate::attention::Dsv4CompressorImage>) {
    match c {
        None => buf.push(0),
        Some(c) => {
            buf.push(1);
            write_bf16_vec(buf, &c.pending_kv);
            write_bf16_vec(buf, &c.pending_score);
            write_bf16_vec(buf, &c.prev_overlap_kv);
            write_bf16_vec(buf, &c.prev_overlap_score);
            write_bf16_vec(buf, &c.compressed);
            buf.extend_from_slice(&(c.compressed_seq_len as u64).to_le_bytes());
            buf.extend_from_slice(&(c.compressed_capacity as u64).to_le_bytes());
            buf.extend_from_slice(&(c.ring_rows as u64).to_le_bytes());
            write_f32_vec(buf, &c.fp32_pending_kv);
            write_f32_vec(buf, &c.fp32_pending_score);
            write_f32_vec(buf, &c.fp32_prev_kv);
            write_f32_vec(buf, &c.fp32_prev_score);
            buf.push(c.fp32_carry_stale as u8);
        }
    }
}

fn write_flashmla(buf: &mut Vec<u8>, f: Option<&crate::attention::Dsv4FlashMlaImage>) {
    match f {
        None => buf.push(0),
        Some(f) => {
            buf.push(1);
            buf.extend_from_slice(&(f.fp8_kv_pool_len as u64).to_le_bytes());
            buf.extend_from_slice(&(f.sw_blocks as u64).to_le_bytes());
            buf.extend_from_slice(&(f.comp_blocks as u64).to_le_bytes());
            buf.extend_from_slice(&(f.max_compressed_keys as u64).to_le_bytes());
            buf.extend_from_slice(&(f.topk_unified as u64).to_le_bytes());
            buf.extend_from_slice(&(f.page_block_size as u64).to_le_bytes());
            buf.push(f.fp8_kv_sw_bootstrapped as u8);
            buf.extend_from_slice(&(f.fp8_kv_comp_packed_rows as u64).to_le_bytes());
            write_i32_vec(buf, &f.topk_length);
            write_i32_vec(buf, &f.sched_meta);
            write_i32_vec(buf, &f.num_splits);
            buf.extend_from_slice(&f.num_sm_parts.to_le_bytes());
            buf.extend_from_slice(&f.fixed_overhead_num_blocks.to_le_bytes());
            buf.extend_from_slice(&f.block_size_topk.to_le_bytes());
            write_i32_vec(buf, &f.device_page_table);
        }
    }
}

fn write_dsa(buf: &mut Vec<u8>, d: Option<&crate::attention::Dsv4DsaImage>) {
    match d {
        None => buf.push(0),
        Some(d) => {
            buf.push(1);
            buf.extend_from_slice(&(d.key_cache_len as u64).to_le_bytes());
            write_bf16_vec(buf, &d.rotated_keys);
            buf.extend_from_slice(&(d.packed_rows as u64).to_le_bytes());
        }
    }
}

fn read_compressor(
    pos: &mut usize,
    read_u8: &impl Fn(&mut usize) -> Result<u8>,
    read_bf16_vec: &impl Fn(&mut usize) -> Result<Vec<half::bf16>>,
    read_f32_vec: &impl Fn(&mut usize) -> Result<Vec<f32>>,
) -> Result<Option<crate::attention::Dsv4CompressorImage>> {
    let flag = read_u8(pos)?;
    if flag == 0 {
        return Ok(None);
    }
    let pending_kv = read_bf16_vec(pos)?;
    let pending_score = read_bf16_vec(pos)?;
    let prev_overlap_kv = read_bf16_vec(pos)?;
    let prev_overlap_score = read_bf16_vec(pos)?;
    let compressed = read_bf16_vec(pos)?;
    let compressed_seq_len = read_u64_helper(pos, read_u8)? as usize;
    let compressed_capacity = read_u64_helper(pos, read_u8)? as usize;
    let ring_rows = read_u64_helper(pos, read_u8)? as usize;
    let fp32_pending_kv = read_f32_vec(pos)?;
    let fp32_pending_score = read_f32_vec(pos)?;
    let fp32_prev_kv = read_f32_vec(pos)?;
    let fp32_prev_score = read_f32_vec(pos)?;
    let fp32_carry_stale = read_u8(pos)? != 0;
    Ok(Some(crate::attention::Dsv4CompressorImage {
        pending_kv,
        pending_score,
        prev_overlap_kv,
        prev_overlap_score,
        compressed,
        compressed_seq_len,
        compressed_capacity,
        ring_rows,
        fp32_pending_kv,
        fp32_pending_score,
        fp32_prev_kv,
        fp32_prev_score,
        fp32_carry_stale,
    }))
}

fn read_u64_helper(pos: &mut usize, read_u8: &impl Fn(&mut usize) -> Result<u8>) -> Result<u64> {
    let mut bytes = [0u8; 8];
    for b in bytes.iter_mut() {
        *b = read_u8(pos)?;
    }
    Ok(u64::from_le_bytes(bytes))
}

fn read_flashmla(
    pos: &mut usize,
    read_u8: &impl Fn(&mut usize) -> Result<u8>,
    read_i32_vec: &impl Fn(&mut usize) -> Result<Vec<i32>>,
) -> Result<Option<crate::attention::Dsv4FlashMlaImage>> {
    let flag = read_u8(pos)?;
    if flag == 0 {
        return Ok(None);
    }
    let fp8_kv_pool_len = read_u64_helper(pos, read_u8)? as usize;
    let sw_blocks = read_u64_helper(pos, read_u8)? as usize;
    let comp_blocks = read_u64_helper(pos, read_u8)? as usize;
    let max_compressed_keys = read_u64_helper(pos, read_u8)? as usize;
    let topk_unified = read_u64_helper(pos, read_u8)? as usize;
    let page_block_size = read_u64_helper(pos, read_u8)? as usize;
    let fp8_kv_sw_bootstrapped = read_u8(pos)? != 0;
    let fp8_kv_comp_packed_rows = read_u64_helper(pos, read_u8)? as usize;
    let topk_length = read_i32_vec(pos)?;
    let sched_meta = read_i32_vec(pos)?;
    let num_splits = read_i32_vec(pos)?;
    let num_sm_parts = read_i32_helper(pos, read_u8)?;
    let fixed_overhead_num_blocks = read_i32_helper(pos, read_u8)?;
    let block_size_topk = read_i32_helper(pos, read_u8)?;
    let device_page_table = read_i32_vec(pos)?;
    Ok(Some(crate::attention::Dsv4FlashMlaImage {
        fp8_kv_pool_len,
        sw_blocks,
        comp_blocks,
        max_compressed_keys,
        topk_unified,
        page_block_size,
        fp8_kv_sw_bootstrapped,
        fp8_kv_comp_packed_rows,
        topk_length,
        sched_meta,
        num_splits,
        num_sm_parts,
        fixed_overhead_num_blocks,
        block_size_topk,
        device_page_table,
    }))
}

fn read_i32_helper(pos: &mut usize, read_u8: &impl Fn(&mut usize) -> Result<u8>) -> Result<i32> {
    let mut bytes = [0u8; 4];
    for b in bytes.iter_mut() {
        *b = read_u8(pos)?;
    }
    Ok(i32::from_le_bytes(bytes))
}

fn read_dsa(
    pos: &mut usize,
    read_u8: &impl Fn(&mut usize) -> Result<u8>,
    read_bf16_vec: &impl Fn(&mut usize) -> Result<Vec<half::bf16>>,
) -> Result<Option<crate::attention::Dsv4DsaImage>> {
    let flag = read_u8(pos)?;
    if flag == 0 {
        return Ok(None);
    }
    let key_cache_len = read_u64_helper(pos, read_u8)? as usize;
    let rotated_keys = read_bf16_vec(pos)?;
    let packed_rows = read_u64_helper(pos, read_u8)? as usize;
    Ok(Some(crate::attention::Dsv4DsaImage {
        key_cache_len,
        rotated_keys,
        packed_rows,
    }))
}

/// Full-prompt-chunk tap capture for the DSpark prefix seed. Each buffer holds
/// one target layer's wide HC stream `[stream_dim * rows]` token-major (column
/// `j*hc_mult + r` at `(j*hc_mult + r) * hidden`), index-aligned with
/// `dspark_target_layer_ids`. Transient: allocated per prefill forward, taken by
/// the seed.
pub(crate) struct Dsv4DsparkPromptTaps {
    pub(crate) bufs: Vec<DeviceVec>,
    pub(crate) rows: usize,
}

struct Dsv4SpecVerifyScratch {
    /// Allocated row capacity ([`Dsv4Model::spec_verify_rows`]) — the ceiling
    /// `set_rows` validates against.
    rows: usize,
    embeddings: HiddenStates,
    initial_stream: HiddenStates,
    layers: Vec<Dsv4SpecVerifyLayerScratch>,
}

struct Dsv4SpecVerifyLayerScratch {
    attn_normed: HiddenStates,
    attn_out: HiddenStates,
    attn_stream: HiddenStates,
    ffn_normed: HiddenStates,
    moe_out: HiddenStates,
    moe_with_shared: HiddenStates,
    ffn_stream: HiddenStates,
}

impl Dsv4SpecVerifyScratch {
    fn new(model: &Dsv4Model) -> Result<Self> {
        let hidden_size = model.config.hidden_size;
        let stream_dim = hidden_size * model.config.hc_mult;
        let rows = model.spec_verify_rows();
        let layers: Vec<_> = (0..model.layers.len())
            .map(|_| Dsv4SpecVerifyLayerScratch::new(&model.ctx, hidden_size, stream_dim, rows))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rows,
            // SAFETY: uninit device scratch; fully written before first read.
            embeddings: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            initial_stream: unsafe { HiddenStates::uninit(&model.ctx, stream_dim, rows)? },
            layers,
        })
    }

    fn set_rows(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.rows,
            "DSv4 spec-verify scratch rows {rows} exceed allocated {}",
            self.rows
        );
        self.embeddings.seq_len = rows;
        self.initial_stream.seq_len = rows;
        for layer in &mut self.layers {
            layer.set_rows(rows);
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn device_bytes(&self) -> usize {
        self.embeddings.device_bytes()
            + self.initial_stream.device_bytes()
            + self
                .layers
                .iter()
                .map(Dsv4SpecVerifyLayerScratch::device_bytes)
                .sum::<usize>()
    }
}

impl Dsv4SpecVerifyLayerScratch {
    fn new(
        ctx: &DeviceContext,
        hidden_size: usize,
        stream_dim: usize,
        rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            // SAFETY: uninit device scratch; fully written before first read.
            attn_normed: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_out: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_stream: unsafe { HiddenStates::uninit(ctx, stream_dim, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            ffn_normed: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            moe_out: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            moe_with_shared: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            ffn_stream: unsafe { HiddenStates::uninit(ctx, stream_dim, rows)? },
        })
    }

    fn set_rows(&mut self, rows: usize) {
        self.attn_normed.seq_len = rows;
        self.attn_out.seq_len = rows;
        self.attn_stream.seq_len = rows;
        self.ffn_normed.seq_len = rows;
        self.moe_out.seq_len = rows;
        self.moe_with_shared.seq_len = rows;
        self.ffn_stream.seq_len = rows;
    }

    #[allow(dead_code)]
    fn device_bytes(&self) -> usize {
        self.attn_normed.device_bytes()
            + self.attn_out.device_bytes()
            + self.attn_stream.device_bytes()
            + self.ffn_normed.device_bytes()
            + self.moe_out.device_bytes()
            + self.moe_with_shared.device_bytes()
            + self.ffn_stream.device_bytes()
    }
}

pub(crate) struct Dsv4DecodeGraphScratch {
    token_ids: CudaSlice<i32>,
    token_ids_u32: CudaSlice<u32>,
    embeddings: HiddenStates,
    initial_stream: HiddenStates,
    layers: Vec<Dsv4DecodeLayerGraphScratch>,
    tail_graph: crate::graph::CudaGraphState,
    whole_graph: crate::graph::CudaGraphState,
    head_stream_row: HiddenStates,
    head_mixes: HiddenStates,
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
    mla: crate::attention::Dsv4MlaDecodeGraphScratch,
    // Kept for the unfused HC-pre path; current decode graph uses fused MHC pre RMS.
    #[allow(dead_code)]
    attn_in: HiddenStates,
    attn_normed: HiddenStates,
    attn_out: HiddenStates,
    attn_stream: HiddenStates,
    // Kept for the unfused HC-pre path; current decode graph uses fused MHC pre RMS.
    #[allow(dead_code)]
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
        // SAFETY: uninit device scratch; fully written before first read.
        let embeddings = unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let initial_stream = unsafe { HiddenStates::uninit(&model.ctx, stream_dim, 1)? };
        let layers: Vec<_> = model
            .layers
            .iter()
            .map(|layer| Dsv4DecodeLayerGraphScratch::new(model, layer))
            .collect::<Result<Vec<_>>>()?;
        let last_hidden = DeviceVec::zeros(&model.ctx, hidden_size)?;
        let last_normed = DeviceVec::zeros(&model.ctx, hidden_size)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let head_stream_row = unsafe { HiddenStates::uninit(&model.ctx, stream_dim, 1)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let head_mixes = unsafe { HiddenStates::uninit(&model.ctx, model.head_hc.mix_fn.rows, 1)? };
        // SAFETY: uninit device scratch; fully written before first read.
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
            head_stream_row,
            head_mixes,
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
            mla: crate::attention::Dsv4MlaDecodeGraphScratch::new(
                &model.ctx,
                &model.config,
                &layer.attention,
                layer.mode,
                layer.compress_ratio,
            )?,
            // SAFETY: uninit device scratch; fully written before first read.
            attn_in: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_normed: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_out: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_stream: unsafe { HiddenStates::uninit(&model.ctx, stream_dim, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            ffn_in: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            ffn_normed: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            moe_out: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            shared: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            moe_with_shared: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
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
        let attention: Vec<_> = model
            .layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                let local_width = layer.attention.wq_b.rows;
                ensure!(
                    local_width.is_multiple_of(model.config.head_dim),
                    "DSv4 slot attention local width {local_width} is not a multiple of head_dim {}",
                    model.config.head_dim
                );
                let pool = kv_adapter.layer(layer_idx)?;
                crate::attention::Dsv4LayerAttentionState::new(
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
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Pre-allocate the per-layer spec-ring snapshot when spec decode is on.
        // No per-step allocation; capture/restore only copies into these buffers.
        let spec_rings = model
            .spec_decode_on
            .then(|| {
                attention
                    .iter()
                    .map(|state| {
                        state.alloc_spec_ring_snapshot(
                            &model.ctx,
                            &model.config,
                            &model.kv_arena,
                            MAX_SPEC_DRAFT_DEPTH,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        // P2 commit-fold scratch: per-layer persisted verify rows.
        let spec_normed = model
            .spec_decode_on
            .then(|| {
                (0..attention.len())
                    .map(|_| {
                        // SAFETY: rows are written by the verify lane before any read.
                        unsafe {
                            HiddenStates::uninit(
                                &model.ctx,
                                model.config.hidden_size,
                                model.spec_verify_rows(),
                            )
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let spec_verify = if model.spec_decode_on {
            Some(Dsv4SpecVerifyScratch::new(model)?)
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
                    .map(|layer| layer.moe.as_ref().expect("DSv4 layer.moe").intermediate)
                    .ok_or_else(|| anyhow!("DSv4 deepep_ll: model has no layers"))?;
                Some(transport.alloc_ll_scratch(&model.ctx, intermediate)?)
            }
            _ => None,
        };
        // DSpark tap buffers: one verify-block capture per target layer.
        // Empty (no alloc) unless this is a DSpark checkpoint.
        let dspark_taps = if model.config.is_dspark() {
            let stream_dim = model.config.hidden_size * model.config.hc_mult;
            let rows = model.config.dspark_block_size + 1;
            model
                .config
                .dspark_target_layer_ids
                .iter()
                .map(|_| DeviceVec::zeros(&model.ctx, stream_dim * rows))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            attention,
            spec_rings,
            spec_normed,
            spec_verify,
            start_pos_device,
            decode_graph: None,
            #[cfg(feature = "deepep")]
            deepep_ll_scratch,
            dspark_taps,
            dspark_prompt_taps: None,
            seq_len: 0,
            max_seq_len,
        })
    }

    /// Exact requested device bytes owned by this ONE slot: the per-layer
    /// attention states + the per-layer spec-ring snapshots + the per-layer
    /// spec-normed commit-fold scratch + the `start_pos_device` scalar.
    ///
    /// EXCLUDED (and why):
    /// - `decode_graph`: `None` at slot construction (lazy, captured on the
    ///   first decode step — well after engine-build reconciliation). Its
    ///   captured CUDA-graph internal allocations are opaque to a byte sum.
    /// - `deepep_ll_scratch`: lives behind `#[cfg(feature = "deepep")]` and is
    ///   only `Some` when the NVSHMEM LL transport is booted; sized off-band.
    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes_breakdown().iter().map(|(_, b)| *b).sum()
    }

    /// Per-component byte breakdown for the VRAM ledger log. The per-layer
    /// `attention` Vec collapses to one summed entry (40+ layers); the spec
    /// scratches are itemized.
    #[allow(dead_code)]
    pub(crate) fn device_bytes_breakdown(&self) -> Vec<(&'static str, usize)> {
        let attention_bytes: usize = self.attention.iter().map(|s| s.device_bytes()).sum();
        let spec_rings_bytes: usize = self
            .spec_rings
            .as_ref()
            .map_or(0, |rings| rings.iter().map(|r| r.device_bytes()).sum());
        let spec_normed_bytes: usize = self
            .spec_normed
            .as_ref()
            .map_or(0, |cache| cache.iter().map(|h| h.device_bytes()).sum());
        let spec_verify_bytes = self
            .spec_verify
            .as_ref()
            .map_or(0, Dsv4SpecVerifyScratch::device_bytes);
        let dspark_taps_bytes: usize = self
            .dspark_taps
            .iter()
            .map(|v| v.len * std::mem::size_of::<half::bf16>())
            .sum();
        vec![
            ("attention(per-layer)", attention_bytes),
            ("spec_rings", spec_rings_bytes),
            ("spec_normed", spec_normed_bytes),
            ("spec_verify", spec_verify_bytes),
            (
                "start_pos_device",
                self.start_pos_device.len() * std::mem::size_of::<i32>(),
            ),
            ("dspark_taps", dspark_taps_bytes),
        ]
    }

    /// DSpark T3: wide-stream taps captured at `config.dspark_target_layer_ids`
    /// (index-aligned), empty off the DSpark path. The T4 draft reads these to
    /// build its fused context; unused until then.
    #[allow(dead_code)]
    pub(crate) fn dspark_taps(&self) -> &[DeviceVec] {
        &self.dspark_taps
    }

    /// Take the transient prompt-chunk taps captured by the last prefill forward
    /// (`None` off the DSpark path or when the forward was single-row). Consumed
    /// once by the DSpark prefix seed.
    #[allow(dead_code)]
    pub(crate) fn take_dspark_prompt_taps(&mut self) -> Option<Dsv4DsparkPromptTaps> {
        self.dspark_prompt_taps.take()
    }

    /// Sub-struct byte totals summed across ALL attention layers (sw_window /
    /// compressor / indexer / flashmla / fused_wqkv / dsa_official). The
    /// [vram-ledger] uses this to attribute the per-slot `attention(per-layer)`
    /// bulk to the exact buffer family — every bit's source named.
    #[allow(dead_code)]
    pub(crate) fn attention_breakdown_total(&self) -> Vec<(&'static str, usize)> {
        let mut totals: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for layer in &self.attention {
            for (name, bytes) in layer.device_bytes_breakdown() {
                *totals.entry(name).or_insert(0) += bytes;
            }
        }
        totals.into_iter().collect()
    }

    /// Snapshot the speculative-verify boundary ring slot across all attention
    /// layers before the depth-K verify forward. No-op when spec decode is off.
    pub(crate) fn capture_spec_rings(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        depth: usize,
    ) -> Result<()> {
        let Some(rings) = self.spec_rings.as_mut() else {
            return Ok(());
        };
        ensure!(
            self.attention.len() == rings.len(),
            "DSv4 spec-ring layer count {} != attention states {}",
            rings.len(),
            self.attention.len()
        );
        for (layer_idx, (state, snap)) in self.attention.iter().zip(rings).enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            state.capture_spec_rings(ctx, pool, snap, start_pos, depth)?;
        }
        Ok(())
    }

    /// Restore the speculative boundary ring slot across all attention layers
    /// AFTER the commit truncate and BEFORE the accepted-prefix fold. No-op
    /// when spec decode is off.
    pub(crate) fn restore_spec_ring_tail(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        accepted_n: usize,
        depth: usize,
    ) -> Result<()> {
        let Some(rings) = self.spec_rings.as_ref() else {
            return Ok(());
        };
        ensure!(
            self.attention.len() == rings.len(),
            "DSv4 spec-ring restore layer count {} != attention states {}",
            rings.len(),
            self.attention.len()
        );
        for (layer_idx, (state, snap)) in self.attention.iter_mut().zip(rings).enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            state.restore_spec_ring_tail(ctx, pool, snap, start_pos, accepted_n, depth)?;
        }
        Ok(())
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
        self.dspark_prompt_taps = None;
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

    /// Re-sync every layer's FlashMLA device page table from the host pool.
    /// Called whenever the adapter reports the slot's host band changed —
    /// dirty-bit driven, prefill AND decode (see
    /// `Dsv4KvAdapter::take_device_table_dirty`); host tables carry only real
    /// pages, padding to the fixed device size happens in
    /// [`crate::attention::Dsv4LayerAttentionState::refresh_flashmla_device_page_table`].
    pub(crate) fn refresh_flashmla_device_page_tables(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
    ) -> Result<()> {
        for (layer_idx, layer) in self.attention.iter_mut().enumerate() {
            let pool = kv_adapter.layer(layer_idx)?;
            layer.refresh_flashmla_device_page_table(ctx, pool)?;
        }
        Ok(())
    }

    /// D2H one completed host page's state across every layer into a
    /// content-keyed pool entry (#154 Phase 2 publish hook). `boundary` = the
    /// forward ended exactly at this page's end (the only moment the page-end
    /// overlap registers + ring are observable). The D2H clones are
    /// stream-ordered; the CALLER syncs once after all pages of the tick
    /// before reading any entry (one sync per tick, not per page).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prefix_page(
        &self,
        ctx: &DeviceContext,
        layers: &[Dsv4Layer],
        kv_adapter: &crate::attention::Dsv4KvAdapter,
        index_head_dim: usize,
        page_tokens: usize,
        page_index: usize,
        boundary: bool,
    ) -> Result<crate::attention::Dsv4PrefixPageEntry> {
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 prefix capture layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        let states = self
            .attention
            .iter()
            .enumerate()
            .map(|(idx, state)| {
                let pool = kv_adapter.layer(idx)?;
                state.capture_prefix_page(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    boundary,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(crate::attention::Dsv4PrefixPageEntry {
            page_index: u32::try_from(page_index)
                .map_err(|_| anyhow!("DSv4 prefix page index {page_index} exceeds u32"))?,
            boundary,
            layers: states,
        })
    }

    /// Capture the finish frontier page (`--dsv4-decode-reuse`): the frontier
    /// page's own content + carry (boundary) PLUS the sub-page tail the radix
    /// match can't cover (`[matched_len, finish_len)`). `page_index =
    /// matched_len/page_tokens − 1`; the whole carry reflects `finish_len`
    /// because capture runs after the finish forward.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_frontier_page(
        &self,
        ctx: &DeviceContext,
        layers: &[Dsv4Layer],
        kv_adapter: &crate::attention::Dsv4KvAdapter,
        index_head_dim: usize,
        page_tokens: usize,
        page_index: usize,
        matched_len: usize,
        finish_len: usize,
    ) -> Result<crate::attention::Dsv4PrefixPageEntry> {
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 frontier capture layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        let states = self
            .attention
            .iter()
            .enumerate()
            .map(|(idx, state)| {
                let pool = kv_adapter.layer(idx)?;
                let mut s = state.capture_prefix_page(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    true,
                )?;
                state.capture_frontier_tail(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    matched_len,
                    finish_len,
                    &mut s,
                )?;
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(crate::attention::Dsv4PrefixPageEntry {
            page_index: u32::try_from(page_index)
                .map_err(|_| anyhow!("DSv4 frontier page index {page_index} exceeds u32"))?,
            boundary: true,
            layers: states,
        })
    }

    /// Restore a cross-request prefix into this slot from content-keyed pool
    /// entries (#154 Phase 2 read hook). `entries[k]` is host page k's state;
    /// the last entry must carry the boundary sections. The caller has already
    /// mirrored the identity band (`Dsv4KvAdapter::mirror_full_band`), which
    /// also set the band cursor and the device-table dirty bit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_prefix_state(
        &mut self,
        ctx: &DeviceContext,
        layers: &[Dsv4Layer],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        index_head_dim: usize,
        entries: &[crate::attention::Dsv4PrefixPageEntry],
        matched_len: usize,
        finish_len: usize,
        page_tokens: usize,
    ) -> Result<()> {
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 prefix restore layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        ensure!(
            page_tokens > 0 && matched_len == entries.len() * page_tokens,
            "DSv4 prefix restore matched_len {matched_len} != {} pages × {page_tokens}",
            entries.len()
        );
        // finish_len is the exact frontier: matched_len (aligned) plus the last
        // entry's sub-page tail. The tail is < page_tokens by construction.
        ensure!(
            (matched_len..matched_len + page_tokens).contains(&finish_len),
            "DSv4 prefix restore finish_len {finish_len} not in [matched_len {matched_len}, +{page_tokens})"
        );
        ensure!(
            finish_len <= self.max_seq_len,
            "DSv4 prefix restore finish_len {finish_len} exceeds slot max_seq_len {}",
            self.max_seq_len
        );
        ensure!(
            entries.last().is_some_and(|e| e.boundary),
            "DSv4 prefix restore: final matched page lacks boundary sections"
        );
        for (k, entry) in entries.iter().enumerate() {
            ensure!(
                entry.page_index as usize == k,
                "DSv4 prefix restore: entry captured at page {} restored at {k} \
                 (recycled host page id?)",
                entry.page_index
            );
            ensure!(
                entry.layers.len() == self.attention.len(),
                "DSv4 prefix restore: entry layer count {} != attention states {}",
                entry.layers.len(),
                self.attention.len()
            );
            let boundary = k + 1 == entries.len();
            for (idx, (state, layer_state)) in
                self.attention.iter_mut().zip(&entry.layers).enumerate()
            {
                let pool = kv_adapter.layer_mut(idx)?;
                state.restore_prefix_page(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    page_tokens,
                    k,
                    layer_state,
                    boundary,
                )?;
            }
        }
        // Frontier tail: the sub-page leftover the last matched page's own rows
        // don't cover. A no-op when finish_len == matched_len (aligned finish /
        // plain restore) — every tail section is then empty.
        if let Some(frontier) = entries.last() {
            for (idx, (state, layer_state)) in
                self.attention.iter_mut().zip(&frontier.layers).enumerate()
            {
                let pool = kv_adapter.layer_mut(idx)?;
                state.restore_frontier_tail(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    matched_len,
                    finish_len,
                    layer_state,
                )?;
            }
        }
        for (idx, state) in self.attention.iter_mut().enumerate() {
            state.restore_prefix_counters(layers[idx].mode, layers[idx].compress_ratio, finish_len);
        }
        self.seq_len = finish_len;
        // Same request-boundary discipline as reset/swap_in: re-arm one eager
        // warm step so per-request host work (SW ring bootstrap, compressed
        // bulk pack) reruns against the restored bands before replay resumes.
        if let Some(graph) = self.decode_graph.as_mut() {
            graph.rearm_for_new_request();
        }
        ctx.sync()?;
        Ok(())
    }

    pub(crate) fn truncate(
        &mut self,
        layers: &[Dsv4Layer],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        new_len: usize,
    ) -> Result<()> {
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
        for (layer_idx, (layer, state)) in layers.iter().zip(&mut self.attention).enumerate() {
            state.truncate_decode_len(layer.mode, layer.compress_ratio, new_len);
            // H2: clamp the FlashMLA pool cursor in lockstep (see flashmla_truncate_slot).
            if let Some(slot_idx) = state.flashmla_slot_idx() {
                kv_adapter
                    .layer_mut(layer_idx)?
                    .flashmla_truncate_slot(slot_idx, new_len)?;
            }
        }
        Ok(())
    }

    pub(crate) fn swap_out_image(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_idx: usize,
    ) -> Result<Dsv4SlotImage> {
        let num_layers = self.attention.len();
        let mut layers = Vec::with_capacity(num_layers);
        let mut kv_pages = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let pool = kv_adapter.layer(layer_idx)?;
            let layer_image = self.attention[layer_idx].swap_out(ctx, pool)?;
            let pages = if let Some(flash) = &self.attention[layer_idx].flashmla {
                debug_assert_eq!(flash.slot_idx, slot_idx);
                let table = pool.flashmla_page_table(slot_idx)?;
                if table.is_empty() {
                    Vec::new()
                } else {
                    pool.flashmla_pool()?.copy_pages_to_host(ctx, table)?
                }
            } else {
                Vec::new()
            };
            layers.push(layer_image);
            kv_pages.push(pages);
        }
        ctx.sync()?;
        for layer_idx in 0..num_layers {
            if let Some(slot_idx_in_layer) = self.attention[layer_idx].flashmla_slot_idx() {
                kv_adapter
                    .layer_mut(layer_idx)?
                    .flashmla_free_slot(slot_idx_in_layer)?;
            }
        }
        let seq_len = self.seq_len;
        self.seq_len = 0;
        Ok(Dsv4SlotImage {
            seq_len,
            layers,
            kv_pages,
        })
    }

    pub(crate) fn swap_in_image(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_idx: usize,
        image: &Dsv4SlotImage,
    ) -> Result<()> {
        ensure!(
            image.layers.len() == self.attention.len(),
            "DSv4 swap-in layer count {} != attention states {}",
            image.layers.len(),
            self.attention.len()
        );
        kv_adapter.mirror_full_band(ctx, slot_idx, image.seq_len)?;
        for layer_idx in 0..self.attention.len() {
            if let Some(flash) = &self.attention[layer_idx].flashmla {
                debug_assert_eq!(flash.slot_idx, slot_idx);
                if !image.kv_pages[layer_idx].is_empty() {
                    let table: Vec<u32> = kv_adapter
                        .layer(layer_idx)?
                        .flashmla_page_table(slot_idx)?
                        .to_vec();
                    let expected = image.layers[layer_idx]
                        .flashmla
                        .as_ref()
                        .map(|f| f.device_page_table.len())
                        .unwrap_or(0);
                    ensure!(
                        table.len() == expected,
                        "DSv4 swap-in layer {layer_idx} page count {} != image {}",
                        table.len(),
                        expected
                    );
                    kv_adapter
                        .layer_mut(layer_idx)?
                        .flashmla_pool_mut()?
                        .copy_pages_from_host(ctx, &table, &image.kv_pages[layer_idx])?;
                }
            }
            let pool = kv_adapter.layer_mut(layer_idx)?;
            self.attention[layer_idx].swap_in(ctx, pool, &image.layers[layer_idx])?;
            self.attention[layer_idx]
                .refresh_flashmla_device_page_table(ctx, kv_adapter.layer(layer_idx)?)?;
        }
        self.seq_len = image.seq_len;
        ctx.sync()?;
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

/// Temporary diagnostic-only trace for the 2026-07 concurrent-decode digit-
/// corruption investigation (`docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`).
/// Env-gated (`ARLE_DSV4_DECODE_TRACE=1`), zero cost otherwise (one env::var
/// read per batched decode call). Rank-gated to `rank==0` only: all TP ranks
/// redundantly execute this per-row loop with identical results, and letting
/// >1 rank's process write to the shared serve-log fd caused byte-level
/// > interleaving between processes even with a single `write_all` call.
/// > Prints one line per call: the batch's slot/start_position/token triples,
/// > so a post-hoc script can (a) pick out a tracked slot's per-step token-id
/// > sequence via its `start_position` == its own prompt-token count, and (b)
/// > see the exact batch composition (which other slots were co-resident) at
/// > each step, without touching the hot sampling path's signature.
pub(crate) fn dsv4_decode_trace(
    rank: usize,
    slot_ids: &[usize],
    start_positions: &[usize],
    out_tokens: &[u32],
) {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    if rank != 0 || std::env::var_os("ARLE_DSV4_DECODE_TRACE").is_none() {
        return;
    }
    static CALL_ID: AtomicU64 = AtomicU64::new(0);
    let call = CALL_ID.fetch_add(1, Ordering::Relaxed);
    let rows: Vec<String> = slot_ids
        .iter()
        .zip(start_positions)
        .zip(out_tokens)
        .map(|((slot, start_pos), tok)| format!("{slot}:{start_pos}:{tok}"))
        .collect();
    // Single pre-formatted `write_all` call (one write() syscall) so the line
    // can't tear even if another writer shares the fd.
    let line = format!(
        "DSV4_TRACE call={call} n={} rows=[{}]\n",
        slot_ids.len(),
        rows.join(",")
    );
    let _ = std::io::stderr().write_all(line.as_bytes());
}

impl Dsv4Model {
    pub(crate) fn begin_mega_moe_forward(&self, _num_tokens: usize) -> Result<Option<u64>> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(mega_moe) = &self.mega_moe {
            return mega_moe.begin_forward(self, _num_tokens).map(Some);
        }
        Ok(None)
    }

    /// Per-forward owned-token cap when the deepep_ll MoE transport is booted
    /// (`None` for allreduce / intranode / non-deepep builds → unbounded). Core
    /// caps decode_rows + prefill chunk tokens to this so the LL dispatch buffer
    /// is never overrun.
    pub(crate) fn max_tokens_per_step(&self) -> Option<usize> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(mega_moe) = &self.mega_moe {
            return mega_moe
                .shape
                .requested_max_tokens_per_rank
                .checked_mul(mega_moe.shape.num_ranks);
        }
        #[cfg(feature = "deepep")]
        {
            self.deepep
                .as_ref()
                .and_then(|t| t.max_owned_tokens_per_forward())
        }
        #[cfg(not(feature = "deepep"))]
        {
            None
        }
    }

    pub(crate) fn boot_mega_moe(&mut self, _requested_max_tokens_per_rank: usize) -> Result<()> {
        if !matches!(
            crate::runtime_flags::dsv4_moe_transport()?,
            crate::runtime_flags::Dsv4MoeTransport::MegaMoe
        ) {
            return Ok(());
        }
        #[cfg(not(all(feature = "cuda", feature = "nccl")))]
        anyhow::bail!("ARLE_DSV4_MOE_TRANSPORT=mega_moe requires infer-cuda features cuda,nccl");
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        {
            ensure!(
                self.mega_moe.is_none(),
                "DSv4 MegaMoE transport already booted"
            );
            ensure!(
                self.tp.config().world_size > 1,
                "ARLE_DSV4_MOE_TRANSPORT=mega_moe requires TP world_size > 1"
            );
            let first = self
                .layers
                .iter()
                .find_map(|layer| layer.moe.as_ref())
                .ok_or_else(|| anyhow!("DSv4 MegaMoE requires at least one routed MoE layer"))?;
            ensure!(
                self.layers
                    .iter()
                    .filter_map(|layer| layer.moe.as_ref())
                    .all(|moe| {
                        moe.hidden_dim == first.hidden_dim
                            && moe.intermediate == first.intermediate
                            && moe.num_groups == first.num_groups
                    }),
                "DSv4 MegaMoE requires one routed-expert shape across all layers"
            );
            if let Some(mtp) = &self.mtp {
                let moe = mtp
                    .layer
                    .moe
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 MegaMoE MTP layer has no routed experts"))?;
                ensure!(
                    moe.hidden_dim == first.hidden_dim
                        && moe.intermediate == first.intermediate
                        && moe.num_groups == first.num_groups,
                    "DSv4 MegaMoE MTP routed-expert shape differs from base layers"
                );
            }
            let shape = cuda_kernels::moe::Sm90MegaMoeShape {
                num_ranks: self.tp.config().world_size,
                num_experts: self.moe_config.num_experts,
                requested_max_tokens_per_rank: _requested_max_tokens_per_rank,
                num_topk: self.moe_config.top_k,
                hidden: first.hidden_dim,
                intermediate_hidden: first.intermediate,
            };
            let layout = cuda_kernels::moe::sm90_mega_moe_workspace_layout(shape)?;
            let bytes = usize::try_from(layout.num_bytes)?;
            let workspace = self.tp.alloc_symmetric_ipc(&self.ctx, bytes)?;
            ensure!(
                workspace.bytes() == bytes,
                "DSv4 MegaMoE workspace bytes {} != layout {bytes}",
                workspace.bytes()
            );
            let mega_moe = Dsv4MegaMoeTransport {
                workspace,
                owned_out: self
                    .ctx
                    .stream
                    .alloc_zeros::<half::bf16>(
                        _requested_max_tokens_per_rank
                            .checked_mul(first.hidden_dim)
                            .ok_or_else(|| anyhow!("DSv4 MegaMoE output scratch overflow"))?,
                    )
                    .map_err(|error| {
                        anyhow!("DSv4 MegaMoE output scratch alloc failed: {error}")
                    })?,
                layout,
                shape,
                fast_math: true,
                enable_pdl: false,
                epoch: std::sync::Mutex::new((0, None)),
            };
            mega_moe.assert_static_spec(self, self.config.swiglu_limit)?;
            self.mega_moe = Some(mega_moe);
            Ok(())
        }
    }

    pub(crate) fn truncate_slot(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        new_len: usize,
    ) -> Result<()> {
        slot.truncate(&self.layers, kv_adapter, new_len)
    }

    /// Snapshot the speculative-verify boundary ring slot before depth-K verify.
    /// No-op when spec decode is off.
    pub(crate) fn capture_spec_rings(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        depth: usize,
    ) -> Result<()> {
        slot.capture_spec_rings(&self.ctx, kv_adapter, start_pos, depth)
    }

    /// Restore the speculative boundary ring slot after truncate and before
    /// accepted-prefix fold. No-op when spec decode is off.
    pub(crate) fn restore_spec_ring_tail(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        accepted_n: usize,
        depth: usize,
    ) -> Result<()> {
        slot.restore_spec_ring_tail(&self.ctx, kv_adapter, start_pos, accepted_n, depth)
    }

    /// Commit the accepted prefix (`accepted_rows` = verify row indices in
    /// schedule order, root first) from the persisted verify rows — per layer:
    /// compressor/indexer re-ingestion + ring K re-derivation — then advance the
    /// slot length. Caller order: truncate → rejected-tail restore → THIS.
    pub(crate) fn commit_accepted_fold<I>(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        accepted_rows: I,
        start_pos: usize,
    ) -> Result<()>
    where
        I: Clone + ExactSizeIterator<Item = usize>,
    {
        let m = accepted_rows.len();
        ensure!(m > 0, "DSv4 commit fold needs at least the pending row");
        let hidden_size = self.config.hidden_size;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        // Gather scratch reused across layers.
        // SAFETY: uninit device scratch; fully written before first read.
        let mut gathered = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, m)? };
        keepalive.keep_hidden(&gathered);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            {
                let cache = slot
                    .spec_normed
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 commit fold without persisted verify rows"))?;
                for (i, row) in accepted_rows.clone().enumerate() {
                    let src = cache[layer_idx]
                        .data
                        .slice(row * hidden_size..(row + 1) * hidden_size);
                    let mut dst = gathered
                        .data
                        .slice_mut(i * hidden_size..(i + 1) * hidden_size);
                    self.ctx
                        .stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSv4 commit fold gather failed: {e}"))?;
                }
            }
            let (layer_pool, flashmla_scratch, fp32_scratch) =
                kv_adapter.layer_and_flashmla_scratch_mut(layer_idx)?;
            crate::attention::commit_layer_fold(
                &self.ctx,
                &self.config,
                &layer.attention,
                layer.mode,
                layer.compress_ratio,
                &mut slot.attention[layer_idx],
                flashmla_scratch,
                fp32_scratch,
                layer_pool,
                &gathered,
                start_pos,
                &mut keepalive,
            )?;
        }
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        // H2: advance the FlashMLA pool cursor by the committed token count so
        // `pool.seq_len(slot) == start_pos + m` matches `slot.seq_len` — the next
        // decode tick's prepare_kv_batch validates `pool.seq_len == append_pos`.
        // commit_layer_fold packs the accepted ring positions but does NOT draw
        // pool tokens; the prior truncate rolled the cursor back to start_pos.
        for (layer_idx, state) in slot.attention.iter().enumerate() {
            if let Some(slot_idx) = state.flashmla_slot_idx() {
                kv_adapter
                    .layer_mut(layer_idx)?
                    .flashmla_alloc_append(slot_idx, m)?;
            }
        }
        slot.seq_len = start_pos + m;
        Ok(())
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
        last_hidden_out: Option<&mut DeviceVec>,
    ) -> Result<u32> {
        let seq_len = tokens.len();
        let moe_transport = crate::runtime_flags::dsv4_moe_transport()?;
        // FlashMLA-decode captures cleanly (capture-safety fixes e95e11b6).
        // The graph routes on-device into fixed scratch and runs the masked
        // MoE tail — no gpu_router/pooled dependency. GLM rides the same FP8 MoE
        // path (routed + shared) with no per-call scratch alloc, so it captures
        // like DSv4 (the bf16-shared per-call-alloc hazard that gated GLM off is
        // gone). ponytail: pod-verify GLM V32 SparseIndexed decode captures+replays
        // cleanly under the decode-graph (no host launches / per-call alloc in the
        // indexer/sparse-attention forward) before relying on it for GLM serving.
        // Lens needs the eager layer loop — a graph replay never executes it
        // (the entropy probe is unaffected: sampling runs outside the graph).
        // DSpark needs the eager per-layer loop to capture the 3 target-layer
        // taps (a graph replay skips the loop), same reason lens / hidden-capture
        // fall through here. Byte-identical when `is_dspark()` is false.
        if dsv4_decode_graph_enabled()
            && last_hidden_out.is_none()
            && !self.config.is_dspark()
            && !self.config.is_glm()
            && seq_len == 1
            && matches!(
                moe_transport,
                crate::runtime_flags::Dsv4MoeTransport::AllReduce
            )
            && crate::probe::lens_layers() == 0
        {
            return self.forward_tokens_decode_graph(
                slot, kv_adapter, tokens[0], start_pos, params, position,
            );
        }

        let (stream, mut keepalive) =
            self.forward_tokens_stream_impl(slot, kv_adapter, tokens, start_pos, false, None)?;
        if seq_len > 1 && crate::probe::token_entropy() {
            self.probe_prefill_entropy(&stream, tokens, start_pos)?;
        }
        if let Some(out) = last_hidden_out {
            self.capture_mtp_stream_hidden(&stream, seq_len - 1, 1, out, &mut keepalive)?;
        }
        let token = self.forward_stream_last_token(
            &stream,
            seq_len,
            params,
            position,
            None,
            &mut keepalive,
        )?;
        // Post-sampling sync point: the emitted token is known, stashed lens
        // logits can D2H without adding a mid-loop sync. seq_len==1 mirrors the
        // arming condition — a prefill call must not drain another step's stash.
        if seq_len == 1 {
            crate::probe::lens_flush(&self.ctx, &[position], &[token]);
        }
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(token)
    }

    /// Commit/selftest forward for a contiguous token prefix. This is not the
    /// frozen MTP verifier: it writes the slot KV state like a normal forward and
    /// returns the per-row logits/hidden view used by verify-forward selftests.
    pub(crate) fn forward_tokens_verify(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        _position: u64,
    ) -> Result<SpecVerifyResult> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 verify forward requires at least one token"
        );
        let n = tokens.len();
        let stream_dim = self.config.hidden_size * self.config.hc_mult;
        let persist_spec_normed = slot.spec_normed.is_some();
        let (stream, mut keepalive) = self.forward_tokens_stream_impl(
            slot,
            kv_adapter,
            tokens,
            start_pos,
            persist_spec_normed,
            None,
        )?;
        let hiddens = (0..n)
            .map(|i| -> Result<_> {
                let mut h = DeviceVec::zeros(&self.ctx, stream_dim)?;
                self.capture_mtp_stream_hidden(&stream, i, 1, &mut h, &mut keepalive)?;
                Ok(h)
            })
            .collect::<Result<Vec<_>>>()?;
        let logits = self.verify_logits_from_stream(&stream, n, &mut keepalive)?;
        let argmax = self.mtp_argmax_batch(&logits)?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(SpecVerifyResult {
            logits,
            argmax,
            hiddens,
        })
    }

    /// Head recipe for stream rows `rows`: per-row HC fold (GLM `hc_mult == 1`
    /// identity) + final RMSNorm, written to `out` rows `0..rows.len()`.
    fn head_normed_rows(
        &self,
        stream: &HiddenStates,
        rows: std::ops::Range<usize>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        ensure!(
            stream.seq_len >= rows.end
                && out.hidden_dim == hidden_size
                && out.seq_len >= rows.len(),
            "DSv4 head rows {rows:?} out of stream rows {} / out {}x{}",
            stream.seq_len,
            out.hidden_dim,
            out.seq_len
        );
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        for (i, row) in rows.enumerate() {
            if self.config.is_glm() {
                // GLM: head hidden = stream row (stream_dim==hidden, no head HC mixer).
                // ponytail: pod-verify GLM hc_mult==1 head hidden = stream row (identity)
                crate::ops::copy_row_to_vec(&self.ctx, stream, row, &mut last_hidden)?;
            } else {
                crate::hc::head_hidden_from_stream(
                    &self.ctx,
                    &self.config,
                    &self.head_hc,
                    stream,
                    row,
                    &mut last_hidden,
                )?;
            }
            crate::ops::rms_norm_vec(&self.ctx, &last_hidden, &self.norm, eps, &mut last_normed)?;
            let mut dst = out.data.slice_mut(i * hidden_size..(i + 1) * hidden_size);
            self.ctx
                .stream
                .memcpy_dtod(&last_normed.data, &mut dst)
                .map_err(|e| anyhow!("DSv4 verify head row copy failed: {e}"))?;
        }
        Ok(())
    }

    /// Fold every row's stream into a full target logits matrix.
    fn verify_logits_from_stream(
        &self,
        stream: &HiddenStates,
        rows: usize,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<HiddenStates> {
        ensure!(rows > 0, "DSv4 verify logits requires at least one row");
        ensure!(
            stream.seq_len >= rows,
            "DSv4 verify stream rows {} < requested logits rows {rows}",
            stream.seq_len
        );
        let hidden_size = self.config.hidden_size;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut head_normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, rows)? };
        self.head_normed_rows(stream, 0..rows, &mut head_normed)?;
        keepalive.keep_hidden(&head_normed);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut logits = unsafe { HiddenStates::uninit(&self.ctx, self.lm_head.rows, rows)? };
        self.lm_head_project_batch(&head_normed, &mut logits)?;
        Ok(logits)
    }

    /// Prefill per-position entropy/NLL: head recipe + lm_head over 256-row
    /// chunks of the full prefill stream, one D2H per chunk (prefill already
    /// syncs at its sample tail; the probe is ON here, so the extra sync is
    /// probe-only cost). Buffers drop after the chunk's D2H sync — safe.
    fn probe_prefill_entropy(
        &self,
        stream: &HiddenStates,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<()> {
        const CHUNK: usize = 256;
        let hidden_size = self.config.hidden_size;
        let vocab = self.lm_head.rows;
        for chunk_start in (0..tokens.len()).step_by(CHUNK) {
            let chunk_end = (chunk_start + CHUNK).min(tokens.len());
            let rows = chunk_end - chunk_start;
            // SAFETY: uninit device scratch; fully written before first read.
            let mut head_normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, rows)? };
            self.head_normed_rows(stream, chunk_start..chunk_end, &mut head_normed)?;
            // SAFETY: uninit device scratch; fully written before first read.
            let mut logits = unsafe { HiddenStates::uninit(&self.ctx, vocab, rows)? };
            self.lm_head_project_batch(&head_normed, &mut logits)?;
            let host = crate::probe::stream_to_host_f32(&self.ctx, &logits)?;
            for i in 0..rows {
                let global = chunk_start + i;
                // Target = the actual next prompt token; None on the call's last
                // row (the target is in the next prefill chunk / the sampled token).
                let target = tokens.get(global + 1).copied();
                let (entropy, nll) =
                    crate::probe::entropy_nll(&host[i * vocab..(i + 1) * vocab], target);
                crate::probe::emit_token(
                    "prefill",
                    (start_pos + global) as u64,
                    target,
                    nll,
                    entropy,
                );
            }
        }
        Ok(())
    }

    /// Verify `tokens` in ONE scheduled sparse forward under `sched`'s per-row
    /// positions. MTP schedules are chain-shaped (`depth + 1` rows), so this path
    /// requires at least two rows and never falls back to a plain single-token
    /// forward. Ordinary non-spec decode uses `forward_tokens_verify` /
    /// `forward_tokens` instead.
    pub(crate) fn forward_tokens_verify_scheduled(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        _position: u64,
        sched: &SpecVerifySchedule,
    ) -> Result<SpecVerifyResult> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 verify forward requires at least one token"
        );
        ensure!(
            tokens.len() >= 2,
            "DSv4 scheduled sparse verify requires at least 2 rows; use ordinary decode for single-token verify"
        );
        ensure!(
            sched.positions.len() == tokens.len() && sched.ancestors.len() == tokens.len(),
            "DSv4 verify schedule rows ({}, {}) != tokens {}",
            sched.positions.len(),
            sched.ancestors.len(),
            tokens.len()
        );
        let _nvtx = crate::nvtx::range("dsv4/lm_head_verify");

        // ONE scheduled sparse verify forward. `hiddens[j]` = row j's MTP
        // stream; `argmax[j]` = the target's argmax AFTER `tokens[j]` (so
        // `argmax[i]` is exactly what an accepted child of node i must equal).
        // Attention uses the frozen sparse verify lane and does not commit slot
        // KV state.
        let stream_dim = self.config.hidden_size * self.config.hc_mult;
        let was_frozen = crate::attention::dsv4_verify_frozen();
        if !was_frozen {
            crate::attention::set_dsv4_verify_frozen(true);
        }
        let result = (|| -> Result<SpecVerifyResult> {
            let (stream, mut keepalive) = self.forward_tokens_stream_impl(
                slot,
                kv_adapter,
                tokens,
                start_pos,
                true,
                Some(sched),
            )?;
            let n = tokens.len();
            let mut hiddens = Vec::with_capacity(n);
            for i in 0..n {
                let mut h = DeviceVec::zeros(&self.ctx, stream_dim)?;
                self.capture_mtp_stream_hidden(&stream, i, 1, &mut h, &mut keepalive)?;
                hiddens.push(h);
            }
            let logits = self.verify_logits_from_stream(&stream, n, &mut keepalive)?;
            let argmax = self.mtp_argmax_batch(&logits)?;
            std::hint::black_box(keepalive.len());
            drop(keepalive);
            Ok(SpecVerifyResult {
                logits,
                argmax,
                hiddens,
            })
        })();
        if !was_frozen {
            crate::attention::set_dsv4_verify_frozen(false);
        }
        result
    }

    /// Layer-major batched decode over N independent slots (one decode token each).
    ///
    /// Each row `r` decodes slot `slot_ids[r]` at `start_positions[r]`. The
    /// point-wise pipeline (embed / HC wrap / rms_norm / MoE / shared expert /
    /// all-reduce) runs over the whole `seq_len = N` batch exactly as the prefill
    /// path does — those ops are token-independent, so stacking N rows is
    /// math-identical to N separate single-row forwards. For MODEL1 B>1, the
    /// attention lane is the canonical batched FlashMLA sparse decode path; B=1
    /// stays on the single-row reference path. Returns one sampled token per row.
    pub(crate) fn forward_decode_batch(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        tokens: &[u32],
        start_positions: &[usize],
        positions: &[u64],
        params: &[SamplingParams],
        // false on the spec-off decode lane: nothing reads these taps.
        capture_taps: bool,
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
            capture_taps,
        )?;
        let _nvtx = crate::nvtx::range("dsv4/lm_head_sample_batched");
        let fast_head = self.lm_head_shard.is_none()
            && params.iter().all(SamplingParams::is_greedy)
            && !crate::probe::token_entropy()
            && crate::probe::lens_layers() == 0
            && std::env::var_os("INFER_DSV4_DUMP_TOPK_POSITIONS").is_none();
        let out_tokens = if fast_head {
            let logits = self.verify_logits_from_stream(&stream, n, &mut keepalive)?;
            self.mtp_argmax_batch(&logits)?
        } else {
            // `forward_stream_last_token` folds row `seq_len - 1`.
            (0..n)
                .map(|r| {
                    self.forward_stream_last_token(
                        &stream,
                        r + 1,
                        &params[r],
                        positions[r],
                        None,
                        &mut keepalive,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };
        // Post-sampling sync point: all rows' emitted tokens are known.
        crate::probe::lens_flush(&self.ctx, positions, &out_tokens);
        dsv4_decode_trace(
            self.tp.config().rank,
            slot_ids,
            start_positions,
            &out_tokens,
        );
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
        capture_taps: bool,
    ) -> Result<(HiddenStates, Dsv4ForwardKeepalive)> {
        let n = slot_ids.len();
        let mega_epoch = self.begin_mega_moe_forward(n)?;
        // Opt-in decode-phase timing probe (env-gated). When unset: zero behavior
        // change, zero extra syncs. Pure instrumentation — see ARLE_DSV4_DECODE_PHASE_TIME.
        let phase_time = matches!(
            std::env::var("ARLE_DSV4_DECODE_PHASE_TIME").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        );
        // Per-GEMV breakdown for this decode step (self-gates on
        // ARLE_DSV4_LINEAR_PROFILE; no-op otherwise). Reset here + print after the
        // phase log scopes the stats to ONE decode forward, exposing which linear
        // (compressor_wkv/wgate, indexer_wq_b/weights, …) dominates compidx. The
        // batched-compressor pre-pass is default-on (P1a/P1b + prepass, since
        // 2026-06-26); this profiler measures its residual per-GEMV cost —
        // pending-remote on TP=4. NOTE: when enabled the profiler syncs per call,
        // inflating absolute step ms; read it for the RELATIVE per-GEMV split, not
        // the clean compidx (use phase_time alone for that).
        crate::linear_profile::reset();
        for r in 0..n {
            let slot = &slots[slot_ids[r]];
            ensure!(
                slot.seq_len == start_positions[r],
                "DSv4 batched decode slot {} seq_len {} != start_pos {}; decode requires contiguous appends",
                slot_ids[r],
                slot.seq_len,
                start_positions[r]
            );
            let next_len = start_positions[r]
                .checked_add(1)
                .ok_or_else(|| anyhow!("DSv4 batched decode start_pos overflow"))?;
            ensure!(
                start_positions[r] < slot.max_seq_len,
                "DSv4 batched decode slot {} sequence {} exceeds max_seq_len {}",
                slot_ids[r],
                next_len,
                slot.max_seq_len
            );
        }

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = n; // batch dimension: N independent decode rows
        let eps = self.config.rms_norm_eps;
        // capture only when the caller (spec anchor) consumes taps, not on
        // is_dspark() alone — plain decode never reads them.
        let dspark = capture_taps && self.config.is_dspark();
        let use_deepep_transport = crate::runtime_flags::dsv4_moe_transport()?.is_deepep();
        // N>1: mirror the prefill keepalive discipline (the per-token decode
        // scratch / comm-overlap fast paths are seq_len==1 only).
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let ctx = &self.ctx;

        // Batched (b=N) FlashMLA decode lane (#60, the #1 concurrency lever).
        // B>1 uses this as the canonical path when the model-wide batched
        // scratch exists (both MODEL1 and V32/GLM). N=1 never reaches this
        // function.
        let batched_attn_lane = crate::attention::dsv4_flashmla_decode_batched_enabled()?
            && kv_adapter.has_flashmla_batch_scratch();

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
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_out_row = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
        keepalive.keep_hidden(&normed_row);
        keepalive.keep_hidden(&attn_out_row);
        // MoE/shared are now grouped over [N] (Phase 6a) — no per-row MoE scratch.

        // PHASE C (#60): contiguous [N] absolute-position array for the batched
        // projection pre-pass RoPE (`positions[token]` = each row's
        // start_positions[r]). The per-slot scalar `slot.start_pos_device`
        // buffers above stay for the per-row compressor/pack/finish; this is the
        // packed form the batched RoPE kernel reads. Uploaded once, reused every
        // layer (positions are fixed across the layer loop). Only allocated when
        // the batched lane is engaged.
        let batched_positions = if batched_attn_lane {
            let positions_host: Vec<i32> = (0..n)
                .map(|r| {
                    i32::try_from(start_positions[r])
                        .map_err(|_| anyhow!("DSv4 start_pos {} overflows i32", start_positions[r]))
                })
                .collect::<Result<Vec<_>>>()?;
            let buf = crate::ops::upload_i32(&self.ctx, &positions_host)?;
            keepalive.keep_i32(&buf);
            Some(buf)
        } else {
            None
        };
        // Batched CSA select-metadata device build (default OFF). Read once; when
        // ON, `csa_select_official_batched` computes block_table/context_lens/
        // positions on-device instead of host-build + per-step H2D.
        let use_device_meta = crate::attention::dsv4_dsa_device_meta_enabled()?;
        // Per-row [*, 1] scratch for the batched projection slice copy-in (each
        // row's c_q_normed / q_prepared / k_prepared sliced out of the batched
        // pre-pass output before the per-slot compressor/pack consume it).
        // Declared once, reused across rows/layers (WAR resolved by stream
        // order, like normed_row). Only allocated when the batched lane runs.
        let local_width = self.layers[0].attention.wq_b.rows;
        let q_lora_rank = self.config.q_lora_rank;
        let mla_head_dim = self.config.head_dim;
        // Reused per-row [q_lora_rank, 1] scratch for this row's batched
        // c_q_normed slice (read by reference for the CSA indexer query proj).
        // q_prepared / k_prepared rows are copied into fresh OWNED buffers inside
        // the loop (taken by value into Dsv4MlaPrepared), so only c_q needs the
        // reused scratch. Only allocated when the batched lane runs.
        let mut c_q_normed_row = if batched_attn_lane {
            // SAFETY: fully written by the per-row slice copy-in before read.
            let cq = unsafe { HiddenStates::uninit(&self.ctx, q_lora_rank, 1)? };
            keepalive.keep_hidden(&cq);
            cq
        } else {
            // Unused (per-row lane); zero-sized stand-in keeps the binding typed.
            // SAFETY: zero-size placeholder; never read.
            unsafe { HiddenStates::uninit(&self.ctx, 0, 1)? }
        };

        // FINISH F1 batched destination: ONE contiguous token-major
        // `[n, local_width]` buffer for `slice_out_batched` (replaces N per-row
        // `slice_out_row` calls). Reused across layers (each layer overwrites all
        // n rows before the finish reads them; WAR resolved by stream order).
        // local_width is uniform across DSv4 layers (all wq_b.rows equal), so
        // layer 0's value sizes it for every layer. Only allocated on the batched
        // lane. Held in keepalive (premature-free hazard under disabled events).
        let mut local_attn_batched = if batched_attn_lane {
            // SAFETY: every [r*local_width, (r+1)*local_width) span is written by
            // slice_out_batched before the finish kernels / O-LoRA read it.
            let lab = unsafe { HiddenStates::uninit(&self.ctx, local_width, n)? };
            keepalive.keep_hidden(&lab);
            lab
        } else {
            // SAFETY: zero-size placeholder; never read.
            unsafe { HiddenStates::uninit(&self.ctx, 0, 1)? }
        };
        // Full-flatten is canonical for MODEL1 B>1 decode. Per layer it batches
        // compressor state update (where a compressor exists), inverse-RoPE, and
        // SW-window write into one launch over N rows. SparseIndexed has no main
        // compressor and is explicitly excluded below.
        // Holds the host-uploaded per-row device-pointer ARRAYS for the batched
        // compressor/finish kernels alive to function return (the N>1 keepalive is
        // inert; pointer arrays must outlive their kernel launch under the
        // disabled-event-tracking premature-free hazard). Only pushed when
        // `full_flatten` is on.
        let mut ptr_keepalive: Vec<CudaSlice<u64>> = Vec::new();
        // Op "c" batched pack: the per-slot device page tables feeding the batched
        // pack's pointer array. Held to function return (same premature-free guard
        // as `ptr_keepalive`; the inert N>1 `keep_i32` would not retain them).
        let mut pack_pt_keepalive: Vec<CudaSlice<i32>> = Vec::new();

        // Decode-phase timing accumulators (only ever touched when phase_time).
        let (mut csa_ms, mut sw_ms, mut moe_ms) = (0f64, 0f64, 0f64);
        // Batched-lane (sw_ms) sub-block split: prep loop / batched fwd / finish loop.
        // These sum ≈ sw_ms. Only touched when phase_time.
        let (mut prep_ms, mut fwd_ms, mut finish_ms) = (0f64, 0f64, 0f64);
        // PREPARE sub-split: batched projection pre-pass vs per-row compressor/indexer.
        // These sum ≈ prep_ms. Only touched when phase_time.
        let (mut proj_ms, mut compidx_ms) = (0f64, 0f64);
        // compidx sub-split (coarse): the whole per-row prepare loop (compressor +
        // Hadamard rotate + cache writes + indexer GEMM + per-row gathers — all
        // fused inside the single `mla_attention_prepare_compressed_only` call, so
        // writes vs gemm-gather are NOT separable without touching the callee) vs
        // the ONE batched `csa_select_official_batched` READ (logits + topk). These
        // two sum ≈ compidx_ms + the batched read. Only touched
        // when phase_time.
        let (mut perrow_ms, mut batchedread_ms) = (0f64, 0f64);

        // Logit lens (batched decode is always a decode step): capture the last
        // N layers' head projections; off = one Option compare per layer.
        let lens_from = match crate::probe::lens_layers() {
            0 => None,
            depth => Some(self.layers.len().saturating_sub(depth)),
        };
        if lens_from.is_some() {
            crate::probe::lens_begin();
        }
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
            let full_flatten = layer.mode != DeepSeekV4AttentionMode::SparseIndexed;

            // Diagnostic-only substage fingerprint (`ARLE_PROBE_STAGES`, see
            // docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md).
            // D2H's one row's slice per call; `stage_want` is a cheap OnceLock
            // compare when off, so this closure costs nothing on the hot path.
            let stage_all = |name: &str, buf: &HiddenStates| {
                let width = buf.hidden_dim;
                for (r, &pos) in start_positions.iter().enumerate().take(buf.seq_len) {
                    let pos = pos as u64;
                    if crate::probe::stage_want(pos) {
                        crate::probe::stage_bf16(
                            ctx,
                            name,
                            layer_idx,
                            r,
                            pos,
                            buf.data.slice(r * width..(r + 1) * width),
                        );
                    }
                }
            };

            // ── Attention half: HC params + pre + norm over the whole [N] batch.
            // GLM (hc_mult==1): no hyper-connection; the stream IS the hidden, so
            // the pre-mixer is identity. Replace the mhc pre+norm with a plain
            // RMSNorm of the stream (the C identity placeholders are 1x1 ZERO
            // mixers — running them through gen_mhc/mhc_pre would zero the stream).
            // ponytail: pod-verify GLM hc_mult==1 plain residual + identity stream (no hyper-connection)
            // SAFETY: uninit device scratch; fully written before first read.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            let attn_mhc = if self.config.is_glm() {
                crate::ops::rms_norm_batch(&self.ctx, &stream, &layer.attn_norm, eps, &mut normed)?;
                None
            } else {
                let mhc =
                    crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_attn, &stream)?;
                keepalive.keep_f32(&mhc.pre);
                keepalive.keep_f32(&mhc.post);
                keepalive.keep_f32(&mhc.comb);
                // SAFETY: fused hc_pre+rms_norm writes the full [seq_len, hidden_size] buffer.
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
                Some(mhc)
            };
            keepalive.keep_hidden(&normed);
            stage_all("attn_norm", &normed);

            // ── Attention: per-row independent single-token MLA into row r.
            // SAFETY: every [r*hidden, (r+1)*hidden) span of attn_out is written by
            // the copy-out below before attn_out is read by hc_post.
            let mut attn_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            // PHASE B (#60): the batched FlashMLA decode lane. For a batched-
            // eligible layer (FlashMLA decode on, ALL modes) the per-row FlashMLA
            // KERNEL launch is replaced by ONE `sparse_decode_fwd(b=N)`:
            //   1. per row: PREPARE (wq/wkv+RoPE, compressor, CSA indexer top-k
            //      `selected`) → pack KV into the shared pool → gather this row's
            //      global-head Q into q_batched[r]; for CSA also gather this row's
            //      `selected` into selected_batched[r];
            //   2. build_layer_batch_meta(b=N) (indices/sched_meta; CSA feeds the
            //      gathered `selected` ptr) → ONE batched fwd;
            //   3. per row: slice the global-head output → local_attn → inverse-rope
            //      + SW-window update (the cheap per-row tail) → O-LoRA into attn_out[r].
            // The prepare/pack/finish/oproj loops stay per-row (cheap); only the
            // attention kernel is batched (the 74k tiny gridX=1 launches).
            // Non-batched (FlashMLA
            // decode off) layers fall through to the per-row `mla_attention` loop,
            // byte-unchanged. SW/HCA index builds are byte-identical (selected_ptr
            // stays 0); only CSA passes the gathered per-row top-k.
            let use_batched_kernel = batched_attn_lane;
            if use_batched_kernel {
                let _sw_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // CSA (cr 1..=15) and HCA (cr≥16) both pack + attend a separate
                // compressed pool. GLM SparseIndexed has NO main compressor — it
                // attends the full latent via the FlashMLA KV pool (no separate
                // compressed pool), so it must NOT request the compressed pack
                // (flashmla_pack_borrow(true) fetches state.compressor == None and
                // would error). Gate on has_compressor() so SparseIndexed packs only
                // SW + current. Byte-identical for SW/HCA/CSA.
                let want_compressed = layer.mode.has_compressor();
                // CSA + GLM SparseIndexed both run the indexer top-k select. The
                // batched-decode DSA select lane widens to has_indexer() so GLM
                // shares it; CSA-only compressor-batch / full-flatten paths below
                // stay gated on CompressedSparse (SparseIndexed has no compressor).
                // ponytail: pod-verify SparseIndexed batched-decode DSA select lane end-to-end
                let runs_indexer = layer.mode.has_indexer();
                // BATCHED CSA-SELECT (#60): when this CSA layer runs on the
                // official DSA path AND the shared scratch is present, defer the
                // per-row READ (paged-MQA logits + topk) into ONE batch_size=N
                // call after the prepare loop. The per-row CACHE WRITES still run
                // inside the prepare loop. When off (non-official DSA, or scratch
                // absent), the existing per-row `csa_select_official` +
                // `gather_selected_row` path runs byte-identically.
                let use_batched_dsa_select = runs_indexer
                    && crate::attention::dsv4_dsa_official_enabled()?
                    && kv_adapter
                        .layer_dsa_and_flashmla_batch_mut(layer_idx)?
                        .1
                        .is_some();
                let _nvtx = crate::nvtx::range("dsv4/mla_attn_batched");
                // N-row staging for the batched CSA select. CompressedSparse uses the
                // batched indexer-query prepass output directly; lanes without that
                // prepass (SparseIndexed) stage per-row q_i/weights here.
                let mut dsa_stage_q_i: Option<HiddenStates> = None;
                let mut dsa_stage_weights: Option<HiddenStates> = None;
                let mut dsa_key_counts =
                    use_batched_dsa_select.then(|| Vec::<i32>::with_capacity(n));
                // ── 1. per-row PREPARE + pack + Q-gather.
                let mut prepared: Vec<crate::attention::Dsv4MlaPrepared> = Vec::with_capacity(n);
                let mut slot_block_offsets = Vec::with_capacity(n);
                let mut page_tables = Vec::with_capacity(n);
                let _prep_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // ── 0. Batched (m=N) slot-INDEPENDENT projection pre-pass
                // (#60 PHASE C): wq_a / wkv / wq_b + RoPE over all N rows at once
                // (weights read ONCE across the N-token grid), replacing the N
                // per-row m=1 projection GEMVs that dominated PREPARE. The per-row
                // loop below then reads each row's SLICE of these batched outputs
                // for the per-slot compressor / pack / gather — no per-row
                // projection recompute. Touches no slot state.
                let positions = batched_positions.as_ref().ok_or_else(|| {
                    anyhow!("DSv4 batched decode lane: batched positions buffer missing")
                })?;
                // Borrow the model-wide shared FP8 prefill DeepGEMM scratch for the
                // batched (m=N) projection. `None` when DeepGEMM is disabled — the
                // pre-pass then takes its scalar fallback. This borrow is scoped to
                // the pre-pass and released before the per-row loop re-borrows the
                // pool / batch scratch via the same accessor.
                let _proj_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let proj = {
                    let (_layer_pool, _dsa, _flash_batch, _flashmla_scratch, prefill_shared) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    crate::attention::mla_attention_prepare_proj_batch(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer.compress_ratio,
                        &normed,
                        positions,
                        &self.tp,
                        prefill_shared,
                        &mut keepalive,
                    )?
                };
                if let Some(t) = _proj_t {
                    ctx.stream.synchronize().ok();
                    proj_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
                // ── 0b. Batched (m=N) compressor/indexer projection pre-pass. The
                // per-row m=1 `compressor_forward` wkv/wgate GEMVs re-read the full
                // weight per decode row; batch them over the same `normed` batch the
                // proj pre-pass used. Each row then reads its `[width,1]` column slice
                // below unless full-flatten consumes the batched outputs directly.
                // Touches NO slot state.
                // Batched (m=N) compressor/indexer key + query projections, routing
                // FP8 weights through tensor-core DeepGEMM via the shared prefill
                // scratch (same accessor as the proj pre-pass; one borrow for all).
                let (compressor_kv_score, indexer_kv_score, indexer_query_kv_score) = {
                    let (_lp, _dsa, _fb, _fs, mut prefill_shared) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    // Decode multi-stream overlap (Aux-stream fork of this prepass)
                    // was KILLed 2026-07-06: pod A/B at n=8 (above its own
                    // activation threshold) showed a ~9% wall-clock REGRESSION
                    // (9.65s -> 10.53s mean, zero overlap across 6 trials) — see
                    // docs/experience/wins/2026-07-06-dsv4-decode-multistream-overlap-pending-remote.md.
                    let main = match layer.attention.compressor.as_ref() {
                        Some(c) => crate::attention::compressor_batch_prepass(
                            &self.ctx,
                            c,
                            &normed,
                            prefill_shared.as_deref_mut(),
                            &mut keepalive,
                        )?,
                        None => None,
                    };
                    let indexer = match (layer.mode, layer.attention.indexer.as_ref()) {
                        (DeepSeekV4AttentionMode::CompressedSparse, Some(idx)) => {
                            crate::attention::compressor_batch_prepass(
                                &self.ctx,
                                idx.compressor
                                    .as_ref()
                                    .expect("DSv4 CSA indexer has a key compressor"),
                                &normed,
                                prefill_shared.as_deref_mut(),
                                &mut keepalive,
                            )?
                        }
                        _ => None,
                    };
                    let query = match (layer.mode, layer.attention.indexer.as_ref()) {
                        (DeepSeekV4AttentionMode::CompressedSparse, Some(idx)) => {
                            Some(crate::attention::indexer_query_batch_prepass(
                                &self.ctx,
                                idx,
                                &proj.c_q_normed,
                                &normed,
                                prefill_shared,
                                &mut keepalive,
                            )?)
                        }
                        _ => None,
                    };
                    (main, indexer, query)
                };
                if let Some((kv, _score)) = compressor_kv_score.as_ref() {
                    stage_all("compressor_out", kv);
                }
                // ── 0c. Full-flatten P1a: per-row compressor
                // STATE-update DEFER (gather each row's ring-state pointers + advance
                // compressed.seq_len, NO per-row FFI) → ONE
                // `dsv4_compressor_update_batched` over all N rows (main + indexer),
                // reading the batched prepass outputs directly. This replaces the N
                // per-row `dsv4_compressor_update_*` launches (the remaining ∝n
                // `perrow`). The batched update runs BEFORE the P1b loop's per-row
                // consumers (pack reads main `compressed`; csa cache-write reads
                // indexer `compressed`), so they see the written keys. `indexer_rows_before`
                // is captured per row here (P1a advances seq_len) and handed to P1b's
                // CSA path. SparseIndexed skips this path because it has no compressor.
                let indexer_rows_before: Vec<usize> = if full_flatten {
                    let mut main_sink = crate::attention::Dsv4CompressorBatchPtrs::with_capacity(n);
                    let mut indexer_sink =
                        crate::attention::Dsv4CompressorBatchPtrs::with_capacity(n);
                    let mut before = Vec::with_capacity(n);
                    let original_seq_len = proj.original_seq_len;
                    for r in 0..n {
                        // Defer mode reads NO `normed_row` data (the projection was
                        // already batched in the prepass; defer only gathers state
                        // pointers + advances seq_len), so the per-row copy-in is
                        // skipped. `normed_row` is a `[hidden,1]` handle with
                        // seq_len==1 — exactly the shape `compressor_forward`'s asserts
                        // require for the token_count==1 decode path.
                        let slot = &mut slots[slot_ids[r]];
                        let b = crate::attention::mla_attention_compressor_defer_row(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            &normed_row,
                            &mut slot.attention[layer_idx],
                            start_positions[r],
                            Some(&slot.start_pos_device),
                            original_seq_len,
                            &mut main_sink,
                            &mut indexer_sink,
                            &mut keepalive,
                        )?;
                        before.push(b);
                    }
                    // ONE batched update each for the main + indexer compressor (only
                    // when this layer's rows actually have that compressor; the sinks
                    // are all-or-nothing per layer since `layer.mode` is uniform).
                    // `ptrs.prev_overlap_*` are per-row per-slot registers (#154).
                    let overlap = layer.compress_ratio < 16;
                    if let (Some((kv, score)), Some(compressor)) = (
                        compressor_kv_score.as_ref(),
                        layer.attention.compressor.as_ref(),
                    ) {
                        let positions = batched_positions.as_ref().ok_or_else(|| {
                            anyhow!("DSv4 full-flatten P1a: batched positions missing")
                        })?;
                        crate::attention::dsv4_compressor_update_batched(
                            &self.ctx,
                            &self.config,
                            compressor,
                            kv,
                            score,
                            &main_sink,
                            positions,
                            n,
                            self.config.head_dim,
                            layer.compress_ratio,
                            overlap,
                            true,
                            proj.original_seq_len,
                            &mut ptr_keepalive,
                        )?;
                    }
                    if layer.mode == DeepSeekV4AttentionMode::CompressedSparse
                        && let (Some((kv, score)), Some(indexer)) =
                            (indexer_kv_score.as_ref(), layer.attention.indexer.as_ref())
                    {
                        let use_official_dsa = crate::attention::dsv4_dsa_official_enabled()?;
                        let indexer_rope_original_seq_len = if use_official_dsa {
                            i32::try_from(
                                self.config.rope_parameters.original_max_position_embeddings,
                            )
                            .map_err(|_| {
                                anyhow!("DSv4 full-flatten indexer rope len overflows i32")
                            })?
                        } else {
                            0
                        };
                        let positions = batched_positions.as_ref().ok_or_else(|| {
                            anyhow!("DSv4 full-flatten P1a: batched positions missing")
                        })?;
                        crate::attention::dsv4_compressor_update_batched(
                            &self.ctx,
                            &self.config,
                            indexer
                                .compressor
                                .as_ref()
                                .expect("DSv4 CSA indexer has a key compressor"),
                            kv,
                            score,
                            &indexer_sink,
                            positions,
                            n,
                            self.config.index_head_dim,
                            layer.compress_ratio,
                            true, // indexer compressor always overlap
                            use_official_dsa,
                            indexer_rope_original_seq_len,
                            &mut ptr_keepalive,
                        )?;
                    }
                    before
                } else {
                    Vec::new()
                };
                // ── 0d. Full-flatten P1b: per-row DSA CACHE WRITE (block (a) of
                // `csa_select_official`: Hadamard rotate of the newly-packed index
                // keys + FP8 fused-store into each slot's cache band + `packed_rows`
                // advance) hoisted out of the prepare loop into ONE batched pre-pass
                // over all N slots. Runs AFTER P1a (the batched compressor-update
                // wrote `indexer.compressed`, the staging ring this reads) and
                // BEFORE the prepare loop (whose per-row `csa_select` then skips the
                // cache write via `cache_writes_in_prepass`). Gated to the
                // CompressedSparse full-flatten lane with the official DSA path
                // engaged; SparseIndexed keeps its per-row cache write in the loop.
                let cache_writes_in_prepass = use_batched_dsa_select
                    && full_flatten
                    && layer.mode == DeepSeekV4AttentionMode::CompressedSparse;
                if cache_writes_in_prepass {
                    let mut cache_ptrs =
                        crate::attention::Dsv4DsaCacheWriteBatchPtrs::with_capacity(n);
                    for r in 0..n {
                        let (layer_pool, dsa_shared, _fb, _fs, _pf) =
                            kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                        let dsa_shared = dsa_shared.ok_or_else(|| {
                            anyhow!("DSv4 full-flatten P1b: shared DSA scratch missing")
                        })?;
                        let slot = &mut slots[slot_ids[r]];
                        let state = &mut slot.attention[layer_idx];
                        // P1a already advanced `compressed.seq_len` to the AFTER
                        // count; `before[r]` is the captured pre-advance value.
                        let indexer_rows_after =
                            state.indexer_compressed_seq_len().ok_or_else(|| {
                                anyhow!("DSv4 full-flatten P1b: indexer state missing")
                            })?;
                        crate::attention::dsv4_dsa_cache_write_gather_row(
                            &self.ctx,
                            &self.config,
                            state,
                            layer_pool,
                            dsa_shared,
                            indexer_rows_before[r],
                            indexer_rows_after,
                            // CompressedSparse compressor-indexer retains full
                            // history → window base 0 (matches csa_select).
                            0,
                            &mut cache_ptrs,
                        )?;
                    }
                    crate::attention::dsv4_dsa_cache_write_batched(
                        &self.ctx,
                        n,
                        &cache_ptrs,
                        &mut ptr_keepalive,
                        &mut keepalive,
                    )?;
                }
                let _compidx_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                {
                    let _nvtx = crate::nvtx::range("dsv4/mla_attn_batched_prepare");
                    // Op "c" batched KV pack (MODEL1 only): when `pack_batched` is on,
                    // the per-row pack's SW one-token + compressed-delta launches are
                    // HOISTED out of this loop into ONE batched launch each after it
                    // (the SW-ring BOOTSTRAP stays per-row — once per slot). Gather
                    // each row's k_prepared NoPE/RoPE base, compressor `compressed`
                    // base (0 = no-op for no-compressor rows), and per-slot device
                    // page-table pointer here; the persistent device page tables live
                    // in `flash.device_page_table` (not per-call temporaries — #8).
                    // V32 (head_dim==576) and SparseIndexed (!full_flatten) keep the
                    // per-row pack byte-for-byte.
                    // Post-restore/bulk gap (codex R3): the batched pack only
                    // packs THIS step's delta; a restored slot re-enters decode
                    // with fp8_kv_comp_packed_rows=0 and needs the single-row
                    // path's [packed_rows, seq_len) bulk rebuild — run the whole
                    // layer per-row this tick. Batched gap-fill is future work.
                    let comp_bulk_gap = (0..n)
                        .any(|r| slots[slot_ids[r]].attention[layer_idx].flashmla_comp_bulk_gap());
                    let pack_batched =
                        full_flatten && self.config.head_dim != 576 && !comp_bulk_gap;
                    let mut pack_nope_ptrs: Vec<u64> = Vec::new();
                    let mut pack_rope_ptrs: Vec<u64> = Vec::new();
                    let mut pack_compressed_ptrs: Vec<u64> = Vec::new();
                    let mut pack_pt_ptrs: Vec<u64> = Vec::new();
                    let mut pack_sw_blocks: usize = 0;
                    let mut pack_num_logical_pages: usize = 0;
                    if pack_batched {
                        pack_nope_ptrs.reserve(n);
                        pack_rope_ptrs.reserve(n);
                        pack_compressed_ptrs.reserve(n);
                        pack_pt_ptrs.reserve(n);
                    }
                    for r in 0..n {
                        if !full_flatten {
                            let src = normed.data.slice(r * hidden_size..(r + 1) * hidden_size);
                            ctx.stream
                                .memcpy_dtod(&src, &mut normed_row.data)
                                .map_err(|e| anyhow!("DSv4 batched attn copy-in failed: {e}"))?;
                            let cq_src = proj
                                .c_q_normed
                                .data
                                .slice(r * q_lora_rank..(r + 1) * q_lora_rank);
                            ctx.stream
                                .memcpy_dtod(&cq_src, &mut c_q_normed_row.data)
                                .map_err(|e| anyhow!("DSv4 batched c_q copy-in failed: {e}"))?;
                        }
                        // This row's q_prepared / k_prepared slices → fresh OWNED
                        // [*, 1] HiddenStates the compressed-only finish takes by
                        // value into the returned Dsv4MlaPrepared (the batched fwd
                        // reads the gathered Q; the finish loop reads
                        // k_prepared/local_attn — both must outlive the reused row
                        // scratch). Cheap [*, 1] bf16 alloc per row.
                        // SAFETY: each dtod copy fills the full buffer before read.
                        let mut q_prepared_owned =
                            unsafe { HiddenStates::uninit(&self.ctx, local_width, 1)? };
                        {
                            let qp_src = proj
                                .q_prepared
                                .data
                                .slice(r * local_width..(r + 1) * local_width);
                            ctx.stream
                                .memcpy_dtod(&qp_src, &mut q_prepared_owned.data)
                                .map_err(|e| {
                                    anyhow!("DSv4 batched q_prepared owned copy failed: {e}")
                                })?;
                        }
                        let mut k_prepared_owned =
                            // SAFETY: uninit device scratch; fully written before first read.
                            unsafe { HiddenStates::uninit(&self.ctx, mla_head_dim, 1)? };
                        {
                            let kp_src = proj
                                .k_prepared
                                .data
                                .slice(r * mla_head_dim..(r + 1) * mla_head_dim);
                            ctx.stream
                                .memcpy_dtod(&kp_src, &mut k_prepared_owned.data)
                                .map_err(|e| {
                                    anyhow!("DSv4 batched k_prepared owned copy failed: {e}")
                                })?;
                        }
                        let (
                            layer_pool,
                            dsa_shared,
                            flash_batch,
                            flashmla_scratch,
                            _prefill_shared,
                        ) = kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                        let flash_batch = flash_batch.ok_or_else(|| {
                            anyhow!("DSv4 batched decode lane: batch scratch missing")
                        })?;
                        let flashmla_scratch = flashmla_scratch.ok_or_else(|| {
                            anyhow!("DSv4 batched decode lane: single-row decode scratch missing")
                        })?;
                        let slot = &mut slots[slot_ids[r]];
                        slot_block_offsets
                            .push(layer_pool.flashmla_slot_first_block_or_zero(slot_ids[r])?);
                        page_tables.push(layer_pool.flashmla_page_table_padded_i32(slot_ids[r])?);
                        // Batched CSA select: thread this row's gather sink (q_i /
                        // weights staging + key_count capture). `csa_select` then
                        // runs cache writes only and returns `selected: None`; the
                        // ONE `csa_select_official_batched` after the loop fills
                        // `selected_batched`. When not batching DSA, `None` →
                        // byte-identical per-row select.
                        let need_dsa_stage =
                            use_batched_dsa_select && indexer_query_kv_score.is_none();
                        if need_dsa_stage && dsa_stage_q_i.is_none() {
                            let index_heads = self.config.index_n_heads;
                            let q_width = index_heads * self.config.index_head_dim;
                            // SAFETY: each row's slice is fully written by the per-row
                            // gather before the batched select reads it.
                            let q_b = unsafe { HiddenStates::uninit(&self.ctx, q_width, n)? };
                            // SAFETY: uninit device scratch; fully written before first read.
                            let w_b = unsafe { HiddenStates::uninit(&self.ctx, index_heads, n)? };
                            keepalive.keep_hidden(&q_b);
                            keepalive.keep_hidden(&w_b);
                            dsa_stage_q_i = Some(q_b);
                            dsa_stage_weights = Some(w_b);
                        }
                        let batched_gather = if use_batched_dsa_select {
                            Some(crate::attention::Dsv4DsaBatchedGather {
                                q_i_batch: dsa_stage_q_i.as_mut(),
                                weights_batch: dsa_stage_weights.as_mut(),
                                row: r,
                                key_counts: dsa_key_counts
                                    .as_mut()
                                    .expect("batched DSA key_counts present"),
                                // CompressedSparse full-flatten: P1b already wrote
                                // the cache; csa_select skips the per-row write.
                                // SparseIndexed (no P1b): false → per-row write.
                                cache_writes_in_prepass,
                            })
                        } else {
                            None
                        };
                        // Slice this row's `[width,1]` column out of the batched
                        // compressor/indexer pre-pass outputs (gate ON). Mirrors the
                        // q_prepared_owned slice convention above: a fresh owned
                        // [width,1] bf16 buffer per row, filled by a single dtod copy,
                        // referenced (not consumed) by the precomputed struct so it
                        // outlives the prepare call. `None` means no batched
                        // projection exists for this layer.
                        let slice_row = |src: &HiddenStates| -> Result<HiddenStates> {
                            let width = src.hidden_dim;
                            // SAFETY: the dtod copy fills the full buffer before read.
                            let mut row_buf = unsafe { HiddenStates::uninit(&self.ctx, width, 1)? };
                            let col = src.data.slice(r * width..(r + 1) * width);
                            self.ctx
                                .stream
                                .memcpy_dtod(&col, &mut row_buf.data)
                                .map_err(|e| {
                                    anyhow!("DSv4 batched compressor slice copy failed: {e}")
                                })?;
                            Ok(row_buf)
                        };
                        // Full-flatten: the compressor STATE update already ran
                        // batched in P1a, so the per-row `[width,1]` slice copies (the
                        // ∝n per-row work this campaign removes) are SKIPPED — pass
                        // `skip_compressor=true` + the P1a-captured `indexer_rows_before`
                        // and NO precomputed slices. Without full-flatten, build the
                        // per-row slices and run the per-row compressor as before.
                        //
                        // The owned slice buffers are bound at iteration scope so they
                        // outlive the prepare call that borrows them through
                        // `compressor_precomputed` (dropping at iteration end). Empty
                        // (`None`) under full-flatten — no per-row slice work.
                        let (comp_main_row, comp_indexer_row) = if full_flatten {
                            (None, None)
                        } else {
                            let main_row = match compressor_kv_score.as_ref() {
                                Some((kv, score)) => Some((slice_row(kv)?, slice_row(score)?)),
                                None => None,
                            };
                            let indexer_row = match indexer_kv_score.as_ref() {
                                Some((kv, score)) => Some((slice_row(kv)?, slice_row(score)?)),
                                None => None,
                            };
                            (main_row, indexer_row)
                        };
                        let compressor_precomputed = comp_main_row.as_ref().map(|(kv, score)| {
                            crate::attention::Dsv4CompressorPrecomputed {
                                main: (kv, score),
                                indexer: comp_indexer_row
                                    .as_ref()
                                    .map(|(ikv, iscore)| (ikv as &_, iscore as &_)),
                            }
                        });
                        // CSA indexer-query batched pre-pass: borrow this row's
                        // `[width,1]` column VIEW of `q_i`/`weights` so `csa_select`
                        // skips the per-row m=1 GEMVs AND the per-row D2D re-copy —
                        // the view feeds the prepass column's device pointer straight
                        // into the gather (zero copy; same address the old per-row
                        // copy produced → byte-identical). `None` when the pre-pass
                        // didn't run (SparseIndexed / no indexer) → byte-identical
                        // per-row GEMVs.
                        let indexer_query_precomputed =
                            indexer_query_kv_score.as_ref().map(|(q_i, weights)| {
                                crate::attention::Dsv4IndexerQueryPrecomputed {
                                    q_i: q_i.col(r),
                                    weights: weights.col(r),
                                }
                            });
                        let skip_compressor = full_flatten;
                        let idx_before_override = if full_flatten {
                            indexer_rows_before.get(r).copied()
                        } else {
                            None
                        };
                        let row_prepared = crate::attention::mla_attention_prepare_compressed_only(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            &normed_row,
                            &c_q_normed_row,
                            q_prepared_owned,
                            k_prepared_owned,
                            &proj,
                            &mut slot.attention[layer_idx],
                            layer_pool,
                            dsa_shared,
                            start_positions[r],
                            Some(&slot.start_pos_device),
                            batched_gather,
                            compressor_precomputed,
                            indexer_query_precomputed,
                            skip_compressor,
                            idx_before_override,
                            &mut keepalive,
                        )?;
                        // Pack this row's KV into the shared pool, then gather its
                        // global-head Q into q_batched[r]. flash arena + pool +
                        // batch scratch are disjoint (slots vs kv_adapter fields).
                        let (flash, sw_window, compressed) =
                            slot.attention[layer_idx].flashmla_pack_borrow(want_compressed)?;
                        if pack_batched {
                            // Op "c": run the once-per-slot SW-ring BOOTSTRAP here
                            // (no-op after the first decode); gather this row's pack
                            // pointers + per-slot device page table for the ONE
                            // batched SW+compressed pack issued after the loop.
                            crate::attention::flashmla_pack_sw_ring(
                                &self.ctx,
                                flash,
                                flashmla_scratch,
                                layer_pool,
                                sw_window,
                                &self.config,
                            )?;
                            pack_sw_blocks = flash.sw_blocks();
                            let (nope_ptr, ng) =
                                row_prepared.k_prepared.data.device_ptr(&self.ctx.stream);
                            pack_nope_ptrs.push(nope_ptr);
                            pack_rope_ptrs.push(
                                nope_ptr
                                    + crate::attention::flashmla_pack_rope_offset_bytes(
                                        &self.config,
                                    ),
                            );
                            drop(ng);
                            match compressed {
                                Some(c) => {
                                    let (cp, cg) = c.data.device_ptr(&self.ctx.stream);
                                    pack_compressed_ptrs.push(cp);
                                    drop(cg);
                                }
                                None => pack_compressed_ptrs.push(0),
                            }
                            // Use the persistent device page table from flash
                            // state (not a per-call temporary that would be freed
                            // before the batched kernel runs — #8 graph UAF).
                            pack_num_logical_pages =
                                pack_num_logical_pages.max(flash.device_page_table.len());
                            let (pt, pg) = flash.device_page_table.device_ptr(&self.ctx.stream);
                            pack_pt_ptrs.push(pt);
                            drop(pg);
                        } else {
                            crate::attention::flashmla_decode_pack_row(
                                &self.ctx,
                                &self.config,
                                layer.compress_ratio,
                                flash,
                                flashmla_scratch,
                                layer_pool,
                                sw_window,
                                &row_prepared.k_prepared,
                                compressed,
                                &slot.start_pos_device,
                            )?;
                        }
                        flash_batch.gather_q_row(
                            &self.ctx,
                            &self.config,
                            &row_prepared.q_prepared,
                            &self.tp,
                            r,
                            row_prepared.local_heads,
                        )?;
                        // CSA only: gather this row's indexer top-k `selected`
                        // (`[index_topk]` i32, == this layer's max_compressed_keys)
                        // into the contiguous `selected_batched[r * index_topk..]`,
                        // so the ONE batched index build can read it per row
                        // (`selected + row * max_compressed_keys`). SW/HCA skip this
                        // (selected is None) and the batched build sees selected_ptr=0
                        // — byte-identical to the prior SW/HCA-only lane.
                        //
                        // When the batched DSA select is engaged, `selected` is
                        // None (the per-row read was deferred); `selected_batched`
                        // is filled by the ONE `csa_select_official_batched` after
                        // this loop, so skip the per-row gather here.
                        // CSA + GLM SparseIndexed (has_indexer) both produce a per-row
                        // `selected`; gather it when the batched select lane is off.
                        if runs_indexer && !use_batched_dsa_select {
                            let sel = row_prepared.selected.as_ref().ok_or_else(|| {
                                anyhow!(
                                    "DSv4 batched indexer decode: row {r} missing indexer selected"
                                )
                            })?;
                            flash_batch.gather_selected_row(&self.ctx, sel, r)?;
                        }
                        // Keep the prepared buffers alive to function return — the
                        // batched fwd (after this loop) reads the gathered Q, and
                        // the finish loop reads k_prepared/local_attn. Mirrors the
                        // keepalive discipline of the per-row intermediates
                        // (guards the disabled-event-tracking premature free).
                        keepalive.keep_hidden(&row_prepared.q_prepared);
                        keepalive.keep_hidden(&row_prepared.k_prepared);
                        keepalive.keep_hidden(&row_prepared.local_attn);
                        if let Some(sel) = row_prepared.selected.as_ref() {
                            keepalive.keep_i32(sel);
                        }
                        prepared.push(row_prepared);
                    }
                    // Op "c": ONE batched SW one-token + compressed-delta pack over
                    // all N rows, replacing the 2×N per-row launches above. Issued
                    // here (after the prepare loop's per-row gathers, before
                    // build_layer_batch_meta / the batched fwd read the pool).
                    // Per-slot device page tables are persistent in
                    // `flash.device_page_table` (#8 fix); uploaded ptr arrays are
                    // kept alive past the launch via `ptr_keepalive` / `keepalive`.
                    if pack_batched && n > 0 {
                        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_batched");
                        let pool_ptr = {
                            let (layer_pool, _dsa, _flash_batch, _flashmla_scratch, _prefill) =
                                kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                            layer_pool.flashmla_pool_base_ptr(&self.ctx)?
                        };
                        let positions = batched_positions.as_ref().ok_or_else(|| {
                            anyhow!("DSv4 op-c batched pack: batched positions missing")
                        })?;
                        let nope_arr = crate::ops::upload_u64(&self.ctx, &pack_nope_ptrs)?;
                        let rope_arr = crate::ops::upload_u64(&self.ctx, &pack_rope_ptrs)?;
                        let compressed_arr =
                            crate::ops::upload_u64(&self.ctx, &pack_compressed_ptrs)?;
                        let pt_arr = crate::ops::upload_u64(&self.ctx, &pack_pt_ptrs)?;
                        crate::attention::flashmla_decode_pack_batched(
                            &self.ctx,
                            &self.config,
                            layer.compress_ratio,
                            pack_sw_blocks,
                            n,
                            pool_ptr,
                            &nope_arr,
                            &rope_arr,
                            &compressed_arr,
                            positions,
                            &pt_arr,
                            pack_num_logical_pages,
                        )?;
                        ptr_keepalive.push(nope_arr);
                        ptr_keepalive.push(rope_arr);
                        ptr_keepalive.push(compressed_arr);
                        ptr_keepalive.push(pt_arr);
                    }
                }
                if let Some(t) = _compidx_t {
                    ctx.stream.synchronize().ok();
                    // `_compidx_t` brackets exactly the per-row prepare loop above
                    // (compressor + Hadamard rotate + cache writes + indexer GEMM +
                    // per-row gathers, all fused inside
                    // `mla_attention_prepare_compressed_only`). Attribute this to the
                    // coarse `perrow_ms`; the batched READ is timed separately below
                    // and both fold into `compidx_ms` so the split sums to compidx.
                    let dt = t.elapsed().as_secs_f64() * 1000.0;
                    perrow_ms += dt;
                    compidx_ms += dt;
                }
                if let Some(t) = _prep_t {
                    ctx.stream.synchronize().ok();
                    prep_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
                // ── 1b. ONE batched CSA select (#60): the per-row READ (paged-MQA
                // logits + topk) for all N rows in ONE batch_size=N call, writing
                // directly into the FlashMLA batch scratch's `selected_batched`.
                // Runs AFTER all N rows' DSA caches are populated (per-row cache
                // writes happened in the prepare loop) and BEFORE
                // `build_layer_batch_meta` reads `selected_batched`.
                let _batchedread_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if use_batched_dsa_select {
                    let (q_i_batch, weights_batch) =
                        if let Some((q_i, weights)) = indexer_query_kv_score.as_ref() {
                            (q_i, weights)
                        } else {
                            (
                                dsa_stage_q_i
                                    .as_ref()
                                    .expect("batched DSA staging q present"),
                                dsa_stage_weights
                                    .as_ref()
                                    .expect("batched DSA staging weights present"),
                            )
                        };
                    let key_counts = dsa_key_counts
                        .take()
                        .expect("batched DSA key_counts present");
                    ensure!(
                        key_counts.len() == n,
                        "DSv4 batched CSA select: captured {} key_counts != n {}",
                        key_counts.len(),
                        n
                    );
                    // GLM SparseIndexed: full-sequence indexer, every token a key
                    // (ratio=1); context_lens = abs_pos / 1 = abs_pos. CSA keeps its
                    // compress_ratio. ensure still holds (1 > 0).
                    let ratio = if layer.mode == DeepSeekV4AttentionMode::SparseIndexed {
                        1
                    } else {
                        layer.compress_ratio
                    };
                    ensure!(
                        ratio > 0,
                        "DSv4 batched indexer select: indexer layer must have ratio>0"
                    );
                    // Byte-equivalent to the single-row GPU fill:
                    //   context_lens[r] = min(key_count_r, abs_pos_r / ratio)
                    //   positions[r]    = abs_pos_r   (abs_pos_r = start_positions[r])
                    // Skipped entirely on the device-meta path (kernel computes the
                    // same values on-device); empty Vecs are ignored there.
                    let mut context_lens_host = Vec::with_capacity(n);
                    let mut positions_host = Vec::with_capacity(n);
                    if !use_device_meta {
                        for r in 0..n {
                            let abs_pos = i32::try_from(start_positions[r]).map_err(|_| {
                                anyhow!(
                                    "DSv4 batched CSA abs_pos {} overflows i32",
                                    start_positions[r]
                                )
                            })?;
                            let avail = (abs_pos / ratio as i32).min(key_counts[r]);
                            context_lens_host.push(avail);
                            positions_host.push(abs_pos);
                        }
                    }
                    let local_index_heads = self.config.index_n_heads;
                    let score_scale = (self.config.index_head_dim as f32).powf(-0.5)
                        * (self.config.index_n_heads as f32).powf(-0.5);
                    let (layer_pool, dsa_shared, flash_batch, _flashmla_scratch, _prefill) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    let dsa_shared = dsa_shared.ok_or_else(|| {
                        anyhow!("DSv4 batched CSA select: shared DSA scratch missing")
                    })?;
                    let flash_batch = flash_batch
                        .ok_or_else(|| anyhow!("DSv4 batched CSA select: batch scratch missing"))?;
                    crate::attention::csa_select_official_batched(
                        &self.ctx,
                        &self.config,
                        layer_idx,
                        q_i_batch,
                        weights_batch,
                        dsa_shared,
                        layer_pool,
                        n,
                        &slot_ids[..n],
                        &context_lens_host,
                        &positions_host,
                        local_index_heads,
                        score_scale,
                        flash_batch.selected_batched_mut(),
                        &mut keepalive,
                        use_device_meta,
                        i32::try_from(ratio)?,
                        batched_positions.as_ref(),
                        &key_counts,
                        &mut pack_pt_keepalive,
                        // Eager batched lane: slot_ids/key_counts uploaded inside
                        // (not graph-replayed). The graph-safe persistent-buffer
                        // lane is the single-row decode-graph path only.
                        None,
                        None,
                    )?;
                }
                if let Some(t) = _batchedread_t {
                    ctx.stream.synchronize().ok();
                    // The ONE batched `csa_select_official_batched` READ (logits +
                    // topk). Folds into `compidx_ms` so the coarse split
                    // (perrow + batchedread) sums to the full PREPARE-tail compidx.
                    let dt = t.elapsed().as_secs_f64() * 1000.0;
                    batchedread_ms += dt;
                    compidx_ms += dt;
                }
                // ── 2. ONE batched indices/sched_meta build + ONE batched fwd.
                let _fwd_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                {
                    let _nvtx = crate::nvtx::range("dsv4/mla_attn_batched_fwd");
                    let (layer_pool, _dsa, flash_batch, _flashmla_scratch, _prefill) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    let flash_batch = flash_batch.ok_or_else(|| {
                        anyhow!("DSv4 batched decode lane: batch scratch missing")
                    })?;
                    // For CSA, build_layer_batch_meta sources the per-row top-k from
                    // the internal `selected_batched` buffer the prepare loop just
                    // filled (`[n, index_topk]`); SW/HCA pass selected_ptr=0 inside.
                    flash_batch.build_layer_batch_meta(
                        &self.ctx,
                        &self.config,
                        layer_idx,
                        layer.mode,
                        layer.compress_ratio,
                        &start_positions[..n],
                        &slot_block_offsets,
                        &page_tables,
                    )?;
                    let sm_scale = prepared[0].sm_scale;
                    flash_batch.decode_lane_fwd(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer_pool,
                        layer_idx,
                        n,
                        sm_scale,
                    )?;
                }
                if let Some(t) = _fwd_t {
                    ctx.stream.synchronize().ok();
                    fwd_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
                // ── 3. per-row slice-out → inverse-rope + SW-update → O-LoRA.
                let _finish_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                {
                    let _nvtx = crate::nvtx::range("dsv4/mla_attn_batched_finish");
                    // FINISH order is identical between the two gate states; only
                    // the inverse-RoPE + SW-window WRITE switch from N per-row
                    // launches to ONE batched launch each when `full_flatten` is on
                    // (the heavy ∝n FINISH tail). Both must run BEFORE the per-row
                    // O-LoRA (which consumes the inverse-roped `local_attn`).
                    //
                    // Pass F1: slice each row's global-head output into its
                    // `local_attn` (per-row, cheap memcpy/TP-slice). When
                    // full_flatten, also gather this row's local_attn / k_prepared /
                    // sw_window device pointers for the batched FINISH kernels.
                    let mut out_ptrs: Vec<u64> = if full_flatten {
                        Vec::with_capacity(n)
                    } else {
                        Vec::new()
                    };
                    let mut kprep_ptrs: Vec<u64> = if full_flatten {
                        Vec::with_capacity(n)
                    } else {
                        Vec::new()
                    };
                    let mut swcache_ptrs: Vec<u64> = if full_flatten {
                        Vec::with_capacity(n)
                    } else {
                        Vec::new()
                    };
                    // F1 batched: when full_flatten, slice ALL n rows' global-head
                    // output into the contiguous `local_attn_batched` in ONE launch,
                    // replacing the N per-row `slice_out_row` memcpy/TP-slices.
                    if full_flatten {
                        let local_heads = prepared[0].local_heads;
                        let (_layer_pool, _dsa, flash_batch, _flashmla_scratch, _prefill) =
                            kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                        let flash_batch = flash_batch.ok_or_else(|| {
                            anyhow!("DSv4 batched decode lane: batch scratch missing")
                        })?;
                        flash_batch.slice_out_batched(
                            &self.ctx,
                            &self.config,
                            &self.tp,
                            n,
                            local_heads,
                            &mut local_attn_batched,
                        )?;
                    }
                    let local_width_row = local_attn_batched.hidden_dim;
                    for r in 0..n {
                        // One mutable borrow of row r's prepared state; field
                        // accesses below are disjoint (k_prepared/scalars `&`,
                        // local_attn `&mut`) so the borrow checker splits them.
                        let p = &mut prepared[r];
                        if full_flatten {
                            // Gather this row's device pointers for the batched
                            // FINISH kernels. The output pointer is row r's slice of
                            // the contiguous `local_attn_batched` (the F2 O-LoRA then
                            // reads the same buffer). Single-stream: ptrs stay valid.
                            let row_view = local_attn_batched
                                .data
                                .slice(r * local_width_row..(r + 1) * local_width_row);
                            let (op, og) = row_view.device_ptr(&ctx.stream);
                            out_ptrs.push(op);
                            drop(og);
                            let (kp, kg) = p.k_prepared.data.device_ptr(&ctx.stream);
                            kprep_ptrs.push(kp);
                            drop(kg);
                            let slot = &mut slots[slot_ids[r]];
                            let sw_window = slot.attention[layer_idx].sw_window_cache_mut();
                            let (cp, cg) = sw_window.device_ptr_mut(&ctx.stream);
                            swcache_ptrs.push(cp);
                            drop(cg);
                        } else {
                            // Per-row lane (unchanged): slice this row out, then
                            // per-row inverse-RoPE + SW-window write, one launch each.
                            {
                                let (_layer_pool, _dsa, flash_batch, _flashmla_scratch, _prefill) =
                                    kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                                let flash_batch = flash_batch.ok_or_else(|| {
                                    anyhow!("DSv4 batched decode lane: batch scratch missing")
                                })?;
                                flash_batch.slice_out_row(
                                    &self.ctx,
                                    &self.config,
                                    &self.tp,
                                    r,
                                    p.local_heads,
                                    &mut p.local_attn,
                                )?;
                            }
                            let slot = &mut slots[slot_ids[r]];
                            let sw_window = slot.attention[layer_idx].sw_window_cache_mut();
                            crate::attention::flashmla_decode_finish_row(
                                &self.ctx,
                                &self.config,
                                sw_window,
                                &p.k_prepared,
                                &mut p.local_attn,
                                start_positions[r],
                                &slot.start_pos_device,
                                p.local_heads,
                                p.rope_base,
                                p.original_seq_len,
                                self.config.rope_parameters.factor,
                                self.config.rope_parameters.beta_fast,
                                self.config.rope_parameters.beta_slow,
                            )?;
                        }
                    }
                    // Batched FINISH tail (gate ON): ONE inverse-RoPE + ONE SW-window
                    // write over all N rows, replacing the per-row launches above.
                    // local_heads / rope params are uniform across the layer's rows.
                    if full_flatten {
                        let positions = batched_positions.as_ref().ok_or_else(|| {
                            anyhow!("DSv4 full-flatten finish: batched positions missing")
                        })?;
                        let local_heads = prepared[0].local_heads;
                        let rope_base = prepared[0].rope_base;
                        let original_seq_len = prepared[0].original_seq_len;
                        let out_arr = crate::ops::upload_u64(&self.ctx, &out_ptrs)?;
                        let kprep_arr = crate::ops::upload_u64(&self.ctx, &kprep_ptrs)?;
                        let swcache_arr = crate::ops::upload_u64(&self.ctx, &swcache_ptrs)?;
                        crate::attention::flashmla_decode_inverse_rope_batched(
                            &self.ctx,
                            &self.config,
                            &out_arr,
                            positions,
                            n,
                            local_heads,
                            rope_base,
                            original_seq_len,
                            self.config.rope_parameters.factor,
                            self.config.rope_parameters.beta_fast,
                            self.config.rope_parameters.beta_slow,
                        )?;
                        crate::attention::flashmla_decode_sw_window_batched(
                            &self.ctx,
                            &self.config,
                            &kprep_arr,
                            &swcache_arr,
                            positions,
                            n,
                        )?;
                        ptr_keepalive.push(out_arr);
                        ptr_keepalive.push(kprep_arr);
                        ptr_keepalive.push(swcache_arr);
                    }
                    // Pass F2: O-LoRA → attn_out (consumes the inverse-roped output).
                    // full_flatten batches over N in ONE mla_oproj(token_count=n):
                    // plain-o (GLM, seq-parametric dsv4_linear), single-output-group
                    // (Qwen3.6 MODEL1 at TP>1, decode-DeepGEMM M=n), AND grouped
                    // (groups>1, gather/GEMM/scatter M=n per group) all converge here
                    // reading the contiguous `local_attn_batched` and writing directly
                    // into `attn_out` ([hidden, n]) — no per-row alloc, no copy-out.
                    // The !full_flatten lane (SparseIndexed) stays per-row over
                    // `prepared[r].local_attn`.
                    if full_flatten {
                        // Batched O-LoRA over all N rows in one call. local_attn_batched
                        // is [local_width, n]; the decode lane reads it token-major.
                        let slot = &mut slots[slot_ids[0]];
                        crate::attention::mla_oproj(
                            &self.ctx,
                            &layer.attention,
                            &mut slot.attention[layer_idx],
                            // Decode: prefill DeepGEMM lane never taken. The decode
                            // lane reuses slot 0's shared transient fused_wqkv scratch
                            // at M=n (active_counts=[n], restored to [1] internally).
                            None,
                            &local_attn_batched,
                            n,
                            &mut keepalive,
                            &mut attn_out,
                        )?;
                    } else {
                        for r in 0..n {
                            let row_src = &prepared[r].local_attn;
                            let slot = &mut slots[slot_ids[r]];
                            crate::attention::mla_oproj(
                                &self.ctx,
                                &layer.attention,
                                &mut slot.attention[layer_idx],
                                None,
                                row_src,
                                1,
                                &mut keepalive,
                                &mut attn_out_row,
                            )?;
                            let mut dst = attn_out
                                .data
                                .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                            ctx.stream
                                .memcpy_dtod(&attn_out_row.data, &mut dst)
                                .map_err(|e| anyhow!("DSv4 batched attn copy-out failed: {e}"))?;
                        }
                    }
                }
                if let Some(t) = _finish_t {
                    ctx.stream.synchronize().ok();
                    finish_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
                if let Some(t) = _sw_t {
                    ctx.stream.synchronize().ok();
                    sw_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
            } else {
                let _csa_t = if phase_time {
                    ctx.stream.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let _nvtx = crate::nvtx::range("dsv4/mla_attn_per_row");
                for r in 0..n {
                    let src = normed.data.slice(r * hidden_size..(r + 1) * hidden_size);
                    ctx.stream
                        .memcpy_dtod(&src, &mut normed_row.data)
                        .map_err(|e| anyhow!("DSv4 batched attn copy-in failed: {e}"))?;
                    let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, _fp32) =
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
                        flashmla_scratch,
                        prefill_shared,
                        // Decode lane (start_pos_device Some): probe unreachable.
                        None,
                        start_positions[r],
                        Some(&slot.start_pos_device),
                        None,
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
                if let Some(t) = _csa_t {
                    ctx.stream.synchronize().ok();
                    csa_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
            }
            keepalive.keep_hidden(&attn_out);
            // Row-parallel O-LoRA: one all-reduce over [N, hidden]. NOT bit-identical
            // to N per-row all-reduces: NCCL tiles a [hidden,N] message differently
            // than N×[hidden,1], so identical-input rows pick up ~1 bf16 ULP of
            // per-row drift here. This is the legitimate batched-numerics seed.
            {
                let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?;
            }
            stage_all("attn_out", &attn_out);
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut attn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            if let Some(mhc) = attn_mhc.as_ref() {
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
            } else {
                // GLM plain residual: stream = attn_out + stream (stream_dim==hidden).
                crate::ops::add_batch(&self.ctx, &attn_out, &stream, &mut attn_stream)?;
            }
            keepalive.keep_hidden(&attn_stream);
            stream = attn_stream;
            stage_all("attn_residual", &stream);

            // ── MoE half: HC-wrap the grouped FP8 DeepGEMM MoE over the [N] batch.
            let _moe_t = if phase_time {
                ctx.stream.synchronize().ok();
                Some(std::time::Instant::now())
            } else {
                None
            };
            // GLM (hc_mult==1): plain RMSNorm of the stream (no ffn hyper-connection).
            // SAFETY: uninit device scratch; fully written before first read.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            let ffn_mhc = if self.config.is_glm() {
                crate::ops::rms_norm_batch(&self.ctx, &stream, &layer.ffn_norm, eps, &mut normed)?;
                None
            } else {
                let mhc =
                    crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)?;
                keepalive.keep_f32(&mhc.pre);
                keepalive.keep_f32(&mhc.post);
                keepalive.keep_f32(&mhc.comb);
                // SAFETY: fused hc_pre+rms_norm writes the full [seq_len, hidden_size] buffer.
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
                Some(mhc)
            };
            keepalive.keep_hidden(&normed);
            stage_all("ffn_input", &normed);
            // Routed MoE over the whole [N] batch. allreduce transport: Phase 6a
            // grouped path (one router gemm + one DeepGEMM grouped expert GEMM
            // over N×topk routes, decode_scratch=None) + one EP all-reduce.
            // deepep transport: the token-owned LL / intranode pipelines, which
            // are natively [N]-batched — same owned-slice structure as the
            // single-row forward (see its collective-participation invariants).
            // Bit-identity vs per-row is NOT expected (grouped GEMM / LL combine
            // tile over N differently); gated on needle retrieval, not byte-parity.
            //
            // GLM dense layer (`per_layer_dense_mlp[i]`): a plain SwiGLU FFN
            // replaces the routed-expert + shared-expert MoE entirely.
            let mut moe_with_shared =
                // SAFETY: uninit device scratch; fully written before first read.
                unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if let Some(dense) = layer.dense_mlp.as_ref() {
                dsv4_dense_mlp_forward(
                    &self.ctx,
                    dense,
                    &normed,
                    &mut moe_with_shared,
                    self.config.swiglu_limit,
                    &mut keepalive,
                )?;
                keepalive.keep_hidden(&moe_with_shared);
            } else {
                // SAFETY: uninit device scratch; fully written before first read.
                let mut moe_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                if use_deepep_transport {
                    #[cfg(feature = "deepep")]
                    {
                        let transport = self.deepep.as_ref().ok_or_else(|| {
                            anyhow!(
                                "ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted"
                            )
                        })?;
                        if transport.is_low_latency() {
                            // Token-owned LL path: DeepEP dispatch/combine are
                            // collectives every rank enters even at owned_n == 0
                            // (seq_len < world). The LL scratch is whole-forward
                            // scratch (fully overwritten per call); it is parked
                            // per-slot for the single-row path, so the batch
                            // borrows row 0's.
                            let scratch = slots[slot_ids[0]]
                                .deepep_ll_scratch
                                .as_mut()
                                .ok_or_else(|| {
                                    anyhow!("deepep_ll selected but slot LL scratch not allocated")
                                })?;
                            self.tp.shard_rows_and_allreduce(
                                &self.ctx,
                                &normed,
                                &mut moe_out,
                                hidden_size,
                                seq_len,
                                |owned_in, owned_out, start, owned_n| {
                                    keepalive.keep_hidden(owned_in);
                                    keepalive.keep_hidden(owned_out);
                                    crate::moe::dsv4_moe_forward_deepep_ll(
                                        self,
                                        transport,
                                        scratch,
                                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                                        &tokens[start..start + owned_n],
                                        tokens.len(),
                                        owned_in,
                                        owned_out,
                                        &mut keepalive,
                                    )
                                },
                            )?;
                        } else {
                            // Intranode normal-mode DeepEP: already [N]-shaped; its
                            // combine reduces across EP, no moe all-reduce needed.
                            crate::moe::dsv4_moe_forward_deepep(
                                self,
                                transport,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                tokens,
                                &normed,
                                &mut moe_out,
                                &mut keepalive,
                            )?;
                        }
                    }
                    #[cfg(not(feature = "deepep"))]
                    anyhow::bail!(
                        "ARLE_DSV4_MOE_TRANSPORT=deepep requires infer-cuda feature deepep"
                    );
                } else {
                    let tail = kv_adapter.moe_tail_scratch_mut();
                    let needs_moe_allreduce = crate::moe::dsv4_moe_forward(
                        self,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        tokens,
                        &normed,
                        &mut moe_out,
                        &mut keepalive,
                        tail,
                        mega_epoch,
                    )?;
                    keepalive.keep_hidden(&moe_out);
                    // Routed experts are EP-sharded → sum, then add the replicated
                    // shared expert once per rank. One all-reduce over [N, hidden].
                    if needs_moe_allreduce {
                        let _nvtx = crate::nvtx::range("dsv4/moe_allreduce");
                        self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
                    }
                }
                keepalive.keep_hidden(&moe_out);
                // Phase 6a: grouped shared expert over [N] (dense FFN, prefill path)
                // — one batched SwiGLU GEMM pair, replacing the per-row loop + N
                // host syncs.
                //
                // Waterfill (per-rank token-sharded shared-expert compute) was
                // KILLed 2026-07-06: pod A/B at n=64 (its own activation
                // threshold) showed no measurable wall-clock gain over this
                // unconditional path (75.9s vs 73.8s mean, well within
                // trial-to-trial noise — see
                // docs/experience/wins/2026-07-06-dsv4-deepep-waterfill-pending-remote.md).
                // Reuse the model-wide shared-expert output + FP8 scratch held on
                // the kv_adapter (#29) instead of a per-layer `uninit` + fresh
                // `dsv4_shared_expert` temporaries — batched decode was the last
                // path still allocating these every layer (launch-bound Step 1).
                // Same kernel sequence as the eager/verify template
                // (dsv4.rs:5440), byte-identical math, pooled provenance only.
                let (shared_out, shared_scratch) = kv_adapter.shared_expert_decode_mut();
                let shared = shared_out.ok_or_else(|| {
                    anyhow!("DSv4 batched decode requires shared-expert output buffer")
                })?;
                let scratch = shared_scratch
                    .ok_or_else(|| anyhow!("DSv4 batched decode requires shared-expert scratch"))?;
                shared.seq_len = seq_len; // rows ≤ MAX_SPEC_VERIFY_ROWS ≤ max_m
                ensure!(
                    shared.hidden_dim == hidden_size,
                    "DSv4 batched shared out hidden {} != {}",
                    shared.hidden_dim,
                    hidden_size
                );
                crate::moe::dsv4_shared_expert_forward_decode_scratch(
                    &self.ctx,
                    &self.ctx.stream,
                    layer.moe.as_ref().expect("DSv4 layer.moe"),
                    &normed,
                    shared,
                    self.config.swiglu_limit,
                    scratch,
                )?;
                // SAFETY: add_batch writes the full [seq_len, hidden_size] buffer.
                crate::ops::add_batch(&self.ctx, &moe_out, shared, &mut moe_with_shared)?;
                keepalive.keep_hidden(&moe_with_shared);
            }
            stage_all("moe_out", &moe_with_shared);
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut ffn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            if let Some(mhc) = ffn_mhc.as_ref() {
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
            } else {
                // GLM plain residual: stream = ffn_out + stream (stream_dim==hidden).
                crate::ops::add_batch(&self.ctx, &moe_with_shared, &stream, &mut ffn_stream)?;
            }
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
            stage_all("ffn_residual", &stream);
            // DSpark T3: capture each row's wide HC stream at this layer's OUTPUT
            // into its slot's tap buffer (decode: 1 token per row). Gated by
            // `dspark` so the default path is byte-identical.
            if dspark
                && let Some(tap_idx) = self
                    .config
                    .dspark_target_layer_ids
                    .iter()
                    .position(|&l| l == layer_idx)
            {
                for r in 0..n {
                    let tap = &mut slots[slot_ids[r]].dspark_taps[tap_idx];
                    self.capture_mtp_stream_hidden(&stream, r, 1, tap, &mut keepalive)?;
                }
            }
            if let Some(t) = _moe_t {
                ctx.stream.synchronize().ok();
                moe_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
            // Lens capture: head-project the layer's final stream for all rows
            // and stash the DEVICE logits (no sync/D2H until lens_flush).
            if lens_from.is_some_and(|from| layer_idx >= from) {
                let logits = self.verify_logits_from_stream(&stream, seq_len, &mut keepalive)?;
                crate::probe::lens_push(layer_idx, logits);
            }
        }

        for r in 0..n {
            slots[slot_ids[r]].seq_len += 1;
        }
        if phase_time {
            log::info!(
                "[decode-phase] n={n} csa_attn={:.1}ms sw_attn={:.1}ms (prep={:.1} [proj={:.1} compidx={:.1} compidx_split=[perrow={:.1} read={:.1}]] fwd={:.1} finish={:.1}) moe={:.1}ms",
                csa_ms,
                sw_ms,
                prep_ms,
                proj_ms,
                compidx_ms,
                perrow_ms,
                batchedread_ms,
                fwd_ms,
                finish_ms,
                moe_ms
            );
        }
        // Per-step per-GEMV breakdown (rank-0 only; self-gates on
        // ARLE_DSV4_LINEAR_PROFILE). Sums the compressor/indexer m=1 GEMVs that
        // compose compidx — the residual after the default-on batched-compressor
        // pre-pass, for the pending-remote TP=4 perf measurement.
        crate::linear_profile::print_rank0("decode-step");
        Ok((stream, keepalive))
    }

    /// Cross-slot batched MTP verify (batched-MTP Stage 1). Verify N draft
    /// chains in ONE layer-major forward over `M = Σ_s row_count_s` rows, grouped
    /// contiguously by slot. The MoE / HC-wrap / norm / head math runs over all
    /// M rows at once; attention stays PER-SLOT but not PER-ROW: each slot's
    /// chain chunk runs through one FlashMLA sparse verify call per layer. Per
    /// slot s, rows `[off_s .. off_s + len_s)` are its chain in schedule order;
    /// ancestor metadata makes row r attend slot s's committed KV + exactly the
    /// chain rows listed in `scheds[s].ancestors[r]`.
    ///
    /// Returns per slot `(argmax, hiddens)`: `argmax[j]` is the target's argmax
    /// AFTER chain row j (so the accepted next chain token must equal it), and
    /// `hiddens[j]` is that row's MTP stream hidden (for the commit path head).
    /// The lm_head extraction batches the greedy fold + projection + argmax over
    /// all M rows (one GEMM, one argmax, one D2H), then slices per slot — the
    /// per-row GEMV loop in the old verify cost N×depth syncs.
    ///
    /// PRECONDITIONS: each slot's `seq_len == start_pos_s` (contiguous append);
    /// allreduce transport only (the deepep_ll owned-shard partition is keyed on
    /// the whole-batch `seq_len`, not per-slot verify chunks — Stage 1/2 stay on
    /// allreduce, matching `mtp_forward_level`); the draft already wrote each
    /// slot's frozen-layer rings (the verify is pure: it READS the frozen KV).
    ///
    /// §0.1 mutated-buffer enumeration (every device buffer this fn WRITES):
    /// - All function-local `HiddenStates` (embeddings / stream / normed /
    ///   normed_chunk / attn_chunk / attn_out / moe_out / shared / *_stream /
    ///   head_normed / logits): SCRATCH — fully written before read each layer;
    ///   freed in stream order at fn return (keepalive guards the disabled-event-
    ///   tracking premature-free). No rollback needed.
    /// - `slots[s].attention[layer_idx]` rings (sw_window / fp8 / compressor):
    ///   NOT mutated by verify. The sparse verify lane is frozen and skips SW
    ///   cache updates; compressor/indexer updates are skipped by
    ///   `set_dsv4_verify_frozen(true)`. Accepted rows are written later by commit.
    /// - `slots[s].seq_len`: NOT mutated here (no commit). The caller advances it
    ///   in the fold commit. (Contrast `forward_decode_batch_stream_impl`, which
    ///   advances seq_len because it IS the commit.)
    /// - `slot.spec_normed`: the combined `[M,hidden]` normed is sliced by slot
    ///   and copied into the owning slot's cache for commit fold.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn forward_decode_batch_verify(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        row_tokens: &[Vec<u32>],
        start_positions: &[usize],
        scheds: &[SpecVerifySchedule],
    ) -> Result<(Vec<(Vec<u32>, Vec<DeviceVec>)>, HiddenStates)> {
        let n = slot_ids.len();
        ensure!(n > 0, "DSv4 batched verify requires at least one chain");
        ensure!(
            row_tokens.len() == n && start_positions.len() == n && scheds.len() == n,
            "DSv4 batched verify surface length mismatch (slots {n}, chains {}, starts {}, scheds {})",
            row_tokens.len(),
            start_positions.len(),
            scheds.len()
        );
        ensure!(
            !crate::runtime_flags::dsv4_moe_transport()?.is_deepep(),
            "DSv4 batched MTP verify supports the allreduce transport only \
             (the deepep_ll owned-shard partition is keyed on the whole-batch \
              seq_len, not per-slot verify chunks)"
        );

        // Per-slot row-block offsets into the combined [M, *] buffers.
        let lens: Vec<usize> = row_tokens.iter().map(|c| c.len()).collect();
        for (s, &len) in lens.iter().enumerate() {
            ensure!(
                len >= 1 && scheds[s].positions.len() == len && scheds[s].ancestors.len() == len,
                "DSv4 batched verify slot {s} chain rows {len} != schedule rows ({}, {})",
                scheds[s].positions.len(),
                scheds[s].ancestors.len()
            );
            scheds[s].validate_sparse_at(start_positions[s])?;
            let slot = &slots[slot_ids[s]];
            ensure!(
                slot.seq_len == start_positions[s],
                "DSv4 batched verify slot {} seq_len {} != start_pos {}; verify requires contiguous appends",
                slot_ids[s],
                slot.seq_len,
                start_positions[s]
            );
            ensure!(
                start_positions[s] + len <= slot.max_seq_len,
                "DSv4 batched verify slot {} sequence {} exceeds max_seq_len {}",
                slot_ids[s],
                start_positions[s] + len,
                slot.max_seq_len
            );
        }
        let mut offsets = Vec::with_capacity(n);
        let mut acc = 0usize;
        for &len in &lens {
            offsets.push(acc);
            acc += len;
        }
        let m = acc; // total rows over all per-slot chains

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = m; // batch dimension: M verify rows
        let mega_epoch = self.begin_mega_moe_forward(seq_len)?;
        let eps = self.config.rms_norm_eps;
        let dspark = self.config.is_dspark();
        let mut keepalive = Dsv4ForwardKeepalive::new(false);

        // Combined token list, grouped by slot (slot s's chain at [off_s..]).
        let tokens: Vec<u32> = row_tokens.iter().flat_map(|c| c.iter().copied()).collect();
        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

        let nvtx_embed = crate::nvtx::range("dsv4/embed");
        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids_host)?;
        keepalive.keep_i32(&token_ids);
        // SAFETY: embedding_batch writes the full [m, hidden_size] buffer.
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

        // Per-slot chain verify: one FlashMLA sparse forward over each slot
        // chunk per layer. The prefix metadata expresses row r -> ancestors
        // explicitly. Current topk does not add rows: D2 verifies 3 rows.
        let max_chunk = *lens.iter().max().unwrap_or(&1);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut normed_chunk = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, max_chunk)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_chunk = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, max_chunk)? };
        keepalive.keep_hidden(&normed_chunk);
        keepalive.keep_hidden(&attn_chunk);
        let sparse_metas: Vec<crate::attention::Dsv4ChainVerifyAttnMeta> = scheds
            .iter()
            .map(|s| {
                crate::attention::Dsv4ChainVerifyAttnMeta::new(
                    &self.ctx,
                    &s.positions,
                    &s.ancestors,
                )
            })
            .collect::<Result<_>>()?;

        crate::attention::set_dsv4_verify_frozen(true);
        let result = (|| -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
                // ── Attention half: HC params + pre-norm over the whole [M] batch.
                // ponytail: pod-verify GLM hc_mult==1 plain residual + identity stream (no hyper-connection)
                // SAFETY: uninit device scratch; fully written before first read.
                let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                let attn_mhc = if self.config.is_glm() {
                    // GLM: plain RMSNorm of the stream (stream IS the hidden).
                    crate::ops::rms_norm_batch(
                        &self.ctx,
                        &stream,
                        &layer.attn_norm,
                        eps,
                        &mut normed,
                    )?;
                    None
                } else {
                    let mhc = crate::hc::gen_mhc_params(
                        &self.ctx,
                        &self.config,
                        &layer.hc_attn,
                        &stream,
                    )?;
                    keepalive.keep_f32(&mhc.pre);
                    keepalive.keep_f32(&mhc.post);
                    keepalive.keep_f32(&mhc.comb);
                    // SAFETY: fused hc_pre+rms_norm writes the full [m, hidden_size] buffer.
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
                    Some(mhc)
                };
                keepalive.keep_hidden(&normed);

                // ── Attention PER SLOT / PER CHUNK: write the slot block into
                // chunk scratch, run the sparse chain-verify lane once, then
                // scatter it back into the combined block.
                // SAFETY: each [off..off+len) block of attn_out is written by the
                // copy-out below before attn_out is read by hc_post.
                let mut attn_out =
                    unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                {
                    let _nvtx = crate::nvtx::range("dsv4/mla_attn_chain_verify");
                    for s in 0..n {
                        let off = offsets[s];
                        let len = lens[s];
                        normed_chunk.seq_len = len;
                        attn_chunk.seq_len = len;
                        let src = normed
                            .data
                            .slice(off * hidden_size..(off + len) * hidden_size);
                        self.ctx
                            .stream
                            .memcpy_dtod(
                                &src,
                                &mut normed_chunk.data.slice_mut(0..len * hidden_size),
                            )
                            .map_err(|e| {
                                anyhow!("DSv4 batched verify chunk copy-in failed: {e}")
                            })?;
                        // Commit-fold scatter: persist THIS slot's per-layer
                        // attn-normed chain rows into the OWNING slot's
                        // `spec_normed[layer_idx]`. The combined `normed` is
                        // sliced per slot, so there is no cross-slot aliasing.
                        let slot = &mut slots[slot_ids[s]];
                        let cache = slot.spec_normed.as_mut().ok_or_else(|| {
                            anyhow!("DSv4 batched verify without spec_normed cache")
                        })?;
                        let mut dst = cache[layer_idx].data.slice_mut(0..len * hidden_size);
                        self.ctx
                            .stream
                            .memcpy_dtod(
                                &normed
                                    .data
                                    .slice(off * hidden_size..(off + len) * hidden_size),
                                &mut dst,
                            )
                            .map_err(|e| {
                                anyhow!("DSv4 batched verify spec_normed persist failed: {e}")
                            })?;
                        let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
                            kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                        let slot = &mut slots[slot_ids[s]];
                        crate::attention::mla_attention(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            &normed_chunk,
                            &mut slot.attention[layer_idx],
                            layer_pool,
                            dsa_shared,
                            flashmla_scratch,
                            prefill_shared,
                            fp32,
                            start_positions[s],
                            None,
                            Some(&sparse_metas[s]),
                            &self.tp,
                            &mut attn_chunk,
                            &mut keepalive,
                        )?;
                        let mut dst = attn_out
                            .data
                            .slice_mut(off * hidden_size..(off + len) * hidden_size);
                        self.ctx
                            .stream
                            .memcpy_dtod(&attn_chunk.data.slice(0..len * hidden_size), &mut dst)
                            .map_err(|e| {
                                anyhow!("DSv4 batched verify chunk copy-out failed: {e}")
                            })?;
                    }
                }
                keepalive.keep_hidden(&attn_out);
                // Row-parallel O-LoRA: one all-reduce over [M, hidden].
                {
                    let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                    self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?;
                }
                // SAFETY: hc_post / add_batch writes the full stream buffer.
                let mut attn_stream =
                    unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
                if let Some(mhc) = attn_mhc.as_ref() {
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
                } else {
                    // GLM plain residual: stream = attn_out + stream (stream_dim==hidden).
                    crate::ops::add_batch(&self.ctx, &attn_out, &stream, &mut attn_stream)?;
                }
                keepalive.keep_hidden(&attn_stream);
                stream = attn_stream;

                // ── MoE half: HC-wrap the grouped MoE over the whole [M] batch
                // (the dominant amortization — math-identical to per-row, byte
                // parity NOT expected: grouped GEMM tiles over M differently;
                // gated on needle, not byte-parity). Allreduce transport only.
                // GLM (hc_mult==1): plain RMSNorm of the stream (no ffn hyper-connection).
                // SAFETY: uninit device scratch; fully written before first read.
                let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                let ffn_mhc = if self.config.is_glm() {
                    crate::ops::rms_norm_batch(
                        &self.ctx,
                        &stream,
                        &layer.ffn_norm,
                        eps,
                        &mut normed,
                    )?;
                    None
                } else {
                    let mhc =
                        crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)?;
                    keepalive.keep_f32(&mhc.pre);
                    keepalive.keep_f32(&mhc.post);
                    keepalive.keep_f32(&mhc.comb);
                    // SAFETY: fused hc_pre+rms_norm writes the full [m, hidden_size] buffer.
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
                    Some(mhc)
                };
                keepalive.keep_hidden(&normed);
                // GLM dense layer (`per_layer_dense_mlp[i]`): a plain SwiGLU FFN
                // replaces the routed-expert + shared-expert MoE entirely.
                let mut moe_with_shared =
                    // SAFETY: uninit device scratch; fully written before first read.
                    unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                if let Some(dense) = layer.dense_mlp.as_ref() {
                    dsv4_dense_mlp_forward(
                        &self.ctx,
                        dense,
                        &normed,
                        &mut moe_with_shared,
                        self.config.swiglu_limit,
                        &mut keepalive,
                    )?;
                    keepalive.keep_hidden(&moe_with_shared);
                } else {
                    // SAFETY: the MoE forward writes the full routed output buffer.
                    let mut moe_out =
                        unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                    let needs_moe_allreduce = crate::moe::dsv4_moe_forward(
                        self,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        &tokens,
                        &normed,
                        &mut moe_out,
                        &mut keepalive,
                        None,
                        mega_epoch,
                    )?;
                    keepalive.keep_hidden(&moe_out);
                    if needs_moe_allreduce {
                        let _nvtx = crate::nvtx::range("dsv4/moe_allreduce");
                        self.tp.all_reduce_sum(&self.ctx, &mut moe_out)?;
                    }
                    // Grouped shared expert over [M] (dense FFN, prefill path).
                    let mut shared =
                        // SAFETY: uninit device scratch; fully written before first read.
                        unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                    crate::moe::dsv4_shared_expert_forward(
                        &self.ctx,
                        &self.ctx.stream,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        &normed,
                        &mut shared,
                        self.config.swiglu_limit,
                        &mut keepalive,
                    )?;
                    keepalive.keep_hidden(&shared);
                    // SAFETY: add_batch writes the full [m, hidden_size] buffer.
                    crate::ops::add_batch(&self.ctx, &moe_out, &shared, &mut moe_with_shared)?;
                    keepalive.keep_hidden(&moe_with_shared);
                }
                // SAFETY: hc_post / add_batch writes the full stream buffer.
                let mut ffn_stream =
                    unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
                if let Some(mhc) = ffn_mhc.as_ref() {
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
                } else {
                    // GLM plain residual: stream = ffn_out + stream (stream_dim==hidden).
                    crate::ops::add_batch(&self.ctx, &moe_with_shared, &stream, &mut ffn_stream)?;
                }
                keepalive.keep_hidden(&ffn_stream);
                stream = ffn_stream;
                // DSpark T3: capture each slot's chain rows' wide HC stream at
                // this layer's OUTPUT into its tap buffer. Row block [off_s..off_s+len_s)
                // belongs to slot s; writes len_s rows (the full chain — the
                // commit fold later reads only the accepted prefix).
                if dspark
                    && let Some(tap_idx) = self
                        .config
                        .dspark_target_layer_ids
                        .iter()
                        .position(|&l| l == layer_idx)
                {
                    for s in 0..n {
                        let tap = &mut slots[slot_ids[s]].dspark_taps[tap_idx];
                        self.capture_mtp_stream_hidden(
                            &stream,
                            offsets[s],
                            lens[s],
                            tap,
                            &mut keepalive,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        crate::attention::set_dsv4_verify_frozen(false);
        result?;

        // ── Per-row MTP stream hidden capture (all M rows), then per-slot slice.
        let mut hiddens_all = Vec::with_capacity(m);
        for i in 0..m {
            let mut h = DeviceVec::zeros(&self.ctx, stream_dim)?;
            self.capture_mtp_stream_hidden(&stream, i, 1, &mut h, &mut keepalive)?;
            hiddens_all.push(h);
        }

        // ── Target logits matrix over all M rows, then its greedy top-1 view.
        let _nvtx = crate::nvtx::range("dsv4/lm_head_verify_batched");
        let logits = self.verify_logits_from_stream(&stream, m, &mut keepalive)?;
        let argmax_all = self.mtp_argmax_batch(&logits)?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);

        // Slice argmax + hiddens back per slot (own row block). hiddens are
        // moved out (DeviceVec owns the data; the caller keeps the accepted one).
        let mut hiddens_iter = hiddens_all.into_iter();
        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let len = lens[s];
            let off = offsets[s];
            let argmax = argmax_all[off..off + len].to_vec();
            let mut hiddens = Vec::with_capacity(len);
            for _ in 0..len {
                hiddens.push(
                    hiddens_iter
                        .next()
                        .ok_or_else(|| anyhow!("DSv4 batched verify hidden slice underflow"))?,
                );
            }
            out.push((argmax, hiddens));
        }
        Ok((out, logits))
    }

    fn capture_mtp_stream_hidden(
        &self,
        stream: &HiddenStates,
        offset: usize,
        len: usize,
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
        let elems = len * stream_dim;
        ensure!(
            out.len >= elems,
            "DSv4 MTP hidden capture len {} < elems {elems}",
            out.len
        );
        let src = stream
            .data
            .slice(offset * stream_dim..offset * stream_dim + elems);
        let mut dst = out.data.slice_mut(0..elems);
        self.ctx
            .stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("DSv4 MTP hidden capture D2D failed: {e}"))?;
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
        if self.config.is_glm() {
            // GLM: head hidden = stream row (stream_dim==hidden, no head HC mixer).
            // ponytail: pod-verify GLM hc_mult==1 head hidden = stream row (identity)
            crate::stage_profile::profile(ctx, "dsv4/stage/head_hc", || {
                crate::ops::copy_row_to_vec(ctx, stream, seq_len - 1, &mut last_hidden)
            })?;
        } else {
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
        }
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
        let token = crate::stage_profile::profile(ctx, "dsv4/stage/sample", || {
            self.sample_logits(&logits, params, position)
        })?;
        Ok(token)
    }

    /// Sample the next token from the decode-tail logits: the standard
    /// full-vocab sampler, or the cross-rank merge when the vocab-sharded
    /// lm_head is active (`logits` then hold this rank's padded slice).
    fn sample_logits(
        &self,
        logits: &DeviceVec,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        match &self.lm_head_shard {
            Some(shard) => crate::executor::sample_cuda_token_vocab_sharded(
                &self.ctx, &self.tp, logits, shard, params, position,
            ),
            None => crate::executor::sample_cuda_token(&self.ctx, logits, params, position),
        }
    }

    fn forward_tokens_verify_stream_persistent(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        sched: &SpecVerifySchedule,
    ) -> Result<(HiddenStates, Dsv4ForwardKeepalive)> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 persistent verify forward requires at least one token"
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
        let seq_len = tokens.len();
        let mega_epoch = self.begin_mega_moe_forward(seq_len)?;
        ensure!(
            seq_len <= MAX_SPEC_VERIFY_ROWS,
            "DSv4 persistent verify rows {seq_len} exceed capacity {MAX_SPEC_VERIFY_ROWS}"
        );
        ensure!(
            sched.positions.len() == seq_len && sched.ancestors.len() == seq_len,
            "DSv4 sparse verify schedule rows ({}, {}) != token rows {seq_len}",
            sched.positions.len(),
            sched.ancestors.len()
        );
        sched.validate_sparse_at(start_pos)?;
        ensure!(
            !crate::runtime_flags::dsv4_moe_transport()?.is_deepep(),
            "DSv4 persistent MTP verify scratch currently supports allreduce transport"
        );

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let eps = self.config.rms_norm_eps;
        let ctx = &self.ctx;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let scratch = slot
            .spec_verify
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 scheduled verify missing persistent scratch"))?;
        scratch.set_rows(seq_len)?;

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let nvtx_embed = crate::nvtx::range("dsv4/embed");
        let token_ids = crate::ops::upload_i32(ctx, &token_ids_host)?;
        crate::stage_profile::profile(ctx, "dsv4/stage/embed", || {
            crate::ops::embedding_batch(
                ctx,
                &self.embed_tokens,
                &token_ids,
                &mut scratch.embeddings,
            )
        })?;
        crate::stage_profile::profile(ctx, "dsv4/stage/embed_hc_expand", || {
            crate::hc::initial_stream_from_embeddings(
                ctx,
                &scratch.embeddings,
                hidden_size,
                hc_mult,
                &mut scratch.initial_stream,
            )
        })?;
        drop(nvtx_embed);

        let sparse_verify_meta = crate::attention::Dsv4ChainVerifyAttnMeta::new(
            ctx,
            &sched.positions,
            &sched.ancestors,
        )?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
            let (prev_layers, current_and_rest) = scratch.layers.split_at_mut(layer_idx);
            let current = current_and_rest
                .first_mut()
                .ok_or_else(|| anyhow!("DSv4 spec-verify layer scratch {layer_idx} missing"))?;

            {
                let stream = if layer_idx == 0 {
                    &scratch.initial_stream
                } else {
                    &prev_layers[layer_idx - 1].ffn_stream
                };
                let normed = &mut current.attn_normed;
                let attn_mhc = if self.config.is_glm() {
                    crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_pre_norm", || {
                        crate::ops::rms_norm_batch(ctx, stream, &layer.attn_norm, eps, normed)
                    })?;
                    None
                } else {
                    let mhc =
                        crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_params", || {
                            crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_attn, stream)
                        })?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_pre_norm", || {
                        crate::hc::mhc_pre_rms_norm(
                            ctx,
                            stream,
                            &mhc.pre,
                            &layer.attn_norm,
                            eps,
                            hidden_size,
                            hc_mult,
                            normed,
                        )
                    })?;
                    Some(mhc)
                };

                if let Some(cache) = slot.spec_normed.as_mut() {
                    let rows = seq_len * hidden_size;
                    let src = normed.data.slice(0..rows);
                    let mut dst = cache[layer_idx].data.slice_mut(0..rows);
                    ctx.stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSv4 commit-fold normed persist failed: {e}"))?;
                }

                let _nvtx = crate::nvtx::range("dsv4/mla_attn_sparse_verify");
                let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
                    kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                crate::attention::mla_attention(
                    ctx,
                    &self.config,
                    &layer.attention,
                    layer.mode,
                    layer.compress_ratio,
                    layer_idx,
                    normed,
                    &mut slot.attention[layer_idx],
                    layer_pool,
                    dsa_shared,
                    flashmla_scratch,
                    prefill_shared,
                    fp32,
                    start_pos,
                    None,
                    Some(&sparse_verify_meta),
                    &self.tp,
                    &mut current.attn_out,
                    &mut keepalive,
                )?;

                let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                crate::stage_profile::profile(ctx, "dsv4/stage/attn_allreduce", || {
                    self.tp.all_reduce_sum(ctx, &mut current.attn_out)
                })?;

                if let Some(mhc) = attn_mhc.as_ref() {
                    crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_post", || {
                        crate::hc::hc_post(
                            ctx,
                            &current.attn_out,
                            stream,
                            &mhc.post,
                            &mhc.comb,
                            hidden_size,
                            hc_mult,
                            &mut current.attn_stream,
                        )
                    })?;
                } else {
                    crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_post", || {
                        crate::ops::add_batch(
                            ctx,
                            &current.attn_out,
                            stream,
                            &mut current.attn_stream,
                        )
                    })?;
                }
            }

            {
                let stream = &current.attn_stream;
                let normed = &mut current.ffn_normed;
                let ffn_mhc = if self.config.is_glm() {
                    crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_pre_norm", || {
                        crate::ops::rms_norm_batch(ctx, stream, &layer.ffn_norm, eps, normed)
                    })?;
                    None
                } else {
                    let mhc =
                        crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_params", || {
                            crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_ffn, stream)
                        })?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_pre_norm", || {
                        crate::hc::mhc_pre_rms_norm(
                            ctx,
                            stream,
                            &mhc.pre,
                            &layer.ffn_norm,
                            eps,
                            hidden_size,
                            hc_mult,
                            normed,
                        )
                    })?;
                    Some(mhc)
                };

                if let Some(dense) = layer.dense_mlp.as_ref() {
                    dsv4_dense_mlp_forward(
                        ctx,
                        dense,
                        normed,
                        &mut current.moe_with_shared,
                        self.config.swiglu_limit,
                        &mut keepalive,
                    )?;
                } else {
                    let needs_moe_allreduce = crate::moe::dsv4_moe_forward(
                        self,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        tokens,
                        normed,
                        &mut current.moe_out,
                        &mut keepalive,
                        None,
                        mega_epoch,
                    )?;

                    if needs_moe_allreduce {
                        let _nvtx = crate::nvtx::range("dsv4/moe_allreduce");
                        crate::stage_profile::profile(ctx, "dsv4/stage/moe_allreduce", || {
                            self.tp.all_reduce_sum(ctx, &mut current.moe_out)
                        })?;
                    }

                    let _nvtx_shared_hc = crate::nvtx::range("dsv4/shared_hc");
                    let (shared_out, shared_scratch) = kv_adapter.shared_expert_decode_mut();
                    let shared = shared_out.ok_or_else(|| {
                        anyhow!("DSv4 verify requires shared-expert output buffer")
                    })?;
                    shared.seq_len = seq_len;
                    ensure!(
                        shared.hidden_dim == hidden_size,
                        "DSv4 shared verify scratch hidden {} != {}",
                        shared.hidden_dim,
                        hidden_size
                    );
                    let shared_scratch = shared_scratch
                        .ok_or_else(|| anyhow!("DSv4 verify requires shared-expert scratch"))?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                        crate::moe::dsv4_shared_expert_forward_decode_scratch(
                            ctx,
                            &ctx.stream,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            normed,
                            shared,
                            self.config.swiglu_limit,
                            shared_scratch,
                        )
                    })?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/shared_add", || {
                        crate::ops::add_batch(
                            ctx,
                            &current.moe_out,
                            shared,
                            &mut current.moe_with_shared,
                        )
                    })?;
                }

                if let Some(mhc) = ffn_mhc.as_ref() {
                    crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_post", || {
                        crate::hc::hc_post(
                            ctx,
                            &current.moe_with_shared,
                            stream,
                            &mhc.post,
                            &mhc.comb,
                            hidden_size,
                            hc_mult,
                            &mut current.ffn_stream,
                        )
                    })?;
                } else {
                    crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_post", || {
                        crate::ops::add_batch(
                            ctx,
                            &current.moe_with_shared,
                            stream,
                            &mut current.ffn_stream,
                        )
                    })?;
                }

                if self.config.is_dspark()
                    && let Some(tap_idx) = self
                        .config
                        .dspark_target_layer_ids
                        .iter()
                        .position(|&l| l == layer_idx)
                {
                    let elems = stream_dim * seq_len;
                    let src = current.ffn_stream.data.slice(0..elems);
                    let mut dst = slot.dspark_taps[tap_idx].data.slice_mut(0..elems);
                    ctx.stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSpark verify tap capture D2D failed: {e}"))?;
                }
            }
        }

        // SAFETY: uninit device scratch; fully written before first read.
        let mut stream_out = unsafe { HiddenStates::uninit(ctx, stream_dim, seq_len)? };
        let elems = stream_dim * seq_len;
        let final_stream = scratch
            .layers
            .last()
            .ok_or_else(|| anyhow!("DSv4 spec-verify scratch has no layers"))?;
        let src = final_stream.ffn_stream.data.slice(0..elems);
        ctx.stream
            .memcpy_dtod(&src, &mut stream_out.data)
            .map_err(|e| anyhow!("DSv4 persistent verify stream export failed: {e}"))?;
        slot.seq_len += seq_len;
        Ok((stream_out, keepalive))
    }

    fn forward_tokens_stream_impl(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        persist_spec_normed: bool,
        // `Some`: verify a scheduled row chunk. Ancestor metadata routes
        // attention through one FlashMLA sparse verify call per layer; point-wise
        // and MoE stay batched for weight-read amortization. Current top-k only
        // widens candidate matching; it does not add scheduled rows.
        verify: Option<&SpecVerifySchedule>,
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
        if let Some(sched) = verify
            && tokens.len() <= MAX_SPEC_VERIFY_ROWS
            && !crate::runtime_flags::dsv4_moe_transport()?.is_deepep()
        {
            return self.forward_tokens_verify_stream_persistent(
                slot, kv_adapter, tokens, start_pos, sched,
            );
        }

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = tokens.len();
        let mega_epoch = self.begin_mega_moe_forward(seq_len)?;
        let eps = self.config.rms_norm_eps;
        let use_deepep_transport = crate::runtime_flags::dsv4_moe_transport()?.is_deepep();
        // DSpark T3: capture the wide stream at each target layer output. One bool
        // check per layer when off; the position scan only runs when on.
        let dspark = self.config.is_dspark();
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let ctx = &self.ctx;
        // DSpark prefix seed: a PREFILL forward (`seq_len > 1`) additionally
        // captures ALL rows per target layer into transient `[stream_dim, seq_len]`
        // buffers (token-major), so the executor can seed the full committed
        // context (the per-slot single-row `dspark_taps` only holds `seq_len-1`).
        // Prefill-scoped; bounded by the chunk size. Stashed on the slot post-loop.
        let mut dspark_prompt_bufs: Option<Vec<DeviceVec>> = if dspark && seq_len > 1 {
            let n_taps = self.config.dspark_target_layer_ids.len();
            Some(
                (0..n_taps)
                    .map(|_| DeviceVec::zeros(ctx, stream_dim * seq_len))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        // DEBUG: catch any deferred CUDA error from init/boot before first forward op.
        if use_deepep_transport {
            self.ctx.stream.synchronize().map_err(|e| {
                anyhow!("DSv4 deepep: stream error at forward entry (before embed): {e}")
            })?;
        }
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
                .map_err(|e| anyhow!("DSv4 embed lookup failed: {e}"))
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
        let sparse_verify_meta = match verify {
            Some(sched) => {
                ensure!(
                    sched.positions.len() == seq_len && sched.ancestors.len() == seq_len,
                    "DSv4 sparse verify schedule rows ({}, {}) != token rows {seq_len}",
                    sched.positions.len(),
                    sched.ancestors.len()
                );
                sched.validate_sparse_at(start_pos)?;
                Some(crate::attention::Dsv4ChainVerifyAttnMeta::new(
                    &self.ctx,
                    &sched.positions,
                    &sched.ancestors,
                )?)
            }
            _ => None,
        };
        // Logit lens: single-row eager DECODE steps only (prefill and
        // spec-decode/MTP verify are not lens-instrumented); off = one Option
        // compare per layer.
        let lens_from = match crate::probe::lens_layers() {
            depth if depth > 0 && seq_len == 1 && verify.is_none() => {
                Some(self.layers.len().saturating_sub(depth))
            }
            _ => None,
        };
        if lens_from.is_some() {
            crate::probe::lens_begin();
        }
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let _layer_nvtx = crate::nvtx::range(&format!("dsv4/layer_{layer_idx:02}"));
            // ── Attention half: HC-wrap MLA attention.
            // ponytail: pod-verify GLM hc_mult==1 plain residual + identity stream (no hyper-connection)
            // SAFETY: fused hc_pre+rms_norm / plain rms_norm writes the full
            // [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            let attn_mhc = if self.config.is_glm() {
                // GLM: plain RMSNorm of the stream (stream IS the hidden).
                crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_pre_norm", || {
                    crate::ops::rms_norm_batch(
                        &self.ctx,
                        &stream,
                        &layer.attn_norm,
                        eps,
                        &mut normed,
                    )
                })?;
                None
            } else {
                let mhc = crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_params", || {
                    crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_attn, &stream)
                })?;
                keepalive.keep_f32(&mhc.pre);
                keepalive.keep_f32(&mhc.post);
                keepalive.keep_f32(&mhc.comb);
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
                Some(mhc)
            };
            keepalive.keep_hidden(&normed);
            if persist_spec_normed {
                let cache = slot
                    .spec_normed
                    .as_mut()
                    .ok_or_else(|| anyhow!("DSv4 verify without spec_normed cache"))?;
                let rows = seq_len * hidden_size;
                let src = normed.data.slice(0..rows);
                let mut dst = cache[layer_idx].data.slice_mut(0..rows);
                self.ctx
                    .stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("DSv4 commit-fold normed persist failed: {e}"))?;
            }
            // SAFETY: mla_attention writes the full [seq_len, hidden_size] buffer.
            let mut attn_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if let Some(meta) = sparse_verify_meta.as_ref() {
                // Scheduled verify: one sparse FlashMLA attention over the
                // whole row chunk. Ancestors carry row `r`'s chain dependency,
                // so no row replay or slot-ring mutation is needed; commit
                // writes accepted rows later.
                let _nvtx = crate::nvtx::range("dsv4/mla_attn_sparse_verify");
                let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
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
                    flashmla_scratch,
                    prefill_shared,
                    fp32,
                    start_pos,
                    None,
                    Some(meta),
                    &self.tp,
                    &mut attn_out,
                    &mut keepalive,
                )?;
            } else {
                let _nvtx = crate::nvtx::range("dsv4/mla_attn");
                crate::stage_profile::profile(ctx, "dsv4/stage/mla_attn", || {
                    let model1_decode = seq_len == 1
                        && start_pos_device.is_some()
                        && layer.attention.w_kc.is_none()
                        && layer.attention.w_vc.is_none()
                        && layer.attention.o_proj.is_none();
                    if model1_decode {
                        let (layer_pool, dsa_shared, flashmla_scratch, mla_scratch) =
                            kv_adapter.layer_flashmla_and_mla_decode_mut(layer_idx)?;
                        let mla_scratch = mla_scratch.ok_or_else(|| {
                            anyhow!("DSv4 MODEL1 layer {layer_idx} missing MLA decode scratch")
                        })?;
                        crate::attention::mla_attention_decode_graph(
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
                            flashmla_scratch,
                            start_pos,
                            start_pos_device,
                            &self.tp,
                            mla_scratch,
                            &mut attn_out,
                            &mut keepalive,
                        )
                    } else {
                        let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
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
                            flashmla_scratch,
                            prefill_shared,
                            fp32,
                            start_pos,
                            start_pos_device,
                            None,
                            &self.tp,
                            &mut attn_out,
                            &mut keepalive,
                        )
                    }
                })?;
            }
            keepalive.keep_hidden(&attn_out);
            // Row-parallel O-LoRA: sum the per-rank partials (no-op single-GPU).
            {
                let _nvtx = crate::nvtx::range("dsv4/attn_allreduce");
                crate::stage_profile::profile(ctx, "dsv4/stage/attn_allreduce", || {
                    self.tp.all_reduce_sum(&self.ctx, &mut attn_out)
                })?;
            }
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut attn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            if let Some(mhc) = attn_mhc.as_ref() {
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
            } else {
                // GLM plain residual: stream = attn_out + stream (stream_dim==hidden).
                crate::stage_profile::profile(ctx, "dsv4/stage/attn_hc_post", || {
                    crate::ops::add_batch(&self.ctx, &attn_out, &stream, &mut attn_stream)
                })?;
            }
            keepalive.keep_hidden(&attn_stream);
            stream = attn_stream;

            // ── MoE half: HC-wrap the FP8 DeepGEMM MoE block.
            // GLM (hc_mult==1): plain RMSNorm of the stream (no ffn hyper-connection).
            // SAFETY: fused hc_pre+rms_norm / plain rms_norm writes the full
            // [seq_len, hidden_size] buffer.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            let ffn_mhc = if self.config.is_glm() {
                crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_pre_norm", || {
                    crate::ops::rms_norm_batch(
                        &self.ctx,
                        &stream,
                        &layer.ffn_norm,
                        eps,
                        &mut normed,
                    )
                })?;
                None
            } else {
                let mhc = crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_params", || {
                    crate::hc::gen_mhc_params(&self.ctx, &self.config, &layer.hc_ffn, &stream)
                })?;
                keepalive.keep_f32(&mhc.pre);
                keepalive.keep_f32(&mhc.post);
                keepalive.keep_f32(&mhc.comb);
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
                Some(mhc)
            };
            keepalive.keep_hidden(&normed);
            // GLM dense layer (`per_layer_dense_mlp[i]`): a plain SwiGLU FFN
            // replaces the routed-expert + shared-expert MoE entirely.
            let mut moe_with_shared =
                // SAFETY: uninit device scratch; fully written before first read.
                unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if let Some(dense) = layer.dense_mlp.as_ref() {
                dsv4_dense_mlp_forward(
                    &self.ctx,
                    dense,
                    &normed,
                    &mut moe_with_shared,
                    self.config.swiglu_limit,
                    &mut keepalive,
                )?;
                keepalive.keep_hidden(&moe_with_shared);
            } else {
                let use_comm_overlap = seq_len == 1 && !use_deepep_transport;
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
                let (mut shared_out, mut shared_scratch) = kv_adapter.shared_expert_decode_mut();
                let needs_moe_allreduce = if use_deepep_transport {
                    #[cfg(feature = "deepep")]
                    {
                        let transport = self.deepep.as_ref().ok_or_else(|| {
                            anyhow!(
                                "ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted"
                            )
                        })?;
                        if transport.is_low_latency() {
                            // ── deepep_ll token-owned path ──────────────────────
                            // TP8 replicates `normed` [hidden, n]; HiddenStates is
                            // token-major (token i at i*hidden), so this rank's owned
                            // shard cols [start..end] is a CONTIGUOUS byte range. The
                            // LL dispatch + combine are COLLECTIVES: EVERY rank must
                            // participate every step or the symmetric protocol
                            // deadlocks (DeepEP "timeout for dispatch receive") — the
                            // shared helper always runs `compute_fn`, even at
                            // owned_n == 0. This REPLACES the moe all-reduce
                            // (needs_moe_allreduce=false).
                            let scratch = slot.deepep_ll_scratch.as_mut().ok_or_else(|| {
                                anyhow!("deepep_ll selected but slot LL scratch not allocated")
                            })?;
                            self.tp.shard_rows_and_allreduce(
                                &self.ctx,
                                &normed,
                                &mut moe_out,
                                hidden_size,
                                seq_len,
                                |owned_in, owned_out, start, owned_n| {
                                    keepalive.keep_hidden(owned_in);
                                    keepalive.keep_hidden(owned_out);
                                    crate::moe::dsv4_moe_forward_deepep_ll(
                                        self,
                                        transport,
                                        scratch,
                                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                                        &tokens[start..start + owned_n],
                                        tokens.len(),
                                        owned_in,
                                        owned_out,
                                        &mut keepalive,
                                    )
                                },
                            )?;
                        } else {
                            crate::moe::dsv4_moe_forward_deepep(
                                self,
                                transport,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                tokens,
                                &normed,
                                &mut moe_out,
                                &mut keepalive,
                            )?;
                        }
                        false
                    }
                    #[cfg(not(feature = "deepep"))]
                    {
                        anyhow::bail!(
                            "ARLE_DSV4_MOE_TRANSPORT=deepep requires infer-cuda feature deepep"
                        );
                    }
                } else {
                    crate::moe::dsv4_moe_forward(
                        self,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        tokens,
                        &normed,
                        &mut moe_out,
                        &mut keepalive,
                        None,
                        mega_epoch,
                    )?
                };
                keepalive.keep_hidden(&moe_out);
                let _nvtx_shared_hc = crate::nvtx::range("dsv4/shared_hc");
                let mut shared_owned = None;
                let shared_ready = if use_comm_overlap {
                    let shared = shared_out.as_deref_mut().ok_or_else(|| {
                        anyhow!("DSv4 decode requires shared-expert output buffer")
                    })?;
                    shared.seq_len = seq_len;
                    ensure!(
                        shared.hidden_dim == hidden_size && shared.seq_len == seq_len,
                        "DSv4 shared decode scratch shape {}x{} != {}x{}",
                        shared.hidden_dim,
                        shared.seq_len,
                        hidden_size,
                        seq_len
                    );
                    let _normed_ready = normed_ready
                        .as_ref()
                        .expect("comm-overlap path records normed fence");
                    let scratch = shared_scratch
                        .as_deref_mut()
                        .ok_or_else(|| anyhow!("DSv4 decode requires shared-expert scratch"))?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                        crate::moe::dsv4_shared_expert_forward_decode_scratch(
                            &self.ctx,
                            &self.ctx.comm_stream,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            &normed,
                            shared,
                            self.config.swiglu_limit,
                            scratch,
                        )
                    })?;
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
                if use_comm_overlap {
                    // Already launched on the comm stream above.
                } else if seq_len == 1 {
                    let shared = shared_out.as_deref_mut().ok_or_else(|| {
                        anyhow!("DSv4 decode requires shared-expert output buffer")
                    })?;
                    shared.seq_len = seq_len;
                    ensure!(
                        shared.hidden_dim == hidden_size && shared.seq_len == seq_len,
                        "DSv4 shared decode scratch shape {}x{} != {}x{}",
                        shared.hidden_dim,
                        shared.seq_len,
                        hidden_size,
                        seq_len
                    );
                    let scratch = shared_scratch
                        .as_deref_mut()
                        .ok_or_else(|| anyhow!("DSv4 decode requires shared-expert scratch"))?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                        crate::moe::dsv4_shared_expert_forward_decode_scratch(
                            &self.ctx,
                            &self.ctx.stream,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            &normed,
                            shared,
                            self.config.swiglu_limit,
                            scratch,
                        )
                    })?;
                } else if !use_deepep_transport
                    && sparse_verify_meta.is_some()
                    && seq_len <= MAX_SPEC_VERIFY_ROWS
                {
                    let shared = shared_out.as_deref_mut().ok_or_else(|| {
                        anyhow!("DSv4 verify requires shared-expert output buffer")
                    })?;
                    shared.seq_len = seq_len;
                    ensure!(
                        shared.hidden_dim == hidden_size,
                        "DSv4 shared verify scratch hidden {} != {}",
                        shared.hidden_dim,
                        hidden_size
                    );
                    let scratch = shared_scratch
                        .ok_or_else(|| anyhow!("DSv4 verify requires shared-expert scratch"))?;
                    crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                        crate::moe::dsv4_shared_expert_forward_decode_scratch(
                            &self.ctx,
                            &self.ctx.stream,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            &normed,
                            shared,
                            self.config.swiglu_limit,
                            scratch,
                        )
                    })?;
                } else {
                    // Keep the default masked path order byte-for-byte with main:
                    // routed MoE → all-reduce → allocate/run shared expert. Moving
                    // shared scratch allocation before all-reduce can reuse in-flight
                    // helper buffers under disabled cudarc event tracking.
                    // SAFETY: dsv4_shared_expert_forward writes the full shared output.
                    let mut shared =
                        unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                    crate::stage_profile::profile(ctx, "dsv4/stage/shared_expert", || {
                        crate::moe::dsv4_shared_expert_forward(
                            &self.ctx,
                            &self.ctx.stream,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            &normed,
                            &mut shared,
                            self.config.swiglu_limit,
                            &mut keepalive,
                        )
                    })?;
                    shared_owned = Some(shared);
                };
                let shared = if seq_len == 1
                    || (!use_deepep_transport
                        && sparse_verify_meta.is_some()
                        && seq_len <= MAX_SPEC_VERIFY_ROWS)
                {
                    shared_out.as_deref().ok_or_else(|| {
                        anyhow!("DSv4 decode requires shared-expert output buffer")
                    })?
                } else {
                    shared_owned
                        .as_ref()
                        .expect("multi-token shared expert allocates owned output")
                };
                keepalive.keep_hidden(shared);
                if let Some(fence) = shared_ready.as_ref() {
                    ctx.wait_on_pipeline_fence(fence, CudaPipelineStreamKind::Compute)?;
                }
                // SAFETY: add_batch writes the full [seq_len, hidden_size] buffer.
                crate::stage_profile::profile(ctx, "dsv4/stage/shared_add", || {
                    crate::ops::add_batch(&self.ctx, &moe_out, shared, &mut moe_with_shared)
                })?;
                keepalive.keep_hidden(&moe_with_shared);
            }
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut ffn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            if let Some(mhc) = ffn_mhc.as_ref() {
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
            } else {
                // GLM plain residual: stream = ffn_out + stream (stream_dim==hidden).
                crate::stage_profile::profile(ctx, "dsv4/stage/ffn_hc_post", || {
                    crate::ops::add_batch(&self.ctx, &moe_with_shared, &stream, &mut ffn_stream)
                })?;
            }
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
            // DSpark T3: capture the wide HC stream at this layer's OUTPUT
            // (hidden_states[layer_idx+1]) for the row being predicted, when this
            // layer is a DSpark target. D2D copy only (graph-safe, no readback);
            // gated by `dspark` so the default path is byte-identical.
            if dspark
                && let Some(tap_idx) = self
                    .config
                    .dspark_target_layer_ids
                    .iter()
                    .position(|&l| l == layer_idx)
            {
                let tap = &mut slot.dspark_taps[tap_idx];
                let elems = stream_dim * seq_len;
                if tap.len >= elems {
                    let src = stream.data.slice(0..elems);
                    let mut dst = tap.data.slice_mut(0..elems);
                    ctx.stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSpark tap capture D2D failed: {e}"))?;
                    keepalive.keep_vec(tap);
                } else {
                    self.capture_mtp_stream_hidden(&stream, seq_len - 1, 1, tap, &mut keepalive)?;
                }
                // Prefill: also capture the FULL stream (all rows) for the seed.
                if let Some(bufs) = dspark_prompt_bufs.as_mut() {
                    ctx.stream
                        .memcpy_dtod(&stream.data, &mut bufs[tap_idx].data)
                        .map_err(|e| anyhow!("DSpark prompt tap capture D2D failed: {e}"))?;
                    keepalive.keep_vec(&bufs[tap_idx]);
                }
            }
            // Lens capture: head-project the layer's final stream (row 0, the
            // decode token) and stash the DEVICE logits (no sync/D2H until
            // lens_flush at the post-sampling sync point).
            if lens_from.is_some_and(|from| layer_idx >= from) {
                let logits = self.verify_logits_from_stream(&stream, seq_len, &mut keepalive)?;
                crate::probe::lens_push(layer_idx, logits);
            }
        }

        if let Some(bufs) = dspark_prompt_bufs.take() {
            slot.dspark_prompt_taps = Some(Dsv4DsparkPromptTaps {
                bufs,
                rows: seq_len,
            });
        }

        slot.seq_len += seq_len;
        Ok((stream, keepalive))
    }

    /// Draft one MTP matrix level. `rows` is the draft batch for this level
    /// (single-slot path uses `m == 1`; cross-slot batched draft uses one row per
    /// slot). The point-wise pipeline batches over `m`, target-layer attention
    /// runs per row at each row's `position`, and candidates come from k rounds
    /// of device argmax + mask — no full-vocab D2H.
    ///
    /// `slot_ids[r]` selects the KV cache slot for row `r`; `positions[r]` is
    /// that row's draft position. Returns per row: top-k candidate tokens
    /// (highest first) + the wide MTP stream the next draft row branches from.
    pub(crate) fn mtp_forward_level(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        rows: &[MtpDraftRow],
        h_prevs: &[&DeviceVec],
        positions: &[u64],
        top_k: usize,
    ) -> Result<Vec<(Vec<u32>, DeviceVec)>> {
        ensure!(
            self.spec_decode_on,
            "DSv4 MTP forward called while spec decode is off (need --spec-type mtp / \
             --mtp-draft-tokens)"
        );
        ensure!(
            !crate::runtime_flags::dsv4_moe_transport()?.is_deepep(),
            "DSv4 MTP Phase 1 supports allreduce transport only"
        );
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP requested but the draft head is not loaded"))?;
        let m = rows.len();
        ensure!(
            m > 0 && h_prevs.len() == m && slot_ids.len() == m && positions.len() == m,
            "DSv4 MTP level shape mismatch (rows {m}, h_prevs {}, slot_ids {}, positions {})",
            h_prevs.len(),
            slot_ids.len(),
            positions.len()
        );
        let mega_epoch = self.begin_mega_moe_forward(m)?;
        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        for h in h_prevs {
            ensure!(
                h.len == stream_dim,
                "DSv4 MTP h_prev len {} != stream dim {stream_dim}",
                h.len
            );
        }
        let eps = self.config.rms_norm_eps;
        let ctx = &self.ctx;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);

        // ── h' = e_proj(enorm(emb(token))) + h_proj(hnorm(h_prev)), batched.
        let token_ids_host: Vec<i32> = rows.iter().map(|r| r.token as i32).collect();
        let token_ids = crate::ops::upload_i32(ctx, &token_ids_host)?;
        // SAFETY: embedding_batch writes the full [m, hidden_size] buffer.
        let mut emb = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::ops::embedding_batch(ctx, &self.embed_tokens, &token_ids, &mut emb)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut emb_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::ops::rms_norm_batch(ctx, &emb, &mtp.enorm, eps, &mut emb_normed)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut e_proj = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::attention::dsv4_linear(ctx, &mtp.e_proj, &emb_normed, &mut e_proj)?;

        // Gather h_prev streams into [m * hc_mult, hidden] (a stream is
        // hc_mult consecutive hidden rows, token-major).
        // SAFETY: uninit device scratch; fully written before first read.
        let mut h_prev_batch = unsafe { HiddenStates::uninit(ctx, hidden_size, m * hc_mult)? };
        for (r, h) in h_prevs.iter().enumerate() {
            let mut dst = h_prev_batch
                .data
                .slice_mut(r * stream_dim..(r + 1) * stream_dim);
            ctx.stream
                .memcpy_dtod(&h.data, &mut dst)
                .map_err(|e| anyhow!("DSv4 MTP h_prev D2D gather failed: {e}"))?;
        }
        // SAFETY: uninit device scratch; fully written before first read.
        let mut h_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m * hc_mult)? };
        crate::ops::rms_norm_batch(ctx, &h_prev_batch, &mtp.hnorm, eps, &mut h_normed)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut h_proj = unsafe { HiddenStates::uninit(ctx, hidden_size, m * hc_mult)? };
        crate::attention::dsv4_linear(ctx, &mtp.h_proj, &h_normed, &mut h_proj)?;

        // SAFETY: uninit device scratch; fully written before first read.
        let mut stream = unsafe { HiddenStates::uninit(ctx, stream_dim, m)? };
        {
            let (e_ptr, _ge) = e_proj.data.device_ptr(&ctx.stream);
            let (h_ptr, _gh) = h_proj.data.device_ptr(&ctx.stream);
            let (out_ptr, _go) = stream.data.device_ptr_mut(&ctx.stream);
            let row_h = (hidden_size * 2) as u64;
            let row_s = (stream_dim * 2) as u64;
            for r in 0..m as u64 {
                // SAFETY: per-row slices of buffers sized above.
                unsafe {
                    ffi::dsv4_mtp_add_eproj_hproj_cuda(
                        (e_ptr + r * row_h) as *const ffi::Half,
                        (h_ptr + r * row_s) as *const ffi::Half,
                        (out_ptr + r * row_s) as *mut ffi::Half,
                        hidden_size as i32,
                        hc_mult as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
        keepalive.keep_hidden(&e_proj);
        keepalive.keep_hidden(&h_proj);
        keepalive.keep_hidden(&stream);

        // ── ONE MTP transformer layer over the batch; attention per row at the
        // draft position. Top-k widens this draft readout; the caller decides
        // which candidate rows are scheduled for target verify.
        let layer = &mtp.layer;
        let target_layer_idx = self.mtp_frozen_target_layer_idx(mtp)?;
        for &sid in slot_ids {
            ensure!(
                target_layer_idx < slots[sid].attention.len(),
                "DSv4 MTP frozen-KV target layer {target_layer_idx} outside slot {sid} attention len {}",
                slots[sid].attention.len()
            );
        }
        let local_width = layer.attention.wq_b.rows;
        ensure!(
            local_width.is_multiple_of(self.config.head_dim),
            "DSv4 MTP attention local width {local_width} is not a multiple of head_dim {}",
            self.config.head_dim
        );

        let attn_mhc = crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_attn, &stream)?;
        // SAFETY: scratch fully written before read.
        let mut attn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
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
        keepalive.keep_hidden(&attn_normed);
        // SAFETY: scratch fully written before read.
        let mut attn_out = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        {
            // SAFETY: scratch fully written before read.
            let mut normed_row = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
            // SAFETY: scratch fully written before read.
            let mut attn_row = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
            keepalive.keep_hidden(&normed_row);
            keepalive.keep_hidden(&attn_row);
            let (layer_pool, mut dsa_shared, mut flashmla_scratch, mut prefill_shared, _fp32) =
                kv_adapter.layer_and_dsa_shared_mut(target_layer_idx)?;
            for r in 0..rows.len() {
                let src = attn_normed
                    .data
                    .slice(r * hidden_size..(r + 1) * hidden_size);
                ctx.stream
                    .memcpy_dtod(&src, &mut normed_row.data)
                    .map_err(|e| anyhow!("DSv4 MTP attn copy-in failed: {e}"))?;
                let pos_dev = ctx
                    .stream
                    .clone_htod(&[positions[r] as i32])
                    .map_err(|e| anyhow!("DSv4 MTP start_pos H2D failed: {e}"))?;
                crate::attention::mla_attention(
                    ctx,
                    &self.config,
                    &layer.attention,
                    layer.mode,
                    layer.compress_ratio,
                    target_layer_idx,
                    &normed_row,
                    &mut slots[slot_ids[r]].attention[target_layer_idx],
                    layer_pool,
                    dsa_shared.as_deref_mut(),
                    flashmla_scratch.as_deref_mut(),
                    prefill_shared.as_deref_mut(),
                    None,
                    positions[r] as usize,
                    Some(&pos_dev),
                    None,
                    &self.tp,
                    &mut attn_row,
                    &mut keepalive,
                )?;
                let mut dst = attn_out
                    .data
                    .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                ctx.stream
                    .memcpy_dtod(&attn_row.data, &mut dst)
                    .map_err(|e| anyhow!("DSv4 MTP attn copy-out failed: {e}"))?;
            }
        }
        self.tp.all_reduce_sum(ctx, &mut attn_out)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, m)? };
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
        keepalive.keep_hidden(&attn_out);
        keepalive.keep_hidden(&attn_stream);

        let ffn_mhc = crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_ffn, &attn_stream)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut ffn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
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
        keepalive.keep_hidden(&ffn_normed);
        let level_tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        // SAFETY: uninit device scratch; fully written before first read.
        let mut moe_out = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        let needs_moe_allreduce = crate::moe::dsv4_moe_forward(
            self,
            layer.moe.as_ref().expect("DSv4 layer.moe"),
            &level_tokens,
            &ffn_normed,
            &mut moe_out,
            &mut keepalive,
            None,
            mega_epoch,
        )?;
        if needs_moe_allreduce {
            self.tp.all_reduce_sum(ctx, &mut moe_out)?;
        }
        // SAFETY: uninit device scratch; fully written before first read.
        let mut shared = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::moe::dsv4_shared_expert_forward(
            ctx,
            &ctx.stream,
            layer.moe.as_ref().expect("DSv4 layer.moe"),
            &ffn_normed,
            &mut shared,
            self.config.swiglu_limit,
            &mut keepalive,
        )?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut moe_with_shared = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::ops::add_batch(ctx, &moe_out, &shared, &mut moe_with_shared)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut ffn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, m)? };
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
        keepalive.keep_hidden(&moe_out);
        keepalive.keep_hidden(&shared);
        keepalive.keep_hidden(&moe_with_shared);
        keepalive.keep_hidden(&ffn_stream);

        // ── Head: per-row HC fold + norm, batched lm_head, device top-k.
        // SAFETY: uninit device scratch; fully written before first read.
        let mut head_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        {
            let mut last_hidden = DeviceVec::zeros(ctx, hidden_size)?;
            let mut last_normed = DeviceVec::zeros(ctx, hidden_size)?;
            for r in 0..m {
                crate::hc::head_hidden_from_stream(
                    ctx,
                    &self.config,
                    &mtp.head_hc,
                    &ffn_stream,
                    r,
                    &mut last_hidden,
                )?;
                crate::ops::rms_norm_vec(ctx, &last_hidden, &mtp.norm, eps, &mut last_normed)?;
                let mut dst = head_normed
                    .data
                    .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                ctx.stream
                    .memcpy_dtod(&last_normed.data, &mut dst)
                    .map_err(|e| anyhow!("DSv4 MTP head row copy failed: {e}"))?;
            }
        }
        keepalive.keep_hidden(&head_normed);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut logits = unsafe { HiddenStates::uninit(ctx, self.lm_head.rows, m)? };
        self.lm_head_project_batch(&head_normed, &mut logits)?;
        keepalive.keep_hidden(&logits);
        let candidates = self.mtp_topk_device(&mut logits, top_k.max(1))?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);

        // Split the level stream into per-row owned vecs (next draft rows
        // branch from their own parent row only).
        let mut out = Vec::with_capacity(m);
        for (r, cand) in candidates.into_iter().enumerate() {
            let mut row_stream = DeviceVec::zeros(ctx, stream_dim)?;
            let src = ffn_stream.data.slice(r * stream_dim..(r + 1) * stream_dim);
            ctx.stream
                .memcpy_dtod(&src, &mut row_stream.data)
                .map_err(|e| anyhow!("DSv4 MTP stream row split failed: {e}"))?;
            out.push((cand, row_stream));
        }
        Ok(out)
    }

    /// Batched lm_head: `[m, hidden] → [m, vocab]` — ONE GEMM for every weight
    /// format (`dsv4_linear`: bf16 → cuBLAS gemm_batch, FP8/FP4 → mla_linear),
    /// weight read once for all `m` rows.
    fn lm_head_project_batch(&self, x: &HiddenStates, out: &mut HiddenStates) -> Result<()> {
        // The batched consumers (MTP verify/draft heads, entropy probes) need
        // full-vocab logits per row; the vocab-sharded lm_head only holds this
        // rank's slice. The load-time gates make this unreachable in serving —
        // keep it loud for probe/selftest paths.
        ensure!(
            self.lm_head_shard.is_none(),
            "DSv4 batched lm_head consumers (MTP verify/draft, entropy probes) are \
             not supported with ARLE_DSV4_LM_HEAD_SHARD=1"
        );
        ensure!(
            self.lm_head.cols == x.hidden_dim
                && self.lm_head.rows == out.hidden_dim
                && x.seq_len == out.seq_len,
            "DSv4 lm_head batch shape mismatch: [{}x{}] x {}x{} out {}x{}",
            self.lm_head.rows,
            self.lm_head.cols,
            x.hidden_dim,
            x.seq_len,
            out.hidden_dim,
            out.seq_len
        );
        // ONE batched GEMM for every weight format (bf16 → cuBLAS gemm_batch,
        // FP8/FP4 → mla_linear) — the lm_head weight is read ONCE for all `m` rows.
        // The old bf16 path looped a per-row GEMV, re-reading the whole ~1 GB bf16
        // lm_head weight PER ROW — at a c=16×depth verify that was ~48× the HBM
        // traffic and the #1 decode GPU kernel (nsys: gemv_handwritten 19%, 558µs).
        crate::attention::dsv4_linear(&self.ctx, &self.lm_head, x, out)
    }

    /// Batched device argmax over `[m, vocab]` verifier logits — one launch,
    /// one D2H of m target top-1 ids.
    fn mtp_argmax_batch(&self, logits: &HiddenStates) -> Result<Vec<u32>> {
        let ctx = &self.ctx;
        let m = logits.seq_len;
        let vocab = logits.hidden_dim;
        let mut ids_dev = ctx
            .stream
            .alloc_zeros::<i32>(m)
            .map_err(|e| anyhow!("DSv4 MTP argmax ids alloc failed: {e}"))?;
        {
            let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
            let (ids_ptr, _ig) = ids_dev.device_ptr_mut(&ctx.stream);
            // SAFETY: logits [m, vocab] and ids [m] sized above.
            unsafe {
                ffi::argmax_batch_cuda(
                    logits_ptr as *const ffi::Half,
                    ids_ptr as *mut i32,
                    m as i32,
                    vocab as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        let ids: Vec<i32> = ctx
            .stream
            .clone_dtoh(&ids_dev)
            .map_err(|e| anyhow!("DSv4 MTP argmax D2H failed: {e}"))?;
        ids.into_iter()
            .map(|id| {
                ensure!(
                    (0..vocab as i32).contains(&id),
                    "DSv4 MTP argmax id {id} out of vocab {vocab}"
                );
                Ok(id as u32)
            })
            .collect()
    }

    /// Batched device top-k over `[m, vocab]` logits. Highest-first per row.
    /// Mutates the logits scratch by masking each selected id to `-inf`; callers
    /// only use draft logits for candidate extraction.
    fn mtp_topk_device(&self, logits: &mut HiddenStates, k: usize) -> Result<Vec<Vec<u32>>> {
        let ctx = &self.ctx;
        let m = logits.seq_len;
        let vocab = logits.hidden_dim;
        ensure!(
            k >= 1 && k < vocab,
            "DSv4 MTP top-k {k} out of range for vocab {vocab}"
        );
        let mut ids_dev = ctx
            .stream
            .alloc_zeros::<i32>(m)
            .map_err(|e| anyhow!("DSv4 MTP top-k ids alloc failed: {e}"))?;
        let mut out = vec![Vec::with_capacity(k); m];
        for round in 0..k {
            {
                let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
                let (ids_ptr, _ig) = ids_dev.device_ptr_mut(&ctx.stream);
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::argmax_batch_cuda(
                        logits_ptr as *const ffi::Half,
                        ids_ptr as *mut i32,
                        m as i32,
                        vocab as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
            let ids: Vec<i32> = ctx
                .stream
                .clone_dtoh(&ids_dev)
                .map_err(|e| anyhow!("DSv4 MTP top-k D2H failed: {e}"))?;
            for (r, &id) in ids.iter().enumerate() {
                ensure!(
                    (0..vocab as i32).contains(&id),
                    "DSv4 MTP top-k id {id} out of vocab {vocab}"
                );
                out[r].push(id as u32);
            }
            if round + 1 < k {
                for (r, &id) in ids.iter().enumerate() {
                    let offset = r * vocab + id as usize;
                    let mut dst = logits.data.slice_mut(offset..offset + 1);
                    ctx.stream
                        .memcpy_htod(&[half::bf16::NEG_INFINITY], &mut dst)
                        .map_err(|e| anyhow!("DSv4 MTP top-k mask failed: {e}"))?;
                }
            }
        }
        Ok(out)
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

    fn forward_tokens_decode_graph(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        // ponytail: pod-verify GLM decode-graph path (not wired; eager path is the GLM lane)
        if self.config.is_glm() {
            anyhow::bail!(
                "DSv4 decode-graph path does not support GLM (hc_mult==1); use the eager decode path"
            );
        }
        ensure!(
            slot.seq_len == start_pos,
            "DSv4 graph decode slot seq_len {} != start_pos {start_pos}",
            slot.seq_len
        );
        let next_len = start_pos
            .checked_add(1)
            .ok_or_else(|| anyhow!("DSv4 graph decode start_pos overflow"))?;
        ensure!(
            start_pos < slot.max_seq_len,
            "DSv4 graph decode sequence {} exceeds slot max_seq_len {}",
            next_len,
            slot.max_seq_len
        );
        ensure!(
            matches!(
                crate::runtime_flags::dsv4_moe_transport()?,
                crate::runtime_flags::Dsv4MoeTransport::AllReduce
            ),
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
        ensure!(
            !(whole_step && self.tp.oneshot_comm_active()),
            "DSv4 whole-step CUDA graph cannot capture ARLE one-shot CustomAllreduce; \
             set ARLE_COMM_BACKEND=nccl or disable ARLE_DSV4_WHOLE_STEP_GRAPH"
        );
        if whole_step {
            for s in graph.layers.iter_mut() {
                s.attn_graph.set_bypass(true);
                s.moe_graph.set_bypass(true);
            }
            graph.tail_graph.set_bypass(true);
        }
        // Graph-safe CSA-read lane (opt-in): the READ is graph-capturable but the
        // cache WRITE (block (a)) is not yet device-shape driven, so run the attn
        // portion EAGER (bypass capture) to exercise + needle-verify the shape-
        // stable read without baking the variable-shape cache write. The MoE
        // portion still captures. When OFF (default), capture as before.
        let csa_read_lane = crate::attention::dsv4_decode_graph_csa_read_enabled()?;
        if csa_read_lane && !whole_step {
            for s in graph.layers.iter_mut() {
                s.attn_graph.set_bypass(true);
            }
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
                        attn_in: _,
                        attn_normed,
                        attn_out,
                        mla,
                        ..
                    } = current;
                    let attn_state = &mut slot.attention[layer_idx];
                    let (attn_pool, dsa_shared_raw, flashmla_scratch, _prefill, _fp32) =
                        kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                    // Default decode stays byte-identical (None) unless the CSA read-lane is
                    // on; the unconditional dsa_shared swap hung eager c=1 decode.
                    let dsa_shared = dsa_shared_raw.filter(|_| csa_read_lane);
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
                        crate::hc::gen_mhc_params_pre_rms_norm_into(
                            &self.ctx,
                            &self.config,
                            &layer.hc_attn,
                            &graph.initial_stream,
                            &layer.attn_norm,
                            eps,
                            hidden_size,
                            attn_mhc,
                            attn_normed,
                        )?;
                        crate::attention::mla_attention_decode_graph(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            attn_normed,
                            attn_state,
                            attn_pool,
                            dsa_shared,
                            flashmla_scratch,
                            start_pos,
                            Some(&slot.start_pos_device),
                            &self.tp,
                            mla,
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
                        attn_in: _,
                        attn_normed,
                        attn_out,
                        mla,
                        ..
                    } = current;
                    let attn_state = &mut slot.attention[layer_idx];
                    let (attn_pool, dsa_shared_raw, flashmla_scratch, _prefill, _fp32) =
                        kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                    // Default decode stays byte-identical (None) unless the CSA read-lane is
                    // on; the unconditional dsa_shared swap hung eager c=1 decode.
                    let dsa_shared = dsa_shared_raw.filter(|_| csa_read_lane);
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
                        crate::hc::gen_mhc_params_pre_rms_norm_into(
                            &self.ctx,
                            &self.config,
                            &layer.hc_attn,
                            &prev.ffn_stream,
                            &layer.attn_norm,
                            eps,
                            hidden_size,
                            attn_mhc,
                            attn_normed,
                        )?;
                        crate::attention::mla_attention_decode_graph(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            attn_normed,
                            attn_state,
                            attn_pool,
                            dsa_shared,
                            flashmla_scratch,
                            start_pos,
                            Some(&slot.start_pos_device),
                            &self.tp,
                            mla,
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
                    ffn_in: _,
                    ffn_normed,
                    moe_out,
                    shared,
                    ..
                } = layer_scratch;
                let moe_scratch = kv_adapter.moe_decode_graph_scratch_mut().ok_or_else(|| {
                    anyhow!("DSv4 decode graph requires shared MoE decode scratch")
                })?;
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
                    crate::hc::gen_mhc_params_pre_rms_norm_into(
                        &self.ctx,
                        &self.config,
                        &layer.hc_ffn,
                        attn_stream,
                        &layer.ffn_norm,
                        eps,
                        hidden_size,
                        ffn_mhc,
                        ffn_normed,
                    )?;
                    crate::moe::dsv4_moe_forward_decode_graph(
                        self,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        &graph.token_ids_u32,
                        ffn_normed,
                        moe_out,
                        moe_scratch,
                    )?;
                    crate::moe::dsv4_shared_expert_forward_decode_graph(
                        &self.ctx,
                        &self.ctx.stream,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        ffn_normed,
                        shared,
                        self.config.swiglu_limit,
                        moe_scratch,
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
                let head_stream_row_ref = &mut graph.head_stream_row;
                let head_mixes_ref = &mut graph.head_mixes;
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
                    crate::hc::head_hidden_from_stream_into(
                        &self.ctx,
                        &self.config,
                        &self.head_hc,
                        &final_scratch.ffn_stream,
                        0,
                        head_stream_row_ref,
                        head_mixes_ref,
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
        self.sample_logits(&graph.logits, params, position)
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

/// GLM dense layer (`config.per_layer_dense_mlp[i]`) forward: a plain SwiGLU
/// FFN that replaces the routed-expert + shared-expert MoE. bf16 throughout
/// (gate/up/down are `DeviceMatrix` DenseBf16). DSv4 never has `dense_mlp`, so
/// this is GLM-only.
///
///   gate = down·SiLU(gate_w·x) ⊙ (up_w·x)
///
/// `out` must be `[hidden, tok]` (== `x` hidden). Writes the FFN output for the
/// MoE half; the caller folds it into the residual.
fn dsv4_dense_mlp_forward(
    ctx: &DeviceContext,
    dense: &Dsv4DenseMlp,
    x: &HiddenStates,
    out: &mut HiddenStates,
    swiglu_limit: f32,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        x.hidden_dim == dense.hidden_dim
            && out.hidden_dim == dense.hidden_dim
            && x.seq_len == out.seq_len,
        "GLM dense MLP shape mismatch: x={}x{} out={}x{} hidden={}",
        x.hidden_dim,
        x.seq_len,
        out.hidden_dim,
        out.seq_len,
        dense.hidden_dim
    );
    let tok = x.seq_len;
    let inter = dense.intermediate;
    // gate = gate_w · x  → [intermediate, tok]
    // SAFETY: uninit device scratch; fully written before first read.
    let mut gate = unsafe { HiddenStates::uninit(ctx, inter, tok)? };
    crate::attention::dsv4_linear(ctx, &dense.gate, x, &mut gate)?;
    keepalive.keep_hidden(&gate);
    // up = up_w · x  → [intermediate, tok]
    // SAFETY: uninit device scratch; fully written before first read.
    let mut up = unsafe { HiddenStates::uninit(ctx, inter, tok)? };
    crate::attention::dsv4_linear(ctx, &dense.up, x, &mut up)?;
    keepalive.keep_hidden(&up);
    // act = SiLU(gate) ⊙ up. GLM dense uses plain SiLU(gate)*up with NO clamp
    // (into_deepseek_v4 sets swiglu_limit=0.0, which the clamped kernel rejects),
    // so use the unclamped silu_mul.
    // ponytail: pod-verify GLM dense FFN activation = silu(gate)*up (unclamped)
    let _ = swiglu_limit;
    // SAFETY: uninit device scratch; fully written before first read.
    let mut act = unsafe { HiddenStates::uninit(ctx, inter, tok)? };
    crate::ops::silu_mul(ctx, &gate, &up, &mut act)?;
    keepalive.keep_hidden(&act);
    // out = down_w · act  → [hidden, tok]
    crate::attention::dsv4_linear(ctx, &dense.down, &act, out)?;
    Ok(())
}

fn dsv4_decode_graph_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

/// Opt-in decode lever #99 (H3):
/// row-shard lm_head across TP ranks (replicated full-vocab GEMV → vocab/N rows
/// per rank). Greedy sampling merges per-rank (max, argmax) via one 8-byte host
/// all-gather; non-greedy all-gathers the bf16 logit slices to full vocab first.
/// OFF by default; perf license pending pod A/B + needle gate. TP=1 no-ops.
fn dsv4_lm_head_shard_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_LM_HEAD_SHARD").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

/// Whole-step decode CUDA graph: capture the ENTIRE per-token forward (all layers
/// and the 86 all-reduces + tail) as ONE graph, replayed with ~0 host orchestration.
///
/// Requires ARLE_DSV4_DECODE_GRAPH=1 (reuses its pre-allocated scratch).
///
/// STATUS (2026-06-08): VALIDATED-CORRECT but WALL-NEUTRAL — kept as gated infra, not
/// default. The 86-NCCL-all-reduces-in-one-graph capture WORKS and is byte-identical,
/// but the A/B is wall-neutral (eager 30.38 / per-portion 30.13 / whole-step 30.32 tok/s):
/// removing ALL host orchestration moves the wall 0%, which is the DEFINITIVE proof that
/// B=1 decode is GPU/critical-path-bound, NOT host-bound (the earlier "94% GPU-idle" was a
/// harness-window artifact). Retained as the re-runnable host-vs-GPU diagnostic and the only
/// concrete capturable-whole-decode reference for future tree-EAGLE/fusion work.
/// See docs/experience/wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md.
fn dsv4_whole_step_graph_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_WHOLE_STEP_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}
