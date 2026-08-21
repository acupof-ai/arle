//! The on-device DSv4 weight tree: one struct per weight sub-block, plus the
//! forwards and config derivations that read only weights.

use super::*;

/// Compressor sub-block for CSA/HCA layers (`compress_ratio > 0`): projects the
/// wide hidden into the compressed-key latent stream sparse attention reads.
pub(crate) struct Dsv4Compressor {
    pub wkv: DeviceMatrix,
    pub wgate: DeviceMatrix,
    pub ape: DeviceMatrix,
    pub fp32_probe: Dsv4CompressorFp32Probe,
    pub norm: DeviceVec,
    /// DeepGEMM repacks of the FP8 `wkv`/`wgate` projections for the batched
    /// (m=N) decode pre-pass. `None` ⇒ scalar `dsv4_fp8_gemv_batch`.
    pub wkv_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub wgate_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
}

pub(crate) struct Dsv4CompressorFp32Probe {
    pub wkv: DeviceMatrix,
    pub wgate: DeviceMatrix,
    pub ape: CudaSlice<f32>,
}

/// Sparse indexer sub-block. DSv4 CompressedSparse: a key compressor over
/// `index_head_dim` keys + `wq_b`/`weights_proj`. GLM DSA (`SparseIndexed`): no
/// compressor — a plain key projection `wk` + key RMSNorm `k_norm`.
pub(crate) struct Dsv4Indexer {
    pub wq_b: DeviceMatrix,
    pub weights_proj: DeviceMatrix,
    /// DSv4 CSA key compressor. `None` ⇒ GLM SparseIndexed (uses `wk`/`k_norm`).
    pub compressor: Option<Dsv4Compressor>,
    /// GLM SparseIndexed only: indexer key projection `[index_n_heads*index_head_dim,
    /// hidden]`. `None` ⇒ DSv4 (keys come through the compressor).
    pub wk: Option<DeviceMatrix>,
    /// GLM SparseIndexed only: indexer key RMSNorm weight. `None` ⇒ DSv4.
    pub k_norm: Option<DeviceVec>,
    /// DeepGEMM repack of `wq_b` for the prefill index-query projection (135ms /
    /// 67% of linear at M=1024). `None` falls back to scalar.
    pub wq_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    /// DeepGEMM repack of `weights_proj` for the batched (m=N) decode
    /// indexer-query pre-pass. `None` ⇒ scalar.
    pub weights_proj_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
}

/// One hyper-connection mixing block (`hc_attn` / `hc_ffn` per layer, `hc_head`
/// at the head). `mix_fn` projects the wide stream into the `(2+hc_mult)*hc_mult`
/// mixing weights; `base`/`scale` feed the sinkhorn `dsv4_mhc_params_cuda`.
pub(crate) struct Dsv4HyperConnection {
    pub base: DeviceVec,
    pub mix_fn: DeviceMatrix,
    pub scale: DeviceVec,
}

impl Dsv4HyperConnection {
    /// Zero placeholder for GLM (`hc_mult == 1`): the mixers are the identity, so
    /// the forward never reads these weights and GLM checkpoints ship no `hc_*`.
    pub(super) fn identity_placeholder(ctx: &DeviceContext) -> Result<Self> {
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
/// compressed latent: `wkv` → `kv_norm`. Output is low-rank: `wo_a` → `wo_b`.
/// `compressor`/`indexer` exist on CSA/HCA layers (indexer on CSA only).
pub(crate) struct Dsv4Attention {
    pub wq_a: DeviceMatrix,
    pub wqkv_a_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub q_norm: DeviceVec,
    pub wq_b: DeviceMatrix,
    /// DeepGEMM-layout FP8 cache of `wq_b` for the decode projection (M=1); the
    /// residual scalar GEMV was nsys #1 at 3.62ms. `None` unless the fused-wqkv
    /// decode alloc gate is on.
    pub wq_b_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>,
    pub wkv: DeviceMatrix,
    pub kv_norm: DeviceVec,
    /// DSv4 low-rank output down-projection. `None` ⇔ GLM (`config.plain_o_proj`,
    /// which uses `o_proj` instead).
    pub wo_a: Option<DeviceMatrix>,
    /// Per-output-group pointer table for `wo_a`. One TP rank may own more than
    /// one full output group (TP1/2/4 on DSv4-Flash), so `wo_a` is launched as
    /// group-routed `[token, group]` rows. `None` ⇔ GLM plain-o.
    pub wo_a_groups: Option<Dsv4WoAGroupTables>,
    /// DSv4 low-rank output up-projection. `None` ⇔ GLM plain-o.
    pub wo_b: Option<DeviceMatrix>,
    /// DeepGEMM-layout FP8 caches of the output projection (`wo_a`/`wo_b`) for
    /// decode. The flat `wo_a` cache exists only when this rank owns exactly one
    /// output group; multi-group ranks use `wo_a_group_deepgemm`.
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
    /// GLM plain output projection `[hidden, num_heads*v_head_dim]`, replacing the
    /// DSv4 `wo_a`/`wo_b` low-rank. `Some` ⇔ `config.plain_o_proj`.
    #[allow(dead_code)]
    pub o_proj: Option<DeviceMatrix>,
    /// GLM `kv_b` absorption split, folded at runtime around the V32 FlashMLA
    /// call. `w_kc[heads, qk_nope_head_dim, kv_lora_rank]` lifts q_nope into the
    /// 512-latent; `w_vc[heads, kv_lora_rank, v_head_dim]` projects back to v.
    /// `Some` ⇔ GLM (`config.kv_lora_rank > 0`); DSv4 is pre-absorbed.
    #[allow(dead_code)]
    pub w_kc: Option<DeviceMatrix>,
    #[allow(dead_code)]
    pub w_vc: Option<DeviceMatrix>,
}

impl Dsv4Attention {
    pub(crate) fn wo_a(&self) -> &DeviceMatrix {
        self.wo_a.as_ref().expect("DSv4 wo_a")
    }

    pub(crate) fn wo_b(&self) -> &DeviceMatrix {
        self.wo_b.as_ref().expect("DSv4 wo_b")
    }

    pub(crate) fn attn_sink(&self) -> &DeviceVec {
        self.attn_sink.as_ref().expect("DSv4 attn_sink")
    }

    pub(crate) fn compressor(&self) -> &Dsv4Compressor {
        self.compressor.as_ref().expect("DSv4 compressor")
    }

    pub(crate) fn indexer(&self) -> &Dsv4Indexer {
        self.indexer.as_ref().expect("DSv4 indexer")
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
/// w1/w3 and w2, the router gate, and the dense shared expert. Only this rank's
/// `ExpertSplit` slice is resident. Routing is per-layer: bias-routed layers
/// carry `gate_bias`; hash-routed layers (`layer_idx < num_hash_layers`) map
/// token id to experts through `hash_tid2eid` and ignore the router gate.
pub(crate) struct Dsv4MoeLayer {
    /// Contiguous per-rank group-major fused gate+up FP8 cache (w1 over w3,
    /// row-stacked) and down cache (w2). `None` for W4A16 checkpoints (routed
    /// experts live in `w13_w4a16`/`w2_w4a16`).
    pub w13_grouped: Option<crate::moe::GroupedCache>,
    pub w2_grouped: Option<crate::moe::GroupedCache>,
    /// W4A16 routed experts: per-expert fused gate+up and down `DeviceMatrix`
    /// (packed INT4 weight + BF16 group scales). `None` for FP8 checkpoints.
    pub w13_w4a16: Option<Vec<DeviceMatrix>>,
    pub w2_w4a16: Option<Vec<DeviceMatrix>>,
    /// W4AFP8 routed experts (SGLang CUTLASS layout): one contiguous packed-INT4
    /// weight + interleaved-BF16-scale buffer per projection across all local
    /// experts. `None` for FP8/W4A16 checkpoints.
    pub w13_w4afp8: Option<crate::moe::W4Afp8ExpertWeights>,
    pub w2_w4afp8: Option<crate::moe::W4Afp8ExpertWeights>,
    pub num_groups: usize,
    pub hidden_dim: usize,
    pub intermediate: usize,
    /// Router gate `[n_routed_experts, hidden]` (BF16). Hash layers ignore it.
    pub gate: DeviceMatrix,
    /// Bias-routed layers only: per-expert `noaux_tc` correction `[n_routed]`.
    pub gate_bias: Option<DeviceVec>,
    /// Hash-routed layers only: device `tid2eid` table for the on-device router.
    pub hash_tid2eid_device: Option<CudaSlice<i64>>,
    pub routing_kind: DeepSeekV4MoeRoutingKind,
    /// Dense shared expert FP8 caches (always-on, n_shared_experts == 1). GLM
    /// ships F32 `weight_scale_inv` block scales the 1D2D FP8 GEMM reads directly.
    pub shared_w13: Dsv4Fp8DeepGemmWeightCache,
    pub shared_w2: Dsv4Fp8DeepGemmWeightCache,
    /// Decode-band grouped-GEMV lane tables, built lazily on first decode-band
    /// MoE forward. `Some(None)` = build failed the lossless UE8M0 check; the
    /// contiguous DeepGEMM lane stays the fallback.
    pub gemv_tables: std::sync::OnceLock<Option<crate::moe::Dsv4GemvTables>>,
    /// W4A16 grouped-GEMV lane tables, built lazily on first W4A16 MoE forward.
    pub w4a16_gemv_tables: std::sync::OnceLock<Option<crate::moe::Dsv4W4A16GemvTables>>,
    /// W4AFP8 GEMV decode lane tables, built lazily on first decode-band forward.
    /// Reuses the W4A16 GEMV kernel with BF16 activations (skips FP8 quant).
    pub w4afp8_gemv_tables: std::sync::OnceLock<crate::moe::Dsv4W4A16GemvTables>,
}

/// GLM dense-MLP layer (the first `first_k_dense_replace` layers): a plain SwiGLU
/// FFN at `intermediate_size` replacing the routed-expert stack. Dequantized to
/// bf16 at load — GLM's F32 `weight_scale_inv` lacks the E8M0 `dsv4_scales`
/// layout the FP8 DeepGEMM caches need. `allow(dead_code)`: the DSv4-only build
/// (every layer MoE) never constructs this.
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

/// One DSv4 transformer layer. `mode` records the attention variant (SW / CSA /
/// HCA) the forward dispatches on.
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
    /// GLM dense layers only (`config.per_layer_dense_mlp[i]`): `Some` ⇒ the
    /// forward runs `dense_mlp` instead of `moe`. DSv4 layers leave this `None`.
    #[allow(dead_code)]
    pub dense_mlp: Option<Dsv4DenseMlp>,
}

/// One shipped DSv4 MTP draft head (`mtp.0.*`): a full transformer layer plus
/// the DeepSeek MTP input-combine and output-head tensors.
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
/// plus position-dependent extras. The draft stacks 3 stages (`mtp.0` → `mtp.2`):
/// entry carries `main_proj` + `main_norm` for the 3-tap context fusion; exit
/// carries `hc_head`, the final `norm`, the low-rank Markov token-transition head
/// (`markov_w1`/`markov_w2`) and `confidence_proj`; middle stages leave all
/// extras `None`. Logits decode through the base model's separate `head.weight`
/// (`tie_word_embeddings=false`), not a draft-local or embed-tied head.
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

/// GLM dense layer (`config.per_layer_dense_mlp[i]`) forward: a plain SwiGLU FFN
/// replacing the routed-expert + shared-expert MoE, bf16 throughout. `out` must
/// be `[hidden, tok]` (== `x` hidden); the caller folds it into the residual.
pub(super) fn dsv4_dense_mlp_forward(
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
    // SAFETY: uninit device scratch; fully written before first read.
    let mut gate = unsafe { HiddenStates::uninit(ctx, inter, tok)? };
    crate::attention::dsv4_linear(ctx, &dense.gate, x, &mut gate)?;
    keepalive.keep_hidden(&gate);
    // SAFETY: uninit device scratch; fully written before first read.
    let mut up = unsafe { HiddenStates::uninit(ctx, inter, tok)? };
    crate::attention::dsv4_linear(ctx, &dense.up, x, &mut up)?;
    keepalive.keep_hidden(&up);
    // GLM dense uses plain SiLU(gate)*up with NO clamp (into_deepseek_v4 sets
    // swiglu_limit=0.0, which the clamped kernel rejects).
    // ponytail: pod-verify GLM dense FFN activation = silu(gate)*up (unclamped)
    let _ = swiglu_limit;
    // SAFETY: uninit device scratch; fully written before first read.
    let mut act = unsafe { HiddenStates::uninit(ctx, inter, tok)? };
    crate::ops::silu_mul(ctx, &gate, &up, &mut act)?;
    keepalive.keep_hidden(&act);
    crate::attention::dsv4_linear(ctx, &dense.down, &act, out)?;
    Ok(())
}

impl Dsv4Model {
    /// MoE config built from the DSv4 router fields (sqrtsoftplus + noaux_tc).
    pub(crate) fn moe_config_from_config(config: &DeepSeekV4Config) -> Result<MoeConfig> {
        let moe = MoeConfig::dsv4(
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
            config.hidden_size,
        );
        moe.validate()
            .map_err(|e| anyhow::anyhow!("DSv4 MoE config invalid: {e}"))?;
        Ok(moe)
    }
}
