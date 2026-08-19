//! GGUF metadata → [`qwen35_spec::Qwen35Config`] mapper for the Vulkan loader.
//!
//! Mirrors the DSv4 mapper (`crates/infer-gguf/src/deepseek4.rs`
//! `dsv4_config_from_gguf`): read `general.architecture`, use it as the
//! metadata-key prefix, and pull each hparam with `gguf.get_*`. A wrong
//! field silently corrupts the forward, so every `Qwen35Config` field is
//! populated explicitly and the result is run through `validate()`.
//!
//! Ground truth is the on-box `Qwen3.6-27B-Q8_0.gguf` (arch `qwen35`, dense)
//! and `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (arch `qwen35moe`). The dense and MoE
//! archs share the same hparam key suffixes; only the MoE arch adds the
//! `expert_*` keys + a shared expert.
//!
//! ── ggml key → Qwen35Config field map (arch = `qwen35` | `qwen35moe`) ──────
//!   {arch}.block_count                → num_hidden_layers
//!   {arch}.embedding_length           → hidden_size
//!   {arch}.feed_forward_length        → intermediate_size (dense MLP width)
//!   {arch}.attention.head_count       → num_attention_heads
//!   {arch}.attention.head_count_kv    → num_key_value_heads
//!   {arch}.attention.key_length       → head_dim
//!   {arch}.attention.layer_norm_rms_epsilon → rms_norm_eps
//!   {arch}.rope.freq_base             → rope_theta
//!   {arch}.context_length             → rope_cache_len_hint
//!   token_embd.weight dims[1]         → vocab_size  (ne1; ggml stores
//!                                       [hidden, vocab])
//!   {arch}.ssm.conv_kernel            → linear_conv_kernel_dim
//!   {arch}.ssm.inner_size             → num_value_heads * value_head_dim
//!   {arch}.ssm.state_size             → linear_key_head_dim = linear_value_head_dim
//!   {arch}.ssm.group_count            → linear_num_key_heads
//!   {arch}.ssm.time_step_rank         → linear_num_value_heads
//!   {arch}moe.expert_count            → num_experts
//!   {arch}moe.expert_used_count       → num_experts_per_tok
//!   {arch}moe.expert_feed_forward_length → moe_intermediate_size
//!   {arch}moe.expert_shared_feed_forward_length → shared_expert_intermediate_size
//!   tokenizer.ggml.eos_token_id       → eos_token_id
//!   tokenizer.ggml.bos_token_id       → bos_token_id
//!   tie_word_embeddings = no `output.weight` tensor (separate head ⇒ untied)
//!
//! ── Linear (gated-delta / SSM) head derivation ────────────────────────────
//! Qwen3.5's linear layers are Qwen3-Next-style GatedDeltaNet. The HF config
//! exposes `linear_num_{key,value}_heads`, `linear_{key,value}_head_dim`, and
//! `linear_conv_kernel_dim`; the ggml conversion folds those into the `ssm.*`
//! keys. The relation (verified against the 27B's known dims) is:
//!   value width   inner_size = linear_num_value_heads * linear_value_head_dim
//!   per-head dim  state_size = linear_key_head_dim     = linear_value_head_dim
//!   key heads     group_count    (GVA grouped key/query heads)
//!   value heads   time_step_rank (one dt per value head)
//! For the 27B: inner_size=6144, state_size=128, group_count=16,
//! time_step_rank=48 ⇒ value_heads=48, value_head_dim=128 (48*128=6144 ✓),
//! key_heads=16, key_head_dim=128, and 48 % 16 == 0 satisfies the spec's
//! `linear_num_value_heads.is_multiple_of(linear_num_key_heads)` guard.
//!
//! If `ssm.time_step_rank` is absent we fall back to deriving the value-head
//! count from `inner_size / state_size`; if `ssm.group_count` is absent we
//! fall back to `num_key_value_heads`. Both fallbacks are documented at the
//! call site and keep `inner_size` consistent.

use anyhow::{Result, anyhow, bail};
use qwen35_spec::{LayerType, Qwen35Config};

use infer_gguf::gguf::GgufFile;

/// Build a [`Qwen35Config`] from an open qwen35 / qwen35moe GGUF.
///
/// Reads `general.architecture` (`qwen35` dense or `qwen35moe`), uses it as
/// the metadata-key prefix, scans per-layer tensors to derive `layer_types`,
/// and populates every `Qwen35Config` field. The result is validated before
/// return so a bad derivation fails at load, not mid-forward.
pub fn qwen35_config_from_gguf(gguf: &GgufFile) -> Result<Qwen35Config> {
    let arch = gguf
        .get_str("general.architecture")
        .ok_or_else(|| anyhow!("GGUF missing general.architecture"))?
        .to_string();
    let is_moe = match arch.as_str() {
        "qwen35" => false,
        "qwen35moe" => true,
        other => bail!("GGUF architecture {other:?} is not qwen35 / qwen35moe"),
    };
    // SSM keys are prefixed with the *dense* arch base in some converters; both
    // dense and MoE GGUFs key their hparams off the model's own arch string, so
    // use `arch` directly as the prefix (matches the on-box files).
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let req_usize = |suffix: &str| -> Result<usize> {
        gguf.get_usize(&key(suffix))
            .ok_or_else(|| anyhow!("GGUF missing required key {}", key(suffix)))
    };
    let req_f32 = |suffix: &str| -> Result<f32> {
        gguf.get_f32(&key(suffix))
            .ok_or_else(|| anyhow!("GGUF missing required key {}", key(suffix)))
    };

    // ── Core transformer dims. ────────────────────────────────────────────
    // `block_count` includes the trailing MTP blocks (Qwen3.8-27B ships one at
    // blk.64), which carry `nextn.*` weights and no attention tensor at all.
    // They are not part of the decode graph, so trim them before any per-layer
    // scan classifies a layer that has neither ssm_conv1d nor attn_q.
    let block_count = req_usize("block_count")?;
    let nextn_predict_layers = gguf.get_usize(&key("nextn_predict_layers")).unwrap_or(0);
    let num_hidden_layers = block_count
        .checked_sub(nextn_predict_layers)
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            anyhow!(
                "qwen35: block_count {block_count} <= nextn_predict_layers {nextn_predict_layers}"
            )
        })?;
    let hidden_size = req_usize("embedding_length")?;
    // Dense MLP width. The MoE GGUF (qwen35moe, all-sparse) ships NO
    // `feed_forward_length` — every layer is a MoE FFN sized by the
    // `expert_*` keys instead — so default it to 0 there (it's unused: the
    // forward branches to the MoE path for every `is_moe_layer`).
    let intermediate_size = gguf.get_usize(&key("feed_forward_length")).unwrap_or(0);
    let num_attention_heads = req_usize("attention.head_count")?;
    let num_key_value_heads = req_usize("attention.head_count_kv")?;
    let head_dim = req_usize("attention.key_length")?;
    let rms_norm_eps = req_f32("attention.layer_norm_rms_epsilon")?;
    let rope_theta = req_f32("rope.freq_base")?;

    // Vocab: ggml stores `token_embd.weight` as [hidden, vocab] (ne0=hidden,
    // ne1=vocab); `output.weight` carries the same vocab on its ne1. Prefer an
    // explicit `vocab_size` key if a converter ever emits one.
    let vocab_size = gguf
        .get_usize(&key("vocab_size"))
        .or_else(|| {
            gguf.tensor("token_embd.weight")
                .or_else(|| gguf.tensor("output.weight"))
                .and_then(|t| t.dims.get(1))
                .map(|&v| v as usize)
        })
        .ok_or_else(|| {
            anyhow!(
                "GGUF has neither {} nor token_embd.weight/output.weight dims",
                key("vocab_size")
            )
        })?;

    // ── RoPE geometry. Qwen3.6 uses PARTIAL rotary: only the first
    // `rope.dimension_count` of each head's `head_dim` dims are rotated, the
    // rest pass through (matches the CUDA reference's rotary_dim = head_dim ×
    // partial_rotary_factor and the HD256 prep kernel's `d >= rotary_dim`
    // branch). The ggml converter writes the rotary width as
    // `{arch}.rope.dimension_count` (= 64 on the 27B, i.e. partial_rotary_factor
    // 0.25). Hard-coding `rotary_dim = head_dim` (256) silently over-rotated
    // every q/k and scrambled attention into garbage logits. Fall back to
    // `head_dim` only if the key is absent (vanilla full-rotation). A
    // `rope.scaling.*` block, if a converter adds one, would be wired here; the
    // on-box files ship unscaled RoPE so we leave `rope_scaling = None`.
    let rotary_dim = gguf
        .get_usize(&key("rope.dimension_count"))
        .filter(|&d| d != 0)
        .unwrap_or(head_dim);
    let partial_rotary_factor = if head_dim == 0 {
        1.0
    } else {
        rotary_dim as f32 / head_dim as f32
    };
    let rope_cache_len_hint = gguf
        .get_usize(&key("context_length"))
        .or_else(|| gguf.get_usize(&key("max_position_embeddings")));

    // ── Linear (gated-delta / SSM) head dims. See module docs for the
    // ssm.* → linear_* derivation and its consistency proof. ───────────────
    let linear_conv_kernel_dim = req_usize("ssm.conv_kernel")?;
    let ssm_inner_size = req_usize("ssm.inner_size")?;
    let ssm_state_size = req_usize("ssm.state_size")?;
    // Per-head dim: ggml `ssm.state_size` is the shared key/value head width.
    let linear_value_head_dim = ssm_state_size;
    let linear_key_head_dim = ssm_state_size;
    // Value heads: prefer `ssm.time_step_rank` (one dt per value head); else
    // derive from the value width so `inner_size` stays exact.
    let linear_num_value_heads = gguf
        .get_usize(&key("ssm.time_step_rank"))
        .filter(|&v| v != 0)
        .unwrap_or_else(|| {
            ssm_inner_size
                .checked_div(linear_value_head_dim)
                .unwrap_or(0)
        });
    // Key/query heads: ggml `ssm.group_count` is the GVA group count; fall back
    // to the full-attention KV-head count if a converter omits it.
    let linear_num_key_heads = gguf
        .get_usize(&key("ssm.group_count"))
        .filter(|&v| v != 0)
        .unwrap_or(num_key_value_heads);

    // Cross-check: inner_size must equal num_value_heads * value_head_dim, or a
    // silently-wrong head split would corrupt the gated-delta forward.
    let recomputed_value_width = linear_num_value_heads * linear_value_head_dim;
    if recomputed_value_width != ssm_inner_size {
        bail!(
            "qwen35: ssm.inner_size {ssm_inner_size} != \
             linear_num_value_heads {linear_num_value_heads} * \
             linear_value_head_dim {linear_value_head_dim} = {recomputed_value_width}; \
             the ssm.* → linear_* head derivation is inconsistent for this GGUF"
        );
    }

    // ── Per-layer types from tensor presence. A layer is LINEAR (gated-delta)
    // iff `blk.N.ssm_conv1d.weight` exists, FULL iff `blk.N.attn_q.weight`
    // exists. Both archs interleave with no fixed pattern, so read it. ──────
    let layer_types = derive_layer_types(gguf, num_hidden_layers)?;

    // Full-attention Q gate: Qwen3.5/3.6 FUSE a per-head sigmoid gate into the
    // q_proj output, so `attn_q.weight` ne1 = num_heads*head_dim*2 (gated) vs
    // num_heads*head_dim (vanilla Qwen3). The real 27B GGUF ships NO separate
    // `attn_gate.weight` tensor — the gate lives in the doubled q_proj — so we
    // detect gating from the first full layer's `attn_q.weight` width, not from
    // tensor presence. (Verified on-box: blk.3.attn_q.weight ne1 = 12288 =
    // 24*256*2.) Default to gated if no full layer / q tensor is found, matching
    // the qwen35-spec HF parser's `full_attn_gated = true` default.
    let q_gate_width = num_attention_heads * head_dim * 2;
    let full_attn_gated = layer_types
        .iter()
        .position(|&t| t == LayerType::FullAttention)
        .and_then(|i| gguf.tensor(&format!("blk.{i}.attn_q.weight")))
        .and_then(|t| t.dims.get(1).copied())
        .map(|ne1| ne1 as usize == q_gate_width)
        .unwrap_or(true);

    // ── MoE block (qwen35moe only; dense leaves num_experts = 0). ──────────
    let (num_experts, num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size) =
        if is_moe {
            (
                req_usize("expert_count")?,
                req_usize("expert_used_count")?,
                req_usize("expert_feed_forward_length")?,
                gguf.get_usize(&key("expert_shared_feed_forward_length"))
                    .unwrap_or(0),
            )
        } else {
            (0, 0, 0, 0)
        };

    // ── Tokenizer special ids + tie. ──────────────────────────────────────
    let eos_token_id = gguf
        .get_u64("tokenizer.ggml.eos_token_id")
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    let bos_token_id = gguf
        .get_u64("tokenizer.ggml.bos_token_id")
        .and_then(|v| u32::try_from(v).ok());
    // Untied iff a separate `output.weight` head tensor exists.
    let tie_word_embeddings = gguf.tensor("output.weight").is_none();

    let config = Qwen35Config {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        vocab_size,
        rms_norm_eps,
        stop_token_ids: vec![eos_token_id],
        bos_token_id,
        eos_token_id,
        tie_word_embeddings,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        linear_num_key_heads,
        linear_key_head_dim,
        linear_num_value_heads,
        linear_value_head_dim,
        linear_conv_kernel_dim,
        rope_theta,
        rope_scaling: None,
        partial_rotary_factor,
        rotary_dim,
        rope_cache_len_hint,
        layer_types,
        num_experts,
        num_experts_per_tok,
        decoder_sparse_step: 1,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated,
    };
    config
        .validate()
        .map_err(|e| anyhow!("GGUF-derived qwen35 config invalid: {e}"))?;
    Ok(config)
}

/// Derive per-layer [`LayerType`] by scanning tensors for `blk.N.*`:
/// `ssm_conv1d.weight` ⇒ LinearAttention, `attn_q.weight` ⇒ FullAttention.
/// Fails loud if a layer has neither (a schema surprise) or both (ambiguous).
fn derive_layer_types(gguf: &GgufFile, num_layers: usize) -> Result<Vec<LayerType>> {
    (0..num_layers)
        .map(|i| {
            let is_linear = gguf.tensor(&format!("blk.{i}.ssm_conv1d.weight")).is_some();
            let is_full = gguf.tensor(&format!("blk.{i}.attn_q.weight")).is_some();
            match (is_linear, is_full) {
                (true, false) => Ok(LayerType::LinearAttention),
                (false, true) => Ok(LayerType::FullAttention),
                (false, false) => Err(anyhow!(
                    "qwen35: layer {i} has neither blk.{i}.ssm_conv1d.weight \
                     nor blk.{i}.attn_q.weight — cannot classify"
                )),
                (true, true) => Err(anyhow!(
                    "qwen35: layer {i} has BOTH ssm_conv1d and attn_q — ambiguous"
                )),
            }
        })
        .collect()
}
