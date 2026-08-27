//! `qwen4_exp` (Qwen3.8-Flash-Next) `config.json` → [`Qwen4ExpConfig`].
//!
//! The Vulkan analogue of [`crate::config::qwen35_config_from_gguf`], but for a
//! checkpoint that ships an HF `config.json` rather than GGUF metadata. Ground
//! truth is the on-box `qwen3.8-flash-next-nvfp4` directory; semantics come from
//! the reference implementation (`configuration_qwen4_exp.py`'s
//! `Qwen4ExpTextConfig.__post_init__` / `validate_architecture`, plus the
//! `modeling_qwen4_exp.py` modules that read each field), not from the field
//! names.
//!
//! # The shape of the file
//!
//! Everything the text forward needs lives under a nested `text_config`; the
//! top level holds only `architectures`, the multimodal token ids, the
//! `quantization_config`, and a `vision_config` we drop. `text_config` in turn
//! nests `rope_parameters` (authoritative for `rope_theta` /
//! `partial_rotary_factor`) and an `mtp` sub-block we also drop. Stop tokens are
//! split across two files: `text_config.eos_token_id` is a scalar, and
//! `generation_config.json` carries the real multi-token stop list.
//!
//! # The five derivations that are easy to get wrong
//!
//! 1. **`ple_layer_ids` is ONE-INDEXED.** The checkpoint says `[2]` and the PLE
//!    hangs off `layer_idx == 1`. The reference validates
//!    `1 <= id <= num_hidden_layers` and then indexes `layer_types[id - 1]`, so
//!    reading it as a zero-index puts the 47.68 GiB n-gram table on the wrong
//!    layer — a layer that still runs, still produces finite logits, and is
//!    wrong. [`Qwen4ExpConfig::ple_layer_ids`] stores the ZERO-indexed ids.
//! 2. **`layer_types` says `full_attention`, the reference means
//!    `qwen_sparse_attention`.** `__post_init__` rewrites the string; those
//!    layers carry a QSA indexer. See [`Qwen4LayerType::FullAttention`].
//! 3. **`rotary_dim` truncates.** The reference computes
//!    `int(head_dim * partial_rotary_factor)` and reads the factor out of
//!    `rope_parameters`, not off the flat `text_config` key (both are present
//!    here and agree at 0.25 ⇒ 64 of 256).
//! 4. **`seed` and `norm_topk_prob` are absent from the file.** They fall back
//!    to the reference's class defaults, 1234 and `true`. `seed` feeds the
//!    n-gram hash multipliers, where a wrong value reads unrelated table rows;
//!    `norm_topk_prob` decides whether the top-10 router weights are
//!    renormalised. Neither failure is visible in a smoke test.
//! 5. **`ple_embed_dim` defaults to `hidden_size`**, and the n-gram row width is
//!    `ple_embed_dim / ngram_heads` where `ngram_heads = (ngram_size - 1) *
//!    heads_per_ngram`. Here 2560 / 16 = 160, which is exactly the `[2500012,
//!    160]` shard geometry `crate::qwen4_names` measured.
//!
//! # MRoPE
//!
//! `rope_parameters` carries `mrope_interleaved` and `mrope_section`, both of
//! which are dropped here. For a text-only decode `position_ids` is 2-D, the
//! reference broadcasts it to three identical grids, and
//! `apply_interleaved_mrope` then overwrites slices of a tensor with slices of
//! an identical tensor — the identity. MRoPE only becomes load-bearing once
//! vision tokens give the three grids different positions, and vision is
//! dropped this round.
//!
//! # No JSON dependency
//!
//! `infer-vulkan` carries no JSON crate (`crate::qwen4_names`'s tests hand-roll
//! a key scanner for the same reason), so [`json`] is a small self-contained
//! reader. It keeps number tokens verbatim rather than routing them through
//! `f64`, so integer fields are read exactly.

use anyhow::{Result, anyhow, bail, ensure};
use std::path::Path;

use crate::qwen4_ple::NGramHashConfig;

/// Default [`Qwen4ExpConfig::max_context`], in tokens.
///
/// The QSA indexer is *provably* the identity below this. For a query with `v`
/// visible tokens the indexer forms `v / compress_ratio` complete blocks and
/// keeps `min(indexer_budget / compress_ratio, blocks)` of them, then appends
/// the `v % compress_ratio` tail tokens unconditionally. So as long as
/// `v / compress_ratio <= indexer_budget / compress_ratio`, i.e.
/// `v <= indexer_budget + compress_ratio - 1` (2048 + 3 = 2051 here), *every*
/// visible token is selected and the emitted mask is exactly the causal mask.
/// ARLE stubs the indexer and gates context here, comfortably under that bound;
/// see [`Qwen4QsaConfig::dense_below_or_equal`].
pub const DEFAULT_MAX_CONTEXT: usize = 2048;

/// Reference default for `seed`, absent from the on-box `config.json`.
const DEFAULT_NGRAM_SEED: u64 = 1234;
/// Reference default for `full_attention_interval`, used only when the
/// checkpoint omits `layer_types`.
const DEFAULT_FULL_ATTENTION_INTERVAL: usize = 4;

/// What a decoder layer's token-mixing sublayer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4LayerType {
    /// `linear_attention` — a Qwen3-Next-style GatedDeltaNet with a depthwise
    /// causal conv. 36 of the 48 layers.
    LinearAttention,
    /// Spelled `full_attention` in the checkpoint, but `__post_init__` rewrites
    /// it to `qwen_sparse_attention`: these layers own a `self_attn.indexer`
    /// (QSA) that narrows the causal mask. 12 of the 48 layers.
    ///
    /// ARLE stubs the indexer and runs plain causal attention, which is exact
    /// while the context stays at or below
    /// [`Qwen4QsaConfig::dense_below_or_equal`].
    FullAttention,
}

impl Qwen4LayerType {
    /// Parse a `layer_types` entry. Accepts both the checkpoint's spelling and
    /// the post-init one, because the reference emits the latter on save.
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "linear_attention" => Ok(Self::LinearAttention),
            "full_attention" | "qwen_sparse_attention" => Ok(Self::FullAttention),
            other => bail!("qwen4_exp: unsupported layer type `{other}`"),
        }
    }
}

/// Activation on the GatedDeltaNet output gate (`Qwen4ExpTextRMSNormGated`).
///
/// The reference takes `output_gate_type or hidden_act` and rejects anything
/// outside this pair, so a checkpoint that omits the key silently inherits
/// `hidden_act`. The on-box file sets `sigmoid` while `hidden_act` is `silu`,
/// which is exactly the case where inheriting would be wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateActivation {
    /// `sigmoid` — what the on-box checkpoint asks for.
    Sigmoid,
    /// `silu`.
    Silu,
}

/// The QSA (quantized sparse attention) indexer hyper-parameters.
///
/// Present as a unit or not at all: the reference errors if some of the five
/// fields are set and others are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen4QsaConfig {
    /// `indexer_n_heads` — index query heads (4).
    pub n_heads: usize,
    /// `indexer_kv_heads` — the reference requires exactly 1.
    pub kv_heads: usize,
    /// `indexer_head_dim` — index q/k head width (128). Must be at least
    /// `rotary_dim`, since the index queries take the same RoPE.
    pub head_dim: usize,
    /// `indexer_budget` — max tokens kept from complete blocks (2048).
    pub budget: usize,
    /// `indexer_compress_ratio` — token keys mean-pooled per index block (4).
    pub compress_ratio: usize,
}

impl Qwen4QsaConfig {
    /// `token_budget / compress_ratio` — how many blocks a query keeps (512).
    #[must_use]
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }

    /// The largest visible-token count for which the indexer is exactly the
    /// identity: `budget + compress_ratio - 1` (2051). See
    /// [`DEFAULT_MAX_CONTEXT`] for the argument.
    #[must_use]
    pub fn dense_below_or_equal(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }
}

/// Everything the `qwen4_exp` Vulkan forward needs out of `config.json` and
/// `generation_config.json`.
///
/// Built by [`Qwen4ExpConfig::from_model_dir`]; every field is populated
/// explicitly and the result runs through [`Qwen4ExpConfig::validate`], so a bad
/// derivation fails at load rather than mid-forward.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpConfig {
    // ── core ─────────────────────────────────────────────────────────────
    /// `hidden_size` — the *block* width (2560), not the residual width. The
    /// inter-layer residual is [`Self::hc_hidden_size`].
    pub hidden_size: usize,
    /// `num_hidden_layers` (48). Excludes the MTP block, which is separate.
    pub num_hidden_layers: usize,
    /// `vocab_size` (248320).
    pub vocab_size: usize,
    /// `rms_norm_eps` (1e-6).
    pub rms_norm_eps: f32,
    /// `hidden_act` — the MLP / expert activation. Only `silu` is implemented.
    pub hidden_act: String,
    /// `tie_word_embeddings` (false here: `lm_head.weight` is its own tensor).
    pub tie_word_embeddings: bool,

    // ── hyper-connections ────────────────────────────────────────────────
    /// `hc_count` (4) — residual streams. The embedding is tiled this many
    /// times to seed the stream.
    pub hc_count: usize,
    /// `hc_lowrank` (320) — inner width of the gated-residual input mixer.
    pub hc_lowrank: usize,

    // ── per-layer plan ───────────────────────────────────────────────────
    /// One entry per layer, length [`Self::num_hidden_layers`].
    pub layer_types: Vec<Qwen4LayerType>,

    // ── full attention ───────────────────────────────────────────────────
    /// `num_attention_heads` (24).
    pub num_attention_heads: usize,
    /// `num_key_value_heads` (2).
    pub num_key_value_heads: usize,
    /// `head_dim` (256). Note `hidden_size / num_attention_heads` is *not*
    /// integral here, so this key is load-bearing rather than derivable.
    pub head_dim: usize,
    /// `attention_bias` (false) — no q/k/v/o bias tensors exist.
    pub attention_bias: bool,
    /// `int(head_dim * partial_rotary_factor)` = 64. The leading `rotary_dim`
    /// lanes of each head rotate; the remaining 192 pass through.
    pub rotary_dim: usize,
    /// `rope_parameters.partial_rotary_factor` (0.25).
    pub partial_rotary_factor: f32,
    /// `rope_parameters.rope_theta` (1e7).
    pub rope_theta: f32,
    /// `max_position_embeddings` (262144) — the checkpoint's claim, not what
    /// ARLE serves. Enforce [`Self::max_context`] instead.
    pub max_position_embeddings: usize,

    // ── linear attention (GatedDeltaNet) ─────────────────────────────────
    /// `linear_num_key_heads` (16).
    pub linear_num_key_heads: usize,
    /// `linear_num_value_heads` (48).
    pub linear_num_value_heads: usize,
    /// `linear_key_head_dim` (128).
    pub linear_key_head_dim: usize,
    /// `linear_value_head_dim` (128).
    pub linear_value_head_dim: usize,
    /// `linear_conv_kernel_dim` (4) — depthwise causal conv width.
    pub linear_conv_kernel_dim: usize,
    /// `output_gate_type or hidden_act`, resolved.
    pub output_gate: GateActivation,

    // ── MoE ──────────────────────────────────────────────────────────────
    /// `num_experts` (512).
    pub num_experts: usize,
    /// `num_experts_per_tok` (10).
    pub num_experts_per_tok: usize,
    /// `moe_intermediate_size` (640) — per routed expert.
    pub moe_intermediate_size: usize,
    /// `shared_expert_intermediate_size` (640). The shared expert is gated by
    /// its own `sigmoid(shared_expert_gate @ x)` scalar.
    pub shared_expert_intermediate_size: usize,
    /// `norm_topk_prob` — absent from the file, so the reference default
    /// `true` applies: the top-k softmax probabilities are renormalised to sum
    /// to 1 before weighting the experts.
    pub norm_topk_prob: bool,

    // ── PLE / n-gram ─────────────────────────────────────────────────────
    /// `ple_layer_ids`, converted to ZERO-indexed layer ids and sorted. The
    /// file's `[2]` becomes `[1]`.
    pub ple_layer_ids: Vec<usize>,
    /// `ple_embed_dim` (2560, defaulting to `hidden_size`).
    pub ple_embed_dim: usize,
    /// `ple_conv_kernel_size` (4) — dilated depthwise conv in the PLE module.
    pub ple_conv_kernel_size: usize,
    /// `ngram_size` (3) — the widest n-gram, and the multiplier count.
    pub ngram_size: usize,
    /// `heads_per_ngram` (8).
    pub heads_per_ngram: usize,
    /// `ngram_vocab_size_base` (20_000_000) — primes are searched above
    /// `base - 1`.
    pub ngram_vocab_size_base: u64,
    /// `make_ngram_vocab_size_divisible_by` (128) — table row-count padding.
    pub make_ngram_vocab_size_divisible_by: u64,
    /// `seed` — absent from the file; the reference default 1234 applies.
    pub ngram_seed: u64,
    /// `split_ngram_parts` (128) — `shard_<i>.weight` tensors per PLE table.
    /// The reference class default is 512, so this key must be read.
    pub split_ngram_parts: usize,

    // ── QSA ──────────────────────────────────────────────────────────────
    /// The indexer hyper-parameters, or `None` if the checkpoint disables QSA.
    pub indexer: Option<Qwen4QsaConfig>,

    // ── tokens ───────────────────────────────────────────────────────────
    /// `text_config.bos_token_id` (248044).
    pub bos_token_id: Option<u32>,
    /// The PLE segment separator and out-of-context fill: the scalar (or first)
    /// `text_config.eos_token_id`, 248044. This is *not* the stop set — see
    /// [`Self::stop_token_ids`].
    pub eos_token_id: u32,
    /// `text_config.eos_token_id` merged with `generation_config.json`'s, in
    /// first-seen order: `[248044, 248046]`.
    pub stop_token_ids: Vec<u32>,

    // ── serving policy ───────────────────────────────────────────────────
    /// Context ARLE will actually serve. Defaults to [`DEFAULT_MAX_CONTEXT`];
    /// the caller enforces it. See that constant for why 2048 keeps the stubbed
    /// QSA indexer exact.
    pub max_context: usize,
}

impl Qwen4ExpConfig {
    /// Read `config.json` and `generation_config.json` from a checkpoint dir.
    ///
    /// # Errors
    /// If `config.json` is missing or malformed, or a derived field fails
    /// [`Self::validate`]. A missing `generation_config.json` is not an error —
    /// the stop set then holds only `text_config.eos_token_id`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let config = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("qwen4_exp: read {}: {e}", config_path.display()))?;
        // Absent is legal: some checkpoints ship stop tokens only in config.json.
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        Self::from_json(&config, generation.as_deref())
    }

    /// Same as [`Self::from_model_dir`], from the file contents.
    ///
    /// # Errors
    /// If either document is not valid JSON, a required key is missing or has
    /// the wrong type, or a derived field fails [`Self::validate`].
    pub fn from_json(config_json: &str, generation_json: Option<&str>) -> Result<Self> {
        let root = json::parse(config_json)?;
        // The real file nests everything under `text_config`. Fall back to the
        // root for a hypothetical text-only re-export that flattened it; a file
        // with neither shape fails on the first required key, with the key name.
        let text = root.get("text_config").unwrap_or(&root);

        let hidden_size = req_usize(text, "hidden_size")?;
        let num_hidden_layers = req_usize(text, "num_hidden_layers")?;
        let num_attention_heads = req_usize(text, "num_attention_heads")?;
        let num_key_value_heads = req_usize(text, "num_key_value_heads")?;
        let vocab_size = req_usize(text, "vocab_size")?;
        let rms_norm_eps = req_f32(text, "rms_norm_eps")?;
        let hidden_act = opt_str(text, "hidden_act")?.unwrap_or("silu").to_owned();
        let tie_word_embeddings = opt_bool(text, "tie_word_embeddings")?.unwrap_or(false);

        // `head_dim` is not derivable here: 2560 / 24 is not an integer. The
        // reference's `head_dim or hidden_size // num_attention_heads` fallback
        // is kept, but only when it divides exactly.
        let head_dim = match opt_usize(text, "head_dim")? {
            Some(d) => d,
            None => {
                ensure!(
                    num_attention_heads > 0 && hidden_size.is_multiple_of(num_attention_heads),
                    "qwen4_exp: no head_dim, and hidden_size {hidden_size} / \
                     num_attention_heads {num_attention_heads} is not exact"
                );
                hidden_size / num_attention_heads
            }
        };

        // RoPE. The reference reads both `rope_theta` and
        // `partial_rotary_factor` out of `rope_parameters`, so that sub-object
        // wins over the flat text_config keys when both exist.
        let rope = text.get("rope_parameters");
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| text.get("rope_theta"))
            .ok_or_else(|| anyhow!("qwen4_exp: missing rope_parameters.rope_theta"))?
            .as_f64()
            .ok_or_else(|| anyhow!("qwen4_exp: rope_theta is not a number"))?;
        let partial_rotary_factor = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| text.get("partial_rotary_factor"))
            .map_or(Ok(1.0), |v| {
                v.as_f64()
                    .ok_or_else(|| anyhow!("qwen4_exp: partial_rotary_factor is not a number"))
            })?;
        if let Some(kind) = rope
            .and_then(|r| r.get("rope_type"))
            .and_then(json::Json::as_str)
        {
            // A scaled rope would need an inv_freq transform the Vulkan lane
            // does not implement; fail at load rather than serve wrong positions.
            ensure!(
                kind == "default",
                "qwen4_exp: unsupported rope_type `{kind}` (only `default`)"
            );
        }
        // `int(...)` in the reference: truncation, not rounding.
        let rotary_dim = (f64::from(u32::try_from(head_dim)?) * partial_rotary_factor) as usize;

        let layer_types = parse_layer_types(text, num_hidden_layers)?;

        // ── PLE. `ple_layer_ids` is ONE-INDEXED in the file. ──────────────
        let mut ple_layer_ids = Vec::new();
        if let Some(ids) = opt_array(text, "ple_layer_ids")? {
            for raw in ids {
                let one_based = raw
                    .as_i64()
                    .ok_or_else(|| anyhow!("qwen4_exp: ple_layer_ids entry is not an integer"))?;
                ensure!(
                    one_based >= 1 && (one_based as usize) <= num_hidden_layers,
                    "qwen4_exp: ple_layer_ids entry {one_based} outside \
                     the one-indexed range [1, {num_hidden_layers}]"
                );
                ple_layer_ids.push(one_based as usize - 1);
            }
            ple_layer_ids.sort_unstable();
            ple_layer_ids.dedup();
        }

        // ── QSA: all five fields, or none. ───────────────────────────────
        let qsa_fields = [
            ("indexer_n_heads", opt_usize(text, "indexer_n_heads")?),
            ("indexer_kv_heads", opt_usize(text, "indexer_kv_heads")?),
            ("indexer_head_dim", opt_usize(text, "indexer_head_dim")?),
            ("indexer_budget", opt_usize(text, "indexer_budget")?),
            (
                "indexer_compress_ratio",
                opt_usize(text, "indexer_compress_ratio")?,
            ),
        ];
        let indexer = if qsa_fields.iter().any(|(_, v)| v.is_some()) {
            let missing: Vec<&str> = qsa_fields
                .iter()
                .filter(|(_, v)| v.is_none())
                .map(|(k, _)| *k)
                .collect();
            ensure!(
                missing.is_empty(),
                "qwen4_exp: partial QSA config, missing {missing:?}"
            );
            Some(Qwen4QsaConfig {
                n_heads: qsa_fields[0].1.unwrap_or_default(),
                kv_heads: qsa_fields[1].1.unwrap_or_default(),
                head_dim: qsa_fields[2].1.unwrap_or_default(),
                budget: qsa_fields[3].1.unwrap_or_default(),
                compress_ratio: qsa_fields[4].1.unwrap_or_default(),
            })
        } else {
            None
        };

        // ── stop tokens: text_config scalar + generation_config list. ────
        let text_eos = eos_list(text.get("eos_token_id"))?;
        let eos_token_id = *text_eos
            .first()
            .ok_or_else(|| anyhow!("qwen4_exp: text_config has no eos_token_id"))?;
        let mut stop_token_ids = text_eos;
        if let Some(gen_json) = generation_json {
            let generation = json::parse(gen_json)?;
            for id in eos_list(generation.get("eos_token_id"))? {
                if !stop_token_ids.contains(&id) {
                    stop_token_ids.push(id);
                }
            }
        }

        let output_gate = match opt_str(text, "output_gate_type")?.unwrap_or(&hidden_act) {
            "sigmoid" => GateActivation::Sigmoid,
            "silu" => GateActivation::Silu,
            other => bail!("qwen4_exp: unsupported output gate activation `{other}`"),
        };

        let cfg = Self {
            hidden_size,
            num_hidden_layers,
            vocab_size,
            rms_norm_eps,
            hidden_act,
            tie_word_embeddings,
            hc_count: opt_usize(text, "hc_count")?.unwrap_or(4),
            hc_lowrank: opt_usize(text, "hc_lowrank")?.unwrap_or(320),
            layer_types,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attention_bias: opt_bool(text, "attention_bias")?.unwrap_or(false),
            rotary_dim,
            #[expect(
                clippy::cast_possible_truncation,
                reason = "config-scale constants; f32 is what the rope kernel consumes"
            )]
            partial_rotary_factor: partial_rotary_factor as f32,
            #[expect(
                clippy::cast_possible_truncation,
                reason = "1e7 is exact in f32; validate() rejects a non-finite theta"
            )]
            rope_theta: rope_theta as f32,
            max_position_embeddings: opt_usize(text, "max_position_embeddings")?.unwrap_or(32768),
            linear_num_key_heads: req_usize(text, "linear_num_key_heads")?,
            linear_num_value_heads: req_usize(text, "linear_num_value_heads")?,
            linear_key_head_dim: req_usize(text, "linear_key_head_dim")?,
            linear_value_head_dim: req_usize(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: opt_usize(text, "linear_conv_kernel_dim")?.unwrap_or(4),
            output_gate,
            num_experts: req_usize(text, "num_experts")?,
            num_experts_per_tok: req_usize(text, "num_experts_per_tok")?,
            moe_intermediate_size: req_usize(text, "moe_intermediate_size")?,
            shared_expert_intermediate_size: req_usize(text, "shared_expert_intermediate_size")?,
            norm_topk_prob: opt_bool(text, "norm_topk_prob")?.unwrap_or(true),
            ple_layer_ids,
            ple_embed_dim: opt_usize(text, "ple_embed_dim")?.unwrap_or(hidden_size),
            ple_conv_kernel_size: opt_usize(text, "ple_conv_kernel_size")?.unwrap_or(4),
            ngram_size: opt_usize(text, "ngram_size")?.unwrap_or(3),
            heads_per_ngram: opt_usize(text, "heads_per_ngram")?.unwrap_or(8),
            ngram_vocab_size_base: opt_u64(text, "ngram_vocab_size_base")?.unwrap_or(20_000_000),
            make_ngram_vocab_size_divisible_by: opt_u64(
                text,
                "make_ngram_vocab_size_divisible_by",
            )?
            .unwrap_or(128),
            ngram_seed: opt_u64(text, "seed")?.unwrap_or(DEFAULT_NGRAM_SEED),
            split_ngram_parts: opt_usize(text, "split_ngram_parts")?.unwrap_or(512),
            indexer,
            bos_token_id: opt_u64(text, "bos_token_id")?
                .map(u32::try_from)
                .transpose()?,
            eos_token_id,
            stop_token_ids,
            // The stub is only the model up to the dense bound, so a checkpoint
            // with a smaller budget than ours gets the smaller default rather
            // than a config that cannot be built. The on-box budget of 2048
            // makes this exactly `DEFAULT_MAX_CONTEXT`.
            max_context: indexer.map_or(DEFAULT_MAX_CONTEXT, |q| {
                DEFAULT_MAX_CONTEXT.min(q.dense_below_or_equal())
            }),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Override [`Self::max_context`], re-validating.
    ///
    /// # Errors
    /// If the new value is zero, or exceeds what the stubbed QSA indexer can
    /// serve exactly ([`Qwen4QsaConfig::dense_below_or_equal`]).
    pub fn with_max_context(mut self, tokens: usize) -> Result<Self> {
        self.max_context = tokens;
        self.validate()?;
        Ok(self)
    }

    /// Check every invariant the reference's `validate_architecture` checks,
    /// plus the ones ARLE's stubs depend on.
    ///
    /// # Errors
    /// If any dimension is degenerate, `layer_types` has the wrong length, the
    /// PLE lands somewhere the reference forbids, the QSA config is
    /// inconsistent, or `max_context` outruns the dense-indexer bound.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.hidden_size > 0, "hidden_size must be positive");
        ensure!(self.num_hidden_layers > 0, "num_hidden_layers must be > 0");
        ensure!(self.vocab_size > 0, "vocab_size must be positive");
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "rms_norm_eps {} is not a positive finite number",
            self.rms_norm_eps
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope_theta {} is not a positive finite number",
            self.rope_theta
        );
        // The MoE / MLP path is SwiGLU-only on this lane.
        ensure!(
            self.hidden_act == "silu",
            "qwen4_exp: unsupported hidden_act `{}` (only `silu`)",
            self.hidden_act
        );

        ensure!(
            self.hc_count > 1,
            "qwen4_exp requires hc_count > 1, got {}",
            self.hc_count
        );
        ensure!(self.hc_lowrank > 0, "hc_lowrank must be positive");

        ensure!(
            self.layer_types.len() == self.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            self.layer_types.len(),
            self.num_hidden_layers
        );

        ensure!(
            self.num_attention_heads > 0 && self.num_key_value_heads > 0,
            "attention head counts must be positive"
        );
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} is not a multiple of num_key_value_heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        ensure!(self.head_dim > 0, "head_dim must be positive");
        ensure!(
            self.rotary_dim > 0 && self.rotary_dim <= self.head_dim,
            "rotary_dim {} outside (0, head_dim {}]",
            self.rotary_dim,
            self.head_dim
        );
        // The rope kernel pairs lanes `i` and `i + rotary_dim/2`.
        ensure!(
            self.rotary_dim.is_multiple_of(2),
            "rotary_dim {} is odd",
            self.rotary_dim
        );

        ensure!(
            self.linear_num_key_heads > 0 && self.linear_num_value_heads > 0,
            "linear head counts must be positive"
        );
        ensure!(
            self.linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads),
            "linear_num_value_heads {} is not a multiple of linear_num_key_heads {}",
            self.linear_num_value_heads,
            self.linear_num_key_heads
        );
        ensure!(
            self.linear_key_head_dim > 0 && self.linear_value_head_dim > 0,
            "linear head dims must be positive"
        );
        ensure!(
            self.linear_conv_kernel_dim > 0,
            "linear_conv_kernel_dim must be positive"
        );

        ensure!(self.num_experts > 0, "num_experts must be positive");
        ensure!(
            self.num_experts_per_tok > 0 && self.num_experts_per_tok <= self.num_experts,
            "num_experts_per_tok {} outside [1, num_experts {}]",
            self.num_experts_per_tok,
            self.num_experts
        );
        ensure!(
            self.moe_intermediate_size > 0 && self.shared_expert_intermediate_size > 0,
            "expert intermediate sizes must be positive"
        );

        if !self.ple_layer_ids.is_empty() {
            let ngram_heads = self.ngram_heads();
            ensure!(
                self.ngram_size >= 2,
                "ngram_size {} < 2 leaves no n-gram to hash",
                self.ngram_size
            );
            ensure!(self.heads_per_ngram > 0, "heads_per_ngram must be positive");
            ensure!(
                self.ple_embed_dim > 0 && self.ple_embed_dim.is_multiple_of(ngram_heads),
                "ple_embed_dim {} is not divisible by the {ngram_heads} n-gram heads",
                self.ple_embed_dim
            );
            ensure!(
                self.ple_conv_kernel_size > 0,
                "ple_conv_kernel_size must be positive"
            );
            ensure!(
                self.split_ngram_parts > 0,
                "split_ngram_parts must be positive"
            );
            ensure!(
                self.ngram_vocab_size_base > 0 && self.make_ngram_vocab_size_divisible_by > 0,
                "n-gram vocab base and divisor must be positive"
            );
            for &layer in &self.ple_layer_ids {
                // Already range-checked at parse; re-checked because
                // `ple_layer_ids` is public and `validate` is the guard.
                ensure!(
                    layer < self.num_hidden_layers,
                    "ple layer {layer} outside [0, {})",
                    self.num_hidden_layers
                );
                ensure!(
                    self.layer_types[layer] == Qwen4LayerType::LinearAttention,
                    "qwen4_exp PLE is only supported on linear_attention layers, \
                     got layer {layer} = {:?}",
                    self.layer_types[layer]
                );
            }
        }

        if let Some(qsa) = self.indexer {
            ensure!(
                qsa.n_heads > 0
                    && qsa.kv_heads > 0
                    && qsa.head_dim > 0
                    && qsa.budget > 0
                    && qsa.compress_ratio > 0,
                "QSA config values must be positive: {qsa:?}"
            );
            ensure!(
                qsa.kv_heads == 1,
                "qwen4_exp QSA requires indexer_kv_heads=1"
            );
            ensure!(
                qsa.budget.is_multiple_of(qsa.compress_ratio),
                "indexer_budget {} is not divisible by indexer_compress_ratio {}",
                qsa.budget,
                qsa.compress_ratio
            );
            // The index queries take the attention RoPE, so they must have room
            // for it.
            ensure!(
                self.rotary_dim <= qsa.head_dim,
                "rotary_dim {} exceeds indexer_head_dim {}",
                self.rotary_dim,
                qsa.head_dim
            );
        }

        ensure!(self.max_context > 0, "max_context must be positive");
        if let Some(qsa) = self.indexer {
            // ARLE stubs the indexer; past this bound the stub is no longer the
            // model. Refuse rather than serve a different architecture.
            ensure!(
                self.max_context <= qsa.dense_below_or_equal(),
                "max_context {} exceeds the {} tokens for which the stubbed QSA \
                 indexer is exactly dense",
                self.max_context,
                qsa.dense_below_or_equal()
            );
        }

        ensure!(
            !self.stop_token_ids.is_empty(),
            "qwen4_exp needs at least one stop token"
        );
        Ok(())
    }

    // ── derived widths ───────────────────────────────────────────────────

    /// `hc_count * hidden_size` = 10240 — the inter-layer residual width.
    #[must_use]
    pub fn hc_hidden_size(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    /// `num_attention_heads * head_dim` = 6144 — the `q_proj` output width.
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// `num_key_value_heads * head_dim` = 512 — the `k_proj` / `v_proj` width.
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// `linear_num_key_heads * linear_key_head_dim` = 2048.
    #[must_use]
    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// `linear_num_value_heads * linear_value_head_dim` = 6144.
    #[must_use]
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Channels of the GatedDeltaNet depthwise conv: `2 * key_dim + value_dim`
    /// = 10240 (q, k, v concatenated).
    #[must_use]
    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    /// `(ngram_size - 1) * heads_per_ngram` = 16 hash heads.
    #[must_use]
    pub fn ngram_heads(&self) -> usize {
        self.ngram_size.saturating_sub(1) * self.heads_per_ngram
    }

    /// `ple_embed_dim / ngram_heads` = 160 — one n-gram table row.
    ///
    /// Returns 0 when there are no n-gram heads, which [`Self::validate`]
    /// already rejects for a PLE-carrying checkpoint.
    #[must_use]
    pub fn ngram_head_dim(&self) -> usize {
        let heads = self.ngram_heads();
        if heads == 0 {
            0
        } else {
            self.ple_embed_dim / heads
        }
    }

    /// Conv states a layer cache must hold: the GatedDeltaNet conv alone, or
    /// that plus the PLE conv and the rolling n-gram token context.
    #[must_use]
    pub fn num_conv_states(&self) -> usize {
        if self.ple_layer_ids.is_empty() { 1 } else { 3 }
    }

    // ── per-layer queries ────────────────────────────────────────────────

    /// The layer's token-mixing type.
    ///
    /// # Panics
    /// If `layer >= num_hidden_layers`.
    #[must_use]
    pub fn layer_type(&self, layer: usize) -> Qwen4LayerType {
        self.layer_types[layer]
    }

    /// Whether this (zero-indexed) layer owns a PLE module.
    #[must_use]
    pub fn is_ple_layer(&self, layer: usize) -> bool {
        self.ple_layer_ids.contains(&layer)
    }

    /// This layer's position within `ple_layer_ids`, which shifts both the
    /// prime window and the hash multipliers — two PLE layers never share a
    /// hash. `None` if the layer has no PLE.
    #[must_use]
    pub fn ple_layer_index(&self, layer: usize) -> Option<usize> {
        self.ple_layer_ids.iter().position(|&l| l == layer)
    }

    /// The [`NGramHashConfig`] for the `ple_layer_index`-th PLE layer, so the
    /// hash is derived once here instead of at each call site.
    #[must_use]
    pub fn ngram_hash_config(&self, ple_layer_index: usize) -> NGramHashConfig {
        NGramHashConfig {
            ngram_size: self.ngram_size,
            heads_per_ngram: self.heads_per_ngram,
            ngram_vocab_size_base: self.ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by: self.make_ngram_vocab_size_divisible_by,
            vocab_size: self.vocab_size as u64,
            seed: self.ngram_seed,
            eos_token_id: i64::from(self.eos_token_id),
            ple_embed_dim: self.ple_embed_dim,
            ple_layer_index,
        }
    }
}

/// `layer_types` if present, else the `full_attention_interval` pattern.
///
/// The reference's fallback is `linear_attention if (i + 1) % interval else
/// qwen_sparse_attention`, i.e. every `interval`-th layer counting from one is
/// the attention layer — layers 3, 7, 11, ... zero-indexed.
fn parse_layer_types(text: &json::Json, num_hidden_layers: usize) -> Result<Vec<Qwen4LayerType>> {
    if let Some(entries) = opt_array(text, "layer_types")? {
        ensure!(
            entries.len() == num_hidden_layers,
            "layer_types has {} entries for {num_hidden_layers} layers",
            entries.len()
        );
        return entries
            .iter()
            .map(|v| {
                let raw = v
                    .as_str()
                    .ok_or_else(|| anyhow!("qwen4_exp: layer_types entry is not a string"))?;
                Qwen4LayerType::parse(raw)
            })
            .collect();
    }
    let interval =
        opt_usize(text, "full_attention_interval")?.unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL);
    ensure!(interval > 0, "full_attention_interval must be positive");
    Ok((0..num_hidden_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(interval) {
                Qwen4LayerType::FullAttention
            } else {
                Qwen4LayerType::LinearAttention
            }
        })
        .collect())
}

/// `eos_token_id` in either spelling — a scalar or a list — as a list.
/// Absent or `null` yields an empty list.
fn eos_list(value: Option<&json::Json>) -> Result<Vec<u32>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let one = |v: &json::Json| -> Result<u32> {
        let id = v
            .as_i64()
            .ok_or_else(|| anyhow!("qwen4_exp: eos_token_id entry is not an integer"))?;
        Ok(u32::try_from(id)?)
    };
    match value {
        json::Json::Null => Ok(Vec::new()),
        json::Json::Arr(items) => items.iter().map(one).collect(),
        other => Ok(vec![one(other)?]),
    }
}

// ── typed lookups ────────────────────────────────────────────────────────
// `null` counts as absent: the checkpoint spells optional integers that way
// (`pad_token_id: null`).

fn lookup<'j>(obj: &'j json::Json, key: &str) -> Option<&'j json::Json> {
    match obj.get(key) {
        None | Some(json::Json::Null) => None,
        some => some,
    }
}

fn req<'j>(obj: &'j json::Json, key: &str) -> Result<&'j json::Json> {
    lookup(obj, key).ok_or_else(|| anyhow!("qwen4_exp: config is missing required key `{key}`"))
}

fn req_usize(obj: &json::Json, key: &str) -> Result<usize> {
    let raw = req(obj, key)?
        .as_i64()
        .ok_or_else(|| anyhow!("qwen4_exp: `{key}` is not an integer"))?;
    usize::try_from(raw).map_err(|_| anyhow!("qwen4_exp: `{key}` = {raw} is negative"))
}

fn req_f32(obj: &json::Json, key: &str) -> Result<f32> {
    let raw = req(obj, key)?
        .as_f64()
        .ok_or_else(|| anyhow!("qwen4_exp: `{key}` is not a number"))?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "eps-scale constants; validate() rejects a non-finite result"
    )]
    Ok(raw as f32)
}

fn opt_usize(obj: &json::Json, key: &str) -> Result<Option<usize>> {
    match lookup(obj, key) {
        None => Ok(None),
        Some(_) => req_usize(obj, key).map(Some),
    }
}

fn opt_u64(obj: &json::Json, key: &str) -> Result<Option<u64>> {
    match lookup(obj, key) {
        None => Ok(None),
        Some(v) => {
            let raw = v
                .as_i64()
                .ok_or_else(|| anyhow!("qwen4_exp: `{key}` is not an integer"))?;
            Ok(Some(u64::try_from(raw).map_err(|_| {
                anyhow!("qwen4_exp: `{key}` = {raw} is negative")
            })?))
        }
    }
}

fn opt_bool(obj: &json::Json, key: &str) -> Result<Option<bool>> {
    match lookup(obj, key) {
        None => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| anyhow!("qwen4_exp: `{key}` is not a bool")),
    }
}

fn opt_str<'j>(obj: &'j json::Json, key: &str) -> Result<Option<&'j str>> {
    match lookup(obj, key) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(Some)
            .ok_or_else(|| anyhow!("qwen4_exp: `{key}` is not a string")),
    }
}

fn opt_array<'j>(obj: &'j json::Json, key: &str) -> Result<Option<&'j [json::Json]>> {
    match lookup(obj, key) {
        None => Ok(None),
        Some(v) => v
            .as_array()
            .map(Some)
            .ok_or_else(|| anyhow!("qwen4_exp: `{key}` is not an array")),
    }
}

/// A minimal JSON reader.
///
/// `infer-vulkan` has no JSON dependency and this module needs typed values out
/// of two small documents, so the grammar is implemented here rather than
/// widening the crate's dependency set. Number tokens are kept verbatim and
/// parsed on demand, so integer fields (`vocab_size`, `ngram_vocab_size_base`,
/// token ids) never round-trip through `f64`.
pub mod json {
    use anyhow::{Result, anyhow, bail};

    /// A parsed JSON value.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Json {
        /// `null`.
        Null,
        /// `true` / `false`.
        Bool(bool),
        /// A number, as its verbatim source token.
        Num(String),
        /// A string, with escapes resolved.
        Str(String),
        /// An array.
        Arr(Vec<Json>),
        /// An object. Insertion-ordered; lookups are linear, which is right for
        /// documents of a few dozen keys.
        Obj(Vec<(String, Json)>),
    }

    impl Json {
        /// The value for `key`, or `None` for a non-object or a missing key.
        #[must_use]
        pub fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }

        /// The entries, if this is an object.
        #[must_use]
        pub fn as_object(&self) -> Option<&[(String, Json)]> {
            match self {
                Json::Obj(entries) => Some(entries),
                _ => None,
            }
        }

        /// The elements, if this is an array.
        #[must_use]
        pub fn as_array(&self) -> Option<&[Json]> {
            match self {
                Json::Arr(items) => Some(items),
                _ => None,
            }
        }

        /// The text, if this is a string.
        #[must_use]
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Json::Str(s) => Some(s),
                _ => None,
            }
        }

        /// The value, if this is a bool.
        #[must_use]
        pub fn as_bool(&self) -> Option<bool> {
            match self {
                Json::Bool(b) => Some(*b),
                _ => None,
            }
        }

        /// This number as `f64`. `None` for any non-number.
        #[must_use]
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Json::Num(raw) => raw.parse::<f64>().ok(),
                _ => None,
            }
        }

        /// This number as an exact `i64`.
        ///
        /// Integral tokens are parsed as integers, so no precision is lost.
        /// A token with a fraction or exponent is accepted only if it is
        /// integral and in range (JSON writers emit `1e3` for 1000), and
        /// `1e-06` correctly yields `None`.
        #[must_use]
        pub fn as_i64(&self) -> Option<i64> {
            let Json::Num(raw) = self else { return None };
            if let Ok(exact) = raw.parse::<i64>() {
                return Some(exact);
            }
            let approx = raw.parse::<f64>().ok()?;
            // 2^53 is where f64 stops representing consecutive integers, so
            // beyond it "integral" no longer implies "the written value".
            if approx.fract() == 0.0 && approx.abs() < 9_007_199_254_740_992.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "guarded: fract() == 0 and |value| < 2^53"
                )]
                return Some(approx as i64);
            }
            None
        }
    }

    /// Parse a complete JSON document.
    ///
    /// # Errors
    /// On any syntax error, or trailing content after the top-level value.
    pub fn parse(src: &str) -> Result<Json> {
        let mut p = Parser {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        };
        p.skip_ws();
        let value = p.value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            bail!("trailing JSON content at byte {}", p.pos);
        }
        Ok(value)
    }

    struct Parser<'a> {
        src: &'a str,
        bytes: &'a [u8],
        pos: usize,
    }

    impl Parser<'_> {
        fn skip_ws(&mut self) {
            while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }

        fn peek(&self) -> Result<u8> {
            self.bytes
                .get(self.pos)
                .copied()
                .ok_or_else(|| anyhow!("unexpected end of JSON at byte {}", self.pos))
        }

        fn eat(&mut self, expect: u8) -> Result<()> {
            let got = self.peek()?;
            if got != expect {
                bail!(
                    "expected `{}` at byte {}, found `{}`",
                    expect as char,
                    self.pos,
                    got as char
                );
            }
            self.pos += 1;
            Ok(())
        }

        fn value(&mut self) -> Result<Json> {
            match self.peek()? {
                b'{' => self.object(),
                b'[' => self.array(),
                b'"' => Ok(Json::Str(self.string()?)),
                b't' => self.literal("true").map(|()| Json::Bool(true)),
                b'f' => self.literal("false").map(|()| Json::Bool(false)),
                b'n' => self.literal("null").map(|()| Json::Null),
                b'-' | b'0'..=b'9' => self.number(),
                other => bail!("unexpected byte `{}` at {}", other as char, self.pos),
            }
        }

        fn literal(&mut self, word: &str) -> Result<()> {
            if self.src[self.pos..].starts_with(word) {
                self.pos += word.len();
                Ok(())
            } else {
                bail!("expected `{word}` at byte {}", self.pos)
            }
        }

        fn number(&mut self) -> Result<Json> {
            let start = self.pos;
            if self.peek()? == b'-' {
                self.pos += 1;
            }
            while matches!(
                self.bytes.get(self.pos),
                Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
            ) {
                self.pos += 1;
            }
            let raw = &self.src[start..self.pos];
            // Validate here so a malformed token fails at parse rather than
            // silently reading as `None` at the first typed access.
            if raw.parse::<f64>().is_err() {
                bail!("malformed JSON number `{raw}` at byte {start}");
            }
            Ok(Json::Num(raw.to_owned()))
        }

        fn string(&mut self) -> Result<String> {
            self.eat(b'"')?;
            let mut out = String::new();
            loop {
                // Runs stop only on ASCII `"` / `\`, so every slice below lands
                // on a char boundary of the (valid UTF-8) source.
                let run = self.pos;
                while !matches!(self.bytes.get(self.pos), None | Some(b'"' | b'\\')) {
                    self.pos += 1;
                }
                out.push_str(&self.src[run..self.pos]);
                if self.peek()? == b'"' {
                    self.pos += 1;
                    return Ok(out);
                }
                self.pos += 1; // the backslash
                self.escape(&mut out)?;
            }
        }

        fn escape(&mut self, out: &mut String) -> Result<()> {
            let esc = self.peek()?;
            self.pos += 1;
            let ch = match esc {
                b'"' => '"',
                b'\\' => '\\',
                b'/' => '/',
                b'b' => '\u{8}',
                b'f' => '\u{c}',
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'u' => {
                    let hi = self.hex4()?;
                    // A high surrogate is only a character together with the
                    // low one that must follow it.
                    if (0xD800..0xDC00).contains(&hi) {
                        self.eat(b'\\')?;
                        self.eat(b'u')?;
                        let lo = self.hex4()?;
                        if !(0xDC00..0xE000).contains(&lo) {
                            bail!("unpaired JSON surrogate \\u{hi:04X} at byte {}", self.pos);
                        }
                        let combined =
                            0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
                        char::from_u32(combined)
                            .ok_or_else(|| anyhow!("bad surrogate pair at byte {}", self.pos))?
                    } else {
                        char::from_u32(u32::from(hi))
                            .ok_or_else(|| anyhow!("bad \\u{hi:04X} escape at byte {}", self.pos))?
                    }
                }
                other => bail!("unknown escape `\\{}` at byte {}", other as char, self.pos),
            };
            out.push(ch);
            Ok(())
        }

        fn hex4(&mut self) -> Result<u16> {
            let end = self.pos + 4;
            let digits = self
                .src
                .get(self.pos..end)
                .ok_or_else(|| anyhow!("truncated \\u escape at byte {}", self.pos))?;
            let value = u16::from_str_radix(digits, 16)
                .map_err(|_| anyhow!("bad \\u escape `{digits}` at byte {}", self.pos))?;
            self.pos = end;
            Ok(value)
        }

        fn array(&mut self) -> Result<Json> {
            self.eat(b'[')?;
            let mut items = Vec::new();
            self.skip_ws();
            if self.peek()? == b']' {
                self.pos += 1;
                return Ok(Json::Arr(items));
            }
            loop {
                self.skip_ws();
                items.push(self.value()?);
                self.skip_ws();
                match self.peek()? {
                    b',' => self.pos += 1,
                    b']' => {
                        self.pos += 1;
                        return Ok(Json::Arr(items));
                    }
                    other => bail!(
                        "expected `,` or `]` at {}, found `{}`",
                        self.pos,
                        other as char
                    ),
                }
            }
        }

        fn object(&mut self) -> Result<Json> {
            self.eat(b'{')?;
            let mut entries = Vec::new();
            self.skip_ws();
            if self.peek()? == b'}' {
                self.pos += 1;
                return Ok(Json::Obj(entries));
            }
            loop {
                self.skip_ws();
                let key = self.string()?;
                self.skip_ws();
                self.eat(b':')?;
                self.skip_ws();
                let value = self.value()?;
                // Last occurrence wins, matching `serde_json`'s map semantics,
                // so swapping this reader for that crate cannot change which
                // value a duplicated key resolves to.
                if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key) {
                    slot.1 = value;
                } else {
                    entries.push((key, value));
                }
                self.skip_ws();
                match self.peek()? {
                    b',' => self.pos += 1,
                    b'}' => {
                        self.pos += 1;
                        return Ok(Json::Obj(entries));
                    }
                    other => bail!(
                        "expected `,` or `}}` at {}, found `{}`",
                        self.pos,
                        other as char
                    ),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
    const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

    fn checkpoint_dir() -> Option<PathBuf> {
        let dir = std::env::var_os(CKPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(CKPT_DEFAULT));
        dir.join("config.json").is_file().then_some(dir)
    }

    // ── the real checkpoint ──────────────────────────────────────────────

    /// Every derived field, against the on-box `config.json` +
    /// `generation_config.json`.
    #[test]
    fn real_config_derives_every_field() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse the real config.json");

        // core
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.vocab_size, 248_320);
        assert!(
            (cfg.rms_norm_eps - 1e-6).abs() < 1e-12,
            "{}",
            cfg.rms_norm_eps
        );
        assert_eq!(cfg.hidden_act, "silu");
        assert!(!cfg.tie_word_embeddings, "lm_head is its own tensor");

        // hyper-connections
        assert_eq!(cfg.hc_count, 4);
        assert_eq!(cfg.hc_lowrank, 320);
        assert_eq!(cfg.hc_hidden_size(), 10_240);

        // full attention
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert!(!cfg.attention_bias);
        assert_eq!(cfg.q_dim(), 6144);
        assert_eq!(cfg.kv_dim(), 512);
        assert_eq!(cfg.rotary_dim, 64, "int(256 * 0.25)");
        assert!((cfg.partial_rotary_factor - 0.25).abs() < 1e-9);
        assert!((cfg.rope_theta - 1e7).abs() < 1.0, "{}", cfg.rope_theta);
        assert_eq!(cfg.max_position_embeddings, 262_144);

        // linear attention
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, 48);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.linear_key_dim(), 2048);
        assert_eq!(cfg.linear_value_dim(), 6144);
        assert_eq!(cfg.linear_conv_dim(), 10_240);
        assert_eq!(
            cfg.output_gate,
            GateActivation::Sigmoid,
            "output_gate_type=sigmoid must beat hidden_act=silu"
        );

        // MoE
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.num_experts_per_tok, 10);
        assert_eq!(cfg.moe_intermediate_size, 640);
        assert_eq!(cfg.shared_expert_intermediate_size, 640);
        assert!(cfg.norm_topk_prob, "absent key ⇒ reference default true");

        // PLE / n-gram
        assert_eq!(cfg.ple_embed_dim, 2560);
        assert_eq!(cfg.ple_conv_kernel_size, 4);
        assert_eq!(cfg.ngram_size, 3);
        assert_eq!(cfg.heads_per_ngram, 8);
        assert_eq!(cfg.ngram_vocab_size_base, 20_000_000);
        assert_eq!(cfg.make_ngram_vocab_size_divisible_by, 128);
        assert_eq!(cfg.ngram_seed, 1234, "absent key ⇒ reference default");
        assert_eq!(cfg.split_ngram_parts, 128, "NOT the class default of 512");
        assert_eq!(cfg.ngram_heads(), 16);
        assert_eq!(cfg.ngram_head_dim(), 160, "the [2500012, 160] shard width");
        assert_eq!(cfg.num_conv_states(), 3);

        // QSA
        let qsa = cfg.indexer.expect("the checkpoint enables QSA");
        assert_eq!(qsa.n_heads, 4);
        assert_eq!(qsa.kv_heads, 1);
        assert_eq!(qsa.head_dim, 128);
        assert_eq!(qsa.budget, 2048);
        assert_eq!(qsa.compress_ratio, 4);
        assert_eq!(qsa.block_topk(), 512);
        assert_eq!(qsa.dense_below_or_equal(), 2051);

        // tokens
        assert_eq!(cfg.bos_token_id, Some(248_044));
        assert_eq!(cfg.eos_token_id, 248_044);
        assert_eq!(
            cfg.stop_token_ids,
            vec![248_044, 248_046],
            "text_config eos first, then generation_config's extra"
        );

        // serving policy
        assert_eq!(cfg.max_context, DEFAULT_MAX_CONTEXT);
        assert!(cfg.max_context <= qsa.dense_below_or_equal());
    }

    /// The interleave, spelled out: 36 linear + 12 full, full on every 4th
    /// layer counting from one.
    #[test]
    fn real_config_layer_types_are_three_linear_then_one_full() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse the real config.json");

        assert_eq!(cfg.layer_types.len(), 48);
        let linear = cfg
            .layer_types
            .iter()
            .filter(|t| **t == Qwen4LayerType::LinearAttention)
            .count();
        let full = cfg.layer_types.len() - linear;
        assert_eq!((linear, full), (36, 12));

        let full_layers: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.layer_type(i) == Qwen4LayerType::FullAttention)
            .collect();
        let expect: Vec<usize> = (0..12).map(|k| 4 * k + 3).collect();
        assert_eq!(
            full_layers, expect,
            "full attention on layers 3, 7, 11, ..."
        );

        // The `layer_types` list and the `full_attention_interval` fallback must
        // agree, or one of the two readings is wrong.
        let derived = parse_layer_types(
            &json::parse(r#"{"full_attention_interval": 4}"#).unwrap(),
            48,
        )
        .unwrap();
        assert_eq!(derived, cfg.layer_types);
    }

    /// The one-indexing trap: the file says `[2]`, the PLE lives on layer 1.
    #[test]
    fn real_config_ple_layer_is_one_not_two() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse the real config.json");

        let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let root = json::parse(&raw).unwrap();
        let file_ids: Vec<i64> = root
            .get("text_config")
            .and_then(|t| t.get("ple_layer_ids"))
            .and_then(json::Json::as_array)
            .expect("ple_layer_ids in the file")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(file_ids, vec![2], "the file's one-indexed value");

        assert_eq!(cfg.ple_layer_ids, vec![1], "zero-indexed, subtract one");
        assert!(cfg.is_ple_layer(1));
        assert!(!cfg.is_ple_layer(2));
        assert_eq!(cfg.ple_layer_index(1), Some(0));
        assert_eq!(cfg.ple_layer_index(2), None);
        assert_eq!(
            cfg.layer_type(1),
            Qwen4LayerType::LinearAttention,
            "the reference forbids PLE on an attention layer"
        );
    }

    /// The hash config handed to `qwen4_ple` must match the values that module
    /// pinned independently against the checkpoint's own hash tensors.
    #[test]
    fn real_config_reproduces_the_pinned_ngram_hash_config() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse the real config.json");
        assert_eq!(cfg.ngram_hash_config(0), NGramHashConfig::qwen4_exp());
    }

    // ── synthetic: each derivation on its own ────────────────────────────

    /// A minimal text_config that parses, for tests that vary one thing.
    fn minimal(extra: &str) -> String {
        format!(
            r#"{{"text_config": {{
                "hidden_size": 64, "num_hidden_layers": 4,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "vocab_size": 100, "rms_norm_eps": 1e-06,
                "head_dim": 16, "rope_theta": 10000.0,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "num_experts": 8, "num_experts_per_tok": 2,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "eos_token_id": 7
                {extra}
            }}}}"#
        )
    }

    #[test]
    fn full_attention_interval_fallback_matches_the_reference_formula() {
        let cfg = Qwen4ExpConfig::from_json(&minimal(r#", "full_attention_interval": 2"#), None)
            .expect("parse");
        use Qwen4LayerType::{FullAttention as F, LinearAttention as L};
        assert_eq!(cfg.layer_types, vec![L, F, L, F]);
    }

    #[test]
    fn missing_interval_defaults_to_four() {
        let cfg = Qwen4ExpConfig::from_json(&minimal(""), None).expect("parse");
        use Qwen4LayerType::{FullAttention as F, LinearAttention as L};
        assert_eq!(cfg.layer_types, vec![L, L, L, F]);
    }

    #[test]
    fn explicit_layer_types_win_and_qsa_spelling_is_accepted() {
        let cfg = Qwen4ExpConfig::from_json(
            &minimal(
                r#", "full_attention_interval": 2,
                   "layer_types": ["qwen_sparse_attention", "linear_attention",
                                   "linear_attention", "full_attention"]"#,
            ),
            None,
        )
        .expect("parse");
        use Qwen4LayerType::{FullAttention as F, LinearAttention as L};
        assert_eq!(cfg.layer_types, vec![F, L, L, F]);
    }

    #[test]
    fn unknown_layer_type_is_rejected() {
        let err = Qwen4ExpConfig::from_json(
            &minimal(
                r#", "layer_types": ["sliding_window", "linear_attention",
                                          "linear_attention", "full_attention"]"#,
            ),
            None,
        )
        .expect_err("sliding_window is not a qwen4_exp layer type");
        assert!(format!("{err}").contains("sliding_window"), "{err}");
    }

    #[test]
    fn layer_types_length_must_match_the_layer_count() {
        let err =
            Qwen4ExpConfig::from_json(&minimal(r#", "layer_types": ["linear_attention"]"#), None)
                .expect_err("1 entry for 4 layers");
        assert!(format!("{err}").contains("layer_types"), "{err}");
    }

    #[test]
    fn ple_layer_ids_are_one_indexed_and_deduped() {
        let cfg = Qwen4ExpConfig::from_json(&minimal(r#", "ple_layer_ids": [3, 1, 3]"#), None)
            .expect("parse");
        assert_eq!(cfg.ple_layer_ids, vec![0, 2]);
        assert_eq!(cfg.num_conv_states(), 3);
        assert_eq!(
            cfg.ple_layer_index(2),
            Some(1),
            "second PLE layer ⇒ index 1"
        );
    }

    #[test]
    fn ple_layer_id_zero_is_rejected_as_out_of_the_one_indexed_range() {
        let err = Qwen4ExpConfig::from_json(&minimal(r#", "ple_layer_ids": [0]"#), None)
            .expect_err("0 is not a valid one-indexed layer");
        assert!(format!("{err}").contains("one-indexed"), "{err}");
    }

    #[test]
    fn ple_on_a_full_attention_layer_is_rejected() {
        // interval 2 ⇒ layers 1 and 3 are full; one-indexed 2 ⇒ layer 1.
        let err = Qwen4ExpConfig::from_json(
            &minimal(r#", "full_attention_interval": 2, "ple_layer_ids": [2]"#),
            None,
        )
        .expect_err("the reference forbids PLE on attention layers");
        assert!(format!("{err}").contains("linear_attention"), "{err}");
    }

    #[test]
    fn no_ple_means_one_conv_state() {
        let cfg = Qwen4ExpConfig::from_json(&minimal(""), None).expect("parse");
        assert!(cfg.ple_layer_ids.is_empty());
        assert_eq!(cfg.num_conv_states(), 1);
    }

    #[test]
    fn rotary_dim_truncates_and_reads_rope_parameters_first() {
        // head_dim 16: the flat 0.5 would give 8, rope_parameters' 0.3 gives
        // int(4.8) = 4. rope_parameters must win, and it must truncate.
        let cfg = Qwen4ExpConfig::from_json(
            &minimal(
                r#", "partial_rotary_factor": 0.5,
                   "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.3}"#,
            ),
            None,
        )
        .expect("parse");
        assert_eq!(cfg.rotary_dim, 4, "int(16 * 0.3), not round");
        assert!((cfg.rope_theta - 1e7).abs() < 1.0);
    }

    #[test]
    fn scaled_rope_is_refused_rather_than_silently_ignored() {
        let err = Qwen4ExpConfig::from_json(
            &minimal(r#", "rope_parameters": {"rope_theta": 1e7, "rope_type": "yarn"}"#),
            None,
        )
        .expect_err("yarn needs an inv_freq transform this lane lacks");
        assert!(format!("{err}").contains("yarn"), "{err}");
    }

    #[test]
    fn output_gate_falls_back_to_hidden_act_when_absent() {
        let cfg =
            Qwen4ExpConfig::from_json(&minimal(r#", "hidden_act": "silu""#), None).expect("parse");
        assert_eq!(cfg.output_gate, GateActivation::Silu);

        let cfg = Qwen4ExpConfig::from_json(
            &minimal(r#", "hidden_act": "silu", "output_gate_type": "sigmoid""#),
            None,
        )
        .expect("parse");
        assert_eq!(cfg.output_gate, GateActivation::Sigmoid);
    }

    #[test]
    fn non_silu_hidden_act_is_refused() {
        let err = Qwen4ExpConfig::from_json(&minimal(r#", "hidden_act": "gelu""#), None)
            .expect_err("the MoE path is SwiGLU-only");
        assert!(format!("{err}").contains("gelu"), "{err}");
    }

    #[test]
    fn stop_tokens_merge_generation_config_without_duplicating() {
        let cfg = Qwen4ExpConfig::from_json(
            &minimal(""),
            Some(r#"{"eos_token_id": [9, 7], "pad_token_id": 7}"#),
        )
        .expect("parse");
        assert_eq!(cfg.eos_token_id, 7, "the PLE separator is text_config's");
        assert_eq!(cfg.stop_token_ids, vec![7, 9]);
    }

    #[test]
    fn a_list_eos_in_text_config_takes_its_head_as_the_ple_separator() {
        let mut cfg_json = minimal("");
        cfg_json = cfg_json.replace(r#""eos_token_id": 7"#, r#""eos_token_id": [7, 9]"#);
        let cfg = Qwen4ExpConfig::from_json(&cfg_json, None).expect("parse");
        assert_eq!(cfg.eos_token_id, 7);
        assert_eq!(cfg.stop_token_ids, vec![7, 9]);
    }

    #[test]
    fn partial_qsa_config_is_refused() {
        let err = Qwen4ExpConfig::from_json(
            &minimal(r#", "indexer_n_heads": 4, "indexer_kv_heads": 1"#),
            None,
        )
        .expect_err("three of the five QSA keys are missing");
        assert!(format!("{err}").contains("indexer_budget"), "{err}");
    }

    #[test]
    fn qsa_requires_a_single_kv_head() {
        let err = Qwen4ExpConfig::from_json(
            &minimal(
                r#", "indexer_n_heads": 4, "indexer_kv_heads": 2, "indexer_head_dim": 16,
                   "indexer_budget": 64, "indexer_compress_ratio": 4"#,
            ),
            None,
        )
        .expect_err("indexer_kv_heads must be 1");
        assert!(format!("{err}").contains("indexer_kv_heads"), "{err}");
    }

    #[test]
    fn max_context_beyond_the_dense_indexer_bound_is_refused() {
        let cfg = Qwen4ExpConfig::from_json(
            &minimal(
                r#", "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 16,
                   "indexer_budget": 64, "indexer_compress_ratio": 4"#,
            ),
            None,
        )
        .expect("parse");
        let bound = cfg.indexer.unwrap().dense_below_or_equal();
        assert_eq!(bound, 67, "64 + 4 - 1");
        assert_eq!(
            cfg.max_context, bound,
            "a budget below ours caps the default instead of failing the build"
        );
        assert!(
            cfg.clone().with_max_context(bound).is_ok(),
            "exactly at the bound is still exact"
        );
        let err = cfg
            .with_max_context(bound + 1)
            .expect_err("one past the bound the stub is no longer the model");
        assert!(format!("{err}").contains("max_context"), "{err}");
    }

    #[test]
    fn json_duplicate_keys_resolve_last_wins_like_serde_json() {
        let v = json::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_i64(), Some(3));
        assert_eq!(
            v.as_object().unwrap().len(),
            2,
            "the duplicate replaces, it does not append"
        );
    }

    #[test]
    fn a_missing_required_key_names_itself() {
        let err = Qwen4ExpConfig::from_json(r#"{"text_config": {"hidden_size": 64}}"#, None)
            .expect_err("num_hidden_layers is missing");
        assert!(format!("{err}").contains("num_hidden_layers"), "{err}");
    }

    #[test]
    fn head_dim_falls_back_only_when_it_divides_exactly() {
        let mut cfg_json = minimal("");
        cfg_json = cfg_json.replace(r#""head_dim": 16,"#, "");
        let cfg = Qwen4ExpConfig::from_json(&cfg_json, None).expect("64 / 4 is exact");
        assert_eq!(cfg.head_dim, 16);

        let ragged = cfg_json.replace(r#""num_attention_heads": 4"#, r#""num_attention_heads": 6"#);
        let err = Qwen4ExpConfig::from_json(&ragged, None)
            .expect_err("64 / 6 is not exact, and the real model is exactly this case");
        assert!(format!("{err}").contains("head_dim"), "{err}");
    }

    // ── the JSON reader ──────────────────────────────────────────────────

    #[test]
    fn json_reads_the_shapes_config_json_actually_uses() {
        let v = json::parse(
            r#"{"a": [1, -2, 3.5e2], "b": {"c": null, "d": true},
                "e": "x\ty\u00e9\ud83d\ude00", "f": 1e-06, "g": []}"#,
        )
        .expect("parse");
        let a = v.get("a").unwrap().as_array().unwrap();
        assert_eq!(a[0].as_i64(), Some(1));
        assert_eq!(a[1].as_i64(), Some(-2));
        assert_eq!(a[2].as_i64(), Some(350), "3.5e2 is integral");
        assert_eq!(v.get("b").unwrap().get("c"), Some(&json::Json::Null));
        assert_eq!(v.get("b").unwrap().get("d").unwrap().as_bool(), Some(true));
        assert_eq!(v.get("e").unwrap().as_str(), Some("x\ty\u{e9}\u{1f600}"));
        assert_eq!(
            v.get("f").unwrap().as_i64(),
            None,
            "1e-06 is not an integer"
        );
        assert!((v.get("f").unwrap().as_f64().unwrap() - 1e-6).abs() < 1e-12);
        assert_eq!(v.get("g").unwrap().as_array(), Some(&[][..]));
        assert_eq!(v.get("missing"), None);
    }

    #[test]
    fn json_keeps_large_integers_exact() {
        // 20_000_003 is one of the n-gram head primes; an f64 round-trip that
        // rounded it would read a different table row.
        let v = json::parse(r#"{"p": 20000003, "q": 9007199254740993}"#).unwrap();
        assert_eq!(v.get("p").unwrap().as_i64(), Some(20_000_003));
        assert_eq!(
            v.get("q").unwrap().as_i64(),
            Some(9_007_199_254_740_993),
            "beyond 2^53, so only the verbatim token is right"
        );
    }

    #[test]
    fn json_rejects_malformed_documents() {
        for bad in [
            "{",
            r#"{"a": }"#,
            r#"{"a": 1,}"#,
            r#"{"a": 1} tail"#,
            r#"{"a": "unterminated}"#,
            r#"{"a": 1.2.3}"#,
            r#"{"a": "\q"}"#,
            r#"{"a": "\ud83d"}"#,
        ] {
            assert!(json::parse(bad).is_err(), "should reject `{bad}`");
        }
    }
}
