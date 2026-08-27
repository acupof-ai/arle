//! `qwen4_exp` (Qwen3.8-Flash-Next) weight residency: checkpoint bytes onto the
//! device, as slab-backed `(buffer, offset, len)` bindings.
//!
//! This is the upload half of [`crate::qwen4_names`]: that module says what each
//! of the checkpoint's 296,475 tensor names *is* and where it is *meant* to
//! live; this one plans the byte budget, sub-allocates a
//! `vulkan_sys::SlabAllocator` and fills it. The forward asks it for a binding
//! by the checkpoint's own HF tensor name.
//!
//! # Why the keys are tensor names and not an invented slot enum
//!
//! Every device tensor is addressed by the string the checkpoint spells it with
//! (`model.language_model.layers.7.self_attn.q_proj.weight`), built by
//! [`layer_tensor_name`] / [`expert_tensor_name`] / [`MIXER_PREFIX`]. A slot
//! enum would be a second, hand-maintained vocabulary that can silently drift
//! from the classifier's; the name is already the checkpoint's authority, and
//! [`crate::qwen4_names::classify_qwen4_tensor`] already refuses one it does not
//! recognise. Missing-weight lookups return `Err`, not `None`, so a typo in the
//! forward fails at the binding and not as a zero-filled buffer.
//!
//! # Device format: three tiers, and the kernel that forces each
//!
//! [`Qwen4DeviceFormat`] is not a preference. Each assignment is the format the
//! kernel that consumes the tensor actually binds:
//!
//! | tier | what | why |
//! |---|---|---|
//! | [`Nvfp4`](Qwen4DeviceFormat::Nvfp4) | the 512 routed experts x 3 projections x 48 layers | `Kernel::GemvIdNvfp4` reads ggml `block_nvfp4` directly; dequantizing would take 63.28 GiB to ~253 GiB |
//! | [`F32`](Qwen4DeviceFormat::F32) | every hyper-connection weight, all norms, the routers, both conv1ds, the linear-attn scalar projections | their consumers — the vendored `rms_norm.comp`, `qwen36_router_gemv.comp`, `qwen4_hc_mix.comp`, `qwen4_hc_combine.comp`, `qwen4_ple_gate.comp`, `qwen4_ple_conv.comp` — all declare `readonly buffer { float w[]; }`. F16 there is silent garbage, not a type error |
//! | [`F16`](Qwen4DeviceFormat::F16) | the dense GEMV weights: attention, linear-attn, shared expert, PLE key/value, `lm_head` | BF16 in the file; F16 halves the 6.70 GiB these would cost at F32, and the model does not otherwise fit |
//!
//! **Dynamic range is not the risk it looks like.** BF16 carries f32's 8-bit
//! exponent and F16 only 5, so the conversion could in principle overflow to
//! inf. Measured over `lm_head`, `layers.0.linear_attn.{in_proj_qkv, out_proj}`,
//! `shared_expert.down_proj`, `attn_hyper_connection.input_mix_weight_up` and
//! `layers.1.ple.key_proj`: the largest magnitude anywhere is 3.94, four orders
//! of magnitude under F16's 65504, and NOTHING overflows. Underflow is the only
//! loss and it is negligible — 2424 of `lm_head`'s 635.7 M elements (4e-6) sit
//! below F16's smallest subnormal and flush to zero, from a value already below
//! 6e-8.
//!
//! **Consumer note for the F16 tier.** As of this writing `vulkan_kernels::Kernel`
//! registers no F16 GEMV — `mul_mat_vec.comp` is built with `DATA_A_NVFP4` and
//! nothing else — so a forward that wants to multiply by these weights needs
//! either a `mul_mat_vec_f16` build of the same vendored shader (the cheap fix,
//! and the reason the tier is F16 rather than something exotic) or, for a
//! few-layer bring-up where the extra 6.7 GiB is not spent,
//! [`Qwen4UploadConfig::dense_format`] = [`Qwen4DeviceFormat::F32`], which
//! `qwen36_router_gemv.comp` can read today.
//!
//! Measured against the on-box checkpoint (header arithmetic, `plan_qwen4_upload`
//! over all 206 shards): 63.28 GiB packed experts + 6.70 GiB F16 + 2.66 GiB F32
//! = **72.64 GiB**, against a 74.43 GiB device-local heap. That is why
//! [`Qwen4UploadConfig::reserve_bytes`] defaults to 1 GiB and not to the 3 GiB
//! the qwen35 loader reserves: this model does not fit with a 3 GiB reserve, and
//! it does not need one.
//!
//! # The slab size is chosen, not taken
//!
//! Those 72.64 GiB are a FLOOR — slabs have tails. Cutting the residency into
//! slabs of the device's `maxMemoryAllocationSize` (2 GiB), which is what a
//! loader naturally does, commits 74.00 GiB and leaves 0.43 GiB of heap: not
//! enough for the reserve, and the kind of shortfall that surfaces as an
//! `ERROR_OUT_OF_DEVICE_MEMORY` from the KV allocation long after the weights
//! are up. [`Qwen4Plan::choose_packing`] sweeps the slab size against the actual
//! item sizes and lands on 1488 MiB / 72.656 GiB / 0.02% waste, recovering 1.34
//! GiB. The tail volume is not monotone in slab size — see [`Qwen4Packing`] for
//! the measured table and why.
//!
//! The 2.66 GiB F32 tier is dominated by the hyper-connection
//! `input_mix_weight_{down,up}` pair: 97 gated-residual sites x 6.55 M elements
//! = 2.37 GiB, which would be 1.18 GiB at F16. It is spent because
//! `qwen4_hc_mix.comp` binding 1 and `qwen36_router_gemv.comp` binding 1 are
//! `float[]`; a F16 variant of those two shaders is worth 1.18 GiB of heap if
//! the headroom ever gets tight.
//!
//! # The `1 + w` norm fold, and the three norms that must NOT get it
//!
//! `Qwen4ExpTextRMSNorm` applies `x * inv_rms * (1.0 + weight)` and the
//! parameter is zero-initialised, so the bias has to be folded somewhere. Where
//! depends on the consumer, and the checkpoint gives no hint:
//!
//! - **Folded at load** (`1.0 + w` in the device buffer): `hc_norm` at all 97
//!   gated-residual sites, plus `self_attn.{q,k}_norm` and
//!   `self_attn.indexer.{q,k}_layernorm`. These are read by the VENDORED
//!   `rms_norm.comp`, which applies the plain weight and which
//!   `vulkan-kernels`' build script never edits.
//! - **Stored raw**: `ple.norm_key`, `ple.norm_query`, `ple.norm_conv`. Their
//!   consumer is ARLE's own `qwen4_ple_gate.comp`, which already spells
//!   `(1.0 + norm_key_w[...])` in the shader — folding here would apply the bias
//!   twice and is not an error any driver can catch.
//! - **Stored raw**: `linear_attn.norm`. That one is `Qwen4ExpTextRMSNormGated`,
//!   a DIFFERENT class with a plain `weight *` and a ones-initialised parameter.
//!   Folding a ones-initialised gain to 2.0 doubles the linear-attention output.
//!
//! [`folds_norm_bias`] is that rule;
//! `norm_bias_fold_covers_exactly_the_vendored_rms_norm_consumers` pins the table
//! and `subset_upload_lands_byte_exact_on_device` pins the actual device bytes,
//! in both directions.
//!
//! # What does not go to the device
//!
//! [`Qwen4HostTables`] keeps `embed_tokens` (1.18 GiB) and the 128-shard FP8
//! n-gram table (47.68 GiB) as slices borrowed straight out of the checkpoint's
//! mmaps. Uploading either is not a tuning choice: the n-gram table alone is
//! 64% of the heap and a token reads 16 of its 320,001,536 rows.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, anyhow, bail, ensure};

use infer_gguf::dequant::{BLOCK_NVFP4_BYTES, QK_NVFP4, QK_NVFP4_SUB};
use infer_gguf::safetensors::{SafeTensorInfo, SafeTensorsDir};

use crate::qwen4_names::{
    ExpertProj, HcPart, HcSite, Nvfp4Part, Qwen4Residency, Qwen4Stream, Qwen4TensorKind,
    Qwen4TensorRole, classify_qwen4_tensor,
};

/// `model.language_model.` — the text stream's prefix.
pub const TEXT_PREFIX: &str = "model.language_model.";
/// The stream-level `hyper_connection_mixer` (`use_combine=false`, so no
/// `block_inject_weight`); it is this model's missing final norm.
pub const MIXER_PREFIX: &str = "model.language_model.hyper_connection_mixer";
/// Untied output projection (`tie_word_embeddings: false`).
pub const LM_HEAD_NAME: &str = "lm_head.weight";
/// The host-resident token embedding table.
pub const EMBED_TOKENS_NAME: &str = "model.language_model.embed_tokens.weight";

/// Step of the slab-size sweep in [`Qwen4Plan::choose_packing`].
///
/// 4 MiB over the ~800 MiB of usable range is ~200 candidate sizes, each a
/// first-fit pass over ~1200 items — well under a second, against the 1.34 GiB
/// it recovers (see [`Qwen4Packing`]).
pub const SLAB_SWEEP_STEP_BYTES: u64 = 4 << 20;

/// Device-local bytes held back for the KV cache, the linear-attention
/// recurrent state, the activation arena and the descriptor pools.
///
/// 1 GiB, not the 3 GiB `crate::loader` reserves. The model does not fit with 3
/// GiB, and it does not need it: measured from the config, this model's
/// device-side state is 48 MiB of KV cache (12 full-attention layers x 2048
/// context x 2 KV heads x 256 head_dim x K+V x f16), 108 MiB of recurrent state
/// (36 linear layers x 48 value heads x 128 x 128 x f32), ~6 MiB of conv rings,
/// and an arena whose widest tensors are the 10240-wide residual and the 248320
/// f32 logits. 1 GiB is roughly 4x that, and — with the slab sweep in
/// [`Qwen4Plan::choose_packing`] — it is available: the full residency commits
/// 72.66 GiB of the 74.43 GiB heap.
pub const DEFAULT_RESERVE_BYTES: u64 = 1 << 30;

// ---------------------------------------------------------------- name builders

/// `model.language_model.layers.<layer>.<suffix>`.
#[must_use]
pub fn layer_tensor_name(layer: usize, suffix: &str) -> String {
    format!("{TEXT_PREFIX}layers.{layer}.{suffix}")
}

/// The HF spelling of one NVFP4 component of one routed expert, e.g.
/// `model.language_model.layers.0.mlp.experts.7.gate_proj.weight_scale`.
#[must_use]
pub fn expert_tensor_name(layer: usize, expert: u32, proj: ExpertProj, part: Nvfp4Part) -> String {
    let proj = match proj {
        ExpertProj::Gate => "gate",
        ExpertProj::Up => "up",
        ExpertProj::Down => "down",
    };
    let part = match part {
        Nvfp4Part::Packed => "weight",
        Nvfp4Part::BlockScale => "weight_scale",
        Nvfp4Part::GlobalScale => "weight_scale_2",
        Nvfp4Part::InputScale => "input_scale",
    };
    format!("{TEXT_PREFIX}layers.{layer}.mlp.experts.{expert}.{proj}_proj.{part}")
}

/// The synthetic name of one layer's 512-expert stack for `proj`.
///
/// No such tensor exists in the checkpoint: `Kernel::GemvIdNvfp4` needs every
/// expert of one projection CONTIGUOUS (`mul_mat_vec_base.glsl` lands on expert
/// `e` at `e * batch_stride_a / QUANT_K`), so the 512 per-expert planes become
/// one suballocation. The `_stack` suffix cannot collide with a real name — the
/// classifier would reject it.
#[must_use]
pub fn expert_stack_name(layer: usize, proj: ExpertProj) -> String {
    let proj = match proj {
        ExpertProj::Gate => "gate",
        ExpertProj::Up => "up",
        ExpertProj::Down => "down",
    };
    format!("{TEXT_PREFIX}layers.{layer}.mlp.experts.{proj}_proj._stack")
}

/// Prefix of one `Qwen4ExpTextGatedResidual`'s four weights.
///
/// `(Some(l), Attn | Mlp)` for a layer's two sites, `(None, Mixer)` for the
/// stream-level one. The other two combinations do not exist in this
/// architecture and are refused rather than formatted into a name that will
/// simply not be resident.
pub fn hyper_connection_prefix(layer: Option<usize>, site: HcSite) -> Result<String> {
    Ok(match (layer, site) {
        (Some(l), HcSite::Attn) => layer_tensor_name(l, "attn_hyper_connection"),
        (Some(l), HcSite::Mlp) => layer_tensor_name(l, "mlp_hyper_connection"),
        (None, HcSite::Mixer) => MIXER_PREFIX.to_string(),
        (None, site) => bail!("qwen4 upload: {site:?} hyper-connection needs a layer index"),
        (Some(l), HcSite::Mixer) => {
            bail!("qwen4 upload: the hyper_connection_mixer is stream-level, not layer {l}")
        }
    })
}

// ------------------------------------------------------------------- policy

/// The three device byte formats this model uses. See the module docs for which
/// kernel forces each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4DeviceFormat {
    /// IEEE binary16, converted from the checkpoint's BF16 with
    /// round-to-nearest-even.
    F16,
    /// IEEE binary32.
    F32,
    /// ggml `block_nvfp4`: four UE4M3 sub-block scales then 32 packed E2M1
    /// nibble bytes per 64 values, 36 B/block.
    Nvfp4,
}

impl Qwen4DeviceFormat {
    /// Device bytes for `n` LOGICAL elements.
    ///
    /// `None` for an NVFP4 width that is not a whole number of 64-value blocks:
    /// `mul_mat_vec.comp` indexes A in block units, so a ragged row width makes
    /// row `r > 0` read the previous row's tail.
    #[must_use]
    pub const fn bytes_for(self, ncols: usize, nrows: usize) -> Option<u64> {
        match self {
            Self::F16 => Some((ncols * nrows * 2) as u64),
            Self::F32 => Some((ncols * nrows * 4) as u64),
            Self::Nvfp4 => match nvfp4_row_bytes(ncols) {
                Some(row) => Some((row * nrows) as u64),
                None => None,
            },
        }
    }
}

/// Device bytes one NVFP4 row of `ncols` values occupies in `block_nvfp4` form.
///
/// Duplicated from `vulkan_kernels::nvfp4_row_bytes` on purpose: planning must
/// work on a box with no `vulkan` feature (and no GPU), and `vulkan-kernels` is
/// an optional dependency. The block constants come from `infer_gguf::dequant`,
/// which is not optional, so the two cannot disagree about the block size.
#[must_use]
pub const fn nvfp4_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_NVFP4) {
        return None;
    }
    Some(ncols / QK_NVFP4 * BLOCK_NVFP4_BYTES)
}

/// The format `kind` lands in, or `None` when it never gets a device buffer of
/// its own.
///
/// `None` covers four different reasons, all deliberate:
/// - `Expert{BlockScale}` — folded INTO the packed stream by the repack.
/// - `Expert{GlobalScale, InputScale}` — f32 scalars that ride host-side
///   ([`Qwen4ExpertStack::weight_scale_2`]).
/// - `HostGather` families — `embed_tokens` and the n-gram table.
/// - `Drop` families — MTP and the vision tower.
///
/// `dense` selects the tier for the big GEMV weights; see
/// [`Qwen4UploadConfig::dense_format`].
#[must_use]
pub fn device_format(kind: Qwen4TensorKind, dense: Qwen4DeviceFormat) -> Option<Qwen4DeviceFormat> {
    use Qwen4DeviceFormat::{F32, Nvfp4};
    use Qwen4TensorKind::*;
    match kind {
        // The one packed tier. `weight_scale` is consumed by the repack, not
        // bound separately; the two f32 scalars stay on the host.
        Expert {
            part: Nvfp4Part::Packed,
            ..
        } => Some(Nvfp4),
        Expert { .. } => None,

        // F32 because the consuming shader declares `readonly buffer { float
        // w[]; }`. Changing any of these to F16 is silent garbage.
        HyperConnection { .. }
        | LinearAttnInProjA
        | LinearAttnInProjB
        | LinearAttnConv1d
        | LinearAttnALog
        | LinearAttnDtBias
        | LinearAttnNorm
        | AttnQNorm
        | AttnKNorm
        | IndexerQNorm
        | IndexerKNorm
        | MoeRouter
        | SharedExpertGate
        | PleNormKey
        | PleNormQuery
        | PleNormConv
        | PleConv1d => Some(F32),

        // The dense GEMV weights — 6.70 GiB at F16, 13.4 GiB at F32.
        LinearAttnInProjQkv | LinearAttnInProjZ | LinearAttnOutProj | AttnQProj | AttnKProj
        | AttnVProj | AttnOProj | IndexerQkProj | SharedExpertGateProj | SharedExpertUpProj
        | SharedExpertDownProj | PleKeyProj | PleValueProj | LmHead => Some(dense),

        // Host-resident tables (see `Qwen4HostTables`).
        EmbedTokens
        | PleNgramShard
        | PleNgramWeightScale
        | PleNgramLayerMultipliers
        | PleNgramHeadsOffsets
        | PleNgramHeadsVocabSizes => None,

        // Not uploaded for a text-only decode.
        ExpertsStackedGateUp
        | ExpertsStackedDown
        | MtpFcEmbedding
        | MtpFcHidden
        | MtpPreFcNormEmbedding
        | MtpPreFcNormHidden
        | Vision(_) => None,
    }
}

/// True when the loader must store `1.0 + w` instead of `w`.
///
/// Exactly the `Qwen4ExpTextRMSNorm` weights whose consumer is the VENDORED
/// `rms_norm.comp`. The three PLE norms are the same reference class but their
/// consumer (`qwen4_ple_gate.comp`) spells the `+ 1` itself, and
/// `linear_attn.norm` is `Qwen4ExpTextRMSNormGated`, which has no bias at all.
/// See the module docs.
#[must_use]
pub const fn folds_norm_bias(kind: Qwen4TensorKind) -> bool {
    use Qwen4TensorKind::*;
    matches!(
        kind,
        HyperConnection {
            part: HcPart::Norm,
            ..
        } | AttnQNorm
            | AttnKNorm
            | IndexerQNorm
            | IndexerKNorm
    )
}

// ------------------------------------------------------------- config / scope

/// Knobs that change the bytes, not the semantics.
#[derive(Debug, Clone, Copy)]
pub struct Qwen4UploadConfig {
    /// Format for the dense GEMV weights. [`Qwen4DeviceFormat::F16`] is the
    /// shipping choice (the model does not fit the heap at F32); F32 is
    /// available for a subset bring-up, where the extra 6.7 GiB is not spent
    /// because only a few layers are resident and there is no registered F16
    /// GEMV yet. [`Qwen4DeviceFormat::Nvfp4`] is rejected — nothing re-quantizes
    /// a BF16 tensor into NVFP4 on this path.
    pub dense_format: Qwen4DeviceFormat,
    /// Nominal slab size, clamped to the device's `maxMemoryAllocationSize`.
    /// `None` — the default — lets [`Qwen4Plan::choose_packing`] pick the size
    /// that commits the fewest bytes for THIS plan, which is worth 1.34 GiB of
    /// heap on the full model; see [`Qwen4Packing`].
    pub slab_bytes: Option<u64>,
    /// Device-local bytes held back from the plan. See [`DEFAULT_RESERVE_BYTES`].
    pub reserve_bytes: u64,
}

impl Default for Qwen4UploadConfig {
    fn default() -> Self {
        Self {
            dense_format: Qwen4DeviceFormat::F16,
            slab_bytes: None,
            reserve_bytes: DEFAULT_RESERVE_BYTES,
        }
    }
}

impl Qwen4UploadConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            !matches!(self.dense_format, Qwen4DeviceFormat::Nvfp4),
            "qwen4 upload: dense_format = Nvfp4 would mean re-quantizing BF16 weights, \
             which this loader does not do"
        );
        ensure!(
            self.slab_bytes != Some(0),
            "qwen4 upload: slab_bytes must be non-zero (or None to choose one)"
        );
        Ok(())
    }
}

/// Which slice of the checkpoint one upload run covers.
///
/// A full run is ~71 GiB and minutes long, so every caller that is not the real
/// model load wants a subset. A subset is a real residency, not a mock: the same
/// plan, the same repack, the same bindings — just fewer of them.
#[derive(Debug, Clone)]
pub struct Qwen4UploadScope {
    /// Decoder layers to upload; `None` = every layer present in `st`.
    pub layers: Option<Vec<usize>>,
    /// Routed experts per layer, taken from expert id 0 upward; `None` = every
    /// expert present. `Some(0)` means "no routed experts", which is the only
    /// way to skip the MoE without tripping the missing-experts guard.
    ///
    /// A partial stack keeps the full `batch_stride_a` contract — expert `e`
    /// still starts at `e * nrows * row_bytes` — so ids at or above the cap are
    /// simply not resident.
    pub experts: Option<usize>,
    /// Upload `lm_head.weight` (1.18 GiB of F16).
    pub lm_head: bool,
}

impl Default for Qwen4UploadScope {
    /// [`Self::full`], so `Qwen4UploadScope { layers: Some(..), ..default() }`
    /// narrows only what it names. A derived `Default` would quietly drop
    /// `lm_head`.
    fn default() -> Self {
        Self::full()
    }
}

impl Qwen4UploadScope {
    /// Everything the checkpoint has.
    #[must_use]
    pub fn full() -> Self {
        Self {
            layers: None,
            experts: None,
            lm_head: true,
        }
    }

    /// Named layers only, with every expert and `lm_head`.
    #[must_use]
    pub fn layers(layers: &[usize]) -> Self {
        Self {
            layers: Some(layers.to_vec()),
            experts: None,
            lm_head: true,
        }
    }

    fn includes_layer(&self, layer: usize) -> bool {
        self.layers.as_ref().is_none_or(|ls| ls.contains(&layer))
    }

    fn includes_expert(&self, expert: u32) -> bool {
        self.experts.is_none_or(|cap| (expert as usize) < cap)
    }
}

// --------------------------------------------------------------------- plan

/// Which attention a decoder layer runs, read off the tensors it actually has
/// rather than off `config.json`'s `layer_types`.
///
/// Deliberately a fact about the RESIDENCY, not about the config: it answers
/// "what did this layer's weights turn out to be", which is the question a
/// partial scope makes interesting and which a config file cannot answer. The
/// config's own `layer_types` is parsed elsewhere (`qwen4_config`);
/// `full_plan_fits_the_device_local_heap` pins that the residency independently
/// arrives at the same 12-of-48 split `full_attention_interval: 4` implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen4LayerKind {
    /// `linear_attn.*` — the gated-delta path, 36 of 48 layers.
    LinearAttention,
    /// `self_attn.*` plus the QSA indexer, 12 of 48 layers.
    FullAttention,
}

/// Where one plan entry's bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen4Source {
    /// One BF16 checkpoint tensor, converted to the entry's format. `fold_bias`
    /// is [`folds_norm_bias`] for its kind.
    Bf16 { fold_bias: bool },
    /// `n_experts` NVFP4 `(weight, weight_scale)` plane pairs, repacked into
    /// ggml `block_nvfp4` and stacked expert-major into one suballocation.
    Nvfp4Stack {
        layer: usize,
        proj: ExpertProj,
        n_experts: usize,
    },
}

/// One suballocation the upload will make.
#[derive(Debug, Clone)]
pub struct Qwen4PlanItem {
    /// The checkpoint tensor name, or [`expert_stack_name`] for a stack.
    pub name: String,
    pub role: Qwen4TensorRole,
    pub format: Qwen4DeviceFormat,
    pub bytes: u64,
    /// Contraction width in LOGICAL elements (GGUF `ne0`), i.e. `ncols` for the
    /// GEMV push constants. For an expert stack this is the per-expert in-dim.
    pub ncols: usize,
    /// Output rows (GGUF `ne1`; `1` for a vector). For an expert stack this is
    /// the PER-EXPERT row count — the value `stride_d` wants — not
    /// `n_experts * rows`.
    pub nrows: usize,
    pub source: Qwen4Source,
}

/// The residency plan: every suballocation, and what it costs.
#[derive(Debug, Clone, Default)]
pub struct Qwen4Plan {
    pub items: Vec<Qwen4PlanItem>,
    /// Sum of `items[..].bytes`, before slab alignment padding.
    pub device_bytes: u64,
    /// Device bytes per decoder layer, for a layer-at-a-time bring-up.
    pub layer_bytes: BTreeMap<usize, u64>,
    /// Device bytes of the stream-level tensors (the mixer, `lm_head`).
    pub global_bytes: u64,
    /// Bytes left borrowed from the mmaps: `embed_tokens` + the n-gram table.
    pub host_bytes: u64,
    /// Bytes skipped entirely: the MTP block and the vision tower.
    pub dropped_bytes: u64,
    /// Attention kind per in-scope layer.
    pub layer_kinds: BTreeMap<usize, Qwen4LayerKind>,
}

impl Qwen4Plan {
    /// Refuse a plan that will not fit the device-local heap, BEFORE any upload
    /// starts.
    ///
    /// Same contract as `crate::loader::ResidencyPlan::ensure_fits`, and for the
    /// same reason: without it the failure arrives tens of GiB and several
    /// minutes deep as an opaque `ERROR_OUT_OF_DEVICE_MEMORY` from whichever
    /// allocation happened to be unlucky. `headroom` is what must survive for
    /// the KV cache, the linear-attention state, the activation arena and the
    /// descriptor pools; a plan that fits with zero bytes to spare does not fit.
    pub fn ensure_fits(&self, device_local_bytes: u64, headroom: u64) -> Result<()> {
        let budget = device_local_bytes.saturating_sub(headroom);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        ensure!(
            self.device_bytes <= budget,
            "qwen4 residency plan needs {:.2} GiB but the device-local heap is {:.2} GiB \
             ({:.2} GiB reserved for KV + state + activations, leaving {:.2} GiB usable); \
             over by {:.2} GiB",
            gib(self.device_bytes),
            gib(device_local_bytes),
            gib(headroom),
            gib(budget),
            gib(self.device_bytes.saturating_sub(budget)),
        );
        Ok(())
    }

    /// Largest suballocation in the plan, i.e. the host scratch buffer the
    /// upload needs (1.18 GiB when `lm_head` is in scope), and the floor on any
    /// slab size — `SlabPlan` refuses a request no slab could ever hold.
    #[must_use]
    pub fn max_item_bytes(&self) -> u64 {
        self.items.iter().map(|i| i.bytes).max().unwrap_or(0)
    }

    /// Dry-run this plan into slabs of `slab_bytes` and report what the heap
    /// would actually give up.
    ///
    /// Runs through `vulkan_sys::SlabPlan`, the same type `SlabAllocator` drives
    /// its real allocations through, so the estimate and the residency cannot
    /// drift. Largest-first, matching [`upload_qwen4`].
    #[cfg(feature = "vulkan")]
    pub fn pack(&self, slab_bytes: u64, alignment: u64) -> Result<Qwen4Packing> {
        let mut sp = vulkan_sys::SlabPlan::new(slab_bytes, alignment)
            .map_err(|e| anyhow!("qwen4 packing at {slab_bytes} B slabs: {e}"))?;
        let mut order: Vec<usize> = (0..self.items.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.items[i].bytes));
        for &i in &order {
            sp.place(self.items[i].bytes).map_err(|e| {
                anyhow!(
                    "qwen4 packing: {} ({} B) into {slab_bytes} B slabs: {e}",
                    self.items[i].name,
                    self.items[i].bytes
                )
            })?;
        }
        Ok(Qwen4Packing {
            slab_bytes,
            committed_bytes: sp.committed_bytes(),
            slab_count: sp.slab_count(),
        })
    }

    /// The slab size that commits the fewest bytes for THIS plan.
    ///
    /// Not a tuning nicety — 1.34 GiB of a 74.43 GiB heap, measured on the full
    /// model (see [`Qwen4Packing`]). And not a constant either: the optimum is a
    /// property of the item multiset, so it moves with
    /// [`Qwen4UploadConfig::dense_format`] and with the scope. Sweeping at load
    /// costs well under a second and re-derives it from whatever is actually
    /// being uploaded.
    #[cfg(feature = "vulkan")]
    pub fn choose_packing(&self, max_slab_bytes: u64, alignment: u64) -> Result<Qwen4Packing> {
        let floor = self.max_item_bytes().max(vulkan_sys::MIN_SLAB_BYTES);
        ensure!(
            floor <= max_slab_bytes,
            "qwen4 packing: the largest suballocation is {floor} B but a slab may not              exceed {max_slab_bytes} B (maxMemoryAllocationSize)"
        );
        let mut best: Option<Qwen4Packing> = None;
        let start = floor.div_ceil(SLAB_SWEEP_STEP_BYTES) * SLAB_SWEEP_STEP_BYTES;
        let mut candidates: Vec<u64> = (0..)
            .map(|k| start + k * SLAB_SWEEP_STEP_BYTES)
            .take_while(|&sz| sz <= max_slab_bytes)
            .collect();
        // The floor and the ceiling are always legal, and rounding up to the
        // step could have skipped both.
        candidates.push(floor);
        candidates.push(max_slab_bytes);
        for sz in candidates {
            let Ok(packing) = self.pack(sz, alignment) else {
                continue;
            };
            if best.is_none_or(|b| packing.committed_bytes < b.committed_bytes) {
                best = Some(packing);
            }
        }
        best.ok_or_else(|| anyhow!("qwen4 packing: no slab size in range fits this plan"))
    }
}

/// What a plan costs the heap once it is cut into slabs.
///
/// `committed_bytes` is what `vkAllocateMemory` gives up, which is NOT
/// [`Qwen4Plan::device_bytes`]: slabs have tails, and the tail volume is a
/// step function of the slab size against the item sizes. Measured on the full
/// `qwen4_exp` residency (1210 items, 144 of them 450 MiB expert stacks and one
/// 1.18 GiB `lm_head`), against a 74.43 GiB heap:
///
/// ```text
///   slab 2048 MiB   74.000 GiB committed   37 slabs   1.83% waste   0.43 GiB free
///   slab 1920 MiB   73.125 GiB committed   39 slabs   0.66% waste   1.31 GiB free
///   slab 1664 MiB   79.625 GiB committed   49 slabs   8.77% waste   DOES NOT FIT
///   slab 1488 MiB   72.656 GiB committed   50 slabs   0.02% waste   1.78 GiB free
/// ```
///
/// The non-monotonicity is why this is swept and not reasoned about: at 1664 MiB
/// only three 450 MiB stacks fit per slab, so 48 slabs each strand a 314 MiB
/// tail — 14.7 GiB of tail against a 9.4 GiB dense tier that could fill it. Take
/// the device's `maxMemoryAllocationSize` as the slab size, as a loader
/// naturally would, and the model reaches the device with 0.43 GiB of heap left
/// for the KV cache, the recurrent state and the arena.
#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen4Packing {
    pub slab_bytes: u64,
    /// Bytes `vkAllocateMemory` will be asked for, summed over slabs.
    pub committed_bytes: u64,
    pub slab_count: usize,
}

#[cfg(feature = "vulkan")]
impl Qwen4Packing {
    /// Refuse a packing that leaves less than `headroom` of the heap.
    ///
    /// The check [`Qwen4Plan::ensure_fits`] cannot make: that one sees the
    /// plan's bytes, this one sees the bytes the slabs will really cost.
    pub fn ensure_fits(&self, device_local_bytes: u64, headroom: u64) -> Result<()> {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        ensure!(
            self.committed_bytes + headroom <= device_local_bytes,
            "qwen4 residency commits {:.2} GiB over {} slabs of {:.0} MiB, which leaves              {:.2} GiB of the {:.2} GiB device-local heap — less than the {:.2} GiB the KV              cache, recurrent state and arena need",
            gib(self.committed_bytes),
            self.slab_count,
            self.slab_bytes as f64 / (1u64 << 20) as f64,
            gib(device_local_bytes.saturating_sub(self.committed_bytes)),
            gib(device_local_bytes),
            gib(headroom),
        );
        Ok(())
    }

    /// `committed - used`, as a fraction of what was committed.
    #[must_use]
    pub fn waste_fraction(&self, plan: &Qwen4Plan) -> f64 {
        if self.committed_bytes == 0 {
            return 0.0;
        }
        self.committed_bytes.saturating_sub(plan.device_bytes) as f64 / self.committed_bytes as f64
    }
}

/// One in-progress expert stack, accumulated while walking the headers.
struct ExpertGroup {
    layer: usize,
    proj: ExpertProj,
    /// Expert ids whose `weight` plane is present and in scope.
    packed: Vec<u32>,
    /// Per-expert logical shape, from the first `weight` seen.
    ncols: usize,
    nrows: usize,
}

/// Plan the upload from the checkpoint's HEADERS alone — no tensor data is
/// touched, so this runs on a box with no GPU and answers "does it fit?" before
/// anything is read.
pub fn plan_qwen4_upload(
    st: &SafeTensorsDir,
    cfg: &Qwen4UploadConfig,
    scope: &Qwen4UploadScope,
) -> Result<Qwen4Plan> {
    cfg.validate()?;
    let mut plan = Qwen4Plan::default();
    let mut groups: HashMap<(usize, ExpertProj), ExpertGroup> = HashMap::new();
    // Per (layer, proj): the block-scale plane's declared bytes, so a scale
    // plane that does not match its weight plane fails at plan time.
    let mut scale_bytes: HashMap<(usize, ExpertProj, u32), u64> = HashMap::new();

    for info in st.tensors() {
        let role = classify_qwen4_tensor(&info.name)
            .with_context(|| format!("classifying {}", info.name))?;

        if role.stream != Qwen4Stream::Text {
            plan.dropped_bytes += info.len;
            continue;
        }
        if role.residency == Qwen4Residency::HostGather {
            plan.host_bytes += info.len;
            continue;
        }
        if let Some(layer) = role.layer {
            if !scope.includes_layer(layer) {
                continue;
            }
            match role.kind {
                Qwen4TensorKind::LinearAttnInProjQkv => {
                    plan.layer_kinds
                        .insert(layer, Qwen4LayerKind::LinearAttention);
                }
                Qwen4TensorKind::AttnQProj => {
                    plan.layer_kinds
                        .insert(layer, Qwen4LayerKind::FullAttention);
                }
                _ => {}
            }
        }
        if role.kind == Qwen4TensorKind::LmHead && !scope.lm_head {
            continue;
        }

        // Routed experts accumulate into a stack instead of getting an entry.
        if let Qwen4TensorKind::Expert { proj, part } = role.kind {
            let layer = role
                .layer
                .ok_or_else(|| anyhow!("{}: routed expert with no layer", info.name))?;
            let expert = role
                .sub_index
                .ok_or_else(|| anyhow!("{}: routed expert with no expert id", info.name))?;
            if !scope.includes_expert(expert) {
                continue;
            }
            match part {
                Nvfp4Part::Packed => {
                    let (ncols, nrows) = nvfp4_plane_shape(info)?;
                    let group = groups.entry((layer, proj)).or_insert(ExpertGroup {
                        layer,
                        proj,
                        packed: Vec::new(),
                        ncols,
                        nrows,
                    });
                    ensure!(
                        group.ncols == ncols && group.nrows == nrows,
                        "{}: expert {expert} is [{nrows}, {ncols}] but expert {} of the same \
                         stack is [{}, {}] — a stacked GEMV needs one shape",
                        info.name,
                        group.packed.first().copied().unwrap_or(0),
                        group.nrows,
                        group.ncols,
                    );
                    group.packed.push(expert);
                }
                Nvfp4Part::BlockScale => {
                    scale_bytes.insert((layer, proj, expert), info.len);
                }
                // f32 scalars; read at upload into the host arrays.
                Nvfp4Part::GlobalScale | Nvfp4Part::InputScale => {}
            }
            continue;
        }

        let Some(format) = device_format(role.kind, cfg.dense_format) else {
            continue;
        };
        ensure!(
            info.dtype == "BF16",
            "{}: expected BF16 on the device-resident text stream, found {}",
            info.name,
            info.dtype
        );
        let (ncols, nrows) = dense_shape(info)?;
        let bytes = format.bytes_for(ncols, nrows).ok_or_else(|| {
            anyhow!(
                "{}: [{nrows}, {ncols}] has no {format:?} device layout",
                info.name
            )
        })?;
        push_item(
            &mut plan,
            Qwen4PlanItem {
                name: info.name.clone(),
                role,
                format,
                bytes,
                ncols,
                nrows,
                source: Qwen4Source::Bf16 {
                    fold_bias: folds_norm_bias(role.kind),
                },
            },
        );
    }

    // Turn each accumulated group into one stacked suballocation.
    let mut group_keys: Vec<(usize, ExpertProj)> = groups.keys().copied().collect();
    group_keys.sort_by_key(|(layer, proj)| (*layer, proj_order(*proj)));
    for key in group_keys {
        let mut group = groups.remove(&key).expect("key came from the map");
        group.packed.sort_unstable();
        let n_experts = group.packed.len();
        // Contiguity is the whole contract: the shader lands on expert `e` at
        // `e * batch_stride_a`, so a hole would silently route a token into its
        // neighbour's weights.
        for (slot, &id) in group.packed.iter().enumerate() {
            ensure!(
                id as usize == slot,
                "layer {} {:?} experts are not 0..{n_experts} (slot {slot} holds id {id}) — \
                 a stacked GEMV indexes by position",
                group.layer,
                group.proj
            );
        }
        if let Some(cap) = scope.experts {
            ensure!(
                n_experts == cap,
                "layer {} {:?}: scope asked for {cap} experts, the opened shards hold {n_experts}",
                group.layer,
                group.proj
            );
        }
        let row_bytes = nvfp4_row_bytes(group.ncols).ok_or_else(|| {
            anyhow!(
                "layer {} {:?}: in-dim {} is not a multiple of {QK_NVFP4}",
                group.layer,
                group.proj,
                group.ncols
            )
        })?;
        // The scale plane is one UE4M3 byte per 16 values; it is consumed by the
        // repack, so its only job here is to prove the two planes agree.
        let want_scale = (group.nrows * group.ncols / QK_NVFP4_SUB) as u64;
        for &id in &group.packed {
            let got = scale_bytes
                .get(&(group.layer, group.proj, id))
                .copied()
                .ok_or_else(|| {
                    anyhow!(
                        "{}: weight plane present but weight_scale is missing",
                        expert_tensor_name(group.layer, id, group.proj, Nvfp4Part::Packed)
                    )
                })?;
            ensure!(
                got == want_scale,
                "{}: weight_scale is {got} B, expected {want_scale} B for [{}, {}]",
                expert_tensor_name(group.layer, id, group.proj, Nvfp4Part::BlockScale),
                group.nrows,
                group.ncols
            );
        }
        let role = classify_qwen4_tensor(&expert_tensor_name(
            group.layer,
            0,
            group.proj,
            Nvfp4Part::Packed,
        ))?;
        push_item(
            &mut plan,
            Qwen4PlanItem {
                name: expert_stack_name(group.layer, group.proj),
                role,
                format: Qwen4DeviceFormat::Nvfp4,
                bytes: (n_experts * group.nrows * row_bytes) as u64,
                ncols: group.ncols,
                nrows: group.nrows,
                source: Qwen4Source::Nvfp4Stack {
                    layer: group.layer,
                    proj: group.proj,
                    n_experts,
                },
            },
        );
    }

    // A scoped layer with no routed experts is almost certainly a shard that was
    // never opened, not a layer that has none — say so instead of uploading a
    // decoder layer with a hole where its MoE should be.
    if scope.experts != Some(0) {
        for &layer in plan.layer_kinds.keys() {
            let have = plan.items.iter().any(
                |i| matches!(i.source, Qwen4Source::Nvfp4Stack { layer: l, .. } if l == layer),
            );
            ensure!(
                have,
                "layer {layer} is in scope but no routed experts were found — is its \
                 `layer-{layer:05}-experts-*.safetensors` shard open? \
                 (use `Qwen4UploadScope {{ experts: Some(0), .. }}` to skip the MoE on purpose)"
            );
        }
    }
    Ok(plan)
}

fn push_item(plan: &mut Qwen4Plan, item: Qwen4PlanItem) {
    plan.device_bytes += item.bytes;
    match item.role.layer {
        Some(layer) => *plan.layer_bytes.entry(layer).or_insert(0) += item.bytes,
        None => plan.global_bytes += item.bytes,
    }
    plan.items.push(item);
}

/// Stable ordering for the `(layer, proj)` group keys so a plan is reproducible.
const fn proj_order(proj: ExpertProj) -> u8 {
    match proj {
        ExpertProj::Gate => 0,
        ExpertProj::Up => 1,
        ExpertProj::Down => 2,
    }
}

/// `(ncols, nrows)` for a dense BF16 weight.
///
/// `dims` is the safetensors shape REVERSED, so `dims[0]` is the contiguous
/// (input) axis. A 1-D norm reports `(len, 1)`; the depthwise `conv1d` is
/// `[kernel, 1, channels]`, whose middle axis is always 1, and it flattens to
/// `(kernel * channels, 1)` because `qwen4_ple_conv.comp` reads it as one
/// channel-major `[channels * kernel]` row.
fn dense_shape(info: &SafeTensorInfo) -> Result<(usize, usize)> {
    let dims: Vec<usize> = info
        .dims
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| anyhow!("{}: dim overflow", info.name)))
        .collect::<Result<_>>()?;
    Ok(match dims.as_slice() {
        [] => (1, 1),
        [n] => (*n, 1),
        [ncols, nrows] => (*ncols, *nrows),
        // Depthwise conv1d: the singleton axis carries no data.
        [kernel, 1, channels] => (kernel * channels, 1),
        other => bail!(
            "{}: unexpected rank-{} shape {other:?}",
            info.name,
            dims.len()
        ),
    })
}

/// `(ncols, nrows)` for an NVFP4 `weight` plane, in LOGICAL elements.
///
/// The plane is U8 with two E2M1 values per byte, so the stored contiguous
/// extent is half the logical width: header `[640, 1280]` -> `dims [1280, 640]`
/// -> a `[640, 2560]` matrix.
fn nvfp4_plane_shape(info: &SafeTensorInfo) -> Result<(usize, usize)> {
    ensure!(
        info.dtype == "U8",
        "{}: NVFP4 weight plane must be U8, found {}",
        info.name,
        info.dtype
    );
    let packed = usize::try_from(info.dims.first().copied().unwrap_or(0))
        .map_err(|_| anyhow!("{}: dim overflow", info.name))?;
    let nrows = usize::try_from(info.dims.get(1).copied().unwrap_or(0))
        .map_err(|_| anyhow!("{}: dim overflow", info.name))?;
    ensure!(
        packed > 0 && nrows > 0,
        "{}: NVFP4 weight plane must be 2-D, got {:?}",
        info.name,
        info.dims
    );
    Ok((packed * 2, nrows))
}

// -------------------------------------------------------------- host tables

/// The two tables that never reach the device, borrowed from the checkpoint's
/// mmaps.
///
/// `embed_tokens` is 1.18 GiB of which a token reads one row; the n-gram table
/// is 47.68 GiB of which a token reads 16. Copying either would be fatal, not
/// merely wasteful, so every accessor here hands back a slice of the mapping (or
/// dequantizes one row into a small `Vec`).
pub struct Qwen4HostTables<'st> {
    embed_tokens: &'st [u8],
    hidden: usize,
    vocab: usize,
    ngram: Option<Qwen4NgramTable<'st>>,
    layer_multipliers: Option<Vec<i64>>,
    heads_offsets: Option<Vec<i64>>,
    heads_vocab_sizes: Option<Vec<i64>>,
}

/// The 128-shard FP8 n-gram embedding table as one logical `[rows, head_dim]`
/// matrix.
///
/// Rows run straight through the shards in SHARD-INDEX order — shard `i` holds
/// rows `[i * rows_per_shard, (i+1) * rows_per_shard)` — which is what makes
/// `padded_vocab_size` (320,001,536 = 128 x 2,500,012) come out exactly right.
///
/// This is the residency-side ACCESSOR, not the gather: it hands back borrowed
/// FP8 bytes (or one dequantized row) and does no threading, batching or
/// prefetch. A sibling module, `qwen4_ngram_gather`, owns the per-token
/// 16-row gather that reads them; nothing here depends on it, so the two can
/// move independently.
pub struct Qwen4NgramTable<'st> {
    shards: Vec<&'st [u8]>,
    rows_per_shard: usize,
    head_dim: usize,
    weight_scale: f32,
}

impl<'st> Qwen4HostTables<'st> {
    /// Collect the host-resident families out of an open checkpoint.
    ///
    /// `embed_tokens` is required — without it there is no token gather. The
    /// n-gram family is optional so a subset load that never opens the
    /// `model-plefp8-*` shards (52 GiB of mapping) still works; asking for it
    /// later returns a named error rather than a wrong row.
    pub fn build(st: &'st SafeTensorsDir) -> Result<Self> {
        let embed = st
            .tensor(EMBED_TOKENS_NAME)
            .ok_or_else(|| anyhow!("{EMBED_TOKENS_NAME} missing from the checkpoint"))?;
        ensure!(
            embed.dtype == "BF16",
            "{EMBED_TOKENS_NAME}: expected BF16, found {}",
            embed.dtype
        );
        // `dims` is the header shape reversed, so dims[0] is the row width.
        let hidden = usize::try_from(embed.dims.first().copied().unwrap_or(0))
            .map_err(|_| anyhow!("{EMBED_TOKENS_NAME}: hidden overflow"))?;
        let vocab = usize::try_from(embed.dims.get(1).copied().unwrap_or(0))
            .map_err(|_| anyhow!("{EMBED_TOKENS_NAME}: vocab overflow"))?;
        ensure!(
            hidden > 0 && vocab > 0,
            "{EMBED_TOKENS_NAME}: expected a 2-D table, got {:?}",
            embed.dims
        );
        let embed_tokens = st.tensor_data(EMBED_TOKENS_NAME)?;
        ensure!(
            embed_tokens.len() == hidden * vocab * 2,
            "{EMBED_TOKENS_NAME}: {} B for a [{vocab}, {hidden}] BF16 table",
            embed_tokens.len()
        );

        let mut shards: BTreeMap<u32, &'st [u8]> = BTreeMap::new();
        let mut shard_shape: Option<(usize, usize)> = None;
        let mut weight_scale: Option<f32> = None;
        let mut layer_multipliers = None;
        let mut heads_offsets = None;
        let mut heads_vocab_sizes = None;
        for info in st.tensors() {
            let role = match classify_qwen4_tensor(&info.name) {
                Ok(role) => role,
                // Not this constructor's job to police the whole checkpoint;
                // `plan_qwen4_upload` already fails loud on an unknown name.
                Err(_) => continue,
            };
            if role.stream != Qwen4Stream::Text {
                continue;
            }
            match role.kind {
                Qwen4TensorKind::PleNgramShard => {
                    let idx = role
                        .sub_index
                        .ok_or_else(|| anyhow!("{}: n-gram shard with no index", info.name))?;
                    ensure!(
                        info.dtype == "F8_E4M3",
                        "{}: expected F8_E4M3, found {}",
                        info.name,
                        info.dtype
                    );
                    let head_dim = usize::try_from(info.dims.first().copied().unwrap_or(0))
                        .map_err(|_| anyhow!("{}: dim overflow", info.name))?;
                    let rows = usize::try_from(info.dims.get(1).copied().unwrap_or(0))
                        .map_err(|_| anyhow!("{}: dim overflow", info.name))?;
                    match shard_shape {
                        None => shard_shape = Some((head_dim, rows)),
                        Some(prev) => ensure!(
                            prev == (head_dim, rows),
                            "{}: shard is [{rows}, {head_dim}] but a sibling is [{}, {}] — \
                             row addressing assumes uniform shards",
                            info.name,
                            prev.1,
                            prev.0
                        ),
                    }
                    let data = st.tensor_data(&info.name)?;
                    ensure!(
                        data.len() == rows * head_dim,
                        "{}: {} B for [{rows}, {head_dim}] FP8",
                        info.name,
                        data.len()
                    );
                    ensure!(
                        shards.insert(idx, data).is_none(),
                        "{}: duplicate n-gram shard index {idx}",
                        info.name
                    );
                }
                Qwen4TensorKind::PleNgramWeightScale => {
                    let v = read_scalar_f32(st, info)?;
                    weight_scale = Some(v);
                }
                Qwen4TensorKind::PleNgramLayerMultipliers => {
                    layer_multipliers = Some(read_i64(st, info)?);
                }
                Qwen4TensorKind::PleNgramHeadsOffsets => {
                    heads_offsets = Some(read_i64(st, info)?);
                }
                Qwen4TensorKind::PleNgramHeadsVocabSizes => {
                    heads_vocab_sizes = Some(read_i64(st, info)?);
                }
                _ => {}
            }
        }

        let ngram = if shards.is_empty() {
            None
        } else {
            let (head_dim, rows_per_shard) = shard_shape.expect("set with the first shard");
            let n = shards.len();
            let keys: Vec<u32> = shards.keys().copied().collect();
            for (i, &k) in keys.iter().enumerate() {
                ensure!(
                    k as usize == i,
                    "n-gram shards are not 0..{n}: position {i} holds shard_{k}"
                );
            }
            let weight_scale = weight_scale.ok_or_else(|| {
                anyhow!(
                    "{n} n-gram shards are present but \
                     `...ngram_embedding.weight_scale` is not — the FP8 codes alone \
                     are not the embedding"
                )
            })?;
            Some(Qwen4NgramTable {
                shards: shards.into_values().collect(),
                rows_per_shard,
                head_dim,
                weight_scale,
            })
        };

        Ok(Self {
            embed_tokens,
            hidden,
            vocab,
            ngram,
            layer_multipliers,
            heads_offsets,
            heads_vocab_sizes,
        })
    }

    #[must_use]
    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    #[must_use]
    pub const fn vocab(&self) -> usize {
        self.vocab
    }

    /// Token `t`'s BF16 row, borrowed from the mmap. No copy.
    pub fn embed_row_bytes(&self, token: usize) -> Result<&'st [u8]> {
        ensure!(
            token < self.vocab,
            "token {token} is outside the {}-entry vocabulary",
            self.vocab
        );
        let row = self.hidden * 2;
        Ok(&self.embed_tokens[token * row..][..row])
    }

    /// Token `t`'s embedding as f32.
    pub fn embed_row(&self, token: usize) -> Result<Vec<f32>> {
        infer_gguf::dequant::dequantize_row_bf16(self.embed_row_bytes(token)?, self.hidden)
    }

    /// The n-gram table, or a named error if the `model-plefp8-*` shards were
    /// never opened.
    pub fn ngram(&self) -> Result<&Qwen4NgramTable<'st>> {
        self.ngram.as_ref().ok_or_else(|| {
            anyhow!(
                "no n-gram embedding shards in this checkpoint view — \
                 open the `model-plefp8-*` shards to gather the PLE"
            )
        })
    }

    /// `ple_embedding.layer_multipliers`, the three per-position hash multipliers.
    pub fn layer_multipliers(&self) -> Result<&[i64]> {
        opt_slice(&self.layer_multipliers, "layer_multipliers")
    }

    /// `ple_embedding.ngram_heads_offsets`: each head's start row in the table.
    pub fn ngram_heads_offsets(&self) -> Result<&[i64]> {
        opt_slice(&self.heads_offsets, "ngram_heads_offsets")
    }

    /// `ple_embedding.ngram_heads_vocab_sizes`: each head's prime modulus.
    pub fn ngram_heads_vocab_sizes(&self) -> Result<&[i64]> {
        opt_slice(&self.heads_vocab_sizes, "ngram_heads_vocab_sizes")
    }
}

impl<'st> Qwen4NgramTable<'st> {
    /// Total rows across all shards — must equal
    /// `qwen4_ple::NGramHash::padded_vocab_size`.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.shards.len() as u64 * self.rows_per_shard as u64
    }

    /// Row width: `ple_embed_dim / ngram_heads` = 160.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// The single BF16 scalar that dequantizes the whole FP8 table.
    #[must_use]
    pub const fn weight_scale(&self) -> f32 {
        self.weight_scale
    }

    /// One row's raw FP8 bytes, borrowed from the mmap.
    pub fn row_bytes(&self, row: u64) -> Result<&'st [u8]> {
        ensure!(
            row < self.rows(),
            "n-gram row {row} is outside the {}-row table",
            self.rows()
        );
        let per = self.rows_per_shard as u64;
        let shard = (row / per) as usize;
        let within = (row % per) as usize;
        Ok(&self.shards[shard][within * self.head_dim..][..self.head_dim])
    }

    /// One row, dequantized and scaled: `f8_e4m3(code) * weight_scale`.
    pub fn row(&self, row: u64) -> Result<Vec<f32>> {
        let mut out =
            infer_gguf::dequant::dequantize_row_f8_e4m3(self.row_bytes(row)?, self.head_dim)?;
        for v in &mut out {
            *v *= self.weight_scale;
        }
        Ok(out)
    }
}

fn opt_slice<'a>(v: &'a Option<Vec<i64>>, what: &str) -> Result<&'a [i64]> {
    v.as_deref().ok_or_else(|| {
        anyhow!(
            "`ple_embedding.{what}` is not in this checkpoint view — open the shard that holds it"
        )
    })
}

fn read_i64(st: &SafeTensorsDir, info: &SafeTensorInfo) -> Result<Vec<i64>> {
    ensure!(
        info.dtype == "I64",
        "{}: expected I64, found {}",
        info.name,
        info.dtype
    );
    let data = st.tensor_data(&info.name)?;
    ensure!(
        data.len().is_multiple_of(8),
        "{}: {} B is not a whole number of i64",
        info.name,
        data.len()
    );
    Ok(data
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

/// A rank-0 (or 1-element) BF16/F32 scalar as f32.
fn read_scalar_f32(st: &SafeTensorsDir, info: &SafeTensorInfo) -> Result<f32> {
    let data = st.tensor_data(&info.name)?;
    Ok(match info.dtype.as_str() {
        "BF16" => {
            ensure!(
                data.len() == 2,
                "{}: {} B for a BF16 scalar",
                info.name,
                data.len()
            );
            bf16_to_f32(u16::from_le_bytes([data[0], data[1]]))
        }
        "F32" => {
            ensure!(
                data.len() == 4,
                "{}: {} B for an F32 scalar",
                info.name,
                data.len()
            );
            f32::from_le_bytes(data[..4].try_into().expect("4 bytes"))
        }
        other => bail!(
            "{}: expected a BF16 or F32 scalar, found {other}",
            info.name
        ),
    })
}

/// BF16 is the top 16 bits of an f32, so the widening is a shift — no table, no
/// rounding, exact in both directions.
#[must_use]
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert one f32 to an IEEE-754 binary16 bit pattern, round-to-nearest-even.
///
/// A verbatim copy of `crate::loader::upload::f32_to_f16` (pinned there against
/// the half-precision corner cases: inf/NaN passthrough, subnormals, overflow to
/// inf). Copied rather than imported because that module is `crate::loader`'s
/// and this file may not widen it; `f16_round_trip_pins_the_corner_cases` below
/// re-pins the behaviour so the two cannot drift silently.
#[must_use]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    // Inf / NaN: keep a set mantissa bit for NaN so it stays NaN.
    if exp == 0xff {
        if mant != 0 {
            return sign | 0x7e00; // qNaN
        }
        return sign | 0x7c00; // +/- inf
    }

    let unbiased = exp - 127;
    let half_exp = unbiased + 15;

    if half_exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign; // too small even for the smallest subnormal
        }
        let significand = mant | 0x0080_0000; // implicit 1.fraction
        let shift = (14 - half_exp) as u32; // in [14, 24]
        let half_mant = significand >> shift;
        let remainder = significand & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut result = half_mant;
        if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
            result += 1; // may carry into the exponent field — that's fine
        }
        return sign | (result as u16);
    }

    let half_mant = mant >> 13;
    let remainder = mant & 0x1fff;
    let halfway = 0x1000u32;
    let mut result = ((half_exp as u32) << 10) | half_mant;
    if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
        result += 1; // carries mantissa->exponent correctly
    }
    sign | (result as u16)
}

/// Fill `dst` with the F16 form of a BF16 tensor.
///
/// Streamed element by element rather than through an intermediate
/// `Vec<f32>`: `lm_head` is 635.7 M elements, so the f32 staging alone would be
/// 2.5 GiB of transient host memory on top of the 1.18 GiB output.
#[cfg(feature = "vulkan")]
fn write_bf16_as_f16(name: &str, src: &[u8], dst: &mut [u8]) -> Result<()> {
    ensure!(
        src.len() == dst.len(),
        "{name}: {} B of BF16 into a {} B F16 buffer",
        src.len(),
        dst.len()
    );
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let v = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
        d.copy_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    Ok(())
}

/// Fill `dst` with the F32 form of a BF16 tensor, optionally folding the
/// `Qwen4ExpTextRMSNorm` bias (`1.0 + w`) on the way — see [`folds_norm_bias`].
#[cfg(feature = "vulkan")]
fn write_bf16_as_f32(name: &str, src: &[u8], dst: &mut [u8], fold_bias: bool) -> Result<()> {
    ensure!(
        src.len() * 2 == dst.len(),
        "{name}: {} B of BF16 into a {} B F32 buffer",
        src.len(),
        dst.len()
    );
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
        let mut v = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
        if fold_bias {
            v += 1.0;
        }
        d.copy_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

// ------------------------------------------------------------------- upload

/// One uploaded tensor: where it sits in the slab, its LOGICAL shape, and what
/// the bytes are.
#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, Copy)]
pub struct Qwen4DeviceTensor {
    /// Feed to [`Qwen4Weights::binding`] for a
    /// `DescriptorSet::storage_buffers_ranged` triple.
    pub alloc: vulkan_sys::SlabAlloc,
    /// Contraction width in logical elements (`ncols` for the GEMV push
    /// constants); the length for a 1-D tensor.
    pub ncols: usize,
    /// Output rows (`1` for a 1-D tensor). For an expert stack this is the
    /// PER-EXPERT row count.
    pub nrows: usize,
    pub format: Qwen4DeviceFormat,
}

/// One layer's 512 experts of one projection, stacked expert-major.
#[cfg(feature = "vulkan")]
#[derive(Debug, Clone)]
pub struct Qwen4ExpertStack {
    pub tensor: Qwen4DeviceTensor,
    pub n_experts: usize,
    /// `weight_scale_2`, one f32 per expert id, in ID ORDER.
    ///
    /// `block_nvfp4` has nowhere to keep the per-tensor second-level scale, so
    /// it rides the fused GEMV's `MAT_VEC_FUSION_SCALE0` output scale, whose
    /// binding is indexed by expert **slot**. Use [`Self::scale0_for_route`] to
    /// reorder; do not bind this array.
    pub weight_scale_2: Vec<f32>,
    /// `input_scale`, one f32 per expert id. The static W4A4 ACTIVATION
    /// quantizer scale, unused by this f32-activation lane; carried because all
    /// 73,728 of them together are 288 KiB and dropping it would foreclose an
    /// activation-quantized path for no measurable saving.
    pub input_scale: Vec<f32>,
}

#[cfg(feature = "vulkan")]
impl Qwen4ExpertStack {
    /// `batch_stride_a` in ELEMENTS, for `vulkan_kernels::gemv_id_params_fused`.
    #[must_use]
    pub const fn batch_stride_a(&self) -> usize {
        self.tensor.ncols * self.tensor.nrows
    }

    /// Gather `weight_scale_2` for one token's routed experts into SLOT order.
    ///
    /// The trap this exists to close: `MAT_VEC_FUSION_SCALE0` makes binding 3 an
    /// `f32[n_experts]` read at `gl_GlobalInvocationID.y` — the expert SLOT,
    /// i.e. the position in `expert_ids`, NOT the expert id.
    /// [`Self::weight_scale_2`] is in ID order, so binding it directly scales
    /// every expert by some other expert's second-level scale and still produces
    /// finite logits. `out` is reused rather than returned so a decode step that
    /// runs this 144 times does not allocate.
    ///
    /// `MAT_VEC_FUSION_SCALE1` is free for the routing weight, which would save
    /// the MoE accumulate a multiply; that is the forward's call, not this one's.
    pub fn scale0_for_route(&self, expert_ids: &[i32], out: &mut Vec<f32>) -> Result<()> {
        out.clear();
        out.reserve(expert_ids.len());
        for &id in expert_ids {
            let id = usize::try_from(id)
                .ok()
                .filter(|&i| i < self.n_experts)
                .ok_or_else(|| {
                    anyhow!(
                        "routed expert id {id} is outside the {}-expert stack",
                        self.n_experts
                    )
                })?;
            out.push(self.weight_scale_2[id]);
        }
        Ok(())
    }

    /// Byte offset of expert `id` inside the stack, i.e. what the shader's
    /// `expert_id * (batch_stride_a / QUANT_K)` lands on.
    pub fn expert_offset(&self, id: usize) -> Result<u64> {
        ensure!(
            id < self.n_experts,
            "expert {id} is outside the {}-expert stack",
            self.n_experts
        );
        let row = nvfp4_row_bytes(self.tensor.ncols)
            .ok_or_else(|| anyhow!("stack in-dim {} is not NVFP4-shaped", self.tensor.ncols))?;
        Ok((id * self.tensor.nrows * row) as u64)
    }
}

/// The four weights of one `Qwen4ExpTextGatedResidual`, resolved in one lookup.
///
/// This site runs 97 times per token; going through
/// [`Qwen4Weights::hyper_connection`] once beats four string formats per site.
#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, Copy)]
pub struct Qwen4HcBinding<'a> {
    /// **Already `1.0 + w`** — see [`folds_norm_bias`].
    pub hc_norm: &'a Qwen4DeviceTensor,
    /// `input_mix_weight_down`, `[hc_lowrank, hc_count*hidden]`.
    pub mix_down: &'a Qwen4DeviceTensor,
    /// `input_mix_weight_up`, `[hc_count*hidden, hc_lowrank]`.
    pub mix_up: &'a Qwen4DeviceTensor,
    /// `block_inject_weight`; `None` on the mixer (`use_combine=false`).
    pub block_inject: Option<&'a Qwen4DeviceTensor>,
}

/// Everything `qwen4_exp` needs to run, resident.
#[cfg(feature = "vulkan")]
pub struct Qwen4Weights<'ctx, 'st> {
    slabs: vulkan_sys::SlabAllocator<'ctx>,
    tensors: BTreeMap<String, Qwen4DeviceTensor>,
    experts: HashMap<(usize, ExpertProj), Qwen4ExpertStack>,
    layer_kinds: BTreeMap<usize, Qwen4LayerKind>,
    host: Qwen4HostTables<'st>,
}

#[cfg(feature = "vulkan")]
impl<'ctx, 'st> Qwen4Weights<'ctx, 'st> {
    /// A device tensor by its checkpoint name. `Err`, not `None`: a weight the
    /// forward asks for and does not get is a bug, and a zero buffer would run.
    pub fn tensor(&self, name: &str) -> Result<&Qwen4DeviceTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("qwen4 weights: `{name}` is not resident"))
    }

    /// `(slab buffer, offset, len)` — straight into
    /// `DescriptorSet::storage_buffers_ranged`.
    pub fn binding(
        &self,
        tensor: &Qwen4DeviceTensor,
    ) -> Result<(&vulkan_sys::DeviceBuffer<'ctx>, u64, u64)> {
        self.slabs
            .binding(&tensor.alloc)
            .map_err(|e| anyhow!("qwen4 weights: binding a suballocation: {e}"))
    }

    /// [`Self::tensor`] then [`Self::binding`], for the common case.
    pub fn binding_by_name(
        &self,
        name: &str,
    ) -> Result<(&vulkan_sys::DeviceBuffer<'ctx>, u64, u64)> {
        let t = *self.tensor(name)?;
        self.binding(&t)
    }

    /// One `Qwen4ExpTextGatedResidual`. Pass `(None, HcSite::Mixer)` for the
    /// stream-level mixer, `(Some(l), Attn|Mlp)` for a layer's two sites.
    pub fn hyper_connection(
        &self,
        layer: Option<usize>,
        site: HcSite,
    ) -> Result<Qwen4HcBinding<'_>> {
        let prefix = hyper_connection_prefix(layer, site)?;
        // `use_combine` is a property of the SITE, not of what happens to be
        // resident. Looking it up as an `Option` would let a layer site whose
        // `block_inject_weight` failed to upload run as if it were the mixer —
        // a residual that never receives its block output, and finite output all
        // the way to the logits.
        let block_inject = match site {
            HcSite::Mixer => None,
            HcSite::Attn | HcSite::Mlp => {
                Some(self.tensor(&format!("{prefix}.block_inject_weight.weight"))?)
            }
        };
        Ok(Qwen4HcBinding {
            hc_norm: self.tensor(&format!("{prefix}.hc_norm.weight"))?,
            mix_down: self.tensor(&format!("{prefix}.input_mix_weight_down.weight"))?,
            mix_up: self.tensor(&format!("{prefix}.input_mix_weight_up.weight"))?,
            block_inject,
        })
    }

    pub fn expert_stack(&self, layer: usize, proj: ExpertProj) -> Result<&Qwen4ExpertStack> {
        self.experts.get(&(layer, proj)).ok_or_else(|| {
            anyhow!("qwen4 weights: layer {layer} has no resident {proj:?} expert stack")
        })
    }

    #[must_use]
    pub fn layer_kind(&self, layer: usize) -> Option<Qwen4LayerKind> {
        self.layer_kinds.get(&layer).copied()
    }

    /// The layers that are resident, ascending.
    pub fn resident_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layer_kinds.keys().copied()
    }

    #[must_use]
    pub const fn host(&self) -> &Qwen4HostTables<'st> {
        &self.host
    }

    #[must_use]
    pub const fn slabs(&self) -> &vulkan_sys::SlabAllocator<'ctx> {
        &self.slabs
    }

    /// Read a suballocation back through a staging buffer. Verification only —
    /// it allocates, submits and blocks; never put it on a per-token path.
    pub fn read_back(&self, tensor: &Qwen4DeviceTensor, dst: &mut [u8]) -> Result<()> {
        self.slabs
            .read_back(&tensor.alloc, dst)
            .map_err(|e| anyhow!("qwen4 weights: read-back: {e}"))
    }
}

/// Execute `plan` against `ctx`, borrowing every source byte from `st`'s mmaps.
///
/// Order of operations matters and is not incidental:
/// 1. [`Qwen4Plan::ensure_fits`] BEFORE a byte is staged.
/// 2. Suballocate LARGEST-FIRST. `vulkan_sys::SlabPlan` documents the measured
///    difference on this exact checkpoint: 62 slabs and 1.01% waste largest-first
///    against 64 slabs and 4.10% in arrival order.
/// 3. Write in plan order, which is checkpoint order, so a shard's mmap pages
///    are touched once and in sequence rather than revisited per slab.
///
/// `st` must include the shard holding `embed_tokens` even for a subset run:
/// [`Qwen4HostTables::build`] treats it as required, because a residency without
/// the token gather cannot produce a token.
#[cfg(feature = "vulkan")]
pub fn upload_qwen4<'ctx, 'st>(
    ctx: &'ctx vulkan_sys::VulkanContext,
    st: &'st SafeTensorsDir,
    plan: &Qwen4Plan,
    cfg: &Qwen4UploadConfig,
) -> Result<Qwen4Weights<'ctx, 'st>> {
    cfg.validate()?;
    // Size against the DEVICE_LOCAL heap, not system RAM: on this APU a 60 GiB
    // device-local allocation moved OS-visible free RAM by only 4.2 GiB, so the
    // host figure says nothing about what the device can hold.
    let device_local: u64 = ctx
        .memory_heaps()
        .into_iter()
        .filter(|&(_, device_local)| device_local)
        .map(|(size, _)| size)
        .max()
        .unwrap_or(u64::MAX);
    plan.ensure_fits(device_local, cfg.reserve_bytes)?;

    // The plan's own bytes are a FLOOR: slabs have tails. Cut them the cheapest
    // way for this particular plan and check what that really costs, before any
    // `vkAllocateMemory` — taking `maxMemoryAllocationSize` as the slab size,
    // which is the obvious choice, costs 1.34 GiB here. See `Qwen4Packing`.
    let max_slab = ctx.max_memory_allocation_size();
    // `SlabAllocator` applies exactly this floor; mirroring it keeps the dry run
    // and the real placement on the same alignment.
    let alignment = ctx.min_storage_buffer_offset_alignment().max(16);
    let packing = match cfg.slab_bytes {
        Some(bytes) => plan.pack(bytes.min(max_slab), alignment)?,
        None => plan.choose_packing(max_slab, alignment)?,
    };
    packing.ensure_fits(device_local, cfg.reserve_bytes)?;
    log::info!(
        "qwen4 residency: {:.2} GiB over {} slabs of {:.0} MiB ({:.2}% packing waste),          {:.2} GiB of heap left",
        packing.committed_bytes as f64 / (1u64 << 30) as f64,
        packing.slab_count,
        packing.slab_bytes as f64 / (1u64 << 20) as f64,
        100.0 * packing.waste_fraction(plan),
        device_local.saturating_sub(packing.committed_bytes) as f64 / (1u64 << 30) as f64,
    );

    let mut slabs = vulkan_sys::SlabAllocator::with_slab_size(ctx, packing.slab_bytes)
        .map_err(|e| anyhow!("qwen4 upload: creating the slab allocator: {e}"))?;

    // Largest-first placement; `order` indexes back into `plan.items`. Same
    // order the dry run used, so the real slab count matches what was checked.
    let mut order: Vec<usize> = (0..plan.items.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(plan.items[i].bytes));
    let mut allocs: Vec<Option<vulkan_sys::SlabAlloc>> = vec![None; plan.items.len()];
    for &i in &order {
        let item = &plan.items[i];
        let alloc = slabs.alloc(item.bytes).map_err(|e| {
            anyhow!(
                "qwen4 upload: reserving {} B for {}: {e}",
                item.bytes,
                item.name
            )
        })?;
        allocs[i] = Some(alloc);
    }

    // One host buffer, sized to the biggest suballocation and reused. Two
    // reasons it is not per-tensor: `SlabAllocator::write` needs one contiguous
    // slice (there is no write-at-offset), and `lm_head` alone would otherwise
    // churn 1.18 GiB of allocate/zero/free.
    let scratch_len = usize::try_from(plan.max_item_bytes())
        .map_err(|_| anyhow!("qwen4 upload: largest suballocation does not fit in usize"))?;
    let mut scratch = vec![0u8; scratch_len];

    let mut tensors = BTreeMap::new();
    let mut experts = HashMap::new();
    for (i, item) in plan.items.iter().enumerate() {
        let alloc = allocs[i].ok_or_else(|| anyhow!("{}: no suballocation", item.name))?;
        let len = usize::try_from(item.bytes)
            .map_err(|_| anyhow!("{}: byte count does not fit in usize", item.name))?;
        let dst = &mut scratch[..len];

        match &item.source {
            Qwen4Source::Bf16 { fold_bias } => {
                let src = st.tensor_data(&item.name)?;
                match item.format {
                    Qwen4DeviceFormat::F16 => write_bf16_as_f16(&item.name, src, dst)?,
                    Qwen4DeviceFormat::F32 => write_bf16_as_f32(&item.name, src, dst, *fold_bias)?,
                    Qwen4DeviceFormat::Nvfp4 => {
                        bail!("{}: a BF16 source cannot land as NVFP4", item.name)
                    }
                }
            }
            Qwen4Source::Nvfp4Stack {
                layer,
                proj,
                n_experts,
            } => {
                let (weight_scale_2, input_scale) =
                    stage_expert_stack(st, item, *layer, *proj, *n_experts, dst)?;
                experts.insert(
                    (*layer, *proj),
                    Qwen4ExpertStack {
                        tensor: Qwen4DeviceTensor {
                            alloc,
                            ncols: item.ncols,
                            nrows: item.nrows,
                            format: item.format,
                        },
                        n_experts: *n_experts,
                        weight_scale_2,
                        input_scale,
                    },
                );
            }
        }
        slabs
            .write(&alloc, dst)
            .map_err(|e| anyhow!("qwen4 upload: writing {}: {e}", item.name))?;

        tensors.insert(
            item.name.clone(),
            Qwen4DeviceTensor {
                alloc,
                ncols: item.ncols,
                nrows: item.nrows,
                format: item.format,
            },
        );
    }

    Ok(Qwen4Weights {
        slabs,
        tensors,
        experts,
        layer_kinds: plan.layer_kinds.clone(),
        host: Qwen4HostTables::build(st)?,
    })
}

/// Repack one layer's expert planes into `dst`, expert-major, and collect the
/// two per-expert f32 scalars.
///
/// The nibble order differs between ModelOpt's and ggml's packing, so the bytes
/// are BUILT by `vulkan_kernels::repack_nvfp4_planes`, not copied — a straight
/// `d || qs` concatenation permutes every group of 16 weights against its
/// activations and still produces plausible, finite output.
#[cfg(feature = "vulkan")]
fn stage_expert_stack(
    st: &SafeTensorsDir,
    item: &Qwen4PlanItem,
    layer: usize,
    proj: ExpertProj,
    n_experts: usize,
    dst: &mut [u8],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let row_bytes = nvfp4_row_bytes(item.ncols)
        .ok_or_else(|| anyhow!("{}: in-dim {} is not NVFP4-shaped", item.name, item.ncols))?;
    let stride = item.nrows * row_bytes;
    ensure!(
        dst.len() == n_experts * stride,
        "{}: {} B of scratch for {n_experts} x {stride} B experts",
        item.name,
        dst.len()
    );
    let mut weight_scale_2 = Vec::with_capacity(n_experts);
    let mut input_scale = Vec::with_capacity(n_experts);
    for (id, slot) in dst.chunks_exact_mut(stride).enumerate() {
        let id = id as u32;
        let qs = st.tensor_data(&expert_tensor_name(layer, id, proj, Nvfp4Part::Packed))?;
        let sc = st.tensor_data(&expert_tensor_name(layer, id, proj, Nvfp4Part::BlockScale))?;
        vulkan_kernels::repack_nvfp4_planes(qs, sc, item.nrows, item.ncols, slot).map_err(|e| {
            anyhow!(
                "{}: repacking expert {id}: {e}",
                expert_stack_name(layer, proj)
            )
        })?;
        weight_scale_2.push(expert_scalar(
            st,
            &expert_tensor_name(layer, id, proj, Nvfp4Part::GlobalScale),
        )?);
        input_scale.push(expert_scalar(
            st,
            &expert_tensor_name(layer, id, proj, Nvfp4Part::InputScale),
        )?);
    }
    Ok((weight_scale_2, input_scale))
}

#[cfg(feature = "vulkan")]
fn expert_scalar(st: &SafeTensorsDir, name: &str) -> Result<f32> {
    let info = st
        .tensor(name)
        .ok_or_else(|| anyhow!("{name} missing from the checkpoint"))?;
    read_scalar_f32(st, info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::qwen4_names::{HcPart, HcSite};

    const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
    const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";
    /// The 8060S's device-local heap, read off the device
    /// (`VulkanContext::memory_heaps`): 74.4322 GiB. The plan is checked against
    /// this rather than against whatever the running box reports so the budget
    /// is a fact under test, not a tautology.
    const HEAP: u64 = 79_920_955_392;
    /// Set to run the whole 71 GiB residency (minutes, and the plan must fit).
    #[cfg(feature = "vulkan")]
    const FULL_ENV: &str = "ARLE_QWEN4_UPLOAD_FULL";

    /// Layer 0 (linear attention) non-expert weights, the stream mixer, and
    /// `layers.1.ple.layer_multipliers`.
    const SHARD_HC: &str = "model-bf16-00001.safetensors";
    /// `embed_tokens` and `lm_head`.
    const SHARD_EMBED: &str = "model-bf16-00012.safetensors";
    /// Layer 0's first 128 routed experts.
    const SHARD_EXPERTS0: &str = "layer-00000-experts-0000-0127.safetensors";

    fn checkpoint_dir() -> Option<PathBuf> {
        let dir = std::env::var_os(CKPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(CKPT_DEFAULT));
        dir.join("model.safetensors.index.json")
            .is_file()
            .then_some(dir)
    }

    /// The three shards a layer-0 subset needs: ~1.7 GiB of mapping, versus the
    /// 122 GiB `open_dir` would map for the whole checkpoint.
    fn open_subset(dir: &std::path::Path) -> SafeTensorsDir {
        SafeTensorsDir::open_files(&[
            dir.join(SHARD_HC),
            dir.join(SHARD_EMBED),
            dir.join(SHARD_EXPERTS0),
        ])
        .expect("open the layer-0 subset shards")
    }

    fn subset_scope() -> Qwen4UploadScope {
        Qwen4UploadScope {
            layers: Some(vec![0]),
            experts: Some(128),
            lm_head: false,
        }
    }

    // -------------------------------------------------------------- policy

    #[test]
    fn device_format_follows_the_consuming_kernel() {
        use Qwen4DeviceFormat::{F16, F32, Nvfp4};
        use Qwen4TensorKind::*;
        let dense = F16;
        let fmt = |k| device_format(k, dense);

        // Packed: the only tier the fused MoE GEMV can read.
        assert_eq!(
            fmt(Expert {
                proj: ExpertProj::Gate,
                part: Nvfp4Part::Packed
            }),
            Some(Nvfp4)
        );
        // The scale plane is folded into the packed stream, not bound.
        assert_eq!(
            fmt(Expert {
                proj: ExpertProj::Gate,
                part: Nvfp4Part::BlockScale
            }),
            None
        );
        assert_eq!(
            fmt(Expert {
                proj: ExpertProj::Down,
                part: Nvfp4Part::GlobalScale
            }),
            None
        );

        // F32 because the shader binds `float w[]`. If any of these ever
        // reports F16, the four fused qwen4 kernels read halves as floats.
        for kind in [
            HyperConnection {
                site: HcSite::Attn,
                part: HcPart::MixUp,
            },
            HyperConnection {
                site: HcSite::Mixer,
                part: HcPart::Norm,
            },
            LinearAttnConv1d,
            LinearAttnALog,
            LinearAttnNorm,
            AttnQNorm,
            IndexerKNorm,
            MoeRouter,
            SharedExpertGate,
            PleNormKey,
            PleConv1d,
        ] {
            assert_eq!(fmt(kind), Some(F32), "{kind:?} must stay F32");
        }

        // The dense GEMV weights follow the config.
        for kind in [
            LinearAttnInProjQkv,
            AttnQProj,
            AttnOProj,
            SharedExpertDownProj,
            PleKeyProj,
            LmHead,
        ] {
            assert_eq!(fmt(kind), Some(F16), "{kind:?} at dense=F16");
            assert_eq!(
                device_format(kind, F32),
                Some(F32),
                "{kind:?} must follow dense_format"
            );
        }

        // Host tables and dropped families get no buffer.
        for kind in [
            EmbedTokens,
            PleNgramShard,
            PleNgramWeightScale,
            ExpertsStackedGateUp,
            MtpFcHidden,
            Vision(crate::qwen4_names::VisionSlot::Merger),
        ] {
            assert_eq!(fmt(kind), None, "{kind:?} must not get a device buffer");
        }
    }

    /// The fold is a per-CONSUMER decision, not a per-class one: three
    /// `Qwen4ExpTextRMSNorm` weights must stay raw because their shader already
    /// applies the `+ 1`.
    #[test]
    fn norm_bias_fold_covers_exactly_the_vendored_rms_norm_consumers() {
        use Qwen4TensorKind::*;
        for kind in [
            HyperConnection {
                site: HcSite::Attn,
                part: HcPart::Norm,
            },
            HyperConnection {
                site: HcSite::Mlp,
                part: HcPart::Norm,
            },
            HyperConnection {
                site: HcSite::Mixer,
                part: HcPart::Norm,
            },
            AttnQNorm,
            AttnKNorm,
            IndexerQNorm,
            IndexerKNorm,
        ] {
            assert!(folds_norm_bias(kind), "{kind:?} needs the 1 + w fold");
        }
        for kind in [
            // `qwen4_ple_gate.comp` spells `(1.0 + norm_key_w[...])` itself.
            PleNormKey,
            PleNormQuery,
            PleNormConv,
            // `Qwen4ExpTextRMSNormGated`: plain `weight *`, ones-initialised.
            LinearAttnNorm,
            // Not norms at all.
            HyperConnection {
                site: HcSite::Attn,
                part: HcPart::MixDown,
            },
            MoeRouter,
        ] {
            assert!(!folds_norm_bias(kind), "{kind:?} must NOT be folded");
        }
    }

    #[test]
    fn nvfp4_row_bytes_matches_the_block_geometry() {
        // 2560 values = 40 blocks x 36 B. Equal to the checkpoint's own
        // `1280 (qs) + 160 (scale)` per row, i.e. the repack costs no VRAM.
        assert_eq!(nvfp4_row_bytes(2560), Some(1440));
        assert_eq!(nvfp4_row_bytes(2560), Some(1280 + 160));
        assert_eq!(nvfp4_row_bytes(640), Some(360));
        assert_eq!(nvfp4_row_bytes(0), None);
        assert_eq!(
            nvfp4_row_bytes(48),
            None,
            "48 is not a whole 64-value block"
        );
        assert_eq!(
            Qwen4DeviceFormat::Nvfp4.bytes_for(2560, 640),
            Some(640 * 1440)
        );
        assert_eq!(Qwen4DeviceFormat::F16.bytes_for(2560, 640), Some(3_276_800));
        assert_eq!(Qwen4DeviceFormat::F32.bytes_for(2560, 640), Some(6_553_600));
    }

    #[test]
    fn f16_round_trip_pins_the_corner_cases() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF); // largest finite half
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(1.0 / 16384.0), 0x0400); // smallest normal
        assert_eq!(f32_to_f16(5.9604645e-8), 0x0001); // smallest subnormal
        assert!(f32_to_f16(f32::NAN) & 0x7C00 == 0x7C00 && f32_to_f16(f32::NAN) & 0x3FF != 0);
        // BF16 widening is exact: 0x3F80 is 1.0.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn ensure_fits_refuses_an_over_budget_plan() {
        let plan = Qwen4Plan {
            device_bytes: 72 << 30,
            ..Default::default()
        };
        // 74.43 GiB heap, 1 GiB reserve -> fits.
        assert!(plan.ensure_fits(79_918_820_000, 1 << 30).is_ok());
        // The qwen35 loader's 3 GiB reserve does NOT fit this model.
        let err = plan
            .ensure_fits(79_918_820_000, 3 << 30)
            .expect_err("3 GiB reserve must not fit");
        assert!(
            err.to_string().contains("over by"),
            "message should quantify the overage: {err}"
        );
        assert!(plan.ensure_fits(60 << 30, 0).is_err());
    }

    #[test]
    fn dense_shape_reads_gguf_order_and_flattens_the_depthwise_axis() {
        let info = |dims: Vec<u64>| SafeTensorInfo {
            name: "t".into(),
            dims,
            dtype: "BF16".into(),
            ggml_type: None,
            offset: 0,
            len: 0,
        };
        // header [12288, 2560] -> dims reversed -> (ncols=2560, nrows=12288)
        assert_eq!(
            dense_shape(&info(vec![2560, 12288])).unwrap(),
            (2560, 12288)
        );
        assert_eq!(dense_shape(&info(vec![10240])).unwrap(), (10240, 1));
        // conv1d header [10240, 1, 4] -> dims [4, 1, 10240] -> one flat row.
        assert_eq!(dense_shape(&info(vec![4, 1, 10240])).unwrap(), (40960, 1));
        assert!(dense_shape(&info(vec![2, 3, 4, 5])).is_err());
    }

    #[test]
    fn hyper_connection_prefix_rejects_impossible_pairs() {
        assert_eq!(
            hyper_connection_prefix(Some(3), HcSite::Attn).unwrap(),
            "model.language_model.layers.3.attn_hyper_connection"
        );
        assert_eq!(
            hyper_connection_prefix(None, HcSite::Mixer).unwrap(),
            MIXER_PREFIX
        );
        assert!(hyper_connection_prefix(Some(3), HcSite::Mixer).is_err());
        assert!(hyper_connection_prefix(None, HcSite::Attn).is_err());
    }

    #[test]
    fn tensor_names_round_trip_through_the_classifier() {
        for name in [
            layer_tensor_name(7, "self_attn.q_proj.weight"),
            layer_tensor_name(0, "linear_attn.in_proj_qkv.weight"),
            expert_tensor_name(0, 511, ExpertProj::Down, Nvfp4Part::GlobalScale),
            expert_tensor_name(47, 0, ExpertProj::Gate, Nvfp4Part::Packed),
            format!("{MIXER_PREFIX}.hc_norm.weight"),
            LM_HEAD_NAME.to_string(),
            EMBED_TOKENS_NAME.to_string(),
        ] {
            classify_qwen4_tensor(&name).unwrap_or_else(|e| panic!("classify {name}: {e}"));
        }
        // The synthetic stack name is deliberately NOT a checkpoint name.
        assert!(classify_qwen4_tensor(&expert_stack_name(0, ExpertProj::Gate)).is_err());
    }

    // ---------------------------------------------------------- real checkpoint

    /// Header-only: the plan for layer 0 has to name every tensor that layer
    /// has, at the byte counts the shapes imply. No device needed.
    #[test]
    fn subset_plan_covers_layer_zero() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = open_subset(&dir);
        let cfg = Qwen4UploadConfig::default();
        let plan = plan_qwen4_upload(&st, &cfg, &subset_scope()).expect("plan the subset");

        assert_eq!(
            plan.layer_kinds.get(&0),
            Some(&Qwen4LayerKind::LinearAttention),
            "layer 0 is a linear-attention layer (`layer_types[0]`)"
        );
        // 3 stacks + 22 dense layer-0 tensors + 3 mixer tensors.
        let stacks = plan
            .items
            .iter()
            .filter(|i| matches!(i.source, Qwen4Source::Nvfp4Stack { .. }))
            .count();
        assert_eq!(stacks, 3, "gate / up / down");
        for (proj, ncols, nrows) in [
            (ExpertProj::Gate, 2560, 640),
            (ExpertProj::Up, 2560, 640),
            (ExpertProj::Down, 640, 2560),
        ] {
            let name = expert_stack_name(0, proj);
            let item = plan
                .items
                .iter()
                .find(|i| i.name == name)
                .unwrap_or_else(|| panic!("{name} missing from the plan"));
            assert_eq!((item.ncols, item.nrows), (ncols, nrows), "{name} shape");
            assert_eq!(
                item.bytes,
                (128 * nrows * nvfp4_row_bytes(ncols).unwrap()) as u64,
                "{name} bytes"
            );
        }
        // `lm_head` was excluded, `embed_tokens` is host-resident, and the
        // mixer is the only global left.
        assert!(
            plan.items.iter().all(|i| i.name != LM_HEAD_NAME),
            "lm_head must honour the scope"
        );
        assert!(
            plan.items.iter().all(|i| i.name != EMBED_TOKENS_NAME),
            "embed_tokens never gets a device buffer"
        );
        // The mixer is `use_combine=false`, so three F32 tensors and no
        // `block_inject_weight`: 10240 + 2 x (320 x 10240) elements.
        assert_eq!(
            plan.global_bytes,
            (10240 + 2 * 320 * 10240) * 4,
            "mixer = hc_norm + mix_down + mix_up, all F32"
        );
        assert!(plan.host_bytes >= 1 << 30, "embed_tokens is 1.18 GiB");
        // `model-bf16-00001` carries the whole 27-block vision tower alongside
        // layer 0, so the Drop tier is exercised here and not hypothetical.
        assert!(
            plan.dropped_bytes > 0,
            "this shard holds the vision tower; the Drop tier must account for it"
        );
        assert!(
            plan.items
                .iter()
                .all(|i| i.name.starts_with(TEXT_PREFIX) || i.name == LM_HEAD_NAME),
            "only the text stream may reach the device"
        );
    }

    /// A scoped layer whose expert shard was never opened must fail loud rather
    /// than upload a decoder layer with a hole where its MoE should be.
    #[test]
    fn missing_expert_shard_fails_loud() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = SafeTensorsDir::open_files(&[dir.join(SHARD_HC), dir.join(SHARD_EMBED)])
            .expect("open two bf16 shards");
        let err = plan_qwen4_upload(
            &st,
            &Qwen4UploadConfig::default(),
            &Qwen4UploadScope {
                layers: Some(vec![0]),
                experts: None,
                lm_head: false,
            },
        )
        .expect_err("layer 0 has no experts here");
        assert!(
            err.to_string().contains("no routed experts"),
            "unexpected error: {err}"
        );
        // Opting out explicitly is allowed.
        plan_qwen4_upload(
            &st,
            &Qwen4UploadConfig::default(),
            &Qwen4UploadScope {
                layers: Some(vec![0]),
                experts: Some(0),
                lm_head: false,
            },
        )
        .expect("experts: Some(0) skips the MoE on purpose");
    }

    #[test]
    fn host_tables_gather_an_embedding_row() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = open_subset(&dir);
        let host = Qwen4HostTables::build(&st).expect("host tables");
        assert_eq!((host.hidden(), host.vocab()), (2560, 248_320));

        // Independent oracle: slice the mmap by hand and widen the BF16.
        let raw = st.tensor_data(EMBED_TOKENS_NAME).expect("embed data");
        let tok = 248_044usize; // eos
        let want: Vec<f32> = raw[tok * 2560 * 2..][..2560 * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(host.embed_row(tok).expect("row"), want);
        assert!(host.embed_row(248_320).is_err(), "out-of-range token");
        // The n-gram shards are not in this view; asking must name that.
        assert!(host.ngram().is_err());
    }

    // -------------------------------------------------------- device upload

    /// A real, small residency: layer 0 with 128 experts plus the stream mixer,
    /// ~0.5 GiB. Asserts what the forward depends on and nothing else:
    /// byte-exactness of an F16 and an F32 tensor, the `1 + w` fold (and its
    /// absence where it must be absent), and that the repacked NVFP4 stack
    /// round-trips through an independent CPU dequantizer at the expert offset
    /// the fused GEMV will use.
    #[cfg(feature = "vulkan")]
    #[test]
    fn subset_upload_lands_byte_exact_on_device() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let ctx = match vulkan_sys::VulkanContext::create() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP: no Vulkan device ({e})");
                return;
            }
        };
        eprintln!("qwen4 upload device: {}", ctx.device_name());

        let st = open_subset(&dir);
        let cfg = Qwen4UploadConfig::default();
        let plan = plan_qwen4_upload(&st, &cfg, &subset_scope()).expect("plan");
        eprintln!(
            "qwen4 subset plan: {:.3} GiB over {} suballocations",
            plan.device_bytes as f64 / (1u64 << 30) as f64,
            plan.items.len()
        );
        let w = upload_qwen4(&ctx, &st, &plan, &cfg).expect("upload");

        assert_eq!(w.layer_kind(0), Some(Qwen4LayerKind::LinearAttention));
        assert_eq!(w.resident_layers().collect::<Vec<_>>(), vec![0]);

        // --- 1. F32 byte-exactness, no fold: the MoE router.
        let router = layer_tensor_name(0, "mlp.gate.weight");
        let t = *w.tensor(&router).expect("router resident");
        assert_eq!(t.format, Qwen4DeviceFormat::F32);
        assert_eq!((t.ncols, t.nrows), (2560, 512));
        let mut back = vec![0u8; 4 * 4096];
        w.read_back(&t, &mut back).expect("router read-back");
        let src = st.tensor_data(&router).expect("router bytes");
        for (i, chunk) in back.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(chunk.try_into().unwrap());
            let want = bf16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
            assert_eq!(got, want, "router element {i}");
        }

        // --- 2. F16 byte-exactness: a dense GEMV weight.
        let out_proj = layer_tensor_name(0, "linear_attn.out_proj.weight");
        let t = *w.tensor(&out_proj).expect("out_proj resident");
        assert_eq!(t.format, Qwen4DeviceFormat::F16);
        assert_eq!((t.ncols, t.nrows), (6144, 2560));
        let mut back = vec![0u8; 2 * 4096];
        w.read_back(&t, &mut back).expect("out_proj read-back");
        let src = st.tensor_data(&out_proj).expect("out_proj bytes");
        for (i, chunk) in back.chunks_exact(2).enumerate() {
            let got = u16::from_le_bytes(chunk.try_into().unwrap());
            let want = f32_to_f16(bf16_to_f32(u16::from_le_bytes([
                src[i * 2],
                src[i * 2 + 1],
            ])));
            assert_eq!(got, want, "out_proj element {i}");
        }

        // --- 3. THE LOADER CONTRACT: `1 + hc_norm` on the device.
        let hc = w
            .hyper_connection(Some(0), HcSite::Attn)
            .expect("layer 0 attn hyper-connection");
        assert!(hc.block_inject.is_some(), "a layer site has block_inject");
        assert!(
            w.hyper_connection(None, HcSite::Mixer)
                .expect("the mixer is resident")
                .block_inject
                .is_none(),
            "the mixer is use_combine=false and must report no block_inject"
        );
        assert_eq!(hc.hc_norm.ncols, 10240);
        let name = layer_tensor_name(0, "attn_hyper_connection.hc_norm.weight");
        let mut back = vec![0u8; 10240 * 4];
        w.read_back(hc.hc_norm, &mut back).expect("hc_norm");
        let src = st.tensor_data(&name).expect("hc_norm bytes");
        let mut differs_from_raw = 0usize;
        for (i, (d, s)) in back.chunks_exact(4).zip(src.chunks_exact(2)).enumerate() {
            let got = f32::from_le_bytes(d.try_into().unwrap());
            let raw = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
            assert_eq!(got, 1.0 + raw, "hc_norm[{i}] must be stored pre-biased");
            if got != raw {
                differs_from_raw += 1;
            }
        }
        // Guards the test itself: if every raw weight were 0 the fold would be
        // indistinguishable from a bug that writes 1.0 everywhere.
        assert!(
            differs_from_raw == 10240,
            "the fold must change every element; {differs_from_raw}/10240 differ"
        );

        // --- 4. ...and the norms that must NOT be folded.
        let gated = layer_tensor_name(0, "linear_attn.norm.weight");
        let t = *w.tensor(&gated).expect("linear_attn norm resident");
        let mut back = vec![0u8; 128 * 4];
        w.read_back(&t, &mut back).expect("linear_attn norm");
        let src = st.tensor_data(&gated).expect("norm bytes");
        for (i, (d, s)) in back.chunks_exact(4).zip(src.chunks_exact(2)).enumerate() {
            let got = f32::from_le_bytes(d.try_into().unwrap());
            let raw = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
            assert_eq!(
                got, raw,
                "linear_attn.norm[{i}] is RMSNormGated — it must stay RAW"
            );
        }

        // --- 5. The NVFP4 stack: repack round-trips, at the right offset.
        let stack = w.expert_stack(0, ExpertProj::Gate).expect("gate stack");
        assert_eq!(stack.n_experts, 128);
        assert_eq!(stack.batch_stride_a(), 2560 * 640);
        assert_eq!(stack.weight_scale_2.len(), 128);
        assert_eq!(stack.input_scale.len(), 128);
        assert!(
            stack
                .weight_scale_2
                .iter()
                .all(|v| v.is_finite() && *v > 0.0),
            "every weight_scale_2 must be a usable positive scale"
        );
        // The scalars are the checkpoint's, in expert-ID order.
        for id in [0u32, 1, 127] {
            let name = expert_tensor_name(0, id, ExpertProj::Gate, Nvfp4Part::GlobalScale);
            let raw = st.tensor_data(&name).expect("weight_scale_2");
            assert_eq!(
                stack.weight_scale_2[id as usize],
                f32::from_le_bytes(raw[..4].try_into().unwrap()),
                "weight_scale_2[{id}]"
            );
        }

        // The fused GEMV indexes its scale binding by SLOT, so the gather has
        // to reorder — binding `weight_scale_2` itself is the bug.
        let mut scale0 = Vec::new();
        stack
            .scale0_for_route(&[7, 3, 3], &mut scale0)
            .expect("scale0 gather");
        assert_eq!(
            scale0,
            vec![
                stack.weight_scale_2[7],
                stack.weight_scale_2[3],
                stack.weight_scale_2[3]
            ]
        );
        assert!(stack.scale0_for_route(&[128], &mut scale0).is_err());
        assert!(stack.scale0_for_route(&[-1], &mut scale0).is_err());

        let row_bytes = nvfp4_row_bytes(2560).unwrap();
        for expert in [0usize, 5, 127] {
            let offset = stack.expert_offset(expert).expect("expert offset");
            assert_eq!(offset, (expert * 640 * row_bytes) as u64);
            // Read this expert's FIRST row back out of the stack...
            let mut row = vec![0u8; row_bytes];
            read_at(&w, stack, offset, &mut row);
            let got =
                infer_gguf::dequant::dequantize_row_nvfp4(&row, 2560).expect("ggml nvfp4 dequant");
            // ...and decode the same row straight from the checkpoint's own
            // planes, in ModelOpt's nibble order. The two agree only if the
            // repack really rewrote the nibble layout.
            let qs = st
                .tensor_data(&expert_tensor_name(
                    0,
                    expert as u32,
                    ExpertProj::Gate,
                    Nvfp4Part::Packed,
                ))
                .expect("qs plane");
            let sc = st
                .tensor_data(&expert_tensor_name(
                    0,
                    expert as u32,
                    ExpertProj::Gate,
                    Nvfp4Part::BlockScale,
                ))
                .expect("scale plane");
            let want = modelopt_row(&qs[..1280], &sc[..160]);
            assert_eq!(got, want, "expert {expert} row 0 after repack");
            assert!(
                want.iter().any(|v| *v != 0.0),
                "a row of zeros would make the comparison vacuous"
            );
        }

        eprintln!(
            "PASS: qwen4 subset residency, {} tensors + {} expert stacks on {}",
            w.tensors.len(),
            w.experts.len(),
            ctx.device_name()
        );
    }

    /// Read `len` bytes at `offset` bytes into a stack's suballocation.
    #[cfg(feature = "vulkan")]
    fn read_at(w: &Qwen4Weights<'_, '_>, stack: &Qwen4ExpertStack, offset: u64, dst: &mut [u8]) {
        // `SlabAllocator::read_back` always starts at the suballocation's own
        // offset, so read the prefix up to and including the row we want and
        // keep the tail. Kept small on purpose: expert 127's prefix is 176 MiB,
        // which is a verification-path cost, not a decode-path one.
        let take = usize::try_from(offset).expect("offset fits usize") + dst.len();
        let mut prefix = vec![0u8; take];
        w.read_back(&stack.tensor, &mut prefix)
            .expect("stack read-back");
        dst.copy_from_slice(&prefix[take - dst.len()..]);
    }

    /// Decode one NVFP4 row from the checkpoint's SEPARATE planes using
    /// MODELOPT's nibble layout: value `2i` in the low nibble of byte `i`,
    /// `2i+1` in the high nibble, one UE4M3 scale per 16 values.
    ///
    /// This is the third opinion the round-trip above needs. What it isolates
    /// is the NIBBLE ORDER: `repack_nvfp4_planes` copies each scale byte
    /// verbatim, so the scale arithmetic here necessarily agrees with ggml's by
    /// construction and cannot be the thing under test. The element order can —
    /// and a straight `d || qs` copy would permute every group of 16 weights
    /// while still producing plausible, finite output.
    #[cfg(feature = "vulkan")]
    fn modelopt_row(qs: &[u8], scales: &[u8]) -> Vec<f32> {
        /// E2M1 magnitudes for nibble codes 0..8, sign in bit 3.
        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mut out = Vec::with_capacity(qs.len() * 2);
        for &byte in qs {
            for nib in [byte & 0xF, byte >> 4] {
                let v = E2M1[(nib & 0x7) as usize];
                // Nibble 8 is a legal -0; ggml keeps the sign off zero.
                out.push(if nib & 0x8 != 0 && v != 0.0 { -v } else { v });
            }
        }
        for (group, chunk) in out.chunks_mut(QK_NVFP4_SUB).enumerate() {
            // UE4M3 `x eeee mmm`: bit 7 unused, 4-bit exponent biased by 7,
            // 3-bit mantissa; 0x00 and 0x7F (E4M3's NaN) both decode to 0.
            let code = u32::from(scales[group] & 0x7F);
            let (exp, mant) = (code >> 3, (code & 0x7) as f32 / 8.0);
            let scale = if code == 0 || code == 0x7F {
                0.0
            } else if exp == 0 {
                mant * 2f32.powi(-6)
            } else {
                (1.0 + mant) * 2f32.powi(exp as i32 - 7)
            };
            for v in chunk {
                *v *= scale;
            }
        }
        out
    }

    /// The answer to "does this model fit?", from the HEADERS of all 206 shards
    /// and nothing else — no device, no tensor data, no upload.
    ///
    /// The numbers here are the ones the module docs quote; if the checkpoint or
    /// the format policy moves, this is where it shows.
    #[test]
    fn full_plan_fits_the_device_local_heap() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = SafeTensorsDir::open_dir(&dir).expect("open the whole checkpoint");
        assert_eq!(st.tensors().len(), 296_475, "checkpoint tensor count");

        let cfg = Qwen4UploadConfig::default();
        let plan =
            plan_qwen4_upload(&st, &cfg, &Qwen4UploadScope::full()).expect("plan the full model");
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "qwen4 full plan: device {:.2} GiB over {} suballocations              (host {:.2} GiB, dropped {:.2} GiB)",
            gib(plan.device_bytes),
            plan.items.len(),
            gib(plan.host_bytes),
            gib(plan.dropped_bytes),
        );

        assert_eq!(plan.layer_kinds.len(), 48, "all 48 decoder layers");
        assert_eq!(
            plan.layer_kinds
                .values()
                .filter(|k| **k == Qwen4LayerKind::FullAttention)
                .count(),
            12,
            "full_attention_interval 4 over 48 layers"
        );
        assert_eq!(
            plan.items
                .iter()
                .filter(|i| matches!(i.source, Qwen4Source::Nvfp4Stack { .. }))
                .count(),
            48 * 3,
            "one stacked suballocation per (layer, projection)"
        );
        // 512 experts x 3 projections x 48 layers of block_nvfp4.
        let expert_bytes: u64 = plan
            .items
            .iter()
            .filter(|i| matches!(i.source, Qwen4Source::Nvfp4Stack { .. }))
            .map(|i| i.bytes)
            .sum();
        assert_eq!(expert_bytes, 67_947_724_800, "the packed expert tier");
        // 47.68 GiB of n-gram table + 1.18 GiB of embed_tokens stay borrowed.
        assert!(
            gib(plan.host_bytes) > 48.0,
            "host tier is {:.2} GiB, expected the whole n-gram table",
            gib(plan.host_bytes)
        );

        // The measured budget, and the reason `DEFAULT_RESERVE_BYTES` is 1 GiB.
        plan.ensure_fits(HEAP, DEFAULT_RESERVE_BYTES)
            .expect("the shipping plan must fit with the default reserve");
        assert!(
            plan.ensure_fits(HEAP, 3 << 30).is_err(),
            "it does NOT fit with the qwen35 loader's 3 GiB reserve — that is why              DEFAULT_RESERVE_BYTES is smaller, and the margin is worth knowing about"
        );
        // F32 everywhere would blow the heap outright; this is what makes the
        // F16 dense tier load-bearing rather than a preference.
        let f32_plan = plan_qwen4_upload(
            &st,
            &Qwen4UploadConfig {
                dense_format: Qwen4DeviceFormat::F32,
                ..Qwen4UploadConfig::default()
            },
            &Qwen4UploadScope::full(),
        )
        .expect("plan at F32");
        assert!(
            f32_plan.ensure_fits(HEAP, DEFAULT_RESERVE_BYTES).is_err(),
            "an all-F32 dense tier is {:.2} GiB and must not fit",
            gib(f32_plan.device_bytes)
        );
    }

    /// The packing sweep is worth more than a tuning nicety, so pin the gap.
    ///
    /// Device-free (`SlabPlan` needs no GPU) but feature-gated, since
    /// `vulkan-sys` is an optional dependency.
    #[cfg(feature = "vulkan")]
    #[test]
    fn choosing_the_slab_size_recovers_over_a_gibibyte() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = SafeTensorsDir::open_dir(&dir).expect("open the whole checkpoint");
        let plan = plan_qwen4_upload(
            &st,
            &Qwen4UploadConfig::default(),
            &Qwen4UploadScope::full(),
        )
        .expect("plan the full model");

        // 2 GiB is `maxMemoryAllocationSize` on this part — the size a loader
        // that did not sweep would naturally pick.
        let naive = plan.pack(2 << 30, 16).expect("pack at the device maximum");
        let chosen = plan.choose_packing(2 << 30, 16).expect("sweep");
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "qwen4 packing: naive {:.3} GiB / {} slabs -> chosen {:.3} GiB / {} slabs              of {:.0} MiB ({:.2}% waste), {:.3} GiB of heap left",
            gib(naive.committed_bytes),
            naive.slab_count,
            gib(chosen.committed_bytes),
            chosen.slab_count,
            chosen.slab_bytes as f64 / (1u64 << 20) as f64,
            100.0 * chosen.waste_fraction(&plan),
            gib(HEAP - chosen.committed_bytes),
        );

        assert!(
            chosen.committed_bytes >= plan.device_bytes,
            "a packing cannot commit less than the plan asks for"
        );
        assert!(
            chosen.committed_bytes <= naive.committed_bytes,
            "the sweep must never be worse than the device maximum"
        );
        assert!(
            naive.committed_bytes - chosen.committed_bytes > 1 << 30,
            "the sweep is only worth carrying if it recovers real heap; it recovered {:.3} GiB",
            gib(naive.committed_bytes - chosen.committed_bytes)
        );
        assert!(
            chosen.waste_fraction(&plan) < 0.005,
            "chosen packing wastes {:.2}%",
            100.0 * chosen.waste_fraction(&plan)
        );
        // The whole point: the default reserve is actually available.
        chosen
            .ensure_fits(HEAP, DEFAULT_RESERVE_BYTES)
            .expect("the chosen packing must leave the reserve");
        assert!(
            naive.ensure_fits(HEAP, DEFAULT_RESERVE_BYTES).is_err(),
            "at the device maximum the reserve is NOT available — that is the bug              this sweep exists to avoid"
        );
    }

    /// The whole ~71 GiB residency, actually uploaded.
    ///
    /// Opt-in: it needs the entire device-local heap and runs for minutes, so it
    /// is a bring-up gate, not a per-commit test.
    /// `full_plan_fits_the_device_local_heap` already proves the arithmetic;
    /// what only this can prove is that 1210 real suballocations of it survive
    /// `vkAllocateMemory` and the staged writes.
    #[cfg(feature = "vulkan")]
    #[test]
    fn full_residency_fits_and_uploads() {
        if std::env::var_os(FULL_ENV).is_none() {
            eprintln!("SKIP: set {FULL_ENV}=1 to run the full ~71 GiB residency");
            return;
        }
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let ctx = match vulkan_sys::VulkanContext::create() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP: no Vulkan device ({e})");
                return;
            }
        };
        let st = SafeTensorsDir::open_dir(&dir).expect("open the whole checkpoint");
        let cfg = Qwen4UploadConfig::default();
        let plan =
            plan_qwen4_upload(&st, &cfg, &Qwen4UploadScope::full()).expect("plan the full model");
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "qwen4 full plan: device {:.2} GiB ({} items), host {:.2} GiB, dropped {:.2} GiB",
            gib(plan.device_bytes),
            plan.items.len(),
            gib(plan.host_bytes),
            gib(plan.dropped_bytes),
        );
        let started = std::time::Instant::now();
        let w = upload_qwen4(&ctx, &st, &plan, &cfg).expect("full upload");
        assert_eq!(w.resident_layers().count(), 48);
        // Every layer's three expert stacks are addressable at the offset the
        // fused GEMV will compute for the last expert.
        for layer in 0..48 {
            for proj in [ExpertProj::Gate, ExpertProj::Up, ExpertProj::Down] {
                let stack = w.expert_stack(layer, proj).expect("expert stack");
                assert_eq!(stack.n_experts, 512);
                assert!(stack.expert_offset(511).is_ok());
                assert!(stack.expert_offset(512).is_err());
            }
        }
        w.hyper_connection(None, HcSite::Mixer)
            .expect("the stream mixer is the final norm");
        assert_eq!(
            w.host().ngram().expect("n-gram table").rows(),
            320_001_536,
            "the derived padded n-gram vocab"
        );
        eprintln!(
            "PASS: qwen4 full residency in {:.1} s, {:.2} GiB committed over {} slabs              ({:.2}% waste)",
            started.elapsed().as_secs_f64(),
            gib(w.slabs().committed_bytes()),
            w.slabs().slab_count(),
            100.0 * w.slabs().wasted_bytes() as f64 / w.slabs().committed_bytes() as f64,
        );
    }
}
