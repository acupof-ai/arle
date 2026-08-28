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
//! [`Qwen4UploadConfig::reserve_bytes`] defaults to 1.5 GiB and not to the 3 GiB
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

use crate::loader::DeviceBudget;
use crate::qwen4_names::{
    ExpertProj, HcPart, HcSite, Nvfp4Part, Qwen4Residency, Qwen4Stream, Qwen4TensorKind,
    Qwen4TensorRole, classify_qwen4_tensor,
};

/// `model.language_model.` — the text stream's prefix.
pub const TEXT_PREFIX: &str = "model.language_model.";
/// The stream-level `hyper_connection_mixer` (`use_combine=false`, so no
/// `block_inject_weight`); it is this model's missing final norm.
pub const MIXER_PREFIX: &str = "model.language_model.hyper_connection_mixer";
/// `mtp.` — the 1-layer multi-token-prediction head's prefix.
pub const MTP_PREFIX: &str = "mtp.";
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
/// 1.5 GiB, not the 3 GiB `crate::loader` reserves. The model does not fit with 3
/// GiB, and it does not need it: measured from the config, this model's
/// device-side state is 48 MiB of KV cache (12 full-attention layers x 2048
/// context x 2 KV heads x 256 head_dim x K+V x f16), 108 MiB of recurrent state
/// (36 linear layers x 48 value heads x 128 x 128 x f32), ~6 MiB of conv rings,
/// and an arena whose widest tensors are the 10240-wide residual and the 248320
/// f32 logits. 1 GiB is roughly 4x that, and — with the slab sweep in
/// [`Qwen4Plan::choose_packing`] — it is available: the full residency commits
/// 72.66 GiB of the 74.43 GiB heap.
/// 1.5 GiB: the KV planes (12 x ~4 MB), the resident linear-attention state
/// (36 x 3.1 MB), the arena, descriptor pools, and slack for the driver's own
/// allocations. The 1 GiB first guess tripped its own fail-loud check the
/// moment the BF16 hyper-connection tier shrank the plan enough for
/// spill_to_fit to keep an extra expert stack on device.
pub const DEFAULT_RESERVE_BYTES: u64 = 3 << 29;

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

/// The pseudo layer index that addresses the MTP head's hyper-connection
/// sites through [`hyper_connection_prefix`].
///
/// The MTP block is `mtp.layers.0.*` — a 49th decoder layer in every respect
/// except its name — and unlike the text stream it has its OWN
/// `hyper_connection_mixer` (use_combine=False, collapsing its 10240 output
/// to 2560 before the shared `lm_head`). `usize::MAX` cannot collide with a
/// real text layer, so the existing `(layer, site)` plumbing (`Qwen4Dev::
/// hc_pre` / `hc_combine`) drives the MTP sites with no second code path.
pub const MTP_HC_LAYER: usize = usize::MAX;

/// `mtp.layers.0.mlp.experts.<expert>.<proj>_proj.weight` — the synthetic name
/// of one MTP routed expert's slice of the stacked BF16 parameters.
///
/// The checkpoint stores the MTP experts STACKED (`experts.gate_up_proj`
/// `[512, 1280, 2560]`, `experts.down_proj` `[512, 2560, 640]` — the
/// quant-excluded `Qwen4ExpTextExperts` layout, gate rows first then up, per
/// `chunk(2, dim=-1)` in modeling_qwen4_exp.py:889). The plan slices them
/// per-expert so each slice is an ordinary dense-GEMV tensor: the MoE of one
/// DRAFTED token touches 10 of 512 experts, and a per-expert suballocation is
/// what lets those ten record as plain `record_dense_at` GEMVs (and lets the
/// spill tier demote the cold 98% without splitting logic). No such per-expert
/// tensor exists in the file — the classifier would reject the name — which is
/// exactly what keeps it collision-free, like [`expert_stack_name`].
#[must_use]
pub fn mtp_expert_slice_name(expert: u32, proj: ExpertProj) -> String {
    let proj = match proj {
        ExpertProj::Gate => "gate",
        ExpertProj::Up => "up",
        ExpertProj::Down => "down",
    };
    format!("{MTP_PREFIX}layers.0.mlp.experts.{expert}.{proj}_proj.weight")
}

/// Prefix of one `Qwen4ExpTextGatedResidual`'s four weights.
///
/// `(Some(l), Attn | Mlp)` for a layer's two sites, `(None, Mixer)` for the
/// stream-level one, and [`MTP_HC_LAYER`] for the MTP head's three (the MTP
/// block carries its own mixer). The remaining combinations do not exist in
/// this architecture and are refused rather than formatted into a name that
/// will simply not be resident.
pub fn hyper_connection_prefix(layer: Option<usize>, site: HcSite) -> Result<String> {
    Ok(match (layer, site) {
        (Some(MTP_HC_LAYER), HcSite::Attn) => format!("{MTP_PREFIX}layers.0.attn_hyper_connection"),
        (Some(MTP_HC_LAYER), HcSite::Mlp) => format!("{MTP_PREFIX}layers.0.mlp_hyper_connection"),
        (Some(MTP_HC_LAYER), HcSite::Mixer) => format!("{MTP_PREFIX}hyper_connection_mixer"),
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
    /// The checkpoint's own BF16 bytes, verbatim — no re-encode at load
    /// (a pure memcpy where F16 walks 686 M elements through a rounding
    /// step), and no precision loss (the bf16->f16 re-encode changes
    /// 221,186 dense weights — the subnormal tail F16 cannot represent).
    Bf16,
    /// IEEE binary16, converted from the checkpoint's BF16 with
    /// round-to-nearest-even.
    F16,
    /// IEEE binary32.
    F32,
    /// ggml `block_q4_K`: 256-value superblocks, 144 B (two f16 super-scales,
    /// twelve bytes of packed 6-bit sub-scales/mins, 128 nibble bytes) —
    /// quantized AT LOAD from BF16 by the ggml-exact `quantize_q4_k_from_bf16`.
    /// 0.5625 B/elem; consumed by `GemvQ4KDense`, whose B operand is the
    /// PLAIN f32 activation (W4A16 — no activation quantization anywhere).
    /// Refuses widths that are not whole superblocks (the 640-wide
    /// shared-expert down stays Q8_0/BF16).
    Q4K,
    /// ggml `block_q8_0`: f16 scale + 32 int8 per 32 values, 34 B/block —
    /// quantized AT LOAD from the checkpoint's BF16 by the ggml-exact
    /// `quantize_q8_0_from_bf16`. Halves the dense bytes at a measured
    /// 6.6-7.4e-3 output vector-rel cost per tensor; consumed by GemvQ8_0,
    /// whose B operand is `block_q8_1_x4` activations (one QuantizeQ8_1
    /// dispatch per distinct width per token).
    Q8_0,
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
            Self::Bf16 | Self::F16 => Some((ncols * nrows * 2) as u64),
            Self::Q4K => {
                if ncols == 0 || !ncols.is_multiple_of(256) {
                    return None;
                }
                Some((ncols / 256 * 144 * nrows) as u64)
            }
            Self::Q8_0 => {
                if ncols == 0 || !ncols.is_multiple_of(32) {
                    return None;
                }
                Some((ncols / 32 * 34 * nrows) as u64)
            }
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

        // The three big hyper-connection tensors follow the dense tier when
        // it is VERBATIM BF16 (the qwen4_hc_*_bf16 kernel variants decode
        // it); any converting dense format falls back to F32 — there is no
        // F16 arm of those shaders, and silently feeding one F16 bytes is
        // exactly the class of bug the format enum exists to prevent.
        HyperConnection {
            part: HcPart::MixDown | HcPart::MixUp | HcPart::BlockInject,
            ..
        } => Some(match dense {
            // Verbatim BF16 rides the qwen4_hc_*_bf16 kernel variants; a
            // Q8_0 dense tier keeps the mix tensors at BF16 too (there is no
            // Q8 arm of those shaders, and 1.27 GB/token of BF16 beats
            // 2.5 GB of F32). Only the float-converting configs (the
            // harness's F32 subset) keep F32.
            Qwen4DeviceFormat::Bf16 | Qwen4DeviceFormat::Q8_0 | Qwen4DeviceFormat::Q4K => {
                Qwen4DeviceFormat::Bf16
            }
            _ => F32,
        }),

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

        // The 640-wide shared-expert down is not a whole number of Q4_K
        // superblocks; under a Q4_K dense tier it rides Q8_0 (whose GEMV
        // takes ncols % 8) instead of silently failing the plan.
        SharedExpertDownProj => Some(match dense {
            Qwen4DeviceFormat::Q4K => Qwen4DeviceFormat::Q8_0,
            other => other,
        }),

        // The dense GEMV weights — 6.70 GiB at F16, 13.4 GiB at F32.
        // (Tried and reverted, with the measurement: lm_head at Q8_0 under a
        // Q4_K tier bought NOTHING — teacher-forced dNLL 0.0857 vs pure
        // Q4_K's 0.0820, noise-level and slightly worse — so the logit-direct
        // family is NOT the dominant quality term and every width-qualified
        // family stays on the same format.)
        LinearAttnInProjQkv | LinearAttnInProjZ | LinearAttnOutProj | AttnQProj | AttnKProj
        | AttnVProj | AttnOProj | IndexerQkProj | SharedExpertGateProj | SharedExpertUpProj
        | PleKeyProj | PleValueProj | LmHead => Some(dense),

        // The two MTP fusion projections are ordinary [2560, 2560] GEMV
        // weights and ride the dense tier like every other plain-GEMV family.
        MtpFcEmbedding | MtpFcHidden => Some(dense),

        // The fusion norms are `Qwen4ExpTextRMSNorm` weights; the MTP forward
        // currently norms on the HOST (the fuse is 12800 elements), so they
        // upload RAW F32 — a future device consumer through the vendored
        // `rms_norm.comp` must move them into `folds_norm_bias` first.
        MtpPreFcNormEmbedding | MtpPreFcNormHidden => Some(F32),

        // Host-resident tables (see `Qwen4HostTables`).
        EmbedTokens
        | PleNgramShard
        | PleNgramWeightScale
        | PleNgramLayerMultipliers
        | PleNgramHeadsOffsets
        | PleNgramHeadsVocabSizes => None,

        // The stacked MTP experts never upload WHOLE — the plan slices them
        // per expert with an explicit format (see `mtp_expert_slice_name`) —
        // and the vision tower is not uploaded for a text-only decode.
        ExpertsStackedGateUp | ExpertsStackedDown | Vision(_) => None,
    }
}

/// Device format of one MTP routed-expert slice under the run's dense tier.
///
/// gate/up (2560-wide) follow the dense format; the 640-wide down is not a
/// whole number of Q4_K superblocks and rides Q8_0 under a Q4_K tier — the
/// same rule as [`Qwen4TensorKind::SharedExpertDownProj`], and for the same
/// reason (`GemvQ8_0Dense` takes `ncols % 8`).
#[must_use]
pub fn mtp_expert_slice_format(proj: ExpertProj, dense: Qwen4DeviceFormat) -> Qwen4DeviceFormat {
    match (proj, dense) {
        (ExpertProj::Down, Qwen4DeviceFormat::Q4K) => Qwen4DeviceFormat::Q8_0,
        _ => dense,
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
    /// Move the coldest read-only weights to the host heap when the device
    /// budget cannot hold them, rather than refusing the load. See
    /// [`Qwen4Tier`] for the measured 2.05% this costs, and
    /// [`Qwen4Plan::spill_to_fit`] for what moves first.
    ///
    /// On by default because the full residency is 2.95 GiB over the driver's
    /// budget; set it false to make an over-budget plan a hard error instead.
    pub spill_to_host: bool,
    /// Override the device-local limit the plan is sized against. `None` — the
    /// default — asks the driver (`crate::loader::device_local_budget`).
    ///
    /// Only for tests: a small artificial budget is what lets the spill path be
    /// exercised on a subset scope in seconds instead of on the 71 GiB full
    /// residency. Nothing in production should set it, and a value ABOVE what
    /// the driver grants is not honoured — the smaller of the two wins, so this
    /// can only ever tighten the guard.
    pub device_budget_bytes: Option<u64>,
}

impl Default for Qwen4UploadConfig {
    fn default() -> Self {
        Self {
            dense_format: Qwen4DeviceFormat::Bf16,
            slab_bytes: None,
            reserve_bytes: DEFAULT_RESERVE_BYTES,
            spill_to_host: true,
            device_budget_bytes: None,
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
    /// Upload the `mtp.*` head — the speculative-decode drafter: its dense
    /// tensors plus the stacked experts sliced per expert
    /// ([`mtp_expert_slice_name`]). Independent of `layers`, which scopes
    /// TEXT layers only.
    pub mtp: bool,
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
    /// Everything the checkpoint has. `mtp` rides along: the MTP head is the
    /// speculative-decode lever and an explicit product keep (see
    /// `qwen4_names`' residency docs) — a run that wants it out sets the
    /// field, it does not get dropped silently.
    #[must_use]
    pub fn full() -> Self {
        Self {
            layers: None,
            experts: None,
            lm_head: true,
            mtp: true,
        }
    }

    /// Named layers only, with every expert and `lm_head`; no MTP (a subset
    /// bring-up opts in explicitly).
    #[must_use]
    pub fn layers(layers: &[usize]) -> Self {
        Self {
            layers: Some(layers.to_vec()),
            experts: None,
            lm_head: true,
            mtp: false,
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
    /// A contiguous row range of one BF16 checkpoint tensor, converted like
    /// [`Qwen4Source::Bf16`] (never bias-folded — the only sliced tensors are
    /// the stacked MTP experts). `tensor` is the real checkpoint name;
    /// `row_offset` counts rows of the flattened `[rows, ncols]` view, so
    /// expert `e`'s gate slice of `experts.gate_up_proj` `[512, 1280, 2560]`
    /// starts at row `e * 1280` and its up slice at `e * 1280 + 640`.
    Bf16Slice { tensor: String, row_offset: usize },
    /// `n_experts` NVFP4 `(weight, weight_scale)` plane pairs, repacked into
    /// ggml `block_nvfp4` and stacked expert-major into one suballocation.
    Nvfp4Stack {
        layer: usize,
        proj: ExpertProj,
        n_experts: usize,
    },
}

/// Which heap one suballocation's slab lives on.
///
/// # Why a spill tier exists at all
///
/// The full text residency is 72.64 GiB of suballocations that pack into 72.656
/// GiB of slabs, against a DRIVER BUDGET of 70.71 GiB (heap 1 reports 74.43 GiB
/// of *size*; see `crate::loader::DeviceBudget`). With
/// [`DEFAULT_RESERVE_BYTES`] held back that is **2.95 GiB over**, and
/// over-committing this UMA part is not `OUT_OF_DEVICE_MEMORY` — it is silent
/// page demotion, i.e. a load that appears to work and then runs several times
/// slow with nothing in any log.
///
/// The alternative to relocating those bytes is shrinking them (requantizing
/// the BF16 dense tier, or an F16 build of the three `float[]` shaders). Both
/// are real work in someone else's file. Relocating is measured to be nearly
/// free: on this part the host heap is the *same* LPDDR5X, and a 512 MiB GPU
/// streaming read costs
///
/// ```text
///   heap 1, alloc_uma          204.4 GB/s
///   heap 0, alloc_host_cached  200.2 GB/s   -2.05%
/// ```
///
/// (same sitting, `vulkan-kernels::device_cache_hierarchy::
/// report_gpu_read_bandwidth_by_memory_flavor`; ratio, not absolute — this box
/// throttles). A spilled weight stays an ordinary storage-buffer binding, so
/// every GEMV consumes it unchanged.
///
/// **A measured trap, since the number above invites the wrong reading.** The
/// `alloc (host WC, heap 0)` row of that same sweep is NOT on heap 0.
/// `memory_type_index` prefers the largest heap among the compatible types, and
/// heap 1 exposes a `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` type, so plain
/// `DeviceBuffer::alloc` lands on heap 1. Confirmed by watching `heapUsage` move
/// under a 512 MiB allocation: `alloc` charges heap 1, `alloc_host_cached`
/// charges heap 0, `alloc_uma` charges heap 1. `alloc_host_cached` is therefore
/// the only public flavour that reaches heap 0 on this part, which is what the
/// spill slabs use — and HOST_CACHED also makes the loader's own verification
/// read-backs fast instead of write-combined-slow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Qwen4Tier {
    /// DEVICE_LOCAL slabs on the big heap. Everything lives here while it fits.
    #[default]
    Device,
    /// `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` slabs on the host heap, for
    /// read-only weights the device budget cannot hold.
    HostSpill,
}

impl Qwen4Tier {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Device => "device-local heap",
            Self::HostSpill => "host heap (spill)",
        }
    }
}

/// One suballocation the upload will make.
#[derive(Debug, Clone)]
pub struct Qwen4PlanItem {
    /// The checkpoint tensor name, or [`expert_stack_name`] for a stack.
    pub name: String,
    pub role: Qwen4TensorRole,
    pub format: Qwen4DeviceFormat,
    pub bytes: u64,
    /// Which heap this entry's slab comes from. Every item plans as
    /// [`Qwen4Tier::Device`]; [`Qwen4Plan::spill_to_fit`] is the only thing that
    /// moves one.
    pub tier: Qwen4Tier,
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
    /// Sum of the [`Qwen4Tier::Device`] items' bytes, before slab alignment
    /// padding. Unchanged from "every item" until something spills.
    pub device_bytes: u64,
    /// Sum of the [`Qwen4Tier::HostSpill`] items' bytes. Zero unless
    /// [`Qwen4Plan::spill_to_fit`] moved something.
    pub spill_bytes: u64,
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
    ///
    /// `budget` carries which limit it is — the driver's `heapBudget - heapUsage`
    /// or the heap size it falls back to — because on this part those differ by
    /// 3.72 GiB and the plan sits between them.
    pub fn ensure_fits(&self, budget: &DeviceBudget, headroom: u64) -> Result<()> {
        let usable = budget.bytes.saturating_sub(headroom);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        ensure!(
            self.device_bytes <= usable,
            "qwen4 residency plan needs {:.2} GiB on the device but heap {} grants {:.2} GiB \
             — {} (heap size {:.2} GiB); {:.2} GiB reserved for KV + state + activations \
             leaves {:.2} GiB usable, so the plan is over by {:.2} GiB. Spill the coldest \
             weights to the host heap (`Qwen4Plan::spill_to_fit`) or narrow the scope.",
            gib(self.device_bytes),
            budget.heap_index,
            gib(budget.bytes),
            budget.source.label(),
            gib(budget.heap_size),
            gib(headroom),
            gib(usable),
            gib(self.device_bytes.saturating_sub(usable)),
        );
        Ok(())
    }

    /// Move the coldest read-only suballocations to [`Qwen4Tier::HostSpill`]
    /// until the device tier fits `budget.bytes - headroom`.
    ///
    /// Idempotent and monotone: it only ever moves Device → HostSpill, and it
    /// stops the moment the device tier fits, so calling it on a plan that
    /// already fits is a no-op that reports zero moves.
    ///
    /// # Spill order, and why it is not "biggest first"
    ///
    /// The router picks 10 of 512 experts per token, so ~98% of an NVFP4 stack's
    /// bytes are not read at all on a given step, while every other item in the
    /// plan — attention, linear-attn, the hyper-connection weights, `lm_head` —
    /// is read in FULL every token. Moving a stack therefore pays the 2.05%
    /// host-heap read penalty on ~2% of its bytes; moving the same volume of
    /// dense weight pays it on all of them. Stacks first, dense last. Within a
    /// class, largest first: fewest suballocations moved per gibibyte recovered,
    /// and it keeps the spill slabs few and full.
    ///
    /// `host_available` is the host heap's own budget (see
    /// `crate::loader::device_local_budget_from` for the device side; heap 0's
    /// figure comes from the same `memory_budgets()` array). A spill that would
    /// not fit THERE is refused rather than moved.
    pub fn spill_to_fit(
        &mut self,
        budget: &DeviceBudget,
        headroom: u64,
        host_available: u64,
    ) -> Result<Qwen4Spill> {
        let usable = budget.bytes.saturating_sub(headroom);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        if self.device_bytes <= usable {
            return Ok(Qwen4Spill {
                items: 0,
                bytes: 0,
                device_bytes: self.device_bytes,
            });
        }

        // Every item is spillable — they are all read-only storage buffers — so
        // the only residency that cannot be rescued is one with no device memory
        // left AT ALL: the KV cache, the recurrent state and the arena are
        // device-side and come out of the reserve, so a reserve at or past the
        // budget is unrunnable however the weights are arranged.
        ensure!(
            usable > 0,
            "qwen4 spill: heap {} grants {:.2} GiB and {:.2} GiB is reserved for KV + state \
             + activations, leaving nothing on the device — no arrangement of the weights \
             makes this residency runnable",
            budget.heap_index,
            gib(budget.bytes),
            gib(headroom),
        );

        let mut order: Vec<usize> = (0..self.items.len())
            .filter(|&i| self.items[i].tier == Qwen4Tier::Device)
            .collect();
        order.sort_by_key(|&i| Self::spill_rank(&self.items[i]));

        // Choose the moves first and apply them only once they are all legal.
        // A refusal that left a half-spilled plan behind would be worse than no
        // spill at all: the caller still holds the `&mut`, and the plan would no
        // longer say what it says on the tin.
        let mut to_move = Vec::new();
        let mut device_after = self.device_bytes;
        let mut moved_bytes = 0u64;
        for i in order {
            if device_after <= usable {
                break;
            }
            device_after -= self.items[i].bytes;
            moved_bytes += self.items[i].bytes;
            to_move.push(i);
        }

        // Postcondition, not a user-facing case: the loop only stops when the
        // device tier fits or the spillable set is exhausted, and every item is
        // spillable, so a `spill_rank` or accounting bug is what would trip it.
        ensure!(
            device_after <= usable,
            "qwen4 spill: even with all {} spillable suballocations ({:.2} GiB) on the \
             host heap the device tier is {:.2} GiB against {:.2} GiB usable",
            to_move.len(),
            gib(moved_bytes),
            gib(device_after),
            gib(usable),
        );
        ensure!(
            self.spill_bytes + moved_bytes <= host_available,
            "qwen4 spill: {:.2} GiB of weights would go to the host heap, which grants only \
             {:.2} GiB",
            gib(self.spill_bytes + moved_bytes),
            gib(host_available),
        );

        for &i in &to_move {
            self.items[i].tier = Qwen4Tier::HostSpill;
        }
        self.device_bytes = device_after;
        self.spill_bytes += moved_bytes;
        Ok(Qwen4Spill {
            items: to_move.len(),
            bytes: moved_bytes,
            device_bytes: self.device_bytes,
        })
    }

    /// Sort key for [`Self::spill_to_fit`]: coldest bytes first, then largest.
    fn spill_rank(item: &Qwen4PlanItem) -> (u8, std::cmp::Reverse<u64>) {
        let class = match (&item.source, item.role.stream) {
            // Coldest of all: an MTP expert slice is read only when a token
            // is being SPECULATED, and then 10 of 512 — heap-0 residency was
            // priced at ~2% of GPU read bandwidth, so these bytes go first.
            (Qwen4Source::Bf16Slice { .. }, _) => 0,
            // The rest of the MTP head: read once per drafted token, never
            // during a plain decode.
            (_, Qwen4Stream::Mtp) => 1,
            // Sparsely read: 10 of `n_experts` slices per token.
            (Qwen4Source::Nvfp4Stack { .. }, _) => 2,
            // Read in full every token.
            (Qwen4Source::Bf16 { .. }, _) => 3,
        };
        (class, std::cmp::Reverse(item.bytes))
    }

    /// Items in `tier`, in plan order.
    pub fn tier_items(&self, tier: Qwen4Tier) -> impl Iterator<Item = &Qwen4PlanItem> {
        self.items.iter().filter(move |i| i.tier == tier)
    }

    /// Bytes in `tier`, recomputed from the items rather than read off the
    /// running totals — the two agreeing is a thing worth being able to assert.
    #[must_use]
    pub fn tier_bytes(&self, tier: Qwen4Tier) -> u64 {
        self.tier_items(tier).map(|i| i.bytes).sum()
    }

    /// Largest suballocation in the plan, i.e. the host scratch buffer the
    /// upload needs (1.18 GiB when `lm_head` is in scope). Across BOTH tiers:
    /// the scratch is shared.
    #[must_use]
    pub fn max_item_bytes(&self) -> u64 {
        self.items.iter().map(|i| i.bytes).max().unwrap_or(0)
    }

    /// Largest suballocation in `tier` — the floor on that tier's slab size,
    /// since `SlabPlan` refuses a request no slab could ever hold.
    #[must_use]
    pub fn max_item_bytes_in(&self, tier: Qwen4Tier) -> u64 {
        self.tier_items(tier).map(|i| i.bytes).max().unwrap_or(0)
    }

    /// Dry-run one tier into slabs of `slab_bytes` and report what that heap
    /// would actually give up.
    ///
    /// Runs through `vulkan_sys::SlabPlan`, the same type `SlabAllocator` drives
    /// its real allocations through, so the estimate and the residency cannot
    /// drift. Largest-first, matching [`upload_qwen4`].
    #[cfg(feature = "vulkan")]
    pub fn pack(&self, tier: Qwen4Tier, slab_bytes: u64, alignment: u64) -> Result<Qwen4Packing> {
        let mut sp = vulkan_sys::SlabPlan::new(slab_bytes, alignment)
            .map_err(|e| anyhow!("qwen4 packing at {slab_bytes} B slabs: {e}"))?;
        let mut items: Vec<&Qwen4PlanItem> = self.tier_items(tier).collect();
        items.sort_by_key(|i| std::cmp::Reverse(i.bytes));
        for item in items {
            sp.place(item.bytes).map_err(|e| {
                anyhow!(
                    "qwen4 packing: {} ({} B) into {slab_bytes} B slabs: {e}",
                    item.name,
                    item.bytes
                )
            })?;
        }
        Ok(Qwen4Packing {
            tier,
            slab_bytes,
            committed_bytes: sp.committed_bytes(),
            used_bytes: sp.used_bytes(),
            slab_count: sp.slab_count(),
        })
    }

    /// The slab size that commits the fewest bytes for THIS tier of THIS plan.
    ///
    /// Not a tuning nicety — 1.34 GiB of a 74.43 GiB heap, measured on the full
    /// model (see [`Qwen4Packing`]). And not a constant either: the optimum is a
    /// property of the item multiset, so it moves with
    /// [`Qwen4UploadConfig::dense_format`], with the scope, and with what has
    /// spilled. Sweeping at load costs well under a second and re-derives it
    /// from whatever is actually being uploaded.
    #[cfg(feature = "vulkan")]
    pub fn choose_packing(
        &self,
        tier: Qwen4Tier,
        max_slab_bytes: u64,
        alignment: u64,
    ) -> Result<Qwen4Packing> {
        let floor = self.max_item_bytes_in(tier).max(vulkan_sys::MIN_SLAB_BYTES);
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
            let Ok(packing) = self.pack(tier, sz, alignment) else {
                continue;
            };
            if best.is_none_or(|b| packing.committed_bytes < b.committed_bytes) {
                best = Some(packing);
            }
        }
        best.ok_or_else(|| anyhow!("qwen4 packing: no slab size in range fits this plan"))
    }
}

/// What [`Qwen4Plan::spill_to_fit`] moved off the device heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Qwen4Spill {
    pub items: usize,
    pub bytes: u64,
    /// The device tier after the move.
    pub device_bytes: u64,
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
    /// Which heap these slabs come from.
    pub tier: Qwen4Tier,
    pub slab_bytes: u64,
    /// Bytes `vkAllocateMemory` will be asked for, summed over slabs.
    pub committed_bytes: u64,
    /// Bytes handed to suballocations, excluding padding and slab tails.
    pub used_bytes: u64,
    pub slab_count: usize,
}

#[cfg(feature = "vulkan")]
impl Qwen4Packing {
    /// Refuse a packing that leaves less than `headroom` of the heap.
    ///
    /// The check [`Qwen4Plan::ensure_fits`] cannot make: that one sees the
    /// plan's bytes, this one sees the bytes the slabs will really cost.
    /// `available` is the heap's grantable bytes — the driver's budget on the
    /// device tier (`crate::loader::DeviceBudget::bytes`), heap 0's on the
    /// spill tier — never a raw `VkMemoryHeap::size`.
    pub fn ensure_fits(&self, available: u64, headroom: u64) -> Result<()> {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        ensure!(
            self.committed_bytes.saturating_add(headroom) <= available,
            "qwen4 residency commits {:.2} GiB over {} slabs of {:.0} MiB on the {}, which \
             leaves {:.2} GiB of the {:.2} GiB granted — less than the {:.2} GiB the KV \
             cache, recurrent state and arena need; over by {:.2} GiB",
            gib(self.committed_bytes),
            self.slab_count,
            self.slab_bytes as f64 / (1u64 << 20) as f64,
            self.tier.label(),
            gib(available.saturating_sub(self.committed_bytes)),
            gib(available),
            gib(headroom),
            gib(self
                .committed_bytes
                .saturating_add(headroom)
                .saturating_sub(available)),
        );
        Ok(())
    }

    /// `committed - used`, as a fraction of what was committed.
    #[must_use]
    pub fn waste_fraction(&self) -> f64 {
        if self.committed_bytes == 0 {
            return 0.0;
        }
        self.committed_bytes.saturating_sub(self.used_bytes) as f64 / self.committed_bytes as f64
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

        match role.stream {
            Qwen4Stream::Text => {}
            // The MTP head plans through its own arm: its stacked experts
            // become per-expert slices and nothing of it may touch the TEXT
            // layer bookkeeping (its `layers.0` would otherwise masquerade as
            // text layer 0).
            Qwen4Stream::Mtp if scope.mtp => {
                plan_mtp_tensor(&mut plan, cfg, info, role)?;
                continue;
            }
            _ => {
                plan.dropped_bytes += info.len;
                continue;
            }
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
                tier: Qwen4Tier::Device,
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
                tier: Qwen4Tier::Device,
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
    match (item.role.stream, item.role.layer) {
        // `layer_bytes` is a TEXT-layer ledger (it feeds the layer-at-a-time
        // bring-up); the MTP head's `layers.0` is a different tree and books
        // as stream-global instead of shadowing text layer 0.
        (Qwen4Stream::Text, Some(layer)) => {
            *plan.layer_bytes.entry(layer).or_insert(0) += item.bytes;
        }
        _ => plan.global_bytes += item.bytes,
    }
    plan.items.push(item);
}

/// Plan one `mtp.*` tensor: the stacked experts slice per expert into dense
/// GEMV suballocations, everything else follows [`device_format`] like the
/// text stream. Fails loud on an MTP family with no device destination — a
/// silently absent draft weight would surface as a wrong draft, which greedy
/// verification then quietly eats as a 0% acceptance rate.
fn plan_mtp_tensor(
    plan: &mut Qwen4Plan,
    cfg: &Qwen4UploadConfig,
    info: &SafeTensorInfo,
    role: Qwen4TensorRole,
) -> Result<()> {
    ensure!(
        info.dtype == "BF16",
        "{}: the MTP head is quant-excluded, expected BF16, found {}",
        info.name,
        info.dtype
    );
    let stacked = match role.kind {
        Qwen4TensorKind::ExpertsStackedGateUp => Some((ExpertProj::Gate, true)),
        Qwen4TensorKind::ExpertsStackedDown => Some((ExpertProj::Down, false)),
        _ => None,
    };
    let Some((_, is_gate_up)) = stacked else {
        let format = device_format(role.kind, cfg.dense_format).ok_or_else(|| {
            anyhow!(
                "{}: MTP family {:?} has no device format — wire it before speculating",
                info.name,
                role.kind
            )
        })?;
        let (ncols, nrows) = dense_shape(info)?;
        let bytes = format.bytes_for(ncols, nrows).ok_or_else(|| {
            anyhow!(
                "{}: [{nrows}, {ncols}] has no {format:?} device layout",
                info.name
            )
        })?;
        push_item(
            plan,
            Qwen4PlanItem {
                name: info.name.clone(),
                role,
                format,
                bytes,
                tier: Qwen4Tier::Device,
                ncols,
                nrows,
                source: Qwen4Source::Bf16 {
                    fold_bias: folds_norm_bias(role.kind),
                },
            },
        );
        return Ok(());
    };

    // Stacked experts: `dims` is GGUF ne order (innermost first), so
    // gate_up `[512, 1280, 2560]` reads back as `[2560, 1280, 512]`.
    ensure!(
        info.dims.len() == 3,
        "{}: expected a 3-D expert stack, dims {:?}",
        info.name,
        info.dims
    );
    let ncols = usize::try_from(info.dims[0])?;
    let rows_per_expert = usize::try_from(info.dims[1])?;
    let n_experts = usize::try_from(info.dims[2])?;
    ensure!(
        n_experts > 0 && rows_per_expert > 0,
        "{}: degenerate expert stack",
        info.name
    );
    // gate_up is the FUSED `[gate; up]` layout: gate rows first, up rows
    // second (`chunk(2, dim=-1)` of the linear output in
    // modeling_qwen4_exp.py's Qwen4ExpTextExperts.forward).
    let slices: &[(ExpertProj, usize, usize)] = if is_gate_up {
        ensure!(
            rows_per_expert.is_multiple_of(2),
            "{}: fused gate_up rows {rows_per_expert} are odd",
            info.name
        );
        &[
            (ExpertProj::Gate, 0, rows_per_expert / 2),
            (ExpertProj::Up, rows_per_expert / 2, rows_per_expert / 2),
        ]
    } else {
        &[(ExpertProj::Down, 0, rows_per_expert)]
    };
    for expert in 0..n_experts {
        for &(proj, at, nrows) in slices {
            let format = mtp_expert_slice_format(proj, cfg.dense_format);
            let bytes = format.bytes_for(ncols, nrows).ok_or_else(|| {
                anyhow!(
                    "{}: expert slice [{nrows}, {ncols}] has no {format:?} layout",
                    info.name
                )
            })?;
            push_item(
                plan,
                Qwen4PlanItem {
                    name: mtp_expert_slice_name(u32::try_from(expert)?, proj),
                    role: Qwen4TensorRole {
                        sub_index: Some(u32::try_from(expert)?),
                        ..role
                    },
                    format,
                    bytes,
                    tier: Qwen4Tier::Device,
                    ncols,
                    nrows,
                    source: Qwen4Source::Bf16Slice {
                        tensor: info.name.clone(),
                        row_offset: expert * rows_per_expert + at,
                    },
                },
            );
        }
    }
    Ok(())
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

/// Slice a BF16 -> quantized conversion across threads by row ranges. Both
/// quantizers are row-self-contained (their docs guarantee it), so each
/// thread owns a disjoint `[row_lo, row_hi)` of src and dst. Single-threaded
/// the whole 13.3 GiB dense tier costs ~30 s of load at the measured
/// ~0.45 GB/s; sixteen ways it is under 3.
#[cfg(feature = "vulkan")]
#[expect(clippy::too_many_arguments, reason = "a conversion plan is this wide")]
fn quantize_rows_threaded(
    name: &str,
    src: &[u8],
    nrows: usize,
    ncols: usize,
    dst: &mut [u8],
    block_vals: usize,
    block_bytes: usize,
    quantize: fn(&[u8], usize, usize, &mut [u8]) -> vulkan_kernels::Result<()>,
) -> Result<()> {
    let src_row = ncols * 2;
    let dst_row = ncols / block_vals * block_bytes;
    ensure!(
        src.len() == nrows * src_row && dst.len() == nrows * dst_row,
        "{name}: quantize buffer geometry"
    );
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(16)
        .min(nrows.max(1));
    let chunk = nrows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (i, dst_chunk) in dst.chunks_mut(chunk * dst_row).enumerate() {
            let rows = dst_chunk.len() / dst_row;
            let src_chunk = &src[i * chunk * src_row..][..rows * src_row];
            handles.push(scope.spawn(move || quantize(src_chunk, rows, ncols, dst_chunk)));
        }
        for h in handles {
            h.join()
                .map_err(|_| anyhow!("{name}: quantize worker panicked"))??;
        }
        Ok(())
    })
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
    /// Which allocator owns `alloc`. A [`vulkan_sys::SlabAlloc`] is only a
    /// `(slab index, offset, len)` triple, so the same handle names a different
    /// buffer in each tier; resolving one against the wrong allocator would
    /// bind whatever happens to sit at that offset and still run.
    pub tier: Qwen4Tier,
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
    /// binding is indexed by expert **id** (like BIAS0). The forward seeds
    /// this array verbatim into its resident scale table once per layer and
    /// binds rows of that table — the router's ids never visit the host.
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

    /// Gather `weight_scale_2` for a routed-expert list into SLOT order.
    ///
    /// Historical seam: `MAT_VEC_FUSION_SCALE0` used to read binding 3 at the
    /// expert SLOT, which forced the decode to fence every layer just to see
    /// the router's ids and build this list. The shader now indexes SCALE0 by
    /// expert id (matching BIAS0) and the forward binds the resident table,
    /// so the hot paths no longer call this; it stays as the host-side
    /// oracle of the slot→id mapping and for its unit test.
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

/// Host-heap slabs for the [`Qwen4Tier::HostSpill`] weights.
///
/// A parallel to `vulkan_sys::SlabAllocator` because that type has no host-heap
/// flavour: its `SlabMemory::Uma` asks for `DEVICE_LOCAL | HOST_VISIBLE`, and on
/// this part that is heap 1 — the heap the spill exists to get off. Measured by
/// watching `heapUsage` move under a 512 MiB allocation: `DeviceBuffer::alloc`
/// and `alloc_uma` both charge heap 1; only `alloc_host_cached` charges heap 0.
/// So the spill tier drives a bare `SlabPlan` over buffers of its own.
///
/// Simpler than `SlabAllocator` in the way that matters here: these slabs are
/// host-mappable, so a write is a `copy_from_host_at` straight into the mapping
/// — no staging window, no command submit, no per-chunk copy.
#[cfg(feature = "vulkan")]
struct Qwen4SpillSlabs<'ctx> {
    ctx: &'ctx vulkan_sys::VulkanContext,
    slabs: Vec<vulkan_sys::DeviceBuffer<'ctx>>,
    plan: vulkan_sys::SlabPlan,
    /// Index of the device-local heap, so [`Self::grow_to_plan`] can catch a
    /// slab that landed on it.
    device_heap: usize,
}

#[cfg(feature = "vulkan")]
impl<'ctx> Qwen4SpillSlabs<'ctx> {
    fn new(
        ctx: &'ctx vulkan_sys::VulkanContext,
        slab_bytes: u64,
        alignment: u64,
        device_heap: usize,
    ) -> Result<Self> {
        Ok(Self {
            ctx,
            slabs: Vec::new(),
            plan: vulkan_sys::SlabPlan::new(slab_bytes, alignment)
                .map_err(|e| anyhow!("qwen4 spill slabs at {slab_bytes} B: {e}"))?,
            device_heap,
        })
    }

    fn alloc(&mut self, len: u64) -> Result<vulkan_sys::SlabAlloc> {
        let alloc = self
            .plan
            .place(len)
            .map_err(|e| anyhow!("qwen4 spill: placing {len} B: {e}"))?;
        self.grow_to_plan()?;
        Ok(alloc)
    }

    /// Back every slab `SlabPlan::place` opened with a real host-heap buffer.
    ///
    /// `place` only ever opens NOMINAL-size slabs, so the plan's capacities and
    /// these buffers' lengths cannot drift.
    fn grow_to_plan(&mut self) -> Result<()> {
        while self.slabs.len() < self.plan.slab_count() {
            let bytes = usize::try_from(self.plan.slab_size())
                .map_err(|_| anyhow!("qwen4 spill: slab size does not fit in usize"))?;
            // `alloc_host_cached` FALLS BACK to plain HOST_VISIBLE|HOST_COHERENT
            // when the device exposes no HOST_CACHED type, and
            // `memory_type_index` then prefers the largest compatible heap —
            // which here is the device-local one. That fallback would spill onto
            // the heap we are fleeing, silently, and the load would die a few
            // GiB later with an opaque allocation error. Trust the accounting,
            // not the flavour: watch whether the device heap absorbed it.
            let before = self.device_heap_usage();
            let buffer = vulkan_sys::DeviceBuffer::alloc_host_cached(self.ctx, bytes)
                .map_err(|e| anyhow!("qwen4 spill: allocating a {bytes} B host slab: {e}"))?;
            if let (Some(before), Some(after)) = (before, self.device_heap_usage()) {
                ensure!(
                    after.saturating_sub(before) < self.plan.slab_size() / 2,
                    "qwen4 spill: a {bytes} B host slab charged {} B to device-local heap {} \
                     — this device has no HOST_CACHED memory type and `alloc_host_cached` fell \
                     back onto the heap the spill is meant to relieve",
                    after - before,
                    self.device_heap,
                );
            }
            self.slabs.push(buffer);
        }
        Ok(())
    }

    fn device_heap_usage(&self) -> Option<u64> {
        self.ctx
            .memory_budgets()
            .and_then(|b| b.get(self.device_heap).map(|&(_, usage)| usage))
    }

    fn write(&mut self, alloc: &vulkan_sys::SlabAlloc, src: &[u8]) -> Result<()> {
        let slab = self
            .slabs
            .get_mut(alloc.slab())
            .ok_or_else(|| anyhow!("qwen4 spill: slab index {} out of range", alloc.slab()))?;
        slab.copy_from_host_at(alloc.offset(), src)
            .map_err(|e| anyhow!("qwen4 spill: writing {} B: {e}", src.len()))
    }

    fn binding(
        &self,
        alloc: &vulkan_sys::SlabAlloc,
    ) -> Result<(&vulkan_sys::DeviceBuffer<'ctx>, u64, u64)> {
        let slab = self
            .slabs
            .get(alloc.slab())
            .ok_or_else(|| anyhow!("qwen4 spill: slab index {} out of range", alloc.slab()))?;
        Ok((slab, alloc.offset(), alloc.len()))
    }

    /// Read a suballocation back. HOST_CACHED, so this is a cached memcpy — not
    /// the write-combined 0.10 GB/s trap the device tier's staged read-back
    /// exists to avoid.
    fn read_back(&self, alloc: &vulkan_sys::SlabAlloc, dst: &mut [u8]) -> Result<()> {
        let slab = self
            .slabs
            .get(alloc.slab())
            .ok_or_else(|| anyhow!("qwen4 spill: slab index {} out of range", alloc.slab()))?;
        slab.copy_to_host_at(alloc.offset(), dst)
            .map_err(|e| anyhow!("qwen4 spill: reading {} B back: {e}", dst.len()))
    }

    fn committed_bytes(&self) -> u64 {
        self.plan.committed_bytes()
    }

    fn slab_count(&self) -> usize {
        self.slabs.len()
    }
}

/// Everything `qwen4_exp` needs to run, resident.
#[cfg(feature = "vulkan")]
pub struct Qwen4Weights<'ctx, 'st> {
    slabs: vulkan_sys::SlabAllocator<'ctx>,
    /// `None` unless something spilled; see [`Qwen4Tier`].
    spill: Option<Qwen4SpillSlabs<'ctx>>,
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
    ///
    /// A spilled weight binds exactly like a device-resident one; the tier is a
    /// property of the memory the slab came from, not of the descriptor.
    pub fn binding(
        &self,
        tensor: &Qwen4DeviceTensor,
    ) -> Result<(&vulkan_sys::DeviceBuffer<'ctx>, u64, u64)> {
        match tensor.tier {
            Qwen4Tier::Device => self
                .slabs
                .binding(&tensor.alloc)
                .map_err(|e| anyhow!("qwen4 weights: binding a suballocation: {e}")),
            Qwen4Tier::HostSpill => self
                .spill
                .as_ref()
                .ok_or_else(|| anyhow!("qwen4 weights: a spilled tensor with no spill tier"))?
                .binding(&tensor.alloc),
        }
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
    /// (The spill tier is host-mapped, so its read-back is a plain cached
    /// memcpy — but the contract is the same: verification, not decode.)
    pub fn read_back(&self, tensor: &Qwen4DeviceTensor, dst: &mut [u8]) -> Result<()> {
        match tensor.tier {
            Qwen4Tier::Device => self
                .slabs
                .read_back(&tensor.alloc, dst)
                .map_err(|e| anyhow!("qwen4 weights: read-back: {e}")),
            Qwen4Tier::HostSpill => self
                .spill
                .as_ref()
                .ok_or_else(|| anyhow!("qwen4 weights: a spilled tensor with no spill tier"))?
                .read_back(&tensor.alloc, dst),
        }
    }

    /// Committed bytes per tier, for a caller that wants to report residency.
    #[must_use]
    pub fn committed_bytes(&self, tier: Qwen4Tier) -> u64 {
        match tier {
            Qwen4Tier::Device => self.slabs.committed_bytes(),
            Qwen4Tier::HostSpill => self
                .spill
                .as_ref()
                .map_or(0, Qwen4SpillSlabs::committed_bytes),
        }
    }

    /// Slab count per tier.
    #[must_use]
    pub fn slab_count(&self, tier: Qwen4Tier) -> usize {
        match tier {
            Qwen4Tier::Device => self.slabs.slab_count(),
            Qwen4Tier::HostSpill => self.spill.as_ref().map_or(0, Qwen4SpillSlabs::slab_count),
        }
    }
}

/// The device-local budget this upload will plan against.
///
/// `cfg.device_budget_bytes` can only ever TIGHTEN what the driver grants — a
/// test that wants to exercise the spill path on a subset asks for a small
/// budget, and a caller that fat-fingers a large one still gets the driver's.
#[cfg(feature = "vulkan")]
fn upload_budget(ctx: &vulkan_sys::VulkanContext, cfg: &Qwen4UploadConfig) -> Result<DeviceBudget> {
    let mut budget = crate::loader::device_local_budget(ctx)?;
    if let Some(cap) = cfg.device_budget_bytes {
        budget.bytes = budget.bytes.min(cap);
    }
    Ok(budget)
}

/// Grantable bytes on the largest NON-device-local heap — where the spill tier
/// lands. `0` when the device reports no host heap, which makes a spill refuse
/// rather than allocate somewhere unintended.
#[cfg(feature = "vulkan")]
fn host_heap_available(ctx: &vulkan_sys::VulkanContext) -> u64 {
    let heaps = ctx.memory_heaps();
    let budgets = ctx.memory_budgets();
    heaps
        .iter()
        .enumerate()
        .filter(|&(_, &(_, device_local))| !device_local)
        .map(
            |(i, &(size, _))| match budgets.as_ref().and_then(|b| b.get(i)) {
                Some(&(budget, usage)) => budget.saturating_sub(usage).min(size),
                None => size,
            },
        )
        .max()
        .unwrap_or(0)
}

/// Execute `plan` against `ctx`, borrowing every source byte from `st`'s mmaps.
///
/// Order of operations matters and is not incidental:
/// 1. Size against the DRIVER'S budget (`crate::loader::device_local_budget`),
///    then spill what will not fit, then [`Qwen4Plan::ensure_fits`] — all BEFORE
///    a byte is staged.
/// 2. Suballocate LARGEST-FIRST, per tier. `vulkan_sys::SlabPlan` documents the
///    measured difference on this exact checkpoint: 62 slabs and 1.01% waste
///    largest-first against 64 slabs and 4.10% in arrival order.
/// 3. Write in plan order, which is checkpoint order, so a shard's mmap pages
///    are touched once and in sequence rather than revisited per slab.
///
/// The plan arrives by reference and is cloned only if something has to spill:
/// a residency that fits pays nothing for the tier existing.
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
    // Size against the DEVICE_LOCAL heap's BUDGET, not system RAM and not the
    // heap size: on this APU a 60 GiB device-local allocation moved OS-visible
    // free RAM by only 4.2 GiB (so the host figure says nothing about what the
    // device can hold), and the driver's budget is 3.72 GiB under the heap size
    // (so the size says nothing about what it will GRANT).
    let budget = upload_budget(ctx, cfg)?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;

    // Spill before the guard, not after it: `spill_to_fit` is a no-op on a plan
    // that already fits, so a subset residency never touches the host heap.
    let spilled;
    let plan = if cfg.spill_to_host && plan.ensure_fits(&budget, cfg.reserve_bytes).is_err() {
        let mut owned = plan.clone();
        let moved = owned.spill_to_fit(&budget, cfg.reserve_bytes, host_heap_available(ctx))?;
        log::warn!(
            "qwen4 residency: {:.2} GiB over heap {}'s {:.2} GiB budget ({}); spilled {} \
             suballocation(s) / {:.2} GiB to the host heap, device tier now {:.2} GiB",
            gib(plan.device_bytes),
            budget.heap_index,
            gib(budget.bytes),
            budget.source.label(),
            moved.items,
            gib(moved.bytes),
            gib(moved.device_bytes),
        );
        spilled = owned;
        &spilled
    } else {
        plan
    };
    plan.ensure_fits(&budget, cfg.reserve_bytes)?;

    // The plan's own bytes are a FLOOR: slabs have tails. Cut them the cheapest
    // way for this particular plan and check what that really costs, before any
    // `vkAllocateMemory` — taking `maxMemoryAllocationSize` as the slab size,
    // which is the obvious choice, costs 1.34 GiB here. See `Qwen4Packing`.
    let max_slab = ctx.max_memory_allocation_size();
    // `SlabAllocator` applies exactly this floor; mirroring it keeps the dry run
    // and the real placement on the same alignment.
    let alignment = ctx.min_storage_buffer_offset_alignment().max(16);
    let pack_tier = |tier| match cfg.slab_bytes {
        Some(bytes) => plan.pack(tier, bytes.min(max_slab), alignment),
        None => plan.choose_packing(tier, max_slab, alignment),
    };
    let packing = pack_tier(Qwen4Tier::Device)?;
    packing.ensure_fits(budget.bytes, cfg.reserve_bytes)?;
    log::info!(
        "qwen4 residency: {:.2} GiB over {} slabs of {:.0} MiB ({:.2}% packing waste),          {:.2} GiB of the {:.2} GiB budget left",
        gib(packing.committed_bytes),
        packing.slab_count,
        packing.slab_bytes as f64 / (1u64 << 20) as f64,
        100.0 * packing.waste_fraction(),
        gib(budget.bytes.saturating_sub(packing.committed_bytes)),
        gib(budget.bytes),
    );

    let mut slabs = vulkan_sys::SlabAllocator::with_slab_size(ctx, packing.slab_bytes)
        .map_err(|e| anyhow!("qwen4 upload: creating the slab allocator: {e}"))?;
    let mut spill = if plan.spill_bytes == 0 {
        None
    } else {
        let spill_packing = pack_tier(Qwen4Tier::HostSpill)?;
        // No reserve on the host heap: nothing else this loader allocates lives
        // there, and the KV cache / arena the reserve protects are device-side.
        spill_packing.ensure_fits(host_heap_available(ctx), 0)?;
        log::info!(
            "qwen4 spill tier: {:.2} GiB over {} host slabs of {:.0} MiB",
            gib(spill_packing.committed_bytes),
            spill_packing.slab_count,
            spill_packing.slab_bytes as f64 / (1u64 << 20) as f64,
        );
        Some(Qwen4SpillSlabs::new(
            ctx,
            spill_packing.slab_bytes,
            alignment,
            budget.heap_index,
        )?)
    };

    // Largest-first placement WITHIN each tier; `order` indexes back into
    // `plan.items`. Same order each tier's dry run used, so the real slab counts
    // match what was checked.
    let mut order: Vec<usize> = (0..plan.items.len()).collect();
    order.sort_by_key(|&i| (plan.items[i].tier, std::cmp::Reverse(plan.items[i].bytes)));
    let mut allocs: Vec<Option<vulkan_sys::SlabAlloc>> = vec![None; plan.items.len()];
    for &i in &order {
        let item = &plan.items[i];
        let alloc = match item.tier {
            Qwen4Tier::Device => slabs.alloc(item.bytes).map_err(|e| {
                anyhow!(
                    "qwen4 upload: reserving {} B for {}: {e}",
                    item.bytes,
                    item.name
                )
            })?,
            Qwen4Tier::HostSpill => spill
                .as_mut()
                .ok_or_else(|| anyhow!("{}: spilled with no spill tier", item.name))?
                .alloc(item.bytes)?,
        };
        allocs[i] = Some(alloc);
    }

    // One host buffer, sized to the biggest suballocation in EITHER tier and
    // reused. Two reasons it is not per-tensor: `SlabAllocator::write` needs one
    // contiguous slice (there is no write-at-offset), and `lm_head` alone would
    // otherwise churn 1.18 GiB of allocate/zero/free.
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
            Qwen4Source::Bf16Slice { tensor, row_offset } => {
                // A row range of a stacked tensor is bytewise just a smaller
                // BF16 matrix, so it stages through the same per-format arms
                // below via a recursive view — minus the bias fold, which no
                // sliced family carries.
                let all = st.tensor_data(tensor)?;
                let row_bytes = item.ncols * 2;
                let start = row_offset * row_bytes;
                let end = start + item.nrows * row_bytes;
                ensure!(
                    end <= all.len(),
                    "{}: slice rows {row_offset}..{} overrun `{tensor}` ({} B)",
                    item.name,
                    row_offset + item.nrows,
                    all.len()
                );
                stage_bf16_dense(&item.name, item, &all[start..end], false, dst)?;
            }
            Qwen4Source::Bf16 { fold_bias } => {
                let src = st.tensor_data(&item.name)?;
                stage_bf16_dense(&item.name, item, src, *fold_bias, dst)?;
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
                            tier: item.tier,
                        },
                        n_experts: *n_experts,
                        weight_scale_2,
                        input_scale,
                    },
                );
            }
        }
        match item.tier {
            Qwen4Tier::Device => slabs
                .write(&alloc, dst)
                .map_err(|e| anyhow!("qwen4 upload: writing {}: {e}", item.name))?,
            Qwen4Tier::HostSpill => spill
                .as_mut()
                .ok_or_else(|| anyhow!("{}: spilled with no spill tier", item.name))?
                .write(&alloc, dst)
                .with_context(|| format!("writing {}", item.name))?,
        }

        tensors.insert(
            item.name.clone(),
            Qwen4DeviceTensor {
                alloc,
                ncols: item.ncols,
                nrows: item.nrows,
                format: item.format,
                tier: item.tier,
            },
        );
    }

    Ok(Qwen4Weights {
        slabs,
        spill,
        tensors,
        experts,
        layer_kinds: plan.layer_kinds.clone(),
        host: Qwen4HostTables::build(st)?,
    })
}

/// Stage one BF16 matrix (`src`, row-major `[item.nrows, item.ncols]`) into
/// `dst` in `item.format` — the shared body of the [`Qwen4Source::Bf16`] and
/// [`Qwen4Source::Bf16Slice`] arms, so a sliced stack cannot drift from the
/// whole-tensor conversion.
#[cfg(feature = "vulkan")]
fn stage_bf16_dense(
    name: &str,
    item: &Qwen4PlanItem,
    src: &[u8],
    fold_bias: bool,
    dst: &mut [u8],
) -> Result<()> {
    match item.format {
        Qwen4DeviceFormat::Bf16 => {
            ensure!(
                !fold_bias,
                "{name}: the 1+w fold needs a converting format, not verbatim BF16"
            );
            ensure!(
                src.len() == dst.len(),
                "{name}: {} B of BF16 into a {} B buffer",
                src.len(),
                dst.len()
            );
            dst.copy_from_slice(src);
        }
        Qwen4DeviceFormat::Q4K => {
            ensure!(
                !fold_bias,
                "{name}: the 1+w fold needs a converting format, not Q4_K"
            );
            quantize_rows_threaded(
                name,
                src,
                item.nrows,
                item.ncols,
                dst,
                256,
                144,
                vulkan_kernels::quantize_q4_k_from_bf16,
            )?;
        }
        Qwen4DeviceFormat::Q8_0 => {
            ensure!(
                !fold_bias,
                "{name}: the 1+w fold needs a float format, not Q8_0"
            );
            quantize_rows_threaded(
                name,
                src,
                item.nrows,
                item.ncols,
                dst,
                32,
                34,
                vulkan_kernels::quantize_q8_0_from_bf16,
            )?;
        }
        Qwen4DeviceFormat::F16 => write_bf16_as_f16(name, src, dst)?,
        Qwen4DeviceFormat::F32 => write_bf16_as_f32(name, src, dst, fold_bias)?,
        Qwen4DeviceFormat::Nvfp4 => {
            bail!("{name}: a BF16 source cannot land as NVFP4")
        }
    }
    Ok(())
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
    /// The 8060S's device-local heap SIZE, read off the device
    /// (`VulkanContext::memory_heaps`): 74.4322 GiB. Pinned rather than read
    /// from the running box so the budget is a fact under test, not a
    /// tautology. NOT what a plan is sized against — see [`BUDGET`].
    const HEAP: u64 = 79_920_955_392;
    /// What the DRIVER grants on that heap (`VK_EXT_memory_budget`'s
    /// `heapBudget`, usage 0 at idle): 70.7107 GiB — 3.72 GiB under the size.
    /// This is the number a residency plan lives or dies by; over-committing a
    /// UMA part is silent page demotion, not `OUT_OF_DEVICE_MEMORY`.
    const BUDGET: u64 = 75_924_905_984;
    /// Set to run the whole 71 GiB residency (minutes, and the plan must fit).
    ///
    /// MEASURED STATE, so the next reader is not surprised: with the spill tier
    /// this gets the plan past every guard — 69.58 GiB of device slabs + 3.08
    /// GiB of host slabs, all within the driver's 70.71 GiB budget — and then
    /// dies ~450 s in, at a `SlabAllocator::write` command SUBMIT, with
    /// `ERROR_OUT_OF_DEVICE_MEMORY` on a 40 KB norm. That is not the budget
    /// guard: a separate probe allocated 74 x 1 GiB DEVICE_LOCAL slabs on this
    /// box before failing, i.e. the budget is advisory and there was room. The
    /// remaining suspect is total system pressure — 72.7 GiB of committed
    /// buffers against 128 GB of LPDDR5X while ~122 GiB of checkpoint is being
    /// streamed through the page cache. Sizing THAT is the next phase's job;
    /// this test is left able to fail rather than weakened to pass.
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
            mtp: false,
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
            // The MTP fusion norms are currently host-applied (raw upload);
            // F32 keeps them exact for a future device consumer.
            MtpPreFcNormEmbedding,
            MtpPreFcNormHidden,
        ] {
            assert_eq!(fmt(kind), Some(F32), "{kind:?} must stay F32");
        }

        // The dense GEMV weights follow the config — the MTP fusion fcs
        // included since the S8 reclassification.
        for kind in [
            LinearAttnInProjQkv,
            AttnQProj,
            AttnOProj,
            SharedExpertDownProj,
            PleKeyProj,
            LmHead,
            MtpFcEmbedding,
            MtpFcHidden,
        ] {
            assert_eq!(fmt(kind), Some(F16), "{kind:?} at dense=F16");
            assert_eq!(
                device_format(kind, F32),
                Some(F32),
                "{kind:?} must follow dense_format"
            );
        }

        // Host tables, the vision tower, and the stacked MTP experts (which
        // upload only as per-expert SLICES with their own format policy) get
        // no whole-tensor buffer.
        for kind in [
            EmbedTokens,
            PleNgramShard,
            PleNgramWeightScale,
            ExpertsStackedGateUp,
            ExpertsStackedDown,
            Vision(crate::qwen4_names::VisionSlot::Merger),
        ] {
            assert_eq!(fmt(kind), None, "{kind:?} must not get a device buffer");
        }

        // The slice policy mirrors SharedExpertDownProj's Q4_K exception.
        use crate::qwen4_names::ExpertProj;
        assert_eq!(
            mtp_expert_slice_format(ExpertProj::Gate, Qwen4DeviceFormat::Q4K),
            Qwen4DeviceFormat::Q4K
        );
        assert_eq!(
            mtp_expert_slice_format(ExpertProj::Up, Qwen4DeviceFormat::Q4K),
            Qwen4DeviceFormat::Q4K
        );
        assert_eq!(
            mtp_expert_slice_format(ExpertProj::Down, Qwen4DeviceFormat::Q4K),
            Qwen4DeviceFormat::Q8_0
        );
        assert_eq!(mtp_expert_slice_format(ExpertProj::Down, F16), F16);
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
    fn ensure_fits_refuses_a_plan_the_driver_will_not_grant() {
        let plan = Qwen4Plan {
            device_bytes: 72 << 30,
            ..Default::default()
        };
        // 72 GiB fits the 74.43 GiB HEAP with a 1 GiB reserve...
        plan.ensure_fits(&heap_size_budget(), 1 << 30)
            .expect("72 GiB fits the heap size");
        // ...and does NOT fit the 70.71 GiB the driver actually grants. Same
        // plan, same reserve, opposite verdicts: this pair is the whole defect.
        let err = plan
            .ensure_fits(&driver_budget(), 1 << 30)
            .expect_err("must not fit the driver budget");
        assert!(
            err.to_string().contains("over by"),
            "message should quantify the overage: {err}"
        );
        // The qwen35 loader's 3 GiB reserve does not fit this model either way.
        assert!(plan.ensure_fits(&heap_size_budget(), 3 << 30).is_err());
    }

    #[test]
    fn ensure_fits_is_exact_at_budget_minus_reserve() {
        let budget = driver_budget();
        let usable = budget.bytes - DEFAULT_RESERVE_BYTES;
        let at = Qwen4Plan {
            device_bytes: usable,
            ..Default::default()
        };
        at.ensure_fits(&budget, DEFAULT_RESERVE_BYTES)
            .expect("exactly budget - reserve fits");
        let over = Qwen4Plan {
            device_bytes: usable + 1,
            ..Default::default()
        };
        assert!(
            over.ensure_fits(&budget, DEFAULT_RESERVE_BYTES).is_err(),
            "one byte past budget - reserve must not fit"
        );
    }

    // ------------------------------------------------------------- spill tier

    /// A plan of `bytes`-sized items, `stacks` of them NVFP4 stacks (spilled
    /// first) and the rest BF16 dense.
    fn synthetic_plan(stacks: &[u64], dense: &[u64]) -> Qwen4Plan {
        let mut plan = Qwen4Plan::default();
        let role = Qwen4TensorRole {
            kind: Qwen4TensorKind::MoeRouter,
            layer: Some(0),
            sub_index: None,
            stream: Qwen4Stream::Text,
            residency: Qwen4Residency::DevicePacked,
        };
        for (n, &bytes) in stacks.iter().enumerate() {
            plan.items.push(Qwen4PlanItem {
                name: format!("stack{n}"),
                role,
                format: Qwen4DeviceFormat::Nvfp4,
                bytes,
                tier: Qwen4Tier::Device,
                ncols: 2560,
                nrows: 640,
                source: Qwen4Source::Nvfp4Stack {
                    layer: 0,
                    proj: ExpertProj::Gate,
                    n_experts: 512,
                },
            });
            plan.device_bytes += bytes;
        }
        for (n, &bytes) in dense.iter().enumerate() {
            plan.items.push(Qwen4PlanItem {
                name: format!("dense{n}"),
                role,
                format: Qwen4DeviceFormat::F16,
                bytes,
                tier: Qwen4Tier::Device,
                ncols: 2560,
                nrows: 2560,
                source: Qwen4Source::Bf16 { fold_bias: false },
            });
            plan.device_bytes += bytes;
        }
        plan
    }

    fn budget_of(bytes: u64) -> DeviceBudget {
        DeviceBudget {
            bytes,
            source: crate::loader::DeviceBudgetSource::DriverBudget,
            heap_index: 1,
            heap_size: HEAP,
        }
    }

    /// heap 1 sized by what the DRIVER grants: 70.71 GiB.
    fn driver_budget() -> DeviceBudget {
        budget_of(BUDGET)
    }

    /// The same heap sized by `VkMemoryHeap::size` instead: 74.43 GiB. What the
    /// guard used before this change, and 3.72 GiB more than it may have.
    fn heap_size_budget() -> DeviceBudget {
        DeviceBudget {
            bytes: HEAP,
            source: crate::loader::DeviceBudgetSource::HeapSize,
            heap_index: 1,
            heap_size: HEAP,
        }
    }

    #[test]
    fn spill_moves_expert_stacks_before_dense_weights() {
        // 4 GiB of plan, 3 GiB of budget, no reserve: exactly 1 GiB must move,
        // and it must come out of the sparsely-read stacks.
        let mut plan = synthetic_plan(&[1 << 30, 1 << 30], &[1 << 30, 1 << 30]);
        let moved = plan
            .spill_to_fit(&budget_of(3 << 30), 0, 64 << 30)
            .expect("1 GiB of 4 fits after one move");
        assert_eq!(moved.items, 1, "one 1 GiB suballocation is enough");
        assert_eq!(moved.bytes, 1 << 30);
        assert_eq!(plan.device_bytes, 3 << 30);
        assert_eq!(plan.spill_bytes, 1 << 30);
        // A stack, not a dense weight: 10 of 512 experts are read per token, so
        // a spilled stack pays the host-heap penalty on ~2% of its bytes.
        let spilled: Vec<&str> = plan
            .tier_items(Qwen4Tier::HostSpill)
            .map(|i| i.name.as_str())
            .collect();
        assert_eq!(spilled, ["stack0"], "the dense items must stay on device");
        // Running totals and the item tags agree.
        assert_eq!(plan.device_bytes, plan.tier_bytes(Qwen4Tier::Device));
        assert_eq!(plan.spill_bytes, plan.tier_bytes(Qwen4Tier::HostSpill));
    }

    #[test]
    fn spill_takes_dense_weights_only_after_every_stack() {
        // 4 GiB of plan against 1 GiB of budget: both stacks plus one dense.
        let mut plan = synthetic_plan(&[1 << 30, 1 << 30], &[1 << 30, 1 << 30]);
        let moved = plan
            .spill_to_fit(&budget_of(1 << 30), 0, 64 << 30)
            .expect("3 of 4 GiB can move");
        assert_eq!(moved.items, 3);
        let spilled: Vec<&str> = plan
            .tier_items(Qwen4Tier::HostSpill)
            .map(|i| i.name.as_str())
            .collect();
        assert_eq!(
            spilled,
            ["stack0", "stack1", "dense0"],
            "stacks exhaust before any dense weight moves"
        );
    }

    #[test]
    fn spill_stops_the_moment_the_device_tier_fits() {
        // Budget already covers the plan: nothing moves, and the call is a
        // no-op rather than a "spill everything eligible".
        let mut plan = synthetic_plan(&[1 << 30], &[1 << 30]);
        let moved = plan
            .spill_to_fit(&budget_of(8 << 30), 0, 64 << 30)
            .expect("a plan that fits needs no spill");
        assert_eq!((moved.items, moved.bytes), (0, 0));
        assert_eq!(plan.spill_bytes, 0);
        assert!(plan.tier_items(Qwen4Tier::HostSpill).next().is_none());
        // And it is idempotent: a second call on the spilled plan moves nothing.
        let mut plan = synthetic_plan(&[1 << 30, 1 << 30], &[]);
        plan.spill_to_fit(&budget_of(1 << 30), 0, 64 << 30).unwrap();
        let again = plan.spill_to_fit(&budget_of(1 << 30), 0, 64 << 30).unwrap();
        assert_eq!(again.items, 0);
        assert_eq!(plan.spill_bytes, 1 << 30);
    }

    #[test]
    fn spill_refuses_when_the_host_heap_cannot_hold_it() {
        let mut plan = synthetic_plan(&[4 << 30], &[1 << 30]);
        let err = plan
            .spill_to_fit(&budget_of(1 << 30), 0, 1 << 30)
            .expect_err("4 GiB must not spill into a 1 GiB host heap");
        assert!(
            err.to_string().contains("host heap"),
            "message should name the host heap: {err}"
        );
        // Untouched: a refusal must not leave a half-spilled plan behind for a
        // caller that still holds the `&mut`.
        assert_eq!(plan.spill_bytes, 0);
        assert_eq!(plan.device_bytes, 5 << 30);
        assert!(plan.tier_items(Qwen4Tier::HostSpill).next().is_none());
    }

    #[test]
    fn spill_refuses_a_budget_the_reserve_swallows_whole() {
        // The reserve alone exceeds the budget, so nothing device-side is left
        // for the KV cache and arena. Spilling every weight would "fit" by
        // arithmetic and then die at the first `vkAllocateMemory`; refuse here
        // instead, with the numbers.
        let mut plan = synthetic_plan(&[1 << 30], &[1 << 30]);
        let err = plan
            .spill_to_fit(&budget_of(1 << 30), 4 << 30, 64 << 30)
            .expect_err("a reserve past the budget cannot be satisfied");
        assert!(
            err.to_string().contains("nothing on the device"),
            "message should say the box has no device memory left: {err}"
        );
        // And the plan is untouched: a refusal must not leave a half-spilled
        // plan behind for a caller that logs the Err and carries on.
        assert_eq!(plan.spill_bytes, 0);
        assert_eq!(plan.device_bytes, 2 << 30);
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
        // The MTP pseudo layer addresses all THREE of the head's sites —
        // unlike a text layer it carries its own mixer.
        assert_eq!(
            hyper_connection_prefix(Some(MTP_HC_LAYER), HcSite::Attn).unwrap(),
            "mtp.layers.0.attn_hyper_connection"
        );
        assert_eq!(
            hyper_connection_prefix(Some(MTP_HC_LAYER), HcSite::Mlp).unwrap(),
            "mtp.layers.0.mlp_hyper_connection"
        );
        assert_eq!(
            hyper_connection_prefix(Some(MTP_HC_LAYER), HcSite::Mixer).unwrap(),
            "mtp.hyper_connection_mixer"
        );
    }

    /// The per-expert slice geometry over the REAL stacked headers: 3 slices
    /// per expert, gate rows before up rows, formats per the dense tier, and
    /// none of it books against a TEXT layer's byte ledger.
    #[test]
    fn mtp_plan_slices_the_stacked_experts() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = SafeTensorsDir::open_dir(&dir).expect("open the whole checkpoint");
        let cfg = Qwen4UploadConfig {
            dense_format: Qwen4DeviceFormat::Q4K,
            ..Qwen4UploadConfig::default()
        };
        let plan = plan_qwen4_upload(&st, &cfg, &Qwen4UploadScope::full()).expect("plan");
        let slice = |name: &str| {
            plan.items
                .iter()
                .find(|i| i.name == name)
                .unwrap_or_else(|| panic!("plan has no `{name}`"))
        };
        let gate = slice("mtp.layers.0.mlp.experts.7.gate_proj.weight");
        let up = slice("mtp.layers.0.mlp.experts.7.up_proj.weight");
        let down = slice("mtp.layers.0.mlp.experts.7.down_proj.weight");
        assert_eq!((gate.nrows, gate.ncols), (640, 2560));
        assert_eq!((up.nrows, up.ncols), (640, 2560));
        assert_eq!((down.nrows, down.ncols), (2560, 640));
        assert_eq!(gate.format, Qwen4DeviceFormat::Q4K);
        assert_eq!(up.format, Qwen4DeviceFormat::Q4K);
        // 640 is not a whole number of Q4_K superblocks.
        assert_eq!(down.format, Qwen4DeviceFormat::Q8_0);
        // Fused [gate; up]: expert 7's gate at row 7 * 1280, up 640 later.
        let row_of = |i: &Qwen4PlanItem| match &i.source {
            Qwen4Source::Bf16Slice { row_offset, .. } => *row_offset,
            s => panic!("expected a slice source, got {s:?}"),
        };
        assert_eq!(row_of(gate), 7 * 1280);
        assert_eq!(row_of(up), 7 * 1280 + 640);
        assert_eq!(row_of(down), 7 * 2560);
        // MTP bytes book stream-global, never against text layer 0.
        let mtp_bytes: u64 = plan
            .items
            .iter()
            .filter(|i| i.role.stream == Qwen4Stream::Mtp)
            .map(|i| i.bytes)
            .sum();
        assert!(
            plan.global_bytes >= mtp_bytes,
            "MTP bytes must be inside the global ledger"
        );
        // And a scope without the head plans none of it.
        let no_mtp = plan_qwen4_upload(
            &st,
            &cfg,
            &Qwen4UploadScope {
                mtp: false,
                ..Qwen4UploadScope::full()
            },
        )
        .expect("plan without mtp");
        assert!(
            no_mtp
                .items
                .iter()
                .all(|i| i.role.stream != Qwen4Stream::Mtp),
            "scope.mtp = false must drop every mtp item"
        );
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
        // The mixer is `use_combine=false`, so three tensors and no
        // `block_inject_weight`: the folded hc_norm stays F32 (10240 x 4 B),
        // the two mix tensors follow the BF16 dense default (320 x 10240 x
        // 2 B each) — the split this pins is exactly the one the format
        // table draws between "folds at load" and "verbatim bytes".
        assert_eq!(
            plan.global_bytes,
            10240 * 4 + 2 * 320 * 10240 * 2,
            "mixer = F32 hc_norm + two BF16 mix tensors"
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
                mtp: false,
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
                mtp: false,
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

    /// The spill tier, end to end on the real device: a weight that could not
    /// fit the budget lands on the HOST heap and reads back byte-identical to
    /// the same weight uploaded device-resident.
    ///
    /// Driven by an artificial `device_budget_bytes` one byte under the subset
    /// plan, which is what lets this run in seconds instead of on the 71 GiB
    /// full residency. The two things it can catch are the two that would be
    /// invisible otherwise: bytes written to the wrong offset in a slab this
    /// module allocates itself, and `alloc_host_cached` quietly falling back
    /// onto the device heap.
    #[cfg(feature = "vulkan")]
    #[test]
    fn spilled_weights_land_on_the_host_heap_byte_identical() {
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
        let st = open_subset(&dir);
        let cfg = Qwen4UploadConfig::default();
        let plan = plan_qwen4_upload(&st, &cfg, &subset_scope()).expect("plan");

        // Device-resident reference.
        let base = upload_qwen4(&ctx, &st, &plan, &cfg).expect("reference upload");
        assert_eq!(base.slab_count(Qwen4Tier::HostSpill), 0, "nothing spills");

        // Pick a slab size and a budget that make the spill tier INTERESTING:
        // every expert stack has to move, and the slab holds two of them, so at
        // least one spilled tensor sits at a NONZERO offset. Both halves matter
        // — with a single spilled item, or one item per slab, every offset is
        // zero and a `copy_from_host_at(0, ..)` bug reads clean.
        let stack_bytes = plan
            .items
            .iter()
            .filter(|i| matches!(i.source, Qwen4Source::Nvfp4Stack { .. }))
            .map(|i| i.bytes)
            .max()
            .expect("the subset has expert stacks");
        let stacks_total: u64 = plan
            .items
            .iter()
            .filter(|i| matches!(i.source, Qwen4Source::Nvfp4Stack { .. }))
            .map(|i| i.bytes)
            .sum();
        let dense_bytes = plan.device_bytes - stacks_total;
        let slab = 2 * stack_bytes + (16 << 20);
        // Low bound: the device tier's own slabs must still fit the budget.
        // High bound: one stack under "everything but the stacks", so no stack
        // can stay. Midpoint, so neither rounding nor a checkpoint reshuffle
        // silently lands outside.
        let lo = dense_bytes.div_ceil(slab) * slab;
        let hi = dense_bytes + stack_bytes;
        assert!(
            lo < hi,
            "no budget forces every stack out while leaving room for {lo} B of device slabs"
        );
        let host_before = host_heap_available(&ctx);
        let tight = Qwen4UploadConfig {
            reserve_bytes: 0,
            device_budget_bytes: Some(lo + (hi - lo) / 2),
            slab_bytes: Some(slab),
            ..cfg
        };
        let spilled = upload_qwen4(&ctx, &st, &plan, &tight).expect("spilled upload");
        let host_after = host_heap_available(&ctx);

        let mut moved: Vec<(&String, &Qwen4DeviceTensor)> = spilled
            .tensors
            .iter()
            .filter(|(_, t)| t.tier == Qwen4Tier::HostSpill)
            .collect();
        moved.sort_by_key(|(_, t)| (t.alloc.slab(), t.alloc.offset()));
        eprintln!(
            "qwen4 spill device test: {} suballocation(s) moved",
            moved.len()
        );
        for (name, t) in &moved {
            eprintln!(
                "  {name}: slab {} offset {} len {}",
                t.alloc.slab(),
                t.alloc.offset(),
                t.alloc.len()
            );
        }
        assert!(
            moved.len() >= 2,
            "the offset comparison below needs more than one spilled suballocation"
        );
        assert!(
            moved.iter().any(|(_, t)| t.alloc.offset() > 0),
            "every spilled tensor is at offset 0, so a write-at-0 bug would read clean"
        );
        assert!(
            moved
                .iter()
                .all(|(_, t)| t.format == Qwen4DeviceFormat::Nvfp4),
            "the coldest tier is the NVFP4 expert stacks; dense weights should not move"
        );
        let spill_bytes: u64 = moved.iter().map(|(_, t)| t.alloc.len()).sum();
        assert!(spilled.slab_count(Qwen4Tier::HostSpill) >= 1);
        assert!(spilled.committed_bytes(Qwen4Tier::HostSpill) >= spill_bytes);

        // The host heap really gave up the bytes. Without this the whole tier
        // could be sitting on heap 1 and every other assertion would still pass.
        assert!(
            host_before.saturating_sub(host_after) >= spill_bytes / 2,
            "host heap available moved by only {} B for a {spill_bytes} B spill — the slabs \
             did not land on heap 0",
            host_before.saturating_sub(host_after),
        );

        // Byte-identical to the device-resident upload of the same tensors,
        // whole-stack: a wrong slab offset shows up as a shifted tail, not as a
        // wrong first block.
        for (name, t) in &moved {
            let reference = *base.tensor(name).expect("resident in the reference upload");
            assert_eq!(reference.tier, Qwen4Tier::Device);
            assert_eq!(reference.alloc.len(), t.alloc.len());
            let len = usize::try_from(t.alloc.len()).unwrap();
            let mut want = vec![0u8; len];
            let mut got = vec![0u8; len];
            base.read_back(&reference, &mut want)
                .expect("device read-back");
            spilled.read_back(t, &mut got).expect("spill read-back");
            assert!(
                want.iter().any(|&b| b != 0),
                "{name}: the reference read-back is all zeros; the comparison would be vacuous"
            );
            // Not `assert_eq!` on the vectors: a mismatch would print 117 MB of
            // bytes. Report the first differing index instead, which is also
            // the number that says WHAT went wrong — 0 for a wrong tensor, the
            // suballocation's own offset for a wrong slab offset.
            let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
            assert_eq!(
                first_diff,
                None,
                "{name}: spilled bytes differ from device bytes at index {:?} of {len}                  (slab {} offset {})",
                first_diff,
                t.alloc.slab(),
                t.alloc.offset(),
            );
        }

        // The binding resolves against the spill allocator, not the device one:
        // a `SlabAlloc` is only (slab index, offset, len), so the same handle
        // names a different buffer in each tier.
        let (name, t) = moved[moved.len() - 1];
        let (spill_buf, off, blen) = spilled.binding(t).expect("spilled binding");
        let (dev_buf, _, _) = spilled
            .binding(
                spilled
                    .tensor(&layer_tensor_name(0, "mlp.gate.weight"))
                    .unwrap(),
            )
            .expect("a device binding");
        assert_eq!((off, blen), (t.alloc.offset(), t.alloc.len()), "{name}");
        assert!(
            !std::ptr::eq(spill_buf, dev_buf),
            "the spilled tensor must not bind the device slab"
        );
    }

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

        // --- 2. Bf16 byte-exactness: a dense GEMV weight is the
        //     checkpoint's OWN bytes on device — no re-encode to round-trip,
        //     so the assert is raw byte identity against the mmap.
        let out_proj = layer_tensor_name(0, "linear_attn.out_proj.weight");
        let t = *w.tensor(&out_proj).expect("out_proj resident");
        assert_eq!(t.format, Qwen4DeviceFormat::Bf16);
        assert_eq!((t.ncols, t.nrows), (6144, 2560));
        let mut back = vec![0u8; 2 * 4096];
        w.read_back(&t, &mut back).expect("out_proj read-back");
        let src = st.tensor_data(&out_proj).expect("out_proj bytes");
        assert_eq!(
            back,
            src[..back.len()],
            "out_proj: device bytes differ from the checkpoint's own"
        );

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

        // The MTP head plans as per-expert slices (3 per expert) plus its
        // dense tensors, and since S8 it is IN the default full scope.
        let slice_items = plan
            .items
            .iter()
            .filter(|i| matches!(i.source, Qwen4Source::Bf16Slice { .. }))
            .count();
        assert_eq!(
            slice_items,
            512 * 3,
            "one gate/up/down slice per MTP expert"
        );
        assert!(
            plan.items
                .iter()
                .any(|i| i.name == "mtp.fc_embedding.weight"),
            "the MTP fusion fcs must plan"
        );

        // With the MTP head aboard the F16-tier plan no longer fits even the
        // heap SIZE — the spill tier is now load-bearing on every full load,
        // not only against the driver budget.
        let over_size = plan
            .ensure_fits(&heap_size_budget(), DEFAULT_RESERVE_BYTES)
            .expect_err("F16 dense + BF16-sized MTP slices must exceed the 74.43 GiB heap");
        eprintln!("qwen4 full plan vs heap size: {over_size}");
        let over = plan
            .ensure_fits(&driver_budget(), DEFAULT_RESERVE_BYTES)
            .expect_err("the plan must not fit a 70.71 GiB budget");
        eprintln!("qwen4 full plan vs driver budget: {over}");

        // Against the heap-size budget the spill takes ONLY the coldest class
        // — the MTP expert slices, read 10-of-512 and only while speculating.
        let mut size_spilled = plan.clone();
        size_spilled
            .spill_to_fit(&heap_size_budget(), DEFAULT_RESERVE_BYTES, 35 << 30)
            .expect("the size overshoot fits inside the MTP slice mass");
        assert!(
            size_spilled
                .tier_items(Qwen4Tier::HostSpill)
                .all(|i| matches!(i.source, Qwen4Source::Bf16Slice { .. })),
            "the heap-size overshoot must be covered by MTP slices alone"
        );
        assert!(
            plan.device_bytes > BUDGET - DEFAULT_RESERVE_BYTES,
            "the plan is {:.3} GiB against {:.3} GiB usable — if this ever stops \
             being true the spill tier has become unnecessary, which is worth noticing",
            gib(plan.device_bytes),
            gib(BUDGET - DEFAULT_RESERVE_BYTES),
        );

        // Spilling the coldest suballocations is what makes it fit, and it takes
        // only expert stacks: 10 of 512 experts are read per token, so a spilled
        // stack pays the 2.05% host-heap read penalty on ~2% of its bytes.
        let mut spilled = plan.clone();
        let moved = spilled
            .spill_to_fit(&driver_budget(), DEFAULT_RESERVE_BYTES, 35 << 30)
            .expect("the full plan fits once the coldest stacks move to the host heap");
        eprintln!(
            "qwen4 spill: {} suballocation(s) / {:.3} GiB -> device {:.3} GiB, host {:.3} GiB",
            moved.items,
            gib(moved.bytes),
            gib(spilled.device_bytes),
            gib(spilled.spill_bytes),
        );
        spilled
            .ensure_fits(&driver_budget(), DEFAULT_RESERVE_BYTES)
            .expect("the spilled plan fits");
        // Cold-first: the MTP head's bytes (slices, then its dense tensors),
        // then NVFP4 stacks — and no stack moves while MTP bytes remain. The
        // rank order IS the residency policy, so pin it.
        assert!(
            spilled.tier_items(Qwen4Tier::HostSpill).all(|i| {
                matches!(i.source, Qwen4Source::Nvfp4Stack { .. })
                    || i.role.stream == Qwen4Stream::Mtp
            }),
            "only the MTP head and NVFP4 stacks should have had to move"
        );
        let any_stack_spilled = spilled
            .tier_items(Qwen4Tier::HostSpill)
            .any(|i| matches!(i.source, Qwen4Source::Nvfp4Stack { .. }));
        let all_mtp_spilled = spilled
            .items
            .iter()
            .filter(|i| i.role.stream == Qwen4Stream::Mtp)
            .all(|i| i.tier == Qwen4Tier::HostSpill);
        assert!(
            !any_stack_spilled || all_mtp_spilled,
            "an NVFP4 stack spilled while MTP bytes stayed on device — the cold-first rank broke"
        );
        assert_eq!(
            spilled.device_bytes + spilled.spill_bytes,
            plan.device_bytes,
            "a spill relocates bytes, it does not create or destroy them"
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
            f32_plan
                .ensure_fits(&heap_size_budget(), DEFAULT_RESERVE_BYTES)
                .is_err(),
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
        let mut plan = plan_qwen4_upload(
            &st,
            &Qwen4UploadConfig::default(),
            &Qwen4UploadScope::full(),
        )
        .expect("plan the full model");
        // Mirror the loader: the MTP-inclusive plan over-commits and spills
        // its coldest suballocations BEFORE any packing question is asked.
        plan.spill_to_fit(&driver_budget(), DEFAULT_RESERVE_BYTES, 35 << 30)
            .expect("spill the full plan to the driver budget");

        // 2 GiB is `maxMemoryAllocationSize` on this part — the size a loader
        // that did not sweep would naturally pick.
        let naive = plan
            .pack(Qwen4Tier::Device, 2 << 30, 16)
            .expect("pack at the device maximum");
        let chosen = plan
            .choose_packing(Qwen4Tier::Device, 2 << 30, 16)
            .expect("sweep");
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "qwen4 packing: naive {:.3} GiB / {} slabs -> chosen {:.3} GiB / {} slabs              of {:.0} MiB ({:.2}% waste), {:.3} GiB of heap left",
            gib(naive.committed_bytes),
            naive.slab_count,
            gib(chosen.committed_bytes),
            chosen.slab_count,
            chosen.slab_bytes as f64 / (1u64 << 20) as f64,
            100.0 * chosen.waste_fraction(),
            gib(BUDGET.saturating_sub(chosen.committed_bytes)),
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
            chosen.waste_fraction() < 0.005,
            "chosen packing wastes {:.2}%",
            100.0 * chosen.waste_fraction()
        );
        // The whole point: the default reserve is actually available — against
        // the DRIVER budget, the number the loader is gated on (post-spill the
        // device tier fits the heap size either way).
        chosen
            .ensure_fits(BUDGET, DEFAULT_RESERVE_BYTES)
            .expect("the chosen packing must leave the reserve");
        assert!(
            naive.ensure_fits(BUDGET, DEFAULT_RESERVE_BYTES).is_err(),
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
        // The residency does not fit the driver's budget on the device alone —
        // `full_plan_fits_the_device_local_heap` pins that it is 2.93 GiB over —
        // so a full load that succeeds MUST have spilled. If this ever reads 0,
        // either the plan shrank or the budget grew, and both are worth knowing.
        let spilled: Vec<&String> = w
            .tensors
            .iter()
            .filter(|(_, t)| t.tier == Qwen4Tier::HostSpill)
            .map(|(name, _)| name)
            .collect();
        assert!(
            !spilled.is_empty(),
            "the full residency fit the device budget without spilling — check whether the \
             budget or the plan moved"
        );
        assert!(
            spilled.iter().all(|n| n.ends_with("_stack")),
            "only NVFP4 expert stacks should be cold enough to spill, got {spilled:?}"
        );
        eprintln!(
            "PASS: qwen4 full residency in {:.1} s — device {:.2} GiB over {} slabs \
             ({:.2}% waste), host spill {:.2} GiB over {} slabs ({} stacks: {:?})",
            started.elapsed().as_secs_f64(),
            gib(w.slabs().committed_bytes()),
            w.slabs().slab_count(),
            100.0 * w.slabs().wasted_bytes() as f64 / w.slabs().committed_bytes() as f64,
            gib(w.committed_bytes(Qwen4Tier::HostSpill)),
            w.slab_count(Qwen4Tier::HostSpill),
            spilled.len(),
            spilled,
        );
    }
}
