//! `qwen4_exp` (Qwen3.8-Flash-Next) HF-safetensors tensor-name classifier.
//!
//! The Vulkan analogue of [`crate::loader::classify_qwen35_tensor`], but for a
//! checkpoint that ships HF names rather than GGUF ones, and for an
//! architecture ARLE has not run before. Ground truth is the on-box
//! `qwen3.8-flash-next-nvfp4` checkpoint; every family below was read out of its
//! `model.safetensors.index.json` weight_map and the
//! `layer-XXXXX-experts-AAAA-BBBB.safetensors` headers, not guessed. Semantics
//! come from the reference implementation, `modeling_qwen4_exp.py`.
//!
//! Three schema surprises worth stating up front, because each one looks like a
//! bug if you expect a Qwen3-shaped checkpoint:
//!
//! 1. **There is no final norm, and no per-layer input/post-attention norm.**
//!    Grep the weight_map for `model.language_model.norm.weight` and it is not
//!    there. The hyper-connection machinery subsumes both: every block is
//!    wrapped by two `Qwen4ExpTextGatedResidual` modules
//!    (`attn_hyper_connection`, `mlp_hyper_connection`) whose `hc_norm` is the
//!    pre-norm, and the stream-level `hyper_connection_mixer` — a
//!    `GatedResidual` built with `use_combine=False`, hence with no
//!    `block_inject_weight` — is the terminal op before `lm_head`. So: 48 x 2 =
//!    96 gated residuals plus 1 mixer.
//! 2. **Routed experts are stored per-expert, not stacked.** HF's
//!    `Qwen4ExpTextExperts` holds `gate_up_proj`/`down_proj` as 3-D parameters,
//!    but the modelopt NVFP4 pass rewrote the text stream into
//!    `experts.<e>.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}`.
//!    The stacked form survives only on the MTP block, which the quant config
//!    excludes. Both spellings classify.
//! 3. **The n-gram (PLE) embedding table is 47.68 GiB of FP8**, split across 128
//!    `shard_<i>.weight` tensors of `[2500012, 160]` — 160 = `ple_embed_dim`
//!    2560 / `ngram_heads` 16, where `ngram_heads = (ngram_size - 1) *
//!    heads_per_ngram = 2 * 8`. It exists on exactly one layer
//!    (`ple_layer_ids: [2]`, one-indexed, so `layer_idx == 1`).
//!
//! # Residency intent
//!
//! [`Qwen4Residency`] is the *plan*, not the upload; nothing here touches a
//! device. The intent is what makes the model fit the 74.43 GiB device-local
//! heap, and the arithmetic is tight enough that the assignment is forced:
//!
//! | tier | what | device bytes |
//! |---|---|---|
//! | [`DevicePacked`](Qwen4Residency::DevicePacked) | 48 x 512 x 3 NVFP4 projections + their FP8 block scales | 63.28 GiB |
//! | [`DeviceDequant`](Qwen4Residency::DeviceDequant) | the rest of the text stream, BF16 -> F16 (incl. `lm_head`) | ~8.0 GiB |
//! | [`HostGather`](Qwen4Residency::HostGather) | `embed_tokens` (1.27 GiB) + the FP8 n-gram table (47.68 GiB) | 0 |
//! | [`Drop`](Qwen4Residency::Drop) | MTP block (~4.8 GiB) + vision tower (~0.83 GiB) | 0 |
//!
//! which lands at ~71.3 GiB. Uploading either host-gather family, or keeping the
//! MTP experts, blows the heap outright — so `HostGather`/`Drop` here are
//! load-bearing decisions, not tidiness.

use anyhow::{Result, anyhow, bail};

/// Which of the three top-level parameter trees a tensor lives in.
///
/// The split matters because it decides droppability: a text-only decode runs
/// `model.language_model.*` and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4Stream {
    /// `model.language_model.*`, plus the global `lm_head.weight`.
    Text,
    /// `mtp.*` — the 1-layer multi-token-prediction block.
    Mtp,
    /// `model.visual.*` — the 27-block vision tower.
    Vision,
}

/// Where a tensor is meant to end up. Data only; the module docs carry the byte
/// budget that forces each assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4Residency {
    /// Uploaded as raw NVFP4 bytes (2 values/byte) plus the FP8 per-16-element
    /// block scales, decoded inside the GEMV. Dequantizing instead would take
    /// the 63.28 GiB of experts to roughly 253 GiB.
    DevicePacked,
    /// BF16 in the file, F16 on the device.
    DeviceDequant,
    /// Stays on the host; the forward gathers the few rows a token needs.
    HostGather,
    /// Not uploaded at all for a text-only load.
    Drop,
}

/// Which `Qwen4ExpTextGatedResidual` a hyper-connection weight belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HcSite {
    /// Wraps the attention sublayer (`attn_hyper_connection`).
    Attn,
    /// Wraps the MoE sublayer (`mlp_hyper_connection`).
    Mlp,
    /// The stream-level `hyper_connection_mixer`, which collapses the
    /// `hc_count`-wide residual back to `hidden_size` and stands in for the
    /// missing final norm.
    Mixer,
}

/// The four weights of a `Qwen4ExpTextGatedResidual`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HcPart {
    /// `hc_norm` — RMSNorm over the `hc_count * hidden_size` = 10240 stream,
    /// grouped at `hidden_size`.
    Norm,
    /// `input_mix_weight_down` — `[hc_lowrank = 320, 10240]`.
    MixDown,
    /// `input_mix_weight_up` — `[10240, 320]`.
    MixUp,
    /// `block_inject_weight` — `[hc_count = 4, 10240]`. Absent on
    /// [`HcSite::Mixer`].
    BlockInject,
}

/// Which projection of a routed expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

/// The four tensors modelopt emits per NVFP4 linear (`quant_algo: "NVFP4"`,
/// `group_size: 16`, from `hf_quant_config.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nvfp4Part {
    /// `.weight` — U8, two FP4 values per byte, so the stored shape is half the
    /// logical width (e.g. `[640, 1280]` for a `[640, 2560]` matrix).
    Packed,
    /// `.weight_scale` — F8_E4M3, one per 16-element group.
    BlockScale,
    /// `.weight_scale_2` — F32 scalar, the second-level scale for the FP8 block
    /// scales themselves.
    GlobalScale,
    /// `.input_scale` — F32 scalar, the *static activation* scale. A weight-only
    /// dequant GEMV ignores it; it rides along anyway because all 73728 of them
    /// together are 288 KiB, and dropping it would foreclose an
    /// activation-quantized path for no measurable saving.
    InputScale,
}

/// Vision-tower module slot. Deliberately coarser than the other families: the
/// whole tower is [`Qwen4Residency::Drop`] for a text load, so resolving
/// weight-vs-bias would add 15 variants nothing dispatches against. The
/// *matcher* is still exhaustive, so a renamed vision tensor still errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionSlot {
    PatchEmbed,
    PosEmbed,
    BlockNorm,
    BlockAttn,
    BlockMlp,
    Merger,
}

/// Every tensor family in the `qwen4_exp` checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4TensorKind {
    // ---- global ----
    /// `model.language_model.embed_tokens.weight`, `[248320, 2560]` BF16.
    EmbedTokens,
    /// `lm_head.weight` — untied (`tie_word_embeddings: false`).
    LmHead,

    // ---- hyper-connections: this model's norm + residual machinery ----
    HyperConnection {
        site: HcSite,
        part: HcPart,
    },

    // ---- gated-delta linear attention: 36 of 48 layers ----
    /// `[10240, 2560]` = 2 x key_dim (16 x 128) + value_dim (48 x 128).
    LinearAttnInProjQkv,
    LinearAttnInProjZ,
    /// `[48, 2560]` — one scalar per value head, the `b` term.
    LinearAttnInProjB,
    /// `[48, 2560]` — one scalar per value head, the `a` term.
    LinearAttnInProjA,
    /// Depthwise, `[10240, 1, 4]`.
    LinearAttnConv1d,
    /// `A_log`, `[48]`. No `.weight` suffix — it is a bare `nn.Parameter`.
    LinearAttnALog,
    /// `dt_bias`, `[48]`. No `.weight` suffix.
    LinearAttnDtBias,
    /// `RMSNormGated` over `linear_value_head_dim` = 128.
    LinearAttnNorm,
    LinearAttnOutProj,

    // ---- full attention: 12 of 48 layers ----
    /// `[12288, 2560]` = 24 heads x head_dim 256 x **2**: the second half is the
    /// output gate that the reference chunks off and sigmoids, not extra query
    /// heads.
    AttnQProj,
    AttnKProj,
    AttnVProj,
    AttnOProj,
    AttnQNorm,
    AttnKNorm,

    // ---- QSA indexer, one per full-attention layer ----
    /// `[640, 2560]` = (indexer_n_heads 4 + indexer_kv_heads 1) x
    /// indexer_head_dim 128.
    IndexerQkProj,
    IndexerQNorm,
    IndexerKNorm,

    // ---- MoE ----
    /// `mlp.gate.weight`, `[512, 2560]` — the top-10 router.
    MoeRouter,
    /// `mlp.shared_expert_gate.weight`, `[1, 2560]`.
    SharedExpertGate,
    SharedExpertGateProj,
    SharedExpertUpProj,
    SharedExpertDownProj,
    /// One NVFP4 component of one routed expert (the text stream's layout).
    Expert {
        proj: ExpertProj,
        part: Nvfp4Part,
    },
    /// Stacked `experts.gate_up_proj`, `[512, 1280, 2560]` BF16. Only the MTP
    /// block keeps this layout, because `mtp.*` is quant-excluded.
    ExpertsStackedGateUp,
    /// Stacked `experts.down_proj`, `[512, 2560, 640]` BF16.
    ExpertsStackedDown,

    // ---- PLE / n-gram injection (one layer only) ----
    PleKeyProj,
    PleValueProj,
    PleNormKey,
    PleNormQuery,
    PleNormConv,
    /// Depthwise **dilated** conv (`dilation = ngram_size = 3`), `[10240, 1, 4]`.
    PleConv1d,
    /// One of 128 FP8 slices of the n-gram table (`split_ngram_parts: 128`).
    PleNgramShard,
    /// Single BF16 scalar dequantizing the whole FP8 table.
    PleNgramWeightScale,
    /// I64 `[3]` — the per-position hash multipliers.
    PleNgramLayerMultipliers,
    /// I64 `[16]` — start offset of each n-gram head's slice of the table.
    PleNgramHeadsOffsets,
    /// I64 `[16]` — each n-gram head's prime modulus.
    PleNgramHeadsVocabSizes,

    // ---- MTP-only ----
    MtpFcEmbedding,
    MtpFcHidden,
    MtpPreFcNormEmbedding,
    MtpPreFcNormHidden,

    // ---- vision tower ----
    Vision(VisionSlot),
}

impl Qwen4TensorKind {
    /// True for the per-expert NVFP4 families — the ones that must stay packed.
    pub const fn is_routed_expert(self) -> bool {
        matches!(self, Self::Expert { .. })
    }

    /// True for the host-resident tables: `embed_tokens`, the n-gram table, and
    /// the small buffers the n-gram hash consumes in the same host-side step.
    pub const fn is_host_table(self) -> bool {
        matches!(self.text_residency(), Qwen4Residency::HostGather)
    }

    /// Residency intent for this family **on the text stream**. MTP and vision
    /// copies of a shared family are overridden to [`Qwen4Residency::Drop`] by
    /// [`Qwen4TensorRole::new`]; see [`Qwen4Stream`].
    pub const fn text_residency(self) -> Qwen4Residency {
        use Qwen4Residency::*;
        match self {
            // 1.27 GiB that only ever contributes one row per token.
            Self::EmbedTokens => HostGather,

            // The NVFP4 record, kept byte-exact for the packed GEMV.
            Self::Expert { .. } => DevicePacked,

            // 47.68 GiB of FP8 across 128 shards, of which a token reads 16 rows
            // of 160. Uploading it is not a tuning choice, it is 64% of the heap.
            Self::PleNgramShard
            // Needed to dequantize those rows, which happens host-side.
            | Self::PleNgramWeightScale
            // The n-gram id hash (shift / xor / mod-prime) runs on the host, so
            // its three index buffers never reach the device.
            | Self::PleNgramLayerMultipliers
            | Self::PleNgramHeadsOffsets
            | Self::PleNgramHeadsVocabSizes => HostGather,

            // Speculative decode is not wired on the Vulkan lane, and the MTP
            // block carries a full 512-expert BF16 MoE (~4.7 GiB).
            Self::MtpFcEmbedding
            | Self::MtpFcHidden
            | Self::MtpPreFcNormEmbedding
            | Self::MtpPreFcNormHidden
            | Self::ExpertsStackedGateUp
            | Self::ExpertsStackedDown => Drop,

            Self::Vision(_) => Drop,

            _ => DeviceDequant,
        }
    }
}

/// A classified tensor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen4TensorRole {
    pub stream: Qwen4Stream,
    pub kind: Qwen4TensorKind,
    /// `Some(n)` for `...layers.<n>....` (text and MTP); `None` for
    /// stream-level tensors and for the vision tower, whose depth index lives in
    /// [`Self::sub_index`] so it cannot be mistaken for a decoder layer.
    pub layer: Option<usize>,
    /// The second index a name can carry: expert id (0..512) for
    /// [`Qwen4TensorKind::Expert`], shard id (0..128) for
    /// [`Qwen4TensorKind::PleNgramShard`], block id (0..27) for
    /// [`Qwen4TensorKind::Vision`]. `None` otherwise.
    pub sub_index: Option<u32>,
    /// Resolved intent: [`Qwen4TensorKind::text_residency`], forced to
    /// [`Qwen4Residency::Drop`] off the text stream.
    pub residency: Qwen4Residency,
}

impl Qwen4TensorRole {
    fn new(
        stream: Qwen4Stream,
        kind: Qwen4TensorKind,
        layer: Option<usize>,
        sub_index: Option<u32>,
    ) -> Self {
        let residency = match stream {
            Qwen4Stream::Text => kind.text_residency(),
            Qwen4Stream::Mtp | Qwen4Stream::Vision => Qwen4Residency::Drop,
        };
        Self {
            stream,
            kind,
            layer,
            sub_index,
            residency,
        }
    }

    /// Not uploaded for a text-only decode.
    pub const fn is_droppable(self) -> bool {
        matches!(self.residency, Qwen4Residency::Drop)
    }
}

/// Map an HF safetensors tensor name to its role.
///
/// Fails loud on anything unrecognized — including a name that is *almost*
/// right — so a schema change surfaces at load rather than as a silently absent
/// weight thousands of tensors later.
pub fn classify_qwen4_tensor(name: &str) -> Result<Qwen4TensorRole> {
    if name == "lm_head.weight" {
        return Ok(Qwen4TensorRole::new(
            Qwen4Stream::Text,
            Qwen4TensorKind::LmHead,
            None,
            None,
        ));
    }
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        if rest == "embed_tokens.weight" {
            return Ok(Qwen4TensorRole::new(
                Qwen4Stream::Text,
                Qwen4TensorKind::EmbedTokens,
                None,
                None,
            ));
        }
        return classify_stream_body(Qwen4Stream::Text, rest, name);
    }
    if let Some(rest) = name.strip_prefix("mtp.") {
        let mtp_only = match rest {
            "fc_embedding.weight" => Some(Qwen4TensorKind::MtpFcEmbedding),
            "fc_hidden.weight" => Some(Qwen4TensorKind::MtpFcHidden),
            "pre_fc_norm_embedding.weight" => Some(Qwen4TensorKind::MtpPreFcNormEmbedding),
            "pre_fc_norm_hidden.weight" => Some(Qwen4TensorKind::MtpPreFcNormHidden),
            _ => None,
        };
        if let Some(kind) = mtp_only {
            return Ok(Qwen4TensorRole::new(Qwen4Stream::Mtp, kind, None, None));
        }
        return classify_stream_body(Qwen4Stream::Mtp, rest, name);
    }
    if let Some(rest) = name.strip_prefix("model.visual.") {
        return classify_vision(rest, name);
    }
    bail!("qwen4_exp: unrecognized tensor name `{name}`")
}

/// The part of a stream shared by `model.language_model.*` and `mtp.*`: the
/// stream-level mixer plus `layers.<n>.*`. The MTP block is a real decoder layer
/// with the same submodule names, so it reuses the same suffix vocabulary; which
/// suffixes actually occur on which stream is asserted by the tests, not
/// enforced here.
fn classify_stream_body(stream: Qwen4Stream, rest: &str, name: &str) -> Result<Qwen4TensorRole> {
    if let Some(part) = rest.strip_prefix("hyper_connection_mixer.") {
        let part = hc_part(part, name)?;
        // Built with `use_combine=False`, so it has no `block_inject_weight`;
        // seeing one means the module changed shape.
        if part == HcPart::BlockInject {
            bail!("qwen4_exp: hyper_connection_mixer has no block_inject_weight (`{name}`)");
        }
        return Ok(Qwen4TensorRole::new(
            stream,
            Qwen4TensorKind::HyperConnection {
                site: HcSite::Mixer,
                part,
            },
            None,
            None,
        ));
    }
    let layers = rest
        .strip_prefix("layers.")
        .ok_or_else(|| anyhow!("qwen4_exp: unrecognized tensor name `{name}`"))?;
    let (idx, suffix) = layers
        .split_once('.')
        .ok_or_else(|| anyhow!("qwen4_exp: malformed layer tensor `{name}`"))?;
    let layer: usize = idx
        .parse()
        .map_err(|_| anyhow!("qwen4_exp: bad layer index `{idx}` in `{name}`"))?;
    let (kind, sub_index) = classify_layer_suffix(suffix, name)?;
    Ok(Qwen4TensorRole::new(stream, kind, Some(layer), sub_index))
}

fn hc_part(part: &str, name: &str) -> Result<HcPart> {
    Ok(match part {
        "hc_norm.weight" => HcPart::Norm,
        "input_mix_weight_down.weight" => HcPart::MixDown,
        "input_mix_weight_up.weight" => HcPart::MixUp,
        "block_inject_weight.weight" => HcPart::BlockInject,
        other => bail!("qwen4_exp: unknown hyper-connection weight `{other}` (in `{name}`)"),
    })
}

fn classify_layer_suffix(suffix: &str, name: &str) -> Result<(Qwen4TensorKind, Option<u32>)> {
    use Qwen4TensorKind::*;

    if let Some(part) = suffix.strip_prefix("attn_hyper_connection.") {
        let part = hc_part(part, name)?;
        return Ok((
            HyperConnection {
                site: HcSite::Attn,
                part,
            },
            None,
        ));
    }
    if let Some(part) = suffix.strip_prefix("mlp_hyper_connection.") {
        let part = hc_part(part, name)?;
        return Ok((
            HyperConnection {
                site: HcSite::Mlp,
                part,
            },
            None,
        ));
    }
    if let Some(tail) = suffix.strip_prefix("linear_attn.") {
        let kind = match tail {
            "in_proj_qkv.weight" => LinearAttnInProjQkv,
            "in_proj_z.weight" => LinearAttnInProjZ,
            "in_proj_b.weight" => LinearAttnInProjB,
            "in_proj_a.weight" => LinearAttnInProjA,
            "conv1d.weight" => LinearAttnConv1d,
            "A_log" => LinearAttnALog,
            "dt_bias" => LinearAttnDtBias,
            "norm.weight" => LinearAttnNorm,
            "out_proj.weight" => LinearAttnOutProj,
            other => bail!("qwen4_exp: unknown linear_attn weight `{other}` (in `{name}`)"),
        };
        return Ok((kind, None));
    }
    if let Some(tail) = suffix.strip_prefix("self_attn.") {
        let kind = match tail {
            "q_proj.weight" => AttnQProj,
            "k_proj.weight" => AttnKProj,
            "v_proj.weight" => AttnVProj,
            "o_proj.weight" => AttnOProj,
            "q_norm.weight" => AttnQNorm,
            "k_norm.weight" => AttnKNorm,
            "indexer.index_qk_proj.weight" => IndexerQkProj,
            "indexer.q_layernorm.weight" => IndexerQNorm,
            "indexer.k_layernorm.weight" => IndexerKNorm,
            other => bail!("qwen4_exp: unknown self_attn weight `{other}` (in `{name}`)"),
        };
        return Ok((kind, None));
    }
    if let Some(tail) = suffix.strip_prefix("mlp.") {
        return classify_moe(tail, name);
    }
    if let Some(tail) = suffix.strip_prefix("ple.") {
        return classify_ple(tail, name);
    }
    bail!("qwen4_exp: unknown layer tensor `{suffix}` (in `{name}`)")
}

fn classify_moe(tail: &str, name: &str) -> Result<(Qwen4TensorKind, Option<u32>)> {
    use Qwen4TensorKind::*;

    if let Some(rest) = tail.strip_prefix("experts.") {
        match rest {
            "gate_up_proj" => return Ok((ExpertsStackedGateUp, None)),
            "down_proj" => return Ok((ExpertsStackedDown, None)),
            _ => {}
        }
        let (idx, rest) = rest
            .split_once('.')
            .ok_or_else(|| anyhow!("qwen4_exp: malformed expert tensor `{name}`"))?;
        let expert: u32 = idx
            .parse()
            .map_err(|_| anyhow!("qwen4_exp: bad expert index `{idx}` in `{name}`"))?;
        let (proj, part) = rest
            .split_once('.')
            .ok_or_else(|| anyhow!("qwen4_exp: malformed expert tensor `{name}`"))?;
        let proj = match proj {
            "gate_proj" => ExpertProj::Gate,
            "up_proj" => ExpertProj::Up,
            "down_proj" => ExpertProj::Down,
            other => bail!("qwen4_exp: unknown expert projection `{other}` (in `{name}`)"),
        };
        let part = match part {
            "weight" => Nvfp4Part::Packed,
            "weight_scale" => Nvfp4Part::BlockScale,
            "weight_scale_2" => Nvfp4Part::GlobalScale,
            "input_scale" => Nvfp4Part::InputScale,
            other => bail!("qwen4_exp: unknown NVFP4 component `{other}` (in `{name}`)"),
        };
        return Ok((Expert { proj, part }, Some(expert)));
    }

    let kind = match tail {
        "gate.weight" => MoeRouter,
        "shared_expert_gate.weight" => SharedExpertGate,
        "shared_expert.gate_proj.weight" => SharedExpertGateProj,
        "shared_expert.up_proj.weight" => SharedExpertUpProj,
        "shared_expert.down_proj.weight" => SharedExpertDownProj,
        other => bail!("qwen4_exp: unknown mlp weight `{other}` (in `{name}`)"),
    };
    Ok((kind, None))
}

fn classify_ple(tail: &str, name: &str) -> Result<(Qwen4TensorKind, Option<u32>)> {
    use Qwen4TensorKind::*;

    if let Some(rest) = tail.strip_prefix("ple_embedding.") {
        if let Some(shard) = rest.strip_prefix("ngram_embedding.shard_") {
            let idx = shard
                .strip_suffix(".weight")
                .ok_or_else(|| anyhow!("qwen4_exp: malformed n-gram shard `{name}`"))?;
            let shard: u32 = idx
                .parse()
                .map_err(|_| anyhow!("qwen4_exp: bad n-gram shard index `{idx}` in `{name}`"))?;
            return Ok((PleNgramShard, Some(shard)));
        }
        let kind = match rest {
            "ngram_embedding.weight_scale" => PleNgramWeightScale,
            "layer_multipliers" => PleNgramLayerMultipliers,
            "ngram_heads_offsets" => PleNgramHeadsOffsets,
            "ngram_heads_vocab_sizes" => PleNgramHeadsVocabSizes,
            other => bail!("qwen4_exp: unknown ple_embedding tensor `{other}` (in `{name}`)"),
        };
        return Ok((kind, None));
    }

    let kind = match tail {
        "key_proj.weight" => PleKeyProj,
        "value_proj.weight" => PleValueProj,
        "norm_key.weight" => PleNormKey,
        "norm_query.weight" => PleNormQuery,
        "norm_conv.weight" => PleNormConv,
        "conv1d.weight" => PleConv1d,
        other => bail!("qwen4_exp: unknown ple weight `{other}` (in `{name}`)"),
    };
    Ok((kind, None))
}

fn classify_vision(rest: &str, name: &str) -> Result<Qwen4TensorRole> {
    use VisionSlot::*;

    let (slot, block) = if let Some(blocks) = rest.strip_prefix("blocks.") {
        let (idx, tail) = blocks
            .split_once('.')
            .ok_or_else(|| anyhow!("qwen4_exp: malformed vision block tensor `{name}`"))?;
        let block: u32 = idx
            .parse()
            .map_err(|_| anyhow!("qwen4_exp: bad vision block index `{idx}` in `{name}`"))?;
        let slot = match tail {
            "norm1.weight" | "norm1.bias" | "norm2.weight" | "norm2.bias" => BlockNorm,
            "attn.qkv.weight" | "attn.qkv.bias" | "attn.proj.weight" | "attn.proj.bias" => {
                BlockAttn
            }
            "mlp.linear_fc1.weight"
            | "mlp.linear_fc1.bias"
            | "mlp.linear_fc2.weight"
            | "mlp.linear_fc2.bias" => BlockMlp,
            other => bail!("qwen4_exp: unknown vision block weight `{other}` (in `{name}`)"),
        };
        (slot, Some(block))
    } else {
        let slot = match rest {
            "patch_embed.proj.weight" | "patch_embed.proj.bias" => PatchEmbed,
            "pos_embed.weight" => PosEmbed,
            "merger.norm.weight"
            | "merger.norm.bias"
            | "merger.linear_fc1.weight"
            | "merger.linear_fc1.bias"
            | "merger.linear_fc2.weight"
            | "merger.linear_fc2.bias" => Merger,
            other => bail!("qwen4_exp: unknown vision tensor `{other}` (in `{name}`)"),
        };
        (slot, None)
    };
    Ok(Qwen4TensorRole::new(
        Qwen4Stream::Vision,
        Qwen4TensorKind::Vision(slot),
        None,
        block,
    ))
}

#[cfg(test)]
mod tests {
    use super::Qwen4TensorKind::*;
    use super::*;
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
    use std::path::{Path, PathBuf};

    // All measured off the on-box checkpoint, not derived from config.json, so a
    // mismatch means either the classifier or the checkpoint moved.
    const TOTAL_NAMES: usize = 296_475;
    const NUM_LAYERS: usize = 48;
    const FULL_ATTENTION_INTERVAL: usize = 4;
    const NUM_EXPERTS: u32 = 512;
    const EXPERTS_PER_SHARD: u32 = 128;
    const NGRAM_SHARDS: u32 = 128;
    /// `ple_layer_ids: [2]` is ONE-indexed in the reference, so the PLE hangs off
    /// `layer_idx == 1`.
    const PLE_LAYER: usize = 1;
    const VISION_DEPTH: u32 = 27;

    const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
    const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

    fn checkpoint_dir() -> Option<PathBuf> {
        let dir = std::env::var_os(CKPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(CKPT_DEFAULT));
        dir.join("model.safetensors.index.json")
            .is_file()
            .then_some(dir)
    }

    /// Collect the keys of the JSON object that starts at the first `{` in `src`.
    ///
    /// `infer-vulkan` carries no JSON dependency, and the only thing these tests
    /// want out of either file is the set of object keys — the values are a
    /// shard filename (the index) or a dtype/shape record (a safetensors
    /// header), neither of which is under test here.
    fn json_object_keys(src: &str) -> Vec<&str> {
        let bytes = src.as_bytes();
        let mut keys = Vec::new();
        let Some(open) = src.find('{') else {
            return keys;
        };
        let mut i = open + 1;
        let mut depth = 1usize;
        while i < bytes.len() {
            match bytes[i] {
                b'{' | b'[' => {
                    depth += 1;
                    i += 1;
                }
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    i += 1;
                }
                b'"' => {
                    let start = i + 1;
                    let mut end = start;
                    while end < bytes.len() && bytes[end] != b'"' {
                        end += if bytes[end] == b'\\' { 2 } else { 1 };
                    }
                    i = end + 1;
                    // A key is a depth-1 string followed by `:`; the same string
                    // in value position (the shard filename) is not.
                    if depth == 1 {
                        let mut colon = i;
                        while colon < bytes.len() && bytes[colon].is_ascii_whitespace() {
                            colon += 1;
                        }
                        if bytes.get(colon) == Some(&b':') {
                            keys.push(&src[start..end]);
                        }
                    }
                }
                _ => i += 1,
            }
        }
        keys
    }

    /// safetensors framing: 8-byte little-endian header length, then that many
    /// bytes of JSON.
    fn read_safetensors_header(path: &Path) -> Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf)?)
    }

    fn index_names(index: &str) -> Vec<&str> {
        let at = index
            .find("\"weight_map\"")
            .expect("index.json has a weight_map");
        json_object_keys(&index[at..])
    }

    /// The whole point of this module: walk every name the real checkpoint
    /// declares and assert both that it classifies and that the per-family
    /// counts are the architecture's, not something that merely parses.
    #[test]
    fn real_checkpoint_name_list_classifies_completely() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let index = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
            .expect("read model.safetensors.index.json");
        let names = index_names(&index);
        assert_eq!(names.len(), TOTAL_NAMES, "weight_map entry count");

        let mut per_kind: HashMap<(Qwen4Stream, Qwen4TensorKind), usize> = HashMap::new();
        let mut per_residency: HashMap<Qwen4Residency, usize> = HashMap::new();
        let mut gated_residuals: HashSet<(Qwen4Stream, usize, HcSite)> = HashSet::new();
        let mut linear_layers: BTreeSet<usize> = BTreeSet::new();
        let mut full_layers: BTreeSet<usize> = BTreeSet::new();
        let mut ple_layers: BTreeSet<usize> = BTreeSet::new();
        let mut expert_ids: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
        let mut ngram_shards: BTreeSet<u32> = BTreeSet::new();
        let mut vision_blocks: BTreeSet<u32> = BTreeSet::new();

        for name in &names {
            let role =
                classify_qwen4_tensor(name).unwrap_or_else(|e| panic!("classify `{name}`: {e}"));
            *per_kind.entry((role.stream, role.kind)).or_default() += 1;
            *per_residency.entry(role.residency).or_default() += 1;

            let text = role.stream == Qwen4Stream::Text;
            match (role.kind, role.layer) {
                (HyperConnection { site, .. }, Some(layer)) => {
                    gated_residuals.insert((role.stream, layer, site));
                }
                (LinearAttnOutProj, Some(layer)) if text => {
                    linear_layers.insert(layer);
                }
                (AttnQProj, Some(layer)) if text => {
                    full_layers.insert(layer);
                }
                (PleNgramShard, Some(layer)) if text => {
                    ple_layers.insert(layer);
                    ngram_shards.insert(role.sub_index.expect("n-gram shard index"));
                }
                (Expert { .. }, Some(layer)) if text => {
                    expert_ids
                        .entry(layer)
                        .or_default()
                        .insert(role.sub_index.expect("expert id"));
                }
                (Vision(_), _) => {
                    if let Some(block) = role.sub_index {
                        vision_blocks.insert(block);
                    }
                }
                _ => {}
            }
        }

        let text = |kind: Qwen4TensorKind| {
            per_kind
                .get(&(Qwen4Stream::Text, kind))
                .copied()
                .unwrap_or(0)
        };
        let mtp = |kind: Qwen4TensorKind| {
            per_kind
                .get(&(Qwen4Stream::Mtp, kind))
                .copied()
                .unwrap_or(0)
        };
        let vision = |slot: VisionSlot| {
            per_kind
                .get(&(Qwen4Stream::Vision, Vision(slot)))
                .copied()
                .unwrap_or(0)
        };

        // ---- global ----
        assert_eq!(text(EmbedTokens), 1);
        assert_eq!(text(LmHead), 1);

        // ---- 48 x 2 gated residuals + 1 mixer ----
        for site in [HcSite::Attn, HcSite::Mlp] {
            for part in [
                HcPart::Norm,
                HcPart::MixDown,
                HcPart::MixUp,
                HcPart::BlockInject,
            ] {
                assert_eq!(
                    text(HyperConnection { site, part }),
                    NUM_LAYERS,
                    "{site:?}/{part:?}"
                );
            }
        }
        for part in [HcPart::Norm, HcPart::MixDown, HcPart::MixUp] {
            assert_eq!(
                text(HyperConnection {
                    site: HcSite::Mixer,
                    part
                }),
                1,
                "mixer/{part:?}"
            );
        }
        assert_eq!(
            text(HyperConnection {
                site: HcSite::Mixer,
                part: HcPart::BlockInject
            }),
            0,
            "the mixer is use_combine=False; a block_inject_weight would be new"
        );
        let text_blocks = gated_residuals
            .iter()
            .filter(|(stream, _, _)| *stream == Qwen4Stream::Text)
            .count();
        assert_eq!(text_blocks, 2 * NUM_LAYERS, "48 layers x 2 GatedResidual");

        // ---- 36 linear-attention layers, 12 full-attention layers ----
        for kind in [
            LinearAttnInProjQkv,
            LinearAttnInProjZ,
            LinearAttnInProjB,
            LinearAttnInProjA,
            LinearAttnConv1d,
            LinearAttnALog,
            LinearAttnDtBias,
            LinearAttnNorm,
            LinearAttnOutProj,
        ] {
            assert_eq!(text(kind), 36, "{kind:?}");
        }
        for kind in [
            AttnQProj,
            AttnKProj,
            AttnVProj,
            AttnOProj,
            AttnQNorm,
            AttnKNorm,
            IndexerQkProj,
            IndexerQNorm,
            IndexerKNorm,
        ] {
            assert_eq!(text(kind), 12, "{kind:?}");
        }
        let expect_full: BTreeSet<usize> = (0..NUM_LAYERS)
            .filter(|l| (l + 1) % FULL_ATTENTION_INTERVAL == 0)
            .collect();
        let expect_linear: BTreeSet<usize> = (0..NUM_LAYERS)
            .filter(|l| (l + 1) % FULL_ATTENTION_INTERVAL != 0)
            .collect();
        assert_eq!(full_layers, expect_full, "full_attention_interval = 4");
        assert_eq!(linear_layers, expect_linear);

        // ---- MoE: router + shared expert dense, 512 routed experts x 3 x 48 ----
        for kind in [
            MoeRouter,
            SharedExpertGate,
            SharedExpertGateProj,
            SharedExpertUpProj,
            SharedExpertDownProj,
        ] {
            assert_eq!(text(kind), NUM_LAYERS, "{kind:?}");
        }
        let per_expert_tensor = NUM_EXPERTS as usize * NUM_LAYERS;
        for proj in [ExpertProj::Gate, ExpertProj::Up, ExpertProj::Down] {
            for part in [
                Nvfp4Part::Packed,
                Nvfp4Part::BlockScale,
                Nvfp4Part::GlobalScale,
                Nvfp4Part::InputScale,
            ] {
                assert_eq!(
                    text(Expert { proj, part }),
                    per_expert_tensor,
                    "{proj:?}/{part:?}"
                );
            }
        }
        assert_eq!(
            expert_ids.len(),
            NUM_LAYERS,
            "every layer has routed experts"
        );
        let all_experts: BTreeSet<u32> = (0..NUM_EXPERTS).collect();
        for (layer, ids) in &expert_ids {
            assert_eq!(ids, &all_experts, "layer {layer} expert ids");
        }
        // The stacked HF layout survives only where modelopt did not quantize.
        assert_eq!(text(ExpertsStackedGateUp), 0);
        assert_eq!(text(ExpertsStackedDown), 0);
        assert_eq!(mtp(ExpertsStackedGateUp), 1);
        assert_eq!(mtp(ExpertsStackedDown), 1);

        // ---- PLE, on exactly one layer ----
        for kind in [
            PleKeyProj,
            PleValueProj,
            PleNormKey,
            PleNormQuery,
            PleNormConv,
            PleConv1d,
            PleNgramWeightScale,
            PleNgramLayerMultipliers,
            PleNgramHeadsOffsets,
            PleNgramHeadsVocabSizes,
        ] {
            assert_eq!(text(kind), 1, "{kind:?}");
        }
        assert_eq!(text(PleNgramShard), NGRAM_SHARDS as usize);
        assert_eq!(ple_layers, BTreeSet::from([PLE_LAYER]));
        assert_eq!(ngram_shards, (0..NGRAM_SHARDS).collect::<BTreeSet<_>>());

        // ---- MTP: a complete 1-layer block plus its four fc/norm weights ----
        for kind in [
            MtpFcEmbedding,
            MtpFcHidden,
            MtpPreFcNormEmbedding,
            MtpPreFcNormHidden,
        ] {
            assert_eq!(mtp(kind), 1, "{kind:?}");
        }
        let mtp_total: usize = per_kind
            .iter()
            .filter(|((stream, _), _)| *stream == Qwen4Stream::Mtp)
            .map(|(_, count)| count)
            .sum();
        assert_eq!(mtp_total, 31, "mtp.* tensor count");
        assert_eq!(mtp(LinearAttnOutProj), 0, "the MTP block is full-attention");

        // ---- vision tower ----
        assert_eq!(vision(VisionSlot::PatchEmbed), 2, "conv weight + bias");
        assert_eq!(vision(VisionSlot::PosEmbed), 1);
        assert_eq!(vision(VisionSlot::BlockNorm), 4 * VISION_DEPTH as usize);
        assert_eq!(vision(VisionSlot::BlockAttn), 4 * VISION_DEPTH as usize);
        assert_eq!(vision(VisionSlot::BlockMlp), 4 * VISION_DEPTH as usize);
        assert_eq!(vision(VisionSlot::Merger), 6);
        assert_eq!(vision_blocks, (0..VISION_DEPTH).collect::<BTreeSet<_>>());

        // ---- residency intent partitions the whole name list ----
        let tier = |r: Qwen4Residency| per_residency.get(&r).copied().unwrap_or(0);
        assert_eq!(
            tier(Qwen4Residency::DevicePacked),
            3 * 4 * per_expert_tensor,
            "NVFP4 experts"
        );
        assert_eq!(
            tier(Qwen4Residency::HostGather),
            1 + NGRAM_SHARDS as usize + 4,
            "embed_tokens + 128 n-gram shards + scale + 3 index buffers"
        );
        assert_eq!(tier(Qwen4Residency::Drop), 31 + 333, "MTP + vision tower");
        assert_eq!(tier(Qwen4Residency::DeviceDequant), 1066);
        assert_eq!(
            tier(Qwen4Residency::DevicePacked)
                + tier(Qwen4Residency::DeviceDequant)
                + tier(Qwen4Residency::HostGather)
                + tier(Qwen4Residency::Drop),
            TOTAL_NAMES
        );

        eprintln!("PASS: {TOTAL_NAMES} qwen4_exp tensor names classified");
    }

    /// The expert shards carry their own safetensors headers; read one directly
    /// so the classifier is checked against the names as physically stored, not
    /// only against the index's copy of them.
    #[test]
    fn expert_shard_header_classifies_and_is_covered_by_the_index() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let shard = dir.join("layer-00000-experts-0000-0127.safetensors");
        if !shard.is_file() {
            eprintln!("SKIP: {} absent", shard.display());
            return;
        }
        let header = read_safetensors_header(&shard).expect("read shard header");
        let names: Vec<&str> = json_object_keys(&header)
            .into_iter()
            .filter(|name| *name != "__metadata__")
            .collect();
        assert_eq!(names.len(), EXPERTS_PER_SHARD as usize * 3 * 4);

        for name in &names {
            let role =
                classify_qwen4_tensor(name).unwrap_or_else(|e| panic!("classify `{name}`: {e}"));
            assert_eq!(role.stream, Qwen4Stream::Text, "{name}");
            assert_eq!(role.layer, Some(0), "{name}");
            assert!(role.kind.is_routed_expert(), "{name}");
            assert_eq!(role.residency, Qwen4Residency::DevicePacked, "{name}");
            let expert = role.sub_index.expect("expert id");
            assert!(expert < EXPERTS_PER_SHARD, "{name} outside shard 0000-0127");
        }

        // The brief for this module said the expert shards are absent from
        // `model.safetensors.index.json`. They are not: the weight_map covers
        // all 192 of them, so a loader can drive the whole checkpoint off the
        // index alone.
        let index = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
            .expect("read model.safetensors.index.json");
        let indexed: HashSet<&str> = index_names(&index).into_iter().collect();
        for name in &names {
            assert!(indexed.contains(name), "`{name}` missing from the index");
        }
    }

    #[test]
    fn unknown_names_fail_loud() {
        for bad in [
            "totally_unknown",
            "model.language_model.layers.3.mystery.weight",
            "model.language_model.layers.3.self_attn.rotary_emb.inv_freq",
            "model.language_model.layers.0.mlp.experts.abc.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.7.gate_proj.weight_scale_3",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_x.weight",
            "model.visual.blocks.0.attn.q_proj.weight",
            // The norm this architecture does not have — a silent skip here
            // would look like a working load that quietly omits the final norm.
            "model.language_model.norm.weight",
            // use_combine=False, so this weight cannot exist.
            "model.language_model.hyper_connection_mixer.block_inject_weight.weight",
        ] {
            assert!(
                classify_qwen4_tensor(bad).is_err(),
                "`{bad}` must not classify"
            );
        }
    }

    #[test]
    fn residency_intent_is_data() {
        use Qwen4Residency::*;
        assert_eq!(EmbedTokens.text_residency(), HostGather);
        assert_eq!(PleNgramShard.text_residency(), HostGather);
        assert!(PleNgramShard.is_host_table());
        assert!(EmbedTokens.is_host_table());
        assert!(!LmHead.is_host_table());
        assert_eq!(LmHead.text_residency(), DeviceDequant);
        assert_eq!(Vision(VisionSlot::Merger).text_residency(), Drop);
        assert_eq!(ExpertsStackedGateUp.text_residency(), Drop);
        for proj in [ExpertProj::Gate, ExpertProj::Up, ExpertProj::Down] {
            for part in [
                Nvfp4Part::Packed,
                Nvfp4Part::BlockScale,
                Nvfp4Part::GlobalScale,
                Nvfp4Part::InputScale,
            ] {
                assert_eq!(Expert { proj, part }.text_residency(), DevicePacked);
                assert!(Expert { proj, part }.is_routed_expert());
            }
        }

        // The same family is device-resident on the text stream and dropped on
        // the MTP one; the stream, not the kind, is what decides.
        let text = classify_qwen4_tensor("model.language_model.layers.0.mlp.gate.weight")
            .expect("text router");
        let mtp = classify_qwen4_tensor("mtp.layers.0.mlp.gate.weight").expect("mtp router");
        assert_eq!(text.kind, mtp.kind);
        assert_eq!(text.residency, DeviceDequant);
        assert_eq!(mtp.residency, Drop);
        assert!(mtp.is_droppable());
        assert!(!text.is_droppable());
    }
}
