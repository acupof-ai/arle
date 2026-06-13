//! qwen35moe / qwen35 GGUF residency loader: the per-tensor name→role
//! "lookup" and the device-format (residency) decision + byte budget.
//!
//! Mirrors the DSv4 HIP loader (`crates/infer-hip/src/loader.rs`) but for the
//! Qwen3.5 hybrid + MoE tensor schema. Ground truth is the on-box
//! `Qwen3.6-35B-A3B` GGUF (arch `qwen35moe`); the same names cover the
//! `Qwen3.5-122B-A10B` target.
//!
//! Residency policy is tied to what the Vulkan kernel surface
//! (`crates/vulkan-kernels`) consumes:
//! - K-quants (Q4_K/Q5_K/Q6_K) stay quantized on device — the registered
//!   `mul_mat_vecq` GEMV path reads them directly (no dequant).
//! - Q8_0/F16/BF16 weights dequantize to F16 (coopmat-FP16 GEMM consumer;
//!   no Q8_0 GEMV is registered yet — Phase 1 may add one to keep them packed).
//! - F32 tensors (norms, SSM params, router) stay F32 on device.
//! - `token_embd.weight` stays as raw GGUF bytes on the host so the embedding
//!   "lookup" gathers + dequantizes a single row per token instead of
//!   uploading the whole (~0.5–1 GB) table.

use anyhow::{Result, bail, ensure};

use infer_gguf::gguf::{GgmlType, GgufFile, TensorInfo};

/// K-quant tiers the Vulkan GEMV surface serves directly (kept packed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuant {
    Q4K,
    Q5K,
    Q6K,
}

/// Per-tensor device-format decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Q4_K/Q5_K/Q6_K uploaded as-is; consumed by the quantized GEMV.
    KeepKQuant(KQuant),
    /// Q8_0/F16/BF16 dequantized to F16 on device.
    DequantF16,
    /// F32 norms / SSM params / router uploaded as F32.
    DequantF32,
    /// `token_embd.weight`: kept as raw GGUF bytes on the host; the forward
    /// gathers + dequantizes one row per token (see [`Qwen35TensorKind::TokenEmbedding`]).
    HostEmbedding,
}

/// Role of a GGUF tensor in the qwen35moe graph. The `attn_*` (full) and
/// `attn_qkv`/`attn_gate` + `ssm_*` (linear) families coexist because the model
/// interleaves full-attention and gated-delta-linear layers; which family a
/// given layer uses is decided by tensor presence at model-build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35TensorKind {
    // --- global ---
    TokenEmbedding,
    OutputNorm,
    LmHead,
    // --- per-layer norms ---
    AttnNorm,
    PostAttentionNorm,
    // --- full-attention layer ---
    AttnQ,
    AttnK,
    AttnV,
    AttnQNorm,
    AttnKNorm,
    AttnOutput,
    AttnGate,
    // --- linear (gated-delta / SSM) layer ---
    AttnQkv,
    SsmConv1d,
    SsmA,
    SsmAlpha,
    SsmBeta,
    SsmDtBias,
    SsmNorm,
    SsmOut,
    // --- MoE FFN ---
    FfnGateInp,
    FfnGateExps,
    FfnUpExps,
    FfnDownExps,
    FfnGateInpShexp,
    FfnGateShexp,
    FfnUpShexp,
    FfnDownShexp,
}

impl Qwen35TensorKind {
    /// 3-D stacked routed-expert weight (`[in, inter, n_experts]`).
    pub fn is_routed_expert(self) -> bool {
        matches!(
            self,
            Self::FfnGateExps | Self::FfnUpExps | Self::FfnDownExps
        )
    }

    /// RMS-norm weight (1-D, F32) — consumed by the rms_norm kernel.
    pub fn is_norm(self) -> bool {
        matches!(
            self,
            Self::OutputNorm
                | Self::AttnNorm
                | Self::PostAttentionNorm
                | Self::AttnQNorm
                | Self::AttnKNorm
                | Self::SsmNorm
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35TensorRole {
    pub kind: Qwen35TensorKind,
    /// `Some(layer)` for `blk.N.*`, `None` for global tensors.
    pub layer: Option<usize>,
}

/// Map a GGUF tensor name to its role + layer. Fails loud on an unknown name so
/// a schema surprise (new tensor, renamed weight) is caught at load, not
/// silently dropped.
pub fn classify_qwen35_tensor(name: &str) -> Result<Qwen35TensorRole> {
    use Qwen35TensorKind::*;

    // Global (no `blk.` prefix).
    let global = match name {
        "token_embd.weight" => Some(TokenEmbedding),
        "output_norm.weight" => Some(OutputNorm),
        "output.weight" => Some(LmHead),
        _ => None,
    };
    if let Some(kind) = global {
        return Ok(Qwen35TensorRole { kind, layer: None });
    }

    // Per-layer: `blk.<N>.<suffix>`.
    let rest = name
        .strip_prefix("blk.")
        .ok_or_else(|| anyhow::anyhow!("qwen35: unrecognized tensor name `{name}`"))?;
    let (idx, suffix) = rest
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("qwen35: malformed layer tensor `{name}`"))?;
    let layer: usize = idx
        .parse()
        .map_err(|_| anyhow::anyhow!("qwen35: bad layer index in `{name}`"))?;

    // Exact-suffix match (no prefixes) — order-independent.
    let kind = match suffix {
        "attn_norm.weight" => AttnNorm,
        "post_attention_norm.weight" => PostAttentionNorm,
        // full attention
        "attn_q.weight" => AttnQ,
        "attn_k.weight" => AttnK,
        "attn_v.weight" => AttnV,
        "attn_q_norm.weight" => AttnQNorm,
        "attn_k_norm.weight" => AttnKNorm,
        "attn_output.weight" => AttnOutput,
        "attn_gate.weight" => AttnGate,
        // linear / gated-delta SSM
        "attn_qkv.weight" => AttnQkv,
        "ssm_conv1d.weight" => SsmConv1d,
        "ssm_a" => SsmA,
        "ssm_alpha.weight" => SsmAlpha,
        "ssm_beta.weight" => SsmBeta,
        "ssm_dt.bias" => SsmDtBias,
        "ssm_norm.weight" => SsmNorm,
        "ssm_out.weight" => SsmOut,
        // MoE FFN
        "ffn_gate_inp.weight" => FfnGateInp,
        "ffn_gate_exps.weight" => FfnGateExps,
        "ffn_up_exps.weight" => FfnUpExps,
        "ffn_down_exps.weight" => FfnDownExps,
        "ffn_gate_inp_shexp.weight" => FfnGateInpShexp,
        "ffn_gate_shexp.weight" => FfnGateShexp,
        "ffn_up_shexp.weight" => FfnUpShexp,
        "ffn_down_shexp.weight" => FfnDownShexp,
        other => bail!("qwen35: unknown layer tensor suffix `{other}` (in `{name}`)"),
    };
    Ok(Qwen35TensorRole {
        kind,
        layer: Some(layer),
    })
}

/// Device-format decision for a classified tensor.
pub fn plan_residency(kind: Qwen35TensorKind, ty: GgmlType) -> Residency {
    if kind == Qwen35TensorKind::TokenEmbedding {
        return Residency::HostEmbedding;
    }
    match ty {
        GgmlType::Q4K => Residency::KeepKQuant(KQuant::Q4K),
        GgmlType::Q5K => Residency::KeepKQuant(KQuant::Q5K),
        GgmlType::Q6K => Residency::KeepKQuant(KQuant::Q6K),
        GgmlType::F32 => Residency::DequantF32,
        // Q8_0 / F16 / BF16 (and anything else dequantizable) → F16 on device.
        _ => Residency::DequantF16,
    }
}

/// Exact device bytes for one tensor under `residency`.
pub fn device_bytes(residency: Residency, info: &TensorInfo) -> Result<u64> {
    let n = info.element_count();
    Ok(match residency {
        Residency::HostEmbedding => 0,
        Residency::DequantF16 => n * 2,
        Residency::DequantF32 => n * 4,
        Residency::KeepKQuant(_) => info.byte_len().ok_or_else(|| {
            anyhow::anyhow!(
                "tensor {}: cannot keep {:?} packed (unaligned ne0 {} for {:?})",
                info.name,
                residency,
                info.dims.first().copied().unwrap_or(0),
                info.ggml_type
            )
        })?,
    })
}

#[derive(Debug, Clone)]
pub struct PlannedTensor {
    pub name: String,
    pub layer: Option<usize>,
    pub kind: Qwen35TensorKind,
    pub ggml_type: GgmlType,
    pub residency: Residency,
    pub bytes: u64,
}

#[derive(Debug, Default)]
pub struct ResidencyPlan {
    pub tensors: Vec<PlannedTensor>,
    /// Indexed by layer; `layer: None` tensors accumulate in `global_bytes`.
    pub layer_bytes: Vec<u64>,
    pub global_bytes: u64,
    pub device_bytes: u64,
}

/// Walk every GGUF tensor, classify it, and compute the device byte budget.
pub fn plan_model(gguf: &GgufFile, num_layers: usize) -> Result<ResidencyPlan> {
    let mut plan = ResidencyPlan {
        layer_bytes: vec![0; num_layers],
        ..Default::default()
    };
    for info in gguf.tensors() {
        let role = classify_qwen35_tensor(&info.name)?;
        let residency = plan_residency(role.kind, info.ggml_type);
        let bytes = device_bytes(residency, info)?;
        match role.layer {
            Some(layer) if layer < num_layers => plan.layer_bytes[layer] += bytes,
            Some(layer) => bail!("tensor {} layer {layer} >= {num_layers}", info.name),
            None => plan.global_bytes += bytes,
        }
        plan.device_bytes += bytes;
        plan.tensors.push(PlannedTensor {
            name: info.name.clone(),
            layer: role.layer,
            kind: role.kind,
            ggml_type: info.ggml_type,
            residency,
            bytes,
        });
    }
    Ok(plan)
}

/// Dequantize a single GGUF row (`n` elements at `data`) to f32, dispatching on
/// type. Covers every type the qwen35moe K-quant GGUFs use; fails loud on an
/// unsupported type (e.g. MXFP4) so the 122B's MXFP4 experts surface a clear
/// "needs a dedicated path" error instead of silently producing garbage.
pub fn dequant_row_f32(ty: GgmlType, data: &[u8], n: usize) -> Result<Vec<f32>> {
    use infer_gguf::dequant;
    match ty {
        GgmlType::F32 => dequant::dequantize_row_f32(data, n),
        GgmlType::F16 => dequant::dequantize_row_f16(data, n),
        GgmlType::Bf16 => dequant::dequantize_row_bf16(data, n),
        GgmlType::Q8_0 => dequant::dequantize_row_q8_0(data, n),
        GgmlType::Q2K => dequant::dequantize_row_q2_k(data, n),
        GgmlType::Q4K => dequant::dequantize_row_q4_k(data, n),
        GgmlType::Q5K => dequant::dequantize_row_q5_k(data, n),
        GgmlType::Q6K => dequant::dequantize_row_q6_k(data, n),
        other => {
            bail!("qwen35: no CPU dequant for {other:?} (e.g. MXFP4 experts need a dedicated path)")
        }
    }
}

/// Host-resident token-embedding table for the per-token gather "lookup".
///
/// `token_embd.weight` is `[hidden, vocab]` in GGUF order (ne0 = `hidden` is the
/// contiguous dim), so token `t`'s vector is the `t`-th `hidden`-element row.
/// Kept on the host (not uploaded) because a forward only ever gathers the few
/// rows for the current tokens — dequantizing one row is far cheaper than
/// materializing the whole (~0.5–1 GB) table on device.
pub struct HostEmbeddingTable {
    pub ggml_type: GgmlType,
    pub hidden: usize,
    pub vocab: usize,
    row_bytes: usize,
    data: Vec<u8>,
}

impl HostEmbeddingTable {
    /// Build from the raw GGUF bytes of `token_embd.weight` and its dims.
    pub fn new(ggml_type: GgmlType, hidden: usize, vocab: usize, data: Vec<u8>) -> Result<Self> {
        let row_bytes = ggml_type.row_bytes(hidden).ok_or_else(|| {
            anyhow::anyhow!("token_embd: {ggml_type:?} row of {hidden} cols is not block-aligned")
        })?;
        let need = row_bytes * vocab;
        ensure!(
            data.len() >= need,
            "token_embd: {} bytes < {need} needed ({vocab} rows x {row_bytes} B)",
            data.len()
        );
        Ok(Self {
            ggml_type,
            hidden,
            vocab,
            row_bytes,
            data,
        })
    }

    /// Gather + dequantize the embedding row for `token` → `hidden` f32 values.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        let t = token as usize;
        ensure!(t < self.vocab, "token id {t} >= vocab {}", self.vocab);
        let off = t * self.row_bytes;
        let row = &self.data[off..off + self.row_bytes];
        dequant_row_f32(self.ggml_type, row, self.hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen35TensorKind::*;
    use super::*;

    fn role(name: &str) -> Qwen35TensorRole {
        classify_qwen35_tensor(name).unwrap_or_else(|e| panic!("classify `{name}`: {e}"))
    }

    #[test]
    fn classifies_global_tensors() {
        assert_eq!(role("token_embd.weight").kind, TokenEmbedding);
        assert_eq!(role("token_embd.weight").layer, None);
        assert_eq!(role("output_norm.weight").kind, OutputNorm);
        assert_eq!(role("output.weight").kind, LmHead);
    }

    #[test]
    fn classifies_full_attention_layer() {
        for (suffix, kind) in [
            ("attn_norm.weight", AttnNorm),
            ("attn_q.weight", AttnQ),
            ("attn_k.weight", AttnK),
            ("attn_v.weight", AttnV),
            ("attn_q_norm.weight", AttnQNorm),
            ("attn_k_norm.weight", AttnKNorm),
            ("attn_output.weight", AttnOutput),
            ("attn_gate.weight", AttnGate),
            ("post_attention_norm.weight", PostAttentionNorm),
        ] {
            let r = role(&format!("blk.7.{suffix}"));
            assert_eq!(r.kind, kind, "{suffix}");
            assert_eq!(r.layer, Some(7), "{suffix}");
        }
    }

    #[test]
    fn classifies_linear_ssm_layer() {
        for (suffix, kind) in [
            ("attn_qkv.weight", AttnQkv),
            ("ssm_conv1d.weight", SsmConv1d),
            ("ssm_a", SsmA),
            ("ssm_alpha.weight", SsmAlpha),
            ("ssm_beta.weight", SsmBeta),
            ("ssm_dt.bias", SsmDtBias),
            ("ssm_norm.weight", SsmNorm),
            ("ssm_out.weight", SsmOut),
        ] {
            let r = role(&format!("blk.0.{suffix}"));
            assert_eq!(r.kind, kind, "{suffix}");
            assert_eq!(r.layer, Some(0), "{suffix}");
        }
    }

    #[test]
    fn classifies_moe_ffn_with_shared_expert() {
        // The `_shexp` shared-expert names must NOT collide with the routed ones.
        assert_eq!(role("blk.3.ffn_gate_inp.weight").kind, FfnGateInp);
        assert_eq!(role("blk.3.ffn_gate_exps.weight").kind, FfnGateExps);
        assert_eq!(role("blk.3.ffn_up_exps.weight").kind, FfnUpExps);
        assert_eq!(role("blk.3.ffn_down_exps.weight").kind, FfnDownExps);
        assert_eq!(
            role("blk.3.ffn_gate_inp_shexp.weight").kind,
            FfnGateInpShexp
        );
        assert_eq!(role("blk.3.ffn_gate_shexp.weight").kind, FfnGateShexp);
        assert_eq!(role("blk.3.ffn_up_shexp.weight").kind, FfnUpShexp);
        assert_eq!(role("blk.3.ffn_down_shexp.weight").kind, FfnDownShexp);
        assert!(role("blk.3.ffn_gate_exps.weight").kind.is_routed_expert());
        assert!(!role("blk.3.ffn_gate_shexp.weight").kind.is_routed_expert());
    }

    #[test]
    fn unknown_tensor_fails_loud() {
        assert!(classify_qwen35_tensor("blk.0.mystery.weight").is_err());
        assert!(classify_qwen35_tensor("totally_unknown").is_err());
    }

    #[test]
    fn residency_policy_matches_kernel_surface() {
        use GgmlType::*;
        // token_embd → host row-gather regardless of its (Q8_0) type.
        assert_eq!(
            plan_residency(TokenEmbedding, Q8_0),
            Residency::HostEmbedding
        );
        // K-quants stay packed for the GEMV.
        assert_eq!(
            plan_residency(FfnGateExps, Q4K),
            Residency::KeepKQuant(KQuant::Q4K)
        );
        assert_eq!(
            plan_residency(FfnDownExps, Q5K),
            Residency::KeepKQuant(KQuant::Q5K)
        );
        assert_eq!(
            plan_residency(LmHead, Q6K),
            Residency::KeepKQuant(KQuant::Q6K)
        );
        // Q8_0 attention/projection weights → F16.
        assert_eq!(plan_residency(AttnQ, Q8_0), Residency::DequantF16);
        assert_eq!(plan_residency(SsmOut, Q8_0), Residency::DequantF16);
        // F32 norms / SSM params / router → F32.
        assert_eq!(plan_residency(AttnNorm, F32), Residency::DequantF32);
        assert_eq!(plan_residency(SsmA, F32), Residency::DequantF32);
        assert_eq!(plan_residency(FfnGateInp, F32), Residency::DequantF32);
    }

    #[test]
    fn embedding_lookup_gathers_correct_row() {
        // vocab=3, hidden=4, F32. Row t = [t*10+0 .. t*10+3].
        let (hidden, vocab) = (4usize, 3usize);
        let mut bytes = Vec::new();
        for t in 0..vocab {
            for i in 0..hidden {
                bytes.extend_from_slice(&((t * 10 + i) as f32).to_le_bytes());
            }
        }
        let tbl = HostEmbeddingTable::new(GgmlType::F32, hidden, vocab, bytes).unwrap();
        assert_eq!(tbl.embed_row(0).unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(tbl.embed_row(2).unwrap(), vec![20.0, 21.0, 22.0, 23.0]);
        assert!(tbl.embed_row(3).is_err(), "out-of-range token must fail");
    }

    #[test]
    fn dequant_row_rejects_mxfp4() {
        assert!(dequant_row_f32(GgmlType::Mxfp4, &[0u8; 64], 32).is_err());
    }
}
