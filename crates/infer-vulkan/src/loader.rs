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
//! - Q8_0 weights ALSO stay packed: `Kernel::GemvQ8_0` (`q8_0_gemv_with_params`)
//!   reads the raw 34 B/32 blocks directly. Dequantizing them to F16 would
//!   double device bytes and decode bandwidth versus llama.cpp, so they ride
//!   the same packed tier as the K-quants.
//! - F16/BF16 weights dequantize to F16 (coopmat-FP16 GEMM consumer).
//! - F32 tensors (norms, SSM params, router) stay F32 on device.
//! - `token_embd.weight` stays as raw GGUF bytes on the host so the embedding
//!   "lookup" gathers + dequantizes a single row per token instead of
//!   uploading the whole (~0.5–1 GB) table.

use anyhow::{Result, bail, ensure};

use infer_gguf::gguf::{GgmlType, GgufFile, TensorInfo};

/// Per-tensor device-format decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Q4_K/Q5_K/Q6_K/Q8_0 uploaded as raw GGUF bytes; consumed directly by the
    /// matching quantized GEMV (`mul_mat_vecq` for K-quants, `q8_0_gemv` for
    /// Q8_0). The held [`GgmlType`] selects the device decode path.
    KeepQuant(GgmlType),
    /// F16/BF16 dequantized to F16 on device.
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
    // --- dense FFN (Qwen3.5 dense, e.g. the 27B) ---
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
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
                | Self::FfnNorm
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35TensorRole {
    pub kind: Qwen35TensorKind,
    /// `Some(layer)` for `blk.N.*`, `None` for global tensors.
    pub layer: Option<usize>,
}

/// A tensor belonging to a trailing MTP block — any `blk.<N>` at or past the
/// decode graph's last layer. Qwen3.8-27B's blk.64 is a *complete* transformer
/// block (attn_q, ffn_down, …) that additionally carries the four `nextn.*`
/// weights, so matching the `nextn.` suffix alone leaves the rest behind.
/// Speculative decode is not wired on Vulkan, so the whole block is dropped
/// rather than spending device bytes nothing dispatches against.
pub fn is_mtp_tensor(name: &str, num_layers: usize) -> bool {
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(index, _)| index.parse::<usize>().ok())
        .is_some_and(|layer| layer >= num_layers)
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
        // dense FFN (Qwen3.5 dense)
        "ffn_norm.weight" => FfnNorm,
        "ffn_gate.weight" => FfnGate,
        "ffn_up.weight" => FfnUp,
        "ffn_down.weight" => FfnDown,
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
        // Quant tiers with a registered device GEMV stay packed (raw bytes):
        // K-quants via `mul_mat_vecq`, Q8_0 via `q8_0_gemv`.
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0 => Residency::KeepQuant(ty),
        GgmlType::F32 => Residency::DequantF32,
        // F16 / BF16 (and anything else dequantizable) → F16 on device.
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
        Residency::KeepQuant(_) => info.byte_len().ok_or_else(|| {
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
        if is_mtp_tensor(&info.name, num_layers) {
            continue;
        }
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

/// Device residency upload: allocate one [`vulkan_sys::DeviceBuffer`] per
/// planned tensor and fill it per residency tier (mirrors the DSv4 HIP
/// `upload` module, but lands F16 not BF16 and targets `vulkan_sys`).
///
/// `token_embd.weight` is the one exception — it never gets a device buffer;
/// its raw GGUF bytes go into the host-resident [`HostEmbeddingTable`] for the
/// per-token row gather.
#[cfg(feature = "vulkan")]
pub mod upload {
    use anyhow::{Context, Result, anyhow, bail, ensure};

    use super::{
        GgufFile, HostEmbeddingTable, Qwen35TensorKind, Residency, ResidencyPlan, dequant_row_f32,
    };

    /// Convert one f32 to an IEEE-754 binary16 (half) bit pattern with
    /// round-to-nearest-even. Handles inf/NaN passthrough, subnormal halves,
    /// and overflow-to-inf.
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

        // Unbias f32 exponent, rebias to f16 (bias 15).
        let unbiased = exp - 127;
        let half_exp = unbiased + 15;

        if half_exp >= 0x1f {
            // Overflow → inf.
            return sign | 0x7c00;
        }
        if half_exp <= 0 {
            // Subnormal half or underflow to zero. Build the 24-bit
            // significand (implicit leading 1 for normal f32 inputs) and
            // shift it into the subnormal range with round-to-nearest-even.
            if half_exp < -10 {
                // Too small even for the smallest subnormal → signed zero.
                return sign;
            }
            let significand = mant | 0x0080_0000; // implicit 1.fraction
            // Right shift so the value aligns to the 10-bit half mantissa.
            let shift = (14 - half_exp) as u32; // in [14, 24]
            let half_mant = significand >> shift;
            // Round-to-nearest-even on the discarded low bits.
            let remainder = significand & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            let mut result = half_mant;
            if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
                result += 1; // may carry into the exponent field — that's fine
            }
            return sign | (result as u16);
        }

        // Normal half. Round the 23-bit mantissa down to 10 bits, RNE.
        let half_mant = mant >> 13;
        let remainder = mant & 0x1fff;
        let halfway = 0x1000u32;
        let mut result = ((half_exp as u32) << 10) | half_mant;
        if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
            result += 1; // carries mantissa→exponent correctly; exp 0x1e→0x1f = inf
        }
        sign | (result as u16)
    }

    /// One device-resident weight tensor plus its provenance.
    pub struct DeviceTensor<'a> {
        pub name: String,
        pub kind: Qwen35TensorKind,
        pub layer: Option<usize>,
        pub residency: Residency,
        pub buffer: vulkan_sys::DeviceBuffer<'a>,
        /// `(ncols=in=ne0, nrows=out=ne1)` for 2-D weight tensors; `(len=ne0,
        /// 1)` for 1-D norm/param tensors. Recorded at upload so the GEMV path
        /// can derive its `[nrows, ncols]` contract — and the norm read-backs
        /// their length — without re-reading the GGUF. `None` only if a tensor
        /// somehow had no dims.
        pub gemv_dims: Option<(usize, usize)>,
        /// For `DequantF32` tensors: the dequantized values kept in CACHED host
        /// RAM. The device buffer for these is HOST_VISIBLE (write-combined), so
        /// reading it back with `copy_to_host` is pathologically slow (~50-300
        /// MB/s uncached CPU reads). The forward re-reads some of these EVERY
        /// layer EVERY token (notably the MoE router `ffn_gate_inp`, 40×/token),
        /// so caching the host copy at upload — we already computed it to fill
        /// the buffer — turns those read-backs into cached-RAM reads (~GB/s).
        /// `None` for non-F32 residencies (their buffers are device-local and
        /// never host-read). See `dequant_f32` / `host_f32_values` in forward.rs.
        pub host_f32: Option<Vec<f32>>,
    }

    /// All device-resident weights for a model plus the host embedding table.
    pub struct ResidentWeights<'a> {
        pub tensors: std::collections::HashMap<String, DeviceTensor<'a>>,
        pub embedding: HostEmbeddingTable,
    }

    impl<'a> ResidentWeights<'a> {
        pub fn get(&self, name: &str) -> Option<&DeviceTensor<'a>> {
            self.tensors.get(name)
        }
    }

    /// Walk the residency plan, uploading each tensor onto `ctx`'s device per
    /// its tier and stashing `token_embd.weight` host-side. Exactly one
    /// `HostEmbedding` tensor (the token embedding) is expected.
    pub fn upload_plan<'a>(
        ctx: &'a vulkan_sys::VulkanContext,
        gguf: &GgufFile,
        plan: &ResidencyPlan,
    ) -> Result<ResidentWeights<'a>> {
        let mut tensors = std::collections::HashMap::new();
        let mut embedding: Option<HostEmbeddingTable> = None;

        for t in &plan.tensors {
            // The host-resident token embedding: never gets a device buffer.
            if t.residency == Residency::HostEmbedding {
                let info = gguf
                    .tensor(&t.name)
                    .ok_or_else(|| anyhow!("token_embd {} vanished from GGUF", t.name))?;
                let hidden = usize::try_from(info.dims.first().copied().unwrap_or(0))
                    .map_err(|_| anyhow!("{}: hidden dim overflow", t.name))?;
                let vocab = usize::try_from(info.dims.get(1).copied().unwrap_or(0))
                    .map_err(|_| anyhow!("{}: vocab dim overflow", t.name))?;
                let raw = gguf.tensor_data(&t.name)?.to_vec();
                let table = HostEmbeddingTable::new(t.ggml_type, hidden, vocab, raw)
                    .with_context(|| format!("building host embedding table from {}", t.name))?;
                ensure!(
                    embedding.replace(table).is_none(),
                    "more than one HostEmbedding tensor in plan (second: {})",
                    t.name
                );
                continue;
            }

            let src = gguf.tensor_data(&t.name)?;
            let n = gguf
                .tensor(&t.name)
                .map(|i| i.element_count() as usize)
                .unwrap_or(0);
            // Build the exact device bytes per residency tier. For DequantF32,
            // we ALSO keep the dequantized f32 Vec (`host_f32`) in cached RAM so
            // the forward never reads it back from the write-combined device
            // buffer (see the field doc on `DeviceTensor::host_f32`).
            let mut host_f32: Option<Vec<f32>> = None;
            let host_bytes: std::borrow::Cow<'_, [u8]> = match t.residency {
                Residency::KeepQuant(_) => std::borrow::Cow::Borrowed(src),
                Residency::DequantF16 => {
                    let f = dequant_row_f32(t.ggml_type, src, n).context(t.name.clone())?;
                    std::borrow::Cow::Owned(
                        f.iter()
                            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                            .collect(),
                    )
                }
                Residency::DequantF32 => {
                    let f = dequant_row_f32(t.ggml_type, src, n).context(t.name.clone())?;
                    let bytes: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
                    host_f32 = Some(f);
                    std::borrow::Cow::Owned(bytes)
                }
                Residency::HostEmbedding => unreachable!("handled above"),
            };
            // F32 norm/SSM-param tensors are read BACK to the host during the
            // forward (`copy_to_host` needs a mappable, host-visible buffer), so
            // they stay HOST_VISIBLE. The large read-only GPU-only weights
            // (KeepQuant / DequantF16) go to DEVICE_LOCAL via a staged upload —
            // routing tens of GB through the smaller host-visible heap is what
            // exhausted it and broke per-dispatch submits on the APU.
            let buffer = if matches!(t.residency, Residency::DequantF32) {
                let mut b = vulkan_sys::DeviceBuffer::alloc(ctx, t.bytes as usize)
                    .map_err(|e| anyhow!("alloc {} ({} B): {e}", t.name, t.bytes))?;
                b.copy_from_host(&host_bytes)
                    .map_err(|e| anyhow!("upload {}: {e}", t.name))?;
                b
            } else {
                vulkan_sys::DeviceBuffer::alloc_device_local_from_host(ctx, &host_bytes)
                    .map_err(|e| anyhow!("staged upload {} ({} B): {e}", t.name, t.bytes))?
            };
            // Record (ncols=in=ne0, nrows=out=ne1) for the GEMV path; 1-D
            // tensors carry (ne0, 1) so norm read-backs know their length.
            let gemv_dims = gguf.tensor(&t.name).map(|info| {
                let ne0 = info.dims.first().copied().unwrap_or(0) as usize;
                let ne1 = info.dims.get(1).copied().map(|v| v as usize).unwrap_or(1);
                (ne0, ne1)
            });
            tensors.insert(
                t.name.clone(),
                DeviceTensor {
                    name: t.name.clone(),
                    kind: t.kind,
                    layer: t.layer,
                    residency: t.residency,
                    buffer,
                    gemv_dims,
                    host_f32,
                },
            );
        }

        let Some(embedding) = embedding else {
            bail!("no HostEmbedding (token_embd.weight) tensor found in plan");
        };
        Ok(ResidentWeights { tensors, embedding })
    }

    /// Minimal synthetic-GGUF writer for the device test below.
    ///
    /// This used to live in `infer_gguf::gguf::test_writer`, `pub` purely so
    /// cross-crate test builds could reach it. `a7c9ee395` ("remove inline unit
    /// tests") deleted it together with its in-crate callers and missed this
    /// one, leaving `infer-vulkan`'s whole test target uncompilable. Since this
    /// is now the only consumer, it lives here under `#[cfg(test)]` instead of
    /// going back to being a `pub` item in a shipping crate.
    #[cfg(test)]
    mod test_writer {
        /// GGUF's default `general.alignment`; tensor data is padded to it.
        const ALIGNMENT: usize = 32;

        pub enum V {
            Str(&'static str),
        }

        pub struct T {
            pub name: String,
            pub dims: Vec<u64>,
            pub type_id: u32,
            pub data: Vec<u8>,
        }

        fn put_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        /// GGUF v3: magic, version, tensor count, kv count, the kv block, the
        /// tensor-info block (each info's `offset` is relative to the start of
        /// the aligned data section), then the padded data section.
        pub fn write(kvs: &[(&str, V)], tensors: &[T]) -> Vec<u8> {
            let mut buf = Vec::new();
            buf.extend_from_slice(b"GGUF");
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
            for (key, value) in kvs {
                put_str(&mut buf, key);
                match value {
                    V::Str(v) => {
                        buf.extend_from_slice(&8u32.to_le_bytes());
                        put_str(&mut buf, v);
                    }
                }
            }
            let mut offset = 0usize;
            for t in tensors {
                put_str(&mut buf, &t.name);
                buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
                for d in &t.dims {
                    buf.extend_from_slice(&d.to_le_bytes());
                }
                buf.extend_from_slice(&t.type_id.to_le_bytes());
                buf.extend_from_slice(&(offset as u64).to_le_bytes());
                offset += t.data.len().div_ceil(ALIGNMENT) * ALIGNMENT;
            }
            while !buf.len().is_multiple_of(ALIGNMENT) {
                buf.push(0);
            }
            for t in tensors {
                buf.extend_from_slice(&t.data);
                while !buf.len().is_multiple_of(ALIGNMENT) {
                    buf.push(0);
                }
            }
            buf
        }

        pub fn write_to_temp(kvs: &[(&str, V)], tensors: &[T], tag: &str) -> std::path::PathBuf {
            let path = std::env::temp_dir().join(format!(
                "arle-infer-vulkan-{tag}-{}.gguf",
                std::process::id()
            ));
            std::fs::write(&path, write(kvs, tensors)).unwrap();
            path
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn f16_pinned() {
            assert_eq!(f32_to_f16(1.0), 0x3C00);
            assert_eq!(f32_to_f16(-2.0), 0xC000);
            assert_eq!(f32_to_f16(0.0), 0x0000);
            assert_eq!(f32_to_f16(-0.0), 0x8000);
            assert_eq!(f32_to_f16(65504.0), 0x7BFF); // largest finite half
            // 2049 = 0x4000_0800: not exactly representable (step is 2 at this
            // magnitude); RNE → 2048.0 = 0x6800.
            assert_eq!(f32_to_f16(2049.0), 0x6800);
            // 2051 rounds up to 2052.0 = 0x6802.
            assert_eq!(f32_to_f16(2051.0), 0x6802);
            // Overflow above the half range → +inf.
            assert_eq!(f32_to_f16(f32::MAX), 0x7C00);
            assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
            assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
            // NaN stays NaN (exp all ones, nonzero mantissa).
            let nan = f32_to_f16(f32::NAN);
            assert_eq!(nan & 0x7C00, 0x7C00);
            assert_ne!(nan & 0x03FF, 0);
            // Smallest normal/subnormal halves round-trip sanity.
            assert_eq!(f32_to_f16(6.103515625e-5), 0x0400); // 2^-14 smallest normal
            assert_eq!(f32_to_f16(5.9604645e-8), 0x0001); // smallest subnormal
        }

        #[cfg(feature = "vulkan")]
        #[test]
        fn upload_plan_lands_bytes_on_device() {
            use super::test_writer::{T, V, write_to_temp};
            use crate::loader::plan_model;
            use infer_gguf::gguf::GgufFile;
            use vulkan_sys::VulkanContext;

            let ctx = match VulkanContext::create() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("no Vulkan device available ({e}); skipping upload_plan device test");
                    return;
                }
            };
            eprintln!("ARLE Vulkan upload_plan device: {}", ctx.device_name());

            // Synthesize a tiny qwen35moe GGUF:
            //   token_embd.weight  F32  [hidden=8, vocab=4]      -> HostEmbedding
            //   blk.0.attn_norm.weight F32 [8]                   -> DequantF32
            //   blk.0.attn_q.weight    Q8_0 [32]  (34 B/32)      -> KeepQuant(Q8_0)
            //   blk.0.ffn_gate_exps.weight Q4_K [256] (144 B/256)-> KeepQuant(Q4K)
            let hidden = 8u64;
            let vocab = 4u64;
            let embd: Vec<u8> = (0..(hidden * vocab))
                .flat_map(|v| (v as f32).to_le_bytes())
                .collect();
            let norm: Vec<u8> = (0..hidden).flat_map(|v| (v as f32).to_le_bytes()).collect();
            // Q8_0: 34 B per 32-element block; one block. Distinct bytes so the
            // raw-vs-dequant path is exercised (we only assert sizes for it).
            let q8_0: Vec<u8> = (0..34u8).collect();
            // Q4_K: 144 B per 256-element block; one block, distinct bytes.
            let q4_k: Vec<u8> = (0..144u32).map(|i| (i % 251) as u8).collect();

            let path = write_to_temp(
                &[("general.architecture", V::Str("qwen35moe"))],
                &[
                    T {
                        name: "token_embd.weight".into(),
                        dims: vec![hidden, vocab],
                        type_id: 0, // F32
                        data: embd,
                    },
                    T {
                        name: "blk.0.attn_norm.weight".into(),
                        dims: vec![hidden],
                        type_id: 0, // F32
                        data: norm,
                    },
                    T {
                        name: "blk.0.attn_q.weight".into(),
                        dims: vec![32],
                        type_id: 8, // Q8_0
                        data: q8_0.clone(),
                    },
                    T {
                        name: "blk.0.ffn_gate_exps.weight".into(),
                        dims: vec![256],
                        type_id: 12, // Q4_K
                        data: q4_k.clone(),
                    },
                ],
                "vulkan-upload",
            );

            let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
            let plan = plan_model(&gguf, 1).expect("plan_model");
            let resident = upload_plan(&ctx, &gguf, &plan).expect("upload_plan");

            // token_embd is NOT in the device tensor map (it's host-resident).
            assert!(
                resident.get("token_embd.weight").is_none(),
                "token_embd must not be a device tensor"
            );
            assert_eq!(
                resident.tensors.len(),
                3,
                "three device tensors expected (norm, attn_q, ffn_gate_exps)"
            );

            // Each device buffer is exactly its planned byte count.
            let by_name = |n: &str| plan.tensors.iter().find(|t| t.name == n).unwrap();
            for name in [
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_gate_exps.weight",
            ] {
                let dt = resident
                    .get(name)
                    .unwrap_or_else(|| panic!("missing {name}"));
                assert_eq!(
                    dt.buffer.len() as u64,
                    by_name(name).bytes,
                    "{name} buffer size != planned bytes"
                );
            }

            // KeepQuant Q4_K round-trips byte-for-byte (raw quant bytes uploaded).
            //
            // Read back STAGED: resident weights live in DEVICE_LOCAL-only
            // memory, so `copy_to_host` — which maps the buffer itself — fails
            // with `ERROR_MEMORY_MAP_FAILED` on them.
            let q4 = resident.get("blk.0.ffn_gate_exps.weight").unwrap();
            assert!(matches!(
                q4.residency,
                Residency::KeepQuant(infer_gguf::gguf::GgmlType::Q4K)
            ));
            let mut back = vec![0u8; q4.buffer.len()];
            q4.buffer.copy_to_host_staged(&mut back).expect("D2H Q4_K");
            assert_eq!(back, q4_k, "Q4_K device round-trip mismatch");

            // Q8_0 now stays PACKED (KeepQuant), not dequantized to F16: the raw
            // 34 B block round-trips byte-for-byte (it would be 64 B of F16 if it
            // had been dequantized — proving the packed Q8_0 tier holds).
            let q8 = resident.get("blk.0.attn_q.weight").unwrap();
            assert!(matches!(
                q8.residency,
                Residency::KeepQuant(infer_gguf::gguf::GgmlType::Q8_0)
            ));
            assert_eq!(
                q8.buffer.len(),
                34,
                "Q8_0 kept packed at 34 B (not 64 B F16)"
            );
            let mut q8_back = vec![0u8; q8.buffer.len()];
            q8.buffer
                .copy_to_host_staged(&mut q8_back)
                .expect("D2H Q8_0");
            assert_eq!(q8_back, q8_0, "Q8_0 device round-trip mismatch");

            // Embedding row 0 gathers `hidden` f32 values.
            let row0 = resident.embedding.embed_row(0).expect("embed_row(0)");
            assert_eq!(row0.len(), hidden as usize);
            assert_eq!(row0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

            std::fs::remove_file(path).ok();
            eprintln!(
                "PASS: upload_plan landed {} device tensors on {}",
                resident.tensors.len(),
                ctx.device_name()
            );
        }
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
        assert_eq!(plan_residency(FfnGateExps, Q4K), Residency::KeepQuant(Q4K));
        assert_eq!(plan_residency(FfnDownExps, Q5K), Residency::KeepQuant(Q5K));
        assert_eq!(plan_residency(LmHead, Q6K), Residency::KeepQuant(Q6K));
        // Q8_0 attention/projection weights now ALSO stay packed (q8_0_gemv
        // reads them directly) — NOT dequantized to F16.
        assert_eq!(plan_residency(AttnQ, Q8_0), Residency::KeepQuant(Q8_0));
        assert_eq!(plan_residency(SsmOut, Q8_0), Residency::KeepQuant(Q8_0));
        // F32 norms / SSM params / router → F32.
        assert_eq!(plan_residency(AttnNorm, F32), Residency::DequantF32);
        assert_eq!(plan_residency(SsmA, F32), Residency::DequantF32);
        assert_eq!(plan_residency(FfnGateInp, F32), Residency::DequantF32);
    }

    #[test]
    fn embedding_lookup_gathers_correct_row() {
        // vocab=3, hidden=4, F32. Row t = [t*10+0 .. t*10+3].
        let (hidden, vocab) = (4usize, 3usize);
        let bytes: Vec<u8> = (0..vocab)
            .flat_map(|t| (0..hidden).flat_map(move |i| ((t * 10 + i) as f32).to_le_bytes()))
            .collect();
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
