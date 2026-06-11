//! Residency planner: per-tensor device-format decision + exact byte
//! budget for the DSv4 2-bit lane, and the `hip`-gated upload walk.
//!
//! Keep-quantized tiers exist ONLY for formats the `hip-kernels` GEMV
//! surface serves (crates/hip-kernels/src/lib.rs): `arle_mmvq_iq2_xxs_cuda`
//! / `arle_mmvq_q2_k_cuda` (csrc/iq2_mmvq.cu) and the q4k/q5k/q6k
//! gemv/dequant/embedding launchers (csrc/gemm/quantized_gemv.cu — q3k
//! also compiles but has no Rust decl, so Q3_K dequantizes). Everything
//! else lands BF16 — every device consumer of the non-matmul tensors
//! (`dsv4_compressor_update` ape/norm, `dsv4_{swa,hybrid}_attention`
//! attn_sink, `dsv4_mhc_params`/`_head_pre` base/scale) takes bf16 per the
//! dsv4_attention.cu/dsv4_mhc.cu signatures. Host-consumed routing data
//! (`exp_probs_b` bias, integer `tid2eid` hash table) stays host-only;
//! the model reads it straight from the GGUF at load.

use anyhow::{Result, bail};

use crate::config::{Dsv4TensorKind, classify_tensor};
use crate::gguf::{GgmlType, GgufFile, TensorInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuant {
    Q4K,
    Q5K,
    Q6K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    KeepIq2Xxs,
    KeepQ2K,
    KeepKQuant(KQuant),
    DequantBf16,
    DequantF32,
    /// Integer routing table (`ffn_gate_tid2eid`) — read on host, 0 device bytes.
    HostOnly,
}

/// f32 → bf16, round-to-nearest-even; NaN keeps a set mantissa bit.
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7FFF + lsb) >> 16) as u16
}

pub fn plan_residency(kind: Dsv4TensorKind, ty: GgmlType) -> Residency {
    // Host-routed data: the executor routes on the CPU, so the bias and
    // hash table never need device bytes.
    if matches!(
        kind,
        Dsv4TensorKind::RouterHashTable | Dsv4TensorKind::RouterBias
    ) {
        return Residency::HostOnly;
    }
    match ty {
        GgmlType::Iq2Xxs => Residency::KeepIq2Xxs,
        GgmlType::Q2K => Residency::KeepQ2K,
        GgmlType::Q4K => Residency::KeepKQuant(KQuant::Q4K),
        GgmlType::Q5K => Residency::KeepKQuant(KQuant::Q5K),
        GgmlType::Q6K => Residency::KeepKQuant(KQuant::Q6K),
        // Everything else is bf16: norms, sinks, APE tables, HC base/scale
        // all enter the dsv4_attention.cu / dsv4_mhc.cu launchers as bf16.
        _ => Residency::DequantBf16,
    }
}

/// Exact device bytes for one tensor under `residency`.
pub fn device_bytes(residency: Residency, info: &TensorInfo) -> Result<u64> {
    let n = info.element_count();
    Ok(match residency {
        Residency::HostOnly => 0,
        Residency::DequantBf16 => n * 2,
        Residency::DequantF32 => n * 4,
        Residency::KeepIq2Xxs | Residency::KeepQ2K | Residency::KeepKQuant(_) => {
            let Some(len) = info.byte_len() else {
                bail!(
                    "tensor {}: cannot keep {:?} (unaligned ne0 {} for {:?})",
                    info.name,
                    residency,
                    info.dims.first().copied().unwrap_or(0),
                    info.ggml_type
                );
            };
            len
        }
    })
}

#[derive(Debug, Clone)]
pub struct PlannedTensor {
    pub name: String,
    pub layer: Option<usize>,
    pub kind: Dsv4TensorKind,
    pub ggml_type: GgmlType,
    pub residency: Residency,
    pub bytes: u64,
}

#[derive(Debug, Default)]
pub struct ResidencyPlan {
    pub tensors: Vec<PlannedTensor>,
    /// Indexed by layer; tensors with `layer: None` accumulate in `global_bytes`.
    pub layer_bytes: Vec<u64>,
    pub global_bytes: u64,
    pub total_bytes: u64,
}

pub fn plan_model(gguf: &GgufFile, num_layers: usize) -> Result<ResidencyPlan> {
    let mut plan = ResidencyPlan {
        layer_bytes: vec![0; num_layers],
        ..Default::default()
    };
    for info in gguf.tensors() {
        let role = classify_tensor(&info.name)?;
        let residency = plan_residency(role.kind, info.ggml_type);
        let bytes = device_bytes(residency, info)?;
        match role.layer {
            Some(layer) if layer < num_layers => plan.layer_bytes[layer] += bytes,
            Some(layer) => bail!("tensor {} layer {layer} >= {num_layers}", info.name),
            None => plan.global_bytes += bytes,
        }
        plan.total_bytes += bytes;
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

/// Upload walk — allocates one device buffer per planned tensor and fills
/// it per residency tier. Compiles on any box; real run = pending-remote
/// (needs ROCm box, #77 on-box phase).
#[cfg(feature = "hip")]
pub mod upload {
    use anyhow::{Context, Result, anyhow};
    use hip_sys::DeviceBuffer;

    use super::{Residency, ResidencyPlan, f32_to_bf16_rne};
    use crate::dequant;
    use crate::gguf::{GgmlType, GgufFile};

    pub struct DeviceTensor {
        pub name: String,
        pub residency: Residency,
        pub buffer: DeviceBuffer,
    }

    fn dequant_f32(ty: GgmlType, data: &[u8], n: usize) -> Result<Vec<f32>> {
        match ty {
            GgmlType::F32 => dequant::dequantize_row_f32(data, n),
            GgmlType::F16 => dequant::dequantize_row_f16(data, n),
            GgmlType::Bf16 => dequant::dequantize_row_bf16(data, n),
            GgmlType::Q8_0 => dequant::dequantize_row_q8_0(data, n),
            GgmlType::Q2K => dequant::dequantize_row_q2_k(data, n),
            GgmlType::Q4K => dequant::dequantize_row_q4_k(data, n),
            GgmlType::Q5K => dequant::dequantize_row_q5_k(data, n),
            GgmlType::Q6K => dequant::dequantize_row_q6_k(data, n),
            other => Err(anyhow!("no CPU dequant for {other:?}")),
        }
    }

    pub fn upload_plan(gguf: &GgufFile, plan: &ResidencyPlan) -> Result<Vec<DeviceTensor>> {
        let mut out = Vec::with_capacity(plan.tensors.len());
        for t in &plan.tensors {
            if t.residency == Residency::HostOnly {
                continue;
            }
            let src = gguf.tensor_data(&t.name)?;
            let n = gguf
                .tensor(&t.name)
                .map(|i| i.element_count() as usize)
                .unwrap_or(0);
            let mut buffer = DeviceBuffer::alloc(t.bytes as usize)
                .map_err(|e| anyhow!("alloc {} ({} B): {e}", t.name, t.bytes))?;
            match t.residency {
                Residency::KeepIq2Xxs | Residency::KeepQ2K | Residency::KeepKQuant(_) => {
                    buffer
                        .copy_from_host(src)
                        .map_err(|e| anyhow!("upload {}: {e}", t.name))?;
                }
                Residency::DequantF32 => {
                    let f = dequant_f32(t.ggml_type, src, n).context(t.name.clone())?;
                    let bytes: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
                    buffer
                        .copy_from_host(&bytes)
                        .map_err(|e| anyhow!("upload {}: {e}", t.name))?;
                }
                Residency::DequantBf16 => {
                    let f = dequant_f32(t.ggml_type, src, n).context(t.name.clone())?;
                    let bytes: Vec<u8> = f
                        .iter()
                        .flat_map(|&v| f32_to_bf16_rne(v).to_le_bytes())
                        .collect();
                    buffer
                        .copy_from_host(&bytes)
                        .map_err(|e| anyhow!("upload {}: {e}", t.name))?;
                }
                Residency::HostOnly => unreachable!("filtered above"),
            }
            out.push(DeviceTensor {
                name: t.name.clone(),
                residency: t.residency,
                buffer,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_writer::{T, V, write_to_temp};

    #[test]
    fn bf16_rne_pinned() {
        assert_eq!(f32_to_bf16_rne(1.0), 0x3F80);
        assert_eq!(f32_to_bf16_rne(-2.0), 0xC000);
        assert_eq!(f32_to_bf16_rne(0.0), 0x0000);
        // Halfway, even target stays: 0x3F80_8000 → 0x3F80.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_8000)), 0x3F80);
        // Halfway, odd target rounds up: 0x3F81_8000 → 0x3F82.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F81_8000)), 0x3F82);
        // Above halfway rounds up.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_8001)), 0x3F81);
        // f32::MAX rounds to bf16 inf.
        assert_eq!(f32_to_bf16_rne(f32::MAX), 0x7F80);
        assert_eq!(f32_to_bf16_rne(f32::INFINITY), 0x7F80);
        let nan = f32_to_bf16_rne(f32::NAN);
        assert_eq!(nan & 0x7F80, 0x7F80);
        assert_ne!(nan & 0x7F, 0);
    }

    #[test]
    fn policy_table_matches_kernel_surface() {
        use Dsv4TensorKind::*;
        use GgmlType::*;
        let cases = [
            (RoutedExpertUp, Iq2Xxs, Residency::KeepIq2Xxs),
            (RoutedExpertGate, Iq2Xxs, Residency::KeepIq2Xxs),
            (RoutedExpertDown, Q2K, Residency::KeepQ2K),
            (AttnQA, Q4K, Residency::KeepKQuant(KQuant::Q4K)),
            (SharedExpertDown, Q5K, Residency::KeepKQuant(KQuant::Q5K)),
            (OutputHead, Q6K, Residency::KeepKQuant(KQuant::Q6K)),
            (TokenEmbedding, Q4K, Residency::KeepKQuant(KQuant::Q4K)),
            // Q3_K has no Rust launcher decl in hip-kernels → dequant.
            (AttnQB, Q3K, Residency::DequantBf16),
            (AttnNorm, F32, Residency::DequantBf16),
            (OutputNorm, F32, Residency::DequantBf16),
            (AttnKvLatent, F16, Residency::DequantBf16),
            (RouterGate, F32, Residency::DequantBf16),
            // Device consumers take these as bf16 (dsv4_attention.cu sink/ape,
            // dsv4_mhc.cu base/scale).
            (AttnSink, F32, Residency::DequantBf16),
            (HyperConnection, F32, Residency::DequantBf16),
            (HeadHyperConnection, F32, Residency::DequantBf16),
            (CompressorApe, F32, Residency::DequantBf16),
            (CompressorNorm, F32, Residency::DequantBf16),
            // Host routing data — never uploaded.
            (RouterBias, F32, Residency::HostOnly),
            (RouterHashTable, I64, Residency::HostOnly),
        ];
        for (kind, ty, expect) in cases {
            assert_eq!(plan_residency(kind, ty), expect, "{kind:?} {ty:?}");
        }
    }

    /// Synthetic table mimicking the ds4 q2-imatrix recipe: routed up/gate
    /// IQ2_XXS, routed down Q2_K, attn + shared experts Q4_K, norms F32.
    #[test]
    fn plan_bytes_match_hand_computed() {
        let t = |name: &str, dims: Vec<u64>, type_id: u32, block_bytes: usize, block: usize| {
            let n: u64 = dims.iter().product();
            let bytes = n as usize / block * block_bytes;
            T {
                name: name.into(),
                dims,
                type_id,
                data: vec![0u8; bytes],
            }
        };
        let path = write_to_temp(
            &[("general.architecture", V::Str("deepseek4"))],
            &[
                // 16 experts, hidden 512, intermediate 256.
                t("blk.0.ffn_up_exps.weight", vec![512, 256, 16], 16, 66, 256),
                t(
                    "blk.0.ffn_gate_exps.weight",
                    vec![512, 256, 16],
                    16,
                    66,
                    256,
                ),
                t(
                    "blk.0.ffn_down_exps.weight",
                    vec![256, 512, 16],
                    10,
                    84,
                    256,
                ),
                t("blk.0.attn_q_a.weight", vec![512, 256], 12, 144, 256),
                t("blk.0.ffn_up_shexp.weight", vec![512, 256], 12, 144, 256),
                t("blk.0.attn_norm.weight", vec![512], 0, 4, 1),
                t("blk.0.attn_sinks", vec![8], 0, 4, 1),
                t("blk.0.ffn_gate_tid2eid", vec![2, 100], 27, 8, 1),
                t("token_embd.weight", vec![512, 1000], 12, 144, 256),
            ],
            "loader",
        );
        let g = GgufFile::open(&path).unwrap();
        let plan = plan_model(&g, 1).unwrap();

        // Hand-computed:
        //   up/gate IQ2_XXS: 512/256=2 blocks x66 = 132 B/row x 256x16 rows = 540672 each
        //   down Q2_K: 1 block x84 = 84 B/row x 512x16 = 688128
        //   q_a / shexp Q4_K: 2x144 = 288 B/row x 256 = 73728 each
        //   norm 512 f32 -> bf16 = 1024; sinks 8 -> bf16 = 16; tid2eid = 0
        //   embd Q4_K: 288 x 1000 = 288000 (global)
        let layer0 = 540_672u64 * 2 + 688_128 + 73_728 * 2 + 1_024 + 16;
        assert_eq!(plan.layer_bytes, vec![layer0]);
        assert_eq!(plan.global_bytes, 288_000);
        assert_eq!(plan.total_bytes, layer0 + 288_000);

        let by_name = |n: &str| plan.tensors.iter().find(|t| t.name == n).unwrap();
        assert_eq!(
            by_name("blk.0.ffn_up_exps.weight").residency,
            Residency::KeepIq2Xxs
        );
        assert_eq!(
            by_name("blk.0.ffn_down_exps.weight").residency,
            Residency::KeepQ2K
        );
        assert_eq!(
            by_name("blk.0.attn_q_a.weight").residency,
            Residency::KeepKQuant(KQuant::Q4K)
        );
        assert_eq!(
            by_name("blk.0.attn_norm.weight").residency,
            Residency::DequantBf16
        );
        assert_eq!(
            by_name("blk.0.attn_sinks").residency,
            Residency::DequantBf16
        );
        assert_eq!(by_name("blk.0.ffn_gate_tid2eid").bytes, 0);
        std::fs::remove_file(path).ok();
    }
}
