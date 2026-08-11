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

use infer_gguf::deepseek4::{Dsv4TensorKind, classify_tensor};
use infer_gguf::gguf::{GgmlType, GgufFile, TensorInfo};

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
    if kind == Dsv4TensorKind::TokenEmbedding && matches!(ty, GgmlType::Iq2Xxs | GgmlType::Q2K) {
        return Residency::DequantBf16;
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
    use infer_gguf::dequant;
    use infer_gguf::gguf::{GgmlType, GgufFile};

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
