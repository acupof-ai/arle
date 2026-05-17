//! Safetensors and GGUF weight loading + RoPE precomputation.
//!
//! Two loading paths:
//! - **Safetensors** (default): `load_tensor_1d`, `load_tensor_2d`, `load_tensor_2d_maybe_quantized`
//! - **GGUF**: `load_tensor_1d_gguf`, `load_tensor_2d_gguf` — dequant to BF16 at load time

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::{info, warn};
use memmap2::Mmap;
use safetensors::{SafeTensors, tensor::Dtype};
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use crate::gguf::{
    self, GgufFile, find_tensor_name, load_matrix_v_reorder_rows_bf16_host, load_vector_bf16_host,
    load_vector_offset_norm_bf16_host,
};
use crate::quant::QuantMeta;
use crate::tp::{TpLoadContext, TpShardAxis};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};

/// Load shard metadata. Returns (shard_file_paths, weight_map: tensor_name -> shard_index)
pub fn load_shard_info(model_path: &str) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let single_path = format!("{}/model.safetensors", model_path);
    let index_path = format!("{}/model.safetensors.index.json", model_path);
    if std::path::Path::new(&single_path).exists() && !std::path::Path::new(&index_path).exists() {
        // Single file, no index — all tensors keyed by name within the file
        return Ok((vec![single_path], HashMap::new()));
    }

    let index_path = format!("{}/model.safetensors.index.json", model_path);
    let index_content = fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weight_map_json = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid index.json: missing weight_map"))?;

    let mut shard_files: Vec<String> = Vec::new();
    let mut file_to_idx: HashMap<String, usize> = HashMap::new();
    let mut weight_map: HashMap<String, usize> = HashMap::new();

    for (tensor_name, shard_file_val) in weight_map_json {
        let shard_file = shard_file_val.as_str().unwrap().to_string();
        let idx = if let Some(&idx) = file_to_idx.get(&shard_file) {
            idx
        } else {
            let idx = shard_files.len();
            shard_files.push(format!("{}/{}", model_path, &shard_file));
            file_to_idx.insert(shard_file, idx);
            idx
        };
        weight_map.insert(tensor_name.clone(), idx);
    }

    Ok((shard_files, weight_map))
}

/// Memory-map shard files. Returns the mmaps; caller deserializes SafeTensors from them.
pub(crate) fn mmap_shards(shard_paths: &[String]) -> Result<Vec<Mmap>> {
    let t0 = Instant::now();
    let mmaps: Vec<Mmap> = shard_paths
        .iter()
        .map(|p| {
            let file = fs::File::open(p)?;
            // SAFETY: we keep the Mmap alive for the duration of model loading,
            // and the file is not modified concurrently.
            unsafe { Mmap::map(&file) }
        })
        .collect::<std::io::Result<_>>()?;

    let total_bytes: usize = mmaps.iter().map(|m| m.len()).sum();
    info!(
        "Memory-mapped {} shard(s) ({:.1} MB) in {:.0}ms",
        mmaps.len(),
        total_bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64() * 1e3
    );
    Ok(mmaps)
}

/// Build a `&'static str` debug label for a 1D weight tensor.
///
/// Leaks a small `String` — acceptable because weight loading is a one-time startup cost
/// and the labels live for the process lifetime.
fn shape_label_1d(name: &str, shape: &[usize]) -> &'static str {
    let dims: String = shape
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let short = name.rsplit('.').next().unwrap_or(name);
    let label = format!("{}[{}]", short, dims);
    // SAFETY: intentional leak — one allocation per weight, bounded by model size.
    Box::leak(label.into_boxed_str())
}

fn find_tensor<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        shards[idx]
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Failed to load tensor '{}': {}", name, e))
    } else {
        // Fallback: try all shards (single-file case)
        for shard in shards {
            if let Ok(t) = shard.tensor(name) {
                return Ok(t);
            }
        }
        Err(anyhow::anyhow!("Tensor '{}' not found in any shard", name))
    }
}

pub(crate) fn load_tensor_1d(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceVec> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    let label = shape_label_1d(name, shape);
    DeviceVec::from_safetensors(ctx, tensor.data()).map(|v| v.with_label(label))
}

#[allow(dead_code)]
pub(crate) fn load_tensor_1d_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tp: &TpLoadContext,
) -> Result<DeviceVec> {
    if tp.is_single() {
        return load_tensor_1d(ctx, shards, weight_map, name);
    }
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    anyhow::ensure!(
        shape.len() == 1,
        "{name}: expected 1D tensor for TP load, got shape {:?}",
        shape
    );
    anyhow::ensure!(
        matches!(tp.axis, TpShardAxis::Column),
        "{name}: 1D TP shard must use column axis"
    );
    anyhow::ensure!(
        tp.sharding.total == shape[0],
        "{name}: shard total {} does not match tensor len {}",
        tp.sharding.total,
        shape[0]
    );
    let all = bytes_to_bf16_vec(tensor.data())?;
    let shard = &all[tp.sharding.range()];
    DeviceVec::from_host(ctx, shard).map(|v| v.with_label(shape_label_1d(name, &[shard.len()])))
}

pub(crate) fn load_tensor_2d(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    DeviceMatrix::from_safetensors(ctx, tensor.data(), shape[0], shape[1])
}

pub(crate) fn load_tensor_2d_concat_rows(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    names: &[&str],
) -> Result<DeviceMatrix> {
    anyhow::ensure!(
        !names.is_empty(),
        "concat_rows load requires at least one tensor"
    );
    let mut rows = 0usize;
    let mut cols = None;
    let mut host = Vec::new();
    for name in names {
        let tensor = find_tensor(shards, weight_map, name)?;
        let shape = tensor.shape();
        anyhow::ensure!(
            shape.len() == 2,
            "{name}: expected 2D tensor for concat_rows load, got shape {:?}",
            shape
        );
        let tensor_cols = shape[1];
        if let Some(expected_cols) = cols {
            anyhow::ensure!(
                tensor_cols == expected_cols,
                "{name}: concat_rows cols mismatch: expected {expected_cols}, got {tensor_cols}"
            );
        } else {
            cols = Some(tensor_cols);
        }
        let tensor_rows = shape[0];
        anyhow::ensure!(
            tensor.data().len() == tensor_rows * tensor_cols * std::mem::size_of::<bf16>(),
            "{name}: bf16 matrix byte length mismatch: expected {}, got {}",
            tensor_rows * tensor_cols * std::mem::size_of::<bf16>(),
            tensor.data().len()
        );
        host.reserve(tensor_rows * tensor_cols);
        push_bf16_range(tensor.data(), 0, tensor_rows * tensor_cols, &mut host);
        rows += tensor_rows;
    }
    DeviceMatrix::from_host(ctx, &host, rows, cols.unwrap_or(0))
}

#[allow(dead_code)]
pub(crate) fn load_tensor_1d_f32_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tp: &TpLoadContext,
) -> Result<CudaSlice<f32>> {
    if tp.is_single() {
        return load_tensor_1d_f32(ctx, shards, weight_map, name);
    }
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    anyhow::ensure!(
        shape.len() == 1,
        "{name}: expected 1D f32 tensor for TP load, got shape {:?}",
        shape
    );
    anyhow::ensure!(
        matches!(tp.axis, TpShardAxis::Column),
        "{name}: 1D TP shard must use column axis"
    );
    anyhow::ensure!(
        tp.sharding.total == shape[0],
        "{name}: shard total {} does not match tensor len {}",
        tp.sharding.total,
        shape[0]
    );
    let bytes = tensor.data();
    anyhow::ensure!(
        bytes.len() == shape[0] * std::mem::size_of::<f32>(),
        "{name}: f32 byte length mismatch: expected {}, got {}",
        shape[0] * std::mem::size_of::<f32>(),
        bytes.len()
    );
    let all: Vec<f32> = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    ctx.stream
        .clone_htod(&all[tp.sharding.range()])
        .map_err(|e| anyhow::anyhow!("H2D copy failed: {e}"))
}

#[allow(dead_code)]
pub(crate) fn load_tensor_2d_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tp: &TpLoadContext,
) -> Result<DeviceMatrix> {
    if tp.is_single() {
        return load_tensor_2d(ctx, shards, weight_map, name);
    }
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    anyhow::ensure!(
        shape.len() == 2,
        "{name}: expected 2D tensor for TP load, got shape {:?}",
        shape
    );
    let (host, rows, cols) = shard_bf16_matrix_host(tensor.data(), shape[0], shape[1], tp)?;
    DeviceMatrix::from_host(ctx, &host, rows, cols)
}

#[allow(dead_code)]
pub(crate) fn load_tensor_2d_fused_column_segments_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    segment_rows: &[usize],
    rank: usize,
    world_size: usize,
) -> Result<DeviceMatrix> {
    if world_size == 1 {
        return load_tensor_2d(ctx, shards, weight_map, name);
    }
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    anyhow::ensure!(
        shape.len() == 2,
        "{name}: expected 2D fused segmented tensor for TP load, got shape {:?}",
        shape
    );
    let rows = shape[0];
    let cols = shape[1];
    anyhow::ensure!(
        segment_rows.iter().sum::<usize>() == rows,
        "{name}: fused segment rows {:?} do not sum to tensor rows {rows}",
        segment_rows
    );
    anyhow::ensure!(
        tensor.data().len() == rows * cols * std::mem::size_of::<bf16>(),
        "{name}: bf16 matrix byte length mismatch: expected {}, got {}",
        rows * cols * std::mem::size_of::<bf16>(),
        tensor.data().len()
    );

    let mut out = Vec::new();
    let mut segment_base = 0usize;
    for &segment_len in segment_rows {
        let tp = TpLoadContext::column(rank, world_size, segment_len)?;
        out.reserve(tp.sharding.size * cols);
        let elem_start = (segment_base + tp.sharding.offset) * cols;
        let elem_end = (segment_base + tp.sharding.end()) * cols;
        push_bf16_range(tensor.data(), elem_start, elem_end, &mut out);
        segment_base += segment_len;
    }
    let out_rows = out.len() / cols;
    DeviceMatrix::from_host(ctx, &out, out_rows, cols)
}

#[allow(dead_code)]
pub(crate) fn load_tensor_1d_fused_segments_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    segment_lens: &[usize],
    rank: usize,
    world_size: usize,
) -> Result<DeviceVec> {
    if world_size == 1 {
        return load_tensor_1d(ctx, shards, weight_map, name);
    }
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    anyhow::ensure!(
        shape.len() == 1,
        "{name}: expected 1D fused segmented tensor for TP load, got shape {:?}",
        shape
    );
    anyhow::ensure!(
        segment_lens.iter().sum::<usize>() == shape[0],
        "{name}: fused segment lengths {:?} do not sum to tensor len {}",
        segment_lens,
        shape[0]
    );

    let all = bytes_to_bf16_vec(tensor.data())?;
    let mut out = Vec::new();
    let mut segment_base = 0usize;
    for &segment_len in segment_lens {
        let tp = TpLoadContext::column(rank, world_size, segment_len)?;
        out.extend_from_slice(
            &all[segment_base + tp.sharding.offset..segment_base + tp.sharding.end()],
        );
        segment_base += segment_len;
    }
    DeviceVec::from_host(ctx, &out).map(|v| v.with_label(shape_label_1d(name, &[out.len()])))
}

#[allow(dead_code)]
pub(crate) fn shard_bf16_matrix_host(
    data: &[u8],
    rows: usize,
    cols: usize,
    tp: &TpLoadContext,
) -> Result<(Vec<bf16>, usize, usize)> {
    anyhow::ensure!(
        data.len() == rows * cols * std::mem::size_of::<bf16>(),
        "bf16 matrix byte length mismatch: expected {}, got {}",
        rows * cols * std::mem::size_of::<bf16>(),
        data.len()
    );
    match tp.axis {
        TpShardAxis::Row => {
            anyhow::ensure!(
                tp.sharding.total == cols,
                "row shard total {} does not match tensor cols {cols}",
                tp.sharding.total
            );
            let mut out = Vec::with_capacity(rows * tp.sharding.size);
            for row in 0..rows {
                let elem_start = row * cols + tp.sharding.offset;
                let elem_end = elem_start + tp.sharding.size;
                push_bf16_range(data, elem_start, elem_end, &mut out);
            }
            Ok((out, rows, tp.sharding.size))
        }
        TpShardAxis::Column => {
            anyhow::ensure!(
                tp.sharding.total == rows,
                "column shard total {} does not match tensor rows {rows}",
                tp.sharding.total
            );
            let elem_start = tp.sharding.offset * cols;
            let elem_end = tp.sharding.end() * cols;
            let mut out = Vec::with_capacity(tp.sharding.size * cols);
            push_bf16_range(data, elem_start, elem_end, &mut out);
            Ok((out, tp.sharding.size, cols))
        }
    }
}

fn push_bf16_range(data: &[u8], elem_start: usize, elem_end: usize, out: &mut Vec<bf16>) {
    let byte_start = elem_start * std::mem::size_of::<bf16>();
    let byte_end = elem_end * std::mem::size_of::<bf16>();
    out.extend(
        data[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<bf16>())
            .map(|bytes| bf16::from_le_bytes([bytes[0], bytes[1]])),
    );
}

fn bytes_to_bf16_vec(data: &[u8]) -> Result<Vec<bf16>> {
    anyhow::ensure!(
        data.len().is_multiple_of(std::mem::size_of::<bf16>()),
        "bf16 byte length must be divisible by 2, got {}",
        data.len()
    );
    Ok(data
        .chunks_exact(std::mem::size_of::<bf16>())
        .map(|bytes| bf16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct QuantLoadConfig {
    pub(crate) group_size: Option<usize>,
    pub(crate) bits: Option<u8>,
    pub(crate) tq_bits: Option<u8>,
    pub(crate) marlin_w4a8: bool,
    pub(crate) marlin_w4_hybrid: bool,
    pub(crate) fp8_weight_scale_inv: bool,
    pub(crate) fp8_block_rows: usize,
    pub(crate) fp8_block_cols: usize,
    pub(crate) unsupported_reason: Option<&'static str>,
}

impl QuantLoadConfig {
    pub(crate) fn from_meta(meta: &QuantMeta) -> Self {
        match meta {
            QuantMeta::Gptq(config) if !config.sym => Self {
                unsupported_reason: Some(
                    "asymmetric GPTQ qzeros are not supported by the current CUDA W2/W4/W8 loader",
                ),
                ..Self::default()
            },
            QuantMeta::Gptq(config) if config.group_size > 0 => Self {
                group_size: Some(config.group_size as usize),
                bits: Some(config.bits),
                ..Self::default()
            },
            QuantMeta::Gptq(config) => Self {
                bits: Some(config.bits),
                ..Self::default()
            },
            QuantMeta::Awq(config) if config.zero_point => Self {
                unsupported_reason: Some(
                    "zero-point AWQ qzeros are not supported by the current CUDA W4 loader",
                ),
                ..Self::default()
            },
            QuantMeta::Awq(config) => Self {
                group_size: Some(config.group_size),
                bits: Some(config.bits),
                ..Self::default()
            },
            QuantMeta::Int8(_) => Self {
                bits: Some(8),
                ..Self::default()
            },
            QuantMeta::MarlinW4A8(config) => Self {
                group_size: Some(config.group_size),
                bits: Some(4),
                marlin_w4a8: true,
                ..Self::default()
            },
            QuantMeta::MarlinW4Hybrid(config) => Self {
                group_size: Some(config.group_size),
                bits: Some(4),
                marlin_w4a8: true,
                marlin_w4_hybrid: true,
                ..Self::default()
            },
            QuantMeta::Fp8(config) => Self {
                fp8_weight_scale_inv: true,
                fp8_block_rows: config.weight_block_size[0],
                fp8_block_cols: config.weight_block_size[1],
                ..Self::default()
            },
            QuantMeta::TurboQuant(config) => Self {
                group_size: Some(config.group_size),
                tq_bits: Some(config.bits),
                ..Self::default()
            },
            _ => Self::default(),
        }
    }

    pub(crate) fn from_model_path(model_path: &str) -> Result<Self> {
        if let Ok(format) = std::env::var("INFER_QUANT_FORMAT_OVERRIDE") {
            match format.as_str() {
                "marlin_w4a8" | "w4a8_marlin" => {
                    return Ok(Self {
                        group_size: Some(128),
                        bits: Some(4),
                        marlin_w4a8: true,
                        ..Self::default()
                    });
                }
                "marlin_w4_hybrid" => {
                    return Ok(Self {
                        group_size: Some(128),
                        bits: Some(4),
                        marlin_w4a8: true,
                        marlin_w4_hybrid: true,
                        ..Self::default()
                    });
                }
                _ => {}
            }
        }
        Ok(Self::from_meta(&crate::quant::load_quant_meta(model_path)?))
    }

    pub(crate) fn enabled(self) -> bool {
        self.group_size.is_some()
            || self.bits.is_some()
            || self.tq_bits.is_some()
            || self.marlin_w4a8
            || self.marlin_w4_hybrid
            || self.fp8_weight_scale_inv
            || self.unsupported_reason.is_some()
    }
}

fn detect_uniform_quant_layout(
    name: &str,
    qw_cols: usize,
    num_groups: usize,
    config: QuantLoadConfig,
) -> Result<(usize, usize, u8)> {
    anyhow::ensure!(
        num_groups > 0,
        "{name}: quantized scales must have at least one group"
    );

    let bits = if let Some(bits) = config.bits {
        bits
    } else if let Some(group_size) = config.group_size {
        let orig_k = num_groups * group_size;
        if qw_cols == orig_k / 4 {
            2
        } else if qw_cols == orig_k / 2 {
            4
        } else if qw_cols == orig_k {
            8
        } else {
            anyhow::bail!(
                "{name}: cannot infer quantized weight bits from qweight cols={qw_cols}, \
                 groups={num_groups}, group_size={group_size}"
            );
        }
    } else {
        let mut candidates = [0u8; 3];
        let mut count = 0usize;
        for bits in [2u8, 4, 8] {
            let elems_per_byte = match bits {
                2 => 4,
                4 => 2,
                8 => 1,
                _ => unreachable!(),
            };
            let orig_k = qw_cols * elems_per_byte;
            if orig_k.is_multiple_of(num_groups) {
                candidates[count] = bits;
                count += 1;
            }
        }
        anyhow::ensure!(
            count == 1,
            "{name}: quantized weight config must specify bits or group_size; \
             inferred {count} possible layouts from qweight cols={qw_cols}, groups={num_groups}"
        );
        candidates[0]
    };

    anyhow::ensure!(
        matches!(bits, 2 | 4 | 8),
        "{name}: unsupported weight quantization bits={bits} (supported: 2, 4, 8)"
    );
    let elems_per_byte = match bits {
        2 => 4,
        4 => 2,
        8 => 1,
        _ => unreachable!(),
    };
    let orig_k = qw_cols * elems_per_byte;
    let group_size = if let Some(group_size) = config.group_size {
        anyhow::ensure!(
            num_groups * group_size == orig_k,
            "{name}: quantized shape mismatch for bits={bits}: qweight cols={qw_cols} \
             implies K={orig_k}, but scales groups={num_groups} and group_size={group_size}"
        );
        group_size
    } else {
        anyhow::ensure!(
            orig_k.is_multiple_of(num_groups),
            "{name}: cannot infer group_size from K={orig_k} and groups={num_groups}"
        );
        orig_k / num_groups
    };
    Ok((orig_k, group_size, bits))
}

fn detect_turboquant_layout(
    name: &str,
    packed_cols: usize,
    num_groups: usize,
    config: QuantLoadConfig,
) -> Result<(usize, usize, u8)> {
    anyhow::ensure!(
        num_groups > 0,
        "{name}: TurboQuant scales must have at least one group"
    );
    let bits = config.tq_bits.unwrap_or(0);
    anyhow::ensure!(
        matches!(bits, 2 | 3 | 4),
        "{name}: TurboQuant requires explicit bits in turboquant_config.json or config.json \
         (got {:?}); supported bits are 2, 3, and 4",
        config.tq_bits
    );
    let elems_per_byte = if bits == 2 { 4 } else { 2 };
    let orig_k = packed_cols * elems_per_byte;
    let group_size = if let Some(group_size) = config.group_size {
        anyhow::ensure!(
            num_groups * group_size == orig_k,
            "{name}: TurboQuant shape mismatch for TQ{bits}: packed cols={packed_cols} \
             implies K={orig_k}, but scales groups={num_groups} and group_size={group_size}"
        );
        group_size
    } else {
        anyhow::ensure!(
            orig_k.is_multiple_of(num_groups),
            "{name}: cannot infer TurboQuant group_size from K={orig_k} and groups={num_groups}"
        );
        orig_k / num_groups
    };
    Ok((orig_k, group_size, bits))
}

/// Load a 2D tensor, trying quantized (.qweight + .scales) first, then bf16.
///
/// If `name` = "model.layers.0.self_attn.q_proj.weight", tries:
///   1. "model.layers.0.self_attn.q_proj.qweight" + ".scales" → INT8 quantized
///   2. "model.layers.0.self_attn.q_proj.weight" → bf16
pub(crate) fn load_tensor_2d_maybe_quantized_with_config(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    config: QuantLoadConfig,
) -> Result<DeviceMatrix> {
    if config.marlin_w4_hybrid {
        anyhow::ensure!(
            config.group_size.unwrap_or(128) == 128,
            "{name}: Marlin W4 hybrid currently supports group_size=128 only, got {:?}",
            config.group_size
        );
        let group_size = config.group_size.unwrap_or(128);

        let w4a16_qweight_name = name.replace(".weight", ".marlin_qweight");
        let w4a16_scales_name = name.replace(".weight", ".marlin_scales");
        let w4a8_qweight_name = name.replace(".weight", ".marlin_w4a8_qweight");
        let w4a8_channel_scales_name = name.replace(".weight", ".marlin_w4a8_s_channel");
        let w4a8_group_scales_name = name.replace(".weight", ".marlin_w4a8_s_group");

        let w4a16_qweight_tensor = find_tensor(shards, weight_map, &w4a16_qweight_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {w4a16_qweight_name}: {e}"))?;
        let w4a16_scales_tensor = find_tensor(shards, weight_map, &w4a16_scales_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {w4a16_scales_name}: {e}"))?;
        let w4a8_qweight_tensor = find_tensor(shards, weight_map, &w4a8_qweight_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {w4a8_qweight_name}: {e}"))?;
        let w4a8_channel_scales_tensor = find_tensor(shards, weight_map, &w4a8_channel_scales_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {w4a8_channel_scales_name}: {e}"))?;
        let w4a8_group_scales_tensor = find_tensor(shards, weight_map, &w4a8_group_scales_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {w4a8_group_scales_name}: {e}"))?;

        let rows = w4a8_channel_scales_tensor.shape().iter().product::<usize>();
        anyhow::ensure!(
            !w4a8_group_scales_tensor.shape().is_empty(),
            "{name}: {w4a8_group_scales_name} must have at least one dimension"
        );
        let num_groups = w4a8_group_scales_tensor.shape()[0];
        let cols = num_groups * group_size;

        let w4a16_qweight = w4a16_qweight_tensor.data();
        let w4a16_scales: &[u16] = unsafe {
            std::slice::from_raw_parts(
                w4a16_scales_tensor.data().as_ptr().cast::<u16>(),
                w4a16_scales_tensor.shape().iter().product::<usize>(),
            )
        };
        let w4a8_qweight = w4a8_qweight_tensor.data();
        let w4a8_channel_scales: &[f32] = unsafe {
            std::slice::from_raw_parts(
                w4a8_channel_scales_tensor.data().as_ptr().cast::<f32>(),
                rows,
            )
        };
        let w4a8_group_scales: &[u16] = unsafe {
            std::slice::from_raw_parts(
                w4a8_group_scales_tensor.data().as_ptr().cast::<u16>(),
                w4a8_group_scales_tensor.shape().iter().product::<usize>(),
            )
        };

        log::info!(
            "Loaded Marlin W4 hybrid {}: [{}x{}] group_size={} w4a16_q {:?} w4a16_s {:?} w4a8_q {:?} w4a8_s_channel {:?} w4a8_s_group {:?}",
            name,
            rows,
            cols,
            group_size,
            w4a16_qweight_tensor.shape(),
            w4a16_scales_tensor.shape(),
            w4a8_qweight_tensor.shape(),
            w4a8_channel_scales_tensor.shape(),
            w4a8_group_scales_tensor.shape()
        );
        return DeviceMatrix::from_hybrid_w4_marlin(
            ctx,
            w4a16_qweight,
            w4a16_scales,
            w4a8_qweight,
            w4a8_channel_scales,
            w4a8_group_scales,
            rows,
            cols,
            group_size,
        );
    }

    if config.marlin_w4a8 {
        anyhow::ensure!(
            config.group_size.unwrap_or(128) == 128,
            "{name}: MarlinW4A8 currently supports group_size=128 only, got {:?}",
            config.group_size
        );
        let packed_name = name.replace(".weight", ".marlin_w4a8_qweight");
        let channel_scales_name = name.replace(".weight", ".marlin_w4a8_s_channel");
        let group_scales_name = name.replace(".weight", ".marlin_w4a8_s_group");
        let packed_tensor = find_tensor(shards, weight_map, &packed_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {packed_name}: {e}"))?;
        let channel_scales_tensor = find_tensor(shards, weight_map, &channel_scales_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {channel_scales_name}: {e}"))?;
        let group_scales_tensor = find_tensor(shards, weight_map, &group_scales_name)
            .map_err(|e| anyhow::anyhow!("{name}: missing {group_scales_name}: {e}"))?;
        let rows = channel_scales_tensor.shape().iter().product::<usize>();
        let group_size = config.group_size.unwrap_or(128);
        let num_groups = group_scales_tensor.shape()[0];
        let cols = num_groups * group_size;

        let packed: &[u8] = unsafe {
            std::slice::from_raw_parts(packed_tensor.data().as_ptr(), packed_tensor.data().len())
        };
        let channel_scales: &[f32] = unsafe {
            std::slice::from_raw_parts(channel_scales_tensor.data().as_ptr().cast::<f32>(), rows)
        };
        let group_scales: &[u16] = unsafe {
            std::slice::from_raw_parts(
                group_scales_tensor.data().as_ptr().cast::<u16>(),
                group_scales_tensor.shape().iter().product::<usize>(),
            )
        };

        log::info!(
            "Loaded MarlinW4A8 {}: [{}x{}] group_size={} packed {:?} s_channel {:?} s_group {:?}",
            name,
            rows,
            cols,
            group_size,
            packed_tensor.shape(),
            channel_scales_tensor.shape(),
            group_scales_tensor.shape()
        );
        return DeviceMatrix::from_marlin_w4a8(
            ctx,
            packed,
            channel_scales,
            group_scales,
            rows,
            cols,
            group_size,
        );
    }

    if config.fp8_weight_scale_inv {
        let scale_name = name.replace(".weight", ".weight_scale_inv");
        if let Ok(weight_tensor) = find_tensor(shards, weight_map, name) {
            match find_tensor(shards, weight_map, &scale_name) {
                Ok(scale_tensor) if weight_tensor.dtype() == Dtype::F8_E4M3 => {
                    let shape = weight_tensor.shape();
                    anyhow::ensure!(
                        shape.len() == 2,
                        "{name}: expected 2D FP8 tensor, got shape {:?}",
                        shape
                    );
                    let host = dequantize_fp8_e4m3_weight_scale_inv_to_bf16_host(
                        name,
                        weight_tensor.data(),
                        shape[0],
                        shape[1],
                        scale_tensor.data(),
                        scale_tensor.dtype(),
                        scale_tensor.shape(),
                        config.fp8_block_rows,
                        config.fp8_block_cols,
                    )?;
                    log::info!(
                        "Loaded FP8 {} via weight_scale_inv: [{}x{}] block=[{},{}] scale {:?}",
                        name,
                        shape[0],
                        shape[1],
                        config.fp8_block_rows,
                        config.fp8_block_cols,
                        scale_tensor.shape()
                    );
                    return DeviceMatrix::from_host(ctx, &host, shape[0], shape[1]);
                }
                Ok(_) => {}
                Err(err) => {
                    anyhow::ensure!(
                        weight_tensor.dtype() != Dtype::F8_E4M3,
                        "{name}: FP8 tensor is missing side tensor {scale_name}: {err}"
                    );
                }
            }
        }
    }

    // Try quantized path: replace ".weight" with ".qweight"
    let qweight_name = name.replace(".weight", ".qweight");
    let scales_name = name.replace(".weight", ".scales");

    if weight_map.contains_key(&qweight_name) && weight_map.contains_key(&scales_name) {
        if let Some(reason) = config.unsupported_reason {
            let qzeros_name = name.replace(".weight", ".qzeros");
            let qzeros_suffix = if weight_map.contains_key(&qzeros_name) {
                format!(" plus {qzeros_name}")
            } else {
                String::new()
            };
            anyhow::bail!(
                "{name}: unsupported quantized checkpoint layout: {reason}; found \
                 {qweight_name} and {scales_name}{qzeros_suffix}; refusing to load it as \
                 symmetric quantization"
            );
        }
        let qw_tensor = find_tensor(shards, weight_map, &qweight_name)?;
        let sc_tensor = find_tensor(shards, weight_map, &scales_name)?;

        let qw_shape = qw_tensor.shape();
        let sc_shape = sc_tensor.shape();
        let rows = qw_shape[0];
        let qw_cols = qw_shape[1];
        let num_groups = sc_shape[1];
        let (orig_k, group_size, bits) =
            detect_uniform_quant_layout(name, qw_cols, num_groups, config)?;

        let sc_data: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(
                sc_tensor.data().as_ptr().cast::<half::bf16>(),
                sc_shape[0] * sc_shape[1],
            )
        };

        if bits == 2 {
            // INT2 packed: 4 values per byte
            let packed: &[u8] =
                unsafe { std::slice::from_raw_parts(qw_tensor.data().as_ptr(), rows * qw_cols) };
            log::info!(
                "Loaded quantized {}: [{}x{}] INT2, group_size={}",
                name,
                rows,
                orig_k,
                group_size
            );
            return DeviceMatrix::from_quantized_int2(
                ctx, packed, sc_data, rows, orig_k, group_size,
            );
        }
        if bits == 4 {
            // INT4 packed: 2 values per byte
            let packed: &[u8] =
                unsafe { std::slice::from_raw_parts(qw_tensor.data().as_ptr(), rows * qw_cols) };
            log::info!(
                "Loaded quantized {}: [{}x{}] INT4, group_size={}",
                name,
                rows,
                orig_k,
                group_size
            );
            let mut mat =
                DeviceMatrix::from_quantized_int4(ctx, packed, sc_data, rows, orig_k, group_size)?;
            // Load pre-computed Marlin weights if available (from scripts/marlin_repack.py)
            let marlin_key = qweight_name.replace(".qweight", ".marlin_qweight");
            let marlin_scales_key = qweight_name.replace(".qweight", ".marlin_scales");
            if let (Ok(mp), Ok(ms)) = (
                find_tensor(shards, weight_map, &marlin_key),
                find_tensor(shards, weight_map, &marlin_scales_key),
            ) {
                let mp_data: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        mp.data().as_ptr(),
                        mp.shape().iter().product::<usize>() * 4, // int32 → bytes
                    )
                };
                let ms_data: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        ms.data().as_ptr().cast::<u16>(),
                        ms.shape().iter().product::<usize>(),
                    )
                };
                let mp_gpu: cudarc::driver::CudaSlice<u8> = ctx
                    .stream
                    .clone_htod(mp_data)
                    .map_err(|e| anyhow::anyhow!("H2D Marlin packed: {}", e))?;
                let ms_gpu: cudarc::driver::CudaSlice<u16> = ctx
                    .stream
                    .clone_htod(ms_data)
                    .map_err(|e| anyhow::anyhow!("H2D Marlin scales: {}", e))?;
                mat.marlin_packed = Some(mp_gpu);
                mat.marlin_scales = Some(ms_gpu);
                log::info!(
                    "  + Marlin repacked: {:?} + scales {:?}",
                    mp.shape(),
                    ms.shape()
                );
            }
            return Ok(mat);
        }

        debug_assert_eq!(bits, 8);
        // INT8
        let qw_data: &[i8] = unsafe {
            std::slice::from_raw_parts(qw_tensor.data().as_ptr().cast::<i8>(), rows * qw_cols)
        };
        log::info!(
            "Loaded quantized {}: [{}x{}] INT8, group_size={}",
            name,
            rows,
            orig_k,
            group_size
        );
        return DeviceMatrix::from_quantized_int8(ctx, qw_data, sc_data, rows, orig_k, group_size);
    }

    // Try TurboQuant path: .tq_packed + .tq_scales + .tq_signs
    let tq_packed_name = name.replace(".weight", ".tq_packed");
    let tq_scales_name = name.replace(".weight", ".tq_scales");
    let tq_signs_name = name.replace(".weight", ".tq_signs");

    if weight_map.contains_key(&tq_packed_name)
        && weight_map.contains_key(&tq_scales_name)
        && weight_map.contains_key(&tq_signs_name)
    {
        let packed_tensor = find_tensor(shards, weight_map, &tq_packed_name)?;
        let scales_tensor = find_tensor(shards, weight_map, &tq_scales_name)?;
        let signs_tensor = find_tensor(shards, weight_map, &tq_signs_name)?;

        let rows = packed_tensor.shape()[0];
        let packed_cols = packed_tensor.shape()[1];
        let num_groups = scales_tensor.shape()[1];
        let (orig_k, group_size, bits) =
            detect_turboquant_layout(name, packed_cols, num_groups, config)?;

        let packed: &[u8] = unsafe {
            std::slice::from_raw_parts(packed_tensor.data().as_ptr(), rows * packed_cols)
        };
        let scales: &[half::f16] = unsafe {
            std::slice::from_raw_parts(
                scales_tensor.data().as_ptr().cast::<half::f16>(),
                rows * num_groups,
            )
        };
        let signs: &[i8] = unsafe {
            std::slice::from_raw_parts(
                signs_tensor.data().as_ptr().cast::<i8>(),
                signs_tensor.shape()[0],
            )
        };

        // Phase 2: keep weights packed on GPU — dequant happens at runtime
        // in fused GEMV (decode) or bulk dequant + cuBLAS GEMM (prefill).
        let num_levels = 1usize << bits;
        let mut centroids_host = vec![0.0f32; num_levels];
        let mut boundaries_host = vec![0.0f32; num_levels + 1];
        unsafe {
            ffi::turboquant_lloyd_max(
                centroids_host.as_mut_ptr(),
                boundaries_host.as_mut_ptr(),
                num_levels as i32,
                group_size as i32,
                200,
            );
        }
        let centroids_gpu: CudaSlice<f32> = ctx
            .stream
            .clone_htod(&centroids_host)
            .map_err(|e| anyhow::anyhow!("H2D centroids failed: {}", e))?;

        let scales_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), std::mem::size_of_val(scales))
        };

        log::info!(
            "Loaded TurboQuant {}: [{}x{}] packed {}-bit on GPU, group_size={}",
            name,
            rows,
            orig_k,
            bits,
            group_size
        );

        let mat = DeviceMatrix::from_quantized_tq(
            ctx,
            packed,
            scales_bytes,
            signs,
            &centroids_gpu,
            rows,
            orig_k,
            group_size,
            bits,
        )?;
        return Ok(mat);
    }

    if config.enabled() {
        warn!(
            "Quant config is present but '{name}' has no supported packed side tensors; loading BF16"
        );
    }

    // Fallback: bf16
    load_tensor_2d(ctx, shards, weight_map, name)
}

fn dequantize_fp8_e4m3_weight_scale_inv_to_bf16_host(
    name: &str,
    weight: &[u8],
    rows: usize,
    cols: usize,
    scales: &[u8],
    scale_dtype: Dtype,
    scale_shape: &[usize],
    block_rows: usize,
    block_cols: usize,
) -> Result<Vec<bf16>> {
    anyhow::ensure!(
        block_rows > 0 && block_cols > 0,
        "{name}: FP8 block shape must be positive, got [{block_rows},{block_cols}]"
    );
    anyhow::ensure!(
        weight.len() == rows * cols,
        "{name}: FP8 byte length mismatch: expected {}, got {}",
        rows * cols,
        weight.len()
    );
    let scale_rows = rows.div_ceil(block_rows);
    let scale_cols = cols.div_ceil(block_cols);
    anyhow::ensure!(
        scale_shape == [scale_rows, scale_cols],
        "{name}: weight_scale_inv shape {:?} does not match expected [{scale_rows},{scale_cols}] for weight [{rows},{cols}] and block [{block_rows},{block_cols}]",
        scale_shape
    );
    let scale_elem_bytes = match scale_dtype {
        Dtype::BF16 => std::mem::size_of::<bf16>(),
        Dtype::F32 => std::mem::size_of::<f32>(),
        dtype => anyhow::bail!("{name}: unsupported FP8 weight_scale_inv dtype {dtype:?}"),
    };
    anyhow::ensure!(
        scales.len() == scale_rows * scale_cols * scale_elem_bytes,
        "{name}: weight_scale_inv byte length mismatch: expected {}, got {}",
        scale_rows * scale_cols * scale_elem_bytes,
        scales.len()
    );

    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let scale_row = row / block_rows;
        for col in 0..cols {
            let scale_col = col / block_cols;
            let scale = fp8_scale_value(scales, scale_dtype, scale_row * scale_cols + scale_col)?;
            let value = decode_fp8_e4m3fn(weight[row * cols + col]) * scale;
            out.push(bf16::from_f32(value));
        }
    }
    Ok(out)
}

fn fp8_scale_value(scales: &[u8], dtype: Dtype, idx: usize) -> Result<f32> {
    match dtype {
        Dtype::BF16 => {
            let offset = idx * std::mem::size_of::<bf16>();
            let bytes = scales
                .get(offset..offset + std::mem::size_of::<bf16>())
                .ok_or_else(|| anyhow::anyhow!("BF16 scale index {idx} out of range"))?;
            Ok(bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
        }
        Dtype::F32 => {
            let offset = idx * std::mem::size_of::<f32>();
            let bytes = scales
                .get(offset..offset + std::mem::size_of::<f32>())
                .ok_or_else(|| anyhow::anyhow!("F32 scale index {idx} out of range"))?;
            Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        dtype => anyhow::bail!("unsupported FP8 scale dtype {dtype:?}"),
    }
}

fn decode_fp8_e4m3fn(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2.0_f32.powi(-6)
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

/// TurboQuant Phase 1: dequantize packed weights at load time on CPU.
///
/// Reverse path: unpack → gather centroids → iFFWT → sign flip → scale by norm.
/// Produces a standard BF16 DeviceMatrix for use with existing GEMM kernels.
#[allow(dead_code)]
fn turboquant_dequant_at_load(
    ctx: &DeviceContext,
    packed: &[u8],
    scales: &[half::f16],
    signs: &[i8],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<DeviceMatrix> {
    let num_groups = cols / group_size;
    let bits = 3u8; // TODO: detect from config
    let effective_bits = if bits == 3 { 4 } else { bits as usize };
    let indices_per_byte = 8 / effective_bits;

    // Compute Lloyd-Max centroids on CPU
    let num_levels = 1usize << bits;
    let mut centroids = vec![0.0f32; num_levels];
    let mut boundaries = vec![0.0f32; num_levels + 1];
    unsafe {
        ffi::turboquant_lloyd_max(
            centroids.as_mut_ptr(),
            boundaries.as_mut_ptr(),
            num_levels as i32,
            group_size as i32,
            200,
        );
    }

    // Dequantize each row
    let mut bf16_data = vec![bf16::ZERO; rows * cols];
    let packed_cols = packed.len() / rows;

    for row in 0..rows {
        for g in 0..num_groups {
            let norm = half::f16::to_f32(scales[row * num_groups + g]);
            let group_start = g * group_size;

            // Unpack indices → centroids
            let mut rotated = vec![0.0f32; group_size];
            for d in 0..group_size {
                let k = group_start + d;
                let byte_idx = k / indices_per_byte;
                let sub_idx = k % indices_per_byte;
                let packed_byte = packed[row * packed_cols + byte_idx];
                let idx = ((packed_byte >> (sub_idx * effective_bits))
                    & ((1 << effective_bits) - 1)) as usize;
                let idx = idx.min(num_levels - 1);
                rotated[d] = centroids[idx] * norm;
            }

            // Inverse FWHT (self-inverse with 1/√n normalization)
            fwht_cpu(&mut rotated);

            // Inverse sign flip
            for d in 0..group_size {
                let k = group_start + d;
                let sign_idx = k % signs.len();
                rotated[d] *= signs[sign_idx] as f32;
                bf16_data[row * cols + k] = bf16::from_f32(rotated[d]);
            }
        }
    }

    DeviceMatrix::from_host(ctx, &bf16_data, rows, cols)
}

/// CPU Fast Walsh-Hadamard Transform (in-place, normalized by 1/√n).
fn fwht_cpu(data: &mut [f32]) {
    #[allow(dead_code)]
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = data[j];
                let b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in data.iter_mut() {
        *x *= scale;
    }
}

/// Precompute RoPE cos/sin cache as contiguous GPU buffers.
/// Layout: [max_seq_len * head_dim] — position `pos` at offset `pos * head_dim`.
pub(crate) const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;

pub(crate) fn resolve_rope_cache_len(config_hint: Option<usize>) -> usize {
    let env_override = std::env::var("INFER_ROPE_CACHE_LEN")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&len| len > 0);

    env_override
        .or(config_hint)
        .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
        .max(DEFAULT_ROPE_CACHE_LEN)
}

#[allow(dead_code)]
pub(crate) fn precompute_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(DeviceVec, DeviceVec)> {
    // Vanilla (no scaling) path. Phase 2 of M_rope-yarn-scaling: caller
    // wires `Option<&RopeScalingConfig>` via `precompute_rope_with_scaling`
    // when the model config has rope_scaling = Some(...).
    precompute_rope_with_scaling(ctx, head_dim, max_seq_len, theta, None)
}

/// Convert a `qwen35_spec::RopeScalingConfig` to its `qwen3_spec` mirror so
/// the same scaled-inv_freq math (in qwen3_spec) can drive both backends.
/// Per `docs/plans/M_rope-yarn-scaling.md` Phase 2 step 2 (2026-05-10):
/// per-crate enum duplication is intentional; a shared rope-spec crate is
/// deferred until DeepSeek-V4 needs the same enum. This conversion is the
/// thin shim that bridges them in the meantime.
fn qwen35_to_qwen3_rope_scaling(
    src: &qwen35_spec::RopeScalingConfig,
) -> qwen3_spec::RopeScalingConfig {
    use qwen3_spec::RopeScalingConfig as Dst;
    use qwen35_spec::RopeScalingConfig as Src;
    match src {
        Src::Yarn {
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            attention_factor,
            mscale,
        } => Dst::Yarn {
            factor: *factor,
            original_max_position_embeddings: *original_max_position_embeddings,
            beta_fast: *beta_fast,
            beta_slow: *beta_slow,
            attention_factor: *attention_factor,
            mscale: *mscale,
        },
        Src::Linear { factor } => Dst::Linear { factor: *factor },
        Src::NtkAware { factor } => Dst::NtkAware { factor: *factor },
    }
}

/// qwen35-spec-typed wrapper over [`precompute_rope_with_scaling`]. The
/// underlying math is identical (qwen3_spec::compute_scaled_inv_freq); only
/// the enum type differs. Caller passes their `Qwen35Config::rope_scaling`
/// directly without manual conversion.
pub(crate) fn precompute_rope_with_qwen35_scaling(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    scaling: Option<&qwen35_spec::RopeScalingConfig>,
) -> Result<(DeviceVec, DeviceVec)> {
    let converted = scaling.map(qwen35_to_qwen3_rope_scaling);
    precompute_rope_with_scaling(ctx, head_dim, max_seq_len, theta, converted.as_ref())
}

/// Long-context-aware variant of [`precompute_rope`] that accepts an
/// optional `RopeScalingConfig` (YARN / Linear / NtkAware). When `scaling`
/// is `None`, this is bit-equivalent to the legacy `precompute_rope`
/// (verified by `qwen3-spec::tests::vanilla_inv_freq_matches_legacy_formula`).
///
/// Phase 2 of M_rope-yarn-scaling. See `docs/plans/M_rope-yarn-scaling.md`.
pub(crate) fn precompute_rope_with_scaling(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    scaling: Option<&qwen3_spec::RopeScalingConfig>,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;
    let inv_freq = qwen3_spec::compute_scaled_inv_freq(head_dim, theta, scaling);

    let total = max_seq_len * head_dim;
    let mut cos_host = vec![bf16::ZERO; total];
    let mut sin_host = vec![bf16::ZERO; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for i in 0..half_dim {
            let freq = pos as f32 * inv_freq[i];
            let cos_val = bf16::from_f32(freq.cos());
            let sin_val = bf16::from_f32(freq.sin());
            // Half-split layout: [cos(0)..cos(63), cos(0)..cos(63)]
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    let cos_cache = DeviceVec::from_host(ctx, &cos_host)?.with_label("rope_cos[seq,dim]");
    let sin_cache = DeviceVec::from_host(ctx, &sin_host)?.with_label("rope_sin[seq,dim]");

    Ok((cos_cache, sin_cache))
}

#[allow(clippy::cast_ptr_alignment)]
/// Load a 1D F32 tensor to GPU as CudaSlice<f32>.
/// For weights stored in float32 (e.g., A_log, norm.weight in linear attention).
pub(crate) fn load_tensor_1d_f32(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<CudaSlice<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let data = tensor.data();
    if data.len() % 4 != 0 {
        return Err(anyhow::anyhow!(
            "F32 tensor '{}': data length {} not multiple of 4",
            name,
            data.len()
        ));
    }
    let len = data.len() / 4;
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), len) };
    let gpu_data = ctx
        .stream
        .clone_htod(slice)
        .map_err(|e| anyhow::anyhow!("H2D copy failed for '{}': {}", name, e))?;
    Ok(gpu_data)
}

/// Load shard info with fixup for mismatched shard filenames in index.json.
///
/// Some models (e.g., Qwen3.5) have index.json with shard filenames like
/// `model.safetensors-00001-of-00002.safetensors` while actual files are
/// `model-00001-of-00002.safetensors`. This function detects and fixes that.
pub(crate) fn load_shard_info_fixed(
    model_path: &str,
) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let (mut shard_files, weight_map) = load_shard_info(model_path)?;

    for path in &mut shard_files {
        if !std::path::Path::new(path).exists() {
            // Try replacing "model.safetensors-" with "model-" in filename
            let filename = std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            if let Some(rest) = filename.strip_prefix("model.safetensors-") {
                let fixed = format!("{}/model-{}", model_path, rest);
                if std::path::Path::new(&fixed).exists() {
                    log::info!(
                        "Fixed shard path: {} -> {}",
                        filename,
                        std::path::Path::new(&fixed)
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                    );
                    *path = fixed;
                    continue;
                }
            }
            return Err(anyhow::anyhow!("Shard file not found: {}", path));
        }
    }

    Ok((shard_files, weight_map))
}

// ============================================================================
// GGUF loading — dequantize to BF16 at load, reuse existing GEMV/GEMM kernels
// ============================================================================

/// Load a 1D tensor (e.g., norm weight) from a GGUF file.
///
/// Looks up the HuggingFace name in the GGUF tensor directory (after
/// reverse name mapping), dequantizes to BF16, uploads to GPU.
pub(crate) fn load_tensor_1d_gguf(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
) -> Result<DeviceVec> {
    let tensor = load_vector_bf16_host(gguf, hf_name)?;
    DeviceVec::from_host(ctx, &tensor.data)
}

/// Load a 1D norm weight from GGUF, subtracting 1.0 (offset RMSNorm correction).
///
/// GGUF stores norm weights with the +1 offset baked in: `w_gguf = 1 + w_hf`.
/// Our engine's offset RMSNorm computes `x * (1 + w)`, so we need `w = w_gguf - 1`
/// to avoid double-offset `x * (1 + w_gguf) = x * (2 + w_hf)`.
pub(crate) fn load_tensor_1d_gguf_offset_norm(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
) -> Result<DeviceVec> {
    let tensor = load_vector_offset_norm_bf16_host(gguf, hf_name)?;
    DeviceVec::from_host(ctx, &tensor.data)
}

fn reorder_packed_v_rows(
    src: &[u8],
    rows: usize,
    row_bytes: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
    hf_name: &str,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        src.len() == rows * row_bytes,
        "unexpected packed byte count for '{}': got {}, expected {}",
        hf_name,
        src.len(),
        rows * row_bytes
    );
    anyhow::ensure!(
        rows == num_k_heads * num_v_per_k * head_dim,
        "unexpected V-row count for '{}': got {}, expected {}",
        hf_name,
        rows,
        num_k_heads * num_v_per_k * head_dim
    );

    let mut dst = vec![0u8; src.len()];
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            let gguf_head = v * num_k_heads + k;
            let hf_head = k * num_v_per_k + v;
            let src_start = gguf_head * head_dim * row_bytes;
            let dst_start = hf_head * head_dim * row_bytes;
            let size = head_dim * row_bytes;
            dst[dst_start..dst_start + size].copy_from_slice(&src[src_start..src_start + size]);
        }
    }
    Ok(dst)
}

fn reorder_v_rows<T: Copy>(
    src: &[T],
    rows: usize,
    row_elems: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
    hf_name: &str,
) -> Result<Vec<T>> {
    anyhow::ensure!(
        src.len() == rows * row_elems,
        "unexpected row element count for '{}': got {}, expected {}",
        hf_name,
        src.len(),
        rows * row_elems
    );
    anyhow::ensure!(
        rows == num_k_heads * num_v_per_k * head_dim,
        "unexpected V-row count for '{}': got {}, expected {}",
        hf_name,
        rows,
        num_k_heads * num_v_per_k * head_dim
    );

    let mut dst = src.to_vec();
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            let gguf_head = v * num_k_heads + k;
            let hf_head = k * num_v_per_k + v;
            let src_start = gguf_head * head_dim * row_elems;
            let dst_start = hf_head * head_dim * row_elems;
            let len = head_dim * row_elems;
            dst[dst_start..dst_start + len].copy_from_slice(&src[src_start..src_start + len]);
        }
    }
    Ok(dst)
}

/// Load a 2D GGUF tensor with Qwen3.5 V-head row reorder reversal.
///
/// Q3_K/Q4_K/Q6_K can stay packed because the permutation moves whole rows,
/// preserving each row's 256-column superblock layout.
pub(crate) fn load_tensor_2d_gguf_v_reorder_rows(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) -> Result<DeviceMatrix> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];
    let (rows, cols) = if info.shape.len() == 2 {
        (info.shape[1] as usize, info.shape[0] as usize)
    } else {
        anyhow::bail!(
            "Expected 2D tensor for '{}', got {}D",
            hf_name,
            info.shape.len()
        );
    };

    let force_bf16 = std::env::var_os("INFER_FORCE_BF16_QUANT").is_some();
    if !force_bf16 && info.dtype == gguf::GgmlType::Q8_0 {
        let (mut qweight, mut scales, group_size) = gguf.read_tensor_q8_packed(&gguf_name)?;
        if num_v_per_k > 1 {
            qweight = reorder_v_rows(
                &qweight,
                rows,
                cols,
                num_k_heads,
                num_v_per_k,
                head_dim,
                hf_name,
            )?;
            scales = reorder_v_rows(
                &scales,
                rows,
                cols / group_size,
                num_k_heads,
                num_v_per_k,
                head_dim,
                hf_name,
            )?;
        }
        return DeviceMatrix::from_quantized_int8(ctx, &qweight, &scales, rows, cols, group_size);
    }

    if !force_bf16 && cols % 256 == 0 {
        if info.dtype == gguf::GgmlType::Q4_K {
            let packed = gguf.read_tensor_q4k_packed(&gguf_name)?;
            if num_v_per_k <= 1 {
                return DeviceMatrix::from_quantized_q4k(ctx, &packed, rows, cols);
            }
            let reordered = reorder_packed_v_rows(
                &packed,
                rows,
                cols * 9 / 16,
                num_k_heads,
                num_v_per_k,
                head_dim,
                hf_name,
            )?;
            return DeviceMatrix::from_quantized_q4k(ctx, &reordered, rows, cols);
        }

        if info.dtype == gguf::GgmlType::Q5_K {
            let packed = gguf.read_tensor_q5k_packed(&gguf_name)?;
            if num_v_per_k <= 1 {
                return DeviceMatrix::from_quantized_q5k(ctx, &packed, rows, cols);
            }
            let reordered = reorder_packed_v_rows(
                &packed,
                rows,
                cols * 11 / 16,
                num_k_heads,
                num_v_per_k,
                head_dim,
                hf_name,
            )?;
            return DeviceMatrix::from_quantized_q5k(ctx, &reordered, rows, cols);
        }

        if info.dtype == gguf::GgmlType::Q3_K {
            let packed = gguf.read_tensor_q3k_packed(&gguf_name)?;
            if num_v_per_k <= 1 {
                return DeviceMatrix::from_quantized_q3k(ctx, &packed, rows, cols);
            }
            let reordered = reorder_packed_v_rows(
                &packed,
                rows,
                cols * 55 / 128,
                num_k_heads,
                num_v_per_k,
                head_dim,
                hf_name,
            )?;
            return DeviceMatrix::from_quantized_q3k(ctx, &reordered, rows, cols);
        }

        if info.dtype == gguf::GgmlType::Q6_K {
            let packed = gguf.read_tensor_q6k_packed(&gguf_name)?;
            if num_v_per_k <= 1 {
                return DeviceMatrix::from_quantized_q6k(ctx, &packed, rows, cols);
            }
            let reordered = reorder_packed_v_rows(
                &packed,
                rows,
                cols * 210 / 256,
                num_k_heads,
                num_v_per_k,
                head_dim,
                hf_name,
            )?;
            return DeviceMatrix::from_quantized_q6k(ctx, &reordered, rows, cols);
        }
    }

    let tensor =
        load_matrix_v_reorder_rows_bf16_host(gguf, hf_name, num_k_heads, num_v_per_k, head_dim)?;
    DeviceMatrix::from_host(ctx, &tensor.data, rows, cols)
}

/// Load a 2D tensor (e.g., linear weight) from a GGUF file.
///
/// For Q8_0: keeps weights packed as INT8 + bf16 scales (uses W8A16 GEMV at runtime).
/// For other formats: dequantizes to BF16 at load time.
/// Load a 2D tensor from GGUF, ALWAYS as BF16 (dequantized). Used for tensors
/// that downstream ops read directly from `DeviceMatrix::data` instead of
/// the packed `qweight` buffer — most importantly `embed_tokens`, whose
/// lookup kernel is not quant-aware and would otherwise read from the
/// 1-element dummy `data` buffer of a quantized matrix.
pub(crate) fn load_tensor_2d_gguf_bf16(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
) -> Result<DeviceMatrix> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];
    let bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
    let (rows, cols) = if info.shape.len() == 2 {
        (info.shape[1] as usize, info.shape[0] as usize)
    } else if info.shape.len() == 1 {
        (1, info.shape[0] as usize)
    } else {
        anyhow::bail!(
            "Expected 1D or 2D tensor for '{}', got {}D",
            hf_name,
            info.shape.len()
        );
    };
    DeviceMatrix::from_host(ctx, &bf16_data, rows, cols)
}

pub(crate) fn load_tensor_2d_gguf(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
) -> Result<DeviceMatrix> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];

    // `INFER_FORCE_BF16_QUANT=1` skips all packed fast paths and forces the
    // BF16 dequant fallback. Kept behind an env var as a bisection tool for
    // "bug in native GPU kernel" vs "bug in downstream forward pass".
    let force_bf16 = std::env::var("INFER_FORCE_BF16_QUANT").is_ok();
    if force_bf16 && info.shape.len() == 2 {
        return load_tensor_2d_gguf_bf16(ctx, gguf, hf_name);
    }

    // Q8_0: keep packed — use existing W8A16 GEMV for on-the-fly dequant.
    if info.dtype == gguf::GgmlType::Q8_0 && info.shape.len() == 2 {
        let (qweight, scales, group_size) = gguf.read_tensor_q8_packed(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        return DeviceMatrix::from_quantized_int8(ctx, &qweight, &scales, rows, cols, group_size);
    }

    // Q4_K_M / Q4_K_S: keep packed — native q4k_gemv kernel.
    // Same column-major → row-major trick as Q8_0: superblocks of 256 live along
    // ne0 (the innermost dimension), so reinterpreting as [ne1, ne0] row-major
    // preserves superblock integrity.
    if info.dtype == gguf::GgmlType::Q4_K && info.shape.len() == 2 {
        let packed = gguf.read_tensor_q4k_packed(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        return DeviceMatrix::from_quantized_q4k(ctx, &packed, rows, cols);
    }

    // Q5_K: keep packed — native q5k_gemv kernel.
    if info.dtype == gguf::GgmlType::Q5_K && info.shape.len() == 2 {
        let packed = gguf.read_tensor_q5k_packed(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        return DeviceMatrix::from_quantized_q5k(ctx, &packed, rows, cols);
    }

    // Q3_K: keep packed — native q3k_gemv kernel.
    if info.dtype == gguf::GgmlType::Q3_K && info.shape.len() == 2 {
        let packed = gguf.read_tensor_q3k_packed(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        return DeviceMatrix::from_quantized_q3k(ctx, &packed, rows, cols);
    }

    // Q6_K: keep packed — native q6k_gemv kernel.
    if info.dtype == gguf::GgmlType::Q6_K && info.shape.len() == 2 {
        let packed = gguf.read_tensor_q6k_packed(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        return DeviceMatrix::from_quantized_q6k(ctx, &packed, rows, cols);
    }

    let bf16_data = gguf.read_tensor_bf16(&gguf_name)?;

    // GGUF 2D layout verified empirically: GGUF stores ne1 "rows" of ne0 elements
    // each in row-major order. data[i * ne0 + j] = element at (row=i, col=j).
    //
    // For weight matrices: ne0=in_dim, ne1=out_dim.
    // HuggingFace: [out_dim, in_dim] row-major = [ne1, ne0].
    // Since GGUF data[i * ne0 + j] directly maps to HF[i][j] with
    // rows=ne1, cols=ne0 — NO transpose needed.
    //
    // Verified: GGUF attn_q data[0] = HF q_proj[0,0], data[1] = HF q_proj[0,1].
    let (rows, cols) = if info.shape.len() == 2 {
        (info.shape[1] as usize, info.shape[0] as usize) // [ne1, ne0]
    } else if info.shape.len() == 1 {
        (1, info.shape[0] as usize)
    } else {
        anyhow::bail!(
            "Expected 1D or 2D tensor for '{}', got {}D",
            hf_name,
            info.shape.len()
        );
    };

    DeviceMatrix::from_host(ctx, &bf16_data, rows, cols)
}

#[cfg(test)]
mod gguf_v_reorder_tests {
    use super::reorder_packed_v_rows;

    #[test]
    fn packed_v_row_reorder_moves_whole_rows() {
        let rows = 12;
        let row_bytes = 3;
        let src = (0..rows)
            .flat_map(|row| [row as u8, 100 + row as u8, 200 + row as u8])
            .collect::<Vec<_>>();

        let dst = reorder_packed_v_rows(&src, rows, row_bytes, 2, 3, 2, "dummy")
            .expect("valid packed row reorder fixture");
        let dst_rows = dst
            .chunks_exact(row_bytes)
            .map(|row| row[0])
            .collect::<Vec<_>>();

        assert_eq!(dst_rows, [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    use crate::ops::OpsBackend;
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    use std::cell::RefCell;
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    use std::path::Path;
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    use std::sync::{Mutex, MutexGuard, OnceLock};

    const QWEN3_4B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
    const QWEN3_8B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-8B");
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    const QWEN3_4B_HYBRID_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/models/Qwen3-4B-W4-hybrid-zpfix"
    );

    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    struct HybridPrefillEnvGuard {
        old: Option<std::ffi::OsString>,
        _lock: MutexGuard<'static, ()>,
    }

    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    impl Drop for HybridPrefillEnvGuard {
        fn drop(&mut self) {
            // SAFETY: the guard holds the process-local env test mutex while
            // restoring this variable, serializing all mutations in this module.
            unsafe {
                if let Some(old) = &self.old {
                    std::env::set_var("INFER_HYBRID_W4A8_PREFILL", old);
                } else {
                    std::env::remove_var("INFER_HYBRID_W4A8_PREFILL");
                }
            }
        }
    }

    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    fn hybrid_prefill_env(value: Option<&str>) -> HybridPrefillEnvGuard {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let old = std::env::var_os("INFER_HYBRID_W4A8_PREFILL");
        // SAFETY: the mutex above serializes all mutations of this variable in
        // these tests. The test only needs process-wide env because the dispatch
        // gate intentionally reads the runtime env on each call.
        unsafe {
            if let Some(value) = value {
                std::env::set_var("INFER_HYBRID_W4A8_PREFILL", value);
            } else {
                std::env::remove_var("INFER_HYBRID_W4A8_PREFILL");
            }
        }
        HybridPrefillEnvGuard { old, _lock: lock }
    }

    #[test]
    fn quant_layout_uses_configured_bits_and_infers_group_size() {
        let cfg = QuantLoadConfig {
            group_size: None,
            bits: Some(4),
            ..QuantLoadConfig::default()
        };
        let (orig_k, group_size, bits) = detect_uniform_quant_layout("w", 64, 2, cfg).unwrap();
        assert_eq!((orig_k, group_size, bits), (128, 64, 4));
    }

    #[test]
    fn turboquant_layout_requires_explicit_bits() {
        let err = detect_turboquant_layout(
            "w",
            64,
            1,
            QuantLoadConfig {
                group_size: Some(128),
                bits: None,
                ..QuantLoadConfig::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires explicit bits"));
    }

    #[test]
    fn turboquant_layout_accepts_tq2_tq3_tq4() {
        for (bits, packed_cols) in [(2, 32), (3, 64), (4, 64)] {
            let (orig_k, group_size, got_bits) = detect_turboquant_layout(
                "w",
                packed_cols,
                2,
                QuantLoadConfig {
                    group_size: Some(64),
                    bits: None,
                    tq_bits: Some(bits),
                    ..QuantLoadConfig::default()
                },
            )
            .unwrap();
            assert_eq!((orig_k, group_size, got_bits), (128, 64, bits));
        }
    }

    #[test]
    fn quant_config_rejects_zero_point_layouts_before_symmetric_loader() {
        let awq =
            QuantLoadConfig::from_meta(&crate::quant::QuantMeta::Awq(crate::quant::AwqConfig {
                bits: 4,
                group_size: 128,
                zero_point: true,
                version: crate::quant::AwqVersion::Gemm,
            }));
        assert!(awq.enabled());
        assert!(
            awq.unsupported_reason
                .expect("zero-point AWQ must be rejected")
                .contains("AWQ")
        );

        let gptq =
            QuantLoadConfig::from_meta(&crate::quant::QuantMeta::Gptq(crate::quant::GptqConfig {
                bits: 4,
                group_size: 128,
                desc_act: false,
                sym: false,
                checkpoint_format: None,
            }));
        assert!(gptq.enabled());
        assert!(
            gptq.unsupported_reason
                .expect("asymmetric GPTQ must be rejected")
                .contains("GPTQ")
        );
    }

    #[test]
    fn quant_config_detects_marlin_w4_hybrid() {
        let hybrid = QuantLoadConfig::from_meta(&crate::quant::QuantMeta::MarlinW4Hybrid(
            crate::quant::MarlinW4A8Config { group_size: 128 },
        ));
        assert!(hybrid.enabled());
        assert!(hybrid.marlin_w4_hybrid);
        assert!(hybrid.marlin_w4a8);
        assert_eq!(hybrid.bits, Some(4));
        assert_eq!(hybrid.group_size, Some(128));
    }

    #[test]
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    fn load_hybrid_w4_marlin_linear_populates_side_tensors() -> Result<()> {
        if !Path::new(QWEN3_4B_HYBRID_PATH).exists() {
            eprintln!("skipping hybrid loader test: {QWEN3_4B_HYBRID_PATH} is absent");
            return Ok(());
        }

        let ctx = DeviceContext::new()?;
        let (shard_paths, weight_map) = load_shard_info(QWEN3_4B_HYBRID_PATH)?;
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = mmaps
            .iter()
            .map(|mmap| {
                SafeTensors::deserialize(mmap)
                    .map_err(|e| anyhow::anyhow!("Deserialize error: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let config = QuantLoadConfig::from_model_path(QWEN3_4B_HYBRID_PATH)?;
        assert!(config.marlin_w4_hybrid);
        assert!(config.marlin_w4a8);

        let matrix = load_tensor_2d_maybe_quantized_with_config(
            &ctx,
            &shards,
            &weight_map,
            "model.layers.0.mlp.gate_proj.weight",
            config,
        )?;

        assert!(matrix.has_marlin());
        assert!(matrix.is_hybrid_w4_marlin());
        assert!(matrix.hybrid_w4a8_qweight.is_some());
        assert!(matrix.hybrid_w4a8_s_channel.is_some());
        assert!(matrix.hybrid_w4a8_s_group.is_some());
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    fn load_hybrid_w4_marlin_dispatches_to_w4a8_prefill() -> Result<()> {
        if !Path::new(QWEN3_4B_HYBRID_PATH).exists() {
            eprintln!("skipping hybrid dispatch test: {QWEN3_4B_HYBRID_PATH} is absent");
            return Ok(());
        }

        let ctx = DeviceContext::new()?;
        let (shard_paths, weight_map) = load_shard_info(QWEN3_4B_HYBRID_PATH)?;
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = mmaps
            .iter()
            .map(|mmap| {
                SafeTensors::deserialize(mmap)
                    .map_err(|e| anyhow::anyhow!("Deserialize error: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let config = QuantLoadConfig::from_model_path(QWEN3_4B_HYBRID_PATH)?;
        let matrix = load_tensor_2d_maybe_quantized_with_config(
            &ctx,
            &shards,
            &weight_map,
            "model.layers.0.mlp.gate_proj.weight",
            config,
        )?;
        assert!(matrix.is_hybrid_w4_marlin());

        let host = vec![bf16::from_f32(0.125); matrix.cols * 16];
        let data = ctx
            .stream
            .clone_htod(&host)
            .map_err(|e| anyhow::anyhow!("H2D test hidden states failed: {e}"))?;
        let input = cuda_kernels::prelude::HiddenStates {
            data,
            hidden_dim: matrix.cols,
            seq_len: 16,
        };

        {
            let _env = hybrid_prefill_env(None);
            let decode_backend = crate::ops::CudaOpsBackend::new(&ctx);
            let decode_host = vec![bf16::from_f32(0.125); matrix.cols];
            let decode_input = DeviceVec::from_host(&ctx, &decode_host)?;
            let mut decode_out = DeviceVec::zeros(&ctx, matrix.rows)?;
            decode_backend.linear_vec_into(&matrix, &decode_input, &mut decode_out)?;
            ctx.sync()?;

            let prefill_backend = crate::ops::CudaOpsBackend::prefill(&ctx);
            let mut out =
                cuda_kernels::prelude::HiddenStates::zeros(&ctx, matrix.rows, input.seq_len)?;
            let err = match prefill_backend.linear_batch_into(&matrix, &input, &mut out) {
                Ok(_) => {
                    anyhow::bail!("default-off hybrid prefill dispatch unexpectedly succeeded")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string().contains("INFER_HYBRID_W4A8_PREFILL=1"),
                "unexpected default-off hybrid dispatch error: {err}"
            );
        }

        {
            let _env = hybrid_prefill_env(Some("1"));
            assert_eq!(
                crate::ops::linear_kernel_plan_for_test(&matrix, 2, false),
                "MarlinW4Gemm"
            );
            assert_eq!(
                crate::ops::linear_kernel_plan_for_test(&matrix, 1, true),
                "MarlinW4Gemm"
            );
            assert_eq!(
                crate::ops::linear_kernel_plan_for_test(&matrix, 2, true),
                "MarlinW4Hybrid"
            );
            let prefill_backend = crate::ops::CudaOpsBackend::prefill(&ctx);
            let mut out =
                cuda_kernels::prelude::HiddenStates::zeros(&ctx, matrix.rows, input.seq_len)?;
            prefill_backend.linear_batch_into(&matrix, &input, &mut out)?;
            assert_eq!(out.hidden_dim, matrix.rows);
            assert_eq!(out.seq_len, input.seq_len);
            ctx.sync()?;
        }

        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    fn load_hybrid_w4_marlin_decode_accepts_preallocated_marlin_scratch() -> Result<()> {
        if !Path::new(QWEN3_4B_HYBRID_PATH).exists() {
            eprintln!("skipping hybrid scratch test: {QWEN3_4B_HYBRID_PATH} is absent");
            return Ok(());
        }

        let ctx = DeviceContext::new()?;
        let (shard_paths, weight_map) = load_shard_info(QWEN3_4B_HYBRID_PATH)?;
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = mmaps
            .iter()
            .map(|mmap| {
                SafeTensors::deserialize(mmap)
                    .map_err(|e| anyhow::anyhow!("Deserialize error: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let config = QuantLoadConfig::from_model_path(QWEN3_4B_HYBRID_PATH)?;
        let matrix = load_tensor_2d_maybe_quantized_with_config(
            &ctx,
            &shards,
            &weight_map,
            "model.layers.0.mlp.gate_proj.weight",
            config,
        )?;
        assert!(matrix.is_hybrid_w4_marlin());

        let seq_len = 2;
        let host = vec![bf16::from_f32(0.125); matrix.cols * seq_len];
        let data = ctx
            .stream
            .clone_htod(&host)
            .map_err(|e| anyhow::anyhow!("H2D scratch test hidden states failed: {e}"))?;
        let input = cuda_kernels::prelude::HiddenStates {
            data,
            hidden_dim: matrix.cols,
            seq_len,
        };
        let mut out = cuda_kernels::prelude::HiddenStates::zeros(&ctx, matrix.rows, seq_len)?;
        let scratch = RefCell::new(crate::ops::MarlinDecodeScratch::new(
            &ctx,
            seq_len,
            matrix.cols.max(matrix.rows),
            matrix.cols.max(matrix.rows),
            crate::ops::MarlinDecodeScratchConfig::new(true, false),
        )?);
        let backend = crate::ops::CudaOpsBackend::decode_with_marlin_scratch(&ctx, &scratch);

        assert_eq!(
            crate::ops::linear_kernel_plan_for_test(&matrix, seq_len, false),
            "MarlinW4Gemm"
        );
        backend.linear_batch_into(&matrix, &input, &mut out)?;
        assert_eq!(out.hidden_dim, matrix.rows);
        assert_eq!(out.seq_len, seq_len);
        ctx.sync()?;
        Ok(())
    }

    fn bf16_matrix_bytes(rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for value in 0..(rows * cols) {
            bytes.extend_from_slice(&bf16::from_f32(value as f32).to_le_bytes());
        }
        bytes
    }

    fn sum_bf16(values: &[bf16]) -> f32 {
        values.iter().map(|value| value.to_f32()).sum()
    }

    #[test]
    fn fp8_weight_scale_inv_dequantizes_block_scaled_matrix() {
        let weights = [0x38, 0xb8, 0x40, 0xc0]; // 1, -1, 2, -2 in E4M3FN
        let mut scales = Vec::new();
        scales.extend_from_slice(&bf16::from_f32(2.0).to_le_bytes());
        scales.extend_from_slice(&bf16::from_f32(0.5).to_le_bytes());

        let out = dequantize_fp8_e4m3_weight_scale_inv_to_bf16_host(
            "test.weight",
            &weights,
            2,
            2,
            &scales,
            Dtype::BF16,
            &[2, 1],
            1,
            2,
        )
        .unwrap();
        let values = out.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
        assert_eq!(values, vec![2.0, -2.0, 1.0, -1.0]);
    }

    #[test]
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    fn fp8_weight_scale_inv_loads_single_file_without_weight_map() -> Result<()> {
        use safetensors::tensor::{TensorView, serialize};

        let ctx = DeviceContext::new()?;
        let weights = [0x38, 0xb8, 0x40, 0xc0]; // 1, -1, 2, -2 in E4M3FN
        let mut scales = Vec::new();
        scales.extend_from_slice(&bf16::from_f32(2.0).to_le_bytes());
        scales.extend_from_slice(&bf16::from_f32(0.5).to_le_bytes());
        let weight = TensorView::new(Dtype::F8_E4M3, vec![2, 2], &weights)?;
        let scale = TensorView::new(Dtype::BF16, vec![2, 1], &scales)?;
        let buf = serialize(
            vec![
                ("layer.weight".to_string(), weight),
                ("layer.weight_scale_inv".to_string(), scale),
            ],
            None,
        )?;
        let shards = vec![SafeTensors::deserialize(&buf)?];
        let weight_map = HashMap::new();

        let matrix = load_tensor_2d_maybe_quantized_with_config(
            &ctx,
            &shards,
            &weight_map,
            "layer.weight",
            QuantLoadConfig {
                fp8_weight_scale_inv: true,
                fp8_block_rows: 1,
                fp8_block_cols: 2,
                ..QuantLoadConfig::default()
            },
        )?;
        assert_eq!((matrix.rows, matrix.cols), (2, 2));
        let host = ctx
            .stream
            .clone_dtoh(&matrix.data)
            .map_err(|e| anyhow::anyhow!("DTOH FP8 single-file fixture failed: {e}"))?;
        ctx.sync()?;
        let values = host.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
        assert_eq!(values, vec![2.0, -2.0, 1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn tp_column_shards_cover_full_bf16_matrix() {
        let rows = 4;
        let cols = 6;
        let bytes = bf16_matrix_bytes(rows, cols);
        let rank0 = TpLoadContext::column(0, 2, rows).unwrap();
        let rank1 = TpLoadContext::column(1, 2, rows).unwrap();

        let (shard0, rows0, cols0) = shard_bf16_matrix_host(&bytes, rows, cols, &rank0).unwrap();
        let (shard1, rows1, cols1) = shard_bf16_matrix_host(&bytes, rows, cols, &rank1).unwrap();

        assert_eq!((rows0, cols0), (2, 6));
        assert_eq!((rows1, cols1), (2, 6));
        assert_eq!(
            sum_bf16(&shard0) + sum_bf16(&shard1),
            (0..24).sum::<i32>() as f32
        );
    }

    #[test]
    fn tp_row_shards_cover_full_bf16_matrix() {
        let rows = 5;
        let cols = 4;
        let bytes = bf16_matrix_bytes(rows, cols);
        let rank0 = TpLoadContext::row(0, 2, cols).unwrap();
        let rank1 = TpLoadContext::row(1, 2, cols).unwrap();

        let (shard0, rows0, cols0) = shard_bf16_matrix_host(&bytes, rows, cols, &rank0).unwrap();
        let (shard1, rows1, cols1) = shard_bf16_matrix_host(&bytes, rows, cols, &rank1).unwrap();

        assert_eq!((rows0, cols0), (5, 2));
        assert_eq!((rows1, cols1), (5, 2));
        assert_eq!(
            sum_bf16(&shard0) + sum_bf16(&shard1),
            (0..20).sum::<i32>() as f32
        );
    }

    #[test]
    fn test_load_shard_info_for_tied_qwen3_4b() {
        let (shards, weight_map) = load_shard_info(QWEN3_4B_PATH).unwrap();

        assert_eq!(shards.len(), 3);
        assert!(weight_map.contains_key("model.embed_tokens.weight"));
        assert!(!weight_map.contains_key("lm_head.weight"));
    }

    #[test]
    #[ignore = "requires Qwen3-8B model"]
    fn test_load_shard_info_for_untied_qwen3_8b() {
        let (shards, weight_map) = load_shard_info(QWEN3_8B_PATH).unwrap();

        assert_eq!(shards.len(), 5);
        assert!(weight_map.contains_key("model.embed_tokens.weight"));
        assert!(weight_map.contains_key("lm_head.weight"));
    }
}
