//! Safetensors weight loading and RoPE precomputation.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::info;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

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
        .map(|d| d.to_string())
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

/// Load a 2D tensor, trying quantized (.qweight + .scales) first, then bf16.
///
/// If `name` = "model.layers.0.self_attn.q_proj.weight", tries:
///   1. "model.layers.0.self_attn.q_proj.qweight" + ".scales" → INT8 quantized
///   2. "model.layers.0.self_attn.q_proj.weight" → bf16
pub(crate) fn load_tensor_2d_maybe_quantized(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    group_size: usize,
) -> Result<DeviceMatrix> {
    // Try quantized path: replace ".weight" with ".qweight"
    let qweight_name = name.replace(".weight", ".qweight");
    let scales_name = name.replace(".weight", ".scales");

    if weight_map.contains_key(&qweight_name) && weight_map.contains_key(&scales_name) {
        let qw_tensor = find_tensor(shards, weight_map, &qweight_name)?;
        let sc_tensor = find_tensor(shards, weight_map, &scales_name)?;

        let qw_shape = qw_tensor.shape();
        let sc_shape = sc_tensor.shape();
        let rows = qw_shape[0];
        let qw_cols = qw_shape[1];
        let num_groups = sc_shape[1];
        let orig_k = num_groups * group_size;

        let sc_data: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(
                sc_tensor.data().as_ptr().cast::<half::bf16>(),
                sc_shape[0] * sc_shape[1],
            )
        };

        // Detect bit width from packed shape: INT2 (K/4), INT4 (K/2), INT8 (K)
        if qw_cols == orig_k / 4 {
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
        if qw_cols == orig_k / 2 {
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
            // Repack for Marlin prefill GEMM (skips if dimensions not aligned)
            if let Err(e) = mat.repack_for_marlin(ctx) {
                log::warn!("Marlin repack failed for {}: {}", name, e);
            }
            return Ok(mat);
        }

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

    // Fallback: bf16
    load_tensor_2d(ctx, shards, weight_map, name)
}

/// Precompute RoPE cos/sin cache as contiguous GPU buffers.
/// Layout: [max_seq_len * head_dim] — position `pos` at offset `pos * head_dim`.
pub(crate) fn precompute_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_4B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
    const QWEN3_8B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-8B");

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
