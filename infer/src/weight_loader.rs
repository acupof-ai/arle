//! Safetensors and GGUF weight loading + RoPE precomputation.
//!
//! Two loading paths:
//! - **Safetensors** (default): `load_tensor_1d`, `load_tensor_2d`, `load_tensor_2d_maybe_quantized`
//! - **GGUF**: `load_tensor_1d_gguf`, `load_tensor_2d_gguf` — dequant to BF16 at load time

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::info;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use crate::gguf::{self, GgufFile};
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
        let orig_k = num_groups * group_size;

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
        let bits = 3u8; // TODO: detect from turboquant_config.json
        let num_levels = 1usize << bits;
        let mut centroids_host = vec![0.0f32; num_levels];
        let mut boundaries_host = vec![0.0f32; num_levels + 1];
        unsafe {
            crate::ffi::turboquant_lloyd_max(
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
            std::slice::from_raw_parts(
                scales.as_ptr().cast::<u8>(),
                scales.len() * std::mem::size_of::<half::f16>(),
            )
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

    // Fallback: bf16
    load_tensor_2d(ctx, shards, weight_map, name)
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
        crate::ffi::turboquant_lloyd_max(
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
    let gguf_name = find_gguf_tensor_name(gguf, hf_name)?;
    let bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
    DeviceVec::from_host(ctx, &bf16_data)
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
    let gguf_name = find_gguf_tensor_name(gguf, hf_name)?;
    let mut bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
    // Subtract 1.0 to convert from GGUF convention (1+w) to HF convention (w)
    for v in &mut bf16_data {
        *v = bf16::from_f32(v.to_f32() - 1.0);
    }
    DeviceVec::from_host(ctx, &bf16_data)
}

/// Reverse llama.cpp's `_reorder_v_heads` permutation for 1D data.
///
/// llama.cpp reorders V heads from grouped (by K head) to tiled:
///   HF:   [G0_v0, G0_v1, G1_v0, G1_v1, ...]  (grouped by K head)
///   GGUF: [G0_v0, G1_v0, G0_v1, G1_v1, ...]  (tiled for ggml broadcast)
///
/// This reverses: reshape [num_v_per_k, num_k_heads, head_dim] → transpose → flatten.
/// For 1D tensors (A_log, dt_bias): head_dim=1.
pub(crate) fn reverse_v_reorder(
    data: &mut [bf16],
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    // GGUF order: [vpk, nk, hd] flattened. Reverse to [nk, vpk, hd].
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            for d in 0..head_dim {
                let gguf_idx = (v * num_k_heads + k) * head_dim + d;
                let hf_idx = (k * num_v_per_k + v) * head_dim + d;
                data[hf_idx] = src[gguf_idx];
            }
        }
    }
}

/// Same as reverse_v_reorder but for f32 data.
pub(crate) fn reverse_v_reorder_f32(
    data: &mut [f32],
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            for d in 0..head_dim {
                let gguf_idx = (v * num_k_heads + k) * head_dim + d;
                let hf_idx = (k * num_v_per_k + v) * head_dim + d;
                data[hf_idx] = src[gguf_idx];
            }
        }
    }
}

/// Reverse llama.cpp's `_reorder_v_heads` for 2D tensor rows.
///
/// Rows are grouped as [num_v_per_k, num_k_heads, head_dim, cols] in GGUF.
/// Reverse to [num_k_heads, num_v_per_k, head_dim, cols] (HF grouped order).
pub(crate) fn reverse_v_reorder_rows(
    data: &mut [bf16],
    rows: usize,
    cols: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    let _ = rows; // used for assertion only
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            let gguf_head = v * num_k_heads + k;
            let hf_head = k * num_v_per_k + v;
            let src_start = gguf_head * head_dim * cols;
            let dst_start = hf_head * head_dim * cols;
            let size = head_dim * cols;
            data[dst_start..dst_start + size].copy_from_slice(&src[src_start..src_start + size]);
        }
    }
}

/// Reverse llama.cpp's `_reorder_v_heads` for 2D tensor columns.
/// Used for out_proj where cols = num_v_heads × value_head_dim.
pub(crate) fn reverse_v_reorder_cols(
    data: &mut [bf16],
    rows: usize,
    cols: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    for r in 0..rows {
        for k in 0..num_k_heads {
            for v in 0..num_v_per_k {
                let gguf_head = v * num_k_heads + k;
                let hf_head = k * num_v_per_k + v;
                for d in 0..head_dim {
                    data[r * cols + hf_head * head_dim + d] =
                        src[r * cols + gguf_head * head_dim + d];
                }
            }
        }
    }
}

/// Load 1D GGUF tensor with V-head reorder reversal (a_log, dt_bias).
pub(crate) fn load_tensor_1d_gguf_v_reorder(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
) -> Result<DeviceVec> {
    let gguf_name = find_gguf_tensor_name(gguf, hf_name)?;
    let mut bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
    reverse_v_reorder(&mut bf16_data, num_k_heads, num_v_per_k, 1);
    DeviceVec::from_host(ctx, &bf16_data)
}

/// Load 2D GGUF tensor with V-head row reorder reversal (in_proj_z, in_proj_a/b).
///
/// For Q4_K_M / Q4_K_S, the permutation happens on packed superblock bytes at
/// row granularity — each row's `(cols/256) * 144` bytes move as one unit,
/// preserving superblock integrity. No BF16 intermediate is materialised.
pub(crate) fn load_tensor_2d_gguf_v_reorder_rows(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) -> Result<DeviceMatrix> {
    let gguf_name = find_gguf_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];
    let (rows, cols) = if info.shape.len() == 2 {
        (info.shape[1] as usize, info.shape[0] as usize)
    } else {
        (1, info.shape[0] as usize)
    };

    // Q4_K packed: byte-level row permutation.
    if info.dtype == gguf::GgmlType::Q4_K && info.shape.len() == 2 && cols % 256 == 0 {
        let src = gguf.read_tensor_q4k_packed(&gguf_name)?;
        let row_bytes = cols * 9 / 16; // (cols/256) * 144
        assert_eq!(src.len(), rows * row_bytes);
        if num_v_per_k <= 1 {
            return DeviceMatrix::from_quantized_q4k(ctx, &src, rows, cols);
        }
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
        drop(src);
        return DeviceMatrix::from_quantized_q4k(ctx, &dst, rows, cols);
    }

    // Q3_K packed: same byte-level row permutation at 110-byte stride.
    if info.dtype == gguf::GgmlType::Q3_K && info.shape.len() == 2 && cols % 256 == 0 {
        let src = gguf.read_tensor_q3k_packed(&gguf_name)?;
        let row_bytes = cols * 55 / 128; // (cols/256) * 110
        assert_eq!(src.len(), rows * row_bytes);
        if num_v_per_k <= 1 {
            return DeviceMatrix::from_quantized_q3k(ctx, &src, rows, cols);
        }
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
        drop(src);
        return DeviceMatrix::from_quantized_q3k(ctx, &dst, rows, cols);
    }

    // Q6_K packed: same byte-level row permutation at 210-byte stride.
    if info.dtype == gguf::GgmlType::Q6_K && info.shape.len() == 2 && cols % 256 == 0 {
        let src = gguf.read_tensor_q6k_packed(&gguf_name)?;
        let row_bytes = cols * 210 / 256;
        assert_eq!(src.len(), rows * row_bytes);
        if num_v_per_k <= 1 {
            return DeviceMatrix::from_quantized_q6k(ctx, &src, rows, cols);
        }
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
        drop(src);
        return DeviceMatrix::from_quantized_q6k(ctx, &dst, rows, cols);
    }

    // BF16 fallback (existing path).
    let mut bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
    reverse_v_reorder_rows(
        &mut bf16_data,
        rows,
        cols,
        num_k_heads,
        num_v_per_k,
        head_dim,
    );
    DeviceMatrix::from_host(ctx, &bf16_data, rows, cols)
}

/// Load 2D GGUF tensor with V-head column reorder reversal (out_proj).
pub(crate) fn load_tensor_2d_gguf_v_reorder_cols(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) -> Result<DeviceMatrix> {
    let gguf_name = find_gguf_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];
    let mut bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
    let (rows, cols) = if info.shape.len() == 2 {
        (info.shape[1] as usize, info.shape[0] as usize)
    } else {
        (1, info.shape[0] as usize)
    };
    reverse_v_reorder_cols(
        &mut bf16_data,
        rows,
        cols,
        num_k_heads,
        num_v_per_k,
        head_dim,
    );
    DeviceMatrix::from_host(ctx, &bf16_data, rows, cols)
}

/// Load a 2D tensor (e.g., linear weight) from a GGUF file.
///
/// For Q8_0: keeps weights packed as INT8 + bf16 scales (uses W8A16 GEMV at runtime).
/// For other formats: dequantizes to BF16 at load time.
pub(crate) fn load_tensor_2d_gguf(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    hf_name: &str,
) -> Result<DeviceMatrix> {
    let gguf_name = find_gguf_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];

    // `PEGAINFER_FORCE_BF16_QUANT=1` skips all packed fast paths and forces the
    // BF16 dequant fallback. Kept behind an env var as a bisection tool for
    // "bug in native GPU kernel" vs "bug in downstream forward pass".
    let force_bf16 = std::env::var("PEGAINFER_FORCE_BF16_QUANT").is_ok();
    if force_bf16 && info.shape.len() == 2 {
        let bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        return DeviceMatrix::from_host(ctx, &bf16_data, ne1, ne0);
    }

    // Q8_0: keep packed — use existing W8A16 GEMV for on-the-fly dequant.
    // GGUF column-major: [ne0, ne1] stores data column-by-column.
    // Reading as row-major gives [ne1, ne0] = [out_dim, in_dim] — no transpose.
    // Q8_0 blocks are stored per-column (32 elements each), so the natural
    // column-major → row-major reinterpretation preserves block alignment.
    if info.dtype == gguf::GgmlType::Q8_0 && info.shape.len() == 2 {
        let (qweight, scales, group_size) = gguf.read_tensor_q8_packed(&gguf_name)?;
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0); // Column-major → row-major: [ne1, ne0]
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

/// Find a tensor in the GGUF file by HuggingFace name.
///
/// Tries: direct lookup → reverse name mapping → prefix stripping.
fn find_gguf_tensor_name(gguf: &GgufFile, hf_name: &str) -> Result<String> {
    // Direct match (some GGUF files use HF naming)
    if gguf.tensors.contains_key(hf_name) {
        return Ok(hf_name.to_string());
    }

    // Reverse mapping: try multiple prefixes (Qwen3 uses "model", Qwen3.5 uses "model.language_model")
    let prefixes = ["model", "model.language_model"];
    for gguf_name in gguf.tensors.keys() {
        for prefix in &prefixes {
            if gguf::map_gguf_name_with_prefix(gguf_name, prefix) == hf_name {
                return Ok(gguf_name.clone());
            }
        }
    }

    anyhow::bail!(
        "Tensor '{}' not found in GGUF (tried {} tensor names with prefixes {:?})",
        hf_name,
        gguf.tensors.len(),
        prefixes
    )
}

/// Public version of find_gguf_tensor_name for use by model-specific loaders.
pub(crate) fn find_gguf_tensor_name_pub(gguf: &GgufFile, hf_name: &str) -> Result<String> {
    find_gguf_tensor_name(gguf, hf_name)
}

/// Detect if a model path contains a GGUF file and return the GgufFile handle.
///
/// Scans for `.gguf` files in the directory. Returns the first one found.
pub(crate) fn try_open_gguf(model_path: &str) -> Option<GgufFile> {
    let dir = std::path::Path::new(model_path);
    if !dir.is_dir() {
        return None;
    }
    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".gguf") {
            let path = entry.path();
            match GgufFile::open(path.to_str()?) {
                Ok(gguf) => return Some(gguf),
                Err(e) => {
                    log::warn!("Failed to parse GGUF {}: {}", path.display(), e);
                }
            }
        }
    }
    None
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
