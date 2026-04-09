use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

#[cfg(feature = "metal")]
use crate::mlx::MlxArray;

use super::{QuantConfig, WeightTensor};

#[cfg(feature = "metal")]
pub(super) type TensorMap = HashMap<String, MlxArray>;

#[cfg(feature = "metal")]
pub(super) fn load_tensor_map(model_dir: &Path) -> Result<TensorMap> {
    use std::ffi::CString;

    let shards = collect_safetensors_shards(model_dir)?;
    anyhow::ensure!(
        !shards.is_empty(),
        "no .safetensors shards found in {}",
        model_dir.display()
    );

    let mut tensors = TensorMap::new();
    log::info!(
        "  loading {} shard(s) via mlx_load_safetensors (mmap) …",
        shards.len()
    );

    for shard in &shards {
        let path_c = CString::new(shard.to_str().unwrap_or("")).unwrap();
        // mlx_load_safetensors requires a CPU stream.
        let cpu_dev = unsafe { mlx_sys::mlx_device_new_type(mlx_sys::mlx_device_type__MLX_CPU, 0) };
        let mut stream = unsafe { mlx_sys::mlx_stream_new() };
        unsafe {
            mlx_sys::mlx_get_default_stream(&mut stream, cpu_dev);
        }
        unsafe {
            mlx_sys::mlx_device_free(cpu_dev);
        }
        let mut map = unsafe { mlx_sys::mlx_map_string_to_array_new() };
        let mut meta = unsafe { mlx_sys::mlx_map_string_to_string_new() };

        let ret =
            unsafe { mlx_sys::mlx_load_safetensors(&mut map, &mut meta, path_c.as_ptr(), stream) };
        if ret != 0 {
            unsafe {
                mlx_sys::mlx_map_string_to_array_free(map);
                mlx_sys::mlx_map_string_to_string_free(meta);
            }
            anyhow::bail!("mlx_load_safetensors failed for {}", shard.display());
        }

        // Iterate over the map to extract tensors.
        let it = unsafe { mlx_sys::mlx_map_string_to_array_iterator_new(map) };
        loop {
            let mut key_ptr: *const std::os::raw::c_char = std::ptr::null();
            let mut val = unsafe { mlx_sys::mlx_array_new() };
            let done = unsafe {
                mlx_sys::mlx_map_string_to_array_iterator_next(&mut key_ptr, &mut val, it)
            };
            if done != 0 {
                unsafe {
                    mlx_sys::mlx_array_free(val);
                }
                break;
            }
            let name = unsafe {
                std::ffi::CStr::from_ptr(key_ptr)
                    .to_string_lossy()
                    .to_string()
            };
            tensors.insert(name, unsafe { MlxArray::from_raw(val) });
        }
        unsafe {
            mlx_sys::mlx_map_string_to_array_iterator_free(it);
            mlx_sys::mlx_map_string_to_array_free(map);
            mlx_sys::mlx_map_string_to_string_free(meta);
        }
    }

    log::info!("  loaded {} tensors (memory-mapped)", tensors.len());
    Ok(tensors)
}

#[cfg(feature = "metal")]
pub(super) fn tensor_get(tensors: &TensorMap, name: &str) -> Result<MlxArray> {
    tensors
        .get(name)
        .cloned()
        .with_context(|| format!("missing weight '{name}'"))
}

#[cfg(feature = "metal")]
pub(super) fn load_proj_from_tensors(
    tensors: &TensorMap,
    base: &str,
    quantization: Option<QuantConfig>,
) -> Result<WeightTensor> {
    use crate::mlx::{eval, transpose_all};

    if let Some(qc) = quantization {
        let scales_key = format!("{base}.scales");
        if let Some(scales) = tensors.get(&scales_key).cloned() {
            let w = tensors
                .get(&format!("{base}.weight"))
                .cloned()
                .with_context(|| format!("missing quantized weight '{base}.weight'"))?;
            let biases = tensors
                .get(&format!("{base}.biases"))
                .cloned()
                .with_context(|| format!("missing quantized biases '{base}.biases'"))?;
            return Ok(WeightTensor::Quantized {
                w,
                scales,
                biases,
                group_size: qc.group_size,
                bits: qc.bits,
            });
        }
    }

    let w = tensor_get(tensors, &format!("{base}.weight"))?;
    let w_t = transpose_all(&w);
    eval(&[&w_t]);
    Ok(WeightTensor::Dense(w_t))
}

#[cfg(feature = "metal")]
pub(super) fn load_embed_tokens_from_tensors(
    tensors: &TensorMap,
    base: &str,
    quantization: Option<QuantConfig>,
) -> Result<MlxArray> {
    let w = tensor_get(tensors, &format!("{base}.weight"))?;
    if let Some(qc) = quantization {
        if let Some(scales) = tensors.get(&format!("{base}.scales")).cloned() {
            let biases = tensors
                .get(&format!("{base}.biases"))
                .cloned()
                .with_context(|| format!("missing biases '{base}.biases'"))?;
            log::info!("  dequantizing embed_tokens at load time");
            return Ok(crate::mlx::dequantize(
                &w,
                &scales,
                &biases,
                qc.group_size,
                qc.bits,
            ));
        }
    }
    Ok(w)
}

#[cfg(feature = "metal")]
pub(super) fn tie_lm_head_from_embed_tokens(embed_tokens: &MlxArray) -> Result<WeightTensor> {
    use crate::mlx::{eval, transpose_all};

    let w_t = transpose_all(embed_tokens);
    eval(&[&w_t]);
    Ok(WeightTensor::Dense(w_t))
}

/// Collect safetensors shard paths. Prefers `model.safetensors.index.json` when
/// present — this avoids loading stray `.safetensors` files that happen to exist
/// in the directory (e.g. leftover shards from a different conversion).
fn collect_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    // Try the index file first.
    let index_path = model_dir.join("model.safetensors.index.json");
    if let Ok(raw) = std::fs::read_to_string(&index_path) {
        let v: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        if let Some(map) = v.get("weight_map").and_then(serde_json::Value::as_object) {
            let mut files: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            files.sort();
            files.dedup();
            let shards: Vec<PathBuf> = files.iter().map(|f| model_dir.join(f)).collect();
            log::info!(
                "  using index: {} shards from {}",
                shards.len(),
                index_path.display()
            );
            return Ok(shards);
        }
    }

    // Fallback: glob all .safetensors files, sorted by name.
    let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("read_dir {}", model_dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "safetensors")
        })
        .collect();
    shards.sort();
    Ok(shards)
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use std::collections::BTreeMap;

    use crate::test_support::metal_test_guard;
    use safetensors::{Dtype, serialize_to_file, tensor::TensorView};
    use tempfile::tempdir;

    use super::load_tensor_map;

    fn write_safetensor_file(path: &std::path::Path, tensors: &[(&str, &[f32], &[usize])]) {
        let views = tensors
            .iter()
            .map(|(name, data, shape)| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(*data),
                    )
                };
                let view = TensorView::new(Dtype::F32, shape.to_vec(), bytes).unwrap();
                ((*name).to_string(), view)
            })
            .collect::<BTreeMap<_, _>>();
        serialize_to_file(&views, None, path).unwrap();
    }

    #[test]
    fn rejects_duplicate_tensor_names_across_shards() {
        let _guard = metal_test_guard();
        let dir = tempdir().unwrap();
        write_safetensor_file(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[("dup.weight", &[1.0f32], &[1])],
        );
        write_safetensor_file(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[("dup.weight", &[2.0f32], &[1])],
        );

        let err = load_tensor_map(dir.path()).unwrap_err().to_string();
        assert!(err.contains("duplicate tensor 'dup.weight'"), "err={err}");
    }
}
