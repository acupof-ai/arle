use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use mlx_rs::Array;

use super::{QuantConfig, WeightTensor};

pub(super) type TensorMap = HashMap<String, Array>;

pub(super) fn load_tensor_map(model_dir: &Path) -> Result<TensorMap> {
    use mlx_rs::Dtype;
    use safetensors::{Dtype as SdType, SafeTensors};

    let shards = collect_safetensors_shards(model_dir)?;
    anyhow::ensure!(
        !shards.is_empty(),
        "no .safetensors shards found in {}",
        model_dir.display()
    );

    let mut tensors = TensorMap::new();
    log::info!("  loading {} shard(s) …", shards.len());

    for shard in &shards {
        let bytes = std::fs::read(shard).with_context(|| format!("read {}", shard.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse {}", shard.display()))?;

        for name in st.names() {
            if tensors.contains_key(name) {
                anyhow::bail!(
                    "duplicate tensor '{}' found while loading {}",
                    name,
                    shard.display()
                );
            }
            let view = st
                .tensor(name)
                .with_context(|| format!("tensor {name} in {}", shard.display()))?;
            let shape: Vec<i32> = view.shape().iter().map(|&x| x as i32).collect();
            let dtype = match view.dtype() {
                SdType::BF16 => Dtype::Bfloat16,
                SdType::F16 => Dtype::Float16,
                SdType::F32 => Dtype::Float32,
                SdType::U8 => Dtype::Uint8,
                SdType::I32 => Dtype::Int32,
                SdType::U32 => Dtype::Uint32,
                other => anyhow::bail!("unsupported safetensors dtype {other:?} for {name}"),
            };
            let arr = unsafe { Array::from_raw_data(view.data().as_ptr().cast(), &shape, dtype) };
            tensors.insert(name.to_string(), arr);
        }
    }

    log::info!("  parsed {} tensors", tensors.len());
    Ok(tensors)
}

pub(super) fn tensor_get(tensors: &TensorMap, name: &str) -> Result<Array> {
    tensors
        .get(name)
        .cloned()
        .with_context(|| format!("missing weight '{name}'"))
}

pub(super) fn load_proj_from_tensors(
    tensors: &TensorMap,
    base: &str,
    quantization: Option<QuantConfig>,
) -> Result<WeightTensor> {
    use mlx_rs::{ops::transpose, transforms::eval};

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
    let w_t = transpose(&w).context("pre-transpose weight")?;
    eval([&w_t]).context("eval pre-transposed weight")?;
    Ok(WeightTensor::Dense(w_t))
}

pub(super) fn load_embed_tokens_from_tensors(
    tensors: &TensorMap,
    base: &str,
    quantization: Option<QuantConfig>,
) -> Result<Array> {
    let w = tensor_get(tensors, &format!("{base}.weight"))?;
    if let Some(qc) = quantization {
        if let Some(scales) = tensors.get(&format!("{base}.scales")).cloned() {
            let biases = tensors
                .get(&format!("{base}.biases"))
                .cloned()
                .with_context(|| format!("missing biases '{base}.biases'"))?;
            log::info!("  dequantizing embed_tokens at load time");
            return mlx_rs::ops::dequantize(
                &w,
                &scales,
                &biases,
                Some(qc.group_size),
                Some(qc.bits),
            )
            .context("dequantize embed_tokens");
        }
    }
    Ok(w)
}

pub(super) fn tie_lm_head_from_embed_tokens(embed_tokens: &Array) -> Result<WeightTensor> {
    use mlx_rs::{ops::transpose, transforms::eval};

    let w_t = transpose(embed_tokens).context("pre-transpose tied lm_head")?;
    eval([&w_t]).context("eval tied lm_head")?;
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
