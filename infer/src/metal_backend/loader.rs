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
    let shards = collect_safetensors_shards(model_dir)?;
    anyhow::ensure!(
        !shards.is_empty(),
        "no .safetensors shards found in {}",
        model_dir.display()
    );

    let mut tensors = TensorMap::new();
    log::info!("  loading {} shard(s) via MLX mmap …", shards.len());

    for shard in &shards {
        let path_str = shard.to_str().context("non-UTF8 path")?;
        let shard_tensors = crate::mlx::load_safetensors(path_str);
        for (name, arr) in shard_tensors {
            if tensors.contains_key(&name) {
                log::warn!(
                    "duplicate tensor '{}' in {} — overwriting",
                    name,
                    shard.display()
                );
            }
            tensors.insert(name, arr);
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
        if let Some(scales) = tensors.get(&format!("{base}.scales")).cloned() {
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

fn collect_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    // Try index file first
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
            log::info!("  using index: {} shards", shards.len());
            return Ok(shards);
        }
    }
    // Fallback: glob
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
