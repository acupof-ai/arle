use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

use crate::config::{QuantConfig, QuantMode};
use crate::mlx::{MlxArray, dequantize, eval, transpose_all};
use crate::weights::WeightTensor;

pub(crate) type TensorMap = HashMap<String, MlxArray>;

pub(crate) fn load_tensor_map(model_dir: &Path) -> Result<TensorMap> {
    let shards = collect_safetensors_shards(model_dir)?;
    anyhow::ensure!(
        !shards.is_empty(),
        "no .safetensors shards found in {}",
        model_dir.display()
    );

    let mut tensors = TensorMap::new();
    log::info!("  loading {} shard(s) via MLX mmap", shards.len());
    for shard in &shards {
        let path = shard.to_str().context("non-UTF8 safetensors path")?;
        let shard_tensors = crate::mlx::load_safetensors(path)?;
        for (name, array) in shard_tensors {
            if tensors.contains_key(&name) {
                log::warn!(
                    "duplicate tensor '{}' in {} -- overwriting",
                    name,
                    shard.display()
                );
            }
            tensors.insert(name, array);
        }
    }
    log::info!("  loaded {} tensors (memory-mapped)", tensors.len());
    Ok(tensors)
}

pub(crate) fn tensor_get(tensors: &TensorMap, name: &str) -> Result<MlxArray> {
    tensors
        .get(name)
        .cloned()
        .with_context(|| format!("missing weight '{name}'"))
}

/// Affine checkpoints always carry biases; mxfp4 never does.
fn biases_for(tensors: &TensorMap, base: &str, qc: &QuantConfig) -> Result<Option<MlxArray>> {
    let biases = tensors.get(&format!("{base}.biases")).cloned();
    if biases.is_none() && qc.mode == QuantMode::Affine {
        anyhow::bail!("missing quantized biases '{base}.biases'");
    }
    Ok(biases)
}

pub(crate) fn load_proj_from_tensors(
    tensors: &TensorMap,
    base: &str,
    quantization: Option<QuantConfig>,
) -> Result<WeightTensor> {
    if let Some(qc) = quantization.as_ref()
        && let Some(scales) = tensors.get(&format!("{base}.scales")).cloned()
    {
        let w = tensors
            .get(&format!("{base}.weight"))
            .cloned()
            .with_context(|| format!("missing quantized weight '{base}.weight'"))?;
        let biases = biases_for(tensors, base, qc)?;
        // Honor per-weight quant overrides (mixed bits/group_size, e.g. OptiQ);
        // fall back to the global config for weights without an override.
        let (bits, group_size) = qc
            .per_weight
            .get(base)
            .copied()
            .unwrap_or((qc.bits, qc.group_size));
        return Ok(WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
            mode: qc.mode,
        });
    }

    let w = tensor_get(tensors, &format!("{base}.weight"))?;
    let w_t = transpose_all(&w);
    eval(&[&w_t]);
    Ok(WeightTensor::Dense(w_t))
}

pub(crate) fn load_embed_tokens_from_tensors(
    tensors: &TensorMap,
    base: &str,
    quantization: Option<QuantConfig>,
) -> Result<MlxArray> {
    let w = tensor_get(tensors, &format!("{base}.weight"))?;
    if let Some(qc) = quantization.as_ref()
        && let Some(scales) = tensors.get(&format!("{base}.scales")).cloned()
    {
        let biases = biases_for(tensors, base, qc)?;
        let (bits, group_size) = qc
            .per_weight
            .get(base)
            .copied()
            .unwrap_or((qc.bits, qc.group_size));
        let dequant = dequantize(&w, &scales, biases.as_ref(), group_size, bits, qc.mode);
        return Ok(dequant);
    }
    Ok(w)
}

pub(crate) fn tie_lm_head_from_embed_tokens(embed_tokens: &MlxArray) -> WeightTensor {
    let w_t = transpose_all(embed_tokens);
    eval(&[&w_t]);
    WeightTensor::Dense(w_t)
}

fn collect_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if let Ok(raw) = std::fs::read_to_string(&index_path) {
        let value: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        if let Some(map) = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
        {
            let mut files: Vec<String> = map
                .values()
                .filter_map(|value| value.as_str().map(String::from))
                .collect();
            files.sort();
            files.dedup();
            let shards: Vec<PathBuf> = files.iter().map(|file| model_dir.join(file)).collect();
            log::info!("  using index: {} shards", shards.len());
            return Ok(shards);
        }
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("read_dir {}", model_dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect();
    shards.sort();
    Ok(shards)
}
