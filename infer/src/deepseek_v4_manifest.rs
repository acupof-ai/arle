//! DeepSeek V4 checkpoint manifest validation.
//!
//! This is a cold-path, CPU-only truth gate for
//! `infer/models/dsv4-mini-1B-init/`. Keeping it outside the CUDA model module
//! lets Phase 0.5 execute the config/tensor-name check under `no-cuda` while
//! the CUDA kernel toolchain is blocked.

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unreachable_pub))]

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use deepseek_spec::{
    DeepSeekV4AttentionTensorNames, DeepSeekV4Config, DeepSeekV4HyperConnectionTensorNames,
    DeepSeekV4MoeTensorNames, DeepSeekV4MtpTensorNames,
};

/// Checkpoint-level manifest summary for the V4 target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV4CheckpointManifest {
    pub tensor_count: usize,
    pub required_tensor_count: usize,
}

/// Parse the safetensors manifest and verify every tensor required by the
/// DeepSeek V4 spec is present. This performs no GPU allocation.
pub(crate) fn validate_deepseek_v4_checkpoint_manifest(
    model_path: impl AsRef<Path>,
    config: &DeepSeekV4Config,
) -> Result<DeepseekV4CheckpointManifest> {
    let model_path = model_path.as_ref();
    let available = safetensor_names(model_path)
        .with_context(|| format!("reading safetensors manifest from {}", model_path.display()))?;
    let required = required_v4_tensor_names(config);
    let missing = required
        .iter()
        .filter(|name| !available.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "DeepSeek V4 checkpoint {} is missing {} required tensor(s): {}",
            model_path.display(),
            missing.len(),
            missing.join(", ")
        );
    }

    Ok(DeepseekV4CheckpointManifest {
        tensor_count: available.len(),
        required_tensor_count: required.len(),
    })
}

fn safetensor_names(model_path: &Path) -> Result<HashSet<String>> {
    let mut names = HashSet::new();
    for path in safetensor_paths(model_path)? {
        let mut file = fs::File::open(&path)
            .with_context(|| format!("opening safetensors shard {}", path.display()))?;
        let mut len_bytes = [0_u8; 8];
        file.read_exact(&mut len_bytes)
            .with_context(|| format!("reading safetensors header len {}", path.display()))?;
        let header_len = u64::from_le_bytes(len_bytes)
            .try_into()
            .context("safetensors header length does not fit usize")?;
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)
            .with_context(|| format!("reading safetensors header {}", path.display()))?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .with_context(|| format!("parsing safetensors header {}", path.display()))?;
        for name in header.keys() {
            if name != "__metadata__" {
                names.insert(name.clone());
            }
        }
    }
    Ok(names)
}

fn safetensor_paths(model_path: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_content = fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&index_content)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{} missing weight_map", index_path.display()))?;
        let mut files = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                anyhow::anyhow!("{} has non-string shard path", index_path.display())
            })?;
            files.insert(model_path.join(shard));
        }
        return Ok(files.into_iter().collect());
    }

    let single = model_path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut paths = fs::read_dir(model_path)
        .with_context(|| format!("listing {}", model_path.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        bail!(
            "{} has no safetensors checkpoint shards",
            model_path.display()
        );
    }
    Ok(paths)
}

fn push_hc(out: &mut Vec<String>, names: &DeepSeekV4HyperConnectionTensorNames) {
    out.push(names.base.clone());
    out.push(names.mix_fn.clone());
    out.push(names.scale.clone());
}

fn push_attention(out: &mut Vec<String>, names: &DeepSeekV4AttentionTensorNames) {
    out.push(names.wq_a.clone());
    out.push(names.q_norm.clone());
    out.push(names.wq_b.clone());
    out.push(names.wkv.clone());
    out.push(names.kv_norm.clone());
    out.push(names.wo_a.clone());
    out.push(names.wo_b.clone());
    out.push(names.attn_sink.clone());
    if let Some(compressor) = &names.compressor {
        out.push(compressor.wkv.clone());
        out.push(compressor.wgate.clone());
        out.push(compressor.ape.clone());
        out.push(compressor.norm.clone());
    }
    if let Some(indexer) = &names.indexer {
        out.push(indexer.wq_b.clone());
        out.push(indexer.weights_proj.clone());
        out.push(indexer.compressor.wkv.clone());
        out.push(indexer.compressor.wgate.clone());
        out.push(indexer.compressor.ape.clone());
        out.push(indexer.compressor.norm.clone());
    }
}

fn push_moe(out: &mut Vec<String>, config: &DeepSeekV4Config, names: &DeepSeekV4MoeTensorNames) {
    out.push(names.gate_weight.clone());
    if let Some(gate_bias) = &names.gate_bias {
        out.push(gate_bias.clone());
    }
    if let Some(gate_tid2eid) = &names.gate_tid2eid {
        out.push(gate_tid2eid.clone());
    }
    for expert_idx in 0..config.n_routed_experts {
        let expert = names.expert(expert_idx);
        out.push(expert.w1);
        out.push(expert.w2);
        out.push(expert.w3);
    }
    if let Some(shared) = &names.shared_experts {
        out.push(shared.w1.clone());
        out.push(shared.w2.clone());
        out.push(shared.w3.clone());
    }
}

fn push_mtp(out: &mut Vec<String>, config: &DeepSeekV4Config, names: &DeepSeekV4MtpTensorNames) {
    out.push(names.enorm.clone());
    out.push(names.hnorm.clone());
    out.push(names.e_proj.clone());
    out.push(names.h_proj.clone());
    out.push(names.attn_norm.clone());
    out.push(names.ffn_norm.clone());
    out.push(names.norm.clone());
    push_hc(out, &names.hc_attn);
    push_hc(out, &names.hc_ffn);
    push_hc(out, &names.hc_head);
    push_attention(out, &names.attn);
    push_moe(out, config, &names.ffn);
}

fn required_v4_tensor_names(config: &DeepSeekV4Config) -> Vec<String> {
    let mut out = Vec::new();
    let top = config.tensor_names();
    out.push(top.embed_tokens().to_string());
    out.push(top.norm().to_string());
    out.push(top.lm_head().to_string());
    push_hc(&mut out, &top.head_hc());

    for layer_idx in 0..config.num_hidden_layers {
        let layer = config.layer_tensor_names(layer_idx);
        out.push(layer.attn_norm);
        out.push(layer.ffn_norm);
        push_hc(&mut out, &layer.hc_attn);
        push_hc(&mut out, &layer.hc_ffn);
        push_attention(&mut out, &layer.attn);
        push_moe(&mut out, config, &layer.ffn);
    }

    for mtp_idx in 0..config.num_nextn_predict_layers {
        let mtp = config.mtp_tensor_names(mtp_idx);
        push_mtp(&mut out, config, &mtp);
    }

    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
fn validate_v4_tensor_name_coverage(config: &DeepSeekV4Config) -> std::result::Result<(), String> {
    for name in required_v4_tensor_names(config) {
        let mut covered = config.shard_for_global_tensor(&name).is_some();
        for layer_idx in 0..config.num_hidden_layers {
            covered |= config
                .layer_tensor_names(layer_idx)
                .shard_for(config, &name, 1)
                .is_some();
        }
        for mtp_idx in 0..config.num_nextn_predict_layers {
            covered |= config
                .mtp_tensor_names(mtp_idx)
                .shard_for(config, &name, 1)
                .is_some();
        }
        if !covered {
            return Err(format!(
                "tensor `{name}` is not covered by any V4 shard rule"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replica_model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/dsv4-mini-1B-init")
    }

    fn replica_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_file(replica_model_path().join("config.json")).unwrap()
    }

    #[test]
    #[ignore = "requires checkpoint at infer/models/dsv4-mini-1B-init"]
    fn v4_config_fields_match_init_checkpoint() {
        let cfg = replica_config();
        assert_eq!(cfg.model_type, "deepseek_v4");
        assert!(
            cfg.architectures
                .iter()
                .any(|arch| arch == "DeepseekV4ForCausalLM")
        );
        assert_eq!(cfg.dtype, "bfloat16");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.q_lora_rank, 384);
        assert_eq!(cfg.o_lora_rank, 384);
        assert_eq!(cfg.n_routed_experts, 16);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert_eq!(cfg.scoring_func, "sqrtsoftplus");
        assert_eq!(cfg.topk_method, "noaux_tc");
        assert_eq!(cfg.num_hash_layers, 2);
        assert_eq!(cfg.sliding_window, 64);
        assert_eq!(cfg.num_nextn_predict_layers, 1);
        assert_eq!(cfg.vocab_size, 129280);
    }

    #[test]
    #[ignore = "requires checkpoint at infer/models/dsv4-mini-1B-init"]
    fn v4_tensor_names_fully_covered() {
        let cfg = replica_config();
        validate_v4_tensor_name_coverage(&cfg).expect("V4 tensor coverage");
    }

    #[test]
    #[ignore = "requires checkpoint at infer/models/dsv4-mini-1B-init"]
    fn v4_checkpoint_manifest_contains_required_tensors() {
        let cfg = replica_config();
        let manifest = validate_deepseek_v4_checkpoint_manifest(replica_model_path(), &cfg)
            .expect("validate V4 tensors");
        assert_eq!(
            manifest.required_tensor_count,
            required_v4_tensor_names(&cfg).len()
        );
        assert!(
            manifest.tensor_count >= manifest.required_tensor_count,
            "checkpoint tensor_count={} required={}",
            manifest.tensor_count,
            manifest.required_tensor_count
        );
    }
}
