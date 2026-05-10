//! DeepSeek V4 1B scaffold smoke test.
//!
//! The target is the actual 2.0 GB `DeepseekV4ForCausalLM` checkpoint at
//! `infer/models/dsv4-mini-1B-init/`, not the old `DeepSeekConfig::nano()`
//! fixture. Phase 0.5 validates config and tensor-name truth; the end-to-end
//! prefill/decode test remains ignored until Phase 2A lands V4 kernels.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use deepseek_spec::DeepSeekV4Config;
use infer::model::ModelForward;
use infer::model::deepseek::{DeepseekModel, DeepseekRuntimeConfig};

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models/dsv4-mini-1B-init")
}

#[test]
fn dsv4_v4_1b_config_and_manifest_parse() {
    let path = model_path();
    let runtime = DeepseekRuntimeConfig::from_model_dir(&path).expect("parse V4 runtime config");
    let spec: &DeepSeekV4Config = &runtime.spec;

    assert_eq!(spec.model_type, "deepseek_v4");
    assert!(
        spec.architectures
            .iter()
            .any(|arch| arch == "DeepseekV4ForCausalLM")
    );
    assert_eq!(spec.dtype, "bfloat16");
    assert_eq!(spec.hidden_size, 1024);
    assert_eq!(spec.num_hidden_layers, 24);
    assert_eq!(spec.num_key_value_heads, 1);
    assert_eq!(spec.q_lora_rank, 384);
    assert_eq!(spec.o_lora_rank, 384);
    assert_eq!(spec.n_routed_experts, 16);
    assert_eq!(spec.num_experts_per_tok, 2);
    assert_eq!(spec.scoring_func, "sqrtsoftplus");
    assert_eq!(spec.topk_method, "noaux_tc");
    assert_eq!(spec.num_nextn_predict_layers, 1);
    assert_eq!(spec.vocab_size, 129_280);

    let manifest =
        DeepseekModel::validate_checkpoint_manifest(&path, spec).expect("validate V4 tensors");
    assert!(manifest.tensor_count >= manifest.required_tensor_count);
}

#[test]
#[ignore = "ignored until DeepSeek V4 forward kernels land in Phase 2A"]
fn dsv4_v4_1b_smoke_prefill_and_greedy_decode() {
    let path = model_path();
    let runtime = DeepseekRuntimeConfig::from_model_dir(&path).expect("parse V4 runtime config");
    let model =
        DeepseekModel::from_safetensors(path.to_str().unwrap(), runtime).expect("load V4 model");

    let tokens: Vec<u32> = vec![0, 1, 2, 3];
    let mut state = model.create_state().expect("create state");
    model
        .forward_prefill(&tokens, &mut state)
        .expect("forward_prefill");

    use infer::model::GenerationState;
    let logits = state.logits();
    let expected = 129_280 * tokens.len();
    assert_eq!(
        logits.len, expected,
        "expected logits len {expected}, got {}",
        logits.len
    );
}
