#![cfg(feature = "cuda")]

use std::path::PathBuf;

use anyhow::Result;
use infer::server_engine::{InferenceEngineOptions, LoadedInferenceEngine};

fn qwen3_06b_model_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ARLE_TEST_QWEN3_06B_DIR") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".cache/modelscope/hub/models/Qwen/Qwen3-0.6B"))
}

#[test]
fn loaded_engine_returns_raw_token_logits_for_qwen3_06b() -> Result<()> {
    let Some(model_dir) = qwen3_06b_model_dir() else {
        eprintln!("skipping raw logits smoke: HOME is not set");
        return Ok(());
    };
    if !model_dir.exists() {
        eprintln!(
            "skipping raw logits smoke: missing model dir {}",
            model_dir.display()
        );
        return Ok(());
    }

    let engine = LoadedInferenceEngine::load_with_options(
        model_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        42,
        InferenceEngineOptions {
            enable_cuda_graph: false,
        },
    )?;
    let input_ids = [1, 3, 8];
    let positions = [0, 1, 2];
    let logits = engine.forward_token_logits(&input_ids, &positions)?;
    assert_eq!(logits.seq_len(), input_ids.len());
    assert!(
        logits.vocab_size() > 0,
        "vocab size should be populated from the loaded model"
    );
    assert_eq!(logits.logits.len, input_ids.len() * logits.vocab_size());

    let host = logits.to_host_f32()?;
    assert_eq!(host.len(), logits.logits.len);
    assert!(
        host.iter().all(|value| value.is_finite()),
        "raw token logits contain NaN/Inf"
    );

    Ok(())
}
