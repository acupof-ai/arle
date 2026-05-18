//! Smoke test for `train::qwen35_loader::load_qwen35_from_hf_dir` against a
//! real on-disk Qwen3 / Qwen3.5 HuggingFace checkpoint.
//!
//! Resolution order for the model directory:
//!
//! 1. `INFER_TEST_QWEN3_06B_DIR` environment variable (explicit override).
//! 2. `~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B` (default ModelScope
//!    cache layout — populated by `arle model download --source modelscope
//!    Qwen/Qwen3-0.6B`).
//!
//! When neither resolves, the test prints a skip note and returns early.
//! The test is not marked `#[ignore]` because we want CI / cargo test runs
//! to surface a clear "skipped — model not present" message rather than
//! silently hide the gate.
//!
//! Expected outcome on the canonical ModelScope `Qwen/Qwen3-0.6B` snapshot:
//!
//! - `Qwen35HfConfig` parses, `to_qwen35_config()` produces a valid config,
//!   `Qwen35Model::new_for_eval` succeeds.
//! - The safetensors walk discovers `model.safetensors`, opens the file,
//!   and starts the per-tensor materialization loop.
//! - The vanilla Qwen3 layout's un-gated `q_proj` triggers a
//!   `LoaderError::ShapeMismatch` for `model.language_model.layers.0.self_attn.q_proj.weight`,
//!   carrying the diagnostic hint pointing at the follow-up tranche needed
//!   to land a non-gated full-attention variant of `Qwen35Model`.
//!
//! The test asserts:
//!
//! - The HF config + name remap pipeline reaches the safetensors layer
//!   (i.e. `Qwen35Model::new_for_eval` succeeded, model has nonzero param
//!   count).
//! - If load succeeds end-to-end (Qwen3.5/3.6-shaped checkpoint), a single
//!   `forward_tokens(&[1, 2, 3], …)` produces finite logits whose length
//!   equals `cfg.vocab_size`.
//! - If load surfaces the expected gated-q_proj mismatch (vanilla Qwen3),
//!   the diagnostic hint is present and references the train-side
//!   limitation, not a generic safetensors error.
//!
//! This is the only place in `crates/train/tests/` that touches a real
//! HF cache; everything else uses synthesized tiny configs.

use std::path::PathBuf;

use autograd::{Tape, TensorStore};
use train::qwen35::Qwen35Model;
use train::qwen35_loader::{LoaderError, load_qwen35_from_hf_dir};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn resolve_qwen3_06b_dir() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("INFER_TEST_QWEN3_06B_DIR") {
        let p = PathBuf::from(explicit);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let cache = PathBuf::from(home).join(".cache/modelscope/hub/models/Qwen/Qwen3-0.6B");
    if cache.is_dir() && cache.join("config.json").is_file() {
        return Some(cache);
    }
    None
}

#[test]
fn loader_smoke_qwen3_0_6b() -> TestResult {
    let dir = match resolve_qwen3_06b_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "loader_smoke_qwen3_0_6b: skipping (set INFER_TEST_QWEN3_06B_DIR or populate \
                 ~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B via \
                 `arle model download --source modelscope Qwen/Qwen3-0.6B`)"
            );
            return Ok(());
        }
    };

    eprintln!("loader_smoke_qwen3_0_6b: loading {}", dir.display());
    let mut store = TensorStore::default();
    let load_result = load_qwen35_from_hf_dir(&dir, &mut store);

    match load_result {
        Ok(model) => {
            // Success path: the checkpoint matched the Qwen3.5/3.6 gated-Q
            // layout. Assert the model has non-zero params and that a single
            // forward pass produces finite logits of vocab_size length.
            let param_count = model.all_parameter_ids().len();
            assert!(
                param_count > 0,
                "loaded model should expose at least one parameter id"
            );
            eprintln!("loader_smoke_qwen3_0_6b: loaded ok, param_count = {param_count}");

            let cfg = model.config().clone();
            let mut tape = Tape::new();
            let logits = model
                .forward_tokens(&[1usize, 2, 3], &mut store, &mut tape)
                .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;
            let host_logits = store.to_host(logits)?;
            // forward_tokens returns the full per-position logits matrix —
            // flat length = seq_len * vocab_size. We only need at least one
            // vocab row to assert finiteness; sample the last row.
            assert_eq!(
                host_logits.len() % cfg.vocab_size,
                0,
                "logits length {} should be a multiple of vocab_size {}",
                host_logits.len(),
                cfg.vocab_size
            );
            assert!(
                !host_logits.is_empty(),
                "forward_tokens returned an empty logits tensor"
            );
            let last_row = &host_logits[host_logits.len() - cfg.vocab_size..];
            let all_finite = last_row.iter().all(|x| x.is_finite());
            let sample: Vec<f32> = last_row.iter().take(5).copied().collect();
            eprintln!(
                "loader_smoke_qwen3_0_6b: last-row logits[..5] = {sample:?}, all_finite = {all_finite}"
            );
            assert!(all_finite, "logits must be finite (no NaN / Inf)");
        }
        Err(LoaderError::ShapeMismatch {
            name,
            expected,
            got,
            hint,
        }) => {
            // Expected on vanilla Qwen3 (0.6B/1.7B/4B) where q_proj has no
            // output gate: train-side Qwen35Model is Qwen3.5/3.6-gated-Q
            // shaped. The diagnostic hint should call this out so the user
            // can wire the follow-up tranche.
            eprintln!(
                "loader_smoke_qwen3_0_6b: structural mismatch as expected \
                 on vanilla Qwen3 layout — tensor={name} expected={expected:?} got={got:?}"
            );
            assert!(
                name.ends_with(".self_attn.q_proj.weight"),
                "shape-mismatch should be on q_proj; got tensor name {name}"
            );
            assert!(
                hint.contains("vanilla Qwen3 ships an un-gated q_proj"),
                "shape-mismatch hint missing diagnostic: {hint}"
            );
            // Sanity-check that what we partially loaded had non-zero param
            // count via a fresh Qwen35Model build (proves the HF config
            // pipeline reached the model-construction step):
            let (hf, _schema) =
                train::qwen35_loader::Qwen35HfConfig::from_json_file(dir.join("config.json"))?;
            let cfg = hf.to_qwen35_config()?;
            let mut fresh = TensorStore::default();
            let probe = Qwen35Model::new_for_eval(&cfg, &mut fresh)?;
            assert!(
                !probe.all_parameter_ids().is_empty(),
                "Qwen35Model::new_for_eval should produce parameters from the HF config"
            );
        }
        Err(other) => {
            return Err(Box::new(other));
        }
    }

    Ok(())
}
