//! DSV4 nano scaffold smoke test.
//!
//! Builds a `DeepseekModel` from `DeepSeekConfig::nano()` and runs a single
//! `forward_prefill` step on tiny synthetic input, asserting the produced
//! logits buffer has the expected `[seq, vocab]` shape.
//!
//! Until the MLA prefill + decode kernels land, every entrypoint inside
//! `infer::model::deepseek` is `todo!("MLA kernel — see
//! docs/plans/2026-05-01-mla-kernel-design.md")`. This test will hit one of
//! those `todo!()`s — that is the intended signal. The test is marked
//! `#[ignore]` so it does not run as part of `cargo test`; remove the
//! `#[ignore]` attribute as part of the diff that lands the MLA kernel.

#![cfg(feature = "cuda")]

use deepseek_spec::DeepSeekConfig;
use infer::model::ModelForward;
use infer::model::deepseek::{DeepseekModel, DeepseekRuntimeConfig};

#[test]
#[ignore = "ignored until MLA forward kernel lands — see docs/plans/2026-05-01-mla-kernel-design.md"]
fn dsv4_nano_smoke_prefill() {
    let spec = DeepSeekConfig::nano();
    let runtime = DeepseekRuntimeConfig::from_spec(spec.clone());
    // `from_config` is currently `todo!()`; this call panics with the
    // documented MLA-kernel pending message until the kernel lands.
    let model = DeepseekModel::from_config(runtime).expect("nano model construction");

    // Tiny synthetic input: 4 tokens within the nano vocab (4096).
    let tokens: Vec<u32> = vec![0, 1, 2, 3];
    let mut state = model.create_state().expect("create state");
    model
        .forward_prefill(&tokens, &mut state)
        .expect("forward_prefill");

    // After prefill, the logits buffer length should equal vocab_size *
    // tokens.len() (the model's `compute_logits_batch` shape contract).
    use infer::model::GenerationState;
    let logits = state.logits();
    let expected = spec.vocab_size * tokens.len();
    assert_eq!(
        logits.len, expected,
        "expected logits len {expected}, got {}",
        logits.len
    );
}
