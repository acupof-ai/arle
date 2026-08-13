//! A backend may skip `sample_token` only when `is_raw_argmax()` holds.

use std::sync::Arc;

use infer_plan::{SamplingParams, sample_token};

fn greedy() -> SamplingParams {
    SamplingParams::default()
}

#[test]
fn logit_bias_vetoes_the_raw_argmax_fast_path() {
    let logits = [0.0f32, 1.0, 0.5];
    let mut params = greedy();
    assert!(params.is_raw_argmax());

    params.logit_bias.insert(0, 5.0);
    assert!(
        params.is_greedy(),
        "bias does not change the sampling policy"
    );
    assert!(
        !params.is_raw_argmax(),
        "a bias rewrites the logits the argmax runs over"
    );
    assert_eq!(
        sample_token(&logits, &params, 0),
        0,
        "the sampler applies the bias"
    );
}

#[test]
fn grammar_bitmask_vetoes_the_raw_argmax_fast_path() {
    let mut params = greedy();
    params.grammar_bitmask = Some(Arc::from(vec![0b101u32].into_boxed_slice()));
    assert!(params.is_greedy());
    assert!(!params.is_raw_argmax());
}

/// Equivalence with the sampler is the whole license for skipping it.
#[test]
fn fast_path_agrees_with_the_sampler_wherever_it_is_taken() {
    let logits = [0.25f32, -1.0, 3.0, 3.0, 0.0];
    let raw_argmax = 2u32; // tie at 2 and 3 resolves to the lower index

    for params in [greedy(), {
        let mut p = greedy();
        p.top_k = 5;
        p.top_p = 0.9;
        p.seed = Some(7);
        p
    }] {
        assert!(params.is_raw_argmax());
        assert_eq!(sample_token(&logits, &params, 0), raw_argmax);
    }
}
