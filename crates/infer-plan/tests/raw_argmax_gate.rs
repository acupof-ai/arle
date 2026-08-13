//! A backend may skip `sample_token` only when `is_raw_argmax()` holds.

use std::sync::Arc;

use infer_plan::{PenaltyHistory, SamplingParams, sample_token, sample_token_penalized};

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

#[test]
fn penalties_veto_the_raw_argmax_fast_path() {
    assert!(greedy().is_raw_argmax());
    let set = [
        (
            "repetition",
            SamplingParams {
                repetition_penalty: 1.2,
                ..greedy()
            },
        ),
        (
            "frequency",
            SamplingParams {
                frequency_penalty: 0.5,
                ..greedy()
            },
        ),
        (
            "presence",
            SamplingParams {
                presence_penalty: 0.5,
                ..greedy()
            },
        ),
    ];
    for (name, mut params) in set {
        assert!(params.has_penalty(), "{name} must register as a penalty");
        assert!(
            params.is_greedy(),
            "{name} does not change the sampling policy"
        );
        assert!(
            !params.is_raw_argmax(),
            "{name} rewrites the logits the argmax runs over"
        );
        params.repetition_penalty = 1.0;
        params.frequency_penalty = 0.0;
        params.presence_penalty = 0.0;
        assert!(params.is_raw_argmax(), "no-op values must not veto");
    }
}

/// Each penalty moves the greedy token in the spec's direction, over the spec's
/// token set: repetition over prompt+generated, the other two over generated.
#[test]
fn penalties_move_the_greedy_token_over_their_own_token_set() {
    let logits = [0.0f32, 3.0, 2.0];
    let generated = PenaltyHistory {
        tokens: &[1],
        prompt_len: 0,
    };
    let prompt_only = PenaltyHistory {
        tokens: &[1],
        prompt_len: 1,
    };

    let mut params = greedy();
    params.presence_penalty = 2.0;
    assert_eq!(sample_token_penalized(&logits, &params, 0, generated), 2);
    assert_eq!(
        sample_token_penalized(&logits, &params, 0, prompt_only),
        1,
        "presence ignores the prompt"
    );

    let mut params = greedy();
    params.repetition_penalty = 2.0;
    assert_eq!(
        sample_token_penalized(&logits, &params, 0, prompt_only),
        2,
        "repetition scores the prompt too"
    );

    // Frequency scales with the occurrence count; presence does not.
    let twice = PenaltyHistory {
        tokens: &[1, 1],
        prompt_len: 0,
    };
    let mut params = greedy();
    params.frequency_penalty = 0.6;
    assert_eq!(sample_token_penalized(&logits, &params, 0, twice), 2);
    let mut params = greedy();
    params.presence_penalty = 0.6;
    assert_eq!(sample_token_penalized(&logits, &params, 0, twice), 1);
}

/// A grammar-masked token must stay unselectable under a penalty: `-inf`
/// arithmetic must not produce a NaN, which would outrank every finite logit.
#[test]
fn repetition_penalty_never_revives_a_grammar_masked_token() {
    let logits = [0.0f32, 3.0, 2.0];
    let mut params = greedy();
    params.grammar_bitmask = Some(Arc::from(vec![0b110u32].into_boxed_slice()));
    params.repetition_penalty = 2.0;
    let history = PenaltyHistory {
        tokens: &[0, 1],
        prompt_len: 0,
    };
    assert_eq!(sample_token_penalized(&logits, &params, 0, history), 2);
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
