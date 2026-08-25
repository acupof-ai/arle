//! `SamplingParams` crosses the multiproc relay as JSON, so every field must
//! survive a round trip. A `HashMap<u32, _>` does not: JSON object keys are
//! strings, and reading them back as `u32` fails at the worker, which unwinds
//! the whole TP group. One biased request killed a live TP=4 serve on
//! 2026-08-13.

#![cfg(feature = "serde")]

use infer_plan::SamplingParams;

fn roundtrip(params: &SamplingParams) -> SamplingParams {
    let wire = serde_json::to_string(params).expect("serialize");
    serde_json::from_str(&wire).expect("deserialize")
}

#[test]
fn logit_bias_survives_the_relay_wire_format() {
    let params = SamplingParams {
        logit_bias: vec![(0, -1.5), (73233, 100.0), (u32::MAX, 0.25)],
        ..Default::default()
    };

    let back = roundtrip(&params);
    assert_eq!(back.logit_bias, params.logit_bias);
    assert!(!back.is_raw_argmax(), "a bias still vetoes the fast path");
}

#[test]
fn penalties_and_stop_tokens_survive_the_relay_wire_format() {
    let params = SamplingParams {
        repetition_penalty: 1.5,
        frequency_penalty: -2.0,
        presence_penalty: 0.75,
        stop_token_ids: vec![1, 2, u32::MAX],
        seed: Some(u64::MAX),
        ..Default::default()
    };

    let back = roundtrip(&params);
    assert_eq!(back.repetition_penalty, 1.5);
    assert_eq!(back.frequency_penalty, -2.0);
    assert_eq!(back.presence_penalty, 0.75);
    assert_eq!(back.stop_token_ids, params.stop_token_ids);
    assert_eq!(back.seed, params.seed);
    assert!(back.has_penalty());
}

/// The logprobs capture request must survive the relay wire, and its absence
/// must deserialize from pre-capture peers (serde default).
#[test]
fn top_logprobs_survives_the_relay_wire_format() {
    let params = SamplingParams {
        top_logprobs: Some(2),
        ..Default::default()
    };
    let back = roundtrip(&params);
    assert_eq!(back.top_logprobs, Some(2));
    assert!(
        !back.is_raw_argmax(),
        "the capture still vetoes the fast path"
    );

    let legacy: SamplingParams =
        serde_json::from_str(r#"{"temperature":0.0,"top_k":-1,"top_p":1.0,"min_p":0.0,"repetition_penalty":1.0,"frequency_penalty":0.0,"presence_penalty":0.0,"ignore_eos":false,"stop_token_ids":[],"seed":null,"max_new_tokens":null,"grammar_bitmask":null,"logit_bias":[],"n":1}"#)
            .expect("pre-capture wire format still parses");
    assert_eq!(legacy.top_logprobs, None);
}
