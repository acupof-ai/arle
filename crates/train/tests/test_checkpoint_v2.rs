//! CheckpointCodec v2 roundtrip + error-path tests.
//!
//! Covers the Phase 2 directory-layout codec: `trainer_state.json` +
//! `optimizer.safetensors`. The legacy `LMCKP003` single-file format is
//! exercised by `test_checkpoint.rs`; this file stays scoped to v2.

use std::borrow::Cow;
use std::collections::HashMap;

use autograd::adamw_state::{AdamWParamState, AdamWState};
use safetensors::{Dtype, serialize_to_file};
use tempfile::tempdir;
use train::checkpoint::{
    CheckpointError, OPTIMIZER_STATE_FILENAME, TRAINER_STATE_CODEC_VERSION, TRAINER_STATE_FILENAME,
    TrainerStateDoc, load_trainer_state_v2, save_trainer_state_v2,
};

fn sample_state() -> TrainerStateDoc {
    TrainerStateDoc {
        step: 1234,
        optim_schema: "adamw-v1".to_string(),
        schedule_name: "cosine-with-warmup".to_string(),
        schedule_params: serde_json::json!({
            "base_lr": 1e-4,
            "warmup_steps": 100,
            "total_steps": 10_000,
        }),
        grad_accum_current: 2,
        rng_seed: 0xDEAD_BEEF,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    }
}

fn sample_optim() -> AdamWState {
    AdamWState {
        step: 1234,
        skipped_export: 3,
        params: vec![
            AdamWParamState {
                name: "layer.0.bias".to_string(),
                m: vec![0.1, -0.2, 0.3, -0.4],
                v: vec![1e-3, 2e-3, 3e-3, 4e-3],
                shape: vec![4],
            },
            AdamWParamState {
                name: "layer.0.weight".to_string(),
                m: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                v: vec![0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
                shape: vec![3, 2],
            },
        ],
    }
}

#[test]
fn roundtrip_v2_minimal() {
    let tmp = tempdir().expect("tempdir");
    let state = sample_state();
    let optim = sample_optim();

    save_trainer_state_v2(tmp.path(), &state, &optim).expect("save v2");

    // Both files land at the expected names.
    assert!(tmp.path().join(TRAINER_STATE_FILENAME).is_file());
    assert!(tmp.path().join(OPTIMIZER_STATE_FILENAME).is_file());

    let (loaded_state, loaded_optim) = load_trainer_state_v2(tmp.path()).expect("load v2");

    assert_eq!(loaded_state.step, state.step);
    assert_eq!(loaded_state.optim_schema, state.optim_schema);
    assert_eq!(loaded_state.schedule_name, state.schedule_name);
    assert_eq!(loaded_state.schedule_params, state.schedule_params);
    assert_eq!(loaded_state.grad_accum_current, state.grad_accum_current);
    assert_eq!(loaded_state.rng_seed, state.rng_seed);
    assert_eq!(loaded_state.codec_version, TRAINER_STATE_CODEC_VERSION);

    assert_eq!(loaded_optim.step, optim.step);
    assert_eq!(loaded_optim.skipped_export, optim.skipped_export);
    assert_eq!(loaded_optim.params.len(), optim.params.len());

    // Build a name-keyed view — on-disk ordering is alphabetical by design
    // (safetensors sorts). Compare by name rather than by index.
    let loaded_by_name: HashMap<&str, &AdamWParamState> = loaded_optim
        .params
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect();
    for expected in &optim.params {
        let got = loaded_by_name
            .get(expected.name.as_str())
            .unwrap_or_else(|| panic!("missing param '{}'", expected.name));
        assert_eq!(got.shape, expected.shape, "shape {}", expected.name);
        bits_eq(&got.m, &expected.m, &format!("{} .m", expected.name));
        bits_eq(&got.v, &expected.v, &format!("{} .v", expected.name));
    }
}

#[test]
fn v2_fails_on_wrong_version() {
    let tmp = tempdir().expect("tempdir");
    let mut state = sample_state();
    save_trainer_state_v2(tmp.path(), &state, &sample_optim()).expect("save v2");

    // Hand-craft a bogus trainer_state.json with codec_version=99. Keeps the
    // safetensors companion file from the previous save so we exercise the
    // version-check gate before anything else.
    state.codec_version = 99;
    let json = serde_json::to_string_pretty(&state).expect("json");
    std::fs::write(tmp.path().join(TRAINER_STATE_FILENAME), json).expect("overwrite json");

    let err = load_trainer_state_v2(tmp.path()).expect_err("should fail on version");
    match err {
        CheckpointError::VersionMismatch(v) => assert_eq!(v, 99),
        other => panic!("expected VersionMismatch(99), got {other:?}"),
    }
}

#[test]
fn v2_fails_on_missing_moment_pair() {
    let tmp = tempdir().expect("tempdir");
    let state = sample_state();

    // Write a valid trainer_state.json so the JSON/version gates pass and we
    // actually reach the moment-pair check.
    let json = serde_json::to_string_pretty(&state).expect("json");
    std::fs::write(tmp.path().join(TRAINER_STATE_FILENAME), json).expect("write json");

    // Safetensors with only "foo.m" — no matching ".v".
    let data = vec![(
        "foo.m".to_string(),
        BytesView {
            dtype: Dtype::F32,
            shape: vec![2],
            bytes: vec![0u8; 8],
        },
    )];
    serialize_to_file(data, None, &tmp.path().join(OPTIMIZER_STATE_FILENAME)).expect("serialize");

    let err = load_trainer_state_v2(tmp.path()).expect_err("should fail on missing pair");
    match err {
        CheckpointError::MissingMomentPair(name) => assert_eq!(name, "foo"),
        other => panic!("expected MissingMomentPair(\"foo\"), got {other:?}"),
    }
}

fn bits_eq(lhs: &[f32], rhs: &[f32], label: &str) {
    assert_eq!(lhs.len(), rhs.len(), "{label} len");
    for (i, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{label}[{i}]: got {a} want {b}");
    }
}

/// Minimal `safetensors::View` for the negative-path test. Lets us write a
/// file containing an arbitrary single tensor without going through the full
/// codec save path.
struct BytesView {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl safetensors::View for BytesView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.bytes.as_slice())
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}
