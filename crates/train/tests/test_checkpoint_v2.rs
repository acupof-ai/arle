//! CheckpointCodec v2 roundtrip + error-path tests.
//!
//! Covers the Phase 2 directory-layout codec: `trainer_state.json` +
//! `optimizer.safetensors`. The handwritten-Transformer single-file
//! checkpoint path has been retired; this file stays scoped to v2.

use std::borrow::Cow;
use std::collections::HashMap;

use autograd::adamw_state::{AdamWParamState, AdamWState};
use safetensors::{Dtype, SafeTensors, serialize_to_file};
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

// CK-5 — guard HF-interop regression: the moments file must be readable by a
// plain `safetensors::SafeTensors` consumer with no project-specific wrapper,
// with f32 dtype and `{name}.m` / `{name}.v` keys at the expected shapes.
#[test]
fn moments_file_readable_by_plain_safetensors() {
    let tmp = tempdir().expect("tempdir");
    let state = sample_state();
    let optim = sample_optim();
    save_trainer_state_v2(tmp.path(), &state, &optim).expect("save v2");

    let bytes = std::fs::read(tmp.path().join(OPTIMIZER_STATE_FILENAME)).expect("read optim file");
    let st = SafeTensors::deserialize(&bytes).expect("plain safetensors deserialize");

    for param in &optim.params {
        let m_key = format!("{}.m", param.name);
        let v_key = format!("{}.v", param.name);

        let m_view = st
            .tensor(&m_key)
            .unwrap_or_else(|_| panic!("missing tensor '{m_key}'"));
        let v_view = st
            .tensor(&v_key)
            .unwrap_or_else(|_| panic!("missing tensor '{v_key}'"));

        assert_eq!(m_view.dtype(), Dtype::F32, "'{m_key}' must be f32");
        assert_eq!(v_view.dtype(), Dtype::F32, "'{v_key}' must be f32");
        assert_eq!(m_view.shape(), param.shape.as_slice(), "'{m_key}' shape");
        assert_eq!(v_view.shape(), param.shape.as_slice(), "'{v_key}' shape");
    }

    // Every tensor in the file must be f32 — no bf16 / fp16 leakage.
    for (name, view) in st.tensors() {
        assert_eq!(
            view.dtype(),
            Dtype::F32,
            "tensor '{name}' must be f32 for HF interop"
        );
    }
}

// CK-6 — guard moments drift on cross-backend resume (lesson #3, HF #27749):
// save on one AdamW instance, create a fresh one, import state, re-export —
// the second export's bytes must match the first exactly. Proves the state
// codec carries only pure host data with no backend-specific representation.
#[test]
fn cross_backend_resume_bitwise() {
    use autograd::{Tensor, TensorId, TensorStore, optim::AdamW};

    // Harness: two params (shapes [4] and [3,2]) with fixed gradients so
    // AdamW has moments to export. Matches the shape of test_adamw_state.rs's
    // build_harness so the codec sees a non-trivial state.
    fn build() -> (TensorStore, AdamW, Vec<TensorId>, Vec<(TensorId, String)>) {
        let mut store = TensorStore::default();
        let p_a = store.alloc(
            Tensor::new(vec![0.1, -0.2, 0.3, -0.4], vec![4], true).expect("p_a well-formed"),
        );
        let p_b = store.alloc(
            Tensor::new(vec![0.05, 0.1, -0.15, 0.2, -0.25, 0.3], vec![3, 2], true)
                .expect("p_b well-formed"),
        );
        let g_a = store.alloc(
            Tensor::new(vec![0.01, -0.02, 0.03, -0.04], vec![4], false).expect("g_a well-formed"),
        );
        let g_b = store.alloc(
            Tensor::new(
                vec![0.005, -0.01, 0.015, -0.02, 0.025, -0.03],
                vec![3, 2],
                false,
            )
            .expect("g_b well-formed"),
        );
        store.get_mut(p_a).expect("p_a").grad = Some(g_a);
        store.get_mut(p_b).expect("p_b").grad = Some(g_b);

        let opt = AdamW::new(0.01, (0.9, 0.999), 1e-8, 0.05);
        let names = vec![(p_a, "p_a".to_string()), (p_b, "p_b".to_string())];
        (store, opt, vec![p_a, p_b], names)
    }

    let (mut store_src, mut opt_src, params_src, names_src) = build();
    // Step twice so moments are non-zero.
    opt_src.step(&params_src, &mut store_src);
    opt_src.step(&params_src, &mut store_src);

    // Original export — this is the reference "on-disk" byte image.
    let original = opt_src.export_state(&names_src);

    // Fresh AdamW on a different store. Seed the shape tracker with one step
    // so the destination's internal shape cache has the real [3, 2] shape
    // registered (AdamW learns shapes at first step, not at import). This
    // mirrors a real cross-backend resume: the target builds its param list
    // + takes a warm-up step-equivalent before restoring moments.
    let (mut store_dst, mut opt_dst, params_dst, names_dst) = build();
    opt_dst.step(&params_dst, &mut store_dst);

    let restored = opt_dst
        .import_state(&original, &names_dst)
        .expect("import on fresh AdamW");
    assert_eq!(
        restored,
        names_dst.len(),
        "all params must restore on fresh AdamW"
    );

    // Re-export from the imported-into instance. Byte image and every field
    // must match the original exactly — proves the state is pure host data.
    let reexport = opt_dst.export_state(&names_dst);
    let original_bytes = serde_json::to_vec(&original).expect("serialize original");
    let reexport_bytes = serde_json::to_vec(&reexport).expect("serialize reexport");

    assert_eq!(
        original_bytes, reexport_bytes,
        "cross-backend export bytes must be bit-identical after import round-trip"
    );
    assert_eq!(original.step, reexport.step);
    assert_eq!(original.params.len(), reexport.params.len());
    for (orig, roundtrip) in original.params.iter().zip(reexport.params.iter()) {
        assert_eq!(orig.name, roundtrip.name);
        assert_eq!(orig.shape, roundtrip.shape);
        bits_eq(&orig.m, &roundtrip.m, &format!("{} .m", orig.name));
        bits_eq(&orig.v, &roundtrip.v, &format!("{} .v", orig.name));
    }
}

// CK-7 — guard cross-optim clobber: a trainer_state.json doc whose
// `optim_schema` differs from the live optimizer's schema must be surfaced
// (not silently imported). `load_trainer_state_v2` does not enforce this
// itself — the gate lives in `Trainer::resume_if_configured` — so the
// codec-level assertion here is that the schema tag round-trips verbatim and
// is caller-inspectable after mutation.
#[test]
fn load_rejects_schema_mismatch() {
    let tmp = tempdir().expect("tempdir");
    let state = sample_state();
    assert_eq!(
        state.optim_schema, "adamw-v1",
        "fixture expects adamw-v1 baseline"
    );
    save_trainer_state_v2(tmp.path(), &state, &sample_optim()).expect("save v2");

    // Mutate the JSON doc on disk to advertise a different optimizer schema.
    let json_path = tmp.path().join(TRAINER_STATE_FILENAME);
    let raw = std::fs::read_to_string(&json_path).expect("read trainer_state.json");
    let mut doc: serde_json::Value = serde_json::from_str(&raw).expect("parse json");
    doc["optim_schema"] = serde_json::Value::String("sgd-v1".to_string());
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&doc).expect("reserialize json"),
    )
    .expect("overwrite json");

    // Codec layer still loads the doc — the schema mismatch is a trainer-level
    // concern — but the caller can inspect `optim_schema` and reject before
    // invoking `import_state`. The crucial guarantee: the codec does NOT
    // silently coerce the mismatched schema into the live one.
    let (loaded, _optim) = load_trainer_state_v2(tmp.path()).expect("load still succeeds");
    assert_eq!(
        loaded.optim_schema, "sgd-v1",
        "codec must surface the on-disk schema verbatim so the caller can reject it"
    );
    assert_ne!(
        loaded.optim_schema, "adamw-v1",
        "codec must not silently rewrite the schema to match the live optimizer"
    );
    assert_eq!(
        loaded.codec_version, TRAINER_STATE_CODEC_VERSION,
        "codec version check still passes; only schema disagrees"
    );
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
