//! Round-trip tests for the opaque AdamW state codec.
//!
//! Setup: two parameters (shapes `[4]` and `[3, 2]`), two AdamW instances A
//! and B with identical hyperparams, fed identical gradients. After step 2
//! export state from A, create a fresh instance C, import, then step all
//! three a third time. A and C should match; B diverges (sanity check — C
//! is supposed to track A, not B, because B never sees the restore).

use autograd::{
    Tensor, TensorId, TensorStore,
    adamw_state::{AdamWParamState, AdamWState},
    optim::AdamW,
};

struct Harness {
    store: TensorStore,
    opt: AdamW,
    params: Vec<TensorId>,
    grad_values: Vec<Vec<f32>>,
}

fn build_harness() -> Harness {
    let mut store = TensorStore::default();
    let param_a = store.alloc(
        Tensor::new(vec![0.1, -0.2, 0.3, -0.4], vec![4], true)
            .expect("param_a tensor is well-formed"),
    );
    let param_b = store.alloc(
        Tensor::new(vec![0.05, 0.1, -0.15, 0.2, -0.25, 0.3], vec![3, 2], true)
            .expect("param_b tensor is well-formed"),
    );

    // Pre-allocate grad tensors; we'll copy fresh values in per-step.
    let grad_a = store
        .alloc(Tensor::new(vec![0.0; 4], vec![4], false).expect("grad_a tensor is well-formed"));
    let grad_b = store
        .alloc(Tensor::new(vec![0.0; 6], vec![3, 2], false).expect("grad_b tensor is well-formed"));
    store.get_mut(param_a).expect("param_a exists").grad = Some(grad_a);
    store.get_mut(param_b).expect("param_b exists").grad = Some(grad_b);

    let grad_values = vec![
        vec![0.01_f32, -0.02, 0.03, -0.04],
        vec![0.005_f32, -0.01, 0.015, -0.02, 0.025, -0.03],
    ];

    Harness {
        store,
        opt: AdamW::new(0.01, (0.9, 0.999), 1e-8, 0.05),
        params: vec![param_a, param_b],
        grad_values,
    }
}

fn step_once(h: &mut Harness) {
    // Copy the constant gradient values back into the grad tensors each step
    // (AdamW does not zero them out; step() reads them host-side).
    for (idx, &param_id) in h.params.iter().enumerate() {
        let grad_id = h
            .store
            .get(param_id)
            .and_then(|t| t.grad)
            .expect("param has grad tensor");
        let grad = h.store.get_mut(grad_id).expect("grad tensor exists");
        grad.data.copy_from_slice(&h.grad_values[idx]);
    }
    h.opt.step(&h.params, &mut h.store);
}

fn param_host(h: &mut Harness, param_id: TensorId) -> Vec<f32> {
    h.store.to_host(param_id).expect("param host copy")
}

fn names(h: &Harness) -> Vec<(TensorId, String)> {
    vec![
        (h.params[0], "p_a".to_string()),
        (h.params[1], "p_b".to_string()),
    ]
}

fn copy_param_weights(src: &mut Harness, dst: &mut Harness) {
    // Simulates the weight half of a checkpoint reload: copy post-step-2
    // param tensor data from `src` into `dst` so only the optimizer state
    // differs between the two.
    let pairs: Vec<(TensorId, TensorId)> = src
        .params
        .iter()
        .copied()
        .zip(dst.params.iter().copied())
        .collect();
    for (src_id, dst_id) in pairs {
        let data = src.store.to_host(src_id).expect("src param host copy");
        let tensor = dst.store.get_mut(dst_id).expect("dst param exists");
        tensor.data.copy_from_slice(&data);
    }
}

#[test]
fn export_import_restores_state_and_matches_reference() {
    // Scenario:
    //   A = reference, takes steps 1, 2, 3.
    //   B = "naive reload" — after step 2, its weights are copied from A but
    //        its AdamW optimizer state is reset to fresh (no import). Then
    //        B takes step 3. B should diverge from A (moments were lost).
    //   C = "full reload" — after step 2, its weights are copied from A AND
    //        its AdamW state is imported from A's export. Then C takes
    //        step 3. C must match A within 1e-6.
    let mut a = build_harness();
    let mut b = build_harness();
    let mut c = build_harness();

    // Steps 1 and 2 on A only.
    step_once(&mut a);
    step_once(&mut a);

    // Snapshot A's state after step 2.
    let state = a.opt.export_state(&names(&a));
    assert_eq!(state.step, 2, "step counter should round-trip as u64");
    assert_eq!(state.params.len(), 2, "both params should be exported");
    assert_eq!(state.skipped_export, 0, "no unmapped internal state");

    // Serde round-trip — this is the real codec test.
    let json = serde_json::to_string(&state).expect("serialize AdamWState");
    let loaded: AdamWState = serde_json::from_str(&json).expect("deserialize AdamWState");

    // B and C both receive A's post-step-2 weights. Only C imports the moments.
    copy_param_weights(&mut a, &mut b);
    copy_param_weights(&mut a, &mut c);

    let restored = c
        .opt
        .import_state(&loaded, &names(&c))
        .expect("import succeeds");
    assert_eq!(restored, 2, "both params should be restored");

    // Step 3 on all three.
    step_once(&mut a);
    step_once(&mut b);
    step_once(&mut c);

    let param_ids: Vec<TensorId> = a.params.clone();
    for param_id in param_ids {
        let va = param_host(&mut a, param_id);
        let vb = param_host(&mut b, param_id);
        let vc = param_host(&mut c, param_id);
        let mut max_ab_delta = 0.0f32;
        for i in 0..va.len() {
            // Full reload (C) must match A.
            assert!(
                (va[i] - vc[i]).abs() < 1e-6,
                "A vs C mismatch at param {param_id}[{i}]: {} vs {}",
                va[i],
                vc[i],
            );
            max_ab_delta = max_ab_delta.max((va[i] - vb[i]).abs());
        }
        // Naive reload (B) lost the moments; with the same weights and one
        // final step it takes a different update than A. Require a strictly
        // non-zero delta (tiny gradients here, so the gap is small but real).
        assert!(
            max_ab_delta > 0.0,
            "B (naive reload, no moment import) should diverge from A on param {param_id} \
             (max |A-B| was {max_ab_delta})",
        );
    }
}

#[test]
fn fresh_without_import_diverges_from_restored() {
    // Complement to the main test: a fresh optimizer that only runs step 3
    // (no imported state, no steps 1-2) should NOT match A after A's step 3.
    // This is the "B diverges" sanity check demanded by the spec, applied to
    // a truly untrained twin.
    let mut a = build_harness();
    let mut untrained = build_harness();

    step_once(&mut a);
    step_once(&mut a);
    step_once(&mut a);

    // `untrained` runs exactly one step — no import, no prior history.
    step_once(&mut untrained);

    let pid_a = a.params[0];
    let pid_u = untrained.params[0];
    let va = param_host(&mut a, pid_a);
    let vu = param_host(&mut untrained, pid_u);

    let diverged = va.iter().zip(vu.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
    assert!(
        diverged,
        "fresh untrained optimizer should visibly diverge from A after 3 vs 1 step"
    );
}

#[test]
fn unknown_name_in_file_is_silently_skipped() {
    // Build a state that has an entry whose name is NOT in the caller's
    // current names slice. `import_state` must skip it silently and return
    // only the count it actually restored.
    let mut a = build_harness();
    step_once(&mut a);

    let mut state = a.opt.export_state(&names(&a));
    state.params.push(AdamWParamState {
        name: "phantom_param".to_string(),
        m: vec![0.0; 4],
        v: vec![0.0; 4],
        shape: vec![4],
    });

    let mut c = build_harness();
    let restored = c
        .opt
        .import_state(&state, &names(&c))
        .expect("import with unknown name still succeeds");
    assert_eq!(
        restored, 2,
        "only the 2 named params should be restored; phantom silently skipped",
    );
}

#[test]
fn shape_mismatch_is_an_error() {
    // After A has recorded moments with shape [4] for p_a, try to import a
    // file that claims p_a has shape [5]. Must error.
    let mut a = build_harness();
    step_once(&mut a);

    let mut state = a.opt.export_state(&names(&a));
    // Corrupt the shape on the first param (p_a expects [4]).
    state.params[0].shape = vec![5];
    state.params[0].m = vec![0.0; 5];
    state.params[0].v = vec![0.0; 5];

    // Fresh instance with a step first so it has a tracked shape to compare.
    let mut c = build_harness();
    step_once(&mut c);

    let err = c.opt.import_state(&state, &names(&c));
    assert!(err.is_err(), "shape mismatch must error");
}

#[test]
fn skipped_export_counts_entries_missing_from_names() {
    // If AdamW tracks state for a TensorId that the caller didn't list in
    // `names`, that entry is skipped and counted.
    let mut a = build_harness();
    step_once(&mut a);

    // Only name one of the two params.
    let partial = vec![(a.params[0], "p_a".to_string())];
    let state = a.opt.export_state(&partial);

    assert_eq!(state.params.len(), 1, "only p_a exported");
    assert_eq!(
        state.skipped_export, 1,
        "p_b was tracked in AdamW but unmapped by names → counted",
    );
}
