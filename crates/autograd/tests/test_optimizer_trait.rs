//! Trait-polymorphism smoke test for the `Optimizer` trait.
//!
//! Exercises AdamW through a `&mut dyn Optimizer`, confirming that every
//! trait method (`step` / `zero_grad` / `lr` / `set_lr` / `state_schema` /
//! `export_state`) dispatches correctly. The goal is to catch regressions if
//! the trait shape drifts from the concrete impl — numeric correctness of
//! AdamW itself is covered by `m1_adamw.rs` and `test_adamw_state.rs`.

use autograd::{AdamW, Optimizer, Tensor, TensorId, TensorStore};

/// Minimal single-param setup: shape [4], constant gradient values copied in
/// per step. Matches the pattern used by `test_adamw_state.rs`.
fn build_single_param() -> (TensorStore, TensorId, Vec<f32>) {
    let mut store = TensorStore::default();
    let param = store.alloc(
        Tensor::new(vec![0.1, -0.2, 0.3, -0.4], vec![4], true)
            .expect("param tensor is well-formed"),
    );
    let grad =
        store.alloc(Tensor::new(vec![0.0; 4], vec![4], false).expect("grad tensor is well-formed"));
    store.get_mut(param).expect("param exists").grad = Some(grad);

    let grad_values = vec![0.01_f32, -0.02, 0.03, -0.04];
    (store, param, grad_values)
}

fn copy_grad(store: &mut TensorStore, param: TensorId, values: &[f32]) {
    let grad_id = store
        .get(param)
        .and_then(|t| t.grad)
        .expect("param has grad");
    let grad = store.get_mut(grad_id).expect("grad tensor exists");
    grad.data.copy_from_slice(values);
}

#[test]
fn adamw_dispatches_through_optimizer_trait() {
    const LR: f32 = 0.01;
    let (mut store, param, grad_values) = build_single_param();
    let mut adamw = AdamW::new(LR, (0.9, 0.999), 1e-8, 0.05);

    // Cast to the trait object and drive the whole lifecycle through it.
    let opt: &mut dyn Optimizer = &mut adamw;

    // Schema tag is stable and identifies the doc layout.
    assert_eq!(opt.state_schema(), "adamw-v1");

    // LR round-trip: read the constructor value, bump it, read back.
    assert!(
        (opt.lr() - LR).abs() < f32::EPSILON,
        "lr() should match ctor"
    );
    opt.set_lr(0.02);
    assert!(
        (opt.lr() - 0.02).abs() < f32::EPSILON,
        "set_lr must update the backing field"
    );
    // Restore for the rest of the test so the step math isn't surprising.
    opt.set_lr(LR);

    // Two steps. After each we reload gradients (AdamW reads host-side).
    copy_grad(&mut store, param, &grad_values);
    opt.step(&mut store, &[param])
        .expect("step 1 dispatches without error");

    copy_grad(&mut store, param, &grad_values);
    opt.step(&mut store, &[param])
        .expect("step 2 dispatches without error");

    // Export produces a one-param doc for the named tensor.
    let names = vec![(param, "p_only".to_string())];
    let state = opt.export_state(&names);
    assert_eq!(
        state.params.len(),
        1,
        "export_state should surface the single tracked param"
    );
    assert_eq!(state.params[0].name, "p_only");
    assert_eq!(state.step, 2, "two trait-dispatched steps recorded");
    assert_eq!(
        state.skipped_export, 0,
        "all tracked internal state is covered by names"
    );

    // zero_grad through the trait should clear the grad buffer in place.
    let grad_id = store
        .get(param)
        .and_then(|t| t.grad)
        .expect("param has grad");
    store
        .get_mut(grad_id)
        .expect("grad tensor exists")
        .data
        .copy_from_slice(&grad_values);
    opt.zero_grad(&mut store, &[param]);
    let cleared = store.to_host(grad_id).expect("host copy of grad");
    assert!(
        cleared.iter().all(|&x| x == 0.0),
        "zero_grad via trait should leave grad tensor all-zero"
    );
}
