//! Tests for the `GradClip` trait and its `NoClip` / `GlobalNorm` impls.
//!
//! Setup: 2 params with hand-filled gradients whose true global L2 norm is
//! exactly sqrt(4) = 2.0 (param A's grad sums-of-squares = 1, param B's = 3).

use autograd::{Tensor, TensorId, TensorStore};
use train::grad_clip::{GlobalNorm, GradClip, NoClip};

/// Build a `TensorStore` with two params and pre-filled gradients.
///
/// Param shapes / grad values are chosen so the global L2 norm is 2.0:
///   * param A grad = `[1.0]`                          (sum-sq = 1)
///   * param B grad = `[1.0, 1.0, 1.0]`                (sum-sq = 3)
///   * total sum-sq = 4, sqrt = 2.0
fn setup_two_params_with_grads() -> (TensorStore, Vec<TensorId>) {
    let mut store = TensorStore::default();

    // Param A: scalar-shaped tensor, grad = [1.0].
    let param_a = store.alloc(
        Tensor::new(vec![0.0], vec![1], /* requires_grad = */ true).expect("param_a tensor"),
    );
    let grad_a = store.alloc(Tensor::new(vec![1.0], vec![1], false).expect("grad_a tensor"));
    store
        .accumulate_grad(param_a, grad_a)
        .expect("accumulate grad_a");

    // Param B: shape [3], grad = [1.0, 1.0, 1.0].
    let param_b = store.alloc(
        Tensor::new(vec![0.0; 3], vec![3], /* requires_grad = */ true).expect("param_b tensor"),
    );
    let grad_b = store.alloc(Tensor::new(vec![1.0; 3], vec![3], false).expect("grad_b tensor"));
    store
        .accumulate_grad(param_b, grad_b)
        .expect("accumulate grad_b");

    (store, vec![param_a, param_b])
}

fn global_grad_l2(params: &[TensorId], store: &TensorStore) -> f32 {
    let mut total_sq = 0.0_f32;
    for &pid in params {
        let grad_id = store.get(pid).and_then(|t| t.grad).expect("param has grad");
        let grad = store.get(grad_id).expect("grad tensor exists");
        total_sq += grad.data.iter().map(|v| v * v).sum::<f32>();
    }
    total_sq.sqrt()
}

fn snapshot_grads(params: &[TensorId], store: &TensorStore) -> Vec<Vec<f32>> {
    params
        .iter()
        .map(|&pid| {
            let grad_id = store.get(pid).and_then(|t| t.grad).expect("param has grad");
            store.get(grad_id).expect("grad tensor").data.clone()
        })
        .collect()
}

#[test]
fn no_clip_returns_zero_and_leaves_grads_untouched() {
    let (mut store, params) = setup_two_params_with_grads();
    let pre_norm = global_grad_l2(&params, &store);
    assert!((pre_norm - 2.0).abs() < 1e-6, "setup pre-norm != 2.0");

    let before = snapshot_grads(&params, &store);
    let mut clip = NoClip;
    let reported = clip.clip(&mut store, &params).expect("no_clip clip");
    let after = snapshot_grads(&params, &store);

    assert_eq!(reported, 0.0, "NoClip must report 0.0");
    assert_eq!(before, after, "NoClip must not modify grads");
}

#[test]
fn global_norm_below_threshold_rescales_grads() {
    let (mut store, params) = setup_two_params_with_grads();

    let mut clip = GlobalNorm { max_norm: 1.0 };
    let pre_clip = clip.clip(&mut store, &params).expect("global_norm clip");

    assert!(
        (pre_clip - 2.0).abs() < 1e-4,
        "pre-clip norm returned {pre_clip}, expected ~2.0"
    );

    let post_clip = global_grad_l2(&params, &store);
    assert!(
        (post_clip - 1.0).abs() < 1e-4,
        "post-clip norm {post_clip}, expected ~1.0"
    );
}

#[test]
fn global_norm_above_threshold_is_noop() {
    let (mut store, params) = setup_two_params_with_grads();
    let before = snapshot_grads(&params, &store);

    let mut clip = GlobalNorm { max_norm: 10.0 };
    let pre_clip = clip.clip(&mut store, &params).expect("global_norm clip");

    assert!(
        (pre_clip - 2.0).abs() < 1e-4,
        "pre-clip norm returned {pre_clip}, expected ~2.0"
    );

    let after = snapshot_grads(&params, &store);
    assert_eq!(
        before, after,
        "GlobalNorm with max_norm > true norm must not modify grads"
    );
}

#[test]
fn global_norm_zero_max_is_noop() {
    // Matches `clip_grad_norm`'s early-return on max_norm <= 0.0.
    let (mut store, params) = setup_two_params_with_grads();
    let before = snapshot_grads(&params, &store);

    let mut clip = GlobalNorm { max_norm: 0.0 };
    let pre_clip = clip.clip(&mut store, &params).expect("global_norm clip");

    assert!(
        (pre_clip - 2.0).abs() < 1e-4,
        "pre-clip norm returned {pre_clip}, expected ~2.0"
    );

    let after = snapshot_grads(&params, &store);
    assert_eq!(
        before, after,
        "GlobalNorm with max_norm=0.0 must be a no-op (matches clip_grad_norm)"
    );
}
