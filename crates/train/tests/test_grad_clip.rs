//! Tests for the `GradClip` trait and its `NoClip` / `GlobalNorm` impls.
//!
//! Setup: 2 params with hand-filled gradients whose true global L2 norm is
//! exactly sqrt(4) = 2.0 (param A's grad sums-of-squares = 1, param B's = 3).

use autograd::{Tensor, TensorId, TensorStore};
use train::grad_clip::{GlobalNorm, GradClip, NoClip, clip_grad_norm};

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
fn no_clip_reports_true_norm_and_leaves_grads_untouched() {
    // NoClip's contract (per the GradClip trait docstring) is to return the
    // pre-clip global L2 norm for logging, *without* mutating gradients.
    // Returning a hard-coded 0.0 would mask explode/vanish gradients in
    // unclipped baselines — see codex review P3 (2026-04-20).
    let (mut store, params) = setup_two_params_with_grads();
    let pre_norm = global_grad_l2(&params, &store);
    assert!((pre_norm - 2.0).abs() < 1e-6, "setup pre-norm != 2.0");

    let before = snapshot_grads(&params, &store);
    let mut clip = NoClip;
    let reported = clip.clip(&mut store, &params).expect("no_clip clip");
    let after = snapshot_grads(&params, &store);

    assert!(
        (reported - 2.0).abs() < 1e-4,
        "NoClip must report true pre-clip norm (~2.0), got {reported}"
    );
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

// GC-4 — guard silent divide-by-zero: GlobalNorm::new(0.0) must panic at
// construction, not at clip time; same for negative/NaN/Inf. The struct
// literal form `GlobalNorm { max_norm: 0.0 }` remains a no-op (covered
// above) for backwards compatibility with the existing call sites — the
// panic is opt-in via the explicit `new` constructor.
#[test]
#[should_panic(expected = "max_norm must be > 0.0")]
fn global_norm_zero_max_norm_panics_early() {
    let _clipper = GlobalNorm::new(0.0);
}

#[test]
#[should_panic(expected = "max_norm must be > 0.0")]
fn global_norm_negative_max_norm_panics_early() {
    let _clipper = GlobalNorm::new(-0.5);
}

#[test]
#[should_panic(expected = "max_norm must be > 0.0")]
fn global_norm_nan_max_norm_panics_early() {
    let _clipper = GlobalNorm::new(f32::NAN);
}

// GC-5 — guard trait-vs-free-fn drift: `GlobalNorm::clip` and the legacy
// `clip_grad_norm` free function must produce bitwise-identical post-clip
// gradients given the same inputs. Prevents silent divergence if one impl
// is optimised without the other.
#[test]
fn norm_computation_matches_legacy_free_fn() {
    let (mut store_a, params_a) = setup_two_params_with_grads();
    let (mut store_b, params_b) = setup_two_params_with_grads();

    // Sanity-check setups start identical.
    let before_a = snapshot_grads(&params_a, &store_a);
    let before_b = snapshot_grads(&params_b, &store_b);
    assert_eq!(before_a, before_b, "initial grads diverge");

    // A goes through the legacy free function.
    clip_grad_norm(&params_a, 1.0, &mut store_a);

    // B goes through the trait surface.
    let mut trait_clip = GlobalNorm { max_norm: 1.0 };
    let _ = trait_clip
        .clip(&mut store_b, &params_b)
        .expect("trait clip");

    let after_a = snapshot_grads(&params_a, &store_a);
    let after_b = snapshot_grads(&params_b, &store_b);
    assert_eq!(after_a.len(), after_b.len(), "shape drift across impls");
    for (pi, (a_grads, b_grads)) in after_a.iter().zip(after_b.iter()).enumerate() {
        assert_eq!(
            a_grads.len(),
            b_grads.len(),
            "param {pi} grad len drift across impls"
        );
        for (i, (a, b)) in a_grads.iter().zip(b_grads.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "param {pi} grad[{i}] bitwise mismatch: free-fn {a} vs trait {b}"
            );
        }
    }
}

// GC-6 — guard NaN from 0 / 0: empty params slice must return 0.0 and leave
// nothing to mutate — no panic, no NaN surfaced into the report.
#[test]
fn norm_on_empty_params_is_zero() {
    let (mut store, _params) = setup_two_params_with_grads();
    let empty: Vec<TensorId> = Vec::new();

    // Trait surface: `NoClip` on empty inputs → 0.0.
    let mut no_clip = NoClip;
    let reported_noclip = no_clip.clip(&mut store, &empty).expect("no_clip on empty");
    assert_eq!(reported_noclip, 0.0);
    assert!(
        reported_noclip.is_finite(),
        "NoClip empty report must be finite, got {reported_noclip}"
    );

    // `GlobalNorm::clip` on empty inputs must also report 0.0 pre-clip
    // without dividing by zero anywhere. Use the struct-literal form so the
    // new-constructor panic doesn't hide this path.
    let mut global = GlobalNorm { max_norm: 1.0 };
    let reported_global = global
        .clip(&mut store, &empty)
        .expect("global_norm on empty");
    assert_eq!(reported_global, 0.0);
    assert!(
        reported_global.is_finite(),
        "GlobalNorm empty pre-clip must be finite, got {reported_global}"
    );

    // Legacy free-function: empty params → early return, no panic, no mutation.
    clip_grad_norm(&empty, 1.0, &mut store);
}

// GC-7 — guard non-finite max_norm: NaN / ±Inf used to bypass the
// `max_norm <= 0.0` gate in `clip_grad_norm` (NaN comparisons always
// false) and then poison every gradient via `scale = max_norm /
// total_norm`. Codex review ef24ca6 P2. Any non-finite (or non-positive)
// value is now a documented no-op — matching the pretrain.rs binary's
// CLI warning path so all call sites stay consistent.
#[test]
fn non_finite_max_norm_is_noop() {
    for max_norm in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0_f32] {
        let (mut store, params) = setup_two_params_with_grads();
        let before = snapshot_grads(&params, &store);
        clip_grad_norm(&params, max_norm, &mut store);
        let after = snapshot_grads(&params, &store);
        assert_eq!(
            before, after,
            "clip_grad_norm({max_norm}) must be a no-op, grads mutated"
        );
        for (pi, grad) in after.iter().enumerate() {
            for (i, v) in grad.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "param {pi} grad[{i}] = {v} after non-finite max_norm={max_norm}"
                );
            }
        }
    }
}
