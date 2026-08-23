//! Tests for the `clip_grad_norm` free function.
//!
//! Setup: 2 params with hand-filled gradients whose true global L2 norm is
//! exactly sqrt(4) = 2.0 (param A's grad sums-of-squares = 1, param B's = 3).

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use autograd::backend_cuda::CudaBackend;
use autograd::{AdamW, Tensor, TensorId, TensorStore};
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use std::sync::Arc;
use train::grad_clip::{FiniteStepError, clip_grad_norm, finite_optimizer_step};

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
    let mut total_sq = 0.0_f64;
    for &pid in params {
        let grad_id = store.get(pid).and_then(|t| t.grad).expect("param has grad");
        let grad = store.get(grad_id).expect("grad tensor exists");
        total_sq += grad
            .data
            .iter()
            .map(|&v| {
                let v = f64::from(v);
                v * v
            })
            .sum::<f64>();
    }
    total_sq.sqrt() as f32
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

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn global_norm_rescales_cuda_device_grads() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping global_norm_rescales_cuda_device_grads: no CUDA device");
        return;
    };
    let mut store = TensorStore::with_backend(Arc::new(backend));

    let param_a = store.alloc(Tensor::new(vec![0.0], vec![1], true).expect("param_a"));
    let grad_a = store.alloc(Tensor::new(vec![1.0], vec![1], false).expect("grad_a"));
    store
        .accumulate_grad(param_a, grad_a)
        .expect("accumulate grad_a");

    let param_b = store.alloc(Tensor::new(vec![0.0; 3], vec![3], true).expect("param_b"));
    let grad_b = store.alloc(Tensor::new(vec![1.0; 3], vec![3], false).expect("grad_b"));
    store
        .accumulate_grad(param_b, grad_b)
        .expect("accumulate grad_b");

    let params = vec![param_a, param_b];
    for &param in &params {
        let grad_id = store
            .get(param)
            .and_then(|tensor| tensor.grad)
            .expect("param has grad");
        store.ensure_device(grad_id).expect("grad upload");
    }

    clip_grad_norm(&params, 1.0, &mut store).unwrap();

    let mut clipped = Vec::new();
    for &param in &params {
        let grad_id = store
            .get(param)
            .and_then(|tensor| tensor.grad)
            .expect("param has grad");
        clipped.push(store.to_host(grad_id).expect("clipped grad readback"));
    }
    assert_eq!(clipped.len(), 2);
    for (param_index, grad) in clipped.iter().enumerate() {
        for (value_index, &value) in grad.iter().enumerate() {
            assert!(
                (value - 0.5).abs() < 1.0e-6,
                "param {param_index} grad[{value_index}] = {value}, expected 0.5"
            );
        }
    }
}

#[test]
fn global_norm_large_finite_grads_do_not_overflow_to_zero() {
    let mut store = TensorStore::default();
    let param = store.alloc(
        Tensor::new(vec![0.0; 2], vec![2], /* requires_grad = */ true).expect("param tensor"),
    );
    let grad =
        store.alloc(Tensor::new(vec![1.0e20, -1.0e20], vec![2], false).expect("large grad tensor"));
    store
        .accumulate_grad(param, grad)
        .expect("accumulate large grad");

    clip_grad_norm(&[param], 1.0e20, &mut store).unwrap();

    let grad_id = store.get(param).and_then(|tensor| tensor.grad).unwrap();
    let clipped = store.get(grad_id).expect("clipped grad");
    assert!(
        clipped.data.iter().all(|value| value.is_finite()),
        "clipped gradients must stay finite: {:?}",
        clipped.data
    );
    assert!(
        clipped.data.iter().all(|value| *value != 0.0),
        "finite large gradients must not be zeroed by norm overflow: {:?}",
        clipped.data
    );
    let post_norm = global_grad_l2(&[param], &store);
    assert!(
        (post_norm - 1.0e20).abs() / 1.0e20 < 1.0e-5,
        "post-clip norm should be about 1e20, got {post_norm:e}"
    );
}

#[test]
fn global_norm_above_f32_max_still_scales_to_finite_grads() {
    let mut store = TensorStore::default();
    let param = store.alloc(
        Tensor::new(vec![0.0; 2], vec![2], /* requires_grad = */ true).expect("param tensor"),
    );
    let grad = store
        .alloc(Tensor::new(vec![f32::MAX, -f32::MAX], vec![2], false).expect("max grad tensor"));
    store
        .accumulate_grad(param, grad)
        .expect("accumulate max grad");

    clip_grad_norm(&[param], 1.0e38, &mut store).unwrap();

    let grad_id = store.get(param).and_then(|tensor| tensor.grad).unwrap();
    let clipped = store.get(grad_id).expect("clipped grad");
    assert!(
        clipped.data.iter().all(|value| value.is_finite()),
        "clipped gradients must stay finite: {:?}",
        clipped.data
    );
    assert!(
        clipped.data.iter().all(|value| *value != 0.0),
        "gradients with finite true scale must not be zeroed: {:?}",
        clipped.data
    );
    let post_norm = global_grad_l2(&[param], &store);
    assert!(
        (post_norm - 1.0e38).abs() / 1.0e38 < 1.0e-5,
        "post-clip norm should be about 1e38, got {post_norm:e}"
    );
}

#[test]
fn finite_step_rejects_non_finite_grad_without_mutating_params() {
    let mut store = TensorStore::default();
    let param = store.alloc(Tensor::new(vec![3.0], vec![1], true).unwrap());
    let grad = store.alloc(Tensor::new(vec![f32::NAN], vec![1], false).unwrap());
    store.accumulate_grad(param, grad).unwrap();
    let mut optimizer = AdamW::new(0.1, (0.9, 0.999), 1e-8, 0.0);

    let err = finite_optimizer_step(1.0, &[param], 1.0, &mut optimizer, &mut store)
        .expect_err("non-finite gradient must reject the whole update");

    assert!(matches!(err, FiniteStepError::NonFiniteGradNorm(norm) if norm.is_nan()));
    assert_eq!(store.get(param).unwrap().data, vec![3.0]);
    assert!(store.get(param).unwrap().grad.is_none());
}

#[test]
fn finite_step_rejects_non_finite_loss_and_clears_pending_grads() {
    let (mut store, params) = setup_two_params_with_grads();
    let before: Vec<Vec<f32>> = params
        .iter()
        .map(|&param| store.get(param).unwrap().data.clone())
        .collect();
    let mut optimizer = AdamW::new(0.1, (0.9, 0.999), 1e-8, 0.0);

    let err = finite_optimizer_step(f32::INFINITY, &params, 1.0, &mut optimizer, &mut store)
        .expect_err("non-finite loss must reject before optimizer mutation");

    assert!(matches!(err, FiniteStepError::NonFiniteLoss(value) if value.is_infinite()));
    let after: Vec<Vec<f32>> = params
        .iter()
        .map(|&param| store.get(param).unwrap().data.clone())
        .collect();
    assert_eq!(after, before);
    assert!(
        params
            .iter()
            .all(|&param| store.get(param).unwrap().grad.is_none())
    );
}

// Non-finite max_norm used to bypass the `max_norm <= 0.0` gate in
// `clip_grad_norm` (NaN comparisons always
// false) and then poison every gradient via `scale = max_norm /
// total_norm`. Codex review ef24ca6 P2. Any non-finite (or non-positive)
// value is now a documented no-op, matching the CLI warning path so all call
// sites stay consistent.
#[test]
fn non_finite_max_norm_is_noop() {
    for max_norm in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0_f32] {
        let (mut store, params) = setup_two_params_with_grads();
        let before = snapshot_grads(&params, &store);
        clip_grad_norm(&params, max_norm, &mut store).unwrap();
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

/// A grad can be device-resident on the CPU backend too — `ChunkSum` allocates
/// its accumulator that way, so `data` is empty and only the handle is real.
/// Reading `data` there silently contributes 0 to the norm and scales nothing
/// when clipping (2026-08-04: the CP gate's f32 anchor read exactly 0.0).
#[test]
fn cpu_backend_device_resident_grad_is_counted_and_clipped() {
    let mut store = TensorStore::default();
    let param = store.alloc(Tensor::new(vec![0.0; 4], vec![4], true).expect("param"));
    let handle = store
        .backend()
        .upload(&[1.0, 1.0, 1.0, 1.0], &[4])
        .expect("device grad handle");
    let grad = store
        .alloc_device_tensor(vec![4], handle)
        .expect("device grad tensor");
    store.accumulate_grad(param, grad).expect("accumulate");

    let params = [param];
    let norm = train::grad_clip::compute_global_norm_f64(&params, &store).unwrap();
    assert!(
        (norm - 2.0).abs() < 1.0e-6,
        "device-resident grad on the CPU backend must count toward the norm, got {norm}"
    );

    clip_grad_norm(&params, 1.0, &mut store).unwrap();
    let clipped = train::grad_clip::compute_global_norm_f64(&params, &store).unwrap();
    assert!(
        (clipped - 1.0).abs() < 1.0e-6,
        "clipping must scale a device-resident grad, norm stayed {clipped}"
    );
}
