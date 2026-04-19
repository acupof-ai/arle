use autograd::{Backend, CpuBackend, Result};
#[cfg(feature = "metal")]
use autograd::{Tape, TensorStore, ops::matmul, ops::sum};
#[cfg(feature = "metal")]
use std::sync::{Arc, Mutex};

// Serialize all Metal-backend tests in this file so the process-global
// `METAL_EVAL_COUNT` counter has a quiet window for the eval-count
// acceptance test (`metal_single_forward_backward_step_triggers_one_eval`).
// Without this, parallel Metal tests concurrently bump the counter and
// the assertion sees a non-deterministic delta. The MLX_GUARD inside the
// backend serializes individual FFI calls but not the read→reset→step→read
// measurement window this test needs.
#[cfg(feature = "metal")]
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>()
}

#[test]
fn cpu_upload_readback_round_trip_matches_bytes() -> Result<()> {
    let backend = CpuBackend;
    let input = vec![1.0_f32, -2.5, 3.25, 0.0];
    let handle = backend.upload(&input, &[2, 2])?;
    let roundtrip = backend.readback(&handle)?;

    assert_eq!(f32_bytes(&roundtrip), f32_bytes(&input));
    Ok(())
}

#[test]
fn cpu_eval_is_noop() -> Result<()> {
    let backend = CpuBackend;
    let handle = backend.upload(&[4.0_f32, 5.0, 6.0], &[3])?;

    backend.eval(&[&handle])?;
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_upload_eval_readback_round_trip_matches_bytes() -> Result<()> {
    use autograd::backend_metal::MetalBackend;

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");
    let backend = MetalBackend;
    let input = vec![0.5_f32, -1.25, 2.0, 4.5];
    let handle = backend.upload(&input, &[2, 2])?;

    backend.eval(&[&handle])?;
    let roundtrip = backend.readback(&handle)?;

    assert_eq!(f32_bytes(&roundtrip), f32_bytes(&input));
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_eval_batches_multiple_handles() -> Result<()> {
    use autograd::backend_metal::MetalBackend;

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");
    let backend = MetalBackend;
    let first_input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let second_input = vec![-4.0_f32, 8.0, 16.0, 32.0];
    let first = backend.upload(&first_input, &[2, 2])?;
    let second = backend.upload(&second_input, &[2, 2])?;

    backend.eval(&[&first, &second])?;
    let first_roundtrip = backend.readback(&first)?;
    let second_roundtrip = backend.readback(&second)?;

    assert_eq!(f32_bytes(&first_roundtrip), f32_bytes(&first_input));
    assert_eq!(f32_bytes(&second_roundtrip), f32_bytes(&second_input));
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_handle_drops_cleanly_on_scope_exit() -> Result<()> {
    use autograd::backend_metal::MetalBackend;

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");
    let backend = MetalBackend;
    {
        let _handle = backend.upload(&[7.0_f32, 8.0, 9.0, 10.0], &[2, 2])?;
    }

    let fresh = backend.upload(&[11.0_f32, 12.0], &[2])?;
    backend.eval(&[&fresh])?;
    let roundtrip = backend.readback(&fresh)?;
    assert_eq!(f32_bytes(&roundtrip), f32_bytes(&[11.0_f32, 12.0]));
    Ok(())
}

/// M5.3a acceptance (`docs/plans/m5.3-device-resident-tensor.md` §5): a
/// single forward+backward pass of `y = x @ w; loss = y.sum()` through the
/// Metal backend must resolve to a small, deterministic number of
/// `mlx_eval` boundaries — not one per op (the pre-M5.3a degenerate path
/// that the 2026-04-18 TinyLM bench flagged as 1.9× slower than CPU).
///
/// Observed breakdown post-M5.3a.2 + matmul-backward-on-GPU
/// (`2026-04-20-matmul-backward-gpu.md`):
///
/// | Op                 | Forward eval | Backward eval |
/// | ------------------ | :----------: | :-----------: |
/// | `matmul` (lazy)    | 0            | —             |
/// | `sum` (host op)    | 1 (readback) | —             |
/// | `matmul_backward`  | —            | 1 (eval+read) |
///
/// `matmul` composes into MLX's lazy graph with zero evals; the only
/// host-forcing points are (a) `sum`'s `ensure_host(y)` and (b) the
/// `mlx_matmul_backward` FFI helper's self-contained eval+readback for
/// each requested gradient side. `need_grad_a=false` here (x is a leaf
/// input, not a parameter) so only one gradient matmul runs on device.
///
/// Upper bound = 2. The stretch goal of 1 requires a lazy
/// `Backend::matmul_backward` that returns unevaluated `DeviceHandle`s
/// and defers the eval to a terminal flush — tracked as a follow-up to
/// M5.3b once more ops go device-resident.
///
/// Regression signal: if this count climbs back above 2, a host round-trip
/// has sneaked into the middle of the graph (MLX's degenerate 1-op-per-eval
/// pattern). Trace the caller and collapse to the end-of-tape flush.
#[cfg(feature = "metal")]
#[test]
fn metal_single_forward_backward_step_has_bounded_eval_count() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let d_in = 32usize;
    let d_out = 16usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    // Inputs: tiny deterministic values so arithmetic fits in f32 noise tol.
    let x_data: Vec<f32> = (0..d_in).map(|i| (i as f32) * 0.01).collect();
    let w_data: Vec<f32> = (0..d_in * d_out)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();

    let x = store.from_slice(&x_data, &[1, d_in])?;
    let w = store.from_slice(&w_data, &[d_in, d_out])?;
    store.get_mut(w).expect("w exists").requires_grad = true;

    // Reset the counter AFTER allocation so from_slice / initialization work
    // doesn't pollute the measurement window.
    reset_eval_count();

    let y = matmul(x, w, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let loss = sum(y, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    // Assert each stage against its expected counter value. The goal of 0
    // after forward matmul (lazy) is the core M5.3a.2 invariant: matmul
    // never forces an eval on its own. If `after_matmul > 0`, the lazy
    // graph has regressed.
    assert_eq!(
        after_matmul, 0,
        "forward matmul must not trigger an eval (lazy graph); saw {after_matmul}"
    );
    assert_eq!(
        after_sum, 1,
        "sum forces one readback of the matmul output; saw {after_sum}"
    );
    assert!(
        after_backward <= 2,
        "forward+backward eval count must be bounded; saw {after_backward} \
         (after_matmul={after_matmul} after_sum={after_sum})"
    );
    Ok(())
}
