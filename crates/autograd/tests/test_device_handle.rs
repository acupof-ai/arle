use autograd::{Backend, CpuBackend, Result};
#[cfg(feature = "metal")]
use autograd::{Tape, TensorStore, ops::exp, ops::log_softmax, ops::matmul, ops::silu, ops::sum};
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

/// M5.3a + M5.3b.1 acceptance (`docs/plans/m5.3-device-resident-tensor.md` §5):
/// a single forward+backward pass of `y = x @ w; loss = y.sum()` through
/// the Metal backend must resolve to a small, deterministic number of
/// `mlx_eval` boundaries — not one per op (the pre-M5.3a degenerate path
/// that the 2026-04-18 TinyLM bench flagged as 1.9× slower than CPU).
///
/// Observed breakdown post-M5.3b.1 (lazy `sum_all` on Metal + batched
/// `tape.backward` flush via `TensorStore::flush_to_host_batch`):
///
/// | Stage                    | Eval count delta | Why                                   |
/// | ------------------------ | :--------------: | ------------------------------------- |
/// | `matmul` (lazy)          | 0                | composes into MLX graph               |
/// | `sum` (lazy, M5.3b.1)    | 0                | `reshape -> sum_axis`, no eval        |
/// | `tape.backward` flush    | 1                | one batched `mlx_eval` for {y, loss}  |
/// | `matmul_backward` (FFI)  | 1                | self-contained eval+readback          |
///
/// Both `matmul` and `sum` now compose into the MLX lazy graph with zero
/// evals. `tape.backward` collapses every Dirty::Device output (here `y`
/// and `loss`) into a single `Backend::eval(&[handle_y, handle_loss])`
/// call before walking the backward graph — MLX realizes shared upstream
/// nodes once, so the per-id `readback`s after that are O(copy). Without
/// this batching, M5.3b.1's lazy `sum` would *increase* the steady-state
/// eval count (y + loss become two Dirty::Device outputs where pre-M5.3b.1
/// only y had to be flushed). The `mlx_matmul_backward` FFI then runs its
/// own internal eval to compute `grad_b = x^T @ dC`; `need_grad_a=false`
/// since `x` is a leaf input, so only one gradient SGEMM runs on device.
///
/// Stretch goal of `after_backward <= 1` requires a lazy
/// `Backend::matmul_backward` that returns unevaluated `DeviceHandle`s
/// (folding the only remaining intra-graph eval into the terminal flush).
/// Out of M5.3b.1 scope.
///
/// Regression signal: if this count climbs back above its bound, either
/// a host round-trip has sneaked into the middle of the graph (trace the
/// new `ensure_host` caller) or the batched-flush helper has been bypassed
/// (check `tape::backward`).
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

    // The two assert_eq! gates are the strict M5.3 invariants; the
    // assert! on `after_backward` is the bounded-flush regression bound.
    assert_eq!(
        after_matmul, 0,
        "forward matmul must not trigger an eval (lazy graph); saw {after_matmul}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 2,
        "forward+backward eval count must be bounded; saw {after_backward} \
         (after_matmul={after_matmul} after_sum={after_sum})"
    );
    Ok(())
}

/// M5.3b.2 acceptance: the forward chain `matmul → log_softmax → sum` now
/// composes fully into the MLX lazy graph — zero `mlx_eval` calls through
/// the forward — mirroring `metal_single_forward_backward_step_has_bounded_eval_count`
/// but with log_softmax folded in.
///
/// Observed breakdown post-M5.3b.2:
///
/// | Stage                      | Eval count delta | Why                                                  |
/// | -------------------------- | :--------------: | ---------------------------------------------------- |
/// | `matmul` (lazy, M5.3a)     | 0                | composes into MLX graph                              |
/// | `log_softmax` (lazy)       | 0                | `mlx_logsumexp_axis + mlx_subtract`, no eval         |
/// | `sum` (lazy, M5.3b.1)      | 0                | `reshape -> sum_axis`, no eval                       |
/// | `tape.backward` flush      | 1                | one batched `mlx_eval` for Dirty::Device outputs     |
/// | `matmul_backward` (FFI)    | 1                | self-contained eval+readback for `grad_b = xᵀ @ dC`  |
/// | `log_softmax_backward`     | ≤1 (slack)       | host-mul readbacks `y` (the tape-saved output)       |
///
/// `after_backward <= 3` budgets the tape flush + matmul_backward FFI
/// eval, plus one unit of slack for the log_softmax backward host-mul
/// which currently forces an `ensure_host(y)`. Pushing that to zero
/// requires a device-side softmax backward (out of M5.3b.2 scope).
///
/// Regression signal: if `after_log_softmax > 0`, an eager path has
/// crept back into `ops/softmax.rs::log_softmax`. If `after_backward`
/// climbs above 3, trace the new `ensure_host` caller (or check that
/// the batched-flush helper in `tape::backward` still groups every
/// Dirty::Device output).
#[cfg(feature = "metal")]
#[test]
fn metal_log_softmax_forward_backward_records_current_eval_cost() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let d_in = 32usize;
    let d_out = 16usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let x_data: Vec<f32> = (0..d_in).map(|i| (i as f32) * 0.01).collect();
    let w_data: Vec<f32> = (0..d_in * d_out)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();

    let x = store.from_slice(&x_data, &[1, d_in])?;
    let w = store.from_slice(&w_data, &[d_in, d_out])?;
    store.get_mut(w).expect("w exists").requires_grad = true;

    reset_eval_count();

    let y = matmul(x, w, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let logp = log_softmax(y, &mut store, &mut tape)?;
    let after_log_softmax = eval_count();
    let loss = sum(logp, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_log_softmax, 0,
        "M5.3b.2: lazy log_softmax must not force an eval; saw {after_log_softmax}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward} \
         (after_matmul={after_matmul} after_log_softmax={after_log_softmax} \
          after_sum={after_sum}). Budget = tape.backward flush + \
          mlx_matmul_backward FFI eval + 1 slack for log_softmax_backward's \
          host-mul readback of the saved `y`."
    );
    Ok(())
}

/// M5.3b.3 acceptance: `silu` forward must compose into the MLX lazy graph
/// with no `mlx_eval` for Dirty::Device inputs. The forward chain
/// `matmul → silu → sum` records zero evals; the backward flush + the
/// `mlx_matmul_backward` FFI eval + one unit of slack for
/// `silu_backward`'s host-mul readback of the saved `x` bound the total.
#[cfg(feature = "metal")]
#[test]
fn metal_silu_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let d_in = 16usize;
    let d_out = 16usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let x_data: Vec<f32> = (0..d_in).map(|i| (i as f32) * 0.01).collect();
    let w_data: Vec<f32> = (0..d_in * d_out)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();

    let x = store.from_slice(&x_data, &[1, d_in])?;
    let w = store.from_slice(&w_data, &[d_in, d_out])?;
    store.get_mut(w).expect("w exists").requires_grad = true;

    reset_eval_count();

    // `y` is Dirty::Device (lazy matmul output); silu must stay lazy on it.
    let y = matmul(x, w, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let act = silu(y, &mut store, &mut tape)?;
    let after_silu = eval_count();
    let loss = sum(act, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_silu, 0,
        "M5.3b.3: lazy silu forward must not force an eval; saw {after_silu}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward} \
         (after_matmul={after_matmul} after_silu={after_silu} \
          after_sum={after_sum}). Budget = tape.backward flush + \
          mlx_matmul_backward FFI eval + 1 slack for silu_backward's \
          host readback of the saved `x`."
    );
    Ok(())
}

/// M5.3b.4 acceptance: `exp` forward must compose into the MLX lazy graph
/// with no `mlx_eval` for Dirty::Device inputs. The forward chain
/// `matmul → exp → sum` records zero evals; the backward flush + the
/// `mlx_matmul_backward` FFI eval + one unit of slack for `exp_backward`'s
/// host-mul readback of the saved output bound the total.
#[cfg(feature = "metal")]
#[test]
fn metal_exp_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let d_in = 16usize;
    let d_out = 16usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    // Keep magnitudes small so exp stays in a well-conditioned range.
    let x_data: Vec<f32> = (0..d_in).map(|i| (i as f32) * 0.01).collect();
    let w_data: Vec<f32> = (0..d_in * d_out)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();

    let x = store.from_slice(&x_data, &[1, d_in])?;
    let w = store.from_slice(&w_data, &[d_in, d_out])?;
    store.get_mut(w).expect("w exists").requires_grad = true;

    reset_eval_count();

    // `y` is Dirty::Device (lazy matmul output); exp must stay lazy on it.
    let y = matmul(x, w, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let act = exp(y, &mut store, &mut tape)?;
    let after_exp = eval_count();
    let loss = sum(act, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_exp, 0,
        "M5.3b.4: lazy exp forward must not force an eval; saw {after_exp}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward} \
         (after_matmul={after_matmul} after_exp={after_exp} \
          after_sum={after_sum}). Budget = tape.backward flush + \
          mlx_matmul_backward FFI eval + 1 slack for exp_backward's \
          host mul_forward on the saved output."
    );
    Ok(())
}
