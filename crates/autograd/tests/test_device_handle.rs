use autograd::{Backend, CpuBackend, Result};
#[cfg(feature = "metal")]
use autograd::{
    Tape, TensorStore, ops::embedding, ops::exp, ops::gather_last_dim, ops::gelu, ops::log_softmax,
    ops::matmul, ops::rmsnorm, ops::rope, ops::silu, ops::sum,
};
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

/// M5.3b.5 acceptance: `rope` forward must compose into the MLX lazy graph
/// with no `mlx_eval` for a Dirty::Device `x`. cos/sin stay host
/// (typical Qwen cache layout). The forward chain
/// `ensure_device(x) → rope → sum` records zero evals; backward pays
/// tape.backward's batch flush + the eager `rope_forward` inside
/// `rope_backward` (one internal eval from its `eval_and_readback` tail).
#[cfg(feature = "metal")]
#[test]
fn metal_rope_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let batch = 1usize;
    let heads = 2usize;
    let seq = 4usize;
    let head_dim = 8usize;
    let half_dim = head_dim / 2;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let x_len = batch * heads * seq * head_dim;
    let x_data: Vec<f32> = (0..x_len).map(|i| (i as f32) * 0.01).collect();
    let cos_data: Vec<f32> = (0..seq * half_dim)
        .map(|i| ((i as f32) * 0.1).cos())
        .collect();
    let sin_data: Vec<f32> = (0..seq * half_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();

    let x = store.from_slice(&x_data, &[batch, heads, seq, head_dim])?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    let cos = store.from_slice(&cos_data, &[seq, half_dim])?;
    let sin = store.from_slice(&sin_data, &[seq, half_dim])?;

    // Force `x` device-resident so the rope dispatch hits the lazy branch.
    // ensure_device uploads but does NOT call mlx_eval; confirm by resetting
    // the counter right after.
    store.ensure_device(x)?;
    reset_eval_count();

    let rotated = rope(x, cos, sin, &mut store, &mut tape)?;
    let after_rope = eval_count();
    let loss = sum(rotated, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_rope, 0,
        "M5.3b.5: lazy rope forward must not force an eval; saw {after_rope}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward} \
         (after_rope={after_rope} after_sum={after_sum}). Budget = \
         tape.backward batch flush + rope_backward's eager \
         `backend.rope_forward` internal eval (one eval_and_readback)."
    );

    // Parity: same inputs on CpuBackend must produce numerically close
    // output. Readback the rotated device tensor via tape.backward's flush
    // side-effect, then rerun through the CPU path.
    let metal_out = store
        .get(rotated)
        .expect("rotated tensor exists")
        .data
        .clone();
    let cpu = CpuBackend;
    let cpu_out = cpu.rope_forward(
        &x_data,
        &[batch, heads, seq, head_dim],
        &cos_data,
        &sin_data,
    )?;
    assert_eq!(metal_out.len(), cpu_out.len());
    for (m, c) in metal_out.iter().zip(cpu_out.iter()) {
        let diff: f32 = (m - c).abs();
        assert!(diff <= 1e-5, "rope parity: metal={m} cpu={c} diff={diff}");
    }

    Ok(())
}

/// M5.3b.6 acceptance: `rmsnorm` forward must compose into the MLX lazy
/// graph with no `mlx_eval` for a Dirty::Device `x`. Weight stays
/// host-resident (typical per-layer RMSNorm weight shape `[hidden]`).
/// `matmul → rmsnorm → sum` records zero evals through the forward.
/// Backward pays tape.backward's batch flush plus the host-eager
/// `rmsnorm_backward` body (which re-reads `x` and `weight`; the lazy
/// branch's empty `inv_rms` sentinel triggers a host recompute using
/// the `eps` now threaded on `RMSNormCtx`).
#[cfg(feature = "metal")]
#[test]
fn metal_rmsnorm_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let rows = 4usize;
    let hidden = 16usize;
    let eps = 1e-5f32;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let x_data: Vec<f32> = (0..rows * hidden).map(|i| (i as f32) * 0.01).collect();
    let w_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
    // Unit matmul weight to keep the lazy op graph minimal but still
    // produce a Dirty::Device output feeding rmsnorm.
    let id_data: Vec<f32> = (0..hidden * hidden)
        .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
        .collect();

    let x = store.from_slice(&x_data, &[rows, hidden])?;
    let id = store.from_slice(&id_data, &[hidden, hidden])?;
    let w = store.from_slice(&w_data, &[hidden])?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    store.get_mut(w).expect("w exists").requires_grad = true;

    reset_eval_count();

    // `y` is Dirty::Device (lazy matmul output); rmsnorm must stay lazy.
    let y = matmul(x, id, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let normed = rmsnorm(y, w, eps, &mut store, &mut tape)?;
    let after_rmsnorm = eval_count();
    let loss = sum(normed, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_rmsnorm, 0,
        "M5.3b.6: lazy rmsnorm forward must not force an eval; saw {after_rmsnorm}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}. \
         Budget = tape.backward batch flush + mlx_matmul_backward FFI eval + \
         1 slack for the host recompute path."
    );

    // Parity: metal lazy output vs CpuBackend::rms_norm_forward on the
    // same inputs.
    let metal_out = store
        .get(normed)
        .expect("normed tensor exists")
        .data
        .clone();
    let cpu = CpuBackend;
    let cpu_out = cpu.rms_norm_forward(&x_data, &w_data, &[rows, hidden], eps)?;
    assert_eq!(metal_out.len(), cpu_out.len());
    for (m, c) in metal_out.iter().zip(cpu_out.iter()) {
        let diff: f32 = (m - c).abs();
        assert!(
            diff <= 1e-4,
            "rmsnorm parity: metal={m} cpu={c} diff={diff}"
        );
    }

    Ok(())
}

/// M5.3b.7 acceptance: `embedding` forward must compose into the MLX
/// lazy graph with no `mlx_eval` when the table is device-resident.
/// `embedding → rmsnorm → sum` records zero evals through the forward.
/// Since the table is brought device-resident here via `ensure_device`
/// (simulating an AdamW step's upload of updated weights), subsequent
/// forward calls would see it as Dirty::Both and still take the lazy
/// branch. Backward stays on the host scatter-add path, bounded by
/// tape.backward's batch flush.
#[cfg(feature = "metal")]
#[test]
fn metal_embedding_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let vocab = 8usize;
    let hidden = 16usize;
    let eps = 1e-5f32;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let table_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.01).collect();
    let w_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let table = store.from_slice(&table_data, &[vocab, hidden])?;
    store.get_mut(table).expect("table exists").requires_grad = true;
    let w = store.from_slice(&w_data, &[hidden])?;
    store.get_mut(w).expect("w exists").requires_grad = true;

    // Force the table device-resident so the lazy embedding branch
    // dispatches. ensure_device uploads but does not call mlx_eval.
    store.ensure_device(table)?;
    reset_eval_count();

    let indices = [3usize, 1, 5, 2];
    let hidden_states = embedding(table, &indices, &mut store, &mut tape)?;
    let after_embedding = eval_count();
    let normed = rmsnorm(hidden_states, w, eps, &mut store, &mut tape)?;
    let after_rmsnorm = eval_count();
    let loss = sum(normed, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_embedding, 0,
        "M5.3b.7: lazy embedding forward must not force an eval; saw {after_embedding}"
    );
    assert_eq!(
        after_rmsnorm, 0,
        "M5.3b.6: lazy rmsnorm forward must not force an eval; saw {after_rmsnorm}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}. \
         Budget = tape.backward batch flush + rmsnorm/embedding host \
         recompute slack."
    );

    // Parity: metal lazy gather vs CpuBackend::embedding_forward on the
    // same inputs.
    let metal_rows = {
        // rmsnorm is downstream of the embedding, so the tape flush has
        // already materialized `hidden_states` to host. Read back through
        // the standard tensor accessor.
        store
            .get(hidden_states)
            .expect("embedding output exists")
            .data
            .clone()
    };
    let cpu = CpuBackend;
    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let cpu_rows = cpu.embedding_forward(&table_data, vocab, hidden, &ids_i32)?;
    assert_eq!(metal_rows.len(), cpu_rows.len());
    for (m, c) in metal_rows.iter().zip(cpu_rows.iter()) {
        let diff: f32 = (m - c).abs();
        assert!(
            diff <= 1e-5,
            "embedding parity: metal={m} cpu={c} diff={diff}"
        );
    }

    Ok(())
}

/// M5.3b.8 acceptance: `gelu` forward must compose into the MLX lazy
/// graph as the erf form (`0.5 * x * (1 + erf(x/sqrt(2)))`) with no
/// `mlx_eval` for a Dirty::Device `x`. `matmul → gelu → sum` records
/// zero evals through the forward. Parity vs the inline erf formula
/// used by `ops::activation::gelu`'s CPU eager path stays ≤ 1e-4 — the
/// same erf formula, different erf implementations (MLX's builtin vs
/// libm::erff) agree to the ULP range both use for f32 erf.
#[cfg(feature = "metal")]
#[test]
fn metal_gelu_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    const INV_SQRT_2: f32 = 0.707_106_77_f32;

    let d_in = 16usize;
    let d_out = 16usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    // Keep magnitudes moderate so erf stays in a well-conditioned range
    // (|x| < 3-ish); matches the scale reaching GELU after a small matmul.
    let x_data: Vec<f32> = (0..d_in).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let id_data: Vec<f32> = (0..d_in * d_out)
        .map(|i| if i / d_out == i % d_out { 1.0 } else { 0.0 })
        .collect();

    let x = store.from_slice(&x_data, &[1, d_in])?;
    let id = store.from_slice(&id_data, &[d_in, d_out])?;
    store.get_mut(id).expect("id exists").requires_grad = true;

    reset_eval_count();

    // `y` is Dirty::Device (lazy matmul output); gelu must stay lazy on it.
    let y = matmul(x, id, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let act = gelu(y, &mut store, &mut tape)?;
    let after_gelu = eval_count();
    let loss = sum(act, &mut store, &mut tape)?;
    let after_sum = eval_count();
    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_gelu, 0,
        "M5.3b.8: lazy gelu forward must not force an eval; saw {after_gelu}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}. \
         Budget = tape.backward batch flush + mlx_matmul_backward FFI eval + \
         1 slack for gelu_backward's host erf derivative."
    );

    // Parity: metal lazy output vs CPU erf formula on the same inputs.
    let metal_out = store.get(act).expect("gelu output exists").data.clone();
    let cpu_out: Vec<f32> = x_data
        .iter()
        .map(|&v| 0.5 * v * (1.0 + libm::erff(v * INV_SQRT_2)))
        .collect();
    assert_eq!(metal_out.len(), cpu_out.len());
    for (m, c) in metal_out.iter().zip(cpu_out.iter()) {
        let diff: f32 = (m - c).abs();
        assert!(diff <= 1e-4, "gelu parity: metal={m} cpu={c} diff={diff}");
    }

    Ok(())
}

/// M5.3b.9 acceptance: `gather_last_dim` forward must compose into the
/// MLX lazy graph when `src` is Dirty::Device — the typical CE-loss
/// situation where logits come straight out of the final matmul. The
/// chain `matmul → gather_last_dim → sum` records zero evals through
/// the forward. Parity vs `CpuBackend::gather_last_dim_forward` on the
/// same inputs stays ≤ 1e-5.
#[cfg(feature = "metal")]
#[test]
fn metal_gather_last_dim_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let seq = 4usize;
    let vocab = 8usize;
    let hidden = 6usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let x_data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) * 0.03 - 0.2).collect();
    let w_data: Vec<f32> = (0..hidden * vocab)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();

    let x = store.from_slice(&x_data, &[seq, hidden])?;
    let w = store.from_slice(&w_data, &[hidden, vocab])?;
    store.get_mut(w).expect("w exists").requires_grad = true;

    reset_eval_count();

    // logits = x @ w  →  [seq, vocab], Dirty::Device from lazy matmul
    let logits = matmul(x, w, &mut store, &mut tape)?;
    let after_matmul = eval_count();

    let indices = [5usize, 2, 7, 0];
    let picked = gather_last_dim(logits, &indices, &mut store, &mut tape)?;
    let after_gather = eval_count();

    let loss = sum(picked, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_gather, 0,
        "M5.3b.9: lazy gather_last_dim forward must not force an eval; saw {after_gather}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}. \
         Budget = tape.backward batch flush + mlx_matmul_backward FFI eval + \
         1 slack for gather_backward's host scatter."
    );

    // Parity: metal lazy output vs CpuBackend::gather_last_dim_forward
    // on the same logits data.
    let metal_picked = store
        .get(picked)
        .expect("gather output exists")
        .data
        .clone();

    // Independently compute logits on the host for the CPU reference.
    let cpu = CpuBackend;
    let (host_logits, host_logits_shape) =
        cpu.matmul_forward(&x_data, &[seq, hidden], &w_data, &[hidden, vocab])?;
    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let cpu_picked = cpu.gather_last_dim_forward(&host_logits, &host_logits_shape, &ids_i32)?;

    assert_eq!(metal_picked.len(), cpu_picked.len());
    for (m, c) in metal_picked.iter().zip(cpu_picked.iter()) {
        let diff: f32 = (m - c).abs();
        assert!(
            diff <= 1e-5,
            "gather_last_dim parity: metal={m} cpu={c} diff={diff}"
        );
    }

    Ok(())
}
