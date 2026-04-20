#[cfg(feature = "metal")]
use autograd::{
    AdamW, Tape, Tensor, TensorStore, ops::add_broadcast, ops::causal_sdpa, ops::embedding,
    ops::exp, ops::gather_last_dim, ops::gelu, ops::log_softmax, ops::matmul, ops::mul_scalar,
    ops::reshape, ops::rmsnorm, ops::rope, ops::silu, ops::slice, ops::sum, ops::transpose,
    tensor::Dirty,
};
use autograd::{Backend, CpuBackend, Result};
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

/// M5.3b.10 acceptance: the device-backed AdamW path keeps `m`, `v`, and
/// the param device-resident across steps on Metal. Five fixed-seed steps
/// through `AdamW::new_with_device` must numerically match the host
/// reference (plain `AdamW::new`) to ≤ 1e-5 L2 diff, and after each step
/// the param's store tensor must be `Dirty::Device` — that's the flag that
/// gates whether the NEXT forward re-uploads or reuses the lazy MLX handle.
///
/// Eval-count budget: the backend `adamw_step` records exactly one terminal
/// `mlx_eval` per param to numerically commit the updated {param, m, v}
/// triple. With a single param under test, the delta from `reset_eval_count`
/// to after-step is therefore ≤ 2 (one optimizer eval + one unit of slack
/// — the test uses no tape.backward here, grads are planted directly, so
/// the backward flush eval is not counted). If a future diff batches
/// multiple params through one `mlx_eval`, this bound will tighten
/// automatically per-param but the total for N params will stay at 1.
///
/// Without this test, a regression that went back to `get_mut`-on-param
/// would silently re-introduce the ~200-param-per-step CPU→GPU re-upload
/// observed in pre-M5.3b.10 profiling, and the only signal would be
/// steady-state throughput dropping at scale. The `Dirty::Device` assertion
/// is the cheap structural catch.
#[cfg(feature = "metal")]
#[test]
fn metal_adamw_step_stays_device_resident() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    const LR: f32 = 1e-3;
    const BETAS: (f32, f32) = (0.9, 0.95);
    const EPS: f32 = 1e-8;
    const WD: f32 = 0.01;
    const STEPS: i32 = 5;

    let shape = vec![8, 8];
    let size: usize = shape.iter().product();

    // Same seed for both sides: produce one param_init + STEPS grad vectors,
    // replay them against host AdamW and device AdamW.
    let seed = 1234_u64;
    let param_init: Vec<f32> = (0..size)
        .map(|i| {
            let x = ((seed as usize + i).wrapping_mul(2654435761)) as u32;
            ((x % 1000) as f32 / 1000.0 - 0.5) * 0.2
        })
        .collect();
    let grads_per_step: Vec<Vec<f32>> = (0..STEPS)
        .map(|step| {
            (0..size)
                .map(|i| {
                    let x =
                        ((seed as usize + step as usize * 17 + i).wrapping_mul(1315423911)) as u32;
                    ((x % 1000) as f32 / 1000.0 - 0.5) * 0.1
                })
                .collect()
        })
        .collect();

    // ---- Host reference ----
    let mut host_store = TensorStore::default();
    let host_param = host_store
        .alloc(Tensor::new(param_init.clone(), shape.clone(), true).expect("host param tensor"));
    let host_grad = host_store
        .alloc(Tensor::new(vec![0.0; size], shape.clone(), false).expect("host grad tensor"));
    host_store
        .get_mut(host_param)
        .expect("host param exists")
        .grad = Some(host_grad);
    let mut host_opt = AdamW::new(LR, BETAS, EPS, WD);

    for step in 0..STEPS {
        let grad_tensor = host_store.get_mut(host_grad).expect("host grad exists");
        grad_tensor
            .data
            .copy_from_slice(&grads_per_step[step as usize]);
        host_opt.step(&[host_param], &mut host_store);
    }
    let host_final = host_store.to_host(host_param).expect("host param readback");

    // ---- Metal device-backed ----
    let backend: Arc<MetalBackend> = Arc::new(MetalBackend);
    let mut dev_store = TensorStore::with_backend(backend.clone());
    let dev_param = dev_store
        .alloc(Tensor::new(param_init.clone(), shape.clone(), true).expect("dev param tensor"));
    let dev_grad = dev_store
        .alloc(Tensor::new(vec![0.0; size], shape.clone(), false).expect("dev grad tensor"));
    dev_store.get_mut(dev_param).expect("dev param exists").grad = Some(dev_grad);
    let mut dev_opt = AdamW::new_with_device(LR, BETAS, EPS, WD, backend.clone());

    // Pre-upload the param so the first step already sees it device-resident
    // (the dispatch path will ensure_device if we don't, but being explicit
    // here makes the Dirty::Device invariant measurable from step 1).
    dev_store.ensure_device(dev_param)?;

    let mut per_step_eval_deltas: Vec<u64> = Vec::with_capacity(STEPS as usize);
    for step in 0..STEPS {
        let grad_tensor = dev_store.get_mut(dev_grad).expect("dev grad exists");
        grad_tensor
            .data
            .copy_from_slice(&grads_per_step[step as usize]);

        reset_eval_count();
        dev_opt.step(&[dev_param], &mut dev_store);
        let after = eval_count();
        per_step_eval_deltas.push(after);

        // Post-condition: the param is device-resident, with host copy cleared.
        let param_tensor = dev_store.get(dev_param).expect("dev param exists");
        assert_eq!(
            param_tensor.dirty,
            Dirty::Device,
            "M5.3b.10: after AdamW step {step} the param must remain Dirty::Device \
             so the next forward reuses the lazy MLX handle without re-upload. \
             Saw dirty={:?}",
            param_tensor.dirty
        );
        assert!(
            param_tensor.device_handle.is_some(),
            "M5.3b.10: param must keep its device handle after AdamW step"
        );
    }

    for (step, &delta) in per_step_eval_deltas.iter().enumerate() {
        assert!(
            delta <= 2,
            "M5.3b.10: AdamW step {step} recorded {delta} mlx_evals; budget ≤ 2 \
             (1 optimizer terminal eval for {{param, m, v}} + 1 slack). If this \
             climbs, trace backend_metal::adamw_step for an accidental intra-graph \
             eval, or check whether a newly-introduced upload-path inside step_device \
             forced an eval."
        );
    }

    // Readback the Metal param once at the end to compare; readback goes
    // through the backend (MLX evaluates the final handle), which is the
    // expected terminal flush the NEXT step would have triggered anyway.
    let dev_final = dev_store.to_host(dev_param).expect("dev param readback");

    assert_eq!(host_final.len(), dev_final.len());
    let mut sq_err = 0.0_f32;
    for (h, d) in host_final.iter().zip(dev_final.iter()) {
        let diff = h - d;
        sq_err += diff * diff;
    }
    let l2_diff = sq_err.sqrt();
    assert!(
        l2_diff <= 1e-5,
        "M5.3b.10 parity: host vs Metal AdamW diverged after {STEPS} steps; \
         L2 diff = {l2_diff} (gate 1e-5)"
    );

    Ok(())
}

/// M5.3b.11: per-step eval count must stay at ~1 regardless of parameter
/// count. Before this milestone, `backend_metal::adamw_step` issued its own
/// terminal `mlx_eval([new_param, new_m, new_v])` inside every call, so the
/// total per step scaled with `num_params` — Qwen3.5-class models (~200
/// trainable params) paid ~200 evals/step from AdamW alone, swamping every
/// other optimization on this device path.
///
/// The fix hoists that eval to `AdamW::step_device`'s post-loop, firing one
/// `backend.eval(&handles)` over every param's `(new_param, new_m, new_v)`
/// at the end of the step. Independent per-param MLX chains share no
/// sub-node, so batching them into one eval is semantically safe.
///
/// The assertion below compares N=1 vs N=8 params and fails if the 8-param
/// case charges more than a small additive slack over the 1-param case. If
/// this test starts failing, grep `backend_metal::adamw_step` for an
/// accidental `mlx_eval` reintroduction, or verify that
/// `AdamW::step_device`'s terminal `backend.eval(&refs)` is still the only
/// eval-point in the optimizer step.
#[cfg(feature = "metal")]
#[test]
fn metal_adamw_step_batches_eval_across_params() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    const LR: f32 = 1e-3;
    const BETAS: (f32, f32) = (0.9, 0.95);
    const EPS: f32 = 1e-8;
    const WD: f32 = 0.01;

    let shape = vec![8, 8];
    let size: usize = shape.iter().product();

    // Measure eval count for N params. Setup is identical to the
    // single-param test, just parameterized.
    let measure = |n_params: usize| -> Result<u64> {
        let backend: Arc<MetalBackend> = Arc::new(MetalBackend);
        let mut store = TensorStore::with_backend(backend.clone());
        let mut param_ids = Vec::with_capacity(n_params);
        for p in 0..n_params {
            let init: Vec<f32> = (0..size)
                .map(|i| ((p * 97 + i) as f32 * 0.001) - 0.05)
                .collect();
            let param_id =
                store.alloc(Tensor::new(init, shape.clone(), true).expect("param tensor"));
            let grad_id = store
                .alloc(Tensor::new(vec![0.0; size], shape.clone(), false).expect("grad tensor"));
            store.get_mut(param_id).expect("param exists").grad = Some(grad_id);
            let grad_tensor = store.get_mut(grad_id).expect("grad exists");
            for (i, value) in grad_tensor.data.iter_mut().enumerate() {
                *value = ((p * 13 + i) as f32 * 0.0003) - 0.015;
            }
            store.ensure_device(param_id)?;
            param_ids.push(param_id);
        }

        let mut opt = AdamW::new_with_device(LR, BETAS, EPS, WD, backend.clone());

        reset_eval_count();
        opt.step(&param_ids, &mut store);
        let step_evals = eval_count();

        // Sanity: every param remains Dirty::Device with a live handle.
        for &pid in &param_ids {
            let t = store.get(pid).expect("param still present");
            assert_eq!(
                t.dirty,
                Dirty::Device,
                "M5.3b.11: param {pid} drifted off-device after AdamW step; \
                 saw dirty={:?}",
                t.dirty
            );
            assert!(
                t.device_handle.is_some(),
                "M5.3b.11: param {pid} lost its device handle after AdamW step"
            );
        }

        Ok(step_evals)
    };

    let one = measure(1)?;
    let eight = measure(8)?;

    // Budget: 1-param case should be ~1 eval (terminal batched eval).
    // 8-param case should be the SAME eval budget — the batched terminal
    // eval composes all 8 params' chains into a single `mlx_eval`.
    assert!(
        one <= 2,
        "M5.3b.11: 1-param AdamW step reported {one} evals; expected ≤ 2. \
         The terminal `backend.eval` should fire exactly once."
    );
    assert!(
        eight <= one + 1,
        "M5.3b.11: 8-param AdamW step reported {eight} evals vs {one} for 1 \
         param. Budget is `one + 1` (tiny slack for platform-level jitter); \
         linear scaling here means the per-op terminal `mlx_eval` crept \
         back into `backend_metal::adamw_step` and is being re-charged per \
         param. Grep for `bump_eval_count` / `mlx_eval` inside adamw_step."
    );

    Ok(())
}

/// M5.3b.12 acceptance: lazy `reshape` on Metal composes `mlx_reshape` into
/// the lazy graph without triggering an eval. The fixture mirrors the
/// Qwen3.5 q/k/v prep shape shuffle: `x @ w → reshape → sum`. The assertion
/// is structural — after-matmul and after-reshape eval counts must both be
/// zero, and backward (which does force the tape batch flush) must stay
/// inside the existing ≤3 budget. If `reshape` regresses back into eager
/// mode (e.g. a caller restores `ensure_host` at the public entry), this
/// test fails on `after_reshape != 0`.
///
/// Parity: metal output is compared against `CpuBackend::matmul_forward`
/// + an identity reshape (reshape is a no-op on contiguous row-major
///   data, so the CPU reference is just the matmul result interpreted at
///   the new shape — same bytes).
#[cfg(feature = "metal")]
#[test]
fn metal_reshape_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let m = 6usize;
    let k = 4usize;
    let n = 8usize;
    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.013 - 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.021).sin()).collect();

    let a = store.from_slice(&a_data, &[m, k])?;
    let b = store.from_slice(&b_data, &[k, n])?;
    store.get_mut(b).expect("b exists").requires_grad = true;

    reset_eval_count();

    // matmul → Dirty::Device lazy output [m, n]
    let logits = matmul(a, b, &mut store, &mut tape)?;
    let after_matmul = eval_count();

    // reshape [m, n] → [m * n] : metadata-only on MLX side, must stay lazy.
    let flat = reshape(logits, &[m * n], &mut store, &mut tape)?;
    let after_reshape = eval_count();

    let loss = sum(flat, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_reshape, 0,
        "M5.3b.12: lazy reshape forward must not force an eval; saw \
         {after_reshape}. Grep for a reintroduced `ensure_host` at \
         ops.rs::reshape or a missing Dirty::Device dispatch in layout.rs."
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}"
    );

    // Parity against CPU reference. Reshape is a pure view, so the bytes
    // are identical to the matmul output.
    let metal_out = store.get(flat).expect("flat output exists").data.clone();
    let cpu = CpuBackend;
    let (cpu_out, _shape) = cpu.matmul_forward(&a_data, &[m, k], &b_data, &[k, n])?;
    assert_eq!(metal_out.len(), cpu_out.len());
    for (a, b) in metal_out.iter().zip(cpu_out.iter()) {
        assert!((a - b).abs() <= 1e-5, "reshape parity: metal={a} cpu={b}");
    }

    Ok(())
}

/// M5.3b.12 acceptance: lazy `transpose` on Metal composes
/// `mlx_transpose_axes` (a lazy view, fused into downstream GEMMs by MLX).
/// Fixture mirrors a Qwen3.5 attention-prep shuffle: rank-3 input transpose
/// of axes 1↔2, feeding into a `sum`. Must not force an eval in forward.
#[cfg(feature = "metal")]
#[test]
fn metal_transpose_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    // Shape mirrors [batch, seq, hidden] → [batch, hidden, seq] shuffle.
    let batch = 2usize;
    let seq = 4usize;
    let hidden = 6usize;
    let m = batch * seq;
    let k = 3usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.011 - 0.15).collect();
    let b_data: Vec<f32> = (0..k * hidden)
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();

    let a = store.from_slice(&a_data, &[m, k])?;
    let b = store.from_slice(&b_data, &[k, hidden])?;
    store.get_mut(b).expect("b exists").requires_grad = true;

    reset_eval_count();

    // matmul → [m, hidden] = [batch*seq, hidden], then reshape to rank-3.
    let logits = matmul(a, b, &mut store, &mut tape)?;
    let after_matmul = eval_count();

    let rank3 = reshape(logits, &[batch, seq, hidden], &mut store, &mut tape)?;
    let after_reshape = eval_count();

    // transpose axes 1↔2: [batch, seq, hidden] → [batch, hidden, seq].
    let transposed = transpose(rank3, 1, 2, &mut store, &mut tape)?;
    let after_transpose = eval_count();

    let loss = sum(transposed, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_reshape, 0,
        "M5.3b.12: lazy reshape forward must not force an eval; saw {after_reshape}"
    );
    assert_eq!(
        after_transpose, 0,
        "M5.3b.12: lazy transpose forward must not force an eval; saw \
         {after_transpose}. Grep for a reintroduced `ensure_host` at \
         ops.rs::transpose or a missing Dirty::Device dispatch in layout.rs."
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}"
    );

    // Parity against CPU: matmul → reshape(no-op view) → transpose(1↔2).
    let metal_out = store
        .get(transposed)
        .expect("transposed output exists")
        .data
        .clone();
    let cpu = CpuBackend;
    let (cpu_mm, _shape) = cpu.matmul_forward(&a_data, &[m, k], &b_data, &[k, hidden])?;
    // Reshape is a contiguous no-op, so cpu_mm already has the rank-3 layout.
    // Now apply transpose(1↔2) via `cpu_transpose_swap` (exposed by
    // `autograd::backend`) for a clean reference.
    use autograd::backend::cpu_transpose_swap;
    let (cpu_out, _new_shape) = cpu_transpose_swap(&cpu_mm, &[batch, seq, hidden], 1, 2)?;
    assert_eq!(metal_out.len(), cpu_out.len());
    for (a, b) in metal_out.iter().zip(cpu_out.iter()) {
        assert!((a - b).abs() <= 1e-5, "transpose parity: metal={a} cpu={b}");
    }

    Ok(())
}

/// M5.3b.13 acceptance: lazy `mul_scalar` on Metal composes
/// `mlx_multiply(x, scalar_arr)` (broadcast rank-0 scalar) into the MLX
/// graph. Fixture mirrors Qwen3.5 attention q-scaling: matmul output
/// scaled by `1/sqrt(d_head)`, feeding into a `sum`. Must not force an
/// eval in forward.
#[cfg(feature = "metal")]
#[test]
fn metal_mul_scalar_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let m = 5usize;
    let k = 4usize;
    let n = 7usize;
    let d_head = 8.0_f32;
    let scale = 1.0 / d_head.sqrt();

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.017 - 0.25).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.023).sin()).collect();

    let a = store.from_slice(&a_data, &[m, k])?;
    let b = store.from_slice(&b_data, &[k, n])?;
    store.get_mut(b).expect("b exists").requires_grad = true;

    reset_eval_count();

    let logits = matmul(a, b, &mut store, &mut tape)?;
    let after_matmul = eval_count();

    let scaled = mul_scalar(logits, scale, &mut store, &mut tape)?;
    let after_mul_scalar = eval_count();

    let loss = sum(scaled, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_mul_scalar, 0,
        "M5.3b.13: lazy mul_scalar forward must not force an eval; saw \
         {after_mul_scalar}. Grep for a reintroduced `ensure_host` at \
         ops.rs::mul_scalar or a missing Dirty::Device dispatch in \
         elementwise.rs."
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 3,
        "forward+backward eval count bounded; saw {after_backward}"
    );

    // Parity against CPU reference.
    let metal_out = store
        .get(scaled)
        .expect("scaled output exists")
        .data
        .clone();
    let cpu = CpuBackend;
    let (cpu_mm, _shape) = cpu.matmul_forward(&a_data, &[m, k], &b_data, &[k, n])?;
    let cpu_out: Vec<f32> = cpu_mm.iter().map(|v| v * scale).collect();
    assert_eq!(metal_out.len(), cpu_out.len());
    for (a, b) in metal_out.iter().zip(cpu_out.iter()) {
        assert!(
            (a - b).abs() <= 1e-5,
            "mul_scalar parity: metal={a} cpu={b}"
        );
    }

    Ok(())
}

/// M5.3b.14 acceptance: lazy `add_broadcast` on Metal when both operands
/// are device-resident. Fixture mirrors Qwen3.5 attention's causal-mask
/// add: `scaled [merged_heads, seq, seq] + mask [1, seq, seq]` where
/// `scaled` is the Dirty::Device result of a prior matmul and `mask` is
/// a host tensor that gets `ensure_device`-ed inside `add_broadcast`.
/// Must not force an eval in forward.
#[cfg(feature = "metal")]
#[test]
fn metal_add_broadcast_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    // Shape mirrors attention scores → mask-broadcast pre-softmax.
    let merged_heads = 2usize;
    let seq = 4usize;
    let k = 5usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    // Build a matmul output that naturally has shape [merged_heads*seq, seq].
    let q_data: Vec<f32> = (0..merged_heads * seq * k)
        .map(|i| (i as f32) * 0.013 - 0.2)
        .collect();
    let k_data: Vec<f32> = (0..k * seq).map(|i| ((i as f32) * 0.021).sin()).collect();

    let q = store.from_slice(&q_data, &[merged_heads * seq, k])?;
    let k_t = store.from_slice(&k_data, &[k, seq])?;
    store.get_mut(k_t).expect("k exists").requires_grad = true;

    // Causal mask [1, seq, seq] (upper triangle = -inf).
    let mut mask_data = vec![0.0_f32; seq * seq];
    for row in 0..seq {
        for col in (row + 1)..seq {
            mask_data[row * seq + col] = f32::NEG_INFINITY;
        }
    }
    let mask = store.from_slice(&mask_data, &[1, seq, seq])?;

    reset_eval_count();

    let scores_flat = matmul(q, k_t, &mut store, &mut tape)?;
    let after_matmul = eval_count();

    // Reshape to rank-3 [merged_heads, seq, seq] so the mask broadcasts
    // along axis 0 (right-aligned against [1, seq, seq]).
    let scores = reshape(
        scores_flat,
        &[merged_heads, seq, seq],
        &mut store,
        &mut tape,
    )?;
    let after_reshape = eval_count();

    let masked = add_broadcast(scores, mask, &mut store, &mut tape)?;
    let after_add_broadcast = eval_count();

    let loss = sum(masked, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_reshape, 0,
        "M5.3b.12: lazy reshape forward must not force an eval; saw {after_reshape}"
    );
    assert_eq!(
        after_add_broadcast, 0,
        "M5.3b.14: lazy add_broadcast forward must not force an eval; \
         saw {after_add_broadcast}. Grep for a reintroduced `ensure_host` \
         at ops.rs::add_broadcast or a missing device-lazy dispatch in \
         broadcast.rs."
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 4,
        "forward+backward eval count bounded; saw {after_backward}"
    );

    // Parity against CPU reference (sum-of-masked-scores is a single f32
    // — comparing the masked tensor element-wise would disagree on -inf
    // entries between metal/cpu representations, so we compare the
    // post-sum scalar which is finite on both paths if any row has
    // non-masked entries).
    let metal_masked = store.get(masked).expect("masked output").data.clone();
    let cpu = CpuBackend;
    let (cpu_scores_flat, _) =
        cpu.matmul_forward(&q_data, &[merged_heads * seq, k], &k_data, &[k, seq])?;
    // Apply mask (broadcast [1, seq, seq] across merged_heads).
    let mut cpu_masked = vec![0.0_f32; merged_heads * seq * seq];
    for h in 0..merged_heads {
        for r in 0..seq {
            for c in 0..seq {
                let s = cpu_scores_flat[(h * seq + r) * seq + c];
                let m = mask_data[r * seq + c];
                cpu_masked[(h * seq + r) * seq + c] = s + m;
            }
        }
    }
    assert_eq!(metal_masked.len(), cpu_masked.len());
    for (a, b) in metal_masked.iter().zip(cpu_masked.iter()) {
        if b.is_finite() {
            assert!(
                (a - b).abs() <= 1e-5,
                "add_broadcast parity: metal={a} cpu={b}"
            );
        } else {
            assert!(
                a.is_infinite() && a.is_sign_negative(),
                "add_broadcast parity: expected -inf, got metal={a}"
            );
        }
    }

    Ok(())
}

/// M5.3b.16: `slice` must stay lazy on Metal. Hot-path use is Qwen3.5's
/// fused `q_full` projection being split into q + gate per attention layer.
/// Must not force an eval in forward.
#[cfg(feature = "metal")]
#[test]
fn metal_slice_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    // Shape mirrors Qwen3.5 q_full projection after reshape:
    // [batch, seq, heads, head_dim*2] → split into q [..., :head_dim] +
    // gate [..., head_dim:]
    let batch = 1usize;
    let seq = 4usize;
    let heads = 2usize;
    let head_dim = 8usize;
    let k = 5usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    // Feed slice off a matmul so its input is Dirty::Device — this is the
    // realistic pattern; previously the `ensure_host` at ops.rs::slice
    // would flush the matmul's lazy graph here.
    let a_data: Vec<f32> = (0..batch * seq * heads * k)
        .map(|i| (i as f32) * 0.019 - 0.1)
        .collect();
    let w_data: Vec<f32> = (0..k * (head_dim * 2))
        .map(|i| ((i as f32) * 0.023).cos())
        .collect();
    let a = store.from_slice(&a_data, &[batch * seq * heads, k])?;
    let w = store.from_slice(&w_data, &[k, head_dim * 2])?;
    store.get_mut(a).expect("a exists").requires_grad = true;

    reset_eval_count();

    let q_full_flat = matmul(a, w, &mut store, &mut tape)?;
    let after_matmul = eval_count();
    let q_full = reshape(
        q_full_flat,
        &[batch, seq, heads, head_dim * 2],
        &mut store,
        &mut tape,
    )?;
    let after_reshape = eval_count();
    let q = slice(
        q_full,
        &[0, 0, 0, 0],
        &[batch, seq, heads, head_dim],
        &mut store,
        &mut tape,
    )?;
    let after_slice_q = eval_count();
    let gate = slice(
        q_full,
        &[0, 0, 0, head_dim],
        &[batch, seq, heads, head_dim * 2],
        &mut store,
        &mut tape,
    )?;
    let after_slice_gate = eval_count();
    let loss_q = sum(q, &mut store, &mut tape)?;
    let loss_gate = sum(gate, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _g1 = tape.backward(loss_q, &mut store)?;
    let _g2 = tape.backward(loss_gate, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_matmul, 0,
        "matmul must stay lazy (M5.3a); saw {after_matmul}"
    );
    assert_eq!(
        after_reshape, 0,
        "M5.3b.12: reshape must stay lazy; saw {after_reshape}"
    );
    assert_eq!(
        after_slice_q, 0,
        "M5.3b.16: lazy slice (q window) must not force an eval; saw \
         {after_slice_q}. Grep for a reintroduced `ensure_host` at \
         ops.rs::slice or a missing device-lazy dispatch in \
         layout.rs::slice."
    );
    assert_eq!(
        after_slice_gate, 0,
        "M5.3b.16: lazy slice (gate window) must not force an eval; saw \
         {after_slice_gate}"
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 6,
        "forward+backward eval count bounded; saw {after_backward}"
    );

    // Parity: compare each sliced window element-wise against a CPU slice.
    let metal_q = store.get(q).expect("q output").data.clone();
    let metal_gate = store.get(gate).expect("gate output").data.clone();

    let cpu = CpuBackend;
    let (cpu_q_full_flat, _) = cpu.matmul_forward(
        &a_data,
        &[batch * seq * heads, k],
        &w_data,
        &[k, head_dim * 2],
    )?;
    // cpu_q_full_flat is [batch*seq*heads, head_dim*2]; reshape is a no-op.
    let row_count = batch * seq * heads;
    let mut cpu_q = vec![0.0_f32; row_count * head_dim];
    let mut cpu_gate = vec![0.0_f32; row_count * head_dim];
    for row in 0..row_count {
        for col in 0..head_dim {
            cpu_q[row * head_dim + col] = cpu_q_full_flat[row * (head_dim * 2) + col];
            cpu_gate[row * head_dim + col] = cpu_q_full_flat[row * (head_dim * 2) + head_dim + col];
        }
    }
    assert_eq!(metal_q.len(), cpu_q.len());
    assert_eq!(metal_gate.len(), cpu_gate.len());
    for (a, b) in metal_q.iter().zip(cpu_q.iter()) {
        assert!((a - b).abs() <= 1e-4, "slice parity (q): metal={a} cpu={b}");
    }
    for (a, b) in metal_gate.iter().zip(cpu_gate.iter()) {
        assert!(
            (a - b).abs() <= 1e-4,
            "slice parity (gate): metal={a} cpu={b}"
        );
    }

    Ok(())
}

/// M5.3b.15: composite `causal_sdpa` must stay lazy end-to-end on Metal.
/// The body decomposes into reshape/transpose/matmul/mul_scalar/
/// add_broadcast/softmax — all individually lazy post M5.3b.1–14. Stripping
/// `ensure_host` at the public entry should let the full attention chain
/// traverse the MLX graph without forcing an eval in forward.
#[cfg(feature = "metal")]
#[test]
fn metal_causal_sdpa_forward_stays_lazy() -> Result<()> {
    use autograd::backend_metal::{MetalBackend, eval_count, reset_eval_count};

    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    // Shape mirrors one attention layer of a small Qwen3.5-like model.
    let batch = 1usize;
    let heads = 2usize;
    let seq = 4usize;
    let head_dim = 8usize;

    let mut store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut tape = Tape::new();

    let q_data: Vec<f32> = (0..batch * heads * seq * head_dim)
        .map(|i| (i as f32) * 0.017 - 0.3)
        .collect();
    let k_data: Vec<f32> = (0..batch * heads * seq * head_dim)
        .map(|i| ((i as f32) * 0.029).cos())
        .collect();
    let v_data: Vec<f32> = (0..batch * heads * seq * head_dim)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();

    let q = store.from_slice(&q_data, &[batch, heads, seq, head_dim])?;
    let k = store.from_slice(&k_data, &[batch, heads, seq, head_dim])?;
    let v = store.from_slice(&v_data, &[batch, heads, seq, head_dim])?;
    store.get_mut(q).expect("q exists").requires_grad = true;

    reset_eval_count();

    let context = causal_sdpa(q, k, v, &mut store, &mut tape)?;
    let after_sdpa = eval_count();

    let loss = sum(context, &mut store, &mut tape)?;
    let after_sum = eval_count();

    let _grads = tape.backward(loss, &mut store)?;
    let after_backward = eval_count();

    assert_eq!(
        after_sdpa, 0,
        "M5.3b.15: lazy causal_sdpa forward must not force an eval; \
         saw {after_sdpa}. Grep for a reintroduced `ensure_host` at \
         ops.rs::causal_sdpa or a regressed inner op in \
         reshape/transpose/matmul/mul_scalar/add_broadcast/softmax."
    );
    assert_eq!(
        after_sum, 0,
        "M5.3b.1: lazy sum must not force a host readback; saw {after_sum}"
    );
    assert!(
        after_backward <= 6,
        "forward+backward eval count bounded; saw {after_backward}"
    );

    Ok(())
}
