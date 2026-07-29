//! Equivalence gate for `fused_linear_ce_loss_indexed`: the forward-once chunked
//! fused cross-entropy (agent-OPD masked-CE writeback core) must match a
//! reference that materializes the full `[P, vocab]` logits and runs the stock
//! autograd CE (`log_softmax` -> `gather` -> `mean` -> `*-1`), on both the loss
//! value and `d_hidden`, to within 1e-3 on a tiny shape.

use std::collections::HashMap;

use autograd::{
    Result, Tape, TensorId, TensorStore,
    ops::{
        fused_linear_distill::fused_linear_ce_loss_indexed, gather_last_dim, log_softmax,
        matmul_bt, mul_scalar, slice,
    },
};

fn deterministic_vec(len: usize, seed: u64) -> Vec<f32> {
    // Small LCG -> [-1, 1); deterministic, no rand dep.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 1.0) * 0.5
        })
        .collect()
}

/// Reference CE over the masked rows: build full logits = hidden @ lm_headᵀ,
/// slice to the masked rows, then stock `cross_entropy_loss` math. Returns the
/// loss and the full-shape `d_hidden` (zeros at non-masked rows). Runs on
/// whatever backend `store` carries (CPU reference or a GPU backend).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn reference_masked_ce_in(
    mut store: TensorStore,
    hidden_data: &[f32],
    weight_data: &[f32],
    seq: usize,
    hidden_dim: usize,
    vocab: usize,
    positions: &[i32],
    targets: &[i32],
) -> Result<(f32, Vec<f32>)> {
    let mut tape = Tape::new();
    let hidden = store.from_slice(hidden_data, &[seq, hidden_dim])?;
    store.get_mut(hidden).expect("hidden").requires_grad = true;
    let weight = store.from_slice(weight_data, &[vocab, hidden_dim])?;

    // [seq, vocab] = hidden @ weightᵀ, matching the fused convention.
    let logits = matmul_bt(hidden, weight, &mut store, &mut tape)?;

    // Per masked position: slice its [1, vocab] logit row and CE against its
    // hard target; sum, then mean over the masked positions (1/N).
    let n = positions.len();
    let mut acc: Option<TensorId> = None;
    for (&p, &t) in positions.iter().zip(targets.iter()) {
        let row = slice(
            logits,
            &[p as usize, 0],
            &[p as usize + 1, vocab],
            &mut store,
            &mut tape,
        )?;
        let lp = log_softmax(row, &mut store, &mut tape)?;
        let gathered = gather_last_dim(lp, &[t as usize], &mut store, &mut tape)?;
        let neg = mul_scalar(gathered, -1.0, &mut store, &mut tape)?;
        acc = Some(match acc {
            None => neg,
            Some(prev) => autograd::ops::add(prev, neg, &mut store, &mut tape)?,
        });
    }
    let summed = acc.expect("non-empty positions");
    let loss = mul_scalar(summed, 1.0 / n as f32, &mut store, &mut tape)?;
    let loss_value = store.to_host(loss)?[0];

    let grads: HashMap<TensorId, TensorId> = tape.backward(loss, &mut store)?;
    let d_hidden = store.to_host(*grads.get(&hidden).expect("d_hidden"))?;
    Ok((loss_value, d_hidden))
}

/// Fused path: one call, chunked over positions, never materializing [seq, vocab].
/// Runs on whatever backend `store` carries.
#[allow(clippy::too_many_arguments)]
fn fused_masked_ce_in(
    mut store: TensorStore,
    hidden_data: &[f32],
    weight_data: &[f32],
    seq: usize,
    hidden_dim: usize,
    vocab: usize,
    positions: &[i32],
    targets: &[i32],
    chunk_rows: usize,
) -> Result<(f32, Vec<f32>)> {
    let mut tape = Tape::new();
    let hidden = store.from_slice(hidden_data, &[1, seq, hidden_dim])?;
    store.get_mut(hidden).expect("hidden").requires_grad = true;
    let weight = store.from_slice(weight_data, &[vocab, hidden_dim])?;

    let loss = fused_linear_ce_loss_indexed(
        hidden, weight, positions, targets, chunk_rows, None, &mut store, &mut tape,
    )?;
    let loss_value = store.to_host(loss)?[0];
    let grads = tape.backward(loss, &mut store)?;
    let d_hidden = store.to_host(*grads.get(&hidden).expect("d_hidden"))?;
    Ok((loss_value, d_hidden))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// CPU-reference convenience wrappers (default backend).
#[allow(clippy::type_complexity)]
fn reference_masked_ce(
    hidden_data: &[f32],
    weight_data: &[f32],
    seq: usize,
    hidden_dim: usize,
    vocab: usize,
    positions: &[i32],
    targets: &[i32],
) -> Result<(f32, Vec<f32>)> {
    reference_masked_ce_in(
        TensorStore::default(),
        hidden_data,
        weight_data,
        seq,
        hidden_dim,
        vocab,
        positions,
        targets,
    )
}

#[allow(clippy::too_many_arguments)]
fn fused_masked_ce(
    hidden_data: &[f32],
    weight_data: &[f32],
    seq: usize,
    hidden_dim: usize,
    vocab: usize,
    positions: &[i32],
    targets: &[i32],
    chunk_rows: usize,
) -> Result<(f32, Vec<f32>)> {
    fused_masked_ce_in(
        TensorStore::default(),
        hidden_data,
        weight_data,
        seq,
        hidden_dim,
        vocab,
        positions,
        targets,
        chunk_rows,
    )
}

#[test]
fn fused_linear_ce_matches_reference_dense() -> Result<()> {
    let (seq, hidden_dim, vocab) = (8usize, 16usize, 32usize);
    let hidden_data = deterministic_vec(seq * hidden_dim, 7);
    let weight_data = deterministic_vec(vocab * hidden_dim, 11);
    // All positions masked (worst case): predicting positions 0..seq.
    let positions: Vec<i32> = (0..seq as i32).collect();
    let targets: Vec<i32> = (0..seq).map(|i| ((i * 5 + 3) % vocab) as i32).collect();

    let (ref_loss, ref_grad) = reference_masked_ce(
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
    )?;
    // Chunk smaller than P so the chunk loop actually iterates.
    let (fused_loss, fused_grad) = fused_masked_ce(
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
        3,
    )?;

    assert!(
        (ref_loss - fused_loss).abs() < 1e-3,
        "loss mismatch: ref={ref_loss} fused={fused_loss}"
    );
    let grad_err = max_abs(&ref_grad, &fused_grad);
    assert!(grad_err < 1e-3, "d_hidden mismatch: max_abs_err={grad_err}");
    Ok(())
}

#[test]
fn fused_linear_ce_matches_reference_sparse_positions() -> Result<()> {
    let (seq, hidden_dim, vocab) = (8usize, 16usize, 32usize);
    let hidden_data = deterministic_vec(seq * hidden_dim, 23);
    let weight_data = deterministic_vec(vocab * hidden_dim, 29);
    // Non-contiguous masked positions (the real masked-writeback pattern).
    let positions: Vec<i32> = vec![1, 2, 5];
    let targets: Vec<i32> = vec![7, 30, 14];

    let (ref_loss, ref_grad) = reference_masked_ce(
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
    )?;
    let (fused_loss, fused_grad) = fused_masked_ce(
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
        2,
    )?;

    assert!(
        (ref_loss - fused_loss).abs() < 1e-3,
        "loss mismatch: ref={ref_loss} fused={fused_loss}"
    );
    // Non-masked rows must be exactly zero in both; masked rows must match.
    let grad_err = max_abs(&ref_grad, &fused_grad);
    assert!(grad_err < 1e-3, "d_hidden mismatch: max_abs_err={grad_err}");
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU-path gate: the CUDA fused chunked-CE (`Device::Cpu` dispatch is bypassed,
// so the device path under test runs) must match the CPU materialize-then-CE
// reference on BOTH the loss and `d_hidden`, on dense and sparse positions.
// Runs only on a CUDA box (the autograd CUDA backend binds cudarc device 0).
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn cuda_store() -> TensorStore {
    use autograd::backend_cuda::CudaBackend;
    use std::sync::Arc;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    TensorStore::with_backend(Arc::new(backend))
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn fused_linear_ce_gpu_matches_reference_dense() -> Result<()> {
    let (seq, hidden_dim, vocab) = (8usize, 16usize, 32usize);
    let hidden_data = deterministic_vec(seq * hidden_dim, 7);
    let weight_data = deterministic_vec(vocab * hidden_dim, 11);
    let positions: Vec<i32> = (0..seq as i32).collect();
    let targets: Vec<i32> = (0..seq).map(|i| ((i * 5 + 3) % vocab) as i32).collect();

    // Reference on CPU (ground truth); fused path on the GPU store.
    let (ref_loss, ref_grad) = reference_masked_ce(
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
    )?;
    let (fused_loss, fused_grad) = fused_masked_ce_in(
        cuda_store(),
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
        3,
    )?;

    assert!(
        (ref_loss - fused_loss).abs() < 1e-3,
        "GPU loss mismatch: ref={ref_loss} fused={fused_loss}"
    );
    let grad_err = max_abs(&ref_grad, &fused_grad);
    assert!(
        grad_err < 1e-3,
        "GPU d_hidden mismatch: max_abs_err={grad_err}"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn fused_linear_ce_gpu_matches_reference_sparse_positions() -> Result<()> {
    let (seq, hidden_dim, vocab) = (8usize, 16usize, 32usize);
    let hidden_data = deterministic_vec(seq * hidden_dim, 23);
    let weight_data = deterministic_vec(vocab * hidden_dim, 29);
    let positions: Vec<i32> = vec![1, 2, 5];
    let targets: Vec<i32> = vec![7, 30, 14];

    let (ref_loss, ref_grad) = reference_masked_ce(
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
    )?;
    let (fused_loss, fused_grad) = fused_masked_ce_in(
        cuda_store(),
        &hidden_data,
        &weight_data,
        seq,
        hidden_dim,
        vocab,
        &positions,
        &targets,
        2,
    )?;

    assert!(
        (ref_loss - fused_loss).abs() < 1e-3,
        "GPU loss mismatch: ref={ref_loss} fused={fused_loss}"
    );
    let grad_err = max_abs(&ref_grad, &fused_grad);
    assert!(
        grad_err < 1e-3,
        "GPU d_hidden mismatch: max_abs_err={grad_err}"
    );
    Ok(())
}
