//! The ring core against a full-softmax reference, on the host.
//!
//! `ring_forward_tile` / `ring_backward_tile` are the correctness-load-bearing
//! math the device merge kernels are gated against, and nothing exercised them
//! after the inline unit tests were removed (`a7c9ee395`). Blocks are ragged and
//! the positions are zigzag so the per-row mask is under test too.

use cuda_kernels::ring_attention::{ring_backward_tile, ring_forward_tile};

const ROWS: usize = 7;
const DIM: usize = 4;
const SCALE: f32 = 0.5;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

/// Dense causal-by-position attention over the concatenated blocks.
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_pos: &[usize],
    k_pos: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let cols = k_pos.len();
    let mut out = vec![0.0f32; ROWS * DIM];
    let mut lse = vec![f32::NEG_INFINITY; ROWS];
    for r in 0..ROWS {
        let mut s = vec![f32::NEG_INFINITY; cols];
        for (c, sc) in s.iter_mut().enumerate() {
            if k_pos[c] > q_pos[r] {
                continue;
            }
            let dot: f32 = (0..DIM).map(|d| q[r * DIM + d] * k[c * DIM + d]).sum();
            *sc = dot * SCALE;
        }
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() {
            continue;
        }
        let mut l = 0.0f32;
        for c in 0..cols {
            if !s[c].is_finite() {
                continue;
            }
            let p = (s[c] - m).exp();
            l += p;
            for d in 0..DIM {
                out[r * DIM + d] += p * v[c * DIM + d];
            }
        }
        for d in 0..DIM {
            out[r * DIM + d] /= l;
        }
        lse[r] = m + l.ln();
    }
    (out, lse)
}

/// Ragged blocks + a zigzag position layout: rows own a front run and a back run,
/// keys are split 3/2/4 across the ring, and one block is entirely in the future
/// for the early rows.
fn fixture() -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
) {
    let q = fill(ROWS * DIM, 0x51ed_2701);
    let k = fill(9 * DIM, 0x9e37_79b9);
    let v = fill(9 * DIM, 0x1234_5678);
    let q_pos = vec![0, 1, 2, 13, 14, 15, 16];
    let k_pos = vec![0, 1, 2, 3, 4, 12, 13, 14, 15];
    let splits = vec![3, 2, 4];
    (q, k, v, q_pos, k_pos, splits)
}

fn as_blocks<'a>(
    k: &'a [f32],
    v: &'a [f32],
    k_pos: &'a [usize],
    splits: &[usize],
) -> Vec<(&'a [f32], &'a [f32], &'a [usize])> {
    let mut blocks = Vec::new();
    let mut off = 0;
    for &n in splits {
        blocks.push((
            &k[off * DIM..(off + n) * DIM],
            &v[off * DIM..(off + n) * DIM],
            &k_pos[off..off + n],
        ));
        off += n;
    }
    blocks
}

#[test]
fn ring_forward_matches_full_softmax() {
    let (q, k, v, q_pos, k_pos, splits) = fixture();
    let blocks = as_blocks(&k, &v, &k_pos, &splits);
    let (got_out, got_lse) = ring_forward_tile(&q, &blocks, ROWS, DIM, SCALE, &q_pos);
    let (want_out, want_lse) = reference(&q, &k, &v, &q_pos, &k_pos);
    for i in 0..ROWS * DIM {
        assert!(
            (got_out[i] - want_out[i]).abs() < 1e-5,
            "out[{i}] ring {} vs full-softmax {}",
            got_out[i],
            want_out[i]
        );
    }
    for r in 0..ROWS {
        assert!(
            (got_lse[r] - want_lse[r]).abs() < 1e-5,
            "lse[{r}] ring {} vs full-softmax {}",
            got_lse[r],
            want_lse[r]
        );
    }
}

/// Finite differences on the same fixture — the adjoint, not a self-comparison.
#[test]
fn ring_backward_matches_finite_differences() {
    let (q, k, v, q_pos, k_pos, splits) = fixture();
    let d_out = fill(ROWS * DIM, 0xdead_beef);

    let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
        let (out, _) = reference(q, k, v, &q_pos, &k_pos);
        out.iter().zip(&d_out).map(|(o, g)| o * g).sum()
    };

    let blocks = as_blocks(&k, &v, &k_pos, &splits);
    let (out, lse) = ring_forward_tile(&q, &blocks, ROWS, DIM, SCALE, &q_pos);
    let (grad_q, per_block) =
        ring_backward_tile(&q, &blocks, &out, &lse, &d_out, ROWS, DIM, SCALE, &q_pos);

    let eps = 1e-3f32;
    for i in 0..ROWS * DIM {
        let (mut lo, mut hi) = (q.clone(), q.clone());
        lo[i] -= eps;
        hi[i] += eps;
        let fd = (loss(&hi, &k, &v) - loss(&lo, &k, &v)) / (2.0 * eps);
        assert!(
            (grad_q[i] - fd).abs() < 2e-2,
            "grad_q[{i}] analytic {} vs finite-diff {}",
            grad_q[i],
            fd
        );
    }

    // Blocks return grads in their own column order; stitch them back.
    let mut grad_k = vec![0.0f32; k.len()];
    let mut grad_v = vec![0.0f32; v.len()];
    let mut off = 0;
    for (bi, &n) in splits.iter().enumerate() {
        let (gk, gv) = &per_block[bi];
        grad_k[off * DIM..(off + n) * DIM].copy_from_slice(gk);
        grad_v[off * DIM..(off + n) * DIM].copy_from_slice(gv);
        off += n;
    }
    for i in 0..k.len() {
        let (mut lo, mut hi) = (k.clone(), k.clone());
        lo[i] -= eps;
        hi[i] += eps;
        let fd = (loss(&q, &hi, &v) - loss(&q, &lo, &v)) / (2.0 * eps);
        assert!(
            (grad_k[i] - fd).abs() < 2e-2,
            "grad_k[{i}] analytic {} vs finite-diff {}",
            grad_k[i],
            fd
        );
    }
    for i in 0..v.len() {
        let (mut lo, mut hi) = (v.clone(), v.clone());
        lo[i] -= eps;
        hi[i] += eps;
        let fd = (loss(&q, &k, &hi) - loss(&q, &k, &lo)) / (2.0 * eps);
        assert!(
            (grad_v[i] - fd).abs() < 2e-2,
            "grad_v[{i}] analytic {} vs finite-diff {}",
            grad_v[i],
            fd
        );
    }
}
