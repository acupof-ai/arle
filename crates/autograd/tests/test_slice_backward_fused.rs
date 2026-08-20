//! Disjoint slices of one tensor must sum to exactly one gradient.
//!
//! Each `slice_backward` used to zero-fill a full-input buffer that `merge_grad`
//! then summed; the fused path scatters into the gradient already held for the
//! input. Double-counting or a missed region is invisible in a memory number and
//! shows up only here.

use autograd::{
    CpuBackend, Tape, Tensor, TensorStore,
    ops::{mul, slice, sum},
};
use std::sync::Arc;

const ROWS: usize = 3;
const COLS: usize = 12;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.5 - 2.0).collect()
}

/// `loss = sum(slice_a * 2 + slice_b * 3 + slice_c * 5)` over three disjoint
/// column bands, so d(loss)/dx is 2 / 3 / 5 on each band and nothing overlaps.
#[test]
fn disjoint_slices_accumulate_once_each() {
    let mut st = TensorStore::with_backend(Arc::new(CpuBackend));
    let mut tp = Tape::new();
    tp.set_enabled(true);

    let shape = vec![1, ROWS, COLS];
    let x = st.alloc(Tensor::new(ramp(ROWS * COLS), shape.clone(), true).unwrap());

    let bands = [(0usize, 2usize, 2.0f32), (2, 6, 3.0), (6, 12, 5.0)];
    let mut total: Option<autograd::TensorId> = None;
    for (lo, hi, w) in bands {
        let part = slice(x, &[0, 0, lo], &[1, ROWS, hi], &mut st, &mut tp).unwrap();
        let wid = st
            .alloc(Tensor::new(vec![w; ROWS * (hi - lo)], vec![1, ROWS, hi - lo], false).unwrap());
        let weighted = mul(part, wid, &mut st, &mut tp).unwrap();
        let s = sum(weighted, &mut st, &mut tp).unwrap();
        total = Some(match total {
            None => s,
            Some(acc) => autograd::ops::add(acc, s, &mut st, &mut tp).unwrap(),
        });
    }

    let grads = tp.backward(total.unwrap(), &mut st).unwrap();
    let gx = st.to_host(*grads.get(&x).expect("grad for x")).unwrap();

    assert_eq!(gx.len(), ROWS * COLS);
    for r in 0..ROWS {
        for c in 0..COLS {
            let want = match c {
                0..=1 => 2.0,
                2..=5 => 3.0,
                _ => 5.0,
            };
            let got = gx[r * COLS + c];
            assert!(
                (got - want).abs() < 1e-6,
                "row {r} col {c}: got {got}, want {want} — a doubled band means the \
                 scatter ran twice, a zero band means it was dropped"
            );
        }
    }
}

/// Overlapping slices must still sum. The fused path writes rather than adds, so
/// an overlap is the case where writing in place would silently lose one.
#[test]
fn overlapping_slices_sum_their_gradients() {
    let mut st = TensorStore::with_backend(Arc::new(CpuBackend));
    let mut tp = Tape::new();
    tp.set_enabled(true);

    let shape = vec![1, ROWS, COLS];
    let x = st.alloc(Tensor::new(ramp(ROWS * COLS), shape.clone(), true).unwrap());

    // [0,8) and [4,12) overlap on [4,8).
    let a = slice(x, &[0, 0, 0], &[1, ROWS, 8], &mut st, &mut tp).unwrap();
    let b = slice(x, &[0, 0, 4], &[1, ROWS, 12], &mut st, &mut tp).unwrap();
    let sa = sum(a, &mut st, &mut tp).unwrap();
    let sb = sum(b, &mut st, &mut tp).unwrap();
    let loss = autograd::ops::add(sa, sb, &mut st, &mut tp).unwrap();

    let grads = tp.backward(loss, &mut st).unwrap();
    let gx = st.to_host(*grads.get(&x).expect("grad for x")).unwrap();

    for r in 0..ROWS {
        for c in 0..COLS {
            let want = if (4..8).contains(&c) { 2.0 } else { 1.0 };
            let got = gx[r * COLS + c];
            assert!(
                (got - want).abs() < 1e-6,
                "row {r} col {c}: got {got}, want {want} — the overlap must add, not overwrite"
            );
        }
    }
}
