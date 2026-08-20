//! `permute_seq_blocks` against the slice+cat form it replaces.
//!
//! The zigzag CP reorder used to be `perm.len()` slices plus a cat, which cost a
//! tensor per block in the forward and a full-input zero buffer per block in the
//! backward. Forward values and the gradient must be unchanged.

use autograd::{
    CpuBackend, Tape, Tensor, TensorStore,
    ops::{cat, layout::permute_seq_blocks, mul, slice, sum},
};
use std::sync::Arc;

const BATCH: usize = 1;
const BLOCKS: usize = 4;
const ROWS: usize = 3;
const DIM: usize = 5;

fn store() -> TensorStore {
    TensorStore::with_backend(Arc::new(CpuBackend))
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.25 - 3.0).collect()
}

/// The form being replaced, kept here as the reference.
fn slice_cat_reorder(
    x: autograd::TensorId,
    perm: &[usize],
    shape: &[usize],
    st: &mut TensorStore,
    tp: &mut Tape,
) -> autograd::TensorId {
    let block = shape[1] / perm.len();
    let blocks: Vec<_> = perm
        .iter()
        .map(|&src| {
            slice(
                x,
                &[0, src * block, 0],
                &[shape[0], (src + 1) * block, shape[2]],
                st,
                tp,
            )
            .unwrap()
        })
        .collect();
    cat(&blocks, 1, st, tp).unwrap()
}

#[test]
fn matches_the_slice_cat_form() {
    let perm = [2usize, 0, 3, 1];
    let shape = vec![BATCH, BLOCKS * ROWS, DIM];
    let data = ramp(BATCH * BLOCKS * ROWS * DIM);

    let mut st = store();
    let mut tp = Tape::new();
    tp.set_enabled(false);

    let a = st.alloc(Tensor::new(data.clone(), shape.clone(), false).unwrap());
    let want = slice_cat_reorder(a, &perm, &shape, &mut st, &mut tp);
    let want = st.to_host(want).unwrap();

    let b = st.alloc(Tensor::new(data, shape, false).unwrap());
    let got = permute_seq_blocks(b, &perm, &mut st, &mut tp).unwrap();
    let got = st.to_host(got).unwrap();

    assert_eq!(got, want);
}

/// Backward is the inverse permutation: a gradient stamped with its slot index
/// must land back on the block that produced it.
#[test]
fn backward_is_the_inverse_permutation() {
    let perm = [2usize, 0, 3, 1];
    let shape = vec![BATCH, BLOCKS * ROWS, DIM];

    let mut st = store();
    let mut tp = Tape::new();
    tp.set_enabled(true);

    let x = st.alloc(Tensor::new(ramp(BATCH * BLOCKS * ROWS * DIM), shape.clone(), true).unwrap());
    let y = permute_seq_blocks(x, &perm, &mut st, &mut tp).unwrap();

    // loss = sum(permute(x) * w) with w stamped by slot, so dL/dx block `src` is
    // the weight of whatever slot `src` was sent to.
    let mut w = vec![0.0f32; BATCH * BLOCKS * ROWS * DIM];
    for (slot, chunk) in w.chunks_mut(ROWS * DIM).enumerate() {
        chunk.fill(slot as f32);
    }
    let w_id = st.alloc(Tensor::new(w, shape, false).unwrap());
    let weighted = mul(y, w_id, &mut st, &mut tp).unwrap();
    let loss = sum(weighted, &mut st, &mut tp).unwrap();
    let grads = tp.backward(loss, &mut st).unwrap();
    let gx = st.to_host(*grads.get(&x).expect("grad for x")).unwrap();

    for (src, chunk) in gx.chunks(ROWS * DIM).enumerate() {
        let slot = perm.iter().position(|&p| p == src).unwrap() as f32;
        assert!(
            chunk.iter().all(|&v| v == slot),
            "source block {src} should carry slot {slot}, got {:?}",
            &chunk[..3]
        );
    }
}

#[test]
fn a_sequence_that_is_not_whole_blocks_is_refused() {
    let mut st = store();
    let mut tp = Tape::new();
    tp.set_enabled(false);
    let x = st.alloc(Tensor::new(ramp(7 * DIM), vec![1, 7, DIM], false).unwrap());
    assert!(permute_seq_blocks(x, &[0, 1, 2, 3], &mut st, &mut tp).is_err());
}
