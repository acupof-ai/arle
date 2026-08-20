//! `checkpoint_seq_chunked` over a dim-changing block must match the unchunked
//! gradients exactly. The chunk loop slices the input at `dim` and the output
//! grad at `out_dim`; conflating the two (the pre-`out_dim` bug shape) slices
//! the output grad with the input width and is invisible until backward.

use autograd::{
    CpuBackend, Result, Tape, Tensor, TensorId, TensorStore,
    ops::{checkpoint_seq_chunked, matmul_bt, mul, reshape, sum},
};
use std::sync::Arc;

const BATCH: usize = 1;
const SEQ: usize = 10;
const IN_DIM: usize = 4;
const OUT_DIM: usize = 6;

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32).sin() * scale).collect()
}

fn proj(x: TensorId, w: TensorId, st: &mut TensorStore, tp: &mut Tape) -> Result<TensorId> {
    let rows = st.get(x).unwrap().shape[1];
    let flat = reshape(x, &[rows, IN_DIM], st, tp)?;
    let y = matmul_bt(flat, w, st, tp)?;
    reshape(y, &[BATCH, rows, OUT_DIM], st, tp)
}

fn grads(chunk: usize) -> (Vec<f32>, Vec<f32>) {
    let mut st = TensorStore::with_backend(Arc::new(CpuBackend));
    let mut tp = Tape::new();
    tp.set_enabled(true);

    let x = st.alloc(
        Tensor::new(
            ramp(BATCH * SEQ * IN_DIM, 1.0),
            vec![BATCH, SEQ, IN_DIM],
            true,
        )
        .unwrap(),
    );
    let w =
        st.alloc(Tensor::new(ramp(OUT_DIM * IN_DIM, 0.3), vec![OUT_DIM, IN_DIM], true).unwrap());

    let y = if chunk == 0 {
        proj(x, w, &mut st, &mut tp).unwrap()
    } else {
        checkpoint_seq_chunked(
            x,
            vec![w],
            chunk,
            &mut st,
            &mut tp,
            move |s, t, _start, inp| proj(inp[0], inp[1], s, t),
        )
        .unwrap()
    };
    // Position-dependent weighting so a row swapped between chunks changes the grad.
    let wgt = st.alloc(
        Tensor::new(
            ramp(BATCH * SEQ * OUT_DIM, 0.7),
            vec![BATCH, SEQ, OUT_DIM],
            false,
        )
        .unwrap(),
    );
    let weighted = mul(y, wgt, &mut st, &mut tp).unwrap();
    let loss = sum(weighted, &mut st, &mut tp).unwrap();
    let grads = tp.backward_collect(loss, &mut st).unwrap();
    (
        st.to_host(grads[&x]).unwrap(),
        st.to_host(grads[&w]).unwrap(),
    )
}

#[test]
fn chunked_projection_matches_unchunked() {
    let (dx_ref, dw_ref) = grads(0);
    for chunk in [3, 10, 64] {
        let (dx, dw) = grads(chunk);
        for (a, b) in dx.iter().zip(&dx_ref) {
            assert!((a - b).abs() < 1e-5, "d_input mismatch at chunk={chunk}");
        }
        for (a, b) in dw.iter().zip(&dw_ref) {
            assert!((a - b).abs() < 1e-5, "d_weight mismatch at chunk={chunk}");
        }
    }
}
