//! Head-group chunked linear-attention core must match the single-call core
//! exactly, forward and gradients. The fused-qkv region slicing (q|k|v offsets,
//! conv-weight group surgery) is new branch logic; a wrong offset silently
//! attends the wrong head and only shows up here.

use autograd::{
    CpuBackend, Tape, Tensor, TensorId, TensorStore,
    ops::{
        LinearAttentionParams, linear_attention_core, linear_attention_core_head_chunked, mul, sum,
    },
};
use std::sync::Arc;

fn params() -> LinearAttentionParams {
    LinearAttentionParams {
        batch: 1,
        seq_len: 6,
        num_key_heads: 4,
        num_value_heads: 8,
        key_dim: 4,
        value_dim: 4,
        conv_kernel: 2,
        eps: 1.0e-5,
    }
}

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 0.5
        })
        .collect()
}

#[allow(clippy::type_complexity)]
fn run(chunked: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
    let p = params();
    let mut st = TensorStore::with_backend(Arc::new(CpuBackend));
    let mut tp = Tape::new();
    tp.set_enabled(true);
    let (b, s) = (p.batch, p.seq_len);
    let (q_dim, v_dim) = (p.num_key_heads * p.key_dim, p.num_value_heads * p.value_dim);
    let mk = |shape: Vec<usize>, seed: u64, st: &mut TensorStore| -> TensorId {
        let n: usize = shape.iter().product();
        st.alloc(Tensor::new(seeded(n, seed), shape, true).unwrap())
    };
    let qkv = mk(vec![b, s, 2 * q_dim + v_dim], 1, &mut st);
    let z = mk(vec![b, s, v_dim], 2, &mut st);
    let b_proj = mk(vec![b, s, p.num_value_heads], 3, &mut st);
    let a_proj = mk(vec![b, s, p.num_value_heads], 4, &mut st);
    let conv = mk(vec![2 * q_dim + v_dim, p.conv_kernel], 5, &mut st);
    let dt = mk(vec![p.num_value_heads], 6, &mut st);
    let a_log = mk(vec![p.num_value_heads], 7, &mut st);
    let norm = mk(vec![p.value_dim], 8, &mut st);

    let out = if chunked {
        linear_attention_core_head_chunked(
            qkv, z, b_proj, a_proj, conv, dt, a_log, norm, p, &mut st, &mut tp,
        )
        .unwrap()
    } else {
        linear_attention_core(
            qkv, z, b_proj, a_proj, conv, dt, a_log, norm, p, &mut st, &mut tp,
        )
        .unwrap()
    };
    let w = {
        let n = b * s * v_dim;
        st.alloc(Tensor::new(seeded(n, 9), vec![b, s, v_dim], false).unwrap())
    };
    let weighted = mul(out, w, &mut st, &mut tp).unwrap();
    let loss = sum(weighted, &mut st, &mut tp).unwrap();
    let grads = tp.backward_collect(loss, &mut st).unwrap();
    let out_host = st.to_host(out).unwrap();
    let inputs = [qkv, z, b_proj, a_proj, conv, dt, a_log, norm];
    let grad_host = inputs
        .iter()
        .map(|id| {
            grads
                .get(id)
                .map(|&g| st.to_host(g).unwrap())
                .unwrap_or_default()
        })
        .collect();
    (out_host, grad_host)
}

#[test]
fn head_chunked_matches_single_call() {
    let (out_ref, grads_ref) = run(false);
    let (out, grads) = run(true);
    for (a, b) in out.iter().zip(&out_ref) {
        assert!((a - b).abs() < 1e-5, "forward mismatch");
    }
    for (gi, (g, gr)) in grads.iter().zip(&grads_ref).enumerate() {
        assert_eq!(g.len(), gr.len(), "grad {gi} presence mismatch");
        for (a, b) in g.iter().zip(gr) {
            assert!((a - b).abs() < 1e-4, "grad {gi} mismatch: {a} vs {b}");
        }
    }
}
