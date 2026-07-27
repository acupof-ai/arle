mod helpers;

use autograd::{
    Result, Tape, Tensor, TensorStore,
    ops::{broadcast_expand, cat_heads, reshape, slice, sum},
};
use helpers::{max_abs_err, num_grad};

#[test]
fn broadcast_expand_grad_matches_central_difference() -> Result<()> {
    // [2,1,3] -> [2,4,3]: gradient must sum-reduce over the expanded axis.
    let src_shape = [2usize, 1, 3];
    let tgt_shape = [2usize, 4, 3];
    let x_data = vec![0.2, -0.1, 0.3, -0.4, 0.7, -0.2];

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &src_shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let y = broadcast_expand(x, &tgt_shape, &mut store, &mut tape)?;
    assert_eq!(store.get(y).expect("y").shape, tgt_shape.to_vec());
    // forward is a pure copy: every expanded slice equals src.
    let y_data = store.to_host(y)?;
    for rep in 0..4 {
        for b in 0..2 {
            for d in 0..3 {
                assert_eq!(
                    y_data[(b * 4 + rep) * 3 + d],
                    x_data[b * 3 + d],
                    "expand copy mismatch"
                );
            }
        }
    }

    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let grad_id = grads
        .get(&x)
        .copied()
        .expect("broadcast_expand grad exists");
    let analytic = store.to_host(grad_id)?;
    // d(sum(expand(x)))/dx = n_rep for each element.
    assert_eq!(analytic, vec![4.0; 6]);

    let mut probe = x_data.clone();
    let numeric = num_grad(
        |xd| {
            let mut s = TensorStore::default();
            let mut t = Tape::new();
            let xi = s.from_slice(xd, &src_shape).expect("x");
            let yi = broadcast_expand(xi, &tgt_shape, &mut s, &mut t).expect("expand");
            let li = sum(yi, &mut s, &mut t).expect("sum");
            s.to_host(li).expect("loss")[0]
        },
        &mut probe,
        1e-3,
    );
    assert!(
        max_abs_err(&analytic, &numeric) < 1e-2,
        "analytic {analytic:?} vs numeric {numeric:?}"
    );
    Ok(())
}

#[test]
fn reshape_backward_restores_input_shape() -> Result<()> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let y = reshape(x, &[3, 2], &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let grad_id = grads.get(&x).copied().expect("reshape grad exists");
    let grad = store.get(grad_id).expect("grad tensor exists");
    assert_eq!(grad.shape, vec![2, 3]);
    let grad_data = store.to_host(grad_id)?;
    assert_eq!(grad_data, vec![1.0; 6]);

    Ok(())
}

#[test]
fn slice_backward_scatter_restores_input_shape() -> Result<()> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let y = slice(x, &[0, 1], &[2, 3], &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let grad_id = grads.get(&x).copied().expect("slice grad exists");
    let grad = store.get(grad_id).expect("grad tensor exists");
    assert_eq!(grad.shape, vec![2, 3]);
    let grad_data = store.to_host(grad_id)?;
    assert_eq!(grad_data, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn slice_grad_matches_central_difference() -> Result<()> {
    let shape = [2, 3];
    let x_data = vec![0.2, -0.1, 0.3, -0.4, 0.7, -0.2];

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let y = slice(x, &[0, 1], &[2, 3], &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let y = slice(x, &[0, 1], &[2, 3], &mut store, &mut tape).expect("slice");
            let loss = sum(y, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn cat_heads_forward_concatenates_along_head_axis() -> Result<()> {
    // a: [1,2,3,4], b: [1,1,3,4] -> [1,3,3,4]; manual concat reference.
    let a_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).map(|i| (100 + i) as f32).collect();

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.alloc(Tensor::new(a_data.clone(), vec![1, 2, 3, 4], false)?);
    let b = store.alloc(Tensor::new(b_data.clone(), vec![1, 1, 3, 4], false)?);

    let out = cat_heads(&[a, b], &mut store, &mut tape)?;
    assert_eq!(store.get(out).expect("out exists").shape, vec![1, 3, 3, 4]);

    let mut expected = a_data.clone();
    expected.extend_from_slice(&b_data);
    assert_eq!(store.to_host(out)?, expected);
    Ok(())
}

#[test]
fn cat_heads_backward_routes_ones_to_each_input() -> Result<()> {
    // loss = sum(cat_heads([a, b])); d(loss)/d(a) and d(loss)/d(b) are all-ones.
    let a_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).map(|i| (100 + i) as f32).collect();

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.alloc(Tensor::new(a_data, vec![1, 2, 3, 4], true)?);
    let b = store.alloc(Tensor::new(b_data, vec![1, 1, 3, 4], true)?);

    let out = cat_heads(&[a, b], &mut store, &mut tape)?;
    let loss = sum(out, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let grad_a = grads.get(&a).copied().expect("grad a exists");
    let grad_b = grads.get(&b).copied().expect("grad b exists");
    assert_eq!(
        store.get(grad_a).expect("grad a tensor").shape,
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        store.get(grad_b).expect("grad b tensor").shape,
        vec![1, 1, 3, 4]
    );
    assert_eq!(store.to_host(grad_a)?, vec![1.0; 24]);
    assert_eq!(store.to_host(grad_b)?, vec![1.0; 12]);
    Ok(())
}
