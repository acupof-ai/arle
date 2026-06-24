mod helpers;

use autograd::{
    Result, Tape, Tensor, TensorStore,
    ops::{cat_heads, reshape, slice, sum},
};
use helpers::{max_abs_err, num_grad};

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
