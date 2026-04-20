mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{reshape, slice, sum},
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
    assert_eq!(grad.data, vec![1.0; 6]);

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
    assert_eq!(grad.data, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

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
