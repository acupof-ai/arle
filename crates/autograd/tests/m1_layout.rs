use autograd::{
    Result, Tape, TensorStore,
    ops::{reshape, sum},
};

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
