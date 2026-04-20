use autograd::{
    Result, Tape, TensorStore,
    ops::{slice, sum},
};

#[test]
fn slice_backward_scatters_grad_into_input_shape() -> Result<()> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    )?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let y = slice(x, &[0, 1, 1], &[2, 2, 3], &mut store, &mut tape)?;
    assert_eq!(store.get(y).expect("slice tensor").shape, vec![2, 1, 2]);

    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let grad_id = grads.get(&x).copied().expect("slice grad exists");
    let grad = store.get(grad_id).expect("grad tensor exists");
    assert_eq!(grad.shape, vec![2, 2, 3]);
    assert_eq!(
        grad.data,
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, //
            0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0,
        ]
    );

    Ok(())
}

#[test]
fn slice_rejects_invalid_bounds() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store
        .from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("x alloc");

    let err = slice(x, &[0, 0], &[2, 3], &mut store, &mut tape).expect_err("bounds should fail");
    assert!(matches!(
        err,
        autograd::AutogradError::IndexOutOfBounds { index: 3, upper: 2 }
    ));
}
