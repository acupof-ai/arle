mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{add, mul_scalar, sum},
};
use helpers::num_grad;

#[test]
fn toy_backward_produces_exact_threes() -> Result<()> {
    let shape = [2, 3];
    let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &shape)?;
    let b = store.from_slice(&b_data, &shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;
    store.get_mut(b).expect("b exists").requires_grad = true;

    let added = add(a, b, &mut store, &mut tape)?;
    let scaled = mul_scalar(added, 3.0, &mut store, &mut tape)?;
    let loss = sum(scaled, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let expected = vec![3.0; 6];
    let a_grad = store.to_host(*grads.get(&a).expect("a grad"))?;
    let b_grad = store.to_host(*grads.get(&b).expect("b grad"))?;

    assert_eq!(a_grad, expected);
    assert_eq!(b_grad, vec![3.0; 6]);

    let mut probe = [0.0_f32];
    let probe_grad = num_grad(|x| x[0], &mut probe, 1e-3);
    assert!((probe_grad[0] - 1.0).abs() < 1e-3);

    Ok(())
}
