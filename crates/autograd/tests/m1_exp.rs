mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{exp, sum},
};
use helpers::{max_abs_err, num_grad};

#[test]
fn exp_grad_matches_central_difference() -> Result<()> {
    let shape = [4];
    let x_data = vec![-1.0, 0.0, 0.5, 1.0];

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let y = exp(x, &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let y = exp(x, &mut store, &mut tape).expect("exp");
            let loss = sum(y, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}
