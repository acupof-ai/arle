mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{add, mul, mul_scalar, sum},
};
use helpers::num_grad;

fn random_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (u32::MAX >> 8) as f32;
            (unit * 2.0) - 1.0
        })
        .collect()
}

fn max_abs_err(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn add_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let a_data = random_vec(6, 7);
    let b_data = random_vec(6, 11);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &shape)?;
    let b = store.from_slice(&b_data, &shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;
    store.get_mut(b).expect("b exists").requires_grad = true;

    let y = add(a, b, &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;
    let analytic_b = store.to_host(*grads.get(&b).expect("grad for b"))?;

    let mut a_numeric_input = a_data.clone();
    let numeric_a = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(x, &shape).expect("a");
            let b = store.from_slice(&b_data, &shape).expect("b");
            let out = add(a, b, &mut store, &mut tape).expect("add");
            let loss = sum(out, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric_input,
        1e-3,
    );
    let mut b_numeric_input = b_data.clone();
    let numeric_b = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(&a_data, &shape).expect("a");
            let b = store.from_slice(x, &shape).expect("b");
            let out = add(a, b, &mut store, &mut tape).expect("add");
            let loss = sum(out, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut b_numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    assert!(max_abs_err(&analytic_b, &numeric_b) < 1e-3);
    Ok(())
}

#[test]
fn mul_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let a_data = random_vec(6, 19);
    let b_data = random_vec(6, 29);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &shape)?;
    let b = store.from_slice(&b_data, &shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;
    store.get_mut(b).expect("b exists").requires_grad = true;

    let y = mul(a, b, &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;
    let analytic_b = store.to_host(*grads.get(&b).expect("grad for b"))?;

    let mut a_numeric_input = a_data.clone();
    let numeric_a = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(x, &shape).expect("a");
            let b = store.from_slice(&b_data, &shape).expect("b");
            let out = mul(a, b, &mut store, &mut tape).expect("mul");
            let loss = sum(out, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric_input,
        1e-3,
    );
    let mut b_numeric_input = b_data.clone();
    let numeric_b = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(&a_data, &shape).expect("a");
            let b = store.from_slice(x, &shape).expect("b");
            let out = mul(a, b, &mut store, &mut tape).expect("mul");
            let loss = sum(out, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut b_numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    assert!(max_abs_err(&analytic_b, &numeric_b) < 1e-3);
    Ok(())
}

#[test]
fn mul_scalar_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let a_data = random_vec(6, 37);
    let scale = 2.75;

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;

    let y = mul_scalar(a, scale, &mut store, &mut tape)?;
    let loss = sum(y, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;
    let mut a_numeric_input = a_data.clone();
    let numeric_a = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(x, &shape).expect("a");
            let out = mul_scalar(a, scale, &mut store, &mut tape).expect("mul_scalar");
            let loss = sum(out, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    Ok(())
}

#[test]
fn sum_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let a_data = random_vec(6, 43);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;

    let loss = sum(a, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;

    let mut a_numeric_input = a_data.clone();
    let numeric_a = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(x, &shape).expect("a");
            let loss = sum(a, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    Ok(())
}
