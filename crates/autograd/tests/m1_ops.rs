mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{
        add_broadcast, embedding, gather_last_dim, gelu, log_softmax, matmul, mean, mul, rmsnorm,
        softmax, sum, transpose,
    },
};
use helpers::{max_abs_err, num_grad, random_vec};

#[test]
fn matmul_grad_matches_numeric_for_2d_and_batched() -> Result<()> {
    let a_shape = [2, 3];
    let b_shape = [3, 4];
    let coeff = random_vec(8, 101);
    let a_data = random_vec(6, 7);
    let b_data = random_vec(12, 11);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &a_shape)?;
    let b = store.from_slice(&b_data, &b_shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;
    store.get_mut(b).expect("b exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &[2, 4])?;

    let y = matmul(a, b, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;
    let analytic_b = store.to_host(*grads.get(&b).expect("grad for b"))?;

    let mut a_numeric = a_data.clone();
    let numeric_a = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(x, &a_shape).expect("a");
            let b = store.from_slice(&b_data, &b_shape).expect("b");
            let coeff_id = store.from_slice(&coeff, &[2, 4]).expect("coeff");
            let y = matmul(a, b, &mut store, &mut tape).expect("matmul");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric,
        1e-3,
    );
    let mut b_numeric = b_data.clone();
    let numeric_b = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(&a_data, &a_shape).expect("a");
            let b = store.from_slice(x, &b_shape).expect("b");
            let coeff_id = store.from_slice(&coeff, &[2, 4]).expect("coeff");
            let y = matmul(a, b, &mut store, &mut tape).expect("matmul");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut b_numeric,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    assert!(max_abs_err(&analytic_b, &numeric_b) < 1e-3);

    let a_shape = [2, 2, 3];
    let b_shape = [2, 3, 2];
    let coeff = random_vec(8, 103);
    let a_data = random_vec(12, 13);
    let b_data = random_vec(12, 17);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &a_shape)?;
    let b = store.from_slice(&b_data, &b_shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;
    store.get_mut(b).expect("b exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &[2, 2, 2])?;

    let y = matmul(a, b, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;
    let analytic_b = store.to_host(*grads.get(&b).expect("grad for b"))?;

    let mut a_numeric = a_data.clone();
    let numeric_a = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(x, &a_shape).expect("a");
            let b = store.from_slice(&b_data, &b_shape).expect("b");
            let coeff_id = store.from_slice(&coeff, &[2, 2, 2]).expect("coeff");
            let y = matmul(a, b, &mut store, &mut tape).expect("matmul");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric,
        1e-3,
    );
    let mut b_numeric = b_data.clone();
    let numeric_b = num_grad(
        |x| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(&a_data, &a_shape).expect("a");
            let b = store.from_slice(x, &b_shape).expect("b");
            let coeff_id = store.from_slice(&coeff, &[2, 2, 2]).expect("coeff");
            let y = matmul(a, b, &mut store, &mut tape).expect("matmul");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut b_numeric,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    assert!(max_abs_err(&analytic_b, &numeric_b) < 1e-3);
    Ok(())
}

#[test]
fn softmax_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let x_data = random_vec(6, 19);
    let coeff = random_vec(6, 23);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &shape)?;

    let y = softmax(x, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let coeff_id = store.from_slice(&coeff, &shape).expect("coeff");
            let y = softmax(x, &mut store, &mut tape).expect("softmax");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn log_softmax_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let x_data = random_vec(6, 29);
    let coeff = random_vec(6, 31);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &shape)?;

    let y = log_softmax(x, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let coeff_id = store.from_slice(&coeff, &shape).expect("coeff");
            let y = log_softmax(x, &mut store, &mut tape).expect("log_softmax");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn gather_last_dim_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3, 4];
    let src_data = random_vec(24, 37);
    let indices = vec![0, 2, 1, 3, 1, 0];

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let src = store.from_slice(&src_data, &shape)?;
    store.get_mut(src).expect("src exists").requires_grad = true;

    let gathered = gather_last_dim(src, &indices, &mut store, &mut tape)?;
    let loss = sum(gathered, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&src).expect("grad for src"))?;

    let mut numeric_input = src_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let src = store.from_slice(values, &shape).expect("src");
            let gathered =
                gather_last_dim(src, &indices, &mut store, &mut tape).expect("gather_last_dim");
            let loss = sum(gathered, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn mean_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let x_data = random_vec(6, 41);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;

    let loss = mean(x, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let loss = mean(x, &mut store, &mut tape).expect("mean");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn rmsnorm_grad_matches_numeric_for_input_and_weight() -> Result<()> {
    let x_shape = [2, 3];
    let weight_shape = [3];
    let coeff = random_vec(6, 43);
    let x_data = random_vec(6, 47);
    let weight_data = random_vec(3, 53);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &x_shape)?;
    let weight = store.from_slice(&weight_data, &weight_shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    store.get_mut(weight).expect("weight exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &x_shape)?;

    let y = rmsnorm(x, weight, 1e-5, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic_x = store.to_host(*grads.get(&x).expect("grad for x"))?;
    let analytic_weight = store.to_host(*grads.get(&weight).expect("grad for weight"))?;

    let mut x_numeric = x_data.clone();
    let numeric_x = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &x_shape).expect("x");
            let weight = store
                .from_slice(&weight_data, &weight_shape)
                .expect("weight");
            let coeff_id = store.from_slice(&coeff, &x_shape).expect("coeff");
            let y = rmsnorm(x, weight, 1e-5, &mut store, &mut tape).expect("rmsnorm");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut x_numeric,
        1e-3,
    );
    let mut weight_numeric = weight_data.clone();
    let numeric_weight = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(&x_data, &x_shape).expect("x");
            let weight = store.from_slice(values, &weight_shape).expect("weight");
            let coeff_id = store.from_slice(&coeff, &x_shape).expect("coeff");
            let y = rmsnorm(x, weight, 1e-5, &mut store, &mut tape).expect("rmsnorm");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut weight_numeric,
        1e-3,
    );

    assert!(max_abs_err(&analytic_x, &numeric_x) < 1e-3);
    assert!(max_abs_err(&analytic_weight, &numeric_weight) < 1e-3);
    Ok(())
}

#[test]
fn gelu_grad_matches_numeric() -> Result<()> {
    let shape = [2, 3];
    let x_data = random_vec(6, 59);
    let coeff = random_vec(6, 61);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &shape)?;

    let y = gelu(x, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let coeff_id = store.from_slice(&coeff, &shape).expect("coeff");
            let y = gelu(x, &mut store, &mut tape).expect("gelu");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn transpose_grad_matches_numeric_round_trip() -> Result<()> {
    let shape = [2, 3, 4];
    let x_data = random_vec(24, 67);
    let coeff = random_vec(24, 71);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let x = store.from_slice(&x_data, &shape)?;
    store.get_mut(x).expect("x exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &shape)?;

    let y = transpose(x, 0, 2, &mut store, &mut tape)?;
    let z = transpose(y, 0, 2, &mut store, &mut tape)?;
    let weighted = mul(z, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&x).expect("grad for x"))?;

    let mut numeric_input = x_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let x = store.from_slice(values, &shape).expect("x");
            let coeff_id = store.from_slice(&coeff, &shape).expect("coeff");
            let y = transpose(x, 0, 2, &mut store, &mut tape).expect("transpose");
            let z = transpose(y, 0, 2, &mut store, &mut tape).expect("transpose");
            let weighted = mul(z, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}

#[test]
fn add_broadcast_grad_matches_numeric_for_broadcast_operand() -> Result<()> {
    let a_shape = [2, 3, 4];
    let b_shape = [4];
    let a_data = random_vec(24, 73);
    let b_data = random_vec(4, 79);
    let coeff = random_vec(24, 83);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let a = store.from_slice(&a_data, &a_shape)?;
    let b = store.from_slice(&b_data, &b_shape)?;
    store.get_mut(a).expect("a exists").requires_grad = true;
    store.get_mut(b).expect("b exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &a_shape)?;

    let y = add_broadcast(a, b, &mut store, &mut tape)?;
    let weighted = mul(y, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic_a = store.to_host(*grads.get(&a).expect("grad for a"))?;
    let analytic_b = store.to_host(*grads.get(&b).expect("grad for b"))?;

    let mut a_numeric = a_data.clone();
    let numeric_a = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(values, &a_shape).expect("a");
            let b = store.from_slice(&b_data, &b_shape).expect("b");
            let coeff_id = store.from_slice(&coeff, &a_shape).expect("coeff");
            let y = add_broadcast(a, b, &mut store, &mut tape).expect("add_broadcast");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut a_numeric,
        1e-3,
    );
    let mut b_numeric = b_data.clone();
    let numeric_b = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let a = store.from_slice(&a_data, &a_shape).expect("a");
            let b = store.from_slice(values, &b_shape).expect("b");
            let coeff_id = store.from_slice(&coeff, &a_shape).expect("coeff");
            let y = add_broadcast(a, b, &mut store, &mut tape).expect("add_broadcast");
            let weighted = mul(y, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut b_numeric,
        1e-3,
    );

    assert!(max_abs_err(&analytic_a, &numeric_a) < 1e-3);
    assert!(max_abs_err(&analytic_b, &numeric_b) < 1e-3);
    Ok(())
}

#[test]
fn embedding_grad_matches_numeric_with_scatter_add() -> Result<()> {
    let table_shape = [5, 3];
    let table_data = random_vec(15, 89);
    let indices = vec![1, 3, 1, 4];
    let coeff = random_vec(12, 97);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let table = store.from_slice(&table_data, &table_shape)?;
    store.get_mut(table).expect("table exists").requires_grad = true;
    let coeff_id = store.from_slice(&coeff, &[1, 4, 3])?;

    let embedded = embedding(table, &indices, &mut store, &mut tape)?;
    let weighted = mul(embedded, coeff_id, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let analytic = store.to_host(*grads.get(&table).expect("grad for table"))?;

    let mut numeric_input = table_data.clone();
    let numeric = num_grad(
        |values| {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let table = store.from_slice(values, &table_shape).expect("table");
            let coeff_id = store.from_slice(&coeff, &[1, 4, 3]).expect("coeff");
            let embedded = embedding(table, &indices, &mut store, &mut tape).expect("embedding");
            let weighted = mul(embedded, coeff_id, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        },
        &mut numeric_input,
        1e-3,
    );

    assert!(max_abs_err(&analytic, &numeric) < 1e-3);
    Ok(())
}
