mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{LinearAttentionParams, linear_attention_core, mul, sum},
};
use helpers::{max_abs_err, num_grad};

#[cfg(feature = "metal")]
use autograd::backend_metal::MetalBackend;
#[cfg(feature = "metal")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "metal")]
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn tiny_params() -> LinearAttentionParams {
    LinearAttentionParams {
        batch: 1,
        seq_len: 3,
        num_key_heads: 1,
        num_value_heads: 1,
        key_dim: 2,
        value_dim: 2,
        conv_kernel: 2,
        eps: 1.0e-5,
    }
}

fn qkv_dim(params: LinearAttentionParams) -> usize {
    params.num_key_heads * params.key_dim * 2 + params.num_value_heads * params.value_dim
}

fn z_dim(params: LinearAttentionParams) -> usize {
    params.num_value_heads * params.value_dim
}

#[derive(Clone)]
struct LinearAttentionFixture {
    qkv: Vec<f32>,
    z: Vec<f32>,
    b_proj: Vec<f32>,
    a_proj: Vec<f32>,
    conv1d_weight: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    norm_weight: Vec<f32>,
    coeff: Vec<f32>,
}

type LinearAttentionLossAndGrads = (f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

impl LinearAttentionFixture {
    fn new(params: LinearAttentionParams) -> Self {
        let qkv_len = params.batch * params.seq_len * qkv_dim(params);
        let z_len = params.batch * params.seq_len * z_dim(params);
        let head_len = params.batch * params.seq_len * params.num_value_heads;
        let conv_len = qkv_dim(params) * params.conv_kernel;
        Self {
            qkv: (0..qkv_len)
                .map(|i| ((i as f32 * 0.17).sin()) * 0.15)
                .collect(),
            z: (0..z_len)
                .map(|i| ((i as f32 * 0.11).cos()) * 0.12)
                .collect(),
            b_proj: (0..head_len)
                .map(|i| ((i as f32 * 0.23).sin()) * 0.08)
                .collect(),
            a_proj: (0..head_len)
                .map(|i| ((i as f32 * 0.19).cos()) * 0.07)
                .collect(),
            conv1d_weight: (0..conv_len)
                .map(|i| ((i as f32 * 0.13).sin()) * 0.09)
                .collect(),
            dt_bias: vec![0.05],
            a_log: vec![-0.3],
            norm_weight: vec![1.0, 0.9],
            coeff: (0..z_len)
                .map(|i| 0.3 + ((i as f32 * 0.07).sin()) * 0.05)
                .collect(),
        }
    }
}

fn loss_and_grads(
    fixture: &LinearAttentionFixture,
    params: LinearAttentionParams,
) -> Result<LinearAttentionLossAndGrads> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let qkv_shape = [params.batch, params.seq_len, qkv_dim(params)];
    let z_shape = [params.batch, params.seq_len, z_dim(params)];
    let head_shape = [params.batch, params.seq_len, params.num_value_heads];

    let qkv = store.from_slice(&fixture.qkv, &qkv_shape)?;
    let z = store.from_slice(&fixture.z, &z_shape)?;
    let b_proj = store.from_slice(&fixture.b_proj, &head_shape)?;
    let a_proj = store.from_slice(&fixture.a_proj, &head_shape)?;
    let conv1d_weight = store.from_slice(
        &fixture.conv1d_weight,
        &[qkv_dim(params), params.conv_kernel],
    )?;
    let dt_bias = store.from_slice(&fixture.dt_bias, &[params.num_value_heads])?;
    let a_log = store.from_slice(&fixture.a_log, &[params.num_value_heads])?;
    let norm_weight = store.from_slice(&fixture.norm_weight, &[params.value_dim])?;
    let coeff = store.from_slice(&fixture.coeff, &z_shape)?;

    for tensor_id in [qkv, z, b_proj, a_proj] {
        store
            .get_mut(tensor_id)
            .expect("tensor exists")
            .requires_grad = true;
    }

    let output = linear_attention_core(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        &mut store,
        &mut tape,
    )?;
    let weighted = mul(output, coeff, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    Ok((
        store.to_host(loss)?[0],
        store.to_host(*grads.get(&qkv).expect("grad for qkv"))?,
        store.to_host(*grads.get(&z).expect("grad for z"))?,
        store.to_host(*grads.get(&b_proj).expect("grad for b_proj"))?,
        store.to_host(*grads.get(&a_proj).expect("grad for a_proj"))?,
    ))
}

fn loss_for_variant(
    fixture: &LinearAttentionFixture,
    params: LinearAttentionParams,
    qkv: &[f32],
    z: &[f32],
    b_proj: &[f32],
    a_proj: &[f32],
) -> f32 {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let qkv_shape = [params.batch, params.seq_len, qkv_dim(params)];
    let z_shape = [params.batch, params.seq_len, z_dim(params)];
    let head_shape = [params.batch, params.seq_len, params.num_value_heads];
    let qkv = store.from_slice(qkv, &qkv_shape).expect("qkv");
    let z = store.from_slice(z, &z_shape).expect("z");
    let b_proj = store.from_slice(b_proj, &head_shape).expect("b_proj");
    let a_proj = store.from_slice(a_proj, &head_shape).expect("a_proj");
    let conv1d_weight = store
        .from_slice(
            &fixture.conv1d_weight,
            &[qkv_dim(params), params.conv_kernel],
        )
        .expect("conv1d_weight");
    let dt_bias = store
        .from_slice(&fixture.dt_bias, &[params.num_value_heads])
        .expect("dt_bias");
    let a_log = store
        .from_slice(&fixture.a_log, &[params.num_value_heads])
        .expect("a_log");
    let norm_weight = store
        .from_slice(&fixture.norm_weight, &[params.value_dim])
        .expect("norm_weight");
    let coeff = store.from_slice(&fixture.coeff, &z_shape).expect("coeff");

    let output = linear_attention_core(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        &mut store,
        &mut tape,
    )
    .expect("linear attention");
    let weighted = mul(output, coeff, &mut store, &mut tape).expect("mul");
    let loss = sum(weighted, &mut store, &mut tape).expect("sum");
    store.to_host(loss).expect("loss")[0]
}

fn max_err_with_index(lhs: &[f32], rhs: &[f32]) -> (usize, f32) {
    lhs.iter()
        .zip(rhs.iter())
        .enumerate()
        .map(|(idx, (a, b))| (idx, (a - b).abs()))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite errors"))
        .expect("non-empty slices")
}

#[test]
fn linear_attention_grad_matches_numeric() -> Result<()> {
    let params = tiny_params();
    let fixture = LinearAttentionFixture::new(params);
    let (_, analytic_qkv, analytic_z, analytic_b, analytic_a) = loss_and_grads(&fixture, params)?;

    let mut qkv_numeric_input = fixture.qkv.clone();
    let numeric_qkv = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                values,
                &fixture.z,
                &fixture.b_proj,
                &fixture.a_proj,
            )
        },
        &mut qkv_numeric_input,
        1.0e-3,
    );
    let mut z_numeric_input = fixture.z.clone();
    let numeric_z = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                values,
                &fixture.b_proj,
                &fixture.a_proj,
            )
        },
        &mut z_numeric_input,
        1.0e-3,
    );
    let mut b_numeric_input = fixture.b_proj.clone();
    let numeric_b = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                &fixture.z,
                values,
                &fixture.a_proj,
            )
        },
        &mut b_numeric_input,
        1.0e-3,
    );
    let mut a_numeric_input = fixture.a_proj.clone();
    let numeric_a = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                &fixture.z,
                &fixture.b_proj,
                values,
            )
        },
        &mut a_numeric_input,
        1.0e-3,
    );

    let (qkv_idx, qkv_err) = max_err_with_index(&analytic_qkv, &numeric_qkv);
    let (z_idx, z_err) = max_err_with_index(&analytic_z, &numeric_z);
    let (b_idx, b_err) = max_err_with_index(&analytic_b, &numeric_b);
    let (a_idx, a_err) = max_err_with_index(&analytic_a, &numeric_a);

    assert!(
        qkv_err < 8.0e-3,
        "qkv grad max abs err {qkv_err} at index {qkv_idx}"
    );
    assert!(
        z_err < 8.0e-3,
        "z grad max abs err {z_err} at index {z_idx}"
    );
    assert!(
        b_err < 8.0e-3,
        "b grad max abs err {b_err} at index {b_idx}"
    );
    assert!(
        a_err < 8.0e-3,
        "a grad max abs err {a_err} at index {a_idx}"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_linear_attention_matches_cpu_with_device_inputs() -> Result<()> {
    let _lock = METAL_TEST_LOCK.lock().expect("metal test lock poisoned");

    let params = tiny_params();
    let fixture = LinearAttentionFixture::new(params);
    let qkv_shape = [params.batch, params.seq_len, qkv_dim(params)];
    let z_shape = [params.batch, params.seq_len, z_dim(params)];
    let head_shape = [params.batch, params.seq_len, params.num_value_heads];

    let mut cpu_store = TensorStore::default();
    let mut cpu_tape = Tape::new();
    let cpu_qkv = cpu_store.from_slice(&fixture.qkv, &qkv_shape)?;
    let cpu_z = cpu_store.from_slice(&fixture.z, &z_shape)?;
    let cpu_b = cpu_store.from_slice(&fixture.b_proj, &head_shape)?;
    let cpu_a = cpu_store.from_slice(&fixture.a_proj, &head_shape)?;
    let cpu_conv = cpu_store.from_slice(
        &fixture.conv1d_weight,
        &[qkv_dim(params), params.conv_kernel],
    )?;
    let cpu_dt = cpu_store.from_slice(&fixture.dt_bias, &[params.num_value_heads])?;
    let cpu_a_log = cpu_store.from_slice(&fixture.a_log, &[params.num_value_heads])?;
    let cpu_norm = cpu_store.from_slice(&fixture.norm_weight, &[params.value_dim])?;
    let cpu_coeff = cpu_store.from_slice(&fixture.coeff, &z_shape)?;
    for tensor_id in [cpu_qkv, cpu_z, cpu_b, cpu_a] {
        cpu_store
            .get_mut(tensor_id)
            .expect("cpu tensor exists")
            .requires_grad = true;
    }
    let cpu_out = linear_attention_core(
        cpu_qkv,
        cpu_z,
        cpu_b,
        cpu_a,
        cpu_conv,
        cpu_dt,
        cpu_a_log,
        cpu_norm,
        params,
        &mut cpu_store,
        &mut cpu_tape,
    )?;
    let cpu_weighted = mul(cpu_out, cpu_coeff, &mut cpu_store, &mut cpu_tape)?;
    let cpu_loss = sum(cpu_weighted, &mut cpu_store, &mut cpu_tape)?;
    let cpu_grads = cpu_tape.backward(cpu_loss, &mut cpu_store)?;
    let cpu_out_host = cpu_store.to_host(cpu_out)?;
    let cpu_qkv_grad = cpu_store.to_host(*cpu_grads.get(&cpu_qkv).expect("cpu qkv grad"))?;
    let cpu_z_grad = cpu_store.to_host(*cpu_grads.get(&cpu_z).expect("cpu z grad"))?;
    let cpu_b_grad = cpu_store.to_host(*cpu_grads.get(&cpu_b).expect("cpu b grad"))?;
    let cpu_a_grad = cpu_store.to_host(*cpu_grads.get(&cpu_a).expect("cpu a grad"))?;

    let mut metal_store = TensorStore::with_backend(Arc::new(MetalBackend));
    let mut metal_tape = Tape::new();
    let metal_qkv = metal_store.from_slice(&fixture.qkv, &qkv_shape)?;
    let metal_z = metal_store.from_slice(&fixture.z, &z_shape)?;
    let metal_b = metal_store.from_slice(&fixture.b_proj, &head_shape)?;
    let metal_a = metal_store.from_slice(&fixture.a_proj, &head_shape)?;
    let metal_conv = metal_store.from_slice(
        &fixture.conv1d_weight,
        &[qkv_dim(params), params.conv_kernel],
    )?;
    let metal_dt = metal_store.from_slice(&fixture.dt_bias, &[params.num_value_heads])?;
    let metal_a_log = metal_store.from_slice(&fixture.a_log, &[params.num_value_heads])?;
    let metal_norm = metal_store.from_slice(&fixture.norm_weight, &[params.value_dim])?;
    let metal_coeff = metal_store.from_slice(&fixture.coeff, &z_shape)?;
    for tensor_id in [
        metal_qkv,
        metal_z,
        metal_b,
        metal_a,
        metal_conv,
        metal_dt,
        metal_a_log,
        metal_norm,
        metal_coeff,
    ] {
        metal_store.ensure_device(tensor_id)?;
    }
    for tensor_id in [metal_qkv, metal_z, metal_b, metal_a] {
        metal_store
            .get_mut(tensor_id)
            .expect("metal tensor exists")
            .requires_grad = true;
    }
    let metal_out = linear_attention_core(
        metal_qkv,
        metal_z,
        metal_b,
        metal_a,
        metal_conv,
        metal_dt,
        metal_a_log,
        metal_norm,
        params,
        &mut metal_store,
        &mut metal_tape,
    )?;
    let metal_weighted = mul(metal_out, metal_coeff, &mut metal_store, &mut metal_tape)?;
    let metal_loss = sum(metal_weighted, &mut metal_store, &mut metal_tape)?;
    let metal_grads = metal_tape.backward(metal_loss, &mut metal_store)?;
    let metal_out_host = metal_store.to_host(metal_out)?;
    let metal_qkv_grad =
        metal_store.to_host(*metal_grads.get(&metal_qkv).expect("metal qkv grad"))?;
    let metal_z_grad = metal_store.to_host(*metal_grads.get(&metal_z).expect("metal z grad"))?;
    let metal_b_grad = metal_store.to_host(*metal_grads.get(&metal_b).expect("metal b grad"))?;
    let metal_a_grad = metal_store.to_host(*metal_grads.get(&metal_a).expect("metal a grad"))?;

    assert!(max_abs_err(&metal_out_host, &cpu_out_host) <= 1.0e-6);
    assert!(max_abs_err(&metal_qkv_grad, &cpu_qkv_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_z_grad, &cpu_z_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_b_grad, &cpu_b_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_a_grad, &cpu_a_grad) <= 1.0e-6);

    Ok(())
}
