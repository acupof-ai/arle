mod helpers;

use autograd::{
    Result, Tape, TensorStore,
    ops::{LinearAttentionParams, linear_attention_core, mul, sum},
};
#[cfg(any(feature = "metal", feature = "cuda"))]
use helpers::max_abs_err;
use helpers::num_grad;

#[cfg(feature = "cuda")]
use autograd::backend_cuda::CudaBackend;
#[cfg(feature = "metal")]
use autograd::backend_metal::MetalBackend;
#[cfg(any(feature = "metal", feature = "cuda"))]
use std::sync::Arc;
#[cfg(feature = "metal")]
use std::sync::Mutex;

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

type LinearAttentionLossAndGrads = (
    f32,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
);

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

    for tensor_id in [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ] {
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
        store.to_host(*grads.get(&conv1d_weight).expect("grad for conv1d_weight"))?,
        store.to_host(*grads.get(&dt_bias).expect("grad for dt_bias"))?,
        store.to_host(*grads.get(&a_log).expect("grad for a_log"))?,
        store.to_host(*grads.get(&norm_weight).expect("grad for norm_weight"))?,
    ))
}

fn loss_for_variant(
    fixture: &LinearAttentionFixture,
    params: LinearAttentionParams,
    qkv: &[f32],
    z: &[f32],
    b_proj: &[f32],
    a_proj: &[f32],
    conv1d_weight: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    norm_weight: &[f32],
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
        .from_slice(conv1d_weight, &[qkv_dim(params), params.conv_kernel])
        .expect("conv1d_weight");
    let dt_bias = store
        .from_slice(dt_bias, &[params.num_value_heads])
        .expect("dt_bias");
    let a_log = store
        .from_slice(a_log, &[params.num_value_heads])
        .expect("a_log");
    let norm_weight = store
        .from_slice(norm_weight, &[params.value_dim])
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
    let (
        _,
        analytic_qkv,
        analytic_z,
        analytic_b,
        analytic_a,
        analytic_conv,
        analytic_dt,
        analytic_a_log,
        analytic_norm,
    ) = loss_and_grads(&fixture, params)?;

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
                &fixture.conv1d_weight,
                &fixture.dt_bias,
                &fixture.a_log,
                &fixture.norm_weight,
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
                &fixture.conv1d_weight,
                &fixture.dt_bias,
                &fixture.a_log,
                &fixture.norm_weight,
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
                &fixture.conv1d_weight,
                &fixture.dt_bias,
                &fixture.a_log,
                &fixture.norm_weight,
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
                &fixture.conv1d_weight,
                &fixture.dt_bias,
                &fixture.a_log,
                &fixture.norm_weight,
            )
        },
        &mut a_numeric_input,
        1.0e-3,
    );
    let mut conv_numeric_input = fixture.conv1d_weight.clone();
    let numeric_conv = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                &fixture.z,
                &fixture.b_proj,
                &fixture.a_proj,
                values,
                &fixture.dt_bias,
                &fixture.a_log,
                &fixture.norm_weight,
            )
        },
        &mut conv_numeric_input,
        1.0e-3,
    );
    let mut dt_numeric_input = fixture.dt_bias.clone();
    let numeric_dt = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                &fixture.z,
                &fixture.b_proj,
                &fixture.a_proj,
                &fixture.conv1d_weight,
                values,
                &fixture.a_log,
                &fixture.norm_weight,
            )
        },
        &mut dt_numeric_input,
        1.0e-3,
    );
    let mut a_log_numeric_input = fixture.a_log.clone();
    let numeric_a_log = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                &fixture.z,
                &fixture.b_proj,
                &fixture.a_proj,
                &fixture.conv1d_weight,
                &fixture.dt_bias,
                values,
                &fixture.norm_weight,
            )
        },
        &mut a_log_numeric_input,
        1.0e-3,
    );
    let mut norm_numeric_input = fixture.norm_weight.clone();
    let numeric_norm = num_grad(
        |values| {
            loss_for_variant(
                &fixture,
                params,
                &fixture.qkv,
                &fixture.z,
                &fixture.b_proj,
                &fixture.a_proj,
                &fixture.conv1d_weight,
                &fixture.dt_bias,
                &fixture.a_log,
                values,
            )
        },
        &mut norm_numeric_input,
        1.0e-3,
    );

    let (qkv_idx, qkv_err) = max_err_with_index(&analytic_qkv, &numeric_qkv);
    let (z_idx, z_err) = max_err_with_index(&analytic_z, &numeric_z);
    let (b_idx, b_err) = max_err_with_index(&analytic_b, &numeric_b);
    let (a_idx, a_err) = max_err_with_index(&analytic_a, &numeric_a);
    let (conv_idx, conv_err) = max_err_with_index(&analytic_conv, &numeric_conv);
    let (dt_idx, dt_err) = max_err_with_index(&analytic_dt, &numeric_dt);
    let (a_log_idx, a_log_err) = max_err_with_index(&analytic_a_log, &numeric_a_log);
    let (norm_idx, norm_err) = max_err_with_index(&analytic_norm, &numeric_norm);

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
    assert!(
        conv_err < 8.0e-3,
        "conv grad max abs err {conv_err} at index {conv_idx}"
    );
    assert!(
        dt_err < 8.0e-3,
        "dt grad max abs err {dt_err} at index {dt_idx}"
    );
    assert!(
        a_log_err < 8.0e-3,
        "a_log grad max abs err {a_log_err} at index {a_log_idx}"
    );
    assert!(
        norm_err < 8.0e-3,
        "norm grad max abs err {norm_err} at index {norm_idx}"
    );

    Ok(())
}

#[test]
fn linear_attention_backward_keeps_tiny_decay_finite() -> Result<()> {
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 20,
        num_key_heads: 1,
        num_value_heads: 1,
        key_dim: 2,
        value_dim: 2,
        conv_kernel: 1,
        eps: 1.0e-6,
    };
    let qkv_dim = qkv_dim(params);
    let z_dim = z_dim(params);
    let head_len = params.batch * params.seq_len * params.num_value_heads;
    let mut store = TensorStore::default();
    let mut tape = Tape::new();

    let qkv = store.from_slice(
        &vec![3.0; params.batch * params.seq_len * qkv_dim],
        &[params.batch, params.seq_len, qkv_dim],
    )?;
    let z = store.from_slice(
        &vec![3.0; params.batch * params.seq_len * z_dim],
        &[params.batch, params.seq_len, z_dim],
    )?;
    let b_proj = store.from_slice(
        &vec![0.0; head_len],
        &[params.batch, params.seq_len, params.num_value_heads],
    )?;
    let a_proj = store.from_slice(
        &vec![0.5; head_len],
        &[params.batch, params.seq_len, params.num_value_heads],
    )?;
    let conv1d_weight = store.from_slice(&vec![1.0; qkv_dim], &[qkv_dim, params.conv_kernel])?;
    let dt_bias = store.from_slice(&[0.0], &[params.num_value_heads])?;
    let a_log = store.from_slice(&[3.0], &[params.num_value_heads])?;
    let norm_weight = store.from_slice(&vec![1.0; params.value_dim], &[params.value_dim])?;

    for tensor_id in [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ] {
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
    let loss = sum(output, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    for (label, tensor_id) in [
        ("qkv", qkv),
        ("z", z),
        ("b_proj", b_proj),
        ("a_proj", a_proj),
        ("conv1d_weight", conv1d_weight),
        ("dt_bias", dt_bias),
        ("a_log", a_log),
        ("norm_weight", norm_weight),
    ] {
        let grad_id = *grads.get(&tensor_id).expect("grad exists");
        let grad = store.to_host(grad_id)?;
        assert!(
            grad.iter().all(|value| value.is_finite()),
            "{label} grad contains non-finite values"
        );
    }

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
    for tensor_id in [
        cpu_qkv, cpu_z, cpu_b, cpu_a, cpu_conv, cpu_dt, cpu_a_log, cpu_norm,
    ] {
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
    let cpu_conv_grad = cpu_store.to_host(*cpu_grads.get(&cpu_conv).expect("cpu conv grad"))?;
    let cpu_dt_grad = cpu_store.to_host(*cpu_grads.get(&cpu_dt).expect("cpu dt grad"))?;
    let cpu_a_log_grad = cpu_store.to_host(*cpu_grads.get(&cpu_a_log).expect("cpu a_log grad"))?;
    let cpu_norm_grad = cpu_store.to_host(*cpu_grads.get(&cpu_norm).expect("cpu norm grad"))?;

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
    for tensor_id in [
        metal_qkv,
        metal_z,
        metal_b,
        metal_a,
        metal_conv,
        metal_dt,
        metal_a_log,
        metal_norm,
    ] {
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
    let metal_conv_grad =
        metal_store.to_host(*metal_grads.get(&metal_conv).expect("metal conv grad"))?;
    let metal_dt_grad = metal_store.to_host(*metal_grads.get(&metal_dt).expect("metal dt grad"))?;
    let metal_a_log_grad =
        metal_store.to_host(*metal_grads.get(&metal_a_log).expect("metal a_log grad"))?;
    let metal_norm_grad =
        metal_store.to_host(*metal_grads.get(&metal_norm).expect("metal norm grad"))?;

    assert!(max_abs_err(&metal_out_host, &cpu_out_host) <= 1.0e-6);
    assert!(max_abs_err(&metal_qkv_grad, &cpu_qkv_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_z_grad, &cpu_z_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_b_grad, &cpu_b_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_a_grad, &cpu_a_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_conv_grad, &cpu_conv_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_dt_grad, &cpu_dt_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_a_log_grad, &cpu_a_log_grad) <= 1.0e-6);
    assert!(max_abs_err(&metal_norm_grad, &cpu_norm_grad) <= 1.0e-6);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linear_attention_matches_cpu_with_device_inputs() -> Result<()> {
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
    for tensor_id in [
        cpu_qkv, cpu_z, cpu_b, cpu_a, cpu_conv, cpu_dt, cpu_a_log, cpu_norm,
    ] {
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
    let cpu_conv_grad = cpu_store.to_host(*cpu_grads.get(&cpu_conv).expect("cpu conv grad"))?;
    let cpu_dt_grad = cpu_store.to_host(*cpu_grads.get(&cpu_dt).expect("cpu dt grad"))?;
    let cpu_a_log_grad = cpu_store.to_host(*cpu_grads.get(&cpu_a_log).expect("cpu a_log grad"))?;
    let cpu_norm_grad = cpu_store.to_host(*cpu_grads.get(&cpu_norm).expect("cpu norm grad"))?;

    let Ok(cuda_backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_linear_attention_matches_cpu_with_device_inputs: no CUDA device");
        return Ok(());
    };
    let mut cuda_store = TensorStore::with_backend(Arc::new(cuda_backend));
    let mut cuda_tape = Tape::new();
    let cuda_qkv = cuda_store.from_slice(&fixture.qkv, &qkv_shape)?;
    let cuda_z = cuda_store.from_slice(&fixture.z, &z_shape)?;
    let cuda_b = cuda_store.from_slice(&fixture.b_proj, &head_shape)?;
    let cuda_a = cuda_store.from_slice(&fixture.a_proj, &head_shape)?;
    let cuda_conv = cuda_store.from_slice(
        &fixture.conv1d_weight,
        &[qkv_dim(params), params.conv_kernel],
    )?;
    let cuda_dt = cuda_store.from_slice(&fixture.dt_bias, &[params.num_value_heads])?;
    let cuda_a_log = cuda_store.from_slice(&fixture.a_log, &[params.num_value_heads])?;
    let cuda_norm = cuda_store.from_slice(&fixture.norm_weight, &[params.value_dim])?;
    let cuda_coeff = cuda_store.from_slice(&fixture.coeff, &z_shape)?;
    for tensor_id in [
        cuda_qkv, cuda_z, cuda_b, cuda_a, cuda_conv, cuda_dt, cuda_a_log, cuda_norm, cuda_coeff,
    ] {
        cuda_store.ensure_device(tensor_id)?;
    }
    for tensor_id in [
        cuda_qkv, cuda_z, cuda_b, cuda_a, cuda_conv, cuda_dt, cuda_a_log, cuda_norm,
    ] {
        cuda_store
            .get_mut(tensor_id)
            .expect("cuda tensor exists")
            .requires_grad = true;
    }
    let cuda_out = linear_attention_core(
        cuda_qkv,
        cuda_z,
        cuda_b,
        cuda_a,
        cuda_conv,
        cuda_dt,
        cuda_a_log,
        cuda_norm,
        params,
        &mut cuda_store,
        &mut cuda_tape,
    )?;
    let cuda_weighted = mul(cuda_out, cuda_coeff, &mut cuda_store, &mut cuda_tape)?;
    let cuda_loss = sum(cuda_weighted, &mut cuda_store, &mut cuda_tape)?;
    let cuda_grads = cuda_tape.backward(cuda_loss, &mut cuda_store)?;
    let cuda_out_host = cuda_store.to_host(cuda_out)?;
    let cuda_qkv_grad = cuda_store.to_host(*cuda_grads.get(&cuda_qkv).expect("cuda qkv grad"))?;
    let cuda_z_grad = cuda_store.to_host(*cuda_grads.get(&cuda_z).expect("cuda z grad"))?;
    let cuda_b_grad = cuda_store.to_host(*cuda_grads.get(&cuda_b).expect("cuda b grad"))?;
    let cuda_a_grad = cuda_store.to_host(*cuda_grads.get(&cuda_a).expect("cuda a grad"))?;
    let cuda_conv_grad =
        cuda_store.to_host(*cuda_grads.get(&cuda_conv).expect("cuda conv grad"))?;
    let cuda_dt_grad = cuda_store.to_host(*cuda_grads.get(&cuda_dt).expect("cuda dt grad"))?;
    let cuda_a_log_grad =
        cuda_store.to_host(*cuda_grads.get(&cuda_a_log).expect("cuda a_log grad"))?;
    let cuda_norm_grad =
        cuda_store.to_host(*cuda_grads.get(&cuda_norm).expect("cuda norm grad"))?;

    assert!(max_abs_err(&cuda_out_host, &cpu_out_host) <= 1.0e-6);
    assert!(max_abs_err(&cuda_qkv_grad, &cpu_qkv_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_z_grad, &cpu_z_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_b_grad, &cpu_b_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_a_grad, &cpu_a_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_conv_grad, &cpu_conv_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_dt_grad, &cpu_dt_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_a_log_grad, &cpu_a_log_grad) <= 1.0e-3);
    assert!(max_abs_err(&cuda_norm_grad, &cpu_norm_grad) <= 1.0e-3);

    Ok(())
}
