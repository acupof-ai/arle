mod helpers;

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use autograd::ops::linear_attention_boundary;
use autograd::{
    Result, Tape, TensorStore,
    ops::{
        LinearAttentionParams, linear_attention_core, linear_attention_core_with_carry,
        linear_attention_core_with_carry_taped, mul, sum,
    },
};
#[cfg(any(feature = "metal", feature = "cuda"))]
use helpers::max_abs_err;
use helpers::num_grad;

#[cfg(feature = "cuda")]
use autograd::backend_cuda::CudaBackend;
#[cfg(feature = "metal")]
use autograd::backend_metal::MetalBackend;
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use autograd::tensor::Dirty;
#[cfg(any(feature = "metal", feature = "cuda"))]
use std::sync::Arc;
#[cfg(feature = "metal")]
use std::sync::Mutex;
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use std::time::Instant;

#[cfg(feature = "metal")]
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(any(feature = "metal", feature = "cuda"))]
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

fn host_gradcheck_params() -> LinearAttentionParams {
    LinearAttentionParams {
        batch: 1,
        seq_len: 80,
        num_key_heads: 1,
        num_value_heads: 2,
        key_dim: 16,
        value_dim: 16,
        conv_kernel: 4,
        eps: 1.0e-5,
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn qwen35_chunked_params(seq_len: usize) -> LinearAttentionParams {
    LinearAttentionParams {
        batch: 1,
        seq_len,
        num_key_heads: 8,
        num_value_heads: 32,
        key_dim: 128,
        value_dim: 128,
        conv_kernel: 4,
        eps: 1.0e-5,
    }
}

// Qwen3.6-27B GatedDeltaNet: 48 value heads / 16 key heads (vs 35B-A3B's 32/8).
// Guards the relaxed head-count check in the device LA dispatch.
#[allow(dead_code)] // only used by the cuda-gated test below
fn qwen36_27b_chunked_params(seq_len: usize) -> LinearAttentionParams {
    LinearAttentionParams {
        batch: 1,
        seq_len,
        num_key_heads: 16,
        num_value_heads: 48,
        key_dim: 128,
        value_dim: 128,
        conv_kernel: 4,
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
            dt_bias: (0..params.num_value_heads)
                .map(|i| 0.03 + ((i as f32 * 0.17).sin()) * 0.02)
                .collect(),
            a_log: (0..params.num_value_heads)
                .map(|i| -0.3 + ((i as f32 * 0.13).cos()) * 0.04)
                .collect(),
            norm_weight: (0..params.value_dim)
                .map(|i| 0.9 + ((i as f32 * 0.07).sin()) * 0.1)
                .collect(),
            coeff: (0..z_len)
                .map(|i| 0.3 + ((i as f32 * 0.07).sin()) * 0.05)
                .collect(),
        }
    }

    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    fn hard_forget(params: LinearAttentionParams) -> Self {
        let mut fixture = Self::new(params);
        fixture.qkv.iter_mut().for_each(|v| *v *= 0.5);
        fixture.z.iter_mut().for_each(|v| *v *= 0.5);
        fixture.b_proj.fill(0.0);
        fixture.a_proj.fill(0.5);
        fixture.dt_bias.fill(0.0);
        fixture.a_log.fill(3.0);
        fixture.conv1d_weight.iter_mut().for_each(|v| *v *= 0.5);
        fixture
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

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn max_rel_err(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0e-6))
        .fold(0.0, f32::max)
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn max_rel_err_qkv_range(
    lhs: &[f32],
    rhs: &[f32],
    params: LinearAttentionParams,
    start: usize,
    len: usize,
) -> f32 {
    let qkv_dim = qkv_dim(params);
    let mut err = 0.0_f32;
    for token in 0..(params.batch * params.seq_len) {
        let base = token * qkv_dim + start;
        err = err.max(max_rel_err(&lhs[base..base + len], &rhs[base..base + len]));
    }
    err
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn max_abs_err_qkv_range(
    lhs: &[f32],
    rhs: &[f32],
    params: LinearAttentionParams,
    start: usize,
    len: usize,
) -> f32 {
    let qkv_dim = qkv_dim(params);
    let mut err = 0.0_f32;
    for token in 0..(params.batch * params.seq_len) {
        let base = token * qkv_dim + start;
        err = err.max(max_abs_err(&lhs[base..base + len], &rhs[base..base + len]));
    }
    err
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn compare_cpu_cuda_device_linear_attention(
    params: LinearAttentionParams,
    fixture: &LinearAttentionFixture,
    label: &str,
    rel_tol: f32,
) -> Result<()> {
    compare_cpu_cuda_device_linear_attention_carry(params, fixture, label, rel_tol, None)
}

/// CPU-vs-CUDA grad compare, optionally seeding an OPD frozen-prompt carry
/// `(initial_state, initial_conv_window)`. With a carry the CUDA path exercises
/// the tranche-2 device carry route (host CPU stays the oracle).
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn compare_cpu_cuda_device_linear_attention_carry(
    params: LinearAttentionParams,
    fixture: &LinearAttentionFixture,
    label: &str,
    rel_tol: f32,
    carry: Option<(&[f32], &[f32])>,
) -> Result<()> {
    // Non-carry compares stay strict at 1e-3. The carry variant's device backward
    // reads the saved bf16 qkv_conv (bit-matching the production forward, which stores
    // conv output as bf16), while the CPU oracle recomputes conv in f32 — a legitimate
    // precision gap, not a logic bug. It concentrates on the carry-fed boundary tokens
    // (dq worst at tok 0-1) and conv boundary taps (dconv worst at taps 0-1), the largest-
    // magnitude rows. A/B (2026-07-26): switching the backward's k/v/q read from bf16 to
    // f32 silu(preact) drops dq 1.74e-3→5.1e-4 and dconv 6.29e-3→3.2e-4, both under 1e-3
    // — confirming pure bf16 rounding. Tolerance covers the measured bf16 residual (dconv
    // 6.29e-3) with margin; the production bf16 gradient is the correct one to ship.
    let abs_tol: f32 = if carry.is_some() { 1.0e-2 } else { 1.0e-3 };
    let qkv_shape = [params.batch, params.seq_len, qkv_dim(params)];
    let z_shape = [params.batch, params.seq_len, z_dim(params)];
    let head_shape = [params.batch, params.seq_len, params.num_value_heads];
    let state_shape = [
        params.batch,
        params.num_value_heads,
        params.key_dim,
        params.value_dim,
    ];
    let conv_win_shape = [params.batch, params.conv_kernel - 1, qkv_dim(params)];

    let nvrtc_started = Instant::now();
    let cuda_backend = CudaBackend::new(0)
        .unwrap_or_else(|err| panic!("{label}: CUDA backend init failed: {err:?}"));
    eprintln!(
        "{label} stage=nvrtc_build seconds={:.3}",
        nvrtc_started.elapsed().as_secs_f64()
    );

    let cpu_started = Instant::now();
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
    let cpu_out = if let Some((state, conv)) = carry {
        let is = cpu_store.from_slice(state, &state_shape)?;
        let ic = cpu_store.from_slice(conv, &conv_win_shape)?;
        linear_attention_core_with_carry_taped(
            cpu_qkv,
            cpu_z,
            cpu_b,
            cpu_a,
            cpu_conv,
            cpu_dt,
            cpu_a_log,
            cpu_norm,
            params,
            Some(is),
            Some(ic),
            &mut cpu_store,
            &mut cpu_tape,
        )?
    } else {
        linear_attention_core(
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
        )?
    };
    let cpu_weighted = mul(cpu_out, cpu_coeff, &mut cpu_store, &mut cpu_tape)?;
    let cpu_loss = sum(cpu_weighted, &mut cpu_store, &mut cpu_tape)?;
    let cpu_grads = cpu_tape.backward(cpu_loss, &mut cpu_store)?;
    let cpu_out_host = cpu_store.to_host(cpu_out)?;
    let cpu_qkv_grad = cpu_store.to_host(*cpu_grads.get(&cpu_qkv).expect("cpu qkv grad"))?;
    let _cpu_z_grad = cpu_store.to_host(*cpu_grads.get(&cpu_z).expect("cpu z grad"))?;
    let cpu_b_grad = cpu_store.to_host(*cpu_grads.get(&cpu_b).expect("cpu b grad"))?;
    let cpu_a_grad = cpu_store.to_host(*cpu_grads.get(&cpu_a).expect("cpu a grad"))?;
    let cpu_conv_grad = cpu_store.to_host(*cpu_grads.get(&cpu_conv).expect("cpu conv grad"))?;
    let cpu_dt_grad = cpu_store.to_host(*cpu_grads.get(&cpu_dt).expect("cpu dt grad"))?;
    let cpu_a_log_grad = cpu_store.to_host(*cpu_grads.get(&cpu_a_log).expect("cpu a_log grad"))?;
    let cpu_norm_grad = cpu_store.to_host(*cpu_grads.get(&cpu_norm).expect("cpu norm grad"))?;
    eprintln!(
        "{label} stage=cpu_fwd_bwd seconds={:.3}",
        cpu_started.elapsed().as_secs_f64()
    );

    let cuda_started = Instant::now();
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
    cuda_store.backend().device_synchronize()?;
    eprintln!(
        "{label} stage=cuda_upload seconds={:.3}",
        cuda_started.elapsed().as_secs_f64()
    );
    for tensor_id in [
        cuda_qkv, cuda_z, cuda_b, cuda_a, cuda_conv, cuda_dt, cuda_a_log, cuda_norm,
    ] {
        cuda_store
            .get_mut(tensor_id)
            .expect("cuda tensor exists")
            .requires_grad = true;
    }
    let cuda_forward_started = Instant::now();
    let cuda_out = if let Some((state, conv)) = carry {
        let is = cuda_store.from_slice(state, &state_shape)?;
        let ic = cuda_store.from_slice(conv, &conv_win_shape)?;
        linear_attention_core_with_carry_taped(
            cuda_qkv,
            cuda_z,
            cuda_b,
            cuda_a,
            cuda_conv,
            cuda_dt,
            cuda_a_log,
            cuda_norm,
            params,
            Some(is),
            Some(ic),
            &mut cuda_store,
            &mut cuda_tape,
        )?
    } else {
        linear_attention_core(
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
        )?
    };
    cuda_store.backend().device_synchronize()?;
    eprintln!(
        "{label} stage=cuda_forward seconds={:.3}",
        cuda_forward_started.elapsed().as_secs_f64()
    );
    let cuda_loss_started = Instant::now();
    let cuda_weighted = mul(cuda_out, cuda_coeff, &mut cuda_store, &mut cuda_tape)?;
    let cuda_loss = sum(cuda_weighted, &mut cuda_store, &mut cuda_tape)?;
    cuda_store.backend().device_synchronize()?;
    eprintln!(
        "{label} stage=cuda_loss_graph seconds={:.3}",
        cuda_loss_started.elapsed().as_secs_f64()
    );
    let cuda_backward_started = Instant::now();
    let cuda_grads = cuda_tape.backward(cuda_loss, &mut cuda_store)?;
    cuda_store.backend().device_synchronize()?;
    eprintln!(
        "{label} stage=cuda_backward seconds={:.3}",
        cuda_backward_started.elapsed().as_secs_f64()
    );
    eprintln!(
        "{label} stage=cuda_fwd_bwd seconds={:.3}",
        cuda_started.elapsed().as_secs_f64()
    );
    for (name, id) in [
        ("qkv", cuda_qkv),
        ("z", cuda_z),
        ("b_proj", cuda_b),
        ("a_proj", cuda_a),
        ("conv1d_weight", cuda_conv),
        ("dt_bias", cuda_dt),
        ("a_log", cuda_a_log),
        ("norm_weight", cuda_norm),
    ] {
        let grad_id = *cuda_grads.get(&id).expect("cuda grad exists");
        let grad = cuda_store.get(grad_id).expect("cuda grad tensor exists");
        assert!(
            grad.dirty != Dirty::Host && grad.device_handle.is_some(),
            "{label} {name} grad fell back to host"
        );
    }

    let compare_started = Instant::now();
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

    let q_dim = params.num_key_heads * params.key_dim;
    let v_dim = params.num_value_heads * params.value_dim;
    let output_abs_err = max_abs_err(&cuda_out_host, &cpu_out_host);
    eprintln!("{label} out max_abs_err={output_abs_err:.6e}");
    assert!(
        output_abs_err.is_finite(),
        "{label} output error is non-finite"
    );

    let checks = [
        (
            "dq",
            max_rel_err_qkv_range(&cuda_qkv_grad, &cpu_qkv_grad, params, 0, q_dim),
            max_abs_err_qkv_range(&cuda_qkv_grad, &cpu_qkv_grad, params, 0, q_dim),
        ),
        (
            "dk",
            max_rel_err_qkv_range(&cuda_qkv_grad, &cpu_qkv_grad, params, q_dim, q_dim),
            max_abs_err_qkv_range(&cuda_qkv_grad, &cpu_qkv_grad, params, q_dim, q_dim),
        ),
        (
            "dv",
            max_rel_err_qkv_range(&cuda_qkv_grad, &cpu_qkv_grad, params, q_dim * 2, v_dim),
            max_abs_err_qkv_range(&cuda_qkv_grad, &cpu_qkv_grad, params, q_dim * 2, v_dim),
        ),
        (
            "dbeta",
            max_rel_err(&cuda_b_grad, &cpu_b_grad),
            max_abs_err(&cuda_b_grad, &cpu_b_grad),
        ),
        (
            "dg",
            max_rel_err(&cuda_a_grad, &cpu_a_grad),
            max_abs_err(&cuda_a_grad, &cpu_a_grad),
        ),
        (
            "da_log",
            max_rel_err(&cuda_a_log_grad, &cpu_a_log_grad),
            max_abs_err(&cuda_a_log_grad, &cpu_a_log_grad),
        ),
        (
            "dnorm",
            max_rel_err(&cuda_norm_grad, &cpu_norm_grad),
            max_abs_err(&cuda_norm_grad, &cpu_norm_grad),
        ),
        (
            "dconv",
            max_rel_err(&cuda_conv_grad, &cpu_conv_grad),
            max_abs_err(&cuda_conv_grad, &cpu_conv_grad),
        ),
    ];
    let mut first_failure = None;
    for (name, err, abs) in checks {
        eprintln!("{label} {name} max_rel_err={err:.6e} max_abs_err={abs:.6e}");
        assert!(err.is_finite(), "{label} {name} error is non-finite");
        if err > rel_tol && abs > abs_tol && first_failure.is_none() {
            first_failure = Some((name, err, abs));
        }
    }
    if let Some((name, err, abs)) = first_failure {
        panic!("{label} {name} max rel err {err} > {rel_tol} and max abs err {abs} > {abs_tol}");
    }
    let ddt_rel = max_rel_err(&cuda_dt_grad, &cpu_dt_grad);
    let ddt_abs = max_abs_err(&cuda_dt_grad, &cpu_dt_grad);
    eprintln!("{label} ddt_bias max_rel_err={ddt_rel:.6e} max_abs_err={ddt_abs:.6e}");
    assert!(
        ddt_rel.is_finite() && (ddt_rel <= rel_tol || ddt_abs <= abs_tol),
        "{label} ddt_bias max rel err {ddt_rel} > {rel_tol} and max abs err {ddt_abs} > {abs_tol}"
    );
    for (name, values) in [
        ("cuda_out", &cuda_out_host),
        ("cuda_dqkv", &cuda_qkv_grad),
        ("cuda_dz", &cuda_z_grad),
        ("cuda_dbeta", &cuda_b_grad),
        ("cuda_da", &cuda_a_grad),
        ("cuda_dconv", &cuda_conv_grad),
        ("cuda_ddt_bias", &cuda_dt_grad),
        ("cuda_dA_log", &cuda_a_log_grad),
        ("cuda_dnorm", &cuda_norm_grad),
    ] {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{label} {name} contains non-finite values"
        );
    }
    eprintln!(
        "{label} stage=analytic_compare seconds={:.3}",
        compare_started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[test]
fn linear_attention_grad_matches_numeric() -> Result<()> {
    let params = host_gradcheck_params();
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

// Frozen-prompt-KV carry gradcheck: seed a NONZERO initial_state + conv window,
// then finite-diff the params the carry backward corrects — a_log/dt_bias (M7:
// initial_state must enter the position-0 decay grad) and conv1d_weight (M8: the
// carried conv window must contribute to the weight grad). The non-carry
// gradcheck cannot catch these (its carry is zero, so the dropped terms are 0).
#[test]
fn linear_attention_carry_grad_matches_numeric() -> Result<()> {
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 12,
        num_key_heads: 1,
        num_value_heads: 2,
        key_dim: 8,
        value_dim: 8,
        conv_kernel: 4,
        eps: 1.0e-5,
    };
    let fixture = LinearAttentionFixture::new(params);
    let state_shape = [
        params.batch,
        params.num_value_heads,
        params.key_dim,
        params.value_dim,
    ];
    let conv_win_shape = [params.batch, params.conv_kernel - 1, qkv_dim(params)];
    let initial_state: Vec<f32> = (0..state_shape.iter().product::<usize>())
        .map(|i| ((i as f32 * 0.09).sin()) * 0.2)
        .collect();
    let initial_conv: Vec<f32> = (0..conv_win_shape.iter().product::<usize>())
        .map(|i| ((i as f32 * 0.15).cos()) * 0.18)
        .collect();

    let forward_loss =
        |a_proj: &[f32], conv1d_weight: &[f32], dt_bias: &[f32], a_log: &[f32]| -> f32 {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let qkv_shape = [params.batch, params.seq_len, qkv_dim(params)];
            let z_shape = [params.batch, params.seq_len, z_dim(params)];
            let head_shape = [params.batch, params.seq_len, params.num_value_heads];
            let qkv = store.from_slice(&fixture.qkv, &qkv_shape).expect("qkv");
            let z = store.from_slice(&fixture.z, &z_shape).expect("z");
            let b_proj = store
                .from_slice(&fixture.b_proj, &head_shape)
                .expect("b_proj");
            let a_proj = store.from_slice(a_proj, &head_shape).expect("a_proj");
            let conv = store
                .from_slice(conv1d_weight, &[qkv_dim(params), params.conv_kernel])
                .expect("conv");
            let dt = store
                .from_slice(dt_bias, &[params.num_value_heads])
                .expect("dt_bias");
            let a_log = store
                .from_slice(a_log, &[params.num_value_heads])
                .expect("a_log");
            let norm = store
                .from_slice(&fixture.norm_weight, &[params.value_dim])
                .expect("norm");
            let coeff = store.from_slice(&fixture.coeff, &z_shape).expect("coeff");
            let is = store
                .from_slice(&initial_state, &state_shape)
                .expect("state");
            let ic = store
                .from_slice(&initial_conv, &conv_win_shape)
                .expect("conv_win");
            let output = linear_attention_core_with_carry_taped(
                qkv,
                z,
                b_proj,
                a_proj,
                conv,
                dt,
                a_log,
                norm,
                params,
                Some(is),
                Some(ic),
                &mut store,
                &mut tape,
            )
            .expect("carry forward");
            let weighted = mul(output, coeff, &mut store, &mut tape).expect("mul");
            let loss = sum(weighted, &mut store, &mut tape).expect("sum");
            store.to_host(loss).expect("loss")[0]
        };

    let (analytic_a, analytic_conv, analytic_dt, analytic_a_log) = {
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
        let is = store.from_slice(&initial_state, &state_shape)?;
        let ic = store.from_slice(&initial_conv, &conv_win_shape)?;
        for id in [a_proj, conv1d_weight, dt_bias, a_log] {
            store.get_mut(id).expect("tensor exists").requires_grad = true;
        }
        let output = linear_attention_core_with_carry_taped(
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
            params,
            Some(is),
            Some(ic),
            &mut store,
            &mut tape,
        )?;
        let weighted = mul(output, coeff, &mut store, &mut tape)?;
        let loss = sum(weighted, &mut store, &mut tape)?;
        let grads = tape.backward(loss, &mut store)?;
        (
            store.to_host(*grads.get(&a_proj).expect("a_proj grad"))?,
            store.to_host(*grads.get(&conv1d_weight).expect("conv grad"))?,
            store.to_host(*grads.get(&dt_bias).expect("dt grad"))?,
            store.to_host(*grads.get(&a_log).expect("a_log grad"))?,
        )
    };

    let mut a_in = fixture.a_proj.clone();
    let numeric_a = num_grad(
        |v| forward_loss(v, &fixture.conv1d_weight, &fixture.dt_bias, &fixture.a_log),
        &mut a_in,
        1.0e-3,
    );
    let mut conv_in = fixture.conv1d_weight.clone();
    let numeric_conv = num_grad(
        |v| forward_loss(&fixture.a_proj, v, &fixture.dt_bias, &fixture.a_log),
        &mut conv_in,
        1.0e-3,
    );
    let mut dt_in = fixture.dt_bias.clone();
    let numeric_dt = num_grad(
        |v| forward_loss(&fixture.a_proj, &fixture.conv1d_weight, v, &fixture.a_log),
        &mut dt_in,
        1.0e-3,
    );
    let mut a_log_in = fixture.a_log.clone();
    let numeric_a_log = num_grad(
        |v| forward_loss(&fixture.a_proj, &fixture.conv1d_weight, &fixture.dt_bias, v),
        &mut a_log_in,
        1.0e-3,
    );

    let (a_idx, a_err) = max_err_with_index(&analytic_a, &numeric_a);
    let (conv_idx, conv_err) = max_err_with_index(&analytic_conv, &numeric_conv);
    let (dt_idx, dt_err) = max_err_with_index(&analytic_dt, &numeric_dt);
    let (a_log_idx, a_log_err) = max_err_with_index(&analytic_a_log, &numeric_a_log);

    assert!(a_err < 8.0e-3, "carry a_proj grad err {a_err} at {a_idx}");
    assert!(
        conv_err < 8.0e-3,
        "carry conv1d_weight grad err {conv_err} at {conv_idx}"
    );
    assert!(
        dt_err < 8.0e-3,
        "carry dt_bias grad err {dt_err} at {dt_idx}"
    );
    assert!(
        a_log_err < 8.0e-3,
        "carry a_log grad err {a_log_err} at {a_log_idx}"
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

    let cuda_backend = CudaBackend::new(0)
        .unwrap_or_else(|err| panic!("cuda_linear_attention_matches_cpu_with_device_inputs: CUDA backend init failed: {err:?}"));
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

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_qwen35_chunked_grad_matches_cpu() -> Result<()> {
    let params = qwen35_chunked_params(80);
    let fixture = LinearAttentionFixture::new(params);
    compare_cpu_cuda_device_linear_attention(
        params,
        &fixture,
        "cuda qwen35 chunked linear_attention",
        2.0e-2,
    )
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_boundary_matches_cpu() -> Result<()> {
    let params = qwen35_chunked_params(17);
    let fixture = LinearAttentionFixture::new(params);
    let (_, cpu_state) = la_full(&fixture, params)?;
    let mut store = TensorStore::with_backend(Arc::new(CudaBackend::new(0)?));
    let qd = qkv_dim(params);
    let hd = params.num_value_heads;
    let ids = [
        store.from_slice(&fixture.qkv, &[1, params.seq_len, qd])?,
        store.from_slice(&fixture.b_proj, &[1, params.seq_len, hd])?,
        store.from_slice(&fixture.a_proj, &[1, params.seq_len, hd])?,
        store.from_slice(&fixture.conv1d_weight, &[qd, params.conv_kernel])?,
        store.from_slice(&fixture.dt_bias, &[hd])?,
        store.from_slice(&fixture.a_log, &[hd])?,
    ];
    for id in ids {
        store.ensure_device(id)?;
    }
    let (state, conv) = linear_attention_boundary(
        ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], params, None, None, &mut store,
    )?;
    let expected_conv =
        &fixture.qkv[(params.seq_len - params.conv_kernel + 1) * qd..params.seq_len * qd];
    assert!(max_abs_err(&store.to_host(state)?, &cpu_state) <= 1.0e-3);
    assert!(max_abs_err(&store.to_host(conv)?, expected_conv) <= 1.0e-6);
    Ok(())
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_hard_forget_grad_matches_cpu_and_stays_finite() -> Result<()> {
    let params = qwen35_chunked_params(64);
    let fixture = LinearAttentionFixture::hard_forget(params);
    compare_cpu_cuda_device_linear_attention(
        params,
        &fixture,
        "cuda qwen35 hard-forget linear_attention",
        2.0e-2,
    )
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_chunk_parallel_grad_matches_cpu() -> Result<()> {
    // 200 tokens = 4 x 64-token chunks (last partial) on a small head count —
    // exercises the staged chunk-parallel backward end to end: parallel
    // per-chunk transfer operators (M_c, B_c), a multi-step reverse boundary
    // carry, and the parallel per-chunk grad replay, vs the CPU reference.
    // (`--la-backward-mono` re-routes to the legacy monolithic scan for
    // manual A/B on the same shapes.)
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 200,
        num_key_heads: 2,
        num_value_heads: 4,
        key_dim: 128,
        value_dim: 128,
        conv_kernel: 4,
        eps: 1.0e-5,
    };
    let fixture = LinearAttentionFixture::new(params);
    compare_cpu_cuda_device_linear_attention(
        params,
        &fixture,
        "cuda chunk-parallel linear_attention",
        2.0e-2,
    )
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_carry_grad_matches_cpu() -> Result<()> {
    // Tranche-2 device carry route: seeded carry → device chunked backward vs CPU host oracle.
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 200,
        num_key_heads: 2,
        num_value_heads: 4,
        key_dim: 128,
        value_dim: 128,
        conv_kernel: 4,
        eps: 1.0e-5,
    };
    let fixture = LinearAttentionFixture::new(params);
    let state_len = params.batch * params.num_value_heads * params.key_dim * params.value_dim;
    let conv_len = params.batch * (params.conv_kernel - 1) * qkv_dim(params);
    let initial_state: Vec<f32> = (0..state_len)
        .map(|i| (i as f32 * 0.09).sin() * 0.2)
        .collect();
    let initial_conv: Vec<f32> = (0..conv_len)
        .map(|i| (i as f32 * 0.15).cos() * 0.18)
        .collect();
    compare_cpu_cuda_device_linear_attention_carry(
        params,
        &fixture,
        "cuda carry linear_attention",
        2.0e-2,
        Some((&initial_state, &initial_conv)),
    )
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_qwen36_27b_chunked_grad_matches_cpu() -> Result<()> {
    // 48 value heads — exercises the device LA kernel after the head-count guard
    // was relaxed from the hardcoded 32. Falls back to the host path (still
    // CPU-matched) if the guard rejects, so a regression surfaces as a perf, not
    // correctness, signal — but the device path must match CPU when taken.
    let params = qwen36_27b_chunked_params(80);
    let fixture = LinearAttentionFixture::new(params);
    compare_cpu_cuda_device_linear_attention(
        params,
        &fixture,
        "cuda qwen36-27b chunked linear_attention",
        2.0e-2,
    )
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_batched_grad_matches_cpu() -> Result<()> {
    // batch=3 x 130 tokens (3 chunks, last partial) — exercises the per-row
    // batched device dispatch: input row slicing, forward/backward result
    // concatenation, and weight-grad accumulation across rows. Before the
    // batched path, batch>1 fell back to host compute and the harness's
    // device-residency asserts would fail.
    let mut params = qwen35_chunked_params(130);
    params.batch = 3;
    let fixture = LinearAttentionFixture::new(params);
    compare_cpu_cuda_device_linear_attention(
        params,
        &fixture,
        "cuda batched linear_attention",
        2.0e-2,
    )
}

// ── OPD frozen-prompt-KV de-risk: prompt-boundary state + conv carry ──────────
// Proves the ONLY genuinely-new math: carrying the gated-delta SSM state + the
// causal-conv window across a [0..k] prompt / [k..N] generated split reproduces
// the full-sequence suffix exactly. This is the license-or-kill gate — if the
// split is bit-equivalent (≤1e-6) to the monolithic run on the generated rows,
// the linear-attn wall is cleared and the rest of the feature is mechanical wiring.

/// Run the full sequence in one call (no carry) and return all `output` rows plus
/// the boundary `final_state`.
fn la_full(
    fixture: &LinearAttentionFixture,
    params: LinearAttentionParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut store = TensorStore::default();
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

    let (output, final_state, _conv_window) = linear_attention_core_with_carry(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        None,
        None,
        true,
        &mut store,
    )?;
    Ok((
        store.to_host(output)?,
        store.to_host(final_state.expect("captured final_state"))?,
    ))
}

/// Run a single segment with an optional carried `(initial_state, initial_conv_window)`,
/// capturing the boundary. `seg_params` carries `seq_len = b - a`; the slices are
/// the fixture's rows `[a..b]` along the sequence axis (batch == 1).
#[allow(clippy::type_complexity)]
fn la_segment(
    fixture: &LinearAttentionFixture,
    full_params: LinearAttentionParams,
    a: usize,
    b: usize,
    initial: Option<(&[f32], &[f32])>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    assert_eq!(full_params.batch, 1, "slice helper assumes batch == 1");
    let seg_len = b - a;
    let mut params = full_params;
    params.seq_len = seg_len;

    let qd = qkv_dim(full_params);
    let zd = z_dim(full_params);
    let hd = full_params.num_value_heads;

    let qkv_slice = &fixture.qkv[a * qd..b * qd];
    let z_slice = &fixture.z[a * zd..b * zd];
    let b_slice = &fixture.b_proj[a * hd..b * hd];
    let a_slice = &fixture.a_proj[a * hd..b * hd];

    let mut store = TensorStore::default();
    let qkv = store.from_slice(qkv_slice, &[1, seg_len, qd])?;
    let z = store.from_slice(z_slice, &[1, seg_len, zd])?;
    let b_proj = store.from_slice(b_slice, &[1, seg_len, hd])?;
    let a_proj = store.from_slice(a_slice, &[1, seg_len, hd])?;
    let conv1d_weight = store.from_slice(&fixture.conv1d_weight, &[qd, full_params.conv_kernel])?;
    let dt_bias = store.from_slice(&fixture.dt_bias, &[hd])?;
    let a_log = store.from_slice(&fixture.a_log, &[hd])?;
    let norm_weight = store.from_slice(&fixture.norm_weight, &[full_params.value_dim])?;

    let conv_window = full_params.conv_kernel - 1;
    let (initial_state, initial_conv_window) = match initial {
        Some((state, window)) => {
            let state_id = store.from_slice(
                state,
                &[
                    1,
                    full_params.num_value_heads,
                    full_params.key_dim,
                    full_params.value_dim,
                ],
            )?;
            let window_id = store.from_slice(window, &[1, conv_window, qd])?;
            (Some(state_id), Some(window_id))
        }
        None => (None, None),
    };

    let (output, final_state, conv_window_id) = linear_attention_core_with_carry(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        initial_state,
        initial_conv_window,
        true,
        &mut store,
    )?;
    Ok((
        store.to_host(output)?,
        store.to_host(final_state.expect("captured final_state"))?,
        store.to_host(conv_window_id.expect("captured conv_window"))?,
    ))
}

fn assert_split_carry_exact(params: LinearAttentionParams, k: usize, label: &str) -> Result<()> {
    let fixture = LinearAttentionFixture::new(params);
    let n = params.seq_len;
    let zd = z_dim(params);

    // BASELINE: full sequence in one shot.
    let (output_full, final_state_full) = la_full(&fixture, params)?;

    // SPLIT: prompt [0..k] with capture, then generated [k..N] seeded with the boundary.
    let (_prompt_out, state_k, window_k) = la_segment(&fixture, params, 0, k, None)?;
    let (gen_out, final_state_split, _gen_window) =
        la_segment(&fixture, params, k, n, Some((&state_k, &window_k)))?;

    // ASSERT: generated-segment outputs == full-sequence rows [k..N], bit-equivalent.
    let full_suffix = &output_full[k * zd..n * zd];
    let (idx, max_diff) = max_err_with_index(&gen_out, full_suffix);
    assert!(
        max_diff <= 1.0e-6,
        "{label}: split suffix output diverged at flat idx {idx} (row {}, ch {}): max_abs_diff={max_diff:.3e} > 1e-6",
        k + idx / zd,
        idx % zd,
    );

    // ASSERT: split's final_state == full final_state (Markov carry is exact).
    let (sidx, smax) = max_err_with_index(&final_state_split, &final_state_full);
    assert!(
        smax <= 1.0e-6,
        "{label}: split final_state diverged at idx {sidx}: max_abs_diff={smax:.3e} > 1e-6",
    );
    Ok(())
}

#[test]
fn linear_attention_prompt_boundary_carry_is_exact() -> Result<()> {
    // Small isolated shapes — no model / LoRA / checkpoint confounders, JUST the
    // gated-delta recurrence + causal conv. conv_kernel=4 forces a real 3-row
    // ring carry across the boundary (the off-by-one risk).
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 32,
        num_key_heads: 1,
        num_value_heads: 2,
        key_dim: 6,
        value_dim: 4,
        conv_kernel: 4,
        eps: 1.0e-5,
    };
    // Split mid-sequence (k=12), and stress the conv ring at boundaries near the
    // kernel width (k < conv_kernel and k just past it) plus a late split.
    assert_split_carry_exact(params, 12, "k=12")?;
    assert_split_carry_exact(params, 1, "k=1")?; // k < conv_kernel-1: short prompt
    assert_split_carry_exact(params, 3, "k=3")?; // k == conv_kernel-1
    assert_split_carry_exact(params, 4, "k=4")?; // k == conv_kernel
    assert_split_carry_exact(params, 31, "k=31")?; // single generated row
    Ok(())
}

#[test]
fn linear_attention_carry_none_matches_full_first_rows() -> Result<()> {
    // Sanity: a carry-less segment [0..k] reproduces the full run's first k rows
    // (the Option=None path is identical to today's absolute-position-0 behavior).
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 20,
        num_key_heads: 2,
        num_value_heads: 4,
        key_dim: 8,
        value_dim: 8,
        conv_kernel: 3,
        eps: 1.0e-5,
    };
    let fixture = LinearAttentionFixture::new(params);
    let (output_full, _) = la_full(&fixture, params)?;
    let k = 7;
    let (prompt_out, _, _) = la_segment(&fixture, params, 0, k, None)?;
    let zd = z_dim(params);
    let (idx, max_diff) = max_err_with_index(&prompt_out, &output_full[0..k * zd]);
    assert!(
        max_diff <= 1.0e-6,
        "carry-none prefix diverged at idx {idx}: max_abs_diff={max_diff:.3e}",
    );
    Ok(())
}

// Context-parallel correctness spec (all-to-all-to-head, per Megatron gated-delta-net):
// running the recurrence on a value-head SUBSET and concatenating over subsets must
// equal running on all heads. This is the math `linear_attention_core_cp` relies on
// after its all-to-all transport hands each rank the full sequence for 1/N of the
// heads — GPU-free, and it fails if the recurrence ever crossed value-heads (it must
// not: state, conv taps, a_log[h], dt_bias[h], beta[h], per-head rmsnorm are all
// head-local). Replaces the old carry-chain spec, which tested the rejected serial ring.
fn assert_head_split_exact(params: LinearAttentionParams, n_shards: usize, label: &str) {
    let fixture = LinearAttentionFixture::new(params);
    let (output_full, _final_full) = la_full(&fixture, params).expect("full");
    assert_eq!(
        params.num_value_heads % n_shards,
        0,
        "{label}: value heads must divide into equal shards"
    );
    assert_eq!(
        params.num_key_heads % n_shards,
        0,
        "{label}: key heads must divide into equal shards"
    );
    let vh_l = params.num_value_heads / n_shards;
    let vd = params.value_dim;
    let seq = params.seq_len;

    // For each shard, run the core on that shard's head slice and place its output
    // rows back into the value-head layout, then compare to the full run.
    let mut reconstructed = vec![0.0_f32; output_full.len()];
    for s in 0..n_shards {
        let (out, _st, _w) = la_head_slice(&fixture, params, s, n_shards).expect("head slice");
        // out is [seq, vh_l*vd]; scatter it into the [seq, num_value_heads*vd] full row.
        for t in 0..seq {
            let dst = t * params.num_value_heads * vd + s * vh_l * vd;
            let src = t * vh_l * vd;
            reconstructed[dst..dst + vh_l * vd].copy_from_slice(&out[src..src + vh_l * vd]);
        }
    }
    let (idx, max_diff) = max_err_with_index(&reconstructed, &output_full);
    assert!(
        max_diff <= 1.0e-6,
        "{label}: head-split reconstruction diverged at idx {idx}: max_abs_diff={max_diff:.3e}",
    );
}

// Run linear_attention_core on shard `s` of `n_shards` value-heads (and the matching
// key-head slice), returning that shard's [seq, vh_l*vd] output. Slices the packed
// qkv/z/b/a channels + conv/dt/a_log weights to the shard's head range — the CPU
// analog of what all-to-all-to-head delivers on device.
fn la_head_slice(
    fixture: &LinearAttentionFixture,
    full: LinearAttentionParams,
    s: usize,
    n_shards: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    assert_eq!(full.batch, 1, "head-slice helper assumes batch == 1");
    let (kh_l, vh_l) = (
        full.num_key_heads / n_shards,
        full.num_value_heads / n_shards,
    );
    let params = LinearAttentionParams {
        num_key_heads: kh_l,
        num_value_heads: vh_l,
        ..full
    };
    let (q_dim, k_dim) = (
        full.num_key_heads * full.key_dim,
        full.num_key_heads * full.key_dim,
    );
    let (kd, vd) = (full.key_dim, full.value_dim);
    let seq = full.seq_len;
    let qkv_full = qkv_dim(full);

    // Per-row gather of this shard's q|k|v channels out of the packed qkv row.
    let (q_off, k_off, v_off) = (
        s * kh_l * kd,
        q_dim + s * kh_l * kd,
        q_dim + k_dim + s * vh_l * vd,
    );
    let mut qkv = Vec::with_capacity(seq * (kh_l * kd * 2 + vh_l * vd));
    for t in 0..seq {
        let base = t * qkv_full;
        qkv.extend_from_slice(&fixture.qkv[base + q_off..base + q_off + kh_l * kd]);
        qkv.extend_from_slice(&fixture.qkv[base + k_off..base + k_off + kh_l * kd]);
        qkv.extend_from_slice(&fixture.qkv[base + v_off..base + v_off + vh_l * vd]);
    }
    let z = head_rows(
        &fixture.z,
        seq,
        full.num_value_heads * vd,
        s * vh_l * vd,
        vh_l * vd,
    );
    let b_proj = head_rows(&fixture.b_proj, seq, full.num_value_heads, s * vh_l, vh_l);
    let a_proj = head_rows(&fixture.a_proj, seq, full.num_value_heads, s * vh_l, vh_l);

    // Weights: conv1d packs [q|k|v] on the channel axis; dt_bias/a_log per value-head;
    // norm shared per value_dim.
    let mut conv = Vec::new();
    for &(base, w) in &[
        (0usize, kh_l * kd),
        (q_dim, kh_l * kd),
        (q_dim + k_dim, vh_l * vd),
    ] {
        let start = base + s * w;
        for ch in start..start + w {
            conv.extend_from_slice(
                &fixture.conv1d_weight[ch * full.conv_kernel..(ch + 1) * full.conv_kernel],
            );
        }
    }
    let dt_bias = fixture.dt_bias[s * vh_l..(s + 1) * vh_l].to_vec();
    let a_log = fixture.a_log[s * vh_l..(s + 1) * vh_l].to_vec();

    let mut store = TensorStore::default();
    let qkv = store.from_slice(&qkv, &[1, seq, kh_l * kd * 2 + vh_l * vd])?;
    let z = store.from_slice(&z, &[1, seq, vh_l * vd])?;
    let b_proj = store.from_slice(&b_proj, &[1, seq, vh_l])?;
    let a_proj = store.from_slice(&a_proj, &[1, seq, vh_l])?;
    let conv1d_weight = store.from_slice(&conv, &[kh_l * kd * 2 + vh_l * vd, full.conv_kernel])?;
    let dt_bias = store.from_slice(&dt_bias, &[vh_l])?;
    let a_log = store.from_slice(&a_log, &[vh_l])?;
    let norm_weight = store.from_slice(&fixture.norm_weight, &[vd])?;

    let (output, final_state, conv_window) = linear_attention_core_with_carry(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        None,
        None,
        true,
        &mut store,
    )?;
    Ok((
        store.to_host(output)?,
        store.to_host(final_state.expect("final_state"))?,
        store.to_host(conv_window.expect("conv_window"))?,
    ))
}

// Gather `width` channels at `offset` from each of `seq` rows of a `[seq, row]` tensor.
fn head_rows(data: &[f32], seq: usize, row: usize, offset: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq * width);
    for t in 0..seq {
        out.extend_from_slice(&data[t * row + offset..t * row + offset + width]);
    }
    out
}

#[test]
fn linear_attention_cp_head_split_reconstructs_full() {
    // key_heads == value_heads == 4 so cp=2 and cp=4 both divide BOTH counts — the
    // regime linear_attention_core_cp asserts. conv_kernel=4 exercises the packed
    // [q|k|v] conv-weight head-slice.
    let params = LinearAttentionParams {
        batch: 1,
        seq_len: 16,
        num_key_heads: 4,
        num_value_heads: 4,
        key_dim: 6,
        value_dim: 4,
        conv_kernel: 4,
        eps: 1.0e-5,
    };
    assert_head_split_exact(params, 2, "cp=2");
    assert_head_split_exact(params, 4, "cp=4");
    // GQA (value_heads > key_heads) at cp=2: key-heads still divide, ratio preserved.
    let gqa = LinearAttentionParams {
        num_key_heads: 2,
        num_value_heads: 4,
        ..params
    };
    assert_head_split_exact(gqa, 2, "cp=2 gqa 2:1");
}

// Sequence-parallel core: two chunks chained through the taped state + conv-window
// carry must match the unchunked device core, gradients included (dh0 / d_window
// flow back through `LinearAttentionFinalState` and the window slice).
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_linear_attention_seq_carry_grads_match_unchunked() -> Result<()> {
    use autograd::TensorId;
    use autograd::ops::{cat, linear_attention_core_carry, slice};
    let params = qwen36_27b_chunked_params(256);
    let fixture = LinearAttentionFixture::new(params);
    let qkv_shape = [params.batch, params.seq_len, qkv_dim(params)];
    let z_shape = [params.batch, params.seq_len, z_dim(params)];
    let head_shape = [params.batch, params.seq_len, params.num_value_heads];
    let mut run = |chunked: bool| -> Result<Vec<Vec<f32>>> {
        let mut store = TensorStore::with_backend(Arc::new(CudaBackend::new(0)?));
        let mut tape = Tape::new();
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
        let leaves = [
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
        ];
        for id in leaves {
            store.get_mut(id).expect("leaf").requires_grad = true;
        }
        let output = if chunked {
            let half = params.seq_len / 2;
            let chunk = LinearAttentionParams {
                seq_len: half,
                ..params
            };
            let mut carry = None;
            let mut outs = Vec::new();
            for c in 0..2 {
                let (r0, r1) = (c * half, (c + 1) * half);
                let mut rows = |x: TensorId, w: usize, st: &mut TensorStore, tp: &mut Tape| {
                    slice(x, &[0, r0, 0], &[params.batch, r1, w], st, tp)
                };
                let qkv_c = rows(qkv, qkv_dim(params), &mut store, &mut tape)?;
                let z_c = rows(z, z_dim(params), &mut store, &mut tape)?;
                let b_c = rows(b_proj, params.num_value_heads, &mut store, &mut tape)?;
                let a_c = rows(a_proj, params.num_value_heads, &mut store, &mut tape)?;
                let (state, window) = carry.map_or((None, None), |(s, w)| (Some(s), Some(w)));
                let (out, final_state, conv_window) = linear_attention_core_carry(
                    qkv_c,
                    z_c,
                    b_c,
                    a_c,
                    conv1d_weight,
                    dt_bias,
                    a_log,
                    norm_weight,
                    chunk,
                    state,
                    window,
                    None,
                    &mut store,
                    &mut tape,
                )?;
                carry = Some((final_state, conv_window));
                outs.push(out);
            }
            cat(&outs, 1, &mut store, &mut tape)?
        } else {
            linear_attention_core(
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
            )?
        };
        let weighted = mul(output, coeff, &mut store, &mut tape)?;
        let loss = sum(weighted, &mut store, &mut tape)?;
        let grads = tape.backward(loss, &mut store)?;
        let mut out = vec![store.to_host(loss)?];
        for id in leaves {
            out.push(store.to_host(*grads.get(&id).expect("leaf grad"))?);
        }
        Ok(out)
    };
    let full = run(false)?;
    let chunked = run(true)?;
    let names = [
        "loss", "qkv", "z", "b_proj", "a_proj", "conv1d", "dt_bias", "a_log", "norm",
    ];
    for ((name, lhs), rhs) in names.iter().zip(&full).zip(&chunked) {
        let (idx, err) = max_err_with_index(lhs, rhs);
        let scale = lhs.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        assert!(
            err / scale < 2.0e-2,
            "{name}: max abs err {err:.3e} at {idx} (scale {scale:.3e}) full={} chunked={}",
            lhs[idx],
            rhs[idx]
        );
    }
    Ok(())
}
