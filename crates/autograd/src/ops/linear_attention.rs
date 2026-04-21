use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionParams {
    pub batch: usize,
    pub seq_len: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

struct LinearAttentionForward {
    output: Vec<f32>,
    preact: Vec<f32>,
    beta: Vec<f32>,
    exp_g: Vec<f32>,
    kv_mem: Vec<f32>,
    final_state: Vec<f32>,
}

pub fn linear_attention_core(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_shapes(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
    )?;

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
        store.ensure_host(tensor_id)?;
    }

    let qkv_tensor = store.tensor(qkv)?.clone();
    let z_tensor = store.tensor(z)?.clone();
    let b_tensor = store.tensor(b_proj)?.clone();
    let a_tensor = store.tensor(a_proj)?.clone();
    let conv_tensor = store.tensor(conv1d_weight)?.clone();
    let dt_tensor = store.tensor(dt_bias)?.clone();
    let a_log_tensor = store.tensor(a_log)?.clone();
    let norm_tensor = store.tensor(norm_weight)?.clone();

    let forward = linear_attention_forward(
        &qkv_tensor.data,
        &z_tensor.data,
        &b_tensor.data,
        &a_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        &dt_tensor.data,
        &a_log_tensor.data,
        &norm_tensor.data,
        params,
    );

    let requires_grad = qkv_tensor.requires_grad
        || z_tensor.requires_grad
        || b_tensor.requires_grad
        || a_tensor.requires_grad;
    let output_shape = vec![
        params.batch,
        params.seq_len,
        params.num_value_heads * params.value_dim,
    ];
    let output_id = store.alloc(Tensor::new(forward.output, output_shape, requires_grad)?);

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::LinearAttention,
            output_id,
            input_ids: smallvec![qkv, z, b_proj, a_proj],
            saved: SavedContext::LinearAttentionCtx {
                qkv,
                z,
                b_proj,
                a_proj,
                conv1d_weight,
                dt_bias,
                a_log,
                norm_weight,
                batch: params.batch,
                seq_len: params.seq_len,
                num_key_heads: params.num_key_heads,
                num_value_heads: params.num_value_heads,
                key_dim: params.key_dim,
                value_dim: params.value_dim,
                conv_kernel: params.conv_kernel,
                eps: params.eps,
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn linear_attention_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::LinearAttentionCtx {
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        batch,
        seq_len,
        num_key_heads,
        num_value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        eps,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "linear attention backward missing saved context",
        ));
    };

    let params = LinearAttentionParams {
        batch,
        seq_len,
        num_key_heads,
        num_value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        eps,
    };
    validate_shapes(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
    )?;

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
        store.ensure_host(tensor_id)?;
    }

    let upstream = store.tensor(output_grad_id)?.clone();
    let qkv_tensor = store.tensor(qkv)?.clone();
    let z_tensor = store.tensor(z)?.clone();
    let b_tensor = store.tensor(b_proj)?.clone();
    let a_tensor = store.tensor(a_proj)?.clone();
    let conv_tensor = store.tensor(conv1d_weight)?.clone();
    let dt_tensor = store.tensor(dt_bias)?.clone();
    let a_log_tensor = store.tensor(a_log)?.clone();
    let norm_tensor = store.tensor(norm_weight)?.clone();

    let expected_shape = vec![batch, seq_len, num_value_heads * value_dim];
    if upstream.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream.shape,
        });
    }

    let forward = linear_attention_forward(
        &qkv_tensor.data,
        &z_tensor.data,
        &b_tensor.data,
        &a_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        &dt_tensor.data,
        &a_log_tensor.data,
        &norm_tensor.data,
        params,
    );

    let q_dim = num_key_heads * key_dim;
    let k_dim = q_dim;
    let v_offset = q_dim + k_dim;
    let mut dqkv = vec![0.0_f32; qkv_tensor.data.len()];
    let mut dz = vec![0.0_f32; z_tensor.data.len()];
    let mut db = vec![0.0_f32; b_tensor.data.len()];
    let mut da = vec![0.0_f32; a_tensor.data.len()];

    for batch_idx in 0..batch {
        for value_head in 0..num_value_heads {
            let key_head = value_head * num_key_heads / num_value_heads;
            let mut state = forward.final_state[state_base(
                batch_idx,
                value_head,
                num_value_heads,
                key_dim,
                value_dim,
            )
                ..state_base(batch_idx, value_head, num_value_heads, key_dim, value_dim)
                    + key_dim * value_dim]
                .to_vec();
            let mut grad_state = vec![0.0_f32; key_dim * value_dim];
            let exp_a = a_log_tensor.data[value_head].exp();

            for seq_idx in (0..seq_len).rev() {
                let preact_row = row3(
                    &forward.preact,
                    batch_idx,
                    seq_idx,
                    seq_len,
                    qkv_tensor.shape[2],
                );
                let q_raw = silu_slice(&preact_row[key_head * key_dim..(key_head + 1) * key_dim]);
                let k_raw = silu_slice(
                    &preact_row[q_dim + key_head * key_dim..q_dim + (key_head + 1) * key_dim],
                );
                let v_raw = silu_slice(
                    &preact_row[v_offset + value_head * value_dim
                        ..v_offset + (value_head + 1) * value_dim],
                );

                let q = l2_normalize_scaled(&q_raw, 1.0 / (key_dim as f32).sqrt());
                let k = l2_normalize_scaled(&k_raw, 1.0);
                let beta =
                    forward.beta[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
                let exp_g =
                    forward.exp_g[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
                let kv_mem = row4(
                    &forward.kv_mem,
                    batch_idx,
                    seq_idx,
                    value_head,
                    seq_len,
                    num_value_heads,
                    value_dim,
                );
                let delta = v_raw
                    .iter()
                    .zip(kv_mem.iter())
                    .map(|(&v_value, &kv_value)| (v_value - kv_value) * beta)
                    .collect::<Vec<_>>();

                let gate_row = row4(
                    &z_tensor.data,
                    batch_idx,
                    seq_idx,
                    value_head,
                    seq_len,
                    num_value_heads,
                    value_dim,
                );
                let upstream_row = row4(
                    &upstream.data,
                    batch_idx,
                    seq_idx,
                    value_head,
                    seq_len,
                    num_value_heads,
                    value_dim,
                );
                let core_out = mat_t_vec(&state, &q.values, key_dim, value_dim);
                let (normed, inv_rms) = rmsnorm_row(&core_out, &norm_tensor.data, eps);
                let gate_silu = silu_slice(gate_row);

                let mut dcore = vec![0.0_f32; value_dim];
                let mut dot_beta = 0.0_f32;
                for value_idx in 0..value_dim {
                    dcore[value_idx] = upstream_row[value_idx] * gate_silu[value_idx];
                    let gate_grad = upstream_row[value_idx] * normed[value_idx];
                    dz[idx4(
                        batch_idx,
                        seq_idx,
                        value_head,
                        value_idx,
                        seq_len,
                        num_value_heads,
                        value_dim,
                    )] += gate_grad * silu_grad_scalar(gate_row[value_idx]);
                    dot_beta +=
                        dcore[value_idx] * core_out[value_idx] * norm_tensor.data[value_idx];
                }
                dcore = rmsnorm_backward_row(
                    &core_out,
                    &norm_tensor.data,
                    &dcore,
                    inv_rms,
                    dot_beta,
                    value_dim,
                );

                let dq = mat_vec(&state, &dcore, key_dim, value_dim);
                add_outer_in_place(&mut grad_state, &q.values, &dcore, key_dim, value_dim);

                let mut s_decay = state.clone();
                subtract_outer_in_place(&mut s_decay, &k.values, &delta, key_dim, value_dim);

                let mut d_delta = vec![0.0_f32; value_dim];
                let mut dk = vec![0.0_f32; key_dim];
                for key_idx in 0..key_dim {
                    let mut accum = 0.0_f32;
                    for value_idx in 0..value_dim {
                        let grad_value = grad_state[key_idx * value_dim + value_idx];
                        accum += grad_value * delta[value_idx];
                        d_delta[value_idx] += grad_value * k.values[key_idx];
                    }
                    dk[key_idx] += accum;
                }

                let mut dkv_mem = vec![0.0_f32; value_dim];
                let v_minus_kv = v_raw
                    .iter()
                    .zip(kv_mem.iter())
                    .map(|(&v_value, &kv_value)| v_value - kv_value)
                    .collect::<Vec<_>>();
                let mut dbeta_scalar = 0.0_f32;
                for value_idx in 0..value_dim {
                    dbeta_scalar += d_delta[value_idx] * v_minus_kv[value_idx];
                    dkv_mem[value_idx] -= d_delta[value_idx] * beta;
                }

                for key_idx in 0..key_dim {
                    let mut accum = 0.0_f32;
                    for value_idx in 0..value_dim {
                        grad_state[key_idx * value_dim + value_idx] +=
                            k.values[key_idx] * dkv_mem[value_idx];
                        accum += s_decay[key_idx * value_dim + value_idx] * dkv_mem[value_idx];
                    }
                    dk[key_idx] += accum;
                }

                let mut dstate_prev = vec![0.0_f32; key_dim * value_dim];
                let mut dexp_g = 0.0_f32;
                for idx in 0..key_dim * value_dim {
                    if exp_g <= 0.0 {
                        continue;
                    }
                    dexp_g += (s_decay[idx] / exp_g) * grad_state[idx];
                    dstate_prev[idx] = grad_state[idx] * exp_g;
                }

                let dg = dexp_g * exp_g;
                let a_value =
                    a_tensor.data[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
                let softplus_grad = sigmoid_scalar(a_value + dt_tensor.data[value_head]);
                da[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)] +=
                    dg * (-exp_a * softplus_grad);
                db[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)] +=
                    dbeta_scalar * beta * (1.0 - beta);

                let dq_raw = l2_normalize_scaled_backward(
                    &q_raw,
                    &dq,
                    q.norm,
                    1.0 / (key_dim as f32).sqrt(),
                );
                let dk_raw = l2_normalize_scaled_backward(&k_raw, &dk, k.norm, 1.0);
                let dv_raw = d_delta
                    .iter()
                    .map(|&d_value| d_value * beta)
                    .collect::<Vec<_>>();

                for key_idx in 0..key_dim {
                    dqkv[idx3(
                        batch_idx,
                        seq_idx,
                        key_head * key_dim + key_idx,
                        seq_len,
                        qkv_tensor.shape[2],
                    )] += dq_raw[key_idx];
                    dqkv[idx3(
                        batch_idx,
                        seq_idx,
                        q_dim + key_head * key_dim + key_idx,
                        seq_len,
                        qkv_tensor.shape[2],
                    )] += dk_raw[key_idx];
                }
                for value_idx in 0..value_dim {
                    dqkv[idx3(
                        batch_idx,
                        seq_idx,
                        v_offset + value_head * value_dim + value_idx,
                        seq_len,
                        qkv_tensor.shape[2],
                    )] += dv_raw[value_idx];
                }

                state = if exp_g > 0.0 {
                    s_decay.iter().map(|value| value / exp_g).collect()
                } else {
                    vec![0.0; key_dim * value_dim]
                };
                grad_state = dstate_prev;
            }
        }
    }

    let dqkv = conv1d_backward_input_only(
        &dqkv,
        &forward.preact,
        &qkv_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        params,
    )?;

    let mut grads = GradPairs::new();
    if qkv_tensor.requires_grad {
        grads.push((
            qkv,
            store.alloc(Tensor::new(dqkv, qkv_tensor.shape.clone(), false)?),
        ));
    }
    if z_tensor.requires_grad {
        grads.push((
            z,
            store.alloc(Tensor::new(dz, z_tensor.shape.clone(), false)?),
        ));
    }
    if b_tensor.requires_grad {
        grads.push((
            b_proj,
            store.alloc(Tensor::new(db, b_tensor.shape.clone(), false)?),
        ));
    }
    if a_tensor.requires_grad {
        grads.push((
            a_proj,
            store.alloc(Tensor::new(da, a_tensor.shape.clone(), false)?),
        ));
    }
    Ok(grads)
}

fn validate_shapes(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    store: &TensorStore,
) -> Result<()> {
    let q_dim = params.num_key_heads * params.key_dim;
    let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
    let z_dim = params.num_value_heads * params.value_dim;
    let expected_rank3 = |tensor: TensorId, dim: usize| -> Result<()> {
        let shape = &store.tensor(tensor)?.shape;
        if shape != &vec![params.batch, params.seq_len, dim] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![params.batch, params.seq_len, dim],
                got: shape.clone(),
            });
        }
        Ok(())
    };
    expected_rank3(qkv, qkv_dim)?;
    expected_rank3(z, z_dim)?;
    expected_rank3(b_proj, params.num_value_heads)?;
    expected_rank3(a_proj, params.num_value_heads)?;

    let conv_shape = &store.tensor(conv1d_weight)?.shape;
    let conv_ok = matches!(
        conv_shape.as_slice(),
        [channels, kernel] if *channels == qkv_dim && *kernel == params.conv_kernel
    ) || matches!(
        conv_shape.as_slice(),
        [channels, kernel, one] if *channels == qkv_dim && *kernel == params.conv_kernel && *one == 1
    );
    if !conv_ok {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![qkv_dim, params.conv_kernel],
            got: conv_shape.clone(),
        });
    }

    for (tensor, expected) in [
        (dt_bias, vec![params.num_value_heads]),
        (a_log, vec![params.num_value_heads]),
        (norm_weight, vec![params.value_dim]),
    ] {
        let shape = &store.tensor(tensor)?.shape;
        if *shape != expected {
            return Err(AutogradError::ShapeMismatch {
                expected,
                got: shape.clone(),
            });
        }
    }
    Ok(())
}

fn linear_attention_forward(
    qkv: &[f32],
    z: &[f32],
    b_proj: &[f32],
    a_proj: &[f32],
    conv1d_weight: &[f32],
    conv1d_shape: &[usize],
    dt_bias: &[f32],
    a_log: &[f32],
    norm_weight: &[f32],
    params: LinearAttentionParams,
) -> LinearAttentionForward {
    let q_dim = params.num_key_heads * params.key_dim;
    let k_dim = q_dim;
    let v_dim = params.num_value_heads * params.value_dim;
    let qkv_dim = q_dim + k_dim + v_dim;
    let z_dim = params.num_value_heads * params.value_dim;
    let mut output = vec![0.0_f32; params.batch * params.seq_len * z_dim];
    let mut preact = vec![0.0_f32; params.batch * params.seq_len * qkv_dim];
    let mut beta = vec![0.0_f32; params.batch * params.seq_len * params.num_value_heads];
    let mut exp_g = vec![0.0_f32; params.batch * params.seq_len * params.num_value_heads];
    let mut kv_mem = vec![0.0_f32; params.batch * params.seq_len * z_dim];
    let mut final_state =
        vec![0.0_f32; params.batch * params.num_value_heads * params.key_dim * params.value_dim];

    for batch_idx in 0..params.batch {
        let mut state = vec![0.0_f32; params.num_value_heads * params.key_dim * params.value_dim];
        for seq_idx in 0..params.seq_len {
            for channel in 0..qkv_dim {
                let mut sum = 0.0_f32;
                for tap in 0..params.conv_kernel {
                    if seq_idx + tap + 1 < params.conv_kernel {
                        continue;
                    }
                    let src_seq = seq_idx + tap + 1 - params.conv_kernel;
                    let input_idx = idx3(batch_idx, src_seq, channel, params.seq_len, qkv_dim);
                    sum +=
                        qkv[input_idx] * conv_weight_at(conv1d_weight, conv1d_shape, channel, tap);
                }
                preact[idx3(batch_idx, seq_idx, channel, params.seq_len, qkv_dim)] = sum;
            }

            for value_head in 0..params.num_value_heads {
                let key_head = value_head * params.num_key_heads / params.num_value_heads;
                let q_raw = (0..params.key_dim)
                    .map(|offset| {
                        preact[idx3(
                            batch_idx,
                            seq_idx,
                            key_head * params.key_dim + offset,
                            params.seq_len,
                            qkv_dim,
                        )]
                    })
                    .map(silu_scalar)
                    .collect::<Vec<_>>();
                let k_raw = (0..params.key_dim)
                    .map(|offset| {
                        preact[idx3(
                            batch_idx,
                            seq_idx,
                            q_dim + key_head * params.key_dim + offset,
                            params.seq_len,
                            qkv_dim,
                        )]
                    })
                    .map(silu_scalar)
                    .collect::<Vec<_>>();
                let v_raw = (0..params.value_dim)
                    .map(|offset| {
                        preact[idx3(
                            batch_idx,
                            seq_idx,
                            q_dim + k_dim + value_head * params.value_dim + offset,
                            params.seq_len,
                            qkv_dim,
                        )]
                    })
                    .map(silu_scalar)
                    .collect::<Vec<_>>();
                let q = l2_normalize_scaled(&q_raw, 1.0 / (params.key_dim as f32).sqrt());
                let k = l2_normalize_scaled(&k_raw, 1.0);
                let beta_value = sigmoid_scalar(
                    b_proj[idx3(
                        batch_idx,
                        seq_idx,
                        value_head,
                        params.seq_len,
                        params.num_value_heads,
                    )],
                );
                beta[idx3(
                    batch_idx,
                    seq_idx,
                    value_head,
                    params.seq_len,
                    params.num_value_heads,
                )] = beta_value;
                let g = -a_log[value_head].exp()
                    * softplus_scalar(
                        a_proj[idx3(
                            batch_idx,
                            seq_idx,
                            value_head,
                            params.seq_len,
                            params.num_value_heads,
                        )] + dt_bias[value_head],
                    );
                let exp_g_value = g.exp();
                exp_g[idx3(
                    batch_idx,
                    seq_idx,
                    value_head,
                    params.seq_len,
                    params.num_value_heads,
                )] = exp_g_value;

                let base = state_base(
                    batch_idx,
                    value_head,
                    params.num_value_heads,
                    params.key_dim,
                    params.value_dim,
                );
                for key_idx in 0..params.key_dim {
                    for value_idx in 0..params.value_dim {
                        state[base + key_idx * params.value_dim + value_idx] *= exp_g_value;
                    }
                }

                let mut kv_row = vec![0.0_f32; params.value_dim];
                for value_idx in 0..params.value_dim {
                    let mut accum = 0.0_f32;
                    for key_idx in 0..params.key_dim {
                        accum += state[base + key_idx * params.value_dim + value_idx]
                            * k.values[key_idx];
                    }
                    kv_row[value_idx] = accum;
                    kv_mem[idx4(
                        batch_idx,
                        seq_idx,
                        value_head,
                        value_idx,
                        params.seq_len,
                        params.num_value_heads,
                        params.value_dim,
                    )] = accum;
                }

                let mut core_out = vec![0.0_f32; params.value_dim];
                for value_idx in 0..params.value_dim {
                    let delta = (v_raw[value_idx] - kv_row[value_idx]) * beta_value;
                    for key_idx in 0..params.key_dim {
                        state[base + key_idx * params.value_dim + value_idx] +=
                            delta * k.values[key_idx];
                    }
                    let mut accum = 0.0_f32;
                    for key_idx in 0..params.key_dim {
                        accum += state[base + key_idx * params.value_dim + value_idx]
                            * q.values[key_idx];
                    }
                    core_out[value_idx] = accum;
                }

                let (normed, _) = rmsnorm_row(&core_out, norm_weight, params.eps);
                for value_idx in 0..params.value_dim {
                    let gate = silu_scalar(
                        z[idx4(
                            batch_idx,
                            seq_idx,
                            value_head,
                            value_idx,
                            params.seq_len,
                            params.num_value_heads,
                            params.value_dim,
                        )],
                    );
                    output[idx4(
                        batch_idx,
                        seq_idx,
                        value_head,
                        value_idx,
                        params.seq_len,
                        params.num_value_heads,
                        params.value_dim,
                    )] = normed[value_idx] * gate;
                }
            }
        }

        let final_base = batch_idx * params.num_value_heads * params.key_dim * params.value_dim;
        final_state[final_base..final_base + state.len()].copy_from_slice(&state);
    }

    LinearAttentionForward {
        output,
        preact,
        beta,
        exp_g,
        kv_mem,
        final_state,
    }
}

fn conv1d_backward_input_only(
    grad_out: &[f32],
    preact: &[f32],
    input: &[f32],
    conv1d_weight: &[f32],
    conv1d_shape: &[usize],
    params: LinearAttentionParams,
) -> Result<Vec<f32>> {
    let q_dim = params.num_key_heads * params.key_dim;
    let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
    if input.len() != params.batch * params.seq_len * qkv_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![params.batch, params.seq_len, qkv_dim],
            got: vec![input.len()],
        });
    }

    let mut grad_input = vec![0.0_f32; input.len()];
    for batch_idx in 0..params.batch {
        for seq_idx in 0..params.seq_len {
            for channel in 0..qkv_dim {
                let preact_idx = idx3(batch_idx, seq_idx, channel, params.seq_len, qkv_dim);
                let dpre = grad_out[preact_idx] * silu_grad_scalar(preact[preact_idx]);
                for tap in 0..params.conv_kernel {
                    if seq_idx + tap + 1 < params.conv_kernel {
                        continue;
                    }
                    let src_seq = seq_idx + tap + 1 - params.conv_kernel;
                    let input_idx = idx3(batch_idx, src_seq, channel, params.seq_len, qkv_dim);
                    grad_input[input_idx] +=
                        dpre * conv_weight_at(conv1d_weight, conv1d_shape, channel, tap);
                }
            }
        }
    }
    Ok(grad_input)
}

struct NormalizedVec {
    values: Vec<f32>,
    norm: f32,
}

fn l2_normalize_scaled(input: &[f32], scale: f32) -> NormalizedVec {
    let norm = (input.iter().map(|value| value * value).sum::<f32>() + 1.0e-12_f32).sqrt();
    let values = input.iter().map(|value| scale * value / norm).collect();
    NormalizedVec { values, norm }
}

fn l2_normalize_scaled_backward(input: &[f32], grad: &[f32], norm: f32, scale: f32) -> Vec<f32> {
    let dot = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| x * g)
        .sum::<f32>();
    let norm_cubed = norm * norm * norm;
    input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| scale * (g / norm - x * dot / norm_cubed))
        .collect()
}

fn rmsnorm_row(input: &[f32], weight: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let inv_rms = 1.0
        / ((input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32) + eps)
            .sqrt();
    let output = input
        .iter()
        .zip(weight.iter())
        .map(|(&value, &w)| value * inv_rms * w)
        .collect();
    (output, inv_rms)
}

fn rmsnorm_backward_row(
    input: &[f32],
    weight: &[f32],
    grad: &[f32],
    inv_rms: f32,
    dot: f32,
    hidden: usize,
) -> Vec<f32> {
    let coeff = inv_rms * inv_rms * inv_rms / hidden as f32;
    input
        .iter()
        .zip(weight.iter())
        .zip(grad.iter())
        .map(|((&x, &w), &g)| g * w * inv_rms - x * coeff * dot)
        .collect()
}

fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

fn silu_grad_scalar(x: f32) -> f32 {
    let sig = sigmoid_scalar(x);
    sig * (1.0 + x * (1.0 - sig))
}

fn silu_slice(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&value| silu_scalar(value)).collect()
}

fn mat_t_vec(matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; cols];
    for row in 0..rows {
        let scalar = vector[row];
        for col in 0..cols {
            output[col] += matrix[row * cols + col] * scalar;
        }
    }
    output
}

fn mat_vec(matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows];
    for row in 0..rows {
        let mut accum = 0.0_f32;
        for col in 0..cols {
            accum += matrix[row * cols + col] * vector[col];
        }
        output[row] = accum;
    }
    output
}

fn add_outer_in_place(matrix: &mut [f32], left: &[f32], right: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            matrix[row * cols + col] += left[row] * right[col];
        }
    }
}

fn subtract_outer_in_place(
    matrix: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    cols: usize,
) {
    for row in 0..rows {
        for col in 0..cols {
            matrix[row * cols + col] -= left[row] * right[col];
        }
    }
}

fn conv_weight_at(conv1d_weight: &[f32], shape: &[usize], channel: usize, tap: usize) -> f32 {
    match shape {
        [_, kernel] => conv1d_weight[channel * kernel + tap],
        [_, kernel, one] if *one == 1 => conv1d_weight[channel * kernel * one + tap * one],
        _ => unreachable!("validated by shape check"),
    }
}

fn idx3(batch: usize, seq: usize, dim: usize, seq_len: usize, width: usize) -> usize {
    (batch * seq_len + seq) * width + dim
}

fn idx4(
    batch: usize,
    seq: usize,
    head: usize,
    dim: usize,
    seq_len: usize,
    heads: usize,
    width: usize,
) -> usize {
    (((batch * seq_len + seq) * heads + head) * width) + dim
}

fn row3(data: &[f32], batch: usize, seq: usize, seq_len: usize, width: usize) -> &[f32] {
    let base = idx3(batch, seq, 0, seq_len, width);
    &data[base..base + width]
}

fn row4(
    data: &[f32],
    batch: usize,
    seq: usize,
    head: usize,
    seq_len: usize,
    heads: usize,
    width: usize,
) -> &[f32] {
    let base = idx4(batch, seq, head, 0, seq_len, heads, width);
    &data[base..base + width]
}

fn state_base(batch: usize, head: usize, heads: usize, key_dim: usize, value_dim: usize) -> usize {
    ((batch * heads + head) * key_dim) * value_dim
}
