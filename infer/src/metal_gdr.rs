//! Gated Delta Rule (GDR) implementation for Metal/MLX backend.
//!
//! Implements the Qwen3.5 linear attention decode step using MLX high-level ops.
//! This is a correctness-first implementation — no custom Metal kernels.
//!
//! # GDR decode step (per layer, seq=1)
//!
//! 1. Project input to get q, k, v, z, beta, alpha
//! 2. Conv1d on [q,k,v] with causal kernel (size 4), SiLU activation
//! 3. RMS-normalize q and k
//! 4. Compute gate g = exp(-exp(A_log) * softplus(alpha + dt_bias))
//! 5. Compute beta = sigmoid(beta_raw)
//! 6. State update: S = diag(g) * S + beta * (v - S^T k) outer k  (delta rule)
//! 7. Output: o = S^T @ q
//! 8. Per-head RMSNorm on o, gated by silu(z)
//! 9. Output projection
//!
//! # Feature flag
//!
//! Gated behind `#[cfg(feature = "metal")]` — only compiled on Metal builds.

#[cfg(feature = "metal")]
use anyhow::{Context, Result};
#[cfg(feature = "metal")]
use mlx_rs::Array;

#[cfg(feature = "metal")]
use crate::metal_backend::WeightTensor;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Qwen3.5 linear attention layers on Metal.
///
/// Mirrors the relevant fields from `Config35` for the GDR computation.
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MetalGdrConfig {
    /// Number of key heads (GQA: fewer key heads than value heads). Qwen3.5-4B: 16.
    pub num_key_heads: usize,
    /// Per-head key dimension. Qwen3.5-4B: 128.
    pub key_dim: usize,
    /// Number of value heads. Qwen3.5-4B: 32.
    pub num_value_heads: usize,
    /// Per-head value dimension. Qwen3.5-4B: 128.
    pub value_dim: usize,
    /// Conv1d kernel size. Qwen3.5-4B: 4.
    pub conv_kernel: usize,
    /// Model hidden size. Qwen3.5-4B: 2560.
    pub hidden_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

#[cfg(feature = "metal")]
impl MetalGdrConfig {
    /// Total QKV projection output dimension: q_dim + k_dim + v_dim.
    pub fn qkv_dim(&self) -> usize {
        let q_dim = self.num_key_heads * self.key_dim;
        let k_dim = q_dim;
        let v_dim = self.num_value_heads * self.value_dim;
        q_dim + k_dim + v_dim
    }

    /// Output dimension of the z projection.
    pub fn z_dim(&self) -> usize {
        self.num_value_heads * self.value_dim
    }

    /// Q dimension (= K dimension).
    pub fn q_dim(&self) -> usize {
        self.num_key_heads * self.key_dim
    }
}

// ── Weight structure ─────────────────────────────────────────────────────────

/// Weight tensors for a single linear attention layer on Metal.
///
/// Weight naming follows the CUDA `LinearAttentionLayer` structure:
/// - `in_proj_qkv`: fused Q+K+V projection
/// - `in_proj_z`: output gate projection
/// - `in_proj_beta`: learning rate projection
/// - `in_proj_alpha`: decay projection
/// - `conv1d_weight`: depthwise causal conv1d kernel
/// - `dt_bias`, `a_log`: per-head bias and log-decay parameters
/// - `norm_weight`: per-head-dim RMSNorm weight
/// - `out_proj`: output projection
#[cfg(feature = "metal")]
pub struct MetalLinearAttnWeights {
    /// Fused QKV projection: [qkv_dim, hidden_size] (pre-transposed for Dense).
    pub in_proj_qkv: WeightTensor,
    /// Z (output gate) projection: [z_dim, hidden_size].
    pub in_proj_z: WeightTensor,
    /// Beta (learning rate) projection: [num_value_heads, hidden_size].
    pub in_proj_beta: WeightTensor,
    /// Alpha (decay) projection: [num_value_heads, hidden_size].
    pub in_proj_alpha: WeightTensor,
    /// Depthwise conv1d weight: [qkv_dim, kernel_size] f32.
    pub conv1d_weight: Array,
    /// Per-head dt bias: [num_value_heads] f32.
    pub dt_bias: Array,
    /// Per-head log-decay: [num_value_heads] f32.
    pub a_log: Array,
    /// RMSNorm weight (per head_dim, broadcast across heads): [value_dim] f32.
    pub norm_weight: Array,
    /// Output projection: [hidden_size, z_dim].
    pub out_proj: WeightTensor,
}

// ── Recurrent state ──────────────────────────────────────────────────────────

/// Per-request recurrent state for all linear attention layers on Metal.
///
/// Each linear attention layer maintains:
/// - Recurrent state matrix: [num_value_heads, key_dim, value_dim] f32
/// - Conv1d rolling buffer: [qkv_dim, conv_kernel - 1] f32
#[cfg(feature = "metal")]
pub struct MetalRecurrentState {
    /// Per-layer recurrent state: [num_value_heads, key_dim, val_dim] f32.
    pub states: Vec<Array>,
    /// Per-layer conv1d rolling buffer: [qkv_dim, conv_kernel - 1] f32.
    pub conv_states: Vec<Array>,
    /// Number of tokens processed so far.
    pub seq_len: usize,
}

#[cfg(feature = "metal")]
impl MetalRecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub fn new(num_linear_layers: usize, config: &MetalGdrConfig) -> Self {
        let mut states = Vec::with_capacity(num_linear_layers);
        let mut conv_states = Vec::with_capacity(num_linear_layers);

        for _ in 0..num_linear_layers {
            // Recurrent state: [num_value_heads, key_dim, val_dim] f32
            states.push(Array::from_slice(
                &vec![0.0f32; config.num_value_heads * config.key_dim * config.value_dim],
                &[
                    config.num_value_heads as i32,
                    config.key_dim as i32,
                    config.value_dim as i32,
                ],
            ));

            // Conv state: [qkv_dim, conv_kernel - 1] f32
            let conv_state_width = config.conv_kernel - 1;
            conv_states.push(Array::from_slice(
                &vec![0.0f32; config.qkv_dim() * conv_state_width],
                &[config.qkv_dim() as i32, conv_state_width as i32],
            ));
        }

        Self {
            states,
            conv_states,
            seq_len: 0,
        }
    }

    /// Reset all state to zeros for a new generation.
    pub fn reset(&mut self, config: &MetalGdrConfig) {
        let num_layers = self.states.len();
        *self = Self::new(num_layers, config);
    }
}

// ── Helper: linear projection ────────────────────────────────────────────────

/// `x @ weight.T` — dispatches to dense matmul or quantized matmul.
/// Same as `metal_backend::linear` but avoids cross-module visibility issues.
#[cfg(feature = "metal")]
#[inline]
fn linear(x: &Array, weight: &WeightTensor) -> Result<Array> {
    match weight {
        WeightTensor::Dense(w_t) => mlx_rs::ops::matmul(x, w_t).context("matmul"),
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => mlx_rs::ops::quantized_matmul(
            x,
            w,
            scales,
            biases,
            Some(true),
            Some(*group_size),
            Some(*bits),
        )
        .context("quantized_matmul"),
    }
}

// ── Helper: RMS normalize ────────────────────────────────────────────────────

/// RMS-normalize along the last axis, matching `mx.fast.rms_norm(x, None, eps)`.
#[cfg(feature = "metal")]
fn rms_normalize(x: &Array, eps: f32) -> Result<Array> {
    let last_dim = x
        .shape()
        .last()
        .copied()
        .context("rms_normalize missing last dimension")? as f32;
    let inv_dim = Array::from_slice(&[1.0f32 / last_dim], &[1]);
    let eps_arr = Array::from_slice(&[eps], &[1]);
    let sq = mlx_rs::ops::multiply(x, x).context("l2 sq")?;
    let sum_sq = sq.sum_axis(-1, true).context("l2 sum")?;
    let mean_sq = mlx_rs::ops::multiply(&sum_sq, &inv_dim).context("rms mean")?;
    let inv_norm = mlx_rs::ops::add(&mean_sq, &eps_arr)
        .context("rms add eps")?
        .sqrt()
        .context("rms sqrt")?
        .reciprocal()
        .context("rms reciprocal")?;
    mlx_rs::ops::multiply(x, &inv_norm).context("rms mul")
}

// ── Helper: softplus ─────────────────────────────────────────────────────────

/// softplus(x) = log(1 + exp(x)), numerically stable.
/// For large x (> 20), exp(x) overflows — return x directly (the CUDA kernel
/// uses the same threshold). Implements: where(x > 20, x, log1p(exp(x))).
#[cfg(feature = "metal")]
fn softplus(x: &Array) -> Result<Array> {
    let threshold = Array::from_slice(&[20.0f32], &[1]);
    let mask = x.gt(&threshold).context("softplus mask")?;
    let exp_x = x.exp().context("softplus exp")?;
    let log1p_exp = exp_x.log1p().context("softplus log1p")?;
    mlx_rs::ops::r#where(&mask, x, &log1p_exp).context("softplus where")
}

// ── Conv1d single-step ───────────────────────────────────────────────────────

/// Causal depthwise conv1d for a single timestep.
///
/// Updates the rolling buffer in-place and returns the convolved + SiLU output.
///
/// # Arguments
/// - `x`: [qkv_dim] current input (f32)
/// - `conv_state`: [qkv_dim, kernel-1] rolling buffer (f32), mutated in place
/// - `kernel`: [qkv_dim, kernel_size] conv weights (f32)
///
/// # Returns
/// - [qkv_dim] convolved output with SiLU activation (f32)
#[cfg(feature = "metal")]
fn conv1d_step(
    x: &Array,
    conv_state: &mut Array,
    kernel: &Array,
    qkv_dim: usize,
    kernel_size: usize,
) -> Result<Array> {
    // conv_state is [qkv_dim, kernel-1], x is [qkv_dim]
    // We need to form [qkv_dim, kernel_size] by concatenating [conv_state, x_col]
    let x_col = x
        .reshape(&[qkv_dim as i32, 1])
        .context("conv1d reshape x")?;

    // full_window = [conv_state | x_col] → [qkv_dim, kernel_size]
    let full_window =
        mlx_rs::ops::concatenate_axis(&[conv_state.clone(), x_col], 1).context("conv1d concat")?;

    // Element-wise multiply by kernel weights, sum along time axis
    let prod = mlx_rs::ops::multiply(&full_window, kernel).context("conv1d mul")?;
    let conv_out = prod.sum_axis(1, None).context("conv1d sum")?; // [qkv_dim]

    // Match CUDA: conv sum is truncated to bf16 before SiLU, and the SiLU output
    // is also stored as bf16 before the next stage consumes it.
    let conv_out = conv_out
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("conv1d sum to bf16")?;
    let activated = mlx_rs::nn::silu(&conv_out).context("conv1d silu")?;
    let activated = activated
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("conv1d silu to bf16")?
        .as_dtype(mlx_rs::Dtype::Float32)
        .context("conv1d silu back to f32")?;

    // Update conv_state: shift left, drop oldest column, append x
    // New state = full_window[:, 1:] = columns [1..kernel_size] of full_window
    let state_width = (kernel_size - 1) as i32;
    if state_width > 0 {
        use mlx_rs::ops::indexing::TryIndexOp;
        *conv_state = full_window
            .try_index((.., 1i32..kernel_size as i32))
            .context("conv1d state update")?;
    }

    Ok(activated)
}

// ── GDR decode step ──────────────────────────────────────────────────────────

/// Single-token GDR decode step using MLX ops.
///
/// Implements the full linear attention layer for one token:
/// project -> conv1d -> normalize -> gate compute -> state update -> output -> norm -> gate.
///
/// # Arguments
/// - `x`: [1, hidden_size] input hidden state
/// - `layer_weights`: projection weights + norms for this linear attention layer
/// - `state`: mutable recurrent state (updated in place)
/// - `layer_idx`: index into the linear attention layer list (0..num_linear_layers-1)
/// - `config`: GDR configuration
///
/// # Returns
/// - [1, hidden_size] output hidden state (after output projection)
#[cfg(feature = "metal")]
pub fn metal_gdr_decode_step(
    x: &Array,
    layer_weights: &MetalLinearAttnWeights,
    state: &mut MetalRecurrentState,
    layer_idx: usize,
    config: &MetalGdrConfig,
) -> Result<Array> {
    use mlx_rs::ops;

    let num_key_heads = config.num_key_heads;
    let key_dim = config.key_dim;
    let num_value_heads = config.num_value_heads;
    let value_dim = config.value_dim;
    let q_dim = num_key_heads * key_dim;
    let k_dim = q_dim;
    let v_dim = num_value_heads * value_dim;

    // ── 1. Projections ───────────────────────────────────────────────────
    // x: [1, hidden_size] → flat [hidden_size] for projections
    let x_flat = x
        .reshape(&[1, config.hidden_size as i32])
        .context("reshape x")?;

    // QKV projection: [1, qkv_dim]
    let qkv_raw = linear(&x_flat, &layer_weights.in_proj_qkv)
        .context("in_proj_qkv")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("in_proj_qkv to bf16")?;
    // Z projection (output gate): [1, z_dim]
    let z = linear(&x_flat, &layer_weights.in_proj_z)
        .context("in_proj_z")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("in_proj_z to bf16")?;
    // Beta projection (learning rate): [1, num_value_heads]
    let beta_raw = linear(&x_flat, &layer_weights.in_proj_beta)
        .context("in_proj_beta")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("in_proj_beta to bf16")?;
    // Alpha projection (decay): [1, num_value_heads]
    let alpha_raw = linear(&x_flat, &layer_weights.in_proj_alpha)
        .context("in_proj_alpha")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("in_proj_alpha to bf16")?;

    // Flatten to 1D for per-element ops
    let qkv_1d = qkv_raw
        .reshape(&[config.qkv_dim() as i32])
        .context("flatten qkv")?;

    // ── 2. Conv1d step ───────────────────────────────────────────────────
    // Cast qkv to f32 for conv1d (state is f32)
    let qkv_f32 = qkv_1d
        .as_dtype(mlx_rs::Dtype::Float32)
        .context("qkv to f32")?;

    let qkv_conv = conv1d_step(
        &qkv_f32,
        &mut state.conv_states[layer_idx],
        &layer_weights.conv1d_weight,
        config.qkv_dim(),
        config.conv_kernel,
    )
    .context("conv1d_step")?;

    // ── 3. Split QKV and RMS-normalize q, k ──────────────────────────────
    // qkv_conv: [qkv_dim] → split into q [q_dim], k [k_dim], v [v_dim]
    use mlx_rs::ops::indexing::TryIndexOp;
    let q_raw = qkv_conv.try_index(0i32..q_dim as i32).context("split q")?;
    let k_raw = qkv_conv
        .try_index(q_dim as i32..(q_dim + k_dim) as i32)
        .context("split k")?;
    let v_raw = qkv_conv
        .try_index((q_dim + k_dim) as i32..(q_dim + k_dim + v_dim) as i32)
        .context("split v")?;

    // MLX normalizes q/k per key head, not across the concatenated q_dim.
    let q_per_key_head = q_raw
        .reshape(&[num_key_heads as i32, key_dim as i32])
        .context("reshape q per key head")?;
    let k_per_key_head = k_raw
        .reshape(&[num_key_heads as i32, key_dim as i32])
        .context("reshape k per key head")?;
    let q_norm = rms_normalize(&q_per_key_head, 1e-6).context("rms norm q per head")?;
    let k_norm = rms_normalize(&k_per_key_head, 1e-6).context("rms norm k per head")?;

    // Match mlx_lm:
    //   q = (inv_scale**2) * rms_norm(q, None, 1e-6)
    //   k = inv_scale * rms_norm(k, None, 1e-6)
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    let q_scale = inv_scale * inv_scale;
    let q_scaled =
        ops::multiply(&q_norm, &Array::from_slice(&[q_scale], &[1])).context("scale q")?;
    let k_scaled =
        ops::multiply(&k_norm, &Array::from_slice(&[inv_scale], &[1])).context("scale k")?;

    // ── 4. Compute gate (g) and beta ─────────────────────────────────────
    // Flatten alpha and beta to [num_value_heads]
    let alpha_1d = alpha_raw
        .reshape(&[num_value_heads as i32])
        .context("flatten alpha")?
        .as_dtype(mlx_rs::Dtype::Float32)
        .context("alpha to f32")?;
    let beta_1d = beta_raw
        .reshape(&[num_value_heads as i32])
        .context("flatten beta")?
        .as_dtype(mlx_rs::Dtype::Float32)
        .context("beta to f32")?;

    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    // From CUDA kernel:
    //   float x = a_val + bias;
    //   float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
    //   float g = -expf(a_log) * softplus_x;
    //   s_exp_g = expf(g);
    let alpha_plus_bias = ops::add(&alpha_1d, &layer_weights.dt_bias).context("alpha + dt_bias")?;
    let sp = softplus(&alpha_plus_bias).context("softplus")?;
    let a_exp = layer_weights.a_log.exp().context("exp(A_log)")?;
    let neg_a_exp = a_exp.negative().context("neg exp(A_log)")?;
    let g_exponent = ops::multiply(&neg_a_exp, &sp).context("g exponent")?;
    let exp_g = g_exponent.exp().context("exp(g)")?; // per-head decay factor [num_value_heads]

    // beta = sigmoid(beta_raw)
    let beta = mlx_rs::nn::sigmoid(&beta_1d).context("sigmoid beta")?;

    // ── 5–6. State update (per value head) ───────────────────────────────
    // State layout: [num_value_heads, key_dim, val_dim] f32
    //
    // From CUDA kernel (two-pass algorithm):
    //   Pass 1: S[j, v] *= exp_g  (decay entire state)
    //           kv_mem[v] = sum_j S[j, v] * k[j]  (query state with k)
    //   Pass 2: delta[v] = (v[v] - kv_mem[v]) * beta
    //           S[j, v] += delta[v] * k[j]  (rank-1 update)
    //           out[v] = sum_j S[j, v] * q[j]  (query updated state with q)

    // Reshape q, k per GQA mapping: value heads share key heads
    // k_norm: [k_dim] = [num_key_heads * key_dim]
    // Expand k to [num_value_heads, key_dim] by repeating key heads
    let heads_per_key = num_value_heads / num_key_heads;
    // Repeat each key head `heads_per_key` times → [num_value_heads, key_dim]
    let k_expanded = if heads_per_key > 1 {
        // [num_key_heads, key_dim] → [num_key_heads, 1, key_dim] → [num_key_heads, heads_per_key, key_dim]
        let k_unsq = k_scaled.expand_dims(1).context("k expand_dims")?;
        let k_broadcast = mlx_rs::ops::broadcast_to(
            &k_unsq,
            &[num_key_heads as i32, heads_per_key as i32, key_dim as i32],
        )
        .context("k broadcast")?;
        k_broadcast
            .reshape(&[num_value_heads as i32, key_dim as i32])
            .context("k reshape expanded")?
    } else {
        k_scaled
    };

    // Same expansion for q
    let q_expanded = if heads_per_key > 1 {
        let q_unsq = q_scaled.expand_dims(1).context("q expand_dims")?;
        let q_broadcast = mlx_rs::ops::broadcast_to(
            &q_unsq,
            &[num_key_heads as i32, heads_per_key as i32, key_dim as i32],
        )
        .context("q broadcast")?;
        q_broadcast
            .reshape(&[num_value_heads as i32, key_dim as i32])
            .context("q reshape expanded")?
    } else {
        q_scaled
    };

    // v: [v_dim] → [num_value_heads, val_dim]
    let v_heads = v_raw
        .reshape(&[num_value_heads as i32, value_dim as i32])
        .context("reshape v")?;

    // Get current state: [num_value_heads, key_dim, val_dim]
    let s = &state.states[layer_idx];

    // Pass 1: Decay state
    // exp_g: [num_value_heads] → [num_value_heads, 1, 1] for broadcasting
    let exp_g_3d = exp_g
        .reshape(&[num_value_heads as i32, 1, 1])
        .context("reshape exp_g")?;
    let s_decayed = ops::multiply(s, &exp_g_3d).context("decay state")?;

    // kv_mem[h, v] = sum_j S_decayed[h, j, v] * k[h, j]
    // k_expanded: [num_value_heads, key_dim] → [num_value_heads, key_dim, 1]
    let k_3d = k_expanded
        .reshape(&[num_value_heads as i32, key_dim as i32, 1])
        .context("reshape k 3d")?;
    // S_decayed: [H, K, V], k_3d: [H, K, 1]
    // kv_mem = sum over K: S_decayed * k_3d → [H, V]
    let s_times_k = ops::multiply(&s_decayed, &k_3d).context("s * k")?;
    let kv_mem = s_times_k.sum_axis(1, None).context("kv_mem sum")?; // [H, V]

    // delta = (v - kv_mem) * beta
    let v_minus_kv = ops::subtract(&v_heads, &kv_mem).context("v - kv_mem")?;
    // beta: [H] → [H, 1]
    let beta_2d = beta
        .reshape(&[num_value_heads as i32, 1])
        .context("reshape beta")?;
    let delta = ops::multiply(&v_minus_kv, &beta_2d).context("delta")?; // [H, V]

    // Pass 2: Rank-1 update: S[h, j, v] += delta[h, v] * k[h, j]
    // delta: [H, V] → [H, 1, V], k_expanded: [H, K] → [H, K, 1]
    // outer product: [H, K, V]
    let delta_3d = delta
        .reshape(&[num_value_heads as i32, 1, value_dim as i32])
        .context("reshape delta 3d")?;
    let update = ops::multiply(&delta_3d, &k_3d).context("rank1 update")?; // [H, K, V]
    let s_updated = ops::add(&s_decayed, &update).context("state + update")?;

    // Store updated state
    state.states[layer_idx] = s_updated.clone();

    // Output: o[h, v] = sum_j S_updated[h, j, v] * q[h, j]
    // q_expanded: [H, K] → [H, K, 1]
    let q_3d = q_expanded
        .reshape(&[num_value_heads as i32, key_dim as i32, 1])
        .context("reshape q 3d")?;
    let s_times_q = ops::multiply(&s_updated, &q_3d).context("s * q")?;
    let output_heads = s_times_q.sum_axis(1, None).context("output sum")?; // [H, V]

    // ── 7. Per-head RMSNorm + output gate ────────────────────────────────
    // output_heads: [num_value_heads, val_dim]
    // norm_weight: [val_dim] (broadcast across heads)
    // z: [1, z_dim] where z_dim = num_value_heads * val_dim

    // RMSNorm per head: for each head h, normalize output_heads[h, :] and scale by norm_weight
    // mlx_rs::fast::rms_norm operates on the last axis, so with shape [H, V] it norms over V.
    let output_heads_bf16 = output_heads
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("gdr output to bf16")?;
    let normed = mlx_rs::fast::rms_norm(
        &output_heads_bf16,
        &layer_weights.norm_weight,
        config.rms_norm_eps,
    )
    .context("gdr rms_norm")?; // [H, V]

    // Flatten: [H, V] → [1, z_dim]
    let normed_flat = normed
        .reshape(&[1, (num_value_heads * value_dim) as i32])
        .context("flatten normed")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("normed to bf16")?;

    // Output gate: o = normed * silu(z)
    let z_f32 = z.as_dtype(mlx_rs::Dtype::Float32).context("z to f32")?;
    let z_silu = mlx_rs::nn::silu(&z_f32).context("silu z")?;
    let gated = ops::multiply(&normed_flat.as_dtype(mlx_rs::Dtype::Float32)?, &z_silu)
        .context("output gate")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("output gate to bf16")?;

    // ── 8. Output projection ─────────────────────────────────────────────
    let output = linear(&gated, &layer_weights.out_proj)
        .context("out_proj")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("out_proj to bf16")?;

    Ok(output) // [1, hidden_size]
}

#[cfg(test)]
#[cfg(feature = "metal")]
mod tests {
    use super::*;
    use crate::test_support::metal_test_guard;

    /// Smoke test: verify shapes through the conv1d step.
    #[test]
    fn test_conv1d_step_shapes() {
        let _guard = metal_test_guard();
        let qkv_dim = 8192; // q=2048 + k=2048 + v=4096
        let kernel_size = 4;
        let state_width = kernel_size - 1;

        let x = Array::from_slice(&vec![1.0f32; qkv_dim], &[qkv_dim as i32]);
        let mut conv_state = Array::from_slice(
            &vec![0.0f32; qkv_dim * state_width],
            &[qkv_dim as i32, state_width as i32],
        );
        let kernel = Array::from_slice(
            &vec![0.25f32; qkv_dim * kernel_size],
            &[qkv_dim as i32, kernel_size as i32],
        );

        let out = conv1d_step(&x, &mut conv_state, &kernel, qkv_dim, kernel_size).unwrap();

        assert_eq!(out.shape(), &[qkv_dim as i32]);
    }

    /// Smoke test: verify state allocation shapes for small dimensions.
    #[test]
    fn test_gdr_state_shapes_small() {
        let _guard = metal_test_guard();
        // Tiny config: 1 key head, 1 value head, dim=2
        let config = MetalGdrConfig {
            num_key_heads: 1,
            key_dim: 2,
            num_value_heads: 1,
            value_dim: 2,
            conv_kernel: 4,
            hidden_size: 4,
            rms_norm_eps: 1e-6,
        };

        // q_dim = 1*2 = 2, k_dim = 2, v_dim = 1*2 = 2, qkv_dim = 6
        assert_eq!(config.qkv_dim(), 6);
        assert_eq!(config.z_dim(), 2);

        let state = MetalRecurrentState::new(1, &config);
        assert_eq!(state.states[0].shape(), &[1, 2, 2]);
        assert_eq!(state.conv_states[0].shape(), &[6, 3]); // [qkv_dim, kernel-1]
    }

    /// Smoke test: verify state shapes for Qwen3.5-4B dimensions.
    #[test]
    fn test_gdr_state_shapes_qwen35() {
        let _guard = metal_test_guard();
        let config = MetalGdrConfig {
            num_key_heads: 16,
            key_dim: 128,
            num_value_heads: 32,
            value_dim: 128,
            conv_kernel: 4,
            hidden_size: 2560,
            rms_norm_eps: 1e-6,
        };

        // q_dim = 16*128 = 2048, k_dim = 2048, v_dim = 32*128 = 4096, qkv = 8192
        assert_eq!(config.qkv_dim(), 8192);
        assert_eq!(config.z_dim(), 4096);

        let state = MetalRecurrentState::new(24, &config);
        assert_eq!(state.states.len(), 24);
        assert_eq!(state.conv_states.len(), 24);
        assert_eq!(state.states[0].shape(), &[32, 128, 128]);
        assert_eq!(state.conv_states[0].shape(), &[8192, 3]);
    }

    // ── Numerical tests ─────────────────────────────────────────────────────

    /// Reference softplus matching the CUDA kernel: x > 20 ? x : log(1 + exp(x))
    fn ref_softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() }
    }

    /// softplus must match the CUDA reference for small, moderate, and large values.
    #[test]
    fn test_softplus_numerical_accuracy() {
        let _guard = metal_test_guard();
        let values: Vec<f32> = vec![
            -10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 19.0, 20.0, 21.0, 50.0, 100.0,
        ];
        let input = Array::from_slice(&values, &[values.len() as i32]);
        let result = softplus(&input).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();

        for (i, &x) in values.iter().enumerate() {
            let expected = ref_softplus(x);
            let actual = result.as_slice::<f32>()[i];
            let tol = expected.abs() * 1e-5 + 1e-6;
            assert!(
                (actual - expected).abs() < tol,
                "softplus({x}): expected {expected}, got {actual}"
            );
        }
    }

    /// softplus must not overflow for large inputs (the original bug).
    #[test]
    fn test_softplus_no_overflow() {
        let _guard = metal_test_guard();
        let large = Array::from_slice(&[88.0f32, 200.0, 1000.0], &[3]);
        let result = softplus(&large).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        let vals = result.as_slice::<f32>();
        for &v in vals {
            assert!(v.is_finite(), "softplus overflow: got {v}");
        }
        // For x >> 20, softplus(x) ≈ x
        assert!((vals[0] - 88.0).abs() < 0.01);
        assert!((vals[1] - 200.0).abs() < 0.01);
    }

    /// RMS normalize should produce unit-norm output (up to epsilon).
    #[test]
    fn test_rms_normalize_unit_norm() {
        let _guard = metal_test_guard();
        let x = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let normed = rms_normalize(&x, 1e-6).unwrap();
        mlx_rs::transforms::eval([&normed]).unwrap();

        let vals = normed.as_slice::<f32>();
        // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        // normed = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        let rms = (3.0f32.powi(2) + 4.0f32.powi(2)) / 2.0;
        let rms = rms.sqrt();
        assert!((vals[0] - 3.0 / rms).abs() < 1e-4);
        assert!((vals[1] - 4.0 / rms).abs() < 1e-4);
    }

    /// Gate computation: g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    /// Verify against scalar reference for a single head.
    #[test]
    fn test_gate_computation_scalar() {
        let _guard = metal_test_guard();
        use mlx_rs::ops;

        let a_log_val = -1.0f32; // exp(A_log) = exp(-1) ≈ 0.3679
        let alpha_val = 2.0f32;
        let dt_bias_val = 1.0f32;
        // softplus(alpha + dt_bias) = softplus(3.0) = log(1 + exp(3)) ≈ 3.0486
        // g_exp = -0.3679 * 3.0486 ≈ -1.1216
        // exp(g) ≈ exp(-1.1216) ≈ 0.3258

        let alpha = Array::from_slice(&[alpha_val], &[1]);
        let dt_bias = Array::from_slice(&[dt_bias_val], &[1]);
        let a_log = Array::from_slice(&[a_log_val], &[1]);

        let alpha_plus_bias = ops::add(&alpha, &dt_bias).unwrap();
        let sp = softplus(&alpha_plus_bias).unwrap();
        let a_exp = a_log.exp().unwrap();
        let neg_a_exp = a_exp.negative().unwrap();
        let g_exponent = ops::multiply(&neg_a_exp, &sp).unwrap();
        let exp_g = g_exponent.exp().unwrap();
        mlx_rs::transforms::eval([&exp_g]).unwrap();

        let expected_sp = ref_softplus(alpha_val + dt_bias_val);
        let expected_g = (-a_log_val.exp() * expected_sp).exp();
        let actual = exp_g.as_slice::<f32>()[0];
        assert!(
            (actual - expected_g).abs() < 1e-4,
            "gate: expected {expected_g}, got {actual}"
        );
    }
}
