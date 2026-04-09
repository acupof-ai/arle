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
use crate::mlx::MlxArray;

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
    /// Merged QKV+Z projection: [qkv_dim + z_dim, hidden_size].
    /// Single matmul replaces two separate projections.
    pub in_proj_qkvz: WeightTensor,
    /// Merged Beta+Alpha projection: [num_value_heads * 2, hidden_size].
    pub in_proj_ba: WeightTensor,
    /// Split dimensions for QKVZ output: qkv_dim, z_dim.
    pub qkvz_split: (i32, i32),
    /// Number of value heads (for BA split).
    pub ba_num_heads: i32,
    /// Depthwise conv1d weight: [qkv_dim, kernel_size] f32.
    pub conv1d_weight: MlxArray,
    /// Per-head dt bias: [num_value_heads] f32.
    pub dt_bias: MlxArray,
    /// Per-head log-decay: [num_value_heads] f32.
    pub a_log: MlxArray,
    /// RMSNorm weight (per head_dim, broadcast across heads): [value_dim] f32.
    pub norm_weight: MlxArray,
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
    pub states: Vec<MlxArray>,
    /// Per-layer conv1d rolling buffer: [qkv_dim, conv_kernel - 1] f32.
    pub conv_states: Vec<MlxArray>,
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
            states.push(MlxArray::from_slice_f32(
                &vec![0.0f32; config.num_value_heads * config.key_dim * config.value_dim],
                &[
                    config.num_value_heads as i32,
                    config.key_dim as i32,
                    config.value_dim as i32,
                ],
            ));

            // Conv state: [qkv_dim, conv_kernel - 1] f32
            let conv_state_width = config.conv_kernel - 1;
            conv_states.push(MlxArray::from_slice_f32(
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
fn linear(x: &MlxArray, weight: &WeightTensor) -> MlxArray {
    use crate::mlx::{matmul, quantized_matmul};
    match weight {
        WeightTensor::Dense(w_t) => matmul(x, w_t),
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => quantized_matmul(x, w, scales, biases, true, *group_size, *bits),
    }
}

// ── Helper: RMS normalize ────────────────────────────────────────────────────

/// RMS-normalize along the last axis, matching `mx.fast.rms_norm(x, None, eps)`.
#[cfg(feature = "metal")]
/// RMS-normalize along the last axis without learnable weight.
/// Uses MLX's fused fast.rms_norm — single op instead of 7 manual ops.
fn rms_normalize(x: &MlxArray, eps: f32) -> MlxArray {
    crate::mlx::rms_norm_no_weight(x, eps)
}

// ── Helper: softplus ─────────────────────────────────────────────────────────

/// softplus(x) = log(1 + exp(x)), numerically stable.
/// For large x (> 20), exp(x) overflows — return x directly (the CUDA kernel
/// uses the same threshold). Implements: where(x > 20, x, log1p(exp(x))).
#[cfg(feature = "metal")]
fn softplus(x: &MlxArray) -> MlxArray {
    use crate::mlx::{exp, greater, log1p, where_};

    let threshold = MlxArray::from_slice_f32(&[20.0f32], &[1]);
    let mask = greater(x, &threshold);
    let exp_x = exp(x);
    let log1p_exp = log1p(&exp_x);
    where_(&mask, x, &log1p_exp)
}

// ── Compiled compute_g ──────────────────────────────────────────────────────
//
// g = exp(-exp(A_log) * softplus(alpha + dt_bias))
// Compiled with shapeless=true matching mlx_lm's pattern.

#[cfg(feature = "metal")]
static COMPILED_COMPUTE_G: std::sync::LazyLock<crate::mlx::CompiledFn> =
    std::sync::LazyLock::new(|| crate::mlx::compile_simple(compute_g_callback, true));

/// C callback for compute_g: inputs = [A_log, alpha, dt_bias], output = [g].
#[cfg(feature = "metal")]
unsafe extern "C" fn compute_g_callback(
    res: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    use crate::mlx::{add, exp, multiply, negative};

    // Extract inputs.
    let mut a_log_h = mlx_sys::mlx_array_new();
    let mut alpha_h = mlx_sys::mlx_array_new();
    let mut dt_bias_h = mlx_sys::mlx_array_new();
    mlx_sys::mlx_vector_array_get(&mut a_log_h, inputs, 0);
    mlx_sys::mlx_vector_array_get(&mut alpha_h, inputs, 1);
    mlx_sys::mlx_vector_array_get(&mut dt_bias_h, inputs, 2);

    let a_log = MlxArray::from_raw(a_log_h);
    let alpha = MlxArray::from_raw(alpha_h);
    let dt_bias = MlxArray::from_raw(dt_bias_h);

    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    let alpha_plus_bias = add(&alpha, &dt_bias);
    let sp = softplus(&alpha_plus_bias);
    let neg_a_exp = negative(&exp(&a_log));
    let g = exp(&multiply(&neg_a_exp, &sp));

    // Set output.
    let out = mlx_sys::mlx_vector_array_new_data(&g.as_raw(), 1);
    mlx_sys::mlx_vector_array_set(res, out);
    mlx_sys::mlx_vector_array_free(out);

    // Don't free inputs — caller owns them. But we took ownership via from_raw,
    // so forget them to avoid double-free.
    std::mem::forget(a_log);
    std::mem::forget(alpha);
    std::mem::forget(dt_bias);

    0
}

/// Compute g using the compiled closure.
#[cfg(feature = "metal")]
fn compiled_compute_g(a_log: &MlxArray, alpha: &MlxArray, dt_bias: &MlxArray) -> MlxArray {
    let results = COMPILED_COMPUTE_G.call(&[a_log, alpha, dt_bias]);
    results
        .into_iter()
        .next()
        .expect("compute_g should return 1 array")
}

// ── Compiled GDR state update ───────────────────────────────────────────────
//
// Compiled version of the delta-rule state update matching mlx_lm's
// @mx.compile on _gated_delta_step_ops.
// inputs = [q, k, v, g, beta, state]  (all properly shaped)
// outputs = [y, new_state]

#[cfg(feature = "metal")]
static COMPILED_GDR_STEP: std::sync::LazyLock<crate::mlx::CompiledFn> =
    std::sync::LazyLock::new(|| crate::mlx::compile_simple(gdr_step_callback, false));

/// C callback for the GDR state update step.
/// inputs[0] = q [H, K], inputs[1] = k [H, K], inputs[2] = v [H, V],
/// inputs[3] = g [H], inputs[4] = beta [H], inputs[5] = state [H, K, V]
/// outputs[0] = y [H, V], outputs[1] = new_state [H, K, V]
#[cfg(feature = "metal")]
unsafe extern "C" fn gdr_step_callback(
    res: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    use crate::mlx::{add, multiply, reshape, subtract, sum_axis};

    // Extract 6 inputs.
    let get = |i: usize| -> MlxArray {
        let mut h = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut h, inputs, i);
        MlxArray::from_raw(h)
    };
    let q = get(0); // [H, K]
    let k = get(1); // [H, K]
    let v = get(2); // [H, V]
    let g = get(3); // [H]
    let beta = get(4); // [H]
    let state = get(5); // [H, K, V]

    let h_dim = state.shape()[0];
    let k_dim = state.shape()[1];
    let v_dim = state.shape()[2];

    // Decay: state * g[:, None, None]
    let g_3d = reshape(&g, &[h_dim, 1, 1]);
    let s_decayed = multiply(&state, &g_3d);

    // kv_mem = sum_k(S_decayed * k[:, :, None])
    let k_3d = reshape(&k, &[h_dim, k_dim, 1]);
    let kv_mem = sum_axis(&multiply(&s_decayed, &k_3d), 1, false); // [H, V]

    // delta = (v - kv_mem) * beta[:, None]
    let beta_2d = reshape(&beta, &[h_dim, 1]);
    let delta = multiply(&subtract(&v, &kv_mem), &beta_2d); // [H, V]

    // Rank-1 update: S += delta[:, None, :] * k[:, :, None]
    let delta_3d = reshape(&delta, &[h_dim, 1, v_dim]);
    let s_updated = add(&s_decayed, &multiply(&delta_3d, &k_3d)); // [H, K, V]

    // Output: y = sum_k(S_updated * q[:, :, None])
    let q_3d = reshape(&q, &[h_dim, k_dim, 1]);
    let y = sum_axis(&multiply(&s_updated, &q_3d), 1, false); // [H, V]

    // Output: [y, s_updated]
    let out_arrays = [y.as_raw(), s_updated.as_raw()];
    let out = mlx_sys::mlx_vector_array_new_data(out_arrays.as_ptr(), 2);
    mlx_sys::mlx_vector_array_set(res, out);
    mlx_sys::mlx_vector_array_free(out);

    // Forget all inputs (caller owns them via mlx_vector_array_get refcount).
    std::mem::forget(q);
    std::mem::forget(k);
    std::mem::forget(v);
    std::mem::forget(g);
    std::mem::forget(beta);
    std::mem::forget(state);

    0
}

/// Call the compiled GDR step.
#[cfg(feature = "metal")]
fn compiled_gdr_step(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> (MlxArray, MlxArray) {
    let results = COMPILED_GDR_STEP.call(&[q, k, v, g, beta, state]);
    let mut it = results.into_iter();
    let y = it.next().expect("gdr_step output y");
    let new_state = it.next().expect("gdr_step output state");
    (y, new_state)
}

// ── Metal GDR kernel ────────────────────────────────────────────────────────
//
// Ported from mlx_lm's gated_delta.py: a custom Metal shader that fuses the
// entire GDR state update into a single GPU dispatch with SIMD reductions.
// This replaces ~12 MLX ops with 1 kernel dispatch.

#[cfg(feature = "metal")]
static GDR_METAL_KERNEL: std::sync::LazyLock<crate::mlx::MetalKernel> =
    std::sync::LazyLock::new(|| {
        crate::mlx::MetalKernel::new(
            "gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            // Metal shader source — scalar gating, no mask (decode-only).
            r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }
        "#,
        )
    });

/// Execute the GDR Metal kernel for a single decode step.
/// inputs: q[B,T,Hk,Dk], k[B,T,Hk,Dk], v[B,T,Hv,Dv], g[B,T,Hv], beta[B,T,Hv], state[B,Hv,Dv,Dk]
/// outputs: y[B,T,Hv,Dv], state_out[B,Hv,Dv,Dk]
#[cfg(feature = "metal")]
fn metal_gdr_kernel_step(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
    config: &MetalGdrConfig,
) -> (MlxArray, MlxArray) {
    use crate::mlx::Dtype;

    let b = 1i32; // batch size always 1 for decode
    let t = 1i32; // single timestep
    let hk = config.num_key_heads as i32;
    let hv = config.num_value_heads as i32;
    let dk = config.key_dim as i32;
    let dv = config.value_dim as i32;

    let y_shape = [b, t, hv, dv];
    let state_shape = [b, hv, dv, dk];

    let t_arr = MlxArray::scalar_i32(t);

    let results = GDR_METAL_KERNEL.apply(
        &[q, k, v, g, beta, state, &t_arr],
        [32, dv, b * hv], // grid
        [32, 4, 1],       // threadgroup
        &[&y_shape, &state_shape],
        &[Dtype::Bfloat16, Dtype::Float32],
        &[("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
        &[("InT", Dtype::Bfloat16), ("StT", Dtype::Float32)],
    );

    let mut it = results.into_iter();
    let y = it.next().expect("GDR kernel output y");
    let new_state = it.next().expect("GDR kernel output state");
    (y, new_state)
}

// ── Conv1d single-step ───────────────────────────────────────────────────────

/// Causal depthwise conv1d for a single timestep.
///
/// Uses manual concatenate+multiply+sum — faster than mlx_conv1d for
/// single-token decode because it avoids reshape/transpose overhead.
#[cfg(feature = "metal")]
fn conv1d_step(
    x: &MlxArray,
    conv_state: &mut MlxArray,
    kernel: &MlxArray,
    qkv_dim: usize,
    kernel_size: usize,
) -> MlxArray {
    use crate::mlx::{Dtype, as_dtype, concatenate_axis, multiply, reshape, silu, slice, sum_axis};

    let x_col = reshape(x, &[qkv_dim as i32, 1]);
    let full_window = concatenate_axis(&[conv_state.clone(), x_col], 1);
    let prod = multiply(&full_window, kernel);
    let conv_out = sum_axis(&prod, 1, false);

    // Truncate to bf16 before SiLU (matching CUDA precision), keep as f32 for downstream.
    let conv_bf16 = as_dtype(&conv_out, Dtype::Bfloat16);
    let activated = as_dtype(&silu(&conv_bf16), Dtype::Float32);

    // Update conv_state: shift left, drop oldest column, append x
    // New state = full_window[:, 1:] = columns [1..kernel_size] of full_window
    // slice(full_window, [0, 1], [qkv_dim, kernel_size], [1, 1])
    let state_width = (kernel_size - 1) as i32;
    if state_width > 0 {
        *conv_state = slice(
            &full_window,
            &[0, 1],
            &[qkv_dim as i32, kernel_size as i32],
            &[1, 1],
        );
    }

    activated
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
    x: &MlxArray,
    layer_weights: &MetalLinearAttnWeights,
    state: &mut MetalRecurrentState,
    layer_idx: usize,
    config: &MetalGdrConfig,
) -> MlxArray {
    use crate::mlx::{
        Dtype, add, as_dtype, broadcast_to, exp, expand_dims, multiply, negative, reshape,
        rms_norm, sigmoid, silu, slice, subtract, sum_axis, transpose_axes,
    };

    let num_key_heads = config.num_key_heads;
    let key_dim = config.key_dim;
    let num_value_heads = config.num_value_heads;
    let value_dim = config.value_dim;
    let q_dim = num_key_heads * key_dim;
    let k_dim = q_dim;
    let v_dim = num_value_heads * value_dim;

    // ── 1. Merged projections (4 matmuls → 2) ─────────────────────────
    // Fused QKVZ: single matmul → split into qkv + z
    // Fused BA: single matmul → split into beta + alpha
    let x_flat = reshape(x, &[1, config.hidden_size as i32]);

    // quantized_matmul returns in input dtype (bf16) — no cast needed.
    let qkvz = linear(&x_flat, &layer_weights.in_proj_qkvz);
    let (qkv_split, z_split) = layer_weights.qkvz_split;
    let qkv_raw = slice(&qkvz, &[0, 0], &[1, qkv_split], &[1, 1]);
    let z = slice(&qkvz, &[0, qkv_split], &[1, qkv_split + z_split], &[1, 1]);

    let ba = linear(&x_flat, &layer_weights.in_proj_ba);
    let nh = layer_weights.ba_num_heads;
    let beta_raw = slice(&ba, &[0, 0], &[1, nh], &[1, 1]);
    let alpha_raw = slice(&ba, &[0, nh], &[1, nh * 2], &[1, 1]);

    // Flatten to 1D for per-element ops
    let qkv_1d = reshape(&qkv_raw, &[config.qkv_dim() as i32]);

    // ── 2. Conv1d step ───────────────────────────────────────────────────
    // Cast qkv to f32 for conv1d (state is f32)
    let qkv_f32 = as_dtype(&qkv_1d, Dtype::Float32);

    let qkv_conv = conv1d_step(
        &qkv_f32,
        &mut state.conv_states[layer_idx],
        &layer_weights.conv1d_weight,
        config.qkv_dim(),
        config.conv_kernel,
    );

    // ── 3. Split QKV and RMS-normalize q, k ──────────────────────────────
    // qkv_conv: [qkv_dim] → split into q [q_dim], k [k_dim], v [v_dim]
    let q_raw = slice(&qkv_conv, &[0], &[q_dim as i32], &[1]);
    let k_raw = slice(&qkv_conv, &[q_dim as i32], &[(q_dim + k_dim) as i32], &[1]);
    let v_raw = slice(
        &qkv_conv,
        &[(q_dim + k_dim) as i32],
        &[(q_dim + k_dim + v_dim) as i32],
        &[1],
    );

    // MLX normalizes q/k per key head, not across the concatenated q_dim.
    let q_per_key_head = reshape(&q_raw, &[num_key_heads as i32, key_dim as i32]);
    let k_per_key_head = reshape(&k_raw, &[num_key_heads as i32, key_dim as i32]);
    let q_norm = rms_normalize(&q_per_key_head, 1e-6);
    let k_norm = rms_normalize(&k_per_key_head, 1e-6);

    // Match mlx_lm:
    //   q = (inv_scale**2) * rms_norm(q, None, 1e-6)
    //   k = inv_scale * rms_norm(k, None, 1e-6)
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    let q_scale = inv_scale * inv_scale;
    let q_scaled = multiply(&q_norm, &MlxArray::from_slice_f32(&[q_scale], &[1]));
    let k_scaled = multiply(&k_norm, &MlxArray::from_slice_f32(&[inv_scale], &[1]));

    // ── 4. Compute gate (g) and beta ─────────────────────────────────────
    // Flatten alpha and beta to [num_value_heads]
    let alpha_1d = as_dtype(
        &reshape(&alpha_raw, &[num_value_heads as i32]),
        Dtype::Float32,
    );
    let beta_1d = as_dtype(
        &reshape(&beta_raw, &[num_value_heads as i32]),
        Dtype::Float32,
    );

    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    // Compiled with shapeless=true matching mlx_lm's pattern.
    let exp_g = compiled_compute_g(&layer_weights.a_log, &alpha_1d, &layer_weights.dt_bias);

    // beta = sigmoid(beta_raw)
    let beta = sigmoid(&beta_1d);

    // Metal kernel disabled by default — crashes on longer sequences (>~50 tokens).
    // The kernel works for short sequences; needs state transposition debugging.
    // Enable with AGENT_INFER_GDR_METAL_KERNEL=1.
    let use_metal_kernel = std::env::var("AGENT_INFER_GDR_METAL_KERNEL")
        .ok()
        .is_some_and(|v| v == "1");

    // ── 5–6. State update ─────────────────────────────────────────────
    let heads_per_key = num_value_heads / num_key_heads;
    let hv = num_value_heads as i32;
    let dk = key_dim as i32;
    let dv = value_dim as i32;

    let output_heads = if use_metal_kernel {
        // Metal kernel path — single GPU dispatch for entire GDR step.
        let b = 1i32;
        let t = 1i32;
        let hk = num_key_heads as i32;

        let q_kernel = as_dtype(&reshape(&q_scaled, &[b, t, hk, dk]), Dtype::Bfloat16);
        let k_kernel = as_dtype(&reshape(&k_scaled, &[b, t, hk, dk]), Dtype::Bfloat16);
        let v_kernel = as_dtype(&reshape(&v_raw, &[b, t, hv, dv]), Dtype::Bfloat16);
        let g_kernel = reshape(&exp_g, &[b, t, hv]);
        let beta_kernel = reshape(&beta, &[b, t, hv]);

        // State: our [Hv, Dk, Dv] → kernel [B=1, Hv, Dv, Dk]
        let state_kernel = transpose_axes(
            &reshape(&state.states[layer_idx], &[b, hv, dk, dv]),
            &[0, 1, 3, 2],
        );

        let (y_kernel, state_out_kernel) = metal_gdr_kernel_step(
            &q_kernel,
            &k_kernel,
            &v_kernel,
            &g_kernel,
            &beta_kernel,
            &state_kernel,
            config,
        );

        // State back: [B=1, Hv, Dv, Dk] → [Hv, Dk, Dv]
        state.states[layer_idx] = reshape(
            &transpose_axes(&state_out_kernel, &[0, 1, 3, 2]),
            &[hv, dk, dv],
        );
        as_dtype(&reshape(&y_kernel, &[hv, dv]), Dtype::Float32)
    } else {
        // Compiled ops fallback.
        let k_expanded = if heads_per_key > 1 {
            let k_unsq = expand_dims(&k_scaled, 1);
            let k_broadcast =
                broadcast_to(&k_unsq, &[num_key_heads as i32, heads_per_key as i32, dk]);
            reshape(&k_broadcast, &[hv, dk])
        } else {
            k_scaled
        };
        let q_expanded = if heads_per_key > 1 {
            let q_unsq = expand_dims(&q_scaled, 1);
            let q_broadcast =
                broadcast_to(&q_unsq, &[num_key_heads as i32, heads_per_key as i32, dk]);
            reshape(&q_broadcast, &[hv, dk])
        } else {
            q_scaled
        };
        let v_heads = reshape(&v_raw, &[hv, dv]);

        let (y, s_updated) = compiled_gdr_step(
            &q_expanded,
            &k_expanded,
            &v_heads,
            &exp_g,
            &beta,
            &state.states[layer_idx],
        );
        state.states[layer_idx] = s_updated;
        y
    };

    // ── 7. Per-head RMSNorm + output gate ────────────────────────────────
    // output_heads: [num_value_heads, val_dim]
    // norm_weight: [val_dim] (broadcast across heads)
    // z: [1, z_dim] where z_dim = num_value_heads * val_dim

    // RMSNorm per head + flatten.
    let normed = rms_norm(
        &as_dtype(&output_heads, Dtype::Bfloat16),
        &layer_weights.norm_weight,
        config.rms_norm_eps,
    );
    let normed_flat = reshape(&normed, &[1, (num_value_heads * value_dim) as i32]);

    // Output gate: o = normed * silu(z).  silu takes f32 input.
    let z_silu = silu(&as_dtype(&z, Dtype::Float32));
    let gated = as_dtype(
        &multiply(&as_dtype(&normed_flat, Dtype::Float32), &z_silu),
        Dtype::Bfloat16,
    );

    // Output projection.
    linear(&gated, &layer_weights.out_proj)
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

        let x = MlxArray::from_slice_f32(&vec![1.0f32; qkv_dim], &[qkv_dim as i32]);
        let mut conv_state = MlxArray::from_slice_f32(
            &vec![0.0f32; qkv_dim * state_width],
            &[qkv_dim as i32, state_width as i32],
        );
        let kernel = MlxArray::from_slice_f32(
            &vec![0.25f32; qkv_dim * kernel_size],
            &[qkv_dim as i32, kernel_size as i32],
        );

        let out = conv1d_step(&x, &mut conv_state, &kernel, qkv_dim, kernel_size);

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
        use crate::mlx::eval;

        let values: Vec<f32> = vec![
            -10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 19.0, 20.0, 21.0, 50.0, 100.0,
        ];
        let input = MlxArray::from_slice_f32(&values, &[values.len() as i32]);
        let result = softplus(&input);
        eval(&[&result]);

        for (i, &x) in values.iter().enumerate() {
            let expected = ref_softplus(x);
            let actual = result.as_slice_f32()[i];
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
        use crate::mlx::eval;

        let large = MlxArray::from_slice_f32(&[88.0f32, 200.0, 1000.0], &[3]);
        let result = softplus(&large);
        eval(&[&result]);
        let vals = result.as_slice_f32();
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
        use crate::mlx::eval;

        let x = MlxArray::from_slice_f32(&[3.0f32, 4.0], &[1, 2]);
        let normed = rms_normalize(&x, 1e-6);
        eval(&[&normed]);

        let vals = normed.as_slice_f32();
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
        use crate::mlx::{add, eval, exp, multiply, negative};

        let a_log_val = -1.0f32; // exp(A_log) = exp(-1) ≈ 0.3679
        let alpha_val = 2.0f32;
        let dt_bias_val = 1.0f32;
        // softplus(alpha + dt_bias) = softplus(3.0) = log(1 + exp(3)) ≈ 3.0486
        // g_exp = -0.3679 * 3.0486 ≈ -1.1216
        // exp(g) ≈ exp(-1.1216) ≈ 0.3258

        let alpha = MlxArray::from_slice_f32(&[alpha_val], &[1]);
        let dt_bias = MlxArray::from_slice_f32(&[dt_bias_val], &[1]);
        let a_log = MlxArray::from_slice_f32(&[a_log_val], &[1]);

        let alpha_plus_bias = add(&alpha, &dt_bias);
        let sp = softplus(&alpha_plus_bias);
        let a_exp = exp(&a_log);
        let neg_a_exp = negative(&a_exp);
        let g_exponent = multiply(&neg_a_exp, &sp);
        let exp_g = exp(&g_exponent);
        eval(&[&exp_g]);

        let expected_sp = ref_softplus(alpha_val + dt_bias_val);
        let expected_g = (-a_log_val.exp() * expected_sp).exp();
        let actual = exp_g.as_slice_f32()[0];
        assert!(
            (actual - expected_g).abs() < 1e-4,
            "gate: expected {expected_g}, got {actual}"
        );
    }
}
