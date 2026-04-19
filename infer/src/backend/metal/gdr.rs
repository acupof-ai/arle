//! Gated Delta Rule (GDR) implementation for Metal/MLX backend.
//!
//! Implements the Qwen3.5 linear attention decode step using MLX high-level ops.
//! This is a correctness-first implementation — no custom Metal kernels.
//!
//! # GDR decode step (per layer, seq=1)
//!
//! 1. Project input to get q, k, v, z, beta, alpha
//! 2. Conv1d on `[q, k, v]` with causal kernel (size 4), SiLU activation
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
use super::mlx::MlxArray;

#[cfg(feature = "metal")]
use super::WeightTensor;

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
    pub in_proj_qkvz: WeightTensor,
    /// Merged Beta+Alpha projection: [num_value_heads * 2, hidden_size].
    pub in_proj_ba: WeightTensor,
    /// Individual projections used by the optional C++ step path.
    pub in_proj_qkv: WeightTensor,
    pub in_proj_z: WeightTensor,
    pub in_proj_b: WeightTensor,
    pub in_proj_a: WeightTensor,
    /// Split dimensions for QKVZ output: qkv_dim, z_dim.
    pub qkvz_split: (i32, i32),
    /// Number of value heads (for BA split).
    pub ba_num_heads: i32,
    /// Depthwise conv1d weight: [qkv_dim, kernel_size, 1] bf16.
    pub conv1d_weight: MlxArray,
    /// Per-head dt bias: `[num_value_heads]` bf16.
    pub dt_bias: MlxArray,
    /// Per-head log-decay: `[num_value_heads]` f32.
    pub a_log: MlxArray,
    /// RMSNorm weight (per head_dim, broadcast across heads): `[value_dim]` bf16.
    pub norm_weight: MlxArray,
    /// Output projection: [hidden_size, z_dim].
    pub out_proj: WeightTensor,
    /// Pre-computed q scale: inv_scale² (scalar f32).
    pub q_scale: MlxArray,
    /// Pre-computed k scale: inv_scale (scalar f32).
    pub k_scale: MlxArray,
}

// ── Recurrent state ──────────────────────────────────────────────────────────

/// Per-request recurrent state for all linear attention layers on Metal.
///
/// Each linear attention layer maintains:
/// - Recurrent state matrix: [1, num_value_heads, value_dim, key_dim] f32
///   (matches Metal kernel layout — no transpose needed)
/// - Conv1d rolling buffer: [1, conv_kernel - 1, qkv_dim] bf16
///   (matches mlx nn.Conv1d input layout — no reshape needed)
#[cfg(feature = "metal")]
pub struct MetalRecurrentState {
    /// Per-layer recurrent state: [1, Hv, Dv, Dk] f32.
    pub states: Vec<MlxArray>,
    /// Per-layer conv1d rolling buffer: [1, conv_kernel-1, qkv_dim] bf16.
    pub conv_states: Vec<MlxArray>,
    /// Number of tokens processed so far.
    pub seq_len: usize,
}

#[cfg(feature = "metal")]
impl MetalRecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub fn new(num_linear_layers: usize, config: &MetalGdrConfig) -> Self {
        use super::mlx::{Dtype, zeros};

        let mut states = Vec::with_capacity(num_linear_layers);
        let mut conv_states = Vec::with_capacity(num_linear_layers);

        for _ in 0..num_linear_layers {
            // Recurrent state: [1, Hv, Dv, Dk] f32 — matches Metal kernel layout
            states.push(zeros(
                &[
                    1,
                    config.num_value_heads as i32,
                    config.value_dim as i32,
                    config.key_dim as i32,
                ],
                Dtype::Float32,
            ));

            // Conv state: [1, conv_kernel-1, qkv_dim] bf16 — matches nn.Conv1d input
            let conv_state_width = (config.conv_kernel - 1) as i32;
            states.len(); // suppress unused
            conv_states.push(zeros(
                &[1, conv_state_width, config.qkv_dim() as i32],
                Dtype::Bfloat16,
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
    use super::mlx::{matmul, quantized_matmul};
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
    super::mlx::rms_norm_no_weight(x, eps)
}

// ── Helper: softplus ─────────────────────────────────────────────────────────

/// softplus(x) = log(1 + exp(x)), numerically stable.
/// For large x (> 20), exp(x) overflows — return x directly (the CUDA kernel
/// uses the same threshold). Implements: where(x > 20, x, log1p(exp(x))).
#[cfg(feature = "metal")]
fn softplus(x: &MlxArray) -> MlxArray {
    use super::mlx::{exp, greater, log1p, where_};

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

// ── Metal GDR kernel ────────────────────────────────────────────────────────
//
// Ported from mlx_lm's gated_delta.py: a custom Metal shader that fuses the
// entire GDR state update into a single GPU dispatch with SIMD reductions.
// This replaces ~12 MLX ops with 1 kernel dispatch.

#[cfg(feature = "metal")]
static GDR_METAL_KERNEL: std::sync::LazyLock<super::mlx::MetalKernel> =
    std::sync::LazyLock::new(|| {
        super::mlx::MetalKernel::new(
            "gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            // Metal shader source — scalar gating, no mask (decode-only).
            r"
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
        ",
        )
    });

/// Execute the GDR Metal kernel for a single decode step.
/// inputs: q[B,T,Hk,Dk], k[B,T,Hk,Dk], v[B,T,Hv,Dv], g[B,T,Hv], beta[B,T,Hv], state[B,Hv,Dv,Dk]
/// outputs: y[B,T,Hv,Dv], state_out[B,Hv,Dv,Dk]
#[cfg(feature = "metal")]
// reason: q/k/v/g/beta/hk/hv/dk/dv follow kernel math notation.
#[allow(clippy::many_single_char_names)]
fn metal_gdr_kernel_step(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
    config: &MetalGdrConfig,
) -> (MlxArray, MlxArray) {
    use super::mlx::Dtype;

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

// ── Conv1d step ─────────────────────────────────────────────────────────────

/// Conv1d step matching mlx_lm approach:
/// 1. Concatenate conv_state and new input along seq axis
/// 2. Update conv_state (keep last kernel_size-1 timesteps)
/// 3. Run standard nn.Conv1d (depthwise, no padding)
/// 4. SiLU activation
///
/// Input/state shapes match mlx_lm: [B=1, T, qkv_dim] in bf16.
#[cfg(feature = "metal")]
fn conv1d_step_v2(
    qkv: &MlxArray,
    conv_state: &mut MlxArray,
    kernel: &MlxArray,
    config: &MetalGdrConfig,
) -> MlxArray {
    use super::mlx::{concatenate_axis, conv1d, silu, slice};

    let qkv_dim = config.qkv_dim() as i32;
    let n_keep = (config.conv_kernel - 1) as i32;
    let total_len = n_keep + 1; // conv_state has n_keep timesteps, we add 1

    // conv_input: [1, conv_kernel, qkv_dim] — concat state [1, n_keep, C] + qkv [1, 1, C]
    let conv_input = concatenate_axis(&[conv_state.clone(), qkv.clone()], 1);

    // Update state: keep last n_keep timesteps
    *conv_state = slice(
        &conv_input,
        &[0, 1, 0],
        &[1, total_len, qkv_dim],
        &[1, 1, 1],
    );

    // Depthwise conv1d: [1, conv_kernel, C] → [1, 1, C] (no padding, groups=C)
    let conv_out = conv1d(&conv_input, kernel, 1, 0, qkv_dim);

    // SiLU activation (works in any dtype)
    silu(&conv_out)
}

// ── GDR decode step ──────────────────────────────────────────────────────────

/// Try the C++ fused GDR forward path. Returns None if weights are not all quantized.
#[cfg(feature = "metal")]
fn try_cpp_gdr_forward(
    x: &MlxArray,
    lw: &MetalLinearAttnWeights,
    state: &mut MetalRecurrentState,
    layer_idx: usize,
    config: &MetalGdrConfig,
) -> Option<MlxArray> {
    use super::WeightTensor;

    // Extract quantized weight params. Bail to Rust path if any are Dense.
    let (qkvz_w, qkvz_s, qkvz_b, qkvz_gs, qkvz_bits) = match &lw.in_proj_qkvz {
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => (w, scales, biases, *group_size, *bits),
        WeightTensor::Dense(_) => return None,
    };
    let (ba_w, ba_s, ba_b, ba_gs, ba_bits) = match &lw.in_proj_ba {
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => (w, scales, biases, *group_size, *bits),
        WeightTensor::Dense(_) => return None,
    };
    let (out_w, out_s, out_b, out_gs, out_bits) = match &lw.out_proj {
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => (w, scales, biases, *group_size, *bits),
        WeightTensor::Dense(_) => return None,
    };

    let use_metal_kernel = std::env::var("AGENT_INFER_GDR_METAL_KERNEL")
        .ok()
        .is_none_or(|value| value != "0");

    let inv_scale = 1.0 / (config.key_dim as f32).sqrt();

    // Reshape x from [1, hidden] to [1, 1, hidden] for the C++ function
    let x_3d = super::mlx::reshape(x, &[1, 1, config.hidden_size as i32]);

    let result = super::mlx::gdr_layer_forward(
        &x_3d,
        qkvz_w,
        qkvz_s,
        qkvz_b,
        qkvz_gs,
        qkvz_bits,
        lw.qkvz_split.0,
        lw.qkvz_split.1,
        ba_w,
        ba_s,
        ba_b,
        ba_gs,
        ba_bits,
        lw.ba_num_heads,
        &lw.conv1d_weight,
        &mut state.conv_states[layer_idx],
        config.conv_kernel as i32,
        &lw.a_log,
        &lw.dt_bias,
        &lw.norm_weight,
        config.rms_norm_eps,
        out_w,
        out_s,
        out_b,
        out_gs,
        out_bits,
        config.num_key_heads as i32,
        config.key_dim as i32,
        config.num_value_heads as i32,
        config.value_dim as i32,
        inv_scale * inv_scale,
        inv_scale,
        &mut state.states[layer_idx],
        GDR_METAL_KERNEL.as_raw(),
        use_metal_kernel,
    );

    Some(result)
}

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
///
/// Optimized GDR decode step matching mlx_lm's approach:
/// - Standard mlx::conv1d instead of manual multiply+sum (saves ~4 ops)
/// - State stored in [B, Hv, Dv, Dk] — no transposes around Metal kernel (saves 4 ops)
/// - Work in bf16 throughout — minimal dtype casts (saves ~8 casts)
/// - 3D/4D tensors — fewer reshapes (saves ~6 reshapes)
///
/// Total ops: ~34 (down from ~60), matching mlx_lm.
#[cfg(feature = "metal")]
// reason: q/k/v/g/beta/qkv/hk/hv/dk/dv match the linear-attention math directly.
#[allow(clippy::many_single_char_names)]
pub fn metal_gdr_decode_step(
    x: &MlxArray,
    layer_weights: &MetalLinearAttnWeights,
    state: &mut MetalRecurrentState,
    layer_idx: usize,
    config: &MetalGdrConfig,
) -> MlxArray {
    use super::mlx::{Dtype, add, as_dtype, multiply, reshape, rms_norm, sigmoid, silu, slice};

    // Try the optional C++ step path first; it requires fully quantized weights.
    if let Some(result) = try_cpp_gdr_forward(x, layer_weights, state, layer_idx, config) {
        return result;
    }
    // Fallback to Rust per-op path for Dense weights.

    let hk = config.num_key_heads as i32;
    let dk = config.key_dim as i32;
    let hv = config.num_value_heads as i32;
    let dv = config.value_dim as i32;

    // ── 1. Projections (2 fused matmuls) ─────────────────────────────────
    let x_flat = reshape(x, &[1, 1, config.hidden_size as i32]);

    let qkvz = linear(&x_flat, &layer_weights.in_proj_qkvz);
    let (qkv_split, z_split) = layer_weights.qkvz_split;
    // qkv: [1, 1, qkv_dim], z: [1, 1, z_dim]
    let qkv = slice(&qkvz, &[0, 0, 0], &[1, 1, qkv_split], &[1, 1, 1]);
    let z = slice(
        &qkvz,
        &[0, 0, qkv_split],
        &[1, 1, qkv_split + z_split],
        &[1, 1, 1],
    );

    let ba = linear(&x_flat, &layer_weights.in_proj_ba);
    let nh = layer_weights.ba_num_heads;
    let b_raw = slice(&ba, &[0, 0, 0], &[1, 1, nh], &[1, 1, 1]);
    let a_raw = slice(&ba, &[0, 0, nh], &[1, 1, nh * 2], &[1, 1, 1]);

    // ── 2. Conv1d (standard mlx::conv1d, depthwise) ──────────────────────
    // qkv is [1, 1, qkv_dim] bf16 — same layout as mlx_lm
    let conv_out = conv1d_step_v2(
        &qkv,
        &mut state.conv_states[layer_idx],
        &layer_weights.conv1d_weight,
        config,
    );
    // conv_out: [1, 1, qkv_dim] bf16 (already activated with SiLU)

    // ── 3. Split QKV and RMS-normalize q, k ──────────────────────────────
    let q_dim = hk * dk;
    let k_dim = q_dim;
    let v_dim = hv * dv;

    // Split conv output: [1, 1, qkv_dim] → q, k, v
    let q_raw = reshape(
        &slice(&conv_out, &[0, 0, 0], &[1, 1, q_dim], &[1, 1, 1]),
        &[1, 1, hk, dk],
    );
    let k_raw = reshape(
        &slice(
            &conv_out,
            &[0, 0, q_dim],
            &[1, 1, q_dim + k_dim],
            &[1, 1, 1],
        ),
        &[1, 1, hk, dk],
    );
    let v_raw = reshape(
        &slice(
            &conv_out,
            &[0, 0, q_dim + k_dim],
            &[1, 1, q_dim + k_dim + v_dim],
            &[1, 1, 1],
        ),
        &[1, 1, hv, dv],
    );

    // RMS-normalize per head + scale (matching mlx_lm exactly)
    let inv_scale = 1.0 / (config.key_dim as f32).sqrt();
    let q = multiply(
        &rms_normalize(&q_raw, 1e-6),
        &MlxArray::scalar_f32(inv_scale * inv_scale),
    );
    let k = multiply(
        &rms_normalize(&k_raw, 1e-6),
        &MlxArray::scalar_f32(inv_scale),
    );

    // ── 4. Compute gate (g) and beta ─────────────────────────────────────
    let beta = sigmoid(&b_raw);

    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    let g = {
        use super::mlx::{exp, negative};
        let a_plus_bias = add(&a_raw, &layer_weights.dt_bias);
        let sp = softplus(&a_plus_bias);
        exp(&multiply(&negative(&exp(&layer_weights.a_log)), &sp))
    };

    // ── 5. Metal kernel GDR state update ─────────────────────────────────
    let use_metal_kernel = std::env::var("AGENT_INFER_GDR_METAL_KERNEL")
        .ok()
        .is_none_or(|value| value != "0");

    let y = if use_metal_kernel {
        // Reshape for kernel: q/k [1,1,Hk,Dk]→bf16, v [1,1,Hv,Dv]→bf16
        // g [1,1,Hv], beta [1,1,Hv]
        // State already in [1, Hv, Dv, Dk] f32 — no transpose needed!
        let q_bf16 = as_dtype(&q, Dtype::Bfloat16);
        let k_bf16 = as_dtype(&k, Dtype::Bfloat16);
        let v_bf16 = as_dtype(&v_raw, Dtype::Bfloat16);
        let g_3d = reshape(&g, &[1, 1, hv]);
        let beta_3d = reshape(&beta, &[1, 1, hv]);

        let (y_out, new_state) = metal_gdr_kernel_step(
            &q_bf16,
            &k_bf16,
            &v_bf16,
            &g_3d,
            &beta_3d,
            &state.states[layer_idx],
            config,
        );

        state.states[layer_idx] = new_state;
        // y_out: [1, 1, Hv, Dv] bf16
        y_out
    } else {
        // Ops fallback — use the compiled step ops pattern
        use super::mlx::{broadcast_to, expand_dims, subtract, sum_axis};
        let heads_per_key = config.num_value_heads / config.num_key_heads;

        // Expand q/k for GQA
        let q_exp = if heads_per_key > 1 {
            let q_unsq = expand_dims(&q, 2);
            reshape(
                &broadcast_to(&q_unsq, &[1, 1, hk, heads_per_key as i32, dk]),
                &[1, hv, dk],
            )
        } else {
            reshape(&q, &[1, hv, dk])
        };
        let k_exp = if heads_per_key > 1 {
            let k_unsq = expand_dims(&k, 2);
            reshape(
                &broadcast_to(&k_unsq, &[1, 1, hk, heads_per_key as i32, dk]),
                &[1, hv, dk],
            )
        } else {
            reshape(&k, &[1, hv, dk])
        };
        let v_3d = reshape(&v_raw, &[1, hv, dv]);

        // State: [1, Hv, Dv, Dk] — decay + delta update
        let g_4d = reshape(&g, &[1, hv, 1, 1]);
        let s = &state.states[layer_idx];
        let s_decayed = multiply(s, &g_4d);
        let k_4d = reshape(&k_exp, &[1, hv, 1, dk]);
        let kv_mem = sum_axis(&multiply(&s_decayed, &k_4d), -1, false);
        let beta_3d = reshape(&beta, &[1, hv, 1]);
        let delta = multiply(&subtract(&v_3d, &kv_mem), &beta_3d);
        let delta_4d = reshape(&delta, &[1, hv, dv, 1]);
        let s_updated = add(&s_decayed, &multiply(&delta_4d, &k_4d));
        let q_4d = reshape(&q_exp, &[1, hv, 1, dk]);
        let y_raw = sum_axis(&multiply(&s_updated, &q_4d), -1, false);
        state.states[layer_idx] = s_updated;
        // y_raw: [1, Hv, Dv] → reshape to [1, 1, Hv, Dv]
        reshape(&y_raw, &[1, 1, hv, dv])
    };

    // ── 6. Per-head RMSNorm + output gate ────────────────────────────────
    // y: [1, 1, Hv, Dv] → reshape to [Hv, Dv] for per-head norm
    let y_heads = reshape(&y, &[hv, dv]);
    let normed = rms_norm(&y_heads, &layer_weights.norm_weight, config.rms_norm_eps);

    // z: [1, 1, z_dim] → reshape [1, 1, Hv, Dv] → silu → multiply with normed
    let z_gated = reshape(&z, &[hv, dv]);
    let out = multiply(&normed, &silu(&z_gated));

    // Output projection: flatten to [1, z_dim] → matmul
    let out_flat = reshape(&out, &[1, hv * dv]);
    linear(&out_flat, &layer_weights.out_proj)
}

#[cfg(test)]
#[cfg(feature = "metal")]
mod tests {
    use super::*;
    use crate::test_support::metal_test_guard;

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max)
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_delta_with_tape_raw(
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        g: &MlxArray,
        beta: &MlxArray,
        state_in: &MlxArray,
        t: i32,
    ) -> (MlxArray, MlxArray, MlxArray) {
        let mut out_y = std::ptr::null_mut();
        let mut out_state = std::ptr::null_mut();
        let mut out_tape = std::ptr::null_mut();
        unsafe {
            mlx_sys::mlx_gated_delta_with_tape(
                q.as_raw(),
                k.as_raw(),
                v.as_raw(),
                g.as_raw(),
                beta.as_raw(),
                state_in.as_raw(),
                t,
                &raw mut out_y,
                &raw mut out_state,
                &raw mut out_tape,
            );
        }
        let y = unsafe { MlxArray::from_raw_checked(out_y) }.expect("gated delta y");
        let state = unsafe { MlxArray::from_raw_checked(out_state) }.expect("gated delta state");
        let tape = unsafe { MlxArray::from_raw_checked(out_tape) }.expect("gated delta tape");
        (y, state, tape)
    }

    fn tape_replay_raw(
        tape: &MlxArray,
        k: &MlxArray,
        g: &MlxArray,
        state_in: &MlxArray,
        t: i32,
    ) -> MlxArray {
        unsafe {
            MlxArray::from_raw_checked(mlx_sys::mlx_tape_replay(
                tape.as_raw(),
                k.as_raw(),
                g.as_raw(),
                state_in.as_raw(),
                t,
            ))
        }
        .expect("tape replay")
    }

    fn tape_replay_varlen_raw(
        tape: &MlxArray,
        k: &MlxArray,
        g: &MlxArray,
        state_in: &MlxArray,
        steps: &MlxArray,
    ) -> MlxArray {
        unsafe {
            MlxArray::from_raw_checked(mlx_sys::mlx_tape_replay_varlen(
                tape.as_raw(),
                k.as_raw(),
                g.as_raw(),
                state_in.as_raw(),
                steps.as_raw(),
            ))
        }
        .expect("tape replay varlen")
    }

    fn make_gdr_test_inputs() -> (MlxArray, MlxArray, MlxArray, MlxArray, MlxArray, MlxArray) {
        use crate::backend::metal::mlx::{Dtype, as_dtype};

        const T: usize = 3;
        const DK: usize = 32;
        const DV: usize = 4;

        let q_data: Vec<f32> = (0..T * DK)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..T * DK)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.04)
            .collect();
        let v_data: Vec<f32> = (0..T * DV).map(|i| ((i % 7) as f32 - 3.0) * 0.07).collect();
        let g_data: Vec<f32> = vec![0.91, 0.73, 0.88];
        let beta_data: Vec<f32> = vec![0.42, 0.57, 0.61];
        let state_data: Vec<f32> = (0..DV * DK)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.03)
            .collect();

        let q = as_dtype(
            &MlxArray::from_slice_f32(&q_data, &[1, T as i32, 1, DK as i32]),
            Dtype::Bfloat16,
        );
        let k = as_dtype(
            &MlxArray::from_slice_f32(&k_data, &[1, T as i32, 1, DK as i32]),
            Dtype::Bfloat16,
        );
        let v = as_dtype(
            &MlxArray::from_slice_f32(&v_data, &[1, T as i32, 1, DV as i32]),
            Dtype::Bfloat16,
        );
        let g = as_dtype(
            &MlxArray::from_slice_f32(&g_data, &[1, T as i32, 1]),
            Dtype::Bfloat16,
        );
        let beta = as_dtype(
            &MlxArray::from_slice_f32(&beta_data, &[1, T as i32, 1]),
            Dtype::Bfloat16,
        );
        let state = MlxArray::from_slice_f32(&state_data, &[1, 1, DV as i32, DK as i32]);
        (q, k, v, g, beta, state)
    }

    fn reference_tape_replay(
        tape: &MlxArray,
        k: &MlxArray,
        g: &MlxArray,
        state_in: &MlxArray,
    ) -> Vec<f32> {
        use crate::backend::metal::mlx::{Dtype, as_dtype, eval};

        let tape_f32 = as_dtype(tape, Dtype::Float32);
        let k_f32 = as_dtype(k, Dtype::Float32);
        let g_f32 = as_dtype(g, Dtype::Float32);
        eval(&[&tape_f32, &k_f32, &g_f32, state_in]);

        let tape_vals = tape_f32.as_slice_f32();
        let k_vals = k_f32.as_slice_f32();
        let g_vals = g_f32.as_slice_f32();
        let mut state = state_in.as_slice_f32().to_vec();

        let shape = tape.shape();
        let t = shape[1] as usize;
        let dv = shape[3] as usize;
        let dk = k.shape()[3] as usize;

        for step in 0..t {
            let g_step = g_vals[step];
            for dv_idx in 0..dv {
                let delta = tape_vals[step * dv + dv_idx];
                for dk_idx in 0..dk {
                    let state_idx = dv_idx * dk + dk_idx;
                    let k_idx = step * dk + dk_idx;
                    state[state_idx] = state[state_idx] * g_step + k_vals[k_idx] * delta;
                }
            }
        }

        state
    }

    /// Smoke test: verify shapes through the conv1d step (v2 — standard conv1d).
    #[test]
    fn test_conv1d_step_shapes() {
        let _guard = metal_test_guard();
        let kernel_size = 4usize;

        let config = MetalGdrConfig {
            num_key_heads: 2,
            key_dim: 16,
            num_value_heads: 2,
            value_dim: 16,
            conv_kernel: kernel_size,
            hidden_size: 64,
            rms_norm_eps: 1e-6,
        };
        let qkv_dim = config.qkv_dim();

        // x: [1, 1, qkv_dim] bf16
        let x = crate::backend::metal::mlx::as_dtype(
            &MlxArray::from_slice_f32(&vec![1.0f32; qkv_dim], &[1, 1, qkv_dim as i32]),
            crate::backend::metal::mlx::Dtype::Bfloat16,
        );
        // conv_state: [1, kernel_size-1, qkv_dim] bf16
        let mut conv_state = crate::backend::metal::mlx::zeros(
            &[1, (kernel_size - 1) as i32, qkv_dim as i32],
            crate::backend::metal::mlx::Dtype::Bfloat16,
        );
        // kernel: [qkv_dim, kernel_size, 1] bf16
        let kernel = crate::backend::metal::mlx::as_dtype(
            &MlxArray::from_slice_f32(
                &vec![0.25f32; qkv_dim * kernel_size],
                &[qkv_dim as i32, kernel_size as i32, 1],
            ),
            crate::backend::metal::mlx::Dtype::Bfloat16,
        );

        let out = conv1d_step_v2(&x, &mut conv_state, &kernel, &config);

        // Output: [1, 1, qkv_dim]
        assert_eq!(out.shape(), &[1, 1, qkv_dim as i32]);
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
        assert_eq!(state.states[0].shape(), &[1, 1, 2, 2]);
        assert_eq!(state.conv_states[0].shape(), &[1, 3, 6]);
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
        assert_eq!(state.states[0].shape(), &[1, 32, 128, 128]);
        assert_eq!(state.conv_states[0].shape(), &[1, 3, 8192]);
    }

    #[test]
    fn test_gated_delta_parallel_matches_serial_state_update() {
        let _guard = metal_test_guard();
        use crate::backend::metal::mlx::{eval, slice};

        let (q, k, v, g, beta, state_in) = make_gdr_test_inputs();
        let (_, parallel_state, _) = gated_delta_with_tape_raw(&q, &k, &v, &g, &beta, &state_in, 3);

        let mut serial_state = state_in.clone();
        for step in 0..3i32 {
            let q_step = slice(&q, &[0, step, 0, 0], &[1, step + 1, 1, 32], &[1, 1, 1, 1]);
            let k_step = slice(&k, &[0, step, 0, 0], &[1, step + 1, 1, 32], &[1, 1, 1, 1]);
            let v_step = slice(&v, &[0, step, 0, 0], &[1, step + 1, 1, 4], &[1, 1, 1, 1]);
            let g_step = slice(&g, &[0, step, 0], &[1, step + 1, 1], &[1, 1, 1]);
            let beta_step = slice(&beta, &[0, step, 0], &[1, step + 1, 1], &[1, 1, 1]);
            let (_, next_state, _) = gated_delta_with_tape_raw(
                &q_step,
                &k_step,
                &v_step,
                &g_step,
                &beta_step,
                &serial_state,
                1,
            );
            serial_state = next_state;
        }

        eval(&[&parallel_state, &serial_state]);
        let diff = max_abs_diff(parallel_state.as_slice_f32(), serial_state.as_slice_f32());
        assert!(
            diff < 1e-5,
            "parallel gated delta state drifted from serial replay by {diff}"
        );
    }

    #[test]
    fn test_tape_replay_matches_reference_fp32_state_update() {
        let _guard = metal_test_guard();
        use crate::backend::metal::mlx::eval;

        let (q, k, v, g, beta, state_in) = make_gdr_test_inputs();
        let (_, _state_out, tape) = gated_delta_with_tape_raw(&q, &k, &v, &g, &beta, &state_in, 3);
        let replayed = tape_replay_raw(&tape, &k, &g, &state_in, 3);
        let expected = reference_tape_replay(&tape, &k, &g, &state_in);

        eval(&[&replayed]);
        let actual = replayed.as_slice_f32();
        let diff = max_abs_diff(actual, &expected);
        assert!(
            diff < 1e-5,
            "tape replay drifted from reference fp32 state update by {diff}"
        );
    }

    #[test]
    fn test_tape_replay_varlen_matches_scalar() {
        let _guard = metal_test_guard();
        use crate::backend::metal::mlx::{Dtype, as_dtype, eval, slice};

        const B: i32 = 3;
        const T_PADDED: i32 = 3;
        const DK: i32 = 32;
        const DV: i32 = 4;

        let tape_data: Vec<f32> = (0..(B * T_PADDED * DV) as usize)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.031)
            .collect();
        let k_data: Vec<f32> = (0..(B * T_PADDED * DK) as usize)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.027)
            .collect();
        let g_data: Vec<f32> = (0..(B * T_PADDED) as usize)
            .map(|i| 0.55 + (i % 7) as f32 * 0.05)
            .collect();
        let state_data: Vec<f32> = (0..(B * DV * DK) as usize)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
            .collect();
        let steps_data = [1, 2, 3];

        let tape = as_dtype(
            &MlxArray::from_slice_f32(&tape_data, &[B, T_PADDED, 1, DV]),
            Dtype::Bfloat16,
        );
        let k = as_dtype(
            &MlxArray::from_slice_f32(&k_data, &[B, T_PADDED, 1, DK]),
            Dtype::Bfloat16,
        );
        let g = as_dtype(
            &MlxArray::from_slice_f32(&g_data, &[B, T_PADDED, 1]),
            Dtype::Bfloat16,
        );
        let state_in = MlxArray::from_slice_f32(&state_data, &[B, 1, DV, DK]);
        let steps = MlxArray::from_slice_i32(&steps_data, &[B]);

        let replayed_varlen = tape_replay_varlen_raw(&tape, &k, &g, &state_in, &steps);

        for (row_idx, &step_count) in steps_data.iter().enumerate() {
            let row = row_idx as i32;
            let tape_prefix = slice(
                &tape,
                &[row, 0, 0, 0],
                &[row + 1, step_count, 1, DV],
                &[1, 1, 1, 1],
            );
            let k_prefix = slice(
                &k,
                &[row, 0, 0, 0],
                &[row + 1, step_count, 1, DK],
                &[1, 1, 1, 1],
            );
            let g_prefix = slice(&g, &[row, 0, 0], &[row + 1, step_count, 1], &[1, 1, 1]);
            let state_row = slice(
                &state_in,
                &[row, 0, 0, 0],
                &[row + 1, 1, DV, DK],
                &[1, 1, 1, 1],
            );
            let expected =
                tape_replay_raw(&tape_prefix, &k_prefix, &g_prefix, &state_row, step_count);
            let actual = slice(
                &replayed_varlen,
                &[row, 0, 0, 0],
                &[row + 1, 1, DV, DK],
                &[1, 1, 1, 1],
            );

            eval(&[&expected, &actual]);
            let diff = max_abs_diff(actual.as_slice_f32(), expected.as_slice_f32());
            assert!(
                diff < 1e-5,
                "varlen tape replay row {row_idx} drifted from scalar replay by {diff}"
            );
        }
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
        use crate::backend::metal::mlx::eval;

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
        use crate::backend::metal::mlx::eval;

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
        use crate::backend::metal::mlx::eval;

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
        use crate::backend::metal::mlx::{add, eval, exp, multiply, negative};

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
