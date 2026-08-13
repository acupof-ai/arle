//! Forward-timing records for the rollout path and the trace sinks that print them.

use super::*;

pub(super) fn trace_model_component(trace: bool, component: &'static str, duration: Duration) {
    if trace {
        println!(
            "qwen35_full_forward_trace scope=model component={} seconds={:.6}",
            component,
            duration.as_secs_f64()
        );
    }
}

pub(super) fn trace_forward_component(
    trace: bool,
    layer_index: usize,
    component: &'static str,
    duration: Duration,
) {
    if trace {
        println!(
            "qwen35_full_forward_trace scope=layer layer={} component={} seconds={:.6}",
            layer_index,
            component,
            duration.as_secs_f64()
        );
    }
}

pub(super) fn trace_attention_component(
    trace: bool,
    layer_index: usize,
    component: &'static str,
    duration: Duration,
) {
    if trace {
        println!(
            "qwen35_full_forward_trace scope=attention layer={} component={} seconds={:.6}",
            layer_index,
            component,
            duration.as_secs_f64()
        );
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35AttentionForwardProfile {
    /// Host-side enqueue/API wall-clock attribution for profile harnesses.
    /// CUDA kernel elapsed time still requires NVTX/nsys cross-checking.
    pub q_proj: Duration,
    pub q_layout: Duration,
    pub k_proj: Duration,
    pub v_proj: Duration,
    pub kv_split: Duration,
    pub qk_norm: Duration,
    pub rope: Duration,
    pub repeat_kv: Duration,
    pub append_kv: Duration,
    pub sdpa: Duration,
    pub gate: Duration,
    pub merge: Duration,
    pub o_proj: Duration,
    pub linear_qkv_proj: Duration,
    pub linear_z_proj: Duration,
    pub linear_b_proj: Duration,
    pub linear_a_proj: Duration,
    pub linear_core: Duration,
    pub linear_out_proj: Duration,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35LayerForwardProfile {
    /// Host-side enqueue/API wall-clock attribution for profile harnesses.
    /// CUDA kernel elapsed time still requires NVTX/nsys cross-checking.
    pub input_rmsnorm: Duration,
    pub attention: Duration,
    pub attention_detail: Qwen35AttentionForwardProfile,
    pub attention_residual: Duration,
    pub post_attention_rmsnorm: Duration,
    pub mlp: Duration,
    pub mlp_residual: Duration,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35RolloutForwardProfile {
    pub total: Duration,
    pub cache_select: Duration,
    pub embedding: Duration,
    pub final_norm: Duration,
    pub lm_head: Duration,
    pub layers: Vec<Qwen35LayerForwardProfile>,
}

#[doc(hidden)]
impl Qwen35RolloutForwardProfile {
    pub fn input_rmsnorm_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.input_rmsnorm).sum()
    }

    pub fn attention_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.attention).sum()
    }

    pub fn attention_residual_total(&self) -> Duration {
        self.layers
            .iter()
            .map(|layer| layer.attention_residual)
            .sum()
    }

    pub fn post_attention_rmsnorm_total(&self) -> Duration {
        self.layers
            .iter()
            .map(|layer| layer.post_attention_rmsnorm)
            .sum()
    }

    pub fn mlp_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.mlp).sum()
    }

    pub fn mlp_residual_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.mlp_residual).sum()
    }
}
