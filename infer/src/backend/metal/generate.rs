use std::time::Instant;

use anyhow::{Context, Result};

use super::config::MetalModelConfig;
use super::forward::build_forward_graph;
use super::kv_pool::MetalKVPool;
use super::mlx::{MlxArray, zeros};
use super::weights::StandardMetalWeights;
use crate::sampler::SamplingParams;

#[cfg(feature = "metal")]
pub(super) const KV_CACHE_CHUNK: i32 = 256;
#[cfg(feature = "metal")]
const METAL_KV_POOL_REQUEST_ID: usize = 0;

#[cfg(feature = "metal")]
pub(super) struct MetalGenerateOutput {
    pub(super) tokens: Vec<u32>,
    pub(super) finish_reason: &'static str,
    pub(super) ttft_ms: f64,
    pub(super) total_time_ms: f64,
}

// TODO: dead code for current use cases — MetalKVPool is only activated via
// the `AGENT_INFER_METAL_KV_POOL=1` env var and only wired into the Qwen3
// fallback path. Qwen3.5 bypasses it entirely (see the warning at the call
// site). Consider removing if the KV pool experiment is abandoned.
#[cfg(feature = "metal")]
pub(super) fn metal_kv_pool_enabled() -> bool {
    std::env::var("AGENT_INFER_METAL_KV_POOL")
        .ok()
        .is_some_and(|value| metal_kv_pool_flag_is_truthy(&value))
}

#[cfg(feature = "metal")]
struct MetalKvPoolRequestCleanup {
    pool: *mut MetalKVPool,
    request_id: usize,
}

#[cfg(feature = "metal")]
impl MetalKvPoolRequestCleanup {
    fn new(pool: &mut MetalKVPool, request_id: usize) -> Self {
        Self {
            pool: std::ptr::from_mut(pool),
            request_id,
        }
    }
}

#[cfg(feature = "metal")]
impl Drop for MetalKvPoolRequestCleanup {
    fn drop(&mut self) {
        if self.pool.is_null() {
            return;
        }

        // SAFETY: the guard is created inside `metal_generate` after the pool
        // and is dropped before the pool goes out of scope.
        unsafe {
            (&mut *self.pool).free_request(self.request_id);
        }
    }
}

#[cfg(feature = "metal")]
fn metal_kv_pool_flag_is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

// GPU required: all tensor operations use MlxArray on Metal unified memory.
#[cfg(feature = "metal")]
pub(super) fn metal_generate(
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<MetalGenerateOutput> {
    if max_new_tokens == 0 {
        return Ok(MetalGenerateOutput {
            tokens: Vec::new(),
            finish_reason: "length",
            ttft_ms: 0.0,
            total_time_ms: 0.0,
        });
    }

    // C++ full generate path (same as Qwen3.5 — batch prefill + double-buffered decode)
    if let Some(ref cpp_model) = weights.cpp_model {
        log::info!("Metal Qwen3: C++ full generate");
        let mut stop_ids: Vec<u32> = params.stop_token_ids.clone();
        if !params.ignore_eos {
            stop_ids.push(config.eos_token_id);
        }
        let (tokens, prefill_ms, decode_ms) = cpp_model.generate(
            input_ids,
            max_new_tokens,
            params.temperature,
            &stop_ids,
            on_token,
        )?;
        let total_time_ms = prefill_ms + decode_ms;
        let finish_reason = if tokens.last().is_some_and(|t| stop_ids.contains(t)) {
            "stop"
        } else {
            "length"
        };
        return Ok(MetalGenerateOutput {
            tokens,
            finish_reason,
            ttft_ms: prefill_ms,
            total_time_ms,
        });
    }

    let n_layers = config.num_hidden_layers;
    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.num_key_value_heads as i32;
    let head_dim = config.head_dim as i32;
    let eps = config.rms_norm_eps as f32;
    let rope_base = config.rope_theta as f32;
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let use_kv_pool = metal_kv_pool_enabled();
    super::sampling::validate_metal_sampling_params(params)?;

    log::info!("Metal transformer path: Rust/MLX");
    if use_kv_pool {
        log::info!("MetalKVPool enabled via AGENT_INFER_METAL_KV_POOL=1");
    }

    // P5: KV cache starts at the next 256-token boundary above the prefill length,
    // plus one chunk for initial decode steps.  Grown lazily in KV_CACHE_CHUNK steps.
    let prefill_len = input_ids.len() as i32;
    let initial_cap = ((prefill_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK + 1) * KV_CACHE_CHUNK;
    let mut kv_capacity = initial_cap;

    let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
    let cache_shape = [1i32, n_kv_heads, initial_cap, head_dim];
    let mut k_caches: Vec<MlxArray> = (0..n_layers)
        .map(|_| zeros(&cache_shape, kv_dtype))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..n_layers)
        .map(|_| zeros(&cache_shape, kv_dtype))
        .collect();
    let mut kv_pool = if use_kv_pool {
        let pool_tokens = std::cmp::max(
            initial_cap as usize,
            input_ids.len().saturating_add(max_new_tokens),
        );
        Some(
            MetalKVPool::new(
                n_layers,
                n_kv_heads as usize,
                head_dim as usize,
                pool_tokens,
                kv_dtype,
            )
            .context("pre-alloc MetalKVPool")?,
        )
    } else {
        None
    };
    let _kv_pool_cleanup = kv_pool
        .as_mut()
        .map(|pool| MetalKvPoolRequestCleanup::new(pool, METAL_KV_POOL_REQUEST_ID));

    let mut generated: Vec<u32> = Vec::new();
    let mut cache_len: i32 = 0;
    let mut ttft_ms = 0.0;

    // ── Phase 1: Prefill — build lazy graph, schedule GPU asynchronously ─────
    let prefill_token = build_forward_graph(
        input_ids,
        weights,
        &mut k_caches,
        &mut v_caches,
        cache_len,
        n_heads,
        n_kv_heads,
        head_dim,
        attn_scale,
        rope_base,
        eps,
        kv_pool.as_mut(),
        METAL_KV_POOL_REQUEST_ID,
        params,
    )?;
    // P6: schedule GPU execution without blocking CPU.
    super::ops::metal_async_eval(&prefill_token);
    cache_len += prefill_len;

    // ── Phase 2: Decode loop (double-buffered — P3/P6) ────────────────────────
    //
    // Each iteration: sync *previous* token via item() (GPU likely done since
    // async_eval), then build *next* graph and async_eval it before looping.
    // CPU graph-build overlaps with GPU execution of the current step.
    let mut pending = prefill_token;
    let mut decode_step: usize = 0;

    let finish_reason = loop {
        let next_token = pending.item_i32() as u32;

        if decode_step == 0 {
            ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "  TTFT: {ttft_ms:.1}ms (prefill {} tokens)",
                input_ids.len()
            );
        }

        let stop = (!params.ignore_eos && config.is_stop_token(next_token))
            || params.stop_token_ids.contains(&next_token);
        generated.push(next_token);
        on_token(next_token)?;

        if stop {
            break "stop";
        }
        if generated.len() >= max_new_tokens {
            break "length";
        }

        // P5: grow KV cache in 256-token chunks when capacity is about to overflow.
        if !use_kv_pool && cache_len + 1 > kv_capacity {
            let new_cap = kv_capacity + KV_CACHE_CHUNK;
            for li in 0..n_layers {
                super::ops::extend_kv_cache(&mut k_caches[li], n_kv_heads, head_dim, new_cap);
                super::ops::extend_kv_cache(&mut v_caches[li], n_kv_heads, head_dim, new_cap);
            }
            kv_capacity = new_cap;
        }

        // P5: release accumulated temporary Metal allocations every 256 steps.
        if decode_step > 0 && decode_step.is_multiple_of(256) {
            super::ops::clear_metal_cache();
        }

        let new_pending = build_forward_graph(
            &[next_token],
            weights,
            &mut k_caches,
            &mut v_caches,
            cache_len,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            kv_pool.as_mut(),
            METAL_KV_POOL_REQUEST_ID,
            params,
        )?;
        cache_len += 1;
        decode_step += 1;

        // P6: kick off GPU — CPU syncs at top of next iteration via item().
        super::ops::metal_async_eval(&new_pending);

        pending = new_pending;
    };

    let elapsed = t0.elapsed().as_secs_f64();
    let total_time_ms = elapsed * 1000.0;
    let decode_elapsed = (elapsed - ttft_ms / 1000.0).max(1e-9);
    let tps = generated.len() as f64 / decode_elapsed;
    log::info!("  generated {} tokens  ({tps:.1} tok/s)", generated.len());

    Ok(MetalGenerateOutput {
        tokens: generated,
        finish_reason,
        ttft_ms,
        total_time_ms,
    })
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;

    #[test]
    fn kv_pool_flag_parser_accepts_common_truthy_values() {
        for value in ["1", "true", "TRUE", "yes", "on", " 1 "] {
            assert!(
                metal_kv_pool_flag_is_truthy(value),
                "{value:?} should be truthy"
            );
        }
    }

    #[test]
    fn kv_pool_flag_parser_rejects_falsey_values() {
        for value in ["", "0", "false", "off", "no", "maybe"] {
            assert!(
                !metal_kv_pool_flag_is_truthy(value),
                "{value:?} should be falsey"
            );
        }
    }
}
