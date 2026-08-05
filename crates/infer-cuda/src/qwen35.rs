//! Qwen3.5 / Qwen3.6 HYBRID model: gated-delta linear attention + periodic
//! full attention, BF16 MoE / dense MLP.
//!
//! Qwen3.5/3.6 production checkpoints interleave two attention kinds per layer
//! (`config.layer_types`):
//!   - `LinearAttention` (the majority): a gated-delta-rule recurrent linear
//!     attention with a depthwise conv1d front end and a gated output RMSNorm.
//!     This carries a per-slot recurrent state (`[V*K*V_head]` f32) + conv ring
//!     (`[qkv_dim*(kernel-1)]` bf16) ACROSS prefill and decode steps.
//!   - `FullAttention` (periodic): standard GQA with a per-head sigmoid gate on
//!     the q_proj output (`q_proj` rows = `heads*head_dim*2`), head_dim 128/256,
//!     on a contiguous per-slot K/V cache.
//!
//! Like [`crate::dsv4`], this model OWNS its KV state (no [`PagedKVPool`]) and
//! runs the uncached full-prefix correctness path: full-attn layers recompute
//! attention over the contiguous cache each step, linear-attn layers advance the
//! recurrent state in place. The continuous-batching paged + packed-batch path
//! (legacy `infer/src/model/qwen35`) is a perf follow-up.
//!
//! Gated-delta uses the RECURRENT kernel, never chunkwise: the chunkwise
//! TileLang WGMMA short-seq path HANGS on sm_90
//! (`errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md`).
//!
//! Precision: BF16 (the shared `moe::moe_forward` grouped GEMM). The two MoE swap
//! points for FP8 / 4-bit (Qwen3.6-4bit q4k) are inside
//! [`crate::moe::moe_forward`]'s two `moe_bf16_grouped_gemm_*` calls — a follow-up.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::ffi;
use cuda_kernels::kv_quant;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tensor::{
    HostMatrixSnapshot, WeightFormat, cache_ptr, offload_raw_slice, reload_raw_slice,
};
use cudarc::driver::{
    CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, sys::CUevent_flags,
};
use half::bf16;
use infer_plan::SamplingParams;
use infer_topo::TpConfig;
use qwen35_spec::{Qwen35AttentionTensorNames, Qwen35Config};
use safetensors::tensor::Dtype;

use crate::executor::sample_cuda_token_scratched;
use crate::loader::SafetensorLoader;
use crate::moe::{
    DEEPGEMM_CONTIG_ALIGN, MoeForwardScratch, QWEN35_DEEPGEMM_MIN_ROUTES, deepgemm_contig_rows_cap,
    moe_forward_into,
};
use crate::moe_config::ExpertSplit;
use crate::ops::{
    add_batch, argmax_into, argmax_row_into, copy_row_to_vec, embedding_batch, gemm_batch, gemv,
    silu_mul_fused, split_qkv, split2, upload_i32, warm_fp8_deepgemm_dense,
};
use crate::workspace::{HiddenSlot, SliceSlot, VecSlot};

#[path = "qwen35/dspark.rs"]
pub(crate) mod dspark;

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;
const QWEN35_BATCHED_DECODE_KV_SPLITS: usize = 4;

/// Longest query the FA3 paged path takes. FA3 zeroes the page stride when
/// `seqused_k` is set, so a ragged batch needs one launch per request; that is
/// free for decode/verify shapes and ruinous for prefill chunks, where the
/// TileLang kernel does the whole batch in one launch (measured 2026-07-28:
/// c=8 TTFT p50 12.07 → 18.23 s with prefill routed here).
const FA3_MAX_QLEN: usize = 64;

/// Floor for the derived FA3 decode split ceiling — the value shipped as a
/// constant before the ceiling was derived, and the measured optimum from
/// batch 4 up (2026-08-04: batch 8 is +0.36% at a ceiling of 20).
const FA3_DECODE_SPLITS_FLOOR: usize = 8;

#[cfg(test)]
pub(crate) mod conv_probe {
    use super::*;
    use std::cell::RefCell;

    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) channels: usize,
        pub(crate) kernel_size: usize,
        pub(crate) input: Vec<bf16>,
        pub(crate) weight: Vec<bf16>,
        pub(crate) output: Vec<bf16>,
        pub(crate) pre_state: Vec<bf16>,
        pub(crate) post_state: Vec<bf16>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "conv probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("conv probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(super) struct Pending {
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
        input: Vec<bf16>,
        weight: Vec<bf16>,
        pre_state: Vec<bf16>,
    }

    pub(super) fn begin(
        ctx: &DeviceContext,
        linear_idx: usize,
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
        input: &CudaView<'_, bf16>,
        weight: &DeviceVec,
        state: &DeviceVec,
    ) -> Result<Option<Pending>> {
        // 只捕获第一个 linear-attention 层的 conv：同一层的 conv 算术在所有层相同，
        // 一层足以验证正确性，避免下载每一层的 state。
        let needed = linear_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let input = ctx
            .stream
            .clone_dtoh(input)
            .map_err(|e| anyhow!("conv input D2H failed: {e}"))?;
        let weight = ctx
            .stream
            .clone_dtoh(&weight.data)
            .map_err(|e| anyhow!("conv weight D2H failed: {e}"))?;
        let pre_state = ctx
            .stream
            .clone_dtoh(&state.data)
            .map_err(|e| anyhow!("conv pre-state D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            channels,
            kernel_size,
            input,
            weight,
            pre_state,
        }))
    }

    pub(super) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        output: &CudaViewMut<'_, bf16>,
        state: &DeviceVec,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let output = ctx
            .stream
            .clone_dtoh(output)
            .map_err(|e| anyhow!("conv output D2H failed: {e}"))?;
        let post_state = ctx
            .stream
            .clone_dtoh(&state.data)
            .map_err(|e| anyhow!("conv post-state D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("conv probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    channels: pending.channels,
                    kernel_size: pending.kernel_size,
                    input: pending.input,
                    weight: pending.weight,
                    output,
                    pre_state: pending.pre_state,
                    post_state,
                });
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod gdr_probe {
    use super::*;
    use std::cell::RefCell;

    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) num_k_heads: usize,
        pub(crate) num_v_heads: usize,
        pub(crate) key_dim: usize,
        pub(crate) val_dim: usize,
        pub(crate) qkv: Vec<bf16>,
        pub(crate) b_proj: Vec<bf16>,
        pub(crate) a_proj: Vec<bf16>,
        pub(crate) dt_bias: Vec<bf16>,
        pub(crate) a_log: Vec<f32>,
        pub(crate) pre_state: Vec<f32>,
        pub(crate) output: Vec<bf16>,
        pub(crate) post_state: Vec<f32>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "gdr probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("gdr probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(super) struct Pending {
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        key_dim: usize,
        val_dim: usize,
        qkv: Vec<bf16>,
        b_proj: Vec<bf16>,
        a_proj: Vec<bf16>,
        dt_bias: Vec<bf16>,
        a_log: Vec<f32>,
        pre_state: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin(
        ctx: &DeviceContext,
        linear_idx: usize,
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        key_dim: usize,
        val_dim: usize,
        qkv: &CudaViewMut<'_, bf16>,
        b_proj: &CudaView<'_, bf16>,
        a_proj: &CudaView<'_, bf16>,
        dt_bias: &DeviceVec,
        a_log: &CudaSlice<f32>,
        state: &CudaSlice<f32>,
    ) -> Result<Option<Pending>> {
        let needed = linear_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let qkv = ctx
            .stream
            .clone_dtoh(qkv)
            .map_err(|e| anyhow!("gdr qkv D2H failed: {e}"))?;
        let b_proj = ctx
            .stream
            .clone_dtoh(b_proj)
            .map_err(|e| anyhow!("gdr b_proj D2H failed: {e}"))?;
        let a_proj = ctx
            .stream
            .clone_dtoh(a_proj)
            .map_err(|e| anyhow!("gdr a_proj D2H failed: {e}"))?;
        let dt_bias = ctx
            .stream
            .clone_dtoh(&dt_bias.data)
            .map_err(|e| anyhow!("gdr dt_bias D2H failed: {e}"))?;
        let a_log = ctx
            .stream
            .clone_dtoh(a_log)
            .map_err(|e| anyhow!("gdr a_log D2H failed: {e}"))?;
        let pre_state = ctx
            .stream
            .clone_dtoh(state)
            .map_err(|e| anyhow!("gdr pre-state D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            num_k_heads,
            num_v_heads,
            key_dim,
            val_dim,
            qkv,
            b_proj,
            a_proj,
            dt_bias,
            a_log,
            pre_state,
        }))
    }

    pub(super) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        output: &CudaViewMut<'_, bf16>,
        state: &CudaSlice<f32>,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let output = ctx
            .stream
            .clone_dtoh(output)
            .map_err(|e| anyhow!("gdr output D2H failed: {e}"))?;
        let post_state = ctx
            .stream
            .clone_dtoh(state)
            .map_err(|e| anyhow!("gdr post-state D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("gdr probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    num_k_heads: pending.num_k_heads,
                    num_v_heads: pending.num_v_heads,
                    key_dim: pending.key_dim,
                    val_dim: pending.val_dim,
                    qkv: pending.qkv,
                    b_proj: pending.b_proj,
                    a_proj: pending.a_proj,
                    dt_bias: pending.dt_bias,
                    a_log: pending.a_log,
                    pre_state: pending.pre_state,
                    output,
                    post_state,
                });
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod prep_probe {
    use super::*;
    use std::cell::RefCell;

    #[allow(dead_code)]
    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) num_q_heads: usize,
        pub(crate) num_kv_heads: usize,
        pub(crate) head_dim: usize,
        pub(crate) rotary_dim: usize,
        pub(crate) rms_eps: f32,
        pub(crate) start_pos: i32,
        pub(crate) q_full: Vec<bf16>,
        pub(crate) k_batch: Vec<bf16>,
        pub(crate) q_norm: Vec<bf16>,
        pub(crate) k_norm: Vec<bf16>,
        pub(crate) cos: Vec<bf16>,
        pub(crate) sin: Vec<bf16>,
        pub(crate) q_prepped: Vec<bf16>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "prep probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("prep probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(super) struct Pending {
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rms_eps: f32,
        start_pos: i32,
        q_full: Vec<bf16>,
        k_batch: Vec<bf16>,
        q_norm: Vec<bf16>,
        k_norm: Vec<bf16>,
        cos: Vec<bf16>,
        sin: Vec<bf16>,
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin(
        ctx: &DeviceContext,
        full_idx: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rms_eps: f32,
        q_full: &CudaSlice<bf16>,
        k_batch: &CudaSlice<bf16>,
        q_norm: &DeviceVec,
        k_norm: &DeviceVec,
        cos: &DeviceVec,
        sin: &DeviceVec,
        start_pos: &CudaSlice<i32>,
    ) -> Result<Option<Pending>> {
        let needed = full_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let q_full = ctx
            .stream
            .clone_dtoh(q_full)
            .map_err(|e| anyhow!("prep q_full D2H failed: {e}"))?;
        let k_batch = ctx
            .stream
            .clone_dtoh(k_batch)
            .map_err(|e| anyhow!("prep k_batch D2H failed: {e}"))?;
        let q_norm = ctx
            .stream
            .clone_dtoh(&q_norm.data)
            .map_err(|e| anyhow!("prep q_norm D2H failed: {e}"))?;
        let k_norm = ctx
            .stream
            .clone_dtoh(&k_norm.data)
            .map_err(|e| anyhow!("prep k_norm D2H failed: {e}"))?;
        let cos = ctx
            .stream
            .clone_dtoh(&cos.data)
            .map_err(|e| anyhow!("prep cos D2H failed: {e}"))?;
        let sin = ctx
            .stream
            .clone_dtoh(&sin.data)
            .map_err(|e| anyhow!("prep sin D2H failed: {e}"))?;
        let start_pos = ctx
            .stream
            .clone_dtoh(start_pos)
            .map_err(|e| anyhow!("prep start_pos D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rms_eps,
            start_pos: start_pos[0],
            q_full,
            k_batch,
            q_norm,
            k_norm,
            cos,
            sin,
        }))
    }

    pub(super) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        q_prepped: &HiddenStates,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let q_prepped = ctx
            .stream
            .clone_dtoh(&q_prepped.data)
            .map_err(|e| anyhow!("prep q_prepped D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("prep probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    num_q_heads: pending.num_q_heads,
                    num_kv_heads: pending.num_kv_heads,
                    head_dim: pending.head_dim,
                    rotary_dim: pending.rotary_dim,
                    rms_eps: pending.rms_eps,
                    start_pos: pending.start_pos,
                    q_full: pending.q_full,
                    k_batch: pending.k_batch,
                    q_norm: pending.q_norm,
                    k_norm: pending.k_norm,
                    cos: pending.cos,
                    sin: pending.sin,
                    q_prepped,
                });
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod attn_probe {
    use super::*;
    use std::cell::RefCell;

    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) num_q_heads: usize,
        pub(crate) num_kv_heads: usize,
        pub(crate) head_dim: usize,
        pub(crate) rotary_dim: usize,
        pub(crate) q_prepped: Vec<bf16>,
        pub(crate) k_raw: Vec<bf16>,
        pub(crate) v_raw: Vec<bf16>,
        pub(crate) k_norm: Vec<bf16>,
        pub(crate) cos: Vec<bf16>,
        pub(crate) sin: Vec<bf16>,
        pub(crate) rms_eps: f32,
        pub(crate) start_pos: i32,
        pub(crate) attn_out: Vec<bf16>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "attn probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("attn probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(super) struct Pending {
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        q_prepped: Vec<bf16>,
        k_raw: Vec<bf16>,
        v_raw: Vec<bf16>,
        k_norm: Vec<bf16>,
        cos: Vec<bf16>,
        sin: Vec<bf16>,
        rms_eps: f32,
        start_pos: i32,
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin(
        ctx: &DeviceContext,
        full_idx: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        q_prepped: &CudaSlice<bf16>,
        k_raw: &CudaSlice<bf16>,
        v_raw: &CudaSlice<bf16>,
        k_norm: &DeviceVec,
        cos: &DeviceVec,
        sin: &DeviceVec,
        rms_eps: f32,
        start_pos: &CudaSlice<i32>,
    ) -> Result<Option<Pending>> {
        let needed = full_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let q_prepped = ctx
            .stream
            .clone_dtoh(q_prepped)
            .map_err(|e| anyhow!("attn q_prepped D2H failed: {e}"))?;
        let k_raw = ctx
            .stream
            .clone_dtoh(k_raw)
            .map_err(|e| anyhow!("attn k_raw D2H failed: {e}"))?;
        let v_raw = ctx
            .stream
            .clone_dtoh(v_raw)
            .map_err(|e| anyhow!("attn v_raw D2H failed: {e}"))?;
        let k_norm = ctx
            .stream
            .clone_dtoh(&k_norm.data)
            .map_err(|e| anyhow!("attn k_norm D2H failed: {e}"))?;
        let cos = ctx
            .stream
            .clone_dtoh(&cos.data)
            .map_err(|e| anyhow!("attn cos D2H failed: {e}"))?;
        let sin = ctx
            .stream
            .clone_dtoh(&sin.data)
            .map_err(|e| anyhow!("attn sin D2H failed: {e}"))?;
        let start_pos = ctx
            .stream
            .clone_dtoh(start_pos)
            .map_err(|e| anyhow!("attn start_pos D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            q_prepped,
            k_raw,
            v_raw,
            k_norm,
            cos,
            sin,
            rms_eps,
            start_pos: start_pos[0],
        }))
    }

    pub(super) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        attn_out: &HiddenStates,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let attn_out = ctx
            .stream
            .clone_dtoh(&attn_out.data)
            .map_err(|e| anyhow!("attn attn_out D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("attn probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    num_q_heads: pending.num_q_heads,
                    num_kv_heads: pending.num_kv_heads,
                    head_dim: pending.head_dim,
                    rotary_dim: pending.rotary_dim,
                    q_prepped: pending.q_prepped,
                    k_raw: pending.k_raw,
                    v_raw: pending.v_raw,
                    k_norm: pending.k_norm,
                    cos: pending.cos,
                    sin: pending.sin,
                    rms_eps: pending.rms_eps,
                    start_pos: pending.start_pos,
                    attn_out,
                });
        });
        Ok(())
    }
}

fn qwen35_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_QWEN35_PROFILE").is_some()
            || std::env::var_os("ARLE_QWEN35_MOE_PROFILE").is_some()
    })
}

fn qwen35_startup_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_CUDA_STARTUP_PROFILE").is_some())
}

fn qwen35_startup_log(phase: &str, start: Instant, extra: std::fmt::Arguments<'_>) {
    if qwen35_startup_profile_enabled() {
        log::info!(
            target: "infer_cuda::startup",
            "cuda_startup phase=qwen35.{phase} elapsed_ms={:.1} {extra}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn qwen35_profile<T>(
    ctx: &DeviceContext,
    label: &'static str,
    layer_idx: Option<usize>,
    seq_len: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !qwen35_profile_enabled() {
        return f();
    }
    let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&ctx.stream)?;
    let host_t0 = Instant::now();
    let result = f();
    let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
    stop.record(&ctx.stream)?;
    stop.synchronize()?;
    let cuda_ms = start.elapsed_ms(&stop)? as f64;
    if std::env::var("INFER_TP_RANK")
        .map(|rank| rank == "0")
        .unwrap_or(true)
    {
        match layer_idx {
            Some(layer_idx) => eprintln!(
                "[qwen-layer-profile] {label} layer={layer_idx} seq={seq_len} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
            ),
            None => eprintln!(
                "[qwen-layer-profile] {label} layer=na seq={seq_len} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
            ),
        }
    }
    result
}

/// Route full-attention prefill chunks (`seq_len > 1`) through the vendored
/// FA3 hopper fwd shim instead of the in-tree `nonpaged_prefill_attention`
/// kernel (42.1% of prefill GPU time at 3k).
/// Default ON (licensed 2026-06-11: 3k prefill −36%, multi-shape verified —
/// see `wins/2026-06-11-qwen35-fa3-prefill-licensed.md`);
/// `--qwen35-fa3 false` is the same-binary fallback arm. A build without an
/// sm_90 target links the stub, whose marker is 0, and the gate keeps the
/// in-tree kernel.
/// The marker is process-wide, but capability is checked on the bound context
/// so mixed-device workers cannot inherit another device's decision.
fn qwen35_fa3_enabled(ctx: &DeviceContext) -> bool {
    if !crate::runtime_flags::qwen35_fa3() {
        return false;
    }
    // SAFETY: pure host query exported by both the real shim and the stub.
    let real = unsafe { ffi::arle_fa3_real_kernel_marker_cuda() } == 1;
    if !real {
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            log::warn!(
                "FA3 stub build (no sm_90 target) — full-attention prefill \
                 stays on the in-tree kernel"
            );
        });
        return false;
    }
    ctx.compute_capability() == (9, 0)
}

fn qwen35_fa3_decode_splits() -> usize {
    crate::runtime_flags::qwen35_fa3_decode_splits()
}

/// `--qwen35-gdr-chunked true`: route GDN prefill chunks (`seq_len > 1`)
/// through the FlashQLA chunked kernels (TileLang AOT, sm_90a) instead of
/// the serial `gated_delta_rule_prefill_recurrent` kernel (28.0% of prefill
/// GPU time pre-FA3).
/// Default ON (licensed 2026-08-02). The call site shape-guards the head
/// geometry, and builds without an sm_90 target link NOT_SUPPORTED stubs that
/// the probe below detects. Decode (`seq_len == 1`) stays on the recurrent
/// kernel.
fn qwen35_gdr_chunked_enabled() -> bool {
    crate::runtime_flags::qwen35_gdr_chunked()
}

/// One-shot probe: stub builds and non-sm90 devices return
/// CUDA_ERROR_NOT_SUPPORTED from the dispatch wrapper before touching any
/// pointer; a real kernel rejects the seq_len=0 launch with a different code.
fn fq_kernels_available(ctx: &DeviceContext, cumsum: ffi::FqCumsumFn) -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // The dispatch wrapper resolves SM via the calling thread's DRIVER
        // context — bind first or the probe is a per-thread lottery.
        if ctx.ctx.bind_to_thread().is_err() {
            return false;
        }
        // SAFETY: seq_len=0 → zero grid; no pointer is dereferenced.
        let r = unsafe { cumsum(std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null_mut()) };
        let ok = r != cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_SUPPORTED;
        if !ok {
            log::warn!("FlashQLA chunked GDR unavailable (stub build or non-sm90); using the recurrent scan");
        }
        ok
    })
}

/// sm_70 (V100) has no FA3 (sm_80+ CUTLASS-3.x) and no BF16 compute — the
/// hand-written `arle_fa2_sm70_attention_cuda` (FA2, FP16 half2 math, BF16 I/O)
/// is the SOTA option. `major < 8` covers sm_70 and sm_75. Latched once.
/// `ARLE_QWEN35_FA2_SM70=0` forces the naive FP32 SIMT fallback (A/B use).
fn qwen35_fa2_sm70_enabled(ctx: &DeviceContext) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (major, _minor) = ctx.compute_capability();
        major < 8
            && !matches!(
                std::env::var("ARLE_QWEN35_FA2_SM70").as_deref(),
                Ok("0") | Ok("false") | Ok("off")
            )
    })
}

/// Raw (un-scaled) LoRA A/B matrices for one student projection, pushed from
/// the train crate's OPD student loop for the per-step re-merge.
///
/// `a` is row-major `[rank, in_features]`, `b` is row-major
/// `[out_features, rank]` — the PEFT on-disk convention (mirrors the legacy
/// `infer/src/model/qwen35/lora.rs::StudentLoraMatrices`). Values are raw;
/// `scale = alpha / rank` is applied exactly once at merge time.
#[derive(Debug, Clone)]
pub struct StudentLoraMatrices {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StudentLoraProjection {
    FullQ,
    FullK,
    FullV,
    FullO,
    LinearQkv,
    LinearZ,
    LinearB,
    LinearA,
    LinearOut,
    MlpGate,
    MlpUp,
    MlpDown,
    MoeRouter,
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
    MoeSharedExpertGate,
    MoeExpertGate { expert_idx: usize },
    MoeExpertUp { expert_idx: usize },
    MoeExpertDown { expert_idx: usize },
}

impl StudentLoraProjection {
    pub fn label(self) -> Cow<'static, str> {
        match self {
            Self::FullQ => Cow::Borrowed("self_attn.q_proj"),
            Self::FullK => Cow::Borrowed("self_attn.k_proj"),
            Self::FullV => Cow::Borrowed("self_attn.v_proj"),
            Self::FullO => Cow::Borrowed("self_attn.o_proj"),
            Self::LinearQkv => Cow::Borrowed("self_attn.in_proj_qkv"),
            Self::LinearZ => Cow::Borrowed("self_attn.in_proj_z"),
            Self::LinearB => Cow::Borrowed("self_attn.in_proj_b"),
            Self::LinearA => Cow::Borrowed("self_attn.in_proj_a"),
            Self::LinearOut => Cow::Borrowed("self_attn.out_proj"),
            Self::MlpGate => Cow::Borrowed("mlp.gate_proj"),
            Self::MlpUp => Cow::Borrowed("mlp.up_proj"),
            Self::MlpDown => Cow::Borrowed("mlp.down_proj"),
            Self::MoeRouter => Cow::Borrowed("mlp.gate"),
            Self::MoeSharedGate => Cow::Borrowed("mlp.shared_expert.gate_proj"),
            Self::MoeSharedUp => Cow::Borrowed("mlp.shared_expert.up_proj"),
            Self::MoeSharedDown => Cow::Borrowed("mlp.shared_expert.down_proj"),
            Self::MoeSharedExpertGate => Cow::Borrowed("mlp.shared_expert_gate"),
            Self::MoeExpertGate { expert_idx } => {
                Cow::Owned(format!("mlp.experts.{expert_idx}.gate_proj"))
            }
            Self::MoeExpertUp { expert_idx } => {
                Cow::Owned(format!("mlp.experts.{expert_idx}.up_proj"))
            }
            Self::MoeExpertDown { expert_idx } => {
                Cow::Owned(format!("mlp.experts.{expert_idx}.down_proj"))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StudentLoraProjectionUpdate {
    pub projection: StudentLoraProjection,
    pub matrices: StudentLoraMatrices,
}

/// One model layer's LoRA adapters for the in-memory re-merge sync.
/// `layer_idx` is the absolute model-layer index.
#[derive(Debug, Clone)]
pub struct StudentLoraLayer {
    pub layer_idx: usize,
    pub projections: Vec<StudentLoraProjectionUpdate>,
}

/// A full LoRA update pushed from the train crate into the infer student
/// engine. Carries raw A/B per full-attention layer plus `rank`/`alpha`; the
/// merge path applies `scale = alpha / rank` once.
#[derive(Debug, Clone)]
pub struct StudentLoraUpdate {
    pub layers: Vec<StudentLoraLayer>,
    pub rank: usize,
    pub alpha: f32,
}

/// One frozen base projection's resident FP8 block-scaled device pointers,
/// exposed read-only for the train-infer weight-sharing path
/// (`--share-frozen-base`). The autograd student's frozen base layers import
/// these pointers as a NON-OWNING view instead of allocating their own ~27 GB
/// copy.
///
/// `layer_idx` + `proj_suffix` (e.g. `"self_attn.q_proj"`, or a per-expert path
/// like `"mlp.experts.3.gate_proj"`) form a stable key the train loader matches
/// against its `train_name`
/// (`model.(language_model.)?layers.{layer_idx}.{proj_suffix}.weight`).
/// `weight_ptr` / `scale_ptr` are raw `CUdeviceptr`s into the engine's resident
/// VRAM; the borrower must keep the engine base resident (no offload of the
/// shared base) for the borrow's lifetime. All fields are plain values (`Send`),
/// so the table crosses the engine-thread control seam safely.
#[derive(Debug, Clone)]
pub struct SharedFp8BaseProjection {
    pub layer_idx: usize,
    pub proj_suffix: String,
    pub weight_ptr: u64,
    pub scale_ptr: u64,
    pub rows: usize,
    pub cols: usize,
    pub block_m: usize,
    pub block_k: usize,
}

/// Host image of one whole slot for G3 capacity spill (mirror of
/// [`crate::dsv4::Dsv4SlotImage`]): the slot's full-attn KV pages plus the
/// per-linear-layer recurrent + conv state and the materialized length. Every
/// device buffer the slot owns is captured here byte-for-byte — a missed buffer
/// is a silently-wrong restore. `k_caches`/`v_caches` are NOT captured: the
/// paged default leaves them empty (full-attn KV is paged), asserted at capture.
pub(crate) struct Qwen35SlotImage {
    /// Full-attn KV bytes in slot-logical page order (from `copy_pages_to_host`).
    full_attn_pages: Vec<u8>,
    /// Logical page count the bytes cover (drives swap-in page allocation).
    full_attn_page_count: usize,
    /// `[num_linear]` gated-delta recurrent states (f32), D2H verbatim.
    gdr_host: Vec<Vec<f32>>,
    /// `[num_linear]` conv1d rings (bf16), D2H verbatim.
    conv_host: Vec<Vec<bf16>>,
    /// Materialized full-attn length the image was captured at.
    seq_len: usize,
}

impl Qwen35SlotImage {
    /// Approximate device-state byte size of a whole-slot image (the unit the G3
    /// `KvTierStore` budgets one entry against). Used only to size the tier's
    /// count cap — exactness isn't required, but it must scale with the image.
    pub(crate) fn dram_bytes(&self) -> usize {
        self.full_attn_pages.len()
            + self.gdr_host.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.conv_host.iter().map(|v| v.len() * 2).sum::<usize>()
            + 8
    }

    /// Flatten the whole-slot image into ONE length-prefixed byte buffer for the
    /// G3 [`kv_native_sys::KvTierStore`] (opaque-`u64`-keyed transport). The
    /// exact byte-inverse of [`Self::from_bytes`] — proven field-for-field in the
    /// `slot_image_byte_inverse` unit test. No serde: a small fixed header
    /// (`seq_len`, `full_attn_page_count`, `full_attn_pages` byte length, the two
    /// linear counts) followed by the full-attn bytes, then each gdr vec's
    /// `[len:u64][f32 LE...]` and each conv vec's `[len:u64][bf16 LE...]`.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.dram_bytes() + 64);
        buf.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.full_attn_page_count as u64).to_le_bytes());
        buf.extend_from_slice(&(self.full_attn_pages.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.gdr_host.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.conv_host.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.full_attn_pages);
        for gdr in &self.gdr_host {
            buf.extend_from_slice(&(gdr.len() as u64).to_le_bytes());
            for &x in gdr {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        for conv in &self.conv_host {
            buf.extend_from_slice(&(conv.len() as u64).to_le_bytes());
            for &x in conv {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        buf
    }

    /// Reconstruct a whole-slot image from [`Self::to_bytes`] — the exact
    /// byte-inverse. A cursor walks the fixed header then the four sized regions;
    /// any short/over-long buffer (a corrupt or foreign payload) errors rather
    /// than restore garbage.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let take_u64 = |pos: &mut usize| -> Result<u64> {
            let end = pos
                .checked_add(8)
                .ok_or_else(|| anyhow!("slot image header overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("slot image truncated reading u64 at {pos}"))?;
            *pos = end;
            Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
        };
        let seq_len = take_u64(&mut pos)? as usize;
        let full_attn_page_count = take_u64(&mut pos)? as usize;
        let full_attn_len = take_u64(&mut pos)? as usize;
        let num_gdr = take_u64(&mut pos)? as usize;
        let num_conv = take_u64(&mut pos)? as usize;
        let pages_end = pos
            .checked_add(full_attn_len)
            .ok_or_else(|| anyhow!("slot image full-attn length overflow"))?;
        let full_attn_pages = bytes
            .get(pos..pages_end)
            .ok_or_else(|| anyhow!("slot image truncated reading full-attn pages"))?
            .to_vec();
        pos = pages_end;
        let mut gdr_host = Vec::with_capacity(num_gdr);
        for _ in 0..num_gdr {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 4)
                .ok_or_else(|| anyhow!("slot image gdr length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("slot image truncated reading gdr state"))?;
            gdr_host.push(
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
            pos = end;
        }
        let mut conv_host = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 2)
                .ok_or_else(|| anyhow!("slot image conv length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("slot image truncated reading conv state"))?;
            conv_host.push(
                raw.chunks_exact(2)
                    .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            );
            pos = end;
        }
        ensure!(
            pos == bytes.len(),
            "slot image has {} trailing bytes after deserialize",
            bytes.len() - pos
        );
        Ok(Self {
            full_attn_pages,
            full_attn_page_count,
            gdr_host,
            conv_host,
            seq_len,
        })
    }
}

/// Per-slot gated-delta recurrent state (state + conv ring per linear-attn
/// layer). Carried across prefill/decode for one request.
///
/// Full-attention K/V is NOT here: since the shared-paged migration (Phase 2)
/// Qwen3.6's full-attn KV lives in the executor's shared [`PagedKVPool`]
/// (`full_attn_kv`), addressed per slot via a page table — never per-slot
/// contiguous (the `max_seq_len × kv_dim × num_full × 2` waste). `k_caches`/
/// `v_caches` stay as (normally-empty) `Vec`s only for the legacy contiguous
/// batched-decode A/B lane, which the paged default bypasses; the default
/// build leaves them empty (`new_slot_state` allocates zero full-attn bytes).
///
/// All dims are this rank's LOCAL shard sizes (= the global config dims on a
/// single GPU): each TP rank caches only its own v-head recurrent slabs and
/// qkv conv channels.
pub(crate) struct Qwen35SlotState {
    /// `[num_full_layers]` contiguous K caches, each `max_seq_len*kv_dim` bf16.
    /// EMPTY by default (full-attn KV is paged); populated only by the legacy
    /// contiguous lane when explicitly requested.
    k_caches: Vec<DeviceVec>,
    v_caches: Vec<DeviceVec>,
    /// `[num_linear_layers]` gated-delta recurrent states (`Vh*Kd*Vd` f32).
    gdr_states: Vec<CudaSlice<f32>>,
    /// `[num_linear_layers]` conv1d rings (`qkv_dim*(kernel-1)` bf16).
    conv_states: Vec<DeviceVec>,
    /// True once `acquire_recurrent` has run for the current occupant (even
    /// when `num_linear == 0` and the state vecs are empty). Guards the
    /// forward's `has_recurrent()` chokepoint against a missed acquire.
    recurrent_acquired: bool,
    /// Tokens materialized into the caches so far (full-attn kv_len).
    seq_len: usize,
}

/// A detached, reusable recurrent state block (`(gdr_states, conv_states)`) on
/// the executor's free-list. Released back by a finished request and popped by
/// the next one — same dims for every slot, so any block fits any slot.
pub(crate) type RecurrentBlock = (Vec<CudaSlice<f32>>, Vec<DeviceVec>);

/// One committed token + its behavior logprob under the filtered sampling
/// dist (`None` = uncaptured: greedy / delta policy).
pub(crate) type CommittedToken = (u32, Option<f32>);

/// D2H snapshot of the recurrent state at a prefix boundary.
/// Used by the sidecar prefix-cache to restore the recurrent layers
/// when reusing a Qwen3.5/3.6 hybrid prefix via the page-radix path.
#[derive(Clone)]
pub(crate) struct Qwen35RecurrentSnapshot {
    /// `[num_linear]` gated-delta states (f32), copied verbatim from device.
    pub(crate) gdr: Vec<Vec<f32>>,
    /// `[num_linear]` conv1d rings (bf16), copied verbatim from device.
    pub(crate) conv: Vec<Vec<bf16>>,
}

impl Qwen35RecurrentSnapshot {
    /// Approximate host byte size (for cap accounting).
    #[allow(dead_code)]
    pub(crate) fn host_bytes(&self) -> usize {
        self.gdr.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.conv.iter().map(|v| v.len() * 2).sum::<usize>()
    }

    /// Flatten for the sidecar tier store — exact byte-inverse of
    /// [`Self::from_bytes`]. Header `[num_gdr][num_conv]` (u64 LE), then each vec
    /// `[len:u64][elems...]`. No full-attention KV: restore mirrors the radix
    /// prefix's own device pages.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.host_bytes() + 64);
        buf.extend_from_slice(&(self.gdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.conv.len() as u64).to_le_bytes());
        for gdr in &self.gdr {
            buf.extend_from_slice(&(gdr.len() as u64).to_le_bytes());
            for &x in gdr {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        for conv in &self.conv {
            buf.extend_from_slice(&(conv.len() as u64).to_le_bytes());
            for &x in conv {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        buf
    }

    /// Reconstruct a snapshot from [`Self::to_bytes`] — the exact byte-inverse.
    /// Any short/over-long buffer (corrupt or foreign payload) errors rather than
    /// restore garbage, so the caller falls through to clean recompute.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let take_u64 = |pos: &mut usize| -> Result<u64> {
            let end = pos
                .checked_add(8)
                .ok_or_else(|| anyhow!("recurrent snapshot header overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("recurrent snapshot truncated reading u64 at {pos}"))?;
            *pos = end;
            Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
        };
        let num_gdr = take_u64(&mut pos)? as usize;
        let num_conv = take_u64(&mut pos)? as usize;
        let mut gdr = Vec::with_capacity(num_gdr);
        for _ in 0..num_gdr {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 4)
                .ok_or_else(|| anyhow!("recurrent snapshot gdr length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("recurrent snapshot truncated reading gdr state"))?;
            gdr.push(
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
            pos = end;
        }
        let mut conv = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 2)
                .ok_or_else(|| anyhow!("recurrent snapshot conv length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("recurrent snapshot truncated reading conv state"))?;
            conv.push(
                raw.chunks_exact(2)
                    .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            );
            pos = end;
        }
        ensure!(
            pos == bytes.len(),
            "recurrent snapshot has {} trailing bytes after deserialize",
            bytes.len() - pos
        );
        Ok(Self { gdr, conv })
    }
}

/// FNV-1a hash of a token id slice — used to key the recurrent sidecar.
pub(crate) fn hash_prefix_tokens(tokens: &[u32]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &t in tokens {
        let bytes = t.to_le_bytes();
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

impl Qwen35SlotState {
    /// A fresh idle slot: allocate NOTHING. The recurrent state (~147 MiB) is a
    /// fixed-size per-request state — not token-addressable like the paged
    /// full-attn KV — so it draws from a request-grained free-list pool
    /// ([`RecurrentBlock`]) lazily on activation ([`Self::acquire_recurrent`]),
    /// not upfront per `num_slots`. Idle slots cost zero recurrent HBM; the win
    /// is partial-load footprint + the foundation for future L2 spill. At full
    /// concurrency the pool grows to `num_slots`, identical to the old upfront
    /// reservation. `k_caches`/`v_caches` stay empty (full-attn KV is paged).
    pub(crate) fn new_linear_only() -> Self {
        Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            gdr_states: Vec::new(),
            conv_states: Vec::new(),
            recurrent_acquired: false,
            seq_len: 0,
        }
    }

    /// Activate this slot's recurrent state for a fresh request (the
    /// `start_pos == 0` prefill): pop a reusable block from `pool` (free-list
    /// reuse, no alloc churn) or allocate fresh, then ZERO it. Idempotent —
    /// no-op if already allocated, so a chunked-prefill's later chunks
    /// (`start_pos > 0`) never re-zero and wipe the prefix's recurrent state;
    /// the zero happens ONLY on fresh acquisition. MUST run before any forward
    /// reads `gdr_states`.
    pub(crate) fn acquire_recurrent(
        &mut self,
        ctx: &DeviceContext,
        num_linear: usize,
        gdr_state_len: usize,
        conv_len: usize,
        pool: &mut Vec<RecurrentBlock>,
    ) -> Result<()> {
        if !self.gdr_states.is_empty() {
            self.recurrent_acquired = true;
            return Ok(()); // already active (a chunked-prefill continuation)
        }
        let (gdr, conv) = match pool.pop() {
            Some(block) => block,
            None => (0..num_linear)
                .map(|_| {
                    let g = ctx
                        .stream
                        .alloc_zeros::<f32>(gdr_state_len)
                        .map_err(|e| anyhow!("alloc gated-delta state failed: {e}"))?;
                    Ok((g, DeviceVec::zeros(ctx, conv_len)?))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip::<_, _, Vec<_>, Vec<_>>(),
        };
        self.gdr_states = gdr;
        self.conv_states = conv;
        self.recurrent_acquired = true;
        self.seq_len = 0;
        // Zero on acquisition (a pooled block carries the prior occupant's
        // state; a fresh alloc is already zero but re-zeroing is cheap/uniform).
        self.zero_recurrent(ctx)
    }

    /// Return this slot's recurrent block to the free-list, leaving the slot's
    /// fields empty. Called ONLY at request finish (the slot's prior occupant is
    /// fully done — no in-flight forward references it), so the block is safe to
    /// hand to the next request.
    pub(crate) fn release_recurrent(&mut self, pool: &mut Vec<RecurrentBlock>) {
        self.recurrent_acquired = false;
        if self.gdr_states.is_empty() {
            return;
        }
        let gdr = std::mem::take(&mut self.gdr_states);
        let conv = std::mem::take(&mut self.conv_states);
        pool.push((gdr, conv));
        self.seq_len = 0;
    }

    /// Zero the recurrent + conv-ring state in place (the per-request fresh
    /// start). Does not touch the full-attn cursor. No-op when unallocated.
    fn zero_recurrent(&mut self, ctx: &DeviceContext) -> Result<()> {
        for s in &mut self.gdr_states {
            ctx.stream
                .memset_zeros(s)
                .map_err(|e| anyhow!("memset gated-delta state failed: {e}"))?;
        }
        for c in &mut self.conv_states {
            ctx.stream
                .memset_zeros(&mut c.data)
                .map_err(|e| anyhow!("memset conv state failed: {e}"))?;
        }
        Ok(())
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Whether this slot's recurrent state is resident (acquired). A forward
    /// that reads `gdr_states` MUST see this true — a false here means an
    /// `acquire_recurrent` hook was missed at the request's `start_pos == 0`.
    pub(crate) fn has_recurrent(&self) -> bool {
        self.recurrent_acquired
    }

    /// D2H snapshot of the current recurrent state. A full-attn-only model
    /// (`num_linear == 0`) has empty `gdr_states`/`conv_states` but still reaches
    /// the prefix-cache sidecar snapshot path — return an empty snapshot (no
    /// device work) so `restore_recurrent_from_snapshot` (0==0 dims, no-op zips)
    /// stays consistent.
    pub(crate) fn snapshot_recurrent(
        &self,
        ctx: &DeviceContext,
    ) -> Result<Qwen35RecurrentSnapshot> {
        if self.gdr_states.is_empty() {
            return Ok(Qwen35RecurrentSnapshot {
                gdr: Vec::new(),
                conv: Vec::new(),
            });
        }
        let gdr = self
            .gdr_states
            .iter()
            .map(|s| {
                ctx.stream
                    .clone_dtoh(s)
                    .map_err(|e| anyhow!("gdr D2H failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv = self
            .conv_states
            .iter()
            .map(|c| {
                ctx.stream
                    .clone_dtoh(&c.data)
                    .map_err(|e| anyhow!("conv D2H failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        ctx.stream
            .synchronize()
            .map_err(|e| anyhow!("sync after recurrent snapshot: {e}"))?;
        Ok(Qwen35RecurrentSnapshot { gdr, conv })
    }

    /// H2D restore from a sidecar snapshot. The slot MUST have acquired recurrent
    /// buffers before calling (call `acquire_recurrent` first). Errors if
    /// dims mismatch (stale snapshot from a different checkpoint).
    pub(crate) fn restore_recurrent_from_snapshot(
        &mut self,
        ctx: &DeviceContext,
        snap: &Qwen35RecurrentSnapshot,
    ) -> Result<()> {
        ensure!(
            snap.gdr.len() == self.gdr_states.len() && snap.conv.len() == self.conv_states.len(),
            "recurrent sidecar dim mismatch: snapshot gdr={}/conv={} vs slot gdr={}/conv={}",
            snap.gdr.len(),
            snap.conv.len(),
            self.gdr_states.len(),
            self.conv_states.len()
        );
        for (s, h) in self.gdr_states.iter_mut().zip(&snap.gdr) {
            ctx.stream
                .memcpy_htod(h, s)
                .map_err(|e| anyhow!("gdr H2D restore failed: {e}"))?;
        }
        for (c, h) in self.conv_states.iter_mut().zip(&snap.conv) {
            ctx.stream
                .memcpy_htod(h, &mut c.data)
                .map_err(|e| anyhow!("conv H2D restore failed: {e}"))?;
        }
        ctx.stream
            .synchronize()
            .map_err(|e| anyhow!("sync after recurrent restore: {e}"))?;
        Ok(())
    }

    /// Advance the materialized length by `n` tokens. The captured decode
    /// graph's caller uses this: the graph body
    /// ([`Qwen35Model::forward_decode_step_captured`]) is host-state-free —
    /// replay re-launches only GPU work — so the host-side length advance
    /// happens exactly once per step at the call site, never inside the
    /// captured closure.
    pub(crate) fn advance_seq_len(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Snapshot the linear-attention recurrent + conv-ring state into caller
    /// scratch, before a speculative verify pass. The 48 gated-delta layers
    /// advance their state **in place, content-based, no position index**
    /// (`gated_delta_rule_decode_cuda` / `_prefill_recurrent_cuda`), so they do
    /// NOT self-heal under a seq_len rewind — they must be restored on reject.
    /// The full-attn K/V caches are position-indexed and self-heal via the
    /// rewind, so they are intentionally not copied here.
    pub(crate) fn snapshot_linear_into(
        &self,
        ctx: &DeviceContext,
        gdr_snap: &mut [CudaSlice<f32>],
        conv_snap: &mut [DeviceVec],
    ) -> Result<()> {
        ensure!(
            gdr_snap.len() == self.gdr_states.len() && conv_snap.len() == self.conv_states.len(),
            "spec snapshot scratch sized {}/{} != slot linear layers {}/{}",
            gdr_snap.len(),
            conv_snap.len(),
            self.gdr_states.len(),
            self.conv_states.len()
        );
        for (dst, src) in gdr_snap.iter_mut().zip(self.gdr_states.iter()) {
            ctx.stream
                .memcpy_dtod(src, dst)
                .map_err(|e| anyhow!("snapshot gated-delta state failed: {e}"))?;
        }
        for (dst, src) in conv_snap.iter_mut().zip(self.conv_states.iter()) {
            ctx.stream
                .memcpy_dtod(&src.data, &mut dst.data)
                .map_err(|e| anyhow!("snapshot conv state failed: {e}"))?;
        }
        Ok(())
    }

    /// Restore the linear-attention recurrent + conv-ring state from a snapshot
    /// taken by [`Self::snapshot_linear_into`] (speculative verify rejected).
    pub(crate) fn restore_linear_from(
        &mut self,
        ctx: &DeviceContext,
        gdr_snap: &[CudaSlice<f32>],
        conv_snap: &[DeviceVec],
    ) -> Result<()> {
        ensure!(
            gdr_snap.len() == self.gdr_states.len() && conv_snap.len() == self.conv_states.len(),
            "spec restore scratch sized {}/{} != slot linear layers {}/{}",
            gdr_snap.len(),
            conv_snap.len(),
            self.gdr_states.len(),
            self.conv_states.len()
        );
        for (dst, src) in self.gdr_states.iter_mut().zip(gdr_snap.iter()) {
            ctx.stream
                .memcpy_dtod(src, dst)
                .map_err(|e| anyhow!("restore gated-delta state failed: {e}"))?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(conv_snap.iter()) {
            ctx.stream
                .memcpy_dtod(&src.data, &mut dst.data)
                .map_err(|e| anyhow!("restore conv state failed: {e}"))?;
        }
        Ok(())
    }

    /// Rewind the full-attn cache cursor (speculative verify accepted `len`
    /// tokens). Stale rows `[len, prev_seq_len)` are position-indexed and get
    /// overwritten by the next real token at that position — no copy needed.
    pub(crate) fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }

    /// Free the per-slot full-attn contiguous K/V caches. Since the shared-paged
    /// migration the DEFAULT build never allocates them (`new_linear_only` leaves
    /// `k_caches`/`v_caches` empty), so this is a no-op there. The linear-attn
    /// recurrent + conv-ring state (`gdr_states`/`conv_states`) is untouched.
    #[allow(dead_code)] // legacy contiguous-lane helper; the paged default never allocates these.
    pub(crate) fn free_full_attn_caches(&mut self) {
        self.k_caches = Vec::new();
        self.v_caches = Vec::new();
    }

    /// Serialize this slot's COMPLETE device state into a host image for G3
    /// whole-slot spill, then FREE the device buffers (pages back to
    /// `full_attn_kv`, recurrent block back to `pool`). The engine frees the slot
    /// right after `demote_slot`, so the trailing `ctx.sync()` (inside
    /// `copy_pages_to_host` for pages, explicit here for the recurrent D2H) makes
    /// the host image complete before any device buffer is reused.
    ///
    /// Captured buffers (every device buffer the slot owns, proven complete):
    ///   (a) full-attn KV pages — `copy_pages_to_host`, then drop the mirror
    ///       (the host pool frees them right after `demote_slot`).
    ///   (b) `gdr_states[0..num_linear]` (f32) — `clone_dtoh` each.
    ///   (c) `conv_states[0..num_linear]` (bf16) — `clone_dtoh` each.
    ///   (d) `seq_len`.
    /// Then `release_recurrent` returns the recurrent block to the free-list.
    /// `k_caches`/`v_caches` are asserted empty (paged default) — not captured.
    pub(crate) fn swap_out_image(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        full_attn_kv: &mut PagedKVPool,
        recurrent_pool: &mut Vec<RecurrentBlock>,
    ) -> Result<Qwen35SlotImage> {
        ensure!(
            self.k_caches.is_empty() && self.v_caches.is_empty(),
            "Qwen3.6 whole-slot swap requires the paged full-attn default; \
             the legacy contiguous K/V caches are not captured (slot {slot})"
        );
        ensure!(
            self.seq_len == full_attn_kv.seq_len(slot),
            "Qwen3.6 swap-out slot {slot} seq_len {} != pool seq_len {}",
            self.seq_len,
            full_attn_kv.seq_len(slot)
        );
        // (a) Full-attn KV pages (slot-logical order). `copy_pages_to_host` ends
        // in `ctx.sync()`, so the host bytes are complete here.
        let pages = full_attn_kv.page_indices(slot).to_vec();
        let full_attn_pages = full_attn_kv.copy_pages_to_host(ctx, &pages)?;
        let full_attn_page_count = pages.len();
        // (b) + (c) recurrent + conv D2H (stream-ordered; sync below covers them).
        let gdr_host = self
            .gdr_states
            .iter()
            .map(|s| {
                ctx.stream
                    .clone_dtoh(s)
                    .map_err(|e| anyhow!("Qwen3.6 swap gdr-state D2H failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_host = self
            .conv_states
            .iter()
            .map(|c| {
                ctx.stream
                    .clone_dtoh(&c.data)
                    .map_err(|e| anyhow!("Qwen3.6 swap conv-state D2H failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        // The clone_dtoh copies above are stream-ordered; drain before the host
        // image is stored/read.
        ctx.sync()?;
        let image = Qwen35SlotImage {
            full_attn_pages,
            full_attn_page_count,
            gdr_host,
            conv_host,
            seq_len: self.seq_len,
        };
        // Free every device buffer now that the image owns the state.
        full_attn_kv.mirror_slot(slot, &[], 0)?;
        self.release_recurrent(recurrent_pool);
        self.seq_len = 0;
        Ok(image)
    }

    /// Restore a whole-slot image — the exact byte-inverse of
    /// [`Self::swap_out_image`]. Mirror the host pages the engine re-allocated,
    /// acquire a recurrent block, H2D the captured bytes verbatim (the SAME
    /// session restores its OWN state), set `seq_len`. The engine resumes
    /// decode immediately after `promote_slot`, so the trailing `ctx.sync()`
    /// makes the device restore complete before the host image can be dropped.
    pub(crate) fn swap_in_image(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        full_attn_kv: &mut PagedKVPool,
        recurrent_pool: &mut Vec<RecurrentBlock>,
        num_linear: usize,
        gdr_state_len: usize,
        conv_len: usize,
        image: &Qwen35SlotImage,
        slot_pages: &[u32],
    ) -> Result<()> {
        // A scheduler-free slot may still hold its FINISHED previous occupant's
        // device state: this arm vacates lazily at the next position-0 prefill,
        // and a swap re-admission is exactly such a fresh occupancy. Release the
        // stale state the same way `submit_prefill_row` does (#134 — the old
        // empty-slot ensure here cost one graceful recompute per rotation pair).
        if self.has_recurrent() {
            self.release_recurrent(recurrent_pool);
        }
        if full_attn_kv.seq_len(slot) != 0 {
            full_attn_kv.mirror_slot(slot, &[], 0)?;
        }
        self.seq_len = 0;
        ensure!(
            image.gdr_host.len() == num_linear && image.conv_host.len() == num_linear,
            "Qwen3.6 swap image linear count {}/{} != num_linear {num_linear}",
            image.gdr_host.len(),
            image.conv_host.len()
        );
        // (a) Mirror the captured page count, H2D the full-attn bytes.
        ensure!(
            slot_pages.len() == image.full_attn_page_count,
            "Qwen3.6 swap-in host slot holds {} pages != captured {}",
            slot_pages.len(),
            image.full_attn_page_count
        );
        full_attn_kv.mirror_slot(slot, slot_pages, image.seq_len)?;
        full_attn_kv.copy_pages_from_host(ctx, slot_pages, &image.full_attn_pages)?;
        // (b) + (c) acquire a fresh recurrent block (alloc+zero) then H2D-restore.
        self.acquire_recurrent(ctx, num_linear, gdr_state_len, conv_len, recurrent_pool)?;
        for (dst, src) in self.gdr_states.iter_mut().zip(&image.gdr_host) {
            ctx.stream
                .memcpy_htod(src, dst)
                .map_err(|e| anyhow!("Qwen3.6 swap gdr-state H2D failed: {e}"))?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(&image.conv_host) {
            ctx.stream
                .memcpy_htod(src, &mut dst.data)
                .map_err(|e| anyhow!("Qwen3.6 swap conv-state H2D failed: {e}"))?;
        }
        // (d) materialized length (both the slot and the pool's allocator agree).
        self.seq_len = image.seq_len;
        // H2D complete before the host image can be dropped (matches DSv4).
        ctx.sync()?;
        Ok(())
    }
}

/// Per-request speculative-decode state for the Qwen3.6 NextN-MTP draft head.
///
/// Holds (a) the draft head's **fresh per-block** K/V cache — the head is a
/// single full-attention layer that attends only over the current draft chain,
/// seeded each block from the last-accepted trunk hidden (the trunk context is
/// baked into that hidden via the `fc` concat, not re-attended); and (b) the
/// pre-verify **snapshot** of the trunk's linear-attn recurrent + conv state,
/// restored on a rejected draft. Allocated only when spec-decode is on, one per
/// concurrent slot, so the baseline decode path never pays for it.
#[allow(dead_code)] // head_k/head_v read by mtp_forward_level; snap by spec_step (next increments)
pub(crate) struct Qwen35SpecSlotState {
    /// Draft head K cache `(depth+1)*kv_dim` bf16, rewritten each draft block.
    head_k: DeviceVec,
    head_v: DeviceVec,
    /// Pre-verify snapshot of the trunk linear-attn recurrent states (f32),
    /// one per linear layer, sized like [`Qwen35SlotState::gdr_states`].
    gdr_snap: Vec<CudaSlice<f32>>,
    /// Pre-verify snapshot of the trunk linear-attn conv rings (bf16).
    conv_snap: Vec<DeviceVec>,
    /// Per-linear-layer capture of the verify forward's gated-delta inputs, for
    /// the cheap partial-accept linear-only replay (see [`Qwen35LinearCapture`]).
    pub(crate) capture: Qwen35LinearCapture,
    /// Persistent 1-element argmax scratch shared by the draft + the two verify
    /// rows, so a spec step performs ZERO per-token argmax allocations and the
    /// verify argmax stays on-device (no full `[seq, vocab]` D2H).
    argmax_scratch: CudaSlice<i32>,
    /// Sampled-mode device buffers (allocated on the first temp>0 spec step
    /// only; greedy never touches them). Mirrors `DsparkScratch`'s sampled
    /// block, sized by the head cap `spec_draft_tokens.max(1)`:
    /// `q_probs [cap, vocab] f32` draft filtered dists (row `level` fully
    /// written by `dspark_draft_sample_cuda` before the chain kernel reads it);
    /// `p_probs [cap+1, vocab] f32` verify filtered dists (leading `depth+1`
    /// rows fully written per accept; the stale tail is never read);
    /// `sample_tok [1]` / `accept_out [2]` fully written before D2H;
    /// `chain_draft [cap]` / `u_accept [cap]` / `u_residual [cap+1]`
    /// host-uploaded prefixes — the kernel reads only the uploaded prefix.
    q_probs: SliceSlot<f32>,
    p_probs: SliceSlot<f32>,
    sample_tok: SliceSlot<i32>,
    accept_out: SliceSlot<i32>,
    chain_draft: SliceSlot<i32>,
    u_accept: SliceSlot<f32>,
    u_residual: SliceSlot<f32>,
}

/// Per-linear-layer capture of the gated-delta-rule inputs from the spec verify
/// forward, sized for the full `depth+1`-row chain — the substrate for the
/// cheap partial-accept replay.
///
/// On a partial accept (`k < depth`) the trunk linear state must be left at the
/// post-`[pending, d1..dk]` position. The old path re-ran a FULL `depth+1`-wide
/// trunk forward (`forward_hidden`) over the accepted prefix purely for that
/// recurrent side-effect (21-47 ms per macro-step on H20 real-fp8). The state
/// the GDR + conv1d kernels advance is a pure function of their per-layer inputs
/// (the post-in_proj `qkv` PRE-conv1d, plus the `b`/`a` gate projections); those
/// inputs already encode the full-stack residual because the verify produced
/// them with the real trunk. So instead of recomputing them we **cache them
/// during verify** and re-run ONLY conv1d + the recurrent GDR over rows
/// `[0..=k]` on a partial accept — bit-identical to the verify's first `k+1`
/// recurrent steps (same kernels, same inputs, same in-place math), skipping
/// every full-attn block, every MLP/MoE, the final norm, and the lm_head.
///
/// All three caches are token-major `[(depth+1), width]` bf16 (token `t` at
/// offset `t*width`), so rows `[0..=k]` slice contiguously as `[0..(k+1)*width]`.
/// Allocated only with the spec state, so the baseline decode path never pays.
#[allow(dead_code)] // populated by linear_attention under capture; read by replay_linear_only
/// Pointer/length staging for [`Qwen35Model::batched_copy`].
#[derive(Default)]
pub(crate) struct Qwen35CopyScratch {
    ptrs: SliceSlot<u64>,
    lens: SliceSlot<i32>,
    host: Vec<u64>,
    hlen: Vec<i32>,
}

/// Per-slot buffer addresses the varlen replay kernels index, one `[B]` table
/// per (kind, linear layer), flat so staging is one H2D. Re-staged every tick:
/// the accepted set changes.
#[derive(Default)]
pub(crate) struct Qwen35ReplayTables {
    ptrs: SliceSlot<u64>,
    row_len: SliceSlot<i32>,
    host: Vec<u64>,
    layout: ReplayLayout,
}

/// Where each table sits in the flat staging buffer — one definition, used by
/// the host writer and the device reader.
#[derive(Clone, Copy, Default)]
struct ReplayLayout {
    base: u64,
    stride: usize,
    batch: usize,
}

impl ReplayLayout {
    fn at(&self, kind: usize, li: usize) -> usize {
        kind * self.stride + li * self.batch
    }

    /// Device address of table `kind`'s `[B]` row for linear layer `li`.
    fn table(&self, kind: usize, li: usize) -> u64 {
        self.base + (self.at(kind, li) as u64) * 8
    }
}

/// [`Qwen35ReplayTables`] kinds, in layout order.
const TBL_QKV: usize = 0;
const TBL_B: usize = 1;
const TBL_A: usize = 2;
const TBL_CONV: usize = 3;
const TBL_GDR: usize = 4;
const REPLAY_TABLES: usize = 5;

impl Qwen35ReplayTables {
    fn stage(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [&mut Qwen35SlotState],
        captures: &[&Qwen35LinearCapture],
        ks: &[usize],
        num_linear: usize,
    ) -> Result<()> {
        let b = slots.len();
        let lay = ReplayLayout {
            base: 0,
            stride: num_linear * b,
            batch: b,
        };
        self.layout = lay;
        self.host.clear();
        self.host.resize(REPLAY_TABLES * lay.stride, 0);
        for li in 0..num_linear {
            for (s, slot) in slots.iter_mut().enumerate() {
                let mut put = |kind: usize, addr: u64| self.host[lay.at(kind, li) + s] = addr;
                put(TBL_QKV, captures[s].qkv[li].data.device_ptr(&ctx.stream).0);
                put(TBL_B, captures[s].b_proj[li].data.device_ptr(&ctx.stream).0);
                put(TBL_A, captures[s].a_proj[li].data.device_ptr(&ctx.stream).0);
                put(
                    TBL_CONV,
                    slot.conv_states[li].data.device_ptr_mut(&ctx.stream).0,
                );
                put(TBL_GDR, slot.gdr_states[li].device_ptr_mut(&ctx.stream).0);
            }
        }
        let dst = self.ptrs.get(ctx, self.host.len())?;
        ctx.stream
            .memcpy_htod(&self.host, dst)
            .map_err(|e| anyhow!("H2D replay pointer tables: {e}"))?;
        let lens: Vec<i32> = ks.iter().map(|k| (k + 1) as i32).collect();
        let dst = self.row_len.get(ctx, b)?;
        ctx.stream
            .memcpy_htod(&lens, dst)
            .map_err(|e| anyhow!("H2D replay row lengths: {e}"))?;
        Ok(())
    }
}

pub(crate) struct Qwen35LinearCapture {
    /// Number of layers (== `num_linear`); the per-row stride is each buffer's
    /// `len / (depth+1)`.
    rows: usize,
    /// Post-in_proj fused `[q|k|v]` (PRE-conv1d) for all `depth+1` rows, one per
    /// linear layer; feeds `conv1d_prefill_cuda` on replay.
    qkv: Vec<DeviceVec>,
    /// `in_proj_b` projection (one scalar per local v-head) for all rows.
    b_proj: Vec<DeviceVec>,
    /// `in_proj_a` projection (one scalar per local v-head) for all rows.
    a_proj: Vec<DeviceVec>,
}

#[allow(dead_code)] // consumed by mtp_forward_level + spec_step (next increments)
impl Qwen35SpecSlotState {
    /// Snapshot the trunk linear state before a verify pass (reject rollback).
    pub(crate) fn snapshot_trunk(
        &mut self,
        ctx: &DeviceContext,
        slot: &Qwen35SlotState,
    ) -> Result<()> {
        slot.snapshot_linear_into(ctx, &mut self.gdr_snap, &mut self.conv_snap)
    }

    /// Restore the trunk linear state after a rejected verify pass.
    pub(crate) fn restore_trunk(
        &self,
        ctx: &DeviceContext,
        slot: &mut Qwen35SlotState,
    ) -> Result<()> {
        slot.restore_linear_from(ctx, &self.gdr_snap, &self.conv_snap)
    }

    /// Append this slot's `(snapshot, live)` linear-state addresses — gdr into
    /// `gdr`, conv into `conv`. The caller picks the direction and issues one
    /// batched copy for the whole speculative batch.
    pub(crate) fn linear_state_addrs(
        &mut self,
        ctx: &DeviceContext,
        slot: &mut Qwen35SlotState,
        bytes: (usize, usize),
        gdr: &mut (Vec<u64>, Vec<u64>),
        conv: &mut (Vec<u64>, Vec<u64>),
    ) -> Result<()> {
        ensure!(
            self.gdr_snap.len() == slot.gdr_states.len()
                && self.conv_snap.len() == slot.conv_states.len(),
            "spec snapshot scratch sized {}/{} != slot linear layers {}/{}",
            self.gdr_snap.len(),
            self.conv_snap.len(),
            slot.gdr_states.len(),
            slot.conv_states.len()
        );
        // The batched copy takes a size, not a slice — every pair must be it.
        ensure!(
            self.gdr_snap.iter().all(|b| b.len() * 4 == bytes.0)
                && slot.gdr_states.iter().all(|b| b.len() * 4 == bytes.0)
                && self.conv_snap.iter().all(|b| b.len * 2 == bytes.1)
                && slot.conv_states.iter().all(|b| b.len * 2 == bytes.1),
            "spec linear state buffers are not uniformly {}/{} bytes",
            bytes.0,
            bytes.1
        );
        for (s, l) in self.gdr_snap.iter_mut().zip(slot.gdr_states.iter_mut()) {
            gdr.0.push(s.device_ptr_mut(&ctx.stream).0);
            gdr.1.push(l.device_ptr_mut(&ctx.stream).0);
        }
        for (s, l) in self.conv_snap.iter_mut().zip(slot.conv_states.iter_mut()) {
            conv.0.push(s.data.device_ptr_mut(&ctx.stream).0);
            conv.1.push(l.data.device_ptr_mut(&ctx.stream).0);
        }
        Ok(())
    }

    /// Mutable access to the persistent 1-element argmax scratch (the warm step
    /// seeds the spec state with the greedy pending token).
    pub(crate) fn argmax_scratch_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.argmax_scratch
    }
}

/// Persistent device workspace for the Qwen3.5/3.6 hybrid forward.
///
/// The forward used to perform ~425 fresh device allocations PER CALL
/// (3 `HiddenStates::zeros` per layer in the residual stream, ~6 per
/// attention block, ~10 per MoE layer, plus the sampling tail) — all fully
/// overwritten before any read, per decode token. This workspace gives every
/// such buffer a persistent exact-shape slot (see [`crate::workspace`] for the
/// reuse/zeroing/free-safety contract and why the cache is exact-shape rather
/// than capacity-sized: `all_reduce_sum` derives the TP collective message
/// length — and the one-shot-vs-NCCL choice — from `data.len()`, so oversized
/// buffers would change the reduction on TP>=2).
///
/// One workspace per executor (forwards are strictly serial); passed `&mut`
/// from the executor like the DSv4 decode scratch pools. Numerics are
/// byte-identical to the per-call path: same buffer sizes, same kernel
/// arguments, same launch order — only the allocation lifetime changes.
///
/// Write-before-read proof per slot (kernels verified in
/// `csrc/misc/elementwise_basic.cu`, `csrc/misc/norm.cu`,
/// `csrc/misc/conv1d.cu`, `csrc/misc/gated_delta_rule.cu`,
/// `csrc/attention/prefill_attention_hd256.cu`,
/// `csrc/attention/nonpaged_prefill_attention.cu`, `csrc/gemm/gemv.cu`;
/// MoE slots carry their own table on [`MoeForwardScratch`]):
///
/// | slot          | shape           | proof of full overwrite before read |
/// |---------------|-----------------|--------------------------------------|
/// | `token_ids`   | `[S]` i32       | H2D upload overwrites every call |
/// | `start_pos`   | `[1]` i32       | H2D upload overwrites every call |
/// | `hidden`      | `[H, S]`        | `embedding_batched` writes all `H*S`; per layer, the residual `add_cuda` rewrites all `H*S` before the next read |
/// | `normed`      | `[H, S]`        | `rms_norm_batched_offset` writes all rows (grid = S blocks × full row) |
/// | `hidden_mid`  | `[H, S]`        | `add_cuda` writes all `H*S` |
/// | `attn_out`    | `[H, S]`        | o/out-proj `gemm_cuda` is beta=0 (writes all `M*N`) |
/// | `mlp_out`     | `[H, S]`        | dense: down `gemm_cuda` beta=0; MoE: combine kernel writes all, then gated-add RMWs |
/// | `full.q_full` | `[2*Hq*D, S]`   | `gemm_cuda` beta=0 |
/// | `full.k_batch`/`v_batch` | `[Hkv*D, S]` | `gemm_cuda` beta=0 |
/// | `full.q_prepped` | `[Hq*D, S]`  | full-attn prep writes every (token, head, d): RoPE pair covers `[0, rotary)`, the `d >= rotary` branch covers the rest |
/// | `full.attn_heads`| `[Hq*D, S]`  | nonpaged attention writes every (token, head, d) from register accumulators; the gate kernel then RMWs in place |
/// | `linear.qkv`/`z`/`b_proj`/`a_proj` | `[*, S]` | `gemm_cuda` beta=0 |
/// | `linear.qkv_conv` | `[QKV, S]`  | conv1d prefill writes every (channel, t) |
/// | `linear.gdr_out`  | `[Vh*Vd, S]`| gated-delta kernels write every (token, v_head, d) (one block per v_head, j_slice 0 writes the reduced output) |
/// | `linear.normed_out`| `[Vh*Vd, S]`| `rms_norm_gated` writes every (head, d) over `Vh*S` blocks |
/// | `dense.gate_up` | `[2I, S]`     | `gemm_cuda` beta=0 (row-fused `[gate; up]`) |
/// | `dense.act`   | `[I, S]`        | `silu_mul_fused` writes all `I*S` |
/// | `last_hidden` | `[H]`           | `memcpy_dtod` overwrites the full vec |
/// | `last_normed` | `[H]`           | `rms_norm_offset` writes all `H` |
/// | `logits`      | `[V]`           | `gemv` writes every output row |
/// | `argmax_out`  | `[1]` i32       | argmax kernel writes it before the D2H read (sampling tail, never captured) |
///
/// The full-logits OPD tail (`[V, S]`) is RETURNED to the caller and stays a
/// per-call allocation by design.
#[derive(Default)]
pub(crate) struct Qwen35Workspace {
    token_ids: SliceSlot<i32>,
    /// GPU-resident `start_pos` for the full-attn prep kernel — uploaded once per
    /// forward (the value is identical for every full-attn layer in the call;
    /// the old path uploaded one identical buffer per layer). The decode graph
    /// also reads it from the devpos attention kernel, so it is the single
    /// per-step position scalar staged pre-replay.
    start_pos: SliceSlot<i32>,
    hidden: HiddenSlot,
    normed: HiddenSlot,
    hidden_mid: HiddenSlot,
    attn_out: HiddenSlot,
    mlp_out: HiddenSlot,
    full: FullAttnScratch,
    linear: LinearAttnScratch,
    dense: DenseMlpScratch,
    moe: MoeForwardScratch,
    last_hidden: VecSlot,
    last_normed: VecSlot,
    logits: VecSlot,
    /// Persistent argmax output (one i32) for the greedy sampling tail —
    /// removes the last steady-state per-token device allocation
    /// (`ops::argmax`'s `alloc_zeros(1)`).
    argmax_out: SliceSlot<i32>,
    /// Buffer-address generation. Bumped whenever cached buffers are dropped
    /// wholesale ([`Self::release`]) — i.e. whenever previously-cached device
    /// ADDRESSES may change on the next `get`. The captured decode graph bakes
    /// buffer addresses, so it records this at capture and recaptures on
    /// mismatch instead of replaying against freed memory.
    epoch: u64,
}

#[derive(Default)]
pub(crate) struct FullAttnScratch {
    /// Fused `[q; k; v]` GEMM output, split into `q_full`/`k_batch`/`v_batch`.
    qkv_fused: HiddenSlot,
    q_full: HiddenSlot,
    k_batch: HiddenSlot,
    v_batch: HiddenSlot,
    q_prepped: HiddenSlot,
    attn_heads: HiddenSlot,
    /// FA3 prefill scratch (`--qwen35-fa3`): fp32 softmax LSE
    /// `[local_q_heads * seq_len]` (write-only output of the fwd kernel) and
    /// the persistent-scheduler semaphore (1 i32, zeroed by the shim per
    /// launch).
    fa3_lse: SliceSlot<f32>,
    /// FA3 split scratch: fp32 partial
    /// outputs `[splits, b=1, local_q_heads, seq_len, head_dim]`.
    fa3_oaccum: SliceSlot<f32>,
    /// FA3 split-decode scratch: fp32 partial LSE
    /// `[splits, b=1, local_q_heads, seq_len]`.
    fa3_lseaccum: SliceSlot<f32>,
    fa3_semaphore: SliceSlot<i32>,
    /// Batched split-KV decode scratch for full attention:
    /// `[B, local_q_heads, QWEN35_BATCHED_DECODE_KV_SPLITS, head_dim]`.
    batch_partial_out: SliceSlot<f32>,
    /// `[B, local_q_heads, QWEN35_BATCHED_DECODE_KV_SPLITS]`.
    batch_partial_m: SliceSlot<f32>,
    batch_partial_l: SliceSlot<f32>,
}

/// Paged full-attn forwarding context for Qwen3.6 — the DEFAULT path since the
/// shared-paged migration. Each full-attn layer reads/writes the shared
/// `PagedKVPool` (`full_attn_kv`) over `meta` (the page table) instead of a
/// per-slot contiguous cache. The default build hands a `for_slot` page table
/// over the slot's FULL resident pages (full attention, no eviction); the
/// `--kv-recall` cycle layers a working-set restriction on top of the SAME
/// pool. `Some` on `layer0_query` opts into the layer-0 post-RoPE query
/// readback for the recall score — a mid-forward D2H, so only the recall
/// prefill asks for it.
pub(crate) struct Qwen35RecallForward<'a> {
    pub(crate) pool: &'a mut PagedKVPool,
    pub(crate) meta: &'a crate::loader::PageMeta,
    pub(crate) layer0_query: Option<Vec<f32>>,
}

#[derive(Default)]
pub(crate) struct LinearAttnScratch {
    /// Staging for the batched spec-capture copies.
    capture_copy: Qwen35CopyScratch,
    /// Fused `[qkv; z]` GEMM output, split into `qkv`/`z`.
    qkvz: HiddenSlot,
    qkv: HiddenSlot,
    z: HiddenSlot,
    /// Fused `[b; a]` GEMM output (`[2*Vh, S]`), split into `b_proj`/`a_proj`.
    ba: HiddenSlot,
    b_proj: HiddenSlot,
    a_proj: HiddenSlot,
    qkv_conv: HiddenSlot,
    gdr_out: HiddenSlot,
    normed_out: HiddenSlot,
    /// FlashQLA chunked-prefill scratch (`--qwen35-gdr-chunked`), all
    /// token-major dense: q/k `[S, Hg, 128]` bf16 l2norm'd, v `[S, H, 128]`
    /// bf16, a_inv `[S, H, 64]` bf16, g/g_cumsum/beta `[S, H]` f32.
    fq_q: HiddenSlot,
    fq_k: HiddenSlot,
    fq_v: HiddenSlot,
    fq_a: HiddenSlot,
    fq_g: SliceSlot<f32>,
    fq_g_cumsum: SliceSlot<f32>,
    fq_beta: SliceSlot<f32>,
}

/// One slot's contiguous column range in a ragged batch. Its `len` token-major
/// columns advance THIS slot's state; its capture receives them from offset 0.
pub(crate) struct LinearRow<'a> {
    pub(crate) slot: &'a mut Qwen35SlotState,
    pub(crate) len: usize,
    pub(crate) capture: Option<&'a mut Qwen35LinearCapture>,
}

/// How [`Qwen35Model::linear_attention`] reaches per-slot conv + recurrent
/// state. Everything else in the layer runs once over all columns.
pub(crate) enum LinearCore<'a, 'r> {
    /// Ragged `B×T`: one single-slot multi-token launch per row.
    Rows(&'a mut [LinearRow<'r>]),
    /// Pure decode, one token per row: staged pointer tables advance all B
    /// states in ONE conv + ONE GDR launch. Row `r`'s channels sit at `r*C`;
    /// conv `[C, K-1]` and GDR `[Vh, Kd, Vd]` match the single-slot layout.
    Tables {
        conv: &'a CudaSlice<u64>,
        gdr: &'a CudaSlice<u64>,
    },
}

#[derive(Default)]
pub(crate) struct DenseMlpScratch {
    gate_up: HiddenSlot,
    act: HiddenSlot,
}

impl Qwen35Workspace {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Buffer-address generation (see the `epoch` field).
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Drop every cached buffer (frees the VRAM). Called by the executor after
    /// the OPD weight offload so the workspace does not hold prefill-shaped
    /// scratch while the student backward needs the headroom. The caller must
    /// have quiesced the device first (`offload_engine_weights` syncs).
    /// Bumps the address epoch: any captured decode graph over these buffers
    /// is stale after this.
    pub(crate) fn release(&mut self) {
        self.epoch += 1;
        let Self {
            token_ids,
            start_pos,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            last_hidden,
            last_normed,
            logits,
            argmax_out,
            epoch: _,
        } = self;
        token_ids.release();
        start_pos.release();
        hidden.release();
        normed.release();
        hidden_mid.release();
        attn_out.release();
        mlp_out.release();
        full.q_full.release();
        full.k_batch.release();
        full.v_batch.release();
        full.q_prepped.release();
        full.attn_heads.release();
        full.fa3_lse.release();
        full.fa3_oaccum.release();
        full.fa3_lseaccum.release();
        full.fa3_semaphore.release();
        full.batch_partial_out.release();
        full.batch_partial_m.release();
        full.batch_partial_l.release();
        linear.qkv.release();
        linear.z.release();
        linear.b_proj.release();
        linear.a_proj.release();
        linear.qkv_conv.release();
        linear.gdr_out.release();
        linear.normed_out.release();
        dense.gate_up.release();
        dense.act.release();
        moe.release();
        last_hidden.release();
        last_normed.release();
        logits.release();
        argmax_out.release();
    }
}

/// Persistent device state for the rows>1 BATCHED DECODE path (stage 1:
/// contiguous per-slot KV kept, no paged migration). Re-port of the deleted
/// monolith's proven design (`e81b98fb~1` `infer/src/model/qwen35/batch_decode.rs`,
/// `BatchDecodeBuffers35` + per-layer pointer tables, lines 87-89/780-815)
/// onto the rewrite's workspace slots, using DSv4's batched-decode executor
/// shape (`dsv4.rs` Step-A per-row attention) as the template.
///
/// Owns a DEDICATED forward workspace: it only ever sees `[*, B]` decode
/// shapes, so the main (prefill-reshaping) workspace never thrashes, and —
/// critically — every buffer that feeds `TpRuntime::all_reduce_sum` is an
/// EXACT-shape `[dim, B]` allocation. `all_reduce_sum` derives the collective
/// message length (and the one-shot-vs-NCCL choice) from `data.len()`
/// (see `workspace.rs`), so exact-shape buffers make the reduced message
/// exactly B valid columns BY CONSTRUCTION — no capacity tail of stale
/// columns can ever enter a reduction. (Deviation from the monolith's
/// capacity-sized `set_batch_size` buffers, deliberate: the monolith had a
/// length-honest collective API; the rewrite's does not.)
///
/// Pointer tables are capacity-sized (`[num_slots]` u64 per linear layer per
/// kind) and uploaded with B valid entries; the batch kernels read entries
/// `[0, gridDim.y)` = `[0, B)` only, so the dead tail is never dereferenced.
/// Tables are a pure function of `(slot_indices, layer)`: the per-slot conv
/// ring / GDR state `CudaSlice`s are allocated once at executor construction
/// and never re-allocated (`Qwen35SlotState::reset` memsets in place; the OPD
/// weight offload leaves slot state untouched), so restaging is needed only
/// when the row→slot mapping changes (monolith `TileLangDecodeMetadata.update`
/// pattern).
pub(crate) struct Qwen35BatchDecodeState {
    /// Dedicated `[*, B]`-shaped forward workspace (see struct docs).
    ws: Qwen35Workspace,
    /// Per-row absolute start positions (`[B]` i32), staged once per step;
    /// batched full-attention kernels read the device array directly.
    positions: SliceSlot<i32>,
    /// Per-row KV lengths after appending the decode token (`positions + 1`).
    seq_lens: SliceSlot<i32>,
    /// Per-FULL-layer `[num_slots]` u64 device tables of K/V cache pointers
    /// (`Half** [B] -> [local_kv_heads, max_seq_len, head_dim]`). The batched
    /// full-attn kernel writes the current row and reads the prefix through
    /// these pointers.
    full_k_cache_ptrs: Vec<CudaSlice<u64>>,
    full_v_cache_ptrs: Vec<CudaSlice<u64>>,
    /// Per-LINEAR-layer `[num_slots]` u64 device tables of conv-ring pointers
    /// (`Half** [B] -> [C, K-1]` as `conv1d_decode_batch_cuda` consumes them).
    conv_state_ptrs: Vec<CudaSlice<u64>>,
    /// Per-LINEAR-layer `[num_slots]` u64 device tables of GDR-state pointers
    /// (`float** [B] -> [Vh, Kd, Vd]` as `gdr_decode_batch_cuda` consumes them).
    gdr_state_ptrs: Vec<CudaSlice<u64>>,
    /// Host staging vecs for the table uploads (monolith pattern: one
    /// `memcpy_htod` per layer per table, no per-row H2D).
    conv_host: Vec<u64>,
    gdr_host: Vec<u64>,
    full_k_host: Vec<u64>,
    full_v_host: Vec<u64>,
    /// Row→slot mapping the tables were last staged for (empty = never).
    staged_slot_indices: Vec<usize>,
    /// Batched logits `[vocab, B]` (final norm + lm_head GEMM over all rows).
    logits_batch: HiddenSlot,
    /// Batched greedy argmax outputs `[B]` i32.
    argmax: SliceSlot<i32>,
}

impl Qwen35BatchDecodeState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_full_layers: usize,
        num_linear_layers: usize,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!(
            max_batch > 0,
            "Qwen3.5 batched decode requires max_batch > 0"
        );
        let (full_k_cache_ptrs, full_v_cache_ptrs) =
            (0..num_full_layers)
                .map(|i| {
                    let k = ctx.stream.alloc_zeros::<u64>(max_batch).map_err(|e| {
                        anyhow!("alloc qwen35 batch full_k_cache_ptrs layer {i}: {e}")
                    })?;
                    let v = ctx.stream.alloc_zeros::<u64>(max_batch).map_err(|e| {
                        anyhow!("alloc qwen35 batch full_v_cache_ptrs layer {i}: {e}")
                    })?;
                    Ok((k, v))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip::<_, _, Vec<_>, Vec<_>>();
        let (conv_state_ptrs, gdr_state_ptrs) = (0..num_linear_layers)
            .map(|i| {
                let c = ctx
                    .stream
                    .alloc_zeros::<u64>(max_batch)
                    .map_err(|e| anyhow!("alloc qwen35 batch conv_state_ptrs layer {i}: {e}"))?;
                let g = ctx
                    .stream
                    .alloc_zeros::<u64>(max_batch)
                    .map_err(|e| anyhow!("alloc qwen35 batch gdr_state_ptrs layer {i}: {e}"))?;
                Ok((c, g))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Ok(Self {
            ws: Qwen35Workspace::new(),
            positions: SliceSlot::default(),
            seq_lens: SliceSlot::default(),
            full_k_cache_ptrs,
            full_v_cache_ptrs,
            conv_state_ptrs,
            gdr_state_ptrs,
            conv_host: vec![0u64; max_batch],
            gdr_host: vec![0u64; max_batch],
            full_k_host: vec![0u64; max_batch],
            full_v_host: vec![0u64; max_batch],
            staged_slot_indices: Vec::new(),
            logits_batch: HiddenSlot::default(),
            argmax: SliceSlot::default(),
        })
    }

    /// Re-upload the per-layer state pointer tables iff the row→slot mapping
    /// changed (tables are a pure function of `(slot_indices, layer)`; see
    /// struct docs for why the slot-state addresses themselves are stable).
    fn stage_pointer_tables(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
    ) -> Result<()> {
        if self.staged_slot_indices == slot_indices {
            return Ok(());
        }
        let b = slot_indices.len();
        ensure!(
            b <= self.conv_host.len(),
            "Qwen3.5 batched decode batch {} exceeds table capacity {}",
            b,
            self.conv_host.len()
        );
        for layer_idx in 0..self.full_k_cache_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.k_caches.len() && layer_idx < slot.v_caches.len(),
                    "Qwen3.5 batched decode full-attn layer {layer_idx} outside slot cache \
                     (k={}, v={})",
                    slot.k_caches.len(),
                    slot.v_caches.len()
                );
                let (k_ptr, _gk) = slot.k_caches[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (v_ptr, _gv) = slot.v_caches[layer_idx].data.device_ptr_mut(&ctx.stream);
                self.full_k_host[r] = k_ptr;
                self.full_v_host[r] = v_ptr;
            }
            ctx.stream
                .memcpy_htod(
                    &self.full_k_host[..b],
                    &mut self.full_k_cache_ptrs[layer_idx],
                )
                .map_err(|e| anyhow!("H2D qwen35 full_k_cache_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(
                    &self.full_v_host[..b],
                    &mut self.full_v_cache_ptrs[layer_idx],
                )
                .map_err(|e| anyhow!("H2D qwen35 full_v_cache_ptrs layer {layer_idx}: {e}"))?;
        }
        for layer_idx in 0..self.conv_state_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.conv_states.len() && layer_idx < slot.gdr_states.len(),
                    "Qwen3.5 batched decode linear layer {layer_idx} outside slot state \
                     (conv={}, gdr={})",
                    slot.conv_states.len(),
                    slot.gdr_states.len()
                );
                let (conv_ptr, _gc) = slot.conv_states[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (gdr_ptr, _gg) = slot.gdr_states[layer_idx].device_ptr_mut(&ctx.stream);
                self.conv_host[r] = conv_ptr;
                self.gdr_host[r] = gdr_ptr;
            }
            ctx.stream
                .memcpy_htod(&self.conv_host[..b], &mut self.conv_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 conv_state_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(&self.gdr_host[..b], &mut self.gdr_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 gdr_state_ptrs layer {layer_idx}: {e}"))?;
        }
        self.staged_slot_indices = slot_indices.to_vec();
        Ok(())
    }

    /// Recurrent-only pointer staging for the PAGED batched-decode lane: stage
    /// the conv-ring + GDR-state tables (linear-attn layers) but SKIP the
    /// contiguous full-attn `k_caches`/`v_caches` tables, which the shared-paged
    /// default never allocates (touching them would deref an empty slice). Paged
    /// full attention reads the shared pool via the per-step `PageMeta` instead,
    /// so the conv/GDR tables are the only per-slot device pointers it needs.
    /// Same `staged_slot_indices` cache key as the contiguous path; both lanes
    /// share the invalidation hook. The two stagers never interleave on one
    /// executor (a build is either paged or contiguous), so the cache key is
    /// unambiguous.
    fn stage_recurrent_pointer_tables(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
    ) -> Result<()> {
        if self.staged_slot_indices == slot_indices {
            return Ok(());
        }
        let b = slot_indices.len();
        ensure!(
            b <= self.conv_host.len(),
            "Qwen3.5 paged batched decode batch {} exceeds table capacity {}",
            b,
            self.conv_host.len()
        );
        for layer_idx in 0..self.conv_state_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.conv_states.len() && layer_idx < slot.gdr_states.len(),
                    "Qwen3.5 paged batched decode linear layer {layer_idx} outside slot state \
                     (conv={}, gdr={})",
                    slot.conv_states.len(),
                    slot.gdr_states.len()
                );
                let (conv_ptr, _gc) = slot.conv_states[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (gdr_ptr, _gg) = slot.gdr_states[layer_idx].device_ptr_mut(&ctx.stream);
                self.conv_host[r] = conv_ptr;
                self.gdr_host[r] = gdr_ptr;
            }
            ctx.stream
                .memcpy_htod(&self.conv_host[..b], &mut self.conv_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 paged conv_state_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(&self.gdr_host[..b], &mut self.gdr_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 paged gdr_state_ptrs layer {layer_idx}: {e}"))?;
        }
        self.staged_slot_indices = slot_indices.to_vec();
        Ok(())
    }

    /// Invalidate the staged pointer-table cache so the next decode batch
    /// restages from the slots' CURRENT recurrent-block addresses. Required at a
    /// request boundary: with the free-list pool a slot's `gdr_states`/
    /// `conv_states` `CudaSlice`s change identity when a new request acquires a
    /// different block (vs the old upfront alloc, where they were fixed for the
    /// executor's lifetime). The cache keys on `slot_indices` alone, so without
    /// this a same-mapping batch would dereference the prior occupant's block.
    pub(crate) fn invalidate_staged_pointers(&mut self) {
        self.staged_slot_indices.clear();
    }

    /// Drop the workspace VRAM (OPD weight time-share hook). The pointer
    /// TABLES and the staged mapping stay: the per-slot state addresses they
    /// hold are executor-owned and untouched by the weight offload, so they
    /// remain valid across an offload/reload cycle (and they are ~KB-scale).
    pub(crate) fn release(&mut self) {
        self.ws.release();
        self.positions.release();
        self.seq_lens.release();
        self.logits_batch.release();
        self.argmax.release();
    }
}

pub(crate) enum Qwen35Attn {
    // Boxed: `FullAttn`/`LinearAttn` are large (multiple DeviceMatrix) and
    // size-skewed; boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<FullAttn>),
    Linear(Box<LinearAttn>),
}

/// Gated full attention (the q rows carry the per-head sigmoid gate → q part
/// rows = `heads*head_dim*2`).
pub(crate) struct FullAttn {
    /// Row-fused `[q; k; v]` (`[q_gated + 2*kv, hidden]`): one GEMM per step,
    /// split into the q/k/v buffers downstream.
    qkv_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
}

/// Gated-delta-rule linear attention.
pub(crate) struct LinearAttn {
    /// Row-fused `[qkv; z]` (`[qkv_dim + z_dim, hidden]`): one GEMM per step,
    /// split into the conv-input qkv and the z gate downstream.
    in_proj_qkvz: DeviceMatrix,
    /// Row-fused `[b; a]` (`[2*Vh, hidden]`): one GEMM per step, split into
    /// the per-head b/a scalars downstream. b = rows `0..Vh`, a = `Vh..2*Vh`.
    in_proj_ba: DeviceMatrix,
    /// Depthwise conv1d weight `[qkv_dim*kernel]` bf16.
    conv1d_weight: DeviceVec,
    dt_bias: DeviceVec,
    /// `A_log` `[num_value_heads]` f32 (this rank's v-head shard under TP).
    a_log: CudaSlice<f32>,
    /// Gated output RMSNorm scale `[value_head_dim]` f32, broadcast across
    /// heads by rms_norm_gated (norm.cu `weight[tid]`) — replicated under TP.
    norm_weight: CudaSlice<f32>,
    out_proj: DeviceMatrix,
}

pub(crate) struct Qwen35Layer {
    input_layernorm: DeviceVec,
    attn: Qwen35Attn,
    post_attention_layernorm: DeviceVec,
    /// Exactly one of `mlp` / `moe` is `Some`.
    mlp: Option<DenseMlp>,
    moe: Option<crate::loader::MoeLayerWeights>,
}

pub(crate) struct DenseMlp {
    /// Row-fused `[gate; up]` (`[2*inter, hidden]`): one GEMM per step feeds
    /// the fused SwiGLU. Gate = rows `0..inter`, up = rows `inter..2*inter`.
    gate_up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

impl DenseMlp {
    /// SwiGLU intermediate width (half the fused projection's output rows).
    fn inter_dim(&self) -> usize {
        self.gate_up_proj.rows / 2
    }
}

/// NextN-MTP draft head for Qwen3.6 speculative decode: one full-attention
/// transformer block (`mtp.layers.0.*`, always Full + dense MLP) preceded by the
/// `fc` concat-projection over `[norm(candidate_embedding); norm(previous_hidden)]`
/// and its two pre-`fc` RMSNorms, with a final RMSNorm before the SHARED
/// `lm_head`. `lm_head` + `embed_tokens` are the base model's and are not stored
/// here. Loaded only when spec-decode is requested; fields are consumed by the
/// draft forward in the next increment.
#[allow(dead_code)] // populated here; read by mtp_forward_level (next increment)
pub(crate) struct Qwen35MtpHead {
    /// RMSNorm over the candidate token embedding, pre-`fc`.
    pre_fc_norm_embedding: DeviceVec,
    /// RMSNorm over the previous-step trunk hidden, pre-`fc`.
    pre_fc_norm_hidden: DeviceVec,
    /// Concat projection `[hidden, 2*hidden]` mapping the two normed inputs to
    /// the transformer-block input.
    fc: DeviceMatrix,
    /// The head's single transformer block (Full attention + dense MLP).
    layer: Qwen35Layer,
    /// Final RMSNorm before the shared lm_head.
    norm: DeviceVec,
}

pub(crate) struct Qwen35Model {
    pub(crate) ctx: DeviceContext,
    pub(crate) config: Qwen35Config,
    embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    layers: Vec<Qwen35Layer>,
    norm: DeviceVec,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    moe_config: Option<infer_moe::MoeConfig>,
    /// Tensor-parallel runtime. `world_size == 1` uses the no-op communicator,
    /// so every `all_reduce_sum` returns immediately (mirrors the dense
    /// [`crate::model::CudaModel`]).
    pub(crate) tp: crate::tp::TpRuntime,
    /// This rank's per-shard head counts (= global config on a single GPU).
    /// The forward sizes its buffers, slot state, and kernel launches from
    /// these, not the config.
    local_q_heads: usize,
    local_kv_heads: usize,
    local_linear_k_heads: usize,
    local_linear_v_heads: usize,
    /// This rank's EP expert ownership (every expert local on a single GPU).
    expert_split: ExpertSplit,
    /// Per-slot cache capacity (full-attn contiguous cache rows).
    max_seq_len: usize,
    /// NextN-MTP draft head for speculative decode. `Some` (and
    /// `spec_draft_tokens > 0`) only when the constructor was asked to load it;
    /// the default decode path never reads it, so the baseline stays
    /// byte-identical when spec-decode is off.
    #[allow(dead_code)] // read by the spec-decode draft/verify path (next increment)
    mtp: Option<Qwen35MtpHead>,
    /// Requested MTP draft depth (`0` = spec-decode off).
    #[allow(dead_code)] // read by the spec-decode orchestrator (next increment)
    spec_draft_tokens: usize,
    /// Host-resident weight snapshot while the engine is offloaded for the OPD
    /// teacher time-share. `Some` iff [`Qwen35Model::offload_engine_weights`] ran
    /// without a matching [`Qwen35Model::reload_engine_weights`]; the device
    /// weight buffers are 1-element placeholders in that state and must NOT be
    /// forwarded through until reloaded.
    offloaded: Option<Box<OffloadedWeights>>,
    /// Pristine *device* copy of each dense-BF16 base weight, captured on the
    /// first touch (device→device clone of the resident matrix, after any
    /// FP8→BF16 promotion). The per-step re-merge then runs entirely on-device:
    /// upload the tiny A/B, GEMM `B·A`, scaled-add `W = base + scale·(B·A)`
    /// straight into the resident matrix — no host triple-loop, no full-W
    /// host→device upload.
    lora_base_dev: HashMap<LoraBaseKey, DeviceVec>,
    /// FP8 side buffers retired by [`Qwen35Model::promote_lora_target_to_bf16`]
    /// whose device pointers may still be aliased by a co-resident autograd
    /// student (`--share-frozen-base` imports NON-OWNING views of the pointers
    /// exported by [`Qwen35Model::frozen_base_fp8_pointers`]). Kept alive so the
    /// student keeps reading the pristine FP8 base; never exported → freed on
    /// promotion instead.
    lora_promoted_fp8_keepalive: Vec<(CudaSlice<u8>, CudaSlice<f32>)>,
    /// Latched by [`Qwen35Model::frozen_base_fp8_pointers`]: resident FP8 base
    /// pointers have been exported for train-infer weight sharing.
    frozen_base_ptrs_exported: AtomicBool,
    /// Reusable device scratch for the `B·A` delta (sized to the largest dense
    /// merged matrix seen). Avoids a per-projection device alloc each step.
    lora_delta_scratch: Option<DeviceVec>,
    /// Projections whose resident device matrix currently includes a non-zero
    /// LoRA delta. Lets all-zero adapter steps skip weight uploads unless they
    /// need to restore a previously merged projection back to base.
    lora_dirty: HashSet<LoraBaseKey>,
    /// Cheap model/weights-version tag (hash of the checkpoint's safetensors
    /// file names + lengths + mtimes). Stamps the durable KV-recall manifest so
    /// a restart after an OPD weight update (which rewrites the checkpoint, so
    /// the mtimes/lengths shift) discards the now-stale recalled KV.
    #[allow(dead_code)] // WIP: durable KV-recall manifest weight-version stamp, not yet wired
    weights_epoch: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LoraBaseKey {
    layer_idx: usize,
    projection: StudentLoraProjection,
}

/// Host-resident snapshot of one transformer block's device weight buffers,
/// captured by [`Qwen35Model::offload_engine_weights`]. Reload restores in the
/// same field order.
struct OffloadedBlock {
    input_layernorm: Vec<bf16>,
    post_attention_layernorm: Vec<bf16>,
    attn: OffloadedAttn,
    /// Dense MLP snapshot — `Some` iff this layer is dense (mutually exclusive
    /// with `moe`, mirroring [`Qwen35Layer`]).
    mlp: Option<OffloadedDenseMlp>,
    moe: Option<crate::loader::MoeLayerHostSnapshot>,
}

struct OffloadedDenseMlp {
    gate_up_proj: HostMatrixSnapshot,
    down_proj: HostMatrixSnapshot,
}

/// Host snapshot of a full-attention block (mirrors [`FullAttn`]).
struct OffloadedFullAttn {
    qkv_proj: HostMatrixSnapshot,
    o_proj: HostMatrixSnapshot,
    q_norm: Vec<bf16>,
    k_norm: Vec<bf16>,
}

/// Host snapshot of a gated-delta linear-attention block (mirrors [`LinearAttn`]).
struct OffloadedLinearAttn {
    in_proj_qkvz: HostMatrixSnapshot,
    in_proj_ba: HostMatrixSnapshot,
    conv1d_weight: Vec<bf16>,
    dt_bias: Vec<bf16>,
    a_log: Vec<f32>,
    norm_weight: Vec<f32>,
    out_proj: HostMatrixSnapshot,
}

enum OffloadedAttn {
    // Boxed: mirrors the device-side `Qwen35Attn` (large, size-skewed variants);
    // boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<OffloadedFullAttn>),
    Linear(Box<OffloadedLinearAttn>),
}

/// Full host-resident snapshot of all model device weights while offloaded.
/// `embed_tokens` doubles as the (tied) lm_head when `lm_head` is `None`.
struct OffloadedWeights {
    embed_tokens: HostMatrixSnapshot,
    lm_head: Option<HostMatrixSnapshot>,
    norm: Vec<bf16>,
    blocks: Vec<OffloadedBlock>,
}

impl std::fmt::Debug for Qwen35Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35Model")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("kv_heads", &self.config.num_key_value_heads)
            .field("head_dim", &self.config.head_dim)
            .field("experts", &self.config.num_experts)
            .finish()
    }
}

fn validate_qwen35_cuda_config(m: &Qwen35Config) -> Result<()> {
    ensure!(
        matches!(m.head_dim, 128 | 256),
        "clean CUDA Qwen3.5 hybrid path supports full-attention head_dim 128/256, got {}",
        m.head_dim
    );
    ensure!(
        m.num_attention_heads.is_multiple_of(m.num_key_value_heads),
        "Qwen3.5 num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
        m.num_attention_heads,
        m.num_key_value_heads
    );
    ensure!(
        m.linear_key_head_dim == 128 && m.linear_value_head_dim == 128,
        "clean CUDA Qwen3.5 gated-delta path supports linear key/value dim 128/128, got {}/{}",
        m.linear_key_head_dim,
        m.linear_value_head_dim
    );
    Ok(())
}

impl Qwen35Model {
    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    /// This rank's full-attention query width (`local_q_heads * head_dim`).
    fn local_full_attn_q_dim(&self) -> usize {
        self.local_q_heads * self.config.head_dim
    }

    /// This rank's GATED q_proj output width: the projection interleaves
    /// `[query; gate]` per head, so each local head contributes `2*head_dim` rows.
    fn local_full_attn_q_proj_dim(&self) -> usize {
        self.local_q_heads * self.config.head_dim * 2
    }

    /// This rank's full-attention K/V width (`local_kv_heads * head_dim`).
    ///
    /// Single source for every full-attn K/V cache / pool size on this rank.
    /// CACHE-IDENTITY INVARIANT (KV replication, kv_heads < world_size): each
    /// rank holds `local_kv_heads == 1` KV head, and ranks in the same replica
    /// group loaded IDENTICAL k/v projection weights ([`infer_topo::kv_load_block_index`])
    /// AND see the same `normed` hidden states, so their per-rank-local cache rows
    /// are bit-identical. GQA then runs independently per rank against its own
    /// copy — no cross-replica KV exchange is needed or done.
    fn local_full_attn_kv_dim(&self) -> usize {
        self.local_kv_heads * self.config.head_dim
    }

    /// This rank's fused gated-delta `[q | k | v]` width.
    fn local_linear_qkv_dim(&self) -> usize {
        let qk = 2 * self.local_linear_k_heads * self.config.linear_key_head_dim;
        qk + self.local_linear_v_heads * self.config.linear_value_head_dim
    }

    /// This rank's gated-delta output / z-gate width.
    fn local_linear_z_dim(&self) -> usize {
        self.local_linear_v_heads * self.config.linear_value_head_dim
    }

    /// `(num_linear, gdr_state_len, conv_len)` for this rank's recurrent state,
    /// sized from LOCAL shard widths (each rank carries only its own v-head
    /// recurrent slabs / qkv conv channels). Single source for both
    /// [`Qwen35SlotState::acquire_recurrent`] and the spec snapshot scratch.
    pub(crate) fn recurrent_dims(&self) -> (usize, usize, usize) {
        let c = &self.config;
        let num_linear = c.num_hidden_layers - c.num_full_attention_layers();
        let gdr_state_len =
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        (num_linear, gdr_state_len, conv_len)
    }

    /// A fresh idle slot — allocates nothing. The recurrent state is drawn from
    /// the executor's free-list on activation ([`Qwen35SlotState::acquire_recurrent`]);
    /// the full-attn K/V lives in the shared `PagedKVPool`.
    pub(crate) fn new_slot_state(&self) -> Qwen35SlotState {
        Qwen35SlotState::new_linear_only()
    }

    /// Allocate per-slot speculative-decode state (draft head KV + trunk linear
    /// snapshot scratch), sized from this rank's local shard widths and the
    /// requested draft depth. Snapshot scratch matches [`Self::new_slot_state`]'s
    /// linear-state dims exactly so a snapshot/restore is a straight D2D copy.
    #[allow(dead_code)] // called by the executor spec-slot init in a later increment
    /// Per-layer bytes of the two linear states, in `linear_state_addrs` order.
    pub(crate) fn linear_state_bytes(&self) -> (usize, usize) {
        let c = &self.config;
        (
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4,
            self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1) * 2,
        )
    }

    pub(crate) fn new_spec_slot_state(&self) -> Result<Qwen35SpecSlotState> {
        let c = &self.config;
        let num_full = c.num_full_attention_layers();
        let num_linear = c.num_hidden_layers - num_full;
        // depth >= 1: the head cache holds the draft chain (level positions
        // 0..depth); +1 leaves room for the seed row at position 0.
        let depth = self.spec_draft_tokens.max(1);
        let kv_dim = self.local_full_attn_kv_dim();
        let gdr_state_len =
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        let qkv_dim = self.local_linear_qkv_dim();
        // b/a are one scalar per LOCAL value head, sharded identically on every
        // linear layer (`in_proj_b`/`in_proj_a` rows == local_linear_v_heads),
        // so the capture stride is uniform across layers.
        let ba_dim = self.local_linear_v_heads;
        let cap_rows = depth + 1;
        let mut gdr_snap = Vec::with_capacity(num_linear);
        let mut conv_snap = Vec::with_capacity(num_linear);
        let mut cap_qkv = Vec::with_capacity(num_linear);
        let mut cap_b = Vec::with_capacity(num_linear);
        let mut cap_a = Vec::with_capacity(num_linear);
        for _ in 0..num_linear {
            gdr_snap.push(
                self.ctx
                    .stream
                    .alloc_zeros::<f32>(gdr_state_len)
                    .map_err(|e| anyhow!("alloc spec gated-delta snapshot failed: {e}"))?,
            );
            conv_snap.push(DeviceVec::zeros(&self.ctx, conv_len)?);
            cap_qkv.push(DeviceVec::zeros(&self.ctx, cap_rows * qkv_dim)?);
            cap_b.push(DeviceVec::zeros(&self.ctx, cap_rows * ba_dim)?);
            cap_a.push(DeviceVec::zeros(&self.ctx, cap_rows * ba_dim)?);
        }
        Ok(Qwen35SpecSlotState {
            head_k: DeviceVec::zeros(&self.ctx, (depth + 1) * kv_dim)?,
            head_v: DeviceVec::zeros(&self.ctx, (depth + 1) * kv_dim)?,
            gdr_snap,
            conv_snap,
            capture: Qwen35LinearCapture {
                rows: cap_rows,
                qkv: cap_qkv,
                b_proj: cap_b,
                a_proj: cap_a,
            },
            argmax_scratch: self
                .ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow!("alloc spec argmax scratch failed: {e}"))?,
            q_probs: SliceSlot::default(),
            p_probs: SliceSlot::default(),
            sample_tok: SliceSlot::default(),
            accept_out: SliceSlot::default(),
            chain_draft: SliceSlot::default(),
            u_accept: SliceSlot::default(),
            u_residual: SliceSlot::default(),
        })
    }

    /// Warm the Qwen FP8 dense DeepGEMM JIT for the CUDA default prefill chunk.
    ///
    /// The routed-expert port already uses DSv4's grouped path. The SLO 4096/256
    /// regression was the divergent dense-projection lane: first request paid
    /// one DeepGEMM JIT compile per `(M,N,K)` dense FP8 projection shape. Warming
    /// unique `(rows, cols)` at `M=2048` mirrors the scheduler's CUDA Qwen chunk
    /// floor and keeps the compile cost out of request TTFT without mutating KV
    /// or recurrent state.
    pub(crate) fn warm_fp8_deepgemm_dense_prefill(&self) -> Result<(usize, usize)> {
        let warm_m = self.max_seq_len.min(2048);
        // sm_120 dense FP8 runs the dequant→BF16 fallback (no DeepGEMM); nothing to warm.
        if self.ctx.is_sm120() {
            return Ok((0, warm_m));
        }
        if warm_m < 1024 {
            return Ok((0, warm_m));
        }
        let mut seen = HashSet::new();
        let mut warmed = 0usize;
        let mut warm = |weight: &DeviceMatrix| -> Result<()> {
            if seen.insert((weight.rows, weight.cols))
                && warm_fp8_deepgemm_dense(&self.ctx, weight, warm_m)?
            {
                // Also JIT-warm the spec-verify row count so the first DSpark
                // block step doesn't compile DeepGEMM M=16 kernels in-request.
                warm_fp8_deepgemm_dense(&self.ctx, weight, 16)?;
                warmed += 1;
            }
            Ok(())
        };

        for layer in &self.layers {
            match &layer.attn {
                Qwen35Attn::Full(full) => {
                    warm(&full.qkv_proj)?;
                    warm(&full.o_proj)?;
                }
                Qwen35Attn::Linear(linear) => {
                    warm(&linear.in_proj_qkvz)?;
                    warm(&linear.in_proj_ba)?;
                    warm(&linear.out_proj)?;
                }
            }
            if let Some(mlp) = &layer.mlp {
                warm(&mlp.gate_up_proj)?;
                warm(&mlp.down_proj)?;
            }
            if let Some(moe) = &layer.moe {
                warm(&moe.router_gate)?;
                warm(&moe.shared_gate)?;
                warm(&moe.shared_up)?;
                warm(&moe.shared_down)?;
                warm(&moe.shared_gate_router)?;
            }
        }
        if warmed > 0 {
            self.ctx.sync()?;
        }
        Ok((warmed, warm_m))
    }

    /// Warm the Qwen FP8 routed-MoE grouped DeepGEMM JIT for the CUDA default
    /// prefill chunk. The helper only launches the two JIT-backed grouped GEMMs;
    /// pack/requant kernels are static CUDA kernels and do not need JIT warmup.
    pub(crate) fn warm_fp8_deepgemm_grouped_prefill(&self) -> Result<(usize, usize, usize, usize)> {
        let warm_tokens = self.max_seq_len.min(2048);
        // sm_120 routes grouped FP8 to the AOT CUTLASS collective — no DeepGEMM
        // JIT kernels to warm (the preflight below is Hopper-only).
        if self.ctx.is_sm120() {
            return Ok((0, warm_tokens, 0, 0));
        }
        let topk = self.config.num_experts_per_tok;
        let mut seen = HashSet::new();
        let mut warmed = 0usize;
        let mut min_rows = usize::MAX;
        let mut max_rows = 0usize;
        for warm_tokens in [warm_tokens, warm_tokens.saturating_sub(16)] {
            let warm_routes = warm_tokens.saturating_mul(topk);
            if warm_routes < QWEN35_DEEPGEMM_MIN_ROUTES {
                continue;
            }
            for layer in &self.layers {
                let Some(moe) = &layer.moe else {
                    continue;
                };
                let (Some(w13), Some(down)) = (&moe.w13_fp8_grouped, &moe.down_fp8_grouped) else {
                    continue;
                };
                let rows = deepgemm_contig_rows_cap(warm_routes, w13.groups, DEEPGEMM_CONTIG_ALIGN);
                let key = (w13.groups, w13.rows, w13.cols, down.rows, down.cols, rows);
                if seen.insert(key) {
                    Self::warm_fp8_deepgemm_grouped_pair(&self.ctx, w13, down, rows)?;
                    min_rows = min_rows.min(rows);
                    max_rows = max_rows.max(rows);
                    warmed += 2;
                }
            }
        }
        if warmed > 0 {
            self.ctx.sync()?;
        }
        if warmed == 0 {
            min_rows = 0;
        }
        Ok((warmed, self.max_seq_len.min(2048), min_rows, max_rows))
    }

    fn warm_fp8_deepgemm_grouped_pair(
        ctx: &DeviceContext,
        w13: &crate::loader::MoeFp8ExpertGroup,
        down: &crate::loader::MoeFp8ExpertGroup,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            w13.groups == down.groups && w13.cols == down.rows && w13.rows == 2 * down.cols,
            "Qwen FP8 grouped DeepGEMM warm shape mismatch: w13={}x{} g={} down={}x{} g={}",
            w13.rows,
            w13.cols,
            w13.groups,
            down.rows,
            down.cols,
            down.groups
        );
        ensure!(
            rows.is_multiple_of(DEEPGEMM_CONTIG_ALIGN),
            "Qwen FP8 grouped DeepGEMM warm rows {rows} not aligned to {DEEPGEMM_CONTIG_ALIGN}"
        );
        cuda_moe::dsv4_deepgemm_native_preflight()?;

        let hidden = w13.cols;
        let intermediate = down.cols;
        let scale_stride_m = rows.div_ceil(4) * 4;
        let hidden_scale_cols = hidden.div_ceil(128);
        let inter_scale_cols = intermediate.div_ceil(128);
        let input_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(rows * hidden)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm input alloc failed: {e}"))?;
        let input_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * hidden_scale_cols)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm input scale alloc failed: {e}"))?;
        let w13_out = ctx
            .stream
            .alloc_zeros::<bf16>(rows * w13.rows)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm w13 output alloc failed: {e}"))?;
        let act_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(rows * intermediate)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm act alloc failed: {e}"))?;
        let act_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * inter_scale_cols)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm act scale alloc failed: {e}"))?;
        let out = ctx
            .stream
            .alloc_zeros::<bf16>(rows * hidden)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm output alloc failed: {e}"))?;
        let m_indices = ctx
            .stream
            .alloc_zeros::<i32>(rows)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm m_indices alloc failed: {e}"))?;
        let stream = ctx.stream.cu_stream();

        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                cache_ptr(&input_fp8, ctx),
                cache_ptr(&input_scales, ctx),
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                cache_ptr(&w13_out, ctx),
                cache_ptr(&m_indices, ctx),
                w13.groups,
                rows,
                w13.rows,
                hidden,
                scale_stride_m,
                DEEPGEMM_CONTIG_ALIGN,
                stream,
            )?;
            cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                cache_ptr(&act_fp8, ctx),
                cache_ptr(&act_scales, ctx),
                cache_ptr(&down.weight, ctx),
                cache_ptr(&down.scales, ctx),
                cache_ptr(&out, ctx),
                cache_ptr(&m_indices, ctx),
                down.groups,
                rows,
                hidden,
                intermediate,
                scale_stride_m,
                DEEPGEMM_CONTIG_ALIGN,
                stream,
            )?;
        }
        Ok(())
    }

    /// Per-slot device-memory cost (bytes) at this rank's local shard widths:
    /// the gated-delta recurrent state + conv rings ONLY. This is the recurrent
    /// block each ACTIVE slot draws from the executor's free-list on activation
    /// (the pool grows to `num_slots` at full concurrency); the budget reserves
    /// for the worst case even though idle slots hold no block.
    ///
    /// Full-attn K/V is not per-slot: it
    /// lives in the executor's one shared profile-sized [`PagedKVPool`], so the
    /// per-slot budget excludes it (the `kv_bytes` term is 0). The slot clamp
    /// then reflects only the small recurrent state; the pool reserves the bulk
    /// of free VRAM separately (profiled AFTER this clamp, dense-style).
    pub(crate) fn per_slot_kv_bytes(&self) -> (usize, usize, usize, usize) {
        let c = &self.config;
        let num_full = c.num_full_attention_layers();
        let num_linear = c.num_hidden_layers - num_full;
        let bf16 = std::mem::size_of::<half::bf16>();
        let f32sz = std::mem::size_of::<f32>();
        // Full-attn K/V is paged (shared pool), not per-slot → 0 here.
        let kv_bytes = 0usize;
        // Gated-delta recurrent state: num_linear × (Vh × Kd × Vd) f32.
        let gdr_len = self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let gdr_bytes = num_linear.saturating_mul(gdr_len).saturating_mul(f32sz);
        // Conv1d rings: num_linear × (qkv_dim × (kernel-1)) bf16.
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        let conv_bytes = num_linear.saturating_mul(conv_len).saturating_mul(bf16);
        let per_slot = kv_bytes
            .saturating_add(gdr_bytes)
            .saturating_add(conv_bytes);
        (per_slot, kv_bytes, gdr_bytes, conv_bytes)
    }

    /// Clamp `requested` slots to what post-weights free VRAM affords — the
    /// dynamic KV-memory budget Qwen3.5/3.6 previously lacked (requested slots
    /// were admitted as-is → OOM at large max_seq_len, the #60 failure class).
    /// Unified with DSv4 through the infer-seam budget kernel; the affordable
    /// count is NCCL min-reduced for TP-consistent slot counts. Call AFTER
    /// weights load so `mem_get_info().free` already excludes them.
    pub(crate) fn kv_budget_num_slots(
        &self,
        requested: usize,
        extra_per_slot_bytes: usize,
    ) -> Result<usize> {
        const MEM_FRACTION: f64 = 0.9;
        let (per_slot, kv_bytes, gdr_bytes, conv_bytes) = self.per_slot_kv_bytes();
        let per_slot = per_slot.saturating_add(extra_per_slot_bytes);
        let affordable_local: i32 = match cudarc::driver::result::mem_get_info() {
            Ok((free, _total)) => {
                // Same neutral kernel as DSv4: floor(free × fraction) / per_slot.
                let budget = infer_seam::SlotBudget::from_free(free, MEM_FRACTION, 0, per_slot);
                log::info!(
                    "Qwen3.5 KV budget: free {}MB, per_slot {}MB (K+V {}MB + gdr {}MB + conv {}MB + draft {}MB)",
                    free >> 20,
                    per_slot >> 20,
                    kv_bytes >> 20,
                    gdr_bytes >> 20,
                    conv_bytes >> 20,
                    extra_per_slot_bytes >> 20,
                );
                budget
                    .affordable()
                    .map_or(i32::MAX, |n| i32::try_from(n).unwrap_or(i32::MAX))
            }
            // Can't query (no active context / driver error) → don't bind the
            // min; the other ranks' budgets still apply.
            Err(_) => i32::MAX,
        };
        let affordable =
            self.tp
                .all_reduce_min_scalar_i32(&self.ctx, affordable_local)? as usize;
        // Reject-below-fixed guard (parity with Metal's fits_fixed + DSv4): a
        // cross-rank-min affordable of 0 means post-weights free VRAM cannot
        // hold even one slot at this max_seq_len. Fail closed uniformly
        // (lockstep-safe — same reduced scalar on every rank) instead of the
        // former `max(1)` admitting one slot and OOMing at slot allocation.
        anyhow::ensure!(
            affordable > 0,
            "Qwen3.5 KV budget rejected startup: post-weights free VRAM affords 0 slots at \
             max_seq_len {} (per_slot ~{}MB exceeds {MEM_FRACTION} of free). Free VRAM or \
             lower --max-total-tokens.",
            self.max_seq_len,
            per_slot >> 20,
        );
        let (planned, clamped) = infer_seam::clamp_to_affordable(requested, affordable);
        if clamped {
            log::warn!(
                "Qwen3.5 KV budget: requested {requested} slots × ~{}MB/slot exceeds the \
                 cross-rank-min affordable {affordable} (local affordable {affordable_local}, \
                 {MEM_FRACTION} of post-weights free); clamping num_slots to {affordable}. \
                 Lower --max-total-tokens (max_seq_len {}) to raise concurrency.",
                per_slot >> 20,
                self.max_seq_len,
            );
        }
        Ok(planned)
    }

    /// Load a BF16 Qwen3.5/3.6 hybrid dense-or-MoE checkpoint, resolving the TP
    /// runtime from the environment (single-GPU when no TP env is set).
    ///
    /// `max_seq_len` sizes the per-slot full-attn contiguous K/V cache.
    pub(crate) fn from_safetensors(
        model_path: &Path,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
    ) -> Result<Self> {
        let tp = crate::loader::build_tp_runtime(false)?;
        Self::from_safetensors_with_tp(model_path, max_seq_len, tp, mtp_draft_tokens)
    }

    /// Load with an explicit [`crate::tp::TpRuntime`] (tests inject a single-GPU
    /// runtime — mirrors the dense loader's `from_safetensors_with_tp`).
    ///
    /// `mtp_draft_tokens`: `Some(n)` loads the NextN-MTP draft head for
    /// speculative decode (draft depth `n`); `None` keeps the baseline decode
    /// path (no MTP head loaded, byte-identical).
    pub(crate) fn from_safetensors_with_tp(
        model_path: &Path,
        max_seq_len: usize,
        tp: crate::tp::TpRuntime,
        mtp_draft_tokens: Option<usize>,
    ) -> Result<Self> {
        let total_t0 = Instant::now();
        let config_t0 = Instant::now();
        let m = Qwen35Config::from_model_dir(model_path)
            .map_err(|e| anyhow!("load Qwen3.5 config from {}: {e}", model_path.display()))?;
        validate_qwen35_cuda_config(&m)?;
        qwen35_startup_log(
            "config",
            config_t0,
            format_args!(
                "layers={} hidden={} moe={} model_path={}",
                m.num_hidden_layers,
                m.hidden_size,
                m.is_moe(),
                model_path.display()
            ),
        );
        // Full attention here is the GATED q_proj variant (Qwen3.5/3.6); the
        // prep+gate kernels assume it. Vanilla un-gated Qwen3 would need
        // the dense path, not this loader.
        ensure!(
            m.full_attn_gated,
            "clean CUDA Qwen3.5 hybrid path expects the gated full-attention q_proj \
             (Qwen3.5/3.6); un-gated Qwen3 uses from_qwen3_bf16_safetensors"
        );
        ensure!(
            m.rope_scaling.is_none(),
            "Qwen3.5 rope_scaling is set but the YaRN bridge is not wired into the \
             clean hybrid RoPE precompute; refusing to silently drop it (pod follow-up)"
        );
        ensure!(
            max_seq_len > 0,
            "Qwen3.5 hybrid model requires a non-zero KV cache budget"
        );

        let tp_cfg = *tp.config();
        let world = tp_cfg.world_size;
        // Per-rank full-attn GQA head counts. `head_shard` shards KV when
        // num_kv_heads >= world (e.g. Qwen3.6-35B kv=8 @ TP8 -> 1/rank) and
        // REPLICATES when num_kv_heads < world (Qwen3.5-122B kv=2 @ TP4 -> every
        // rank holds 1 replicated KV head + its Q-head shard). Replicas load
        // identical K/V weights (`kv_load_block_index`) so each computes GQA
        // independently; the divisible case stays byte-identical.
        let (local_q_heads, local_kv_heads) = if tp_cfg.is_single() {
            (m.num_attention_heads, m.num_key_value_heads)
        } else {
            infer_topo::head_shard(m.num_attention_heads, m.num_key_value_heads, &tp_cfg)
                .map_err(|e| anyhow!("Qwen3.5 TP full-attention head shard failed: {e}"))?
        };
        // KV-head block index this rank loads (== rank in the shard regime;
        // shared within a replica group in the replication regime). Q always
        // partitions by `tp_cfg.rank`.
        let kv_block = if tp_cfg.is_single() {
            0
        } else {
            infer_topo::kv_load_block_index(m.num_key_value_heads, &tp_cfg)
                .map_err(|e| anyhow!("Qwen3.5 TP full-attention KV block index: {e}"))?
        };
        // Gated-delta head counts. The linear (gated-delta) heads are large
        // (Qwen3.5/3.6: Kh=16, Vh=32), so they SHARD cleanly at the TP sizes that
        // need full-attn KV replication (122B @ TP4: 16->4, 32->8). We keep the
        // strict divisibility contract here — a contiguous head-major block shard
        // preserves the v-per-k grouping (gated_delta_rule.cu maps
        // k_head = v_head * Kh / Vh): each rank's v-head range reads exactly its
        // own k-head range. Linear-head replication would need a different shard
        // (the k/v grouping can't be split by replica), so reject it loudly
        // rather than silently mis-shard.
        ensure!(
            m.linear_num_key_heads.is_multiple_of(world),
            "Qwen3.5 TP: linear_num_key_heads ({}) not divisible by world_size ({world}) \
             — gated-delta linear heads must shard (replication unsupported on the linear path)",
            m.linear_num_key_heads
        );
        ensure!(
            m.linear_num_value_heads.is_multiple_of(world),
            "Qwen3.5 TP: linear_num_value_heads ({}) not divisible by world_size ({world}) \
             — gated-delta linear heads must shard (replication unsupported on the linear path)",
            m.linear_num_value_heads
        );
        let local_linear_k_heads = m.linear_num_key_heads / world;
        let local_linear_v_heads = m.linear_num_value_heads / world;
        // Shared expert is column/row-sharded like a dense MLP (its partial
        // joins the routed partial in one post-MoE all-reduce).
        ensure!(
            m.shared_expert_intermediate_size.is_multiple_of(world),
            "Qwen3.5 TP: shared_expert_intermediate_size ({}) not divisible by world_size ({world})",
            m.shared_expert_intermediate_size
        );
        // Dense-MLP layers (mlp_only_layers / sparse-step gaps) shard their
        // intermediate dim; only constrain it when such a layer exists.
        if (0..m.num_hidden_layers).any(|i| !m.is_moe_layer(i)) {
            ensure!(
                m.intermediate_size.is_multiple_of(world),
                "Qwen3.5 TP: dense intermediate_size ({}) not divisible by world_size ({world})",
                m.intermediate_size
            );
        }

        let moe_config = if m.is_moe() {
            Some(crate::moe_config::moe_config_from_qwen35(&m)?)
        } else {
            None
        };
        // EP mirrors TP for MoE: each rank owns `num_experts / world` whole
        // experts (`ExpertSplit::new` rejects an indivisible expert count
        // loudly). Dense Qwen3.5 has no expert-owned buffers; keep an inert
        // split so the struct layout stays uniform.
        let split = if !m.is_moe() {
            ExpertSplit::single(0)
        } else if tp_cfg.is_single() {
            ExpertSplit::single(m.num_experts)
        } else {
            ExpertSplit::new(m.num_experts, world, tp_cfg.rank)
                .map_err(|e| anyhow!("Qwen3.5 TP expert split: {e}"))?
        };

        let loader_t0 = Instant::now();
        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;
        qwen35_startup_log("ctx_loader", loader_t0, format_args!(""));

        let embed_t0 = Instant::now();
        let embed_tokens = loader.load_matrix(&ctx, m.embed_tokens_tensor_name())?;
        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, m.lm_head_tensor_name())?)
        };
        qwen35_startup_log(
            "embeddings",
            embed_t0,
            format_args!("tie_word_embeddings={}", m.tie_word_embeddings),
        );

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for layer_idx in 0..m.num_hidden_layers {
            let layer_t0 = Instant::now();
            let names = m.layer_tensor_names(layer_idx);
            let attn_t0 = Instant::now();
            let attn = match &names.attention {
                // Single GPU: full tensors, byte-identical to the pre-TP path.
                Qwen35AttentionTensorNames::Full(full) if tp_cfg.is_single() => {
                    Qwen35Attn::Full(Box::new(FullAttn {
                        qkv_proj: loader.load_matrices_row_fused(
                            &ctx,
                            &[
                                (full.q_proj.as_str(), None),
                                (full.k_proj.as_str(), None),
                                (full.v_proj.as_str(), None),
                            ],
                        )?,
                        o_proj: loader.load_matrix_quant_aware(&ctx, &full.o_proj)?,
                        q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                        k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                    }))
                }
                Qwen35AttentionTensorNames::Full(full) => Qwen35Attn::Full(Box::new(FullAttn {
                    // The GATED q_proj interleaves [query(HD); gate(HD)] PER
                    // HEAD (prefill_attention_hd256.cu reads q at
                    // `head*2*HD + d`, the gate kernel at `head*2*HD + HD + d`),
                    // so a whole-head slice with per-head row block 2*head_dim
                    // carries each head's query rows AND its matching gate rows.
                    // Q partitions by rank; K/V load the replica-aware KV block
                    // (== rank in the shard regime; shared within a replica group
                    // when kv_heads < world_size, giving identical K/V weights).
                    qkv_proj: {
                        let head_spec = |name: &str, local_rows: usize, block: usize| {
                            let total = loader.logical_rows(name)?;
                            Ok::<_, anyhow::Error>(infer_topo::ShardingSpec {
                                offset: block * local_rows,
                                size: local_rows,
                                total,
                            })
                        };
                        let q_spec =
                            head_spec(&full.q_proj, local_q_heads * m.head_dim * 2, tp_cfg.rank)?;
                        let k_spec =
                            head_spec(&full.k_proj, local_kv_heads * m.head_dim, kv_block)?;
                        let v_spec =
                            head_spec(&full.v_proj, local_kv_heads * m.head_dim, kv_block)?;
                        loader.load_matrices_row_fused(
                            &ctx,
                            &[
                                (full.q_proj.as_str(), Some(q_spec)),
                                (full.k_proj.as_str(), Some(k_spec)),
                                (full.v_proj.as_str(), Some(v_spec)),
                            ],
                        )?
                    },
                    // Row-parallel: each rank holds the o_proj input columns of
                    // its own heads; the forward all-reduces the partial sums.
                    o_proj: loader.load_matrix_sharded_quant_aware(
                        &ctx,
                        &full.o_proj,
                        infer_topo::ParallelLinearKind::Row,
                        &tp_cfg,
                    )?,
                    // q/k_norm are `[head_dim]`, broadcast across heads by the
                    // full-attention prep kernel — replicated.
                    q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                })),
                Qwen35AttentionTensorNames::Linear(lin) if tp_cfg.is_single() => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        in_proj_qkvz: loader.load_matrix_pair_fused(
                            &ctx,
                            &lin.in_proj_qkv,
                            &lin.in_proj_z,
                        )?,
                        in_proj_ba: loader.load_matrix_pair_fused(
                            &ctx,
                            &lin.in_proj_b,
                            &lin.in_proj_a,
                        )?,
                        conv1d_weight: loader.load_conv1d_vec(&ctx, &lin.conv1d_weight)?,
                        dt_bias: loader.load_vec_any(&ctx, &lin.dt_bias)?,
                        a_log: loader.load_f32_vec(&ctx, &lin.a_log)?,
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        out_proj: loader.load_matrix_quant_aware(&ctx, &lin.out_proj)?,
                    }))
                }
                Qwen35AttentionTensorNames::Linear(lin) => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        // Fused [q | k | v] blocks: shard EACH block on whole-head
                        // boundaries and re-stack this rank's three slices (a flat
                        // column shard would cut across the block boundaries).
                        in_proj_qkvz: {
                            let qkv = load_linear_qkv_sharded(
                                &loader,
                                &ctx,
                                &lin.in_proj_qkv,
                                &m,
                                &tp_cfg,
                            )?;
                            // z gate is v-head-major `[Vh*Vd]` (rms_norm_gated
                            // reads the gate at `head*Vd + d`).
                            let z = loader.load_qkv_head_sharded_quant_aware(
                                &ctx,
                                &lin.in_proj_z,
                                local_linear_v_heads,
                                m.linear_value_head_dim,
                                tp_cfg.rank,
                            )?;
                            DeviceMatrix::fuse_rows(&ctx, &qkv, &z)
                                .map_err(|e| anyhow!("fuse TP in_proj_qkv + in_proj_z: {e}"))?
                        },
                        // b/a are ONE SCALAR PER V HEAD (gated_delta_rule.cu reads
                        // `b_proj[token*Vh + v_head]`) → per-head row count 1;
                        // the local head shards row-fuse into one `[2*Vh, H]`.
                        in_proj_ba: {
                            let b = loader.load_qkv_head_sharded(
                                &ctx,
                                &lin.in_proj_b,
                                local_linear_v_heads,
                                1,
                                tp_cfg.rank,
                            )?;
                            let a = loader.load_qkv_head_sharded(
                                &ctx,
                                &lin.in_proj_a,
                                local_linear_v_heads,
                                1,
                                tp_cfg.rank,
                            )?;
                            DeviceMatrix::fuse_rows(&ctx, &b, &a)?
                        },
                        // Depthwise conv: channel rows mirror the qkv block shard.
                        conv1d_weight: load_conv1d_sharded(
                            &loader,
                            &ctx,
                            &lin.conv1d_weight,
                            &m,
                            &tp_cfg,
                        )?,
                        // Per-v-head vectors `[Vh]`.
                        dt_bias: load_v_head_vec_sharded(
                            &loader,
                            &ctx,
                            &lin.dt_bias,
                            m.linear_num_value_heads,
                            &tp_cfg,
                        )?,
                        a_log: load_v_head_f32_sharded(
                            &loader,
                            &ctx,
                            &lin.a_log,
                            m.linear_num_value_heads,
                            &tp_cfg,
                        )?,
                        // Gated-norm scale is `[Vd]`, broadcast across heads by
                        // rms_norm_gated (norm.cu `weight[tid]`) — replicated,
                        // matching the qwen35-spec Shard contract.
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        // Row-parallel: input columns follow this rank's v heads;
                        // the forward all-reduces the partial sums.
                        out_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &lin.out_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }))
                }
            };
            qwen35_startup_log(
                "layer.attn",
                attn_t0,
                format_args!("layer={layer_idx} type={:?}", m.layer_types[layer_idx]),
            );

            let ffn_t0 = Instant::now();
            let (mlp, moe) = if m.is_moe_layer(layer_idx) {
                let moe = loader.load_moe_layer_experts(
                    &ctx,
                    &names.common.moe_tensor_names(),
                    &split,
                    &tp_cfg,
                    m.moe_intermediate_size,
                    m.hidden_size,
                )?;
                (None, Some(moe))
            } else if tp_cfg.is_single() {
                (
                    Some(DenseMlp {
                        gate_up_proj: loader.load_matrix_pair_fused(
                            &ctx,
                            &names.common.mlp_gate_proj,
                            &names.common.mlp_up_proj,
                        )?,
                        down_proj: loader
                            .load_matrix_quant_aware(&ctx, &names.common.mlp_down_proj)?,
                    }),
                    None,
                )
            } else {
                (
                    Some(DenseMlp {
                        gate_up_proj: loader.load_matrix_pair_fused_column_sharded(
                            &ctx,
                            &names.common.mlp_gate_proj,
                            &names.common.mlp_up_proj,
                            &tp_cfg,
                        )?,
                        down_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &names.common.mlp_down_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }),
                    None,
                )
            };
            qwen35_startup_log(
                "layer.ffn",
                ffn_t0,
                format_args!("layer={layer_idx} moe={}", m.is_moe_layer(layer_idx)),
            );

            layers.push(Qwen35Layer {
                input_layernorm: loader.load_vec(&ctx, &names.common.input_layernorm)?,
                attn,
                post_attention_layernorm: loader
                    .load_vec(&ctx, &names.common.post_attention_layernorm)?,
                mlp,
                moe,
            });
            qwen35_startup_log(
                "layer.total",
                layer_t0,
                format_args!("layer={layer_idx} moe={}", m.is_moe_layer(layer_idx)),
            );
        }
        let tail_t0 = Instant::now();
        let norm = loader.load_vec(&ctx, m.norm_tensor_name())?;

        let rope_len = m
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        ensure!(
            max_seq_len <= rope_len,
            "Qwen3.5 max_seq_len ({max_seq_len}) exceeds the RoPE cache length ({rope_len}); \
             positions beyond the table would read out of bounds"
        );
        // PARTIAL RoPE: the table must be built over `rotary_dim` (= head_dim ×
        // partial_rotary_factor, 64 on Qwen3.6), not head_dim — the prep
        // kernel indexes `cos_cache[pos * rotary_dim + d]` and expects inv_freq
        // computed over rotary_dim dims (`precompute_rope` is generic over its
        // dim arg and emits the half-duplicated stride-dim layout it reads).
        let (cos_cache, sin_cache) =
            crate::ops::precompute_rope(&ctx, m.rotary_dim, rope_len, m.rope_theta, None)?;
        ctx.sync()?;
        qwen35_startup_log(
            "tail_norm_rope_sync",
            tail_t0,
            format_args!("rope_len={rope_len} max_seq_len={max_seq_len}"),
        );
        qwen35_startup_log(
            "total",
            total_t0,
            format_args!("layers={} max_seq_len={max_seq_len}", m.num_hidden_layers),
        );

        // NextN-MTP draft head (speculative decode). Loaded only on request; the
        // default decode path never touches it, so the baseline stays
        // byte-identical when off. Single-GPU only for now — the head shares the
        // base lm_head/embed and the 27B-FP8 fits on one H20; TP-sharded MTP is a
        // follow-up once spec-decode proves out single-GPU.
        let mtp = if mtp_draft_tokens.is_some() {
            let mtp_t0 = Instant::now();
            ensure!(
                tp_cfg.is_single(),
                "Qwen3.5 MTP spec-decode is single-GPU only for now \
                 (TP-sharded MTP draft head not yet wired)"
            );
            let head = load_qwen35_mtp_head(&loader, &ctx, &m, &split, &tp_cfg)?;
            qwen35_startup_log(
                "mtp_head",
                mtp_t0,
                format_args!("draft_tokens={}", mtp_draft_tokens.unwrap_or(0)),
            );
            Some(head)
        } else {
            None
        };
        let spec_draft_tokens = mtp_draft_tokens.unwrap_or(0);

        Ok(Self {
            ctx,
            config: m,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            moe_config,
            tp,
            local_q_heads,
            local_kv_heads,
            local_linear_k_heads,
            local_linear_v_heads,
            expert_split: split,
            max_seq_len,
            mtp,
            spec_draft_tokens,
            offloaded: None,
            lora_base_dev: HashMap::new(),
            lora_promoted_fp8_keepalive: Vec::new(),
            frozen_base_ptrs_exported: AtomicBool::new(false),
            lora_delta_scratch: None,
            lora_dirty: HashSet::new(),
            weights_epoch: kv_native_sys::weights_epoch_tag(model_path),
        })
    }

    /// Cheap model/weights-version tag for the durable KV-recall manifest. See
    /// the `weights_epoch` field.
    #[allow(dead_code)] // WIP: paired with the durable KV-recall manifest stamp
    pub(crate) fn weights_epoch(&self) -> &str {
        &self.weights_epoch
    }

    /// Move every device weight buffer to host RAM and free the VRAM (OPD teacher
    /// time-share). Idempotent: a no-op (returns 0) if already offloaded.
    ///
    /// Returns the total device VRAM (bytes) freed. After this returns the model
    /// must NOT be forwarded through until [`Qwen35Model::reload_engine_weights`].
    /// The freed weight blocks return to the device's async memory pool, which the
    /// co-resident OPD student/autograd allocator reuses for the backward pass —
    /// the VRAM headroom the time-share buys. Per-slot KV / recurrent state
    /// ([`Qwen35SlotState`]) is owned by the executor and untouched here.
    ///
    pub(crate) fn offload_engine_weights(&mut self) -> Result<usize> {
        if self.offloaded.is_some() {
            return Ok(0);
        }
        let ctx = self.ctx.clone();
        // Drain ALL in-flight GPU work before snapshotting. The OPD step has
        // co-resident allocators (infer-teacher + train autograd) sharing one
        // device/pool on separate streams; a full synchronize quiesces every
        // stream so the D2H snapshot and the subsequent block frees do not race
        // other-stream allocations from the shared async pool.
        ctx.sync()?;
        let mut freed = 0usize;

        let embed_tokens = self.embed_tokens.offload_to_host(&ctx)?;
        freed += embed_tokens.freed_bytes();
        let lm_head = match self.lm_head.as_mut() {
            Some(head) => {
                let snap = head.offload_to_host(&ctx)?;
                freed += snap.freed_bytes();
                Some(snap)
            }
            None => None,
        };
        let (norm, norm_n) = self.norm.offload_to_host(&ctx)?;
        freed += norm_n;

        let mut blocks = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            // (MoE guard hoisted to the preflight above — no mid-loop bail.)
            let (input_layernorm, in_ln_n) = layer.input_layernorm.offload_to_host(&ctx)?;
            let (post_attention_layernorm, post_ln_n) =
                layer.post_attention_layernorm.offload_to_host(&ctx)?;
            freed += in_ln_n + post_ln_n;

            let mlp = match layer.mlp.as_mut() {
                Some(dense) => {
                    let gate_up_proj = dense.gate_up_proj.offload_to_host(&ctx)?;
                    let down_proj = dense.down_proj.offload_to_host(&ctx)?;
                    freed += gate_up_proj.freed_bytes() + down_proj.freed_bytes();
                    Some(OffloadedDenseMlp {
                        gate_up_proj,
                        down_proj,
                    })
                }
                None => None,
            };
            let moe = match layer.moe.as_mut() {
                Some(moe) => {
                    let snap = moe.offload_to_host(&ctx)?;
                    freed += snap.freed_bytes();
                    Some(snap)
                }
                None => None,
            };

            let attn = match &mut layer.attn {
                Qwen35Attn::Full(full) => {
                    let qkv_proj = full.qkv_proj.offload_to_host(&ctx)?;
                    let o_proj = full.o_proj.offload_to_host(&ctx)?;
                    let (q_norm, qn) = full.q_norm.offload_to_host(&ctx)?;
                    let (k_norm, kn) = full.k_norm.offload_to_host(&ctx)?;
                    freed += qkv_proj.freed_bytes() + o_proj.freed_bytes() + qn + kn;
                    OffloadedAttn::Full(Box::new(OffloadedFullAttn {
                        qkv_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    }))
                }
                Qwen35Attn::Linear(lin) => {
                    let in_proj_qkvz = lin.in_proj_qkvz.offload_to_host(&ctx)?;
                    let in_proj_ba = lin.in_proj_ba.offload_to_host(&ctx)?;
                    let (conv1d_weight, conv_n) = lin.conv1d_weight.offload_to_host(&ctx)?;
                    let (dt_bias, dt_n) = lin.dt_bias.offload_to_host(&ctx)?;
                    let (a_log, al) = offload_raw_slice(&ctx, &mut lin.a_log)?;
                    let (norm_weight, nw) = offload_raw_slice(&ctx, &mut lin.norm_weight)?;
                    let out_proj = lin.out_proj.offload_to_host(&ctx)?;
                    freed += in_proj_qkvz.freed_bytes()
                        + in_proj_ba.freed_bytes()
                        + out_proj.freed_bytes()
                        + conv_n
                        + dt_n
                        + al
                        + nw;
                    OffloadedAttn::Linear(Box::new(OffloadedLinearAttn {
                        in_proj_qkvz,
                        in_proj_ba,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm_weight,
                        out_proj,
                    }))
                }
            };

            blocks.push(OffloadedBlock {
                input_layernorm,
                post_attention_layernorm,
                attn,
                mlp,
                moe,
            });
        }

        // Quiesce again after the block frees so reload (or a co-resident
        // backward) sees a settled pool. We deliberately do NOT trim the pool to
        // the OS: the freed blocks must stay in the shared async pool for the
        // student backward to reuse.
        ctx.sync()?;

        self.offloaded = Some(Box::new(OffloadedWeights {
            embed_tokens,
            lm_head,
            norm,
            blocks,
        }));
        Ok(freed)
    }

    /// Restore every device weight buffer from the host snapshot, re-allocating
    /// VRAM (OPD teacher time-share). Idempotent: a no-op if not offloaded.
    pub(crate) fn reload_engine_weights(&mut self) -> Result<()> {
        let Some(snapshot) = self.offloaded.take() else {
            return Ok(());
        };
        let ctx = self.ctx.clone();
        // Quiesce the whole device before re-allocating weight VRAM so the H2D
        // restores do not race the train/optimizer allocations still draining
        // from the shared async pool (see offload note).
        ctx.sync()?;
        let OffloadedWeights {
            embed_tokens,
            lm_head,
            norm,
            blocks,
        } = *snapshot;

        self.embed_tokens.reload_from_host(&ctx, &embed_tokens)?;
        match (self.lm_head.as_mut(), &lm_head) {
            (Some(head), Some(snap)) => head.reload_from_host(&ctx, snap)?,
            (None, None) => {}
            _ => anyhow::bail!("offload/reload lm_head presence mismatch"),
        }
        self.norm.reload_from_host(&ctx, &norm)?;

        ensure!(
            blocks.len() == self.layers.len(),
            "offload/reload layer count mismatch: snapshot {} vs model {}",
            blocks.len(),
            self.layers.len()
        );
        for (layer, block) in self.layers.iter_mut().zip(blocks) {
            let OffloadedBlock {
                input_layernorm,
                post_attention_layernorm,
                attn,
                mlp,
                moe,
            } = block;
            layer
                .input_layernorm
                .reload_from_host(&ctx, &input_layernorm)?;
            layer
                .post_attention_layernorm
                .reload_from_host(&ctx, &post_attention_layernorm)?;

            match (layer.mlp.as_mut(), layer.moe.as_mut(), mlp, moe) {
                (Some(dense), None, Some(snap), None) => {
                    dense
                        .gate_up_proj
                        .reload_from_host(&ctx, &snap.gate_up_proj)?;
                    dense.down_proj.reload_from_host(&ctx, &snap.down_proj)?;
                }
                (None, Some(moe), None, Some(snap)) => {
                    moe.reload_from_host(&ctx, &snap)?;
                }
                _ => anyhow::bail!("offload/reload MLP/MoE presence mismatch"),
            }

            match (&mut layer.attn, attn) {
                (Qwen35Attn::Full(full), OffloadedAttn::Full(snap)) => {
                    let OffloadedFullAttn {
                        qkv_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    } = *snap;
                    full.qkv_proj.reload_from_host(&ctx, &qkv_proj)?;
                    full.o_proj.reload_from_host(&ctx, &o_proj)?;
                    full.q_norm.reload_from_host(&ctx, &q_norm)?;
                    full.k_norm.reload_from_host(&ctx, &k_norm)?;
                }
                (Qwen35Attn::Linear(lin), OffloadedAttn::Linear(snap)) => {
                    let OffloadedLinearAttn {
                        in_proj_qkvz,
                        in_proj_ba,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm_weight,
                        out_proj,
                    } = *snap;
                    lin.in_proj_qkvz.reload_from_host(&ctx, &in_proj_qkvz)?;
                    lin.in_proj_ba.reload_from_host(&ctx, &in_proj_ba)?;
                    lin.conv1d_weight.reload_from_host(&ctx, &conv1d_weight)?;
                    lin.dt_bias.reload_from_host(&ctx, &dt_bias)?;
                    reload_raw_slice(&ctx, &mut lin.a_log, &a_log)?;
                    reload_raw_slice(&ctx, &mut lin.norm_weight, &norm_weight)?;
                    lin.out_proj.reload_from_host(&ctx, &out_proj)?;
                }
                _ => anyhow::bail!("offload/reload attention-kind mismatch"),
            }
        }
        ctx.sync()?;
        Ok(())
    }

    /// Shared layer stack for [`Self::forward_tokens`] /
    /// [`Self::forward_token_logits_full`]: stages the per-step host inputs
    /// (token ids + start_pos), runs [`Self::forward_hidden_staged`], and
    /// advances `slot.seq_len`. Leaves the final hidden states in `ws.hidden`
    /// (`[hidden, seq_len]`).
    fn forward_hidden(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<()> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, None, None, None)
    }

    /// [`Self::forward_hidden`] with an optional per-linear-layer gated-delta
    /// input capture (spec verify only). When `capture` is `Some`, each linear
    /// layer copies its post-in_proj `qkv` (PRE-conv1d) + `b`/`a` projections
    /// for all rows into the capture so a partial-accept replay can re-run only
    /// conv1d + GDR (see [`Qwen35LinearCapture`]). `None` is byte-for-byte the
    /// old `forward_hidden` (the capture branch is fully elided).
    fn forward_hidden_capture(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        capture: Option<&mut Qwen35LinearCapture>,
        recall: Option<&mut Qwen35RecallForward>,
        taps: Option<&mut dspark::Qwen35DsparkTaps>,
    ) -> Result<()> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3.5 hybrid forward requires at least one token"
        );
        ensure!(
            slot.seq_len == start_pos,
            "Qwen3.5 hybrid slot seq_len {} != start_pos {start_pos} (uncached full-prefix \
             path requires contiguous appends)",
            slot.seq_len
        );
        let seq_len = tokens.len();
        ensure!(
            start_pos + seq_len <= self.max_seq_len,
            "Qwen3.5 hybrid sequence {} exceeds KV cache budget {}",
            start_pos + seq_len,
            self.max_seq_len
        );
        self.stage_step_inputs(ws, tokens, start_pos)?;
        let mut rows = [LinearRow {
            slot,
            len: seq_len,
            capture,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, recall, taps)?;
        rows[0].slot.seq_len += seq_len;
        Ok(())
    }

    /// Stage the per-step HOST inputs (token ids, start_pos) into their
    /// persistent device slots. This is the ONLY H2D traffic of a decode step;
    /// the captured decode graph runs it OUTSIDE the capture/replay (the dense
    /// `stage1_write` pattern), so the graph body below is a pure GPU kernel
    /// sequence. Returns the `(token_ids, start_pos)` device addresses for the
    /// graph-bake fingerprint (length-matched slot reuse keeps them stable; a
    /// change means the captured graph reads stale memory and must recapture).
    pub(crate) fn stage_step_inputs(
        &self,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(u64, u64)> {
        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = ws.token_ids.upload(&self.ctx, &token_ids_host)?;
        let (token_ids_ptr, _g0) = token_ids.device_ptr(&self.ctx.stream);
        let start_pos_dev = ws.start_pos.upload(&self.ctx, &[start_pos as i32])?;
        let (start_pos_ptr, _g1) = start_pos_dev.device_ptr(&self.ctx.stream);
        Ok((token_ids_ptr, start_pos_ptr))
    }

    /// The pure-GPU layer stack over already-staged inputs: embeds the staged
    /// token ids, runs every layer over the workspace buffers, and advances
    /// each row's recurrent/conv/KV device state in place. Does NOT advance
    /// `slot.seq_len` and performs NO H2D/D2H/sync — at one row of one token
    /// this is the CUDA-graph-capturable decode body (every per-step scalar is
    /// read from the staged device buffers; see
    /// [`Self::forward_decode_step_captured`] for the capture-safety table).
    ///
    /// `rows` are the ragged per-slot token spans, token-major and in the same
    /// order as the staged ids; the full-attn layers get their identity from
    /// `recall`'s page table instead. `start_pos` (host) is consumed only by
    /// the multi-token NON-paged attention launch, which is single-row.
    fn forward_hidden_staged(
        &self,
        rows: &mut [LinearRow<'_>],
        ws: &mut Qwen35Workspace,
        start_pos: usize,
        mut recall: Option<&mut Qwen35RecallForward>,
        mut taps: Option<&mut dspark::Qwen35DsparkTaps>,
    ) -> Result<()> {
        // Single chokepoint for every recurrent-reading forward (prefill /
        // decode / spec-capture / OPD): each row's recurrent block MUST be
        // resident — a missed `acquire_recurrent` hook would otherwise read an
        // empty `gdr_states` as a silent no-op.
        ensure!(
            rows.iter().all(|r| r.slot.has_recurrent()),
            "Qwen3.6 forward: slot recurrent state not acquired (missing \
             acquire_recurrent at the start_pos==0 prefill)"
        );
        let seq_len: usize = rows.iter().map(|r| r.len).sum();
        ensure!(
            recall.is_some() || rows.len() == 1,
            "Qwen3.6 forward: the contiguous full-attn cache is single-row, got {} rows",
            rows.len()
        );
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;

        // Destructure once: each binding borrows its own workspace field, so
        // the residual-stream buffers, the per-block scratch, and the MoE
        // scratch stay simultaneously borrowable across the layer loop.
        let Qwen35Workspace {
            token_ids,
            start_pos: start_pos_slot,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            ..
        } = ws;

        // Shape-matched re-gets return the SAME buffers `stage_step_inputs`
        // just wrote (no H2D here — a mismatch would mean staging was skipped,
        // which the two call paths above make impossible).
        let token_ids = &*token_ids.get(&self.ctx, seq_len)?;
        let start_pos_dev = &*start_pos_slot.get(&self.ctx, 1)?;

        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        qwen35_profile(&self.ctx, "qwen/embedding", None, seq_len, || {
            embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)
        })?;
        // DSpark tap: `-1` = the embedding output (residual stream pre-layer-0).
        if let Some(t) = taps.as_deref_mut() {
            t.capture(&self.ctx, -1, hidden)?;
        }
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, seq_len)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, seq_len)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, seq_len)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            qwen35_profile(
                &self.ctx,
                "qwen/input_norm",
                Some(layer_idx),
                seq_len,
                || rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed),
            )?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    qwen35_profile(
                        &self.ctx,
                        "qwen/full_attention",
                        Some(layer_idx),
                        seq_len,
                        || {
                            if let Some(rc) = recall.as_deref_mut() {
                                self.full_attention_paged(
                                    full_attn,
                                    normed,
                                    full_idx,
                                    rc.pool,
                                    rc.meta,
                                    full,
                                    attn_out,
                                    rc.layer0_query.as_mut(),
                                )
                            } else {
                                self.full_attention(
                                    full_attn,
                                    normed,
                                    &mut *rows[0].slot,
                                    full_idx,
                                    start_pos,
                                    start_pos_dev,
                                    full,
                                    attn_out,
                                )
                            }
                        },
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    qwen35_profile(
                        &self.ctx,
                        "qwen/linear_attention",
                        Some(layer_idx),
                        seq_len,
                        || {
                            self.linear_attention(
                                lin,
                                normed,
                                LinearCore::Rows(&mut *rows),
                                linear_idx,
                                linear,
                                attn_out,
                            )
                        },
                    )?;
                    linear_idx += 1;
                }
            }

            // Post-attn residual add + post_attention_layernorm via the
            // `add_batch` + `rms_norm_offset` pair (`hidden_mid`/`normed`).
            qwen35_profile(
                &self.ctx,
                "qwen/post_attn_norm",
                Some(layer_idx),
                seq_len,
                || {
                    add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
                    rms_norm_offset(
                        &self.ctx,
                        hidden_mid,
                        &layer.post_attention_layernorm,
                        eps,
                        normed,
                    )
                },
            )?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                qwen35_profile(&self.ctx, "qwen/moe_ffn", Some(layer_idx), seq_len, || {
                    moe_forward_into(
                        &self.ctx,
                        moe_weights,
                        mlp_in,
                        cfg,
                        &self.expert_split,
                        moe,
                        mlp_out,
                    )
                })?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                qwen35_profile(
                    &self.ctx,
                    "qwen/dense_ffn",
                    Some(layer_idx),
                    seq_len,
                    || self.dense_mlp(mlp, mlp_in, dense, mlp_out),
                )?;
            }
            // ONE all-reduce covers the whole FFN partial: the MoE buffer already
            // sums this rank's routed experts (non-local routes contribute zero)
            // + the column/row-sharded shared expert; the dense branch is a
            // row-parallel down_proj partial. No-op on a single GPU.
            qwen35_profile(
                &self.ctx,
                "qwen/ffn_allreduce",
                Some(layer_idx),
                seq_len,
                || self.tp.all_reduce_sum(&self.ctx, mlp_out),
            )?;

            // MLP residual add producing the next layer's residual stream.
            // The post-attn sum lives in `hidden_mid`; add_batch reads
            // hidden_mid/mlp_out and writes `hidden` (whose previous value is
            // dead).
            qwen35_profile(
                &self.ctx,
                "qwen/ffn_residual",
                Some(layer_idx),
                seq_len,
                || add_batch(&self.ctx, hidden_mid, mlp_out, hidden),
            )?;
            // DSpark tap: the residual-stream OUTPUT of this layer.
            if let Some(t) = taps.as_deref_mut() {
                t.capture(&self.ctx, layer_idx as i64, hidden)?;
            }
        }

        Ok(())
    }

    /// Run prefill or decode for one row. `start_pos` is the absolute position of
    /// the first token; `tokens` are the new tokens (whole prompt on prefill, one
    /// token on decode). Advances `slot.seq_len` and the recurrent state. Returns
    /// the next sampled token + its behavior logprob (`None` for greedy). `ws` is
    /// the executor's persistent forward workspace (serial forwards share one).
    pub(crate) fn forward_tokens(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<(u32, Option<f32>)> {
        qwen35_profile(&self.ctx, "qwen/forward_hidden", None, tokens.len(), || {
            self.forward_hidden(slot, ws, tokens, start_pos)
        })?;
        qwen35_profile(&self.ctx, "qwen/lm_head", None, tokens.len(), || {
            self.lm_head_logits(ws, tokens.len())
        })?;
        qwen35_profile(&self.ctx, "qwen/sample", None, tokens.len(), || {
            self.sample_workspace_logits(ws, params, position)
        })
    }

    /// [`Self::forward_tokens`] over the opt-in paged recall KV pool
    /// (`--kv-recall`): the full-attn layers read/write `recall.pool` over
    /// `recall.meta` instead of the contiguous slot cache. `slot.seq_len` is
    /// still advanced in lockstep (the executor's decode invariant reads it);
    /// the pool's own `seq_len` is advanced separately via `alloc_tokens`.
    pub(crate) fn forward_tokens_recall(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        recall: &mut Qwen35RecallForward,
    ) -> Result<(u32, Option<f32>)> {
        self.forward_tokens_recall_tapped(
            slot, ws, tokens, start_pos, params, position, recall, None,
        )
    }

    /// [`Self::forward_tokens_recall`] with an optional DSpark trunk-tap
    /// capture (`--spec-type dspark` prefill/warm steps). `None` is
    /// byte-identical to the untapped path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tokens_recall_tapped(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        recall: &mut Qwen35RecallForward,
        taps: Option<&mut dspark::Qwen35DsparkTaps>,
    ) -> Result<(u32, Option<f32>)> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3.5 recall forward requires at least one token"
        );
        let seq_len = tokens.len();
        self.stage_step_inputs(ws, tokens, start_pos)?;
        let mut rows = [LinearRow {
            slot,
            len: seq_len,
            capture: None,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, Some(recall), taps)?;
        rows[0].slot.seq_len += seq_len;
        self.lm_head_logits(ws, seq_len)?;
        self.sample_workspace_logits(ws, params, position)
    }

    /// Final norm (offset) + LM head on the last token only, into `ws.logits`.
    /// Last stage of the captured decode graph (the capture ends at the logits
    /// buffer, dense-style); sampling stays outside.
    ///
    /// TP invariant: embed/lm_head are replicated and `hidden` is
    /// post-all-reduce (every row-parallel output above was summed), so the
    /// logits — and therefore the sampled token — are identical on every
    /// rank. No rank ever needs to broadcast its sample.
    fn lm_head_logits(&self, ws: &mut Qwen35Workspace, seq_len: usize) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        let Qwen35Workspace {
            hidden,
            last_hidden,
            last_normed,
            logits,
            ..
        } = ws;
        // Shape-matched re-get returns the SAME buffer forward_hidden filled.
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let last_hidden = last_hidden.get(&self.ctx, hidden_size)?;
        copy_row_to_vec(&self.ctx, hidden, seq_len - 1, last_hidden)?;
        let last_normed = last_normed.get(&self.ctx, hidden_size)?;
        rms_norm_offset_vec(&self.ctx, last_hidden, &self.norm, eps, last_normed)?;
        let logits = logits.get(&self.ctx, self.output_projection().rows)?;
        gemv(&self.ctx, self.output_projection(), last_normed, logits)?;
        Ok(())
    }

    /// Sample the next token from `ws.logits` (written by
    /// [`Self::lm_head_logits`] — eagerly or by a decode-graph replay). Greedy
    /// uses the persistent `argmax_out` slot (zero per-token allocations);
    /// non-greedy reads the logits to host. Always OUTSIDE any capture (syncs).
    pub(crate) fn sample_workspace_logits(
        &self,
        ws: &mut Qwen35Workspace,
        params: &SamplingParams,
        position: u64,
    ) -> Result<(u32, Option<f32>)> {
        let Qwen35Workspace {
            logits, argmax_out, ..
        } = ws;
        let logits = logits.get(&self.ctx, self.output_projection().rows)?;
        let argmax_out = argmax_out.get(&self.ctx, 1)?;
        sample_cuda_token_scratched(&self.ctx, logits, params, position, argmax_out)
    }

    /// Device address of the workspace logits buffer (allocating it at vocab
    /// size if absent) — the decode-graph bake fingerprint's output anchor.
    pub(crate) fn workspace_logits_ptr(&self, ws: &mut Qwen35Workspace) -> Result<u64> {
        let logits = ws.logits.get(&self.ctx, self.output_projection().rows)?;
        let (ptr, _g) = logits.data.device_ptr(&self.ctx.stream);
        Ok(ptr)
    }

    /// Per-slot KV-cache capacity (tokens).
    pub(crate) fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// MTP spec-decode draft depth this model was built for (`0` = spec off).
    /// The executor's `mtp_decode_row` uses this to size `spec_step`'s depth.
    pub(crate) fn spec_draft_tokens(&self) -> usize {
        self.spec_draft_tokens
    }

    /// This rank's local full-attention KV head count (= global config on a
    /// single GPU). Used by the opt-in KV-recall pool sizing + paged kernels.
    pub(crate) fn local_kv_heads(&self) -> usize {
        self.local_kv_heads
    }

    /// This rank's local full-attention query head count (= global config on a
    /// single GPU). Used by the opt-in KV-recall scoring + paged kernels.
    pub(crate) fn local_q_heads(&self) -> usize {
        self.local_q_heads
    }

    /// Whole-step captured decode body: embedding → every layer (norms,
    /// full/linear attention, dense/MoE FFN) → final norm → lm_head GEMV,
    /// ending at `ws.logits`. One `seq_len == 1` pass over already-staged
    /// inputs; sampling stays outside (dense pattern). The host-side
    /// `slot.seq_len` advance is the CALLER's job (replay never re-runs host
    /// code), and the per-step scalars (token id, position) live in the staged
    /// device buffers, so ONE capture replays across positions.
    ///
    /// Why this is the big lever (formula, vs the DSv4 whole-step WASH):
    /// Qwen3.5/3.6 B=1 decode measured 24.5 ms/token (40.8 tok/s) against a
    /// ~1.7 ms HBM active-weight floor — ~94% orchestration: ~1,074 kernel
    /// launches per token, each paying serialized host issue (~3-8 us) plus
    /// inter-launch gaps. predicted = 24.5 ms − (host issue + gap reclaim by
    /// graph scheduling) → ~14-19 ms/token (+30-75% tok/s) at TP=1. DSv4's
    /// whole-step graph was wall-neutral because ITS decode is GPU-bound;
    /// this one is host-bound, the opposite regime. License threshold:
    /// ≥ +10% tok/s with needle-gate pass AND replay-reuse evidence (the
    /// capture/replay counters in the executor), per the bench spec.
    ///
    /// Captured-kernel enumeration (capture-safety proof per kernel; the
    /// captured-bodies-allocate-nothing rule). All workspace slots are at
    /// steady decode shapes (allocated by the warm eager run `CudaGraphState`
    /// forces before first capture), so every `get`/`upload_const` inside is a
    /// pure cache hit; the only stream ops recorded are kernels, D2D memcpys,
    /// and device memsets — no H2D, no host callback, no alloc (the
    /// `audit_capturing_graph` host-memcpy census enforces this at capture).
    ///
    /// | # | kernel (per occurrence) | capture-safety justification |
    /// |---|-------------------------|------------------------------|
    /// | 1 | `embedding_batched_cuda` | reads staged `token_ids` device buffer; writes ws.hidden |
    /// | 2 | `rms_norm_batched_offset_cuda` ×2/layer | stateless; fixed ws buffers |
    /// | 3 | `gemm_cuda` (q/k/v/o, qkv/z/b/a/out, gate/up/down, router, shared ×4) | cuBLASLt with load-time workspace + algo cache warmed by the eager warm run (heuristic query happens outside capture; autotune self-suppresses during capture) |
    /// | 4 | `prefill_attention_hd256_prep_cuda` | position read from staged `start_pos` DEVICE buffer (already a device-pointer arg); writes K/V cache rows + q_prepped in place |
    /// | 5 | `nonpaged_prefill_attention_devpos_cuda` | NEW devpos entry: kv walk bounded by `*start_pos_dev` read on device; grid `(heads, 1)` shape-constant |
    /// | 6 | `attention_gate_batch_hd256_cuda` | stateless gate RMW on ws buffers |
    /// | 7 | `conv1d_prefill_cuda` (one fused `conv1d_decode_kernel` at seq_len 1) | depthwise conv + ring shift are content-based in-place device writes; each replay advances the ring by one token exactly like eager |
    /// | 8 | `gated_delta_rule_decode_cuda` | recurrent-state advance is a content-based in-place device write; no position arg |
    /// | 9 | `rms_norm_gated_cuda` | stateless; fixed ws buffers |
    /// | 10 | `dsv4_route` + `qwen36_renorm_topk_weights` | device router (gate requires `qwen35_decode_moe_graph_capturable`): all-zero bias table is `upload_const` (warm, no H2D node); writes route indices/weights slots unconditionally |
    /// | 11 | memset(counts), memset(cursors) | `get_zeroed` device memsets — legal graph memset nodes, re-executed per replay (atomicAdd accumulators NEED the per-replay re-zero) |
    /// | 12 | `dsv4_count_local_experts`, `dsv4_exclusive_scan_i32`, `dsv4_pack_local_experts_with_slots` | device counts/offsets/pack; grids sized by `total_routes = top_k` (shape constant, 8); counts live on device |
    /// | 13 | `moe_bf16_grouped_gemm_pair_batch` + `moe_bf16_grouped_gemm_batch` | HAND grouped kernels (hybrid dispatch: R=8 < `QWEN35_DEEPGEMM_MIN_ROUTES`, DeepGEMM JIT never runs at decode); weight-ptr tables are load-time device buffers |
    /// | 14 | `silu_mul_cuda` ×2 (routed + shared) | stateless |
    /// | 15 | `dsv4_scatter_all_route_slots` + `dsv4_combine_route_slot_outputs` | single-GPU (graph gated `tp.is_single()`): scatter writes ALL `top_k` slots (no EP sentinel/zeroed-tail path taken) |
    /// | 16 | `qwen36_add_shared_expert_gated` | stateless RMW over fully-written `mlp_out` |
    /// | 17 | `add_cuda` ×2/layer | stateless residual adds |
    /// | 18 | `memcpy_dtod` (last row), `rms_norm_offset_cuda`, `gemv_cuda` (lm_head) | D2D memcpy node + stateless kernels; capture ends at `ws.logits` |
    ///
    /// NOT in the captured region (stays eager): `SliceSlot::upload` staging
    /// H2D (pre-replay), argmax + D2H + `ctx.sync` sampling tail, the MoE
    /// host-route fallback (gated off), NCCL all-reduce (`tp.is_single()`
    /// only — `TpComm::Single` is a literal `Ok(())`).
    pub(crate) fn forward_decode_step_captured(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        start_pos: usize,
    ) -> Result<()> {
        let mut rows = [LinearRow {
            slot,
            len: 1,
            capture: None,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, None, None)?;
        self.lm_head_logits(ws, 1)
    }

    /// Paged twin of [`Self::forward_decode_step_captured`]: the same staged,
    /// host-state-free decode body, but full-attn layers attend the paged pool
    /// through `recall`'s page table. Capture-safe iff the meta is a
    /// [`crate::loader::PageMeta::persistent_decode`] (stable device
    /// addresses, `seqlen_k_capture` pinned) and the FA3 BF16 lane is active
    /// (the TileLang fallback bakes `num_pages` as a host arg).
    pub(crate) fn forward_decode_step_paged_captured(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        start_pos: usize,
        recall: &mut Qwen35RecallForward,
    ) -> Result<()> {
        let mut rows = [LinearRow {
            slot,
            len: 1,
            capture: None,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, Some(recall), None)?;
        self.lm_head_logits(ws, 1)
    }

    /// Whether the paged decode step will take the FA3 BF16 hd256 lane — the
    /// only paged-attention lane whose launch is CUDA-graph-capturable (the
    /// TileLang fallback bakes `num_pages` as a host arg).
    pub(crate) fn paged_decode_fa3_active(&self) -> bool {
        self.config.head_dim == 256 && qwen35_fa3_enabled(&self.ctx)
    }

    /// Whether every layer of this model can run the captured decode body —
    /// i.e. each MoE layer's decode step is a pure device-kernel sequence.
    /// Dense-MLP layers are always capturable; MoE layers need the device
    /// router + hand grouped kernels
    /// ([`crate::moe::qwen35_decode_moe_graph_capturable`]).
    pub(crate) fn decode_graph_unsupported_reason(&self) -> Option<&'static str> {
        let has_moe = self.layers.iter().any(|l| l.moe.is_some());
        if !has_moe {
            return None;
        }
        let Some(cfg) = self.moe_config.as_ref() else {
            return Some("MoE layers present but no moe_config");
        };
        if !crate::moe::qwen35_decode_moe_graph_capturable(cfg) {
            return Some(
                "MoE decode is not device-routable (host router fallback active — \
                 --qwen35-gpu-router false or non-greedy/grouped routing)",
            );
        }
        None
    }

    /// Run the full forward over `tokens` and return the FULL `[seq_len, vocab]`
    /// logits (every row, not just the last) WITHOUT sampling. Mirrors
    /// [`Self::forward_tokens`]'s layer stack but, instead of slicing the last
    /// row + a single `gemv`, applies the final offset-RMSNorm over the whole
    /// batch and a batched lm-head GEMM, returning the device logits buffer plus
    /// its `[seq_len, vocab]` shape.
    ///
    /// This is the OPD teacher raw-logits path: the distillation loss needs the
    /// teacher's per-position logit distribution over the prompt, so it cannot
    /// use the sampling tail. `slot` carries the per-slot KV + recurrent state
    /// exactly like [`Self::forward_tokens`].
    pub(crate) fn forward_token_logits_full(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        recall: Option<&mut Qwen35RecallForward>,
    ) -> Result<(DeviceVec, [usize; 2])> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, None, recall, None)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;

        // Final norm (offset) over the WHOLE batch, then the batched lm-head GEMM
        // produces every row's logits — no last-row slice, no sampling.
        // (TP invariant as in `forward_tokens`: replicated lm_head over
        // post-all-reduce hidden ⇒ identical logits on every rank.)
        let Qwen35Workspace { hidden, normed, .. } = ws;
        // Shape-matched re-gets return the SAME buffers forward_hidden used;
        // rms_norm fully overwrites `normed` before the lm-head GEMM reads it.
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let vocab = self.output_projection().rows;
        // The logits buffer is RETURNED to the OPD caller (ownership leaves the
        // forward), so it stays a per-call allocation — not a workspace slot.
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)?;
        self.ctx.sync()?;

        // `HiddenStates` is a `[vocab, seq_len]` column-batch over a flat device
        // buffer; reinterpret it as a `[seq_len, vocab]` row-major `DeviceVec`
        // (the train OPD bridge reads it as `[1, seq_len, vocab]`).
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_token_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab]))
    }

    /// Like [`Self::forward_token_logits_full`] but ALSO returns the LAST token's
    /// RAW trunk hidden (pre-final-norm) in a freshly-owned `DeviceVec`. The
    /// NextN-MTP draft head consumes that hidden as its level-0 input
    /// (`fc(concat[embed(tok), h])`), so spec-decode needs both the verify logits
    /// and the trunk hidden from a single forward. Byte-identical to
    /// `forward_token_logits_full` for the logits; the extra `copy_row_to_vec`
    /// captures the hidden BEFORE the lm-head norm reuses the workspace.
    /// (Spec-decode port increment 3a; gated path, no default-decode change.)
    #[allow(dead_code)] // consumed by mtp_forward_level / spec_step (next increment)
    pub(crate) fn forward_tokens_with_hidden(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        recall: Option<&mut Qwen35RecallForward>,
    ) -> Result<(DeviceVec, [usize; 2], DeviceVec)> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, None, recall, None)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        // Capture the last row's raw trunk hidden into an OWNED buffer first —
        // the lm-head norm below overwrites `ws.normed` from `ws.hidden`.
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        {
            let hidden = ws.hidden.get(&self.ctx, hidden_size, seq_len)?;
            copy_row_to_vec(&self.ctx, hidden, seq_len - 1, &mut last_hidden)?;
        }
        let Qwen35Workspace { hidden, normed, .. } = ws;
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)?;
        self.ctx.sync()?;
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_token_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab], last_hidden))
    }

    /// Like [`Self::forward_token_logits_full`] but ALSO returns EVERY row's raw
    /// trunk hidden (pre-final-norm) as one owned `[seq_len, hidden]` (token-major)
    /// `DeviceVec`. The depth>1 spec-decode verify needs the ACCEPTED row's hidden
    /// (not just the last) to seed the next block's level-0 draft — the accepted
    /// prefix length is only known after host-argmax, so all rows are captured.
    /// Byte-identical logits to `forward_token_logits_full`; the extra D2D copy of
    /// `ws.hidden` happens BEFORE the lm-head norm reuses the workspace.
    #[allow(dead_code)] // consumed by spec_step (next increment)
    pub(crate) fn forward_tokens_verify(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        capture: Option<&mut Qwen35LinearCapture>,
        recall: Option<&mut Qwen35RecallForward>,
    ) -> Result<(DeviceVec, [usize; 2], DeviceVec)> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, capture, recall, None)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        // Capture ALL rows' raw (pre-final-norm) trunk hidden into an owned
        // [seq_len, hidden] (token-major) buffer — the spec step seeds the next
        // block's level-0 draft from the accepted row's hidden.
        let mut hiddens = DeviceVec::zeros(&self.ctx, seq_len * hidden_size)?;
        {
            let hidden = ws.hidden.get(&self.ctx, hidden_size, seq_len)?;
            self.ctx
                .stream
                .memcpy_dtod(&hidden.data, &mut hiddens.data)
                .map_err(|e| anyhow!("verify hidden batch capture failed: {e}"))?;
        }
        let Qwen35Workspace { hidden, normed, .. } = ws;
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)?;
        self.ctx.sync()?;
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_verify_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab], hiddens))
    }

    /// One NextN-MTP draft level: given the previous chain token `token` and its
    /// hidden `h_prev`, produce the next greedy draft token + the head's output
    /// hidden (the input to the next level). The head is a single full-attention
    /// transformer block attending only over the **fresh per-block** draft chain
    /// at **local** positions `0..depth` (verified against the Metal
    /// `prepare_draft_block_mtp` reference: `draft_state.reset()` + rope_offset
    /// starts at 0), with the trunk context injected via
    /// `fc(concat[norm(embed(token)), norm(h_prev)])` — NOT re-attended. `level`
    /// is the 0-based position within the draft block (RoPE position + head-KV
    /// row). Greedy rows argmax; sampled rows draw on device from the
    /// engine-sampler-filtered head dist, retaining the filtered row in
    /// `spec.q_probs[level]` for the rejection accept.
    ///
    /// The head's KV is `spec.head_k`/`head_v`; processing levels `0,1,2,…` in
    /// order overwrites rows `0,1,2,…`, so each block self-seeds without an
    /// explicit reset (rows beyond the current chain are never read).
    //
    // ponytail: per-level owned scratch (emb/concat/normed/logits/…). Correct &
    // simple for the correctness gate; fold into a spec workspace before the
    // perf bench (the [vocab,1] logits alloc is the only non-trivial one).
    #[allow(dead_code)] // consumed by spec_step (next increment)
    fn mtp_forward_level(
        &self,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        token: u32,
        h_prev: &DeviceVec,
        level: usize,
        params: &SamplingParams,
        start_pos: usize,
    ) -> Result<(u32, DeviceVec)> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow!("mtp_forward_level called without a loaded MTP head"))?;
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden = c.hidden_size;
        ensure!(
            h_prev.len == hidden,
            "mtp_forward_level h_prev len {} != hidden {hidden}",
            h_prev.len
        );

        // Candidate token embedding.
        let token_ids = upload_i32(&self.ctx, &[token as i32])?;
        let mut emb = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut emb)?;

        // Pre-fc RMSNorms over [embedding ; previous hidden], concat.
        // fc is [hidden, 2*hidden]; the concat input is [norm(emb) ; norm(h_prev)]
        // (embedding first — matches the Metal `qwen3_5_mtp` loader order).
        let mut concat = HiddenStates::zeros(&self.ctx, 2 * hidden, 1)?;
        let mut emb_n = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        rms_norm_offset(&self.ctx, &emb, &mtp.pre_fc_norm_embedding, eps, &mut emb_n)?;
        {
            let mut dst = concat.data.slice_mut(0..hidden);
            self.ctx
                .stream
                .memcpy_dtod(&emb_n.data, &mut dst)
                .map_err(|e| anyhow!("mtp concat emb half failed: {e}"))?;
        }
        let mut h_n = DeviceVec::zeros(&self.ctx, hidden)?;
        rms_norm_offset_vec(&self.ctx, h_prev, &mtp.pre_fc_norm_hidden, eps, &mut h_n)?;
        {
            let mut dst = concat.data.slice_mut(hidden..2 * hidden);
            self.ctx
                .stream
                .memcpy_dtod(&h_n.data, &mut dst)
                .map_err(|e| anyhow!("mtp concat hidden half failed: {e}"))?;
        }
        let mut h_fc = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        gemm_batch(&self.ctx, &mtp.fc, &concat, &mut h_fc)?;

        // One transformer block (Full attn over the head's own per-block
        //       KV + dense MLP), mirroring the trunk layer body.
        let layer = &mtp.layer;
        let Qwen35Attn::Full(full_attn) = &layer.attn else {
            unreachable!("MTP head layer is always full attention");
        };

        let mut normed = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        rms_norm_offset(&self.ctx, &h_fc, &layer.input_layernorm, eps, &mut normed)?;

        // Local position within the fresh draft block (== head-KV row + RoPE pos).
        let start_pos_dev = upload_i32(&self.ctx, &[level as i32])?;
        let mut attn_out = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        self.full_attention_into(
            full_attn,
            &normed,
            &mut spec.head_k,
            &mut spec.head_v,
            0, // profiling label
            level,
            &start_pos_dev,
            &mut ws.full,
            &mut attn_out,
        )?;

        let mut hidden_mid = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        add_batch(&self.ctx, &h_fc, &attn_out, &mut hidden_mid)?;
        rms_norm_offset(
            &self.ctx,
            &hidden_mid,
            &layer.post_attention_layernorm,
            eps,
            &mut normed,
        )?;
        let mut mlp_out = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        if let Some(moe) = &layer.moe {
            let moe_cfg = self
                .moe_config
                .as_ref()
                .ok_or_else(|| anyhow!("MTP MoE layer but no moe_config"))?;
            crate::moe::moe_forward_into(
                &self.ctx,
                moe,
                &normed,
                moe_cfg,
                &self.expert_split,
                &mut ws.moe,
                &mut mlp_out,
            )?;
        } else {
            let mlp = layer
                .mlp
                .as_ref()
                .ok_or_else(|| anyhow!("MTP head layer missing MLP"))?;
            self.dense_mlp(mlp, &normed, &mut ws.dense, &mut mlp_out)?;
        }
        let mut h_layer = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        add_batch(&self.ctx, &hidden_mid, &mlp_out, &mut h_layer)?;

        // Final head RMSNorm + SHARED lm_head + token selection.
        rms_norm_offset(&self.ctx, &h_layer, &mtp.norm, eps, &mut normed)?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, 1)?;
        gemm_batch(&self.ctx, self.output_projection(), &normed, &mut logits)?;
        let logits_vec = DeviceVec {
            data: logits.data,
            len: vocab,
            label: "qwen35_mtp_draft_logits",
        };
        let next = if params.is_greedy() {
            argmax_into(&self.ctx, &logits_vec, &mut spec.argmax_scratch)?
        } else {
            // Filter + q-row retain + multinomial draw in one device call; the
            // uniform is the salted (seed, position) stream plain decode would
            // consume at this position (mirrors the DSpark draft draw).
            let u =
                dspark::unit_uniform(params.seed, dspark::SALT_DRAW, (start_pos + level) as u64);
            let cap = self.spec_draft_tokens.max(1);
            let q_all = spec.q_probs.get(&self.ctx, cap * vocab)?;
            let tok_out = spec.sample_tok.get(&self.ctx, 1)?;
            {
                let (l_ptr, _gl) = logits_vec.data.device_ptr(&self.ctx.stream);
                let (q_ptr, _gq) = q_all.device_ptr_mut(&self.ctx.stream);
                let (t_ptr, _gt) = tok_out.device_ptr_mut(&self.ctx.stream);
                // SAFETY: logits_vec holds `vocab` bf16; q row `level` < cap
                // (spec_step's depth guard bounds every level).
                unsafe {
                    ffi::dspark_draft_sample_cuda(
                        l_ptr as *const ffi::Half,
                        (q_ptr + (level * vocab * 4) as u64) as *mut f32,
                        t_ptr as *mut i32,
                        vocab as i32,
                        1.0 / params.temperature,
                        params.top_k,
                        params.top_p,
                        params.min_p,
                        u,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
            self.ctx.sync()?;
            self.ctx
                .stream
                .clone_dtoh(tok_out)
                .map_err(|e| anyhow!("D2H mtp draft token failed: {e}"))?[0] as u32
        };

        // The head's own output hidden seeds the next level (autoregressive chain,
        // per `dflash.rs` `step_hidden = flat`).
        let mut h_out = DeviceVec::zeros(&self.ctx, hidden)?;
        copy_row_to_vec(&self.ctx, &h_layer, 0, &mut h_out)?;
        Ok((next, h_out))
    }

    /// One depth-`depth` NextN-MTP speculative decode step (single CHAIN).
    ///
    /// Drafts a `depth`-token chain with the MTP head (each level autoregressive
    /// on the previous level's head hidden), verifies `[pending, d1..dD]` in a
    /// SINGLE trunk forward, accepts, and commits the accepted drafts + the
    /// trunk's bonus. Greedy rows (`params.is_greedy()`) accept the longest
    /// prefix whose draft token equals the trunk's argmax at that row (STRICT,
    /// k=1 top-1 match) — **token-exact to greedy no-spec decode** (every
    /// committed token is a trunk argmax); the correctness gate is spec greedy
    /// ≡ no-spec greedy (MoE non-determinism caveat applies to the 35B/27B MoE
    /// shapes). Sampled rows draft by device multinomial from the filtered head
    /// dist (q retained per level) and accept by chain rejection sampling
    /// ([`Self::mtp_accept_commit_sampled`]) — committed tokens are distributed
    /// exactly as filtered target sampling. A single chain (no sibling
    /// branching) keeps the 48 gated-delta linear layers' sequential recurrence
    /// correct; tree/top-k acceptance is a later, lossy increment.
    ///
    /// State contract (caller threads `pending`/`hidden` across steps):
    /// - `pending`: the last already-emitted token, (re)written into the KV at
    ///   `start_pos` by the verify (its KV is not yet materialized).
    /// - `hidden`: the trunk hidden that PRODUCED `pending` — the head's level-0
    ///   seed (matches Metal `prepare_draft_block_mtp` + DSv4 `spec.hidden`).
    /// - entry invariant: `slot.seq_len() == start_pos`.
    ///
    /// Returns `(emitted_tokens, next_pending, next_hidden)` with `k` = accepted
    /// draft count: emitted `[d1..dk, bonus]` (k+1 tokens); next_pending = bonus;
    /// next_hidden = the verify hidden of accepted row `k`. seq_len → `start_pos+k+1`.
    /// On full accept (`k==depth`) the verify already left the trunk state correct;
    /// on partial accept the trunk linear state is rolled back to post-`[pending,
    /// d1..dk]` via the pre-verify snapshot + a `(k+1)`-token replay (the full-attn
    /// KV self-heals via the seq_len rewind).
    #[allow(dead_code)] // wired into the executor decode dispatch in a later increment
    pub(crate) fn spec_step(
        &self,
        slot: &mut Qwen35SlotState,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        pending: u32,
        hidden: &DeviceVec,
        start_pos: usize,
        depth: usize,
        params: &SamplingParams,
        recall: Option<&mut Qwen35RecallForward>,
    ) -> Result<(Vec<CommittedToken>, u32, DeviceVec)> {
        ensure!(depth >= 1, "spec_step requires depth >= 1, got {depth}");
        // The MTP head KV (spec.head_k/head_v) was sized (spec_draft_tokens+1)
        // rows by new_spec_slot_state; a depth beyond that would overflow the
        // head KV in mtp_forward_level (row = level, 0..depth-1). Guard it.
        ensure!(
            depth <= self.spec_draft_tokens.max(1),
            "spec_step depth {depth} exceeds the MTP head KV capacity (model built for {} draft tokens)",
            self.spec_draft_tokens.max(1)
        );
        ensure!(
            slot.seq_len() == start_pos,
            "spec_step entry seq_len {} != start_pos {start_pos}",
            slot.seq_len()
        );
        let vocab = self.output_projection().rows;
        let hidden_size = self.config.hidden_size;
        let mut pt = mtp_phase_start(&self.ctx);

        // 1. Draft a depth-token chain: each level feeds the prior level's head
        //    hidden (autoregressive), starting from (pending, seed hidden).
        let mut h_prev = DeviceVec::zeros(&self.ctx, hidden_size)?;
        self.ctx
            .stream
            .memcpy_dtod(&hidden.data, &mut h_prev.data)
            .map_err(|e| anyhow!("spec seed hidden copy failed: {e}"))?;
        let mut chain: Vec<u32> = Vec::with_capacity(depth + 1);
        chain.push(pending);
        for level in 0..depth {
            let last_tok = *chain.last().unwrap();
            let (tok, h_out) =
                self.mtp_forward_level(spec, ws, last_tok, &h_prev, level, params, start_pos)?;
            chain.push(tok);
            h_prev = h_out;
        }
        let draft_ms = mtp_phase_lap(&self.ctx, &mut pt);

        // 2. Snapshot the trunk linear state BEFORE the verify (partial-accept base).
        spec.snapshot_trunk(&self.ctx, slot)?;
        let snap_ms = mtp_phase_lap(&self.ctx, &mut pt);

        // 3. Verify the whole chain in ONE trunk forward → per-row logits + hiddens.
        //    Advances the full-attn KV + 48 linear states by depth+1 tokens, and
        //    captures each linear layer's gated-delta inputs for ALL depth+1 rows
        //    (the cheap partial-accept replay reads them; see step 5).
        let (logits, dims, hiddens) = self.forward_tokens_verify(
            slot,
            ws,
            &chain,
            start_pos,
            Some(&mut spec.capture),
            recall,
        )?;
        ensure!(
            dims == [depth + 1, vocab],
            "spec verify dims {dims:?} != [{}, {vocab}]",
            depth + 1
        );
        let verify_ms = mtp_phase_lap(&self.ctx, &mut pt);

        // 4+5. Accept + commit, rolling the trunk back on partial accept (the
        //    verify over-advanced by depth+1 > k+1). Greedy: longest prefix
        //    where the draft == the trunk's argmax at that row. Sampled: chain
        //    rejection sampling over the shared-filter p/q dists.
        let (emitted, bonus, k) = if params.is_greedy() {
            let mut k = 0usize;
            let bonus;
            loop {
                let am = argmax_row_into(&self.ctx, &logits, k, vocab, &mut spec.argmax_scratch)?;
                if k < depth && am == chain[k + 1] {
                    k += 1;
                } else {
                    bonus = am;
                    break;
                }
            }
            // Greedy: delta policy, no behavior logprob (P6 sidecar skips greedy).
            let mut emitted: Vec<CommittedToken> =
                chain[1..=k].iter().map(|&t| (t, None)).collect();
            emitted.push((bonus, None));
            if k < depth {
                spec.restore_trunk(&self.ctx, slot)?;
                // LINEAR-ONLY replay: restore_trunk just rewound the 48 gated-delta
                // recurrent + conv rings to S_{start_pos}; re-advance ONLY them over
                // the accepted prefix `[pending, d1..dk]` (k+1 rows) from the verify
                // capture, skipping the full-attn blocks, MLP/MoE, final norm, and
                // lm_head — the dominant avoidable cost of the old full replay. The
                // 16 full-attn KV caches self-heal via position-indexing under the
                // explicit seq_len rewind below; MLP/MoE/lm_head leave no state.
                self.replay_linear_only(slot, ws, &spec.capture, k)?;
                slot.set_seq_len(start_pos + k + 1);
            }
            // else k==depth: verify already left seq_len=start_pos+depth+1, state correct.
            (emitted, bonus, k)
        } else {
            self.mtp_accept_commit_sampled(slot, spec, ws, &chain, &logits, start_pos, params)?
        };
        let accept_ms = mtp_phase_lap(&self.ctx, &mut pt);

        // next_hidden = the verify hidden of accepted row k (produced `bonus`).
        let mut next_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        {
            let src = hiddens.data.slice(k * hidden_size..(k + 1) * hidden_size);
            self.ctx
                .stream
                .memcpy_dtod(&src, &mut next_hidden.data)
                .map_err(|e| anyhow!("spec next-hidden copy failed: {e}"))?;
        }

        if pt.is_some() {
            eprintln!(
                "[mtp-phase] depth={depth} accept={k} draft={draft_ms:.2} snap={snap_ms:.2} verify={verify_ms:.2} accept_commit={accept_ms:.2} ms"
            );
        }
        Ok((emitted, bonus, next_hidden))
    }

    /// Rejection-sampling twin of the greedy accept scan in [`Self::spec_step`]
    /// — the port of [`Self::dspark_accept_commit_sampled`] onto the NextN-MTP
    /// lane (mirrors flashinfer/SGLang `chain_speculative_sampling`): accept
    /// `chain[j+1]` with prob min(1, p_j(tok)/q_j(tok)); the first reject
    /// commits a residual `max(0, p−q)` renormalized draw, full accept a bonus
    /// draw from the last row. Exactness invariant: p and q pass the SAME
    /// engine-sampler filter (temp/top_k/top_p/min_p), so committed tokens are
    /// distributed exactly as filtered target sampling. Identical rollback set
    /// to the greedy path: `restore_trunk` + `replay_linear_only` +
    /// `set_seq_len` (the full-attn KV self-heals under the caller's seq
    /// rewind / pool truncate). Returns `(emitted-with-logprobs, bonus, k)`.
    fn mtp_accept_commit_sampled(
        &self,
        slot: &mut Qwen35SlotState,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        chain: &[u32],
        logits: &DeviceVec,
        start_pos: usize,
        params: &SamplingParams,
    ) -> Result<(Vec<CommittedToken>, u32, usize)> {
        let ctx = &self.ctx;
        let depth = chain.len() - 1;
        let cap = self.spec_draft_tokens.max(1);
        ensure!(
            depth <= cap,
            "mtp sampled verify: depth {depth} > head cap {cap}"
        );
        let vocab = self.output_projection().rows;
        // Uniform streams at pos = start_pos + j + 1 (identical to the host
        // path's per-step draws — position-salted, so batching changes nothing).
        let pos = |j: usize| (start_pos + j + 1) as u64;
        let u_acc: Vec<f32> = (0..depth)
            .map(|j| dspark::unit_uniform(params.seed, dspark::SALT_ACCEPT, pos(j)))
            .collect();
        let u_res: Vec<f32> = (0..=depth)
            .map(|j| dspark::unit_uniform(params.seed, dspark::SALT_RESIDUAL, pos(j)))
            .collect();
        let draft: Vec<i32> = chain[1..].iter().map(|&t| t as i32).collect();

        let p_all = spec.p_probs.get(ctx, (cap + 1) * vocab)?;
        let q_all = spec.q_probs.get(ctx, cap * vocab)?;
        let draft_dev = spec.chain_draft.get(ctx, cap)?;
        let ua_dev = spec.u_accept.get(ctx, cap)?;
        let ur_dev = spec.u_residual.get(ctx, cap + 1)?;
        let out_dev = spec.accept_out.get(ctx, 2)?;
        ctx.stream
            .memcpy_htod(&draft, &mut draft_dev.slice_mut(0..depth))
            .and_then(|()| {
                ctx.stream
                    .memcpy_htod(&u_acc, &mut ua_dev.slice_mut(0..depth))
            })
            .and_then(|()| {
                ctx.stream
                    .memcpy_htod(&u_res, &mut ur_dev.slice_mut(0..=depth))
            })
            .map_err(|e| anyhow!("H2D mtp chain inputs failed: {e}"))?;
        {
            let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
            let (p_ptr, _gp) = p_all.device_ptr_mut(&ctx.stream);
            let (q_ptr, _gq) = q_all.device_ptr(&ctx.stream);
            let (d_ptr, _gd) = draft_dev.device_ptr(&ctx.stream);
            let (ua_ptr, _gua) = ua_dev.device_ptr(&ctx.stream);
            let (ur_ptr, _gur) = ur_dev.device_ptr(&ctx.stream);
            let (o_ptr, _go) = out_dev.device_ptr_mut(&ctx.stream);
            // SAFETY: logits holds chain.len()*vocab bf16; p/q scratches hold
            // (cap+1)/cap vocab-rows and depth <= cap (ensured above); the q
            // rows were written by this step's draft; draft/u prefixes uploaded
            // just above.
            unsafe {
                ffi::dspark_filter_probs_cuda(
                    l_ptr as *const ffi::Half,
                    p_ptr as *mut f32,
                    chain.len() as i32,
                    vocab as i32,
                    1.0 / params.temperature,
                    params.top_k,
                    params.top_p,
                    params.min_p,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::dspark_chain_accept_cuda(
                    q_ptr as *const f32,
                    p_ptr as *const f32,
                    d_ptr as *const i32,
                    ua_ptr as *const f32,
                    ur_ptr as *const f32,
                    o_ptr as *mut i32,
                    depth as i32,
                    vocab as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        ctx.sync()?;
        let out = ctx
            .stream
            .clone_dtoh(out_dev)
            .map_err(|e| anyhow!("D2H mtp chain verdict failed: {e}"))?;
        let (k, bonus) = (out[0] as usize, out[1] as u32);
        ensure!(
            k <= depth,
            "mtp chain kernel returned k {k} > depth {depth}"
        );
        let mut tokens: Vec<u32> = chain[1..=k].to_vec();
        tokens.push(bonus);
        // Behavior logprobs: committed token j is marginally distributed as the
        // filtered target dist p_j (chain rejection-sampling exactness), and the
        // p rows are still materialized + final (verdict D2H synced above).
        let logprobs = chain_commit_logprobs(ctx, p_all, vocab, &tokens)?;
        let emitted = tokens
            .into_iter()
            .zip(logprobs)
            .map(|(t, lp)| (t, Some(lp)))
            .collect();
        if k < depth {
            spec.restore_trunk(ctx, slot)?;
            self.replay_linear_only(slot, ws, &spec.capture, k)?;
            slot.set_seq_len(start_pos + k + 1);
        }
        Ok((emitted, bonus, k))
    }

    /// Per-step student LoRA re-merge (OPD P2).
    ///
    /// Folds a fresh [`StudentLoraUpdate`] into the resident student projection
    /// weights in place. On the first call the pristine base weight of every
    /// touched projection is cached host-side; each call then recomputes
    /// `W = base + (alpha/rank)·(B·A)` from that pristine base — so re-merging
    /// never compounds onto an already-merged weight.
    ///
    /// Read-only borrow of every resident FP8 block-scaled base projection's
    /// device pointers, for train-infer weight sharing (`--share-frozen-base`).
    ///
    /// Walks each layer's full-attention (q/k/v/o), linear-attention
    /// (in_proj_qkv/z + out_proj — in_proj_a/b are tiny BF16 and skipped by the
    /// FP8 filter), dense-MLP (gate/up/down), and MoE projections, emitting a
    /// [`SharedFp8BaseProjection`] for every one stored as resident block-scaled
    /// FP8. MoE routed experts come from either
    /// the per-expert `DeviceMatrix` vecs (DeepGEMM disabled) or sliced out of
    /// the fused `w13`/`down` grouped FP8 buffers (default), and the shared
    /// expert from its individual FP8 matrices. The train side picks the subset
    /// it actually shares (frozen, non-LoRA-target tensors) by matching
    /// `(layer_idx, proj_suffix)`. Read-only: returns raw pointers, never
    /// exposes mutation. Single-GPU only — TP/EP shards would split the base, so
    /// group index equals global expert index.
    pub(crate) fn frozen_base_fp8_pointers(&self) -> Result<Vec<SharedFp8BaseProjection>> {
        ensure!(
            self.tp.is_single(),
            "frozen-base FP8 sharing is single-GPU only; got TP world_size={}",
            self.tp.config().world_size
        );
        // From here on a LoRA FP8→BF16 promotion must retire (not free) the FP8
        // buffers: the importer holds non-owning views of these pointers.
        self.frozen_base_ptrs_exported
            .store(true, Ordering::Relaxed);
        let ctx = &self.ctx;
        // Plain helper (not a closure) so the grouped-buffer `out.push` calls
        // below don't conflict with a captured `&mut out` borrow.
        fn push(
            out: &mut Vec<SharedFp8BaseProjection>,
            ctx: &DeviceContext,
            layer_idx: usize,
            suffix: String,
            m: &DeviceMatrix,
        ) {
            if let Some((weight_ptr, scale_ptr, rows, cols, block_m, block_k)) =
                m.fp8_block_scaled_ptrs(ctx)
            {
                out.push(SharedFp8BaseProjection {
                    layer_idx,
                    proj_suffix: suffix,
                    weight_ptr,
                    scale_ptr,
                    rows,
                    cols,
                    block_m,
                    block_k,
                });
            }
        }
        let mut out = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Qwen35Attn::Full(full) = &layer.attn {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "self_attn.qkv_proj".to_string(),
                    &full.qkv_proj,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "self_attn.o_proj".to_string(),
                    &full.o_proj,
                );
            }
            if let Qwen35Attn::Linear(lin) = &layer.attn {
                // Gated-delta linear attention. in_proj_qkv/z + out_proj ship as
                // resident FP8 in the Qwen3.5/3.6 hybrid checkpoint; in_proj_a/b
                // are tiny per-head BF16 (fp8_block_scaled_ptrs skips them via the
                // format filter). The train student names these
                // `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (HF spec).
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.in_proj_qkvz".to_string(),
                    &lin.in_proj_qkvz,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.in_proj_ba".to_string(),
                    &lin.in_proj_ba,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.out_proj".to_string(),
                    &lin.out_proj,
                );
            }
            if let Some(mlp) = &layer.mlp {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.gate_up_proj".to_string(),
                    &mlp.gate_up_proj,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.down_proj".to_string(),
                    &mlp.down_proj,
                );
            }
            if let Some(moe) = &layer.moe {
                // Shared expert: always individual FP8 DeviceMatrix (mirror dense).
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.gate_proj".to_string(),
                    &moe.shared_gate,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.up_proj".to_string(),
                    &moe.shared_up,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.down_proj".to_string(),
                    &moe.shared_down,
                );

                // Routed experts. Two storage layouts:
                //  (A) per-expert Vec<DeviceMatrix> populated (DeepGEMM disabled) -> borrow each directly.
                //  (B) fused FP8 grouped buffers (default) -> slice per-expert ptrs into the group.
                if !moe.gate.is_empty() {
                    // Layout A: per-expert DeviceMatrix retained.
                    for (e, m) in moe.gate.iter().enumerate() {
                        push(
                            &mut out,
                            ctx,
                            layer_idx,
                            format!("mlp.experts.{e}.gate_proj"),
                            m,
                        );
                    }
                    for (e, m) in moe.up.iter().enumerate() {
                        push(
                            &mut out,
                            ctx,
                            layer_idx,
                            format!("mlp.experts.{e}.up_proj"),
                            m,
                        );
                    }
                    for (e, m) in moe.down.iter().enumerate() {
                        push(
                            &mut out,
                            ctx,
                            layer_idx,
                            format!("mlp.experts.{e}.down_proj"),
                            m,
                        );
                    }
                } else {
                    // Layout B: slice into the fused grouped FP8 buffers.
                    // Single-GPU: group index == global expert index.
                    if let Some(w13) = &moe.w13_fp8_grouped {
                        // rows = 2*moe_intermediate; gate = rows[0..mi], up = rows[mi..2*mi].
                        let mi = w13.rows / 2;
                        for e in 0..w13.groups {
                            if let Some(p) = w13.expert_slice_fp8_ptrs(ctx, e, 0, mi) {
                                out.push(SharedFp8BaseProjection {
                                    layer_idx,
                                    proj_suffix: format!("mlp.experts.{e}.gate_proj"),
                                    weight_ptr: p.0,
                                    scale_ptr: p.1,
                                    rows: p.2,
                                    cols: p.3,
                                    block_m: p.4,
                                    block_k: p.5,
                                });
                            }
                            if let Some(p) = w13.expert_slice_fp8_ptrs(ctx, e, mi, mi) {
                                out.push(SharedFp8BaseProjection {
                                    layer_idx,
                                    proj_suffix: format!("mlp.experts.{e}.up_proj"),
                                    weight_ptr: p.0,
                                    scale_ptr: p.1,
                                    rows: p.2,
                                    cols: p.3,
                                    block_m: p.4,
                                    block_k: p.5,
                                });
                            }
                        }
                    }
                    if let Some(down) = &moe.down_fp8_grouped {
                        for e in 0..down.groups {
                            if let Some(p) = down.expert_slice_fp8_ptrs(ctx, e, 0, down.rows) {
                                out.push(SharedFp8BaseProjection {
                                    layer_idx,
                                    proj_suffix: format!("mlp.experts.{e}.down_proj"),
                                    weight_ptr: p.0,
                                    scale_ptr: p.1,
                                    rows: p.2,
                                    cols: p.3,
                                    block_m: p.4,
                                    block_k: p.5,
                                });
                            }
                        }
                    }
                }
            }
        }
        if std::env::var("ARLE_SHARE_BASE_DIAG").is_ok() {
            let mut attn = 0usize;
            let mut lin = 0usize;
            let mut mlp = 0usize;
            let mut shared = 0usize;
            let mut experts = 0usize;
            for e in &out {
                if e.proj_suffix.starts_with("self_attn") {
                    attn += 1;
                } else if e.proj_suffix.starts_with("linear_attn") {
                    lin += 1;
                } else if e.proj_suffix.starts_with("mlp.shared_expert") {
                    shared += 1;
                } else if e.proj_suffix.starts_with("mlp.experts") {
                    experts += 1;
                } else if e.proj_suffix.starts_with("mlp.") {
                    mlp += 1;
                }
            }
            eprintln!(
                "[share-base-diag] emitted {} entries: full_attn={attn} linear_attn={lin} dense_mlp={mlp} shared_expert={shared} routed_experts={experts}; sample suffixes: {:?}",
                out.len(),
                out.iter()
                    .take(10)
                    .map(|e| (
                        e.layer_idx,
                        e.proj_suffix.as_str(),
                        e.rows,
                        e.cols,
                        e.block_m,
                        e.block_k
                    ))
                    .collect::<Vec<_>>()
            );
        }
        Ok(out)
    }

    /// `A` is `[rank, in]`, `B` is `[out, rank]`, matching the train-side
    /// `LinearWithLora` contract. The next `forward_tokens` picks up the merged
    /// resident matrices automatically.
    pub(crate) fn remerge_student_lora(&mut self, update: StudentLoraUpdate) -> Result<()> {
        ensure!(update.rank > 0, "student LoRA update has rank=0");
        ensure!(
            self.tp.is_single(),
            "student LoRA re-merge is currently single-GPU only; got TP world_size={}",
            self.tp.config().world_size
        );
        let scale = update.alpha / update.rank as f32;
        let num_layers = self.config.num_hidden_layers;

        for layer in &update.layers {
            let layer_idx = layer.layer_idx;
            ensure!(
                layer_idx < num_layers,
                "student LoRA references layer {layer_idx} but model has {num_layers} layers"
            );
            ensure!(
                !layer.projections.is_empty(),
                "student LoRA layer {layer_idx} carries no projection updates"
            );

            for projection in &layer.projections {
                self.merge_lora_proj(
                    layer_idx,
                    projection.projection,
                    &projection.matrices,
                    scale,
                )?;
            }
        }
        self.ctx.sync()?;
        Ok(())
    }

    /// Promote an FP8-block-scaled LoRA target to dense BF16 on first touch:
    /// dequantize on-device into a fresh dense buffer and swap the matrix's
    /// resident storage in place (same rows/cols). Replaces the former host
    /// remerge lane (O(rows·cols·rank) triple loop + re-quant + full-W upload,
    /// 60-83s/round) with a one-time kernel; every later re-merge rides the
    /// on-device dense lane. Dense targets are a no-op; grouped targets error
    /// in [`Qwen35Model::lora_matrix_mut`].
    ///
    /// VRAM: trades FP8→BF16 storage (2×) for the touched projections only;
    /// their rollout GEMMs ride the dense-BF16 path thereafter (both formats
    /// are first-class in serving). If `--share-frozen-base` exported the FP8
    /// pointers, the retired buffers are kept alive (aliased non-owningly by
    /// the autograd student); otherwise they are freed.
    fn promote_lora_target_to_bf16(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<()> {
        let label = projection.label();
        let ctx = self.ctx.clone();
        let matrix = self.lora_matrix_mut(layer_idx, projection)?;
        if matrix.is_dense_bf16() {
            return Ok(());
        }
        ensure!(
            matrix.weight_format() == WeightFormat::Fp8BlockScaled,
            "layer {layer_idx} {label}: LoRA merge supports dense BF16 or FP8 block-scaled \
             weights; got {:?}",
            matrix.weight_format()
        );
        ensure!(
            matrix.quant_block_m > 0
                && matrix.quant_block_k > 0
                && matrix.quant_scale_rows > 0
                && matrix.quant_scale_cols > 0,
            "layer {layer_idx} {label}: FP8 LoRA target missing block-scale metadata"
        );
        let mut dense = ctx
            .stream
            .alloc_zeros::<bf16>(matrix.rows * matrix.cols)
            .map_err(|e| anyhow!("layer {layer_idx} {label}: BF16 promotion alloc failed: {e}"))?;
        {
            let qweight = matrix.qweight_u8.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 LoRA target missing qweight")
            })?;
            let scales = matrix.scale_f32.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 LoRA target missing f32 scales")
            })?;
            ensure!(
                qweight.len() == matrix.rows * matrix.cols,
                "layer {layer_idx} {label}: FP8 qweight len {} != rows*cols {}",
                qweight.len(),
                matrix.rows * matrix.cols
            );
            let (qw_ptr, _gq) = qweight.device_ptr(&ctx.stream);
            let (scale_ptr, _gs) = scales.device_ptr(&ctx.stream);
            let (dense_ptr, _gd) = dense.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
                    qw_ptr as *const u8,
                    scale_ptr as *const f32,
                    dense_ptr as *mut ffi::Half,
                    matrix.rows as i32,
                    matrix.cols as i32,
                    matrix.quant_scale_rows as i32,
                    matrix.quant_scale_cols as i32,
                    matrix.quant_block_m as i32,
                    matrix.quant_block_k as i32,
                    ctx.stream.cu_stream(),
                )
            }
            .result()
            .map_err(|e| {
                anyhow!("layer {layer_idx} {label}: FP8→BF16 promotion dequant failed: {e}")
            })?;
        }
        // Success — swap the resident storage to dense BF16 (infallible).
        let retired = (matrix.qweight_u8.take(), matrix.scale_f32.take());
        matrix.data = dense;
        matrix.weight_format = WeightFormat::DenseBf16;
        matrix.quant_scale_rows = 0;
        matrix.quant_scale_cols = 0;
        matrix.quant_block_m = 0;
        matrix.quant_block_k = 0;
        if self.frozen_base_ptrs_exported.load(Ordering::Relaxed)
            && let (Some(qweight), Some(scales)) = retired
        {
            self.lora_promoted_fp8_keepalive.push((qweight, scales));
        }
        Ok(())
    }

    /// Base-cache key for a projection. Projections that live inside a
    /// row-fused matrix (MlpUp shares `gate_up_proj` with MlpGate) map to one
    /// canonical key so the pristine device base is cached once per underlying
    /// buffer — captured before either half's first merge.
    fn lora_base_cache_key(layer_idx: usize, projection: StudentLoraProjection) -> LoraBaseKey {
        let projection = match projection {
            StudentLoraProjection::MlpUp => StudentLoraProjection::MlpGate,
            StudentLoraProjection::LinearA => StudentLoraProjection::LinearB,
            StudentLoraProjection::FullK | StudentLoraProjection::FullV => {
                StudentLoraProjection::FullQ
            }
            StudentLoraProjection::LinearZ => StudentLoraProjection::LinearQkv,
            other => other,
        };
        LoraBaseKey {
            layer_idx,
            projection,
        }
    }

    /// `(row_offset, rows)` of a projection inside its (possibly row-fused)
    /// resident matrix: e.g. MlpUp occupies `[inter_dim, inter_dim)` of
    /// `gate_up_proj`, FullK the `[q_gated, kv)` window of `qkv_proj`. A
    /// projection that owns its whole matrix spans `[0, matrix_rows)`.
    fn lora_row_window(
        &self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        matrix_rows: usize,
    ) -> (usize, usize) {
        let layer = &self.layers[layer_idx];
        let inter = || layer.mlp.as_ref().map(DenseMlp::inter_dim).unwrap_or(0);
        let vh = || match &layer.attn {
            Qwen35Attn::Linear(lin) => lin.in_proj_ba.rows / 2,
            _ => 0,
        };
        let q_gated = self.local_full_attn_q_proj_dim();
        let kv = self.local_kv_heads * self.config.head_dim;
        let qkv = self.local_linear_qkv_dim();
        match projection {
            StudentLoraProjection::MlpGate => (0, inter()),
            StudentLoraProjection::MlpUp => (inter(), inter()),
            StudentLoraProjection::LinearB => (0, vh()),
            StudentLoraProjection::LinearA => (vh(), vh()),
            StudentLoraProjection::FullQ => (0, q_gated),
            StudentLoraProjection::FullK => (q_gated, kv),
            StudentLoraProjection::FullV => (q_gated + kv, kv),
            StudentLoraProjection::LinearQkv => (0, qkv),
            StudentLoraProjection::LinearZ => (qkv, self.local_linear_z_dim()),
            _ => (0, matrix_rows),
        }
    }

    /// Merge `W = base + scale·(B·A)` for one projection, entirely on device.
    /// FP8-stored targets are promoted to dense BF16 on first touch, so every
    /// projection rides one lane: pristine device base cache → rank-`rank`
    /// GEMM + row-window scaled-add into the resident matrix.
    fn merge_lora_proj(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        adapter: &StudentLoraMatrices,
        scale: f32,
    ) -> Result<()> {
        let label = projection.label();
        let key = LoraBaseKey {
            layer_idx,
            projection,
        };

        let adapter_is_zero = adapter.b.iter().all(|&value| value == 0.0);
        if adapter_is_zero && !self.lora_dirty.contains(&key) {
            return Ok(());
        }

        // Shape checks against the adapter's declared features (cheap; no copy).
        let rows = adapter.out_features;
        let cols = adapter.in_features;
        ensure!(
            adapter.a.len() == adapter.rank * cols,
            "layer {layer_idx} {label}: lora_A len {} != rank*in {}",
            adapter.a.len(),
            adapter.rank * cols
        );
        ensure!(
            adapter.b.len() == rows * adapter.rank,
            "layer {layer_idx} {label}: lora_B len {} != out*rank {}",
            adapter.b.len(),
            rows * adapter.rank
        );

        self.promote_lora_target_to_bf16(layer_idx, projection)?;

        if adapter_is_zero {
            // Restore the pristine *device* base (no host round-trip).
            if self.lora_dirty.remove(&key) {
                let cache_key = Self::lora_base_cache_key(layer_idx, projection);
                self.restore_lora_base_dev(layer_idx, projection, &cache_key)?;
            }
            return Ok(());
        }
        self.merge_lora_proj_device(layer_idx, projection, adapter, scale, &key)
    }

    /// On-device dense-BF16 LoRA merge. Caches the pristine base on-device on
    /// first touch, then per step uploads the tiny A/B, runs `B·A` on the GPU,
    /// and folds `W = base + scale·(B·A)` straight into the resident matrix.
    fn merge_lora_proj_device(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        adapter: &StudentLoraMatrices,
        scale: f32,
        key: &LoraBaseKey,
    ) -> Result<()> {
        let label = projection.label();
        let rows = adapter.out_features;
        let cols = adapter.in_features;
        let rank = adapter.rank;

        // Pristine device base (device→device clone of the resident matrix on
        // first touch — runs before any merge mutates the weight). Row-fused
        // targets share one base under the canonical cache key.
        let cache_key = Self::lora_base_cache_key(layer_idx, projection);
        self.ensure_lora_base_dev_cached(layer_idx, projection, &cache_key)?;
        let (row_offset, _) = self.lora_row_window(layer_idx, projection, rows);

        // Tiny host→device uploads. `a_t` is A transposed to [cols, rank]
        // row-major (== W[c,k]=A[k,c] for `lora_device_gemm`); `b` uploads as-is
        // ([rows, rank] row-major == col-major X[k,r]=B[r,k]).
        let mut a_t = vec![bf16::ZERO; cols * rank];
        for k in 0..rank {
            let a_row = &adapter.a[k * cols..k * cols + cols];
            for (c, &a_kc) in a_row.iter().enumerate() {
                a_t[c * rank + k] = bf16::from_f32(a_kc);
            }
        }
        let b_host: Vec<bf16> = adapter.b.iter().map(|&v| bf16::from_f32(v)).collect();
        let a_t_dev = DeviceVec::from_host(&self.ctx, &a_t)?;
        let b_dev = DeviceVec::from_host(&self.ctx, &b_host)?;

        // Reusable delta scratch, grown to the largest dense matrix seen.
        let needed = rows * cols;
        if self
            .lora_delta_scratch
            .as_ref()
            .map(|s| s.len < needed)
            .unwrap_or(true)
        {
            self.lora_delta_scratch = Some(DeviceVec::zeros(&self.ctx, needed)?);
        }
        let ctx = self.ctx.clone();

        // GEMM `B·A` into the scratch buffer.
        {
            let scratch = self
                .lora_delta_scratch
                .as_mut()
                .expect("scratch allocated above");
            crate::ops::lora_device_gemm(
                &ctx,
                &a_t_dev.data,
                &b_dev.data,
                &mut scratch.data,
                rows,
                cols,
                rank,
            )?;
        }

        // Clone the (cheap, Arc-backed) device handles for the pristine base and
        // the delta scratch before taking the mutable target borrow — sidesteps
        // the whole-`self` borrow `lora_weight_target_mut` requires.
        let base_data = self
            .lora_base_dev
            .get(&cache_key)
            .ok_or_else(|| anyhow!("layer {layer_idx} {label}: device base not cached"))?
            .data
            .clone();
        let delta_data = self
            .lora_delta_scratch
            .as_ref()
            .expect("scratch allocated above")
            .data
            .clone();
        let delta_view = delta_data.slice(0..needed);

        let matrix = self.lora_matrix_mut(layer_idx, projection)?;
        ensure!(
            matrix.is_dense_bf16() && row_offset + rows <= matrix.rows && matrix.cols == cols,
            "layer {layer_idx} {label}: dense device merge shape/format mismatch \
             ({}x{} {:?} vs window [{row_offset}..{}]x{cols})",
            matrix.rows,
            matrix.cols,
            matrix.weight_format(),
            row_offset + rows
        );
        let window = row_offset * cols..row_offset * cols + needed;
        let base_view = base_data.slice(window.clone());
        let mut out_view = matrix.data.slice_mut(window);
        crate::ops::lora_scaled_add_into(
            &ctx,
            &base_view,
            &delta_view,
            &mut out_view,
            needed,
            scale,
        )?;

        self.lora_dirty.insert(*key);
        Ok(())
    }

    /// Capture a pristine *device* copy of one dense-BF16 base weight on first
    /// touch (device→device clone of the resident matrix).
    fn ensure_lora_base_dev_cached(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        key: &LoraBaseKey,
    ) -> Result<()> {
        if self.lora_base_dev.contains_key(key) {
            return Ok(());
        }
        let label = projection.label();
        let matrix = self.lora_matrix(layer_idx, projection)?;
        ensure!(
            matrix.is_dense_bf16(),
            "layer {layer_idx} {label}: device base cache requires dense BF16; got {:?}",
            matrix.weight_format()
        );
        let mut base = DeviceVec::zeros(&self.ctx, matrix.rows * matrix.cols)?;
        self.ctx
            .stream
            .memcpy_dtod(&matrix.data, &mut base.data)
            .map_err(|e| anyhow!("layer {layer_idx} {label}: device base D2D clone failed: {e}"))?;
        self.lora_base_dev.insert(*key, base);
        Ok(())
    }

    /// Restore a dense projection's resident matrix from its pristine device
    /// base (device→device copy; no host round-trip). Only the projection's
    /// own row window is restored, so the other half of a row-fused matrix
    /// keeps its (possibly merged) state.
    fn restore_lora_base_dev(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        key: &LoraBaseKey,
    ) -> Result<()> {
        let label = projection.label();
        let ctx = self.ctx.clone();
        let base_dev = self
            .lora_base_dev
            .get(key)
            .ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: device base not cached for restore")
            })?
            .data
            .clone();
        let matrix_rows = self.lora_matrix(layer_idx, projection)?.rows;
        // The base always spans the full (possibly fused) matrix; the window is
        // the whole buffer whenever this projection is the sole occupant.
        let (row_offset, rows) = self.lora_row_window(layer_idx, projection, matrix_rows);
        let matrix = self.lora_matrix_mut(layer_idx, projection)?;
        let window = row_offset * matrix.cols..(row_offset + rows) * matrix.cols;
        let src = base_dev.slice(window.clone());
        let mut dst = matrix.data.slice_mut(window);
        ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|e| {
            anyhow!("layer {layer_idx} {label}: device base restore D2D failed: {e}")
        })?;
        Ok(())
    }

    fn local_expert_idx(&self, global_expert: usize) -> Result<usize> {
        ensure!(
            self.expert_split.owns(global_expert),
            "Qwen3.6 LoRA sync expert {global_expert} is not local to this rank \
             (local range {}..{})",
            self.expert_split.local_expert_start,
            self.expert_split.local_expert_end()
        );
        Ok(global_expert - self.expert_split.local_expert_start)
    }

    fn lora_matrix(
        &self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<&DeviceMatrix> {
        let layer = &self.layers[layer_idx];
        match projection {
            StudentLoraProjection::FullQ
            | StudentLoraProjection::FullK
            | StudentLoraProjection::FullV
            | StudentLoraProjection::FullO => {
                let Qwen35Attn::Full(full) = &layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a full-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    // q/k/v live in the row-fused `qkv_proj`; callers address
                    // their window via `lora_row_window`.
                    StudentLoraProjection::FullQ
                    | StudentLoraProjection::FullK
                    | StudentLoraProjection::FullV => &full.qkv_proj,
                    StudentLoraProjection::FullO => &full.o_proj,
                    _ => unreachable!("full projection arm checked above"),
                })
            }
            StudentLoraProjection::LinearQkv
            | StudentLoraProjection::LinearZ
            | StudentLoraProjection::LinearB
            | StudentLoraProjection::LinearA
            | StudentLoraProjection::LinearOut => {
                let Qwen35Attn::Linear(lin) = &layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a linear-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    StudentLoraProjection::LinearQkv | StudentLoraProjection::LinearZ => {
                        &lin.in_proj_qkvz
                    }
                    StudentLoraProjection::LinearB | StudentLoraProjection::LinearA => {
                        &lin.in_proj_ba
                    }
                    StudentLoraProjection::LinearOut => &lin.out_proj,
                    _ => unreachable!("linear projection arm checked above"),
                })
            }
            StudentLoraProjection::MlpGate
            | StudentLoraProjection::MlpUp
            | StudentLoraProjection::MlpDown => {
                let dense = layer.mlp.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a dense MLP layer; MoE student LoRA sync is not supported",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    // Gate/up live in the row-fused `gate_up_proj`; callers
                    // address their half via `lora_row_offset`.
                    StudentLoraProjection::MlpGate | StudentLoraProjection::MlpUp => {
                        &dense.gate_up_proj
                    }
                    StudentLoraProjection::MlpDown => &dense.down_proj,
                    _ => unreachable!("mlp projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeRouter
            | StudentLoraProjection::MoeSharedGate
            | StudentLoraProjection::MoeSharedUp
            | StudentLoraProjection::MoeSharedDown
            | StudentLoraProjection::MoeSharedExpertGate => {
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    StudentLoraProjection::MoeRouter => &moe.router_gate,
                    StudentLoraProjection::MoeSharedGate => &moe.shared_gate,
                    StudentLoraProjection::MoeSharedUp => &moe.shared_up,
                    StudentLoraProjection::MoeSharedDown => &moe.shared_down,
                    StudentLoraProjection::MoeSharedExpertGate => &moe.shared_gate_router,
                    _ => unreachable!("shared MoE projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeExpertGate { expert_idx }
            | StudentLoraProjection::MoeExpertUp { expert_idx }
            | StudentLoraProjection::MoeExpertDown { expert_idx } => {
                let local_idx = self.local_expert_idx(expert_idx)?;
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                let experts = match projection {
                    StudentLoraProjection::MoeExpertGate { .. } => &moe.gate,
                    StudentLoraProjection::MoeExpertUp { .. } => &moe.up,
                    StudentLoraProjection::MoeExpertDown { .. } => &moe.down,
                    _ => unreachable!("expert MoE projection arm checked above"),
                };
                experts.get(local_idx).ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} expert matrix is not resident as a per-expert \
                         BF16 DeviceMatrix; grouped/FP8 MoE LoRA sync is not supported by this \
                         re-merge path",
                        projection.label()
                    )
                })
            }
        }
    }

    fn lora_matrix_mut(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<&mut DeviceMatrix> {
        let layer = &mut self.layers[layer_idx];
        match projection {
            StudentLoraProjection::FullQ
            | StudentLoraProjection::FullK
            | StudentLoraProjection::FullV
            | StudentLoraProjection::FullO => {
                let Qwen35Attn::Full(full) = &mut layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a full-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    StudentLoraProjection::FullQ
                    | StudentLoraProjection::FullK
                    | StudentLoraProjection::FullV => &mut full.qkv_proj,
                    StudentLoraProjection::FullO => &mut full.o_proj,
                    _ => unreachable!("full projection arm checked above"),
                })
            }
            StudentLoraProjection::LinearQkv
            | StudentLoraProjection::LinearZ
            | StudentLoraProjection::LinearB
            | StudentLoraProjection::LinearA
            | StudentLoraProjection::LinearOut => {
                let Qwen35Attn::Linear(lin) = &mut layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a linear-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    StudentLoraProjection::LinearQkv | StudentLoraProjection::LinearZ => {
                        &mut lin.in_proj_qkvz
                    }
                    StudentLoraProjection::LinearB | StudentLoraProjection::LinearA => {
                        &mut lin.in_proj_ba
                    }
                    StudentLoraProjection::LinearOut => &mut lin.out_proj,
                    _ => unreachable!("linear projection arm checked above"),
                })
            }
            StudentLoraProjection::MlpGate
            | StudentLoraProjection::MlpUp
            | StudentLoraProjection::MlpDown => {
                let dense = layer.mlp.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a dense MLP layer; MoE student LoRA sync is not supported",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    StudentLoraProjection::MlpGate | StudentLoraProjection::MlpUp => {
                        &mut dense.gate_up_proj
                    }
                    StudentLoraProjection::MlpDown => &mut dense.down_proj,
                    _ => unreachable!("mlp projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeRouter
            | StudentLoraProjection::MoeSharedGate
            | StudentLoraProjection::MoeSharedUp
            | StudentLoraProjection::MoeSharedDown
            | StudentLoraProjection::MoeSharedExpertGate => {
                let moe = layer.moe.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    StudentLoraProjection::MoeRouter => &mut moe.router_gate,
                    StudentLoraProjection::MoeSharedGate => &mut moe.shared_gate,
                    StudentLoraProjection::MoeSharedUp => &mut moe.shared_up,
                    StudentLoraProjection::MoeSharedDown => &mut moe.shared_down,
                    StudentLoraProjection::MoeSharedExpertGate => &mut moe.shared_gate_router,
                    _ => unreachable!("shared MoE projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeExpertGate { expert_idx }
            | StudentLoraProjection::MoeExpertUp { expert_idx }
            | StudentLoraProjection::MoeExpertDown { expert_idx } => {
                let local_start = self.expert_split.local_expert_start;
                let local_end = self.expert_split.local_expert_end();
                ensure!(
                    (local_start..local_end).contains(&expert_idx),
                    "Qwen3.6 LoRA sync expert {expert_idx} is not local to this rank \
                     (local range {local_start}..{local_end})"
                );
                let local_idx = expert_idx - local_start;
                let moe = layer.moe.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                let experts = match projection {
                    StudentLoraProjection::MoeExpertGate { .. } => &mut moe.gate,
                    StudentLoraProjection::MoeExpertUp { .. } => &mut moe.up,
                    StudentLoraProjection::MoeExpertDown { .. } => &mut moe.down,
                    _ => unreachable!("expert MoE projection arm checked above"),
                };
                experts.get_mut(local_idx).ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} expert matrix is not resident as a per-expert \
                         BF16 DeviceMatrix; grouped/FP8 MoE LoRA sync is not supported by this \
                         re-merge path",
                        projection.label()
                    )
                })
            }
        }
    }

    /// Dense SwiGLU MLP into `out` (`[hidden, seq]`). One GEMM over the
    /// row-fused `[gate; up]` weight, then the fused SwiGLU reads each row's
    /// halves in place. Every stage fully overwrites its scratch buffer.
    fn dense_mlp(
        &self,
        mlp: &DenseMlp,
        normed: &HiddenStates,
        dw: &mut DenseMlpScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let inter = mlp.inter_dim();
        let seq_len = normed.seq_len;
        let gate_up = dw.gate_up.get(&self.ctx, 2 * inter, seq_len)?;
        gemm_batch(&self.ctx, &mlp.gate_up_proj, normed, gate_up)?;
        let act = dw.act.get(&self.ctx, inter, seq_len)?;
        silu_mul_fused(&self.ctx, gate_up, act)?;
        gemm_batch(&self.ctx, &mlp.down_proj, act, out)?;
        Ok(())
    }

    /// Thin trunk wrapper: full attention writing this rank's per-slot K/V cache
    /// for full-attn layer `full_idx`. The MTP draft head bypasses this and calls
    /// [`Self::full_attention_into`] directly with its own per-block KV.
    #[allow(clippy::too_many_arguments)]
    fn full_attention(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        full_idx: usize,
        start_pos: usize,
        start_pos_dev: &CudaSlice<i32>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let k_cache = &mut slot.k_caches[full_idx];
        let v_cache = &mut slot.v_caches[full_idx];
        self.full_attention_into(
            attn,
            normed,
            k_cache,
            v_cache,
            full_idx,
            start_pos,
            start_pos_dev,
            fw,
            out,
        )
    }

    /// Gated full attention over an explicit contiguous K/V cache (`max_seq_len`
    /// derived from `k_cache.len / kv_dim`), uncached recompute over
    /// `[0, start_pos+seq_len)` each call, into `out` (`[hidden, seq]`, beta=0
    /// o_proj GEMM). The prep kernel fuses q/k RMSNorm + RoPE + cache write; the
    /// gate kernel applies the per-head sigmoid gate carried in `q_full`.
    /// `start_pos_dev` is the GPU-resident `start_pos` (same for every layer of
    /// one call). Prefill chunks (`seq_len > 1`) route through the vendored FA3
    /// hopper fwd when [`qwen35_fa3_enabled`]; decode keeps the devpos kernel
    /// (graph-captured) untouched. The trunk passes its per-slot cache via
    /// [`Self::full_attention`]; the MTP draft head passes its own fresh
    /// per-block cache. `full_idx` is only a profiling label. Numerics, kernels,
    /// and launch order are identical to the pre-extraction trunk path.
    #[allow(clippy::too_many_arguments)]
    fn full_attention_into(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        k_cache: &mut DeviceVec,
        v_cache: &mut DeviceVec,
        full_idx: usize,
        start_pos: usize,
        start_pos_dev: &CudaSlice<i32>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        // LOCAL per-rank widths (= global config on a single GPU): the sharded
        // q/k/v GEMM outputs, the per-slot caches, and the kernel launches must
        // all agree on this rank's head shard.
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();

        let FullAttnScratch {
            qkv_fused,
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse,
            fa3_oaccum: _,
            fa3_lseaccum: _,
            fa3_semaphore,
            batch_partial_out: _,
            batch_partial_m: _,
            batch_partial_l: _,
        } = fw;
        let qkv_fused = qkv_fused.get(&self.ctx, q_proj_dim + 2 * kv_dim, seq_len)?;
        let q_full = q_full.get(&self.ctx, q_proj_dim, seq_len)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, seq_len)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, seq_len)?;
        qwen35_profile(
            &self.ctx,
            "qwen/full/qkv_gemm",
            Some(full_idx),
            seq_len,
            || {
                gemm_batch(&self.ctx, &attn.qkv_proj, normed, qkv_fused)?;
                split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;
                Ok(())
            },
        )?;

        let q_prepped = q_prepped.get(&self.ctx, q_dim, seq_len)?;
        let attn_out = attn_heads.get(&self.ctx, q_dim, seq_len)?;

        let max_seq_len = k_cache.len / kv_dim;
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let kv_len = start_pos + seq_len;

        // Prep: q/k RMSNorm + RoPE + write K/V into the contiguous cache.
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (kc_ptr, _g8) = k_cache.data.device_ptr_mut(&self.ctx.stream);
            let (vc_ptr, _g9) = v_cache.data.device_ptr_mut(&self.ctx.stream);
            let (sp_ptr, _g10) = start_pos_dev.device_ptr(&self.ctx.stream);
            qwen35_profile(&self.ctx, "qwen/full/prep", Some(full_idx), seq_len, || {
                // SAFETY: all buffers valid on ctx.stream; cache sized max_seq_len*kv_dim.
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qf_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        qp_ptr as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        seq_len as i32,
                        sp_ptr as *const i32,
                        c.rotary_dim as i32,
                        c.rms_norm_eps,
                        max_seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        // Attention over the contiguous cache (causal; decode = qlen 1).
        // Decode (`seq_len == 1`) takes the devpos entry: the kv length is read
        // from the staged `start_pos` DEVICE buffer inside the kernel (same
        // math — kv_len = start_pos + token + 1 either way), so the launch is
        // CUDA-graph capture-safe and ONE captured graph replays across
        // positions. Eager decode uses the same entry, keeping the graph lane
        // kernel-for-kernel identical to its eager warm runs. Prefill keeps
        // the host-scalar entry (multi-token, never captured).
        {
            let (q_ptr, _g0) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (kc_ptr, _g1) = k_cache.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _g2) = v_cache.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_prepped/caches/out valid on ctx.stream for the shapes
            // above; `start_pos_dev` is the forward-level staged position (one
            // i32, value == start_pos).
            qwen35_profile(
                &self.ctx,
                "qwen/full/attention",
                Some(full_idx),
                seq_len,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        if seq_len == 1
                            && qwen35_fa2_sm70_enabled(&self.ctx)
                            && !crate::runtime_flags::qwen35_decode_graph()
                        {
                            // FA2 sm_70 decode (eager). The host kv_len arg is
                            // not graph-replay safe, so captured decode falls
                            // through to the devpos kernel below.
                            ffi::arle_fa2_sm70_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else if seq_len == 1 {
                            let (sp_ptr, _g4) = start_pos_dev.device_ptr(&self.ctx.stream);
                            ffi::nonpaged_prefill_attention_devpos_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                sp_ptr as *const i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else if c.head_dim == 256 && qwen35_fa3_enabled(&self.ctx) {
                            // FA3 fwd over the SAME buffers the in-tree kernel uses:
                            // q/out token-major [S, h, 256] (HD256 prep layout),
                            // cache head-major [h_k, max_seq, 256]. Passing the
                            // exact `kv_len` as seqlen_k keeps the shim on the
                            // non-varlen path; causal is bottom-right aligned =
                            // chunked-prefill semantics. Gate + o_proj follow
                            // unchanged.
                            let lse = fa3_lse.get(&self.ctx, self.local_q_heads * seq_len)?;
                            let sem = fa3_semaphore.get(&self.ctx, 5)?;
                            let (lse_ptr, _g4) = lse.device_ptr_mut(&self.ctx.stream);
                            let (sem_ptr, _g5) = sem.device_ptr_mut(&self.ctx.stream);
                            let head_dim = c.head_dim as i64;
                            let args = ffi::ArleFa3FwdHd256Args {
                                q: q_ptr as *const ffi::Half,
                                k: kc_ptr as *const ffi::Half,
                                v: vc_ptr as *const ffi::Half,
                                o: o_ptr as *mut ffi::Half,
                                softmax_lse: lse_ptr as *mut f32,
                                out_accum: std::ptr::null_mut(),
                                softmax_lse_accum: std::ptr::null_mut(),
                                tile_count_semaphore: sem_ptr as *mut i32,
                                metadata_capacity: 5,
                                cu_seqlens_q: std::ptr::null(),
                                seqused_k: std::ptr::null(),
                                batch: 1,
                                total_q: seq_len as i32,
                                seqlen_q: seq_len as i32,
                                seqlen_k: kv_len as i32,
                                num_heads: self.local_q_heads as i32,
                                num_heads_k: self.local_kv_heads as i32,
                                head_dim: c.head_dim as i32,
                                q_row_stride: q_dim as i64,
                                k_row_stride: head_dim,
                                v_row_stride: head_dim,
                                o_row_stride: q_dim as i64,
                                q_head_stride: head_dim,
                                k_head_stride: max_seq_len as i64 * head_dim,
                                v_head_stride: max_seq_len as i64 * head_dim,
                                o_head_stride: head_dim,
                                softmax_scale: sm_scale,
                                is_causal: 1,
                                num_splits: 1,
                                page_table: std::ptr::null(),
                                page_table_batch_stride: 0,
                                page_size: 0,
                                num_pages: 0,
                                k_page_stride: 0,
                                v_page_stride: 0,
                            };
                            ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, self.ctx.stream.cu_stream())
                                .result()?;
                        } else if qwen35_fa2_sm70_enabled(&self.ctx) {
                            // FA2 sm_70 prefill (SOTA on V100; FA3 is sm_80+).
                            // Causal chunked-prefill; same buffers as naive.
                            ffi::arle_fa2_sm70_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else {
                            ffi::nonpaged_prefill_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                    }
                    Ok(())
                },
            )?;
        }

        // Per-head sigmoid gate from q_full's gate half.
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            qwen35_profile(&self.ctx, "qwen/full/gate", Some(full_idx), seq_len, || {
                // SAFETY: q_full/attn_out valid on ctx.stream; gate layout per full-attn prep.
                unsafe {
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        c.head_dim as i32,
                        seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        qwen35_profile(
            &self.ctx,
            "qwen/full/o_proj",
            Some(full_idx),
            seq_len,
            || gemm_batch(&self.ctx, &attn.o_proj, attn_out, out),
        )?;
        // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
        qwen35_profile(
            &self.ctx,
            "qwen/full/allreduce",
            Some(full_idx),
            seq_len,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// Paged full attention over `meta`'s ragged page table: qkv GEMM → prep →
    /// paged attention → sigmoid gate → o_proj → all-reduce. The dispatch
    /// `(batch, total_q, max_qlen)` makes 1×T, B×1 and the B×T middle one path.
    /// RoPE is baked into the cached K at write time, so a recall-restricted
    /// page subset attends exactly those pages (the dense-Qwen3 argument).
    ///
    /// `layer0_query` is the `--kv-recall` sink: on a multi-row prefill the
    /// post-RoPE layer-0 Q is read back for the next step's recall score
    /// (head-major `[num_q_heads * head_dim]`, matching `recompute_recall_plan`).
    #[allow(clippy::too_many_arguments)]
    fn full_attention_paged(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        full_idx: usize,
        pool: &PagedKVPool,
        meta: &crate::loader::PageMeta,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
        layer0_query: Option<&mut Vec<f32>>,
    ) -> Result<()> {
        let c = &self.config;
        let rows = normed.seq_len;
        ensure!(
            meta.total_q == rows,
            "Qwen3.6 paged full attention: page table covers {} query tokens != {rows} rows",
            meta.total_q
        );
        // `max_qlen == 1` is the decode kernel's contract (one q row per batch
        // element); anything longer goes through the ragged prefill kernel.
        let decode = meta.seq_len == 1;
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let stride_page = pool.kv_dim * pool.page_size;

        let FullAttnScratch {
            qkv_fused,
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse,
            fa3_oaccum,
            fa3_lseaccum,
            fa3_semaphore,
            ..
        } = fw;
        let qkv_fused = qkv_fused.get(&self.ctx, q_proj_dim + 2 * kv_dim, rows)?;
        let q_full = q_full.get(&self.ctx, q_proj_dim, rows)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, rows)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, rows)?;
        qwen35_profile(
            &self.ctx,
            "qwen/full_paged/qkv_gemm",
            Some(full_idx),
            rows,
            || {
                gemm_batch(&self.ctx, &attn.qkv_proj, normed, qkv_fused)?;
                split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;
                Ok(())
            },
        )?;

        let q_prepped = q_prepped.get(&self.ctx, q_dim, rows)?;
        let attn_out = attn_heads.get(&self.ctx, q_dim, rows)?;

        let k_pool_ptr = pool.k_ptr(full_idx, &self.ctx.stream);
        let v_pool_ptr = pool.v_ptr(full_idx, &self.ctx.stream);

        // Prep: q/k RMSNorm + RoPE; write each row's K/V into its tail page(s).
        {
            #[cfg(test)]
            let prep_capture = if !decode && meta.batch == 1 {
                prep_probe::begin(
                    &self.ctx,
                    full_idx,
                    rows,
                    self.local_q_heads,
                    self.local_kv_heads,
                    c.head_dim,
                    c.rotary_dim,
                    c.rms_norm_eps,
                    &q_full.data,
                    &k_batch.data,
                    &attn.q_norm,
                    &attn.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    &meta.start_positions,
                )?
            } else {
                None
            };
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (positions_ptr, _gp) = meta.positions.device_ptr(&self.ctx.stream);
            let (kv_indices_ptr, _gi) = meta.kv_indices.device_ptr(&self.ctx.stream);
            let (kv_indptr_ptr, _gpi) = meta.kv_indptr.device_ptr(&self.ctx.stream);
            let (last_page_len_ptr, _gl) = meta.kv_last_page_len.device_ptr(&self.ctx.stream);
            let (start_pos_ptr, _gs) = meta.start_positions.device_ptr(&self.ctx.stream);
            {
                let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                qwen35_profile(
                    &self.ctx,
                    "qwen/full_paged/prep",
                    Some(full_idx),
                    rows,
                    || {
                        // SAFETY: all buffers valid on ctx.stream; pool tail page
                        // allocated; per-row offsets come from the meta's own
                        // prefix sums, so each launch stays inside its row.
                        unsafe {
                            if decode {
                                ffi::decode_prep_paged_hd256_cuda(
                                    qf_ptr as *const ffi::Half,
                                    qp_ptr as *mut ffi::Half,
                                    k_ptr as *const ffi::Half,
                                    v_ptr as *const ffi::Half,
                                    qn_ptr as *const ffi::Half,
                                    kn_ptr as *const ffi::Half,
                                    cos_ptr as *const ffi::Half,
                                    sin_ptr as *const ffi::Half,
                                    positions_ptr as *const i32,
                                    k_pool_ptr as *mut ffi::Half,
                                    v_pool_ptr as *mut ffi::Half,
                                    kv_indices_ptr as *const i32,
                                    kv_indptr_ptr as *const i32,
                                    last_page_len_ptr as *const i32,
                                    self.local_q_heads as i32,
                                    self.local_kv_heads as i32,
                                    pool.page_size as i32,
                                    stride_page as i32,
                                    meta.batch as i32,
                                    c.rotary_dim as i32,
                                    c.rms_norm_eps,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            } else {
                                // The prep reads ONE scalar start_pos off a
                                // table based at element 0 — launch per row.
                                let elem = std::mem::size_of::<ffi::Half>() as u64;
                                for b in 0..meta.batch {
                                    let (col, pages) = (meta.q_offsets[b], meta.page_offsets[b]);
                                    let len = meta.q_offsets[b + 1] - col;
                                    ffi::prefill_attention_paged_prep_hd256_cuda(
                                        (qf_ptr + (col * q_proj_dim) as u64 * elem)
                                            as *const ffi::Half,
                                        (qp_ptr + (col * q_dim) as u64 * elem) as *mut ffi::Half,
                                        (k_ptr + (col * kv_dim) as u64 * elem) as *const ffi::Half,
                                        (v_ptr + (col * kv_dim) as u64 * elem) as *const ffi::Half,
                                        qn_ptr as *const ffi::Half,
                                        kn_ptr as *const ffi::Half,
                                        cos_ptr as *const ffi::Half,
                                        sin_ptr as *const ffi::Half,
                                        (kv_indices_ptr + (pages * 4) as u64) as *const i32,
                                        pool.page_size as i32,
                                        k_pool_ptr as *mut ffi::Half,
                                        v_pool_ptr as *mut ffi::Half,
                                        self.local_q_heads as i32,
                                        self.local_kv_heads as i32,
                                        len as i32,
                                        (start_pos_ptr + (b * 4) as u64) as *const i32,
                                        c.rotary_dim as i32,
                                        c.rms_norm_eps,
                                        self.ctx.stream.cu_stream(),
                                    )
                                    .result()?;
                                }
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            #[cfg(test)]
            prep_probe::finish(&self.ctx, prep_capture, q_prepped)?;
        }

        // For quantized pools: BF16 work buffer → quantized data buffer.
        // The prep kernel above wrote the new tokens BF16 into `k_work` / `v_work`
        // (= `k_ptr` / `v_ptr` for FP8/INT8 pools). Quantize them into `k_data[layer]`
        // so the FP8 attention kernel can read the complete token history (prefix from
        // prior steps + new tokens just written) from one contiguous FP8 pool.
        if pool.format != KVFormat::BF16 {
            let new_rows = meta.new_token_rows.as_ref().ok_or_else(|| {
                anyhow!(
                    "Qwen35 full-attn FP8/INT8 pool missing new_token_rows in PageMeta \
                     (format={:?})",
                    pool.format
                )
            })?;
            let kv_dim = self.local_full_attn_kv_dim();
            match pool.format {
                KVFormat::FP8E4M3 => {
                    kv_quant::quantize_paged_kv_fp8(
                        &self.ctx,
                        pool.k_ptr(full_idx, &self.ctx.stream),
                        pool.k_data_ptr(full_idx, &self.ctx.stream),
                        pool.k_scales_ptr(full_idx, &self.ctx.stream),
                        new_rows,
                        self.local_kv_heads,
                        c.head_dim,
                        kv_dim,
                        rows,
                    )?;
                    kv_quant::quantize_paged_kv_fp8(
                        &self.ctx,
                        pool.v_ptr(full_idx, &self.ctx.stream),
                        pool.v_data_ptr(full_idx, &self.ctx.stream),
                        pool.v_scales_ptr(full_idx, &self.ctx.stream),
                        new_rows,
                        self.local_kv_heads,
                        c.head_dim,
                        kv_dim,
                        rows,
                    )?;
                }
                other => anyhow::bail!(
                    "Qwen35 full-attn paged: unsupported pool format {other:?} \
                     (only BF16 and FP8E4M3 are wired)"
                ),
            }
        }

        // Paged attention over the recall page table (RoPE pre-baked).
        {
            #[cfg(test)]
            let attn_capture = if !decode && meta.batch == 1 {
                attn_probe::begin(
                    &self.ctx,
                    full_idx,
                    rows,
                    self.local_q_heads,
                    self.local_kv_heads,
                    c.head_dim,
                    c.rotary_dim,
                    &q_prepped.data,
                    &k_batch.data,
                    &v_batch.data,
                    &attn.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    c.rms_norm_eps,
                    &meta.start_positions,
                )?
            } else {
                None
            };
            let (bsz, total_q, max_q) =
                (meta.batch as i32, meta.total_q as i32, meta.seq_len as i32);
            let (q_indptr_ptr, _g1) = meta.q_indptr.device_ptr(&self.ctx.stream);
            let (kv_indptr_ptr, _g2) = meta.kv_indptr.device_ptr(&self.ctx.stream);
            let (kv_indices_ptr, _g3) = meta.kv_indices.device_ptr(&self.ctx.stream);
            let (last_page_len_ptr, _g4) = meta.kv_last_page_len.device_ptr(&self.ctx.stream);
            let phase = if decode {
                ffi::AttnPhase::Decode
            } else {
                ffi::AttnPhase::Prefill
            };
            {
                let (qp_ptr, _g0) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                let (ao_ptr, _g5) = attn_out.data.device_ptr_mut(&self.ctx.stream);
                qwen35_profile(
                    &self.ctx,
                    "qwen/full_paged/attention",
                    Some(full_idx),
                    rows,
                    || {
                        // sm_90 short queries — decode (1 row) and spec verify
                        // (block+1): FA3 paged split-KV + PackGQA. The TileLang
                        // kernel it replaces pads BLOCK_M=64 around the real rows
                        // and gives one CTA per query head, 6× the KV traffic.
                        // batch==1 prefill chunks also route here: the 2026-07-28
                        // kill (c=8 TTFT 12.07→18.23 s) was the one-launch-per-
                        // request cost on ragged batches; a single request is a
                        // single launch either way. Ragged multi-request prefill
                        // and non-Hopper keep TileLang.
                        if (meta.seq_len <= FA3_MAX_QLEN || meta.batch == 1)
                            && pool.format == KVFormat::BF16
                            && c.head_dim == 256
                            && qwen35_fa3_enabled(&self.ctx)
                        {
                            // ONE launch for the whole batch: q/o are packed
                            // [total_q, h, d] behind `q_indptr`, and each row's KV
                            // extent comes from `kv_lens_dev` against the
                            // rectangular page table. `seqused_k` does not drop the
                            // K/V batch strides (only `cu_seqlens_k` does,
                            // flash_api.cpp:105-108), which is what lets a paged
                            // batch share a launch — per row it was 16 CTAs on 78
                            // SMs, serialized.
                            // Split-KV pays only when q is tiny vs kv; a long
                            // prefill chunk saturates SMs on the q axis alone.
                            // Upper bound only: FA3 picks the live value itself
                            // (flash_prepare_scheduler.cu `num_splits_dynamic`),
                            // clamped by what we pass. pack_gqa leaves batch ×
                            // kv_heads tiles before splitting, so one tile per SM
                            // is where the ceiling stops costing anything — raising
                            // it further only grows the combine scratch, which
                            // measured +0.36% at batch 8. The historical 8 is the
                            // floor: it is the measured optimum from batch 4 up,
                            // and it bound the scheduler only at batch 1, where
                            // 4×8 = 32 tiles left 46 of 78 SMs idle.
                            let splits = if meta.seq_len <= FA3_MAX_QLEN {
                                match qwen35_fa3_decode_splits() {
                                    0 => self
                                        .ctx
                                        .sm_count()
                                        .div_ceil(meta.batch.max(1) * self.local_kv_heads.max(1))
                                        .max(FA3_DECODE_SPLITS_FLOOR)
                                        .clamp(2, 256),
                                    n => n,
                                }
                            } else {
                                1
                            };
                            let accum_rows = self.local_q_heads * meta.total_q;
                            let lse = fa3_lse.get(&self.ctx, accum_rows)?;
                            let oaccum =
                                fa3_oaccum.get(&self.ctx, splits * accum_rows * c.head_dim)?;
                            let lseaccum = fa3_lseaccum.get(&self.ctx, splits * accum_rows)?;
                            let meta_cap = meta.batch.div_ceil(4) * 4 * 4 + 1;
                            let sem = fa3_semaphore.get(&self.ctx, meta_cap)?;
                            let (lse_ptr, _f0) = lse.device_ptr_mut(&self.ctx.stream);
                            let (oaccum_ptr, _f1) = oaccum.device_ptr_mut(&self.ctx.stream);
                            let (lseaccum_ptr, _f2) = lseaccum.device_ptr_mut(&self.ctx.stream);
                            let (sem_ptr, _f3) = sem.device_ptr_mut(&self.ctx.stream);
                            let (kv_lens_ptr, _f4) = meta.kv_lens_dev.device_ptr(&self.ctx.stream);
                            let (rect_ptr, _f5) = meta.page_table_rect.device_ptr(&self.ctx.stream);
                            let head_dim = c.head_dim as i64;
                            let args = ffi::ArleFa3FwdHd256Args {
                                q: qp_ptr as *const ffi::Half,
                                k: k_pool_ptr as *const ffi::Half,
                                v: v_pool_ptr as *const ffi::Half,
                                o: ao_ptr as *mut ffi::Half,
                                softmax_lse: lse_ptr as *mut f32,
                                out_accum: oaccum_ptr as *mut f32,
                                softmax_lse_accum: lseaccum_ptr as *mut f32,
                                tile_count_semaphore: sem_ptr as *mut i32,
                                metadata_capacity: meta_cap as i32,
                                cu_seqlens_q: q_indptr_ptr as *const i32,
                                seqused_k: kv_lens_ptr as *const i32,
                                batch: meta.batch as i32,
                                total_q: meta.total_q as i32,
                                seqlen_q: meta.seq_len as i32,
                                seqlen_k: meta.seqlen_k_capture.unwrap_or_else(|| {
                                    meta.kv_lens.iter().copied().max().unwrap_or(0)
                                }) as i32,
                                num_heads: self.local_q_heads as i32,
                                num_heads_k: self.local_kv_heads as i32,
                                head_dim: c.head_dim as i32,
                                q_row_stride: (self.local_q_heads * c.head_dim) as i64,
                                // HND pool [page, h_k, page_size, d]: tokens are
                                // contiguous inside a (page, head), heads stride
                                // by a page's worth, pages by the whole page.
                                k_row_stride: head_dim,
                                v_row_stride: head_dim,
                                o_row_stride: (self.local_q_heads * c.head_dim) as i64,
                                q_head_stride: head_dim,
                                k_head_stride: pool.page_size as i64 * head_dim,
                                v_head_stride: pool.page_size as i64 * head_dim,
                                o_head_stride: head_dim,
                                softmax_scale: sm_scale,
                                // Bottom-right aligned; the shim demotes to
                                // non-causal at qlen 1.
                                is_causal: 1,
                                num_splits: splits as i32,
                                page_table: rect_ptr as *const i32,
                                page_table_batch_stride: meta.page_table_stride as i64,
                                page_size: pool.page_size as i32,
                                num_pages: pool.max_total_pages as i32,
                                k_page_stride: stride_page as i64,
                                v_page_stride: stride_page as i64,
                            };
                            // SAFETY: q/o are the live prepped/out buffers; k/v are
                            // the layer's pool base; the page table is the meta's
                            // rectangular mirror, `batch * page_table_stride` long.
                            unsafe {
                                ffi::arle_fa3_fwd_hd256_bf16_cuda(
                                    &args,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            }
                            return Ok(());
                        }
                        // A persistent-decode meta means a graph capture may be
                        // active — the TileLang lane below bakes `num_pages`
                        // as a host arg and would replay stale, so refuse (the
                        // capture error downgrades the lane to eager).
                        ensure!(
                            meta.seqlen_k_capture.is_none(),
                            "paged decode graph capture requires the FA3 BF16 lane"
                        );
                        match pool.format {
                            KVFormat::BF16 => {
                                // SAFETY: kernel signature from paged_attn_v1 ABI (18-arg BF16).
                                let kernel = ffi::resolve_paged_attn_v1(
                                    c.head_dim as u32,
                                    self.local_q_heads as u32,
                                    self.local_kv_heads as u32,
                                    phase,
                                )
                                .ok_or_else(|| {
                                    anyhow!(
                                        "no HD256 paged {} kernel for q{}_kv{}",
                                        if decode { "decode" } else { "prefill" },
                                        self.local_q_heads,
                                        self.local_kv_heads
                                    )
                                })?;
                                // SAFETY: ptrs from live device allocations sized to the dims passed.
                                unsafe {
                                    kernel(
                                        qp_ptr as *mut ffi::Half,
                                        q_indptr_ptr as *const i32,
                                        k_pool_ptr as *mut ffi::Half,
                                        v_pool_ptr as *mut ffi::Half,
                                        kv_indptr_ptr as *const i32,
                                        kv_indices_ptr as *const i32,
                                        last_page_len_ptr as *const i32,
                                        ao_ptr as *mut ffi::Half,
                                        bsz,
                                        total_q,
                                        max_q,
                                        pool.max_total_pages as i32,
                                        meta.num_pages as i32,
                                        self.local_q_heads as i32,
                                        self.local_kv_heads as i32,
                                        pool.page_size as i32,
                                        sm_scale,
                                        self.ctx.stream.cu_stream(),
                                    )
                                    .result()?;
                                }
                            }
                            KVFormat::FP8E4M3 => {
                                // SAFETY: kernel signature from paged_attn_fp8_v1 ABI (20-arg).
                                // k/v data buffers are FP8; scales are per-token per-kv-head f32.
                                let kernel = ffi::resolve_paged_attn_fp8_v1(
                                    c.head_dim as u32,
                                    self.local_q_heads as u32,
                                    self.local_kv_heads as u32,
                                    phase,
                                )
                                .ok_or_else(|| {
                                    anyhow!(
                                        "no HD256 FP8 paged {} kernel for q{}_kv{}",
                                        if decode { "decode" } else { "prefill" },
                                        self.local_q_heads,
                                        self.local_kv_heads
                                    )
                                })?;
                                let k_data = pool.k_data_ptr(full_idx, &self.ctx.stream);
                                let v_data = pool.v_data_ptr(full_idx, &self.ctx.stream);
                                let k_scales = pool.k_scales_ptr(full_idx, &self.ctx.stream);
                                let v_scales = pool.v_scales_ptr(full_idx, &self.ctx.stream);
                                // SAFETY: ptrs from live device allocations sized to the dims passed.
                                unsafe {
                                    kernel(
                                        qp_ptr as *mut ffi::Half,
                                        q_indptr_ptr as *const i32,
                                        k_data as *const u8,
                                        v_data as *const u8,
                                        k_scales as *const f32,
                                        v_scales as *const f32,
                                        kv_indptr_ptr as *const i32,
                                        kv_indices_ptr as *const i32,
                                        last_page_len_ptr as *const i32,
                                        ao_ptr as *mut ffi::Half,
                                        bsz,
                                        total_q,
                                        max_q,
                                        pool.max_total_pages as i32,
                                        meta.num_pages as i32,
                                        self.local_q_heads as i32,
                                        self.local_kv_heads as i32,
                                        pool.page_size as i32,
                                        sm_scale,
                                        self.ctx.stream.cu_stream(),
                                    )
                                    .result()?;
                                }
                            }
                            other => anyhow::bail!(
                                "Qwen35 full-attn paged attention: unsupported pool format {other:?}"
                            ),
                        }
                        Ok(())
                    },
                )?;
            }
            #[cfg(test)]
            attn_probe::finish(&self.ctx, attn_capture, attn_out)?;
        }

        // Per-head sigmoid gate from q_full's gate half.
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_full/attn_out valid on ctx.stream; gate iterates
            // rows * num_q_heads.
            qwen35_profile(
                &self.ctx,
                "qwen/full_paged/gate",
                Some(full_idx),
                rows,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        ffi::attention_gate_paged_hd256_cuda(
                            qf_ptr as *const ffi::Half,
                            o_ptr as *mut ffi::Half,
                            self.local_q_heads as i32,
                            rows as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }

        // Layer-0 PREFILL: read back the post-RoPE prepped Q for the recall score
        // (head-major `[num_q_heads * head_dim]`). Under the write-through model the
        // whole recall cycle (score → evict → prefetch) runs ONCE per prefill, not
        // per decode step, so only the multi-token prefill needs this signal — the
        // D2H stays off every other paged forward. `q_prepped` is token-major
        // `[rows, q_dim]`; the recall query is the mean of the last `m` prompt
        // tokens' queries (R3 — "what am I about to generate").
        if let Some(dst) = layer0_query
            && full_idx == 0
            && rows > 1
        {
            let host: Vec<bf16> = self
                .ctx
                .stream
                .clone_dtoh(&q_prepped.data)
                .map_err(|e| anyhow!("recall layer0 q dtoh: {e}"))?;
            const RECALL_PREFILL_Q_TOKENS: usize = 16; // R3 default `m`.
            let m = RECALL_PREFILL_Q_TOKENS.min(rows);
            let mut q = vec![0.0_f32; q_dim];
            for t in (rows - m)..rows {
                let base = t * q_dim;
                for (d, slot) in q.iter_mut().enumerate() {
                    *slot += host[base + d].to_f32();
                }
            }
            let inv = 1.0_f32 / m as f32;
            for v in &mut q {
                *v *= inv;
            }
            *dst = q;
        }

        qwen35_profile(
            &self.ctx,
            "qwen/full_paged/o_proj",
            Some(full_idx),
            rows,
            || gemm_batch(&self.ctx, &attn.o_proj, attn_out, out),
        )?;
        // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
        qwen35_profile(
            &self.ctx,
            "qwen/full_paged/allreduce",
            Some(full_idx),
            rows,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// Gated-delta-rule linear attention into `out` (`[hidden, rows]`, beta=0
    /// out-proj GEMM): in-proj → depthwise conv1d → RECURRENT gated-delta
    /// (advances the per-slot state in place) → gated output RMSNorm →
    /// out-proj. The conv ring + recurrent state carry across prefill/decode.
    ///
    /// `rows = normed.seq_len` is the FLAT column count. Every weight-heavy
    /// step runs once over all of them; only [`LinearCore`] is per-slot.
    fn linear_attention(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        core: LinearCore<'_, '_>,
        linear_idx: usize,
        lw: &mut LinearAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let rows = normed.seq_len;
        // LOCAL per-rank widths (= global config on a single GPU): the fused
        // [q|k|v] shard, conv channels, recurrent state, and kernel launches all
        // follow this rank's linear k/v head shard. b/a widths come off the
        // sharded projection rows directly (`[local_Vh, hidden]`).
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let b_dim = attn.in_proj_ba.rows / 2;
        let a_dim = b_dim;

        let LinearAttnScratch {
            capture_copy,
            qkvz,
            qkv,
            z,
            ba,
            b_proj,
            a_proj,
            qkv_conv,
            gdr_out,
            normed_out,
            fq_q,
            fq_k,
            fq_v,
            fq_a,
            fq_g,
            fq_g_cumsum,
            fq_beta,
        } = lw;
        let qkvz = qkvz.get(&self.ctx, qkv_dim + z_dim, rows)?;
        let qkv = qkv.get(&self.ctx, qkv_dim, rows)?;
        let z = z.get(&self.ctx, z_dim, rows)?;
        let ba = ba.get(&self.ctx, b_dim + a_dim, rows)?;
        let b_proj = b_proj.get(&self.ctx, b_dim, rows)?;
        let a_proj = a_proj.get(&self.ctx, a_dim, rows)?;
        qwen35_profile(
            &self.ctx,
            "qwen/linear/in_proj",
            Some(linear_idx),
            rows,
            || {
                gemm_batch(&self.ctx, &attn.in_proj_qkvz, normed, qkvz)?;
                split2(&self.ctx, qkvz, qkv, z)?;
                gemm_batch(&self.ctx, &attn.in_proj_ba, normed, ba)?;
                split2(&self.ctx, ba, b_proj, a_proj)?;
                Ok(())
            },
        )?;

        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, rows)?;
        let gdr_out = gdr_out.get(&self.ctx, z_dim, rows)?;
        match core {
            LinearCore::Rows(rs) => {
                let total: usize = rs.iter().map(|r| r.len).sum();
                ensure!(
                    total == rows,
                    "linear rows total {total} != {rows} staged columns"
                );
                // Each row's columns land at ITS capture offset 0, so a
                // partial-accept replay re-runs only that slot's prefix. Three
                // launches for the whole batch, not three per row.
                if rs.iter().any(|r| r.capture.is_some()) {
                    let (mut dst, mut src, mut sz) = (Vec::new(), Vec::new(), Vec::new());
                    let mut off = 0usize;
                    for r in rs.iter_mut() {
                        let len = r.len;
                        let at = off;
                        off += len;
                        let Some(cap) = r.capture.as_deref_mut() else {
                            continue;
                        };
                        ensure!(
                            linear_idx < cap.qkv.len() && len <= cap.rows,
                            "spec capture is {} layers x {} rows, cannot hold layer \
                             {linear_idx} x {len} rows",
                            cap.qkv.len(),
                            cap.rows
                        );
                        for (s_ptr, w, d) in [
                            (&qkv.data, qkv_dim, &mut cap.qkv[linear_idx]),
                            (&b_proj.data, b_dim, &mut cap.b_proj[linear_idx]),
                            (&a_proj.data, a_dim, &mut cap.a_proj[linear_idx]),
                        ] {
                            let elem = std::mem::size_of::<bf16>();
                            dst.push(d.data.device_ptr_mut(&self.ctx.stream).0);
                            src.push(s_ptr.device_ptr(&self.ctx.stream).0 + (at * w * elem) as u64);
                            sz.push(len * w * elem);
                        }
                    }
                    self.batched_copy(capture_copy, &dst, &src, &sz)?;
                }
                let mut off = 0usize;
                for r in rs.iter_mut() {
                    self.advance_linear_conv_gdr(
                        attn,
                        &qkv.data.slice(off * qkv_dim..(off + r.len) * qkv_dim),
                        &b_proj.data.slice(off * b_dim..(off + r.len) * b_dim),
                        &a_proj.data.slice(off * a_dim..(off + r.len) * a_dim),
                        r.slot,
                        linear_idx,
                        r.len,
                        &mut qkv_conv
                            .data
                            .slice_mut(off * qkv_dim..(off + r.len) * qkv_dim),
                        &mut gdr_out.data.slice_mut(off * z_dim..(off + r.len) * z_dim),
                        fq_q,
                        fq_k,
                        fq_v,
                        fq_a,
                        fq_g,
                        fq_g_cumsum,
                        fq_beta,
                    )?;
                    off += r.len;
                }
            }
            LinearCore::Tables { conv, gdr } => {
                let (x_ptr, _g0) = qkv.data.device_ptr(&self.ctx.stream);
                let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
                let (b_ptr, _g2) = b_proj.data.device_ptr(&self.ctx.stream);
                let (a_ptr, _g3) = a_proj.data.device_ptr(&self.ctx.stream);
                let (dt_ptr, _g4) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                let (alog_ptr, _g5) = attn.a_log.device_ptr(&self.ctx.stream);
                let (conv_tbl, _g6) = conv.device_ptr(&self.ctx.stream);
                let (gdr_tbl, _g7) = gdr.device_ptr(&self.ctx.stream);
                let (cv_ptr, _g8) = qkv_conv.data.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g9) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
                qwen35_profile(
                    &self.ctx,
                    "qwen/linear/conv1d",
                    Some(linear_idx),
                    rows,
                    || {
                        // SAFETY: x/weight/out are live `[B, C]`/`[C*K]` buffers on
                        // ctx.stream; the table's first B entries point at live
                        // `[C, K-1]` conv rings.
                        unsafe {
                            ffi::conv1d_decode_batch_cuda(
                                x_ptr as *const ffi::Half,
                                w_ptr as *const ffi::Half,
                                conv_tbl as *mut *mut ffi::Half,
                                cv_ptr as *mut ffi::Half,
                                qkv_dim as i32,
                                c.linear_conv_kernel_dim as i32,
                                rows as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                        Ok(())
                    },
                )?;
                qwen35_profile(
                    &self.ctx,
                    "qwen/linear/gdr_recurrent",
                    Some(linear_idx),
                    rows,
                    || {
                        // SAFETY: all buffers live on ctx.stream; the table's first
                        // B entries point at live `[Vh, Kd, Vd]` f32 states.
                        unsafe {
                            ffi::gdr_decode_batch_cuda(
                                cv_ptr as *const ffi::Half,
                                b_ptr as *const ffi::Half,
                                a_ptr as *const ffi::Half,
                                dt_ptr as *const ffi::Half,
                                alog_ptr as *const f32,
                                gdr_tbl as *mut *mut f32,
                                o_ptr as *mut ffi::Half,
                                self.local_linear_k_heads as i32,
                                self.local_linear_v_heads as i32,
                                c.linear_key_head_dim as i32,
                                c.linear_value_head_dim as i32,
                                rows as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                        Ok(())
                    },
                )?;
            }
        }

        // ── gated output RMSNorm (per value head; gate = z). ──
        let normed_out = normed_out.get(&self.ctx, z_dim, rows)?;
        {
            let (x_ptr, _g0) = gdr_out.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.norm_weight.device_ptr(&self.ctx.stream);
            let (gate_ptr, _g2) = z.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = normed_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: gdr_out/norm/z/out valid on ctx.stream; per-head layout from config.
            // The kernel launches exactly `num_heads` blocks, each normalizing one
            // flat `[val_dim]` slice at `blockIdx.x * val_dim` — gdr_out/z are
            // `[rows, Vh*Vd]` row-major, so the grid must cover all
            // rows*Vh (token, head) slices, not just token 0. `weight[tid]`
            // is a per-[Vd] broadcast (no blockIdx dependence), so the
            // extension is exact (the monolith's `rms_norm_gated_batch_into`
            // passed `seq_len * num_heads` identically).
            qwen35_profile(
                &self.ctx,
                "qwen/linear/norm",
                Some(linear_idx),
                rows,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        ffi::rms_norm_gated_cuda(
                            x_ptr as *const ffi::Half,
                            w_ptr as *const f32,
                            gate_ptr as *const ffi::Half,
                            o_ptr as *mut ffi::Half,
                            (self.local_linear_v_heads * rows) as i32,
                            c.linear_value_head_dim as i32,
                            c.rms_norm_eps,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }

        qwen35_profile(
            &self.ctx,
            "qwen/linear/out_proj",
            Some(linear_idx),
            rows,
            || gemm_batch(&self.ctx, &attn.out_proj, normed_out, out),
        )?;
        // Row-parallel out_proj: ONE all-reduce over the exact `[hidden, rows]`
        // buffer (no-op single-GPU).
        qwen35_profile(
            &self.ctx,
            "qwen/linear/allreduce",
            Some(linear_idx),
            rows,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// Conv1d (advances `slot.conv_states[linear_idx]`) + gated-delta rule
    /// (advances `slot.gdr_states[linear_idx]`) for one linear layer over
    /// `seq_len` rows. The ONLY persistent-state-mutating core of
    /// [`Self::linear_attention`], factored out so the partial-accept replay
    /// re-runs the IDENTICAL kernel dispatch (same conv1d + same
    /// recurrent/chunked GDR branch, same inputs) — guaranteeing the conv and
    /// recurrent state advance is byte-identical between the trunk forward and
    /// the replay.
    ///
    /// `qkv_in` is the post-in_proj fused `[q|k|v]` PRE-conv1d (`qkv_dim` wide,
    /// token-major); conv1d reads it and writes the SEPARATE `qkv_conv`, which
    /// the GDR then consumes. `b_in`/`a_in` are the `in_proj_b`/`in_proj_a`
    /// gate projections. Every view spans EXACTLY this slot's `seq_len` rows —
    /// in a ragged batch that is a column-range slice of the shared scratch, so
    /// each slot's state sees only its own tokens.
    /// `gdr_out` is written but discarded by the replay (only the state
    /// side-effect matters); the trunk path norms it.
    #[allow(clippy::too_many_arguments)]
    fn advance_linear_conv_gdr(
        &self,
        attn: &LinearAttn,
        qkv_in: &CudaView<'_, bf16>,
        b_in: &CudaView<'_, bf16>,
        a_in: &CudaView<'_, bf16>,
        slot: &mut Qwen35SlotState,
        linear_idx: usize,
        seq_len: usize,
        qkv_conv: &mut CudaViewMut<'_, bf16>,
        gdr_out: &mut CudaViewMut<'_, bf16>,
        fq_q: &mut HiddenSlot,
        fq_k: &mut HiddenSlot,
        fq_v: &mut HiddenSlot,
        fq_a: &mut HiddenSlot,
        fq_g: &mut SliceSlot<f32>,
        fq_g_cumsum: &mut SliceSlot<f32>,
        fq_beta: &mut SliceSlot<f32>,
    ) -> Result<()> {
        let c = &self.config;
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();

        // ── conv1d (advances the per-slot conv ring). ──
        let conv_state = &mut slot.conv_states[linear_idx];
        ensure!(
            conv_state.len == qkv_dim * (c.linear_conv_kernel_dim - 1),
            "Qwen3.5 conv state len {} != qkv_dim*(kernel-1) {}",
            conv_state.len,
            qkv_dim * (c.linear_conv_kernel_dim - 1)
        );
        {
            #[cfg(test)]
            let conv_capture = conv_probe::begin(
                &self.ctx,
                linear_idx,
                seq_len,
                qkv_dim,
                c.linear_conv_kernel_dim,
                qkv_in,
                &attn.conv1d_weight,
                conv_state,
            )?;
            {
                let (x_ptr, _g0) = qkv_in.device_ptr(&self.ctx.stream);
                let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
                let (s_ptr, _g2) = conv_state.data.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g3) = qkv_conv.device_ptr_mut(&self.ctx.stream);
                // SAFETY: qkv/weight/state/out valid on ctx.stream; weight len checked
                // by the kernel against num_channels*kernel.
                qwen35_profile(
                    &self.ctx,
                    "qwen/linear/conv1d",
                    Some(linear_idx),
                    seq_len,
                    || {
                        // SAFETY: ptrs from live device allocations sized to the dims passed.
                        unsafe {
                            ffi::conv1d_prefill_cuda(
                                x_ptr as *const ffi::Half,
                                w_ptr as *const ffi::Half,
                                s_ptr as *mut ffi::Half,
                                o_ptr as *mut ffi::Half,
                                qkv_dim as i32,
                                seq_len as i32,
                                c.linear_conv_kernel_dim as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                        Ok(())
                    },
                )?;
            }
            #[cfg(test)]
            conv_probe::finish(&self.ctx, conv_capture, qkv_conv, conv_state)?;
        }

        // ── gated-delta rule. Decode (seq_len==1) is always the recurrent
        //    kernel. Prefill chunks default to the recurrent kernel; the
        //    FlashQLA chunked path (--qwen35-gdr-chunked) replaces the
        //    serial token scan with chunk-parallel TileLang kernels, one AOT
        //    instantiation per (Hg, H) geometry — unknown geometry falls back
        //    to recurrent. The legacy in-tree chunkwise TileLang path stays
        //    dead (sm_90 hang was in ITS kernels). ──
        let fq_fns: Option<(ffi::FqCumsumFn, ffi::FqKktFn, ffi::FqFwdFn)> =
            match (self.local_linear_k_heads, self.local_linear_v_heads) {
                (16, 32) => Some((
                    ffi::gdr_fq_cumsum_cuda as _,
                    ffi::gdr_fq_kkt_cuda as _,
                    ffi::gdr_fq_fwd_cuda as _,
                )),
                (16, 48) => Some((
                    ffi::gdr_fq_cumsum_h48_cuda as _,
                    ffi::gdr_fq_kkt_h48_cuda as _,
                    ffi::gdr_fq_fwd_h48_cuda as _,
                )),
                _ => None,
            };
        let use_fq_chunked = seq_len > 1
            && qwen35_gdr_chunked_enabled()
            && c.linear_key_head_dim == 128
            && c.linear_value_head_dim == 128
            && fq_fns.is_some_and(|(cumsum, _, _)| fq_kernels_available(&self.ctx, cumsum));
        if use_fq_chunked {
            // The AOT dispatch wrapper resolves SM + module via the DRIVER
            // context of the calling thread; the engine forward thread is not
            // guaranteed to have one bound (runtime-API kernels don't need
            // it), which made the fq path a per-thread lottery returning
            // NOT_SUPPORTED. Bind explicitly.
            self.ctx
                .ctx
                .bind_to_thread()
                .map_err(|e| anyhow!("bind CUDA context for chunked GDR failed: {e}"))?;
            let (fq_cumsum, fq_kkt, fq_fwd) = fq_fns.unwrap();
            let fq_parity = std::env::var_os("ARLE_FQ_PARITY").is_some();
            let mut fq_par_snap: Option<cudarc::driver::CudaSlice<f32>> = None;
            if fq_parity {
                let st = &slot.gdr_states[linear_idx];
                let mut snap = self.ctx.stream.alloc_zeros::<f32>(st.len())?;
                self.ctx.stream.memcpy_dtod(st, &mut snap)?;
                fq_par_snap = Some(snap);
            }
            let hg_dim = self.local_linear_k_heads * c.linear_key_head_dim;
            let fq_q = fq_q.get(&self.ctx, hg_dim, seq_len)?;
            let fq_k = fq_k.get(&self.ctx, hg_dim, seq_len)?;
            let fq_v = fq_v.get(&self.ctx, z_dim, seq_len)?;
            let fq_a = fq_a.get(&self.ctx, self.local_linear_v_heads * 64, seq_len)?;
            let g_len = self.local_linear_v_heads * seq_len;
            let fq_g = fq_g.get(&self.ctx, g_len)?;
            let fq_g_cumsum = fq_g_cumsum.get(&self.ctx, g_len)?;
            let fq_beta = fq_beta.get(&self.ctx, g_len)?;
            let gdr_state = &mut slot.gdr_states[linear_idx];

            let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let (q_ptr, _g5) = fq_q.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _g6) = fq_k.data.device_ptr_mut(&self.ctx.stream);
            let (v_ptr, _g7) = fq_v.data.device_ptr_mut(&self.ctx.stream);
            let (a_inv_ptr, _g8) = fq_a.data.device_ptr_mut(&self.ctx.stream);
            let (g_ptr, _g9) = fq_g.device_ptr_mut(&self.ctx.stream);
            let (gc_ptr, _g10) = fq_g_cumsum.device_ptr_mut(&self.ctx.stream);
            let (beta_ptr, _g11) = fq_beta.device_ptr_mut(&self.ctx.stream);
            let (s_ptr, _g12) = gdr_state.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g13) = gdr_out.device_ptr_mut(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream, shapes per the slot
            // `.get` calls above. The slot state pointer is passed as BOTH
            // h0 and ht (in-place chunk chaining): each fwd CTA reads its h0
            // slice fully before writing the same ht slice.
            qwen35_profile(
                &self.ctx,
                "qwen/linear/gdr_fq",
                Some(linear_idx),
                seq_len,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        ffi::gdr_fq_prep_cuda(
                            qkv_ptr as *const ffi::Half,
                            b_ptr as *const ffi::Half,
                            a_ptr as *const ffi::Half,
                            dt_ptr as *const ffi::Half,
                            alog_ptr as *const f32,
                            q_ptr as *mut ffi::Half,
                            k_ptr as *mut ffi::Half,
                            v_ptr as *mut ffi::Half,
                            g_ptr as *mut f32,
                            beta_ptr as *mut f32,
                            self.local_linear_k_heads as i32,
                            self.local_linear_v_heads as i32,
                            c.linear_key_head_dim as i32,
                            c.linear_value_head_dim as i32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        fq_cumsum(
                            g_ptr as *const f32,
                            gc_ptr as *mut f32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        fq_kkt(
                            k_ptr as *const ffi::Half,
                            beta_ptr as *const f32,
                            a_inv_ptr as *mut ffi::Half,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        fq_fwd(
                            q_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            v_ptr as *const ffi::Half,
                            a_inv_ptr as *const ffi::Half,
                            gc_ptr as *const f32,
                            beta_ptr as *const f32,
                            s_ptr as *const f32,
                            o_ptr as *mut ffi::Half,
                            s_ptr as *mut f32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
            if let Some(mut snap) = fq_par_snap {
                drop(_g12);
                drop(_g13);
                let o_ref = DeviceVec::zeros(&self.ctx, seq_len * z_dim)?;
                {
                    let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
                    let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
                    let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
                    let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                    let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
                    let (s_ptr, _g5) = snap.device_ptr_mut(&self.ctx.stream);
                    let (o_ptr, _g6) = o_ref.data.device_ptr(&self.ctx.stream);
                    // SAFETY: same buffers/dims as the fq calls above.
                    unsafe {
                        ffi::gated_delta_rule_prefill_recurrent_cuda(
                            qkv_ptr as *const ffi::Half,
                            b_ptr as *const ffi::Half,
                            a_ptr as *const ffi::Half,
                            dt_ptr as *const ffi::Half,
                            alog_ptr as *const f32,
                            s_ptr as *mut f32,
                            o_ptr as *mut ffi::Half,
                            self.local_linear_k_heads as i32,
                            self.local_linear_v_heads as i32,
                            c.linear_key_head_dim as i32,
                            c.linear_value_head_dim as i32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
                let a = self.ctx.stream.clone_dtoh(&*gdr_state)?;
                let b = self.ctx.stream.clone_dtoh(&snap)?;
                let (mut n2, mut d2) = (0f64, 0f64);
                for (x, y) in a.iter().zip(b.iter()) {
                    let d = (*x - *y) as f64;
                    n2 += d * d;
                    d2 += (*y as f64) * (*y as f64);
                }
                let oc = self.ctx.stream.clone_dtoh(&*gdr_out)?;
                let orf = self.ctx.stream.clone_dtoh(&o_ref.data)?;
                let (mut on2, mut od2) = (0f64, 0f64);
                let (mut worst, mut worst_i) = (0f64, 0usize);
                for (i, (x, y)) in oc.iter().zip(orf.iter()).enumerate() {
                    let d = (x.to_f32() - y.to_f32()) as f64;
                    on2 += d * d;
                    od2 += (y.to_f32() as f64) * (y.to_f32() as f64);
                    if d.abs() > worst {
                        worst = d.abs();
                        worst_i = i;
                    }
                }
                eprintln!(
                    "[fq-parity] layer={linear_idx} seq={seq_len} state_rel={:.3e} o_rel={:.3e} o_worst={:.3}@tok{}",
                    (n2 / (d2 + 1e-30)).sqrt(),
                    (on2 / (od2 + 1e-30)).sqrt(),
                    worst,
                    worst_i / z_dim
                );
            }
        }
        if !use_fq_chunked {
            let gdr_state = &mut slot.gdr_states[linear_idx];
            #[cfg(test)]
            let gdr_capture = gdr_probe::begin(
                &self.ctx,
                linear_idx,
                seq_len,
                self.local_linear_k_heads,
                self.local_linear_v_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
                qkv_conv,
                b_in,
                a_in,
                &attn.dt_bias,
                &attn.a_log,
                gdr_state,
            )?;
            {
                let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
                let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
                let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
                let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
                let (s_ptr, _g5) = gdr_state.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g6) = gdr_out.device_ptr_mut(&self.ctx.stream);
                qwen35_profile(
                    &self.ctx,
                    "qwen/linear/gdr_recurrent",
                    Some(linear_idx),
                    seq_len,
                    || {
                        // SAFETY: all buffers valid on ctx.stream; head dims from config.
                        unsafe {
                            if seq_len == 1 {
                                ffi::gated_delta_rule_decode_cuda(
                                    qkv_ptr as *const ffi::Half,
                                    b_ptr as *const ffi::Half,
                                    a_ptr as *const ffi::Half,
                                    dt_ptr as *const ffi::Half,
                                    alog_ptr as *const f32,
                                    s_ptr as *mut f32,
                                    o_ptr as *mut ffi::Half,
                                    self.local_linear_k_heads as i32,
                                    self.local_linear_v_heads as i32,
                                    c.linear_key_head_dim as i32,
                                    c.linear_value_head_dim as i32,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            } else {
                                ffi::gated_delta_rule_prefill_recurrent_cuda(
                                    qkv_ptr as *const ffi::Half,
                                    b_ptr as *const ffi::Half,
                                    a_ptr as *const ffi::Half,
                                    dt_ptr as *const ffi::Half,
                                    alog_ptr as *const f32,
                                    s_ptr as *mut f32,
                                    o_ptr as *mut ffi::Half,
                                    self.local_linear_k_heads as i32,
                                    self.local_linear_v_heads as i32,
                                    c.linear_key_head_dim as i32,
                                    c.linear_value_head_dim as i32,
                                    seq_len as i32,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            #[cfg(test)]
            gdr_probe::finish(&self.ctx, gdr_capture, gdr_out, gdr_state)?;
        }
        Ok(())
    }

    /// Partial-accept linear-only replay: advance ONLY the 48 gated-delta
    /// recurrent + conv states over the accepted prefix `[pending, d1..dk]`
    /// (`k+1` rows) from the verify capture, leaving them byte-identical to the
    /// old full-trunk `forward_hidden` replay at a tiny fraction of its cost.
    ///
    /// Precondition: the caller has just `restore_trunk`-ed the conv + gdr
    /// states to `S_{start_pos}` (the pre-verify snapshot), so re-running the
    /// first `k+1` recurrent steps reproduces the verify's `S_{start_pos+k+1}`.
    /// This re-uses the SAME [`Self::advance_linear_conv_gdr`] dispatch the
    /// verify forward used, fed the captured per-layer inputs — same conv1d +
    /// same recurrent/chunked GDR branch + same in-place accumulation, so the
    /// result is bit-for-bit the verify's first `k+1` steps. The
    /// recurrent-vs-chunked branch is re-selected from `seq_len = k+1`
    /// IDENTICALLY to how the old `forward_hidden` replay's `linear_attention`
    /// selected it (preserving any k==0 decode-vs-chunked path choice).
    ///
    /// Mutated device buffers (full enumeration):
    ///   - `slot.conv_states[li]` for every linear layer `li` — advanced k+1
    ///     steps from the restored S_start (conv1d ring shift, content-based).
    ///   - `slot.gdr_states[li]` for every linear layer `li` — advanced k+1
    ///     steps from the restored S_start (recurrent in-place).
    ///   - `ws.linear` scratch (`qkv_conv`/`gdr_out`/`fq_*`) — fully overwritten
    ///     per layer before read; transient, no persistent meaning.
    ///
    /// Explicitly NOT touched: the 16 full-attn KV caches (self-heal via the
    /// caller's `set_seq_len` position rewind), `slot.seq_len` (caller sets it),
    /// MLP/MoE/lm_head (no persistent state). Numerics-only; no H2D/D2H/sync.
    fn replay_linear_only(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        capture: &Qwen35LinearCapture,
        k: usize,
    ) -> Result<()> {
        let num_linear = slot.conv_states.len();
        ensure!(
            capture.qkv.len() == num_linear
                && capture.b_proj.len() == num_linear
                && capture.a_proj.len() == num_linear,
            "spec capture linear count {}/{}/{} != slot linear layers {num_linear}",
            capture.qkv.len(),
            capture.b_proj.len(),
            capture.a_proj.len()
        );
        let rows = k + 1;
        ensure!(
            rows <= capture.rows,
            "spec replay needs {rows} rows but capture holds {}",
            capture.rows
        );
        // Each slot's capture holds ITS OWN rows token-major from offset 0, so
        // the accepted prefix is the leading `rows = k+1` columns.
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let Qwen35Workspace { linear, .. } = ws;
        let LinearAttnScratch {
            qkv_conv,
            gdr_out,
            fq_q,
            fq_k,
            fq_v,
            fq_a,
            fq_g,
            fq_g_cumsum,
            fq_beta,
            ..
        } = linear;
        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, rows)?;
        let gdr_out = gdr_out.get(&self.ctx, z_dim, rows)?;
        let mut li = 0usize;
        for layer in &self.layers {
            if let Qwen35Attn::Linear(attn) = &layer.attn {
                let b_dim = attn.in_proj_ba.rows / 2;
                let a_dim = b_dim;
                self.advance_linear_conv_gdr(
                    attn,
                    &capture.qkv[li].data.slice(0..rows * qkv_dim),
                    &capture.b_proj[li].data.slice(0..rows * b_dim),
                    &capture.a_proj[li].data.slice(0..rows * a_dim),
                    slot,
                    li,
                    rows,
                    &mut qkv_conv.data.slice_mut(..),
                    &mut gdr_out.data.slice_mut(..),
                    fq_q,
                    fq_k,
                    fq_v,
                    fq_a,
                    fq_g,
                    fq_g_cumsum,
                    fq_beta,
                )?;
                li += 1;
            }
        }
        ensure!(
            li == num_linear,
            "spec replay advanced {li} linear layers != slot count {num_linear}"
        );
        Ok(())
    }

    /// `dst[i] <- src[i]` for `n` buffers in one launch. `bytes` is one size
    /// for all or one per buffer. A spec snapshot is 48 layers x B slots of
    /// ~3 MB, and each `memcpy_dtod` costs ~11 µs of host driver time against
    /// ~2 µs of bandwidth.
    pub(crate) fn batched_copy(
        &self,
        s: &mut Qwen35CopyScratch,
        dst: &[u64],
        src: &[u64],
        bytes: &[usize],
    ) -> Result<()> {
        ensure!(
            dst.len() == src.len(),
            "batched copy dst/src length mismatch"
        );
        ensure!(
            bytes.len() == 1 || bytes.len() == dst.len(),
            "batched copy {} sizes for {} buffers",
            bytes.len(),
            dst.len()
        );
        if dst.is_empty() || bytes.iter().all(|b| *b == 0) {
            return Ok(());
        }
        ensure!(
            bytes.iter().all(|b| b % 16 == 0),
            "batched copy sizes must be 16B multiples"
        );
        let ctx = &self.ctx;
        let n = dst.len();
        s.host.clear();
        s.host.extend_from_slice(dst);
        s.host.extend_from_slice(src);
        let tbl = s.ptrs.get(ctx, 2 * n)?;
        ctx.stream
            .memcpy_htod(&s.host, tbl)
            .map_err(|e| anyhow!("H2D batched copy tables: {e}"))?;
        let (base, _g) = tbl.device_ptr(&ctx.stream);
        let (len_ptr, max_words) = if bytes.len() == 1 {
            (0u64, 0usize)
        } else {
            s.hlen.clear();
            s.hlen.extend(bytes.iter().map(|b| (b / 16) as i32));
            let max = s.hlen.iter().copied().max().unwrap_or(0) as usize;
            let d = s.lens.get(ctx, n)?;
            ctx.stream
                .memcpy_htod(&s.hlen, d)
                .map_err(|e| anyhow!("H2D batched copy sizes: {e}"))?;
            (d.device_ptr(&ctx.stream).0, max)
        };
        // SAFETY: the table holds `n` dst then `n` src live addresses, each
        // buffer at least its `bytes` entry and cudaMalloc-aligned.
        unsafe {
            ffi::batched_copy_uniform_cuda(
                base as *const *mut std::ffi::c_void,
                (base + (n as u64) * 8) as *const *const std::ffi::c_void,
                len_ptr as *const i32,
                bytes[0],
                max_words,
                n as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }

    /// [`Self::replay_linear_only`] for a whole batch: one conv1d and one
    /// gated-delta launch per layer instead of two per slot per layer. Each
    /// slot keeps its own capture and state, reached through `tables`. The
    /// launches are sub-100 µs, so the win is their count.
    pub(crate) fn replay_linear_only_batched(
        &self,
        slots: &mut [&mut Qwen35SlotState],
        captures: &[&Qwen35LinearCapture],
        ks: &[usize],
        tables: &mut Qwen35ReplayTables,
        ws: &mut Qwen35Workspace,
    ) -> Result<()> {
        let b = slots.len();
        ensure!(
            b == captures.len() && b == ks.len(),
            "batched replay: {b} slots vs {} captures / {} ks",
            captures.len(),
            ks.len()
        );
        let num_linear = slots[0].conv_states.len();
        let max_len = ks.iter().map(|k| k + 1).max().unwrap_or(0);
        ensure!(max_len >= 1, "batched replay with no rows");
        for (s, cap) in captures.iter().enumerate() {
            ensure!(
                cap.qkv.len() == num_linear && ks[s] < cap.rows,
                "batched replay slot {s}: capture {} layers / {} rows cannot hold {} rows of \
                 {num_linear} layers",
                cap.qkv.len(),
                cap.rows,
                ks[s] + 1
            );
        }
        let ctx = &self.ctx;
        tables.stage(ctx, slots, captures, ks, num_linear)?;

        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let rows = b * max_len;
        let Qwen35Workspace { linear, .. } = ws;
        let qkv_conv = linear.qkv_conv.get(ctx, qkv_dim, rows)?;
        let gdr_out = linear.gdr_out.get(ctx, z_dim, rows)?;
        let (cv_ptr, _gc) = qkv_conv.data.device_ptr_mut(&ctx.stream);
        let (go_ptr, _gg) = gdr_out.data.device_ptr_mut(&ctx.stream);
        let stride = num_linear * b;
        let (tbl, _gt) = tables
            .ptrs
            .get(ctx, REPLAY_TABLES * stride)?
            .device_ptr(&ctx.stream);
        let lay = ReplayLayout {
            base: tbl,
            ..tables.layout
        };
        let (len_ptr, _gl) = tables.row_len.get(ctx, b)?.device_ptr(&ctx.stream);
        let c = &self.config;
        let mut li = 0usize;
        for layer in &self.layers {
            let Qwen35Attn::Linear(attn) = &layer.attn else {
                continue;
            };
            let (w_ptr, _g0) = attn.conv1d_weight.data.device_ptr(&ctx.stream);
            let (dt_ptr, _g1) = attn.dt_bias.data.device_ptr(&ctx.stream);
            let (alog_ptr, _g2) = attn.a_log.device_ptr(&ctx.stream);
            let qkv_tbl = lay.table(TBL_QKV, li);
            let b_tbl = lay.table(TBL_B, li);
            let a_tbl = lay.table(TBL_A, li);
            let conv_tbl = lay.table(TBL_CONV, li);
            let gdr_tbl = lay.table(TBL_GDR, li);
            // SAFETY: each table holds `b` pointers staged above; the shared
            // scratch is `[b * max_len, dim]`.
            unsafe {
                ffi::conv1d_prefill_varlen_cuda(
                    qkv_tbl as *const *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    conv_tbl as *const *mut ffi::Half,
                    len_ptr as *const i32,
                    cv_ptr as *mut ffi::Half,
                    qkv_dim as i32,
                    max_len as i32,
                    c.linear_conv_kernel_dim as i32,
                    b as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::gated_delta_rule_prefill_recurrent_varlen_cuda(
                    cv_ptr as *const ffi::Half,
                    b_tbl as *const *const ffi::Half,
                    a_tbl as *const *const ffi::Half,
                    dt_ptr as *const ffi::Half,
                    alog_ptr as *const f32,
                    gdr_tbl as *const *mut f32,
                    len_ptr as *const i32,
                    go_ptr as *mut ffi::Half,
                    self.local_linear_k_heads as i32,
                    self.local_linear_v_heads as i32,
                    c.linear_key_head_dim as i32,
                    c.linear_value_head_dim as i32,
                    max_len as i32,
                    b as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            li += 1;
        }
        ensure!(
            li == num_linear,
            "batched replay advanced {li} linear layers != slot count {num_linear}"
        );
        Ok(())
    }

    /// BATCHED DECODE (stage 1, contiguous KV): execute a whole multi-row
    /// pure-decode plan in ONE forward — B rows packed as `seq_len == B`.
    ///
    /// Token-parallel ops run batched exactly like a B-token prefill chunk
    /// (they already support `seq_len > 1`): embedding, every GEMM (q/k/v/o,
    /// qkv/z/b/a/out, router, experts, lm_head), `rms_norm_batched_offset`,
    /// residual adds, MoE (`moe_forward_into` with `num_tokens = B` → `R = 8B`
    /// routes — stays on the hand grouped kernels below
    /// `QWEN35_DEEPGEMM_MIN_ROUTES = 1024`), final norm + batched argmax.
    /// Per-row handling ONLY where state is per-slot:
    ///   - full attention (per-row loop over the existing single-row kernels
    ///     against each row's contiguous cache — DSv4 Step-A pattern, but with
    ///     column-offset pointers instead of copy-in/out:
    ///     [`Self::full_attention_batch_rows`]);
    ///   - conv1d + GDR (one batched kernel each via per-layer device pointer
    ///     tables: [`LinearCore::Tables`]);
    ///   - sampling (batched greedy argmax; non-greedy rows host-sample).
    ///
    /// FORMULA (the c=4 prediction this path is licensed against): at B=4 the
    /// token-parallel ops amortize ~4x (GEMM/MoE/norm launch count AND time
    /// per generated token /4), while per-row ops (full attention on 10
    /// layers x 3 kernels, sampling) stay linear in B → predicted aggregate
    /// ~2.3-3.2x single-stream tok/s at B=4 (vs ~1.0x today, because a
    /// rows>1 plan previously hard-errored the executor). License: pod
    /// c-sweep 1/2/4, aggregate tok/s + per-stream ITL + needle PASS at c=2.
    ///
    /// TP: safe under TP=2 by the same lockstep argument as single-row decode
    /// — every rank executes the identical plan (slot order fixed by the
    /// deterministic scheduler), the layer loop and the per-row attention
    /// loop have a fixed order, and every collective is an `all_reduce_sum`
    /// over an exact-shape `[hidden, B]` buffer (identical message length and
    /// call sequence on every rank; the per-row loop itself contains NO
    /// collectives). Complete all-reduce enumeration for one batched step:
    ///   1. full-attn o_proj partial   — `attn_out` `[hidden, B]` x num_full
    ///   2. linear-attn out_proj partial — `attn_out` `[hidden, B]` x num_linear
    ///   3. FFN (MoE routed+shared / dense down_proj) partial — `mlp_out`
    ///      `[hidden, B]` x num_layers
    ///
    /// All three are exact-shape `HiddenSlot` buffers at `seq_len == B`, so
    /// `all_reduce_sum`'s `data.len()`-derived message is exactly B columns.
    ///
    /// Advances every row's slot state (KV rows, conv ring, GDR state,
    /// `seq_len`) by one token, exactly like B sequential single-row decodes.
    /// Returns the B sampled tokens in row order.
    pub(crate) fn forward_decode_batch(
        &self,
        slots: &mut [Qwen35SlotState],
        bd: &mut Qwen35BatchDecodeState,
        slot_indices: &[usize],
        tokens: &[u32],
        kv_seq_lens: &[usize],
        params: &[SamplingParams],
        sample_positions: &[u64],
    ) -> Result<Vec<(u32, Option<f32>)>> {
        let b = tokens.len();
        ensure!(b >= 1, "Qwen3.5 batched decode requires at least one row");
        ensure!(
            slot_indices.len() == b
                && kv_seq_lens.len() == b
                && params.len() == b
                && sample_positions.len() == b,
            "Qwen3.5 batched decode surface length mismatch: slots={} tokens={} kv_lens={} params={} positions={}",
            slot_indices.len(),
            b,
            kv_seq_lens.len(),
            params.len(),
            sample_positions.len()
        );
        // Pre-mutation validation: every row in bounds and in budget BEFORE
        // any device state is touched.
        for (r, &si) in slot_indices.iter().enumerate() {
            ensure!(
                si < slots.len(),
                "Qwen3.5 batched decode slot {si} outside executor slots {}",
                slots.len()
            );
            ensure!(
                slots[si].seq_len() == kv_seq_lens[r],
                "Qwen3.5 batched decode materialized seq_len {} != scheduler kv_seq_len {} for slot {si}",
                slots[si].seq_len(),
                kv_seq_lens[r]
            );
            // A decode-batch row's slot was activated at its start_pos==0 prefill;
            // its recurrent block MUST still be resident (the pointer tables below
            // dereference `gdr_states`).
            ensure!(
                slots[si].has_recurrent(),
                "Qwen3.6 batched decode: slot {si} recurrent state not acquired"
            );
            ensure!(
                kv_seq_lens[r] < self.max_seq_len,
                "Qwen3.5 batched decode sequence {} exceeds KV cache budget {}",
                kv_seq_lens[r] + 1,
                self.max_seq_len
            );
        }

        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;
        let vocab = self.output_projection().rows;

        // ── Stage pointer tables (no-op when the row→slot mapping is unchanged). ──
        bd.stage_pointer_tables(&self.ctx, slots, slot_indices)?;

        // ── Stage per-step inputs: token ids + per-row absolute positions. ──
        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let positions_host: Vec<i32> = kv_seq_lens.iter().map(|&len| len as i32).collect();
        let seq_lens_host: Vec<i32> = positions_host.iter().map(|&p| p + 1).collect();

        let Qwen35BatchDecodeState {
            ws,
            positions,
            seq_lens,
            full_k_cache_ptrs,
            full_v_cache_ptrs,
            conv_state_ptrs,
            gdr_state_ptrs,
            logits_batch,
            argmax,
            ..
        } = bd;
        let Qwen35Workspace {
            token_ids,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            logits: row_logits,
            ..
        } = ws;
        let token_ids = token_ids.upload(&self.ctx, &token_ids_host)?;
        let positions_dev = positions.upload(&self.ctx, &positions_host)?;
        let seq_lens_dev = seq_lens.upload(&self.ctx, &seq_lens_host)?;

        // ── Forward body: identical layer stack to `forward_hidden_staged`
        //    at seq_len == B, with the two per-slot dispatch differences. ──
        let hidden = hidden.get(&self.ctx, hidden_size, b)?;
        embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)?;
        let normed = normed.get(&self.ctx, hidden_size, b)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, b)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, b)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, b)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for layer in self.layers.iter() {
            rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed)?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    self.full_attention_batch_rows(
                        full_attn,
                        normed,
                        slots,
                        slot_indices,
                        full_idx,
                        positions_dev,
                        seq_lens_dev,
                        &full_k_cache_ptrs[full_idx],
                        &full_v_cache_ptrs[full_idx],
                        full,
                        attn_out,
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    ensure!(
                        linear_idx < conv_state_ptrs.len(),
                        "Qwen3.5 batched decode linear layer {linear_idx} outside pointer tables {}",
                        conv_state_ptrs.len()
                    );
                    self.linear_attention(
                        lin,
                        normed,
                        LinearCore::Tables {
                            conv: &conv_state_ptrs[linear_idx],
                            gdr: &gdr_state_ptrs[linear_idx],
                        },
                        linear_idx,
                        linear,
                        attn_out,
                    )?;
                    linear_idx += 1;
                }
            }

            // Post-attn residual add + post_attention_layernorm via the
            // `add_batch` + `rms_norm_offset` pair (`hidden_mid`/`normed`).
            add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
            rms_norm_offset(
                &self.ctx,
                hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                normed,
            )?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                moe_forward_into(
                    &self.ctx,
                    moe_weights,
                    mlp_in,
                    cfg,
                    &self.expert_split,
                    moe,
                    mlp_out,
                )?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                self.dense_mlp(mlp, mlp_in, dense, mlp_out)?;
            }
            // ONE all-reduce covers the whole FFN partial (see the per-layer
            // enumeration in the method docs); exact `[hidden, B]` message.
            self.tp.all_reduce_sum(&self.ctx, mlp_out)?;

            // MLP residual add: the post-attn sum lives in `hidden_mid`;
            // add_batch reads hidden_mid/mlp_out and writes `hidden`.
            add_batch(&self.ctx, hidden_mid, mlp_out, hidden)?;
        }

        // ── Final norm over ALL rows + batched lm_head GEMM. ──
        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let logits_buf = logits_batch.get(&self.ctx, vocab, b)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, logits_buf)?;

        // Host seq_len advance: the device state (KV rows, conv rings, GDR
        // states) advanced in-stream above, so the host counters advance here
        // regardless of how sampling below fares — host and device stay
        // consistent (mirrors `forward_hidden`).
        for &si in slot_indices {
            slots[si].advance_seq_len(1);
        }

        // ── Sampling: ONE batched argmax over `[B, vocab]` (greedy fast
        //    path); any non-greedy row falls back to per-row host sampling. ──
        let argmax_buf = argmax.get(&self.ctx, b)?;
        {
            let (l_ptr, _gl) = logits_buf.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _ga) = argmax_buf.device_ptr_mut(&self.ctx.stream);
            // SAFETY: logits is a live `[B, vocab]` bf16 buffer and argmax a
            // live `[B]` i32 buffer on ctx.stream.
            unsafe {
                ffi::argmax_batch_cuda(
                    l_ptr as *const ffi::Half,
                    a_ptr as *mut i32,
                    b as i32,
                    vocab as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        self.ctx.sync()?;
        let greedy_ids = self
            .ctx
            .stream
            .clone_dtoh(argmax_buf)
            .map_err(|e| anyhow!("D2H qwen35 batched argmax failed: {e}"))?;
        let out = params
            .iter()
            .enumerate()
            .map(|(r, p)| -> anyhow::Result<(u32, Option<f32>)> {
                if p.is_greedy() {
                    return Ok((greedy_ids[r] as u32, None));
                }
                let row_vec = row_logits.get(&self.ctx, vocab)?;
                copy_row_to_vec(&self.ctx, logits_buf, r, row_vec)?;
                let host = row_vec.to_host(&self.ctx)?;
                Ok(infer_plan::sample_token_logprob(
                    &host,
                    p,
                    sample_positions[r],
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(out)
    }

    /// PAGED batched decode (the shared-paged default lane): the exact body of
    /// [`Self::forward_decode_batch`] (embed → layer loop → final norm → batched
    /// lm_head + argmax), but full-attn layers route through
    /// [`Self::full_attention_paged`] against the shared `pool` + the
    /// B-row `meta` ([`PageMeta::for_decode_batch`]) instead of the contiguous
    /// per-slot caches. The engine has already appended this step's token to
    /// each row's slot in the pool and built `meta`; this method only runs the
    /// forward + samples. Linear (recurrent) layers are pool-independent and use
    /// the SAME batched conv1d/GDR kernels as the contiguous lane.
    ///
    /// Caller contract (mirrors `forward_decode_batch`): one row per
    /// `slot_indices`/`tokens`/`params`/`sample_positions`; `kv_seq_lens[r]` is
    /// the pre-append length (== the engine's `DecodeRow.kv_seq_len`); `meta`
    /// describes the same B rows POST-append (its `positions`/page slices carry
    /// `kv_seq_lens[r]` as the query position). B==1 stays on the single-row
    /// paged path; this only runs for B>1.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_decode_batch_paged(
        &self,
        slots: &mut [Qwen35SlotState],
        bd: &mut Qwen35BatchDecodeState,
        pool: &mut PagedKVPool,
        meta: &crate::loader::PageMeta,
        slot_indices: &[usize],
        tokens: &[u32],
        kv_seq_lens: &[usize],
        params: &[SamplingParams],
        sample_positions: &[u64],
    ) -> Result<Vec<(u32, Option<f32>)>> {
        let b = tokens.len();
        ensure!(
            b >= 1,
            "Qwen3.6 paged batched decode requires at least one row"
        );
        ensure!(
            slot_indices.len() == b
                && kv_seq_lens.len() == b
                && params.len() == b
                && sample_positions.len() == b,
            "Qwen3.6 paged batched decode surface length mismatch: slots={} tokens={} kv_lens={} params={} positions={}",
            slot_indices.len(),
            b,
            kv_seq_lens.len(),
            params.len(),
            sample_positions.len()
        );
        ensure!(
            meta.batch == b && meta.total_q == b,
            "Qwen3.6 paged batched decode meta (batch {}, total_q {}) != {} one-token rows",
            meta.batch,
            meta.total_q,
            b
        );
        // Pre-mutation validation: every row in bounds, recurrent resident, and
        // the pool already holds this row's POST-append length (the engine
        // appended one token per row before building `meta`).
        for (r, &si) in slot_indices.iter().enumerate() {
            ensure!(
                si < slots.len(),
                "Qwen3.6 paged batched decode slot {si} outside executor slots {}",
                slots.len()
            );
            ensure!(
                slots[si].seq_len() == kv_seq_lens[r],
                "Qwen3.6 paged batched decode materialized seq_len {} != scheduler kv_seq_len {} for slot {si}",
                slots[si].seq_len(),
                kv_seq_lens[r]
            );
            ensure!(
                slots[si].has_recurrent(),
                "Qwen3.6 paged batched decode: slot {si} recurrent state not acquired"
            );
            ensure!(
                pool.seq_len(si) == kv_seq_lens[r] + 1,
                "Qwen3.6 paged batched decode: pool seq_len {} != kv_seq_len+1 {} for slot {si}",
                pool.seq_len(si),
                kv_seq_lens[r] + 1
            );
        }

        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;
        let vocab = self.output_projection().rows;

        // Stage the recurrent pointer tables ONLY (paged full-attn needs no
        // contiguous K/V tables); no-op when the row→slot mapping is unchanged.
        bd.stage_recurrent_pointer_tables(&self.ctx, slots, slot_indices)?;

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

        let Qwen35BatchDecodeState {
            ws,
            conv_state_ptrs,
            gdr_state_ptrs,
            logits_batch,
            argmax,
            ..
        } = bd;
        let Qwen35Workspace {
            token_ids,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            logits: row_logits,
            ..
        } = ws;
        let token_ids = token_ids.upload(&self.ctx, &token_ids_host)?;

        let hidden = hidden.get(&self.ctx, hidden_size, b)?;
        embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)?;
        let normed = normed.get(&self.ctx, hidden_size, b)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, b)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, b)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, b)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for layer in &self.layers {
            rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed)?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    self.full_attention_paged(
                        full_attn, normed, full_idx, pool, meta, full, attn_out, None,
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    ensure!(
                        linear_idx < conv_state_ptrs.len(),
                        "Qwen3.6 paged batched decode linear layer {linear_idx} outside pointer tables {}",
                        conv_state_ptrs.len()
                    );
                    self.linear_attention(
                        lin,
                        normed,
                        LinearCore::Tables {
                            conv: &conv_state_ptrs[linear_idx],
                            gdr: &gdr_state_ptrs[linear_idx],
                        },
                        linear_idx,
                        linear,
                        attn_out,
                    )?;
                    linear_idx += 1;
                }
            }

            add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
            rms_norm_offset(
                &self.ctx,
                hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                normed,
            )?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                moe_forward_into(
                    &self.ctx,
                    moe_weights,
                    mlp_in,
                    cfg,
                    &self.expert_split,
                    moe,
                    mlp_out,
                )?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                self.dense_mlp(mlp, mlp_in, dense, mlp_out)?;
            }
            self.tp.all_reduce_sum(&self.ctx, mlp_out)?;
            add_batch(&self.ctx, hidden_mid, mlp_out, hidden)?;
        }

        rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)?;
        let logits_buf = logits_batch.get(&self.ctx, vocab, b)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, logits_buf)?;

        // Host seq_len advance (device KV/conv/GDR advanced in-stream above).
        for &si in slot_indices {
            slots[si].advance_seq_len(1);
        }

        let argmax_buf = argmax.get(&self.ctx, b)?;
        {
            let (l_ptr, _gl) = logits_buf.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _ga) = argmax_buf.device_ptr_mut(&self.ctx.stream);
            // SAFETY: logits `[B, vocab]` bf16, argmax `[B]` i32, both on ctx.stream.
            unsafe {
                ffi::argmax_batch_cuda(
                    l_ptr as *const ffi::Half,
                    a_ptr as *mut i32,
                    b as i32,
                    vocab as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        self.ctx.sync()?;
        let greedy_ids = self
            .ctx
            .stream
            .clone_dtoh(argmax_buf)
            .map_err(|e| anyhow!("D2H qwen36 paged batched argmax failed: {e}"))?;
        let out = params
            .iter()
            .enumerate()
            .map(|(r, p)| -> anyhow::Result<(u32, Option<f32>)> {
                if p.is_greedy() {
                    return Ok((greedy_ids[r] as u32, None));
                }
                let row_vec = row_logits.get(&self.ctx, vocab)?;
                copy_row_to_vec(&self.ctx, logits_buf, r, row_vec)?;
                let host = row_vec.to_host(&self.ctx)?;
                Ok(infer_plan::sample_token_logprob(
                    &host,
                    p,
                    sample_positions[r],
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(out)
    }

    /// Batched-decode full attention: batched q/k/v projections over all B
    /// rows, then one split-KV fused decode launch over grid.z = B. The kernel
    /// reads per-row positions / seq_lens and per-row K/V cache pointers from
    /// device arrays, so the host never loops over rows for full-attn decode.
    /// head_dim != 256 keeps the per-row path.
    #[allow(clippy::too_many_arguments)]
    fn full_attention_batch_rows(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
        full_idx: usize,
        positions_dev: &CudaSlice<i32>,
        seq_lens_dev: &CudaSlice<i32>,
        k_cache_table: &CudaSlice<u64>,
        v_cache_table: &CudaSlice<u64>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let b = normed.seq_len;
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();

        let FullAttnScratch {
            qkv_fused,
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse: _,
            fa3_oaccum: _,
            fa3_lseaccum: _,
            fa3_semaphore: _,
            batch_partial_out,
            batch_partial_m,
            batch_partial_l,
        } = fw;
        let qkv_fused = qkv_fused.get(&self.ctx, q_proj_dim + 2 * kv_dim, b)?;
        let q_full = q_full.get(&self.ctx, q_proj_dim, b)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, b)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, b)?;
        gemm_batch(&self.ctx, &attn.qkv_proj, normed, qkv_fused)?;
        split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;

        let attn_heads = attn_heads.get(&self.ctx, q_dim, b)?;

        if c.head_dim == 256 {
            ensure!(
                self.local_kv_heads > 0 && self.local_q_heads.is_multiple_of(self.local_kv_heads),
                "Qwen3.5 batched decode full-attn requires integral GQA ratio: q_heads={} kv_heads={}",
                self.local_q_heads,
                self.local_kv_heads
            );
            let partial_scalars = b * self.local_q_heads * QWEN35_BATCHED_DECODE_KV_SPLITS;
            let partial_out = batch_partial_out.get(&self.ctx, partial_scalars * c.head_dim)?;
            let partial_m = batch_partial_m.get(&self.ctx, partial_scalars)?;
            let partial_l = batch_partial_l.get(&self.ctx, partial_scalars)?;
            let (qf_base, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_base, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_base, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (pos_ptr, _g7) = positions_dev.device_ptr(&self.ctx.stream);
            let (seq_ptr, _g8) = seq_lens_dev.device_ptr(&self.ctx.stream);
            let (ktbl_ptr, _g9) = k_cache_table.device_ptr(&self.ctx.stream);
            let (vtbl_ptr, _g10) = v_cache_table.device_ptr(&self.ctx.stream);
            let (po_ptr, _g11) = partial_out.device_ptr_mut(&self.ctx.stream);
            let (pm_ptr, _g12) = partial_m.device_ptr_mut(&self.ctx.stream);
            let (pl_ptr, _g13) = partial_l.device_ptr_mut(&self.ctx.stream);
            let (ao_base, _g14) = attn_heads.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q/k/v projections and output are live `[B, *]` buffers;
            // positions/seq_lens are live `[B]` i32 device arrays; K/V pointer
            // tables hold the first B live per-slot caches staged for this
            // row→slot mapping. The fused kernel writes one new K/V row per
            // batch item and all partial slots; reduce writes every attn_heads
            // element before gate reads it.
            unsafe {
                ffi::fused_gqa_attention_decode_batched(
                    qf_base as *const ffi::Half,
                    k_base as *const ffi::Half,
                    v_base as *const ffi::Half,
                    qn_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    seq_ptr as *const i32,
                    ktbl_ptr as *const *const ffi::Half,
                    vtbl_ptr as *const *const ffi::Half,
                    po_ptr as *mut f32,
                    pm_ptr as *mut f32,
                    pl_ptr as *mut f32,
                    self.local_q_heads as i32,
                    self.local_kv_heads as i32,
                    (self.local_q_heads / self.local_kv_heads) as i32,
                    c.head_dim as i32,
                    c.rotary_dim as i32,
                    self.max_seq_len as i32,
                    b as i32,
                    c.rms_norm_eps,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::attention_decode_reduce_batched(
                    po_ptr as *const f32,
                    pm_ptr as *const f32,
                    pl_ptr as *const f32,
                    ao_base as *mut ffi::Half,
                    self.local_q_heads as i32,
                    c.head_dim as i32,
                    b as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::attention_gate_batch_hd256_cuda(
                    qf_base as *const ffi::Half,
                    ao_base as *mut ffi::Half,
                    self.local_q_heads as i32,
                    c.head_dim as i32,
                    b as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        } else {
            let q_prepped = q_prepped.get(&self.ctx, q_dim, b)?;
            let (qf_base, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_base, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_base, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_base, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (ao_base, _g8) = attn_heads.data.device_ptr_mut(&self.ctx.stream);
            let (pos_base, _g9) = positions_dev.device_ptr(&self.ctx.stream);

            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                let k_cache = &mut slot.k_caches[full_idx];
                let v_cache = &mut slot.v_caches[full_idx];
                let max_seq_len = k_cache.len / kv_dim;
                let (kc_ptr, _gk) = k_cache.data.device_ptr_mut(&self.ctx.stream);
                let (vc_ptr, _gv) = v_cache.data.device_ptr_mut(&self.ctx.stream);
                let qf_r = qf_base + (r * q_proj_dim * 2) as u64;
                let k_r = k_base + (r * kv_dim * 2) as u64;
                let v_r = v_base + (r * kv_dim * 2) as u64;
                let qp_r = qp_base + (r * q_dim * 2) as u64;
                let ao_r = ao_base + (r * q_dim * 2) as u64;
                let pos_r = pos_base + (r * 4) as u64;
                // SAFETY: every pointer is a live device allocation on
                // ctx.stream; offsets stay inside the `[*, B]` buffers for
                // r < B; each kernel runs at seq_len == 1 so it touches only
                // row r's block + slot r's caches.
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qf_r as *const ffi::Half,
                        k_r as *const ffi::Half,
                        v_r as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        qp_r as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        1,
                        pos_r as *const i32,
                        c.rotary_dim as i32,
                        c.rms_norm_eps,
                        max_seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                    ffi::nonpaged_prefill_attention_devpos_cuda(
                        qp_r as *const ffi::Half,
                        kc_ptr as *const ffi::Half,
                        vc_ptr as *const ffi::Half,
                        ao_r as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        1,
                        pos_r as *const i32,
                        max_seq_len as i32,
                        sm_scale,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_r as *const ffi::Half,
                        ao_r as *mut ffi::Half,
                        self.local_q_heads as i32,
                        c.head_dim as i32,
                        1,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }

        gemm_batch(&self.ctx, &attn.o_proj, attn_heads, out)?;
        // Row-parallel o_proj: ONE all-reduce over the exact `[hidden, B]`
        // buffer — message length is B valid columns by construction (no-op
        // single-GPU).
        self.tp.all_reduce_sum(&self.ctx, out)?;
        Ok(())
    }
}

/// The gated-delta fused `in_proj_qkv` row blocks `[q(Kh×Kd); k(Kh×Kd); v(Vh×Vd)]`
/// (gated_delta_rule.cu reads q at `k_head*Kd + d`, k at `q_dim + k_head*Kd + d`,
/// v at `q_dim + k_dim + v_head*Vd + d`). Shared by the qkv weight slice and the
/// depthwise conv1d slice — conv channel `c` filters in_proj_qkv output row `c`,
/// so both must shard the SAME row ranges.
fn linear_qkv_head_blocks(m: &Qwen35Config) -> [crate::shard_slice::HeadBlock; 3] {
    let k_block = crate::shard_slice::HeadBlock {
        heads: m.linear_num_key_heads,
        head_rows: m.linear_key_head_dim,
    };
    let v_block = crate::shard_slice::HeadBlock {
        heads: m.linear_num_value_heads,
        head_rows: m.linear_value_head_dim,
    };
    [k_block, k_block, v_block]
}

/// Load the fused gated-delta `in_proj_qkv` (`[2*Kh*Kd + Vh*Vd, hidden]` BF16)
/// sharded for this TP rank: each of the `[q; k; v]` blocks is head-sharded
/// independently and re-stacked, preserving the k↔v head grouping (a flat
/// column shard would cut across the block boundaries).
fn load_linear_qkv_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    m: &Qwen35Config,
    tp: &TpConfig,
) -> Result<DeviceMatrix> {
    let head_blocks = linear_qkv_head_blocks(m);
    // FP8 block-scaled checkpoints (e.g. Qwen3.6-27B-FP8) carry the fused qkv as
    // F8_E4M3 + a `weight_scale_inv` sidecar; shard both with the same head-block
    // helper as the BF16 path. `None` → no quant view, keep the BF16 path below
    // byte-for-byte. head_dim == block_m makes head-block boundaries land on
    // scale-row boundaries, so the 3-block re-stack mirrors 1:1 in scale units.
    if let Some(matrix) = loader.load_linear_qkv_fp8_head_sharded(ctx, name, &head_blocks, tp)? {
        return Ok(matrix);
    }
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "{name}: expected BF16 fused qkv projection, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape.len() == 2 && tensor.shape[0] == m.linear_attn_qkv_dim(),
        "{name}: expected [{}, hidden] fused qkv projection, got shape {:?}",
        m.linear_attn_qkv_dim(),
        tensor.shape
    );
    let sharded = crate::shard_slice::shard_head_blocks_column_parallel(
        &tensor.bytes,
        tensor.shape[1],
        2,
        &head_blocks,
        tp,
    )?;
    DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
        .map_err(|e| anyhow!("upload sharded fused qkv {name}: {e}"))
}

/// Load the depthwise conv1d weight (`[qkv_dim, 1, kernel]` BF16) sharded for
/// this TP rank as a flat `[local_qkv_dim*kernel]` [`DeviceVec`]: the channel
/// rows mirror [`load_linear_qkv_sharded`]'s `[q; k; v]` block shard exactly
/// (conv1d.cu reads `conv_weight[c*kernel + k]` for channel `c`).
fn load_conv1d_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    m: &Qwen35Config,
    tp: &TpConfig,
) -> Result<DeviceVec> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "{name}: expected BF16 conv1d weight, got {:?}",
        tensor.dtype
    );
    let channels = tensor.shape.first().copied().unwrap_or(0);
    ensure!(
        channels == m.linear_attn_qkv_dim(),
        "{name}: conv1d channels {channels} != qkv_dim {} (shape {:?})",
        m.linear_attn_qkv_dim(),
        tensor.shape
    );
    // `[channels, 1, kernel]` (HF) or `[channels, kernel]`: the singleton middle
    // dim is squeezed by treating each channel's row as `kernel` elements.
    let kernel: usize = tensor.shape[1..].iter().product();
    ensure!(
        kernel == m.linear_conv_kernel_dim,
        "{name}: conv1d kernel {kernel} != linear_conv_kernel_dim {} (shape {:?})",
        m.linear_conv_kernel_dim,
        tensor.shape
    );
    let sharded = crate::shard_slice::shard_head_blocks_column_parallel(
        &tensor.bytes,
        kernel,
        2,
        &linear_qkv_head_blocks(m),
        tp,
    )?;
    DeviceVec::from_safetensors(ctx, &sharded.bytes)
        .map_err(|e| anyhow!("upload sharded conv1d {name}: {e}"))
}

/// Load a per-v-head 1D vector (`[Vh]`, BF16 or F32 normalized to BF16 exactly
/// like the single-GPU `load_vec_any`) sliced to this rank's contiguous v-head
/// range (gated_delta_rule.cu indexes `dt_bias[v_head]`).
fn load_v_head_vec_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    total_v_heads: usize,
    tp: &TpConfig,
) -> Result<DeviceVec> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.shape.len() == 1 && tensor.shape[0] == total_v_heads,
        "{name}: expected 1D [{total_v_heads}] per-v-head vector, got shape {:?}",
        tensor.shape
    );
    let bf16_bytes = SafetensorLoader::dsv4_bytes_to_bf16(name, &tensor)?;
    let (start, len) = v_head_shard_range(name, total_v_heads, tp)?;
    DeviceVec::from_safetensors(ctx, &bf16_bytes[start * 2..(start + len) * 2])
        .map_err(|e| anyhow!("upload sharded per-v-head vec {name}: {e}"))
}

/// Load a per-v-head 1D F32 tensor (`A_log`; F32 passthrough or BF16 widened,
/// matching the single-GPU `load_f32_vec`) sliced to this rank's v-head range,
/// uploaded as a device `f32` slice.
fn load_v_head_f32_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    total_v_heads: usize,
    tp: &TpConfig,
) -> Result<CudaSlice<f32>> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.shape.len() == 1 && tensor.shape[0] == total_v_heads,
        "{name}: expected 1D [{total_v_heads}] per-v-head tensor, got shape {:?}",
        tensor.shape
    );
    let host: Vec<f32> = match tensor.dtype {
        Dtype::F32 => tensor
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::BF16 => tensor
            .bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => anyhow::bail!("{name}: expected F32/BF16 1D tensor, got {other:?}"),
    };
    let (start, len) = v_head_shard_range(name, total_v_heads, tp)?;
    ctx.stream
        .clone_htod(&host[start..start + len])
        .map_err(|e| anyhow!("upload sharded per-v-head f32 {name}: {e}"))
}

/// Load the NextN-MTP draft head (single-GPU): one Full-attention transformer
/// block, the `fc` concat-projection, two pre-`fc` RMSNorms, and the final
/// pre-lm_head RMSNorm. The block's FFN is MoE when the base model is MoE
/// (`m.is_moe()`), mirroring the trunk layer's `load_moe_layer_experts`;
/// otherwise a dense MLP. `lm_head`/`embed_tokens` are SHARED with the base
/// model and are not reloaded here.
fn load_qwen35_mtp_head(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    m: &Qwen35Config,
    split: &ExpertSplit,
    tp: &TpConfig,
) -> Result<Qwen35MtpHead> {
    let names = m.mtp_tensor_names();
    let Qwen35AttentionTensorNames::Full(full) = &names.layer.attention else {
        unreachable!("MTP head layer is always full attention");
    };
    let attn = Qwen35Attn::Full(Box::new(FullAttn {
        qkv_proj: loader.load_matrices_row_fused(
            ctx,
            &[
                (full.q_proj.as_str(), None),
                (full.k_proj.as_str(), None),
                (full.v_proj.as_str(), None),
            ],
        )?,
        o_proj: loader.load_matrix_quant_aware(ctx, &full.o_proj)?,
        q_norm: loader.load_vec(ctx, &full.q_norm)?,
        k_norm: loader.load_vec(ctx, &full.k_norm)?,
    }));
    let (mlp, moe) = if m.is_moe() {
        let moe = loader.load_moe_layer_experts(
            ctx,
            &names.layer.common.moe_tensor_names(),
            split,
            tp,
            m.moe_intermediate_size,
            m.hidden_size,
        )?;
        (None, Some(moe))
    } else {
        let mlp = DenseMlp {
            gate_up_proj: loader.load_matrix_pair_fused(
                ctx,
                &names.layer.common.mlp_gate_proj,
                &names.layer.common.mlp_up_proj,
            )?,
            down_proj: loader.load_matrix_quant_aware(ctx, &names.layer.common.mlp_down_proj)?,
        };
        (Some(mlp), None)
    };
    let layer = Qwen35Layer {
        input_layernorm: loader.load_vec(ctx, &names.layer.common.input_layernorm)?,
        attn,
        post_attention_layernorm: loader
            .load_vec(ctx, &names.layer.common.post_attention_layernorm)?,
        mlp,
        moe,
    };
    Ok(Qwen35MtpHead {
        pre_fc_norm_embedding: loader.load_vec(ctx, &names.pre_fc_norm_embedding)?,
        pre_fc_norm_hidden: loader.load_vec(ctx, &names.pre_fc_norm_hidden)?,
        fc: loader.load_matrix_quant_aware(ctx, &names.fc)?,
        layer,
        norm: loader.load_vec(ctx, &names.norm)?,
    })
}

/// This rank's contiguous v-head range `(start, len)` within `[0, total_v_heads)`.
fn v_head_shard_range(name: &str, total_v_heads: usize, tp: &TpConfig) -> Result<(usize, usize)> {
    ensure!(
        total_v_heads.is_multiple_of(tp.world_size),
        "{name}: {total_v_heads} v heads not divisible by world_size {}",
        tp.world_size
    );
    let local = total_v_heads / tp.world_size;
    Ok((tp.rank * local, local))
}

/// Offset RMSNorm (1+weight) over a batch — Qwen3.5 norms store `weight - 1`.
/// Spec-decode phase attribution: returns `Some(Instant)` only when
/// `ARLE_MTP_PHASE` is set (the per-phase sync needed for accurate GPU timing is
/// opt-in, so the default spec-decode path pays nothing).
fn mtp_phase_start(ctx: &DeviceContext) -> Option<std::time::Instant> {
    phase_start(ctx, "ARLE_MTP_PHASE")
}

/// Same opt-in phase timer keyed on `ARLE_DSPARK_PHASE` (DSpark block step).
pub(crate) fn dspark_phase_start(ctx: &DeviceContext) -> Option<std::time::Instant> {
    phase_start(ctx, "ARLE_DSPARK_PHASE")
}

fn phase_start(ctx: &DeviceContext, var: &str) -> Option<std::time::Instant> {
    if std::env::var(var).is_ok() {
        let _ = ctx.sync();
        Some(std::time::Instant::now())
    } else {
        None
    }
}

/// Sync + return ms since the last lap (or 0.0 when phase timing is off).
pub(crate) fn mtp_phase_lap(ctx: &DeviceContext, t: &mut Option<std::time::Instant>) -> f64 {
    match t {
        Some(prev) => {
            let _ = ctx.sync();
            let now = std::time::Instant::now();
            let ms = now.duration_since(*prev).as_secs_f64() * 1000.0;
            *t = Some(now);
            ms
        }
        None => 0.0,
    }
}

/// log p_filtered of each committed chain token, read from the materialized
/// filtered `p` rows (`dspark_filter_probs_cuda` output; row j produced
/// `tokens[j]`). Caller contract: the accept verdict's D2H + sync already ran,
/// so the rows are final and these 4-byte reads add no new sync. Committed
/// tokens always carry filtered mass > 0; the floor clamp only guards f32
/// underflow at `ln`.
fn chain_commit_logprobs(
    ctx: &DeviceContext,
    p_all: &CudaSlice<f32>,
    vocab: usize,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    tokens
        .iter()
        .enumerate()
        .map(|(j, &tok)| {
            let off = j * vocab + tok as usize;
            let p = ctx
                .stream
                .clone_dtoh(&p_all.slice(off..off + 1))
                .map_err(|e| anyhow!("D2H chain commit prob failed: {e}"))?[0];
            Ok(p.max(f32::MIN_POSITIVE).ln())
        })
        .collect()
}

fn rms_norm_offset(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: x/weight/out valid on ctx.stream; out matches x shape.
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Offset RMSNorm (1+weight) over a single vector (the final norm before lm_head).
fn rms_norm_offset_vec(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: x/weight/out valid on ctx.stream; out matches x len.
    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_spec::LayerType;

    /// G3 whole-slot spill: `from_bytes(to_bytes(img))` must reproduce EVERY
    /// field byte-for-byte (the new correctness surface — the device buffers and
    /// the round-trip are independently proven). Ragged per-vec lengths +
    /// non-trivial bytes exercise the length-prefixed layout and the f32/bf16
    /// LE round-trips; a truncated buffer must error, never restore garbage.
    #[test]
    fn slot_image_byte_inverse() {
        let img = Qwen35SlotImage {
            full_attn_pages: (0u8..=233).cycle().take(777).collect(),
            full_attn_page_count: 3,
            gdr_host: vec![
                vec![1.5_f32, -2.0, f32::from_bits(0x4048_f5c3)],
                vec![],
                vec![0.0, -0.0, 123.456],
            ],
            conv_host: vec![
                vec![bf16::from_f32(1.0), bf16::from_f32(-3.5)],
                vec![bf16::from_f32(0.25)],
                vec![],
            ],
            seq_len: 1234,
        };
        let bytes = img.to_bytes();
        let back = Qwen35SlotImage::from_bytes(&bytes).expect("round-trip");
        assert_eq!(back.full_attn_pages, img.full_attn_pages);
        assert_eq!(back.full_attn_page_count, img.full_attn_page_count);
        assert_eq!(back.seq_len, img.seq_len);
        // f32 / bf16 compared on bit patterns so a -0.0 or NaN can't false-pass.
        assert_eq!(back.gdr_host.len(), img.gdr_host.len());
        for (b, a) in back.gdr_host.iter().zip(&img.gdr_host) {
            let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
            let ab: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
            assert_eq!(bb, ab);
        }
        assert_eq!(back.conv_host.len(), img.conv_host.len());
        for (b, a) in back.conv_host.iter().zip(&img.conv_host) {
            let bb: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
            let ab: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
            assert_eq!(bb, ab);
        }
        // A truncated payload errors rather than restoring a partial image.
        assert!(Qwen35SlotImage::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        // A foreign trailing byte is rejected (exact-length invariant).
        let mut extra = bytes.clone();
        extra.push(0xAB);
        assert!(Qwen35SlotImage::from_bytes(&extra).is_err());
    }

    fn qwen35_dense_4b_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 2,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            stop_token_ids: vec![151_645],
            bos_token_id: None,
            eos_token_id: 151_645,
            tie_word_embeddings: true,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 128,
            rope_cache_len_hint: Some(32_768),
            layer_types: vec![LayerType::FullAttention, LayerType::LinearAttention],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: vec![],
            full_attn_gated: true,
        }
    }

    #[test]
    fn cuda_qwen35_guard_accepts_dense_hybrid_4b_shape() {
        let cfg = qwen35_dense_4b_config();
        cfg.validate().unwrap();
        validate_qwen35_cuda_config(&cfg).unwrap();
    }

    /// First index where two token streams differ (or the shorter length).
    fn first_divergence(a: &[u32], b: &[u32]) -> usize {
        a.iter()
            .zip(b.iter())
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()))
    }

    /// On-GPU correctness gate + perf A/B for the depth-1 NextN-MTP spec decode.
    ///
    /// Gated by `INFER_MTP_BENCH=1` (a no-op otherwise, so CI stays green). Run on
    /// a CUDA box with the 27B-FP8 checkpoint:
    /// ```text
    /// INFER_MTP_BENCH=1 INFER_MTP_MODEL=/data01/models/Qwen3.6-27B-FP8 \
    ///   INFER_MTP_PROMPT_IDS="<comma-separated real prompt token ids>" \
    ///   cargo test -p infer-cuda --release --features cuda \
    ///   mtp_spec_decode_gate_and_bench -- --ignored --nocapture
    /// ```
    /// Correctness (MoE → token-exact is confounded by atomic-scatter
    /// non-determinism, per the KV parity gate): run greedy no-spec TWICE to
    /// measure the self-consistency floor, then assert spec greedy diverges from
    /// the no-spec reference NO EARLIER than that floor. Perf: same binary, same
    /// prompt, same token budget — clean in-process A/B (no HTTP/scheduler noise).
    #[test]
    #[ignore = "GPU + 27B-FP8 checkpoint; opt in via INFER_MTP_BENCH=1"]
    fn mtp_spec_decode_gate_and_bench() {
        use std::time::Instant;
        if std::env::var("INFER_MTP_BENCH").is_err() {
            return;
        }
        let path = std::env::var("INFER_MTP_MODEL")
            .unwrap_or_else(|_| "/data01/models/Qwen3.6-27B-FP8".to_string());
        let prompt: Vec<u32> = std::env::var("INFER_MTP_PROMPT_IDS")
            .expect("INFER_MTP_PROMPT_IDS (comma-separated token ids) required")
            .split(',')
            .map(|s| s.trim().parse::<u32>().expect("token id"))
            .collect();
        let n_decode: usize = std::env::var("INFER_MTP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
        let depth: usize = std::env::var("INFER_MTP_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1);
        let max_seq = (prompt.len() + n_decode + 8).next_power_of_two().max(2048);
        let greedy = SamplingParams {
            temperature: 0.0,
            top_k: -1,
            ..Default::default()
        };

        // Load with the MTP head sized for `depth` draft tokens (the head KV
        // needs depth+1 slots; new_spec_slot_state derives this from the param).
        let model = Qwen35Model::from_safetensors(Path::new(&path), max_seq, Some(depth))
            .expect("load Qwen3.6-27B-FP8 with MTP head");

        // Greedy no-spec: prefill the prompt, then decode `n_decode` tokens.
        let (num_linear, gdr_len, conv_len) = model.recurrent_dims();
        let run_nospec = || -> (Vec<u32>, f64) {
            let mut slot = model.new_slot_state();
            let mut acq_pool = Vec::new();
            slot.acquire_recurrent(&model.ctx, num_linear, gdr_len, conv_len, &mut acq_pool)
                .unwrap();
            let mut ws = Qwen35Workspace::new();
            let mut out = Vec::with_capacity(n_decode);
            let mut tok = model
                .forward_tokens(&mut slot, &mut ws, &prompt, 0, &greedy, 0)
                .unwrap()
                .0;
            out.push(tok);
            model.ctx.sync().unwrap();
            let t0 = Instant::now();
            for step in 1..n_decode {
                let sp = slot.seq_len();
                tok = model
                    .forward_tokens(&mut slot, &mut ws, &[tok], sp, &greedy, step as u64)
                    .unwrap()
                    .0;
                out.push(tok);
            }
            model.ctx.sync().unwrap();
            (out, t0.elapsed().as_secs_f64())
        };

        // Spec: prefill seeds (pending, hidden), then loop spec_step(depth).
        let run_spec = || -> (Vec<u32>, f64, usize, usize) {
            let mut slot = model.new_slot_state();
            let mut acq_pool = Vec::new();
            slot.acquire_recurrent(&model.ctx, num_linear, gdr_len, conv_len, &mut acq_pool)
                .unwrap();
            let mut spec = model.new_spec_slot_state().unwrap();
            let mut ws = Qwen35Workspace::new();
            // Prefill returns the next token's logits + the producing hidden.
            let (logits, dims, mut hidden) = model
                .forward_tokens_with_hidden(&mut slot, &mut ws, &prompt, 0, None)
                .unwrap();
            let vocab = dims[1];
            // The prefill's last-row argmax IS the first decoded token = the seed
            // `pending`; `hidden` is its producing trunk hidden (last prompt row).
            // On-device argmax (same path as the live decode), no full-logits D2H.
            let mut pending = argmax_row_into(
                &model.ctx,
                &logits,
                dims[0] - 1,
                vocab,
                &mut spec.argmax_scratch,
            )
            .unwrap();
            let mut out = vec![pending];
            let mut accepts = 0usize;
            let mut steps = 0usize;
            model.ctx.sync().unwrap();
            let t0 = Instant::now();
            while out.len() < n_decode {
                let sp = slot.seq_len();
                let (emitted, next_pending, next_hidden) = model
                    .spec_step(
                        &mut slot, &mut spec, &mut ws, pending, &hidden, sp, depth, &greedy, None,
                    )
                    .unwrap();
                // Accepted drafts this step = emitted.len() - 1 (the bonus is the
                // trunk's own token, not a draft). accept_rate = accepts/(steps*depth).
                accepts += emitted.len().saturating_sub(1);
                steps += 1;
                out.extend(emitted.iter().map(|&(t, _)| t));
                pending = next_pending;
                hidden = next_hidden;
            }
            model.ctx.sync().unwrap();
            (out, t0.elapsed().as_secs_f64(), accepts, steps)
        };

        let (ref1, _) = run_nospec();
        let (ref2, t_nospec) = run_nospec();
        let floor = first_divergence(&ref1, &ref2);

        let (spec_out, t_spec, accepts, steps) = run_spec();
        let spec_div = first_divergence(&ref1, &spec_out);

        let n = n_decode as f64;
        let drafted = (steps * depth).max(1);
        eprintln!(
            "[mtp-bench] depth={} | no-spec {:.1} tok/s ({:.3}s/{} tok) | spec {:.1} tok/s ({:.3}s) | speedup {:.2}x | accept {}/{} drafts = {:.1}% | {:.2} tok/step",
            depth,
            n / t_nospec,
            t_nospec,
            n_decode,
            n / t_spec,
            t_spec,
            t_nospec / t_spec,
            accepts,
            drafted,
            100.0 * accepts as f64 / drafted as f64,
            spec_out.len() as f64 / steps.max(1) as f64,
        );
        eprintln!(
            "[mtp-gate] MoE self-consistency floor: no-spec runs diverge @{floor}/{} | spec-vs-ref diverge @{spec_div}/{}",
            ref1.len(),
            ref1.len(),
        );
        // Spec is correct iff it tracks the no-spec reference at least as far as
        // two no-spec runs track each other (the MoE non-determinism floor).
        assert!(
            spec_div >= floor,
            "MTP spec greedy diverged from no-spec @{spec_div} BEFORE the MoE floor @{floor} — real spec bug, not non-determinism"
        );
    }

    #[test]
    fn cuda_qwen35_guard_rejects_unknown_full_attention_head_dim() {
        let mut cfg = qwen35_dense_4b_config();
        cfg.head_dim = 64;
        cfg.rotary_dim = 64;
        assert!(validate_qwen35_cuda_config(&cfg).is_err());
    }

    /// Reference host merge: `W[r,c] = base[r,c] + scale·Σ_k B[r,k]·A[k,c]`.
    fn host_merge_reference(
        base: &[bf16],
        a: &[f32],
        b: &[f32],
        rows: usize,
        cols: usize,
        rank: usize,
        scale: f32,
    ) -> Vec<bf16> {
        let mut merged = vec![bf16::ZERO; rows * cols];
        for row in 0..rows {
            let b_row = &b[row * rank..row * rank + rank];
            for col in 0..cols {
                let mut delta = 0.0f32;
                for (k, &b_rk) in b_row.iter().enumerate() {
                    delta += b_rk * a[k * cols + col];
                }
                let idx = row * cols + col;
                merged[idx] = bf16::from_f32(base[idx].to_f32() + scale * delta);
            }
        }
        merged
    }

    /// GPU gate: the on-device dense LoRA merge (`lora_device_gemm` +
    /// `lora_scaled_add_into`, including the host-side A transpose used by
    /// `merge_lora_proj_device`) must match the host triple-loop reference
    /// within BF16 tolerance. Requires a real CUDA device (run on GPU7).
    #[test]
    fn device_lora_merge_matches_host_reference() {
        let Ok(ctx) = DeviceContext::new() else {
            eprintln!("[device_lora_merge] no CUDA device; skipping");
            return;
        };

        // A representative dense projection shape (qkv-ish): rows=out, cols=in,
        // rank=32. Deterministic pseudo-random A/B/base.
        let rows = 512usize;
        let cols = 384usize;
        let rank = 32usize;
        let scale = 0.5f32 / rank as f32; // alpha=0.5 / rank

        let prng = |seed: usize| -> f32 {
            // Cheap deterministic LCG-ish value in [-0.5, 0.5).
            let x = (seed as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9_7F4A_7C15;
            ((x >> 33) as f32 / u32::MAX as f32) - 0.5
        };

        let base: Vec<bf16> = (0..rows * cols)
            .map(|i| bf16::from_f32(prng(i + 1) * 2.0))
            .collect();
        let a: Vec<f32> = (0..rank * cols).map(|i| prng(i + 1_000_003)).collect();
        let b: Vec<f32> = (0..rows * rank).map(|i| prng(i + 7_000_019)).collect();

        let reference = host_merge_reference(&base, &a, &b, rows, cols, rank, scale);

        // --- device path (replicates merge_lora_proj_device) ---
        // A transposed to [cols, rank] row-major: a_t[c*rank+k] = a[k*cols+c].
        let mut a_t = vec![bf16::ZERO; cols * rank];
        for k in 0..rank {
            for c in 0..cols {
                a_t[c * rank + k] = bf16::from_f32(a[k * cols + c]);
            }
        }
        let b_host: Vec<bf16> = b.iter().map(|&v| bf16::from_f32(v)).collect();

        let a_t_dev = DeviceVec::from_host(&ctx, &a_t).unwrap();
        let b_dev = DeviceVec::from_host(&ctx, &b_host).unwrap();
        let base_dev = DeviceVec::from_host(&ctx, &base).unwrap();
        let mut delta = DeviceVec::zeros(&ctx, rows * cols).unwrap();
        let mut out = DeviceMatrix::from_host(&ctx, &base, rows, cols).unwrap();

        crate::ops::lora_device_gemm(
            &ctx,
            &a_t_dev.data,
            &b_dev.data,
            &mut delta.data,
            rows,
            cols,
            rank,
        )
        .unwrap();
        crate::ops::lora_scaled_add_into(
            &ctx,
            &base_dev.data,
            &delta.data,
            &mut out.data,
            rows * cols,
            scale,
        )
        .unwrap();
        ctx.sync().unwrap();

        let device_merged = ctx.stream.clone_dtoh(&out.data).unwrap();
        ctx.sync().unwrap();
        assert_eq!(device_merged.len(), reference.len());

        // Cosine similarity + max-abs-err vs the host reference.
        let mut dot = 0.0f64;
        let mut nr = 0.0f64;
        let mut nd = 0.0f64;
        let mut max_abs = 0.0f32;
        for (r, d) in reference.iter().zip(device_merged.iter()) {
            let rf = r.to_f32();
            let df = d.to_f32();
            dot += (rf as f64) * (df as f64);
            nr += (rf as f64) * (rf as f64);
            nd += (df as f64) * (df as f64);
            max_abs = max_abs.max((rf - df).abs());
        }
        let cosine = dot / (nr.sqrt() * nd.sqrt());
        eprintln!(
            "[device_lora_merge] rows={rows} cols={cols} rank={rank} cosine={cosine:.8} max_abs_err={max_abs:e}"
        );
        assert!(
            cosine >= 0.9999,
            "device merge cosine {cosine:.8} < 0.9999 (max_abs_err {max_abs:e})"
        );
        // BF16 mantissa is 8 bits; a rank-32 reduction lands well within ~1e-2.
        assert!(
            max_abs <= 2.0e-2,
            "device merge max-abs-err {max_abs:e} exceeds BF16 tolerance"
        );
    }

    /// Microbench (ignored by default): replay the full per-step all-linear LoRA
    /// merge at Qwen3.5-4B dense shapes and time host triple-loop vs the
    /// on-device path. Run with `--ignored --nocapture` on GPU7.
    #[test]
    #[ignore = "GPU microbench; run explicitly with --ignored"]
    fn bench_lora_remerge_host_vs_device() {
        let Ok(ctx) = DeviceContext::new() else {
            eprintln!("[lora_remerge_bench] no CUDA device; skipping");
            return;
        };
        let hidden = 2560usize;
        let rank = 32usize;
        let scale = 16.0f32 / rank as f32;
        // Qwen3.5-4B all-linear per-layer projection [out, in] shapes
        // (full-attn QKVO + linear-attn in/out + dense MLP gate/up/down).
        let proj_shapes: &[(usize, usize)] = &[
            (4096, hidden), // q_proj  (32 heads × 128)
            (1024, hidden), // k_proj  (8  × 128)
            (1024, hidden), // v_proj
            (hidden, 4096), // o_proj
            (hidden, 9216), // mlp.down_proj
            (9216, hidden), // mlp.gate_proj
            (9216, hidden), // mlp.up_proj
        ];
        let num_layers = 40usize;

        let prng = |seed: usize| -> f32 {
            let x = (seed as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9_7F4A_7C15;
            ((x >> 33) as f32 / u32::MAX as f32) - 0.5
        };

        // Pre-stage device base + resident matrices once (the per-step cost is
        // the merge, not the one-time base capture).
        struct Proj {
            rows: usize,
            cols: usize,
            base: Vec<bf16>,
            a: Vec<f32>,
            b: Vec<f32>,
            base_dev: DeviceVec,
            matrix: DeviceMatrix,
        }
        let mut projs: Vec<Proj> = Vec::new();
        for layer in 0..num_layers {
            for (pi, &(rows, cols)) in proj_shapes.iter().enumerate() {
                let s = layer * 97 + pi * 13 + 1;
                let base: Vec<bf16> = (0..rows * cols)
                    .map(|i| bf16::from_f32(prng(i + s) * 2.0))
                    .collect();
                let a: Vec<f32> = (0..rank * cols).map(|i| prng(i + s + 11)).collect();
                let b: Vec<f32> = (0..rows * rank).map(|i| prng(i + s + 23)).collect();
                let base_dev = DeviceVec::from_host(&ctx, &base).unwrap();
                let matrix = DeviceMatrix::from_host(&ctx, &base, rows, cols).unwrap();
                projs.push(Proj {
                    rows,
                    cols,
                    base,
                    a,
                    b,
                    base_dev,
                    matrix,
                });
            }
        }
        let max_n = proj_shapes.iter().map(|(r, c)| r * c).max().unwrap();
        let mut delta = DeviceVec::zeros(&ctx, max_n).unwrap();

        // --- HOST triple-loop path (the old merge) ---
        let t0 = std::time::Instant::now();
        for p in &projs {
            let mut merged = vec![bf16::ZERO; p.rows * p.cols];
            for row in 0..p.rows {
                let b_row = &p.b[row * rank..row * rank + rank];
                for col in 0..p.cols {
                    let mut d = 0.0f32;
                    for (k, &b_rk) in b_row.iter().enumerate() {
                        d += b_rk * p.a[k * p.cols + col];
                    }
                    let idx = row * p.cols + col;
                    merged[idx] = bf16::from_f32(p.base[idx].to_f32() + scale * d);
                }
            }
            std::hint::black_box(&merged);
        }
        let host_ms = t0.elapsed().as_secs_f64() * 1e3;

        // --- DEVICE path (the new merge) ---
        ctx.sync().unwrap();
        let t1 = std::time::Instant::now();
        for p in &mut projs {
            let mut a_t = vec![bf16::ZERO; p.cols * rank];
            for k in 0..rank {
                for c in 0..p.cols {
                    a_t[c * rank + k] = bf16::from_f32(p.a[k * p.cols + c]);
                }
            }
            let b_host: Vec<bf16> = p.b.iter().map(|&v| bf16::from_f32(v)).collect();
            let a_t_dev = DeviceVec::from_host(&ctx, &a_t).unwrap();
            let b_dev = DeviceVec::from_host(&ctx, &b_host).unwrap();
            let n = p.rows * p.cols;
            crate::ops::lora_device_gemm(
                &ctx,
                &a_t_dev.data,
                &b_dev.data,
                &mut delta.data,
                p.rows,
                p.cols,
                rank,
            )
            .unwrap();
            let delta_view = delta.data.slice(0..n);
            crate::ops::lora_scaled_add_into(
                &ctx,
                &p.base_dev.data,
                &delta_view,
                &mut p.matrix.data,
                n,
                scale,
            )
            .unwrap();
        }
        ctx.sync().unwrap();
        let dev_ms = t1.elapsed().as_secs_f64() * 1e3;

        eprintln!(
            "[lora_remerge_bench] {} projections × {num_layers} layers ({} total)\n\
             [lora_remerge_bench]   HOST triple-loop: {host_ms:.1} ms\n\
             [lora_remerge_bench]   DEVICE merge:     {dev_ms:.1} ms\n\
             [lora_remerge_bench]   speedup:          {:.1}×",
            proj_shapes.len(),
            projs.len(),
            host_ms / dev_ms
        );
    }
}
