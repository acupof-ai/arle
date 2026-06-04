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
//!     the q_proj output (`q_proj` rows = `heads*head_dim*2`), HD256 / kv2, on a
//!     contiguous per-slot K/V cache.
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
//! points for FP8 (`dsv4_fp8_grouped_gemm`) / 4-bit (Qwen3.6-4bit q4k) are inside
//! [`crate::moe::moe_forward`]'s two `moe_bf16_grouped_gemm_*` calls — a follow-up.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use infer_plan::SamplingParams;
use qwen35_spec::{Qwen35AttentionTensorNames, Qwen35Config};

use crate::executor::sample_cuda_token;
use crate::loader::SafetensorLoader;
use crate::moe::moe_forward;
use crate::ops::{
    add_batch, copy_row_to_vec, embedding_batch, gemm_batch, gemv, silu_mul, upload_i32,
};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;

/// Per-slot full-attention K/V cache (one contiguous bf16 cache per full-attn
/// layer) + per-slot gated-delta recurrent state (state + conv ring per
/// linear-attn layer). Carried across prefill/decode for one request.
pub(crate) struct Qwen35SlotState {
    /// `[num_full_layers]` contiguous K caches, each `max_seq_len*kv_dim` bf16.
    k_caches: Vec<DeviceVec>,
    v_caches: Vec<DeviceVec>,
    /// `[num_linear_layers]` gated-delta recurrent states (`V*K*Vh` f32).
    gdr_states: Vec<CudaSlice<f32>>,
    /// `[num_linear_layers]` conv1d rings (`qkv_dim*(kernel-1)` bf16).
    conv_states: Vec<DeviceVec>,
    /// Tokens materialized into the caches so far (full-attn kv_len).
    seq_len: usize,
}

impl Qwen35SlotState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Qwen35Config,
        max_seq_len: usize,
    ) -> Result<Self> {
        let num_full = config.num_full_attention_layers();
        let num_linear = config.num_hidden_layers - num_full;
        let kv_dim = config.full_attn_kv_dim();
        let gdr_state_len = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_len = config.linear_attn_qkv_dim() * (config.linear_conv_kernel_dim - 1);

        let mut k_caches = Vec::with_capacity(num_full);
        let mut v_caches = Vec::with_capacity(num_full);
        for _ in 0..num_full {
            k_caches.push(DeviceVec::zeros(ctx, max_seq_len * kv_dim)?);
            v_caches.push(DeviceVec::zeros(ctx, max_seq_len * kv_dim)?);
        }
        let mut gdr_states = Vec::with_capacity(num_linear);
        let mut conv_states = Vec::with_capacity(num_linear);
        for _ in 0..num_linear {
            gdr_states.push(
                ctx.stream
                    .alloc_zeros::<f32>(gdr_state_len)
                    .map_err(|e| anyhow!("alloc gated-delta state failed: {e}"))?,
            );
            conv_states.push(DeviceVec::zeros(ctx, conv_len)?);
        }
        Ok(Self {
            k_caches,
            v_caches,
            gdr_states,
            conv_states,
            seq_len: 0,
        })
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset for a fresh generation in this slot (zeros recurrent + conv state,
    /// rewinds the full-attn cache cursor; stale cache rows are overwritten on
    /// the next prefill).
    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.seq_len = 0;
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
}

pub(crate) enum Qwen35Attn {
    // Boxed: `FullAttn`/`LinearAttn` are large (multiple DeviceMatrix) and
    // size-skewed; boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<FullAttn>),
    Linear(Box<LinearAttn>),
}

/// Gated full attention (q_proj carries the per-head sigmoid gate → rows =
/// `heads*head_dim*2`).
pub(crate) struct FullAttn {
    q_proj: DeviceMatrix,
    k_proj: DeviceMatrix,
    v_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
}

/// Gated-delta-rule linear attention.
pub(crate) struct LinearAttn {
    in_proj_qkv: DeviceMatrix,
    in_proj_z: DeviceMatrix,
    in_proj_b: DeviceMatrix,
    in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight `[qkv_dim*kernel]` bf16.
    conv1d_weight: DeviceVec,
    dt_bias: DeviceVec,
    /// `A_log` `[num_value_heads]` f32.
    a_log: CudaSlice<f32>,
    /// Gated output RMSNorm scale `[V*Vh]` f32.
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
    gate_proj: DeviceMatrix,
    up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
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
    /// Per-slot cache capacity (full-attn contiguous cache rows).
    max_seq_len: usize,
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

impl Qwen35Model {
    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    pub(crate) fn new_slot_state(&self) -> Result<Qwen35SlotState> {
        Qwen35SlotState::new(&self.ctx, &self.config, self.max_seq_len)
    }

    /// Load a single-GPU BF16 Qwen3.5/3.6 HYBRID MoE checkpoint.
    ///
    /// `max_seq_len` sizes the per-slot full-attn contiguous K/V cache.
    pub(crate) fn from_qwen35_moe_safetensors(
        model_path: &Path,
        max_seq_len: usize,
    ) -> Result<Self> {
        let m = Qwen35Config::from_model_dir(model_path)
            .map_err(|e| anyhow!("load Qwen3.5 config from {}: {e}", model_path.display()))?;
        ensure!(
            m.is_moe(),
            "from_qwen35_moe_safetensors requires a MoE checkpoint (num_experts > 0)"
        );
        ensure!(
            m.head_dim == 256 && m.num_attention_heads == 16 && m.num_key_value_heads == 2,
            "clean CUDA Qwen3.5 hybrid path only wires the HD256 q16/kv2 TileLang \
             full-attention kernels (the only HD256 SUPPORTED_HEADS that cover \
             Qwen3.6-35B-A3B); got heads={} kv_heads={} head_dim={}",
            m.num_attention_heads,
            m.num_key_value_heads,
            m.head_dim
        );
        // Full attention here is the GATED q_proj variant (Qwen3.5/3.6); the
        // HD256 prep+gate kernels assume it. Vanilla un-gated Qwen3 would need
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

        let moe_config = crate::moe_config::moe_config_from_qwen35(&m)?;
        let split = crate::moe_config::ExpertSplit::single(m.num_experts);

        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;

        let embed_tokens = loader.load_matrix(&ctx, m.embed_tokens_tensor_name())?;
        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, m.lm_head_tensor_name())?)
        };

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for layer_idx in 0..m.num_hidden_layers {
            let names = m.layer_tensor_names(layer_idx);
            let attn = match &names.attention {
                Qwen35AttentionTensorNames::Full(full) => Qwen35Attn::Full(Box::new(FullAttn {
                    q_proj: loader.load_matrix(&ctx, &full.q_proj)?,
                    k_proj: loader.load_matrix(&ctx, &full.k_proj)?,
                    v_proj: loader.load_matrix(&ctx, &full.v_proj)?,
                    o_proj: loader.load_matrix(&ctx, &full.o_proj)?,
                    q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                })),
                Qwen35AttentionTensorNames::Linear(lin) => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        in_proj_qkv: loader.load_matrix(&ctx, &lin.in_proj_qkv)?,
                        in_proj_z: loader.load_matrix(&ctx, &lin.in_proj_z)?,
                        in_proj_b: loader.load_matrix(&ctx, &lin.in_proj_b)?,
                        in_proj_a: loader.load_matrix(&ctx, &lin.in_proj_a)?,
                        conv1d_weight: loader.load_conv1d_vec(&ctx, &lin.conv1d_weight)?,
                        dt_bias: loader.load_vec_any(&ctx, &lin.dt_bias)?,
                        a_log: loader.load_f32_vec(&ctx, &lin.a_log)?,
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        out_proj: loader.load_matrix(&ctx, &lin.out_proj)?,
                    }))
                }
            };

            let (mlp, moe) = if m.is_moe_layer(layer_idx) {
                let moe =
                    loader.load_moe_layer_experts(&ctx, &names.common.layer_prefix, &split)?;
                (None, Some(moe))
            } else {
                (
                    Some(DenseMlp {
                        gate_proj: loader.load_matrix(&ctx, &names.common.mlp_gate_proj)?,
                        up_proj: loader.load_matrix(&ctx, &names.common.mlp_up_proj)?,
                        down_proj: loader.load_matrix(&ctx, &names.common.mlp_down_proj)?,
                    }),
                    None,
                )
            };

            layers.push(Qwen35Layer {
                input_layernorm: loader.load_vec(&ctx, &names.common.input_layernorm)?,
                attn,
                post_attention_layernorm: loader
                    .load_vec(&ctx, &names.common.post_attention_layernorm)?,
                mlp,
                moe,
            });
        }
        let norm = loader.load_vec(&ctx, m.norm_tensor_name())?;

        let rope_len = m
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        let (cos_cache, sin_cache) =
            crate::ops::precompute_rope(&ctx, m.head_dim, rope_len, m.rope_theta, None)?;
        ctx.sync()?;

        Ok(Self {
            ctx,
            config: m,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            moe_config: Some(moe_config),
            max_seq_len,
        })
    }

    /// Run prefill or decode for one row. `start_pos` is the absolute position of
    /// the first token; `tokens` are the new tokens (whole prompt on prefill, one
    /// token on decode). Advances `slot.seq_len` and the recurrent state. Returns
    /// the next sampled token.
    pub(crate) fn forward_tokens(
        &self,
        slot: &mut Qwen35SlotState,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "forward_tokens requires at least one token"
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
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for layer in &self.layers {
            let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            rms_norm_offset(&self.ctx, &hidden, &layer.input_layernorm, eps, &mut normed)?;

            let attn_out = match &layer.attn {
                Qwen35Attn::Full(full) => {
                    let out = self.full_attention(full, &normed, slot, full_idx, start_pos)?;
                    full_idx += 1;
                    out
                }
                Qwen35Attn::Linear(lin) => {
                    let out = self.linear_attention(lin, &normed, slot, linear_idx)?;
                    linear_idx += 1;
                    out
                }
            };

            let mut hidden_mid = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            add_batch(&self.ctx, &hidden, &attn_out, &mut hidden_mid)?;

            rms_norm_offset(
                &self.ctx,
                &hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut normed,
            )?;
            let mlp_out = if let Some(moe) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                moe_forward(&self.ctx, moe, &normed, cfg)?
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                self.dense_mlp(mlp, &normed)?
            };

            let mut hidden_next = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
            add_batch(&self.ctx, &hidden_mid, &mlp_out, &mut hidden_next)?;
            hidden = hidden_next;
        }

        slot.seq_len += seq_len;

        // Final norm (offset) + LM head on the last token only.
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        copy_row_to_vec(&self.ctx, &hidden, seq_len - 1, &mut last_hidden)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        rms_norm_offset_vec(&self.ctx, &last_hidden, &self.norm, eps, &mut last_normed)?;
        let mut logits = DeviceVec::zeros(&self.ctx, self.output_projection().rows)?;
        gemv(
            &self.ctx,
            self.output_projection(),
            &last_normed,
            &mut logits,
        )?;
        sample_cuda_token(&self.ctx, &logits, params, position)
    }

    fn dense_mlp(&self, mlp: &DenseMlp, normed: &HiddenStates) -> Result<HiddenStates> {
        let inter = mlp.gate_proj.rows;
        let seq_len = normed.seq_len;
        let mut gate = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut up = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        gemm_batch(&self.ctx, &mlp.gate_proj, normed, &mut gate)?;
        gemm_batch(&self.ctx, &mlp.up_proj, normed, &mut up)?;
        let mut act = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        silu_mul(&self.ctx, &gate, &up, &mut act)?;
        let mut out = HiddenStates::zeros(&self.ctx, self.config.hidden_size, seq_len)?;
        gemm_batch(&self.ctx, &mlp.down_proj, &act, &mut out)?;
        Ok(out)
    }

    /// Gated full attention over the contiguous per-slot K/V cache (uncached
    /// recompute over `[0, start_pos+seq_len)` each call). HD256 prep fuses
    /// q/k RMSNorm + RoPE + cache write; the gate kernel applies the per-head
    /// sigmoid gate carried in `q_full`.
    fn full_attention(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        full_idx: usize,
        start_pos: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        let q_dim = c.full_attn_q_dim();
        let kv_dim = c.full_attn_kv_dim();
        let q_proj_dim = c.full_attn_q_proj_dim();

        let mut q_full = HiddenStates::zeros(&self.ctx, q_proj_dim, seq_len)?;
        let mut k_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut v_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        gemm_batch(&self.ctx, &attn.q_proj, normed, &mut q_full)?;
        gemm_batch(&self.ctx, &attn.k_proj, normed, &mut k_batch)?;
        gemm_batch(&self.ctx, &attn.v_proj, normed, &mut v_batch)?;

        let mut q_prepped = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut attn_out = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let k_cache = &mut slot.k_caches[full_idx];
        let v_cache = &mut slot.v_caches[full_idx];

        let start_pos_buf = upload_i32(&self.ctx, &[start_pos as i32])?;
        let max_seq_len = k_cache.len / kv_dim;
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let kv_len = start_pos + seq_len;

        // ── 1. Prep: q/k RMSNorm + RoPE + write K/V into the contiguous cache. ──
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
            let (sp_ptr, _g10) = start_pos_buf.device_ptr(&self.ctx.stream);
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
                    c.num_attention_heads as i32,
                    c.num_key_value_heads as i32,
                    seq_len as i32,
                    sp_ptr as *const i32,
                    c.rotary_dim as i32,
                    c.rms_norm_eps,
                    max_seq_len as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // ── 2. Attention over the contiguous cache (causal; decode = qlen 1). ──
        {
            let (q_ptr, _g0) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (kc_ptr, _g1) = k_cache.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _g2) = v_cache.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_prepped/caches/out valid on ctx.stream for the shapes above.
            unsafe {
                ffi::nonpaged_prefill_attention_cuda(
                    q_ptr as *const ffi::Half,
                    kc_ptr as *const ffi::Half,
                    vc_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    c.num_attention_heads as i32,
                    c.num_key_value_heads as i32,
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

        // ── 3. Per-head sigmoid gate from q_full's gate half. ──
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_full/attn_out valid on ctx.stream; gate layout per HD256 prep.
            unsafe {
                ffi::attention_gate_batch_hd256_cuda(
                    qf_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    c.num_attention_heads as i32,
                    seq_len as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        let mut out = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        gemm_batch(&self.ctx, &attn.o_proj, &attn_out, &mut out)?;
        Ok(out)
    }

    /// Gated-delta-rule linear attention: in-proj → depthwise conv1d → RECURRENT
    /// gated-delta (advances the per-slot state in place) → gated output RMSNorm
    /// → out-proj. The conv ring + recurrent state carry across prefill/decode.
    fn linear_attention(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        linear_idx: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let b_dim = attn.in_proj_b.rows;
        let a_dim = attn.in_proj_a.rows;

        let mut qkv = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        let mut z = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        let mut b_proj = HiddenStates::zeros(&self.ctx, b_dim, seq_len)?;
        let mut a_proj = HiddenStates::zeros(&self.ctx, a_dim, seq_len)?;
        gemm_batch(&self.ctx, &attn.in_proj_qkv, normed, &mut qkv)?;
        gemm_batch(&self.ctx, &attn.in_proj_z, normed, &mut z)?;
        gemm_batch(&self.ctx, &attn.in_proj_b, normed, &mut b_proj)?;
        gemm_batch(&self.ctx, &attn.in_proj_a, normed, &mut a_proj)?;

        // ── conv1d (advances the per-slot conv ring). ──
        let mut qkv_conv = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        let conv_state = &mut slot.conv_states[linear_idx];
        ensure!(
            conv_state.len == qkv_dim * (c.linear_conv_kernel_dim - 1),
            "Qwen3.5 conv state len {} != qkv_dim*(kernel-1) {}",
            conv_state.len,
            qkv_dim * (c.linear_conv_kernel_dim - 1)
        );
        {
            let (x_ptr, _g0) = qkv.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
            let (s_ptr, _g2) = conv_state.data.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g3) = qkv_conv.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: qkv/weight/state/out valid on ctx.stream; weight len checked
            // by the kernel against num_channels*kernel.
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
        }

        // ── gated-delta RECURRENT (never chunkwise: WGMMA short-seq path hangs
        //    on sm_90; the recurrent kernel handles seq_len==1 decode too). ──
        let mut gdr_out = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        let gdr_state = &mut slot.gdr_states[linear_idx];
        {
            let (qkv_ptr, _g0) = qkv_conv.data.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_proj.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_proj.data.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let (s_ptr, _g5) = gdr_state.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g6) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
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
                        c.linear_num_key_heads as i32,
                        c.linear_num_value_heads as i32,
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
                        c.linear_num_key_heads as i32,
                        c.linear_num_value_heads as i32,
                        c.linear_key_head_dim as i32,
                        c.linear_value_head_dim as i32,
                        seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }

        // ── gated output RMSNorm (per value head; gate = z). ──
        let mut normed_out = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        {
            let (x_ptr, _g0) = gdr_out.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.norm_weight.device_ptr(&self.ctx.stream);
            let (gate_ptr, _g2) = z.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = normed_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: gdr_out/norm/z/out valid on ctx.stream; per-head layout from config.
            unsafe {
                ffi::rms_norm_gated_cuda(
                    x_ptr as *const ffi::Half,
                    w_ptr as *const f32,
                    gate_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    c.linear_num_value_heads as i32,
                    c.linear_value_head_dim as i32,
                    c.rms_norm_eps,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        let mut out = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        gemm_batch(&self.ctx, &attn.out_proj, &normed_out, &mut out)?;
        Ok(out)
    }
}

/// Offset RMSNorm (1+weight) over a batch — Qwen3.5 norms store `weight - 1`.
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
