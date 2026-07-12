//! DSpark block drafter for DeepSeek-V4-Flash CUDA speculative decode (T4.1).
//!
//! The MLA-geometry analog of the shipped Qwen3.6 DSpark draft
//! (`crate::qwen35::dspark`). One draft step proposes a whole block
//! (`block_size` positions) from a single non-causal DSv4 transformer forward
//! conditioned on trunk "context features". The key difference from Qwen is MLA:
//! there is NO separate V — K and V are the SAME compressed latent
//! (`head_dim` = NoPE + RoPE). So where the Qwen draft caches separate
//! `k_ctx`/`v_ctx`, this caches ONE `latent_kv` per stage.
//!
//! THIS tranche = ONE stage's dual-stream forward: given the noise-block
//! hyper-connection `stream` + the fused `context` feature rows (provided by the
//! caller — the `main_proj` 3-tap fuse + the 3-stage stack are later tranches),
//! project q/latent, RoPE them, append context+block to the explicit `latent_kv`,
//! run the isolated dense MLA-latent attention (`ffi::dsv4_dspark_draft_attention_cuda`,
//! kernel body pod-stubbed), o-project, then MoE + hyper-connection exactly as
//! `Dsv4Model::mtp_forward_level`. Everything is `dead_code` until the stacking
//! tranche wires a caller.

use super::*;
use deepseek_spec::DeepSeekV4RopeParameters;

/// Per-slot draft context cache. For each draft stage, ONE `latent_kv` buffer
/// (K==V, MLA has no separate V) laid out `[position][head_dim]`, LINEAR from
/// `ctx_base` (`row = abs − ctx_base`). Sized `context_capacity + block_size`:
/// the accepted context plus one speculative noise block. Noise rows are written
/// speculatively at their positions and self-heal — rejected rows are overwritten
/// by the next block's context/noise writes before any read.
#[allow(dead_code)]
pub(crate) struct Dsv4DsparkSlotState {
    /// One compressed KV latent buffer per stage (`[cap * head_dim]` bf16).
    latent_kv: Vec<DeviceVec>,
    /// Absolute trunk position of `latent_kv` row 0.
    ctx_base: usize,
    /// Absolute end (exclusive) of materialized context rows; the buffer holds
    /// `ctx_end - ctx_base` context rows before the noise block is appended.
    ctx_end: usize,
    /// Last emitted token (the next block's anchor), staged by prefill/verify.
    pending: Option<u32>,
}

#[allow(dead_code)]
impl Dsv4DsparkSlotState {
    /// Allocate one latent cache per stage. `context_capacity` is the per-request
    /// context ceiling (accepted tokens the draft ever caches); `block_size` the
    /// speculative noise block. Call only under `config.is_dspark()`.
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        num_stages: usize,
        context_capacity: usize,
        block_size: usize,
    ) -> Result<Self> {
        let cap = context_capacity + block_size;
        let latent_kv = (0..num_stages)
            .map(|_| DeviceVec::zeros(ctx, cap * config.head_dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latent_kv,
            ctx_base: 0,
            ctx_end: 0,
            pending: None,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.rebase(0);
    }

    /// Re-key the (empty) latent buffer so row 0 is absolute position `pos`. No
    /// zeroing: stale rows are overwritten by post-rebase appends before any read.
    pub(crate) fn rebase(&mut self, pos: usize) {
        self.ctx_base = pos;
        self.ctx_end = pos;
        self.pending = None;
    }
}

/// Draft-side persistent scratch (exact-shape reuse; spec steps are serial). The
/// fixed block-shaped hot buffers are pooled here so the per-block forward does
/// not churn device allocations; the variable-length context buffers are
/// allocated inline (once per stage per step) in the append path.
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct Dsv4DsparkScratch {
    /// RoPE'd query for the noise block `[local_width, block]`.
    q_prepped: HsSlot,
    /// Attention output `[local_heads * nope_dim, block]` — the NoPE-latent value
    /// side fed to `mla_oproj`.
    attn_heads: HsSlot,
}

/// A single lazily (re)allocated `HiddenStates` slot — the DSv4 analog of the
/// Qwen executor's `HiddenSlot`. Reallocates only when the requested shape
/// changes (never, at fixed block/config), so steady state is one buffer reused.
#[derive(Default)]
struct HsSlot(Option<HiddenStates>);

impl HsSlot {
    fn get(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<&mut HiddenStates> {
        let stale = self
            .0
            .as_ref()
            .is_none_or(|h| h.hidden_dim != hidden_dim || h.seq_len != seq_len);
        if stale {
            // SAFETY: fully written before first read at every call site (RoPE /
            // attention kernels write the whole buffer).
            self.0 = Some(unsafe { HiddenStates::uninit(ctx, hidden_dim, seq_len)? });
        }
        Ok(self.0.as_mut().expect("just populated"))
    }
}

impl Dsv4Model {
    /// One DSpark stage's dual-stream forward. Appends the fused `context` rows
    /// and the noise `block` to the stage's explicit `latent_kv`, runs the dense
    /// non-causal MLA-latent attention over the whole `[context ++ block]` range,
    /// and returns the stage's hyper-connection output stream `[stream_dim, block]`
    /// (after MoE + HC), ready to feed the next stage.
    ///
    /// `stream_in` is the noise block's HC stream `[stream_dim, block]` (built by
    /// the fuse/stack tranche; passed in here). `context` is the fused feature
    /// rows `[hidden, ctx_rows]` at absolute positions `df.ctx_end..`. `block_tokens`
    /// are the block's anchor+draft ids (MoE hash routing; unused for the draft's
    /// `LearnedBias` layers). The attention math lives in the pod-stubbed FFI —
    /// this only wires shapes/pointers.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_stage_forward(
        &self,
        draft: &Dsv4DsparkDraft,
        stage_idx: usize,
        df: &mut Dsv4DsparkSlotState,
        scratch: &mut Dsv4DsparkScratch,
        attn_state: &mut crate::attention::Dsv4LayerAttentionState,
        stream_in: &HiddenStates,
        context: &HiddenStates,
        block_tokens: &[u32],
    ) -> Result<HiddenStates> {
        let ctx = &self.ctx;
        let config = &self.config;
        let eps = config.rms_norm_eps;
        let hidden_size = config.hidden_size;
        let hc_mult = config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let head_dim = config.head_dim;
        let rope_dim = config.qk_rope_head_dim;
        let nope_dim = head_dim - rope_dim;

        let stage = draft
            .stages
            .get(stage_idx)
            .ok_or_else(|| anyhow!("DSpark stage {stage_idx} out of range"))?;
        let layer = &stage.layer;
        let attention = &layer.attention;
        let block = stream_in.seq_len;
        let ctx_rows = context.seq_len;
        ensure!(
            stream_in.hidden_dim == stream_dim,
            "DSpark stage stream dim {} != {stream_dim}",
            stream_in.hidden_dim
        );
        ensure!(
            context.hidden_dim == hidden_size,
            "DSpark context dim {} != hidden {hidden_size}",
            context.hidden_dim
        );
        ensure!(
            block_tokens.len() == block,
            "DSpark block_tokens {} != block {block}",
            block_tokens.len()
        );
        let local_width = attention.wq_b.rows;
        ensure!(
            local_width.is_multiple_of(head_dim),
            "DSpark local width {local_width} not a multiple of head_dim {head_dim}"
        );
        let local_heads = local_width / head_dim;

        // The draft layer is compress_ratio == 0 (pure attention, no CSA/HCA), so
        // RoPE is plain rope_theta with no YaRN (matches the mla_attention cr==0
        // branch). block/ctx caches must fit the latent buffer.
        let rope = &config.rope_parameters;
        let ctx_start = df.ctx_end;
        let block_start = ctx_start + ctx_rows;
        let cap = df.latent_kv[stage_idx].len / head_dim;
        ensure!(
            (block_start + block) - df.ctx_base <= cap,
            "DSpark latent overflow: rows {} > cap {cap}",
            (block_start + block) - df.ctx_base
        );

        let mut keepalive = Dsv4ForwardKeepalive::new(false);

        // ── Append the fused context rows' latent (wkv → kv_norm → RoPE) into
        // latent_kv[ctx_start..ctx_start+ctx_rows]. Context has no query — drive a
        // discarded dummy q through the fused prep (mirrors the Qwen draft
        // dspark_append_ctx dummy-q pattern).
        if ctx_rows > 0 {
            self.dspark_append_latent(
                df,
                stage_idx,
                attention,
                context,
                ctx_start,
                local_width,
                local_heads,
                head_dim,
                rope_dim,
                rope,
                eps,
                &mut keepalive,
            )?;
        }
        // NOTE (codex T4.1 P1 ×2): do NOT advance the persistent `df.ctx_end`
        // here. It is SHARED across stages but `latent_kv` is per-stage — a
        // per-stage advance would push stage>0's ctx_start past its own
        // still-unwritten rows (stale reads), and advancing before the attention
        // below (currently the `NotYetImplemented` stub) commits state on the
        // error path. The stacking caller advances `ctx_end` to `block_start`
        // ONCE, after all stages of the block succeed.

        // ── Noise block. HC pre-norm the stream to the attention input.
        let attn_mhc = crate::hc::gen_mhc_params(ctx, config, &layer.hc_attn, stream_in)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, block)? };
        crate::hc::mhc_pre_rms_norm(
            ctx,
            stream_in,
            &attn_mhc.pre,
            &layer.attn_norm,
            eps,
            hidden_size,
            hc_mult,
            &mut attn_normed,
        )?;
        keepalive.keep_hidden(&attn_normed);

        // q: wq_a → q_norm → wq_b (pre-absorbed, w_kc None → local_heads*head_dim).
        // SAFETY: dsv4_linear writes the full buffers.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, block)? };
        crate::attention::dsv4_linear(ctx, &attention.wq_a, &attn_normed, &mut c_q)?;
        keepalive.keep_hidden(&c_q);
        let c_q_normed = crate::attention::mla_rms_norm(ctx, &c_q, &attention.q_norm, eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // SAFETY: dsv4_linear writes the full buffer.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, block)? };
        crate::attention::dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)?;
        keepalive.keep_hidden(&q_raw);

        // latent: wkv → kv_norm (the single compressed K==V latent).
        // SAFETY: dsv4_linear writes the full buffer.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, block)? };
        crate::attention::dsv4_linear(ctx, &attention.wkv, &attn_normed, &mut kv_raw)?;
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = crate::attention::mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, eps)?;
        keepalive.keep_hidden(&kv_normed);

        // Partial RoPE on q + latent at absolute positions block_start..+block; the
        // RoPE'd latent K writes directly into latent_kv[block range].
        let q_prepped = scratch.q_prepped.get(ctx, local_width, block)?;
        let positions: Vec<i32> = (block_start..block_start + block)
            .map(|p| p as i32)
            .collect();
        let pos_dev = ctx
            .stream
            .clone_htod(&positions)
            .map_err(|e| anyhow!("DSpark block positions H2D failed: {e}"))?;
        {
            let latent_off = ((block_start - df.ctx_base) * head_dim) as u64 * 2;
            let (q_ptr, _gq) = q_raw.data.device_ptr(&ctx.stream);
            let (k_ptr, _gk) = kv_normed.data.device_ptr(&ctx.stream);
            let (qo_ptr, _gqo) = q_prepped.data.device_ptr_mut(&ctx.stream);
            let (lat_ptr, _gl) = df.latent_kv[stage_idx].data.device_ptr_mut(&ctx.stream);
            let (sp_ptr, _gs) = pos_dev.device_ptr(&ctx.stream);
            // SAFETY: q/latent buffers sized above; k_out targets the contiguous
            // latent_kv block slice (row block_start-ctx_base .. +block < cap).
            unsafe {
                ffi::dsv4_prepare_qk_fused_batch_start_pos_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    qo_ptr as *mut ffi::Half,
                    (lat_ptr + latent_off) as *mut ffi::Half,
                    block as i32,
                    local_heads as i32,
                    head_dim as i32,
                    rope_dim as i32,
                    sp_ptr as *const i32,
                    eps,
                    config.rope_theta,
                    0,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // ── Dense non-causal MLA-latent attention over [context ++ block]. Every
        // block row attends the whole kv_len latent range; K==V head-shared latent.
        let kv_len = (block_start + block) - df.ctx_base;
        let attn_heads = scratch.attn_heads.get(ctx, local_heads * nope_dim, block)?;
        {
            let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
            let (q_ptr, _gq) = q_prepped.data.device_ptr(&ctx.stream);
            let (lat_ptr, _gl) = df.latent_kv[stage_idx].data.device_ptr(&ctx.stream);
            let (o_ptr, _go) = attn_heads.data.device_ptr_mut(&ctx.stream);
            // SAFETY: q [block,local_heads,head_dim]; latent_kv holds kv_len rows
            // from ctx_base; out [block,local_heads,nope_dim]. Kernel is pod-stubbed.
            unsafe {
                ffi::dsv4_dspark_draft_attention_cuda(
                    q_ptr as *const ffi::Half,
                    lat_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    kv_len as i32,
                    block as i32,
                    local_heads as i32,
                    head_dim as i32,
                    nope_dim as i32,
                    rope_dim as i32,
                    sm_scale,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // ── O-projection (wo_a → wo_b) back to hidden, then TP all-reduce.
        // SAFETY: uninit device scratch; fully written by mla_oproj.
        let mut attn_out = unsafe { HiddenStates::uninit(ctx, hidden_size, block)? };
        {
            let attn_heads = scratch.attn_heads.get(ctx, local_heads * nope_dim, block)?;
            crate::attention::mla_oproj(
                ctx,
                attention,
                attn_state,
                None,
                attn_heads,
                block,
                &mut keepalive,
                &mut attn_out,
            )?;
        }
        self.tp.all_reduce_sum(ctx, &mut attn_out)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, block)? };
        crate::hc::hc_post(
            ctx,
            &attn_out,
            stream_in,
            &attn_mhc.post,
            &attn_mhc.comb,
            hidden_size,
            hc_mult,
            &mut attn_stream,
        )?;
        keepalive.keep_hidden(&attn_out);
        keepalive.keep_hidden(&attn_stream);

        // ── FFN (MoE + shared expert) hyper-connection, identical to mtp_forward_level.
        let ffn_mhc = crate::hc::gen_mhc_params(ctx, config, &layer.hc_ffn, &attn_stream)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut ffn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, block)? };
        crate::hc::mhc_pre_rms_norm(
            ctx,
            &attn_stream,
            &ffn_mhc.pre,
            &layer.ffn_norm,
            eps,
            hidden_size,
            hc_mult,
            &mut ffn_normed,
        )?;
        keepalive.keep_hidden(&ffn_normed);
        let moe = layer.moe.as_ref().expect("DSpark draft layer.moe");
        // SAFETY: uninit device scratch; fully written before first read.
        let mut moe_out = unsafe { HiddenStates::uninit(ctx, hidden_size, block)? };
        crate::moe::dsv4_moe_forward(
            self,
            moe,
            block_tokens,
            &ffn_normed,
            &mut moe_out,
            &mut keepalive,
            None,
        )?;
        self.tp.all_reduce_sum(ctx, &mut moe_out)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut shared = unsafe { HiddenStates::uninit(ctx, hidden_size, block)? };
        crate::moe::dsv4_shared_expert_forward(
            ctx,
            &ctx.stream,
            moe,
            &ffn_normed,
            &mut shared,
            config.swiglu_limit,
            &mut keepalive,
        )?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut moe_with_shared = unsafe { HiddenStates::uninit(ctx, hidden_size, block)? };
        crate::ops::add_batch(ctx, &moe_out, &shared, &mut moe_with_shared)?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut ffn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, block)? };
        crate::hc::hc_post(
            ctx,
            &moe_with_shared,
            &attn_stream,
            &ffn_mhc.post,
            &ffn_mhc.comb,
            hidden_size,
            hc_mult,
            &mut ffn_stream,
        )?;
        keepalive.keep_hidden(&moe_out);
        keepalive.keep_hidden(&shared);
        keepalive.keep_hidden(&moe_with_shared);
        std::hint::black_box(keepalive.len());
        Ok(ffn_stream)
    }

    /// Compute the compressed latent for `feats` (already hidden-wide fused
    /// features) via `wkv → kv_norm → partial-RoPE` and write it into
    /// `latent_kv[start..start+rows]` at absolute positions `start..`. Context
    /// rows carry no query, so a discarded dummy q is driven through the fused
    /// prep kernel (which requires q heads).
    #[allow(clippy::too_many_arguments)]
    fn dspark_append_latent(
        &self,
        df: &mut Dsv4DsparkSlotState,
        stage_idx: usize,
        attention: &Dsv4Attention,
        feats: &HiddenStates,
        start: usize,
        local_width: usize,
        local_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        rope: &DeepSeekV4RopeParameters,
        eps: f32,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let rows = feats.seq_len;
        // SAFETY: dsv4_linear writes the full buffer.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, rows)? };
        crate::attention::dsv4_linear(ctx, &attention.wkv, feats, &mut kv_raw)?;
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = crate::attention::mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, eps)?;
        keepalive.keep_hidden(&kv_normed);
        // Dummy q (discarded q_out) — the fused prep requires q heads. Zeroed,
        // not uninit: the kernel RoPE's this buffer in place, and reading
        // uninitialized device memory is UB / can surface NaN (codex T4.1 P2).
        // RoPE of zeros is zeros; the result is discarded anyway.
        let mut q_dummy = HiddenStates::zeros(ctx, local_width, rows)?;
        keepalive.keep_hidden(&q_dummy);
        let positions: Vec<i32> = (start..start + rows).map(|p| p as i32).collect();
        let pos_dev = ctx
            .stream
            .clone_htod(&positions)
            .map_err(|e| anyhow!("DSpark ctx positions H2D failed: {e}"))?;
        let latent_off = ((start - df.ctx_base) * head_dim) as u64 * 2;
        let (k_ptr, _gk) = kv_normed.data.device_ptr(&ctx.stream);
        // q_raw == q_out == the dummy buffer (in-place, discarded).
        let (qo_ptr, _gqo) = q_dummy.data.device_ptr_mut(&ctx.stream);
        let (lat_ptr, _gl) = df.latent_kv[stage_idx].data.device_ptr_mut(&ctx.stream);
        let (sp_ptr, _gs) = pos_dev.device_ptr(&ctx.stream);
        // SAFETY: k_out targets latent_kv[start-ctx_base .. +rows] (< cap); q_out
        // reuses the discarded dummy buffer (in-place RoPE on garbage q).
        unsafe {
            ffi::dsv4_prepare_qk_fused_batch_start_pos_cuda(
                qo_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                qo_ptr as *mut ffi::Half,
                (lat_ptr + latent_off) as *mut ffi::Half,
                rows as i32,
                local_heads as i32,
                head_dim as i32,
                rope_dim as i32,
                sp_ptr as *const i32,
                eps,
                self.config.rope_theta,
                0,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }
}
