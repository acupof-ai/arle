//! DSpark block drafter for DeepSeek-V4-Flash CUDA speculative decode.
//!
//! MLA-geometry analog of the shipped Qwen3.6 DSpark draft
//! (`crate::qwen35::dspark`). One step proposes a whole block (`block_size`
//! positions) from a single non-causal DSv4 forward conditioned on trunk context
//! features. Unlike Qwen, MLA has NO separate V — K and V are the SAME compressed
//! latent (`head_dim` = NoPE + RoPE), so this caches ONE `latent_kv` per stage,
//! not separate `k_ctx`/`v_ctx`.
//!
//! Stack: [`Dsv4Model::dspark_forward_block`] embeds the noise block, fuses the
//! 3-tap trunk context, chains the 3 stages' dual-stream forwards
//! ([`Dsv4Model::dspark_stage_forward`] — q/latent project + RoPE, append to
//! `latent_kv`, dense MLA-latent attention, o-proj, MoE + hyper-connection as
//! `mtp_forward_level`), then folds the exit head.
//! [`Dsv4Model::dspark_build_proposal`] turns the block hidden into a draft chain
//! via tied-head logits → Markov semi-AR sampling → confidence truncation.
//! `dead_code` until the verify tranche wires a caller.

use super::*;
use deepseek_spec::DeepSeekV4RopeParameters;

/// Per-slot draft context cache. Per stage, ONE `latent_kv` buffer (K==V) laid
/// out `[position][head_dim]`, LINEAR from `ctx_base` (`row = abs − ctx_base`).
/// Sized `context_capacity + block_size` (accepted context + one speculative
/// noise block). Noise rows self-heal: rejected rows are overwritten by the next
/// block's context/noise writes before any read.
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
    /// Allocate one latent cache per stage. `context_capacity` = per-request
    /// context ceiling (accepted tokens the draft ever caches); `block_size` =
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

/// Draft-side persistent scratch (exact-shape reuse; spec steps are serial).
/// Fixed block-shaped hot buffers pool here so the per-block forward does not
/// churn allocations; variable-length context buffers are allocated inline (once
/// per stage per step) in the append path.
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct Dsv4DsparkScratch {
    /// RoPE'd query for the noise block `[local_width, block]`.
    q_prepped: HsSlot,
    /// Attention output `[local_heads * head_dim, block]` — the full-latent value
    /// side fed to `mla_oproj` (Flag #1: `local_heads*head_dim == local_width`,
    /// matching the main model's `local_attn`).
    attn_heads: HsSlot,
}

/// A single lazily (re)allocated `HiddenStates` slot — DSv4 analog of the Qwen
/// executor's `HiddenSlot`. Reallocates only on a shape change (never, at fixed
/// block/config), so steady state is one reused buffer.
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
    /// and the noise `block` to the stage's `latent_kv`, runs dense non-causal
    /// MLA-latent attention over the whole `[context ++ block]` range, and returns
    /// the stage's HC output stream `[stream_dim, block]` (after MoE + HC), ready
    /// to feed the next stage.
    ///
    /// `stream_in` = the noise block's HC stream `[stream_dim, block]`. `context`
    /// = fused feature rows `[hidden, ctx_rows]` at absolute positions
    /// `df.ctx_end..`. `block_tokens` = anchor+draft ids (MoE hash routing; unused
    /// for the draft's `LearnedBias` layers). Attention math lives in the
    /// pod-stubbed FFI — this only wires shapes/pointers.
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

        // Draft layer is compress_ratio == 0 (pure attention, no CSA/HCA), so RoPE
        // is plain rope_theta with no YaRN (matches mla_attention's cr==0 branch).
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

        // Append fused context latent to latent_kv[ctx_start..+ctx_rows].
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
        // codex P1 ×2: do NOT advance the SHARED `df.ctx_end` here. `latent_kv` is
        // per-stage, so a per-stage advance would push stage>0's ctx_start past its
        // own still-unwritten rows (stale reads); advancing before the attention
        // below also commits state on the error path. Caller advances to
        // `block_start` ONCE, after all stages succeed.

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
        // SAFETY: dsv4_linear writes the full buffer.
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
        // RoPE'd latent K writes directly into latent_kv's block range.
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

        // ── Dense non-causal MLA-latent attention over [context ++ block]: every
        // block row attends the whole kv_len range; K==V head-shared latent.
        let kv_len = (block_start + block) - df.ctx_base;
        let attn_heads = scratch.attn_heads.get(ctx, local_heads * head_dim, block)?;
        {
            let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
            let (q_ptr, _gq) = q_prepped.data.device_ptr(&ctx.stream);
            let (lat_ptr, _gl) = df.latent_kv[stage_idx].data.device_ptr(&ctx.stream);
            let (o_ptr, _go) = attn_heads.data.device_ptr_mut(&ctx.stream);
            // SAFETY: q [block,local_heads,head_dim]; latent_kv holds kv_len rows
            // from ctx_base; out [block,local_heads,head_dim] (Flag #1).
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
            let attn_heads = scratch.attn_heads.get(ctx, local_heads * head_dim, block)?;
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

        // ── FFN (MoE + shared expert) HC, identical to mtp_forward_level.
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

    /// The 3-stage DSpark backbone (`mtp.0 → mtp.1 → mtp.2`). Builds the noise
    /// block's HC stream + the fused `context` ONCE, chains every stage's
    /// dual-stream forward ([`Dsv4Model::dspark_stage_forward`]), then advances the
    /// shared `df.ctx_end` ONE time and folds the exit head. Returns `block_hidden`
    /// `[hidden, block]` (base-embed logits are the next tranche).
    ///
    /// `attn_states` = one caller-provisioned [`Dsv4LayerAttentionState`] per stage
    /// (no per-block alloc — spec steps are serial). `taps` = the 3 wide HC-stream
    /// taps from `Dsv4SlotState::dspark_taps()` (trunk layers `[40,41,42]`);
    /// `anchor` = the last accepted token (noise block position 0).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_forward_block(
        &self,
        draft: &Dsv4DsparkDraft,
        df: &mut Dsv4DsparkSlotState,
        scratch: &mut Dsv4DsparkScratch,
        attn_states: &mut [crate::attention::Dsv4LayerAttentionState],
        taps: &[DeviceVec],
        anchor: u32,
    ) -> Result<HiddenStates> {
        let ctx = &self.ctx;
        let config = &self.config;
        let eps = config.rms_norm_eps;
        let hidden = config.hidden_size;
        let hc_mult = config.hc_mult;
        let stream_dim = hidden * hc_mult;
        let block = config.dspark_block_size;
        let num_stages = config.dspark_num_stages();
        ensure!(
            num_stages > 0,
            "DSpark backbone requires >= 1 stage (dspark_num_stages == 0)"
        );
        ensure!(block > 0, "DSpark backbone called on a non-DSpark config");
        ensure!(
            draft.stages.len() == num_stages && attn_states.len() == num_stages,
            "DSpark backbone stages {} / attn_states {} != num_stages {num_stages}",
            draft.stages.len(),
            attn_states.len()
        );

        // ── Noise embedding: pos0 = anchor, pos1.. = mask token; embed via the
        // base table (draft has no own embed), then HC-expand hidden → stream.
        let mut ids = vec![config.dspark_noise_token_id as i32; block];
        ids[0] = anchor as i32;
        let ids_dev = crate::ops::upload_i32(ctx, &ids)?;
        // SAFETY: embedding_batch writes the full [hidden, block] buffer.
        let mut emb = unsafe { HiddenStates::uninit(ctx, hidden, block)? };
        crate::ops::embedding_batch(ctx, &self.embed_tokens, &ids_dev, &mut emb)?;
        // SAFETY: initial_stream_from_embeddings writes the full stream buffer.
        let mut hidden_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, block)? };
        crate::hc::initial_stream_from_embeddings(ctx, &emb, hidden, hc_mult, &mut hidden_stream)?;

        // ── Context fuse (main_proj @ stage 0). POD-VERIFY tap layout: each tap is
        // the WIDE HC stream `[hc_mult, hidden]` for ONE trunk position (row r at
        // `r*hidden`). main_proj `[hidden, n_taps*hidden]` takes `n_taps*hidden` PER
        // hc_mult row (NOT `n_taps*stream_dim`): per row concat the taps'
        // hidden-slices → `[n_taps*hidden, hc_mult]`; main_proj → `[hidden,
        // hc_mult]`; main_norm → context.
        let stage0 = &draft.stages[0];
        let main_proj = stage0
            .main_proj
            .as_ref()
            .ok_or_else(|| anyhow!("DSpark stage 0 missing main_proj"))?;
        let main_norm = stage0
            .main_norm
            .as_ref()
            .ok_or_else(|| anyhow!("DSpark stage 0 missing main_norm"))?;
        let n_taps = taps.len();
        // POD-VERIFY preconditions — the tap stride/interleave and main_proj
        // input width are not locally verifiable (attention kernel is stubbed).
        ensure!(
            main_proj.cols == n_taps * hidden,
            "DSpark main_proj cols {} != n_taps {n_taps} * hidden {hidden}",
            main_proj.cols
        );
        ensure!(
            main_proj.rows == hidden,
            "DSpark main_proj rows {} != hidden {hidden}",
            main_proj.rows
        );
        for (t, tap) in taps.iter().enumerate() {
            ensure!(
                tap.len == hidden * hc_mult,
                "DSpark tap {t} len {} != hidden {hidden} * hc_mult {hc_mult}",
                tap.len
            );
        }
        // SAFETY: uninit device scratch; every [row r, tap t] slice written below.
        let mut fuse_in = unsafe { HiddenStates::uninit(ctx, n_taps * hidden, hc_mult)? };
        for r in 0..hc_mult {
            for (t, tap) in taps.iter().enumerate() {
                let src = tap.data.slice(r * hidden..(r + 1) * hidden);
                let base = r * n_taps * hidden + t * hidden;
                let mut dst = fuse_in.data.slice_mut(base..base + hidden);
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("DSpark tap concat D2D failed: {e}"))?;
            }
        }
        // SAFETY: dsv4_linear writes the full [hidden, hc_mult] buffer.
        let mut context_raw = unsafe { HiddenStates::uninit(ctx, hidden, hc_mult)? };
        crate::attention::dsv4_linear(ctx, main_proj, &fuse_in, &mut context_raw)?;
        let context = crate::attention::mla_rms_norm(ctx, &context_raw, main_norm, eps)?;

        // Draft MoE routing is LearnedBias, so these hash ids are unused — but the
        // stage forward's shape gate requires `block_tokens.len() == block`.
        let block_tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();

        // ── Chain the stages. `df.ctx_end` is the SHARED cursor; every stage
        // appends `context` to its OWN `latent_kv` at the same `ctx_start` and must
        // NOT advance it — the single advance happens AFTER all stages succeed.
        let ctx_start = df.ctx_end;
        let block_start = ctx_start + context.seq_len;
        for (stage_idx, attn_state) in attn_states.iter_mut().enumerate() {
            hidden_stream = self.dspark_stage_forward(
                draft,
                stage_idx,
                df,
                scratch,
                attn_state,
                &hidden_stream,
                &context,
                &block_tokens,
            )?;
        }

        // ── Exit: block_hidden = mtp.{n-1}.norm(mtp.{n-1}.hc_head(stream)). DSv4
        // folds the wide stream through the stage's head HC (vs qwen3's bare norm)
        // then RMSNorm — mirrors `head_normed_rows`, stage-scoped.
        let exit = &draft.stages[num_stages - 1];
        let hc_head = exit
            .hc_head
            .as_ref()
            .ok_or_else(|| anyhow!("DSpark exit stage missing hc_head"))?;
        let norm = exit
            .norm
            .as_ref()
            .ok_or_else(|| anyhow!("DSpark exit stage missing norm"))?;
        // SAFETY: uninit device scratch; every row written before first read.
        let mut block_hidden = unsafe { HiddenStates::uninit(ctx, hidden, block)? };
        let mut row_hidden = DeviceVec::zeros(ctx, hidden)?;
        let mut row_normed = DeviceVec::zeros(ctx, hidden)?;
        for row in 0..block {
            crate::hc::head_hidden_from_stream(
                ctx,
                config,
                hc_head,
                &hidden_stream,
                row,
                &mut row_hidden,
            )?;
            crate::ops::rms_norm_vec(ctx, &row_hidden, norm, eps, &mut row_normed)?;
            let mut dst = block_hidden
                .data
                .slice_mut(row * hidden..(row + 1) * hidden);
            ctx.stream
                .memcpy_dtod(&row_normed.data, &mut dst)
                .map_err(|e| anyhow!("DSpark exit head row copy failed: {e}"))?;
        }
        // Advance the shared context cursor ONCE, only after the whole block (all
        // stages AND the exit head) succeeds — a mid-forward advance would commit
        // persistent state on an error path (codex T4.2 P2).
        df.ctx_end = block_start;
        Ok(block_hidden)
    }

    /// Compute the compressed latent for hidden-wide `feats` via
    /// `wkv → kv_norm → partial-RoPE`, writing it into
    /// `latent_kv[start..start+rows]` at absolute positions `start..`. Context rows
    /// carry no query, so a discarded dummy q is driven through the fused prep
    /// kernel (which requires q heads).
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
        // Dummy q (discarded q_out) — the fused prep requires q heads. Zeroed, not
        // uninit: the kernel RoPE's this buffer in place, and reading uninitialized
        // device memory is UB / can surface NaN (codex T4.1 P2). RoPE of zeros is
        // zeros; the result is discarded anyway.
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

// ── Markov semi-AR sampling + confidence truncation ──────────────────────────
//
// TODO(P2): dedup with qwen35::dspark into a shared target-independent
// dspark_heads module. Kept separate to protect the shipped Qwen3.6 B track —
// DO NOT touch `qwen35/dspark.rs` for this.
//
// The draft head is DeepSpec's `VanillaMarkov` (hidden-INDEPENDENT: exit stage
// `mtp.2` carries `markov_w1`/`markov_w2` but no `gate_proj`/`joint_proj`, so the
// step bias is `markov_w2 · markov_w1[prev]`, ignoring the hidden). Confidence
// head = `AcceptRatePredictor` (`proj = Linear(in_dim, 1)`, no bias in this
// checkpoint) — `sigmoid(logit) < threshold` gives the dynamic draft length.

/// Proposal from one DSpark block draft. `chain[0]` = anchor, `chain[1..]` = the
/// drafted tokens surviving confidence truncation (`draft_len`). Consumed by the
/// verify path: `chain` feeds [`Dsv4Model::forward_tokens_verify`] as `tokens`;
/// `draft_logits` = the corrected per-step logits (base + Markov bias) retained
/// for the sampled-mode accept test's draft distribution `q`.
#[allow(dead_code)]
pub(crate) struct Dsv4DsparkProposal {
    /// Verify input ids `[anchor, d0 .. d_{L-1}]` (len == `draft_len + 1`).
    pub chain: Vec<u32>,
    /// Number of drafted tokens `L` surviving confidence truncation.
    pub draft_len: usize,
    /// Corrected per-step logits `[vocab, draft_len]` for the accepted prefix.
    /// `None` when the block truncated to zero drafts.
    pub draft_logits: Option<HiddenStates>,
}

impl Dsv4Model {
    /// Turn the backbone's `block_hidden` `[hidden, block]` (from
    /// [`Dsv4Model::dspark_forward_block`]) into a proposed draft-token block:
    /// tied-head base logits → Markov semi-AR L→R sampling → confidence truncation.
    /// `dead_code` until the verify wiring calls it.
    #[allow(dead_code)]
    pub(crate) fn dspark_build_proposal(
        &self,
        draft: &Dsv4DsparkDraft,
        block_hidden: &HiddenStates,
        anchor: u32,
        temperature: f32,
        conf_threshold: f32,
    ) -> Result<Dsv4DsparkProposal> {
        let ctx = &self.ctx;
        let hidden = self.config.hidden_size;
        let block = self.config.dspark_block_size;
        let vocab = self.lm_head.rows;
        ensure!(block > 0, "dspark_build_proposal on a non-DSpark config");
        ensure!(
            block_hidden.hidden_dim == hidden && block_hidden.seq_len == block,
            "DSpark block_hidden [{}, {}] != [hidden {hidden}, block {block}]",
            block_hidden.hidden_dim,
            block_hidden.seq_len
        );
        let exit = draft
            .stages
            .last()
            .ok_or_else(|| anyhow!("DSpark draft has no stages"))?;

        // Base logits via the tied head (no `lm_head` on the draft — it shares the
        // target's output projection over the exit-normed `block_hidden`).
        // SAFETY: uninit scratch; `lm_head_project_batch` writes every row.
        let mut base_logits = unsafe { HiddenStates::uninit(ctx, vocab, block)? };
        self.lm_head_project_batch(block_hidden, &mut base_logits)?;

        // Markov semi-AR L→R: step bias = markov_w2 · markov_w1[prev], added to the
        // base row; sample via the shared DSv4 sampler; feed the sampled token as
        // the next step's prev. First prev = anchor.
        let markov = exit.markov_w1.as_ref().zip(exit.markov_w2.as_ref());
        let params = SamplingParams {
            temperature,
            ..Default::default()
        };
        // Retained corrected logits (base + bias) for every step, for draft_probs.
        // SAFETY: uninit scratch; every row written in the loop below.
        let mut corrected = unsafe { HiddenStates::uninit(ctx, vocab, block)? };
        let mut base_row = HiddenStates::zeros(ctx, vocab, 1)?;
        let mut sum_row = HiddenStates::zeros(ctx, vocab, 1)?;
        let mut step_vec = DeviceVec::zeros(ctx, vocab)?;
        // Markov bias buffers (allocated once when the head is present).
        let mut markov_bufs = markov
            .map(|(w1, _)| -> Result<_> {
                Ok((
                    HiddenStates::zeros(ctx, w1.cols, 1)?, // emb [rank, 1]
                    HiddenStates::zeros(ctx, vocab, 1)?,   // bias [vocab, 1]
                ))
            })
            .transpose()?;

        let mut drafts = Vec::with_capacity(block);
        let mut prev = anchor;
        for i in 0..block {
            // base_row = base_logits[:, i].
            let src = base_logits.data.slice(i * vocab..(i + 1) * vocab);
            ctx.stream
                .memcpy_dtod(&src, &mut base_row.data)
                .map_err(|e| anyhow!("DSpark base row copy failed: {e}"))?;

            // corrected_row = base_row (+ Markov bias when the head is present).
            let corrected_row =
                if let (Some((w1, w2)), Some((emb, bias))) = (markov, markov_bufs.as_mut()) {
                    let tok = crate::ops::upload_i32(ctx, &[prev as i32])?;
                    crate::ops::embedding_batch(ctx, w1, &tok, emb)?; // markov_w1[prev]
                    crate::ops::gemm_batch(ctx, w2, emb, bias)?; // markov_w2 · emb
                    crate::ops::add_batch(ctx, &base_row, bias, &mut sum_row)?;
                    &sum_row
                } else {
                    &base_row
                };

            // Retain the corrected row + stage it for the sampler.
            let dst = i * vocab..(i + 1) * vocab;
            ctx.stream
                .memcpy_dtod(&corrected_row.data, &mut corrected.data.slice_mut(dst))
                .map_err(|e| anyhow!("DSpark corrected retain copy failed: {e}"))?;
            ctx.stream
                .memcpy_dtod(&corrected_row.data, &mut step_vec.data)
                .map_err(|e| anyhow!("DSpark step row copy failed: {e}"))?;

            let tok = crate::executor::sample_cuda_token(ctx, &step_vec, &params, i as u64)?;
            drafts.push(tok);
            prev = tok;
        }

        // Confidence truncation → dynamic draft length. prev_tokens =
        // [anchor, d0 .. d_{L-2}] (the token preceding each draft position).
        let prev_tokens: Vec<u32> = std::iter::once(anchor)
            .chain(drafts.iter().copied())
            .take(drafts.len())
            .collect();
        let keep =
            self.dspark_confident_prefix_len(exit, block_hidden, conf_threshold, &prev_tokens)?;
        let draft_len = keep.min(drafts.len());
        drafts.truncate(draft_len);

        let draft_logits = if draft_len == 0 {
            None
        } else {
            // SAFETY: uninit scratch; the leading `draft_len` rows are copied.
            let mut out = unsafe { HiddenStates::uninit(ctx, vocab, draft_len)? };
            let src = corrected.data.slice(0..draft_len * vocab);
            ctx.stream
                .memcpy_dtod(&src, &mut out.data)
                .map_err(|e| anyhow!("DSpark draft logits copy failed: {e}"))?;
            Some(out)
        };

        let mut chain = Vec::with_capacity(1 + draft_len);
        chain.push(anchor);
        chain.extend(drafts);
        Ok(Dsv4DsparkProposal {
            chain,
            draft_len,
            draft_logits,
        })
    }

    /// Confidence-head seam: acceptance-confident prefix length over the block
    /// rows — the first `i` with `sigmoid(proj(feat_i)) < threshold` truncates the
    /// proposal there. `feat_i = block_hidden[:, i]` for the plain head, or
    /// `[block_hidden[:, i] ; markov_w1[prev_i]]` for the with-markov head (keyed
    /// on the loaded `proj` input width: `hidden` vs `hidden + rank`). Head absent
    /// or `threshold <= 0` → the full block survives (`usize::MAX`).
    fn dspark_confident_prefix_len(
        &self,
        exit: &Dsv4DsparkStage,
        block_hidden: &HiddenStates,
        threshold: f32,
        prev_tokens: &[u32],
    ) -> Result<usize> {
        let Some(conf) = &exit.confidence_proj else {
            return Ok(usize::MAX);
        };
        if threshold <= 0.0 {
            return Ok(usize::MAX);
        }
        let ctx = &self.ctx;
        let hidden = self.config.hidden_size;
        let block = self.config.dspark_block_size;
        let in_dim = conf.cols;
        ensure!(conf.rows == 1, "confidence proj rows {} != 1", conf.rows);
        let with_markov = in_dim > hidden;
        if with_markov {
            let rank =
                exit.markov_w1.as_ref().map(|m| m.cols).ok_or_else(|| {
                    anyhow!("confidence head expects a markov head (in > hidden)")
                })?;
            ensure!(
                in_dim == hidden + rank,
                "confidence in {in_dim} != hidden {hidden} + markov rank {rank}"
            );
        } else {
            ensure!(
                in_dim == hidden,
                "confidence in {in_dim} != hidden {hidden}"
            );
        }

        let mut feat = DeviceVec::zeros(ctx, in_dim)?;
        let mut out = DeviceVec::zeros(ctx, 1)?;
        for (i, &prev) in prev_tokens.iter().enumerate().take(block) {
            // feat[..hidden] = block_hidden[:, i].
            let src = block_hidden.data.slice(i * hidden..(i + 1) * hidden);
            ctx.stream
                .memcpy_dtod(&src, &mut feat.data.slice_mut(0..hidden))
                .map_err(|e| anyhow!("DSpark conf hidden copy failed: {e}"))?;
            if with_markov {
                let w1 = exit.markov_w1.as_ref().expect("validated above");
                let rank = w1.cols;
                let tok = crate::ops::upload_i32(ctx, &[prev as i32])?;
                let mut emb = HiddenStates::zeros(ctx, rank, 1)?;
                crate::ops::embedding_batch(ctx, w1, &tok, &mut emb)?;
                ctx.stream
                    .memcpy_dtod(&emb.data, &mut feat.data.slice_mut(hidden..hidden + rank))
                    .map_err(|e| anyhow!("DSpark conf markov copy failed: {e}"))?;
            }
            // No bias tensor in this checkpoint — the raw proj output is the logit.
            crate::ops::gemv(ctx, conf, &feat, &mut out)?;
            let logit = out.to_host(ctx)?[0];
            if 1.0 / (1.0 + (-logit).exp()) < threshold {
                return Ok(i);
            }
        }
        Ok(usize::MAX)
    }
}
