//! DSpark block drafter for Qwen3.6 CUDA speculative decode.
//!
//! One draft step proposes a whole block (`block_size` positions) in a single
//! 5-layer non-causal forward conditioned on trunk "context features": per
//! accepted trunk token the residual stream is tapped at
//! `target_layer_ids`, projected through `fc` (+ `hidden_norm`) to one 5120-d
//! feature, and cached as the draft's per-layer K/V context. Covers both
//! checkpoint flavors (see [`qwen35_spec::DsparkConfig`]): z-lab DFlash
//! (backbone only, same-position denoising) and DeepSpec DSpark (next-token
//! labels + optional Markov / confidence heads). The trunk verify + rollback
//! substrate ([`Qwen35SpecSlotState`], `replay_linear_only`) is reused
//! unchanged — this module only swaps the draft-token source.

use super::*;

use crate::ops::rms_norm_batch;
use qwen35_spec::{DsparkConfig, DsparkLayerType, dspark_tensor_names};

struct DsparkLayer {
    /// Gate-padded `[2*q_dim, hidden]`: head `h` occupies rows
    /// `2h*head_dim..(2h+1)*head_dim`, odd bands zero — so the trunk's fused
    /// prep kernel (which assumes the gated q layout) reads the real q values
    /// and the unused gate half stays untouched.
    q_proj: DeviceMatrix,
    k_proj: DeviceMatrix,
    v_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    /// Stored `w - 1` (the prep kernel applies the trunk's offset `(1+w)` norm).
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    input_layernorm: DeviceVec,
    post_attention_layernorm: DeviceVec,
    mlp: DenseMlp,
    sliding: bool,
}

struct DsparkMarkovHead {
    /// `[vocab, rank]` embedding table (`markov_w1`).
    w1: DeviceMatrix,
    /// `[vocab, rank]` bias projection (`markov_w2`).
    w2: DeviceMatrix,
    rank: usize,
}

struct DsparkConfidenceHead {
    /// `[1, hidden (+ rank)]` acceptance-logit projection.
    weight: DeviceMatrix,
    bias: f32,
    with_markov: bool,
}

pub(crate) struct Qwen35DsparkHead {
    pub(crate) cfg: DsparkConfig,
    /// `fc` split input-wise into one `[hidden, hidden]` matrix per tap, so the
    /// concat becomes a sum of per-tap GEMMs (no per-token gather).
    fc: Vec<DeviceMatrix>,
    hidden_norm: DeviceVec,
    norm: DeviceVec,
    layers: Vec<DsparkLayer>,
    markov: Option<DsparkMarkovHead>,
    confidence: Option<DsparkConfidenceHead>,
    /// sigmoid(conf) < threshold truncates the proposal (`ARLE_DSPARK_CONF_THRESHOLD`).
    confidence_threshold: f32,
    /// Draft RoPE tables (full rotary over head_dim, `rope_theta`), `cap` positions.
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Per-layer ctx-cache rows: `max_seq_len + block_size` (noise rows extend
    /// past the trunk cap by at most one block).
    cap: usize,
}

impl Qwen35DsparkHead {
    fn q_dim(&self) -> usize {
        self.cfg.num_attention_heads * self.cfg.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.cfg.num_key_value_heads * self.cfg.head_dim
    }
    pub(crate) fn block_size(&self) -> usize {
        self.cfg.block_size
    }
    pub(crate) fn target_layer_ids(&self) -> &[i64] {
        &self.cfg.target_layer_ids
    }
    pub(crate) fn mode_label(&self) -> &'static str {
        match (
            self.cfg.next_token_heads,
            self.markov.is_some(),
            self.confidence.is_some(),
        ) {
            (false, _, _) => "dflash-backbone",
            (true, false, false) => "dspark-backbone",
            (true, true, false) => "dspark+markov",
            (true, false, true) => "dspark+confidence",
            (true, true, true) => "dspark+markov+confidence",
        }
    }
}

/// Per-slot draft context K/V cache: for each draft layer, the RoPE'd/normed K
/// and raw V of every accepted position, laid out `[kv_head][cap][head_dim]`
/// (the trunk contiguous-cache layout, so the existing prep + attention
/// kernels index it directly). Noise rows are written speculatively at their
/// absolute positions and self-heal: rejected rows are overwritten by the next
/// block's ctx/noise writes at those positions before any read.
pub(crate) struct Qwen35DsparkSlotState {
    k_ctx: Vec<DeviceVec>,
    v_ctx: Vec<DeviceVec>,
    /// Accepted ctx rows materialized as `[0, ctx_len)`. The spec step requires
    /// `ctx_len == start_pos`; a slot restored without its draft cache (sidecar
    /// / whole-slot promote) decodes non-speculatively until the next fresh
    /// prefill rebuilds it.
    pub(crate) ctx_len: usize,
    /// Last emitted token (the next block's anchor), staged by prefill/verify.
    pub(crate) pending: Option<u32>,
}

impl Qwen35DsparkSlotState {
    pub(crate) fn new(ctx: &DeviceContext, head: &Qwen35DsparkHead) -> Result<Self> {
        let per_layer = head.cfg.num_key_value_heads * head.cap * head.cfg.head_dim;
        let alloc = || -> Result<Vec<DeviceVec>> {
            (0..head.cfg.num_hidden_layers)
                .map(|_| DeviceVec::zeros(ctx, per_layer))
                .collect()
        };
        Ok(Self {
            k_ctx: alloc()?,
            v_ctx: alloc()?,
            ctx_len: 0,
            pending: None,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.ctx_len = 0;
        self.pending = None;
    }
}

/// Trunk residual-stream tap capture. `prepare` arms it for one forward; the
/// layer loop D2D-copies the residual stream after each target layer (`-1` =
/// the embedding output). `None` in the forward keeps baseline decode
/// byte-identical (no kernels, no allocations).
#[derive(Default)]
pub(crate) struct Qwen35DsparkTaps {
    targets: Vec<i64>,
    bufs: Vec<HiddenSlot>,
    hidden: usize,
    seq: usize,
    armed: bool,
}

impl Qwen35DsparkTaps {
    pub(crate) fn prepare(&mut self, targets: &[i64], hidden: usize, seq: usize) {
        if self.targets != targets {
            self.targets = targets.to_vec();
            self.bufs = std::iter::repeat_with(HiddenSlot::default)
                .take(targets.len())
                .collect();
        }
        self.hidden = hidden;
        self.seq = seq;
        self.armed = true;
    }

    /// Called by the forward after the residual stream for `layer` is complete
    /// (`-1` right after embedding). No-op for non-target layers.
    pub(crate) fn capture(
        &mut self,
        ctx: &DeviceContext,
        layer: i64,
        hidden: &HiddenStates,
    ) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        let Some(i) = self.targets.iter().position(|&t| t == layer) else {
            return Ok(());
        };
        ensure!(
            hidden.hidden_dim == self.hidden && hidden.seq_len == self.seq,
            "dspark tap shape [{}, {}] != armed [{}, {}]",
            hidden.hidden_dim,
            hidden.seq_len,
            self.hidden,
            self.seq
        );
        let buf = self.bufs[i].get(ctx, self.hidden, self.seq)?;
        ctx.stream
            .memcpy_dtod(&hidden.data, &mut buf.data)
            .map_err(|e| anyhow!("dspark tap capture failed: {e}"))?;
        Ok(())
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

/// Draft-side persistent scratch (exact-shape reuse, one per executor — spec
/// steps are serial). Separate from [`Qwen35Workspace`] so the draft's shapes
/// (`[.., block]`, draft intermediate size) never thrash the trunk's slots.
#[derive(Default)]
pub(crate) struct DsparkScratch {
    ids: SliceSlot<i32>,
    start_pos: SliceSlot<i32>,
    hidden: HiddenSlot,
    normed: HiddenSlot,
    hidden_mid: HiddenSlot,
    attn_out_h: HiddenSlot,
    q_full: HiddenSlot,
    k_new: HiddenSlot,
    v_new: HiddenSlot,
    q_prepped: HiddenSlot,
    attn_heads: HiddenSlot,
    dense: DenseMlpScratch,
    final_normed: HiddenSlot,
    logits: HiddenSlot,
    feat_a: HiddenSlot,
    feat_b: HiddenSlot,
    feat_rows: HiddenSlot,
    ctx_q_dummy: HiddenSlot,
    ctx_q_out: HiddenSlot,
    ctx_k: HiddenSlot,
    ctx_v: HiddenSlot,
    markov_tok: SliceSlot<i32>,
    markov_emb: HiddenSlot,
    markov_bias: HiddenSlot,
    step_logits: HiddenSlot,
    step_sum: HiddenSlot,
    conf_feat: VecSlot,
    conf_out: VecSlot,
    argmax: SliceSlot<i32>,
}

/// On-device argmax over one token row of a `[vocab, seq]` bf16 buffer
/// (`ops::argmax_row_into` for the `HiddenStates` shape — same kernel, no copy).
fn argmax_hs_row(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    row: usize,
    scratch: &mut CudaSlice<i32>,
) -> Result<u32> {
    let vocab = logits.hidden_dim;
    ensure!(row < logits.seq_len, "dspark argmax row {row} oob");
    {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = scratch.device_ptr_mut(&ctx.stream);
        // SAFETY: row bounds checked; the kernel reads `vocab` bf16 from the
        // row offset and writes one i32.
        unsafe {
            ffi::argmax_cuda(
                (l_ptr + (row * vocab * 2) as u64) as *const ffi::Half,
                o_ptr as *mut i32,
                vocab as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    ctx.sync()?;
    let token = ctx
        .stream
        .clone_dtoh(scratch)
        .map_err(|e| anyhow!("D2H dspark argmax failed: {e}"))?;
    Ok(token[0] as u32)
}

/// Executor-side DSpark runtime: the loaded head + per-slot draft state + the
/// shared tap/scratch buffers. Built only under `--spec-type dspark`, so the
/// baseline executor allocates nothing.
pub(crate) struct Qwen35DsparkExec {
    pub(crate) head: Qwen35DsparkHead,
    pub(crate) slots: Vec<Option<Qwen35DsparkSlotState>>,
    pub(crate) spec: Vec<Option<Qwen35SpecSlotState>>,
    pub(crate) taps: Qwen35DsparkTaps,
    pub(crate) scratch: DsparkScratch,
    pub(crate) accepts: usize,
    pub(crate) rejects: usize,
}

impl Qwen35DsparkExec {
    pub(crate) fn new(head: Qwen35DsparkHead, num_slots: usize) -> Self {
        Self {
            head,
            slots: (0..num_slots).map(|_| None).collect(),
            spec: (0..num_slots).map(|_| None).collect(),
            taps: Qwen35DsparkTaps::default(),
            scratch: DsparkScratch::default(),
            accepts: 0,
            rejects: 0,
        }
    }
}

fn dspark_confidence_threshold() -> f32 {
    std::env::var("ARLE_DSPARK_CONF_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5)
}

/// Load a raw 1D bf16/f32 vector as host bf16.
fn load_host_vec(loader: &SafetensorLoader, name: &str) -> Result<Vec<bf16>> {
    let t = loader.load_raw_tensor(name)?;
    let bytes = SafetensorLoader::tensor_bytes_to_bf16(name, t.dtype, &t.bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Load a norm vector as `w - 1` so the trunk's offset `(1+w)` kernels apply
/// the draft's plain Qwen3 RMSNorm weight (bf16-rounding-exact).
fn load_vec_minus_one(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
) -> Result<DeviceVec> {
    let host: Vec<bf16> = load_host_vec(loader, name)?
        .iter()
        .map(|w| bf16::from_f32(w.to_f32() - 1.0))
        .collect();
    DeviceVec::from_host(ctx, &host)
}

fn load_host_matrix(
    loader: &SafetensorLoader,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<bf16>> {
    let t = loader.load_raw_tensor(name)?;
    ensure!(
        t.shape == [rows, cols],
        "{name}: shape {:?} != [{rows}, {cols}]",
        t.shape
    );
    let bytes = SafetensorLoader::tensor_bytes_to_bf16(name, t.dtype, &t.bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub(crate) fn load_dspark_head(
    ctx: &DeviceContext,
    dir: &Path,
    max_seq_len: usize,
    trunk_hidden: usize,
    trunk_layers: usize,
    trunk_vocab: usize,
) -> Result<Qwen35DsparkHead> {
    let cfg = DsparkConfig::from_dir(dir)
        .map_err(|e| anyhow!("dspark draft config at {}: {e}", dir.display()))?;
    ensure!(
        cfg.hidden_size == trunk_hidden,
        "dspark hidden {} != trunk hidden {trunk_hidden}",
        cfg.hidden_size
    );
    for &t in &cfg.target_layer_ids {
        ensure!(
            t == -1 || (0..trunk_layers as i64).contains(&t),
            "dspark target layer {t} outside trunk [-1, {trunk_layers})"
        );
    }
    let loader = SafetensorLoader::new(dir)?;
    let names = dspark_tensor_names(cfg.num_hidden_layers);
    let hidden = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;

    // fc [hidden, n_taps*hidden] → one [hidden, hidden] per tap.
    let n_taps = cfg.target_layer_ids.len();
    let fc_host = load_host_matrix(&loader, &names.fc, hidden, n_taps * hidden)?;
    let fc = (0..n_taps)
        .map(|t| {
            let part: Vec<bf16> = (0..hidden)
                .flat_map(|r| {
                    let row = &fc_host[r * n_taps * hidden..];
                    row[t * hidden..(t + 1) * hidden].iter().copied()
                })
                .collect();
            DeviceMatrix::from_host(ctx, &part, hidden, hidden)
        })
        .collect::<Result<Vec<_>>>()?;

    let layers = names
        .layers
        .iter()
        .enumerate()
        .map(|(i, n)| {
            // Gate-pad q_proj rows into the trunk's gated layout (odd bands zero).
            let q_host = load_host_matrix(&loader, &n.q_proj, q_dim, hidden)?;
            let mut q_padded = vec![bf16::ZERO; 2 * q_dim * hidden];
            for h in 0..cfg.num_attention_heads {
                let src = h * cfg.head_dim * hidden;
                let dst = 2 * h * cfg.head_dim * hidden;
                q_padded[dst..dst + cfg.head_dim * hidden]
                    .copy_from_slice(&q_host[src..src + cfg.head_dim * hidden]);
            }
            Ok(DsparkLayer {
                q_proj: DeviceMatrix::from_host(ctx, &q_padded, 2 * q_dim, hidden)?,
                k_proj: loader.load_matrix(ctx, &n.k_proj)?,
                v_proj: loader.load_matrix(ctx, &n.v_proj)?,
                o_proj: loader.load_matrix(ctx, &n.o_proj)?,
                q_norm: load_vec_minus_one(&loader, ctx, &n.q_norm)?,
                k_norm: load_vec_minus_one(&loader, ctx, &n.k_norm)?,
                input_layernorm: loader.load_vec_any(ctx, &n.input_layernorm)?,
                post_attention_layernorm: loader.load_vec_any(ctx, &n.post_attention_layernorm)?,
                mlp: DenseMlp {
                    gate_proj: loader.load_matrix(ctx, &n.gate_proj)?,
                    up_proj: loader.load_matrix(ctx, &n.up_proj)?,
                    down_proj: loader.load_matrix(ctx, &n.down_proj)?,
                },
                sliding: cfg.layer_types[i] == DsparkLayerType::Sliding,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    ensure!(
        !loader.has_tensor(&names.markov_gate_proj) && !loader.has_tensor(&names.markov_joint_proj),
        "gated/RNN markov heads are not wired; only the vanilla w1/w2 head is supported"
    );
    let markov = if loader.has_tensor(&names.markov_w1) && loader.has_tensor(&names.markov_w2) {
        let w1 = loader.load_matrix(ctx, &names.markov_w1)?;
        let w2 = loader.load_matrix(ctx, &names.markov_w2)?;
        ensure!(
            w1.rows == trunk_vocab && w2.rows == trunk_vocab && w1.cols == w2.cols,
            "markov head shapes w1 [{}, {}] / w2 [{}, {}] vs vocab {trunk_vocab}",
            w1.rows,
            w1.cols,
            w2.rows,
            w2.cols
        );
        let rank = w1.cols;
        Some(DsparkMarkovHead { w1, w2, rank })
    } else {
        None
    };

    let confidence = if loader.has_tensor(&names.confidence_weight) {
        let weight = loader.load_matrix(ctx, &names.confidence_weight)?;
        let bias = load_host_vec(&loader, &names.confidence_bias)
            .map(|b| b.first().map_or(0.0, |v| v.to_f32()))
            .unwrap_or(0.0);
        let with_markov = weight.cols > hidden;
        if with_markov {
            let rank = markov
                .as_ref()
                .map(|m| m.rank)
                .ok_or_else(|| anyhow!("confidence head expects a markov head (in > hidden)"))?;
            ensure!(
                weight.cols == hidden + rank,
                "confidence in {} != hidden {hidden} + markov rank {rank}",
                weight.cols
            );
        } else {
            ensure!(
                weight.cols == hidden,
                "confidence in {} != hidden {hidden}",
                weight.cols
            );
        }
        Some(DsparkConfidenceHead {
            weight,
            bias,
            with_markov,
        })
    } else {
        None
    };

    if loader.has_tensor("lm_head.weight") || loader.has_tensor("embed_tokens.weight") {
        log::warn!(
            "dspark draft checkpoint carries its own embeddings/lm_head; \
             sharing the trunk's (DeepSpec trains them frozen-copied)"
        );
    }

    let cap = max_seq_len + cfg.block_size;
    let (cos_cache, sin_cache) =
        crate::ops::precompute_rope(ctx, cfg.head_dim, cap, cfg.rope_theta, None)?;
    Ok(Qwen35DsparkHead {
        confidence_threshold: dspark_confidence_threshold(),
        cfg,
        fc,
        hidden_norm: loader.load_vec_any(ctx, &names.hidden_norm)?,
        norm: loader.load_vec_any(ctx, &names.norm)?,
        layers,
        markov,
        confidence,
        cos_cache,
        sin_cache,
        cap,
    })
}

/// Elementwise add over two same-length device buffers.
fn add_vec_into(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch(ctx, a, b, &mut out)?;
    Ok(out)
}

impl Qwen35Model {
    /// Compute ctx features for `rows` leading tap rows and append their draft
    /// K/V at positions `[start, start+rows)`. `taps` must hold a capture of a
    /// forward whose row 0 sits at trunk position `start`.
    pub(crate) fn dspark_append_ctx(
        &self,
        head: &Qwen35DsparkHead,
        df: &mut Qwen35DsparkSlotState,
        taps: &mut Qwen35DsparkTaps,
        scratch: &mut DsparkScratch,
        rows: usize,
        start: usize,
    ) -> Result<()> {
        ensure!(taps.armed, "dspark ctx append without an armed tap capture");
        ensure!(
            df.ctx_len == start,
            "dspark ctx append at {start} but ctx_len {} (draft cache not contiguous)",
            df.ctx_len
        );
        ensure!(rows >= 1 && rows <= taps.seq, "dspark ctx rows {rows}");
        ensure!(start + rows <= head.cap, "dspark ctx overflow");
        let ctx = &self.ctx;
        let hidden = head.cfg.hidden_size;
        let seq = taps.seq;
        let eps = head.cfg.rms_norm_eps;

        // feat = hidden_norm(Σ_t fc_t · tap_t), over the full captured seq.
        let feat = {
            let acc = scratch.feat_a.get(ctx, hidden, seq)?;
            gemm_batch(ctx, &head.fc[0], taps.bufs[0].get(ctx, hidden, seq)?, acc)?;
            for (t, fc_t) in head.fc.iter().enumerate().skip(1) {
                let tmp = scratch.feat_b.get(ctx, hidden, seq)?;
                gemm_batch(ctx, fc_t, taps.bufs[t].get(ctx, hidden, seq)?, tmp)?;
                let sum = add_vec_into(ctx, scratch.feat_a.get(ctx, hidden, seq)?, tmp)?;
                ctx.stream
                    .memcpy_dtod(&sum.data, &mut scratch.feat_a.get(ctx, hidden, seq)?.data)
                    .map_err(|e| anyhow!("dspark fc accumulate failed: {e}"))?;
            }
            let acc = scratch.feat_a.get(ctx, hidden, seq)?;
            let normed = scratch.feat_b.get(ctx, hidden, seq)?;
            rms_norm_batch(ctx, acc, &head.hidden_norm, eps, normed)?;
            normed
        };
        // Leading `rows` tokens are contiguous in the token-major layout.
        let feat_rows = scratch.feat_rows.get(ctx, hidden, rows)?;
        {
            let src = feat.data.slice(0..rows * hidden);
            ctx.stream
                .memcpy_dtod(&src, &mut feat_rows.data)
                .map_err(|e| anyhow!("dspark feat prefix copy failed: {e}"))?;
        }
        taps.disarm();

        let kv_dim = head.kv_dim();
        let start_dev = scratch.start_pos.upload(&self.ctx, &[start as i32])?;
        let (sp_ptr, _gsp) = start_dev.device_ptr(&ctx.stream);
        for (li, layer) in head.layers.iter().enumerate() {
            let k_new = scratch.ctx_k.get(ctx, kv_dim, rows)?;
            gemm_batch(ctx, &layer.k_proj, feat_rows, k_new)?;
            let v_new = scratch.ctx_v.get(ctx, kv_dim, rows)?;
            gemm_batch(ctx, &layer.v_proj, feat_rows, v_new)?;
            // K-norm + RoPE + cache write via the fused prep kernel. The
            // wrapper requires q heads, so drive `num_kv_heads` dummy q heads
            // over a zeroed gated-layout buffer (0/rms(0+eps)=0, discarded).
            let q_dummy = scratch.ctx_q_dummy.get_zeroed(ctx, 2 * kv_dim, rows)?;
            let q_out = scratch.ctx_q_out.get(ctx, kv_dim, rows)?;
            let (qd_ptr, _g0) = q_dummy.data.device_ptr(&ctx.stream);
            let (k_ptr, _g1) = k_new.data.device_ptr(&ctx.stream);
            let (v_ptr, _g2) = v_new.data.device_ptr(&ctx.stream);
            let (kn_ptr, _g3) = layer.k_norm.data.device_ptr(&ctx.stream);
            let (cos_ptr, _g4) = head.cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _g5) = head.sin_cache.data.device_ptr(&ctx.stream);
            let (qo_ptr, _g6) = q_out.data.device_ptr_mut(&ctx.stream);
            let (kc_ptr, _g7) = df.k_ctx[li].data.device_ptr_mut(&ctx.stream);
            let (vc_ptr, _g8) = df.v_ctx[li].data.device_ptr_mut(&ctx.stream);
            // SAFETY: buffers valid on ctx.stream; caches sized cap*kv_dim and
            // start+rows <= cap (ensured above).
            unsafe {
                ffi::prefill_attention_hd256_prep_cuda(
                    qd_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    qo_ptr as *mut ffi::Half,
                    kc_ptr as *mut ffi::Half,
                    vc_ptr as *mut ffi::Half,
                    head.cfg.num_key_value_heads as i32,
                    head.cfg.num_key_value_heads as i32,
                    head.cfg.head_dim as i32,
                    rows as i32,
                    sp_ptr as *const i32,
                    head.cfg.head_dim as i32,
                    head.cfg.rms_norm_eps,
                    head.cap as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        df.ctx_len = start + rows;
        Ok(())
    }

    /// One DSpark block draft: propose up to `max_draft_tokens` greedy tokens
    /// from a single non-causal 5-layer forward over `[ctx cache ++ block]`.
    /// Returns the verify chain `[anchor, d1..dL]` (`L` truncated by the
    /// confidence head when present).
    pub(crate) fn dspark_draft_block(
        &self,
        head: &Qwen35DsparkHead,
        df: &mut Qwen35DsparkSlotState,
        scratch: &mut DsparkScratch,
        anchor: u32,
        start: usize,
    ) -> Result<Vec<u32>> {
        let cfg = &head.cfg;
        let ctx = &self.ctx;
        let block = cfg.block_size;
        ensure!(df.ctx_len == start, "dspark draft: ctx_len != start");
        ensure!(start + block <= head.cap, "dspark draft past cache cap");
        let hidden = cfg.hidden_size;
        let (q_dim, kv_dim) = (head.q_dim(), head.kv_dim());
        let eps = cfg.rms_norm_eps;
        let kv_len_total = start + block;

        let mut ids = vec![cfg.mask_token_id as i32; block];
        ids[0] = anchor as i32;
        let ids_dev = scratch.ids.upload(ctx, &ids)?;
        let h = scratch.hidden.get(ctx, hidden, block)?;
        embedding_batch(ctx, &self.embed_tokens, ids_dev, h)?;

        let start_dev = scratch.start_pos.upload(&self.ctx, &[start as i32])?;
        for (li, layer) in head.layers.iter().enumerate() {
            let h = scratch.hidden.get(ctx, hidden, block)?;
            let normed = scratch.normed.get(ctx, hidden, block)?;
            rms_norm_batch(ctx, h, &layer.input_layernorm, eps, normed)?;

            let q_full = scratch.q_full.get(ctx, 2 * q_dim, block)?;
            gemm_batch(ctx, &layer.q_proj, normed, q_full)?;
            let k_new = scratch.k_new.get(ctx, kv_dim, block)?;
            gemm_batch(ctx, &layer.k_proj, normed, k_new)?;
            let v_new = scratch.v_new.get(ctx, kv_dim, block)?;
            gemm_batch(ctx, &layer.v_proj, normed, v_new)?;

            // q/k head-RMSNorm + RoPE at absolute positions start..start+block;
            // noise K/V land at their positions in the ctx cache (self-healing
            // speculative rows — see the slot-state doc).
            let q_prepped = scratch.q_prepped.get(ctx, q_dim, block)?;
            {
                let (qf_ptr, _g0) = q_full.data.device_ptr(&ctx.stream);
                let (k_ptr, _g1) = k_new.data.device_ptr(&ctx.stream);
                let (v_ptr, _g2) = v_new.data.device_ptr(&ctx.stream);
                let (qn_ptr, _g3) = layer.q_norm.data.device_ptr(&ctx.stream);
                let (kn_ptr, _g4) = layer.k_norm.data.device_ptr(&ctx.stream);
                let (cos_ptr, _g5) = head.cos_cache.data.device_ptr(&ctx.stream);
                let (sin_ptr, _g6) = head.sin_cache.data.device_ptr(&ctx.stream);
                let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&ctx.stream);
                let (kc_ptr, _g8) = df.k_ctx[li].data.device_ptr_mut(&ctx.stream);
                let (vc_ptr, _g9) = df.v_ctx[li].data.device_ptr_mut(&ctx.stream);
                let (sp_ptr, _g10) = start_dev.device_ptr(&ctx.stream);
                // SAFETY: shapes as above; cache holds cap*kv_dim, start+block <= cap.
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
                        cfg.num_attention_heads as i32,
                        cfg.num_key_value_heads as i32,
                        cfg.head_dim as i32,
                        block as i32,
                        sp_ptr as *const i32,
                        cfg.head_dim as i32,
                        eps,
                        head.cap as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }

            // Non-causal attention: every noise row attends the whole
            // `[ctx ++ block]` key range. Per-row launches with `seq_len=1` and
            // `kv_len = start+block` express the non-causal window through the
            // causal kernel; sliding layers shift the K/V base by `lo` rows
            // (a uniform in-band offset in the `[head][cap][dim]` layout).
            let attn_heads = scratch.attn_heads.get(ctx, q_dim, block)?;
            {
                let sm_scale = 1.0 / (cfg.head_dim as f32).sqrt();
                let elem = std::mem::size_of::<ffi::Half>() as u64;
                let (q_ptr, _g0) = q_prepped.data.device_ptr(&ctx.stream);
                let (kc_ptr, _g1) = df.k_ctx[li].data.device_ptr(&ctx.stream);
                let (vc_ptr, _g2) = df.v_ctx[li].data.device_ptr(&ctx.stream);
                let (o_ptr, _g3) = attn_heads.data.device_ptr_mut(&ctx.stream);
                for row in 0..block {
                    let q_pos = start + row;
                    let lo = if layer.sliding {
                        // HF sliding window keeps keys with q_pos - k_pos < window.
                        q_pos.saturating_sub(cfg.sliding_window - 1)
                    } else {
                        0
                    };
                    let kv_len = kv_len_total - lo;
                    let base = (lo * cfg.head_dim) as u64 * elem;
                    // SAFETY: row/lo offsets stay inside the per-head bands
                    // (lo + kv_len == start+block <= cap).
                    unsafe {
                        ffi::nonpaged_prefill_attention_cuda(
                            (q_ptr + (row * q_dim) as u64 * elem) as *const ffi::Half,
                            (kc_ptr + base) as *const ffi::Half,
                            (vc_ptr + base) as *const ffi::Half,
                            (o_ptr + (row * q_dim) as u64 * elem) as *mut ffi::Half,
                            cfg.num_attention_heads as i32,
                            cfg.num_key_value_heads as i32,
                            cfg.head_dim as i32,
                            1,
                            kv_len as i32,
                            head.cap as i32,
                            sm_scale,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
            }

            let attn_out_h = scratch.attn_out_h.get(ctx, hidden, block)?;
            gemm_batch(ctx, &layer.o_proj, attn_heads, attn_out_h)?;
            let hidden_mid = scratch.hidden_mid.get(ctx, hidden, block)?;
            add_batch(
                ctx,
                scratch.hidden.get(ctx, hidden, block)?,
                attn_out_h,
                hidden_mid,
            )?;
            let normed = scratch.normed.get(ctx, hidden, block)?;
            rms_norm_batch(
                ctx,
                hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                normed,
            )?;
            let mlp_out = scratch.attn_out_h.get(ctx, hidden, block)?;
            self.dense_mlp(&layer.mlp, normed, &mut scratch.dense, mlp_out)?;
            add_batch(
                ctx,
                scratch.hidden_mid.get(ctx, hidden, block)?,
                mlp_out,
                scratch.hidden.get(ctx, hidden, block)?,
            )?;
        }

        let final_normed = scratch.final_normed.get(ctx, hidden, block)?;
        rms_norm_batch(
            ctx,
            scratch.hidden.get(ctx, hidden, block)?,
            &head.norm,
            eps,
            final_normed,
        )?;
        let vocab = self.output_projection().rows;
        let logits = scratch.logits.get(ctx, vocab, block)?;
        gemm_batch(ctx, self.output_projection(), final_normed, logits)?;

        // Greedy left-to-right sampling. Next-token heads (DSpark) draft from
        // every row; same-position (DFlash) rows 1.. fill their own positions.
        let first_row = usize::from(!cfg.next_token_heads);
        let mut drafts = Vec::with_capacity(block);
        let mut prev = anchor;
        for row in first_row..block {
            let tok = if let Some(m) = &head.markov {
                // step_logits = base_row + markov_w2 · markov_w1[prev]
                let tok_dev = scratch.markov_tok.upload(ctx, &[prev as i32])?;
                let emb = scratch.markov_emb.get(ctx, m.rank, 1)?;
                embedding_batch(ctx, &m.w1, tok_dev, emb)?;
                let bias = scratch.markov_bias.get(ctx, vocab, 1)?;
                gemm_batch(ctx, &m.w2, emb, bias)?;
                let step = scratch.step_logits.get(ctx, vocab, 1)?;
                {
                    let logits = scratch.logits.get(ctx, vocab, block)?;
                    let src = logits.data.slice(row * vocab..(row + 1) * vocab);
                    ctx.stream
                        .memcpy_dtod(&src, &mut step.data)
                        .map_err(|e| anyhow!("dspark markov row copy failed: {e}"))?;
                }
                let sum = scratch.step_sum.get(ctx, vocab, 1)?;
                add_batch(
                    ctx,
                    scratch.step_logits.get(ctx, vocab, 1)?,
                    scratch.markov_bias.get(ctx, vocab, 1)?,
                    sum,
                )?;
                let am = scratch.argmax.get(ctx, 1)?;
                argmax_hs_row(ctx, scratch.step_sum.get(ctx, vocab, 1)?, 0, am)?
            } else {
                let logits = scratch.logits.get(ctx, vocab, block)?;
                let am = scratch.argmax.get(ctx, 1)?;
                argmax_hs_row(ctx, logits, row, am)?
            };
            drafts.push(tok);
            prev = tok;
        }

        // Confidence seam: truncate the proposal at the first low-confidence
        // position; absent head = keep the whole block.
        let prev_tokens: Vec<u32> = std::iter::once(anchor)
            .chain(drafts.iter().copied())
            .take(drafts.len())
            .collect();
        let keep = self.dspark_confident_prefix_len(head, scratch, &prev_tokens)?;
        drafts.truncate(keep.min(drafts.len()));

        let mut chain = Vec::with_capacity(1 + drafts.len());
        chain.push(anchor);
        chain.extend(drafts);
        Ok(chain)
    }

    /// Confidence-head seam: acceptance-confident prefix length over the block
    /// rows (`sigmoid(proj([hidden_i ; markov_w1[prev_i]])) >= threshold`).
    /// Head absent → the full block survives.
    fn dspark_confident_prefix_len(
        &self,
        head: &Qwen35DsparkHead,
        scratch: &mut DsparkScratch,
        prev_tokens: &[u32],
    ) -> Result<usize> {
        let Some(conf) = &head.confidence else {
            return Ok(usize::MAX);
        };
        let ctx = &self.ctx;
        let hidden = head.cfg.hidden_size;
        let block = head.cfg.block_size;
        let in_dim = conf.weight.cols;
        for (i, &prev) in prev_tokens.iter().enumerate().take(block) {
            let feat = scratch.conf_feat.get(ctx, in_dim)?;
            {
                let final_normed = scratch.final_normed.get(ctx, hidden, block)?;
                let src = final_normed.data.slice(i * hidden..(i + 1) * hidden);
                let mut dst = feat.data.slice_mut(0..hidden);
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("dspark conf feature copy failed: {e}"))?;
            }
            if conf.with_markov {
                let m = head.markov.as_ref().expect("validated at load");
                let tok_dev = scratch.markov_tok.upload(ctx, &[prev as i32])?;
                let emb = scratch.markov_emb.get(ctx, m.rank, 1)?;
                embedding_batch(ctx, &m.w1, tok_dev, emb)?;
                let feat = scratch.conf_feat.get(ctx, in_dim)?;
                let src = emb.data.slice(0..m.rank);
                let mut dst = feat.data.slice_mut(hidden..hidden + m.rank);
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("dspark conf markov copy failed: {e}"))?;
            }
            let out = scratch.conf_out.get(ctx, 1)?;
            gemv(ctx, &conf.weight, scratch.conf_feat.get(ctx, in_dim)?, out)?;
            let mut host = [bf16::ZERO];
            scratch
                .conf_out
                .get(ctx, 1)?
                .copy_region_to_host(ctx, 0, 1, &mut host)?;
            ctx.sync()?;
            let logit = host[0].to_f32() + conf.bias;
            if 1.0 / (1.0 + (-logit).exp()) < head.confidence_threshold {
                return Ok(i);
            }
        }
        Ok(usize::MAX)
    }

    /// Size the reused per-slot spec state (`new_spec_slot_state` capture rows,
    /// verify-depth guard) for the DSpark block depth. Called once at attach.
    pub(crate) fn set_spec_draft_tokens(&mut self, n: usize) {
        self.spec_draft_tokens = n;
    }

    /// Accept scan + trunk rollback over a DSpark verify (`spec_step` steps 4-5
    /// with the draft source swapped out). `logits` is the `[chain_len, vocab]`
    /// verify output; the pre-verify trunk snapshot must already be in `spec`.
    /// Returns `(emitted, bonus, k)`; the caller crops the paged pool to
    /// `start_pos + k + 1` when `k + 1 < chain.len()`.
    pub(crate) fn dspark_accept_commit(
        &self,
        slot: &mut Qwen35SlotState,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        chain: &[u32],
        logits: &DeviceVec,
        start_pos: usize,
    ) -> Result<(Vec<u32>, u32, usize)> {
        let depth = chain.len() - 1;
        let vocab = self.output_projection().rows;
        // Longest prefix where each draft equals the trunk argmax at its row.
        let mut k = 0usize;
        let bonus;
        loop {
            let am = argmax_row_into(&self.ctx, logits, k, vocab, &mut spec.argmax_scratch)?;
            if k < depth && am == chain[k + 1] {
                k += 1;
            } else {
                bonus = am;
                break;
            }
        }
        let mut emitted: Vec<u32> = chain[1..=k].to_vec();
        emitted.push(bonus);
        if k < depth {
            // Rewind the 48 gated-delta recurrent/conv states to the pre-verify
            // snapshot, then linear-only replay of the accepted prefix from the
            // verify capture; the paged full-attn KV self-heals under the pool
            // truncate + seq_len rewind (position-indexed rows).
            spec.restore_trunk(&self.ctx, slot)?;
            self.replay_linear_only(slot, ws, &spec.capture, k)?;
            slot.set_seq_len(start_pos + k + 1);
        }
        Ok((emitted, bonus, k))
    }

    /// Trunk verify over the paged pool with linear-capture + tap capture:
    /// per-row logits `[chain_len, vocab]` for the accept scan. The paged twin
    /// of [`Self::forward_tokens_verify`] (that one drives the legacy
    /// contiguous lane and the MTP gate test).
    pub(crate) fn dspark_verify_logits(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        chain: &[u32],
        start_pos: usize,
        spec: &mut Qwen35SpecSlotState,
        recall: &mut Qwen35RecallForward<'_>,
        taps: &mut Qwen35DsparkTaps,
    ) -> Result<DeviceVec> {
        let seq_len = chain.len();
        ensure!(
            slot.seq_len() == start_pos,
            "dspark verify entry seq_len {} != start_pos {start_pos}",
            slot.seq_len()
        );
        self.stage_step_inputs(ws, chain, start_pos)?;
        self.forward_hidden_staged(
            slot,
            ws,
            seq_len,
            start_pos,
            Some(&mut spec.capture),
            Some(recall),
            Some(taps),
        )?;
        slot.advance_seq_len(seq_len);
        let hidden_size = self.config.hidden_size;
        let Qwen35Workspace { hidden, normed, .. } = ws;
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        super::rms_norm_offset(
            &self.ctx,
            hidden,
            &self.norm,
            self.config.rms_norm_eps,
            normed,
        )?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)?;
        self.ctx.sync()?;
        Ok(DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "dspark_verify_logits[seq,vocab]",
        })
    }
}
