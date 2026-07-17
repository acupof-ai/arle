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
    /// sigmoid(conf) < threshold truncates the proposal (`--dspark-conf-threshold`).
    confidence_threshold: f32,
    /// Draft RoPE tables (full rotary over head_dim, `rope_theta`), `rope_cap`
    /// absolute positions (== the full-attention layer cap).
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Per-layer ctx-cache rows (== per-head cache stride): sliding layers hold
    /// `sliding_window + block_size` addressed as an absolute-position ring
    /// (`row = pos % cap`); the full-attention layer holds `ctx_cap +
    /// block_size` addressed linearly (`row = pos − ctx_base`).
    caps: Vec<usize>,
    /// RoPE table length + full-layer cap = `ctx_cap + block_size`, where
    /// `ctx_cap = min(max_seq_len, max_total_tokens)` — the per-request token
    /// ceiling, NOT the whole-pool `max_seq_len`. Sizing the full draft layer
    /// from the pool floor (128K) cost 512 MB/slot; the scheduler admits nothing
    /// past `max_total_tokens`, so that is all the full layer ever caches.
    rope_cap: usize,
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
    /// Per-slot draft ctx K/V cache bytes ([`Qwen35DsparkSlotState`], lazily
    /// allocated at first prefill) — reserved out of the KV budget so slot
    /// admission and pool profiling account for them (else startup passes and
    /// the first dspark prefill OOMs).
    pub(crate) fn slot_state_bytes(&self) -> usize {
        let per_head = self.cfg.num_key_value_heads * self.cfg.head_dim;
        2 * self.caps.iter().map(|c| c * per_head).sum::<usize>() * std::mem::size_of::<bf16>()
    }
    pub(crate) fn mode_label(&self) -> &'static str {
        match (
            self.cfg.next_token_heads,
            self.markov.is_some(),
            self.confidence.is_some(),
        ) {
            (false, false, false) => "dflash-backbone",
            (false, true, false) => "dspark-sp+markov",
            (false, false, true) => "dspark-sp+confidence",
            (false, true, true) => "dspark-sp+markov+confidence",
            (true, false, false) => "dspark-backbone",
            (true, true, false) => "dspark+markov",
            (true, false, true) => "dspark+confidence",
            (true, true, true) => "dspark+markov+confidence",
        }
    }
}

/// Per-slot draft context K/V cache: for each draft layer, the RoPE'd/normed K
/// and raw V of accepted positions, laid out `[kv_head][cap][head_dim]`.
/// Addressing is per layer type (see `Qwen35DsparkHead::caps`): the full layer
/// is LINEAR (`row = abs − ctx_base`, `cap = max_seq + block`); sliding layers
/// are an absolute-position RING (`row = abs % cap`, `cap = window + block`),
/// keeping only the last `window` keys the sliding attention ever reads. Noise
/// rows are written speculatively at their positions and self-heal: rejected
/// rows are overwritten by the next block's ctx/noise writes before any read.
/// A ring never aliases a live key within a block — the live span
/// (window ctx + block noise) equals `cap`, so distinct positions never collide
/// mod `cap`.
pub(crate) struct Qwen35DsparkSlotState {
    k_ctx: Vec<DeviceVec>,
    v_ctx: Vec<DeviceVec>,
    /// Absolute trunk position of ctx buffer row 0. Non-zero after a
    /// prefix-restore rebase: the suffix-only ctx is exact for sliding layers
    /// once the tail ≥ window, approximate only for the full-attention layer.
    pub(crate) ctx_base: usize,
    /// Absolute end (exclusive) of materialized ctx rows; buffer holds
    /// `ctx_end - ctx_base` rows. The spec step requires `ctx_end == start_pos`.
    pub(crate) ctx_end: usize,
    /// Last emitted token (the next block's anchor), staged by prefill/verify.
    pub(crate) pending: Option<u32>,
}

impl Qwen35DsparkSlotState {
    pub(crate) fn new(ctx: &DeviceContext, head: &Qwen35DsparkHead) -> Result<Self> {
        let per_head = head.cfg.num_key_value_heads * head.cfg.head_dim;
        let alloc = || -> Result<Vec<DeviceVec>> {
            head.caps
                .iter()
                .map(|&cap| DeviceVec::zeros(ctx, cap * per_head))
                .collect()
        };
        Ok(Self {
            k_ctx: alloc()?,
            v_ctx: alloc()?,
            ctx_base: 0,
            ctx_end: 0,
            pending: None,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.rebase(0);
    }

    /// Re-key the (empty) ctx buffer so row 0 is absolute position `pos`. No
    /// buffer zeroing needed: stale rows are overwritten by post-rebase
    /// appends before any read (attention lo never drops below `ctx_base`).
    pub(crate) fn rebase(&mut self, pos: usize) {
        self.ctx_base = pos;
        self.ctx_end = pos;
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
    /// Absolute (unshifted) start position for sliding-ring prep launches.
    start_pos_abs: SliceSlot<i32>,
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
    /// Sampled-mode device buffers (allocated on first temp>0 spec step only;
    /// greedy never touches them). Fixed caps — no per-step realloc:
    /// `q_probs [block, vocab] f32` draft filtered dists (row i fully written
    /// by the markov-step filter kernel before the chain kernel reads it);
    /// `p_probs [block+1, vocab] f32` verify filtered dists (leading
    /// `chain_len` rows fully written per accept; the stale tail is never
    /// read — the chain kernel indexes rows `<= depth < chain_len`);
    /// `sample_tok [1]` / `accept_out [2]` fully written before D2H;
    /// `chain_draft [block]` / `u_accept [block]` / `u_residual [block+1]`
    /// host-uploaded prefixes — the kernel reads only the uploaded prefix.
    q_probs: SliceSlot<f32>,
    p_probs: SliceSlot<f32>,
    sample_tok: SliceSlot<i32>,
    accept_out: SliceSlot<i32>,
    chain_draft: SliceSlot<i32>,
    u_accept: SliceSlot<f32>,
    u_residual: SliceSlot<f32>,
    /// Valid draft rows in `q_probs` (set by the draft, checked by the accept).
    q_rows: usize,
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

/// Uniform-stream salts: draft draw / accept test / residual+bonus draw at the
/// same position must be independent or the rejection identity breaks.
/// `pub(super)`: the MTP lane's rejection twin shares the same streams.
pub(super) const SALT_DRAW: u64 = 0;
pub(super) const SALT_ACCEPT: u64 = 0x9E37_79B9_7F4A_7C15;
pub(super) const SALT_RESIDUAL: u64 = 0xC2B2_AE3D_27D4_EB4F;

/// SplitMix64 — mirrors `infer_plan::sample`'s private mixer bit-for-bit.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic uniform in [0, 1) from `(seed, salt, position)` — the engine
/// sampler's `(seed, position)` stream (`infer_plan::sample_token`), so
/// same-config-twice reproduces. `SALT_DRAW = 0` makes the draft draw consume
/// exactly the uniform plain decode would at that position.
pub(super) fn unit_uniform(seed: Option<u64>, salt: u64, position: u64) -> f32 {
    let bits = splitmix64(
        seed.unwrap_or(0)
            .wrapping_add(salt)
            .wrapping_add(position)
            .wrapping_add(1),
    );
    (bits >> 40) as f32 / (1u32 << 24) as f32
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
    /// Verified draft chains, and the subset drafted from a partial
    /// (ctx_base > 0) draft context.
    pub(crate) chains: usize,
    pub(crate) partial_ctx_chains: usize,
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
            chains: 0,
            partial_ctx_chains: 0,
        }
    }
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
    max_total_tokens: usize,
    trunk_hidden: usize,
    trunk_layers: usize,
    trunk_vocab: usize,
    confidence_threshold: f32,
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

    // Per-layer cache stride: sliding layers only ever read the last `window`
    // keys, so a `window + block` ring suffices; the full-attention layer caches
    // one request's accepted tokens — capped at the per-request ceiling
    // `min(max_seq_len, max_total_tokens)`, NOT the whole KV-pool `max_seq_len`
    // (128K floor = 512 MB/slot; the scheduler admits nothing longer).
    let ctx_cap = max_seq_len.min(max_total_tokens.max(1));
    let rope_cap = ctx_cap + cfg.block_size;
    let caps: Vec<usize> = layers
        .iter()
        .map(|l| {
            if l.sliding {
                cfg.sliding_window + cfg.block_size
            } else {
                rope_cap
            }
        })
        .collect();
    let (cos_cache, sin_cache) =
        crate::ops::precompute_rope(ctx, cfg.head_dim, rope_cap, cfg.rope_theta, None)?;
    Ok(Qwen35DsparkHead {
        confidence_threshold,
        cfg,
        fc,
        hidden_norm: loader.load_vec_any(ctx, &names.hidden_norm)?,
        norm: loader.load_vec_any(ctx, &names.norm)?,
        layers,
        markov,
        confidence,
        cos_cache,
        sin_cache,
        caps,
        rope_cap,
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
            df.ctx_end == start,
            "dspark ctx append at {start} but ctx_end {} (draft cache not contiguous)",
            df.ctx_end
        );
        ensure!(rows >= 1 && rows <= taps.seq, "dspark ctx rows {rows}");
        // Absolute positions must index the shared cos/sin table.
        ensure!(
            start + rows <= head.rope_cap,
            "dspark ctx overflow past rope_cap"
        );
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
        // Full layers: buffer-relative start + `ctx_base`-shifted RoPE tables →
        // linear write at `abs − ctx_base`. Sliding layers: ABSOLUTE start into
        // the ring prep kernel (RoPE unshifted, cache row = `abs % cap`).
        let rope_off = (df.ctx_base * head.cfg.head_dim) as u64 * 2;
        let start_rel = scratch
            .start_pos
            .upload(&self.ctx, &[(start - df.ctx_base) as i32])?;
        let (sp_rel, _gr) = start_rel.device_ptr(&ctx.stream);
        let start_abs = scratch.start_pos_abs.upload(&self.ctx, &[start as i32])?;
        let (sp_abs, _ga) = start_abs.device_ptr(&ctx.stream);
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.caps[li];
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
            let nkv = head.cfg.num_key_value_heads as i32;
            let hd = head.cfg.head_dim as i32;
            if layer.sliding {
                // One launch must not wrap the ring (else two tokens alias one
                // row); chunk sizes (≤32) sit far below cap here.
                ensure!(
                    rows <= cap_li,
                    "dspark sliding append rows {rows} > ring cap {cap_li}; lower chunked_prefill_size"
                );
                // SAFETY: ring cache sized cap_li*kv_dim; rows <= cap_li (no
                // aliasing); abs positions < rope_cap index the cos/sin tables.
                unsafe {
                    ffi::prefill_attention_hd256_prep_ring_cuda(
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
                        nkv,
                        nkv,
                        hd,
                        rows as i32,
                        sp_abs as *const i32,
                        hd,
                        head.cfg.rms_norm_eps,
                        cap_li as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            } else {
                // SAFETY: linear cache sized cap_li*kv_dim; abs − ctx_base <
                // cap_li; rope_off stays inside the cos/sin tables.
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qd_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        (cos_ptr + rope_off) as *const ffi::Half,
                        (sin_ptr + rope_off) as *const ffi::Half,
                        qo_ptr as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        nkv,
                        nkv,
                        hd,
                        rows as i32,
                        sp_rel as *const i32,
                        hd,
                        head.cfg.rms_norm_eps,
                        cap_li as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
        df.ctx_end = start + rows;
        Ok(())
    }

    /// One DSpark block draft: propose up to `max_draft_tokens` tokens from a
    /// single non-causal 5-layer forward over `[ctx cache ++ block]`.
    /// Returns the verify chain `[anchor, d1..dL]` (`L` truncated by the
    /// confidence head when present). In sampling mode (`!params.is_greedy()`)
    /// each draft is a device draw from the engine-sampler-filtered dist `q`,
    /// retained on device in `scratch.q_probs` row `i` for `chain[i+1]`
    /// (greedy: argmax path, byte-identical, no q buffers touched).
    pub(crate) fn dspark_draft_block(
        &self,
        head: &Qwen35DsparkHead,
        df: &mut Qwen35DsparkSlotState,
        scratch: &mut DsparkScratch,
        anchor: u32,
        start: usize,
        params: &SamplingParams,
    ) -> Result<Vec<u32>> {
        let cfg = &head.cfg;
        let ctx = &self.ctx;
        let block = cfg.block_size;
        ensure!(df.ctx_end == start, "dspark draft: ctx_end != start");
        ensure!(
            start + block <= head.rope_cap,
            "dspark draft past cache cap"
        );
        let hidden = cfg.hidden_size;
        let (q_dim, kv_dim) = (head.q_dim(), head.kv_dim());
        let eps = cfg.rms_norm_eps;
        let kv_len_total = start + block;

        let mut pt = super::dspark_phase_start(ctx);
        let (mut prep_ms, mut attn_ms, mut mlp_ms) = (0.0f64, 0.0f64, 0.0f64);
        let mut ids = vec![cfg.mask_token_id as i32; block];
        ids[0] = anchor as i32;
        let ids_dev = scratch.ids.upload(ctx, &ids)?;
        let h = scratch.hidden.get(ctx, hidden, block)?;
        embedding_batch(ctx, &self.embed_tokens, ids_dev, h)?;
        let embed_ms = super::mtp_phase_lap(ctx, &mut pt);

        // Full layers: buffer-relative start + ctx_base-shifted RoPE. Sliding
        // layers: absolute start into the ring prep (see `dspark_append_ctx`).
        let rope_off = (df.ctx_base * cfg.head_dim) as u64 * 2;
        let start_rel = scratch
            .start_pos
            .upload(&self.ctx, &[(start - df.ctx_base) as i32])?;
        let start_abs = scratch.start_pos_abs.upload(&self.ctx, &[start as i32])?;
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.caps[li];
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
                let nq = cfg.num_attention_heads as i32;
                let nkv = cfg.num_key_value_heads as i32;
                let hd = cfg.head_dim as i32;
                if layer.sliding {
                    let (sp_ptr, _g10) = start_abs.device_ptr(&ctx.stream);
                    // block (≤ cap_li) noise rows write distinct ring rows.
                    // SAFETY: ring cache cap_li*kv_dim; abs pos < rope_cap.
                    unsafe {
                        ffi::prefill_attention_hd256_prep_ring_cuda(
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
                            nq,
                            nkv,
                            hd,
                            block as i32,
                            sp_ptr as *const i32,
                            hd,
                            eps,
                            cap_li as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                } else {
                    let (sp_ptr, _g10) = start_rel.device_ptr(&ctx.stream);
                    // SAFETY: shapes as above; cache holds cap_li*kv_dim,
                    // start+block-ctx_base <= cap_li.
                    unsafe {
                        ffi::prefill_attention_hd256_prep_cuda(
                            qf_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            v_ptr as *const ffi::Half,
                            qn_ptr as *const ffi::Half,
                            kn_ptr as *const ffi::Half,
                            (cos_ptr + rope_off) as *const ffi::Half,
                            (sin_ptr + rope_off) as *const ffi::Half,
                            qp_ptr as *mut ffi::Half,
                            kc_ptr as *mut ffi::Half,
                            vc_ptr as *mut ffi::Half,
                            nq,
                            nkv,
                            hd,
                            block as i32,
                            sp_ptr as *const i32,
                            hd,
                            eps,
                            cap_li as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
            }

            prep_ms += super::mtp_phase_lap(ctx, &mut pt);
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
                let nq = cfg.num_attention_heads as i32;
                let nkv = cfg.num_key_value_heads as i32;
                let hd = cfg.head_dim as i32;
                for row in 0..block {
                    let q_pos = start + row;
                    let q_off = (q_ptr + (row * q_dim) as u64 * elem) as *const ffi::Half;
                    let o_off = (o_ptr + (row * q_dim) as u64 * elem) as *mut ffi::Half;
                    if layer.sliding {
                        // HF sliding window keeps keys with q_pos - k_pos < window;
                        // never below ctx_base. The ctx buffer is an absolute
                        // ring: walk `[lo, start+block)` mapped through `% cap_li`
                        // (order-independent softmax → identical to a linear walk).
                        let lo = q_pos
                            .saturating_sub(cfg.sliding_window - 1)
                            .max(df.ctx_base);
                        let kv_len = kv_len_total - lo;
                        // SAFETY: kv_len == q_pos+1−lo ≤ window+block == cap_li,
                        // so the ring read never revisits a physical row.
                        unsafe {
                            ffi::nonpaged_prefill_attention_ring_cuda(
                                q_off,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_off,
                                nq,
                                nkv,
                                hd,
                                1,
                                kv_len as i32,
                                lo as i32,
                                cap_li as i32,
                                sm_scale,
                                ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                    } else {
                        let lo = df.ctx_base;
                        let kv_len = kv_len_total - lo;
                        let base = ((lo - df.ctx_base) * cfg.head_dim) as u64 * elem;
                        // SAFETY: lo-ctx_base + kv_len == start+block-ctx_base <= cap_li.
                        unsafe {
                            ffi::nonpaged_prefill_attention_cuda(
                                q_off,
                                (kc_ptr + base) as *const ffi::Half,
                                (vc_ptr + base) as *const ffi::Half,
                                o_off,
                                nq,
                                nkv,
                                hd,
                                1,
                                kv_len as i32,
                                cap_li as i32,
                                sm_scale,
                                ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                    }
                }
            }

            attn_ms += super::mtp_phase_lap(ctx, &mut pt);
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
            mlp_ms += super::mtp_phase_lap(ctx, &mut pt);
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
        let head_ms = super::mtp_phase_lap(ctx, &mut pt);

        // Left-to-right token selection: argmax (greedy) or a rejection-ready
        // device draw from the engine-sampler-filtered distribution q (full
        // row retained in `scratch.q_probs` — only the token id comes back).
        // Next-token heads (DSpark) draft from every row; same-position
        // (DFlash) rows 1.. fill their own positions.
        let sampling = !params.is_greedy();
        scratch.q_rows = 0;
        let first_row = usize::from(!cfg.next_token_heads);
        let mut drafts = Vec::with_capacity(block);
        let mut prev = anchor;
        for row in first_row..block {
            // Corrected logits row: base + markov bias when the head is present.
            let (src, src_row) = if let Some(m) = &head.markov {
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
                (scratch.step_sum.get(ctx, vocab, 1)?, 0)
            } else {
                (scratch.logits.get(ctx, vocab, block)?, row)
            };
            let tok = if sampling {
                // Filter + q-row store + multinomial draw in one device call;
                // uniform from the host (seed, position) stream plain decode
                // would consume at this position (SALT_DRAW = 0).
                let u = unit_uniform(params.seed, SALT_DRAW, (start + row) as u64);
                let q_all = scratch.q_probs.get(ctx, block * vocab)?;
                let tok_out = scratch.sample_tok.get(ctx, 1)?;
                {
                    let elem = std::mem::size_of::<bf16>() as u64;
                    let (l_ptr, _gl) = src.data.device_ptr(&ctx.stream);
                    let (q_ptr, _gq) = q_all.device_ptr_mut(&ctx.stream);
                    let (t_ptr, _gt) = tok_out.device_ptr_mut(&ctx.stream);
                    // SAFETY: `src` row holds `vocab` bf16 (src_row bounded by
                    // its seq_len); q row index == drafts.len() < block.
                    unsafe {
                        ffi::dspark_draft_sample_cuda(
                            (l_ptr + (src_row * vocab) as u64 * elem) as *const ffi::Half,
                            (q_ptr + (drafts.len() * vocab * 4) as u64) as *mut f32,
                            t_ptr as *mut i32,
                            vocab as i32,
                            1.0 / params.temperature,
                            params.top_k,
                            params.top_p,
                            params.min_p,
                            u,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
                ctx.sync()?;
                let tok = ctx
                    .stream
                    .clone_dtoh(tok_out)
                    .map_err(|e| anyhow!("D2H dspark draft token failed: {e}"))?[0]
                    as u32;
                scratch.q_rows += 1;
                tok
            } else {
                argmax_hs_row(ctx, src, src_row, scratch.argmax.get(ctx, 1)?)?
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
        let argmax_ms = super::mtp_phase_lap(ctx, &mut pt);
        if pt.is_some() {
            eprintln!(
                "[dspark-draft] embed={embed_ms:.2} prep={prep_ms:.2} attn={attn_ms:.2} mlp={mlp_ms:.2} head={head_ms:.2} argmax={argmax_ms:.2} ms"
            );
        }
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

    /// Rejection-sampling twin of [`Self::dspark_accept_commit`] (mirrors
    /// flashinfer/SGLang `chain_speculative_sampling`): accept `chain[j+1]`
    /// with prob min(1, p_j(tok)/q_j(tok)) under the engine-sampler-filtered
    /// distributions; the first reject commits a residual `max(0, p−q)` draw
    /// (falling back to `p` on ~0 mass), full accept a bonus draw from the
    /// last row — committed tokens are distributed exactly as filtered target
    /// sampling. Fully on device: one batched filter launch over the verify
    /// logits + one chain kernel; the host receives only `[accepted_len,
    /// token]` (8 bytes) plus one 4-byte filtered prob per committed token
    /// (the P6 behavior logprob, read after the verdict sync). All uniforms
    /// come from host salted `(seed, position)` streams, so same-config-twice
    /// reproduces.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_accept_commit_sampled(
        &self,
        slot: &mut Qwen35SlotState,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        head: &Qwen35DsparkHead,
        scratch: &mut DsparkScratch,
        chain: &[u32],
        logits: &DeviceVec,
        start_pos: usize,
        params: &SamplingParams,
    ) -> Result<(Vec<super::CommittedToken>, u32, usize)> {
        let ctx = &self.ctx;
        let depth = chain.len() - 1;
        let block = head.cfg.block_size;
        ensure!(
            scratch.q_rows >= depth && depth <= block,
            "dspark sampled verify: {} q rows for depth {depth} (block {block})",
            scratch.q_rows
        );
        let vocab = self.output_projection().rows;
        // Uniform streams at pos = start_pos + j + 1 (identical to the host
        // path's per-step draws — position-salted, so batching changes nothing).
        let pos = |j: usize| (start_pos + j + 1) as u64;
        let u_acc: Vec<f32> = (0..depth)
            .map(|j| unit_uniform(params.seed, SALT_ACCEPT, pos(j)))
            .collect();
        let u_res: Vec<f32> = (0..=depth)
            .map(|j| unit_uniform(params.seed, SALT_RESIDUAL, pos(j)))
            .collect();
        let draft: Vec<i32> = chain[1..].iter().map(|&t| t as i32).collect();

        let p_all = scratch.p_probs.get(ctx, (block + 1) * vocab)?;
        let q_all = scratch.q_probs.get(ctx, block * vocab)?;
        let draft_dev = scratch.chain_draft.get(ctx, block)?;
        let ua_dev = scratch.u_accept.get(ctx, block)?;
        let ur_dev = scratch.u_residual.get(ctx, block + 1)?;
        let out_dev = scratch.accept_out.get(ctx, 2)?;
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
            .map_err(|e| anyhow!("H2D dspark chain inputs failed: {e}"))?;
        {
            let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
            let (p_ptr, _gp) = p_all.device_ptr_mut(&ctx.stream);
            let (q_ptr, _gq) = q_all.device_ptr(&ctx.stream);
            let (d_ptr, _gd) = draft_dev.device_ptr(&ctx.stream);
            let (ua_ptr, _gua) = ua_dev.device_ptr(&ctx.stream);
            let (ur_ptr, _gur) = ur_dev.device_ptr(&ctx.stream);
            let (o_ptr, _go) = out_dev.device_ptr_mut(&ctx.stream);
            // SAFETY: logits holds chain.len()*vocab bf16; p/q scratches hold
            // (block+1)/block vocab-rows and depth <= block (ensured above);
            // draft/u prefixes uploaded just above.
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
            .map_err(|e| anyhow!("D2H dspark chain verdict failed: {e}"))?;
        let (k, bonus) = (out[0] as usize, out[1] as u32);
        ensure!(
            k <= depth,
            "dspark chain kernel returned k {k} > depth {depth}"
        );
        let mut tokens: Vec<u32> = chain[1..=k].to_vec();
        tokens.push(bonus);
        // Behavior logprobs from the still-materialized filtered p rows
        // (verdict D2H synced above) — see `chain_commit_logprobs`.
        let logprobs = super::chain_commit_logprobs(ctx, p_all, vocab, &tokens)?;
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
