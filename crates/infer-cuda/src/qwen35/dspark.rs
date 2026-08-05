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
use qwen35_spec::{DsparkConfig, DsparkSps, dspark_tensor_names, dspark_verify_lens};

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
    /// Verify-step cost model driving the goodput budget (`--dspark-sps-*-ms`).
    sps: DsparkSps,
    /// Draft RoPE tables (full rotary over head_dim, `rope_theta`), `rope_cap`
    /// absolute positions.
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Per-layer ctx-cache rows (== per-head cache stride): `sliding_window +
    /// block_size` addressed as an absolute-position ring (`row = pos % cap`).
    cap: usize,
    /// RoPE table length = `min(max_seq_len, max_total_tokens) + block_size` —
    /// the per-request token ceiling, NOT the whole KV-pool `max_seq_len`.
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
    /// Per-slot bytes: draft ctx K/V plus the draft outputs. All lazily
    /// allocated, so they must be reserved out of the KV budget or the first
    /// dspark step OOMs behind an already-sized pool.
    pub(crate) fn slot_state_bytes(&self, vocab: usize) -> usize {
        let per_head = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let ctx = 2 * self.cap * per_head * self.layers.len() * std::mem::size_of::<bf16>();
        let draft = self.cfg.block_size
            * vocab
            * (std::mem::size_of::<bf16>() + std::mem::size_of::<f32>());
        ctx + draft
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

    /// Hot-swap the Markov head weights from a host f32 snapshot.
    ///
    /// Called by the train sidecar after each acceptance-weighted step. The new
    /// weights are uploaded to a staging buffer, then the `data` pointer is
    /// flipped under the device stream's ordering guarantee (the hot path
    /// reads `markov.w1`/`markov.w2` only after a stream sync at step entry).
    /// `w1` is `[vocab * rank]`, `w2` is `[vocab * rank]` (both `[vocab, rank]`
    /// row-major — the `gemm_batch` weight frame).
    pub(crate) fn update_markov_weights(
        &mut self,
        ctx: &DeviceContext,
        w1: &[f32],
        w2: &[f32],
    ) -> Result<()> {
        let markov = self
            .markov
            .as_mut()
            .ok_or_else(|| anyhow!("dspark head has no Markov head to update"))?;
        let w1_len = markov.w1.rows * markov.w1.cols;
        let w2_len = markov.w2.rows * markov.w2.cols;
        ensure!(
            w1.len() == w1_len,
            "markov w1 size mismatch: got {}, expected {w1_len}",
            w1.len()
        );
        ensure!(
            w2.len() == w2_len,
            "markov w2 size mismatch: got {}, expected {w2_len}",
            w2.len()
        );
        // The bias is added in bf16, so under half an ulp of the base logit it is
        // discarded whole and the head is a no-op no matter how well it trained.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        log::info!(
            "dspark markov publish: rms|w1|={:.3e} rms|w2|={:.3e} est|bias|={:.3e} (bf16 floor ~3e-2)",
            rms(w1),
            rms(w2),
            (markov.rank as f32).sqrt() * rms(w1) * rms(w2)
        );
        let w1_bf16: Vec<half::bf16> = w1.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let w2_bf16: Vec<half::bf16> = w2.iter().map(|&x| half::bf16::from_f32(x)).collect();
        markov.w1.data = ctx
            .stream
            .clone_htod(&w1_bf16)
            .map_err(|e| anyhow!("markov w1 H2D upload failed: {e}"))?;
        markov.w2.data = ctx
            .stream
            .clone_htod(&w2_bf16)
            .map_err(|e| anyhow!("markov w2 H2D upload failed: {e}"))?;
        ctx.sync()?;
        Ok(())
    }
}

/// Per-slot draft context K/V cache: for each draft layer, the RoPE'd/normed K
/// and raw V of accepted positions, laid out `[kv_head][cap][head_dim]`.
/// All layers are sliding-window: an absolute-position RING (`row = abs % cap`,
/// `cap = window + block`), keeping only the last `window` keys the sliding
/// attention ever reads. Noise rows are written speculatively at their positions
/// and self-heal: rejected rows are overwritten by the next block's ctx/noise
/// writes before any read. A ring never aliases a live key within a block — the
/// live span (window ctx + block noise) equals `cap`, so distinct positions
/// never collide mod `cap`.
pub(crate) struct Qwen35DsparkSlotState {
    k_ctx: Vec<DeviceVec>,
    v_ctx: Vec<DeviceVec>,
    /// Absolute trunk position of ctx buffer row 0. Non-zero after a
    /// prefix-restore rebase: the suffix-only ctx is exact once the tail ≥ window.
    pub(crate) ctx_base: usize,
    /// Absolute end (exclusive) of materialized ctx rows; buffer holds
    /// `ctx_end - ctx_base` rows. The spec step requires `ctx_end == start_pos`.
    pub(crate) ctx_end: usize,
    /// Last emitted token (the next block's anchor), staged by prefill/verify.
    pub(crate) pending: Option<u32>,
    /// Draft outputs that outlive the draft call — a batched step drafts every
    /// slot before verifying any, so a shared scratch would hand row `i`'s
    /// accept row `B-1`'s distributions. Lazy: greedy never allocates `q_probs`.
    pub(crate) logits: HiddenSlot,
    q_probs: SliceSlot<f32>,
    q_rows: usize,
}

impl Qwen35DsparkSlotState {
    pub(crate) fn new(ctx: &DeviceContext, head: &Qwen35DsparkHead) -> Result<Self> {
        let per_head = head.cfg.num_key_value_heads * head.cfg.head_dim;
        let bytes = head.cap * per_head;
        let alloc = || -> Result<Vec<DeviceVec>> {
            (0..head.layers.len())
                .map(|_| DeviceVec::zeros(ctx, bytes))
                .collect()
        };
        Ok(Self {
            k_ctx: alloc()?,
            v_ctx: alloc()?,
            ctx_base: 0,
            ctx_end: 0,
            pending: None,
            logits: HiddenSlot::default(),
            q_probs: SliceSlot::default(),
            q_rows: 0,
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

    /// Captured tap `i` as host f32, token-major `[seq, hidden]`. Offline draft
    /// training reads the taps raw; serving fuses them on device through
    /// [`Qwen35Model::dspark_tap_features`] instead.
    pub(crate) fn tap_to_host(&mut self, ctx: &DeviceContext, i: usize) -> Result<Vec<f32>> {
        ensure!(self.armed, "dspark tap readback without an armed capture");
        ensure!(i < self.bufs.len(), "tap {i} of {}", self.bufs.len());
        self.bufs[i].get(ctx, self.hidden, self.seq)?.to_host(ctx)
    }

    pub(crate) fn release(&mut self) {
        self.disarm();
    }
}

/// Draft-side persistent scratch, one per executor — every buffer here is
/// written then read inside one draft or one accept, and those never
/// interleave. Separate from [`Qwen35Workspace`] so the shapes never thrash.
#[derive(Default)]
pub(crate) struct DsparkScratch {
    ids: SliceSlot<i32>,
    /// Absolute (unshifted) start position for sliding-ring prep launches.
    start_pos_abs: SliceSlot<i32>,
    /// Per-row draft attention windows, `[ring_base; block] ++ [kv_len; block]`
    /// — identical for every layer, so one upload serves the whole forward.
    attn_win: SliceSlot<i32>,
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
    /// Row-shaped twins of the four above, grouped so a caller can borrow the
    /// base logits out of this same scratch and pass these beside them.
    mk: MarkovScratch,
    conf: ConfScratch,
    /// Batched-draft logits `[B*block, vocab]`; per-slot `df.logits` copy from here.
    logits_b: HiddenSlot,
    /// Row count of the tap features held in `feat_b` (the whole batch).
    feat_seq: usize,
    /// Sampled-mode device buffers (allocated on first temp>0 spec step only;
    /// greedy never touches them). Fixed caps — no per-step realloc:
    /// `p_probs [block+1, vocab] f32` verify filtered dists (leading
    /// `chain_len` rows fully written per accept; the stale tail is never
    /// read — the chain kernel indexes rows `<= depth < chain_len`);
    /// `sample_tok [1]` / `accept_out [2]` fully written before D2H;
    /// `chain_draft [block]` / `u_accept [block]` / `u_residual [block+1]`
    /// host-uploaded prefixes — the kernel reads only the uploaded prefix.
    p_probs: SliceSlot<f32>,
    sample_tok: SliceSlot<i32>,
    accept_out: SliceSlot<i32>,
    chain_draft: SliceSlot<i32>,
    u_accept: SliceSlot<f32>,
    u_residual: SliceSlot<f32>,
}

#[derive(Default)]
struct MarkovScratch {
    prevs: SliceSlot<i32>,
    emb: HiddenSlot,
    bias: HiddenSlot,
    sum: HiddenSlot,
    ids: SliceSlot<i32>,
}

/// Own `prevs`/`emb`: the settle's are `block`-shaped, these `block - first_row`.
#[derive(Default)]
struct ConfScratch {
    prevs: SliceSlot<i32>,
    emb: HiddenSlot,
    feat: HiddenSlot,
    out: HiddenSlot,
    copy: Qwen35CopyScratch,
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
    /// Pointer tables for the batched rollback replay.
    pub(crate) replay_tables: Qwen35ReplayTables,
    /// Staging for the batched state snapshot/restore.
    pub(crate) copy: Qwen35CopyScratch,
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
            replay_tables: Qwen35ReplayTables::default(),
            copy: Qwen35CopyScratch::default(),
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
    sps: DsparkSps,
    markov_head_rank: Option<usize>,
    block_size_cap: Option<usize>,
) -> Result<Qwen35DsparkHead> {
    let mut cfg = DsparkConfig::from_dir(dir)
        .map_err(|e| anyhow!("dspark draft config at {}: {e}", dir.display()))?;
    // A block longer than the accepted prefix is pure waste: the chain stops at
    // the first rejection, so every position past it costs a draft forward and a
    // verify row and can never be committed. Measured on TC-27B + DFlash:
    // accept_rate 0.205 at block 16 keeps 3.28 tokens, discarding 79.5% of the
    // drafted work. Clamping here is the single correct point — rope_cap, the ctx
    // ring, and the per-slot scratch all size off `cfg.block_size`.
    if let Some(cap) = block_size_cap {
        let capped = cfg.block_size.min(cap.max(1));
        if capped != cfg.block_size {
            log::info!(
                "CUDA Qwen3.6 DSpark: block_size {} -> {capped} (--dspark-block-size)",
                cfg.block_size
            );
            cfg.block_size = capped;
        }
    }
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
        .map(|n| {
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
                    gate_up_proj: loader.load_matrix_pair_fused(ctx, &n.gate_proj, &n.up_proj)?,
                    down_proj: loader.load_matrix(ctx, &n.down_proj)?,
                },
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
    } else if let Some(rank) = markov_head_rank {
        // `--dspark-markov-init` installs a head over a DFlash backbone that
        // ships none; only the slot's shape matters, both halves are overwritten.
        ensure!(rank > 0, "dspark markov head rank must be positive");
        let zeros = vec![0u8; trunk_vocab * rank * 2];
        Some(DsparkMarkovHead {
            w1: DeviceMatrix::from_safetensors(ctx, &zeros, trunk_vocab, rank)?,
            w2: DeviceMatrix::from_safetensors(ctx, &zeros, trunk_vocab, rank)?,
            rank,
        })
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

    // All draft layers are sliding-window: fixed `window + block` ring cache.
    // RoPE tables cover the full per-request token ceiling (absolute positions).
    let rope_cap = max_seq_len.min(max_total_tokens.max(1)) + cfg.block_size;
    let cap = cfg.sliding_window + cfg.block_size;
    let full = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, qwen35_spec::DsparkLayerType::Full))
        .count();
    if full > 0 {
        // Honoring them needs a ctx ring the length of the request (671 MB/slot
        // at 32k vs 42 MB), so they are windowed on purpose — but silently
        // windowing a full-attention layer moves acceptance, so say it.
        log::warn!(
            "dspark draft: {full}/{} layers declare full attention; all run the \
             {}-token sliding window (ctx ring is {cap} rows)",
            cfg.layer_types.len(),
            cfg.sliding_window
        );
    }
    let (cos_cache, sin_cache) =
        crate::ops::precompute_rope(ctx, cfg.head_dim, rope_cap, cfg.rope_theta, None)?;
    Ok(Qwen35DsparkHead {
        sps,
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
        rope_cap,
    })
}

/// One partially-accepted chain to rewind (`k` of its depth accepted).
pub(crate) struct DsparkRollback<'a> {
    pub(crate) slot: &'a mut Qwen35SlotState,
    pub(crate) spec: &'a mut Qwen35SpecSlotState,
    pub(crate) start_pos: usize,
    pub(crate) k: usize,
}

/// Elementwise add over two same-length device buffers.
fn add_vec_into(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch(ctx, a, b, &mut out)?;
    Ok(out)
}

impl Qwen35Model {
    /// `feat = hidden_norm(Σ_t fc_t · tap_t)` over the WHOLE tap — the fc GEMMs
    /// are batch-wide, so this runs once per forward. Disarms the tap.
    pub(crate) fn dspark_tap_features(
        &self,
        head: &Qwen35DsparkHead,
        taps: &mut Qwen35DsparkTaps,
        scratch: &mut DsparkScratch,
    ) -> Result<()> {
        ensure!(taps.armed, "dspark ctx append without an armed tap capture");
        let ctx = &self.ctx;
        let hidden = head.cfg.hidden_size;
        let seq = taps.seq;
        let eps = head.cfg.rms_norm_eps;
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
        scratch.feat_seq = seq;
        taps.disarm();
        Ok(())
    }

    /// Append this slot's draft K/V at `[start, start+rows)` from tap feature
    /// rows `[tap_off, tap_off+rows)`.
    pub(crate) fn dspark_append_ctx(
        &self,
        head: &Qwen35DsparkHead,
        df: &mut Qwen35DsparkSlotState,
        scratch: &mut DsparkScratch,
        tap_off: usize,
        rows: usize,
        start: usize,
    ) -> Result<()> {
        ensure!(
            df.ctx_end == start,
            "dspark ctx append at {start} but ctx_end {} (draft cache not contiguous)",
            df.ctx_end
        );
        ensure!(
            rows >= 1 && tap_off + rows <= scratch.feat_seq,
            "dspark ctx rows {rows} at tap offset {tap_off} outside features {}",
            scratch.feat_seq
        );
        // Absolute positions must index the shared cos/sin table.
        ensure!(
            start + rows <= head.rope_cap,
            "dspark ctx overflow past rope_cap"
        );
        let ctx = &self.ctx;
        let hidden = head.cfg.hidden_size;
        let feat_seq = scratch.feat_seq;
        let feat = scratch.feat_b.get(ctx, hidden, feat_seq)?;
        let src = feat.data.slice(tap_off * hidden..(tap_off + rows) * hidden);
        let feat_rows = scratch.feat_rows.get(ctx, hidden, rows)?;
        ctx.stream
            .memcpy_dtod(&src, &mut feat_rows.data)
            .map_err(|e| anyhow!("dspark feat row copy failed: {e}"))?;

        let kv_dim = head.kv_dim();
        let start_abs = scratch.start_pos_abs.upload(&self.ctx, &[start as i32])?;
        let (sp_abs, _ga) = start_abs.device_ptr(&ctx.stream);
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.cap;
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

        let start_abs = scratch.start_pos_abs.upload(&self.ctx, &[start as i32])?;
        // Per-row attention windows. HF sliding window keeps keys with
        // q_pos - k_pos < window, never below `ctx_base`; the ctx buffer is an
        // absolute ring, so row `row` walks `[lo, start+block)` mapped through
        // `% cap`. `lo` reads only start/row/window/ctx_base — layer-independent,
        // so one table serves all 5 layers and one ragged-window launch replaces
        // `block` single-row ones.
        let mut win = vec![0i32; 2 * block];
        for row in 0..block {
            let lo = (start + row)
                .saturating_sub(cfg.sliding_window - 1)
                .max(df.ctx_base);
            let kv_len = kv_len_total - lo;
            // kv_len == q_pos+1−lo ≤ window+block == cap, so the ring read never
            // revisits a physical row (the kernel cannot check this host-side).
            ensure!(kv_len <= head.cap, "dspark draft row window {kv_len} > cap");
            win[row] = lo as i32;
            win[block + row] = kv_len as i32;
        }
        let win_dev = scratch.attn_win.upload(&self.ctx, &win)?;
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.cap;
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
            }

            prep_ms += super::mtp_phase_lap(ctx, &mut pt);
            // Non-causal attention: every noise row attends its whole window of
            // the `[ctx ++ block]` key range, in one ragged-window launch.
            let attn_heads = scratch.attn_heads.get(ctx, q_dim, block)?;
            {
                let sm_scale = 1.0 / (cfg.head_dim as f32).sqrt();
                let (q_ptr, _g0) = q_prepped.data.device_ptr(&ctx.stream);
                let (kc_ptr, _g1) = df.k_ctx[li].data.device_ptr(&ctx.stream);
                let (vc_ptr, _g2) = df.v_ctx[li].data.device_ptr(&ctx.stream);
                let (o_ptr, _g3) = attn_heads.data.device_ptr_mut(&ctx.stream);
                let (w_ptr, _g4) = win_dev.device_ptr(&ctx.stream);
                let nq = cfg.num_attention_heads as i32;
                let nkv = cfg.num_key_value_heads as i32;
                let hd = cfg.head_dim as i32;
                // SAFETY: `win` holds 2*block i32 (bases then lengths, each
                // length ≤ cap_li as asserted above).
                unsafe {
                    ffi::nonpaged_prefill_attention_ring_varlen_cuda(
                        q_ptr as *const ffi::Half,
                        kc_ptr as *const ffi::Half,
                        vc_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        nq,
                        nkv,
                        hd,
                        block as i32,
                        w_ptr as *const i32,
                        (w_ptr as *const i32).add(block),
                        cap_li as i32,
                        sm_scale,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
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
        let logits = df.logits.get(ctx, vocab, block)?;
        gemm_batch(ctx, self.output_projection(), final_normed, logits)?;
        let head_ms = super::mtp_phase_lap(ctx, &mut pt);

        // Left-to-right token selection: argmax (greedy) or a rejection-ready
        // device draw from the engine-sampler-filtered distribution q (full
        // row retained in `df.q_probs` — only the token id comes back).
        // Next-token heads (DSpark) draft from every row; same-position
        // (DFlash) rows 1.. fill their own positions.
        let sampling = !params.is_greedy();
        df.q_rows = 0;
        let first_row = usize::from(!cfg.next_token_heads);
        let mut drafts = Vec::with_capacity(block);
        // Greedy resolves the whole block in one batched pass; sampling walks
        // the chain because a markov bias makes row r depend on row r-1's draw.
        if !sampling {
            let am = self.dspark_settle_rows(
                head.markov.as_ref(),
                df.logits.get(ctx, vocab, block)?,
                &mut scratch.mk,
                &[anchor],
                block,
                first_row,
            )?;
            drafts.extend_from_slice(&am[first_row..block]);
        }
        // No markov head: every row reads a precomputed logits row, so the
        // draws do not depend on each other — issue them all, then sync once.
        if sampling && head.markov.is_none() {
            let n = block - first_row;
            {
                let elem = std::mem::size_of::<bf16>() as u64;
                let logits = df.logits.get(ctx, vocab, block)?;
                let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
                let q_all = df.q_probs.get(ctx, n * vocab)?;
                let (q_ptr, _gq) = q_all.device_ptr_mut(&ctx.stream);
                let (t_ptr, _gt) = scratch.sample_tok.get(ctx, n)?.device_ptr_mut(&ctx.stream);
                for i in 0..n {
                    let u = unit_uniform(params.seed, SALT_DRAW, (start + first_row + i) as u64);
                    // SAFETY: logits row `first_row + i` of `block`; q row `i`
                    // of `n`; one i32 out slot per row.
                    unsafe {
                        ffi::dspark_draft_sample_cuda(
                            (l_ptr + ((first_row + i) * vocab) as u64 * elem) as *const ffi::Half,
                            (q_ptr + (i * vocab * 4) as u64) as *mut f32,
                            (t_ptr + (i * 4) as u64) as *mut i32,
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
            }
            ctx.sync()?;
            let drawn: Vec<i32> = ctx
                .stream
                .clone_dtoh(scratch.sample_tok.get(ctx, n)?)
                .map_err(|e| anyhow!("D2H dspark draws failed: {e}"))?;
            for tok in drawn {
                ensure!(
                    (0..vocab as i32).contains(&tok),
                    "dspark sampled draft token {tok} oob"
                );
                drafts.push(tok as u32);
            }
            df.q_rows = n;
        }
        let mut prev = anchor;
        let sample_rows = if sampling && head.markov.is_some() {
            first_row..block
        } else {
            0..0
        };
        for row in sample_rows {
            // Corrected logits row: base + markov bias when the head is present
            // (`step_logits = base_row + markov_w2 · markov_w1[prev]`).
            let (src, src_row) = if let Some(m) = &head.markov {
                let tok_dev = scratch.markov_tok.upload(ctx, &[prev as i32])?;
                let emb = scratch.markov_emb.get(ctx, m.rank, 1)?;
                embedding_batch(ctx, &m.w1, tok_dev, emb)?;
                let bias = scratch.markov_bias.get(ctx, vocab, 1)?;
                gemm_batch(ctx, &m.w2, emb, bias)?;
                let step = scratch.step_logits.get(ctx, vocab, 1)?;
                {
                    let logits = df.logits.get(ctx, vocab, block)?;
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
                (df.logits.get(ctx, vocab, block)?, row)
            };
            let tok = {
                // Filter + q-row store + multinomial draw in one device call;
                // uniform from the host (seed, position) stream plain decode
                // would consume at this position (SALT_DRAW = 0).
                let u = unit_uniform(params.seed, SALT_DRAW, (start + row) as u64);
                let q_all = df.q_probs.get(ctx, block * vocab)?;
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
                df.q_rows += 1;
                tok
            };
            drafts.push(tok);
            prev = tok;
        }

        // Confidence seam: goodput-budget the proposal (R=1 on this per-slot
        // path); absent head = keep the whole block.
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
        let keep = self.dspark_verify_keeps(
            head,
            scratch.final_normed.get(ctx, hidden, block)?,
            &mut scratch.conf,
            &prev_tokens,
            1,
            block,
            first_row,
        )?[0];
        drafts.truncate(keep.min(drafts.len()));

        let mut chain = Vec::with_capacity(1 + drafts.len());
        chain.push(anchor);
        chain.extend(drafts);
        Ok(chain)
    }

    /// Batched [`Self::dspark_draft_block`]: one forward over every slot's
    /// block. The draft GEMMs are weight-bound at `block` rows, so B serial
    /// drafts re-read the same weights B times. Only the ring kernels stay
    /// per-slot. Greedy only — a sampled draw syncs per row.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_draft_blocks(
        &self,
        head: &Qwen35DsparkHead,
        dfs: &mut [&mut Qwen35DsparkSlotState],
        scratch: &mut DsparkScratch,
        anchors: &[u32],
        starts: &[usize],
        params: &[&SamplingParams],
    ) -> Result<Vec<Vec<u32>>> {
        debug_assert!(
            params.iter().all(|p| p.is_greedy()),
            "batched draft is greedy-only"
        );
        let cfg = &head.cfg;
        let ctx = &self.ctx;
        let block = cfg.block_size;
        let b = dfs.len();
        ensure!(
            b == anchors.len() && b == starts.len() && b == params.len(),
            "dspark batched draft: {b} slots vs {} anchors / {} starts / {} params",
            anchors.len(),
            starts.len(),
            params.len()
        );
        let rows = b * block;
        let hidden = cfg.hidden_size;
        let (q_dim, kv_dim) = (head.q_dim(), head.kv_dim());
        let eps = cfg.rms_norm_eps;

        let mut ids = vec![cfg.mask_token_id as i32; rows];
        // `[all bases][all lengths]`: one upload for every slot and layer.
        let mut win = vec![0i32; 2 * rows];
        let mut pos = vec![0i32; b];
        for (s, df) in dfs.iter().enumerate() {
            let start = starts[s];
            ensure!(df.ctx_end == start, "dspark draft: ctx_end != start");
            ensure!(
                start + block <= head.rope_cap,
                "dspark draft past cache cap"
            );
            ids[s * block] = anchors[s] as i32;
            pos[s] = start as i32;
            for row in 0..block {
                let lo = (start + row)
                    .saturating_sub(cfg.sliding_window - 1)
                    .max(df.ctx_base);
                let kv_len = start + block - lo;
                ensure!(kv_len <= head.cap, "dspark draft row window {kv_len} > cap");
                win[s * block + row] = lo as i32;
                win[rows + s * block + row] = kv_len as i32;
            }
        }

        let mut pt = super::dspark_phase_start(ctx);
        let ids_dev = scratch.ids.upload(ctx, &ids)?;
        let h = scratch.hidden.get(ctx, hidden, rows)?;
        embedding_batch(ctx, &self.embed_tokens, ids_dev, h)?;
        let pos_dev = scratch.start_pos_abs.upload(ctx, &pos)?;
        let win_dev = scratch.attn_win.upload(ctx, &win)?;
        let embed_ms = super::mtp_phase_lap(ctx, &mut pt);

        let elem = std::mem::size_of::<half::f16>() as u64;
        let (mut qkv_ms, mut attn_ms, mut mlp_ms) = (0.0f64, 0.0f64, 0.0f64);
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.cap;
            let h = scratch.hidden.get(ctx, hidden, rows)?;
            let normed = scratch.normed.get(ctx, hidden, rows)?;
            rms_norm_batch(ctx, h, &layer.input_layernorm, eps, normed)?;

            let q_full = scratch.q_full.get(ctx, 2 * q_dim, rows)?;
            gemm_batch(ctx, &layer.q_proj, normed, q_full)?;
            let k_new = scratch.k_new.get(ctx, kv_dim, rows)?;
            gemm_batch(ctx, &layer.k_proj, normed, k_new)?;
            let v_new = scratch.v_new.get(ctx, kv_dim, rows)?;
            gemm_batch(ctx, &layer.v_proj, normed, v_new)?;
            qkv_ms += super::mtp_phase_lap(ctx, &mut pt);

            let q_prepped = scratch.q_prepped.get(ctx, q_dim, rows)?;
            let attn_heads = scratch.attn_heads.get(ctx, q_dim, rows)?;
            {
                let (qf_ptr, _g0) = q_full.data.device_ptr(&ctx.stream);
                let (k_ptr, _g1) = k_new.data.device_ptr(&ctx.stream);
                let (v_ptr, _g2) = v_new.data.device_ptr(&ctx.stream);
                let (qn_ptr, _g3) = layer.q_norm.data.device_ptr(&ctx.stream);
                let (kn_ptr, _g4) = layer.k_norm.data.device_ptr(&ctx.stream);
                let (cos_ptr, _g5) = head.cos_cache.data.device_ptr(&ctx.stream);
                let (sin_ptr, _g6) = head.sin_cache.data.device_ptr(&ctx.stream);
                let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&ctx.stream);
                let (ao_ptr, _g8) = attn_heads.data.device_ptr_mut(&ctx.stream);
                let (w_ptr, _g9) = win_dev.device_ptr(&ctx.stream);
                let (sp_ptr, _g10) = pos_dev.device_ptr(&ctx.stream);
                let nq = cfg.num_attention_heads as i32;
                let nkv = cfg.num_key_value_heads as i32;
                let hd = cfg.head_dim as i32;
                let sm_scale = 1.0 / (cfg.head_dim as f32).sqrt();
                for (s, df) in dfs.iter_mut().enumerate() {
                    let off = (s * block) as u64;
                    let (kc_ptr, _gk) = df.k_ctx[li].data.device_ptr_mut(&ctx.stream);
                    let (vc_ptr, _gv) = df.v_ctx[li].data.device_ptr_mut(&ctx.stream);
                    // SAFETY: offset by this slot's `block` rows inside a
                    // `rows`-row buffer; ring bounds guarded above.
                    unsafe {
                        ffi::prefill_attention_hd256_prep_ring_cuda(
                            (qf_ptr + off * 2 * q_dim as u64 * elem) as *const ffi::Half,
                            (k_ptr + off * kv_dim as u64 * elem) as *const ffi::Half,
                            (v_ptr + off * kv_dim as u64 * elem) as *const ffi::Half,
                            qn_ptr as *const ffi::Half,
                            kn_ptr as *const ffi::Half,
                            cos_ptr as *const ffi::Half,
                            sin_ptr as *const ffi::Half,
                            (qp_ptr + off * q_dim as u64 * elem) as *mut ffi::Half,
                            kc_ptr as *mut ffi::Half,
                            vc_ptr as *mut ffi::Half,
                            nq,
                            nkv,
                            hd,
                            block as i32,
                            (sp_ptr + s as u64 * 4) as *const i32,
                            hd,
                            eps,
                            cap_li as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    // SAFETY: q/out offset as above; the window table is
                    // `rows` bases then `rows` lengths.
                    unsafe {
                        ffi::nonpaged_prefill_attention_ring_varlen_cuda(
                            (qp_ptr + off * q_dim as u64 * elem) as *const ffi::Half,
                            kc_ptr as *const ffi::Half,
                            vc_ptr as *const ffi::Half,
                            (ao_ptr + off * q_dim as u64 * elem) as *mut ffi::Half,
                            nq,
                            nkv,
                            hd,
                            block as i32,
                            (w_ptr as *const i32).add(s * block),
                            (w_ptr as *const i32).add(rows + s * block),
                            cap_li as i32,
                            sm_scale,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
            }

            attn_ms += super::mtp_phase_lap(ctx, &mut pt);
            let attn_out_h = scratch.attn_out_h.get(ctx, hidden, rows)?;
            gemm_batch(ctx, &layer.o_proj, attn_heads, attn_out_h)?;
            let hidden_mid = scratch.hidden_mid.get(ctx, hidden, rows)?;
            add_batch(
                ctx,
                scratch.hidden.get(ctx, hidden, rows)?,
                attn_out_h,
                hidden_mid,
            )?;
            let normed = scratch.normed.get(ctx, hidden, rows)?;
            rms_norm_batch(
                ctx,
                hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                normed,
            )?;
            let mlp_out = scratch.attn_out_h.get(ctx, hidden, rows)?;
            self.dense_mlp(&layer.mlp, normed, &mut scratch.dense, mlp_out)?;
            add_batch(
                ctx,
                scratch.hidden_mid.get(ctx, hidden, rows)?,
                mlp_out,
                scratch.hidden.get(ctx, hidden, rows)?,
            )?;
            mlp_ms += super::mtp_phase_lap(ctx, &mut pt);
        }
        let layers_ms = qkv_ms + attn_ms + mlp_ms;

        let final_normed = scratch.final_normed.get(ctx, hidden, rows)?;
        rms_norm_batch(
            ctx,
            scratch.hidden.get(ctx, hidden, rows)?,
            &head.norm,
            eps,
            final_normed,
        )?;
        let vocab = self.output_projection().rows;
        let logits = scratch.logits_b.get(ctx, vocab, rows)?;
        gemm_batch(ctx, self.output_projection(), final_normed, logits)?;
        let head_ms = super::mtp_phase_lap(ctx, &mut pt);

        // One D2H for the whole tick, and one markov settle over every slot.
        let first_row = usize::from(!cfg.next_token_heads);
        let am = self.dspark_settle_rows(
            head.markov.as_ref(),
            scratch.logits_b.get(ctx, vocab, rows)?,
            &mut scratch.mk,
            anchors,
            block,
            first_row,
        )?;
        let settle_ms = super::mtp_phase_lap(ctx, &mut pt);

        let n = block - first_row;
        let mut prevs = Vec::with_capacity(b * n);
        for (s, df) in dfs.iter_mut().enumerate() {
            df.q_rows = 0;
            let src = scratch
                .logits_b
                .get(ctx, vocab, rows)?
                .data
                .slice(s * block * vocab..(s + 1) * block * vocab);
            let dst = df.logits.get(ctx, vocab, block)?;
            ctx.stream
                .memcpy_dtod(&src, &mut dst.data)
                .map_err(|e| anyhow!("dspark batched draft logits copy failed: {e}"))?;
            if n > 0 {
                prevs.push(anchors[s]);
                prevs.extend_from_slice(&am[s * block + first_row..(s + 1) * block - 1]);
            }
        }
        let keeps = self.dspark_verify_keeps(
            head,
            scratch.final_normed.get(ctx, hidden, rows)?,
            &mut scratch.conf,
            &prevs,
            b,
            block,
            first_row,
        )?;
        if pt.is_some() {
            let conf_ms = super::mtp_phase_lap(ctx, &mut pt);
            eprintln!(
                "[dspark-draft-b] rows={rows} embed={embed_ms:.2} qkv={qkv_ms:.2} \
                 attn={attn_ms:.2} mlp={mlp_ms:.2} layers={layers_ms:.2} \
                 head={head_ms:.2} settle={settle_ms:.2} conf={conf_ms:.2} ms"
            );
        }
        let mut chains = Vec::with_capacity(b);
        for (s, &anchor) in anchors.iter().enumerate() {
            let drafts = &am[s * block + first_row..(s + 1) * block];
            let mut chain = Vec::with_capacity(1 + drafts.len());
            chain.push(anchor);
            chain.extend(&drafts[..keeps[s].min(drafts.len())]);
            chains.push(chain);
        }
        Ok(chains)
    }

    /// Greedy token for every row of a `[vocab, b*block]` draft output. A markov
    /// `bias = w2·w1[prev]` makes row r depend on row r-1, so the chain
    /// speculates on itself: the base argmax is a near-perfect guess for `prev`,
    /// so guess every predecessor, correct all rows at once, and re-run while a
    /// guess disagreed. Each round confirms one more row, so `block` bounds it.
    ///
    /// Every slot settles in the same rounds — `w2` is `[vocab, rank]`, so the
    /// GEMM is weight-bound and B serial settles re-read it B times. Costs two
    /// `[vocab, b*block]` buffers, 48 MB each at block 6 / B 16.
    fn dspark_settle_rows(
        &self,
        m: Option<&DsparkMarkovHead>,
        base: &HiddenStates,
        mk: &mut MarkovScratch,
        anchors: &[u32],
        block: usize,
        first_row: usize,
    ) -> Result<Vec<u32>> {
        let ctx = &self.ctx;
        let (rows, vocab) = (base.seq_len, base.hidden_dim);
        let b = anchors.len();
        ensure!(
            rows == b * block,
            "dspark settle: {rows} rows != {b} slots x {block}"
        );
        let ids = mk.ids.get(ctx, rows)?;
        let mut toks = self.argmax_rows_into(base, ids)?;
        let Some(m) = m else {
            return Ok(toks);
        };
        // The anchor feeds the slot's first drafted row, then its own tokens.
        let mut prevs = vec![0i32; rows];
        for _ in 0..block {
            for (s, &anchor) in anchors.iter().enumerate() {
                for row in 0..block {
                    prevs[s * block + row] = if row <= first_row {
                        anchor as i32
                    } else {
                        toks[s * block + row - 1] as i32
                    };
                }
            }
            let tok_dev = mk.prevs.upload(ctx, &prevs)?;
            let emb = mk.emb.get(ctx, m.rank, rows)?;
            embedding_batch(ctx, &m.w1, tok_dev, emb)?;
            let bias = mk.bias.get(ctx, vocab, rows)?;
            gemm_batch(ctx, &m.w2, mk.emb.get(ctx, m.rank, rows)?, bias)?;
            let sum = mk.sum.get(ctx, vocab, rows)?;
            add_batch(ctx, base, mk.bias.get(ctx, vocab, rows)?, sum)?;
            let ids = mk.ids.get(ctx, rows)?;
            let next = self.argmax_rows_into(mk.sum.get(ctx, vocab, rows)?, ids)?;
            // "the correction moved no row another row's bias depends on".
            let settled = (0..b).all(|s| {
                (first_row..block.saturating_sub(1))
                    .all(|r| toks[s * block + r] == next[s * block + r])
            });
            toks = next;
            if settled {
                return Ok(toks);
            }
        }
        Err(anyhow!(
            "dspark markov chain failed to settle in {block} rounds"
        ))
    }

    /// Per-slot draft-keep lengths: confidence
    /// `sigmoid(proj([hidden_i ; markov_w1[prev_i]]))` over a `[hidden,
    /// b*block]` draft forward, cumprod'd into survival and fed to the goodput
    /// budget ([`dspark_verify_lens`]). `prevs` is slot-major, `b*(block -
    /// first_row)` long. Head absent → the whole block survives. One GEMM, two
    /// copy launches and one sync for the batch.
    #[allow(clippy::too_many_arguments)]
    fn dspark_verify_keeps(
        &self,
        head: &Qwen35DsparkHead,
        final_normed: &HiddenStates,
        cs: &mut ConfScratch,
        prevs: &[u32],
        b: usize,
        block: usize,
        first_row: usize,
    ) -> Result<Vec<usize>> {
        let n = block - first_row;
        let Some(conf) = &head.confidence else {
            return Ok(vec![usize::MAX; b]);
        };
        if n == 0 || b == 0 {
            return Ok(vec![usize::MAX; b]);
        }
        ensure!(
            prevs.len() == b * n,
            "dspark confidence: {} prevs != {b} slots x {n}",
            prevs.len()
        );
        let ctx = &self.ctx;
        let hidden = head.cfg.hidden_size;
        let in_dim = conf.weight.cols;
        let elem = std::mem::size_of::<bf16>();
        let total = b * n;
        // batched_copy moves 16B words and checks its sizes, not these strides.
        ensure!(
            (in_dim * elem).is_multiple_of(16) && (hidden * elem).is_multiple_of(16),
            "dspark confidence rows {in_dim}/{hidden} are not 16B-aligned"
        );
        // Feature row `s*n + i` reads the drafted row `s*block + first_row + i`.
        let fbase = cs
            .feat
            .get(ctx, in_dim, total)?
            .data
            .device_ptr_mut(&ctx.stream)
            .0;
        let hbase = final_normed.data.device_ptr(&ctx.stream).0;
        let (mut dst, mut src) = (Vec::with_capacity(total), Vec::with_capacity(total));
        for s in 0..b {
            for i in 0..n {
                dst.push(fbase + ((s * n + i) * in_dim * elem) as u64);
                src.push(hbase + ((s * block + first_row + i) * hidden * elem) as u64);
            }
        }
        self.batched_copy(&mut cs.copy, &dst, &src, &[hidden * elem])?;
        if conf.with_markov {
            let m = head.markov.as_ref().expect("validated at load");
            let toks: Vec<i32> = prevs.iter().map(|&t| t as i32).collect();
            let tok_dev = cs.prevs.upload(ctx, &toks)?;
            let emb = cs.emb.get(ctx, m.rank, total)?;
            embedding_batch(ctx, &m.w1, tok_dev, emb)?;
            let ebase = cs
                .emb
                .get(ctx, m.rank, total)?
                .data
                .device_ptr(&ctx.stream)
                .0;
            dst.clear();
            src.clear();
            for r in 0..total {
                dst.push(fbase + (r * in_dim * elem + hidden * elem) as u64);
                src.push(ebase + (r * m.rank * elem) as u64);
            }
            self.batched_copy(&mut cs.copy, &dst, &src, &[m.rank * elem])?;
        }
        let out = cs.out.get(ctx, 1, total)?;
        gemm_batch(ctx, &conf.weight, cs.feat.get(ctx, in_dim, total)?, out)?;
        let host = ctx
            .stream
            .clone_dtoh(&cs.out.get(ctx, 1, total)?.data)
            .map_err(|e| anyhow!("D2H dspark confidence failed: {e}"))?;
        ctx.sync()?;
        let survivals: Vec<Vec<f32>> = (0..b)
            .map(|s| {
                let mut acc = 1.0f32;
                host[s * n..(s + 1) * n]
                    .iter()
                    .map(|h| {
                        let logit = h.to_f32() + conf.bias;
                        acc *= 1.0 / (1.0 + (-logit).exp());
                        acc
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = survivals.iter().map(Vec::as_slice).collect();
        Ok(dspark_verify_lens(&refs, head.sps))
    }

    /// Size the reused per-slot spec state (`new_spec_slot_state` capture rows,
    /// verify-depth guard) for the DSpark block depth. Called once at attach.
    pub(crate) fn set_spec_draft_tokens(&mut self, n: usize) {
        self.spec_draft_tokens = n;
    }

    /// Rank of `token` in the draft's row-`row` logits (0 == the draft already
    /// proposed it). Answers "would a width-w candidate tree have survived this
    /// rejection" without building one. Opt-in probe: a whole-vocab D2H per call.
    pub(crate) fn draft_token_rank(
        &self,
        logits: &HiddenStates,
        row: usize,
        token: u32,
    ) -> Result<usize> {
        let vocab = logits.hidden_dim;
        ensure!(
            row < logits.seq_len && (token as usize) < vocab,
            "dspark rank probe: row {row} / token {token} outside [{}, {vocab})",
            logits.seq_len
        );
        let host: Vec<bf16> = self
            .ctx
            .stream
            .clone_dtoh(&logits.data.slice(row * vocab..(row + 1) * vocab))
            .map_err(|e| anyhow!("D2H dspark rank row failed: {e}"))?;
        let target = host[token as usize];
        Ok(host.iter().filter(|v| **v > target).count())
    }

    /// Per-row argmax over a whole verify output in one launch + one D2H. The
    /// accept scan is host arithmetic once this lands, so a batched tick costs
    /// one sync instead of one per chain row per slot.
    pub(crate) fn argmax_rows(&self, logits: &HiddenStates) -> Result<Vec<u32>> {
        let mut ids_dev = self
            .ctx
            .stream
            .alloc_zeros::<i32>(logits.seq_len)
            .map_err(|e| anyhow!("dspark argmax ids alloc failed: {e}"))?;
        self.argmax_rows_into(logits, &mut ids_dev)
    }

    /// [`Self::argmax_rows`] into caller-held scratch — the draft path calls it
    /// once per speculation round, so it must not allocate.
    pub(crate) fn argmax_rows_into(
        &self,
        logits: &HiddenStates,
        ids_dev: &mut CudaSlice<i32>,
    ) -> Result<Vec<u32>> {
        let ctx = &self.ctx;
        let (rows, vocab) = (logits.seq_len, logits.hidden_dim);
        ensure!(
            ids_dev.len() == rows,
            "dspark argmax scratch {} != {rows} rows",
            ids_dev.len()
        );
        {
            let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
            let (o_ptr, _go) = ids_dev.device_ptr_mut(&ctx.stream);
            // SAFETY: logits [rows, vocab] bf16; ids [rows] sized above.
            unsafe {
                ffi::argmax_batch_cuda(
                    l_ptr as *const ffi::Half,
                    o_ptr as *mut i32,
                    rows as i32,
                    vocab as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        let ids: Vec<i32> = ctx
            .stream
            .clone_dtoh(ids_dev)
            .map_err(|e| anyhow!("D2H dspark argmax failed: {e}"))?;
        ids.into_iter()
            .map(|id| {
                ensure!((0..vocab as i32).contains(&id), "dspark argmax id {id} oob");
                Ok(id as u32)
            })
            .collect()
    }

    /// Accept scan + trunk rollback over a DSpark verify (`spec_step` steps 4-5
    /// with the draft source swapped out). This chain occupies `logits` rows
    /// `[row0, row0 + chain_len)` of the (possibly batched) verify output; the
    /// pre-verify trunk snapshot must already be in `spec`.
    /// Returns `(emitted, bonus, k)`; the caller crops the paged pool to
    /// `start_pos + k + 1` when `k + 1 < chain.len()`.
    /// `argmax` is the whole verify output's per-row argmax (see
    /// [`Self::argmax_rows`]) — reading it row by row on device would drain the
    /// pipeline once per chain row per slot.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_accept_commit(
        &self,
        chain: &[u32],
        argmax: &[u32],
        row0: usize,
    ) -> Result<(Vec<u32>, u32, usize)> {
        let depth = chain.len() - 1;
        ensure!(
            row0 + chain.len() <= argmax.len(),
            "dspark accept: chain rows outside the verify argmax"
        );
        // Longest prefix where each draft equals the trunk argmax at its row.
        let mut k = 0usize;
        let bonus;
        loop {
            let am = argmax[row0 + k];
            if k < depth && am == chain[k + 1] {
                k += 1;
            } else {
                bonus = am;
                break;
            }
        }
        let mut emitted: Vec<u32> = chain[1..=k].to_vec();
        emitted.push(bonus);
        Ok((emitted, bonus, k))
    }

    /// Rewind a partially-accepted batch: restore every slot's gated-delta
    /// state, then one linear-only replay of every accepted prefix. The paged
    /// full-attn KV self-heals under the truncate + seq_len rewind.
    pub(crate) fn dspark_rollback_batch(
        &self,
        rolls: &mut [DsparkRollback<'_>],
        tables: &mut Qwen35ReplayTables,
        copy: &mut Qwen35CopyScratch,
        ws: &mut Qwen35Workspace,
    ) -> Result<()> {
        if rolls.is_empty() {
            return Ok(());
        }
        let mut pt = super::dspark_phase_start(&self.ctx);
        let (mut gdr, mut conv) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
        let (gdr_bytes, conv_bytes) = self.linear_state_bytes();
        for r in rolls.iter_mut() {
            r.spec.linear_state_addrs(
                &self.ctx,
                r.slot,
                (gdr_bytes, conv_bytes),
                &mut gdr,
                &mut conv,
            )?;
        }
        // Restore: live <- snapshot, so the snapshot side is the source.
        self.batched_copy(copy, &gdr.1, &gdr.0, &[gdr_bytes])?;
        self.batched_copy(copy, &conv.1, &conv.0, &[conv_bytes])?;
        let restore_ms = super::mtp_phase_lap(&self.ctx, &mut pt);
        // The varlen replay is the recurrent kernel; leave the opt-in chunked
        // path on its own route.
        if super::qwen35_gdr_chunked_enabled() {
            for r in rolls.iter_mut() {
                self.replay_linear_only(r.slot, ws, &r.spec.capture, r.k)?;
                r.slot.set_seq_len(r.start_pos + r.k + 1);
            }
            return Ok(());
        }
        let ks: Vec<usize> = rolls.iter().map(|r| r.k).collect();
        let mut slots: Vec<&mut Qwen35SlotState> = Vec::with_capacity(rolls.len());
        let mut captures: Vec<&Qwen35LinearCapture> = Vec::with_capacity(rolls.len());
        for DsparkRollback { slot, spec, .. } in rolls.iter_mut() {
            slots.push(&mut **slot);
            captures.push(&spec.capture);
        }
        self.replay_linear_only_batched(&mut slots, &captures, &ks, tables, ws)?;
        let replay_ms = super::mtp_phase_lap(&self.ctx, &mut pt);
        if pt.is_some() {
            eprintln!(
                "[dspark-rollback] rows={} restore={restore_ms:.2} replay={replay_ms:.2} ms",
                rolls.len()
            );
        }
        for r in rolls.iter_mut() {
            r.slot.set_seq_len(r.start_pos + r.k + 1);
        }
        Ok(())
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
        df: &mut Qwen35DsparkSlotState,
        scratch: &mut DsparkScratch,
        chain: &[u32],
        logits: &HiddenStates,
        row0: usize,
        start_pos: usize,
        params: &SamplingParams,
    ) -> Result<(Vec<super::CommittedToken>, u32, usize)> {
        let ctx = &self.ctx;
        let depth = chain.len() - 1;
        let block = head.cfg.block_size;
        ensure!(
            df.q_rows >= depth && depth <= block,
            "dspark sampled verify: {} q rows for depth {depth} (block {block})",
            df.q_rows
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
        let q_all = df.q_probs.get(ctx, block * vocab)?;
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
            // SAFETY: logits rows [row0, row0+chain.len()) are this chain's;
            // p/q scratches hold (block+1)/block vocab-rows and depth <= block
            // (ensured above); draft/u prefixes uploaded just above.
            unsafe {
                ffi::dspark_filter_probs_cuda(
                    (l_ptr + (row0 * vocab * std::mem::size_of::<bf16>()) as u64)
                        as *const ffi::Half,
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

    /// Trunk verify of every chain in ONE forward over the paged pool, with
    /// per-slot linear capture + a batch-wide tap: logits `[total_q, vocab]`
    /// where chain `i` owns rows `[Σ_{j<i} len_j, ..)`. The paged twin of
    /// [`Self::forward_tokens_verify`] (that one drives the legacy
    /// contiguous lane and the MTP gate test).
    pub(crate) fn dspark_verify_logits(
        &self,
        rows: &mut [super::LinearRow<'_>],
        ws: &mut Qwen35Workspace,
        chains: &[u32],
        recall: &mut Qwen35RecallForward<'_>,
        taps: &mut Qwen35DsparkTaps,
    ) -> Result<HiddenStates> {
        let seq_len: usize = rows.iter().map(|r| r.len).sum();
        ensure!(
            seq_len == chains.len(),
            "dspark verify: {} staged tokens != {seq_len} chain rows",
            chains.len()
        );
        let start_pos = rows[0].slot.seq_len();
        self.stage_step_inputs(ws, chains, start_pos)?;
        self.forward_hidden_staged(rows, ws, start_pos, Some(recall), Some(taps))?;
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
        // Last: the caller rolls the pool back on any error here, and that only
        // restores consistency while the trunk seq_lens have not moved.
        for r in rows.iter_mut() {
            r.slot.advance_seq_len(r.len);
        }
        Ok(logits)
    }
}
