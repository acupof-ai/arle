//! DSpark block drafter for Qwen3.6 CUDA speculative decode.
//!
//! One draft step proposes `block_size` positions in a single 5-layer
//! non-causal forward: per accepted trunk token the residual stream is tapped
//! at `target_layer_ids`, projected through `fc` (+ `hidden_norm`) to one
//! feature, and cached as the draft's per-layer K/V context. Covers both
//! checkpoint flavors of [`qwen35_spec::DsparkConfig`] — DFlash (same-position
//! denoising) and DSpark (next-token labels + optional Markov / confidence).

use super::*;

use crate::ops::rms_norm_batch;
use cuda_kernels::attention as cuda_attn;
use qwen35_spec::{DsparkConfig, DsparkSps, dspark_tensor_names, dspark_verify_lens};

struct DsparkLayer {
    /// Gate-padded `[2*q_dim, hidden]`: head `h` at rows
    /// `2h*head_dim..(2h+1)*head_dim`, odd bands zero — the trunk's fused prep
    /// kernel assumes the gated q layout.
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
    w1: DeviceMatrix,
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
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Per-layer ctx-cache rows, addressed as an absolute-position ring
    /// (`row = pos % cap`).
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
    /// Per-slot bytes: draft ctx K/V plus the draft outputs. Lazily allocated,
    /// so they must be reserved out of the KV budget or the first dspark step
    /// OOMs behind an already-sized pool.
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

    /// Hot-swap the Markov head weights from a host f32 snapshot. `w1`/`w2` are
    /// `[vocab, rank]` row-major. Safe mid-run because the hot path reads them
    /// only after a stream sync at step entry.
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
        let w1_bf16: Vec<bf16> = w1.iter().map(|&x| bf16::from_f32(x)).collect();
        let w2_bf16: Vec<bf16> = w2.iter().map(|&x| bf16::from_f32(x)).collect();
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
/// and raw V of accepted positions, laid out `[kv_head][cap][head_dim]` as an
/// absolute-position ring (`row = abs % cap`, `cap = window + block`).
/// Speculative noise rows self-heal: rejected rows are overwritten by the next
/// block's writes before any read, and the live span (window ctx + block noise)
/// equals `cap`, so distinct live positions never collide mod `cap`.
pub(crate) struct Qwen35DsparkSlotState {
    k_ctx: Vec<DeviceVec>,
    v_ctx: Vec<DeviceVec>,
    /// Absolute trunk position of ctx buffer row 0. Non-zero after a
    /// prefix-restore rebase: the suffix-only ctx is exact once the tail ≥ window.
    pub(crate) ctx_base: usize,
    /// Absolute end (exclusive) of materialized ctx rows. The spec step
    /// requires `ctx_end == start_pos`.
    pub(crate) ctx_end: usize,
    /// Last emitted token (the next block's anchor), staged by prefill/verify.
    pub(crate) pending: Option<u32>,
    /// Per-slot, not shared scratch: a batched step drafts every slot before
    /// verifying any. Lazy — greedy never allocates `q_probs`.
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
    /// zeroing needed: attention `lo` never drops below `ctx_base`, so stale
    /// rows are overwritten before any read.
    pub(crate) fn rebase(&mut self, pos: usize) {
        self.ctx_base = pos;
        self.ctx_end = pos;
        self.pending = None;
    }
}

/// Trunk residual-stream tap capture. `prepare` arms it for one forward; the
/// layer loop D2D-copies the residual stream after each target layer (`-1` =
/// the embedding output).
#[derive(Default)]
pub(crate) struct Qwen35DsparkTaps {
    targets: Vec<i64>,
    bufs: Vec<HiddenSlot>,
    captured: Vec<bool>,
    hidden: usize,
    seq: usize,
    armed: bool,
}

impl Qwen35DsparkTaps {
    /// Every id must name a reachable layer exactly once — a duplicate or
    /// out-of-range id leaves `capture` nothing to write and hands the reader a
    /// zero buffer indistinguishable from a real one.
    pub(crate) fn validate(targets: &[i64], num_layers: usize) -> Result<()> {
        let mut seen = targets.to_vec();
        seen.sort_unstable();
        seen.dedup();
        ensure!(
            seen.len() == targets.len(),
            "duplicate dspark target_layer_ids: {targets:?}"
        );
        for &t in targets {
            ensure!(
                t == -1 || (0..num_layers as i64).contains(&t),
                "dspark target layer {t} outside -1..{num_layers}"
            );
        }
        Ok(())
    }

    pub(crate) fn prepare(&mut self, targets: &[i64], hidden: usize, seq: usize) {
        if self.targets != targets {
            self.targets = targets.to_vec();
            self.bufs = std::iter::repeat_with(HiddenSlot::default)
                .take(targets.len())
                .collect();
        }
        self.captured = vec![false; targets.len()];
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
        self.captured[i] = true;
        Ok(())
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    /// Captured tap `i` as host f32, token-major `[seq, hidden]`.
    pub(crate) fn tap_to_host(&mut self, ctx: &DeviceContext, i: usize) -> Result<Vec<f32>> {
        ensure!(self.armed, "dspark tap readback without an armed capture");
        ensure!(i < self.bufs.len(), "tap {i} of {}", self.bufs.len());
        ensure!(
            self.captured[i],
            "dspark tap {i} (layer {}) was never captured; reading it would \
             hand the trainer a zero feature",
            self.targets[i]
        );
        self.bufs[i].get(ctx, self.hidden, self.seq)?.to_host(ctx)
    }

    pub(crate) fn release(&mut self) {
        self.disarm();
    }
}

/// Draft-side persistent scratch, one per executor — every buffer is written
/// then read inside one draft or one accept, and those never interleave.
/// Separate from [`Qwen35Workspace`] so the shapes never thrash.
#[derive(Default)]
pub(crate) struct DsparkScratch {
    ids: SliceSlot<i32>,
    /// Absolute (unshifted) start position for sliding-ring prep launches.
    start_pos_abs: SliceSlot<i32>,
    /// Per-row draft attention windows, `[ring_base; block] ++ [kv_len; block]`
    /// — identical for every layer, so one upload serves the whole forward.
    attn_win: SliceSlot<i32>,
    /// `[k_ctx base; slots] ++ [v_ctx base; slots]` for the current draft layer.
    attn_kv_slots: SliceSlot<u64>,
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
    mk: MarkovScratch,
    conf: ConfScratch,
    /// Batched-draft logits `[B*block, vocab]`; per-slot `df.logits` copy from here.
    logits_b: HiddenSlot,
    /// Row count of the tap features held in `feat_b` (the whole batch).
    feat_seq: usize,
    /// Sampled-mode buffers, fixed caps, allocated on the first temp>0 step.
    /// Only written prefixes are ever read: `p_probs` rows past `chain_len` and
    /// the `chain_draft`/`u_*` tails past `depth` hold stale data.
    p_probs: SliceSlot<f32>,
    sample_tok: SliceSlot<i32>,
    accept_out: SliceSlot<i32>,
    chain_draft: SliceSlot<i32>,
    u_accept: SliceSlot<f32>,
    u_residual: SliceSlot<f32>,
    /// Pinned: `clone_dtoh` into a `Vec` page-locks and frees per call.
    tok_host: PinnedSlot<i32>,
    /// Device argmax ids; verify runs one per tick, so it must not allocate.
    argmax_ids: SliceSlot<i32>,
}

#[derive(Default)]
struct MarkovScratch {
    prevs: SliceSlot<i32>,
    emb: HiddenSlot,
    bias: HiddenSlot,
    sum: HiddenSlot,
    ids: SliceSlot<i32>,
    host: PinnedSlot<i32>,
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
/// sampler's stream (`infer_plan::sample_token`), so same-config-twice
/// reproduces. `SALT_DRAW = 0` makes the draft draw consume exactly the uniform
/// plain decode would at that position.
pub(super) fn unit_uniform(seed: Option<u64>, salt: u64, position: u64) -> f32 {
    let bits = splitmix64(
        seed.unwrap_or(0)
            .wrapping_add(salt)
            .wrapping_add(position)
            .wrapping_add(1),
    );
    (bits >> 40) as f32 / (1u32 << 24) as f32
}

/// Built only under `--spec-type dspark`, so the baseline executor allocates
/// nothing.
pub(crate) struct Qwen35DsparkExec {
    pub(crate) head: Qwen35DsparkHead,
    pub(crate) slots: Vec<Option<Qwen35DsparkSlotState>>,
    pub(crate) spec: Vec<Option<Qwen35SpecSlotState>>,
    pub(crate) taps: Qwen35DsparkTaps,
    pub(crate) scratch: DsparkScratch,
    pub(crate) replay_tables: Qwen35ReplayTables,
    pub(crate) copy: Qwen35CopyScratch,
    pub(crate) accepts: usize,
    pub(crate) rejects: usize,
    /// Verified chains, and the subset drafted from a partial
    /// (`ctx_base > 0`) draft context.
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

fn load_host_vec(loader: &SafetensorLoader, name: &str) -> Result<Vec<bf16>> {
    let t = loader.load_raw_tensor(name)?;
    let bytes = SafetensorLoader::tensor_bytes_to_bf16(name, t.dtype, &t.bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_le_bytes(*c))
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_le_bytes(*c))
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
    // Positions past the accepted prefix are pure waste: TC-27B + DFlash keeps
    // 3.28 tokens at block 16 (accept_rate 0.205), discarding 79.5% of the
    // drafted work. Clamp here — rope_cap, the ctx ring and the per-slot
    // scratch all size off `cfg.block_size`.
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
        // Zeros: `--dspark-markov-init` only needs the slot shape, both halves
        // are overwritten before use.
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

    let rope_cap = max_seq_len.min(max_total_tokens.max(1)) + cfg.block_size;
    let cap = cfg.sliding_window.map_or(rope_cap, |w| w + cfg.block_size);
    match cfg.sliding_window {
        // No window costs a request-length ring: 671 MB/slot at 32k vs 42 MB.
        None => log::info!(
            "dspark draft: no sliding_window declared; every layer runs full \
             attention over a {cap}-row ctx ring"
        ),
        Some(window) => {
            let full = cfg
                .layer_types
                .iter()
                .filter(|t| matches!(t, qwen35_spec::DsparkLayerType::Full))
                .count();
            if full > 0 {
                // Windowed on purpose (honoring them needs a request-length
                // ring), but it moves acceptance, so say it.
                log::warn!(
                    "dspark draft: {full}/{} layers declare full attention; all run the \
                     {window}-token sliding window (ctx ring is {cap} rows)",
                    cfg.layer_types.len(),
                );
            }
        }
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

/// Lowest absolute key position a draft row at `pos` may read: HF sliding
/// window keeps keys with `q_pos - k_pos < window`.
fn window_lo(cfg: &DsparkConfig, pos: usize) -> usize {
    cfg.sliding_window
        .map_or(0, |w| pos.saturating_sub(w.saturating_sub(1)))
}

fn add_vec_into(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch(ctx, a, b, &mut out)?;
    Ok(out)
}

impl Qwen35Model {
    /// `feat = hidden_norm(Σ_t fc_t · tap_t)` over the whole tap; runs once per
    /// forward and disarms the tap.
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
        crate::profile::profile_op(ctx, "mtp_fc", None, seq, || {
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
            Ok(())
        })?;
        let acc = scratch.feat_a.get(ctx, hidden, seq)?;
        let normed = scratch.feat_b.get(ctx, hidden, seq)?;
        crate::profile::profile_op(ctx, "mtp_hidden_norm", None, seq, || {
            rms_norm_batch(ctx, acc, &head.hidden_norm, eps, normed)
        })?;
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
            let v_new = scratch.ctx_v.get(ctx, kv_dim, rows)?;
            crate::profile::profile_op(ctx, "ctx_kv_proj", Some(li), rows, || {
                gemm_batch(ctx, &layer.k_proj, feat_rows, k_new)?;
                gemm_batch(ctx, &layer.v_proj, feat_rows, v_new)
            })?;
            // The fused prep wrapper requires q heads, so drive `num_kv_heads`
            // dummy ones over a zeroed gated-layout buffer (0/rms(0+eps)=0).
            let q_dummy = scratch.ctx_q_dummy.get_zeroed(ctx, 2 * kv_dim, rows)?;
            let q_out = scratch.ctx_q_out.get(ctx, kv_dim, rows)?;
            crate::profile::profile_op(ctx, "ctx_prep", Some(li), rows, || {
                let (qd_ptr, _g0) = q_dummy.data.device_ptr(&ctx.stream);
                let (k_ptr, _g1) = k_new.data.device_ptr(&ctx.stream);
                let (v_ptr, _g2) = v_new.data.device_ptr(&ctx.stream);
                let (kn_ptr, _g3) = layer.k_norm.data.device_ptr(&ctx.stream);
                let (cos_ptr, _g4) = head.cos_cache.data.device_ptr(&ctx.stream);
                let (sin_ptr, _g5) = head.sin_cache.data.device_ptr(&ctx.stream);
                let (qo_ptr, _g6) = q_out.data.device_ptr_mut(&ctx.stream);
                let (kc_ptr, _g7) = df.k_ctx[li].data.device_ptr_mut(&ctx.stream);
                let (vc_ptr, _g8) = df.v_ctx[li].data.device_ptr_mut(&ctx.stream);
                let nkv = head.cfg.num_key_value_heads;
                let hd = head.cfg.head_dim;
                // One launch must not wrap the ring, else two tokens alias one row.
                ensure!(
                    rows <= cap_li,
                    "dspark sliding append rows {rows} > ring cap {cap_li}; lower chunked_prefill_size"
                );
                // Ring cache sized cap_li*kv_dim; rows <= cap_li (no
                // aliasing); abs positions < rope_cap index the cos/sin tables.
                cuda_attn::prefill_attention_hd256_prep_ring_raw(
                    &ctx.stream,
                    qd_ptr,
                    k_ptr,
                    v_ptr,
                    kn_ptr,
                    kn_ptr,
                    cos_ptr,
                    sin_ptr,
                    qo_ptr,
                    kc_ptr,
                    vc_ptr,
                    nkv,
                    nkv,
                    hd,
                    rows,
                    sp_abs,
                    hd,
                    head.cfg.rms_norm_eps,
                    cap_li,
                )
            })?;
        }
        df.ctx_end = start + rows;
        Ok(())
    }

    /// One DSpark block draft over `[ctx cache ++ block]`. Returns the verify
    /// chain `[anchor, d1..dL]` (`L` truncated by the confidence head when
    /// present). Sampling mode draws each draft from the engine-sampler-filtered
    /// dist `q`, retained in `df.q_probs` row `i` for `chain[i+1]`.
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

        let mut ids = vec![cfg.mask_token_id as i32; block];
        ids[0] = anchor as i32;
        let ids_dev = scratch.ids.upload(ctx, &ids)?;
        let h = scratch.hidden.get(ctx, hidden, block)?;
        crate::profile::profile_op(ctx, "embedding", None, block, || {
            embedding_batch(ctx, &self.embed_tokens, ids_dev, h)
        })?;

        let start_abs = scratch.start_pos_abs.upload(&self.ctx, &[start as i32])?;
        // Per-row attention windows, never below `ctx_base`. `lo` is
        // layer-independent, so one table serves all layers and one
        // ragged-window launch replaces `block` single-row ones.
        let mut win = vec![0i32; 2 * block];
        for row in 0..block {
            let lo = window_lo(cfg, start + row).max(df.ctx_base);
            let kv_len = kv_len_total - lo;
            // kv_len == q_pos+1−lo ≤ cap, so the ring read never revisits a
            // physical row (the kernel cannot check this host-side).
            ensure!(kv_len <= head.cap, "dspark draft row window {kv_len} > cap");
            win[row] = lo as i32;
            win[block + row] = kv_len as i32;
        }
        let win_dev = scratch.attn_win.upload(&self.ctx, &win)?;
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.cap;
            let h = scratch.hidden.get(ctx, hidden, block)?;
            let normed = scratch.normed.get(ctx, hidden, block)?;
            crate::profile::profile_op(ctx, "input_norm", Some(li), block, || {
                rms_norm_batch(ctx, h, &layer.input_layernorm, eps, normed)
            })?;

            crate::profile::profile_op(ctx, "full_attention", Some(li), block, || {
                let q_full = scratch.q_full.get(ctx, 2 * q_dim, block)?;
                gemm_batch(ctx, &layer.q_proj, normed, q_full)?;
                let k_new = scratch.k_new.get(ctx, kv_dim, block)?;
                gemm_batch(ctx, &layer.k_proj, normed, k_new)?;
                let v_new = scratch.v_new.get(ctx, kv_dim, block)?;
                gemm_batch(ctx, &layer.v_proj, normed, v_new)?;

                // Noise K/V land at their absolute positions in the ctx ring
                // (self-healing speculative rows — see the slot-state doc).
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
                    let nq = cfg.num_attention_heads;
                    let nkv = cfg.num_key_value_heads;
                    let hd = cfg.head_dim;
                    let (sp_ptr, _g10) = start_abs.device_ptr(&ctx.stream);
                    // Ring cache cap_li*kv_dim; abs pos < rope_cap;
                    // block (≤ cap_li) noise rows write distinct ring rows.
                    cuda_attn::prefill_attention_hd256_prep_ring_raw(
                        &ctx.stream,
                        qf_ptr,
                        k_ptr,
                        v_ptr,
                        qn_ptr,
                        kn_ptr,
                        cos_ptr,
                        sin_ptr,
                        qp_ptr,
                        kc_ptr,
                        vc_ptr,
                        nq,
                        nkv,
                        hd,
                        block,
                        sp_ptr,
                        hd,
                        eps,
                        cap_li,
                    )?;
                }

                // Non-causal: every noise row attends its whole window of the
                // `[ctx ++ block]` key range.
                let attn_heads = scratch.attn_heads.get(ctx, q_dim, block)?;
                {
                    let sm_scale = 1.0 / (cfg.head_dim as f32).sqrt();
                    let (q_ptr, _g0) = q_prepped.data.device_ptr(&ctx.stream);
                    let (kc_ptr, _g1) = df.k_ctx[li].data.device_ptr(&ctx.stream);
                    let (vc_ptr, _g2) = df.v_ctx[li].data.device_ptr(&ctx.stream);
                    let (o_ptr, _g3) = attn_heads.data.device_ptr_mut(&ctx.stream);
                    let (w_ptr, _g4) = win_dev.device_ptr(&ctx.stream);
                    let nq = cfg.num_attention_heads;
                    let nkv = cfg.num_key_value_heads;
                    let hd = cfg.head_dim;
                    // `win` holds 2*block i32 (bases then lengths, each
                    // length ≤ cap_li as asserted above).
                    cuda_attn::nonpaged_prefill_attention_ring_varlen_raw(
                        &ctx.stream,
                        q_ptr,
                        kc_ptr,
                        vc_ptr,
                        o_ptr,
                        nq,
                        nkv,
                        hd,
                        block,
                        w_ptr,
                        w_ptr + (block * std::mem::size_of::<i32>()) as u64,
                        cap_li,
                        sm_scale,
                    )?;
                }

                let attn_out_h = scratch.attn_out_h.get(ctx, hidden, block)?;
                gemm_batch(ctx, &layer.o_proj, attn_heads, attn_out_h)?;
                Ok(())
            })?;

            let hidden_mid = scratch.hidden_mid.get(ctx, hidden, block)?;
            crate::profile::profile_op(ctx, "post_attn_norm", Some(li), block, || {
                add_batch(
                    ctx,
                    scratch.hidden.get(ctx, hidden, block)?,
                    scratch.attn_out_h.get(ctx, hidden, block)?,
                    hidden_mid,
                )?;
                let normed = scratch.normed.get(ctx, hidden, block)?;
                rms_norm_batch(
                    ctx,
                    hidden_mid,
                    &layer.post_attention_layernorm,
                    eps,
                    normed,
                )
            })?;
            let mlp_out = scratch.attn_out_h.get(ctx, hidden, block)?;
            crate::profile::profile_op(ctx, "dense_ffn", Some(li), block, || {
                self.dense_mlp(
                    &layer.mlp,
                    scratch.normed.get(ctx, hidden, block)?,
                    &mut scratch.dense,
                    mlp_out,
                )
            })?;
            crate::profile::profile_op(ctx, "ffn_residual", Some(li), block, || {
                add_batch(
                    ctx,
                    scratch.hidden_mid.get(ctx, hidden, block)?,
                    mlp_out,
                    scratch.hidden.get(ctx, hidden, block)?,
                )
            })?;
        }

        let final_normed = scratch.final_normed.get(ctx, hidden, block)?;
        crate::profile::profile_op(ctx, "final_norm", None, block, || {
            rms_norm_batch(
                ctx,
                scratch.hidden.get(ctx, hidden, block)?,
                &head.norm,
                eps,
                final_normed,
            )
        })?;
        let vocab = self.output_projection().rows;
        let logits = df.logits.get(ctx, vocab, block)?;
        crate::profile::profile_op(ctx, "lm_head_gemm", None, block, || {
            gemm_batch(ctx, self.output_projection(), final_normed, logits)
        })?;

        // Next-token heads (DSpark) draft from every row; same-position
        // (DFlash) rows 1.. fill their own positions.
        let sampling = !params.is_greedy();
        df.q_rows = 0;
        let first_row = usize::from(!cfg.next_token_heads);
        let mut drafts = Vec::with_capacity(block);
        // Greedy resolves the whole block in one batched pass; sampling walks
        // the chain because a markov bias makes row r depend on row r-1's draw.
        if !sampling {
            crate::profile::profile_op(ctx, "sample", None, block, || {
                self.dspark_settle_rows(
                    head.markov.as_ref(),
                    df.logits.get(ctx, vocab, block)?,
                    &mut scratch.mk,
                    &[anchor],
                    block,
                    first_row,
                )
            })
            .map(|am| drafts.extend_from_slice(&am[first_row..block]))?;
        }
        // No markov head: every row reads a precomputed logits row, so the
        // draws do not depend on each other — issue them all, then sync once.
        if sampling && head.markov.is_none() {
            let n = block - first_row;
            crate::profile::profile_op(ctx, "sample", None, n, || {
                let logits = df.logits.get(ctx, vocab, block)?;
                let q_all = df.q_probs.get(ctx, n * vocab)?;
                let tok_dev = scratch.sample_tok.get(ctx, n)?;
                let filter = cuda_kernels::sampling::DsparkFilter {
                    inv_temperature: 1.0 / params.temperature,
                    top_k: params.top_k,
                    top_p: params.top_p,
                    min_p: params.min_p,
                };
                for i in 0..n {
                    let u = unit_uniform(params.seed, SALT_DRAW, (start + first_row + i) as u64);
                    // Logits row `first_row + i` of `block`; q row `i` of `n`;
                    // one i32 out slot per row.
                    let row = first_row + i;
                    let logits_row = logits.data.slice(row * vocab..(row + 1) * vocab);
                    let mut q_row = q_all.slice_mut(i * vocab..(i + 1) * vocab);
                    let mut tok = tok_dev.slice_mut(i..i + 1);
                    cuda_kernels::sampling::dspark_draft_sample(
                        ctx,
                        &logits_row,
                        &mut q_row,
                        &mut tok,
                        vocab,
                        filter,
                        u,
                    )?;
                }
                ctx.sync()?;
                Ok(())
            })?;
            let src = scratch.sample_tok.get(ctx, n)?.clone();
            let host = scratch.tok_host.get(ctx, n)?;
            ctx.stream
                .memcpy_dtoh(&src, host)
                .map_err(|e| anyhow!("D2H dspark draws failed: {e}"))?;
            let drawn: Vec<i32> = host.as_slice()?.to_vec();
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
            let (src, src_row) = if let Some(m) = &head.markov {
                crate::profile::profile_op(ctx, "mtp_fc", None, 1, || {
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
                    Ok(())
                })?;
                (scratch.step_sum.get(ctx, vocab, 1)?, 0)
            } else {
                (df.logits.get(ctx, vocab, block)?, row)
            };
            let tok = crate::profile::profile_op(ctx, "sample", None, 1, || {
                // Uniform from the host (seed, position) stream plain decode
                // would consume at this position (SALT_DRAW = 0).
                let u = unit_uniform(params.seed, SALT_DRAW, (start + row) as u64);
                let q_all = df.q_probs.get(ctx, block * vocab)?;
                let tok_out = scratch.sample_tok.get(ctx, 1)?;
                // `src` row holds `vocab` bf16 (src_row bounded by its
                // seq_len); q row index == drafts.len() < block.
                let logits_row = src.data.slice(src_row * vocab..(src_row + 1) * vocab);
                let q_row_idx = drafts.len();
                let mut q_row = q_all.slice_mut(q_row_idx * vocab..(q_row_idx + 1) * vocab);
                cuda_kernels::sampling::dspark_draft_sample(
                    ctx,
                    &logits_row,
                    &mut q_row,
                    tok_out,
                    vocab,
                    cuda_kernels::sampling::DsparkFilter {
                        inv_temperature: 1.0 / params.temperature,
                        top_k: params.top_k,
                        top_p: params.top_p,
                        min_p: params.min_p,
                    },
                    u,
                )?;
                ctx.sync()?;
                let src = tok_out.clone();
                let host = scratch.tok_host.get(ctx, 1)?;
                ctx.stream
                    .memcpy_dtoh(&src, host)
                    .map_err(|e| anyhow!("D2H dspark draft token failed: {e}"))?;
                let tok = host.as_slice()?[0] as u32;
                df.q_rows += 1;
                Ok(tok)
            })?;
            drafts.push(tok);
            prev = tok;
        }

        // Goodput-budget the proposal (R=1 here); absent head keeps the block.
        let prev_tokens: Vec<u32> = std::iter::once(anchor)
            .chain(drafts.iter().copied())
            .take(drafts.len())
            .collect();
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
    /// block, since the draft GEMMs are weight-bound at `block` rows. Only the
    /// ring kernels stay per-slot. Greedy only — a sampled draw syncs per row.
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
                let lo = window_lo(cfg, start + row).max(df.ctx_base);
                let kv_len = start + block - lo;
                ensure!(kv_len <= head.cap, "dspark draft row window {kv_len} > cap");
                win[s * block + row] = lo as i32;
                win[rows + s * block + row] = kv_len as i32;
            }
        }

        let ids_dev = scratch.ids.upload(ctx, &ids)?;
        let h = scratch.hidden.get(ctx, hidden, rows)?;
        crate::profile::profile_op(ctx, "embedding", None, rows, || {
            embedding_batch(ctx, &self.embed_tokens, ids_dev, h)
        })?;
        let pos_dev = scratch.start_pos_abs.upload(ctx, &pos)?;
        let win_dev = scratch.attn_win.upload(ctx, &win)?;

        let elem = std::mem::size_of::<bf16>() as u64;
        for (li, layer) in head.layers.iter().enumerate() {
            let cap_li = head.cap;
            let h = scratch.hidden.get(ctx, hidden, rows)?;
            let normed = scratch.normed.get(ctx, hidden, rows)?;
            crate::profile::profile_op(ctx, "input_norm", Some(li), rows, || {
                rms_norm_batch(ctx, h, &layer.input_layernorm, eps, normed)
            })?;

            crate::profile::profile_op(ctx, "full_attention", Some(li), rows, || {
                let q_full = scratch.q_full.get(ctx, 2 * q_dim, rows)?;
                gemm_batch(ctx, &layer.q_proj, normed, q_full)?;
                let k_new = scratch.k_new.get(ctx, kv_dim, rows)?;
                gemm_batch(ctx, &layer.k_proj, normed, k_new)?;
                let v_new = scratch.v_new.get(ctx, kv_dim, rows)?;
                gemm_batch(ctx, &layer.v_proj, normed, v_new)?;

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
                    let nq = cfg.num_attention_heads;
                    let nkv = cfg.num_key_value_heads;
                    let hd = cfg.head_dim;
                    let sm_scale = 1.0 / (cfg.head_dim as f32).sqrt();
                    // `[k bases][v bases]`, one entry per slot, so the attention
                    // below runs once at gridZ = slots instead of once per slot.
                    let mut kv_bases = vec![0u64; 2 * b];
                    let mut ring_guards = Vec::with_capacity(2 * b);
                    for (s, df) in dfs.iter_mut().enumerate() {
                        let off = (s * block) as u64;
                        let (kc_ptr, gk) = df.k_ctx[li].data.device_ptr_mut(&ctx.stream);
                        let (vc_ptr, gv) = df.v_ctx[li].data.device_ptr_mut(&ctx.stream);
                        kv_bases[s] = kc_ptr;
                        kv_bases[b + s] = vc_ptr;
                        // Offset by this slot's `block` rows inside a
                        // `rows`-row buffer; ring bounds guarded above.
                        cuda_attn::prefill_attention_hd256_prep_ring_raw(
                            &ctx.stream,
                            qf_ptr + off * 2 * q_dim as u64 * elem,
                            k_ptr + off * kv_dim as u64 * elem,
                            v_ptr + off * kv_dim as u64 * elem,
                            qn_ptr,
                            kn_ptr,
                            cos_ptr,
                            sin_ptr,
                            qp_ptr + off * q_dim as u64 * elem,
                            kc_ptr,
                            vc_ptr,
                            nq,
                            nkv,
                            hd,
                            block,
                            sp_ptr + s as u64 * 4,
                            hd,
                            eps,
                            cap_li,
                        )?;
                        ring_guards.push(gk);
                        ring_guards.push(gv);
                    }
                    let slots_dev = scratch.attn_kv_slots.upload(ctx, &kv_bases)?;
                    let (sl_ptr, _gs) = slots_dev.device_ptr(&ctx.stream);
                    // `kv_bases` holds this layer's `b` k rings then `b` v
                    // rings, each staged above; the window table is `rows` bases
                    // then `rows` lengths, slot-major as the kernel indexes it.
                    cuda_attn::nonpaged_prefill_attention_ring_varlen_batched_raw(
                        &ctx.stream,
                        qp_ptr,
                        sl_ptr,
                        sl_ptr + (b * std::mem::size_of::<u64>()) as u64,
                        ao_ptr,
                        nq,
                        nkv,
                        hd,
                        block,
                        b,
                        w_ptr,
                        w_ptr + (rows * std::mem::size_of::<i32>()) as u64,
                        cap_li,
                        sm_scale,
                    )?;
                }

                let attn_out_h = scratch.attn_out_h.get(ctx, hidden, rows)?;
                gemm_batch(ctx, &layer.o_proj, attn_heads, attn_out_h)?;
                Ok(())
            })?;

            let hidden_mid = scratch.hidden_mid.get(ctx, hidden, rows)?;
            crate::profile::profile_op(ctx, "post_attn_norm", Some(li), rows, || {
                add_batch(
                    ctx,
                    scratch.hidden.get(ctx, hidden, rows)?,
                    scratch.attn_out_h.get(ctx, hidden, rows)?,
                    hidden_mid,
                )?;
                let normed = scratch.normed.get(ctx, hidden, rows)?;
                rms_norm_batch(
                    ctx,
                    hidden_mid,
                    &layer.post_attention_layernorm,
                    eps,
                    normed,
                )
            })?;
            let mlp_out = scratch.attn_out_h.get(ctx, hidden, rows)?;
            crate::profile::profile_op(ctx, "dense_ffn", Some(li), rows, || {
                self.dense_mlp(
                    &layer.mlp,
                    scratch.normed.get(ctx, hidden, rows)?,
                    &mut scratch.dense,
                    mlp_out,
                )
            })?;
            crate::profile::profile_op(ctx, "ffn_residual", Some(li), rows, || {
                add_batch(
                    ctx,
                    scratch.hidden_mid.get(ctx, hidden, rows)?,
                    mlp_out,
                    scratch.hidden.get(ctx, hidden, rows)?,
                )
            })?;
        }

        let final_normed = scratch.final_normed.get(ctx, hidden, rows)?;
        crate::profile::profile_op(ctx, "final_norm", None, rows, || {
            rms_norm_batch(
                ctx,
                scratch.hidden.get(ctx, hidden, rows)?,
                &head.norm,
                eps,
                final_normed,
            )
        })?;
        let vocab = self.output_projection().rows;
        let logits = scratch.logits_b.get(ctx, vocab, rows)?;
        crate::profile::profile_op(ctx, "lm_head_gemm", None, rows, || {
            gemm_batch(ctx, self.output_projection(), final_normed, logits)
        })?;

        let first_row = usize::from(!cfg.next_token_heads);
        let am = crate::profile::profile_op(ctx, "sample", None, rows, || {
            self.dspark_settle_rows(
                head.markov.as_ref(),
                scratch.logits_b.get(ctx, vocab, rows)?,
                &mut scratch.mk,
                anchors,
                block,
                first_row,
            )
        })?;

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
    /// speculates on itself: guess every predecessor from the base argmax,
    /// correct all rows at once, re-run while a guess disagreed. Each round
    /// confirms one more row, so `block` bounds it. Costs two `[vocab, b*block]`
    /// buffers, 48 MB each at block 6 / B 16.
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
        let mut host = std::mem::take(&mut mk.host);
        let mut toks = {
            let ids = mk.ids.get(ctx, rows)?;
            self.argmax_rows_into(base, ids, &mut host)
        }?;
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
            let next = {
                let MarkovScratch { ids, sum, .. } = &mut *mk;
                let logits = sum.get(ctx, vocab, rows)?;
                let ids = ids.get(ctx, rows)?;
                self.argmax_rows_into(logits, ids, &mut host)
            }?;
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
    /// `sigmoid(proj([hidden_i ; markov_w1[prev_i]]))` cumprod'd into survival
    /// and fed to the goodput budget ([`dspark_verify_lens`]). `prevs` is
    /// slot-major, `b*(block - first_row)` long. Head absent → whole block.
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

    /// Per-row argmax over a whole verify output in one launch + one D2H, so a
    /// batched tick costs one sync instead of one per chain row per slot.
    pub(crate) fn argmax_rows(
        &self,
        logits: &HiddenStates,
        scratch: &mut DsparkScratch,
    ) -> Result<Vec<u32>> {
        let mut ids_dev = std::mem::take(&mut scratch.argmax_ids);
        let out = {
            let dev = ids_dev.get(&self.ctx, logits.seq_len)?;
            self.argmax_rows_into(logits, dev, &mut scratch.tok_host)
        };
        scratch.argmax_ids = ids_dev;
        out
    }

    /// [`Self::argmax_rows`] into caller-held scratch — the draft path calls it
    /// once per speculation round, so it must not allocate.
    pub(crate) fn argmax_rows_into(
        &self,
        logits: &HiddenStates,
        ids_dev: &mut CudaSlice<i32>,
        host: &mut PinnedSlot<i32>,
    ) -> Result<Vec<u32>> {
        let ctx = &self.ctx;
        let (rows, vocab) = (logits.seq_len, logits.hidden_dim);
        ensure!(
            ids_dev.len() == rows,
            "dspark argmax scratch {} != {rows} rows",
            ids_dev.len()
        );
        cuda_kernels::sampling::argmax_batch(ctx, &logits.data, ids_dev, rows, vocab)?;
        let hbuf = host.get(ctx, rows)?;
        ctx.stream
            .memcpy_dtoh(&*ids_dev, hbuf)
            .map_err(|e| anyhow!("D2H dspark argmax failed: {e}"))?;
        let ids: Vec<i32> = hbuf.as_slice()?.to_vec();
        ids.into_iter()
            .map(|id| {
                ensure!((0..vocab as i32).contains(&id), "dspark argmax id {id} oob");
                Ok(id as u32)
            })
            .collect()
    }

    /// Accept scan over a DSpark verify. This chain occupies `logits` rows
    /// `[row0, row0 + chain_len)` of the (possibly batched) verify output;
    /// `argmax` is that whole output's per-row argmax. Returns `(emitted,
    /// bonus, k)`; the caller crops the paged pool to `start_pos + k + 1` when
    /// `k + 1 < chain.len()`.
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
        let ks: Vec<usize> = rolls.iter().map(|r| r.k).collect();
        let mut slots: Vec<&mut Qwen35SlotState> = Vec::with_capacity(rolls.len());
        let mut captures: Vec<&Qwen35LinearCapture> = Vec::with_capacity(rolls.len());
        for DsparkRollback { slot, spec, .. } in rolls.iter_mut() {
            slots.push(&mut **slot);
            captures.push(&spec.capture);
        }
        self.replay_linear_only_batched(&mut slots, &captures, &ks, tables, ws)?;
        for r in rolls.iter_mut() {
            r.slot.set_seq_len(r.start_pos + r.k + 1);
        }
        Ok(())
    }

    /// Rejection-sampling twin of [`Self::dspark_accept_commit`] (mirrors
    /// flashinfer/SGLang `chain_speculative_sampling`): accept `chain[j+1]`
    /// with prob min(1, p_j(tok)/q_j(tok)) under the engine-sampler-filtered
    /// distributions; the first reject commits a residual `max(0, p−q)` draw
    /// (falling back to `p` on ~0 mass), full accept a bonus draw from the last
    /// row. All uniforms come from host salted `(seed, position)` streams, so
    /// same-config-twice reproduces.
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
        // Position-salted, so batching draws the same uniforms as the host path.
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
        // Logits rows [row0, row0+chain.len()) are this chain's; p/q scratches
        // hold (block+1)/block vocab-rows and depth <= block (ensured above);
        // draft/u prefixes uploaded just above.
        let chain_logits = logits
            .data
            .slice(row0 * vocab..(row0 + chain.len()) * vocab);
        cuda_kernels::sampling::dspark_filter_probs(
            ctx,
            &chain_logits,
            p_all,
            chain.len(),
            vocab,
            cuda_kernels::sampling::DsparkFilter {
                inv_temperature: 1.0 / params.temperature,
                top_k: params.top_k,
                top_p: params.top_p,
                min_p: params.min_p,
            },
        )?;
        cuda_kernels::sampling::dspark_chain_accept(
            ctx,
            &*q_all,
            &*p_all,
            &*draft_dev,
            &*ua_dev,
            &*ur_dev,
            out_dev,
            depth,
            vocab,
        )?;
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
        // Behavior logprobs read from the still-materialized filtered p rows.
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

    /// Trunk verify of every chain in one forward over the paged pool, with
    /// per-slot linear capture + a batch-wide tap: logits `[total_q, vocab]`
    /// where chain `i` owns rows `[Σ_{j<i} len_j, ..)`.
    pub(crate) fn dspark_verify_logits(
        &self,
        rows: &mut [super::LinearRow<'_>],
        ws: &mut Qwen35Workspace,
        chains: &[u32],
        recall: &mut Qwen35PagedForward<'_>,
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
        crate::profile::profile_op(&self.ctx, "final_norm", None, seq_len, || {
            super::rms_norm_offset(
                &self.ctx,
                hidden,
                &self.norm,
                self.config.rms_norm_eps,
                normed,
            )
        })?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemm", None, seq_len, || {
            gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)
        })?;
        self.ctx.sync()?;
        // Last: the caller's error rollback only restores consistency while the
        // trunk seq_lens have not moved.
        for r in rows.iter_mut() {
            r.slot.advance_seq_len(r.len);
        }
        Ok(logits)
    }
}
