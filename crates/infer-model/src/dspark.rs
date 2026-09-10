//! DSpark/DFlash drafter host arithmetic: the deterministic RNG stream,
//! attention-window math, confidence survival, Markov-chain
//! settle logic, and the load-time host tensor transforms. The device
//! launches stay in `infer-cuda`; this module holds only the decisions and
//! the shapes.

use anyhow::{Result, ensure};
use half::bf16;
use qwen35_spec::{DsparkConfig, DsparkSps, dspark_verify_lens};

/// Uniform-stream salts: draft draw / accept test / residual+bonus draw at the
/// same position must be independent or the rejection identity breaks.
pub const SALT_DRAW: u64 = 0;
pub const SALT_ACCEPT: u64 = 0x9E37_79B9_7F4A_7C15;
pub const SALT_RESIDUAL: u64 = 0xC2B2_AE3D_27D4_EB4F;

/// SplitMix64 — mirrors `infer_plan::sample`'s private mixer bit-for-bit.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic uniform in [0, 1) from `(seed, salt, position)` — the engine
/// sampler's stream (`infer_plan::sample_token`), so same-config-twice
/// reproduces. `SALT_DRAW = 0` makes the draft draw consume exactly the
/// uniform plain decode would at that position.
pub fn unit_uniform(seed: Option<u64>, salt: u64, position: u64) -> f32 {
    let bits = splitmix64(
        seed.unwrap_or(0)
            .wrapping_add(salt)
            .wrapping_add(position)
            .wrapping_add(1),
    );
    (bits >> 40) as f32 / (1u32 << 24) as f32
}

/// Lowest absolute key position a draft row at `pos` may read: HF sliding
/// window keeps keys with `q_pos - k_pos < window`.
pub fn window_lo(sliding_window: Option<usize>, pos: usize) -> usize {
    sliding_window.map_or(0, |w| pos.saturating_sub(w.saturating_sub(1)))
}

/// Per-row draft attention window table for one slot: `[lo; block] ++
/// [kv_len; block]`. `lo` never drops below `ctx_base`; every `kv_len` must fit
/// the ctx ring (the kernel cannot check this host-side).
pub fn draft_attention_windows(
    sliding_window: Option<usize>,
    start: usize,
    block: usize,
    ctx_base: usize,
    cap: usize,
) -> Result<Vec<i32>> {
    let mut win = vec![0i32; 2 * block];
    for row in 0..block {
        let lo = window_lo(sliding_window, start + row).max(ctx_base);
        let kv_len = start + block - lo;
        ensure!(kv_len <= cap, "dspark draft row window {kv_len} > cap");
        win[row] = lo as i32;
        win[block + row] = kv_len as i32;
    }
    Ok(win)
}

/// First drafted row: same-position (DFlash) rows 1.. fill their own positions.
pub fn first_row(next_token_heads: bool) -> usize {
    usize::from(!next_token_heads)
}

pub fn q_dim(cfg: &DsparkConfig) -> usize {
    cfg.num_attention_heads * cfg.head_dim
}

pub fn kv_dim(cfg: &DsparkConfig) -> usize {
    cfg.num_key_value_heads * cfg.head_dim
}

/// RoPE table length = `min(max_seq_len, max_total_tokens) + block_size` — the
/// per-request token ceiling, NOT the whole KV-pool `max_seq_len`.
pub fn rope_cap(max_seq_len: usize, max_total_tokens: usize, block_size: usize) -> usize {
    max_seq_len.min(max_total_tokens.max(1)) + block_size
}

/// Per-layer ctx-cache ring rows, addressed as an absolute-position ring
/// (`row = pos % cap`).
pub fn ctx_cap(sliding_window: Option<usize>, rope_cap: usize, block_size: usize) -> usize {
    sliding_window.map_or(rope_cap, |w| w + block_size)
}

/// Per-slot bytes: draft ctx K/V plus the draft outputs. Lazily allocated, so
/// they must be reserved out of the KV budget or the first dspark step OOMs
/// behind an already-sized pool.
pub fn slot_state_bytes(cfg: &DsparkConfig, cap: usize, num_layers: usize, vocab: usize) -> usize {
    let per_head = cfg.num_key_value_heads * cfg.head_dim;
    let ctx = 2 * cap * per_head * num_layers * std::mem::size_of::<bf16>();
    let draft = cfg.block_size * vocab * (std::mem::size_of::<bf16>() + std::mem::size_of::<f32>());
    ctx + draft
}

/// Which checkpoint flavor the head is.
pub fn mode_label(next_token_heads: bool, has_markov: bool, has_confidence: bool) -> &'static str {
    match (next_token_heads, has_markov, has_confidence) {
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

/// How a DSpark block draft resolves its rows — the stage-4 kernel-selection
/// decision (batch shape decides kernel selection).
// TODO(step-3b): a0's settled-shape descriptor will carry this alongside the
// batch shape; this enum is the decision half of that type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftRoute {
    /// Greedy: whole block in one batched argmax pass.
    Greedy,
    /// Sampled, no Markov head: per-row draws are independent.
    SampledIndependent,
    /// Sampled with a Markov head: row r depends on row r-1's draw.
    SampledMarkov,
}

pub fn draft_route(is_greedy: bool, has_markov: bool) -> DraftRoute {
    match (is_greedy, has_markov) {
        (true, _) => DraftRoute::Greedy,
        (false, false) => DraftRoute::SampledIndependent,
        (false, true) => DraftRoute::SampledMarkov,
    }
}

/// Per-slot draft-keep lengths from the confidence logits: sigmoid cumprod'd
/// into survival and fed to the goodput budget. `None` head (or empty
/// batch/rows) keeps the whole block (`usize::MAX`).
pub fn confidence_keep_lengths(
    confidence: Option<(&[f32], f32)>,
    b: usize,
    n: usize,
    sps: DsparkSps,
) -> Vec<usize> {
    let Some((logits, bias)) = confidence else {
        return vec![usize::MAX; b];
    };
    if n == 0 || b == 0 {
        return vec![usize::MAX; b];
    }
    let survivals: Vec<Vec<f32>> = (0..b)
        .map(|s| {
            let mut acc = 1.0f32;
            logits[s * n..(s + 1) * n]
                .iter()
                .map(|&h| {
                    let logit = h + bias;
                    acc *= 1.0 / (1.0 + (-logit).exp());
                    acc
                })
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = survivals.iter().map(Vec::as_slice).collect();
    dspark_verify_lens(&refs, sps)
}

/// Markov settle: the predecessor each row's bias depends on. Row 0..=first_row
/// anchor on the slot's anchor token; later rows on the previous row's token.
pub fn markov_prevs(anchors: &[u32], toks: &[u32], block: usize, first_row: usize) -> Vec<i32> {
    let b = anchors.len();
    let mut prevs = vec![0i32; b * block];
    for (s, &anchor) in anchors.iter().enumerate() {
        for row in 0..block {
            prevs[s * block + row] = if row <= first_row {
                anchor as i32
            } else {
                toks[s * block + row - 1] as i32
            };
        }
    }
    prevs
}

/// "the correction moved no row another row's bias depends on".
pub fn markov_settled(
    anchors: &[u32],
    toks: &[u32],
    next: &[u32],
    block: usize,
    first_row: usize,
) -> bool {
    let b = anchors.len();
    (0..b).all(|s| {
        (first_row..block.saturating_sub(1)).all(|r| toks[s * block + r] == next[s * block + r])
    })
}

/// Every id must name a reachable layer exactly once — a duplicate or
/// out-of-range id leaves `capture` nothing to write and hands the reader a
/// zero buffer indistinguishable from a real one.
pub fn validate_target_layer_ids(targets: &[i64], num_layers: usize) -> Result<()> {
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

/// Split the packed `[hidden, n_taps*hidden]` fc into per-tap `[hidden, hidden]`
/// matrices, so the concat becomes a sum of per-tap GEMMs (no per-token
/// gather).
pub fn split_fc_taps(fc_host: &[bf16], hidden: usize, n_taps: usize) -> Vec<Vec<bf16>> {
    (0..n_taps)
        .map(|t| {
            (0..hidden)
                .flat_map(|r| {
                    let row = &fc_host[r * n_taps * hidden..];
                    row[t * hidden..(t + 1) * hidden].iter().copied()
                })
                .collect()
        })
        .collect()
}

/// Gate-pad a draft q projection: head `h` lands at rows
/// `2h*head_dim..(2h+1)*head_dim`, odd bands zero — the trunk's fused prep
/// kernel assumes the gated q layout.
pub fn pad_gated_q(q_host: &[bf16], num_heads: usize, head_dim: usize, hidden: usize) -> Vec<bf16> {
    let mut q_padded = vec![bf16::ZERO; 2 * num_heads * head_dim * hidden];
    for h in 0..num_heads {
        let src = h * head_dim * hidden;
        let dst = 2 * h * head_dim * hidden;
        q_padded[dst..dst + head_dim * hidden]
            .copy_from_slice(&q_host[src..src + head_dim * hidden]);
    }
    q_padded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dspark_cfg(sliding_window: Option<usize>) -> DsparkConfig {
        DsparkConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window,
            layer_types: vec![],
            block_size: 4,
            mask_token_id: 0,
            target_layer_ids: vec![],
            next_token_heads: true,
        }
    }

    #[test]
    fn rng_deterministic_and_bounded() {
        assert_eq!(
            unit_uniform(Some(42), SALT_DRAW, 7),
            unit_uniform(Some(42), SALT_DRAW, 7)
        );
        for seed in [None, Some(0), Some(12345)] {
            for pos in 0..32 {
                let u = unit_uniform(seed, SALT_ACCEPT, pos);
                assert!((0.0..1.0).contains(&u));
            }
        }
        // Salts must give independent streams.
        assert_ne!(
            unit_uniform(Some(1), SALT_DRAW, 0),
            unit_uniform(Some(1), SALT_ACCEPT, 0)
        );
    }

    #[test]
    fn window_math() {
        assert_eq!(window_lo(None, 100), 0);
        assert_eq!(window_lo(Some(128), 200), 73); // 200 - (128-1)
        assert_eq!(window_lo(Some(128), 50), 0);
        let win = draft_attention_windows(Some(64), 100, 4, 10, 128).unwrap();
        assert_eq!(&win[..4], &[37, 38, 39, 40]); // lo = 100+r - 63
        assert_eq!(&win[4..], &[67, 66, 65, 64]); // kv_len = 104 - lo
        assert!(draft_attention_windows(None, 100, 4, 0, 8).is_err());
    }

    #[test]
    fn caps_and_bytes() {
        assert_eq!(rope_cap(32768, 8192, 16), 8208);
        assert_eq!(rope_cap(32768, 0, 16), 17); // max_total_tokens floored at 1
        assert_eq!(ctx_cap(None, 100, 16), 100);
        assert_eq!(ctx_cap(Some(64), 100, 16), 80);
        let cfg = dspark_cfg(None);
        assert_eq!(q_dim(&cfg), 8);
        assert_eq!(kv_dim(&cfg), 4);
        let bytes = slot_state_bytes(&cfg, 80, 2, 128000);
        assert_eq!(bytes, 2 * 80 * 4 * 2 * 2 + 4 * 128000 * (2 + 4));
    }

    #[test]
    fn confidence_keeps() {
        let sps = DsparkSps::default();
        assert_eq!(
            confidence_keep_lengths(None, 3, 4, sps),
            vec![usize::MAX; 3]
        );
        assert_eq!(
            confidence_keep_lengths(Some((&[1.0], 0.0)), 0, 1, sps),
            Vec::<usize>::new()
        );
        // High logits → survival ~1 → every row kept.
        let logits = [10.0f32; 8];
        assert_eq!(
            confidence_keep_lengths(Some((&logits, 0.0)), 2, 4, sps),
            vec![4, 4]
        );
        // Very negative logits → survival ~0 → nothing kept.
        let logits = [-100.0f32; 8];
        assert_eq!(
            confidence_keep_lengths(Some((&logits, 0.0)), 2, 4, sps),
            vec![0, 0]
        );
    }

    #[test]
    fn markov() {
        let anchors = [9u32, 19];
        let toks = [1, 2, 3, 4, 11, 12, 13, 14];
        let prevs = markov_prevs(&anchors, &toks, 4, 1);
        assert_eq!(&prevs[..4], &[9, 9, 2, 3]); // row<=1 anchors
        assert_eq!(&prevs[4..], &[19, 19, 12, 13]);
        assert!(markov_settled(&anchors, &toks, &toks, 4, 1));
        let mut next = toks;
        next[7] ^= 1; // row 3 of slot 1 — outside the checked range (..block-1)
        assert!(markov_settled(&anchors, &toks, &next, 4, 1));
        next[5] ^= 1; // row 1 — inside
        assert!(!markov_settled(&anchors, &toks, &next, 4, 1));
    }

    #[test]
    fn route_and_label() {
        assert_eq!(draft_route(true, true), DraftRoute::Greedy);
        assert_eq!(draft_route(false, false), DraftRoute::SampledIndependent);
        assert_eq!(draft_route(false, true), DraftRoute::SampledMarkov);
        assert_eq!(mode_label(true, true, true), "dspark+markov+confidence");
        assert_eq!(mode_label(false, false, false), "dflash-backbone");
        assert_eq!(first_row(false), 1);
        assert_eq!(first_row(true), 0);
    }

    #[test]
    fn target_layer_validation() {
        assert!(validate_target_layer_ids(&[-1, 0, 3], 4).is_ok());
        assert!(validate_target_layer_ids(&[0, 0], 4).is_err());
        assert!(validate_target_layer_ids(&[4], 4).is_err());
        assert!(validate_target_layer_ids(&[-2], 4).is_err());
    }

    #[test]
    fn fc_split_and_q_pad() {
        // 2 taps, hidden 4: fc[r, t*4..] interleaved per row; 4 rows × 8 cols.
        let fc: Vec<bf16> = (0..32u32).map(|i| bf16::from_f32(i as f32)).collect();
        let parts = split_fc_taps(&fc, 4, 2);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 16); // 4 rows × 4 cols per tap
        // tap 0 gets columns 0..4 of each row: row0 [0,1,2,3], row1 [4+2..]
        assert_eq!(bf16::to_f32(parts[0][0]), 0.0);
        assert_eq!(bf16::to_f32(parts[0][4]), 8.0); // row 1 starts at idx 8
        assert_eq!(bf16::to_f32(parts[1][0]), 4.0); // tap1 row0 col0
        // q pad: 2 heads, head_dim 2, hidden 2 → head h at rows 2h*2.
        let q: Vec<bf16> = (0..8u32).map(|i| bf16::from_f32(i as f32)).collect();
        let padded = pad_gated_q(&q, 2, 2, 2);
        assert_eq!(padded.len(), 16);
        assert_eq!(bf16::to_f32(padded[0]), 0.0);
        assert_eq!(bf16::to_f32(padded[4]), 0.0); // odd band zero
        assert_eq!(bf16::to_f32(padded[8]), 4.0); // head 1 at row 2*head_dim
    }
}
