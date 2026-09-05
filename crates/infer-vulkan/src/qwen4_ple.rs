//! Host reference for the `qwen4_exp` (Qwen3.8-Flash-Next) n-gram hash and PLE
//! layer — the component that decides what a token even *looks up*.
//!
//! Semantics are transcribed from `modeling_qwen4_exp.py`
//! (`Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextPLELayer`, and the
//! `_splitmix64` / `_build_layer_multipliers` / `_find_nth_prime_after`
//! helpers), not inferred from tensor names. No Vulkan here: this is the oracle
//! a future shader is diffed against.
//!
//! # Why the hash has to be bit-exact
//!
//! The n-gram table is 320,001,536 rows of 160 FP8 values. A hash that is off
//! by one bit does not degrade the output, it reads a *different, unrelated*
//! row — 47.68 GiB of table means every wrong index still lands on plausible
//! numbers. There is no "close enough" here, which is why every constant below
//! is stated and pinned rather than left implicit at a call site.
//!
//! # Derived constants for the on-box checkpoint
//!
//! With `vocab_size = 248320`, `ngram_size = 3`, `seed = 1234`,
//! `ple_layer_index = 0`:
//!
//! - `multiplier_max = (2^63 - 1) / 248320 = 37_143_089_710_272`
//! - `half_bound     = multiplier_max / 2   = 18_571_544_855_136`
//! - `layer_multipliers = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]`
//!
//! and the 16 per-head vocab sizes are the first 16 primes strictly above
//! `ngram_vocab_size_base - 1 = 19_999_999`:
//!
//! ```text
//! 20000003 20000023 20000033 20000047 20000059 20000063 20000069 20000077
//! 20000081 20000093 20000107 20000147 20000153 20000159 20000161 20000171
//! ```
//!
//! summing to `total_vocab_size = 320_001_446`, padded up to the next multiple
//! of `make_ngram_vocab_size_divisible_by = 128` to give **320,001,536**.
//!
//! All three derivations are confirmed by the checkpoint itself, which ships
//! `layer_multipliers` (I64[3]), `ngram_heads_vocab_sizes` (I64[16]) and
//! `ngram_heads_offsets` (I64[16]) as ordinary tensors, and by the table's own
//! geometry: 128 `shard_<i>.weight` tensors of `[2500012, 160]`, and
//! `128 * 2_500_012 == 320_001_536` exactly. The padding divisor being 128 is
//! what makes that 128-way split land on a whole number of rows.
//! `real_checkpoint_pins_the_hash_tables` re-reads all of it.
//!
//! # The two traps in the PLE forward
//!
//! 1. The residual branch adds the **un-normed** gated value; only the conv
//!    branch sees `norm_conv`'s output. Feeding the normed tensor to both is
//!    the natural-looking rewrite and it is wrong.
//! 2. `torch.sign(0) == 0`, so a gate that is exactly zero stays zero and does
//!    **not** become `sqrt(1e-6)`. `f32::signum` returns `1.0` for `+0.0`, so
//!    the obvious Rust spelling silently changes the model.
//!
//! # Precision
//!
//! The hash is integer and exact. The forward is a *host f32 reference*: dot
//! products and RMS means accumulate in `f64` and round once to `f32`, which is
//! more accurate than torch's f32 tree reduction, not bit-identical to it.
//! Compare against it with a tolerance, not with `==`.

use anyhow::{Result, ensure};

// --------------------------------------------------------------------------
// splitmix64 and the layer multipliers
// --------------------------------------------------------------------------

/// splitmix64's increment, `floor(2^64 / phi)`. The reference calls it
/// `_SPLITMIX_GAMMA` and reuses it as the per-index seed stride below.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// First finalizer multiplier (`_SPLITMIX_M1`).
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second finalizer multiplier (`_SPLITMIX_M2`).
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// `_PRIME_1`: the stride that decorrelates one PLE layer's seed from the next.
/// Only `ple_layer_index == 0` exists in this checkpoint, so it multiplies out
/// to zero — it is transcribed anyway because a second PLE layer would need it.
const SPLITMIX_LAYER_STRIDE: u64 = 10007;

/// `2^63 - 1`, the `torch.long` maximum the reference divides by to bound the
/// multipliers so `token * multiplier` cannot overflow int64.
const MAX_LONG: u64 = (1u64 << 63) - 1;

/// splitmix64's mix, exactly as `_splitmix64` spells it.
///
/// Python ints are unbounded and the reference masks to 64 bits after every
/// step; in Rust the wrapping is the type, so `wrapping_add`/`wrapping_mul`
/// *are* the `& _MASK64`.
#[must_use]
pub fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

/// `_build_layer_multipliers`: one odd multiplier per n-gram position.
///
/// The `2 * (x % half_bound) + 1` form is doing two jobs at once. The `+ 1`
/// forces the multiplier odd, so multiplying by it is a bijection modulo any
/// power of two and the low bits of `token` keep contributing. The
/// `% half_bound` with `half_bound = (i64::MAX / vocab) / 2` keeps
/// `2 * (x % half_bound) + 1` below `i64::MAX / vocab`, so `token * multiplier`
/// fits in an i64 for every in-range token — that bound is the only thing
/// standing between this hash and a silent wraparound.
#[must_use]
pub fn build_layer_multipliers(
    unigram_vocab_size: u64,
    ngram_size: usize,
    ple_layer_index: usize,
    seed: u64,
) -> Vec<i64> {
    let multiplier_max = MAX_LONG / unigram_vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(SPLITMIX_LAYER_STRIDE.wrapping_mul(ple_layer_index as u64));
    (0..ngram_size)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            // Bounded by `multiplier_max <= i64::MAX`, so the cast is exact.
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

/// `_is_prime`: trial division, transcribed. Kept simple on purpose — it runs
/// 16 times at construction over a ~200-wide window near 2e7.
fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// `_find_nth_prime_after(start, count)`: the `count`-th prime strictly greater
/// than `start`. Transcribed with the reference's restart-from-`start` shape
/// rather than an ascending single pass, so the two cannot drift.
#[must_use]
pub fn find_nth_prime_after(start: u64, count: usize) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

// --------------------------------------------------------------------------
// n-gram hash
// --------------------------------------------------------------------------

/// The `config` fields `Qwen4ExpTextNGramEmbedding` actually reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGramHashConfig {
    /// `ngram_size`: the widest n-gram, and also the number of multipliers.
    pub ngram_size: usize,
    /// `heads_per_ngram`: how many independent hash heads each n-gram order gets.
    pub heads_per_ngram: usize,
    /// `ngram_vocab_size_base`: primes are searched strictly above `base - 1`.
    pub ngram_vocab_size_base: u64,
    /// `make_ngram_vocab_size_divisible_by`: row-count padding of the table.
    pub make_ngram_vocab_size_divisible_by: u64,
    /// `vocab_size` of the *unigram* tokenizer — this bounds the multipliers.
    pub vocab_size: u64,
    /// `seed`.
    pub seed: u64,
    /// `eos_token_id`, the segment separator *and* the out-of-context fill.
    pub eos_token_id: i64,
    /// `ple_embed_dim`; `ple_embed_dim / ngram_heads` is the table's row width.
    pub ple_embed_dim: usize,
    /// Which entry of `ple_layer_ids` this is; shifts both the prime window and
    /// the multiplier seed so two PLE layers never share a hash.
    pub ple_layer_index: usize,
}

impl NGramHashConfig {
    /// The on-box `qwen3.8-flash-next-nvfp4` values, read from
    /// `config.json:text_config` — with `seed` absent there, so the
    /// `Qwen4ExpTextConfig` default of 1234 applies.
    #[must_use]
    pub fn qwen4_exp() -> Self {
        Self {
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            vocab_size: 248_320,
            seed: 1234,
            eos_token_id: 248_044,
            ple_embed_dim: 2560,
            ple_layer_index: 0,
        }
    }
}

/// Rolling token context for the hash, i.e. the reference's `conv_states[2]`.
///
/// The reference seeds it with `eos_token_id` and, on a first call whose
/// `input_ids` are shorter than `context_len`, left-pads the cached tokens with
/// `eos_token_id` before trimming. Starting from an EOS-filled buffer and
/// always keeping the last `context_len` of `context ++ input_ids` reproduces
/// both branches with one rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGramContext {
    tokens: Vec<i64>,
}

impl NGramContext {
    /// A fresh (sequence-start) context: `context_len` copies of EOS.
    #[must_use]
    pub fn new(hash: &NGramHash) -> Self {
        Self {
            tokens: vec![hash.cfg.eos_token_id; hash.context_len()],
        }
    }

    /// The `context_len` tokens that precede the next call's `input_ids`.
    #[must_use]
    pub fn tokens(&self) -> &[i64] {
        &self.tokens
    }

    /// Roll `input_ids` in, keeping the last `context_len` tokens.
    pub fn push(&mut self, input_ids: &[i64]) {
        let keep = self.tokens.len();
        self.tokens.extend_from_slice(input_ids);
        let drop = self.tokens.len() - keep;
        self.tokens.drain(..drop);
    }
}

/// `Qwen4ExpTextNGramEmbedding`'s index arithmetic, without the table.
#[derive(Debug, Clone)]
pub struct NGramHash {
    cfg: NGramHashConfig,
    head_vocab_sizes: Vec<i64>,
    head_offsets: Vec<i64>,
    layer_multipliers: Vec<i64>,
    total_vocab_size: u64,
    padded_vocab_size: u64,
}

impl NGramHash {
    /// Build the per-head prime table, offsets and multipliers.
    ///
    /// # Errors
    /// If the config cannot produce a usable table: a degenerate `ngram_size`
    /// or `heads_per_ngram`, a zero vocab, or a `ple_embed_dim` that does not
    /// divide evenly into `ngram_heads` row widths.
    pub fn new(cfg: NGramHashConfig) -> Result<Self> {
        ensure!(
            cfg.ngram_size >= 2,
            "ngram_size {} < 2 leaves no n-gram to hash",
            cfg.ngram_size
        );
        ensure!(cfg.heads_per_ngram > 0, "heads_per_ngram must be positive");
        ensure!(cfg.vocab_size > 0, "vocab_size must be positive");
        ensure!(
            cfg.ngram_vocab_size_base > 0,
            "ngram_vocab_size_base must be positive"
        );
        ensure!(
            cfg.make_ngram_vocab_size_divisible_by > 0,
            "make_ngram_vocab_size_divisible_by must be positive"
        );
        let ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        ensure!(
            cfg.ple_embed_dim.is_multiple_of(ngram_heads),
            "ple_embed_dim {} is not divisible by the {ngram_heads} n-gram heads",
            cfg.ple_embed_dim
        );

        let mut head_vocab_sizes = Vec::with_capacity(ngram_heads);
        let mut head_offsets = Vec::with_capacity(ngram_heads);
        let mut total_vocab_size = 0u64;
        for head_idx in 0..ngram_heads {
            // The prime window is indexed GLOBALLY across PLE layers, so a
            // second layer continues where this one stopped instead of
            // repeating it — same tokens, different rows.
            let global_head_idx = cfg.ple_layer_index * ngram_heads + head_idx;
            let size = find_nth_prime_after(cfg.ngram_vocab_size_base - 1, global_head_idx + 1);
            head_vocab_sizes.push(size as i64);
            head_offsets.push(total_vocab_size as i64);
            total_vocab_size += size;
        }

        let divisor = cfg.make_ngram_vocab_size_divisible_by;
        let padded_vocab_size = total_vocab_size.div_ceil(divisor) * divisor;
        let layer_multipliers = build_layer_multipliers(
            cfg.vocab_size,
            cfg.ngram_size,
            cfg.ple_layer_index,
            cfg.seed,
        );

        Ok(Self {
            cfg,
            head_vocab_sizes,
            head_offsets,
            layer_multipliers,
            total_vocab_size,
            padded_vocab_size,
        })
    }

    #[must_use]
    pub fn config(&self) -> &NGramHashConfig {
        &self.cfg
    }

    /// `(ngram_size - 1) * heads_per_ngram` — 16 for this checkpoint.
    #[must_use]
    pub fn ngram_heads(&self) -> usize {
        (self.cfg.ngram_size - 1) * self.cfg.heads_per_ngram
    }

    /// `ngram_size - 1`: how many previous tokens the widest n-gram needs.
    #[must_use]
    pub fn context_len(&self) -> usize {
        self.cfg.ngram_size - 1
    }

    /// `ple_embed_dim / ngram_heads` — the width of one table row (160 here).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.cfg.ple_embed_dim / self.ngram_heads()
    }

    /// `ngram_heads_vocab_sizes`: the per-head prime modulus.
    #[must_use]
    pub fn head_vocab_sizes(&self) -> &[i64] {
        &self.head_vocab_sizes
    }

    /// `ngram_heads_offsets`: where each head's block starts in the flat table.
    #[must_use]
    pub fn head_offsets(&self) -> &[i64] {
        &self.head_offsets
    }

    /// `layer_multipliers`, one per n-gram position (`ngram_size` of them).
    #[must_use]
    pub fn layer_multipliers(&self) -> &[i64] {
        &self.layer_multipliers
    }

    /// Sum of the per-head primes, before padding.
    #[must_use]
    pub fn total_vocab_size(&self) -> u64 {
        self.total_vocab_size
    }

    /// Row count of the actual embedding table, i.e. `total_vocab_size` rounded
    /// up to `make_ngram_vocab_size_divisible_by`.
    #[must_use]
    pub fn padded_vocab_size(&self) -> u64 {
        self.padded_vocab_size
    }

    /// `_shift_right_ignore_eos`: shift `tokens` right by `shift`, but never
    /// across an EOS.
    ///
    /// A position whose distance from its segment start is less than `shift`
    /// would have to read a token belonging to the *previous* document, so it
    /// reads `eos_token_id` instead. The segment start is one past the last EOS
    /// strictly before the position, which means an EOS token still sees its
    /// own left context (it ends a segment) while the token after it starts
    /// fresh.
    #[must_use]
    pub fn shift_right_ignore_eos(&self, tokens: &[i64], shift: usize) -> Vec<i64> {
        if shift == 0 {
            return tokens.to_vec();
        }
        let eos = self.cfg.eos_token_id;
        let shift_i = shift as i64;
        // torch builds this as `cummax(eos_positions)` shifted right by one;
        // scanning left to right, that is just "last EOS index seen so far",
        // read before the current token is folded in.
        let mut previous_eos: i64 = -1;
        let mut out = Vec::with_capacity(tokens.len());
        for (t, &token) in tokens.iter().enumerate() {
            let position_in_segment = t as i64 - (previous_eos + 1);
            let source = t as i64 - shift_i;
            let valid = position_in_segment >= shift_i && source >= 0;
            out.push(if valid { tokens[source as usize] } else { eos });
            if token == eos {
                previous_eos = t as i64;
            }
        }
        out
    }

    /// Row indices into the n-gram table for every token of `input_ids`.
    ///
    /// Returns `input_ids.len() * ngram_heads` ids, row-major and head-minor:
    /// `out[t * ngram_heads + h]`. That is the order `flatten(-2)` produces, so
    /// the embedding fed to the PLE layer is head 0's 160 floats, then head
    /// 1's, and so on — heads `0..heads_per_ngram` are the bigram hashes and
    /// `heads_per_ngram..2*heads_per_ngram` the trigram ones.
    ///
    /// `context` is *not* advanced; call [`NGramContext::push`] afterwards.
    ///
    /// # Errors
    /// If the context is the wrong length, or a token falls outside
    /// `0..vocab_size`. The range check is not pedantry: the multiplier bound
    /// that keeps `token * multiplier` inside an i64 is derived from
    /// `vocab_size`, so an out-of-range id would wrap and hash somewhere else.
    pub fn row_ids(&self, context: &NGramContext, input_ids: &[i64]) -> Result<Vec<i64>> {
        ensure!(
            context.tokens.len() == self.context_len(),
            "context holds {} tokens, expected {}",
            context.tokens.len(),
            self.context_len()
        );
        let vocab = self.cfg.vocab_size as i64;
        for &token in context.tokens.iter().chain(input_ids) {
            ensure!(
                (0..vocab).contains(&token),
                "token id {token} outside 0..{vocab}"
            );
        }

        let mut history = context.tokens.clone();
        history.extend_from_slice(input_ids);
        let shifted: Vec<Vec<i64>> = (0..self.cfg.ngram_size)
            .map(|shift| self.shift_right_ignore_eos(&history, shift))
            .collect();

        let heads = self.ngram_heads();
        let first_new = history.len() - input_ids.len();
        let mut out = vec![0i64; input_ids.len() * heads];
        for ngram in 2..=self.cfg.ngram_size {
            let head_start = (ngram - 2) * self.cfg.heads_per_ngram;
            for (row, out_row) in out.chunks_exact_mut(heads).enumerate() {
                let t = first_new + row;
                // torch int64 multiply wraps; `wrapping_mul` says so out loud.
                // With in-range tokens the multiplier bound above makes every
                // product non-negative and overflow-free, so `mixed >= 0`.
                let mut mixed = shifted[0][t].wrapping_mul(self.layer_multipliers[0]);
                for (row_shifted, &multiplier) in shifted
                    .iter()
                    .zip(&self.layer_multipliers)
                    .take(ngram)
                    .skip(1)
                {
                    mixed ^= row_shifted[t].wrapping_mul(multiplier);
                }
                // The invariant the `rem_euclid` below leans on. XOR of
                // non-negative i64s keeps the sign bit clear, so this can only
                // trip if a product overflowed, i.e. if the multiplier bound
                // or the token range check were wrong.
                debug_assert!(mixed >= 0, "n-gram mix overflowed into the sign bit");
                let block = head_start..head_start + self.cfg.heads_per_ngram;
                let dsts = out_row[block.clone()].iter_mut();
                let sizes = &self.head_vocab_sizes[block.clone()];
                let offsets = &self.head_offsets[block];
                for ((dst, &size), &offset) in dsts.zip(sizes).zip(offsets) {
                    // `torch.remainder` follows the divisor's sign (Python
                    // modulo), which for a positive prime is `rem_euclid`.
                    *dst = mixed.rem_euclid(size) + offset;
                }
            }
        }
        Ok(out)
    }
}

// --------------------------------------------------------------------------
// PLE layer
// --------------------------------------------------------------------------

/// Shapes `Qwen4ExpTextPLELayer` is built from.
#[derive(Debug, Clone, PartialEq)]
pub struct PleConfig {
    /// `hidden_size` — also the RMSNorm `group_size`.
    pub hidden_size: usize,
    /// `hc_count`: how many hyper-connection streams share one PLE value.
    pub hc_count: usize,
    /// `ple_embed_dim`: width of the concatenated n-gram embedding.
    pub ple_embed_dim: usize,
    /// `ple_conv_kernel_size` (4).
    pub conv_kernel_size: usize,
    /// The conv's dilation, which the reference sets to `ngram_size` (3).
    pub conv_dilation: usize,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
}

impl PleConfig {
    /// The on-box `qwen3.8-flash-next-nvfp4` values.
    #[must_use]
    pub fn qwen4_exp() -> Self {
        Self {
            hidden_size: 2560,
            hc_count: 4,
            ple_embed_dim: 2560,
            conv_kernel_size: 4,
            conv_dilation: 3,
            rms_norm_eps: 1e-6,
        }
    }

    /// `hidden_size * hc_count` — 10240, the width of every PLE tensor except
    /// `value_proj`'s output.
    #[must_use]
    pub fn hc_hidden(&self) -> usize {
        self.hidden_size * self.hc_count
    }

    /// `(conv_kernel_size - 1) * conv_dilation` = 9 steps of history.
    #[must_use]
    pub fn short_conv_state_len(&self) -> usize {
        (self.conv_kernel_size - 1) * self.conv_dilation
    }
}

/// PLE weights as f32, in the checkpoint's own layout.
///
/// Linear weights are `[out][in]` row-major — the safetensors `shape` as
/// written, i.e. `SafeTensorInfo::dims` reversed back — so `key_proj` is
/// indexed `[o * ple_embed_dim + i]`.
#[derive(Debug, Clone)]
pub struct PleWeights {
    /// `key_proj.weight`, `[hc_hidden][ple_embed_dim]`.
    pub key_proj: Vec<f32>,
    /// `value_proj.weight`, `[hidden_size][ple_embed_dim]`.
    pub value_proj: Vec<f32>,
    /// `norm_key.weight`, `[hc_hidden]`.
    pub norm_key: Vec<f32>,
    /// `norm_query.weight`, `[hc_hidden]`.
    pub norm_query: Vec<f32>,
    /// `norm_conv.weight`, `[hc_hidden]`.
    pub norm_conv: Vec<f32>,
    /// `conv1d.weight`, `[hc_hidden][1][kernel]` flattened to
    /// `[hc_hidden][kernel]` — it is depthwise, so the middle axis is always 1.
    pub conv1d: Vec<f32>,
}

/// The PLE short conv's rolling history, i.e. the reference's `conv_states[1]`.
///
/// `short_conv_state_len` time steps of `hc_hidden` channels, time-major and
/// zero-filled at sequence start. Zero rather than EOS because this state holds
/// activations, not tokens, and `update_conv_state` left-pads a short prefill
/// with literal zeros.
#[derive(Debug, Clone, PartialEq)]
pub struct PleConvState {
    rows: Vec<f32>,
    hc_hidden: usize,
}

impl PleConvState {
    /// A fresh, all-zero state.
    #[must_use]
    pub fn zeros(cfg: &PleConfig) -> Self {
        Self {
            rows: vec![0.0; cfg.short_conv_state_len() * cfg.hc_hidden()],
            hc_hidden: cfg.hc_hidden(),
        }
    }

    /// The stored history, time-major: `rows()[t * hc_hidden + c]`.
    #[must_use]
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    /// The device PLE advances the ring on the GPU and hands it back; this is
    /// how the host state stays canonical without re-deriving the shift.
    pub fn rows_mut(&mut self) -> &mut [f32] {
        &mut self.rows
    }
}

/// `Qwen4ExpTextRMSNorm` with a `group_size`, one row at a time.
///
/// Normalisation is per group of `group_size` channels — with `hc_hidden`
/// 10240 and `group_size` 2560 that is four independent RMS statistics, one per
/// hyper-connection stream — but the learned scale applies to the flat row, and
/// it is `1.0 + weight` (the parameter is zero-initialised in this model, not
/// one-initialised).
///
/// # Panics
/// If `x.len() != weight.len()`, or `x.len()` is not a multiple of
/// `group_size`.
#[must_use]
pub fn rms_norm_grouped(x: &[f32], weight: &[f32], group_size: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "RMSNorm weight width");
    assert!(
        group_size > 0 && x.len().is_multiple_of(group_size),
        "group_size"
    );
    let mut out = Vec::with_capacity(x.len());
    for group in x.chunks_exact(group_size) {
        let mean: f64 = group
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / group_size as f64;
        let scale = (mean + f64::from(eps)).sqrt().recip();
        out.extend(group.iter().map(|&v| (f64::from(v) * scale) as f32));
    }
    for (o, &w) in out.iter_mut().zip(weight) {
        *o *= 1.0 + w;
    }
    out
}

/// `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`.
///
/// TRAP: `torch.sign(0) == 0`, so an exactly-zero gate stays zero. Rust's
/// `f32::signum` returns `+1.0` for `+0.0` and `-1.0` for `-0.0`, which would
/// turn a zero gate into `±1e-3` and shift `sigmoid` off 0.5. Hence the
/// explicit three-way comparison.
#[must_use]
pub fn signed_sqrt_gate(gate: f32) -> f32 {
    if gate.is_nan() {
        return gate; // torch.sign(NaN) is NaN; propagate rather than sign it 0.
    }
    let sign = if gate > 0.0 {
        1.0
    } else if gate < 0.0 {
        -1.0
    } else {
        0.0
    };
    sign * gate.abs().max(1e-6).sqrt()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `y[o] = sum_i weight[o * in_dim + i] * x[i]`, accumulating in f64.
fn linear_row(weight: &[f32], in_dim: usize, x: &[f32], out: &mut [f32]) {
    for (row, y) in weight.chunks_exact(in_dim).zip(out.iter_mut()) {
        let acc: f64 = row
            .iter()
            .zip(x)
            .map(|(&w, &v)| f64::from(w) * f64::from(v))
            .sum();
        *y = acc as f32;
    }
}

/// `Qwen4ExpTextPLELayer`, host f32.
#[derive(Debug, Clone)]
pub struct PleLayer {
    cfg: PleConfig,
    weights: PleWeights,
}

impl PleLayer {
    /// Validate every weight against the config once, so the forward can index
    /// without re-checking.
    ///
    /// # Errors
    /// If any weight's element count disagrees with the config's shapes.
    pub fn new(cfg: PleConfig, weights: PleWeights) -> Result<Self> {
        ensure!(cfg.hidden_size > 0, "hidden_size must be positive");
        ensure!(cfg.hc_count > 0, "hc_count must be positive");
        ensure!(cfg.ple_embed_dim > 0, "ple_embed_dim must be positive");
        ensure!(
            cfg.conv_kernel_size > 0,
            "conv_kernel_size must be positive"
        );
        ensure!(cfg.conv_dilation > 0, "conv_dilation must be positive");
        let hc_hidden = cfg.hc_hidden();
        let checks: [(&str, usize, usize); 6] = [
            (
                "key_proj",
                weights.key_proj.len(),
                hc_hidden * cfg.ple_embed_dim,
            ),
            (
                "value_proj",
                weights.value_proj.len(),
                cfg.hidden_size * cfg.ple_embed_dim,
            ),
            ("norm_key", weights.norm_key.len(), hc_hidden),
            ("norm_query", weights.norm_query.len(), hc_hidden),
            ("norm_conv", weights.norm_conv.len(), hc_hidden),
            (
                "conv1d",
                weights.conv1d.len(),
                hc_hidden * cfg.conv_kernel_size,
            ),
        ];
        for (name, got, want) in checks {
            ensure!(
                got == want,
                "{name}.weight has {got} elements, expected {want}"
            );
        }
        Ok(Self { cfg, weights })
    }

    #[must_use]
    pub fn config(&self) -> &PleConfig {
        &self.cfg
    }

    /// The dilated depthwise conv plus its SiLU, and the state roll.
    ///
    /// `x` is `[seq][hc_hidden]`, time-major. With kernel 4 and dilation 3 the
    /// receptive field is 10 steps wide but only 4 are read: **t-9, t-6, t-3,
    /// t**, weighted by `conv1d[c][0..4]` in that order. The reference gets
    /// there by concatenating the 9-step state, padding 9 more zeros and
    /// slicing the last `9 + seq` — which is exactly "state then x" whenever
    /// the state is full, and a zero left-pad at sequence start.
    ///
    /// # Panics
    /// If `x` is not a whole number of `hc_hidden`-wide rows, or the state was
    /// built for a different width.
    pub fn short_conv(&self, x: &[f32], state: &mut PleConvState) -> Vec<f32> {
        let hc_hidden = self.cfg.hc_hidden();
        assert_eq!(state.hc_hidden, hc_hidden, "conv state width");
        assert_eq!(x.len() % hc_hidden, 0, "conv input width");
        let state_len = self.cfg.short_conv_state_len();
        let kernel = self.cfg.conv_kernel_size;
        let dilation = self.cfg.conv_dilation;

        let mut history = Vec::with_capacity(state.rows.len() + x.len());
        history.extend_from_slice(&state.rows);
        history.extend_from_slice(x);

        let mut out = vec![0.0f32; x.len()];
        for (t, out_row) in out.chunks_exact_mut(hc_hidden).enumerate() {
            for (c, y) in out_row.iter_mut().enumerate() {
                let taps = &self.weights.conv1d[c * kernel..(c + 1) * kernel];
                let acc: f64 = taps
                    .iter()
                    .enumerate()
                    .map(|(k, &w)| {
                        // history row `t + k*dilation` is absolute time
                        // `t - state_len + k*dilation`, so k = kernel-1 is the
                        // current step and k = 0 reaches back `state_len`.
                        let v = history[(t + k * dilation) * hc_hidden + c];
                        f64::from(w) * f64::from(v)
                    })
                    .sum();
                *y = silu(acc as f32);
            }
        }

        // Keep the last `state_len` rows of state ++ x, matching
        // `update_conv_state`'s trim. A prefill shorter than the state keeps
        // its leading zeros, which is the same thing.
        let keep_from = history.len() - state_len * hc_hidden;
        state.rows.clear();
        state.rows.extend_from_slice(&history[keep_from..]);
        out
    }

    /// Full PLE forward.
    ///
    /// `embeddings` is `[seq][ple_embed_dim]` — the gathered n-gram rows,
    /// concatenated head-major, as [`NGramHash::row_ids`] documents.
    /// `hidden_states` is `[seq][hc_hidden]`, the hyper-connection stream.
    /// `conv_mask` is the reference's `apply_mask_to_padding_states` mask: one
    /// flag per step, `false` zeroing that step before it can enter the conv or
    /// the state.
    ///
    /// Returns `[seq][hc_hidden]`, which the decoder *adds* to `hidden_states`.
    ///
    /// # Errors
    /// If the inputs are not whole rows of the expected widths, or disagree on
    /// the sequence length.
    pub fn forward(
        &self,
        embeddings: &[f32],
        hidden_states: &[f32],
        state: &mut PleConvState,
        conv_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let hidden = self.cfg.hidden_size;
        let hc_hidden = self.cfg.hc_hidden();
        let embed = self.cfg.ple_embed_dim;
        ensure!(
            embeddings.len().is_multiple_of(embed),
            "embeddings length {} is not a multiple of ple_embed_dim {embed}",
            embeddings.len()
        );
        let seq = embeddings.len() / embed;
        ensure!(
            hidden_states.len() == seq * hc_hidden,
            "hidden_states holds {} floats, expected {seq} x {hc_hidden}",
            hidden_states.len()
        );
        if let Some(mask) = conv_mask {
            ensure!(
                mask.len() == seq,
                "conv_mask holds {} flags, expected {seq}",
                mask.len()
            );
        }
        ensure!(state.hc_hidden == hc_hidden, "conv state width");

        // The gate divisor is sqrt(hidden_size), scaled like an attention
        // logit: one dot product per hyper-connection stream.
        let gate_scale = f64::from(hidden as f32).sqrt();
        let mut gated = vec![0.0f32; seq * hc_hidden];
        let mut key = vec![0.0f32; hc_hidden];
        let mut value = vec![0.0f32; hidden];

        for (t, row) in gated.chunks_exact_mut(hc_hidden).enumerate() {
            let emb = &embeddings[t * embed..(t + 1) * embed];
            linear_row(&self.weights.key_proj, embed, emb, &mut key);
            linear_row(&self.weights.value_proj, embed, emb, &mut value);

            let key_normed =
                rms_norm_grouped(&key, &self.weights.norm_key, hidden, self.cfg.rms_norm_eps);
            let query_normed = rms_norm_grouped(
                &hidden_states[t * hc_hidden..(t + 1) * hc_hidden],
                &self.weights.norm_query,
                hidden,
                self.cfg.rms_norm_eps,
            );

            for (stream, chunk) in row.chunks_exact_mut(hidden).enumerate() {
                let k = &key_normed[stream * hidden..(stream + 1) * hidden];
                let q = &query_normed[stream * hidden..(stream + 1) * hidden];
                let dot: f64 = k
                    .iter()
                    .zip(q)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let weight = sigmoid(signed_sqrt_gate((dot / gate_scale) as f32));
                for (dst, &v) in chunk.iter_mut().zip(&value) {
                    *dst = weight * v;
                }
            }
        }

        if let Some(mask) = conv_mask {
            for (row, &keep) in gated.chunks_exact_mut(hc_hidden).zip(mask) {
                if !keep {
                    row.fill(0.0);
                }
            }
        }

        let mut gated_normed = Vec::with_capacity(gated.len());
        for src in gated.chunks_exact(hc_hidden) {
            gated_normed.extend(rms_norm_grouped(
                src,
                &self.weights.norm_conv,
                hidden,
                self.cfg.rms_norm_eps,
            ));
        }
        // The reference masks both branches. The norm of a zeroed row is zero
        // anyway, but `1.0 + norm_conv.weight` is not, so re-masking matters
        // only if eps ever let a zero row through — do it regardless, because
        // the reference does.
        if let Some(mask) = conv_mask {
            for (row, &keep) in gated_normed.chunks_exact_mut(hc_hidden).zip(mask) {
                if !keep {
                    row.fill(0.0);
                }
            }
        }

        let conv = self.short_conv(&gated_normed, state);
        // TRAP: the residual is the UN-normed gated value. Only the conv branch
        // consumes `norm_conv`'s output; reusing the normed tensor here is the
        // natural-looking rewrite and it changes the model.
        let mut out = gated;
        for (o, &c) in out.iter_mut().zip(&conv) {
            *o += c;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 16 per-head moduli, computed outside this crate and then confirmed
    /// against the checkpoint's own `ngram_heads_vocab_sizes`.
    const EXPECTED_PRIMES: [i64; 16] = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];
    const EXPECTED_MULTIPLIERS: [i64; 3] =
        [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
    const EXPECTED_TOTAL: u64 = 320_001_446;
    const EXPECTED_PADDED: u64 = 320_001_536;
    /// Measured: 128 `shard_<i>.weight` tensors of `[2500012, 160]`.
    const CHECKPOINT_SHARDS: u64 = 128;
    const CHECKPOINT_ROWS_PER_SHARD: u64 = 2_500_012;
    const EOS: i64 = 248_044;

    fn hash() -> NGramHash {
        NGramHash::new(NGramHashConfig::qwen4_exp()).expect("build hash")
    }

    /// Independent trial-division search, written here so the prime table is
    /// not checked against the code that produced it.
    fn primes_after(start: u64, count: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(count);
        let mut candidate = start + 1;
        while out.len() < count {
            let mut prime = candidate >= 2;
            let mut d = 2u64;
            while d * d <= candidate {
                if candidate.is_multiple_of(d) {
                    prime = false;
                    break;
                }
                d += 1;
            }
            if prime {
                out.push(candidate);
            }
            candidate += 1;
        }
        out
    }

    /// splitmix64 re-derived from the algorithm's published form, using u128
    /// arithmetic and an explicit 64-bit mask instead of Rust's wrapping ops —
    /// a different spelling, so a transcription slip in either shows up.
    fn splitmix64_via_u128(value: u64) -> u64 {
        const MASK: u128 = (1u128 << 64) - 1;
        let mut v = (u128::from(value) + u128::from(SPLITMIX_GAMMA)) & MASK;
        v = ((v ^ (v >> 30)) * u128::from(SPLITMIX_M1)) & MASK;
        v = ((v ^ (v >> 27)) * u128::from(SPLITMIX_M2)) & MASK;
        ((v ^ (v >> 31)) & MASK) as u64
    }

    #[test]
    fn splitmix64_matches_an_independent_spelling_and_pinned_outputs() {
        // The three seeds `_build_layer_multipliers` feeds it for
        // ple_layer_index 0: 1234 + GAMMA * (index + 1).
        let seeds: [u64; 3] = [
            0x9E37_79B9_7F4A_80E7,
            0x3C6E_F372_FE94_FCFC,
            0xDAA6_6D2C_7DDF_7911,
        ];
        let expected: [u64; 3] = [
            0x97C7_A136_4DF0_6524,
            0x33BE_FAE4_9BC0_25DA,
            0x4E62_41F2_52D0_A033,
        ];
        for (&seed, &want) in seeds.iter().zip(&expected) {
            assert_eq!(splitmix64(seed), want, "splitmix64(0x{seed:016X})");
            assert_eq!(splitmix64(seed), splitmix64_via_u128(seed));
        }
        // Pin the seed schedule too, so a wrong stride cannot hide behind a
        // right mix.
        for (index, &seed) in seeds.iter().enumerate() {
            let built = 1234u64.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            assert_eq!(built, seed, "seed schedule at index {index}");
        }
    }

    #[test]
    fn layer_multipliers_are_odd_bounded_and_pinned() {
        let h = hash();
        assert_eq!(h.layer_multipliers(), EXPECTED_MULTIPLIERS);

        // The bound that keeps `token * multiplier` inside an i64.
        let multiplier_max = ((1u64 << 63) - 1) / 248_320;
        assert_eq!(multiplier_max, 37_143_089_710_272);
        assert_eq!((multiplier_max / 2).max(1), 18_571_544_855_136);
        for &m in h.layer_multipliers() {
            assert_eq!(m % 2, 1, "multipliers must be odd: {m}");
            assert!(m < multiplier_max as i64, "multiplier {m} breaks the bound");
            let widest = (248_320i64 - 1).checked_mul(m).expect("no i64 overflow");
            assert!(widest > 0, "widest product must stay non-negative");
        }
        // A second PLE layer must not reuse layer 0's multipliers.
        let other = build_layer_multipliers(248_320, 3, 1, 1234);
        assert_ne!(other, EXPECTED_MULTIPLIERS.to_vec());
    }

    #[test]
    fn head_vocab_sizes_are_the_first_sixteen_primes_after_the_base() {
        let h = hash();
        assert_eq!(h.ngram_heads(), 16);
        assert_eq!(h.head_dim(), 160);
        assert_eq!(h.head_vocab_sizes(), EXPECTED_PRIMES);

        // Independent search, and the "strictly after base - 1" boundary: the
        // window opens at 19_999_999, so 20_000_003 is in and nothing below.
        let independent = primes_after(19_999_999, 16);
        let as_i64: Vec<i64> = independent.iter().map(|&p| p as i64).collect();
        assert_eq!(as_i64, EXPECTED_PRIMES);
        assert!(independent[0] > 19_999_999);

        // Offsets are the running sum, so head h's block is contiguous.
        let mut running = 0i64;
        for (idx, (&off, &size)) in h
            .head_offsets()
            .iter()
            .zip(h.head_vocab_sizes())
            .enumerate()
        {
            assert_eq!(off, running, "offset for head {idx}");
            running += size;
        }
        assert_eq!(running as u64, EXPECTED_TOTAL);
    }

    /// `_find_nth_prime_after(start, count)` searches STRICTLY after `start`,
    /// which is why the reference passes `base - 1` rather than `base`. On this
    /// checkpoint `base` is 20_000_000 and composite, so the off-by-one is
    /// invisible — pin it with a config whose base is itself prime, where
    /// including or excluding it changes every head.
    #[test]
    fn the_prime_window_opens_strictly_after_its_argument() {
        assert_eq!(find_nth_prime_after(6, 1), 7);
        assert_eq!(find_nth_prime_after(7, 1), 11, "strictly after, not from");
        assert_eq!(find_nth_prime_after(1, 3), 5);

        let mut cfg = NGramHashConfig::qwen4_exp();
        cfg.ngram_vocab_size_base = 20_000_003; // prime, unlike the real base
        let h = NGramHash::new(cfg).expect("hash");
        assert_eq!(
            h.head_vocab_sizes()[0],
            20_000_003,
            "a prime base is head 0"
        );
        assert_eq!(h.head_vocab_sizes()[1], 20_000_023);
    }

    /// A second PLE layer must hash into a DIFFERENT window of primes: the
    /// per-head index is global, so layer 1's head 0 is the 17th prime after
    /// the base, not the 1st. Only `ple_layer_index == 0` exists on this
    /// checkpoint, which is exactly why the offset needs pinning here.
    #[test]
    fn a_second_ple_layer_gets_the_next_window_of_primes() {
        let mut cfg = NGramHashConfig::qwen4_exp();
        cfg.ple_layer_index = 1;
        let h = NGramHash::new(cfg).expect("hash");
        // Primes 17, 18, 19 after 19_999_999, computed independently below.
        let window = primes_after(19_999_999, 32);
        assert_eq!(h.head_vocab_sizes()[0], window[16] as i64);
        assert_eq!(h.head_vocab_sizes()[0], 20_000_213);
        assert_eq!(h.head_vocab_sizes()[1], 20_000_221);
        assert_eq!(h.total_vocab_size(), 320_005_046);
        // ...and it must not collide with layer 0's table.
        let zero = hash();
        assert_ne!(h.head_vocab_sizes(), zero.head_vocab_sizes());
        assert_ne!(h.layer_multipliers(), zero.layer_multipliers());
    }

    /// The multiplier bound only just fits: 248_319 x 23_703_573_157_769 is
    /// ~5.9e18, inside i64 but with under a factor of 1.6 to spare. If a
    /// product wrapped, `mixed` would go negative and — with a truncating `%`
    /// instead of `rem_euclid` — the ids would fall below their head offsets.
    #[test]
    fn the_largest_token_still_hashes_inside_its_head_block() {
        let h = hash();
        let max = h.config().vocab_size as i64 - 1;
        let ids = h
            .row_ids(&NGramContext::new(&h), &[max, max, max])
            .expect("row ids");
        for (t, row) in ids.chunks_exact(16).enumerate() {
            for (head, &id) in row.iter().enumerate() {
                let lo = h.head_offsets()[head];
                let hi = lo + h.head_vocab_sizes()[head];
                assert!((lo..hi).contains(&id), "t{t} head{head} id {id} escaped");
            }
        }
    }

    #[test]
    fn total_vocab_pads_up_to_the_checkpoint_row_count() {
        let h = hash();
        assert_eq!(h.total_vocab_size(), EXPECTED_TOTAL);
        assert_eq!(h.padded_vocab_size(), EXPECTED_PADDED);
        // The padding divisor is what lets the table split 128 ways evenly.
        assert_eq!(h.padded_vocab_size() % 128, 0);
        assert_eq!(
            h.padded_vocab_size(),
            CHECKPOINT_SHARDS * CHECKPOINT_ROWS_PER_SHARD
        );
        assert!(h.padded_vocab_size() >= h.total_vocab_size());
        assert!(h.padded_vocab_size() - h.total_vocab_size() < 128);
    }

    #[test]
    fn shift_zero_is_the_identity() {
        let h = hash();
        let tokens = [1i64, 2, EOS, 3];
        assert_eq!(h.shift_right_ignore_eos(&tokens, 0), tokens);
    }

    #[test]
    fn shift_does_not_bleed_across_eos() {
        let h = hash();
        //           idx:  0   1    2   3   4   5
        let tokens = [10i64, 11, EOS, 12, 13, 14];

        // shift 1: index 0 has no source at all; index 3 is the first token of
        // a new segment, so its "previous token" is the fill, not the EOS.
        assert_eq!(
            h.shift_right_ignore_eos(&tokens, 1),
            [EOS, 10, 11, EOS, 12, 13]
        );
        // shift 2: indices 3 and 4 are less than 2 steps into their segment.
        assert_eq!(
            h.shift_right_ignore_eos(&tokens, 2),
            [EOS, EOS, 10, EOS, EOS, 12]
        );
        // The EOS at index 2 still sees its OWN left context — the segment
        // boundary is one PAST it, not at it.
        assert_eq!(h.shift_right_ignore_eos(&tokens, 1)[2], 11);

        // With no EOS the same call is a plain right shift with EOS fill, which
        // is what makes the assertions above about segments and not clamping.
        let plain = [10i64, 11, 12, 13];
        assert_eq!(h.shift_right_ignore_eos(&plain, 2), [EOS, EOS, 10, 11]);
    }

    /// Row ids for `context = [EOS, EOS]`, `input = [7, 11, EOS, 5]`, computed
    /// outside this crate from the reference formula and pasted in whole. A
    /// shared bug between implementation and expectation is impossible: these
    /// are literals, not a recomputation.
    #[test]
    fn row_ids_are_bit_exact() {
        let h = hash();
        let ctx = NGramContext::new(&h);
        assert_eq!(ctx.tokens(), [EOS, EOS]);
        let ids = h.row_ids(&ctx, &[7, 11, EOS, 5]).expect("row ids");

        #[rustfmt::skip]
        let expected: [[i64; 16]; 4] = [
            [2_927_653, 34_980_843, 54_748_278, 66_612_378, 97_814_964, 109_013_870,
             126_560_393, 151_352_333, 167_888_935, 182_235_580, 215_170_017, 237_467_519,
             247_510_681, 278_779_700, 296_141_806, 304_994_522],
            [3_518_994, 23_525_975, 43_529_577, 79_534_759, 87_539_255, 103_540_813,
             127_543_174, 159_546_370, 168_682_726, 193_417_972, 204_093_272, 222_522_584,
             258_665_398, 275_167_591, 280_748_216, 309_250_551],
            [4_679_147, 20_352_667, 52_599_150, 74_683_044, 98_198_172, 106_977_280,
             121_027_876, 141_408_284, 161_918_623, 184_949_976, 213_836_989, 230_977_695,
             243_605_751, 277_292_288, 282_089_598, 307_840_018],
            [15_389_869, 39_778_609, 55_713_969, 62_213_332, 88_817_728, 118_483_999,
             133_731_511, 155_458_159, 179_763_390, 197_956_758, 205_378_969, 220_499_474,
             242_466_248, 265_658_744, 293_662_119, 315_720_898],
        ];
        assert_eq!(ids, expected.concat());

        // Every id must land inside its own head's block, or the gather reads
        // another head's rows.
        for (t, row) in ids.chunks_exact(16).enumerate() {
            for (head, &id) in row.iter().enumerate() {
                let lo = h.head_offsets()[head];
                let hi = lo + h.head_vocab_sizes()[head];
                assert!((lo..hi).contains(&id), "t{t} head{head} id {id} escaped");
            }
            assert!(row.iter().all(|&id| (id as u64) < EXPECTED_PADDED));
        }
    }

    #[test]
    fn a_token_after_eos_hashes_as_if_it_started_the_sequence() {
        let h = hash();
        // Token 5 opening a fresh context vs. token 5 right after an EOS: both
        // have an all-EOS window, so both must hash identically. That is the
        // whole point of the segment-aware shift.
        let fresh = h.row_ids(&NGramContext::new(&h), &[5]).expect("fresh");
        let after_eos = h
            .row_ids(&NGramContext::new(&h), &[7, 11, EOS, 5])
            .expect("seq");
        assert_eq!(fresh, &after_eos[3 * 16..]);
        // ...and it must NOT match a token 5 that has real left context, or the
        // assertion above would be vacuous.
        let with_context = h.row_ids(&NGramContext::new(&h), &[7, 11, 5]).expect("seq");
        assert_ne!(fresh, &with_context[2 * 16..]);
    }

    #[test]
    fn context_rolls_so_chunked_prefill_matches_one_shot() {
        let h = hash();
        let tokens: Vec<i64> = vec![7, 11, 13, EOS, 17, 19, 23];
        let one_shot = h
            .row_ids(&NGramContext::new(&h), &tokens)
            .expect("one shot");

        let mut ctx = NGramContext::new(&h);
        let mut chunked = Vec::new();
        for chunk in tokens.chunks(2) {
            chunked.extend(h.row_ids(&ctx, chunk).expect("chunk"));
            ctx.push(chunk);
        }
        assert_eq!(chunked, one_shot);
        assert_eq!(ctx.tokens(), [19, 23]);

        // A first chunk shorter than the context keeps the EOS left-pad the
        // reference applies explicitly on that branch.
        let mut short = NGramContext::new(&h);
        short.push(&[7]);
        assert_eq!(short.tokens(), [EOS, 7]);
    }

    #[test]
    fn out_of_range_tokens_are_rejected_not_wrapped() {
        let h = hash();
        assert!(h.row_ids(&NGramContext::new(&h), &[248_320]).is_err());
        assert!(h.row_ids(&NGramContext::new(&h), &[-1]).is_err());
        assert!(h.row_ids(&NGramContext::new(&h), &[248_319]).is_ok());
    }

    // ---------------- PLE forward ----------------

    /// A deliberately tiny PLE: hidden 2, 2 streams, embed 3, so every number
    /// below can be worked out by hand. Kernel/dilation stay at the real 4/3.
    fn tiny_cfg() -> PleConfig {
        PleConfig {
            hidden_size: 2,
            hc_count: 2,
            ple_embed_dim: 3,
            conv_kernel_size: 4,
            conv_dilation: 3,
            rms_norm_eps: 1e-6,
        }
    }

    fn tiny_weights(cfg: &PleConfig) -> PleWeights {
        let hc_hidden = cfg.hc_hidden();
        PleWeights {
            key_proj: vec![0.0; hc_hidden * cfg.ple_embed_dim],
            value_proj: vec![0.0; cfg.hidden_size * cfg.ple_embed_dim],
            norm_key: vec![0.0; hc_hidden],
            norm_query: vec![0.0; hc_hidden],
            norm_conv: vec![0.0; hc_hidden],
            conv1d: vec![0.0; hc_hidden * cfg.conv_kernel_size],
        }
    }

    fn assert_close(got: f32, want: f32, what: &str) {
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{what}: got {got}, want {want}"
        );
    }

    #[test]
    fn rms_norm_groups_are_independent_and_scale_is_one_plus_weight() {
        // Two groups of 2 with wildly different magnitudes. If the statistic
        // were taken over the flat row, the small group would be crushed.
        let x = [3.0f32, 4.0, 0.03, 0.04];
        let w = [0.0f32, 0.0, 1.0, -0.5];
        let out = rms_norm_grouped(&x, &w, 2, 0.0);
        // RMS of (3,4) is sqrt(12.5); of (0.03,0.04) is sqrt(0.00125).
        assert_close(out[0], 3.0 / 12.5f32.sqrt(), "g0[0]");
        assert_close(out[1], 4.0 / 12.5f32.sqrt(), "g0[1]");
        assert_close(out[2], (0.03 / 0.00125f32.sqrt()) * 2.0, "g1[0] * (1+1)");
        assert_close(out[3], (0.04 / 0.00125f32.sqrt()) * 0.5, "g1[1] * (1-0.5)");
        // Both groups reach the same normalised shape before the weight, which
        // is only true if the statistic is per group.
        let bare = rms_norm_grouped(&x, &[0.0; 4], 2, 0.0);
        assert_close(bare[0], bare[2], "group shapes match");
    }

    #[test]
    fn zero_gate_stays_zero_rather_than_becoming_sqrt_epsilon() {
        assert_eq!(signed_sqrt_gate(0.0), 0.0);
        assert_eq!(signed_sqrt_gate(-0.0), 0.0);
        // f32::signum would have produced these instead:
        assert_ne!(signed_sqrt_gate(0.0), 1e-6f32.sqrt());
        assert_ne!(signed_sqrt_gate(-0.0), -(1e-6f32.sqrt()));
        // Non-zero gates keep their sign and get the clamp floor.
        assert_close(signed_sqrt_gate(4.0), 2.0, "sqrt(4)");
        assert_close(signed_sqrt_gate(-4.0), -2.0, "-sqrt(4)");
        assert_close(signed_sqrt_gate(1e-12), 1e-3, "clamped to sqrt(1e-6)");
        assert_close(signed_sqrt_gate(-1e-12), -1e-3, "clamped, negative");
        assert!(signed_sqrt_gate(f32::NAN).is_nan());
    }

    /// With `key_proj = 0` the key is all zeros, so every gate is exactly zero
    /// and `sigmoid(0) = 0.5`. If `sign(0)` were 1 the gate would be 1e-3 and
    /// the factor 0.50025 — a 5e-4 relative shift this test can see.
    #[test]
    fn a_zero_key_gates_the_value_by_exactly_one_half() {
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&cfg);
        // value_proj picks embedding dim 0 into channel 0, dim 1 into channel 1.
        w.value_proj = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        let out = layer
            .forward(&[6.0, 8.0, 0.0], &[1.0, 2.0, 3.0, 4.0], &mut state, None)
            .expect("forward");
        // conv weights are zero -> silu(0) = 0, so the output is the residual.
        for (stream, chunk) in out.chunks_exact(2).enumerate() {
            assert_close(chunk[0], 3.0, &format!("stream {stream} ch0"));
            assert_close(chunk[1], 4.0, &format!("stream {stream} ch1"));
        }
        assert!((out[0] - 3.0015).abs() > 1e-4, "sign(0) leaked in as 1");
    }

    /// The bilinear gate is divided by `sqrt(hidden_size)`. Every other forward
    /// test drives the gate to zero, where the divisor is invisible; this one
    /// makes it the whole answer.
    #[test]
    fn the_gate_is_divided_by_sqrt_hidden_size() {
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&cfg);
        // key = 0.1 * emb[0] = 1 on every channel, so key_normed is all ones
        // (RMS 1). hidden_states (1,1,2,2) normalises to all ones as well, so
        // each stream's dot product is exactly 2 = hidden_size.
        for row in w.key_proj.chunks_exact_mut(cfg.ple_embed_dim) {
            row[0] = 0.1;
        }
        w.value_proj = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        let out = layer
            .forward(&[10.0, 20.0, 0.0], &[1.0, 1.0, 2.0, 2.0], &mut state, None)
            .expect("forward");

        let scaled = 2.0f32 / (cfg.hidden_size as f32).sqrt();
        let weight = 1.0 / (1.0 + (-scaled.sqrt()).exp());
        assert_close(weight, 0.766_599_2, "sigmoid of the scaled gate");
        for (stream, chunk) in out.chunks_exact(cfg.hidden_size).enumerate() {
            assert_close(chunk[0], weight * 10.0, &format!("stream {stream} ch0"));
            assert_close(chunk[1], weight * 20.0, &format!("stream {stream} ch1"));
        }
        // Without the divisor the factor would be 0.80443 — 5% away, far
        // outside the tolerance above.
        let unscaled = 1.0 / (1.0 + (-2.0f32.sqrt()).exp());
        assert!((out[0] - unscaled * 10.0).abs() > 0.3);
    }

    /// Both `norm_key` and `norm_query` take their statistic per `hidden_size`
    /// group, not over the flat `hc_count * hidden_size` row. Give stream 1 a
    /// 10x key and a 5x query and grouped normalisation cancels both, so the
    /// two streams must gate identically. A whole-row norm would not cancel:
    /// the streams would come out at 5.585 and 8.404 instead of both at 7.666.
    #[test]
    fn key_and_query_norms_group_per_stream() {
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&cfg);
        // key = (1, 1, 10, 10) from emb[0] = 10.
        for (o, row) in w.key_proj.chunks_exact_mut(cfg.ple_embed_dim).enumerate() {
            row[0] = if o < cfg.hidden_size { 0.1 } else { 1.0 };
        }
        w.value_proj = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        let out = layer
            .forward(&[10.0, 20.0, 0.0], &[1.0, 1.0, 5.0, 5.0], &mut state, None)
            .expect("forward");

        // Every group normalises to all-ones, so each stream's dot is
        // hidden_size = 2 and the gate is the same one as the test above.
        let weight = 1.0 / (1.0 + (-(2.0f32 / 2.0f32.sqrt()).sqrt()).exp());
        for (stream, chunk) in out.chunks_exact(cfg.hidden_size).enumerate() {
            assert!(
                (chunk[0] - weight * 10.0).abs() < 1e-4,
                "stream {stream}: got {}, want {}",
                chunk[0],
                weight * 10.0
            );
        }
        // A whole-row norm would leave the 10x/5x uncancelled and the two
        // streams would disagree outright (5.585 vs 8.404).
        assert!(
            (out[0] - out[2]).abs() < 1e-4,
            "streams gated differently: {} vs {}",
            out[0],
            out[2]
        );
    }

    /// The residual branch adds the UN-normed gated value. Zeroing the conv
    /// weights isolates it, and a non-zero `norm_conv` weight puts the normed
    /// tensor orders of magnitude away, so confusing the two cannot pass.
    #[test]
    fn residual_branch_takes_the_unnormed_gated_value() {
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&cfg);
        w.value_proj = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // norm_conv scale of +2 everywhere -> the normed tensor is 3x a unit
        // RMS shape, nowhere near the raw magnitudes below.
        w.norm_conv = vec![2.0; cfg.hc_hidden()];
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        let out = layer
            .forward(
                &[600.0, 800.0, 0.0],
                &[1.0, 1.0, 1.0, 1.0],
                &mut state,
                None,
            )
            .expect("forward");
        // Un-normed: 0.5 * (600, 800) = (300, 400).
        assert_close(out[0], 300.0, "residual ch0");
        assert_close(out[1], 400.0, "residual ch1");
        // The normed-and-scaled value would have been ~(1.8, 2.4): a wrong
        // branch here is orders of magnitude off, not a rounding blur.
        let normed = rms_norm_grouped(
            &[300.0, 400.0, 300.0, 400.0],
            &[2.0; 4],
            cfg.hidden_size,
            cfg.rms_norm_eps,
        );
        assert!((out[0] - normed[0]).abs() > 100.0);
    }

    /// The conv branch consumes the NORMED gated value, step for step. Two
    /// steps whose values differ in shape (not just scale — RMSNorm is
    /// scale-invariant, so a scale difference would hide a row swap) with the
    /// conv reading only the current step: each output must be its own
    /// `gated + silu(norm_conv(gated))`.
    #[test]
    fn conv_branch_consumes_the_normed_value_step_for_step() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let mut w = tiny_weights(&cfg);
        w.value_proj = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        w.norm_conv = vec![2.0; hc_hidden];
        for taps in w.conv1d.chunks_exact_mut(cfg.conv_kernel_size) {
            taps[cfg.conv_kernel_size - 1] = 1.0; // tap at t, nothing historical
        }
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        // key_proj is zero, so every gate is 0.5: value (10,20) then (20,10).
        let out = layer
            .forward(
                &[10.0, 20.0, 0.0, 20.0, 10.0, 0.0],
                &vec![1.0f32; 2 * hc_hidden],
                &mut state,
                None,
            )
            .expect("forward");

        for (t, value) in [[10.0f32, 20.0], [20.0, 10.0]].into_iter().enumerate() {
            let gated: Vec<f32> = (0..hc_hidden).map(|i| 0.5 * value[i % 2]).collect();
            let normed = rms_norm_grouped(
                &gated,
                &vec![2.0f32; hc_hidden],
                cfg.hidden_size,
                cfg.rms_norm_eps,
            );
            for c in 0..hc_hidden {
                let want = gated[c] + silu(normed[c]);
                assert_close(out[t * hc_hidden + c], want, &format!("t{t} c{c}"));
            }
        }
    }

    /// A one-hot conv tap must read exactly `t - (state_len - k*dilation)`:
    /// k=0 -> t-9, k=1 -> t-6, k=2 -> t-3, k=3 -> t.
    #[test]
    fn conv_taps_land_at_minus_nine_six_three_and_zero() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let state_len = cfg.short_conv_state_len();
        assert_eq!(state_len, 9);

        for k in 0..cfg.conv_kernel_size {
            let mut w = tiny_weights(&cfg);
            for taps in w.conv1d.chunks_exact_mut(cfg.conv_kernel_size) {
                taps[k] = 1.0;
            }
            let layer = PleLayer::new(cfg.clone(), w).expect("layer");
            let mut state = PleConvState::zeros(&cfg);
            // A ramp: step t carries the value t+1 on channel 0, so the tap's
            // lag is readable straight off the output.
            let seq = 12;
            let mut x = vec![0.0f32; seq * hc_hidden];
            for (t, row) in x.chunks_exact_mut(hc_hidden).enumerate() {
                row[0] = (t + 1) as f32;
            }
            let out = layer.short_conv(&x, &mut state);
            let lag = state_len - k * cfg.conv_dilation;
            for (t, row) in out.chunks_exact(hc_hidden).enumerate() {
                let source = if t >= lag { (t - lag + 1) as f32 } else { 0.0 };
                assert_close(row[0], silu(source), &format!("k={k} lag={lag} t={t}"));
            }
        }
    }

    #[test]
    fn conv_state_makes_chunked_and_one_shot_agree() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let mut w = tiny_weights(&cfg);
        // Distinct taps per channel so an off-by-one in the state roll shows up.
        for (c, taps) in w.conv1d.chunks_exact_mut(cfg.conv_kernel_size).enumerate() {
            for (k, tap) in taps.iter_mut().enumerate() {
                *tap = 0.25 * (k as f32 + 1.0) - 0.1 * c as f32;
            }
        }
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");

        let seq = 14;
        let mut x = vec![0.0f32; seq * hc_hidden];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.5;
        }

        let mut one = PleConvState::zeros(&cfg);
        let full = layer.short_conv(&x, &mut one);

        let mut split = PleConvState::zeros(&cfg);
        let mut chunked = Vec::new();
        for chunk in x.chunks(4 * hc_hidden) {
            chunked.extend(layer.short_conv(chunk, &mut split));
        }
        assert_eq!(chunked.len(), full.len());
        for (i, (&a, &b)) in chunked.iter().zip(&full).enumerate() {
            assert_close(a, b, &format!("chunked step {i}"));
        }
        assert_eq!(split.rows(), one.rows());
        assert_eq!(one.rows().len(), cfg.short_conv_state_len() * hc_hidden);
    }

    /// A masked step must contribute nothing — not to its own output, and not
    /// to any later step through the conv state.
    #[test]
    fn conv_mask_zeroes_both_branches() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let mut w = tiny_weights(&cfg);
        w.value_proj = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        w.norm_conv = vec![2.0; hc_hidden];
        // Tap k=3 is the current step, so the masked step's own conv output
        // depends only on the masked value.
        for taps in w.conv1d.chunks_exact_mut(cfg.conv_kernel_size) {
            taps[3] = 1.0;
        }
        let layer = PleLayer::new(cfg.clone(), w).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        let emb = [600.0f32, 800.0, 0.0, 600.0, 800.0, 0.0];
        let hidden = vec![1.0f32; 2 * hc_hidden];
        let out = layer
            .forward(&emb, &hidden, &mut state, Some(&[true, false]))
            .expect("forward");
        for v in &out[hc_hidden..] {
            assert_eq!(*v, 0.0, "masked step must contribute nothing");
        }
        // Step 0 survives, so the mask is not just zeroing everything.
        assert!(out[0].abs() > 1.0);
        // The masked activation must not have entered the state either.
        let last_row = (cfg.short_conv_state_len() - 1) * hc_hidden;
        assert!(state.rows()[last_row].abs() < 1e-12);
    }

    #[test]
    fn forward_rejects_shape_mismatches() {
        let cfg = tiny_cfg();
        let layer = PleLayer::new(cfg.clone(), tiny_weights(&cfg)).expect("layer");
        let mut state = PleConvState::zeros(&cfg);
        // embeddings not a whole number of ple_embed_dim rows
        assert!(
            layer
                .forward(&[1.0, 2.0], &[0.0; 4], &mut state, None)
                .is_err()
        );
        // hidden_states the wrong width
        assert!(
            layer
                .forward(&[1.0, 2.0, 3.0], &[0.0; 3], &mut state, None)
                .is_err()
        );
        // mask longer than the sequence
        assert!(
            layer
                .forward(&[1.0, 2.0, 3.0], &[0.0; 4], &mut state, Some(&[true, true]))
                .is_err()
        );
        let mut bad = tiny_weights(&cfg);
        bad.norm_key.pop();
        assert!(PleLayer::new(cfg, bad).is_err());
    }
}

#[cfg(test)]
mod on_box_tests {
    use super::*;
    use infer_gguf::safetensors::SafeTensorsDir;

    const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
    const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";
    const PLE_PREFIX: &str = "model.language_model.layers.1.ple.";

    /// Not `#[ignore]`d: a hash whose only cross-check never runs is unchecked.
    /// Off-box the directory is absent and the test returns immediately.
    fn open() -> Option<SafeTensorsDir> {
        let dir = std::env::var(CKPT_ENV).unwrap_or_else(|_| CKPT_DEFAULT.into());
        let path = std::path::Path::new(&dir);
        if !path.is_dir() {
            eprintln!("skip: {} not present (set {CKPT_ENV})", path.display());
            return None;
        }
        Some(SafeTensorsDir::open_dir(path).expect("open checkpoint"))
    }

    fn read_i64(st: &SafeTensorsDir, name: &str) -> Vec<i64> {
        let bytes = st
            .tensor_data(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(bytes.len() % 8, 0, "{name} is not whole i64s");
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()
    }

    /// The checkpoint ships the hash tables as ordinary tensors, which makes
    /// this a real oracle rather than a restatement: if the primes, the running
    /// offsets or the splitmix64 multipliers derived here disagree with the
    /// ones the exporter baked in, every lookup is wrong and this fails.
    #[test]
    fn real_checkpoint_pins_the_hash_tables() {
        let Some(st) = open() else { return };
        let h = NGramHash::new(NGramHashConfig::qwen4_exp()).expect("hash");

        let multipliers = read_i64(&st, &format!("{PLE_PREFIX}ple_embedding.layer_multipliers"));
        assert_eq!(multipliers, h.layer_multipliers(), "layer_multipliers");

        let sizes = read_i64(
            &st,
            &format!("{PLE_PREFIX}ple_embedding.ngram_heads_vocab_sizes"),
        );
        assert_eq!(sizes, h.head_vocab_sizes(), "ngram_heads_vocab_sizes");

        let offsets = read_i64(
            &st,
            &format!("{PLE_PREFIX}ple_embedding.ngram_heads_offsets"),
        );
        assert_eq!(offsets, h.head_offsets(), "ngram_heads_offsets");
    }

    /// The derived padded vocab has to equal the table that is actually on
    /// disk. If it does not, the finding is the mismatch — do not paper over it
    /// by padding differently.
    #[test]
    fn real_checkpoint_row_count_matches_the_derived_vocab() {
        let Some(st) = open() else { return };
        let h = NGramHash::new(NGramHashConfig::qwen4_exp()).expect("hash");

        let prefix = format!("{PLE_PREFIX}ple_embedding.ngram_embedding.shard_");
        let mut shards = 0u64;
        let mut rows = 0u64;
        for t in st.tensors() {
            let Some(rest) = t.name.strip_prefix(&prefix) else {
                continue;
            };
            assert!(
                rest.ends_with(".weight"),
                "unexpected shard tensor {}",
                t.name
            );
            // `dims` is the safetensors shape REVERSED, so dims[0] is the row
            // width and dims[1] the row count.
            assert_eq!(t.dims.len(), 2, "{} rank", t.name);
            assert_eq!(t.dims[0], h.head_dim() as u64, "{} row width", t.name);
            assert_eq!(t.dtype, "F8_E4M3", "{} dtype", t.name);
            shards += 1;
            rows += t.dims[1];
        }
        assert_eq!(shards, 128, "n-gram table shard count");
        assert_eq!(
            rows,
            h.padded_vocab_size(),
            "checkpoint has {rows} n-gram rows but the derived padded vocab is {}",
            h.padded_vocab_size()
        );
        // One byte per FP8 element: the table really is 47.68 GiB.
        assert_eq!(rows * h.head_dim() as u64, 51_200_245_760);
    }

    /// The rest of the PLE block, so a drift in `PleConfig::qwen4_exp` cannot
    /// pass unnoticed.
    #[test]
    fn real_checkpoint_pins_the_ple_weight_shapes() {
        let Some(st) = open() else { return };
        let cfg = PleConfig::qwen4_exp();
        let hc_hidden = cfg.hc_hidden() as u64;
        let hidden = cfg.hidden_size as u64;
        let embed = cfg.ple_embed_dim as u64;

        // `dims` is reversed from the header, so a Linear reads [in, out].
        for (suffix, want) in [
            ("key_proj.weight", vec![embed, hc_hidden]),
            ("value_proj.weight", vec![embed, hidden]),
            ("norm_key.weight", vec![hc_hidden]),
            ("norm_query.weight", vec![hc_hidden]),
            ("norm_conv.weight", vec![hc_hidden]),
            // conv1d header shape is [hc_hidden, 1, kernel]; reversed that is
            // [kernel, 1, hc_hidden]. Depthwise, hence the 1.
            (
                "conv1d.weight",
                vec![cfg.conv_kernel_size as u64, 1, hc_hidden],
            ),
        ] {
            let name = format!("{PLE_PREFIX}{suffix}");
            let t = st.tensor(&name).unwrap_or_else(|| panic!("{name} missing"));
            assert_eq!(t.dims, want, "{name} dims");
        }
    }
}
