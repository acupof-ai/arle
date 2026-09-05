//! Host-f32 reference for `qwen4_exp` (Qwen3.8-Flash-Next) **hyper-connections**
//! — the oracle a device kernel gets checked against.
//!
//! This is the architecture's structural novelty, so it is worth stating what
//! actually changes relative to every other model ARLE runs. The inter-layer
//! residual is not `hidden_size` wide. It is `hc_count * hidden_size` =
//! `4 * 2560` = 10240, **stream-major** (stream `s` occupies the contiguous
//! range `[s * 2560, (s + 1) * 2560)`), seeded by tiling the token embedding
//! `hc_count` times (`hidden_states.repeat(1, 1, hc_count)` in the reference's
//! `Qwen4ExpTextModel.forward`). Every sublayer is wrapped by a
//! `Qwen4ExpTextGatedResidual` that collapses those 4 streams down to one
//! `hidden_size` block input, and scatters the sublayer's output back across
//! the 4 streams with a per-stream gate:
//!
//! ```text
//!   hn  = grouped_rmsnorm(h, group = hidden_size)   # 4 independent norms
//!   u   = silu(W_down @ hn / hc_count)              # divide BEFORE silu
//!   m   = sigmoid(W_up  @ u)
//!   x   = mean_s( m[s] * hn[s] )                    # [hidden] -> the block input
//!   inj = 2 * sigmoid(W_inject @ hn / hc_count)     # [hc_count], in (0, 2)
//!   ... y = sublayer(x) ...
//!   h[s] += inj[s] * y                              # for each stream s
//! ```
//!
//! Transcribed line for line from `Qwen4ExpTextGatedResidual.forward` and the
//! `Qwen4ExpTextDecoderLayer.forward` that calls it in the reference
//! implementation (`modeling_qwen4_exp.py`), in the same style as the host
//! gated-delta references in `crates/vulkan-kernels/tests/`.
//!
//! # The three details that fail silently
//!
//! 1. **`/ hc_count` is inside, before the nonlinearity.** `silu(z / 4)` and
//!    `silu(z) / 4` are both plausible-looking and both finite. Measured on the
//!    real layer-0 weights with the real seeded state, the raw inject logit is
//!    `W_inject @ hn = -16.6`: dividing first gives `2*sigmoid(-4.16) = 0.031`,
//!    dividing after gives `2*sigmoid(-16.6) = 1.2e-7`. Five orders of
//!    magnitude, and neither output looks wrong on inspection. The tests below
//!    pin the ordering on both the mix path and the inject path.
//! 2. **`hc_norm` is a GROUPED RMSNorm.** Four independent norms of width 2560
//!    sharing one 10240-long weight vector — not one norm over 10240. On the
//!    *seeded* state the two agree exactly (all four streams are copies, so the
//!    per-group and global mean-of-squares coincide), which is precisely why
//!    the bug survives a first-token smoke test. It only diverges once the
//!    streams differ, i.e. from the first `inj[s] * y` onward.
//! 3. **The gain is `1 + weight`, not `weight`.** `Qwen4ExpTextRMSNorm` holds a
//!    zero-initialised parameter and applies `output * (1.0 + self.weight)`.
//!    Measured on the checkpoint: the 96 per-layer `hc_norm` weights have mean
//!    ~ -0.06 to -0.11, i.e. they sit around zero, not around one.
//!
//! # The mixer, and the missing `model.norm`
//!
//! `Qwen4ExpTextModel` builds one extra `Qwen4ExpTextGatedResidual` with
//! `use_combine=False`: it has no `block_inject_weight`, returns only the
//! `[hidden]` mixed input, and is the terminal op before `lm_head`. Here that
//! is [`GatedResidualWeights::block_inject`] being `None`.
//!
//! **There is no separate `model.norm` in this checkpoint** — verified, not
//! assumed. `model.safetensors.index.json` (296,475 entries) contains no
//! `model.language_model.norm.weight`; the only text-stream `*norm.weight`
//! outside `layers.<n>.` is `model.language_model.hyper_connection_mixer.
//! hc_norm.weight`, `[10240]` BF16. The mixer's own `hc_norm` *is* the final
//! norm, and its trained weight looks the part: mean +2.75 (range -0.37 ..
//! +12.4) against ~ -0.06 for the per-layer norms, i.e. a gain of ~3.75x rather
//! than ~0.94x. `real_checkpoint_has_no_separate_final_norm` re-checks this
//! against the on-box files every run.
//!
//! # Numerics
//!
//! Public signatures are `f32` in / `f32` out — the device buffers are f32 —
//! but every reduction (`sum(x^2)`, every dot product, the mean over streams)
//! accumulates in `f64`. A reference exists to make a device mismatch
//! unambiguous, and a sequential `f32` sum over 10240 terms has an error of its
//! own that is the same order as the shader's tree reduction, which would leave
//! a disagreement unattributable. Stage boundaries (`normed`, `lowrank`,
//! `mix_gate`, `block_input`, `injection_weights`) round to `f32`, matching
//! what a device kernel would hand to the next dispatch.
//!
//! One token per call. Hyper-connections are per-position and share nothing
//! across a sequence, so a batched form would be a loop and nothing else.

use anyhow::{Result, ensure};

/// Shape constants of a `Qwen4ExpTextGatedResidual`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperConnectionConfig {
    /// `config.hidden_size` — the width of one stream, and of the block input.
    pub hidden_size: usize,
    /// `config.hc_count` — how many residual streams run in parallel.
    pub hc_count: usize,
    /// `config.hc_lowrank` — the bottleneck the stream gate is computed through.
    pub hc_lowrank: usize,
    /// `config.rms_norm_eps`, inside the `rsqrt` of `hc_norm`.
    pub rms_norm_eps: f32,
}

impl HyperConnectionConfig {
    /// The on-box `qwen3.8-flash-next-nvfp4` checkpoint's values, read from its
    /// `config.json`.
    pub const QWEN4_EXP: Self = Self {
        hidden_size: 2560,
        hc_count: 4,
        hc_lowrank: 320,
        rms_norm_eps: 1e-6,
    };

    /// Width of the hyper-connection residual: `hc_count * hidden_size`.
    #[must_use]
    pub const fn hc_hidden(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.hidden_size > 0, "hidden_size must be non-zero");
        ensure!(self.hc_count > 0, "hc_count must be non-zero");
        ensure!(self.hc_lowrank > 0, "hc_lowrank must be non-zero");
        Ok(())
    }
}

/// The weights of one `Qwen4ExpTextGatedResidual`, in HF row-major order.
///
/// Row-major means `w[out * in_features + in]`, which is exactly the byte order
/// `SafeTensorsDir::tensor_data` hands back for an `nn.Linear` whose header
/// `shape` is `[out_features, in_features]`. (Note that
/// `SafeTensorInfo::dims` reverses that to GGUF innermost-first order, so the
/// same tensor reads as `dims == [in_features, out_features]`.)
#[derive(Debug, Clone, PartialEq)]
pub struct GatedResidualWeights {
    /// `hc_norm.weight`, `[hc_hidden]`. The applied gain is `1 + hc_norm`.
    pub hc_norm: Vec<f32>,
    /// `input_mix_weight_down.weight`, `[hc_lowrank, hc_hidden]`.
    pub mix_down: Vec<f32>,
    /// `input_mix_weight_up.weight`, `[hc_hidden, hc_lowrank]`.
    pub mix_up: Vec<f32>,
    /// `block_inject_weight.weight`, `[hc_count, hc_hidden]`.
    ///
    /// `None` is the `use_combine=False` module — the stream-level
    /// `hyper_connection_mixer`, which produces no injection weights because
    /// nothing is injected back after it.
    pub block_inject: Option<Vec<f32>>,
}

impl GatedResidualWeights {
    /// Fail loud on any weight whose length is not what `cfg` implies.
    pub fn validate(&self, cfg: &HyperConnectionConfig) -> Result<()> {
        cfg.validate()?;
        let hc_hidden = cfg.hc_hidden();
        ensure!(
            self.hc_norm.len() == hc_hidden,
            "hc_norm is {} long, expected hc_count * hidden_size = {hc_hidden}",
            self.hc_norm.len()
        );
        ensure!(
            self.mix_down.len() == cfg.hc_lowrank * hc_hidden,
            "mix_down is {} long, expected hc_lowrank * hc_hidden = {}",
            self.mix_down.len(),
            cfg.hc_lowrank * hc_hidden
        );
        ensure!(
            self.mix_up.len() == hc_hidden * cfg.hc_lowrank,
            "mix_up is {} long, expected hc_hidden * hc_lowrank = {}",
            self.mix_up.len(),
            hc_hidden * cfg.hc_lowrank
        );
        if let Some(inject) = &self.block_inject {
            ensure!(
                inject.len() == cfg.hc_count * hc_hidden,
                "block_inject is {} long, expected hc_count * hc_hidden = {}",
                inject.len(),
                cfg.hc_count * hc_hidden
            );
        }
        Ok(())
    }

    /// True for the `use_combine=False` module (`hyper_connection_mixer`).
    #[must_use]
    pub const fn is_mixer(&self) -> bool {
        self.block_inject.is_none()
    }
}

/// Everything one `Qwen4ExpTextGatedResidual` produces.
///
/// The intermediates are returned, not discarded, because they are what a
/// device port is checked stage by stage against: a kernel that gets
/// `block_input` right by luck while `normed` is wrong is a kernel that will
/// break on the next layer.
#[derive(Debug, Clone, PartialEq)]
pub struct GatedResidual {
    /// `hyper_input_normed`, `[hc_hidden]` — grouped RMSNorm of the residual.
    pub normed: Vec<f32>,
    /// `silu(W_down @ normed / hc_count)`, `[hc_lowrank]`.
    pub lowrank: Vec<f32>,
    /// `sigmoid(W_up @ lowrank)`, `[hc_hidden]` — the per-stream, per-channel
    /// mixing gate.
    pub mix_gate: Vec<f32>,
    /// `mean_s(mix_gate[s] * normed[s])`, `[hidden_size]` — what the attention
    /// or MoE sublayer actually consumes.
    pub block_input: Vec<f32>,
    /// `2 * sigmoid(W_inject @ normed / hc_count)`, `[hc_count]`, every element
    /// strictly in `(0, 2)`. `None` for the mixer.
    pub injection_weights: Option<Vec<f32>>,
}

/// Logistic sigmoid. Rounds once, at the end.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    sigmoid64(f64::from(x)) as f32
}

/// SiLU / swish, `x * sigmoid(x)`. Rounds once, at the end.
#[must_use]
pub fn silu(x: f32) -> f32 {
    silu64(f64::from(x)) as f32
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu64(x: f64) -> f64 {
    x * sigmoid64(x)
}

/// `Qwen4ExpTextRMSNorm` with `group_size` set: `x.reshape(-1, group_size)` is
/// normalised row by row, the result is flattened, and the *whole* flattened
/// vector is then scaled by `1 + weight`.
///
/// So `weight` is `x.len()` long (one gain per channel per stream), while the
/// statistic is per group. With `group_size == x.len()` this degenerates to an
/// ordinary RMSNorm, which is what makes the grouped-vs-ungrouped bug so quiet:
/// it only shows up once the groups carry different energy.
pub fn grouped_rmsnorm(x: &[f32], weight: &[f32], group_size: usize, eps: f32) -> Result<Vec<f32>> {
    ensure!(group_size > 0, "group_size must be non-zero");
    ensure!(
        x.len() == weight.len(),
        "grouped_rmsnorm: {} values against {} gains",
        x.len(),
        weight.len()
    );
    ensure!(
        x.len().is_multiple_of(group_size),
        "grouped_rmsnorm: {} values is not a multiple of group_size {group_size}",
        x.len()
    );

    let eps = f64::from(eps);
    let mut out = Vec::with_capacity(x.len());
    for (group, gains) in x
        .chunks_exact(group_size)
        .zip(weight.chunks_exact(group_size))
    {
        let sum_sq: f64 = group.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        // `torch.rsqrt(x.pow(2).mean(-1) + eps)` — the mean is over the group,
        // not over the flattened vector.
        let scale = (sum_sq / group_size as f64 + eps).sqrt().recip();
        out.extend(
            group
                .iter()
                .zip(gains)
                // One rounding: the norm and the `1 + weight` gain are a single
                // f64 expression, as they are a single f32 expression in torch.
                .map(|(&v, &w)| (f64::from(v) * scale * (1.0 + f64::from(w))) as f32),
        );
    }
    Ok(out)
}

/// `y = W @ x` for a row-major `[out_features, in_features]` weight, f64
/// accumulation, f64 out so the caller can apply `/ hc_count` and the
/// nonlinearity before rounding.
fn matvec(w: &[f32], in_features: usize, x: &[f32]) -> Vec<f64> {
    debug_assert_eq!(x.len(), in_features);
    w.chunks_exact(in_features)
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        })
        .collect()
}

/// One `Qwen4ExpTextGatedResidual.forward`, for a single token.
///
/// `hyper_state` is the `[hc_count * hidden_size]` inter-layer residual. It is
/// read, never written: the reference returns `hyper_input` unchanged and the
/// decoder layer adds to it afterwards, which is what
/// [`inject_block_output`] does.
pub fn gated_residual(
    cfg: &HyperConnectionConfig,
    weights: &GatedResidualWeights,
    hyper_state: &[f32],
) -> Result<GatedResidual> {
    weights.validate(cfg)?;
    let hc_hidden = cfg.hc_hidden();
    ensure!(
        hyper_state.len() == hc_hidden,
        "expected {hc_hidden} hyper-connection features, got {}",
        hyper_state.len()
    );

    let normed = grouped_rmsnorm(
        hyper_state,
        &weights.hc_norm,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    )?;

    let hc_count = cfg.hc_count as f64;
    // `F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)`.
    // The divide is inside the activation; moving it outside changes the answer,
    // not just its scale.
    let lowrank: Vec<f32> = matvec(&weights.mix_down, hc_hidden, &normed)
        .into_iter()
        .map(|d| silu64(d / hc_count) as f32)
        .collect();
    let mix_gate: Vec<f32> = matvec(&weights.mix_up, cfg.hc_lowrank, &lowrank)
        .into_iter()
        .map(|z| sigmoid64(z) as f32)
        .collect();

    // `(gate.unflatten(-1, (hc_count, hidden)) * normed.unflatten(...)).mean(-2)`
    // — stream-major, so stream `s` is the contiguous slice at `s * hidden`.
    let mut block_input = Vec::with_capacity(cfg.hidden_size);
    for i in 0..cfg.hidden_size {
        let acc: f64 = (0..cfg.hc_count)
            .map(|s| {
                let k = s * cfg.hidden_size + i;
                f64::from(mix_gate[k]) * f64::from(normed[k])
            })
            .sum();
        block_input.push((acc / hc_count) as f32);
    }

    // `2 * torch.sigmoid(self.block_inject_weight(normed) / self.hc_count)`.
    let injection_weights = weights.block_inject.as_ref().map(|inject| {
        matvec(inject, hc_hidden, &normed)
            .into_iter()
            .map(|z| (2.0 * sigmoid64(z / hc_count)) as f32)
            .collect()
    });

    Ok(GatedResidual {
        normed,
        lowrank,
        mix_gate,
        block_input,
        injection_weights,
    })
}

/// The decoder layer's half of the residual:
/// `hidden_states = hyper_input + (block_output.unsqueeze(-2) *
/// injection_weights.unsqueeze(-1)).flatten(-2)`.
///
/// Adds `injection_weights[s] * block_output` into stream `s` of `hyper_state`,
/// in place. Note that this accumulates onto the *raw* residual, not onto
/// `GatedResidual::normed` — the norm is consumed by the gate and thrown away.
pub fn inject_block_output(
    cfg: &HyperConnectionConfig,
    hyper_state: &mut [f32],
    injection_weights: &[f32],
    block_output: &[f32],
) -> Result<()> {
    cfg.validate()?;
    ensure!(
        hyper_state.len() == cfg.hc_hidden(),
        "hyper_state is {} long, expected {}",
        hyper_state.len(),
        cfg.hc_hidden()
    );
    ensure!(
        injection_weights.len() == cfg.hc_count,
        "expected {} injection weights, got {}",
        cfg.hc_count,
        injection_weights.len()
    );
    ensure!(
        block_output.len() == cfg.hidden_size,
        "block_output is {} long, expected {}",
        block_output.len(),
        cfg.hidden_size
    );

    for (stream, &inj) in injection_weights.iter().enumerate() {
        let inj = f64::from(inj);
        let base = stream * cfg.hidden_size;
        for (h, &y) in hyper_state[base..base + cfg.hidden_size]
            .iter_mut()
            .zip(block_output)
        {
            *h = (f64::from(*h) + inj * f64::from(y)) as f32;
        }
    }
    Ok(())
}

/// Seed the residual from a token embedding: `hidden_states.repeat(1, 1,
/// hc_count)`.
///
/// All `hc_count` streams start as byte-identical copies. That is why the first
/// layer cannot distinguish a grouped `hc_norm` from an ungrouped one — see the
/// module docs.
pub fn seed_hyper_state(cfg: &HyperConnectionConfig, embedding: &[f32]) -> Result<Vec<f32>> {
    cfg.validate()?;
    ensure!(
        embedding.len() == cfg.hidden_size,
        "embedding is {} long, expected hidden_size {}",
        embedding.len(),
        cfg.hidden_size
    );
    Ok(embedding.repeat(cfg.hc_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
    const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

    /// Independent logistic sigmoid via the tanh half-angle identity,
    /// `sigmoid(x) = (1 + tanh(x/2)) / 2`. Shares no code with [`sigmoid64`],
    /// so it can disagree with it.
    fn sigmoid_via_tanh(x: f64) -> f64 {
        0.5 * (1.0 + (0.5 * x).tanh())
    }

    fn assert_close(label: &str, got: f64, want: f64, tol: f64) {
        let denom = want.abs().max(1e-6);
        assert!(
            (got - want).abs() / denom < tol,
            "{label}: got {got} want {want} (rel {})",
            (got - want).abs() / denom
        );
    }

    fn assert_slice_close(label: &str, got: &[f32], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_close(&format!("{label}[{i}]"), f64::from(g), w, tol);
        }
    }

    /// Deterministic xorshift, so a failure reproduces.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn fill(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * scale).collect()
        }
    }

    // ---------------------------------------------------------------- analytic

    /// A group whose values are all the constant `c` normalises to `c /
    /// sqrt(c^2 + eps)` — a closed form, not a re-run of the implementation.
    /// Four groups with different `c` therefore all collapse to `sign(c)`,
    /// which one norm over the whole vector could not do.
    #[test]
    fn grouped_rmsnorm_normalizes_each_group_independently() {
        const GROUP: usize = 3;
        let consts = [1.0f64, -2.0, 4.0, 0.5];
        let eps = 1e-6f32;
        let x: Vec<f32> = consts
            .iter()
            .flat_map(|&c| std::iter::repeat_n(c as f32, GROUP))
            .collect();
        let weight = vec![0.0f32; x.len()];

        let got = grouped_rmsnorm(&x, &weight, GROUP, eps).expect("grouped norm");
        let want: Vec<f64> = consts
            .iter()
            .flat_map(|&c| std::iter::repeat_n(c / (c * c + f64::from(eps)).sqrt(), GROUP))
            .collect();
        assert_slice_close("grouped", &got, &want, 1e-6);

        // The same call with group_size == len() is the ungrouped norm, and it
        // must disagree — otherwise this test proves nothing.
        let global = grouped_rmsnorm(&x, &weight, x.len(), eps).expect("global norm");
        let mean_sq: f64 = consts.iter().map(|c| c * c).sum::<f64>() / consts.len() as f64;
        let inv = (mean_sq + f64::from(eps)).sqrt().recip();
        let want_global: Vec<f64> = consts
            .iter()
            .flat_map(|&c| std::iter::repeat_n(c * inv, GROUP))
            .collect();
        assert_slice_close("global", &global, &want_global, 1e-6);
        assert!(
            (f64::from(global[0]) - f64::from(got[0])).abs() > 0.1,
            "grouped and ungrouped must differ on unequal groups, else the test is blind"
        );
    }

    /// The gain is `1 + weight`. A zero weight is the identity gain (so the
    /// per-group RMS of the output is 1), and a weight of exactly -1 zeroes the
    /// output. Under a bare-`weight` convention both statements would be false.
    #[test]
    fn grouped_rmsnorm_gain_is_one_plus_weight() {
        const GROUP: usize = 8;
        let mut rng = Rng(0x51ED_2701);
        let x = rng.fill(4 * GROUP, 3.0);

        let unit = grouped_rmsnorm(&x, &vec![0.0; x.len()], GROUP, 0.0).expect("zero-weight norm");
        for (g, group) in unit.chunks_exact(GROUP).enumerate() {
            let ms: f64 = group
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / GROUP as f64;
            assert_close(&format!("group {g} mean-square"), ms, 1.0, 1e-6);
        }

        let zeroed =
            grouped_rmsnorm(&x, &vec![-1.0; x.len()], GROUP, 0.0).expect("minus-one-weight norm");
        assert!(
            zeroed.iter().all(|&v| v == 0.0),
            "weight == -1 must cancel the gain exactly"
        );
    }

    #[test]
    fn activations_match_the_tanh_identity() {
        for i in -400..=400 {
            let x = f64::from(i) / 20.0;
            assert_close(
                "sigmoid",
                f64::from(sigmoid(x as f32)),
                sigmoid_via_tanh(x),
                1e-6,
            );
            assert_close(
                "silu",
                f64::from(silu(x as f32)),
                x * sigmoid_via_tanh(x),
                1e-6,
            );
        }
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(silu(0.0), 0.0);
    }

    /// A tiny config where every stage is hand-computable.
    fn tiny_cfg() -> HyperConnectionConfig {
        HyperConnectionConfig {
            hidden_size: 2,
            hc_count: 4,
            hc_lowrank: 1,
            // eps = 0 so a constant stream normalises to exactly +/-1 and the
            // hand arithmetic below is exact rather than approximate.
            rms_norm_eps: 0.0,
        }
    }

    /// With both mix weights zero the gate is `sigmoid(silu(0)) = sigmoid(0) =
    /// 0.5` everywhere, so the block input is exactly half the stream-mean of
    /// the normed residual. With `block_inject` zero the injection weight is
    /// exactly `2 * sigmoid(0) = 1`, i.e. an unweighted residual add.
    #[test]
    fn zero_mix_weights_give_a_half_gate_and_unit_injection() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let weights = GatedResidualWeights {
            hc_norm: vec![0.0; hc_hidden],
            mix_down: vec![0.0; cfg.hc_lowrank * hc_hidden],
            mix_up: vec![0.0; hc_hidden * cfg.hc_lowrank],
            block_inject: Some(vec![0.0; cfg.hc_count * hc_hidden]),
        };
        // Stream s is the constant c_s, so normed stream s is sign(c_s) exactly.
        let consts = [1.0f32, -2.0, 4.0, 0.5];
        let state: Vec<f32> = consts.iter().flat_map(|&c| [c, c]).collect();

        let out = gated_residual(&cfg, &weights, &state).expect("gated residual");
        assert!(out.mix_gate.iter().all(|&m| m == 0.5), "gate must be 0.5");
        let mean_sign: f64 = consts.iter().map(|&c| f64::from(c.signum())).sum::<f64>() / 4.0;
        assert_slice_close(
            "block_input",
            &out.block_input,
            &[0.5 * mean_sign, 0.5 * mean_sign],
            1e-6,
        );
        let inj = out.injection_weights.expect("has block_inject");
        assert_eq!(inj, vec![1.0; 4], "2 * sigmoid(0) is exactly 1");
    }

    /// `silu(z / hc_count)` vs `silu(z) / hc_count`, made visible.
    ///
    /// All four normed streams are `+1`, `mix_down` is all ones, so the
    /// down-projection logit is `z = hc_hidden = 8`. The correct gate is
    /// `sigmoid(silu(2))`; the transposed-divide gate would be
    /// `sigmoid(silu(8) / 4)`. Both are checked, and the wrong one is asserted
    /// to be a different number, so the test cannot pass by accident.
    #[test]
    fn the_divide_by_hc_count_precedes_the_silu() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let weights = GatedResidualWeights {
            hc_norm: vec![0.0; hc_hidden],
            mix_down: vec![1.0; cfg.hc_lowrank * hc_hidden],
            mix_up: vec![1.0; hc_hidden * cfg.hc_lowrank],
            block_inject: None,
        };
        let state = vec![1.0f32; hc_hidden];

        let out = gated_residual(&cfg, &weights, &state).expect("gated residual");
        let z = hc_hidden as f64;
        let want_u = (z / 4.0) * sigmoid_via_tanh(z / 4.0);
        let wrong_u = z * sigmoid_via_tanh(z) / 4.0;
        assert_slice_close("lowrank", &out.lowrank, &[want_u], 1e-6);
        assert!(
            (want_u - wrong_u).abs() > 0.2,
            "the two orderings must be distinguishable here, else the test is blind"
        );

        // normed is all +1, so the block input is just the gate.
        let want_gate = sigmoid_via_tanh(want_u);
        assert_slice_close(
            "block_input",
            &out.block_input,
            &[want_gate, want_gate],
            1e-6,
        );
    }

    /// The same trap on the inject path: `2 * sigmoid(z / hc_count)`.
    #[test]
    fn the_divide_by_hc_count_precedes_the_inject_sigmoid() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let weights = GatedResidualWeights {
            hc_norm: vec![0.0; hc_hidden],
            mix_down: vec![0.0; cfg.hc_lowrank * hc_hidden],
            mix_up: vec![0.0; hc_hidden * cfg.hc_lowrank],
            block_inject: Some(vec![1.0; cfg.hc_count * hc_hidden]),
        };
        let state = vec![1.0f32; hc_hidden];

        let inj = gated_residual(&cfg, &weights, &state)
            .expect("gated residual")
            .injection_weights
            .expect("has block_inject");
        let z = hc_hidden as f64;
        let want = 2.0 * sigmoid_via_tanh(z / 4.0);
        let wrong = 2.0 * sigmoid_via_tanh(z);
        assert_slice_close("inject", &inj, &[want; 4], 1e-6);
        assert!(
            (want - wrong).abs() > 0.2,
            "the two orderings must be distinguishable here, else the test is blind"
        );
    }

    /// `2 * sigmoid(.)` is bounded by construction; assert it on random weights
    /// so a lost factor of 2 or a missing sigmoid surfaces as a range violation.
    #[test]
    fn injection_weights_lie_strictly_between_zero_and_two() {
        let cfg = HyperConnectionConfig {
            hidden_size: 16,
            hc_count: 4,
            hc_lowrank: 5,
            rms_norm_eps: 1e-6,
        };
        let hc_hidden = cfg.hc_hidden();
        let mut rng = Rng(0xC0FF_EE01);
        for _ in 0..64 {
            let weights = GatedResidualWeights {
                hc_norm: rng.fill(hc_hidden, 1.0),
                mix_down: rng.fill(cfg.hc_lowrank * hc_hidden, 1.0),
                mix_up: rng.fill(hc_hidden * cfg.hc_lowrank, 1.0),
                // Kept modest on purpose: `2 * sigmoid` saturates to exactly
                // 2.0f32 past a logit of ~19, and a spurious failure there
                // would be about f32 rounding, not about the bound under test.
                block_inject: Some(rng.fill(cfg.hc_count * hc_hidden, 0.5)),
            };
            let state = rng.fill(hc_hidden, 4.0);
            let inj = gated_residual(&cfg, &weights, &state)
                .expect("gated residual")
                .injection_weights
                .expect("has block_inject");
            assert_eq!(inj.len(), cfg.hc_count);
            for &w in &inj {
                assert!(w > 0.0 && w < 2.0, "injection weight {w} left (0, 2)");
            }
        }
    }

    /// The mean over streams of a quantity that is constant across streams is
    /// that quantity. Arranged by repeating the same stream and making `mix_up`
    /// stream-periodic, so the gate is stream-periodic too; the block input must
    /// then equal stream 0's `gate * normed` exactly.
    #[test]
    fn mean_over_streams_of_a_stream_constant_is_that_constant() {
        let cfg = HyperConnectionConfig {
            hidden_size: 6,
            hc_count: 4,
            hc_lowrank: 3,
            rms_norm_eps: 1e-6,
        };
        let hc_hidden = cfg.hc_hidden();
        let mut rng = Rng(0x1234_5678);

        let one_stream_up = rng.fill(cfg.hidden_size * cfg.hc_lowrank, 1.0);
        let weights = GatedResidualWeights {
            // A stream-periodic hc_norm keeps `normed` stream-periodic too.
            hc_norm: rng.fill(cfg.hidden_size, 0.5).repeat(cfg.hc_count),
            mix_down: rng.fill(cfg.hc_lowrank * hc_hidden, 1.0),
            mix_up: one_stream_up.repeat(cfg.hc_count),
            block_inject: None,
        };
        let state = rng.fill(cfg.hidden_size, 2.0).repeat(cfg.hc_count);

        let out = gated_residual(&cfg, &weights, &state).expect("gated residual");
        for s in 1..cfg.hc_count {
            let base = s * cfg.hidden_size;
            assert_eq!(
                &out.mix_gate[base..base + cfg.hidden_size],
                &out.mix_gate[..cfg.hidden_size],
                "gate must be stream-periodic here"
            );
        }
        let want: Vec<f64> = (0..cfg.hidden_size)
            .map(|i| f64::from(out.mix_gate[i]) * f64::from(out.normed[i]))
            .collect();
        assert_slice_close("block_input", &out.block_input, &want, 1e-6);
        assert!(out.injection_weights.is_none(), "mixer injects nothing");
    }

    #[test]
    fn seeding_tiles_the_embedding_stream_major() {
        let cfg = HyperConnectionConfig {
            hidden_size: 3,
            hc_count: 4,
            hc_lowrank: 1,
            rms_norm_eps: 1e-6,
        };
        let emb = [1.0f32, 2.0, 3.0];
        let seeded = seed_hyper_state(&cfg, &emb).expect("seed");
        assert_eq!(
            seeded,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        // Interleaved (channel-major) tiling would have been [1,1,1,1,2,2,...].
        assert_eq!(&seeded[cfg.hidden_size..2 * cfg.hidden_size], &emb);
    }

    #[test]
    fn injection_scales_each_stream_by_its_own_weight() {
        let cfg = HyperConnectionConfig {
            hidden_size: 3,
            hc_count: 4,
            hc_lowrank: 1,
            rms_norm_eps: 1e-6,
        };
        let mut state = vec![0.0f32; cfg.hc_hidden()];
        let y = [1.0f32, 2.0, 3.0];
        inject_block_output(&cfg, &mut state, &[1.0, 2.0, 3.0, 4.0], &y).expect("inject");
        assert_eq!(
            state,
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0]
        );
        // Accumulates onto the existing residual rather than replacing it.
        inject_block_output(&cfg, &mut state, &[1.0, 1.0, 1.0, 1.0], &y).expect("inject twice");
        assert_eq!(state[0], 2.0);
        assert_eq!(state[3], 3.0);
    }

    #[test]
    fn wrong_shapes_fail_loud() {
        let cfg = tiny_cfg();
        let hc_hidden = cfg.hc_hidden();
        let good = GatedResidualWeights {
            hc_norm: vec![0.0; hc_hidden],
            mix_down: vec![0.0; cfg.hc_lowrank * hc_hidden],
            mix_up: vec![0.0; hc_hidden * cfg.hc_lowrank],
            block_inject: None,
        };
        assert!(gated_residual(&cfg, &good, &vec![0.0; hc_hidden - 1]).is_err());

        let mut short = good.clone();
        short.hc_norm.pop();
        assert!(short.validate(&cfg).is_err());

        // Documented blind spot, asserted so nobody assumes otherwise:
        // `mix_down` and `mix_up` hold the same number of elements, so swapping
        // them passes `validate`. Only the loader's dims check (see
        // `real_checkpoint_hyper_connection_shapes`) can catch that.
        let mut swapped = good.clone();
        std::mem::swap(&mut swapped.mix_down, &mut swapped.mix_up);
        assert!(swapped.validate(&cfg).is_ok());

        let mut bad_inject = good.clone();
        bad_inject.block_inject = Some(vec![0.0; hc_hidden]);
        assert!(bad_inject.validate(&cfg).is_err());

        assert!(grouped_rmsnorm(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 2, 0.0).is_err());
        assert!(grouped_rmsnorm(&[1.0, 2.0], &[1.0], 1, 0.0).is_err());
    }

    // -------------------------------------------------------- real checkpoint

    fn checkpoint_dir() -> Option<PathBuf> {
        let dir = std::env::var_os(CKPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(CKPT_DEFAULT));
        dir.join("model.safetensors.index.json")
            .is_file()
            .then_some(dir)
    }

    const ATTN0: &str = "model.language_model.layers.0.attn_hyper_connection";
    const MLP0: &str = "model.language_model.layers.0.mlp_hyper_connection";
    const MIXER: &str = "model.language_model.hyper_connection_mixer";
    /// Both `layers.0.*` hyper-connections and the stream-level mixer live here.
    const HC_SHARD: &str = "model-bf16-00001.safetensors";
    /// `embed_tokens` lives here; the seeded state is one of its rows, tiled.
    const EMBED_SHARD: &str = "model-bf16-00012.safetensors";

    fn open_hc_shards(dir: &std::path::Path) -> infer_gguf::safetensors::SafeTensorsDir {
        infer_gguf::safetensors::SafeTensorsDir::open_files(&[
            dir.join(HC_SHARD),
            dir.join(EMBED_SHARD),
        ])
        .expect("open the two bf16 shards that carry the hyper-connection weights")
    }

    fn load_bf16(
        st: &infer_gguf::safetensors::SafeTensorsDir,
        name: &str,
        want_dims: &[u64],
    ) -> Vec<f32> {
        let info = st
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} missing from the checkpoint"));
        assert_eq!(info.dtype, "BF16", "{name} dtype");
        assert_eq!(
            info.dims, want_dims,
            "{name} dims (GGUF order, innermost first)"
        );
        let n = info.element_count() as usize;
        infer_gguf::dequant::dequantize_row_bf16(st.tensor_data(name).expect("tensor data"), n)
            .expect("bf16 -> f32")
    }

    fn load_gated(
        st: &infer_gguf::safetensors::SafeTensorsDir,
        prefix: &str,
        with_inject: bool,
    ) -> GatedResidualWeights {
        // Every name goes through the classifier, so a rename shows up here and
        // not as a silently absent weight.
        for part in ["hc_norm", "input_mix_weight_down", "input_mix_weight_up"] {
            let name = format!("{prefix}.{part}.weight");
            crate::qwen4_names::classify_qwen4_tensor(&name)
                .unwrap_or_else(|e| panic!("classify {name}: {e}"));
        }
        GatedResidualWeights {
            hc_norm: load_bf16(st, &format!("{prefix}.hc_norm.weight"), &[10240]),
            // Header shape [320, 10240]; `dims` is that reversed.
            mix_down: load_bf16(
                st,
                &format!("{prefix}.input_mix_weight_down.weight"),
                &[10240, 320],
            ),
            // Header shape [10240, 320].
            mix_up: load_bf16(
                st,
                &format!("{prefix}.input_mix_weight_up.weight"),
                &[320, 10240],
            ),
            block_inject: with_inject.then(|| {
                load_bf16(
                    st,
                    &format!("{prefix}.block_inject_weight.weight"),
                    &[10240, 4],
                )
            }),
        }
    }

    /// Row `tok` of `embed_tokens`, straight out of the mmap.
    fn embed_row(
        st: &infer_gguf::safetensors::SafeTensorsDir,
        tok: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let name = "model.language_model.embed_tokens.weight";
        let info = st.tensor(name).expect("embed_tokens");
        assert_eq!(info.dims, vec![2560, 248_320], "embed_tokens dims");
        let data = st.tensor_data(name).expect("embed_tokens data");
        infer_gguf::dequant::dequantize_row_bf16(
            &data[tok * hidden * 2..(tok + 1) * hidden * 2],
            hidden,
        )
        .expect("bf16 -> f32")
    }

    /// Shapes, dtypes and the `use_combine=False` asymmetry, read off the real
    /// files rather than off `config.json`.
    #[test]
    fn real_checkpoint_hyper_connection_shapes() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let st = open_hc_shards(&dir);
        let cfg = HyperConnectionConfig::QWEN4_EXP;

        for (prefix, inject) in [(ATTN0, true), (MLP0, true), (MIXER, false)] {
            let w = load_gated(&st, prefix, inject);
            w.validate(&cfg)
                .unwrap_or_else(|e| panic!("{prefix} against QWEN4_EXP: {e}"));
            assert_eq!(w.is_mixer(), !inject, "{prefix} use_combine");
        }
        assert!(
            st.tensor(&format!("{MIXER}.block_inject_weight.weight"))
                .is_none(),
            "the mixer is use_combine=False and must carry no block_inject_weight"
        );
        assert!(
            crate::qwen4_names::classify_qwen4_tensor(&format!(
                "{MIXER}.block_inject_weight.weight"
            ))
            .is_err(),
            "the classifier must reject a mixer block_inject_weight too"
        );
    }

    /// The claim in the module docs, re-checked every run: this checkpoint has
    /// no `model.norm`, and the mixer's `hc_norm` stands in for it.
    #[test]
    fn real_checkpoint_has_no_separate_final_norm() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let index = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
            .expect("read model.safetensors.index.json");
        for absent in [
            "\"model.language_model.norm.weight\"",
            "\"model.norm.weight\"",
            "\"model.language_model.final_layernorm.weight\"",
        ] {
            assert!(
                !index.contains(absent),
                "expected no {absent} in the weight_map — the mixer is the final norm"
            );
        }
        assert!(
            index.contains("\"model.language_model.hyper_connection_mixer.hc_norm.weight\""),
            "the mixer's hc_norm must exist, since nothing else normalises the last hidden state"
        );

        let st = open_hc_shards(&dir);
        // The mixer's hc_norm is trained to a large positive gain (mean ~ +2.75
        // over 1 + w), the per-layer ones sit near the zeros-init they started
        // from. If either mean moved to ~1.0 the `1 + weight` reading would be
        // the thing to re-check.
        let mixer_mean = mean(&load_bf16(
            &st,
            &format!("{MIXER}.hc_norm.weight"),
            &[10240],
        ));
        let layer_mean = mean(&load_bf16(
            &st,
            &format!("{ATTN0}.hc_norm.weight"),
            &[10240],
        ));
        assert!(
            mixer_mean > 2.0,
            "mixer hc_norm weight mean {mixer_mean} — expected a large trained gain"
        );
        assert!(
            layer_mean.abs() < 0.5,
            "layer-0 hc_norm weight mean {layer_mean} — expected it near the zeros init"
        );
    }

    fn mean(v: &[f32]) -> f64 {
        v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64
    }

    // Golden values for the real layer-0 `attn_hyper_connection` and for the
    // stream-level mixer, produced by an independent NumPy transcription of
    // `Qwen4ExpTextGatedResidual.forward` (float64 accumulation, float32 at each
    // stage boundary, bf16 weights widened bit-exactly). Input: row 1234 of
    // `embed_tokens`, tiled 4x, with stream `s` scaled by `2^s` so the four
    // grouped norms see different energy — an ungrouped 10240-wide norm would
    // divide stream 0 by sqrt(21.25) instead of by its own RMS and miss these by
    // ~5x, and the unscaled/tiled goldens below differ from these by ~1.4%
    // because `rms_norm_eps` is not negligible against this row's tiny scale.
    //
    // Reproduce: read the four bf16 tensors + the embedding row with numpy,
    // then hn = (x.reshape(4,2560) * rsqrt(mean(x^2,-1,keepdims)+1e-6)).ravel()
    // * (1+N); u = silu((D@hn)/4); m = sigmoid(U@u);
    // mixed = (m.reshape(4,2560)*hn.reshape(4,2560)).mean(0);
    // inj = 2*sigmoid((B@hn)/4).
    const GOLD_TOKEN: usize = 1234;
    /// Shortest round-trip decimals for the exact bf16 values
    /// -0.01007080078125, 0.0025634765625, -0.0029144287109375, 0.001617431640625.
    const GOLD_EMBED_HEAD: [f32; 4] = [
        -0.010_070_801,
        0.002_563_476_6,
        -0.002_914_428_7,
        0.001_617_431_6,
    ];
    const GOLD_SCALED_ATTN_HEAD: [f64; 4] = [
        -1.647_156_238_555_908_2,
        0.344_711_959_362_030_03,
        -0.321_865_022_182_464_6,
        0.221_010_729_670_524_6,
    ];
    const GOLD_SCALED_ATTN_LAST: f64 = 0.764_234_840_869_903_6;
    const GOLD_SCALED_ATTN_PEAK: (usize, f64) = (1028, 6.176_609_039_306_641);
    const GOLD_SCALED_ATTN_SUM: f64 = 128.312_868_754_059_3;
    const GOLD_SCALED_INJECT: [f64; 4] = [
        0.030_726_430_937_647_82,
        0.027_499_154_210_090_637,
        0.143_514_588_475_227_36,
        0.031_444_463_878_870_01,
    ];
    const GOLD_TILED_ATTN_HEAD: [f64; 4] = [
        -1.624_533_295_631_408_7,
        0.339_794_427_156_448_36,
        -0.319_046_080_112_457_3,
        0.218_300_178_647_041_32,
    ];
    const GOLD_TILED_INJECT: [f64; 4] = [
        0.032_229_006_290_435_79,
        0.028_864_203_020_930_29,
        0.147_542_253_136_634_83,
        0.032_933_581_620_452_79,
    ];
    const GOLD_SCALED_MIXER_HEAD: [f64; 4] = [
        -4.603_444_099_426_269_5,
        1.644_690_155_982_971_2,
        -2.098_404_645_919_8,
        0.915_295_422_077_179,
    ];
    const GOLD_SCALED_MIXER_PEAK: (usize, f64) = (2187, -21.140_928_268_432_617);
    const GOLD_SCALED_MIXER_SUM: f64 = 554.841_492_993_873_5;
    /// f64 accumulation on both sides, f32 at the stage boundaries; the residual
    /// gap is NumPy's pairwise summation against this module's sequential one.
    const GOLD_TOL: f64 = 2e-5;

    #[test]
    fn real_checkpoint_layer0_attn_matches_the_numpy_oracle() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let cfg = HyperConnectionConfig::QWEN4_EXP;
        let st = open_hc_shards(&dir);
        let weights = load_gated(&st, ATTN0, true);
        let emb = embed_row(&st, GOLD_TOKEN, cfg.hidden_size);
        assert_eq!(&emb[..4], &GOLD_EMBED_HEAD, "wrong embed row or stride");

        // The state as the model actually seeds it, and the same state with the
        // streams pulled apart so the grouped norm has something to prove.
        let tiled = seed_hyper_state(&cfg, &emb).expect("seed");
        let mut scaled = tiled.clone();
        for s in 0..cfg.hc_count {
            let f = (1u32 << s) as f32;
            for v in &mut scaled[s * cfg.hidden_size..(s + 1) * cfg.hidden_size] {
                *v *= f;
            }
        }

        let out = gated_residual(&cfg, &weights, &scaled).expect("gated residual");
        assert_eq!(out.normed.len(), cfg.hc_hidden());
        assert_eq!(out.lowrank.len(), cfg.hc_lowrank);
        assert_eq!(out.block_input.len(), cfg.hidden_size);
        assert_slice_close(
            "scaled block_input head",
            &out.block_input[..4],
            &GOLD_SCALED_ATTN_HEAD,
            GOLD_TOL,
        );
        assert_close(
            "scaled block_input last",
            f64::from(out.block_input[cfg.hidden_size - 1]),
            GOLD_SCALED_ATTN_LAST,
            GOLD_TOL,
        );
        let (peak_i, peak_v) = GOLD_SCALED_ATTN_PEAK;
        assert_close(
            "scaled block_input peak",
            f64::from(out.block_input[peak_i]),
            peak_v,
            GOLD_TOL,
        );
        assert_close(
            "scaled block_input sum",
            out.block_input.iter().map(|&v| f64::from(v)).sum::<f64>(),
            GOLD_SCALED_ATTN_SUM,
            1e-4,
        );
        let inj = out.injection_weights.expect("attn hc has block_inject");
        assert_slice_close("scaled inject", &inj, &GOLD_SCALED_INJECT, GOLD_TOL);

        // On the real weights the inject logit is ~ -4.16, so `2*sigmoid(z/4)`
        // lands near 0.03 while `2*sigmoid(z)` would land near 1e-7. The golden
        // above already pins it; assert the separation so the pin is meaningful.
        assert!(
            inj.iter().all(|&w| w > 1e-3 && w < 0.5),
            "layer-0 injection weights {inj:?} are the /hc_count-before-sigmoid values"
        );

        let tiled_out = gated_residual(&cfg, &weights, &tiled).expect("gated residual, tiled");
        assert_slice_close(
            "tiled block_input head",
            &tiled_out.block_input[..4],
            &GOLD_TILED_ATTN_HEAD,
            GOLD_TOL,
        );
        assert_slice_close(
            "tiled inject",
            &tiled_out.injection_weights.expect("has block_inject"),
            &GOLD_TILED_INJECT,
            GOLD_TOL,
        );

        // A 10240-wide norm would put the scaled and tiled answers within f32
        // noise of one another (a global mean-of-squares is scale-blind in a way
        // the per-stream one is not); they are ~1.4% apart.
        let gap = (f64::from(out.block_input[0]) - f64::from(tiled_out.block_input[0])).abs()
            / f64::from(tiled_out.block_input[0]).abs();
        assert!(
            gap > 1e-3,
            "scaled and tiled block inputs must differ ({gap}); a global norm would collapse them"
        );
    }

    #[test]
    fn real_checkpoint_mixer_matches_the_numpy_oracle() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no qwen3.8-flash-next checkpoint (set {CKPT_ENV})");
            return;
        };
        let cfg = HyperConnectionConfig::QWEN4_EXP;
        let st = open_hc_shards(&dir);
        let weights = load_gated(&st, MIXER, false);
        let emb = embed_row(&st, GOLD_TOKEN, cfg.hidden_size);

        let mut state = seed_hyper_state(&cfg, &emb).expect("seed");
        for s in 0..cfg.hc_count {
            let f = (1u32 << s) as f32;
            for v in &mut state[s * cfg.hidden_size..(s + 1) * cfg.hidden_size] {
                *v *= f;
            }
        }

        let out = gated_residual(&cfg, &weights, &state).expect("mixer");
        assert!(
            out.injection_weights.is_none(),
            "use_combine=False returns only the mixed input"
        );
        assert_eq!(out.block_input.len(), cfg.hidden_size);
        assert_slice_close(
            "mixer head",
            &out.block_input[..4],
            &GOLD_SCALED_MIXER_HEAD,
            GOLD_TOL,
        );
        let (peak_i, peak_v) = GOLD_SCALED_MIXER_PEAK;
        assert_close(
            "mixer peak",
            f64::from(out.block_input[peak_i]),
            peak_v,
            GOLD_TOL,
        );
        assert_close(
            "mixer sum",
            out.block_input.iter().map(|&v| f64::from(v)).sum::<f64>(),
            GOLD_SCALED_MIXER_SUM,
            1e-4,
        );
    }
}
