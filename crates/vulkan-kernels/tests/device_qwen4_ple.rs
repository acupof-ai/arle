//! On-device Qwen3.8-Flash-Next (`qwen4_exp`) PLE correctness proof.
//!
//! Oracle-gates the two fused PLE halves — `qwen4_ple_gate` (projections →
//! gated value + `norm_conv`'s output) and `qwen4_ple_conv` (dilated depthwise
//! conv + SiLU + ring roll + residual add) — against the host f32 reference
//! `infer_vulkan::qwen4_ple::PleLayer`, itself transcribed line-for-line from
//! `Qwen4ExpTextPLELayer`.
//!
//! The oracle is TRANSCRIBED here rather than imported: `infer-vulkan` depends
//! on `vulkan-kernels`, so a dev-dependency the other way would need a
//! Cargo.toml change, and `device_router_topk.rs` already establishes the
//! transcribe-the-reference pattern for this crate. The transcription is the
//! part that can rot, so every function below names the oracle item it copies.
//!
//! Three things this file exists to catch, beyond "the numbers are close":
//!   * `Qwen4ExpTextRMSNorm` scales by `1.0 + weight`, not `weight`
//!     (`grouped_rms_norm`'s only multiply, exercised with non-zero weights).
//!   * the residual branch adds the UN-normed gated value while only the conv
//!     branch consumes the normed one — `swapping_the_normed_branches_diverges`
//!     asserts the swapped answers are far outside tolerance, so the test can
//!     actually fail if the shader aliases them.
//!   * `sign(0) == 0`, not `+1` — `a_zero_gate_stays_exactly_one_half` pins the
//!     5e-4 relative shift a `+1` would introduce.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, launch_cached, qwen4_ple_conv_dispatch, qwen4_ple_conv_params,
    qwen4_ple_conv_ring_advance, qwen4_ple_conv_state_len, qwen4_ple_gate_dispatch,
    qwen4_ple_gate_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

// The on-box `qwen3.8-flash-next-nvfp4` shapes, mirroring
// `infer_vulkan::qwen4_ple::PleConfig::qwen4_exp`.
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const KERNEL_SIZE: usize = 4;
const DILATION: usize = 3;
const STATE_LEN: usize = (KERNEL_SIZE - 1) * DILATION;
const EPS: f32 = 1e-6;

// --------------------------------------------------------------------------
// host oracle, transcribed from `infer_vulkan::qwen4_ple`
// --------------------------------------------------------------------------

/// `qwen4_ple::rms_norm_grouped`. Per-group RMS statistic over `group_size`
/// channels, but the learned scale applies to the flat row and it is
/// `1.0 + weight` — the parameter is zero-initialised in this model.
fn grouped_rms_norm(x: &[f32], weight: &[f32], group_size: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "RMSNorm weight width");
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

/// `qwen4_ple::signed_sqrt_gate` = `gate.abs().clamp_min(1e-6).sqrt() *
/// gate.sign()`, with the three-way comparison that keeps `sign(0) == 0`.
/// `f32::signum` returns `+1.0` for `+0.0` and would turn a zero gate into
/// `+1e-3`, which is exactly the mistake this oracle exists to detect.
fn signed_sqrt_gate(gate: f32) -> f32 {
    if gate.is_nan() {
        return gate;
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

/// The half of `PleLayer::forward` that `qwen4_ple_gate.comp` fuses: from the
/// two projections' outputs to `(gated, gated_normed)`.
///
/// `key` and `hidden_states` are `[seq][HC_HIDDEN]`, `value` is `[seq][HIDDEN]`
/// (no stream axis — one value row gates into every stream).
fn host_gate(
    key: &[f32],
    hidden_states: &[f32],
    value: &[f32],
    norm_key: &[f32],
    norm_query: &[f32],
    norm_conv: &[f32],
    seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    // The gate divisor is sqrt(hidden_size), scaled like an attention logit.
    let gate_scale = f64::from(HIDDEN as f32).sqrt();
    let mut gated = vec![0.0f32; seq * HC_HIDDEN];

    for (t, row) in gated.chunks_exact_mut(HC_HIDDEN).enumerate() {
        let key_normed =
            grouped_rms_norm(&key[t * HC_HIDDEN..][..HC_HIDDEN], norm_key, HIDDEN, EPS);
        let query_normed = grouped_rms_norm(
            &hidden_states[t * HC_HIDDEN..][..HC_HIDDEN],
            norm_query,
            HIDDEN,
            EPS,
        );
        for (stream, chunk) in row.chunks_exact_mut(HIDDEN).enumerate() {
            let k = &key_normed[stream * HIDDEN..][..HIDDEN];
            let q = &query_normed[stream * HIDDEN..][..HIDDEN];
            let dot: f64 = k
                .iter()
                .zip(q)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let weight = sigmoid(signed_sqrt_gate((dot / gate_scale) as f32));
            for (dst, &v) in chunk.iter_mut().zip(&value[t * HIDDEN..][..HIDDEN]) {
                *dst = weight * v;
            }
        }
    }

    let mut gated_normed = Vec::with_capacity(gated.len());
    for src in gated.chunks_exact(HC_HIDDEN) {
        gated_normed.extend(grouped_rms_norm(src, norm_conv, HIDDEN, EPS));
    }
    (gated, gated_normed)
}

/// `PleLayer::short_conv` plus the residual add at the tail of `forward`.
///
/// `state` is the oracle's time-major history, oldest row first, and is rolled
/// in place. `x_normed` is `gated_normed`; `residual` is the UN-normed `gated`.
/// Returns `[seq][HC_HIDDEN]`.
fn host_conv(
    x_normed: &[f32],
    conv_w: &[f32],
    state: &mut Vec<f32>,
    residual: &[f32],
    seq: usize,
) -> Vec<f32> {
    let mut history = Vec::with_capacity(state.len() + x_normed.len());
    history.extend_from_slice(state);
    history.extend_from_slice(x_normed);

    let mut out = vec![0.0f32; seq * HC_HIDDEN];
    for (t, out_row) in out.chunks_exact_mut(HC_HIDDEN).enumerate() {
        for (c, y) in out_row.iter_mut().enumerate() {
            let taps = &conv_w[c * KERNEL_SIZE..][..KERNEL_SIZE];
            let acc: f64 = taps
                .iter()
                .enumerate()
                .map(|(k, &w)| {
                    // history row `t + k*dilation` is absolute time
                    // `t - state_len + k*dilation`, so k = KERNEL_SIZE-1 is the
                    // current step and k = 0 reaches back `state_len`.
                    let v = history[(t + k * DILATION) * HC_HIDDEN + c];
                    f64::from(w) * f64::from(v)
                })
                .sum();
            // TRAP: the residual is the UN-normed tensor. Only the taps above
            // see `x_normed`.
            *y = residual[t * HC_HIDDEN + c] + silu(acc as f32);
        }
    }

    let keep_from = history.len() - STATE_LEN * HC_HIDDEN;
    state.clear();
    state.extend_from_slice(&history[keep_from..]);
    out
}

// --------------------------------------------------------------------------
// device plumbing
// --------------------------------------------------------------------------

/// Deterministic xorshift PRNG so failures reproduce.
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

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

fn upload<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc f32 buffer");
    b.copy_from_host(&bytes).expect("upload f32 buffer");
    b
}

/// HOST_CACHED, not the write-combined default: everything allocated here is
/// read back, and a WC read-back runs ~0.1 GB/s on this part.
fn upload_readback<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b =
        DeviceBuffer::alloc_host_cached(ctx, bytes.len().max(4)).expect("alloc host-cached");
    b.copy_from_host(&bytes).expect("upload host-cached buffer");
    b
}

fn read_f32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).expect("read back f32 buffer");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Max relative error against the oracle: `|got - want| / max(|want|, floor)`.
///
/// `floor` should be the scale of the TERMS the result was built from, not an
/// arbitrary epsilon. Above it this is a true relative error; below it — where
/// the result is a near-total cancellation of larger terms and its own
/// magnitude says nothing about the accuracy of the arithmetic — it degrades to
/// an absolute-error criterion at that scale. Passing a floor far under the
/// operand scale makes the metric report f32 noise as a large error; passing
/// one far over it hides real divergence.
fn max_rel_err(got: &[f32], want: &[f32], floor: f32) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "non-finite device value at {i}: {g}");
        let err = (g - w).abs() / w.abs().max(floor);
        if err > worst {
            worst = err;
            at = i;
        }
    }
    (worst, at)
}

fn context() -> Option<VulkanContext> {
    match VulkanContext::create() {
        Ok(c) => {
            eprintln!("ARLE Vulkan qwen4 PLE proof on: {}", c.device_name());
            Some(c)
        }
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping qwen4 PLE oracle test");
            None
        }
    }
}

/// Weights shared by the gate cases. Non-zero on every norm, because with
/// zero-initialised weights `1.0 + weight` and `weight` differ by exactly the
/// factor that makes the whole output vanish — a test with zero weights cannot
/// tell the two spellings apart in the direction that matters.
struct GateWeights {
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
}

impl GateWeights {
    fn random(rng: &mut Rng) -> Self {
        Self {
            norm_key: rng.vec(HC_HIDDEN, 0.4),
            norm_query: rng.vec(HC_HIDDEN, 0.4),
            norm_conv: rng.vec(HC_HIDDEN, 0.4),
        }
    }
}

/// Dispatch `qwen4_ple_gate` and return `(gated, gated_normed)`.
fn run_gate(
    ctx: &VulkanContext,
    key: &[f32],
    hidden_states: &[f32],
    value: &[f32],
    w: &GateWeights,
    seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cache = KernelCache::new();
    let buf_key = upload(ctx, key);
    let buf_hidden = upload(ctx, hidden_states);
    let buf_value = upload(ctx, value);
    let buf_nk = upload(ctx, &w.norm_key);
    let buf_nq = upload(ctx, &w.norm_query);
    let buf_nc = upload(ctx, &w.norm_conv);
    let mut buf_gated = upload_readback(ctx, &vec![0.0f32; seq * HC_HIDDEN]);
    let mut buf_normed = upload_readback(ctx, &vec![0.0f32; seq * HC_HIDDEN]);

    let push = qwen4_ple_gate_params(HIDDEN as u32, HC_COUNT as u32, seq as u32, EPS).to_le_bytes();
    launch_cached(
        &mut cache,
        ctx,
        Kernel::Qwen4PleGate,
        &[
            &buf_key,
            &buf_hidden,
            &buf_value,
            &buf_nk,
            &buf_nq,
            &buf_nc,
            &mut buf_gated,
            &mut buf_normed,
        ],
        qwen4_ple_gate_dispatch(HC_COUNT as u32, seq as u32),
        &push,
        Kernel::Qwen4PleGate.specialization_u32(),
    )
    .expect("ple_gate dispatch");

    (
        read_f32(&buf_gated, seq * HC_HIDDEN),
        read_f32(&buf_normed, seq * HC_HIDDEN),
    )
}

/// Dispatch `qwen4_ple_conv` and return `(out, rolled_ring)`. `ring` is
/// slot-major with `ring_pos` naming the oldest slot; with `ring_pos == 0` that
/// is byte-identical to the oracle's time-major state.
#[allow(clippy::too_many_arguments)]
fn run_conv(
    ctx: &VulkanContext,
    x_normed: &[f32],
    conv_w: &[f32],
    ring: &[f32],
    ring_pos: u32,
    residual: &[f32],
    seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cache = KernelCache::new();
    let buf_x = upload(ctx, x_normed);
    let buf_w = upload(ctx, conv_w);
    let mut buf_ring = upload_readback(ctx, ring);
    let buf_res = upload(ctx, residual);
    let mut buf_out = upload_readback(ctx, &vec![0.0f32; seq * HC_HIDDEN]);

    let push = qwen4_ple_conv_params(
        HC_HIDDEN as u32,
        seq as u32,
        KERNEL_SIZE as u32,
        DILATION as u32,
        ring_pos,
    )
    .to_le_bytes();
    launch_cached(
        &mut cache,
        ctx,
        Kernel::Qwen4PleConv,
        &[&buf_x, &buf_w, &mut buf_ring, &buf_res, &mut buf_out],
        qwen4_ple_conv_dispatch(HC_HIDDEN as u32),
        &push,
        Kernel::Qwen4PleConv.specialization_u32(),
    )
    .expect("ple_conv dispatch");

    (
        read_f32(&buf_out, seq * HC_HIDDEN),
        read_f32(&buf_ring, STATE_LEN * HC_HIDDEN),
    )
}

// --------------------------------------------------------------------------
// tests
// --------------------------------------------------------------------------

/// The gate half at the real shapes: decode (seq 1) and a short prefill.
///
/// Tolerance is relative, because the oracle accumulates its 2560-wide dot and
/// its RMS means in f64 while the shader runs an f32 strided-then-tree
/// reduction; the two are close, never bit-identical.
#[test]
fn gate_matches_host_oracle() {
    let Some(ctx) = context() else { return };
    let mut rng = Rng(0x4E4C_5045_0001_0001);
    let w = GateWeights::random(&mut rng);

    for seq in [1usize, 3] {
        let key = rng.vec(seq * HC_HIDDEN, 1.0);
        let hidden_states = rng.vec(seq * HC_HIDDEN, 1.0);
        let value = rng.vec(seq * HIDDEN, 1.0);
        let (want_gated, want_normed) = host_gate(
            &key,
            &hidden_states,
            &value,
            &w.norm_key,
            &w.norm_query,
            &w.norm_conv,
            seq,
        );
        let (got_gated, got_normed) = run_gate(&ctx, &key, &hidden_states, &value, &w, seq);

        let (err_g, at_g) = max_rel_err(&got_gated, &want_gated, 1e-3);
        let (err_n, at_n) = max_rel_err(&got_normed, &want_normed, 1e-3);
        eprintln!(
            "[ple_gate seq={seq}] max rel err: gated {err_g:.3e} (i={at_g}), \
             gated_normed {err_n:.3e} (i={at_n})"
        );
        assert!(
            err_g < 2e-5,
            "gated: max rel err {err_g:.3e} at {at_g} (got {} vs want {})",
            got_gated[at_g],
            want_gated[at_g]
        );
        assert!(
            err_n < 2e-5,
            "gated_normed: max rel err {err_n:.3e} at {at_n} (got {} vs want {})",
            got_normed[at_n],
            want_normed[at_n]
        );

        // The two outputs must not be the same tensor. `norm_conv` is random
        // and non-zero, so a shader that wrote `gated` to both bindings would
        // pass every per-element check above against the wrong reference.
        let identical = got_gated
            .iter()
            .zip(&got_normed)
            .all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(!identical, "gate wrote the same tensor to both outputs");
    }
}

/// The conv half at the real shapes, including a NON-ZERO `ring_pos`.
///
/// A ring at `ring_pos == 0` is laid out exactly like the oracle's time-major
/// state, so that case alone cannot see a wrong modulus. Feeding four
/// successive single-token steps walks `ring_pos` through 1, 2, 3 and compares
/// each against the oracle's own rolled state.
#[test]
fn conv_matches_host_oracle_across_the_ring() {
    let Some(ctx) = context() else { return };
    let mut rng = Rng(0x4E4C_5045_0002_0002);

    // The ring this test sizes and rotates by hand must be the one the shader
    // indexes, so take the length from the crate rather than restating it.
    assert_eq!(
        qwen4_ple_conv_state_len(KERNEL_SIZE as u32, DILATION as u32) as usize,
        STATE_LEN
    );

    // Distinct taps per channel, so an off-by-one in the ring indexing cannot
    // hide behind a symmetric kernel.
    let conv_w: Vec<f32> = (0..HC_HIDDEN * KERNEL_SIZE)
        .map(|i| 0.25 * ((i % KERNEL_SIZE) as f32 + 1.0) - 0.05 * ((i / KERNEL_SIZE) % 7) as f32)
        .collect();

    // A pre-filled (not zero) state, so every tap reads something the output
    // depends on from the first step.
    let mut host_state = rng.vec(STATE_LEN * HC_HIDDEN, 1.0);
    let mut ring = host_state.clone();
    let mut ring_pos = 0u32;
    let mut worst = 0.0f32;

    // `out = residual + silu(conv)` adds two terms of scale ~3 and ~1, and for
    // some channels they cancel to ~1e-3. Measuring relative error against a
    // result that small would report ordinary f32 noise (~2e-8 absolute) as a
    // 2e-5 "error" while saying nothing about the kernel; the operand scale is
    // the honest denominator there. Above 1.0 the metric is still fully
    // relative, and a wrong tap or ring slot moves outputs by O(1) — three
    // orders of magnitude past the 2e-5 gate below.
    const CONV_ERR_FLOOR: f32 = 1.0;

    for step in 0..5 {
        let seq = if step == 4 { 2 } else { 1 };
        let x_normed = rng.vec(seq * HC_HIDDEN, 1.0);
        let residual = rng.vec(seq * HC_HIDDEN, 3.0);

        let want = host_conv(&x_normed, &conv_w, &mut host_state, &residual, seq);
        let (got, rolled) = run_conv(&ctx, &x_normed, &conv_w, &ring, ring_pos, &residual, seq);

        let (err, at) = max_rel_err(&got, &want, CONV_ERR_FLOOR);
        worst = worst.max(err);
        eprintln!("[ple_conv step={step} seq={seq} ring_pos={ring_pos}] max rel err {err:.3e}");
        assert!(
            err < 2e-5,
            "step {step}: max rel err {err:.3e} at {at} (got {} vs want {})",
            got[at],
            want[at]
        );

        ring = rolled;
        ring_pos =
            qwen4_ple_conv_ring_advance(ring_pos, seq as u32, KERNEL_SIZE as u32, DILATION as u32);

        // The ring holds the same rows as the oracle's state, just rotated by
        // `ring_pos`. Checking it every step is what proves the roll, not only
        // the outputs (a kernel that never wrote the ring would still match the
        // first step's output exactly).
        //
        // BIT-exact, not within a tolerance: the shader copies `x_normed`
        // into the ring with no arithmetic, so any difference at all is a wrong
        // slot rather than rounding.
        for age_row in 0..STATE_LEN {
            let slot = (ring_pos as usize + age_row) % STATE_LEN;
            let got_row = &ring[slot * HC_HIDDEN..][..HC_HIDDEN];
            let want_row = &host_state[age_row * HC_HIDDEN..][..HC_HIDDEN];
            if let Some((i, (&g, &w))) = got_row
                .iter()
                .zip(want_row)
                .enumerate()
                .find(|(_, (g, w))| g != w)
            {
                panic!(
                    "step {step}: ring slot {slot} (age row {age_row}) holds the wrong row \
                     at channel {i}: {g} vs {w}"
                );
            }
        }
    }
    assert_eq!(
        ring_pos, 6,
        "ring_pos must have walked 0→1→2→3→4→6 across the five dispatches"
    );
    eprintln!("[ple_conv] worst rel err across the ring sweep: {worst:.3e}");
}

/// gate → conv end to end against the full `PleLayer::forward`, and — with the
/// same inputs — proof that BOTH branch swaps are far outside tolerance.
///
/// `norm_conv` is pinned at `+2.0` and the values are large, which puts the
/// normed tensor around 3 while the un-normed one is in the hundreds. That
/// separation is what makes the two swapped answers below fail by orders of
/// magnitude instead of by rounding.
#[test]
fn forward_matches_host_oracle_and_swapping_the_normed_branches_diverges() {
    let Some(ctx) = context() else { return };
    let mut rng = Rng(0x4E4C_5045_0003_0003);
    let seq = 1usize;

    let w = GateWeights {
        norm_key: rng.vec(HC_HIDDEN, 0.4),
        norm_query: rng.vec(HC_HIDDEN, 0.4),
        norm_conv: vec![2.0; HC_HIDDEN],
    };
    let key = rng.vec(seq * HC_HIDDEN, 1.0);
    let hidden_states = rng.vec(seq * HC_HIDDEN, 1.0);
    // Large values: `gated` lands around 300, `norm_conv(gated)` around 3.
    let value = rng.vec(seq * HIDDEN, 800.0);
    let conv_w = rng.vec(HC_HIDDEN * KERNEL_SIZE, 0.3);

    let (want_gated, want_normed) = host_gate(
        &key,
        &hidden_states,
        &value,
        &w.norm_key,
        &w.norm_query,
        &w.norm_conv,
        seq,
    );
    // The magnitude separation the swap assertions below depend on. If a future
    // change to this fixture collapsed it, the swap test would go vacuous.
    let mag = |v: &[f32]| {
        (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let (mag_gated, mag_normed) = (mag(&want_gated), mag(&want_normed));
    assert!(
        mag_gated > 50.0 * mag_normed,
        "fixture lost its magnitude separation: gated rms {mag_gated:.3}, normed rms {mag_normed:.3}"
    );

    let (got_gated, got_normed) = run_gate(&ctx, &key, &hidden_states, &value, &w, seq);
    let zero_state = vec![0.0f32; STATE_LEN * HC_HIDDEN];

    let mut state = zero_state.clone();
    let want_out = host_conv(&want_normed, &conv_w, &mut state, &want_gated, seq);
    let (got_out, _) = run_conv(&ctx, &got_normed, &conv_w, &zero_state, 0, &got_gated, seq);

    let (err, at) = max_rel_err(&got_out, &want_out, 1e-3);
    eprintln!(
        "[ple forward] max rel err {err:.3e} (i={at}); gated rms {mag_gated:.3}, \
         normed rms {mag_normed:.3}"
    );
    assert!(
        err < 2e-5,
        "forward: max rel err {err:.3e} at {at} (got {} vs want {})",
        got_out[at],
        want_out[at]
    );

    // Swap A: the residual adds the NORMED tensor. Every element moves by
    // (gated - gated_normed), i.e. by hundreds.
    let mut state_a = zero_state.clone();
    let swapped_residual = host_conv(&want_normed, &conv_w, &mut state_a, &want_normed, seq);
    let (err_a, _) = max_rel_err(&swapped_residual, &want_out, 1e-3);
    assert!(
        err_a > 1e-2,
        "swap A is inside tolerance ({err_a:.3e}) — this test cannot fail"
    );

    // Swap B: the conv taps the UN-NORMED tensor. Its SiLU saturates on inputs
    // in the hundreds, so the conv branch stops being a small correction.
    let mut state_b = zero_state.clone();
    let swapped_taps = host_conv(&want_gated, &conv_w, &mut state_b, &want_gated, seq);
    let (err_b, _) = max_rel_err(&swapped_taps, &want_out, 1e-3);
    assert!(
        err_b > 1e-2,
        "swap B is inside tolerance ({err_b:.3e}) — this test cannot fail"
    );
    eprintln!("[ple forward] swapped-branch divergence: residual {err_a:.3e}, taps {err_b:.3e}");
}

/// `sign(0) == 0`: a gate that is exactly zero must stay zero, so the sigmoid
/// stays exactly 0.5 — NOT `sigmoid(sqrt(1e-6)) = 0.50025`.
///
/// An all-zero `key_proj` output makes `key_normed` exactly zero on every
/// stream, so every dot is exactly zero regardless of the query. The kernel's
/// answer must be `0.5 * value`; the `+1`-sign answer is `0.50025 * value`,
/// a 5e-4 relative shift the assertions below bracket from both sides.
///
/// The second half plants a gate that is large and NEGATIVE (key = -query with
/// identity norm weights), which a shader that dropped `sign` entirely — using
/// only `sqrt(max(|g|, 1e-6))` — would get wrong by a factor of ~1200. Without
/// it, "sign(0) == 0" would also be satisfied by never applying a sign at all.
#[test]
fn a_zero_gate_stays_exactly_one_half() {
    let Some(ctx) = context() else { return };
    let mut rng = Rng(0x4E4C_5045_0004_0004);
    let seq = 1usize;

    let w = GateWeights {
        norm_key: rng.vec(HC_HIDDEN, 0.4),
        norm_query: rng.vec(HC_HIDDEN, 0.4),
        norm_conv: vec![0.0; HC_HIDDEN],
    };
    let key = vec![0.0f32; seq * HC_HIDDEN];
    let hidden_states = rng.vec(seq * HC_HIDDEN, 1.0);
    let value = rng.vec(seq * HIDDEN, 1.0);

    let (got_gated, _) = run_gate(&ctx, &key, &hidden_states, &value, &w, seq);
    // What a `sign(0) == +1` shader would gate by: the 1e-6 magnitude clamp
    // survives, so the gate becomes +1e-3 instead of 0 and the sigmoid moves
    // off 0.5 by 2.5e-4. Derived, not pasted, so it tracks the clamp constant.
    let wrong_weight = sigmoid(1e-6f32.sqrt());

    let mut worst = 0.0f32;
    let mut closest_to_wrong = f32::INFINITY;
    for (stream, chunk) in got_gated.chunks_exact(HIDDEN).enumerate() {
        for (i, (&g, &v)) in chunk.iter().zip(&value).enumerate() {
            let want = 0.5 * v;
            worst = worst.max((g - want).abs() / want.abs().max(1e-3));
            let wrong = wrong_weight * v;
            if v.abs() > 0.5 {
                closest_to_wrong = closest_to_wrong.min((g - wrong).abs() / wrong.abs().max(1e-3));
            }
            assert!(
                (g - want).abs() <= 1e-6 * want.abs().max(1.0),
                "stream {stream} ch {i}: got {g}, want {want} (sign(0) leaked in as +1?)"
            );
        }
    }
    eprintln!(
        "[ple_gate sign(0)] max rel err vs 0.5*value {worst:.3e}; \
         closest approach to the sign(0)=+1 answer {closest_to_wrong:.3e}"
    );
    assert!(
        closest_to_wrong > 1e-4,
        "the sign(0)=+1 answer is within {closest_to_wrong:.3e} — this test cannot fail"
    );

    // Now the large-negative gate. `norm_key`/`norm_query` are zero here so the
    // grouped norms are pure RMS: key = -hidden_states makes key_normed the
    // exact negation of query_normed, and the dot is -hidden.
    let w_plain = GateWeights {
        norm_key: vec![0.0; HC_HIDDEN],
        norm_query: vec![0.0; HC_HIDDEN],
        norm_conv: vec![0.0; HC_HIDDEN],
    };
    let neg_key: Vec<f32> = hidden_states.iter().map(|v| -v).collect();
    let (want_gated, _) = host_gate(
        &neg_key,
        &hidden_states,
        &value,
        &w_plain.norm_key,
        &w_plain.norm_query,
        &w_plain.norm_conv,
        seq,
    );
    let (got_neg, _) = run_gate(&ctx, &neg_key, &hidden_states, &value, &w_plain, seq);
    let (err, at) = max_rel_err(&got_neg, &want_gated, 1e-3);
    eprintln!("[ple_gate negative gate] max rel err {err:.3e}");
    assert!(
        err < 2e-5,
        "negative gate: max rel err {err:.3e} at {at} (got {} vs want {})",
        got_neg[at],
        want_gated[at]
    );

    // The sign really is doing work: dropping it would gate by
    // sigmoid(+sqrt(hidden^0.5)) ~= 0.999 instead of ~8e-4.
    let unsigned_weight = sigmoid((HIDDEN as f32).sqrt().sqrt());
    let signed_weight = want_gated[0] / value[0];
    assert!(
        (unsigned_weight - signed_weight).abs() > 0.9,
        "the negative-gate fixture is not exercising sign(): {signed_weight} vs {unsigned_weight}"
    );
    for (i, (&g, &v)) in got_neg
        .iter()
        .zip(value.iter().cycle())
        .enumerate()
        .take(HIDDEN)
    {
        assert!(
            (g - unsigned_weight * v).abs() > 0.2 * v.abs().max(1e-3),
            "ch {i}: device answer {g} matches the SIGN-DROPPED gate — sign() is missing"
        );
    }
}
