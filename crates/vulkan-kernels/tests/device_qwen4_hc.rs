//! On-device proof for the Qwen3.8-Flash-Next (`qwen4_exp`) hyper-connection.
//!
//! Runs the FULL four-dispatch `Qwen4ExpTextGatedResidual` — grouped RMSNorm,
//! down-projection GEMV, `qwen4_hc_mix`, `qwen4_hc_combine` — as ONE recorded
//! command buffer, and diffs every stage against a host f32/f64 reference.
//!
//! ## The oracle is transcribed, not imported, and that is forced
//!
//! The reference is `infer_vulkan::qwen4_hc` (`gated_residual` +
//! `inject_block_output`). `infer-vulkan` cannot be a dev-dependency of this
//! crate: `infer-vulkan/Cargo.toml` already depends on `vulkan-kernels`, so the
//! edge would close a cycle — and `crates/vulkan-kernels/Cargo.toml` is outside
//! this change's file scope in any case. The host path below is therefore
//! transcribed from that module line for line (same f64 accumulation, same
//! `1 + weight` gain, same `/ hc_count` INSIDE each nonlinearity). If the two
//! ever drift, this test is the one that is wrong.
//!
//! ## What a passing run actually establishes
//!
//! * `hn` — the grouped norm is grouped: `hc_count` independent statistics over
//!   one `hc_count * hidden` weight vector, not one broadcast norm.
//! * `x` — `qwen4_hc_mix` computes all `hc_count` mixing gates per hidden
//!   channel, applies the sigmoid, multiplies by `hn` and means across streams,
//!   with the bottleneck's `silu(u / hc_count)` folded into its read of `u_raw`.
//! * `h` — `qwen4_hc_combine` gets the `2*sigmoid(dot / hc_count)` injection
//!   weights right and scatters `inj[s] * y` onto the RAW residual.
//! * the dispatch count is FOUR, in one submit. That is the design constraint,
//!   not a side effect: this site occurs 97 times per token on a decode path
//!   that is dispatch-bound, so a fifth dispatch here is ~330us/token of host
//!   recording. A regression that splits a fusion still passes the numerics and
//!   must fail here.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, qwen4_hc_combine_dispatch, qwen4_hc_combine_params, qwen4_hc_mix_dispatch,
    qwen4_hc_mix_params, qwen36_router_gemv_dispatch, qwen36_router_gemv_params, record_dispatch,
    rms_norm_dispatch_rows, rms_norm_params_grouped,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Shape of one `Qwen4ExpTextGatedResidual`.
#[derive(Clone, Copy)]
struct Cfg {
    hidden: usize,
    hc_count: usize,
    hc_lowrank: usize,
    eps: f32,
}

impl Cfg {
    /// The on-box `qwen3.8-flash-next-nvfp4` checkpoint's values.
    const QWEN4_EXP: Self = Self {
        hidden: 2560,
        hc_count: 4,
        hc_lowrank: 320,
        eps: 1e-6,
    };

    const fn hc_hidden(&self) -> usize {
        self.hc_count * self.hidden
    }
}

/// Weights of one gated residual, HF row-major (`w[out * in_features + in]`).
struct Weights {
    /// `hc_norm.weight`, `[hc_hidden]`. The APPLIED gain is `1 + hc_norm`.
    hc_norm: Vec<f32>,
    /// `input_mix_weight_down.weight`, `[hc_lowrank, hc_hidden]`.
    mix_down: Vec<f32>,
    /// `input_mix_weight_up.weight`, `[hc_hidden, hc_lowrank]`.
    mix_up: Vec<f32>,
    /// `block_inject_weight.weight`, `[hc_count, hc_hidden]`.
    block_inject: Vec<f32>,
}

/// Deterministic xorshift PRNG so a failure reproduces exactly.
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

    /// Uniform in `[-scale, scale]`.
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

// ---------------------------------------------------------------------------
// Host oracle — transcribed from `infer_vulkan::qwen4_hc`.
// ---------------------------------------------------------------------------

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu64(x: f64) -> f64 {
    x * sigmoid64(x)
}

/// `Qwen4ExpTextRMSNorm` with a group size: the statistic is per group, the
/// gain is per channel, and the gain is `1 + weight` — NOT `weight`.
fn grouped_rmsnorm(x: &[f32], weight: &[f32], group_size: usize, eps: f32) -> Vec<f32> {
    let eps = f64::from(eps);
    let mut out = Vec::with_capacity(x.len());
    for (group, gains) in x
        .chunks_exact(group_size)
        .zip(weight.chunks_exact(group_size))
    {
        let sum_sq: f64 = group.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = (sum_sq / group_size as f64 + eps).sqrt().recip();
        out.extend(
            group
                .iter()
                .zip(gains)
                .map(|(&v, &w)| (f64::from(v) * scale * (1.0 + f64::from(w))) as f32),
        );
    }
    out
}

/// `y = W @ x` for a row-major `[out_features, in_features]` weight, f64
/// accumulation, f64 out so the caller applies `/ hc_count` and the
/// nonlinearity before rounding.
fn matvec(w: &[f32], in_features: usize, x: &[f32]) -> Vec<f64> {
    w.chunks_exact(in_features)
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        })
        .collect()
}

/// Everything the four dispatches are checked against.
struct HostSite {
    normed: Vec<f32>,
    block_input: Vec<f32>,
    /// The residual AFTER `h[s] += inj[s] * y`.
    hyper_out: Vec<f32>,
}

fn host_site(cfg: &Cfg, w: &Weights, hyper_state: &[f32], block_output: &[f32]) -> HostSite {
    let hc_hidden = cfg.hc_hidden();
    let hc_count = cfg.hc_count as f64;

    let normed = grouped_rmsnorm(hyper_state, &w.hc_norm, cfg.hidden, cfg.eps);

    // The divide is INSIDE the activation: `silu(z / hc_count)`.
    let lowrank: Vec<f32> = matvec(&w.mix_down, hc_hidden, &normed)
        .into_iter()
        .map(|d| silu64(d / hc_count) as f32)
        .collect();
    let mix_gate: Vec<f32> = matvec(&w.mix_up, cfg.hc_lowrank, &lowrank)
        .into_iter()
        .map(|z| sigmoid64(z) as f32)
        .collect();

    let mut block_input = Vec::with_capacity(cfg.hidden);
    for i in 0..cfg.hidden {
        let acc: f64 = (0..cfg.hc_count)
            .map(|s| {
                let k = s * cfg.hidden + i;
                f64::from(mix_gate[k]) * f64::from(normed[k])
            })
            .sum();
        block_input.push((acc / hc_count) as f32);
    }

    let injection: Vec<f32> = matvec(&w.block_inject, hc_hidden, &normed)
        .into_iter()
        .map(|z| (2.0 * sigmoid64(z / hc_count)) as f32)
        .collect();

    let mut hyper_out = hyper_state.to_vec();
    for (stream, &inj) in injection.iter().enumerate() {
        let inj = f64::from(inj);
        let base = stream * cfg.hidden;
        for (h, &y) in hyper_out[base..base + cfg.hidden]
            .iter_mut()
            .zip(block_output)
        {
            *h = (f64::from(*h) + inj * f64::from(y)) as f32;
        }
    }

    HostSite {
        normed,
        block_input,
        hyper_out,
    }
}

// ---------------------------------------------------------------------------
// Device plumbing.
// ---------------------------------------------------------------------------

/// A buffer the GPU reads and the host only writes: device-local + host-visible
/// on this APU, so there is no staging copy.
fn upload<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc_uma(ctx, bytes.len().max(4)).expect("alloc uma buffer");
    b.copy_from_host(&bytes).expect("upload buffer");
    b
}

/// A buffer the HOST READS. `alloc`/`alloc_uma` memory is write-combined here —
/// host reads out of it run ~0.10 GB/s — so anything read back belongs in
/// HOST_CACHED.
fn readable<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b =
        DeviceBuffer::alloc_host_cached(ctx, bytes.len().max(4)).expect("alloc host-cached buffer");
    b.copy_from_host(&bytes).expect("seed host-cached buffer");
    b
}

fn read_f32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).expect("read back buffer");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Largest `|got - want| / max(|want|, floor)` over a vector, and where it was.
fn max_rel_err(got: &[f32], want: &[f32], floor: f32) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "length mismatch in comparison");
    got.iter()
        .zip(want)
        .enumerate()
        .map(|(i, (&g, &w))| ((g - w).abs() / w.abs().max(floor), i))
        .fold((0.0f32, 0usize), |acc, e| if e.0 > acc.0 { e } else { acc })
}

struct DeviceSite {
    normed: Vec<f32>,
    block_input: Vec<f32>,
    hyper_out: Vec<f32>,
    dispatches: u64,
    submits: u64,
}

/// Record and run one whole gated-residual site.
///
/// Four `KernelCache`s rather than one because `KernelCache::get` hands back a
/// reference borrowed from `&mut self`: all four pipelines have to be live at
/// once to record them into a single command buffer, which one cache cannot do.
/// The decode path fetches and records a step at a time and needs only one.
fn run_site(
    ctx: &VulkanContext,
    cfg: &Cfg,
    w: &Weights,
    hyper_state: &[f32],
    block_output: &[f32],
) -> DeviceSite {
    let hc_hidden = cfg.hc_hidden();

    // THE `1 + weight` FOLD. `Qwen4ExpTextRMSNorm` scales by `(1.0 + weight)`;
    // the vendored `rms_norm.comp` applies the plain weight and this crate never
    // edits `vendor/`. So the bias is folded once, host-side, exactly as the
    // loader must do at model load.
    let norm_gain: Vec<f32> = w.hc_norm.iter().map(|&v| 1.0 + v).collect();

    let buf_h = readable(ctx, hyper_state);
    let buf_norm_gain = upload(ctx, &norm_gain);
    let buf_w_down = upload(ctx, &w.mix_down);
    let buf_w_up = upload(ctx, &w.mix_up);
    let buf_w_inject = upload(ctx, &w.block_inject);
    let buf_y = upload(ctx, block_output);
    let buf_hn = readable(ctx, &vec![0.0; hc_hidden]);
    let buf_u_raw = upload(ctx, &vec![0.0; cfg.hc_lowrank]);
    let buf_x = readable(ctx, &vec![0.0; cfg.hidden]);

    let push_norm =
        rms_norm_params_grouped(cfg.hidden as u32, cfg.hc_count as u32, cfg.eps).to_le_bytes();
    let push_down =
        qwen36_router_gemv_params(cfg.hc_lowrank as u32, hc_hidden as u32, false).to_le_bytes();
    let push_mix = qwen4_hc_mix_params(
        cfg.hidden as u32,
        cfg.hc_count as u32,
        cfg.hc_lowrank as u32,
    )
    .to_le_bytes();
    let push_combine =
        qwen4_hc_combine_params(cfg.hidden as u32, cfg.hc_count as u32).to_le_bytes();

    let mut cache_norm = KernelCache::new();
    let mut cache_down = KernelCache::new();
    let mut cache_mix = KernelCache::new();
    let mut cache_combine = KernelCache::new();

    let (pipe_norm, layout_norm) = cache_norm
        .get(
            ctx,
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            push_norm.len() as u32,
            3,
        )
        .expect("rms_norm pipeline");
    let (pipe_down, layout_down) = cache_down
        .get(
            ctx,
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            push_down.len() as u32,
            3,
        )
        .expect("router_gemv pipeline");
    let (pipe_mix, layout_mix) = cache_mix
        .get(
            ctx,
            Kernel::Qwen4HcMix,
            Kernel::Qwen4HcMix.specialization_u32(),
            push_mix.len() as u32,
            4,
        )
        .expect("qwen4_hc_mix pipeline");
    let (pipe_combine, layout_combine) = cache_combine
        .get(
            ctx,
            Kernel::Qwen4HcCombine,
            Kernel::Qwen4HcCombine.specialization_u32(),
            push_combine.len() as u32,
            4,
        )
        .expect("qwen4_hc_combine pipeline");

    // 1. grouped RMSNorm: A = raw residual, B = (1 + hc_norm), D = hn.
    let set_norm =
        DescriptorSet::storage_buffers(ctx, layout_norm, &[&buf_h, &buf_norm_gain, &buf_hn])
            .expect("bind rms_norm set");
    // 2. down-projection: x = hn, W = mix_down, y = u_raw (NOT silu'd — the
    //    activation is folded into the mix kernel's read of it).
    let set_down =
        DescriptorSet::storage_buffers(ctx, layout_down, &[&buf_hn, &buf_w_down, &buf_u_raw])
            .expect("bind router_gemv set");
    // 3. mix: hn, W_up, u_raw -> x (the block input).
    let set_mix =
        DescriptorSet::storage_buffers(ctx, layout_mix, &[&buf_hn, &buf_w_up, &buf_u_raw, &buf_x])
            .expect("bind mix set");
    // 4. combine: hn, W_inject, h (in place), y (the sublayer output).
    let set_combine = DescriptorSet::storage_buffers(
        ctx,
        layout_combine,
        &[&buf_hn, &buf_w_inject, &buf_h, &buf_y],
    )
    .expect("bind combine set");

    let d_norm = rms_norm_dispatch_rows(cfg.hc_count as u32);
    let d_down = qwen36_router_gemv_dispatch(cfg.hc_lowrank as u32);
    let d_mix = qwen4_hc_mix_dispatch(cfg.hidden as u32);
    let d_combine = qwen4_hc_combine_dispatch(cfg.hidden as u32);

    let mut recorder = CommandRecorder::new(ctx).expect("recorder");
    recorder.begin().expect("recorder begin");
    record_dispatch(
        &mut recorder,
        pipe_norm,
        &set_norm,
        &push_norm,
        [d_norm.x, d_norm.y, d_norm.z],
    );
    // Every step reads the previous step's writes, so each boundary needs a
    // compute->compute barrier — but only ONE submit for the whole site.
    recorder.barrier();
    record_dispatch(
        &mut recorder,
        pipe_down,
        &set_down,
        &push_down,
        [d_down.x, d_down.y, d_down.z],
    );
    recorder.barrier();
    record_dispatch(
        &mut recorder,
        pipe_mix,
        &set_mix,
        &push_mix,
        [d_mix.x, d_mix.y, d_mix.z],
    );
    recorder.barrier();
    record_dispatch(
        &mut recorder,
        pipe_combine,
        &set_combine,
        &push_combine,
        [d_combine.x, d_combine.y, d_combine.z],
    );
    recorder.submit_and_wait().expect("submit site");

    DeviceSite {
        normed: read_f32(&buf_hn, hc_hidden),
        block_input: read_f32(&buf_x, cfg.hidden),
        hyper_out: read_f32(&buf_h, hc_hidden),
        dispatches: recorder.dispatches_in_batch(),
        submits: recorder.submit_count(),
    }
}

/// Weights scaled so no pre-activation saturates: a saturated sigmoid hides
/// arithmetic errors behind its own flat tail, which would make this test pass
/// on a broken kernel. `1/sqrt(fan_in)` keeps every dot O(1).
fn random_weights(rng: &mut Rng, cfg: &Cfg) -> Weights {
    let hc_hidden = cfg.hc_hidden();
    let in_scale = 1.0 / (hc_hidden as f32).sqrt();
    Weights {
        // `hc_norm` is zero-initialised in the real checkpoint; a small spread
        // around 0 keeps the `1 + w` gain near 1 the way the model does.
        hc_norm: rng.vec(hc_hidden, 0.1),
        mix_down: rng.vec(cfg.hc_lowrank * hc_hidden, in_scale),
        mix_up: rng.vec(
            hc_hidden * cfg.hc_lowrank,
            1.0 / (cfg.hc_lowrank as f32).sqrt(),
        ),
        block_inject: rng.vec(cfg.hc_count * hc_hidden, in_scale),
    }
}

fn check_site(ctx: &VulkanContext, label: &str, cfg: &Cfg, rng: &mut Rng) -> f32 {
    let w = random_weights(rng, cfg);
    // Per-stream scales so the four RMSNorm groups carry DIFFERENT energy. With
    // identical streams (which is exactly how the residual is seeded, by tiling
    // the embedding `hc_count` times) a grouped norm and a broadcast norm agree
    // to the last bit, and dispatch 1 would be untested.
    let mut hyper_state = Vec::with_capacity(cfg.hc_hidden());
    for s in 0..cfg.hc_count {
        let scale = 0.25 * (1 + s) as f32;
        hyper_state.extend(rng.vec(cfg.hidden, scale));
    }
    let block_output = rng.vec(cfg.hidden, 1.0);

    let want = host_site(cfg, &w, &hyper_state, &block_output);
    let got = run_site(ctx, cfg, &w, &hyper_state, &block_output);

    assert_eq!(
        got.dispatches, 4,
        "{label}: the gated residual must be FOUR dispatches, recorded {}",
        got.dispatches
    );
    assert_eq!(
        got.submits, 1,
        "{label}: the whole site must be one submit, was {}",
        got.submits
    );

    // The device and the host agree to within a rounding of each other on the
    // residual, so "matches the oracle" alone would also be satisfied by a
    // combine kernel that never wrote anything IF the oracle's injection were
    // itself negligible. Require the injection to have actually moved the
    // residual, so a no-op phase 2 fails here rather than passing quietly.
    let moved = got
        .hyper_out
        .iter()
        .zip(&hyper_state)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 1e-3,
        "{label}: the residual never moved (max |h_out - h_in| = {moved:.3e}) — \
         the injection scatter did not run"
    );

    // Floors are ~the RMS of each vector, so a near-zero element cannot inflate
    // the relative error into a false failure while a real error still shows.
    let (e_norm, i_norm) = max_rel_err(&got.normed, &want.normed, 0.1);
    let (e_x, i_x) = max_rel_err(&got.block_input, &want.block_input, 0.05);
    let (e_h, i_h) = max_rel_err(&got.hyper_out, &want.hyper_out, 0.05);

    // The device sums in f32 with a tree reduction where the host sums serially
    // in f64, over contractions up to 10240 long. That is a few ULP per term;
    // 2e-4 is far above it and far below any structural error (a broadcast norm,
    // a missing `/ hc_count`, a `silu(z)/4` instead of `silu(z/4)` all move the
    // answer by O(1)).
    const TOL: f32 = 2e-4;
    assert!(
        e_norm < TOL,
        "{label}: grouped RMSNorm max rel err {e_norm:.3e} at {i_norm} (got {} vs want {})",
        got.normed[i_norm],
        want.normed[i_norm]
    );
    assert!(
        e_x < TOL,
        "{label}: block_input max rel err {e_x:.3e} at {i_x} (got {} vs want {})",
        got.block_input[i_x],
        want.block_input[i_x]
    );
    assert!(
        e_h < TOL,
        "{label}: injected residual max rel err {e_h:.3e} at {i_h} (got {} vs want {})",
        got.hyper_out[i_h],
        want.hyper_out[i_h]
    );

    eprintln!(
        "[{label}] PASS 4 dispatches / 1 submit — max rel err: hn {e_norm:.3e}, \
         block_input {e_x:.3e}, residual {e_h:.3e}"
    );
    e_norm.max(e_x).max(e_h)
}

#[test]
fn qwen4_hyper_connection_matches_host_oracle_in_four_dispatches() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping qwen4 hyper-connection proof");
            return;
        }
    };
    // A missing `.spv` is a box without the shader corpus, not a failure.
    if KernelCache::new()
        .get(
            &ctx,
            Kernel::Qwen4HcMix,
            Kernel::Qwen4HcMix.specialization_u32(),
            12,
            4,
        )
        .is_err()
    {
        eprintln!("qwen4_hc_mix .spv unavailable; skipping qwen4 hyper-connection proof");
        return;
    }
    eprintln!(
        "ARLE Vulkan qwen4 hyper-connection proof on: {}",
        ctx.device_name()
    );

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0.0f32;

    // The real shapes first: hidden 2560, hc_count 4, hc_lowrank 320.
    for trial in 0..3 {
        worst = worst.max(check_site(
            &ctx,
            &format!("qwen4_exp trial={trial}"),
            &Cfg::QWEN4_EXP,
            &mut rng,
        ));
    }

    // A ragged shape: hidden is not a multiple of the combine kernel's 256-wide
    // workgroup, hc_lowrank is under the mix kernel's 64 lanes (so most lanes
    // contribute nothing to the dot), and hc_count is not a power of two (so the
    // grouped norm's `fastmod` takes its slow branch). Every tail guard in both
    // shaders is live here and dead at the real shape.
    let ragged = Cfg {
        hidden: 100,
        hc_count: 3,
        hc_lowrank: 40,
        eps: 1e-6,
    };
    worst = worst.max(check_site(&ctx, "ragged", &ragged, &mut rng));

    eprintln!("qwen4 hyper-connection: worst max-rel-err across all shapes {worst:.3e}");
}
