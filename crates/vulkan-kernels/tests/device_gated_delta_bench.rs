//! Throughput bench for `qwen35_gated_delta_net` at the PRODUCTION prefill
//! shape, with the traffic decomposition that says whether the kernel is where
//! it should be.
//!
//! Why this exists: after the coopmat GEMM landed, the per-op profile put
//! `lin_gdr` at ~34% of a prefill chunk — second only to the GEMM — and the
//! recurrence is token-serial by construction (one workgroup per value head,
//! stepping tokens in order). Before rewriting it into the chunkwise matrix
//! form, this bench answers the only question that licenses that work: is the
//! kernel slow, or is it already at its roofline?
//!
//! The shape is not a microbenchmark shape — it is read off the on-box GGUF
//! (`qwen35.ssm.*`): 48 value heads, 16 key heads, key_dim = state_size = 128,
//! val_dim = inner_size / n_value_heads = 6144 / 48 = 128.
//!
//! ## What this bench has already settled
//!
//! **The kernel is at its roofline; micro-optimizing it is dead.** `val_dim ==
//! local_size_x == 128`, so every thread owns exactly one value column and all
//! 128 threads per workgroup redundantly (a) run the `q_sumsq`/`k_sumsq`
//! reduction over all 128 `j` and (b) re-load the same `q[j]`/`k[j]` inside both
//! inner loops. That redundancy is **56% of addressed bytes** — an obvious
//! target, and a mirage: measured addressed bandwidth is ~610 GB/s, **238% of
//! the 256 GB/s LPDDR5X peak**, i.e. those lines are cache-resident and free.
//! Eliminating them buys nothing.
//!
//! Two further hypotheses died here, each worth not re-testing:
//! - *"the bench is flattered by MALL residency"* — rebuilt to walk all 48
//!   layer slices of a production-sized 150 MB arena (`alloc_uma`, as the
//!   forward does). No change: 11.19 ms vs 11.70 ms at T=256. The state stays
//!   resident **within** a dispatch because all T tokens re-touch the same 3 MB,
//!   which is intrinsic to the algorithm, not an artifact.
//! - *"production is 4.3x slower than this bench, so something is broken"* —
//!   that gap was the **Armoury Crate power mode**, not the code. Under Silent
//!   the profile showed 50.89 ms/dispatch at T=256 against this bench's 12 ms;
//!   under Performance production shows 3.70 ms at T=64 against this bench's
//!   ~3.0 ms. Sustained load costs only ~7% here (see `SUSTAIN_SECS`), so the
//!   bench was never the liar — the box was in a different clock state.
//!
//! What is left is irreducible **in the recurrent formulation**: 2 reads + 2
//! writes of the `[key_dim, val_dim]` state per token per head, 3.22 GB per
//! dispatch at T=256, sustained at ~270-290 GB/s. The only lever that moves
//! that number is the chunkwise matrix form, which touches the state once per
//! CHUNK instead of once per token. This bench is the license-or-kill gate for
//! that rewrite: it must show state GB dropping by ~the chunk size, not just
//! ms going down.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_kernels::{
    Dispatch, Kernel, KernelCache, qwen35_gated_delta_net_dispatch, qwen35_gated_delta_net_params,
    record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Seconds of continuous GPU load before the reported pass, so the clock has
/// settled into the same state a multi-second prefill chunk puts it in.
const SUSTAIN_SECS: f64 = 6.0;

/// Linear (gated-delta) layers in Qwen3.8-27B — the number of state slices in
/// the production arena, and the number of dispatches one prefill chunk issues.
const N_LINEAR_LAYERS: usize = 48;

/// Deterministic xorshift PRNG so a regression reproduces bit-for-bit.
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
}

fn upload_f32<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc f32 buffer");
    b.copy_from_host(&bytes).expect("upload f32 buffer");
    b
}

/// Time one dispatch, averaged over one pass across ALL `n_layers` state
/// slices, with a full warm pass discarded first.
///
/// Rotating the slice is the whole point. An earlier version of this bench
/// hammered a single 3 MB state buffer `ITERS` times and reported 619 GB/s of
/// addressed traffic — 242% of the 256 GB/s LPDDR5X peak — i.e. it was
/// measuring MALL residency, not the kernel. Production never sees that: the
/// 48 linear layers hold a 150 MB state arena, each slice is touched once per
/// chunk, and a ~16 GB weight sweep runs between consecutive touches. Walking
/// every slice once per pass reproduces the cold-ish regime the profile saw.
#[allow(clippy::too_many_arguments)]
fn time_dispatch_over_layers<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    kernel: Kernel,
    ro: &[&DeviceBuffer<'a>],
    state: &DeviceBuffer<'a>,
    out: &DeviceBuffer<'a>,
    state_stride: u64,
    n_layers: usize,
    dispatch: Dispatch,
    push: &[u8],
    spec: &[(u32, u32)],
) -> f64 {
    let (pipeline, layout) = cache
        .get(ctx, kernel, spec, push.len() as u32, 7)
        .expect("build pipeline");

    // One descriptor set per layer slice, built once and reused across passes
    // so set-creation never lands inside the timed region.
    let sets: Vec<DescriptorSet<'_>> = (0..n_layers)
        .map(|layer| {
            let binds = [
                (ro[0], 0u64, ro[0].len() as u64),
                (ro[1], 0, ro[1].len() as u64),
                (ro[2], 0, ro[2].len() as u64),
                (ro[3], 0, ro[3].len() as u64),
                (ro[4], 0, ro[4].len() as u64),
                (state, state_stride * layer as u64, state_stride),
                (out, 0, out.len() as u64),
            ];
            DescriptorSet::storage_buffers_ranged(ctx, layout, &binds).expect("bind layer set")
        })
        .collect();

    let run = || {
        let mut recorder = CommandRecorder::new(ctx).expect("recorder");
        recorder.begin().expect("begin");
        for set in &sets {
            record_dispatch(
                &mut recorder,
                pipeline,
                set,
                push,
                [dispatch.x, dispatch.y, dispatch.z],
            );
            // The recurrence carries `state` forward; overlapping dispatches
            // would race and would also hide the serial dependency under test.
            recorder.barrier();
        }
        let t0 = Instant::now();
        recorder.submit_and_wait().expect("submit");
        t0.elapsed().as_secs_f64() / n_layers as f64
    };
    // Sustained: keep going until the GPU has been busy for `SUSTAIN_SECS`, and
    // report the LAST pass. A single 5 ms burst runs at boost clock; a real
    // prefill chunk keeps this box (Armoury Crate Silent) under load for
    // seconds, and on an APU that is the difference between two clock states,
    // not a rounding error. Reporting first-pass numbers here would compare a
    // boost-clock microbenchmark against a throttled production profile.
    let mut per_dispatch = run();
    let t_start = Instant::now();
    let mut passes = 1usize;
    while t_start.elapsed().as_secs_f64() < SUSTAIN_SECS {
        per_dispatch = run();
        passes += 1;
    }
    eprintln!("                       (sustained {passes} passes over {SUSTAIN_SECS:.0}s)");
    per_dispatch
}

#[test]
fn gated_delta_throughput_at_prefill_shape() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping gated-delta bench");
            return;
        }
    };
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // Read off the on-box GGUF (`qwen35.ssm.*`) for Qwen3.8-27B-Q4_K_M.
    let (nk, nv, kd, vd) = (16usize, 48usize, 128usize, 128usize);
    let v_dim_total = nv * vd;
    let qkv_stride = 2 * nk * kd + v_dim_total;

    for &seq_len in &[128usize, 256, 384] {
        let qkv: Vec<f32> = (0..seq_len * qkv_stride).map(|_| rng.next_f32()).collect();
        let b_proj: Vec<f32> = (0..seq_len * nv).map(|_| rng.next_f32()).collect();
        let a_proj: Vec<f32> = (0..seq_len * nv).map(|_| rng.next_f32()).collect();
        let dt_bias: Vec<f32> = (0..nv).map(|_| rng.next_f32()).collect();
        // `ssm_a` is stored already negated (A = -exp(A_log)); a positive value
        // here would make the decay explode and the timing meaningless.
        let a_log: Vec<f32> = (0..nv).map(|_| -rng.next_f32().abs() - 0.5).collect();
        let state0: Vec<f32> = (0..nv * kd * vd).map(|_| rng.next_f32()).collect();

        let buf_qkv = upload_f32(&ctx, &qkv);
        let buf_b = upload_f32(&ctx, &b_proj);
        let buf_a = upload_f32(&ctx, &a_proj);
        let buf_dt = upload_f32(&ctx, &dt_bias);
        let buf_alog = upload_f32(&ctx, &a_log);
        // Production-sized arena (48 layers), allocated the way the forward
        // does it (`alloc_uma` = DEVICE_LOCAL|HOST_VISIBLE) rather than plain
        // `alloc`, so the memory type matches too.
        let state_stride = (nv * kd * vd * 4) as u64;
        let mut buf_state = DeviceBuffer::alloc_uma(&ctx, state_stride as usize * N_LINEAR_LAYERS)
            .expect("alloc gdr state arena");
        {
            let one: Vec<u8> = state0.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut all = Vec::with_capacity(one.len() * N_LINEAR_LAYERS);
            for _ in 0..N_LINEAR_LAYERS {
                all.extend_from_slice(&one);
            }
            buf_state.copy_from_host(&all).expect("upload state arena");
        }
        let buf_out = upload_f32(&ctx, &vec![0.0f32; seq_len * v_dim_total]);

        let push = qwen35_gated_delta_net_params(
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            seq_len as u32,
        )
        .to_le_bytes();
        let secs = time_dispatch_over_layers(
            &ctx,
            &mut cache,
            Kernel::Qwen35GatedDeltaNet,
            &[&buf_qkv, &buf_b, &buf_a, &buf_dt, &buf_alog],
            &buf_state,
            &buf_out,
            state_stride,
            N_LINEAR_LAYERS,
            qwen35_gated_delta_net_dispatch(nv as u32),
            &push,
            Kernel::Qwen35GatedDeltaNet.specialization_u32(),
        );

        // Addressed bytes, split so a change can be attributed rather than
        // just observed. `state` is the irreducible term: 2 reads + 2 writes of
        // the [key_dim, val_dim] state per token per head. The other two are
        // pure per-thread redundancy — same values, all 128 threads.
        let state_b = seq_len * nv * kd * vd * 4 * 4;
        let qk_b = seq_len * nv * vd * (kd * 3) * 4;
        let scal_b = seq_len * nv * vd * (2 * kd) * 4;
        let total_b = state_b + qk_b + scal_b;
        let gbps = |b: usize| b as f64 / 1e9 / secs;

        eprintln!(
            "[gated_delta T={seq_len:>3}] {:>7.2} ms  |  addressed {:>5.2} GB @ {:>6.1} GB/s \
             ({:>3.0}% of 256)  |  irreducible state {:>5.2} GB @ {:>6.1} GB/s  |  \
             redundant q/k+scalars {:>4.0}% of bytes",
            secs * 1e3,
            total_b as f64 / 1e9,
            gbps(total_b),
            100.0 * gbps(total_b) / 256.0,
            state_b as f64 / 1e9,
            gbps(state_b),
            100.0 * (qk_b + scal_b) as f64 / total_b as f64,
        );

        // Per-token cost must be flat in `seq_len`: the recurrence is serial in
        // tokens and does identical work per token. If this ever stops holding,
        // the shape is no longer what the profile measured and the numbers
        // above describe a different kernel than the one in prefill.
        let per_token_us = secs * 1e6 / seq_len as f64;
        eprintln!("                       {per_token_us:>7.2} us/token");
    }
}
