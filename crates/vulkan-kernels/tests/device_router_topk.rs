//! On-device Qwen3.6 MoE router top-k correctness proof.
//!
//! Oracle-gates `qwen36_router_topk` (softmax-over-all → top-k by prob → renorm)
//! against a host reference transcribed line-for-line from
//! `infer-vulkan`'s `qwen36_topk_routes` (model_qwen36.rs). The device kernel may
//! replace the host routing only if it picks the SAME expert ids and matching
//! weights. IDs must match exactly; weights match within tolerance (GPU `exp`
//! vs host `exp` differ by a few ULP, which never flips a well-separated top-k).
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, launch_cached, qwen36_moe_weighted_accum_dispatch,
    qwen36_moe_weighted_accum_params, qwen36_router_gemv_dispatch, qwen36_router_gemv_params,
    qwen36_router_topk_dispatch, qwen36_router_topk_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

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
}

fn upload_f32<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc f32 buffer");
    b.copy_from_host(&bytes).expect("upload f32 buffer");
    b
}

fn zeroed<'a>(ctx: &'a VulkanContext, bytes: usize) -> DeviceBuffer<'a> {
    let mut b = DeviceBuffer::alloc(ctx, bytes.max(4)).expect("alloc buffer");
    b.copy_from_host(&vec![0u8; bytes.max(4)])
        .expect("zero buffer");
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

fn read_i32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<i32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).expect("read back i32 buffer");
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Host reference transcribed line-for-line from `qwen36_topk_routes`.
fn host_routes(logits: &[f32], top_k: usize, norm_topk_prob: bool) -> (Vec<i32>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    let mut scored: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    scored.sort_by(|(a_idx, a), (b_idx, b)| b.total_cmp(a).then_with(|| a_idx.cmp(b_idx)));
    scored.truncate(top_k.min(scored.len()));
    if norm_topk_prob {
        const F16_MIN: f32 = 6.103_516e-5;
        let denom = scored.iter().map(|(_, p)| *p).sum::<f32>().max(F16_MIN);
        for (_, p) in &mut scored {
            *p /= denom;
        }
    }
    let ids = scored.iter().map(|(e, _)| *e as i32).collect();
    let weights = scored.iter().map(|(_, w)| *w).collect();
    (ids, weights)
}

#[test]
fn qwen36_router_topk_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping router_topk oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan router_topk proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x5151_A0A0_F00D_1234);

    // The real qwen36 config (256 experts, top-8, renorm) plus smaller shapes and
    // the k==n edge. Logits scaled x4 so the top-k is well separated (no ties).
    let configs = [
        (256usize, 8usize, true),
        (256, 8, false),
        (8, 2, true),
        (4, 4, true),
        (64, 8, true),
    ];
    for &(n_expert, top_k, norm) in &configs {
        for trial in 0..4 {
            let logits: Vec<f32> = (0..n_expert).map(|_| rng.next_f32() * 4.0).collect();
            let (want_ids, want_weights) = host_routes(&logits, top_k, norm);

            let buf_logits = upload_f32(&ctx, &logits);
            let mut buf_ids = zeroed(&ctx, top_k * 4);
            let mut buf_weights = zeroed(&ctx, top_k * 4);
            let push = qwen36_router_topk_params(n_expert as u32, top_k as u32, norm).to_le_bytes();
            launch_cached(
                &mut cache,
                &ctx,
                Kernel::Qwen36RouterTopk,
                &[&buf_logits, &mut buf_ids, &mut buf_weights],
                qwen36_router_topk_dispatch(),
                &push,
                Kernel::Qwen36RouterTopk.specialization_u32(),
            )
            .expect("router_topk dispatch");

            let got_ids = read_i32(&buf_ids, top_k);
            let got_weights = read_f32(&buf_weights, top_k);
            let label = format!("router_topk n={n_expert} k={top_k} norm={norm} trial={trial}");

            assert_eq!(
                got_ids, want_ids,
                "{label}: expert ids mismatch (got {got_ids:?} vs want {want_ids:?})"
            );
            for (j, (&g, &w)) in got_weights.iter().zip(&want_weights).enumerate() {
                assert!(
                    (g - w).abs() < 1e-5 || (g - w).abs() / w.abs().max(1e-4) < 1e-4,
                    "{label}: weight[{j}] got {g} vs want {w}"
                );
            }
            // Weights must sum to ~1 when renormalized.
            if norm {
                let s: f32 = got_weights.iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "{label}: renorm weights sum {s} != 1"
                );
            }
            eprintln!("[{label}] PASS ids+weights");
        }
    }
}

/// Host reference for `qwen36_router_gemv`: `y[e] = Σ_c W[e,c]·x[c]`, transcribed
/// from `gemv_f32_host` (forward.rs), optional sigmoid (shared-expert gate).
fn host_router_gemv(x: &[f32], w: &[f32], n_out: usize, apply_sigmoid: bool) -> Vec<f32> {
    let hidden = x.len();
    (0..n_out)
        .map(|e| {
            let mut s = 0.0f32;
            for c in 0..hidden {
                s += w[e * hidden + c] * x[c];
            }
            if apply_sigmoid {
                1.0 / (1.0 + (-s).exp())
            } else {
                s
            }
        })
        .collect()
}

#[test]
fn qwen36_router_gemv_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping router_gemv oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan router_gemv proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // The real qwen36 router (256 experts × 2048 hidden), the shared-expert gate
    // (n_out=1, sigmoid), and a small shape.
    let configs = [
        (256usize, 2048usize, false),
        (1, 2048, true),
        (8, 64, false),
    ];
    for &(n_out, hidden, sigmoid) in &configs {
        let x: Vec<f32> = (0..hidden).map(|_| rng.next_f32()).collect();
        // Scale weights down so the dot magnitude stays moderate (real router
        // weights are small; keeps sigmoid off its saturated tails).
        let w: Vec<f32> = (0..n_out * hidden).map(|_| rng.next_f32() * 0.05).collect();
        let want = host_router_gemv(&x, &w, n_out, sigmoid);

        let buf_x = upload_f32(&ctx, &x);
        let buf_w = upload_f32(&ctx, &w);
        let mut buf_y = zeroed(&ctx, n_out * 4);
        let push = qwen36_router_gemv_params(n_out as u32, hidden as u32, sigmoid).to_le_bytes();
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::Qwen36RouterGemv,
            &[&buf_x, &buf_w, &mut buf_y],
            qwen36_router_gemv_dispatch(n_out as u32),
            &push,
            Kernel::Qwen36RouterGemv.specialization_u32(),
        )
        .expect("router_gemv dispatch");

        let got = read_f32(&buf_y, n_out);
        let label = format!("router_gemv n_out={n_out} hidden={hidden} sigmoid={sigmoid}");
        for (e, (&g, &wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - wv).abs() < 1e-4 || (g - wv).abs() / wv.abs().max(1e-4) < 1e-4,
                "{label}: y[{e}] got {g} vs want {wv}"
            );
        }
        eprintln!("[{label}] PASS ({} rows)", n_out);
    }
}

#[test]
fn qwen36_moe_weighted_accum_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping weighted_accum oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan weighted_accum proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    // Routed (top-8, init from 0) and shared (count 1, accumulate into existing).
    let configs = [(2048usize, 8usize, true), (2048, 1, false), (512, 8, true)];
    for &(hidden, count, init) in &configs {
        let src: Vec<f32> = (0..count * hidden).map(|_| rng.next_f32()).collect();
        let weights: Vec<f32> = (0..count).map(|_| rng.next_f32().abs()).collect();
        let acc0: Vec<f32> = (0..hidden).map(|_| rng.next_f32()).collect();

        // Host reference.
        let mut want = vec![0.0f32; hidden];
        for (i, w) in want.iter_mut().enumerate() {
            let mut s = if init { 0.0 } else { acc0[i] };
            for e in 0..count {
                s += weights[e] * src[e * hidden + i];
            }
            *w = s;
        }

        let buf_src = upload_f32(&ctx, &src);
        let buf_w = upload_f32(&ctx, &weights);
        let mut buf_acc = upload_f32(&ctx, &acc0);
        let push =
            qwen36_moe_weighted_accum_params(hidden as u32, count as u32, init).to_le_bytes();
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::Qwen36MoeWeightedAccum,
            &[&buf_src, &buf_w, &mut buf_acc],
            qwen36_moe_weighted_accum_dispatch(hidden as u32),
            &push,
            Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
        )
        .expect("weighted_accum dispatch");

        let got = read_f32(&buf_acc, hidden);
        let label = format!("weighted_accum hidden={hidden} count={count} init={init}");
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4 || (g - w).abs() / w.abs().max(1e-4) < 1e-4,
                "{label}: acc[{i}] got {g} vs want {w}"
            );
        }
        eprintln!("[{label}] PASS");
    }
}
