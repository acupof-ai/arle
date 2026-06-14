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
    Kernel, KernelCache, launch_cached, qwen36_router_topk_dispatch, qwen36_router_topk_params,
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
