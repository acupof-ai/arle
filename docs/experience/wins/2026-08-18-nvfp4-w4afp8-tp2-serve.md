# NVFP4→W4AFP8 load-time conversion serves DSv4-Flash-0731 at TP=2 — CUDA, 2026-08-18

> Status: Shipped

## Context

DeepSeek-V4-Flash-0731 ships routed MoE experts as NVFP4 (E2M1 float4 packed
in I8 + F8_E8M0 per-1×32-block scales). The SGLang W4A8 CUTLASS kernel
expects signed INT4 two's complement + BF16 per-1×128-block interleaved
scales. Converting at load time keeps weights at 4-bit (0.5 byte/elem),
enabling TP=2 on H20 (96 GB/GPU).

## What Worked

1. GPU load-time conversion kernel (`nvfp4_to_w4afp8.cu`): E2M1+E8M0 →
   INT4+BF16 per expert. Grid (N, K/128), 128 threads/block. Block-reduce
   amax, scale = amax/8.0, pack two's complement INT4.
2. Loader detection: NVFP4 detected via F8_E8M0 scale dtype. Conversion runs
   per-expert during weight load; results fused (w1+w3) and uploaded as
   W4AFP8. Forward path unchanged — `dsv4_moe_forward_w4afp8` dispatches on
   `w13_w4afp8.is_some()`.
3. CUTLASS 4.x switch: SGLang mixed-input extensions were written for
   CUTLASS 4.x but backported to 3.7.0, introducing TMA descriptor lifecycle
   bugs (-729 crash on first MoE forward). Switching to FlashMLA vendored
   CUTLASS 4.x fixed it.
4. W4AFP8 workspace right-sized 64MB→32MB (actual need <16MB + 17KB metadata
   at E=128).
5. Stale NCCL shared memory files (`/dev/shm/nccl-*`) from crashed runs cause
   silent SIGKILL on new serve launches. Clean before launch.

## Results

- Model serves at TP=2 on 2×H20 (96GB), 95.7GB/GPU used
- Math: 17×23=391, +19=410 — correct
- Chinese: coherent, well-structured
- Decode: ~37 tok/s
- Prefill: 2109 tok/s (1K), 3647 tok/s (4K)
- Bench: 5/5 cases status 200

The 64MB workspace build showed 36s TTFT on 1K prefill (memory pressure from
the oversized workspace); right-sizing to 32MB fixed it (0.48s TTFT).

## Rule

NVFP4 checkpoints can be served via W4AFP8 by converting at load time on GPU.
The conversion is a per-expert kernel that runs during weight load, adding
~90s to startup but keeping weights at 4-bit. CUTLASS 4.x is required for the
SGLang mixed-input kernel — the 3.7.0 backport has TMA descriptor lifecycle
bugs. Clean stale NCCL shm files before launching a new serve after a crash.
