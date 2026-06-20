# Autograd GPU CE speedup — design (infra-first unlock for 27B OPD)

**Goal (ckl, 2026-06-21):** make the 27B-dense autograd CE step fast on GPU so
rubric-OPD (and OPD generally) can run large curves. Today a 27B CE step is
**~4.5 min** (capped curve: round-0 CE 1/33→7/33 in 27 min) — the bottleneck that
forced tiny capped runs. **Design first, then implement** (ckl's call).

## Hard evidence (the only measured numbers — everything else is hypothesis)
- 27B-dense rubric-OPD CE: **~4.5 min/step** (wall-clock, from the curve log).
- suffix-detach win (2026-06-20): 35B-A3B at seq-2048, **forward = 488 s**, backward
  327–1196 s (autograd, single "GPU").
- During CE: arle main thread **99.9% CPU** (host-bound orchestration).

488 s for a 35B-A3B (3B active) forward is **~500× slower than GPU compute** (~1 s
of actual FP32 GEMM) and **~10⁴× slower than kernel-launch latency** (~ms for a few
thousand kernels). So it is neither pure compute nor pure launch latency — **the real
breakdown must be measured**, not inferred.

## §0 gate — MEASURE before fixing
The scoping workflow (read-only code analysis = hypothesis) found: ops are
device-resident GPU (cuBLAS + custom kernels), no per-op host round-trip in the
normal path, profiling off. That does NOT explain 488 s. Before implementing any
lever, **profile one 27B CE step** to attribute the time:
- `ARLE_OPD_STEP_PROFILE` forward/backward seconds (already wired).
- nsys (or CUDA-event timers around the GEMM vs elementwise vs the per-op host
  dispatch) on a single CE step — which kernels/host-sections dominate.
- Specifically resolve: (i) is the frozen-base GEMM the custom `fp8_block_scaled_
  matmul_bt_f32` kernel, and is THAT kernel slow (naive vs tensor-core)? (ii) are the
  trainable/LoRA + elementwise ops FP32 `sgemm` on H20 (FP32-weak)? (iii) host
  per-op dispatch / sync time; (iv) gradient-checkpoint recompute share.

The lever ranking below is the **hypothesis order**; the measurement re-ranks it.

## Candidate levers (impact × tractability; pick by measurement)
- **L1 — frozen-base FP8 GEMM kernel quality.** The 27B's bulk is the FP8 frozen base,
  whose autograd forward uses the custom `fp8_block_scaled_matmul_bt_f32` kernel
  (backend_cuda.rs:888/936). If that kernel is naive (not tensor-core / not DeepGEMM-
  class), it dominates. Fix: route the autograd frozen-base GEMM through an optimized
  FP8 GEMM (DeepGEMM / TileLang, already vendored for infer-cuda). **Likely highest
  impact if (i) confirms.**
- **L2 — bf16 compute for trainable/activation GEMMs.** If activations + LoRA matmuls
  go through FP32 `sgemm` (backend_cuda.rs:595), H20 FP32 throughput is a fraction of
  bf16; a bf16 GEMM path exists (`matmul_bt_device_f32_bf16`). Switch compute to bf16
  (f32 accumulate). Tractable; impact depends on the FP32-GEMM share.
- **L3 — batched CE writeback.** Train B accepted rollouts in ONE forward+backward
  (batch dim; `forward_batch_indices` already batches) instead of B sequential steps →
  amortize per-op host dispatch over B. Tractable, high-value if host-dispatch is a
  large share; needs length padding/bucketing for variable rollouts.
- **L4 — op fusion.** Fuse rmsnorm/rope/silu·mul elementwise chains into fewer kernels
  (cuts launch + dispatch count). Moderate.
- **L5 — CUDA-graph capture.** Capture the tape's kernel stream once, replay
  (cuStreamBeginCapture/instantiate/launch) → eliminate per-op host launch. Constraints
  to handle: variable rollout len (re-capture per shape or pad), data-dependent control
  flow (checkpoint recompute, conditional readback), allocation address stability (fixed
  scratch arena). Only relevant if the measurement shows launch-count is the bottleneck
  after L1–L4.

## The 门 (non-negotiable)
- **Opt-in + byte-identical default**: the validated OPD path (4B math 0.518→0.792,
  the suffix-detach byte-identity) must not change without a flag. Each lever gated.
- **Correctness gate**: needle/loss parity — a CE step's loss under the fast path must
  match the slow path within the MoE/FP non-determinism floor (per the KV parity gate
  pattern), on a fixed (prompt, completion).
- **License-or-kill on wall-clock**: each lever lands only with a paired A/B on the CE
  step-time (`ARLE_OPD_STEP_PROFILE` forward/backward seconds, same shape, two env
  flips) showing the speedup. No "should be faster" — measured Δ.

## Plan
1. **Measure** one 27B CE step (forward/backward seconds + a kernel/host breakdown) →
   attribute the 4.5 min. (No code; a profiled GPU run.)
2. Implement the measured-#1 lever behind a flag; A/B the CE step-time + loss parity.
3. Re-measure; stack the next lever if the step is still the curve bottleneck.
4. Once a CE step is ~seconds: run the BIG rubric-OPD curve (and the autograd unlock
   serves all OPD). Then (b) Flash judge multi-GPU + Mode B correction.

## Status
Design only. Next: the measurement run (§0 gate) to pick the lever — no autograd code
until the 4.5 min is attributed.
