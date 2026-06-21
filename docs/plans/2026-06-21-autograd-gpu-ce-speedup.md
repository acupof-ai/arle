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
- **L3 — batched CE writeback.** DONE + MEASURED: `rubric_writeback_ce_step_batched`
  amortizes the host dispatch — **batch-4 = 2.7×** (400 s/4 vs 270 s/1), confirming the
  overhead-bound diagnosis. BUT it **OOMs at B>1 on long prompts**: the `[B, seq, vocab]`
  logits with **vocab=248320** is huge (a batch hitting a ~2800-tok math prompt →
  `[4,2866,248320]` = 11.4 GB device upload → OOM; the engine-offload frees to a pool the
  autograd allocator can't fully reuse, ~4 GB device headroom). batch-1 fits (~2.85 GB).
  → **L3b (the real fix): completion-only logits** — run `lm_head` only on the completion
  positions (forward to hidden via `forward_batch_hidden_indices`, slice completion-pos
  hidden, then lm_head), so CE device memory is `[Σ completion_tok, vocab]` (~3 GB at
  B=4) — independent of prompt length × B. Then B>1 batching is safe. OR token-budget
  micro-batch grouping (group by `count×maxlen ≤ budget`). The current curve runs at
  batch-1 (proven-fit) to deliver a result; L3b makes batching usable.
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

## MEASURED 2026-06-21 (§0 gate complete — overturns L1–L5)
Profiled one 27B-dense CE step (`ARLE_OPD_BACKWARD_PROFILE=1`, B=1, suffix-detach 48, grad-ckpt on):
- **Step total 441 s = forward 277 s + backward 164 s** (timestamps phase-C→phase-D minus backward).
- **Backward 164 s = 99.98% `op=Checkpoint` (count=16)** — i.e. 100% gradient-checkpoint recompute.
  `lm_head` MatmulBT backward = **0.013%** → **L3b/lm_head saves ≈ nothing on backward**; GEMM-quality
  (L1/L2) never appears → **deprioritized**.
- Inside the 16 checkpoint recomputes, **12 are linear-attention (GatedDeltaNet) layers**, each ~6.1 s:
  `fwd_recompute` 3.5 s (57.8%) + **`host_materialize` 2.05 s (33.5%, runs on HOST CPU)** + scan 0.4 s.
- Externally: GPU bursty (1–3 s bursts, 3–4 s host-busy gaps), main thread 70–97% CPU — the LA layers
  run the **host scan fallback**, not the device kernel.

**Root cause (not in the L1–L5 list): the device LA dispatch bailed on a hardcoded head-count guard.**
`cuda_linear_attention_{forward,backward}_device` (backend_cuda.rs:3477/3937) returned `Ok(None)`
(→ host scan) unless `num_value_heads == 32`. Qwen3.6-27B GatedDeltaNet has **48** value heads
(35B-A3B has 32). `layer_types` = 3 linear : 1 full → **48 of 64 layers** were on the host scan,
explaining both the 277 s forward and the 67 s host backward.

**Fix (landed, gated by parity test):** drop the `num_value_heads != 32` term from both guards — the
kernels are head-count-generic (per-head grid blocks, dim-sized shared mem; only `key_dim/value_dim==128`
are baked in, which the 27B satisfies). New `cuda_linear_attention_qwen36_27b_chunked_grad_matches_cpu`
parity test (48 value heads) is the correctness gate.

## Status
§0 gate complete; root cause = LA device-kernel head-count guard. Guard relaxed + 48-head parity test
added. Next: pod parity test (correctness) → rebuild arle → re-profile the CE step (expect forward+backward
to collapse as 48 LA layers move host→device) → relaunch the clean larger-n capability curve.
