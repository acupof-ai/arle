# W4AFP8 GEMV Killed item covers DSv4 MoE only; dense NVFP4 M=1 GEMV untested — CUDA, 2026-09-09

> Status: Confirmed (docs/code archaeology, no GPU used); perf implication pending-remote

## Context

B=1 decode on Qwen3.8-27B-NVFP4 measures 84.5 tok/s (`docs/baselines.md:292`).
tileRL reports 92.4 tok/s B=1 on the same model and GPU class (H20), with a
dedicated scalar fp4 GEMV at M=1; their tensor-core mma8 path measured 2.2x
slower at M=1 (39.9 vs 87 tok/s) (`tileRL/docs/roadmap.md:20`,
`tileRL/docs/experience/wins/2026-08-28-mma8-decode-gemm.md:30`, read-only
reference). A candidate explanation for the 84.5 vs 92.4 gap is a kernel-
structure difference at M=1. The roadmap Killed list appeared to foreclose
this lever: "W4AFP8 GEMV restructuring — memory-bound at M=1, custom kernel
perf-neutral" (`docs/plans/2026-08-24-roadmap.md:56`).

## Root Cause

The Killed item's evidence base is scoped to DSv4 MoE grouped GEMV:

- Its sole citation (`docs/experience/wins/2026-08-22-w4afp8-custom-decode-kernel.md`)
  is DSv4-Flash-0731, W4AFP8, kernel `w4afp8_grouped_swiglu_decode_kernel`,
  grid `(N/8, max_count/8, num_experts)`, TP=2/4. The companion win
  (`2026-08-21-w4afp8-gemv-decode-lane.md`) is the same model and structure.
  W4AFP8 is the DSv4 MoE quant format. Dense Qwen3.8 uses NVFP4 (E2M1 nibbles,
  per-128-block FP8 scales, one F32 per-tensor scale), a different format with
  a different kernel.
- The perf-neutral result has a MoE-specific explanation: ~36 MB/layer of
  expert weights loaded once per step; H20 bandwidth floors the load time
  regardless of compute structure.

The dense NVFP4 production path at M=1 is a tensor-core GEMM:

- `crates/infer-cuda/src/ops/quant_linear.rs:308` routes `Fp4E2M1Group` to
  `fp4::run`; `crates/infer-cuda/src/ops/quant_linear_fp4.rs:272` sends M=1 to
  `marlin_fp4_gemm_raw`, the vendored SGLang Marlin kernel using Ampere
  `mma.sync.m16n8k16` (`crates/cuda-kernels/csrc/gemm/marlin_gemm.cu:14`). The
  DeepGEMM arm requires prefill shape (`m >= QWEN_FP4_DEEPGEMM_MIN_M`).
- A dense NVFP4 GEMV kernel exists in the tree:
  `fp4_e2m1_group_gemv_batch_kernel`
  (`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu:1193`), with FFI bindings
  (`crates/cuda-kernels/src/ffi/gemm.rs:422`) and a kernel test
  (`gemm_tests.rs:343`). It has no production caller: a grep across `crates/`
  finds only the binding and the test. The dense dispatch has been
  Marlin-only since the 2026-08-20 quant-linear consolidation (`b7432e52a`).
- No wins/errors entry A/Bs Marlin against this GEMV at M=1 for dense NVFP4.
  The 2026-08-19 issue-bound finding
  (`errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md`) varied
  occupancy on the Marlin kernel; it did not test the GEMV.

One doc sentence amplified the overbroad reading: `docs/baselines.md:317`
states in the present tense that FP8 and NVFP4 "run the same hand-written
warp-per-row scalar GEMV at M=1". That passage originates from 2026-08-19
(`ea301015a`) and predates the Marlin-only consolidation; the same section's
2026-08-23 Marlin tiebreaker win reflects the current path.

## Fix

- The roadmap Killed item is amended to state its scope explicitly (DSv4 MoE
  grouped GEMV, W4AFP8).
- The dense NVFP4 M=1 structure question is recorded as open. The experiment
  vehicle already exists: wire `fp4_e2m1_group_gemv_batch_kernel` into the
  dense dispatch behind a matched A/B. No kernel was written for this entry.
- The stale baselines.md:317 sentence is flagged here for a separate doc
  correction.

Caveat on the candidate explanation: tileRL's 2.2x is their mma8 kernel
against their own GEMV on their stack; our Marlin kernel is a different
tensor-core implementation, and the 84.5/92.4 comparison crosses stacks
(tileRL's 92.4 also includes their split-KV occupancy win). The 2026-08-19
analysis characterizes our Marlin decode as issue-bound on the per-value
group-scale multiply — a cost a scalar GEMV structure removes — so the lever
is plausible. It is unmeasured.

## Bench (pending-remote)

Gate for wiring the existing dense GEMV into the M=1 dispatch:

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model Qwen3.8-27B \
  --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1 \
  --requests-per-concurrency 16 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900
```

- Baseline: `marlin_fp4_gemm` at M=1 (current production). Treatment:
  `fp4_e2m1_group_gemv_batch_kernel` at M=1.
- Same shell, >=3 trials per arm, needle ladder x3 before timing.
- Record ncu issue efficiency / warp active cycles on `marlin_fp4_gemm` at
  M=1 to re-confirm the issue-bound characterization on the current kernel.
- H20 (sm_90, 78 SMs), TP=1, `--kv-cache-dtype fp8`.

## Rule

A Killed item names the exact kernel, shape, quant format, and workload its
evidence killed. A perf-neutral result on one structure (MoE grouped GEMV,
W4AFP8) does not transfer to another structure (dense GEMV, NVFP4): the
second is untested, and an untested lever is open, not killed.
