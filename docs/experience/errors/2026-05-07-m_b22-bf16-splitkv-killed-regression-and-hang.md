# M_b.2.2 BF16 Split-KV Killed — Regression and Hang

## Context

M_b.2.2 tested the opt-in TileLang HD128 BF16 split-KV decode substrate from
`85d3751`. The path is enabled by `INFER_TILELANG_BF16_SPLIT_KV=1` and was
licensed against longctx 4k/c=4 on Qwen3-4B before any runtime default cutover.

Baseline references:

- ARLE P0' default Split baseline: TTFT p50 `1976.4 ms`, ITL p50 `19.4 ms`,
  out tok/s `153.8`.
- vLLM longctx 4k/c=4 baseline:
  `docs/experience/wins/2026-05-07-m_b22-vllm-longctx-baseline.md`.
- SGLang P0.1 baseline: TTFT p50 `972.9 ms`, making the real gap `2.03x`.

## Failure

The split-KV opt-in run regressed decode-side metrics instead of improving
them:

| run | TTFT p50 | ITL p50 | out tok/s | delta |
|---|---:|---:|---:|---|
| P0' default Split | 1976.4 ms | 19.4 ms | 153.8 | baseline |
| BF16 split-KV opt-in | ~2005 ms | 25.53 ms | 124.9 | ITL +31.6%, out tok/s -18.8% |

The opt-in path also triggered a severe validation failure: the e2e test was
interrupted after hanging for 33m+. Treat this as a correctness/runtime bug, not
only a perf miss. The dirty graph-capture experiment that attempted to force the
split branch was discarded and is not shipped.

## Root Cause

The immediate root cause is that the BF16 split-KV path is not licensed for the
canonical 4k/c=4 shape:

- The two-phase partial+merge path adds launch and merge overhead.
- At this shape, the extra split parallelism does not beat the existing runtime
baseline.
- The CUDA graph interaction around split/no-split branch capture is fragile;
short warmup metadata and long-context eligibility can diverge, and the opt-in
path did not survive e2e validation.

The broader M_world1 P0.1 evidence also changed the strategic priority:
SGLang's BF16 prefill GEMMs are cuBLAS like ARLE, and the observed TTFT lead is
now attributed primarily to SGLang's piecewise CUDA graph capture for prefill,
not to a dense GEMM kernel advantage.

## Fix

- Kill M_b.2.2 as a runtime path-cut.
- Keep the existing substrate default-off from `85d3751`; do not promote
  `INFER_TILELANG_BF16_SPLIT_KV=1`.
- Discard the dirty `infer/src/model/qwen3/batch_decode.rs` graph-branch patch.
- Move next CUDA decode/prefill substrate work to M_b.3 G1 segment-aware grid
  and the new P0 prefill CUDA graph-capture direction.

## Rule

An opt-in kernel path is not enough for promotion. It must clear both gates:

1. e2e + greedy consistency without hangs.
2. Same-shape GuideLLM license bench that beats the current production
   baseline.

If either gate fails, record the result, keep the path default-off or remove the
route, and move to the next hypothesis.
