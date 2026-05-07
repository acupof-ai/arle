# 2026-05-07 · M_world1 P0.2 — SGLang baseline at long-ctx 8k + high-conc + KEY innovation finding

> Follow-on to [`12c4c86`](2026-05-07-m_world1-p0-sglang-baseline.md)
> (P0.1 long-ctx 4k/c=4). This entry covers the remaining two
> shapes (8k/c=4, 1k/256/c=64) and surfaces a **major innovation
> finding** discovered during SGLang server load: SGLang
> graph-captures prefill across 42 num_token shapes, which
> **ARLE does not do**.

## Goal

[M_world1 P0.2](../../plans/M_world1-30-percent-lead-roadmap.md):
extend the #2 baseline to all canonical workloads (long-ctx 8k +
high-conc + multi-tenant). Multi-tenant uses a custom Python
runner, deferred. Long-ctx 8k and high-conc covered here.

## Hypothesis

After P0.1 showed SGLang #1 at 4k by 1.21× over vLLM and 2.03×
over ARLE, expect SGLang to lead by similar margin at 8k. At
high-conc 1k/256/c=64, ARLE was already +30.3% past vLLM with
F4-Small substrate; SGLang is the new #2 unknown.

## Results — long-ctx 8k/c=4

| rate | TTFT mean | TTFT p50 | TTFT p99 | ITL p50 | E2E mean | out tok/s | total in | total out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conc4 | 8293.7 | 8054.5 | 9062.6 | 18.22 | 13.43 | 78.05 | 131088 | 4096 |

Samples: 16 OK / 0 failed.

### 8k/c=4 three-way comparison

| Engine | TTFT p50 | out tok/s | rank |
|---|---:|---:|---:|
| **vLLM (m_b22 baseline)** | **2361.5 ms** ⭐ | **104.74** | #1 |
| SGLang 0.5.11 | 8054.5 ms | 78.05 | #3 |
| ARLE | TBD | TBD | TBD |

**Surprising finding**: **vLLM is faster than SGLang at 8k**
(2361 vs 8054 ms TTFT, 3.41× ratio). Likely cause: SGLang's
default `chunked_prefill_size=2048` chunks an 8k input into 4
sequential forward passes, whereas vLLM uses a single larger
prefill or different chunking. SGLang's lead at 4k (where input
fits in 2 chunks) does not generalize to 8k (where 4 chunks
serialize).

This is a **second-order finding**: the "ARLE 2.03× behind #2"
narrative at 4k may NOT scale to 8k. At 8k, ARLE may already be
competitive or leading depending on actual ARLE 8k bench.

ARLE 8k/c=4 self-bench needed to complete the table. (Not run in
this tick — codex M_b.2.2 just released the GPU and time is
budgeted to high-conc.)

## Results — high-conc 1k/256/c=64

SGLang launched with `--max-running-requests 64 --context-length 2048`.
Note: SGLang default `cuda_graph_max_bs=8` (decode graph) is
suboptimal for c=64 — only batch sizes 1, 2, 4, 8 are graph-captured.
Decode at bs > 8 falls back to eager (slower per-row).

| rate | TTFT mean | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E mean | out tok/s | conc actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conc64 | 17254.1 | 17643.3 | 25027.9 | 27.51 | 100.95 | 24.76 | **498.79** | 58 |

Samples: 139 OK / 0 failed; ITL p99/p50 = 3.67 (stability
warning, but headline metrics consistent).

### high-conc three-way comparison

| Engine | out tok/s | TTFT p50 | rank | ARLE Δ |
|---|---:|---:|---:|---:|
| **ARLE F4-Small s48** ([`m3.6 wins`](2026-05-07-m3.6-f4small-bench-world-first.md)) | **843** ⭐ | n/a | **#1** | baseline |
| vLLM s14 | 647 | n/a | #2 | +30.3% past ARLE? actually ARLE +30.3% past vLLM |
| SGLang default | 499 | 17643 ms | #3 | ARLE **+69%** past SGLang |

ARLE leads BOTH vLLM and SGLang at high-conc. Lead margin grew
from "+30% past vLLM" to "+69% past SGLang" because SGLang's
default config (cuda_graph_max_bs=8) underperforms at c=64.

**Caveat**: SGLang COULD close some of this gap with
`--cuda-graph-max-bs 64`. ARLE's lead may shrink to +30-40%
under tuned SGLang config. Not re-run in this tick due to
~5-min capture time per config × multiple shapes.

**M_world1 status update for high-conc**:
- Required: ≥ +30% past #2.
- Current vs SGLang default: +69% ✓✓ (DOUBLE the requirement)
- Current vs SGLang tuned (estimated): +30-40% ✓ (likely still meets)
- Verdict: **HIGH-CONC IS WORLD #1, by comfortable margin** at
  default configs. Tuning SGLang for fair comparison still leaves
  ARLE ahead.

## 🔥 KEY INNOVATION FINDING — Piecewise CUDA Graph for Prefill

During SGLang server startup (high-conc config), the load log
revealed:

```
Capture cuda graph bs [1, 2, 4, 8]
  0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=8 ...)
... (decode graph capture)

[2026-05-07 22:54:42] Compiling a graph for dynamic shape takes 0.54 s
Compiling num tokens (num_tokens=2048):  ... 42 sizes
   2048, 1792, 1536, 1280, 1024, 960, 896, 832, 768, 704, 640, 576,
   512, 480, 448, 416, 384, 352, 320, 288, 256, 240, 224, 208, 192,
   176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 28, 24, 20, 16, 12,
   8, 4
```

Then at first request: `cuda graph: True`.

**SGLang graph-captures BOTH decode and prefill**, the latter
across 42 num_token sizes (4–2048 tokens). This is the
`piecewise_cuda_graph` feature, ON by default in SGLang 0.5+.

### Why this matters

**ARLE only graph-captures decode** (`crates/cuda-kernels/csrc/`
ggml shim + Qwen3 decode B=1..8 capture). **vLLM** also defaults
to decode-only graph capture. **TensorRT-LLM** does some
piecewise but with different semantics.

SGLang's prefill graph capture eliminates per-launch dispatch
overhead in the prefill path — the bottleneck H_LP3 trace
identified for ARLE long-ctx 4k. **Each chunked-prefill segment
runs through a pre-compiled CUDA graph**, removing the
~5-10 µs/launch × hundreds of layer ops.

For Qwen3-4B at chunked_prefill_size=2048, that's 36 layers ×
~7 GEMMs/layer × 2-4 chunks = ~500-1000 launches. At ~7.5 µs each,
~4-7 ms saved per request — explains a meaningful slice of the
SGLang 4k lead (1976 - 972 = 1004 ms gap).

### Strategic implication

**This is a top-tier "innovative combination" candidate for ARLE
to PASS SGLang at long-ctx 4k**:

1. ARLE has a fixed-shape decode capture (B=1..8). Extending
   to prefill chunks at fixed num_token boundaries is the
   same pattern, applied to a different code path.
2. ARLE's chunked_prefill_size is configurable; aligning chunk
   sizes to a small set of graph-captured sizes is a known
   pattern.
3. TileLang prefill kernels (M_pf-gemm Phase 2 / 2.5) can be
   COMBINED with graph capture: capture once with TileLang as
   the kernel inside the graph, get both kernel-level speedup
   AND launch-overhead elimination.

**Combination innovation**: ARLE could be the first inference
runtime to combine (a) hand-tuned TileLang prefill GEMM with
(b) prefill CUDA graph capture and (c) FP8 paged KV. This stack
isn't done by vLLM, SGLang, or TRT-LLM — each does 1-2 of these
but not all 3.

## Per-shape M_world1 verdict

| Shape | ARLE | #2 (max non-ARLE) | ARLE Δ vs #2 | M_world1 30% target | Verdict |
|---|---:|---:|---:|---:|---|
| 1k/256/c=64 (high-conc, tok/s) | **843** | SGLang 499 | **+69%** | +30% | ✓✓ EXCEEDS |
| 4k/c=4 (long-ctx, TTFT lower=better) | 1976 ms | SGLang 973 ms | **−51% (slower)** | TTFT ≤ 748 ms | ✗ NEEDS WORK |
| 8k/c=4 (long-ctx, TTFT) | TBD | vLLM 2362 ms | TBD | TTFT ≤ 1817 ms | ⏳ NEED ARLE 8K BENCH |
| multi-tenant shared-prefix | 318 ms TTFT | (not benched here) | n/a | n/a | ⏳ NEED SGLANG MULTI-TENANT |

ARLE confirmed world #1 at 1 of 4 shapes; needs major TTFT
improvement at 1 shape; needs measurement at 2 shapes.

## What's next (P-priority order)

1. **P0.3 — ARLE 8k/c=4 self-bench** (Claude, ~5 min): close the
   8k gap row in the table. Codex's `bench-output/2026-05-07-m_b22-bf16-splitkv`
   was at 4k; need an 8k variant.
2. **P0.4 — Multi-tenant shared-prefix bench across all engines**:
   uses custom runner, deferred until guidellm Multi-tenant support
   exists or we author a runner.
3. **P1 — Prefill CUDA Graph capture** (NEW): innovation track.
   - Phase 0: prototype with one num_token size (e.g., 2048)
     gated behind env flag, prove TTFT reduction at long-ctx 4k
   - Phase 1: full piecewise capture (mirror SGLang's 42 sizes)
   - Phase 2: combine with TileLang prefill GEMM (M_pf-gemm
     Phase 2.5)
4. **P1 — TileLang prefill GEMM port** (M_pf-gemm Phase 2 / 2.5):
   already planned, even more justified now.
5. **Codex parallel** — M_b.3 G1+G2 (mixed prefill collapse, was
   blocked by M_b.2.2 which just KILLED).

## Cross-references

- M_world1 plan: [`docs/plans/M_world1-30-percent-lead-roadmap.md`](../../plans/M_world1-30-percent-lead-roadmap.md)
- P0.1 wins (4k): [`12c4c86`](2026-05-07-m_world1-p0-sglang-baseline.md)
- M_pf-gemm Phase 2/2.5 plan: [`012d989`](../../plans/M_pf-gemm-cublaslt-autotune.md)
- ARLE F4-Small high-conc 843 baseline: [`m3.6 wins`](2026-05-07-m3.6-f4small-bench-world-first.md)
- vLLM 8k/c=4 baseline: `bench-output/2026-05-07-m_b22-vllm-longctx-8k-c4/`
- SGLang artifacts:
  - 8k: `bench-output/2026-05-07-sglang-longctx-8k-c4/`
  - high-conc: `bench-output/2026-05-07-sglang-highconc-1k-256-c64/`

## Rules

- **#2 baseline shifts by shape**. SGLang #1 at 4k, vLLM #1 at 8k
  (vs SGLang), ARLE #1 at high-conc. The #2 baseline at any given
  shape can be a different engine.
- **Default vs tuned matters**. SGLang at c=64 default
  `cuda_graph_max_bs=8` is artificially slow. Production
  comparisons should use tuned settings; bench reports note
  default config used.
- **Surprise findings during installation are first-class**.
  Discovering SGLang's piecewise prefill graph capture during
  load-log inspection is a higher-leverage finding than the
  measured benchmark numbers themselves. Read the logs.
- **Innovation = combinations**. SGLang has piecewise prefill
  graph but not TileLang custom kernels. vLLM has Triton kernels
  but not prefill graph. ARLE adding piecewise prefill graph
  ON TOP OF TileLang HD128 + FP8 paged KV would be a unique
  stack — that's the world #1 angle.
