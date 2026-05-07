# 2026-05-07 · M_world1 P0.1 — SGLang IS #2 (and TTFT gap is 2.03×, bigger than vLLM 1.65×)

## Goal

[M_world1 P0.1](../../plans/M_world1-30-percent-lead-roadmap.md):
establish actual #2 competitor at long-ctx 4k/c=4 by benchmarking
SGLang. The vLLM-only baseline assumed vLLM was #2; before
committing to "+30% past #2" engineering, verify.

## Hypothesis

SGLang v0.5+ (zero-overhead scheduler claim) at long-ctx 4k/c=4
on RTX 4070 Ti SUPER is **at least as fast as vLLM**, and could
be #1. ARLE's actual gap to #1 might be larger than the 1.65×
vLLM-only number suggested.

## Command

Server:

```bash
PATH=/home/ckl/sglang-venv/bin:$PATH \
LD_LIBRARY_PATH=/home/ckl/sglang-venv/lib_extra:$LD_LIBRARY_PATH \
NVCC_CCBIN=/usr/bin/g++-14 \
TORCH_CUDA_ARCH_LIST=8.9 \
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 \
TMPDIR=/var/tmp \
  /home/ckl/sglang-venv/bin/python -m sglang.launch_server \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8001 --max-running-requests 8 --context-length 12288
```

GuideLLM:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh sglang-longctx-4k-c4 \
  --target http://localhost:8001 \
  --model /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --concurrencies 4 --max-seconds 60 --warmup 10
```

## Environment

- GPU: RTX 4070 Ti SUPER 16 GiB, SM 8.9
- CUDA: 13.2, gcc-14 host (gcc-16 incompatible with nvcc)
- Model: Qwen3-4B BF16 weights, BF16 KV cache (SGLang default)
- SGLang: 0.5.11 with sgl-kernel SM89 (cu130 official wheel)
  + flashinfer 0.6.8.post1
  + flash-attn-4 4.0.0b12
- Toolchain workarounds:
  - `libnuma.so.1` extracted from Arch `numactl-2.0.19` package
    into venv-local `lib_extra/` (no system pkg available)
  - Flashinfer JIT compile fails on gcc-16; needs g++-14 via
    `NVCC_CCBIN=/usr/bin/g++-14`
- KV cache: 22156 tokens, 3.04 GB total (1.52 GB K + 1.52 GB V)
- mem_fraction_static = 0.752, cuda_graph_max_bs=8 (decode)
- attention_backend = flashinfer
- Raw artifacts: `bench-output/2026-05-07-sglang-longctx-4k-c4/`

## Results — long-ctx 4k/c=4

| rate | TTFT mean | TTFT p50 | TTFT p99 | ITL p50 | E2E mean | out tok/s | total in | total out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conc4 | **1117.4** | **972.9** | 1828.4 | 19.44 | 6.29 | **164.3** | 147492 | 9216 |

Samples: 75 OK / 0 failed.

## Three-way comparison

| Engine | TTFT p50 | TTFT mean | out tok/s | E2E mean | rank |
|---|---:|---:|---:|---:|---:|
| **SGLang 0.5.11** | **972.9 ms** ⭐ | 1117.4 ms | **164.3** ⭐ | 6.29 s | #1 |
| vLLM (s8) | 1177 ms | n/a | 159.1 | n/a | #2 |
| ARLE post-`3e0ed5a` | 2005.9 ms | 2009.6 ms | 152.49 | 6.96 s | #3 |
| **ARLE pre-fuse `2026-05-07-longctx-4k-c4`** | 1976.4 ms | n/a | 153.83 | n/a | (baseline) |

ARLE vs each engine at long-ctx 4k/c=4:
- vs SGLang: TTFT **2.03× slower** (972.9 → 1976.4 ms), out tok/s **−6.4%**
- vs vLLM: TTFT 1.68× slower (1177 → 1976.4 ms), out tok/s −3.4%

SGLang TTFT p50 of 972.9 ms is the new **#2 baseline** for the
M_world1 30%-lead target. To lead #2 by 30%:
- ARLE TTFT p50 must be ≤ 972.9 / 1.30 = **748 ms** (37.6% faster than SGLang)
- ARLE out tok/s must be ≥ 164.3 × 1.30 = **213.6 tok/s** (+38.8% above ARLE current)

## What this rules out / rules in

**Ruled out**:
- "vLLM is #2" — SGLang is comfortably ahead of vLLM on TTFT
  (172% improvement: 972.9 vs 1177 ms) AND out tok/s (164.3 vs 159.1).
- "Gap is closeable via 1-2 fusion fixes" — too much absolute
  delta. Even M_pf-fuse Phase 1 QKV (predicted -14% TTFT) would
  only put ARLE at ~1700 ms, still 1.75× slower than SGLang.

**Strongly suggests** (now with #2 baseline data):
- **M_pf-gemm Phase 2 (TileLang prefill GEMM port)** — JUSTIFIED.
  cuBLAS is leaving large performance on the table; SGLang's
  flashinfer prefill kernel ≈ 2× faster on this shape.
- **M_pf-gemm Phase 2.5 (Hybrid TileLang+cuBLAS dispatch)** —
  JUSTIFIED. Smaller scope than full Phase 2, targets the
  cuBLAS large-N blind spot evidence from M_pf-fuse Phase 0
  KILL ([`3e0ed5a`](2026-05-07-m_pf-fuse-phase0-gateup-killed.md)).
- **Investigate what SGLang does**:
  - flashinfer prefill kernels (CUDA + Triton hybrid)
  - chunked_prefill_size=2048 (vs ARLE chunk policy?)
  - cuda_graph_max_bs=8 for decode (ARLE has this)
  - **Does SGLang graph-capture prefill?** — If so, that's a
    novel target (M_world1 innovation candidate).

**Still in play**:
- M_b.2.2 split-KV BF16 (codex active) — improves decode at
  long-ctx, complementary to prefill TTFT work.
- M_b.3 G1+G2 mixed prep collapse — scheduler-side improvement.
- ARLE's existing **wins** at high-conc 1k/256/c=64 (+30.3%
  vs vLLM) and multi-tenant shared-prefix (+80% vs vLLM TTFT)
  remain. SGLang baseline at those shapes is still pending —
  could close those leads.

## What's next (P-priority order)

1. **P0.2 — SGLang baseline at remaining shapes** (Claude, ~30 min):
   - high-conc 1k/256/c=64 (verify ARLE +30.3% lead vs vLLM holds vs SGLang)
   - long-ctx 8k/c=4
   - multi-tenant shared-prefix (custom Python runner)
2. **P0.3 — TRT-LLM bench**: deferred until SGLang results at all
   shapes (per M_world1 plan).
3. **P1 — Decide kernel-impl Phase 2 vs Phase 2.5**: based on full
   #2 baseline table, pick scope. Likely Phase 2.5 first (lower
   risk, smaller LOC) then Phase 2 if ROI plateaus.
4. **P1 (parallel codex)** — M_b.2.2 split-KV BF16 (already active).

## Cross-references

- M_world1 plan: [`docs/plans/M_world1-30-percent-lead-roadmap.md`](../../plans/M_world1-30-percent-lead-roadmap.md)
- M_pf-fuse KILL: [`3e0ed5a`](2026-05-07-m_pf-fuse-phase0-gateup-killed.md)
- M_pf-gemm Phase 2 / 2.5 plan: [`012d989`](../../plans/M_pf-gemm-cublaslt-autotune.md)
- ARLE long-ctx 4k baseline: `bench-output/2026-05-07-longctx-4k-c4/`
- vLLM long-ctx 4k baseline: `bench-output/2026-05-07-vllm-longctx-4k-c4/`
- SGLang artifacts: `bench-output/2026-05-07-sglang-longctx-4k-c4/`
- vLLM longctx 4k wins: [`d13d2b3`](2026-05-07-m_b22-vllm-longctx-baseline.md)

## Rule

- **Phase 0 baseline measurement matters**. Before this bench, the
  assumption was "vLLM is #2 → 1.65× gap to close". The reality
  is "SGLang is #2 → 2.03× gap to close". Engineering scope and
  priority both shift. Without the measurement, M_pf-fuse-style
  cheap fusion experiments are scoped to the wrong gap target.
- **#2 baseline must come from real benchmarks, not vendor
  claims.** SGLang installs are non-trivial on consumer hardware
  (libnuma + g++-14 + sgl-kernel SM89 wheels) — but the data is
  worth the install effort.
- **Rank by TTFT *and* throughput.** SGLang wins on both. If they
  diverged (e.g. SGLang lower TTFT but lower tok/s), the #2
  decision would need clarification.
