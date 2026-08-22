# sm_120 FP8 MoE first-light + baseline — CUDA (RTX PRO 6000), 2026-07-22

> Status: Shipped (baseline anchor for the sm_120 FP8 peak-GEMM work; G1 verified real-hardware)

## Goal

Anchor the sm_120 (Blackwell RTX PRO 6000) champion row for `Qwen3.6-35B-A3B-FP8`,
and verify the G1 DeepGEMM SM-gate fix on real hardware. Load-bearing metric:
**cold-prefill TTFT** (3 k-token prompt) — the FP8 peak-GEMM kernels (§6 Cut-2/G2)
must beat it.

## Hypothesis

None (clean baseline, no treatment). Records where the FP8 fallback path stands
before any peak kernel lands. Secondary: confirm G1 (`quant_linear.rs:174`
`major==9` gate) routes sm_120 to the portable path without a
`CUDA_ERROR_NOT_SUPPORTED` abort.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --model qwen36moe --prompts-jsonl bench-prompts-64.jsonl \
  --concurrency-grid 1,4,8,16 --seconds-per-concurrency 120 \
  --max-tokens 256 --seed 20260416 --output bench-output/baseline/bench
```

- Baseline: origin/main `9dbcb54`, binary sha256 `6c588981…`, kernel bundle `058202352a…`, features `cuda` (no nccl, world=1). **No treatment** — single-arm baseline.
- Prompt tokens: p50 **3013** / ~13.4 k chars, 64 unique docs.
- Completion tokens: 256 target (c1 256 / c4 248 / c8 219 avg).
- Trials: n = 3 / 6 / 8 / 0 (small — TTFT 85–175 s vs a 120 s window; inherent to the fallback's slowness, not a harness fault. TTFT/prefill is the load-bearing metric and is unambiguous).

## Environment

- Host / GPU: Colab, NVIDIA **RTX PRO 6000 Blackwell Server Edition, sm_120**, 97,887 MiB, driver 580.82.07, 48-core.
- CUDA: 12.8 (V12.8.93); tilelang 0.1.12 (`/usr/bin/python3`).
- Model / dtype: Qwen3.6-35B-A3B-FP8 (~35 GB), **FP8 on FALLBACKS** — dense `DeepGEMM SM-gated OFF on sm_120 … dequant→BF16 GEMM / scalar GEMV`; MoE `DeepGEMM disabled … hand grouped kernels` (native preflight `CUDA_ERROR_NOT_SUPPORTED`); whole-step decode graph off.
- TP/EP/slots/KV: world=1, num_slots 256, **BF16 KV** 118,351 pages @ page_size 16 (1.89 M max tok, 38.8 GB), mem_fraction_static 0.9.

## Results

| concurrency | arm | completed | errors | req/s | TTFT p50/p99 ms | ITL p50/p99 ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | 3 | 0 | 0.017 | **84,634 / 85,724** | 11.1 / 11.2 |
| 4 | baseline | 6 | 0 | 0.023 | 119,293 / 172,340 | ~0 / 20.1 |
| 8 | baseline | 8 | 0 | 0.045 | 175,404 / 175,738 | ~0 / 19.0 |
| 16 | baseline | 0 | 16 | 0.000 | n/a | n/a |

Correctness gate PASSED: coherent non-repeating output, `finish_reason=length`,
`correctness_failed=0`, `error=0` at every valid point.

Raw artifacts: VM `~/bench-output/baseline/{bench.json,bench.csv}` (643 KB per-request
detail) + serve log. (Re-fetch into repo on the next VM cycle if a raw A/B needs it.)

## Problems

- **c=16 collapsed** (0/16 complete) — prefill starvation, no request finished 256 tok
  in 120 s. Per spec §5 incompletes are not throughput → c=16 is the saturation/collapse
  point, no valid sample.
- **ITL p50 ≈ 0 at c4/c8** = bursty batched SSE delivery; true inter-step latency is p99
  (~17–20 ms).
- Minor prefix-cache contamination: 1 warmup prompt ∈ the 64 docs, so ≤1 req/point hits
  the cached prefix (the c1 low-TTFT 0.19 s sample). Does not move the cold p50/p99.

## Learnings

**PASS (baseline anchored) + G1 verified real-hardware.**
- **Prefill is the bottleneck; decode is already fine.** Cold prefill **~85 s / 3013 tok
  = ~35 tok/s** on the scalar/dequant FP8 fallback (no tensor-core FP8 GEMM). Decode ITL
  ~11 ms = ~90 tok/s, healthy. **This isolates the §6 target to the PREFILL large-M GEMM**
  (dense proj on dequant→BF16 + MoE grouped on hand-grouped), NOT decode.
- **Corrects the §6 FLOP hypothesis.** A back-of-envelope said "decode → G2 (MoE grouped)
  dominates"; the measurement says decode is fine and PREFILL is the disaster. At M=3013
  the dense projections are large-M on the worst fallback → **Cut-2 (dense block-scaled)
  is both the de-risk AND the first prefill win**; G2 (MoE grouped block-scaled) follows.
  Build order = win order.
- **G1 confirmed** (`quant_linear.rs:176` log verbatim on sm_120): routes to the portable
  path, no abort. On Hopper (major 9) byte-identical.
- **Next wall:** confirm bundled CUTLASS ≥ 3.8 has the sm_120a block-scaled collectives
  (§6 Cut-2 gate), + an nsys prefill breakdown (dense-proj vs MoE-grouped vs attention
  share at M=3013) to split Cut-2 vs G2 effort. Then implement Cut-2.
