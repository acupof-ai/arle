# TTFT vs prompt size — infer vs sglang on L4 Qwen3-4B (single-request)

## Context

Follow-up from `2026-04-17-sglang-p99-parity-qwen3-4b.md`. That win showed
a ~420ms TTFT-p99 gap at steady-state under load; this run isolates the
**single-request prefill cost** across prompt sizes to localise the gap.

- Hardware: NVIDIA L4 24GB, CUDA 12.8
- Model: Qwen/Qwen3-4B bf16
- Both servers: 10 slots, 5120 max seq, `--mem-fraction-static 0.88`
- Probe: `/tmp/ttft_probe.py` — streaming `/v1/completions`, first-byte latency,
  **unique prompt per run** (seeded random token substitution to defeat
  prefix cache), 5 runs per point, max_tokens=16
- Serial bench: kill infer, then start sglang. Same GPU.

## Data — median first-byte TTFT (5 runs, no-prefix-hit)

| prompt tokens | infer (ms) | sglang (ms) | Δ abs (ms) | Δ % |
|---|---|---|---|---|
| 128  | **63**  | 100 | -37 | **-37% (infer faster)** |
| 512  | **102** | 145 | -43 | **-30%** |
| 1024 | **190** | 210 | -20 | -10% |
| 2048 | 380 | **350** | +30 | +8.6% (infer slower) |
| 4096 | 797 | **697** | +100 | **+14.3% (infer slower)** |

### Per-token prefill cost (excluding 128-token fixed overhead)

| prompt | infer ms/tok | sglang ms/tok |
|---|---|---|
| 512  | 0.102 | 0.117 |
| 1024 | 0.142 | 0.121 |
| 2048 | 0.164 | 0.130 |
| 4096 | 0.186 | 0.150 |

## Findings

### 1. Fixed per-request overhead — infer wins by ~37ms

At 128 tokens the GPU prefill work is negligible, so the TTFT floor is
pure CPU/framework overhead:

- **infer: 63ms**  (HTTP parse + tokenizer + scheduler enqueue +
  forward_prefill setup + first-token sample + SSE emit)
- **sglang: 100ms** (equivalent stack, Python/Triton-backed)

**infer's Rust-native path is ~37ms cheaper at admission.** Likely due to no
Python GIL, no torch dispatch overhead per kernel, a tighter HTTP handler.

### 2. Per-token prefill cost — sglang scales better past ~1500 tokens

The crossover is around **n ≈ 1500-2000 tokens**. Below: infer wins by the
fixed-overhead advantage. Above: sglang's attention kernel is better.

Per-token cost ratio (infer / sglang):
- 512: **0.87** (infer faster)
- 1024: 1.17
- 2048: 1.26
- 4096: **1.24** (infer 24% slower per-token)

Not a linear-in-n divergence — the ratio settles ~1.24-1.26 from 2048 onward,
suggesting a constant-factor kernel efficiency gap, not an algorithmic
complexity difference. We use the same FlashInfer backend; sglang's wrapper
is just tighter.

### 3. Chunking is NOT the cause of the steady-state TTFT p99 gap

Ran the same 4096-token probe with `--decode-prefill-cap 4096` (disables
sub-chunking when decode is active). Median TTFT unchanged: 800ms vs 820ms
default. So the `decode_active_prefill_cap: 512` policy is not what's
eating 400ms in the guidellm p99.

### 4. What IS the guidellm p99 400ms gap, then?

Our single-request probe is 820ms (infer) vs 700ms (sglang) — a 120ms delta.
guidellm's p99 at 0.135 r/s is 1234ms (infer) vs 819ms (sglang) — a 415ms delta.

The extra 300ms on infer at steady-state rate appears only when **decode is
concurrently active** AND **multiple prefill chunks** interleave with decode
steps. Hypothesis: our `has_decode` scheduler branch pessimises prefill
throughput, spending extra time on the readback/emit path per-chunk.

Specifically, `execution.rs:110` measures `prefill_us` per step, and
`step_breakdown` logs show ~250ms total prefill time per 4096-token request
when isolated — so the extra 300ms in guidellm p99 is NOT from prefill
kernels themselves but from **step interleaving overhead** when prefill and
decode coexist.

This is consistent with the deferred `docs/plans/scheduler-gpu-cpu-overlap.md`
plan, which targets exactly this case.

## Ranked surpass opportunities

By projected ROI, lowest-effort first:

1. **Short-prompt workloads already win** — no action, just market the agent
   use-case where chat/tool-call prompts are <1024 tokens. Infer is
   already 10-37% faster than sglang at admission.

2. **GPU/CPU overlap for mixed prefill+decode steps** (~300ms of the 420ms
   guidellm p99 gap). Deferred plan exists at
   `docs/plans/scheduler-gpu-cpu-overlap.md`. **Highest ROI for p99 work.**

3. **FlashInfer prefill wrapper** (~100ms of single-request gap at 4096 tok).
   sglang cache's the wrapper `plan()` across calls; ours rebuilds per-call.
   Audit `crates/cuda-kernels/csrc/attention/flashinfer_prefill*.cu`
   bindings for a reusable plan cache.

4. **Prefill kernel selection for mid-range (1024-4096)** — crossover at
   n≈1500 suggests we could pick the better kernel variant for each range.
   FlashInfer has `BatchPrefillWithRaggedKVCache` (no paging overhead) vs
   `BatchPrefillWithPagedKVCache`; the former is faster for single-request
   full-prompt prefill.

5. **Qwen3.5 specialisation** — the hybrid linear+full attention model gives
   asymptotically faster scaling (linear layers are O(n) not O(n²)). Our
   adaptation here is the true "surpass" path for long-context workloads,
   but first we need to confirm sglang's Qwen3.5 support in 0.5.10.

## Artefacts

- Probe script: `/tmp/ttft_probe.py`
- Raw step breakdowns: `/tmp/infer-multi.log`, `/tmp/sglang-multi.log`

## Rule

**TTFT is two-humped: a fixed CPU/framework cost (wins for infer at short
prompts) and a per-token attention cost (wins for sglang at long prompts).**
Report both humps in future parity docs — single-number comparisons hide
which audience segment each engine serves best.
