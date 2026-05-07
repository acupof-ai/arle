# 2026-05-07 · Split-plan tax IS the prefill bottleneck — 10× per-token slowdown

## Goal

Disambiguate the long-context TTFT gap between two hypotheses
identified in
[`67f9bcb`](2026-05-07-prefill-batching-confirmed.md):

1. Multi-chunk serialization (4 chunks of one request processed
   sequentially)
2. Mixed-step (split) tax (steps that mix prefill + decode are
   slower than pure prefill)

Method: bench longctx 4k/c=4 (only 2 chunks per request — half of
8k case) and compare to longctx 8k/c=4. Linear scaling = chunk
serialization dominates. Sub-linear = split-tax dominates.

## Results

| Workload | TTFT mdn | ITL mdn | out tok/s | server log step breakdowns |
|---|---:|---:|---:|---|
| longctx 4k/c=4 | **3403 ms** | 20.1 ms | 122.5 | split steps 1263 ms each, batch=4 |
| longctx 8k/c=4 | 4961 ms | 23.9 ms | 92.2 | split steps similar (per `2026-05-07-m3.7-b12-longctx-8k`) |
| Ratio 8k/4k | **1.46×** | 1.19× | 0.75× | (sub-linear in chunk count) |
| **vLLM 8k/c=4** | 2367 ms | 26.7 ms | 105.6 | (control) |
| ARLE 8k vs vLLM 8k | 2.10× slower | -10% (faster) | -14% | TTFT gap is the issue |

**TTFT scales sub-linearly with chunk count (1.46× for 2× chunks)**
→ multi-chunk serialization is NOT the dominant factor. Split-tax IS.

## Server log step-breakdown decomposition

c=4/4k bench (`/tmp/longctx4k.log`):
```
14:54:54  batch=1  plan=prefill prefill=563ms       # only req 0 admitted
14:55:04  batch=4  plan=split   prefill=1263ms      # all 4 in split, 9s gap
14:55:12  batch=4  plan=split   prefill=1264ms      # repeats every 8.5s
14:55:21  batch=4  plan=split   prefill=1265ms
... [6 such super-steps total during the bench]
```

Per split step at batch=4:
- prefill phase = 1263 ms
- 4 requests × 2048 tokens prefill chunk = 8192 tokens
- **per-token = 1263 / 8192 = 154 µs/token**

Compare to pure-prefill batched (from
[`67f9bcb`](2026-05-07-prefill-batching-confirmed.md)):
- batch=8 × 2048 tokens = 16384 tokens
- 252 ms total → **15.4 µs/token**

**Split is 10× slower per-token than pure prefill at the same
underlying GPU work shape.**

## Why split is 10× slower

Hypothesis: split-step combines prefill + decode kernels. The
decode rows (already-prefilling-complete requests' next-token
generation) require their full attention+gemm chain, which
launches kernels NOT in the same CUDA-graph as prefill. The two
kernel chains compete for SMs, and the ARLE step machinery
launches them sequentially within the step, paying the launch
overhead twice.

Confirming evidence in same log:
```
plan=split admission=58us decode=990us emit=0us prefill=1263402us total=1264450us batch=4
                                       ^^^^^^         ^^^^^^^^^^^
                                       decode time   prefill time
```

decode part = 990 µs (negligible compared to prefill's 1.26 s).
The cost is the prefill kernel running with mixed metadata.

vLLM does NOT have this problem because:
- vLLM uses continuous batching: prefill chunks are SCHEDULED
  alongside decode rows, but the underlying attention kernel
  treats them uniformly (same shape, same launch).
- ARLE's `step_mixed_launch` retains a separate prefill+decode
  kernel split — historically efficient but not at high-conc
  long-prompt mixed workloads.

## Implications

| Track | Pre-experiment | Post-experiment |
|---|---|---|
| M3.8 cross-request batching | proposed | **NOT NEEDED** (already works) |
| M3.8 multi-chunk batching across requests | considered | **NOT THE BOTTLENECK** (4k vs 8k sub-linear) |
| **NEW: M3.9 — eliminate split-plan tax** | n/a | **HIGHEST PRIORITY** |
| M_b.2 (FP8 prefill TileLang) | medium | LOWER (kernel is fast at 15 µs/tok in pure mode) |
| F4-Big | optional | optional |

## Proposed M3.9 (rough sketch)

Two angles to attack the 10× split tax:

### A) Pure-prefill priority scheduling
Stay in `plan=prefill` mode until prefill queue empties OR
admission allows decode rows to dominate. ARLE today seems to
greedily mix as soon as ANY decode is ready. By delaying split
entry, all chunks across all c=N requests can prefill at the
high-throughput rate (15 µs/tok) before any decode tax is paid.

Cost: TTFT for the FIRST request might increase slightly (delays
its decode start until other prefills complete).
Gain: TTFT for all c=N requests is ~10× faster on the prefill
phase, at the cost of 1-step decode latency offset.

### B) Unify prefill+decode kernel call (true continuous batch)
The fundamental fix: make the kernel receive a uniform
"sequences with varying QO lengths" tensor. Decode rows have
qlen=1, prefill chunks have qlen=2048. The HD128 prefill TileLang
kernel ALREADY supports this (qlen as a per-row parameter via
`q_indptr`). The dispatch needs to use the prefill kernel for
mixed batches instead of the current split.

Cost: Larger refactor (~200-400 LOC), changes the scheduler/
model interface.
Gain: Step time becomes proportional to `total_tokens` not
`max(decode_step_time, prefill_step_time)`, eliminating the
10× tax.

## Bench Status

- Bench artifact: `bench-output/2026-05-07-longctx-4k-c4/`
  (gitignored).
- Server log: `/tmp/longctx4k.log`.
- Direct comparison vs F4-Small longctx 8k/c=4 (`c63c31c`) and
  prefill-batching-confirmed (`67f9bcb`).

## Rule

- **Sub-linear scaling in a sweep across the suspected variable
  is evidence the suspected variable is NOT the bottleneck**.
  TTFT only 1.46× at 2× chunks falsifies multi-chunk
  serialization as primary cause.
- **Step-breakdown logs are 50% of the diagnostic toolkit**.
  Server-side logs are deterministic and complete; they should
  be the first stop for scheduler/runtime questions, before nsys
  (which has been unreliable at low-conc workloads).
- **Mixed-step tax (split-plan 10×) is a structural ARLE-vs-vLLM
  difference**. Closing this gap is the highest-leverage prefill
  work left after F4-Small landed.
