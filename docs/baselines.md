# Rolling performance baselines

> Status: Active — champion row + step budget per fingerprint. Best numbers
> only; verdicts, rejected arms and analysis live in the linked entries.

Screening compares a new run against the champion row — no second arm.

1. **Effect > ~10%** (2× the measured drift band): verdict valid, update the
   champion, archive the binary.
2. **Inside the ±3% drift band**: never kill on ambiguity. Escalate to a
   same-shell A/B against the archived champion (≥3 trials/arm, median + range).
3. **Fingerprint change re-anchors**: model, TP/EP, GPU set, serve flags, slot
   line, dataset, driver/CUDA. Re-measure before comparing.
4. **Anchor audit** every ~5 accepted updates and before any default flip: one
   A/B against the oldest archived binary bounds accumulated drift.
5. **One workload**: the multi-turn long-agent dataset at the TraceLab medians,
   cold and warm turns reported separately.

**Stated deviation: rows run 32K, not the spec's 119K median.** Dense KV is
64 KB/token, so 119K×c16 needs 122 GB against ~69 GB free after weights.
A 119K row is a new anchor, not a re-measure.

```
python3 scripts/gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8
```

---

## Qwen3.6 · 1×H20 · single-GPU · eager — LONG-AGENT ANCHOR

### CHAMPION (DSpark) — `51985031d` (2026-07-30) · `arle-mk`

Features on: batched draft · replay · snapshot · capture · markov+confidence
head driving the goodput budget. Serve adds `--spec-type dspark
--mtp-draft-model Qwen3.6-27B-DFlash --dspark-block-size 6`; `--spec-max-batch`
is the shipped default 16.

A spec row carries `tok/row` (committed tokens per verify row; plain decode
= 1.0) and `burst`, never ITL p50 — a spec step emits `k+1` tokens back-to-back,
so most recorded ITLs are the within-chain gap.

| c | pt | TTFT cold | TTFT warm | TPOT | burst | decode tok/s | total tok/s | occ | prefix hit | accept | tok/row |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1st | 19.3 s | 1.1 s | 9.80 ms | 34.8 ms | 102.0 | 7440.7 | 0.26 | 0.883 | 0.509 | 0.591 |
| 2 | 1st | — | 1.2 s | 31.26 ms | 110.8 ms | 32.0 | 8292.3 | 0.47 | 0.883 | 0.509 | 0.591 |
| 4 | 2nd | — | 0.5 s | 32.10 ms | 78.2 ms | 31.2 | 25432.8 | 0.85 | 1.000 | 0.287 | 0.406 |
| 8 | 2nd | — | 0.7 s | 60.70 ms | 145.7 ms | 16.5 | 31754.1 | 0.87 | 1.000 | 0.280 | 0.400 |
| 16 | 3rd | 6.8 s | 1.2 s | 109.43 ms | 262.7 ms | 9.1 | 32559.0 | 0.87 | 1.000 | 0.280 | 0.400 |

Gate exact=3 DET at 512/4k/16k/32k. 0 errors. 126/128. `prompt_tokens` p50 34963.

Two properties of this row are load-bearing when reading it:

- **`accept` tracks `pt`, not `c`.** A serve's first point misses the dataset's
  16 turn-0 sessions; later points inherit the cache. At matched c=16: 0.532 as
  a fresh serve's sole point vs 0.313 as a later point — **+70% from cache
  state alone**. "Accept halves at concurrency" is withdrawn.
- **`occ` = `out tok/s / (c × decode tok/s)`** is the fraction of wall clock a
  slot decodes rather than waits on prefill. At 0.26–0.47 (c=1/2) `burst` is
  inflated ~1/occ. Never read `burst` as a kernel cost.

### CHAMPION (no-spec) — `a956f69b1` (2026-07-28) · `arle-fa3b2`

Features on: host-authoritative KV mirror · batched FA3 (one launch per layer).

| c | arm | TTFT cold | TTFT warm | TPOT | ITL p50 | decode tok/s | total tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | MoE | 9.2 s | 0.7 s | 16.22 ms | 16.17 ms | 61.7 | 6707.2 |
| 8 | MoE | 0.6 s | 0.5 s | 44.10 ms | 38.31 ms | 22.7 | 27967.8 |
| 16 | MoE | 1.8 s | 0.6 s | 73.74 ms | 60.90 ms | 13.6 | 33858.9 |
| 1 | dense | 19.0 s | 1.0 s | 28.83 ms | 28.78 ms | 34.7 | 5028.2 |
| 2 | dense | — | — | 78.40 ms | — | 12.8 | — |
| 4 | dense | — | — | 64.67 ms | — | 15.5 | — |
| 8 | dense | 1.2 s | 0.5 s | 81.87 ms | 52.30 ms | 12.2 | 24702.9 |
| 16 | dense | 4.2 s | 0.7 s | 122.15 ms | 66.55 ms | 8.2 | 30045.6 |

ITL p50 fit: MoE `15.7 + 2.82·B` ms, dense `38.0 + 1.78·B` ms.
Gate exact=3 DET at 512/4k/16k/32k. 0 errors. MoE 128/128, dense 126/128.
Anchor audit 2026-07-30 re-ran this binary at −0.03 / +0.40 / +2.26% — accumulated
drift over five accepted DSpark updates is bounded under 2.3%.

DSpark over no-spec at matched c: 2.9× (c=1), 2.5× (c=2), 2.0× (c=4), 1.4× (c=8),
1.1× (c=16).

### Step budget — where the time goes (2026-08-01, `nsys`, dense FP8)

> **Superseded for decode by the W8A16 anchor below (2026-08-03).** Two of
> this section's conclusions were overturned there: the decode graph is no
> longer unreachable (it is default-on and worth −7.9%), and "widening the
> grid is not the lever" for `gated_delta_rule_decode` is false at c=1 decode,
> where there is no token axis left to shorten. The prefill half stands.

The champion tables say how fast; this says what to fix. Both captures are
GPU-idle, ThinkingCap-Qwen3.6-27B-FP8, one H20.

**Decode, 25 ms/step** (plain, no spec, 59 steps, 1094 `cudaLaunchKernel`/step):

| kernel | launches/step | ms | share |
|---|---:|---:|---:|
| `fp8_gemv_batch_kernel` | 400 | 13.8 | 66% |
| `gemv_handwritten_kernel` (bf16) | 97 | 4.3 | 21% |
| `gated_delta_rule_decode` | 48 | 0.80 | 4% |
| rms_norm / add / silu | ~250 | 0.79 | 4% |
| flash attn (16 of 64 layers are full-attn) | 16 | 0.20 | 1% |
| GPU idle between launches | — | ~4 | 16% |

Weights are 31.2 GB and H20's **measured achievable read is 3.5 TB/s** (not the
4.02 spec sheet), so the per-step floor is 8.9 ms. The GEMVs take 18.1 ms —
**49% of achievable**, reproducing the 51% attributed on 2026-07-10.

**Prefill, 33K in 28.6 s** (single request, 24.0 s GPU-busy, ~37K launches,
2328 `cuMemcpyDtoH` costing 1.58 s):

| kernel | launches | s | share |
|---|---:|---:|---:|
| `gated_delta_rule_prefill_recurrent` | 1152 | 9.37 | 33% |
| DeepGEMM FP8, all shapes | 7936 | 8.33 | 29% |
| TileLang full attention | 368 | 3.93 | 14% |
| `pack_quantize` bf16→fp8 | 9600 | 1.50 | 5% |
| conv1d / norm / silu | 3840 | 0.55 | 2% |
| GPU idle (includes host tokenization) | — | ≤4.6 | ≤16% |

Efficiency of each part against its own ceiling: DeepGEMM `gate_up` 199 TFLOPS
and `down` 189 TFLOPS = **64–67% of the FP8 peak, healthy**; full attention
54 TFLOPS = **36% of the BF16 peak**; the linear-attention recurrence is a
latency chain, **5.9 µs per token per layer** (~6 dependent `__syncthreads`
each). No free parallel axis is left — the block is already 512 threads
(`val_dim 128 × j_slice 4`) and the token axis is the recurrence. Its
`<<<48, ...>>>` grid starves a **78-SM** GPU only at c=1; varlen launches
`grid(num_value_heads, batch)`. Shortening the chain (chunked matmul form) is
the lever; widening the grid is not.

Verify decomposes as **22 ms intercept + 2.48 ms/row** (5.18 ms/row at 33K), and
the intercept equals one plain non-spec step: verifying 8 speculative tokens
costs what decoding 1 costs. Spec decode is working; the intercept is the wall.

[decode + graph-flag profile](experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md) ·
[prefill profile](experience/wins/2026-08-01-prefill-and-decode-step-budget.md) ·
[FP8 small-M attribution](experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md)

### Environment

- **Box** 1×H20 (sm_90, 78 SM, 96 GB), TP=1, eager, `--max-running-requests 16`.
- **Models** `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (dense, 64 layers, 16
  full-attn, kv_heads 4, head_dim 256, KV 64 KB/token) · `Qwen3.6-35B-A3B-FP8`
  (MoE, 40 layers, 10 full-attn, kv_heads 2, 256 experts, top_k 8).
- **Dataset** `bench-agent-32k-16x8.jsonl`, sha256 `8867f63eaac2f053…`,
  `prompt_tokens` p50 34828.
- **Runner** `bench_throughput.py`, 128 req/point, max_tokens 214, greedy,
  seed 20260416. **Gate** `needle_gate.py 512,4096,16384,32768 3 0.0`.
- **Metrics** TTFT and decode are separate SLOs, never averaged. Decode =
  token-weighted mean ITL (`Σ itl_s / count`); never `e2e − ttft` (this harness
  carries ~4.7 s teardown, inflating TPOT ~1.85×). Cold = session turn 0,
  warm = turns 1–7. `total tok/s` = prompt+generated over wall: capacity, not
  latency.

**Inert flag — do not cost this into a plan.** `--qwen35-decode-graph` prints
`ARMED` but produces zero `cuGraph*` calls (its call site sits below an
unconditional paged-KV early return).

**`--qwen35-gdr-chunked` is DEFAULT-ON** (2026-08-02, `c2eb5de9e`): 33K cold
prefill −26%; license = chat GSM8K 100 **95/100 both arms, zero
disagreements** + chat MMLU 80 vs 81 + needle 9/9 ×2 + stub-probe fallback.
Named trade: raw-completion few-shot can flip knife-edge boundary tokens
(the 11/100-vs-46/100 collapse was that, not a kernel bug —
[error](experience/errors/2026-08-02-gdr-chunked-gsm-collapse-was-a-knife-edge-harness.md));
chat/agentic serving is parity. **TTFT-cold champion rows predate this flip
and need a re-anchor sweep.**

---

## Qwen3.6-27B-W8A16 (Marlin) · 1×H20 · single-GPU — MATCHED-vs-SGLANG ANCHOR

New anchor (rule 3): different checkpoint, dataset and metric from the
long-agent rows above — not comparable to them. Its purpose is one question:
against SGLang running the **same gptq_marlin kernel over the same int8
weights**, how much of the gap is our runtime?

Model `iso-tc-huihui-w8a16` (Huihui-Qwen3.6-27B abliterated, W8A16 gs=128,
29 GB), GPU 6, `bench-agent-32k-64.jsonl`, c=1, 16 requests × 256 tokens,
temperature 0, seed 20260416. Metric is decode ITL p50 (TTFT is prefill and
not in scope here). SGLang 0.5.13 serves the mechanically repacked GPTQ v1
twin (`scripts/w8a16_to_gptq.py`) — identical int8 values, identical kernel.

### CHAMPION — `f6820efa9` (2026-08-03)

| arm | ITL p50 | ITL p99 |
|---|---:|---:|
| **ARLE, all #196 tranches + engine fix** | **18.98** | 19.52 |
| SGLang, same kernel + same weights | 17.07 | 18.67 |

Ladder, each step a matched same-protocol run:

| tranche | ITL p50 | cum. |
|---|---:|---|
| pre-#196 | 26.88 | — |
| T1 gate+up fusion | 26.31 | −2.1% |
| T3 in_proj_ba fusion | 25.08 | −6.7% |
| T5 small-M GEMV → cuBLASLt | 23.80 | −11.5% |
| T2 qkv/qkvz fusion | 23.21 | −13.7% |
| T4 whole-step decode graph | 21.37 | −20.5% |
| T5b lm_head → cuBLASLt | 20.77 | −22.7% |
| T6 GDN decode kernel | 20.19 | −24.9% |
| resident-page O(1) counter | **18.98** | **−29.4%** |

Greedy output byte-identical across every tranche except T5b (accumulation
order changes; gated by an f32 anchor and MMLU 84/100 instead). Graphed lane
verified by counted API events (17 captures / 4100+ replays per run), never
by the ARMED log line.

Remaining 1.9 ms vs SGLang, from the two-sided nsys ledger: FA3 decode
config ~0.2, marlin per-launch prologue residue, and inter-kernel gap on
~1100 graph nodes vs their ~980.

[matched A/B method](experience/wins/2026-08-02-w8a16-sglang-matched-ab.md) ·
[module ledger + T4](experience/wins/2026-08-03-t4-paged-decode-graph.md) ·
[the host-tail find](experience/wins/2026-08-03-resident-page-scan-per-token.md)

---

## DSv4-Flash-FP8 · 4×H20 · TP=4/EP=4 · eager

### CHAMPION — Base, `d0525cb06` (re-anchored 2026-07-25, #180)

> Short-prompt fingerprint, retired 2026-07-26 under rule 5 — the dataset is no
> longer reproducible from the repo. Evidence for what it licensed, not a
> comparison target.

Dataset `bench-prompts-20.jsonl`, sha256 `e095ddf1fcc9325a…`, 60 s/point,
max_tokens 256, seed 20260416. Slot line `59 slots / per_slot 338MB / 84736 tok`.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---:|---|---|
| 1 | 10 | 38.66 | 456 | 1085 / 1113 | 21.9 / 41.0 |
| 4 | 20 | 74.67 | 876 | 1447 / 2985 | 43.8 / 89.2 |
| 8 | 40 | 152.82 | 1793 | 1069 / 1204 | 47.5 / 93.2 |
| 16 | 48 | 197.51 | 2319 | 2238 / 2265 | 71.4 / 119.0 |

0 errors / 0 incomplete / 0 correctness_failed at every point. c32 needs
`--max-running-requests 32`; without it host-admission oversubscription degrades
to preemption, not a crash (#164/#162 closed).

**Spec decode is c=1-only on this fingerprint and not a default-flip candidate**
— DSpark +5.0% at c=1, −23/−44/−48% at c=4/8/16; MTP negative everywhere. The
crossover is the compute-bound transition: verify is free only while the GPU has
idle compute.

---

## Qwen3.6-27B-W4A16 · 1×V100 (sm_70) · eager

**`aec71ef16` (2026-07-21)** — V100 kernel opts + KV pool floor fix. Synthetic
prompts 64, 60 s/point, max_tokens 256, seed 20260416. KV pool 16384 tok BF16
(1.1 GB), 86 slots. Serve `--max-total-tokens 16384`.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---:|---|---|
| 1 | 11 | 22.8 | 24.4 | 251 / 304 | 40.4 / 41.6 |
| 4 | 12 | 25.5 | 27.4 | 17799 / 25769 | 0.02\* / 270 |
| 8 | 17 | 28.4 | 30.4 | 30818 / 54318 | 0.02\* / 335 |
| 16 | 16 | 30.1 | 32.1 | 72270 / 72933 | 0.02\* / 452 |

\* ITL p50 ≈ 0.02 ms is a streaming-sampling artifact at c≥4; read out tok/s.
Decode-bound at every concurrency (+32% from c=1 to c=16); TTFT grows linearly
with concurrency (queueing).

**DSpark on V100 is KILLED (−91% at c=1, errors at c≥8).** ITL 40 → 499 ms;
c=16 produced 131204 errors in 60 s with `[coordinator] lockstep stalled`. The
TP lockstep proposal path deadlocks at world_size=1 — needs a TP=1 fast path
before this arm is retried.
