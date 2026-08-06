# Rolling performance baselines

> Status: Active — **one SOTA row per model, plus its config**. A superseded
> row is deleted, not archived here; verdicts, rejected arms, prior champions
> and analysis live in the linked `experience/` entries.

Screening compares a new run against the SOTA row — no second arm.

1. **Effect > ~10%** (2× the measured drift band): verdict valid, replace the
   SOTA row, archive the binary.
2. **Inside the ±3% drift band**: never kill on ambiguity. Escalate to a
   same-shell A/B against the archived binary (≥3 trials/arm, median + range).
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

## Qwen3.6-27B-FP8 · 1×H20 · single-GPU · eager — LONG-AGENT ANCHOR

### SOTA — DSpark, `4933e1bf4` (2026-08-07)

> **Part of the c≥4 decode regression is fixed; c=16 still carries about half.**
> The verify linear core ran a per-row host loop, costing B× the kernel launches
> per layer; batching it is worth −12.7% TPOT at c=8 and +10.0% total tok/s at
> c=16 in a counterbalanced A/B
> ([`wins/2026-08-07-dspark-verify-linear-core-batched.md`](experience/wins/2026-08-07-dspark-verify-linear-core-batched.md)).
> c=8 is now within 1.4% of the 07-30 champion, c=16 still 5.5% above it. The
> audit that opened the regression is in
> [`wins/2026-08-06-dspark-anchor-remeasure-c1-plus-40-percent.md`](experience/wins/2026-08-06-dspark-anchor-remeasure-c1-plus-40-percent.md).
> Screen a candidate against c=1 and c=8 freely; c=16 is still against a
> partly-sick champion.

Features on: batched draft · replay · snapshot · capture · markov+confidence
head driving the goodput budget. Serve adds `--spec-type dspark
--mtp-draft-model Qwen3.6-27B-DFlash --dspark-block-size 6`; `--spec-max-batch`
is the shipped default 16.

**Schema changed with this row.** It is driven by `bench_throughput.py`, which
does not emit `burst`, `occ`, `tok/row`, or per-point `prefix hit`. Those
columns are dropped rather than carried forward at their 07-30 values. `accept`
is the run's cumulative rate, not per-point.

A spec row carries `tok/row` (committed tokens per verify row; plain decode
= 1.0) and `burst`, never ITL p50 — a spec step emits `k+1` tokens back-to-back,
so most recorded ITLs are the within-chain gap.

One serve, ascending, so `pt` is 1st through 5th — every point but c=1 inherits
a warm cache. `TTFT cold` is the p90 of the 128 requests (16 of them are turn 0);
`TTFT warm` is the p50.

**Every cell is the mean of two sweeps run in opposite orders.** The arm that
runs second on a shared box loads cold (1173 s to ready against 15 s) and
measures 3–6.5% slower — larger than the drift band, so a single sweep confounds
the binary with its position. Counterbalance any A/B on this row.

`TPOT` is `itl_mean`, and for a spec row that is the only honest per-token
number: `itl_p50` reads 0.02 ms because it samples the within-chain gap.
`decode tok/s = 1000 / TPOT` is per request and excludes prefill; `total tok/s`
is the whole-run figure and does not.

| c | pt | TTFT cold | TTFT warm | TPOT | decode tok/s | total tok/s | req/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1st | 10.79 s | 0.80 s | 8.57 ms | **116.7** | **10486.0** | 0.300 |
| 2 | 2nd | 0.66 s | 0.57 s | 18.66 ms | 53.6 | 21941.2 | 0.629 |
| 4 | 3rd | 1.03 s | 0.60 s | 35.58 ms | 28.1 | 26853.3 | 0.769 |
| 8 | 4th | 1.51 s | 0.74 s | **64.61 ms** | 15.5 | 31457.0 | 0.901 |
| 16 | 5th | 3.46 s | 1.42 s | **117.68 ms** | 8.5 | **32953.9** | 0.944 |

128/128 at every point, both sweeps, 0 errors. `prompt_tokens` 34782/request
against a 32000 target (+8.7%, inside the ±10% bar). Cumulative `accept` 0.2996
— 1.2% below the row this replaces, because the batched recurrent core is not
bit-identical to per-row FlashQLA and the shifted numerics move which draft
tokens verify. Correctness is gated by `scripts/needle_concurrent.py` (a
distinct needle per row at c=2/8/16, ×3) rather than by acceptance.

**Measured drift band, ±2.7%.** The same sweep run twice gave total tok/s
10444.2 / 20484.9 / 24666.9 / 29450.8 / 31313.7 — the row above is the idle-box
repeat, and the spread across the pair is what a candidate has to clear. Use
this, not the ±3% constant at the top of the file, for this workload.

**The superseded row, kept as the regression's other arm** — `51985031d`
(2026-07-30) · `arle-mk`, re-run on the same box and dataset on 2026-08-06:

total tok/s (prefill + decode):

| c | recorded 07-30 | `arle-mk` re-run | this row | Δ vs `arle-mk` |
|---|---:|---:|---:|---:|
| 1 | 7440.7 | 7287.4 | 10321.5 | **+41.6%** |
| 2 | 8292.3 | 20740.4 | 19967.0 | −3.7% |
| 4 | 25432.8 | 25265.7 | 23750.0 | −6.0% |
| 8 | 31754.1 | 30621.3 | 27517.6 | **−10.1%** |
| 16 | 32559.0 | 31890.9 | 29583.1 | **−7.2%** |

**TPOT, decode only — this is where the regression actually lives:**

| c | recorded TPOT | `arle-mk` re-run | this row | Δ decode tok/s |
|---|---:|---:|---:|---:|
| 1 | 9.80 ms | 9.69 ms | **8.46 ms** | **+14.5%** |
| 2 | 31.26 ms | 19.82 ms | 18.89 ms | +4.8% |
| 4 | 32.10 ms | 35.69 ms | 36.23 ms | −1.5% |
| 8 | 60.70 ms | 63.68 ms | **69.89 ms** | **−8.9%** |
| 16 | 109.43 ms | 111.53 ms | **135.51 ms** | **−17.7%** |

The champion reproduces its own 07-30 row on both metrics — total tok/s within
2.1% at c=1/4/16, TPOT within 1.1% at c=1 and 1.9% at c=16 — so the deficit is
a regression rather than drift or a fingerprint difference.

Separating the two metrics changes what has to be bisected. **The regression is
in decode and scales with concurrency** (−1.5 / −8.9 / −17.7% at c=4/8/16),
while c=1 decode got 14.5% *faster*. A blended `out tok/s` hid both halves. Any
candidate mechanism has to explain a per-token cost that grows with batch and
is absent at batch 1 — which rules out the prefill-side changes that landed in
the same window, `SIDECAR_SNAPSHOT_STRIDE_PAGES` included, since a
cross-conversation restore is TTFT work and cannot move steady-state TPOT.

Both `this row` columns come from the audit's back-to-back shell; the standalone
sweep above is a separate run of the same binary, which is why c=1 reads 10406.8
there and 10321.5 here.

Two properties of this row are load-bearing when reading it:

- **`accept` tracks `pt`, not `c`.** A serve's first point misses the dataset's
  16 turn-0 sessions; later points inherit the cache. At matched c=16: 0.532 as
  a fresh serve's sole point vs 0.313 as a later point — **+70% from cache
  state alone**. "Accept halves at concurrency" is withdrawn.
- **`occ` and `burst` are not in this row's schema.** When a driver does emit
  them: `occ = out tok/s / (c × decode tok/s)` is the fraction of wall clock a
  slot decodes rather than waits on prefill, and at low `occ` `burst` is
  inflated ~1/occ. Never read `burst` as a kernel cost.

DSpark over the same binary with spec off: 2.9× (c=1), 2.5× (c=2), 2.0× (c=4),
1.4× (c=8), 1.1× (c=16).

### Step budget — where the time goes (2026-08-01, `nsys`, dense FP8)

The SOTA table says how fast; this says what to fix.

> **STALE — do not rank prefill work off this table.** Measured before chunked
> GDR went default-on (`c2eb5de9e`, 08-02). Its #1 row is a kernel that no
> longer runs at the shipped defaults: the FlashQLA chunked path replaced
> `gated_delta_rule_prefill_recurrent` and measured **1.06 s against its 9.37 s**
> ([`wins/2026-08-02-flashqla-chunked-gdr-h48.md`](experience/wins/2026-08-02-flashqla-chunked-gdr-h48.md)),
> which moves 33% of the prefill budget and reorders everything below it. Two
> more prefill changes landed after (`0ac780495`, `301d0c074`). A re-measure
> needs one `nsys` capture; the two attempts that failed and the delay to use
> are recorded in
> [`errors/2026-08-06-decode-lever-board-rebuilt-gemm-is-not-the-top-lever.md`](experience/errors/2026-08-06-decode-lever-board-rebuilt-gemm-is-not-the-top-lever.md).
> The verify decomposition below is also superseded: measured directly on the
> current binary as **5.69 ms/row verify + 3.04 ms/row draft**, and the per-row
> term is not the GDN lane.

**Prefill, 33K in 28.6 s** (single request, 24.0 s GPU-busy, ~37K launches,
2328 `cuMemcpyDtoH` costing 1.58 s):

| kernel | launches | s | share |
|---|---:|---:|---:|
| `gated_delta_rule_prefill_recurrent` (no longer on the default path) | 1152 | 9.37 | 33% |
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
the prefill lever.

Verify decomposes as **22 ms intercept + 2.48 ms/row** (5.18 ms/row at 33K), and
the intercept equals one plain non-spec step: verifying 8 speculative tokens
costs what decoding 1 costs. Spec decode is working; the intercept is the wall.

[decode + graph-flag profile](experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md) ·
[prefill profile](experience/wins/2026-08-01-prefill-and-decode-step-budget.md) ·
[FP8 small-M attribution](experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md)

---

## Qwen3.6-35B-A3B-FP8 (MoE) · 1×H20 · single-GPU · eager

### SOTA — `a956f69b1` (2026-07-28) · `arle-fa3b2`

No spec. Features on: host-authoritative KV mirror · batched FA3 (one launch
per layer).

| c | TTFT cold | TTFT warm | TPOT | ITL p50 | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 9.2 s | 0.7 s | 16.22 ms | 16.17 ms | 61.7 | 6707.2 |
| 8 | 0.6 s | 0.5 s | 44.10 ms | 38.31 ms | 22.7 | 27967.8 |
| 16 | 1.8 s | 0.6 s | 73.74 ms | 60.90 ms | 13.6 | 33858.9 |

ITL p50 fit `15.7 + 2.82·B` ms. Gate exact=3 DET at 512/4k/16k/32k. 0 errors,
128/128. Anchor audit 2026-07-30 bounds accumulated drift under 2.3%.

---

## Environment — both 1×H20 FP8 rows

- **Box** 1×H20 (sm_90, 78 SM, 96 GB), TP=1, eager, `--max-running-requests 16`.
- **Models** `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (dense, 64 layers, 16
  full-attn, kv_heads 4, head_dim 256, KV 64 KB/token) · `Qwen3.6-35B-A3B-FP8`
  (MoE, 40 layers, 10 full-attn, kv_heads 2, 256 experts, top_k 8).
- **Dataset** `bench-agent-32k-16x8.jsonl`, sha256 `8867f63eaac2f053…`,
  `prompt_tokens` p50 34828.
- **Runner** `bench_throughput.py`, 128 req/point, max_tokens 214, greedy,
  seed 20260416. **Gate** `needle_gate.py 512,4096,16384,32768 3 0.0`; a change
  that only fires at concurrency needs `needle_concurrent.py` as well, since the
  ladder is single-request and would run the untouched path.
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

## Qwen3.6-27B-W8A16 (Marlin) · 1×H20 · single-GPU

Model `iso-tc-huihui-w8a16` (Huihui-Qwen3.6-27B abliterated, W8A16 gs=128,
29 GB), GPU 6, `bench-agent-32k-64.jsonl`, c=1, 16 requests × 256 tokens,
temperature 0, seed 20260416. TTFT is cold — 16 distinct prompts, no prefix
hits. SGLang 0.5.13 row serves the GPTQ v1 twin repacked by
`scripts/w8a16_to_gptq.py` — identical int8 values, identical kernel.

### SOTA — snapshot stride 8192 (2026-08-06)

Two reps per arm; reps agree to 0.10 s TTFT / 0.02 ms ITL. P/D reported
separately: `prefill tok/s = prompt_tokens / TTFT` (33000 prompt tokens),
`decode tok/s = 1 / ITL`.

| arm | TTFT p50 | prefill tok/s | ITL p50 | decode tok/s | ITL p99 | e2e p50 |
|---|---:|---:|---:|---:|---:|---:|
| ARLE | 23.01 s | 1434 | **16.70** | **59.9** | 20.50 | 27.4 s |
| SGLang, same kernel + same weights | **21.03 s** | **1568** | 17.16 | 58.3 | **19.19** | **25.44 s** |

Decode leads by 2.8%, TTFT is 1.09× behind (was 1.48× then 1.19×), p99 7% behind.
Gate: needle 512/4k/16k/32k ×3, all `exact=3 miss=0 DET`.

Prefill idle split, cold 33K, `--cuda-graph-trace=node`:

| | ARLE | SGLang |
|---|---:|---:|
| GPU busy | 21.9 s | within 0.93 s of ARLE |
| in-span idle, stride 2048 | 1.675 s | 0.19 s |
| D2H, pinned staging (was pageable) | 2.771 GB / **0.062 s** (0.577 s) | ~0 |

Periodic snapshot cost by stride, 33K prefill: 18 snapshots 3.13 s · 4
snapshots 0.85 s · 0 snapshots 0 s. Each retains ~150 MB until publish, so the
cost is the count. Remaining gap against SGLang: 1.98 s.

`--chunked-prefill-size` is not a lever on either stack: 2048 vs 4096 (ARLE)
and 4096 vs 8192 (SGLang) all land inside 0.07 s TTFT.

The row is from a build with no `ARLE_CUDA_*` set — FA3 and FlashQLA now build
from vendored-tree + sm_90 detection alone. Confirmed against the env-set
build: TTFT 24.94/24.95 s vs 24.97/25.05, zero fallback lines in the serve log.

[FlashQLA stub build + prefill ledger](experience/wins/2026-08-05-flashqla-was-never-compiled-into-the-pod-binary.md) ·
[decode budget, both stacks](experience/wins/2026-08-04-w8a16-decode-step-kernel-budget.md) ·
[FA3 splits](experience/wins/2026-08-04-fa3-decode-splits-fill-the-sms.md) ·
[conv1d fusion](experience/wins/2026-08-04-conv1d-decode-fusion.md) ·
[repack method](experience/wins/2026-08-02-w8a16-sglang-matched-ab.md)

---

## DSv4-Flash-FP8 · 4×H20 · TP=4/EP=4 · eager

### SOTA — Base, `d0525cb06` (re-anchored 2026-07-25, #180)

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

---

# Training baselines — OPD writeback

> Status: Active. Same screening rules as the inference rows above. Training
> numbers were previously scattered across wins entries and task descriptions;
> this section is the single truth. A row not listed here has not been measured.

**Fingerprint** — re-anchor when any of these move: model, LoRA target set,
`cp_size`/`dp_size`, GPU set, sequence length, commit.

- Model `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (64 layers = 16 full-attn +
  48 gated-delta, kv_heads 4, head_dim 256), or `qwen35-08b-clean`
  (24 layers, `full_attention_interval: 4`, dense MLP) for the correctness rows.
- LoRA `attention-qv`. Workload = `--synthetic-writeback-seq N` (one masked-CE
  writeback on a synthetic trajectory; no rollout).
- Box 8×H20 (sm_90, 97.9 GB). Two ranks = one CP group unless stated.

**Before trusting any CP row, verify the binary.** `nm -D <bin> | grep
ncclCommInitRank` and `ldd <bin> | grep libnccl`. A shared build target was
silently overwritten by a `cuda`-only build on 2026-08-05 and the resulting run
failed in a way that reads as a code bug. FA3 additionally needs the vendored
hopper tree and an sm_90 target at build time — without them
`ring_fa3_route`'s real-kernel marker returns 0 and the ring falls back to the
scalar kernels.

## SOTA — 27B, cp=2, seq=32768 · `15caff0d0` (2026-08-05)

| | |
|---|---:|
| forward | 34.2 s |
| fused CE | 0.92 s |
| backward | 190.0 s |
| optimizer | 0.05 s |
| **step** | **225.2 s** |
| checkpoint peak | 61,396 MiB/rank |
| loss | 10.871086 |
| grad_norm | 2.263385 |

Both ranks print identical loss and grad_norm (post-all-reduce). Reproduces the
2026-08-04 FA3 reference (10.871086 / 2.264733 / ~212 s) to 6-decimal loss and
0.06% grad-norm; the +6% on step is shared-box variance.

## SOTA — 27B, cp=2, seq=81920 · FlashQLA default-on `fa742a038` (2026-08-05)

FlashQLA GDN chunkwise backward is the default (`--gdr-chunkwise-prefill=true`).
Same harness (`/host/fqgate.sh perf_on`), same seq, only variable is the flag.

| | rank 0 | rank 1 | recurrent (below) | speedup |
|---|---:|---:|---:|---:|
| forward | 64.124 s | 64.125 s | 81.0 s | 1.26× |
| fused CE | 0.83 s | 0.83 s | 1.91 s | — |
| backward | 312.643 s | 312.648 s | 670.275 s | **2.14×** |
| **step** | **378.723 s** | **378.723 s** | 752.956 s | **1.99×** |

Peak host RSS 55.4 GB, loss 4.537510, grad_norm 7.976866, RUN_EXIT=0. The 71%
`linear_attention_chunked_scan_backward_f32` row is gone.

The recurrent column is `--la-backward-mono` on `e675f031b`: device peak
91,547 MiB/rank (93.5% of the card), loss 4.536131, grad_norm 7.202155.

### Step budget — where the time goes

`nsys cuda_gpu_kern_sum`, one step, both ranks combined, FA3 engaged.

| share | time | instances | kernel |
|---|---:|---:|---|
| 71.0% | 707.345 s | 90 | `linear_attention_chunked_scan_backward_f32` |
| 6.7% | 66.316 s | 238,080 | `gated_delta_rule_prefill_recurrent` |
| 3.9% | 38.365 s | 7,436 | nvjet GEMM 128×256 |
| 3.2% | 32.096 s | 4,194 | nvjet GEMM 320×128 TNT |
| 1.9% | 19.134 s | 2,886 | nvjet GEMM 320×128 NNT |
| 1.5% | 15.271 s | 11,664 | `transpose_axes_swap_f32` |
| 1.5% | 14.635 s | 47 | `FlashAttnBwdSm90` |
| 1.4% | 13.553 s | 25,196 | `slice_f32` |

The two gated-delta rows are 77.7% of the step. Both ride the route the
FlashQLA port (`4846f8046`) replaces.

## Correctness rows — 0.8B dense, seq=2048 · `15caff0d0` (2026-08-05)

Mean of 3 serial reps per cell, post-all-reduce (all ranks identical). Within-cell
spread is 5.2e-5 to 2.1e-4 relative across every cell.

| arm | grad_norm | deviation from cp=1 |
|---|---:|---:|
| cp=1 | 3.464900 | — |
| cp=2 | 3.459982 | −1.419e-3 |
| cp=4 | 3.464276 | −1.80e-4 |

FA3 is inert at cp=1 (no ring exists there): toggling it leaves loss identical at
8.963640 and grad_norm inside the spread. The deviation does not compound with
ring-step count — it collapses into the noise floor at cp=4, while the pre-flip
scalar path's grows (+1.085e-3 at cp=2 to +1.655e-3 at cp=4). See
[the gate entry](experience/wins/2026-08-05-fa3-cp-gate-compounding-not-sign.md).

## Known walls

| shape | outcome |
|---|---|
| 27B cp=1 seq=81920 | forward completes (3972.216 s), **backward OOMs** on `cuda alloc_zeros failed`. Host RSS 104.5 GB. The failing tensor is not named by the log. |
| 27B cp=2 seq=131072 | fits — backward peak 94,175 MiB (96.6%), ~3.3 GB headroom (2026-08-02, older commit) |
| 27B cp=4 seq=131072 | full step ~3100 s, host RSS 170.4 GiB total / ~44.6 GB per rank (2026-08-03, scalar ring, older commit) |

