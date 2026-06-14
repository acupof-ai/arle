# DSv4-Flash long-context concurrency — decode aggregate FLAT across c (worse than short-ctx 1.4× cap), prefill fully serial

## Goal
Measure DSv4-Flash *current* concurrent throughput at long context — c∈{1,2,4,8}
× prompt∈{32K,64K,128K} — to answer "is c≥2 usable at long context?" and
quantify the gap the batched-FlashMLA-decode lever (#1, [throughput
plan](../../plans/dsv4-concurrency-throughput.md)) must close. The prior
[short-context baseline](2026-06-13-dsv4-concurrency-baseline-serial-capped.md)
found a 1.4× aggregate cap; long context (attention fraction rises) was
unmeasured.

## Hypothesis
Long-context concurrency is ≥ as bad as the short-context 1.4× cap, because the
batched lane loses FlashMLA at the `seq_len==1` gate → per-row attention, which
costs more as context (= attention work) grows.

## Params / Env
8×H20, TP=8/EP=8, tree `/data01/build/arle` @ `7d660f66` (zero source edits).
`arle serve --backend cuda --num-slots 8 --max-total-tokens 140000 --page-size
64 --total-pages 24000 --spec-type mtp --mtp-draft-tokens 2 --dsv4-batched-decode`
(ON unless noted), `INFER_DSV4_MAX_SEQ_LEN=140000`, allreduce MoE + native
DeepGEMM experts. Prompts = repeated-paragraph filler, server-confirmed
prompt_tokens (32755 / 65505 / 131005). Decode-agg = Δgenerated_tokens /
(batch_wall − TTFT_wall) via `/v1/stats` deltas. No OOM at any cell (peak 68.7 /
97.9 GB). All 12 cells completed; no timeouts/errors.

## Results — batched-decode ON

| len | c | TTFT (s) | agg decode tok/s | per-req tok/s | tok/step | agg scaling vs c=1 |
|-----|---|----------|------------------|---------------|----------|--------------------|
| 32K  | 1 | 5.98  | 48.5 | 48.5 | 1.80 | 1.00× |
| 32K  | 2 | 11.9  | 45.8 | 22.9 | 3.24 | 0.94× |
| 32K  | 4 | 23.8  | 48.1 | 12.0 | 6.48 | 0.99× |
| 32K  | 8 | 47.7  | 45.7 | 5.71 | 6.74 | 0.94× |
| 64K  | 1 | 12.4  | 46.2 | 46.2 | 1.60 | 1.00× |
| 64K  | 2 | 24.8  | 47.4 | 23.7 | 3.12 | 1.03× |
| 64K  | 4 | 49.4  | 43.2 | 10.8 | 5.64 | 0.94× |
| 64K  | 8 | 98.9  | 45.4 | 5.67 | 5.99 | 0.98× |
| 128K | 1 | 26.9  | 39.7 | 39.7 | 1.22 | 1.00× |
| 128K | 2 | 53.6  | 45.6 | 22.8 | 2.56 | 1.15× |
| 128K | 4 | 106.9 | 44.6 | 11.2 | 4.86 | 1.12× |
| 128K | 8 | 213.6 | 43.9 | 5.49 | 5.17 | 1.11× |

### Batched ON vs OFF (c=8)
| cell | TTFT ON/OFF (s) | agg decode ON/OFF (tok/s) |
|------|-----------------|---------------------------|
| 32K c=8 | 47.7 / 48.0 | 45.7 / 47.1 |
| 64K c=8 | 98.9 / 99.0 | 45.4 / 45.5 |

## Findings
1. **Aggregate decode does NOT scale with c — flat ~44-48 tok/s at every length.**
   Per-request collapses 1/c (48→23→12→6). This is ~1.0× scaling — **worse than
   the short-context 1.4× cap** (zero concurrency benefit on decode). tok/step
   rises ~c× (a step advances all rows) while agg holds → ms/step rises ~c× → the
   per-row-serial-over-rows MLA-decode signature, NOT a true batched kernel.
2. **TTFT linear in BOTH length AND concurrency — prefill fully serial.** Single
   prefill 6/12/27s @ 32K/64K/128K; at c=8 it is 48/99/**214s**. No cross-request
   prefill batching/overlap. This is a SEPARATE, additive bottleneck that
   batched-*decode* does not address.
3. **Slot clamp 8→6**: every serve logs `executor clamped slots 8 -> 6; scheduler
   follows` — true max concurrency is 6 (6 in-flight + 2 queued), not 8.
4. **MTP acceptance degrades with context**: tok/step @c=1 1.80 (32K) → 1.60
   (64K) → 1.22 (128K) — at 128K MTP barely accepts beyond the bonus token.
5. **`--dsv4-batched-decode` is a no-op at long context** — ON vs OFF within
   run-to-run noise at 32K/64K c=8.

## Verdict
Long-context concurrent serving is **not usable today**: decode aggregate is
constant, so adding users only divides fixed throughput + stretches TTFT
linearly. Hypothesis CONFIRMED and then some (long-ctx is ~1.0×, not just ≥1.4×).
The gap batched-FlashMLA-decode (#1 lever) must close: make aggregate decode
tok/s **rise with c** (per-request held, aggregate ≈ c× until HBM/compute bound).
Prefill-serial TTFT (#2) is a distinct lever, not in batched-decode scope.

## Rule
- **Concurrency scaling is shape-specific — re-measure at the production context
  length, never extrapolate from a smoke shape.** The short-context 1.4× cap
  *understated* the problem; at 32K-128K decode aggregate is flat (~1.0×). Same
  lesson as [SLO-from-SLO-workload](../errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md).
- **An opt-in "batched" flag that leaves aggregate flat is reachability, not a
  win.** `--dsv4-batched-decode` runs but is a no-op at long ctx — `plan_label`
  ≠ license ([axis2 kill](../errors/2026-05-25-axis2-mixed-default-kill.md)).
  Aggregate-rises-with-c is the only acceptance bar for the decode lever.
- Use **aggregate decode tok/s vs c** as the scaling metric and **TTFT vs (len,c)**
  as the separate prefill-serialization metric; report both per cell.
