# DSpark verify linear core batched: c=8 TPOT −12.7%, c=16 TPOT −5.8% — CUDA, 2026-08-07

> Status: **accepted at c≥2, null at c=1 as the mechanism predicts.**
> Counterbalanced A/B — two sweeps per arm in both orders, same box, same
> dataset, single-commit delta. Closes the c=8 half of the regression that
> [`2026-08-06-dspark-anchor-remeasure-c1-plus-40-percent.md`](2026-08-06-dspark-anchor-remeasure-c1-plus-40-percent.md)
> opened and about half of c=16.

## Problem

`Qwen35Model::linear_attention`'s `LinearCore::Rows` arm ran a **per-row host
loop**, calling `advance_linear_conv_gdr` once per row. The FlashQLA chunked
block lives inside that function and operates on one slot, so a DSpark verify
tick issued five launches per row per linear layer — conv1d, then FlashQLA
prep / cumsum / kkt / fwd — each carrying a 7-token chain.

| | c=1 | c=16 |
|---|---:|---:|
| launches per linear layer | 5 | 80 |
| tokens per launch | 7 | 7 |

Cost grows linearly with the row count and is absent at one row. That is the
shape of the measured regression: TPOT +9.7% at c=8, +21.5% at c=16, −12.7%
(faster) at c=1.

`0ac780495` (08-05) is where FlashQLA first compiled into the pod binary, so
the per-row call pattern only started being paid then — the same commit that
delivered the c=1 TTFT win. Fix forward, not revert.

## What worked

The batched kernels already existed. `replay_linear_only_batched` drives
`conv1d_prefill_varlen_cuda` + `gated_delta_rule_prefill_recurrent_varlen_cuda`
through per-slot pointer tables. Reading their indexing settled the question
without a kernel change:

- conv varlen reads row `s` through `x_ptrs[s]` (any base) and writes at
  `s * max_len * C`; state and length come from `state_ptrs[s]` / `row_len[s]`.
- GDR varlen reads one contiguous `qkv` at `s * max_len`, takes b/a through
  pointer tables, state through a table, writes output at `s * max_len`.

When every row has the same length, that `s * len` stride **is** the trunk's
ragged packing, byte for byte. So the trunk can take the same path with no
repack: one pointer upload and two launches per layer for the whole batch.

The gate is `rows > 1 && all lengths equal && len <= 64`. Sixty-four is one
FlashQLA chunk — below it there is no chunk parallelism to win back against B
times the launches. Prefill chunks are 2048 and keep the per-row FlashQLA loop;
single-row forwards are untouched, which is what preserves the c=1 win.

## Parameters

```bash
python3 scripts/gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8

arle serve --backend cuda --model-path ThinkingCap-Qwen3.6-27B-FP8 \
  --spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash \
  --dspark-block-size 6 --max-running-requests 16 --port 8321

python3 scripts/bench_throughput.py --url http://127.0.0.1:8321 \
  --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,2,4,8,16 --requests-per-concurrency 128 \
  --max-tokens 214 --temperature 0 --seed 20260416 --timeout-seconds 900
```

- 1× H20 GPU 6, TP=1, eager, 16 slots. Co-tenants on GPUs 0/4/5.
- Arm A `010af0ede` (parent), arm B `4933e1bf4` (this change). The only
  source delta between them is `crates/infer-cuda/src/qwen35.rs`.
- Dataset regenerated on the pod; `gen_bench_prompts.py` has no RNG, so it
  reproduces byte for byte (md5 `0f0d67222baa50c884ee3468a66d0df6`).
- **Two rounds in opposite orders** (round 1 A→B, round 2 B→A), because the
  arm that runs second is measurably slower — see below.
- 128/128 complete at every point, both arms, both rounds, 0 errors.

## The ordering effect, and why the design is counterbalanced

Round to round, each arm moved in the direction its position changed:

| c | A r1 → r2 | B r1 → r2 |
|---:|---:|---:|
| 8 | 72.124 → 75.826 (**+5.1%**) | 66.788 → 62.429 (**−6.5%**) |
| 16 | 123.097 → 126.875 (**+3.1%**) | 119.421 → 115.934 (**−2.9%**) |

The second arm of a round loaded cold (1173 s and 1133 s to ready, against
15 s for the first) and measured slower. One round alone therefore confounds
the treatment with position. Running both orders and averaging cancels it.

**In round 1 the ordering handicap worked against arm B, and arm B still won
at every c ≥ 2.** The effect survives its own disadvantage.

## Results

TPOT is `itl_mean`, the only honest per-token figure on a spec row — `itl_p50`
samples the within-chain gap at 0.02 ms. Each cell is the mean of the two
counterbalanced rounds.

| c | A TPOT ms | B TPOT ms | Δ TPOT |
| ---: | ---: | ---: | ---: |
| 1 | 8.539 | 8.566 | +0.3% |
| 2 | 19.248 | 18.661 | −3.1% |
| 4 | 36.685 | 35.584 | −3.0% |
| 8 | 73.975 | **64.608** | **−12.7%** |
| 16 | 124.986 | **117.678** | **−5.8%** |

Against the ±2.7% drift band measured yesterday, c=4 sits at its edge and
c=2/8/16 are outside it on TPOT.

**The c=1 null is the control, not a disappointment.** The batched lane is
gated on more than one row, so a single-row forward must be unchanged. It is,
to 0.3%. Every point from c=2 up moves the same direction.

Arm B's full row, for `docs/baselines.md`:

| c | TPOT ms | ttft p50 | ttft p90 | itl p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.566 | 802 | 10790 | 31.59 |
| 2 | 18.661 | 571 | 658 | 205.91 |
| 4 | 35.584 | 596 | 1035 | 505.70 |
| 8 | 64.608 | 743 | 1508 | 545.90 |
| 16 | 117.678 | 1419 | 3459 | 773.57 |

## Correctness gate

The standard needle ladder (`needle_gate.py`, 241 → 32000, ×3, `RAW=1
TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=64`) is `exact=3 miss=0 DET` at every
length on both arms. **It does not cover this change.** The ladder issues one
request at a time, and the batched lane is gated on more than one row, so the
ladder runs exactly the path that was left alone. A gate that only counts
absent symptoms passes on code that never ran.

The path needs a gate that reaches it, and the failure it must catch is
specific: the batched core reaches each row's conv ring and recurrent state
through a pointer table, so a mis-indexed table makes row `i` advance row `j`'s
state. Every row still emits fluent text — just the wrong content. Degeneracy
checks cannot see it (`bench_throughput.py`'s `correctness_failed` only flags
empty output and repetition loops, and it was 0 on all four sweeps), and a
single-request ladder cannot see it either.

`scripts/needle_concurrent.py`: N in-flight requests, a **different needle per
row**, row-unique filler so a prefix-cache hit cannot mask a mix-up, and an
explicit report of which other row's needle came back. c=2 / 8 / 16, ×3 rounds,
both arms:

| arm | c=2 | c=8 | c=16 | cross-row |
|---|---|---|---|---|
| A `010af0ede` | 6/6 exact | 24/24 exact | 48/48 exact | none |
| B `4933e1bf4` | 6/6 exact | 24/24 exact | 48/48 exact | none |

## Problems

**Acceptance drops 1.2%.** `accept_rate` is 0.3036 / 0.3031 on arm A against
0.2996 / 0.2996 on arm B — consistent within each arm across rounds, so the
difference is the treatment, not noise. The batched recurrent kernel is not
bit-identical to per-row FlashQLA, and the shifted numerics move which draft
tokens verify. Arm B ran ~1% more chains for the same output and was still
12.7% faster per token at c=8, so the trade is clearly net positive, but this
is a real behavior change on the default path and is why the needle gate below
is required rather than optional.

**About half of c=16's regression is still there.** Yesterday's champion
measured 63.685 / 111.529 ms at c=8 / c=16. Arm B is 64.608 / 117.678 —
within 1.4% at c=8, still 5.5% above at c=16. That comparison is cross-day and
the champion binary no longer exists on the pod, so treat it as indicative.
The per-row loop was the dominant cause at c=8 and a partial one at c=16.

## Learnings

**Look for the batched sibling before writing one.** The whole fix is a
routing change: `replay_linear_only_batched` had been calling the varlen
kernels for this exact shape all along, and the trunk was not. The cost of
finding that was reading two kernels' indexing; the cost of not finding it
would have been a new pair of CUDA kernels.

**Uniformity is what makes a ragged buffer batchable.** The varlen kernels
assume a `s * max_len` row stride, which normally forces a repack. A verify
tick's rows are all the chain length, so the two layouts coincide and the
repack disappears. The gate encodes that condition rather than assuming it.

**A concurrency-gated change needs a concurrent gate.** The needle ladder was
green on both arms before the batched lane was ever entered — it is
single-request, and the lane requires more than one row. Yesterday's lesson was
that a gate counting absent symptoms passes on code that never ran; here the
same shape appeared one layer up, in a gate that runs but on the wrong path.
The fix is to pick the gate from the failure mode: a pointer table can hand row
`i` row `j`'s state, so each row carries its own needle and the gate reports
which other row's needle came back.

**Counterbalance the order when the second run loads cold.** Position was
worth ±3 to ±6.5% here — larger than the whole c=16 effect and larger than the
drift band. A single-order A/B would have reported c=8 at −7.4% instead of
−12.7%, and c=16 at −3.0%, which is inside the band and would have read as a
null. Two rounds in opposite orders cost one extra hour and turned a marginal
result into a decisive one.

**A same-window A/B is worth two extra builds when the window is dirty.**
13 commits touch the serve path between yesterday's measured HEAD row and this
one, so comparing against that row would have attributed their sum to this
change. Building the parent commit as arm A cost about 40 minutes and made the
delta single-commit.

**Archives expire.** `/host/spec-phase/arle-mk` and
`/host/gdr-gates/arle-gdr2-3d80dd4`, both cited yesterday as the arms that made
the champion row falsifiable, are gone from the pod. The A/B above had to
rebuild its own control. An archived binary is only a control while it exists.
