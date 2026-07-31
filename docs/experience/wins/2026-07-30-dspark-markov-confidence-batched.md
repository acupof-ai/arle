# The markov settle and the confidence head batch across slots — the head is free, its verdict is what loses

## Context

The 2026-07-29 batched draft gated itself on `batchable_draft()`
(`markov.is_none()`), so installing a markov head sent the arm back to
`--spec-max-batch 1` — no speculation at all above c=1. I accepted that gate on
the grounds that the markov path was unverifiable for want of a checkpoint. It
was on the host the whole time: `dspark-fr-native` (`DSparkDraftModel`, 65
tensors, `markov_rank` 256, `enable_confidence_head`), same 5-layer / 5120 /
32-head geometry as `Qwen3.6-27B-DFlash`, differing only in its two extra heads.

## What Worked

Both remaining per-slot loops are the shape of the four already batched.

**The markov settle.** `bias = w2·w1[prev]` makes row r depend on row r-1;
[2026-07-26](2026-07-26-dspark-markov-chain-self-speculation.md) removed the
serial scan *within* a block by speculating the chain on itself, but each slot
still ran its own rounds — and `w2` is `[248320, 256]`, so its GEMM is
weight-bound and B slots re-read 127 MB B times, each round ending in an argmax
whose D2H drains the pipeline. One settle over `[vocab, b*block]` with `prevs`
laid out slot-major runs every slot's rounds together: a slot that has already
settled reproduces its own tokens, so looping until all of them agree returns
exactly what B separate settles would.

**The confidence prefix.** One D2H sync and `block` D2D feature copies per slot
became two `batched_copy` launches and one sync for the batch, with the per-slot
prefix scan falling out of one host-side vector.

`dspark_block_greedy` and `dspark_confident_prefix_len{,_at}` collapse into
`dspark_settle_rows` / `dspark_confident_keeps`, which the b=1 row path also
calls; `batchable_draft()` is deleted.

## Measurement — the shipped arm is untouched

1×H20 GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + 27B-DFlash block 6,
`bench-agent-32k-16x8`, 128 req/point, max_tokens 214, greedy, seed 20260416,
`prompt_tokens` p50 34963. Against the `d05d0aee6` champion row:

| c | TPOT before | TPOT after | Δ |
|---|---:|---:|---:|
| 1 | 9.77 ms | 9.80 ms | +0.3% |
| 8 | 60.74 ms | 60.70 ms | −0.07% |
| 16 | 107.94 ms | 109.43 ms | +1.4% |

All inside the ±3% band, which is the point: the new code is a path DFlash never
enters. Gate exact=3 DET at 512/4k/16k/32k, 0 errors, 126/128.

## The markov settle is free; so is the confidence head — its verdict is not

Nine arms, one binary (`arle-mk`), run concurrently on one GPU each (180 cores,
load 16, so the only shared resource is idle). All three drafts have identical
geometry — 5 layers, hidden 5120, intermediate 17408, 32 heads, 8 kv, head_dim
128, `markov_rank` 256 on both markov checkpoints. `dspark-aeon` and
`dspark-fr-native` also share `layer_types` (5×`full_attention`, all forced to
the 2048 window), so they differ **only** in `enable_confidence_head`; DFlash
alone is 4 sliding + 1 true full.

Chain length is exactly `block - 1` on every arm that does not truncate, so
`rows/step` and `depth` come straight out of the free `/stats` counters
(`steps = chains/c`, `rows/step = (drafted+chains)/steps`,
`step_ms = itl × (accepted+chains)/chains`). `t0` = `--dspark-conf-threshold 0`:
sigmoid never falls below 0, so the head runs every step and truncates nothing.

| arm | heads | block | c=1 | **c=16** | depth | rows/step | step @16 | tok/row |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no-spec | — | — | 34.8 | 8.0 | — | 16.0 | 124.8 | 1.000 |
| DFlash | none | 6 | 102.0 | 9.1 | 5.00 | 96.0 | 262.7 | 0.400 |
| aeon | markov | 6 | 103.0 | 9.2 | 5.00 | 96.0 | 269.9 | 0.415 |
| **FR t0** | **markov+conf** | 6 | **107.2** | **9.5** | 5.00 | 96.0 | **268.7** | 0.428 |
| FR | markov+conf | 6 | 92.3 | 6.3 | 3.75 | 76.0 | **430.8** | 0.568 |
| DFlash | none | 16 | 109.1 | 6.9 | 15.00 | 256.0 | 352.3 | 0.152 |
| **FR t0** | **markov+conf** | 16 | **115.4** | 7.5 | 15.00 | 256.0 | **354.1** | 0.166 |
| FR | markov+conf | 16 | 97.0 | 5.8 | 4.22 | 83.5 | 499.3 | 0.554 |
| aeon | markov | 11 | 113.0 | 7.7 | 10.00 | 176.0 | 335.7 | 0.235 |

**The batched markov settle is nearly free.** `aeon6` and `dflash6` verify the
same 96 rows at the same depth 5.00 and step in 269.9 vs 262.7 ms — **+7.2 ms,
+2.7%**. Read that as a lower bound, not an isolation: DFlash's one *true*
full-attention layer over a 35k context should make its base step the dearer of
the two, so the markov head's own share is ≥7.2 ms. Either way it is the first
measurement of a markov checkpoint speculating above c=1 at all, because until
now the gate forbade it.

**The confidence head's execution is free; acting on its verdict costs 162
ms/step.** `t0` keeps every GEMM, copy, embedding lookup, D2H and sync and
removes only the truncation, at rows and depth matched exactly to a head-free
control:

| pair (same rows, same depth) | step @16 | Δ |
|---|---:|---:|
| `FR t0` 6 vs `aeon6` — 96 rows, depth 5.00 | 268.7 vs 269.9 | **−0.4%** |
| `FR t0` 16 vs `dflash16` — 256 rows, depth 15.00 | 354.1 vs 352.3 | **+0.5%** |
| `FR` 6 vs `FR t0` 6 — one flag apart | 430.8 vs 268.7 | **+162 ms on 20 FEWER rows** |

So the head does exactly what it is for — truncating lifts tokens per verify row
33% (0.568 vs 0.428 at c=16, 0.731 vs 0.620 at c=1) — and the wall clock of a
shorter, *variable-length* chain eats roughly four times what the saved rows are
worth.

**Root cause CLOSED 2026-07-31 — and it was never raggedness.** See
"Raggedness is not the cost" at the end of this entry: the arms that looked
ragged-penalized were the arms that truncated chains to the bare anchor, and
those fell out of the batched verify into a serial per-slot decode
(`dspark_warm_decode_row`) — one forward per dropped slot, which is the only
mechanism in this path with a 56 ms/step magnitude. `7358cb06c` keeps
budget-zeroed greedy chains in the batch, and the penalty does not reproduce.

## DEFAULT FLIP — `--dspark-conf-threshold` 0.5 → 0 (flag since deleted)

**The shipped default was the worst arm in the table**: at c≥8 it made spec
decode slower than not speculating. TPOT ms, one flag apart, same binary and
dataset; no-spec is the mean of two runs that agreed to 2.7%:

| arm | c=1 | c=8 | c=16 |
|---|---:|---:|---:|
| no-spec | 28.77 | 83.34 | 124.86 |
| block 6, **0.5 (old)** | 10.84 | 84.67 **+1.6%** | 159.68 **+27.9%** |
| block 6, **0 (new)** | **9.33** | **60.41 −27.5%** | **104.73 −16.1%** |
| block 16, 0.5 | 10.31 | 91.92 **+10.3%** | 172.63 **+38.3%** |
| block 16, 0 | **8.67** | — | **133.54** +7.0% |

Worth −13.9/−28.7/−34.4% at c=1/8/16 on block 6, −15.9/−22.6% at c=1/16 on
block 16. Only the c=1 pairs match on point order (c=16 is point 3 vs 2, c=8 is
cross-serve), but five comparisons carry one sign at 14-34% against 2.7% noise.
Gate at `0`: **exact=3 DET at 512/4k/16k/32k**. (The flag itself was deleted
the same day — the head now always feeds the goodput scheduler; see the final
section.)

Whether `FR t0` displaces `dflash6` as champion *draft* is separate and still
open — its three c=16 trials read 104.73 / 184.77 / 235.54 ms.

**Reject the static threshold as a policy — not the head.** The head's
predictions are good (truncating lifts tok/row 33%); a fixed per-position cut
is the wrong consumer of them, and a looser cut only widens the gap because it
leaves a longer variable-length chain:

| threshold | depth | tok/row | rows/step | step @16 | TPOT c=16 |
|---|---:|---:|---:|---:|---:|
| 0 (`t0`, no cut) | 15.00 | 0.166 | 256.0 | 354.1 | 133.54 ms |
| 0.50 | 4.22 | 0.554 | 83.5 | 499.3 | 172.63 ms |
| 0.30 | 12.13 | 0.335 | 210.0 | 1088.2 | 247.76 ms |
| 0.15 | 14.87 | 0.274 | 253.9 | 1095.7 | 251.94 ms |

`t=0.15` and `dflash16` verify the same work — 253.9 vs 256.0 rows, depth 14.87
vs 15.00 — and step 1095.7 vs 352.3 ms, **3.1×**. Nothing about the head's
arithmetic changes across those four rows; only how much the chain length moves.

**`FR t0` block 6 beats the shipped champion at matched rows and depth**, and is
the first arm to do so:

| c | `dflash6` (champion) | `FR t0` 6 | Δ |
|---|---:|---:|---:|
| 1 | 9.80 ms TPOT | **9.33 ms** | **−4.8%** |
| 8 | 60.70 ms | **60.41 ms** | −0.5% |
| 16 | 109.43 ms | **104.73 ms** | **−4.3%** |
| 16 | 0.400 tok/row | **0.428** | **+7.0%** |

Same 96 rows, same depth 5.00, same binary — so this is the FR *weights* being
better than DFlash's, not the head, and not batching. It also beats `aeon6`
(108.33 at c=16).

Not a champion flip yet: c=8 is a wash and the third trial disagrees with itself
(three `fr6t0` c=16 points read 104.73 / 184.77 / 235.54 ms). Correctness gate on
`FR t0` block 6: **exact=3 DET at 512 / 4096 / 16384 / 32768**, 0 miss.

## Still open

- **Block size wants opposite values at the two ends.** Head-free: block 16 wins
  c=1 (109.1 vs 102.0) and block 6 wins c=16 (9.1 vs 6.9), because at c=1 the GPU
  is idle and rows are nearly free. `--dspark-block-size` is static; the decode
  batch is not.
- **Accept takes two values, switching between c=2 and c=4.** `fr6t0` reads
  0.544 at c=1 *and* c=2 (chains bit-identical, 4250) then 0.313 at c=4/8/16
  (mean_k 1.566/1.565/1.566 across 6859/6187/6666 chains — 20× tighter than
  sampling noise). `dflash6` shows the same two regimes. A gradient cannot do
  that; look for what changes state at c=4.
- **The accept rate falls with concurrency** on *every* draft (DFlash 0.509 →
  0.280), and every chain at c≥8 drafts on a rebased context
  (`partial_ctx_chains/chains` 0.75 → 1.00) while prefix reuse *improves*
  (0.883 → 1.000). A prefix-cache or sidecar restore skips the trunk prefill, so
  `df.rebase()` (`executor/qwen35.rs:1460`, `:1842`) leaves the draft holding a
  suffix-only context. Next probe: bucket accept by `ctx_end - ctx_base` at chain
  time. Not an FR-specific handicap: `dspark.rs:676` warns that its 5/5 declared
  full-attention layers all run the 2048 window, but `dspark-aeon` declares the
  same five and takes the same forcing, and DFlash keeps one true full layer —
  so the window costs FR nothing that `aeon` does not also pay.
- The two draft attention kernels (`dspark.rs:1290-1340`) are the only per-slot
  launches left in the batched draft — 160 at c=16. Same shape as the varlen
  conv1d/GDR pair already built, so a `blockIdx.y`-per-slot pointer table
  collapses them to two per layer.

## Build the paper's scheduler

(An earlier revision argued "do not build Algorithm 1 here" off the 0.53 ms
marginal row cost. Wrong twice: 160 rows is 85 ms, **32% of the step** — a
small per-row figure is not a small row budget; and the disproof ran the
*static threshold*, precisely the prior art DSpark (arXiv 2607.05147) §3.2.2
names and replaces.)

The paper's premise holds on our hardware. §3.2: an extra verified token is
near-free under light load and costs batch capacity under high concurrency.
Measured, same draft, block the only variable:

| | c=1 TPOT | c=16 TPOT |
|---|---:|---:|
| `dflash6` | 9.80 ms | **109.43** |
| `dflash16` | **9.16 ms** | 144.37 |

The optimum moves with load, and `--dspark-block-size` is static — the "opposite
values at the two ends" item below is this, unsolved. In the paper's objective
`Θ = τ·SPS(B)` (system tok/s at c=16) there is a real maximum:

| arm | B | τ | Θ |
|---|---:|---:|---:|
| no-spec | 16 | 1.000 | 128.1 |
| block 2 | 16 | 1.734 | 138.6 |
| block 4 | 48 | 2.277 | **154.0** |
| block 6 | 96 | 2.400 | **146.2** (repeat trial 152.8) |
| block 11 | 176 | 2.584 | 123.2 |
| block 16 | 256 | 2.440 | **110.8 — under no-spec** |

Block 2/4 close the curve's low end: the c=16 interior maximum sits at
block 4–6 (154.0 vs 152.8, within trial noise), rising monotonically from
no-spec below it and falling monotonically above it — exactly the unimodal
shape Algorithm 1's break-when-falling search assumes.

Algorithm 1 initializes `Θ_best ← R·SPS(R)` (line 6, the no-spec point) and
breaks the moment throughput stops rising (line 13), so **it cannot ship the
failure the 0.5 default shipped** — losing to not speculating at all. That alone
answers the rejection.

What looked like a blocker: Algorithm 1 emits unequal `ℓ_r` by construction, and
two measurements said this engine charges 3.1× for that (`fr16` threshold 0 vs
0.15: 354.1 vs 1095.7 ms at 253.9 vs 256.0 rows; blk4 uniform 64 rows 103.91 ms
≈ fr6t0 uniform 96 rows 104.73 vs fr6@0.5 ragged **76** rows 159.68). **Both
were confounded — see "Raggedness is not the cost" below.** Every penalized arm
is a *thresholded* arm, and thresholding is the one thing that cuts chains to
the bare anchor, which drops them out of the batched verify entirely.

The port target is sglang's deployed pipeline
(`speculative/dspark_components/`), not a home-grown subset: the confidence
head always runs (no threshold switch — static truncation is an eval-only
convenience, default-off there too), STS per-position temperatures calibrate
it, survival is `cumprod(confidence)`, and the batch token budget
`argmax_B τ*(B)/step_time(B)` uses **lagged** survival (1–2 steps prior,
`_shift_to_lag`) — which dissolves the "head output arrives post-draft, too
late to schedule" objection; only the per-request top-k allocation reads the
current step. Their SPS cost model is `bias + α(num_reqs) + θ(total rows)` —
the same additive decomposition we measured (116 ms + 0.53 ms/row at c=16).

## Scheduler core landed — measured, it wins

The port's first tranche is in (same day): static truncation deleted
(`--dspark-conf-threshold` removed), confidence → survival cumprod →
`qwen35_spec::dspark_verify_lens` (goodput argmax + global admission, B=0 arm
seeded), cost model `--dspark-sps-bias-ms`/`--dspark-sps-row-ms` defaulting to
the measured 211 + 0.53. Both stacks call the one function: Qwen3.5/3.6
batch-global (R=b), DSv4 per-slot (R=1). No lag ring (sync engine — current
survival precedes verify), no STS (no fitted temperatures yet). Host unit gate
`budget_scales_with_survival`.

**GPU A/B** (H20 GPU 3, ThinkingCap-27B-FP8, block 6, agent-32k prompts,
128 reqs/point, `scripts/`-equivalent probe at `/host/spec-phase/probe.sh`;
needle ladder `exact=3 DET` at 512/4k/16k/32k on both spec arms):

| arm | c=1 TPOT | c=16 TPOT | accept c=1 | accept c=16 | depth c=16 |
|---|---|---|---|---|---|
| no-spec | 78.25 ms | 262.49 ms | — | — | — |
| DFlash blk6 (fixed 5 rows) | 21.77 ms | 162.50 ms | 0.504 | 0.275 | 5.00 |
| fr-native blk6 + goodput | **20.93 ms** | **153.96 ms** | 0.539 | **0.402** | **3.79** |

The scheduler does exactly what the model says it should: at c=1 rows are
nearly free, it admits the full block (depth 5.00, TPOT −3.9% vs the fixed
control); at c=16 rows are dear, it cuts the average chain to 3.79 rows, and
accept rises 0.275 → 0.402 while TPOT drops 5.3%. Trading depth for accept
under load is the whole point of `argmax_B τ*(B)/step_time(B)`, and it is the
first arm to shrink the c=1 → c=16 accept collapse (−0.14 vs the control's
−0.23) instead of merely suffering it.

## Raggedness is not the cost — the serial fallback was

The scheduler was supposed to be blocked behind a 3.1× ragged-chain penalty.
That penalty does not exist. Three hypotheses, killed in order:

- **Workspace-slot realloc** (`HiddenSlot::get` exact-shape reuse): dead on
  arithmetic. A shape change re-`zeros` the buffer, but the biggest one is
  `[248320, rows]` logits = 47 MB ≈ 30 µs of memset; every slot in the path
  together is ~1.5 ms against a 56 ms/step gap. Off by 40×.
- **CUDA-graph tier miss** (the sglang `round_up_grid` story): structurally
  impossible here. The whole-step decode graph covers `rows == 1` only
  (`executor/qwen35.rs`), so a spec verify — uniform or ragged — is always
  eager. There is no tier to fall out of.
- **Serial fallback**: the only mechanism at the right magnitude. A chain cut
  to the bare anchor used to fail the batch filter and take
  `dspark_warm_decode_row`, one per-slot trunk forward each. Every "ragged"
  arm in the earlier table is a *thresholded* arm — thresholding is what
  produces bare-anchor chains — so the two variables were never separated.
  `7358cb06c` keeps those chains in the batch as one decode row.

The isolating A/B (zero code — `--dspark-sps-row-ms 0` makes rows free, so the
budget always admits the full block; same binary, weights, prompts, GPU):

| | c=1 TPOT | c=16 TPOT | c=16 depth | c=16 accept |
|---|---:|---:|---:|---:|
| uniform (`row_ms 0`) | 20.99 ms | 160.75 ms | 5.00 | 0.309 |
| ragged (goodput) | 20.93 ms | **153.96 ms** | 3.79 | 0.402 |

At c=1 the two arms are the same configuration by construction, and they land
0.3% apart — that is this campaign's noise floor. At c=16 the ragged arm is
4.2% **faster**, by about what its 1.2 fewer rows/request are worth at
0.53 ms/row. Raggedness costs nothing measurable, `round_up_grid` is not
needed, and the remaining c=16 cost is row count and accept rate — which is
exactly what the budget optimizes. Spec still costs 2× TPOT at c=16 vs c=1;
that gap is now unattributed and needs its own decomposition.

## Rule

**Name the mechanism and check its magnitude before believing a measurement's
label.** "Ragged chains cost 3.1×" survived a week and blocked a scheduler
because nobody asked what in this engine could plausibly cost 56 ms/step. Of
the three candidates, two die on arithmetic and structure alone — a memset is
40× too small, and the decode graph never covers a verify — leaving the one
that does a whole trunk forward per dropped slot. A number attached to the
wrong noun is worse than no number: it points the fix at the wrong code.

**When every arm showing an effect also shares a second setting, you measured
the second setting.** Every "ragged" arm was a *thresholded* arm. The isolating
A/B cost zero code — an existing flag (`--dspark-sps-row-ms 0`) makes rows free
and turns the budget into a uniform-block emitter — and the same-config c=1
pair fell out of it as a free 0.3% noise floor.

**A claim that a path cannot be verified is a claim about the disk, and disks are
cheap to check.** I filed "no markov checkpoint exists" after looking at one
directory, and that sentence kept the best-predicting draft gated at c=1 for a
week. `ls /host` was the whole investigation.

**Price a feature with a flag that removes the behaviour and keeps the
instructions, before reaching for a profiler.** `--dspark-conf-threshold 0` runs
every GEMM, copy and sync the confidence head runs and cuts nothing — one
existing flag, no code, no `nsys`, and it separated "the head executes" from "the
head decides" in one A/B. I had instead filed an nsys capture and a named suspect
against a cost the head does not pay.

**A derived quantity is not a measurement, and a decomposition built on two of
them is an invention.** `rows/step` and `step_ms` both came out of an ITL formula;
one arithmetic slip put the whole 7-arm cost table 15-40% off and turned a
per-row cost into an imaginary fixed one. `rows/step` is `block - 1` plus the
anchor, exactly, on any arm that does not truncate — a number that must equal
`c × block` is a free check, and it disagreed.

**Tokens per verify row is the number that decides a draft, and TPOT is the
number that decides shipping it.** FR wins the first by 38% and loses the second
by 34%; either metric alone picks the wrong draft. A spec baseline row without
its accept rate cannot be compared against another draft at all — which is why
`docs/baselines.md` now carries one.
