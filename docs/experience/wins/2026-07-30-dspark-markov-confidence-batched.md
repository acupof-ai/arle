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

**Hypothesis, not a root cause:** truncation is the only thing that makes the
trunk verify's row count vary step to step, and `HiddenSlot`/`VecSlot`/
`SliceSlot::get` (`workspace.rs:56`, `:109`, `:135`) reuse a cached buffer only
on an **exact** size match — any change, including a shrink, frees and
re-allocates (`HiddenStates::zeros` = cudaMalloc + memset), the logits buffer
among them at `[248320, rows]` = 47 MB. Not settled: the one within-arm test
(steps whose row count repeats the previous step's should skip every realloc) saw
no difference, but it ran on the phase instrument below, which covers only 17% of
the step. The cheap next probe is a counter on `HiddenStates::zeros` per step.

## Two instruments that could not carry this, and one that could

Corrects the first version of this entry, which reported the excess as
**+127 ms/step paid once per step, 56% of the step, block-independent**, and named
`gemm_batch` routing a quantized `conf.weight` into `quant_linear::gemm_batch` as
the first suspect. All three parts were wrong:

- **The suspect does not exist.** `confidence_head.proj.weight` is BF16
  `[1, 5376]` in the checkpoint, so `gemm_batch` takes the plain `gemm_cuda`
  path. One safetensors header read, no GPU.
- **`rows/step` and `step @16` were miscomputed** — the old table read 81.1 rows
  and 222.2 ms where the counters give exactly 96.0 and 262.7. 96.0 is
  16 slots × 6 rows on the nose, and the phase instrument's `chain_rows`
  independently reads 96.0.
- **"Block-independent, so it is a fixed per-step cost" was an artifact.** fr6
  +202 ms and fr16 +258 ms look alike because their `rows/step` are alike
  (76 vs 83.5), not because the cost is fixed. Across four FR arms at three
  thresholds the excess per row is 2.66/3.09/3.55/2.93 ms — per row, not per
  step. And with the truncation removed it is zero, so it was never the head.
- **`ARLE_DSPARK_PHASE` cannot attribute a 30% wall-clock gap.** It syncs at
  every lap: aeon6's step goes 269.9 → **931.2 ms** and the four phases sum to
  159.20, leaving 83% outside any span. A reading off it ("fr6's step is cheaper
  than aeon6's") is not a claim about the real step.

What settled it was a flag that removes one behaviour and keeps every
instruction: `--dspark-conf-threshold 0`, matched to a head-free arm on rows and
depth. No profiler.

## Reject the truncation; `FR t0` is a champion candidate

**Reject the confidence head as a policy.** Every arm that acts on its verdict
falls below *not speculating* — FR 6.3 and 5.8 tok/s against no-spec's 8.0 — and
loosening the threshold only widens the gap, because a looser cut means a longer
variable-length chain:

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
| 1 | 102.0 tok/s | **107.2** | **+5.1%** |
| 16 | 109.43 ms TPOT | **104.73 ms** | **−4.3%** |
| 16 | 0.400 tok/row | **0.428** | **+7.0%** |

Same 96 rows, same depth 5.00, same binary — so this is the FR *weights* being
better than DFlash's, not the head, and not batching. It also beats `aeon6`
(9.5 vs 9.2 tok/s). Both ends move the same way and c=1 clears the ±3% band,
which is what a real draft-quality win looks like.

Not a default flip yet: needs the third trial per the bench spec, the c=8 point,
and a licence that shipping a checkpoint with its own confidence head switched
off is the shape we want (the alternative is retraining the head or dropping it
from the checkpoint). Correctness gate on `FR t0` block 6:
**exact=3 DET at 512 / 4096 / 16384 / 32768**, 0 miss.

## Still open

- **Block size wants opposite values at the two ends.** Head-free: block 16 wins
  c=1 (109.1 vs 102.0) and block 6 wins c=16 (9.1 vs 6.9), because at c=1 the GPU
  is idle and rows are nearly free. `--dspark-block-size` is static; the decode
  batch is not.
- **The accept rate halves at concurrency** on *every* draft (DFlash 0.509 →
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

## Rule

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
