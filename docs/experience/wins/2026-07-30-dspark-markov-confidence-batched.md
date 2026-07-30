# The markov settle and the confidence head batch across slots — and the accurate draft still loses on wall clock

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

## The markov settle is free; the confidence head costs 130 ms a step

Six arms, one binary, one fingerprint, run concurrently on one GPU each (180
cores, load 16, so the only shared resource is idle). All three drafts have
identical geometry — 5 layers, hidden 5120, intermediate 17408, 32 heads, 8 kv,
head_dim 128 — so block size and the two heads are the only variables, and
`dspark-aeon` (markov, no confidence) separates them.

Decode tok/s, and the marginal cost of a verify row against the no-spec step
(115.2 ms at c=16):

| arm | heads | block | c=1 | c=8 | **c=16** | rows/step @16 | **ms/row @16** |
|---|---|---:|---:|---:|---:|---:|---:|
| no-spec | — | — | 34.8 | 11.8 | 8.0 | 0.0 | — |
| DFlash | none | 6 | 102.0 | **16.5** | **9.1** | 81.1 | 1.32 |
| DFlash | none | 16 | 109.1 | 12.6 | 6.9 | 221.2 | 0.86 |
| aeon | markov | 11 | **113.0** | 14.2 | 7.7 | 149.5 | **1.14** |
| FR | markov+conf | 6 | 92.3 | 11.8 | 6.3 | 47.6 | **4.04** |
| FR | markov+conf | 16 | 97.0 | 10.9 | 5.8 | 41.8 | **4.47** |

**The batched markov settle costs nothing** — `aeon` sits at 1.14 ms/row, among
the head-free arms, and this is the first measurement of a markov checkpoint
speculating above c=1 at all.

**The confidence head costs a fixed ~130 ms/step at c=16, block-independent.**
Priced at DFlash's row cost, FR block 6 should step in 178 ms (115.2 + 47.6×1.32)
and block 16 in 170; they measure 307.7 and 302.1. Equal excess at unequal blocks
means a per-step cost, and `dspark_confident_keeps` runs exactly once per step.
Not diagnosed further — the device work in it is one 5376×96 GEMM and two copy
launches, which cannot be 130 ms, so the next step is an nsys capture of the FR
draft, not another guess.

The prize is large enough to be worth it: at DFlash's row cost FR block 6 reaches
**10.8 tok/s at c=16 against the champion's 9.14 (+18%)**, because its accuracy
*comes from* the truncation — 0.568 tokens per verify row against DFlash's 0.400
is the largest accuracy gap measured on this model, with plain decode's
break-even at 1.0.

## Reject, for now — DFlash block 6 stays the default draft

Every markov arm loses at c=16 today, and two lose to *not speculating*: FR 6.3
and FR 16 5.8 against no-spec's 8.0. `--spec-max-batch` is a user flag, so a
markov user wanting the old behaviour sets it to 1; the shipped default draft has
no confidence head and is unaffected.

Loosening the threshold makes it worse — the head at 0.5 already cuts where
marginal accuracy stops covering a verify row:

| threshold | depth | mean k | tok/row | TPOT c=16 |
|---|---:|---:|---:|---:|
| 0.50 | 4.22 | 1.893 | 0.554 | 172.63 ms |
| 0.30 | 12.13 | 3.392 | 0.335 | 247.76 ms |
| 0.15 | 14.87 | 3.349 | 0.274 | 251.94 ms |

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
  time. FR is hit hardest — `dspark.rs:676` warns its 5/5 declared full-attention
  layers all run the 2048 sliding window, so its measured accept is a floor.
- The two draft attention kernels (`dspark.rs:1290-1340`) are the only per-slot
  launches left in the batched draft — 160 at c=16. Same shape as the varlen
  conv1d/GDR pair already built, so a `blockIdx.y`-per-slot pointer table
  collapses them to two per layer.

## Rule

**A claim that a path cannot be verified is a claim about the disk, and disks are
cheap to check.** I filed "no markov checkpoint exists" after looking at one
directory, and that sentence kept the best-predicting draft gated at c=1 for a
week. `ls /host` was the whole investigation.

**Tokens per verify row is the number that decides a draft, and TPOT is the
number that decides shipping it.** FR wins the first by 38% and loses the second
by 34%; either metric alone picks the wrong draft. A spec baseline row without
its accept rate cannot be compared against another draft at all — which is why
`docs/baselines.md` now carries one.
