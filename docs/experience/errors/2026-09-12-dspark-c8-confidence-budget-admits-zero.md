# DSpark c=8 zero acceptance: the confidence budget admits no first draft — scheduling candidate, pending the stats readout

Date: 2026-09-12. Task `dspark-accept-collapse`: `RadixArk/Qwen3.8-27B-DSpark`
drafting for `Qwen3.8-27B-FP8` accepts 13% of drafts at c=1 and 0% at c=8
(`2026-08-23-dspark-rope-fix-was-not-the-cause.md`). Status: **scheduling
candidate, not established** — one by-design path can print the 0.0000 number;
whether it fired is decided by one `/v1/stats` snapshot, scripted in
`scripts/dspark_c8_accept_branch.py`.

## Context

Kernel hypotheses have thinned: the kernel-level ring-attention gate passed
its first GPU run, the sampler gate failure was a gate bug, and the
whole-drafter-step gate (#374) covers cross-slot arithmetic. The unexamined
competing hypothesis is that the drafter behaves correctly and a scheduling
rule — the goodput budget that decides how many draft rows even reach the
verify — drives the proposal length to zero as concurrency rises.

The served drafter has the confidence head: the model store's
`Qwen3.8-27B-DSpark/config.json` has `block_size 7`, 40 q / 8 kv heads,
head_dim 128, `markov_rank 256`, `enable_confidence_head true`,
`confidence_head_with_markov true`. So the
budget described below is live on the 13%/0% run; an absent head would keep
the whole block and this mechanism could not fire.

## The site and the arithmetic

`dspark_verify_lens` (`crates/qwen35-spec/src/lib.rs:1299-1322`) implements
the DSpark §3.2.2 goodput budget. Every slot's cumulative per-row survival
probabilities (sigmoid of the confidence head, cumprod'd in
`confidence_keep_lengths`, `crates/infer-model/src/dspark.rs:140-167`) are
pooled across ALL r slots in the tick and sorted; a draft row is verified only
while adding it raises

```
Θ(k) = (r + Σ top-k survival) / (bias_ms + row_ms·(r + k))
```

over the no-spec baseline Θ0 = r/(bias + row_ms·r). The additive verify-step
cost model `step_ms = bias + row·verify_rows` uses the H20
ThinkingCap-27B c=16 profile `bias_ms = 211`, `row_ms = 0.53`
(`crates/qwen35-spec/src/lib.rs:1283-1290`; `--dspark-sps-bias-ms` /
`--dspark-sps-row-ms`, `crates/cli/src/args.rs:758,762`).

A token enters the pool iff its survival exceeds the marginal bar
`row_ms · Θ_at_k`. For the FIRST pooled draft row (k=1, prior sum 0) that bar
is `row_ms · Θ0`:

| r (slots) | Θ0 = r/(211+0.53r) | first-row admission bar = 0.53·Θ0 |
|---:|---:|---:|
| 1 | 1/211.53 = **0.004728** tok/ms | **0.002506** |
| 8 | 8/215.24 = **0.037168** tok/ms | **0.019699** |

The bar rises **7.86x** from c=1 to c=8 (the ratio equals Θ0(8)/Θ0(1)).
A slot whose first-row confidence survival sits in `[0.0025, 0.0197)` is
admitted — drafted and verified — at c=1 and refused at c=8. Refusal means
`keeps[s] = 0`; the chain is truncated to the bare anchor
(`crates/infer-cuda/src/qwen35/dspark.rs:1450` batched, `:1182` single). A
bare-anchor chain contributes nothing to the `drafted` counter (the counter
counts accepted+rejected draft rows only, `spec_accept_totals`,
`crates/infer-plan/src/spec.rs:97-99`, aggregated at
`crates/infer-cuda/src/executor.rs:268-280`), and the acceptance bench prints
`accepted/drafted` as `0.0000` when drafted is 0
(`scripts/bench_dspark_accept.py:106-108`).

The two observations are therefore mutually compatible: the same weakly
calibrated first-row confidence around 1-2% accepts ~13% at c=1 (draft rows
proposed, prefix hits 13% of the time) and proposes nothing at c=8. With
well-calibrated confidence (~0.13) the cut settles at ~0.13 and every slot
still keeps at least one row at r=8, so this mechanism does not fire for a
confident head. The head's actual first-row confidence distribution is not
readable from code and is the unknown this entry does not paper over.

## Every other by-design batch-sensitive site

| Site (file:line) | What it computes | c=1 | c=8 | Can it zero acceptance alone |
|---|---|---|---|---|
| goodput budget `dspark_verify_lens` (`qwen35-spec/src/lib.rs:1299`), truncation `qwen35/dspark.rs:1450,:1182` | pooled survival cut, keep length per slot | bar 0.002506 | bar 0.019699 | **Yes, displayed-rate**: keeps=0 → drafted=0 → 0.0000 |
| concurrency gate `decide_decode` (`infer-plan/src/spec.rs:240-269`), default `spec_max_batch=16` (`infer-seam/src/runtime_flags.rs:28-32`) | route to Plain above the gate | 1≤16, Dspark | 8≤16, Dspark | No |
| KV-class gate (`infer-plan/src/spec.rs:251-252`) | batched Dspark only on BF16 FA3/no-FA3 | BF16 (CUDA Auto→BF16, `executor.rs:105`) | same | No for this run (BF16 confirmed in boot log); would zero c=8 on an FP8 serve |
| block-size clamp (`qwen35/dspark.rs:466-477`) | min(config 7, `--dspark-block-size`), load time | 7 unless flagged | identical | No |
| per-row seed conditions (`executor/qwen35.rs:1843-1850`) | paged pool, no top_logprobs, chain fits, `pending==last_token`, `kv_seq_len-ctx_end ≤ 7` | checked per row | not a function of batch | Not by batch size; but mixed prefill/decode ticks historically drafted almost nothing (11 chains/3440 tokens, `2026-08-22-batched-dspark-quant-kv-verify-loses.md`) — a second drafted=0 channel via ticks that never seed |
| ≥2 greedy rows to batch (`dspark_draft_plan`, `infer-plan/src/spec.rs:167-180`) | below 2 rows use the single-slot draft | single path | batch path | No — the fallback still drafts |

Scheduling changes what gets VERIFIED (the keep length and whether a row
seeds). It cannot make a proposed draft token unequal to the trunk argmax:
there is no batch-dependent arithmetic on the accept scan
(`spec_accept_greedy`, `infer-plan/src/spec.rs:126-151`).

## Decision rule — one stats snapshot at c=8

`GET /v1/stats` → `spec_decode.{chains,drafted,accepted}`:

1. `chains = 0` over a decode window — **rows not seeding**: mixed-step
   starvation or a seed-condition mismatch; read the `[dspark-seed]` info
   lines (`executor/qwen35.rs:1855-1865`) for the per-row reason.
2. `chains > 0, drafted = 0` — **budget zero-keeps**: every chain was
   truncated to its anchor. Scheduling/confidence calibration explains the
   number; the next move is a confidence-distribution probe or an SPS
   bar sweep, not a kernel fix.
3. `chains > 0, drafted > 0, accepted = 0` — drafts are proposed and
   rejected: scheduling is exonerated, the verify-side cross-chain KV
   isolation hypothesis survives, and the whole-step gate
   (`dspark_drafter_batch_invariance`, #374) plus the GPU batch are the
   instrument.

`scripts/dspark_c8_accept_branch.py` automates the c=1/c=8 run and prints
which branch landed. The run is pending-remote (no free H20); the bench
prereg row fixes this hypothesis before the counters are read.

## Rule

A displayed acceptance rate of 0.0000 is `accepted/drafted`, and drafted=0
means no draft row was verified — a scheduling verdict, indistinguishable in
the rate alone from drafts that all missed. Read the three counters before
naming the kernel.
