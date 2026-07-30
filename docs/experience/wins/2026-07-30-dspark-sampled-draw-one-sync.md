# The sampled DSpark draft needs one sync, not one per draw — and stays gated at c=1

## Context

Every other DSpark path got batched across slots this week; sampling was
excluded, and the code said why:

> Greedy resolves the whole block in one batched pass; sampling has to walk the
> chain a row at a time (it syncs per draw regardless).

That is only true with a markov head. Without one, `src` is a **precomputed**
logits row from the single block-parallel draft forward and `prev` is never read
— every draw in the block is independent, so the per-draw sync bought nothing.

## What Worked

Issue the whole block's draws, then sync once: `block - 1` fewer full pipeline
drains per draft. No kernel change, no new buffers.

## Measurement

GPU 0 (verified 0 MiB before the run), ThinkingCap-Qwen3.6-27B-FP8 +
27B-DFlash, block 6, `bench-agent-32k-16x8`, 48 req, max_tokens 214,
temperature 0.7, seed 20260416, c=1.

| | TPOT | decode tok/s |
|---|---:|---:|
| no-spec | 32.24 ms | 31.0 |
| DSpark, per-draw sync | 16.72 ms | 59.8 |
| **DSpark, one sync** | **14.25 ms** | **70.2** |

Sampled DSpark goes 1.93× → **2.26×** over no-spec. The two DSpark arms ran on
different GPUs in different sessions, so that −14.8% TPOT step is a
cross-session number; the 2.26× against no-spec is same-GPU, same-session.
Greedy gate exact=3 DET at 512/4k/16k/32k on the same binary.

## Reject — sampling does not speculate above c=1

Same fingerprint at temperature 0.7, spec vs no-spec:

| c | no-spec | DSpark | |
|---|---:|---:|---:|
| 1 | 13.1 | 15.9 | +21% |
| 8 | 70.2 | 59.3 | **−15.5%** |
| 16 | 84.1 | 61.9 | **−26.4%** |

So `--spec-max-batch` keeps clamping sampled batches to 1, and the batched-draft
path dropped the sampled branch it would never reach. Greedy at c=16 is +17% on
this model; the rejection test commits far less per verify row, and that
difference is the whole gap. **A spec-decode verdict does not carry across
temperatures** any more than it carries across accept rates.

## Rule

**A comment explaining why something cannot be batched is a claim, not a
constraint** — this one had been true of the markov path and was copied onto the
path without a markov head. Re-derive the dependency from the code that runs
before believing the loop has to be serial.
