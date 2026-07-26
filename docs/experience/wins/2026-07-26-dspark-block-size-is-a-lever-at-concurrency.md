# DSpark block size IS a lever — once the batched verify leaves the latency-bound regime

## Context

[2026-07-25-dspark-block-truncation-is-not-a-throughput-win](../errors/2026-07-25-dspark-block-truncation-is-not-a-throughput-win.md)
measured block 4/6/8/16 **at c=1**, found throughput falling monotonically as the
block shrank, and concluded the flag "is not a lever; it is an instrument". That
conclusion is correct at c=1 and was generalized one step too far: it fixed a
default for concurrent serving from a single-stream measurement.

Attribution came first. Splitting `commit` (16.2 ms, 17% of the c=8 tick) showed
`accept` at 81% of it, and splitting `accept` showed **k_med = 2 out of depth
15** — the draft's useful reach is two tokens, and the other 13 rows ride a
verify forward that can never commit them.

Same rig throughout: ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash, 1×H20
GPU 0, `--spec-max-batch 16 --max-running-requests 16`, greedy, `conc_drive.py`.

## What Worked

Aggregate tok/s, 3 interleaved trials per arm (16, 8, 16, 8, …), medians:

| conc | block 16 | block 8 | Δ |
|---|---:|---:|---:|
| 1 | 63.8 (65.6/63.8/63.2) | 64.3 (65.7/64.3/64.0) | +0.8% |
| 4 | 88.9 (91.5/88.9/88.9) | 92.2 (89.6/92.2/92.5) | +3.7% |
| 8 | 87.2 (89.3/83.7/87.2) | **93.1** (91.8/94.6/93.1) | **+6.8%** |

Block 8 wins c=8 in all 3 trials and c=4 in 2 of 3; c=1 is a wash. A wider sweep
put 6 and 8 together at the top and 4 far below (c=8: 67.1 / 93.4 / 93.7 / 87.1
for block 4 / 6 / 8 / 16).

The mechanism, measured at c=8 with the phase timer:

| | block 16 | block 8 |
|---|---:|---:|
| k_med (accepted drafts per chain) | 2.0 | **2.0** |
| chain rows per tick | 96 | 48 |
| verify | 62.09 ms | 39.11 ms (−37%) |
| commit | 16.20 ms | 11.59 ms (−28%) |
| draft | 14.73 ms | 14.59 ms (−1%) |
| tick | 96.02 ms | 67.67 ms (−30%) |

The accepted prefix is identical at both widths, so block 8 buys the same tokens
for 30% less tick. Halving the rows cuts verify by only 37%, not 50% — the
forward is still partly latency-bound at 48 rows, which is exactly why the c=1
measurement (17 rows) saw it as free.

`draft` is flat across the halving: the draft head's cost is weight reads and
launches, not columns. That is consistent with the ragged-window result — after
`attn` collapsed, what remains is `mlp` + `head` re-reading draft weights once
per slot.

## Problems

- **This does not overturn the c=1 finding.** The old entry's root cause — a
  block shares one verify forward, so discarded tokens are not discarded work —
  holds whenever that forward is latency-bound. It stops holding once B·block
  fills the GPU. The error entry now carries a pointer to this crossover.
- No default flip here. `--dspark-block-size` unset takes the checkpoint value;
  changing that needs the long-agent re-measure (bench spec §3.3) plus a needle
  gate, and this is one model pair on short prompts.
- Block 4 collapses (c=1 40.0 tok/s, −38%) and was not diagnosed.
- k_med = 2 at both widths is the deeper problem: DFlash reaches two tokens. A
  smaller block is damage control, not a fix — raising acceptance is.

## Rule

A speculative-decoding parameter measured at c=1 is measured in the wrong
regime. Batch width multiplies the block: 16 rows at c=1 and 16 rows at c=8 are
136 rows, and the second one is not free. Re-run any spec-decode knob across the
concurrency axis before fixing a serving default from it — and attribute the
phase first, which is how `k_med = 2` surfaced at all.
