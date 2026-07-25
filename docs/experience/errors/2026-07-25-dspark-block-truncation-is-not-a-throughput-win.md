# DSpark block truncation is not a throughput win — the cost unit is the verify forward, not the token

## Context

`--dspark-block-size` (49b02be24) landed on the premise that block 16 at
accept_rate ~0.21 keeps only ~3.3 tokens per chain, so **79.5% of drafted work is
discarded** — and that shrinking the block would recover it. It was also pitched
as the fixed-length stand-in for a confidence head, and as the baseline that head
would have to beat.

Measured c=1 on 8×H20 GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash
draft, 40 requests × 64 max_tokens, temperature 0, `--max-running-requests 8`,
training OFF, one serve per block size:

| block | drafted | accepted | accept_rate | accepted/verify | median tok/s | aggregate tok/s |
|-------|---------|----------|-------------|-----------------|--------------|-----------------|
| **16** (default) | 7800 | 2054 | 0.263 | **4.21** | **78.9** | **73.5** |
| 8  | 4179 | 1975 | 0.473 | 3.78 | 75.8 | 72.7 |
| 6  | 3320 | 1906 | 0.574 | 3.45 | 69.4 | 69.2 |
| 4  | 2457 | 1740 | 0.708 | 2.83 | 41.9 | 42.0 |

accept_rate rises exactly as predicted (0.263 → 0.708). Throughput falls
monotonically. **Block 16 is the fastest configuration measured.**

## Root Cause

The waste was counted in the wrong unit. Discarded *tokens* are not discarded
*work*: every position in a block — accepted or not — rides in the **same single
batched verify forward**. At these widths that forward is latency-bound, so 16
rows cost nearly what 4 rows cost. Shrinking the block does not remove work, it
removes yield: accepted tokens per verify forward drop 4.21 → 2.83, and the
verify count per generated token rises to match.

The one real per-position cost is the draft head's autoregressive steps, which
are cheap relative to a trunk verify.

## Fix

Keep the flag (it works — `block_size 16 -> 8` confirmed in the serve log) as a
measurement knob, keep the default at 16, and drop the throughput claim. The flag
is not a lever; it is an instrument.

Consequence for the confidence head: its motivation cannot be "stop drafting past
the accepted prefix to save compute" — that saving does not exist at the verify
level. If it is worth building, the value must be in saving draft-head serial
steps, or in raising block size above 16 where per-row verify cost finally bites.
The trend here says **larger** blocks are worth probing, which the current
downward-only clamp cannot express.

## Rule

Before claiming discarded work is recoverable, name the unit that actually costs
money. Tokens are not forwards: in speculative decoding a whole block shares one
verify pass, so a low accept_rate is not by itself waste — measure
accepted-per-forward, and only then decide whether truncation buys anything.
