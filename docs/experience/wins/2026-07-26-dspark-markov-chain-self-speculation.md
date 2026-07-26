# The DSpark markov path batches by speculating its own chain — argmax 8.97 → 0.14 ms

## Context

Installing the trained markov head cost **22.5% throughput** with no acceptance
gain (`2026-07-26-markov-head-online-selfrl-cannot-reach-scale.md`), which made
the head a debt before it could be an asset. The cause is a real dependency:
`bias = w2·w1[prev]` means draft row `r`'s logits depend on the token chosen at
row `r-1`, so the greedy scan drops out of the batched-argmax path
(`block_argmax` is gated on `markov.is_none()`) and pays a `[248320, 256]` gemm
plus a row memcpy and an add **per block row per slot**.

The dependency is real but the *uncertainty* is not. The same entry measured the
learned bias at 0.052 logits of within-row spread against an O(1) top-2 gap — so
the base argmax is a near-perfect predictor of each row's own predecessor.

## What Worked

Speculate the chain on itself. Guess every row's predecessor from the base
argmax, correct all rows in **one** batched embedding + gemm + add + argmax, and
re-run only while a guess disagreed with what that pass produced. Each round
confirms at least one more row, so `block` rounds is the hard bound and the
result is exactly what walking the chain produces — no approximation.

ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash, 1×H20 GPU 0, block 16,
`--spec-max-batch 16 --max-running-requests 16`, c=8, open-perfectblend eval
split, greedy, trained head installed via `--dspark-markov-init`. Two interleaved
rounds, `ARLE_DSPARK_PHASE` medians over ~2400 draft ticks per arm:

| draft sub-phase (ms/tick) | per-row | batched | |
|---|---:|---:|---|
| embed | 0.01 | 0.01 | |
| prep | 0.39 | 0.40 | |
| attn | 0.54 | 0.55 | |
| mlp | 1.07 | 1.07 | |
| head | 0.70 | 0.70 | |
| **argmax** | **8.97** | **0.14** | **64×** |
| **draft total** | **11.68** | **2.88** | **−75%** |

Round 2 reproduces it to two decimals (8.96 → 0.14). 0.14 ms is one batched
pass, so the speculation settles on the first round essentially always — as the
0.052-logit bias predicted. That is the load-bearing number: the worst case
(`block` rounds) costs exactly what the per-row path did, so the win *is* the
round count, and 0.14 ms is what bounds it.

A simplify pass (`0ade41244`) followed: greedy became a slice copy out of
`dspark_block_greedy` rather than a per-row scan, the round loop stopped
allocating a device buffer per argmax (`argmax_rows_into`), and the four markov
scratch slots gained block-shaped twins so the row-shaped path (sampling, and
the confidence head in the same call) no longer frees and rezeroes a
`[vocab, block]` buffer every tick. Re-measured: argmax 0.13 ms, draft total
2.85 ms, and all four output hashes still identical.

**Output is bit-identical.** Four eval prompts, 300 greedy tokens each, sha256
over the whole `choices[0]` object: all four hashes match between the two
binaries in both rounds. `k_mean` 3.823 (per-row) vs 3.827 (batched) over ~4200
rejections — the same chain, as designed.

## Problems

- **Wall tok/s on this box is not a usable measurement right now.** Interleaved
  3×3 rounds swung 117–205 tok/s within a single arm, and one round returned the
  same 262.0 s wall for both arms — an external wall, not the code. Excluding
  that round: per-row 138.5, batched 187.6, no-markov 190.8 tok/s, i.e. the tax
  drops from ~27% to ~2%. Directionally consistent with the phase split and with
  the earlier 22.5% reading, but the phase medians are the evidence.
- Sampling keeps the per-row path. It syncs and D2Hs a token per row regardless,
  so batching the bias buys nothing there; the same speculation would apply if
  that ever changes.
- The confidence head's own per-row markov lookup
  (`dspark_confident_prefix_len`) is untouched — no checkpoint here ships one,
  so there is nothing to measure. It is a strictly easier batch than this one
  (`prev_tokens` is fully known before the call, so no fixed point to iterate),
  and it would re-spend part of this win on any checkpoint that does ship one.
- Serving under `--dspark-train` runs the full correction against a `w2` that
  is identically zero at cold start, so the corrected argmax equals the base
  argmax by construction. A `w2_all_zero` flag captured at load would skip it.

## Rule

A serial data dependency is not automatically a serial *computation*. When the
dependency is cheap to predict — and the previous entry had already measured
exactly how cheap, 0.052 logits against an O(1) gap — guess it, verify the guess
in the same batched pass, and re-run only on disagreement. That keeps the
result exact while paying batch cost once. The measurement that killed the head
as a quality lever is the same measurement that unlocked its performance fix.
