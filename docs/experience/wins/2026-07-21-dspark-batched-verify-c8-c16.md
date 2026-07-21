# DSv4 DSpark batched anchor+verify — c=8 -36.2% (was -41.5%), c=16 stable

> Status: Shipped — batched verify path verified on H20 TP=4; 0 errors at c=8/16.
> c=8 regression improved ~5pp; c=16 no-spec baseline collapsed (server overwhelmed),
> DSpark ran clean.

## Context

The B>1 DSpark dispatch path (`executor/dsv4.rs:1898`, pre-fix) dispatched every
row individually: N serial anchor forwards + N serial verify forwards vs the
no-spec baseline's 1 batched forward. At c=8 this regressed -41.5%, at c=16
-60.4% (`2026-07-20-dspark-sliding-window-c1-win-c8-regress.md`).

## Fix

Two commits batch the two heaviest target-model phases:

- `13fe251cb` — `dspark_decode_tokens_batched`: ONE `forward_decode_batch`
  (anchor) + ONE `forward_decode_batch_verify` (all chains) over N slots,
  replacing 2N serial target forwards. Draft (small 3-stage model) and
  accept/commit (proven fold) remain per-slot.
- `9edfcb234` — capture DSpark T3 taps in `forward_decode_batch_verify` so
  Phase 4's `dspark_append_context(taps, accepted+1)` reads the full chain,
  not the stale 1-token anchor tap.
- `4e2a852b0` — `mla_attention`: when `chain_verify` is set but `token_count <= 1`
  (draft_len=0, anchor-only chain), fall through to the normal decode path
  instead of `ensure!(used, "requires FlashMLA sparse prefill")`. The sparse
  optimization only applies to multi-row chains.

## Params

- 4×H20 TP=4, GPUs 0,1,2,3; DSv4-Flash FP8 + DSpark FP8 draft, block 5, greedy
- `bench-prompts-64.jsonl` (~3.4k tok), 60s/point, max_tokens 256
- `--dspark-conf-threshold 0` (full block survives)
- DSpark server clamped slots 32→22 (draft model VRAM ~82GB/GPU vs ~75GB no-spec)

## Results

| c | No-spec tok/s | DSpark tok/s | Δ% | Prev Δ% | Errors (no-spec / dspark) |
|---|-------------:|-------------:|---:|--------:|--------------------------:|
| 8 | 146.5 | **93.5** | **−36.2%** | −41.5% | 0 / 0 |
| 16 | 32.0* | **95.4** | n/a | −60.4% | 82286 / 0 |

\* c=16 no-spec invalid: server overwhelmed (15/82303 complete, 82286 connection
errors). Not reproducible vs the 162.0 tok/s baseline — likely KV cache
exhaustion at 32 slots under high concurrency.

DSpark spec stats:
- c=8: accept_rate 0.43, drafted 7610, accepted 3270, rejected 4340
- c=16: accept_rate 0.45, drafted 18325, accepted 8224, rejected 10101

## Analysis

1. **Batched verify helps at c=8** (−41.5% → −36.2%, +5.3pp). The 2N→2 target
   forward reduction amortizes launch overhead; at c=8 the batch is large enough
   to benefit.
2. **DSpark ITL is 5–8× no-spec** (c=8: 286ms vs 49.5ms; c=16: 581ms vs 73.6ms).
   The draft forward + verify per step dominates; low accept_rate (~0.44) means
   most drafting overhead is wasted.
3. **DSpark is more stable under load**: at c=16 the no-spec server collapsed
   while DSpark handled 64/70 requests with 0 errors.
4. **Slot clamping** (32→22) reduces DSpark's concurrency ceiling; a fair c=16
   comparison needs matched slot counts.

## Rule

- Keep the batched verify path: correct, 0 errors, +5pp at c=8.
- c=1 is still the strong win (+64%); c=4 neutral; c=8/16 still negative but
  improved.
- Full c=8/16 recovery requires batching the DSpark draft forward (currently
  serial per-slot) or raising accept_rate.
- c=16 no-spec baseline needs re-measurement with a stable config.
