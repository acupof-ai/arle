# Metal DSpark: window the prefill target hidden + why the breaker fires — metal, 2026-08-26

> Status: Confirmed (local M4 Pro 48 GB)
>
> Baseline: `59d91a2f6`, `arle serve --backend metal --model-path
> LiquidAI/LFM2.5-8B-A1B-MLX-4bit --draft-model LiquidAI/LFM2.5-8B-A1B-DSpark
> --kv-cache-dtype bf16 --kv-disk=<dir> --chunked-prefill-size 1024`,
> M4 Pro 48 GB. 15 030-token prompt, 126 generated tokens.

## Context

`DSpark disabled after 3 consecutive rejections (slot 0)` on every request. The
obvious suspect was the prefix cache landed in `59d91a2f6` — a restored prefix
leaving the draft out of sync with the target. It is not that:

- the warning fires once per request in the **pre-change** baseline log, where
  `licensed_blocks=0` on every lookup and no prefix is ever restored;
- it fires on a **cold** request in the same run as a warm one;
- it fires with a 20-token prompt that never touches the cache.

The breaker is reporting the truth. `INFER_METAL_DFLASH_DRAFT_TRACE=1`,
15 k context, `block_size=4`:

```
matched=1 accepted=2/4  draft_ms=274.32 verify_ms=0.35 accept_ms=27.95
matched=0 accepted=1/4  draft_ms=22.71  verify_ms=0.32 accept_ms=23.63
matched=0 accepted=1/4  draft_ms=9.71   verify_ms=0.22 accept_ms=23.70
matched=2 accepted=3/4  draft_ms=9.44   verify_ms=0.30 accept_ms=22.83
```

`verify_ms` measures nothing — the graph is lazy and the cost lands in
`accept_ms`, the first `materialize_i32_tokens` sync. A block really costs
~33 ms (9.5 draft + 23 verify) and returns a mean of ~1.7 tokens. Plain decode
is 8.9 ms/token (112 tok/s measured with `--no-speculative`), so a block has to
accept 3.7 tokens to break even, out of a maximum of 4. It accepts 1.7. DSpark
is a ~2x loss on this checkpoint at this block size, and the circuit breaker is
the only reason the damage stops at three blocks.

## What worked

The trace does expose one real defect: the **first** block after prefill costs
274 ms, 28x every block after it. The draft prepends every target-hidden row
into its own KV, and prefill seeded the store with the *whole prompt* — 15 030
rows — while decode has always rolled it to the last 64
(`MetalSlotState::roll_target_hidden`). Prefill now goes through the same
window:

- `executor.rs`: `submit_prefill` calls `roll_target_hidden` instead of
  assigning/concatenating the raw capture, and the window size becomes a named
  constant (`DFLASH_TARGET_HIDDEN_ROWS`).
- `slot.rs`: `roll_target_hidden` handles an empty history. It used to do
  `take().unwrap_or_else(|| new_rows.clone())` and then concatenate old with
  new, so the first seed was silently doubled. Unreachable while prefill always
  assigned first; reachable the moment prefill routes through it.

| | first-block draft_ms | decode tok/s (4 requests) |
|---|---|---|
| DSpark, prompt-wide seed | 274 | 89.2 / 100.3 / 93.0 / 82.1 |
| DSpark, 64-row seed | 47.7 | **108.9 / 108.7 / 105.6 / 100.0** |
| `--no-speculative` | n/a | 112.0 / 112.0 / 112.5 / 110.4 |

TTFT is unchanged by the window (cold 11.0 s, warm 0.552 / 0.545 s, after
restart 0.732 s) — the 274 ms sits in the first *decode* block, not TTFT.

Correctness on the windowed build: restored-prefix output byte-identical to
cold; needle ladder 512 / 2000 / 4000 / 8000 x3 exact 3/3 DET at every length;
`cargo test -p infer-metal --features metal` 12/12.

## Recommendation

On LFM2.5-8B-A1B at long context, serve **without** `--draft-model`: 111.7
tok/s mean and flat, against 105.8 with DSpark after this fix (and 91-102
before it). The window closes most of the gap but cannot close it — that needs
acceptance near 4/4, which is a drafter-quality question, not an engine one.

## Rule

- A circuit breaker firing every request is a measurement, not a bug. Read what
  it measured before hunting for what broke it — this one was right, and the
  prefix cache it was blamed on was not involved.
- Lazy graphs put the cost in the next sync, not in the section you timed.
  `verify_ms=0.3` next to `accept_ms=23` is one number, mislabelled.
- State that decode keeps in a window must be *seeded* through the same window.
  Prefill wrote 15 030 rows into a store every other writer capped at 64, and it
  cost 28x a normal block exactly once per request — big enough to matter,
  rare enough to hide.
- Speculative decode has a hard break-even: block cost / single-token cost. At
  33 ms vs 8.9 ms with `block_size=4` the drafter needs 93% acceptance to be
  worth loading. Measure that ratio before tuning anything else.
