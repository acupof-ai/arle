# DSv4 DSpark sliding-window draft latent — c=1 +64%, c=4 +3%, c=8/16 regress

> Status: Shipped — sliding-window fix verified; no crash at any concurrency.
> c=1 win confirmed; c=8/16 regression was measured on the July-20 path.
> Superseded 2026-07-21 for current-path attribution: batched anchor + one batched target verify is now implemented. Keep the July-20 measurements, but “no batched verify” is historical; corrected c=8 is 93.5 vs valid no-spec 101.2 tok/s (−7.6%). Draft generation remains per-slot.

## Context

DSpark draft latent KV was sized `max_seq_len + block` (linear in prompt
length), so long prompts OOMed the per-slot budget and the
`--dspark-max-prompt-tokens 64` router disabled DSpark on production ~2.8k tok
prompts. The sliding-window fix sized latent KV to `sliding_window + block`
(fixed) and shifted the live context forward (memmove) on overflow. The
`--dspark-max-prompt-tokens` router was then deleted (DSpark runs for all
prompt lengths).

## Params

- 4×H20 TP=4, GPUs 2,3,4,5; DSv4-Flash FP8 + DSpark FP8 draft, block 5, greedy
- `bench-prompts-64.jsonl` (~3.4k tok), 60s/point, max_tokens 256
- No `--dspark-max-prompt-tokens` (router deleted in `591772a43`)

## Results

| c | No-spec tok/s | DSpark tok/s | Δ% | Errors |
|---|-------------:|-------------:|---:|------:|
| 1 | 38.9 | **63.7** | **+63.8%** | 0 |
| 4 | 74.7 | **77.1** | **+3.2%** | 0 |
| 8 | 127.6 | 74.7 | −41.5% | 0 |
| 16 | 162.0 | 64.2 | −60.4% | 0 |

Zero errors / crashes / illegal-address at all concurrencies. The prior
`CUDA_ERROR_ILLEGAL_ADDRESS` at c=4+ is resolved by the sliding-window
overflow fix (rebase + memmove on oversized prefill chunks).

## Regression (c=8/16)

Not a crash — throughput loss vs the already-batched no-spec baseline. DSpark
draft generation is serial (N stages × depth sequential forwards, no batching
across slots), and the B>1 dispatch path (`executor/dsv4.rs:1898`) dispatches
every row individually without batching the verify phase. See
`docs/experience/errors/2026-07-19-dsv4-mtp-dspark-high-concurrency-regression.md`.

## Rule

- DSpark c=1 is a strong win (+64%); keep the sliding-window change.
- c=4 is neutral-to-positive (+3%).
- c=8/16 regression is structural, not a correctness issue; do not revert.
- Default flip requires batching the draft + verify phases (separate work).
