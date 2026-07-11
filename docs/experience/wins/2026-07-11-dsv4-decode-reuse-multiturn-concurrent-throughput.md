# DSv4 decode-region reuse — multi-turn concurrent throughput +25% at c=16 → default flip

## Context

The measurement guidellm couldn't produce (its synthetic prompts are
independent — no shared growing prefix, reuse never fires). The multi-turn
concurrent harness (`eval_harness multiturn_concurrent`, token-preserving
history feed-back) A/B's `--dsv4-decode-reuse` OFF vs ON on a real high-
concurrency multi-turn agent shape. H20 TP=4 GPUs 4-7, DSv4-Flash-FP8, HEAD
`d9cc1bfbf`, binary `873101640824…`, CONCURRENCY=1/4/8/16, TURNS=4,
PROMPT=500, GEN=128.

## Result — the reuse win scales monotonically with concurrency

Δ% (ON vs OFF; TTFT/TPOT negative = faster, agg positive = higher throughput):

| c | TTFT p50 | TPOT p50 | TPOT p99 | agg tok/s |
|---|---|---|---|---|
| 1 | −35.2% | −4.3% | −4.4% | +5.7% |
| 4 | −46.8% | −8.6% | −10.5% | +12.4% |
| 8 | −52.7% | −13.1% | −16.4% | +19.1% |
| 16 | −52.5% | −18.7% | −18.9% | **+25.3%** |

At c=16: **TTFT p50 halved, TPOT −18.7%, aggregate throughput +25.3%.** The win
grows with concurrency — more concurrent multi-turn conversations = more
prior-turn KV reused = less prefill contention. TTFT p99 is flat (~+1-3%, tail
is queueing under load, not prefill).

## Reuse fired (the mechanism, not page-rounding)

Per-conversation turn-2 reuse: OFF = floor + **1 page** (64 tok, prompt
page-rounding only); ON = floor + **2 pages** (128 tok) — the extra page is the
prior turn's GENERATED region, the decode-region reuse the flag enables. (The
`reuse_ok > floor` gate reads true on both because page-rounding alone clears
floor by one page; the delta magnitude 1-vs-2 pages is the real signal, and the
perf table confirms it fired.) Needle-exact throughout (no corruption).

## No downside (the flip is safe)

- Independent-prompt A/B (guidellm, prior run): reuse-ON is byte-wash on the
  non-reuse path — c=1 arms identical, the finish-capture D2H costs ~0 when
  reuse doesn't fire.
- ON-path correctness pod-verified across the campaign: OFF-vs-ON crash-repro
  24/24, needle-exact multi-turn + concurrent, zero `seq_len != append_pos`.
- Concurrency ceiling ≥16 for this shape (scheduler queues, not rejects, at
  mtt=2048) — the fixed-band admission ceiling was not reached; #160 (band
  cascade) is not the c≤16 bottleneck.

## Verdict — default flip LICENSED

Two binding shapes: multi-turn concurrent (big win, +25% at c=16) + single-shot
(no regression). `--dsv4-decode-reuse` flipped **default ON**. The flag stays
(`--dsv4-decode-reuse false` restores the old path). This is the campaign's
delivered throughput win — the lever was the reuse feature itself, not the
killed pinned-DRAM (#5) / admission-watermark (#6) knobs.

## Rule

The throughput lever for a high-concurrency AGENT workload is prefix/decode
reuse (measured on a multi-turn concurrent harness), not admission/alloc micro-
tuning — and it must be measured on multi-turn, not guidellm's independent
prompts. The win scales with concurrency (more conversations → more reuse).
