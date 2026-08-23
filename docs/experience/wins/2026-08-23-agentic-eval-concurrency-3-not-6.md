# Agentic eval: concurrency 3 beats 6 on the same GPU

## Context

Agent-OPD held-out eval, `Qwen3.8-27B-NVFP4` on one H20, Claude Code as the
agent. At `--eval-concurrency 6`, four of twelve tasks hit the 1800 s cap with
`edited=false` — thirty minutes of agent time that produced no file change.
Suspected a hung harness; it was queueing.

## What Worked

Dropping `--eval-concurrency` from 6 to 3. Same binary, same corpus, same
`Qwen3.8-27B-NVFP4`, same box, back to back:

| | c=6 | c=3 |
|---|---|---|
| TTFT (mean) | 19.65 s | 2.28 s |
| decode | 66.3 ms/step | 28.6 ms/step |
| tokens/step | 7.87 | 2.78 |
| aggregate | 118.7 tok/s | 97.2 tok/s |
| `pr_330` wall (passed both) | 624.7 s | 146.9 s |

Doubling concurrency buys 1.22x aggregate throughput and costs 4.3x in
per-request wall clock. For a batch scorer that trade is fine; for an agent
that must finish inside a timeout it is not, because every turn pays the TTFT
and a 30-turn trajectory pays it thirty times.

Three hypotheses ruled out first, each with its own counter-number:

- context recompute — `prefix_cache hit_rate=0.959`, ~27k tokens hit per
  request against 2.2k newly prefilled;
- KV exhaustion — `kv_free_pages=1930/15625`, and `kv_tier` reported
  `available=true` with `demoted_pages=0 promoted_pages=0`, so the L2/L3 tier
  never engaged;
- admission gating — permissive governor, `queue_depth=0`.

`forward_busy` was 1,952 s over a ~2,060 s serving window: the GPU was ~95%
busy and a new turn waited roughly 300 decode steps for its prefill to land in
a mixed step.

Because c=6 and c=3 share the same workload and the same ~27k mean context,
the pair isolates concurrency. Aggregate throughput scaling 1.22x for 2x
concurrency points at the batching path rather than long-context attention —
a single-stream decode reads the weights once per step, so batch should be
nearly free until it hits a wall. Which wall is not established here; the cell
that separates it is c=1 at 27k context, which nobody has measured yet.

## Rule

Report the measured floor, not the aggregate, when the consumer is an agent.
Aggregate tok/s is the right metric for a batch scorer and the wrong one for a
trajectory that pays TTFT once per turn — the two peak at different
concurrencies, and optimizing the wrong one silently converts a capability
measurement into a timeout measurement.
