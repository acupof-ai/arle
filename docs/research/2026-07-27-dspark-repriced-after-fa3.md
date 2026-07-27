# DSpark, repriced

FA3 paged decode cut the decode step 76.98 → 27.94 ms at c=1. Every DSpark
number measured before that was priced against a step 2.8× too expensive.

## What changed

| 32k long-agent, total tok/s | DSpark vs no-spec |
|---|---:|
| c=1 | **+1.5%** |
| c=8 | **−6.3%** |
| c=16 | **−7.1%** |

Before FA3, DSpark was +57.5% at c=1 on short prompts and +2.04× on this
dataset. It is now a rounding error at c=1 and a **loss** at every concurrency
that matters.

Acceptance did not move — same draft, same `k`. Speculation's value is
`(k+1) × step_cost` saved against `verify_cost`; shrinking `step_cost` shrinks
the numerator. **DSpark was never mostly a speculation win.** It was paying for
a kernel defect, and two thirds of its edge belonged to the kernel.

At c≥4 the ceiling is elsewhere entirely: TPOT 1177.9 ms against ITL p50
63.89 ms means ~95% of the decode span is queueing behind other requests'
chunked prefill. **Speculation makes you compute faster, not wait faster**, so
it cannot touch that 95%.

## What this kills

- **Anything justified by "decode is expensive at c=1."** The online markov-head
  work already failed on its own terms
  ([entry](../experience/errors/2026-07-26-markov-head-online-selfrl-cannot-reach-scale.md));
  at 1.33× a retry needs a much larger win to clear the bar.
- **Block-size tuning as a headline.** Block 8 beats 16 by 24.9% decode tok/s at
  c=8 and loses at c=1 ([entry](../experience/wins/2026-07-26-dspark-block-size-is-a-lever-at-concurrency.md)),
  but it is a ±15% knob on a 1.33× arm inside a 5%-of-wall-clock budget. Keep the
  finding, do not flip a default on it.

## What survives

**Candidate width is still the one real lever.** The rank probe measured the
trunk's token inside the draft's top-2 **47.0%** of the time
([entry](../experience/wins/2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax.md)):
one alternative per position projects `E[k]` 2.19 → ~5.1. DSpark's block is a
single non-causal forward, so the draft side is free — only verify pays.

It is blocked, and the blocker is structural: Qwen3.6 has 48 gated-delta-net
layers whose recurrent state is path-dependent (~126 MB/slot), so a 16-node
candidate tree needs ~2 GB/slot. Medusa/EAGLE-style trees assume pure
transformers. Nothing here is cheap.

## Order

1. **Stop spending on DSpark for concurrent serving.** It is measured negative at
   c=8 and c=16 against the champion row. Keep it available for c=1 latency
   work, where it is +1.5% and costs nothing to leave in.
2. **Scheduling.** The machine saturates at c=8 (+1.0% from c=8 to c=16 while
   ITL p90 reaches 8.1 s). That is where the remaining throughput is.
3. **Revisit tree speculation only if a single-stream latency target appears.**
   The 47% top-2 rescue is real, but it buys `E[k]`, and `E[k]` is now worth a
   third of what it was — against ~2 GB/slot of GDN state.

## Rule

Re-price a speculative-decode result whenever the step cost moves. Acceleration
from speculation is a ratio against the thing it replaces — a kernel fix
underneath silently deflates it, and a stale ratio will justify work that no
longer pays.
