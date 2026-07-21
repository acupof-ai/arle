# hd256 q/k RMSNorm convention flip — root-caused + fixed

> Status: Shipped/Resolved (`e4d5580ca`). Agentic-rollout / long-context
> degeneration = a single kernel bug: `b4b293f0c` flipped hd256 q/k RMSNorm
> OFFSET→STANDARD, collapsing 27B attention at length. Binary bisect (clean
> adjacent flip) → fix restores `(1+w)` at 5 hd256-only sites → pod-verified base
> greedy agentic rollout emits tool calls, 7 turns, reward 1.0. A SEPARATE temp>0
> sampling defect (#167 Type-B) was the same commit's `qwen35.rs` `w-1` load,
> fixed `d703b5240`. Full record in the errors + wins entries.

## Verdict

`b4b293f0c` (OFFSET→STANDARD) was a misattributed "fix": 27B q/k_norm weights are
OFFSET (mean|w|=0.49<0.75, Metal-reference-confirmed), and its "verified vs 4B"
claim is void (hd256 kernels are 27B-only). Dropping `+1` shrank q/k ~3× →
attention collapse, length-dependent (short greedy fine → its smoke passed; long
agentic/temp>0 broke). Fix `e4d5580ca` restores OFFSET at all 5 hd256 sites.

Eight hypotheses were killed before the bisect (router/FP8-scale/FP8-value/config/
ThinkingCap-weights/temperature/prompt-render/sampler); #48's day-one relay
("b4b293f0c breaks temp>0") was right all along.

## Follow-ups

- **#167 Type-B (fixed `d703b5240`):** the same commit's `qwen35.rs` `w-1` final-
  norm load — sign-corrupts the STANDARD final-norm's negative channels → temp=1.0
  sampled-tail garbage (greedy survived, hid behind the greedy gate). Revert to
  `load_vec`. temp=1.0 grpo unblocked.
- **OPD P4:** unblocked — rejection-ce (greedy) AND grpo (temp=1.0) both runnable.
- ThinkingCap-FP8 re-evaluate fairly (it was never the problem).

## Links

- Errors (full chase + rules):
  [errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md](../experience/errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md).
- Wins (acceptance A/B):
  [wins/2026-07-20-hd256-qk-rmsnorm-offset-restore.md](../experience/wins/2026-07-20-hd256-qk-rmsnorm-offset-restore.md).
- Fix: `e4d5580ca`. Regressor: `b4b293f0c`. Bisect GOOD parent: `67e15b0a6`.
  Relay/root: #48. Temp>0 follow-up: #167 (fixed `d703b5240`).
