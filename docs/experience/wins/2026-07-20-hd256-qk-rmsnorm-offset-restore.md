# hd256 q/k RMSNorm OFFSET restore unblocks agentic-OPD

> Fix `e4d5580ca` (reverts the convention flip in `b4b293f0c`). Pod-verified.

## Context

Agentic-OPD rollouts on Qwen3.6-27B degenerated (no tool calls, no edits) and
temp>0 completions saladed at length. Root-caused by binary bisect to a single
kernel commit `b4b293f0c`, which flipped hd256 q/k RMSNorm from OFFSET
`x·rms·(1+w)` to STANDARD `x·rms·w`, shrinking 27B q/k ~3× (weights are OFFSET:
mean|w|=0.49<0.75) and collapsing attention at length. Full chase +
eight-hypothesis retro in the paired errors entry.

## What Worked

Restore `(1+weight)` at all 5 hd256-only q/k sites; 4B/hd128 and the MTP
`pre_fc_norm` fix untouched.

**Acceptance A/B (pod, isolated sm_90 build, HEAD e4d5580ca):** base
Qwen3.6-27B-FP8, greedy agentic cc-harness rollout on the sqlparse task.

| | before (b4b293f0c / current champion) | after (e4d5580ca) |
|---|---|---|
| tool calls | none (drifts to generic chat) | proper `<tool_call>` Glob/Grep/Read |
| turns | 1 | 7 |
| edited / passed | false / 0 | true / hidden tests pass |
| reward | 0.0 | 1.000 |

Matches the GOOD bisect parent `67e15b0a6`. Greedy temp=0 control coherent to
2000 tokens. **Agentic-OPD lane unblocked at greedy.**

## Rule

Gate kernel-numerics changes on the SLO shape (long agentic / long generation),
never greedy-short alone — b4b293f0c passed a short-prompt smoke and shipped a
length-dependent attention collapse. A named prior suspect (#48 relay named
b4b293f0c day one) earns the first experiment: `git revert <sha> + rebuild + A/B`
is one build vs a multi-probe forensic tower. See
[errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md](../errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md).

## Follow-up

A separate temp>0 sampling defect (#167 Type-B) was ALSO in `b4b293f0c` — its
`qwen35.rs` `w-1` final-norm load, sign-corrupting the STANDARD final-norm's
negative channels → temp=1.0 sampled-tail garbage (greedy survived). Fixed
`d703b5240` (revert to `load_vec`). temp=1.0 grpo unblocked.
