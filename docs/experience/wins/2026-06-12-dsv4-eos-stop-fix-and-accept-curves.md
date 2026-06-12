# DSv4 CUDA EOS-stop fix + measured MTP acceptance curves (clean workload)

Commits: `ea71e060` (EOS fix). Pod: 8×H20 TP=8, DeepSeek-V4-Flash, binary at
`a68aa0ee`+fix, 2026-06-12.

## Context

The MTP P0 probe revealed every CUDA request ran to `max_tokens`: the CUDA
executor never overrode the seam's `model_stop_token_ids` (default empty;
Metal always had it), so engine-core had no fallback stop set — every
response was padded with post-EOS degenerate text, polluting user latency
and every recorded CUDA bench number.

## What Worked

One trait-method override on all 3 arms (`RealCudaExecutor`): Qwen → new
scalar-or-list-tolerant `Qwen3Config::eos_token_ids`; Qwen3.5 → existing
`stop_token_ids`; DSv4 → config `eos_token_id` (=1). Verified:
`completion_tokens=9, finish=stop` (was 64/64 with garbage tail).

## Results — clean 256-token essay workload (finish=length, real prose)

| config | per-level accept | A (tok/step) | tok/s p50 |
|---|---|---|---|
| no-spec | — | 1.0 | **32.52** |
| chain d1 | q₁=0.695 | 1.70 | 19.68 |
| chain d4 | 0.651 / **0.357** / 0.333 / 0.200 | 1.98 | 13.89 |
| tree d2k2 | **0.933** / 0.531 (top-2) | **2.43** | 14.84 |

- The earlier "36% accept" was the degenerate-tail artifact; clean-text
  chain accept is 0.695, and top-2 coverage at level 1 is **0.933**.
- Depth decay (nextn-1 head is depth-1-trained) kills deep chains at level 2
  (0.36); tree width holds level 2 at 0.53.
- A=2.43 at d2k2 is banked; the wall-clock loss (14.84 vs 32.52) is the
  known per-row verify cost — the P1/P2/P3 fast-path plan
  ([2026-06-12-dsv4-mtp-tree-fast-path](../../plans/2026-06-12-dsv4-mtp-tree-fast-path.md))
  budgets ~66 tok/s (2.0×) at this A.

## Rule

- **A backend that implements a seam trait must be audited against every
  defaulted trait method** — the empty `model_stop_token_ids` default
  silently disabled EOS for the whole CUDA backend; the Metal twin had the
  override from day one.
- **Acceptance curves are per-level conditional probabilities, measured on
  EOS-honoring real text** — a single accept ratio on an unbounded
  generation mixes regimes and misled the perf budget twice (36% artifact,
  then 86% small-n).
