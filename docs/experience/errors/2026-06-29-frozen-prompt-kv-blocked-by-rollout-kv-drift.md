# Frozen-prompt-KV agent-OPD writeback blocked upstream by a Qwen3.6 rollout-engine KV-seqlen drift

## Context

Goal: exercise the new **frozen-prompt-KV masked-writeback** feature
(`ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV=1`, commits `677cecd6`/`f1750d10`/`e41d4758`,
Gate A `0154a23f`) end-to-end on the H20 box, to capture the payoff: forward+backward
only the GENERATED segment (~few hundred tokens) instead of the full ~15129-token
trajectory → the ~90 GB writeback OOM disappears, ~80 min/step collapses to (hypothesized)
<1 min, the writeback COMPLETES → `trained_pairs>0` → AdamW → the first held-out eval Δ.

Build tree `/host/arle-ckl-aopd` (synced from local HEAD via `git archive` overlay,
26 MB source tarball, keeping warm `target/`). Run: `arle train agent-opd`, student
`/host/Qwen3.6-27B-FP8`, GPU 1, tmux session `arleCKL`, config `--samples-per-prompt 4
--writeback-cap 1 --rounds 1 --max-turns 16 --max-tokens 768 --lora-layer-start 32
--rollout-num-slots 1`, 1 train task + 3 held-out eval tasks (SWE-bench-Pro ansible).

## Build — PASSED, feature symbol-verified

- `BUILD_EXIT=0` (warm incremental `--release --features cuda --bin arle`, 1m 43s),
  binary `/host/arle-ckl-aopd/target/release/arle` (Jun 29 07:48).
- New binary `strings` confirms the gated path is present and reachable:
  `ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV`, `[masked-writeback-frozen] phase=forward_gen_segment seconds=`,
  `pre/post forward_hidden_states_gen_segment`. The feature is correctly gated and compiled in.

## Root Cause — pre-existing rollout-engine KV-bookkeeping drift, NOT the feature

The very first eval rollout died with a single fatal engine error:

```
ERROR infer_server::execution: execution.rs:272 infer-server engine step failed:
  Qwen3.5 materialized state len 4520 != DecodeRow.kv_seq_len 4478 for slot 0
```

`crates/infer-cuda/src/executor.rs:4766` (`submit_decode_row`) asserts the slot's
materialized KV state length equals the scheduler's `DecodeRow.kv_seq_len`. At ~4.5K
tokens of agentic multi-turn context they diverged by **Δ=42** (slot=4520 vs
scheduler=4478) → hard error → the engine thread closed → every subsequent request
(2 more eval tasks + all 4 round-0 rollout samples) returned `engine thread closed;
cannot submit`. Zero rollouts completed → `passed=0` → `trained_pairs=0` →
**the frozen-prompt-KV writeback never executed.**

Isolation (case-as-fact, confounders ruled out):
- **Not the feature.** The feature commits' own diffs touch only autograd / train / cli /
  tests (8 files) — none touch `executor.rs`. The assertion lives in the `a96db69e`
  "Phase 2 Qwen3.6 full-attn KV → shared paged pool" code, unrelated to the writeback.
- **Not ELKEID / crash-loop / load-death.** The pre-CUDA sandbox-spawner came up
  (`pid 368317`), the model loaded (train-infer FP8 sharing: 400 resident base
  projections borrowed zero-copy), GPU 1 reached 36→88 GB at 100% util. It was one
  deterministic assertion during decode, not a fork-kill or a reaped launch.
- **Not a real capability measurement.** The held-out `pass_rate=0.0 (0/3)` is a harness
  artifact — all three eval dumps read `note: "rollout error: engine thread closed"`,
  i.e. zero genuine model attempts. (CLAUDE.md §0: audit the harness; a 0/3 that is
  3/3 engine-crash is not a capability signal.)
- **GPU clean.** After the crash GPU 1 freed to 0 MiB; the 87785 MiB later observed was
  a FOREIGN `arle serve --model-path /host/Qwen3-4B --port 8765` (host PID 1264862,
  started after I vacated) — left untouched.

The drift is in the Qwen3.6 (qwen35 codepath) rollout engine's paged-KV accounting under
the agentic re-prefill pattern (tool output appended → re-prefill → decode): the slot
seq_len advanced 42 tokens past what the scheduler's DecodeRow tracked. Mechanism not yet
attributed to a specific line — the next step is to instrument where slot seq_len and
`DecodeRow.kv_seq_len` separate across a turn boundary (prefill-row vs decode-row append).

## Fix

NOT FIXED — out of scope for this run (the feature under test is fine; the blocker is
upstream). The writeback payoff (forward_gen_segment time, peak VRAM, trained_pairs, Δ)
remains **unmeasured** because no trajectory was ever generated. Re-run is blocked until
the Qwen3.6 rollout-engine seq_len drift is root-caused and fixed, OR the agentic context
is kept short enough that the drift does not surface (the failure was at ~4.5K tokens).

## Rule

- **A held-out `pass_rate=0` whose per-case notes are all "engine thread closed" is an
  engine-crash artifact, not a capability number.** Decode the per-case dump before
  trusting any aggregate (CLAUDE.md §0 case-as-fact).
- **The frozen-prompt-KV writeback cannot be exercised through the agentic-OPD loop until
  the Qwen3.6 rollout engine survives a ~4.5K-token multi-turn rollout.** Validate the
  rollout path in isolation (a single long agentic rollout completing) BEFORE attributing
  any writeback win/null — the writeback is downstream of a passing rollout.
- **`materialized state len != DecodeRow.kv_seq_len` = scheduler↔executor KV drift**, in
  `executor.rs` `submit_decode_row`/`submit_decode_batch`; tied to the `a96db69e` shared
  paged-pool path, surfaces on agentic re-prefill, not on plain single-turn decode.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
