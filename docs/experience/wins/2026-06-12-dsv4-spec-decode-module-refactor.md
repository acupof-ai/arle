# DSv4 MTP spec-decode module — topk=1 refactor validated

**VALIDATED on 8×H20** (`ddbbdc8c`, `scripts/needle_gate.py`).

## Context

The MTP draft→verify→accept→commit orchestration was inline in
`executor.rs::forward_decode_tokens`, tangled with depth-K debugging cruft
(`chain_fresh`, `draft_dump`, the `SKIP_REFORWARD` forward-cost diagnostic).
The perf lever for spec decode is **accepted tokens per verify forward** (the
forward is weight-read-bound — a 1-token and a whole-tree forward cost ~the
same), which a **topk≥2 draft tree** raises. To get there without disturbing
the validated frozen-KV primitives, the orchestration was lifted into its own
module designed tree-general from the start, with topk=1 as the degenerate
one-chain case.

## What worked

`crates/infer-cuda/src/executor/spec_decode.rs` — a child module of `executor`
(reaches the executor's private fields without exposing them):

- `DraftTree` (flattened parent-pointer form) — `topk=1` = a chain, `topk≥2`
  branches; verify + accept are shape-agnostic.
- `longest_accepted_path` — **pure**, unit-tested (chain prefix, reject-first,
  and a topk=2 tree branch all green) — no device state.
- `spec_step` orchestrates `draft_tree → verify_tree → longest_accepted_path →
  commit_path`, driving the **untouched** model primitives (`mtp_forward`,
  `forward_tokens_verify`, the frozen-KV ring snapshot).
- `forward_decode_tokens` just delegates.

**topk=1 needle parity** — same long-context gate as the frozen-KV win:

| length | result |
|--------|--------|
| 3000 (depth 0.5) | exact ×3 |
| 6000 (depth 0.5) | exact ×3 |

Behaviorally identical to the pre-refactor inline path. `cargo check` + clippy +
fmt clean; 3 spec_decode unit tests pass.

## Rule

- **Lift a tangled hot path into its own module *before* extending it, with the
  general shape baked in and the special case (topk=1) reproducing the validated
  behavior exactly.** The extension (topk≥2 tree) then becomes three localized
  changes — branch the draft, tree-mask the verify, generalize the restore — on
  a confirmed foundation, not a rewrite. Validate the refactor against the
  existing gate (needle parity) before building on it. See
  [[2026-06-11-dsv4-mtp-frozen-kv-p1-longctx-fix]].
