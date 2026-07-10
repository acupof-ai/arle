# Medusa Phase 1.B Qwen3.5 Step 0 Audit

## Context

User direction on 2026-05-10 changed the Medusa pickup from the older
Qwen3/Qwen3.6 brief to Qwen3.5. This audit checks whether the Phase 1.B
substrate can be safely implemented against the CUDA Qwen3.5 model without
leaving a half-state.

Scope audited:

- `infer/src/model/qwen35/forward.rs`
- `infer/src/model/qwen35/batch_decode.rs`
- `infer/src/model/qwen35/recurrent_state.rs`
- `infer/src/scheduler/cuda/spec_path.rs`
- `infer/src/scheduler/AGENTS.md`

## Evidence

1. Qwen3.5 does not currently implement the shared verifier hook.

   `rg forward_spec_verify_batch infer/src/model` only finds the Qwen3
   implementation in `infer/src/model/qwen3/forward.rs`. Qwen3.5 would need a
   new verifier path before external-draft or Medusa verification can run.

2. Qwen3.5 can expose the hidden state Medusa needs.

   `infer/src/model/qwen35/batch_decode.rs:578` to `:594` applies the final
   RMS norm into `bufs.common.normed`, then projects that same tensor through
   `embed_tokens` into logits. A Medusa head could consume `common.normed`
   after the first target step.

3. Qwen3.5 decode mutates per-request recurrent state.

   `infer/src/model/qwen35/batch_decode.rs:447` to `:485` uploads per-slot
   pointers for every linear-attention layer's conv and GDR recurrent state,
   then `decode_batch_body` runs with those live pointers. Verifier tokens
   therefore advance both paged KV and recurrent state.

4. The existing speculative commit path only rolls back paged KV.

   `infer/src/scheduler/cuda/spec_path.rs:251` to `:258` computes
   `keep_target_len` after greedy verification and calls only
   `paged_kv_pool.truncate_slot`. There is no model-owned commit/rollback hook
   for non-KV state.

5. Qwen3.5 recurrent state has no partial truncate operation.

   `infer/src/model/qwen35/forward.rs:122` to `:129` implements
   `truncate_to` by truncating base state and resetting recurrent state to
   zeros. `supports_partial_prefix()` returns false at `:132` to `:134`.
   `infer/src/scheduler/AGENTS.md:57` to `:60` records the project invariant:
   hybrid models cannot truncate recurrent state; only full-prefix snapshots
   are safe.

6. The existing recurrent snapshot API is not a decode-token commit primitive.

   `infer/src/model/qwen35/recurrent_state.rs:91` to `:134` saves/restores a
   single post-prefill snapshot for full-prefix reuse. The comment records a
   roughly 49 MB GPU copy cost for Qwen3.5-4B. Reusing this per speculative
   decode step would add a large memory-copy tax and would overwrite the
   prefix-cache snapshot semantics unless a separate snapshot ring is added.

## Decision

Do not implement Qwen3.5 Medusa Phase 1.B as the older brief is written.

The hidden-state part is feasible, but the commit semantics are not safe.
If the verifier runs `K+1` target steps and only some draft tokens are
accepted, paged KV can be truncated to the accepted length, but Qwen3.5
linear-attention recurrent state remains advanced through all verifier tokens.
That would violate greedy consistency and risks anti-pattern #26
same-output-but-garbage failures.

This is a scope expansion beyond the original "about 350 LOC Rust + 0 kernels"
substrate. The missing unit is a Qwen3.5-safe speculative commit contract.

## Required Design Before Implementation

Any Qwen3.5 Medusa substrate must first choose and license one commit model:

1. Model-owned verifier commit API.

   Add a model trait method where Qwen3.5 owns verification and receives the
   accepted length per row before mutating live state permanently. This likely
   needs recurrent snapshots per verifier step or an equivalent state ring.

2. Shadow verifier state.

   Run speculative target verification on shadow states, then replay only
   accepted tokens plus bonus into live state. This is correctness-safe but
   may erase the throughput win unless replay cost is proven small.

3. Qwen3.5 Medusa kill or defer.

   Keep Medusa on full-attention models only, and do not expose a Qwen3.5
   CLI/runtime mode until recurrent rollback has a measured design.

## License-Or-Kill Gaps

- No runtime code was landed from this audit.
- No bench was run because this is a docs-only blocker record.
- The next executable step is a small prototype that measures recurrent
  snapshot-ring overhead per decode step before wiring Medusa heads.
- Minimum evidence before unblocking: greedy consistency with targeted reject
  cases, manual sample inspect, and a memory-copy cost measurement for the
  chosen recurrent rollback strategy.
