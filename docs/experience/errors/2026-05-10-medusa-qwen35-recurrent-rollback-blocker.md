# Medusa Qwen3.5 Recurrent Rollback Blocker

## Context

Medusa Phase 1.B was re-scoped on 2026-05-10 from the older Qwen3/Qwen3.6
pickup brief to Qwen3.5. The first audit checked whether the existing CUDA
Qwen3.5 runtime can safely host target-integrated Medusa verification.

## Root Cause

Qwen3.5 is a hybrid model. Decode advances both paged KV for full-attention
layers and recurrent state for linear-attention layers.

The existing speculative scheduler commit path rolls back only paged KV after
verification. It calls `paged_kv_pool.truncate_slot(...)` after computing the
accepted draft length, but it has no corresponding model-owned rollback for
Qwen3.5 recurrent state.

Qwen3.5 explicitly cannot partially truncate recurrent state:

- `Qwen35State::truncate_to` resets recurrent state to zeros.
- `supports_partial_prefix()` returns false.
- The scheduler guide records that hybrid models cannot truncate recurrent
  state and only full-prefix snapshot restore is supported.

Therefore a Medusa verifier that runs `K+1` target steps would leave recurrent
state advanced through rejected draft tokens. That can break greedy
consistency even if the visible token stream initially looks plausible.

## Fix

Stopped before runtime implementation. No Qwen3.5 Medusa CLI, model hook, or
scheduler path was landed.

Documented the Step 0 audit:

- `docs/research/2026-05-10-medusa-phase1b-qwen35-step0-audit.md`

The next implementation must first add a Qwen3.5-safe speculative commit
contract, such as a measured recurrent snapshot ring or a shadow verifier
state, before exposing Medusa as a runtime mode.

## Rule

For hybrid models, speculative verification is not safe unless every mutable
state advanced by verifier tokens has an accepted-length commit or rollback
mechanism. Paged-KV rollback alone is insufficient for Qwen3.5.
