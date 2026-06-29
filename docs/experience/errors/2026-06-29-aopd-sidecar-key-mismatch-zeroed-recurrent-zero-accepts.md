# Agent-OPD: Sidecar Key Mismatch Zeroed Recurrent State → 0 Rollout Accepts

## Context

Qwen3.6 is a hybrid model (linear-attn + full-attn). The linear-attn layers carry
a recurrent state (gdr + conv) across agentic re-prefills. This state is
snapshot/restored via a `prefix_sidecar` HashMap keyed by `FNV(tokens[..mat_len])`.

After fixing the rollout seq_len drift bug (`1b0f0459`), rollout runs survived but
the model produced 0 edits across every task — no accepts ever fired even for tasks
the model had previously solved 4–8/8.

## Root Cause

`capture_recurrent_sidecar` (executor.rs:3358) keyed on **raw `slot.seq_len()`**:
```rust
let mat_len = slot_state.seq_len().min(tokens.len());
```

`restore_recurrent_sidecar` looked up by **`matched_len`** — the page-16-aligned
prefix length returned by the radix cache:
```rust
let key = hash_prefix_tokens(&tokens[..matched_len]);
```

Raw `seq_len` equals a multiple of `SUPPORTED_PAGE_SIZE=16` only ~1/16 of the time.
Every restore missed → recurrent state was zeroed at each re-prefill → the model
operated without turn-to-turn memory → wandered, stopped editing files, produced 0
accepts.

This is distinct from the seq_len drift bug (which caused an assertion crash at
`materialized state len != DecodeRow.kv_seq_len`). This bug is silent: rollout runs
to completion with 0 accepts and no assertion, so it looks like a capability failure.

## Fix

`2bc6b44d` — floor `mat_len` to page size before hashing:
```rust
let mat_len = (slot_state.seq_len().min(tokens.len()) / SUPPORTED_PAGE_SIZE)
    * SUPPORTED_PAGE_SIZE;
```

Capture key and restore key now always agree. Residual: the snapshot is taken at
`mat_len` (≤15 tokens before the true seq_len boundary) — a small double-count on
restore; negligible versus full-zeroing. Exact boundary snapshot deferred.

## Verification

Pending-remote: rebuild on H20 pod with `2bc6b44d`, run OPD with
`ARLE_KVDRIFT_DEBUG=1` and confirm:
- `[kvdrift] SIDECAR-LOOKUP ... -> HIT` at every re-prefill (was always MISS)
- ≥1 accept in the first round (was 0/N every round post seq_len-drift fix)

## Rule

**Silent 0-accept ≠ model capability failure.** Check sidecar HIT rate before
concluding the model can't solve the task. A sidecar key divergence zeroes recurrent
state silently — no crash, no assertion, just a model that forgets every turn.
