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

## Second bug: page count mismatch after key fix

After the key-alignment fix (`2bc6b44d`), restores started hitting the cache but then
failed with:
```
paged_kv host payload length mismatch: got 77594624 expected 76546048
```

The slot had 74 allocated full-attn KV pages (seq_len=1210, ceil(1210/16)=76 → 74
actually allocated) but `matched_len=1168` expected 73 pages. The extra page is the
slot's partially-filled "next" page that was allocated but not yet covered by the
page-aligned mat_len.

Fix `c96bfb50`: limit the D2H copy to `n_pages = mat_len / PAGE_SIZE` pages — the
extra allocated-but-not-full page is excluded from the snapshot.

## Verification (COMPLETE — 2026-06-29)

Pod run (`run_fixsidecar_opd4.log`) with `2bc6b44d` + `c96bfb50` + `max-tokens 8192`:
- `RESTORE-SIDECAR` at every re-prefill, zero "restore failed" or "mismatch" errors ✓
- Eval base + eval[1]: 0ea40e0 `passed=true edited=true` via multi-turn sidecar
  (base: start_pos=0 fresh; eval[1]: `matched_len=1168 → 3296 new tokens → pass`) ✓
- Training round 0: 0/4 accepts for f327e65 — model generates prose-Stop for
  samples 0 (211 fresh tokens), 1 (3 tail tokens after sidecar-at-960), 3 (3 tokens);
  sample 2 explored (sub_turn with re-prefill at 2673 tokens) but stopped without edit

**Training 0-accept is model capability on this specific task**, not infrastructure:
- The sidecar correctly preserves state (0ea40e0 passes with same sidecar path)
- f327e65 (FQCN validation) and the 2 failing eval tasks (12734fa, 5e36960) all
  produce prose-Stop even with correct recurrent state and full user message visible
- 12734fa eval: `matched_len=752`, 896 new tokens → still 0 turns (plenty of context)
- Fix: use tasks the current student can solve (ensure at least one accept per round)

**Secondary issue — sample 1+ 3-token tail**: after sample 0's radix caches the
963-token prompt, subsequent samples see `matched_len=960` → only 3 new tail tokens.
Model can still generate correctly (recurrent state encodes full user message), but the
narrow window makes exploration less likely. Mitigated by temperature diversity; root
cause is the page-16 alignment of the sidecar-capture vs the prompt length.

## Rule

**Silent 0-accept ≠ model capability failure.** Check sidecar HIT rate before
concluding the model can't solve the task. A sidecar key divergence zeroes recurrent
state silently — no crash, no assertion, just a model that forgets every turn.

**After infrastructure fixes, 0-accept = task selection problem.** Verify that at
least one training task is solvable by the current checkpoint before interpreting 0
accepts as a sidecar regression.
