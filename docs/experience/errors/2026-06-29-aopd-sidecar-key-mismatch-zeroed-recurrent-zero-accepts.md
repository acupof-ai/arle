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

## Third bug: sidecar miss still zeroes linear attention (cross-attention mismatch)

After `2bc6b44d` + `c96bfb50`, restores HIT the sidecar for within-session re-prefills.
But eval tasks (0ea40e0) and training tasks (f327e65) share the same 752-token system
prompt prefix. The radix cache returns `matched_len=752` for f327e65's first turn
(after 0ea40e0 built radix pages), but there is NO sidecar entry at 752 tokens —
0ea40e0's first prefill captured at 1168, not 752.

On miss, the previous code silently zeroed linear-attention recurrent state while
leaving full-attention KV intact (from the radix cache):
```
// miss branch — original code:
log::debug!("no recurrent sidecar ... starting with zeroed recurrent state");
// → then sets seq_len=752, allocs KV pool for 752 tokens
```

The model then processes the remaining 319 tokens (user message) with:
- Full attention: sees all 1071 tokens via KV cache (752 radix + 319 new) ✓
- Linear attention: only "knows" the last 319 tokens (recurrent state zero at t=0–751) ✗

This cross-attention-type mismatch corrupts output: the model generates plain text
instead of JSON tool calls, producing 0 edits.

Fix `f8e861bb`: return `Err` on miss → `prefix.rs:162-173` catches it, calls
`kv.free_slot(slot)`, returns `Ok(0)` (matched_len=0). Both attention types then
start from scratch via full re-prefill.

## Verification

**Before fix (run_fixsidecar_opd4.log, hinted run)**: training tasks hit matched_len=752
sidecar MISS → 0/4 accepts for f327e65; model outputs plain text (prose-Stop).

**After fix (run_freshtest.log, --eval-every 999)**: running training tasks without
eval[base] first → empty sidecar → all tasks start fresh at matched_len=0 → **3/4
accepts** (samples 0, 2, 3 passed; sample 1 failed exit 4):
```
[agent-opd] ansible__ansible-f327e65 sample 0: passed=true (turns=12)
[agent-opd] ansible__ansible-f327e65 sample 1: passed=false (turns=14) [exit 4]
[agent-opd] ansible__ansible-f327e65 sample 2: passed=true (turns=5)
[agent-opd] ansible__ansible-f327e65 sample 3: passed=true (turns=6)
```

**With fix + eval**: `f8e861bb` binary running OPD with eval-every 1 — pending (run
killed previously; new run started post-build).

## Rule

**Silent 0-accept ≠ model capability failure.** Check sidecar HIT rate AND sidecar
miss behavior before concluding. A miss with zeroed linear state vs. non-zero full-
attention KV corrupts model outputs silently — no crash, 0 accepts, looks like a
capability regression.

**On sidecar miss, fall back to full recompute.** Never let a partial (zeroed linear +
non-zero full) attention state reach the model. The cross-type mismatch is worse than
the recompute cost.

**Incorrect intermediate attribution (2026-06-29)**: after the first two fixes, 0/4
training accepts were attributed to "model capability on this task" — wrong. The
third bug (miss → zeroed state) was still present; fresh-start test immediately
showed 3/4 accepts.
