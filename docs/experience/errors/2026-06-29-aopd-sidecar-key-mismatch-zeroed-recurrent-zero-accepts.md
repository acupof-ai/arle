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

**Fourth bug: soundness crash on partial page cleanup (f8e861bb introduced)**:
`restore_recurrent_sidecar` returned Err without cleaning up `full_attn_kv` or
resetting `slot.seq_len()`. The page-radix path in `prefix.rs:89-98` caught the
Err but CONTINUED (log warn + proceed) — kept stale `full_attn_kv` seq_len (4777)
with `start_pos=4704` → soundness gate: "device pool seq_len 4777 != start_pos 4704".

Fix `16a8247f`: (a) executor `None` branch also frees `full_attn_kv` + resets
`seq_len=0`; (b) `prefix.rs` sidecar-error branch now releases pages
(`kv.free_slot + radix.release_blocks + kv.release_pages`) and returns Ok() with
`prefill_start_pos=0`.

**Performance side-effect of full-recompute fallback**: sidecar captures at
PREFILL completion only, not decode. Decode tokens advance the radix cache past
the last sidecar capture. Each turn where decode generated ≥16 tokens → radix
matches past the sidecar entry → fallback → full re-prefill from 0. 17
fallbacks observed in sidecarfix run (10 during eval, 7 during training).

Fix `7c461c80`: capture sidecar in `finish_slot` using prompt + generated tokens
before `publish_prefix_blocks`. Subsequent agentic turns that match at the
decode-extended seq_len will hit the sidecar instead of falling back.

**With fix `16a8247f` + eval (sidecarfix run, GPU 7)**: f327e65 → **4/4 accepts**:
- sample 0: passed=true (turns=5)
- sample 1: passed=true (turns=8)
- sample 2: passed=true (turns=5)
- sample 3: passed=true (turns=7)

Writeback: seq_len=7184, total_targets=604, chunk_rows=2048 — **OOM during backward**
(`cuda alloc_zeros failed` at 94778/97871 MiB, 2026-06-30). Root cause: fifth bug below.

**Fifth bug: SDPA chunk intermediates pile up in inner backward (inner_tape)**

`head_chunked_sdpa_recompute` with `tape.enabled` (inner_tape active inside
`checkpoint_backward`) kept ALL chunks' `scores/scaled/masked/probs` alive
simultaneously for the inner backward. At seq=7184 with ATTN_HEAD_CHUNK=8:
7 chunks × 4 × [8, 7184, 7184] × 4B = 46 GiB for ONE layer's SDPA alone.
With model (27 GB) + lm_head (4.3 GB) + grad accum (6.6 GB) + Adam (1 GB)
= 84.9 GB base + 46 GB SDPA transients → 94 GB peak → OOM.

The pile-up happened because the "tape-on: old behavior, intermediates survive"
comment at `qwen35.rs:354` (pre-fix) was accepted as correct — but it's only
correct for EAGER forward. In the inner backward's recompute, ALL chunk
intermediates must survive until the inner backward completes. At long seq, this
is too large.

Fix `0b7a1d89`: when `tape.enabled`, wrap each `causal_sdpa_recompute` call in
a nested `checkpoint` so the inner backward re-executes each chunk on demand
(one at a time, ~6.6 GiB) instead of keeping all 7 simultaneously (~46 GiB).
Peak drops from 94 GiB to ~48 GiB for seq=7184.

**allfix run launched (2026-06-30)**: `ARLE_OPD_WRITEBACK_OFFLOAD=0` (GPU-bound
backward) + `0b7a1d89` (nested SDPA checkpoint). Expected: backward GPU-bound
in ~15 min vs 54 min CPU-bound. Status: running on GPU 7.

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
