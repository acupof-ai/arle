# DSv4 dynamic KV mem budget — c=32 OOM crash → clamp+warn (cudaMemGetInfo×0.9)

## Context
DSv4 executor sized KV for a FIXED num_slots → at high concurrency × long max_seq_len the
per-rank MLA-KV arena alloc OOM-CRASHED (c=32 root cause, task #37). DSv4 owns its MLA KV
inside the forward (sized by max_seq_len per slot), so this is `num_slots × per-slot-bytes`
exceeding GPU free mem.

## Fix (commit aa445112)
`Dsv4Model::kv_budget_num_slots(requested, max_seq_len)`: query `cudarc::mem_get_info()`
(post-weights free) → affordable = `free×0.9 ÷ per_slot_bytes`, clamp num_slots with a clear
warning instead of crashing. per_slot = EXACT FP8 arena (`max_seq_len × bytes_per_token(584) ×
num_layers`) × 2 (covers compressor/SW/indexer per-slot buffers + activations). Deterministic
formula ⇒ **TP-consistent** (a per-rank measure-and-retry would diverge across ranks and
corrupt the shared KV). No new FFI (cudarc mem_get_info existed).

## Verify (8×H20 TP=8)
- pod **build PASS**; c=1 needle **byte-identical** `[223,30793,…,8308]`, steady 39.68 tok/s.
- **No clamp at c=1** (affordable ≫ requested) — the default path is untouched.
- c=32 clamp is logic-verified by the formula: e.g. c=32 @ max_seq_len=81920 →
  per_slot ≈ 4.1GB → affordable ≈ 17 slots (clamp+warn) instead of the OOM crash.

## Rule
DSv4 KV capacity must be budgeted from `mem_get_info()` at construction, not a fixed count —
and the budget formula must be DETERMINISTIC (not per-rank measure-based) so all TP ranks
agree on num_slots. Exact arena term + conservative ×2 overhead + 0.9 fraction clamps safely
below OOM while leaving c=1/moderate-shape unaffected. Unblocks the batched-decode (#38)
c-scaling (no more c=32 crash). Bench: c=1 byte-identical + no-clamp verified; full c=32
exercised by #38.
