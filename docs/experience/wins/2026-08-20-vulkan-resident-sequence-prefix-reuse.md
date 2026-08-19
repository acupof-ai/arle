# Vulkan resident-sequence prefix reuse — turn-2 prefill 156.5 s → 3.2 s (49.7×)

## Context

Qwen3.8-27B-Q4_K_M on Strix Halo (Radeon 8060S, Vulkan-only, 63.6 GB unified).
Multi-turn chat re-prefilled the entire conversation every turn. The Vulkan lane
prefills through a serial `forward_token` loop (`executor.rs:203`), so a cold
536-token prompt costs ~156 s — prefill is *slower* than decode (3.5 vs 9.4
tok/s). Prefix reuse is therefore worth more here than on any batched backend.

The CUDA-style route was unreachable. `crates/infer-vulkan/src/forward.rs`'s
`DeviceKvCache` is ONE flat UMA buffer `[K block | V block]` indexed
`[full_layer, kv_head, pos, head_dim]` by **absolute position**. There is no page
indirection: `VulkanKvPool`'s page ids are host bookkeeping that never reach the
device, so "attach these radix pages at these positions" names no device bytes.
The model also holds one `Qwen35ForwardState` + one `DecodeResources` — the lane
is genuinely single-slot.

## What Worked

`infer_seam::PrefixReuse::{cached_prefix_match_len, restore_cached_prefix}` is
documented as exactly this seam — "backends whose KV cannot be page-reattached at
arbitrary positions" — and was half-wired: `cached_prefix_match_len` was called
for budgeting only, and `restore_cached_prefix` had **zero call sites**. Both
sides had to be built.

- **Lane side** (`model_qwen35.rs`): replaced the `(slot, epoch)` reset key with
  `resident_tokens: Vec<u32>` — the tokens actually materialized into the device
  state. Epoch changes on every new request *including the ones that continue the
  sequence*, so it could never key reuse. Reset now triggers on `start_pos == 0`.
- **Engine side** (`prefix.rs::restore_cached_prefix_image`): consulted only when
  the page route returns 0, so that route stays byte-for-byte unchanged. Every
  failure degrades to `Ok(0)` (full recompute) — this runs on the admission path
  where an error is fatal to the whole TP group (#164).
- **Finish write-through** (`materialize_finish`): the last sampled token is never
  fed back, so without it the resident image is always one token short of the next
  turn's prefix and the match lands at |PT|+|GT|-1.

Reuse is **all-or-nothing**. Full-attention KV is positional and genuinely holds
`[0, len)`, but `gdr_state` (gated-delta S matrix) and `conv_ring` are a running
fold with no rewind and no snapshot, so a partial match is worth nothing.

New gate `scripts/eval_harness/resident_reuse.py`:

```
|PT|=488 |GT|=48 ceiling=536 on_hit=536 off_hit=0 on=3.15s off=156.51s  → 49.7×
```

## Rule

On a position-locked backend, key prefix reuse on **the token list actually
materialized into the device state**, never on a request-scoped id (slot, epoch,
sequence id) — those turn over on exactly the requests reuse exists to serve.
Then write through the final sampled token, or every match lands one short.

Corollary for gate design: `token_reuse` cannot measure this lane, because it
round-trips its suffix ids through a third request between turn 1 and turn 2, and
on a single-resident lane that request *is* an eviction. `prefix_reuse` likewise
declines, because its warm arm ends on a generated token the reuse prompt does
not contain. Both are correct refusals, not regressions — neither reused anything
on Vulkan before this change either (`prefix_reuse()` returned `None`).
