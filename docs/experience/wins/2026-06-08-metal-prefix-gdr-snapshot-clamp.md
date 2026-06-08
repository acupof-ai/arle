# Metal: clamp host prefix-cache reuse to the backend's GDR-snapshot coverage (multi-turn chat crash fix)

**Date:** 2026-06-08. **Backend:** Metal (MLX), Apple Silicon. **Model:**
`mlx-community/Qwen3.6-35B-A3B-4bit` (canonical). **Status:** landed,
correctness-verified by the exact two-turn repro. No default flag.

## Context

The agent REPL crashed on the **second** conversational turn:

```
ERROR infer_server::execution: Metal prefix attach missing GDR snapshot for slot 0,
  prefix_tokens=208, pages=[0..12]
Error: engine thread closed before request 1 completed
```

Turn 1 (`你好`) succeeded; turn 2 (`继续`) killed the engine thread. Basic
multi-turn chat was broken on Metal.

## Root cause

Two prefix caches with **different boundary granularity**:

- The host `RadixCache` (`infer-core`) caches a block at **every** page boundary
  (16 tokens) — `publish_prefix_blocks` inserts `prompt_len / 16` blocks.
- The Metal executor's `MetalPageStore.prefixes` only holds a **GDR
  (linear-attention recurrent/conv) snapshot** at the page boundaries a *forward
  pass actually landed on*. GDR state is prefix-wide, not page-sliceable, so it
  can only be exported at the exact length the session computed.

Chunked prefill processes a whole chunk in one `prefill_session` call, so
`cache_len` jumps over interior page boundaries. Probe evidence (turn 1):
snapshots landed at tokens `{64,128,192,224,240,256,272}` (prefill chunks
64/128/192/**219**, then decode crossings) — **208 was skipped** because it falls
inside the final prefill chunk `[192,219)`. Turn 2's prompt shares the 208-token
system-prompt prefix; the radix offered all 13 pages, but Metal had no GDR
snapshot at 208 → `materialize_slot_from_prefix` hard-errored → engine thread died.

An executor-only fallback is impossible: by `submit_prefill` the `PrefillRow`
only carries `tokens[208..]` (`planner.rs`), so the executor cannot re-prefill the
`[192,208)` gap, and submit/poll has no "I attached less" return channel. **The
host must clamp the offered prefix to what the backend can serve.**

## What Worked

Three files, backend-isolated:

1. **`BackendExecutor::reusable_prefix_pages(block_ids) -> usize`** (`infer-seam`)
   — default returns `block_ids.len()` (reuse everything), so **CUDA / paged
   attention is byte-for-byte unchanged** (its KV *is* page-sliceable).
2. **Metal impl** (`infer-metal/src/executor.rs`) — `MetalPageStore` returns the
   longest leading prefix of `block_ids` that has a snapshot key. It reads the
   **same `self.prefixes`** that `materialize_slot_from_prefix` reads, so the
   clamp can never disagree with what the executor will accept (single source of
   truth).
3. **Engine clamp** (`infer-core/src/{prefix,lib}.rs`) — `clamp_prefix_to_backend`
   trims `PrefixMatch` to the reusable page count at both the admission peek and
   the real attach. On the repro: 208 → 192, reuse 12 pages, re-prefill only the
   16-token tail.

Also removed the 3 `[metal-debug]` eprintln probes from the prior diagnosis pass
and restored `publish_slot` to its clean form.

## Verification (wall-clock ground truth, §0)

Exact repro — `printf '你好\n继续\n/quit\n' | arle --model-path
mlx-community/Qwen3.6-35B-A3B-4bit run`:

```
Turn 1 (你好):  in 219 tok · ttft 21.6s · 10.1 tok/s   out 59 tok
Turn 2 (继续):  in 515 tok · ttft 189ms · 2723 tok/s   out 38 tok   ← no crash
```

Turn 2's 189ms TTFT / 2723 tok/s on 515 input tokens proves the clamped prefix
attached cleanly and most of the prompt was a cache hit (only the tail
recomputed). Pre-fix this turn returned `engine thread closed before request 1
completed`.

**Correctness gate — reuse is bit-exact, not just non-crashing.** The open
question (and the reason `702454fe` disabled recurrent-KV reuse on CUDA) was
whether a slot resumed at `start_pos > 0` reconstructs the gated-delta recurrent
+ conv-ring state correctly, or silently decodes wrong. `agent-bench::tests::
metal_prefix_reuse_parity_qwen36` (canonical Qwen3.6 MoE) drives prompt `P` cold
(caches it) then drives `P` again — the second request reconstructs the slot at
the deepest cached page boundary (192/200), re-prefills only the `[192,200)`
tail, and decodes. Metal greedy is deterministic, so the floor is exact match:

```
cold  (ticks=28 fp=0x56076e604b0bf0e6)
reuse (ticks=25 fp=0x56076e604b0bf0e6)   ← identical fingerprint
```

Bit-for-bit identical greedy continuation ⇒ `materialize_slot_from_prefix`
reconstructs the recurrent/GDR state correctly; `ticks 25 < 28` ⇒ reuse engaged
(skipped prefill). This is the difference from CUDA: the CUDA executor advances
KV in place and asserts contiguous appends (`seq_len == start_pos`), so it cannot
honor `start_pos > 0` and `702454fe` rightly disables it there; the Metal
executor reconstructs a fresh contiguous slot, so reuse is sound. The two fixes
are complementary, not contradictory.

Regression test (CPU, no GPU): `infer-core::tests::
prefix_match_clamped_to_backend_reusable_pages` — a `LimitedPrefixExecutor` that
can attach only 1 page while the radix offers 2; asserts `prefill_start_pos` is
clamped 8→4. `cargo test -p infer-core` 33 passed; `clippy -D warnings` clean on
`infer-core`/`infer-seam`/`infer-metal`.

Why not a guidellm sweep: guidellm fires **independent** requests and never
exercises conversational multi-turn prefix reuse, which is the only path this bug
lives on. The two-turn agent repro is the discriminating instrument. The hot
decode loop is untouched; the change only adds a bounded prefix-key scan on the
admission/attach path (zero cost on the CUDA default).

## Follow-up landed: planner page-aligns prefill chunk ends (perf layer)

The clamp alone leaves a per-turn tail re-prefill: the radix's deepest cached
boundary (`floor(prompt_len/16)*16`) lands inside the final prefill chunk, which
has no GDR snapshot, so the clamp drops it back to the previous boundary and
re-prefills the gap (~6ms/turn). Fix (`infer-core/src/planner.rs`,
`build_forward_plan`): stop a prefill chunk on a page boundary when it would
otherwise cross one mid-chunk (`aligned_end = chunk_end - chunk_end % page_size`;
shrink to it when `aligned_end > start_pos`). A chunk can no longer skip a page
boundary without landing on it, so Metal snapshots GDR at every boundary the
radix later offers as its deepest match. The sub-page tail (no boundary left)
is emitted as-is — one extra tiny tick at most. Page-sliceable backends (CUDA
paged attention) are unaffected functionally; the only change is chunk
boundaries shift to page multiples.

This is the perf layer **on top of** the clamp, not a replacement: chunks are
larger than a page (64 in the repro), so page boundaries *inside* a chunk
(16/32/48/80…) are still radix-offerable but unsnapshotted — a prefix hit there
still relies on the clamp for safety. Alignment only guarantees the *deepest*
boundary (the common agent-REPL case where the shared prefix = the full system
prompt) is reusable. Regression test: `prefill_chunk_stops_on_page_boundary`.

CUDA bench: `pending-remote` — this planner change touches the shared CPU
scheduler so it affects the CUDA path's chunk boundaries; cannot run CUDA on
this Mac. Expected delta ≈ 0 (one extra ≤page-size prefill tick per request at
most). Verify on the 8×H20 pod before relying on it for CUDA.

## Honest read / deferred (not silent)

- **Snapshot staleness under eviction (latent, separate bug).** `MetalPageStore.
  pages/prefixes` are not pruned when the host radix LRU-evicts a page; if a page
  id is recycled, a stale snapshot could be read. The 2-turn repro has no
  eviction pressure so it doesn't trigger here. Follow-up, out of this scope.

## Rule

The host radix offers prefix reuse at page granularity, but a recurrent-state
backend (Metal GDR / linear attention) can only attach at boundaries it
snapshotted during a forward pass. **Any prefix the host offers must be clamped
to the backend's actual reuse coverage via a seam query** — never assume a
page-aligned KV-present boundary is attachable. Default the seam method to
full-reuse so page-sliceable backends are unaffected.
