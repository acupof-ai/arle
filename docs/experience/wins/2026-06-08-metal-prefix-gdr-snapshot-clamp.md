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

Regression test (CPU, no GPU): `infer-core::tests::
prefix_match_clamped_to_backend_reusable_pages` — a `LimitedPrefixExecutor` that
can attach only 1 page while the radix offers 2; asserts `prefill_start_pos` is
clamped 8→4. `cargo test -p infer-core` 32 passed; `clippy -D warnings` clean on
`infer-core`/`infer-seam`/`infer-metal`.

Why not a guidellm sweep: guidellm fires **independent** requests and never
exercises conversational multi-turn prefix reuse, which is the only path this bug
lives on. The two-turn agent repro is the discriminating instrument. The hot
decode loop is untouched; the change only adds a bounded prefix-key scan on the
admission/attach path (zero cost on the CUDA default).

## Honest read / deferred (not silent)

- **Perpetual tail re-prefill.** Each turn the radix's deepest cached boundary
  (`floor(prompt_len/16)*16`) lands inside the final prefill chunk, so ≤1 chunk
  is re-prefilled every turn. Inherent to chunked-prefill × GDR; the clamp makes
  it *correct* but doesn't eliminate it. Full fix = split the final prefill chunk
  at the last page boundary so Metal snapshots there too — a planner change
  affecting all backends, deferred to its own evaluation.
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
