# Gap-injection KV memory system (finalized design, 2026-06-24)

> **Status: DESIGN ONLY — implementation DEFERRED (ckl 2026-06-24).** Build the
> infrastructure first (persistent NVMe memory store + pool consolidation); this
> gap-injection recall *mechanism* lands on top of that foundation later. Kept here as
> the finalized blueprint (the re-RoPE result below is the load-bearing insight).

Supersedes the original-position recall in
[`2026-06-23-writethrough-tiered-kv-memory.md`](2026-06-23-writethrough-tiered-kv-memory.md)
§④. The tier/budget model there still holds; this finalizes **the recall mechanism**.

## Why gap-injection (not original-position)
Original-position recall keeps a recalled block at its true position `p`. For a
million-token session `p` exceeds the model's trained RoPE range → the attention is
garbage. Gap-injection **re-RoPEs** the recalled memory into a fixed in-range slot, so
positions are always in range regardless of session length. This is what makes
"unlimited text" actually correct, and it bounds prefill (no full-history re-prefill).

## The prompt layout + the key result
```
positions:  [0 .. S)          [S .. S+G)            [S+G .. )
content:    [system prompt]   [memory gap]          [user input]
```
The gap (`G = top_k · l_bs` tokens, default 8·256 = 2048; tunable) is filled with recalled
memory KV. **The attention is STANDARD** — `[system][gap][user]` are contiguous, in-range
positions; ordinary paged attention runs unchanged. No sparse kernel.

**Re-RoPE math (the load-bearing result).** A block cached at original `[p, p+L)` has
`K_cached[i] = R(p+i)·k[i]`. To place it at gap `[g, g+L)` we need `K_gap[i] = R(g+i)·k[i] =
R(g−p)·K_cached[i]` — a **single uniform rotation `R(g−p)` per block** (the delta is the
same for every token in the block). V carries no RoPE — copied as-is. So recall =
fetch block → rotate K by the per-block delta → write into the gap pages.

## Components (two parallel tracks against one interface)

**Interface — the memory store (`MemStore`, session-keyed; per-user deferred):**
```
put(session, block_id, kv_blob, rep, orig_start_pos)   // write-through a frozen block
score(session, query_rep) -> [(block_id, score)]        // relevance = rep · query, top-k
get(session, [block_id]) -> [(kv_blob, orig_start_pos)] // fetch for re-RoPE
persist() / load(session)                               // durable index + manifest
evict()                                                 // bounded total bytes (per-session cap)
```

### Track A — recall core (device side, in `recall_exec.rs`)
- **Bounded pool**: `system + gap (G) + user + local` per slot — NOT max_seq_len. This is
  the pool consolidation + the bounded budget (one device working pool).
- **Write-through**: as generation freezes a block (leaves the local window), compute its
  rep + `put` it to the store (K is final; rep is mean-pooled layer-0 K).
- **Recall at prefill**: build `[system]` + reserve the gap + `[user]`; `score` the store
  with the user query rep → top-k blocks → `get` → **δ-rotate each block's K** (new re-RoPE
  kernel) → write into the gap pages (original temporal order) → standard paged attention.
- **Decode unchanged**: append + attend the fixed `[system][gap][user]+local` set; zero tier I/O.
- **New kernel**: `rope_delta_rotate(K_block, delta)` — uniform rotation of a block's cached
  K by `g−p`. Reuse the RoPE machinery (apply a rotation by an integer position delta).

### Track B — persistent SSD memory store (storage side)
- Implement `MemStore` over the existing `CudaKvTierStore` + `kv-native-sys`: DRAM (hot) →
  NVMe (cold), block blobs sharded on disk.
- **Persist the index + reps + manifest** (the gap ckl flagged): today the key→location map +
  reps are in-memory and lost on restart → disk blobs orphaned. Write a durable manifest
  (block_id → {rep, orig_start_pos, disk offset/len, kv_dtype}) so a session's memory survives
  restart and is scorable without re-reading blobs.
- **Wire NVMe for recall** (`set_disk` for the recall store; per-process namespace — per-user
  deferred). Bounded total bytes per session (eviction by cold/oldest); the 64 GB/user cap is
  the per-user layer, deferred.
- **Invalidate on weight epoch** (OPD): a model/RoPE-version tag in the manifest; a new epoch
  drops stale memory (see prefix-cache-stale-across-weight-epochs).

## Open knobs (defaults chosen; tunable, flag in benchmarks)
- Gap size `G = top_k · l_bs` (default 2048). Bounds memory injected per inference.
- Within-gap order: **original temporal order** (preserve relative structure) — default.
- Re-RoPE: **δ-rotate cached (post-RoPE) K** (no pre-RoPE storage; reuses RoPE kernel).
- Cross-block positions: originally-scattered blocks become contiguous in the gap (a "fact
  bag"); validate with the needle grid.

## Out of scope here (deferred)
- per-user 64 GB model / cap / cross-session sharing (ckl: 先不做).
- proactive eviction watermark, admission control (later steps).
