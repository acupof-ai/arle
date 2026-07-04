# DSv4 KV storage — three-layer redesign (Shape / Index / Physical)

> Status: Active — 2026-07-04. **Architecture cleanup, decoupled from #146.**
> An audit of the value-pool write→read path (below) proves the storage layer is
> race-free and bounded on the #146 repro config, so #146 is NOT a storage/mapping
> bug — it lives in compressed VALUE *content* (compressor high-position compute /
> FP8 quant / model), tracked separately. This refactor's value is purely
> structural: collapse 4 value layouts to 1 paged pool, delete the quality-
> defective scalar lane, single-source the block map, and unblock page-granular
> L2/L3 tiering (P5). P1/P2/P4 landed (commits 611d18cd, 2dd9d07c); P3 remains.

## Audit — the storage layer is constructionally correct (why #146 ≠ storage)

First principles: on HBM, a KV access is correct iff **index correct + physical
correct + write-happens-before-read**. GPU adds the third term — kernels are async,
so a cross-stream write/read without an event fence can race. Audit of the
eager-decode / spec-none path (the #146 baseline):

| Check | Finding | Source |
|---|---|---|
| write (pack) stream | all `ctx.stream` | `flashmla_pack_compressed_delta` |
| read (FlashMLA decode) stream | build-indices + pool_ptr + kernel launch all `ctx.stream` | `try_flashmla_decode_attention`, flashmla.rs:238 `cu_stream()` |
| comm_stream / pipeline fence touch value pool? | **never** (zero hits in attention/flashmla) | grep |
| read range | `start_row = fp8_kv_comp_packed_rows`; reads `[0, packed_rows)` only | attention.rs:1936 |
| bound assert | `packed_rows ≤ indexer_rows_before` | attention.rs:7935 |
| index/physical | P1 BlockMap single-source + P4 CSA assert + kernel page_table routing masked by `flashmla_total_pages()` | — |

**Single stream ⇒ stream order guarantees write-before-read ⇒ no race.** The only
other CUDA streams (`comm_stream` for NCCL, the MTP persistent verify stream) never
touch the value pool on this path (MTP is off at spec-none). All three correctness
terms hold. **#146 is therefore compressed VALUE content, not storage** — the
refactor below does not target it.

(The "race" in the 07-03 #138 note — MTP-off context-129 wall, lens-maskable — is a
*different* symptom, most likely a scratch-buffer premature-free, not the value pool.)

## Current state — 5 KV stores, 2 read lanes

Per (slot, layer), the same logical KV is materialized in multiple layouts:

| Store | Type | Role | Read by |
|---|---|---|---|
| `sw_window_cache` | `CudaSlice<bf16>` | sliding-window ring (128 tok), bf16 staging | scalar lane **+** packed into FP8 pool SW region |
| `state.compressor.compressed` | `HiddenStates` bf16 [head_dim, rows] | **VALUE** compressor output | scalar lane **+** packed into FP8 pool comp region |
| `state.indexer.compressor.compressed` | `HiddenStates` bf16 [index_head_dim=128, rows] | **INDEX-KEY** (scoring only) | DSA top-k selector |
| `dsa_key_cache` | `CudaSlice<u8>` | rotated keys for official DSA selection | DSA selector |
| `flashmla_kv_pool` | `TokenKVPool` FP8 PackedBytes, 584 B/tok, page=64 | paged pool: SW region `[0,sw_blocks)` + comp region `[sw_blocks,total)` | FlashMLA lane (default) |

Two attention read lanes over the SAME logical KV:
- **Scalar lane** (`dsv4_swa_attention` / `dsv4_hybrid_attention`, `if !flashmla_used`): reads `sw_window_cache` + `compressor.compressed` (bf16). Documented quality-defective (2026-07-03 win) — inferior, effectively dead.
- **FlashMLA lane** (default): reads `flashmla_kv_pool` (FP8) via host page_table.

Problem: the VALUE lives in 3 bf16 buffers (`sw_window_cache`, `compressor.compressed`) *and* the FP8 pool, each with its own write path. Not a correctness bug (audit above) but latent debt: 3× the value memory, two divergent read lanes, and the pool can't be tiered per-page while a parallel bf16 copy exists.

## Final state — one paged value pool, three layers

```
Shape      block k  ==  tokens [k·cr, k·cr+cr-1]        pure semantics
   │  Dsv4BlockMap  (the ONE index — landed P1)
Index      comp row r → (slot-logical page, in-page row)  single map
   │  page_table  (host-built, drawn at admission)
Physical   page → FP8 pool byte offset                    real memory
```

- **Value collapses to `flashmla_kv_pool` only** (FP8). `sw_window_cache` and
  `compressor.compressed` lose their VALUE-read role; the compressor packs straight
  into the pool via `Dsv4BlockMap`. bf16 value shadows deleted.
- **Scalar lane deleted** (both kernels + 6 dispatch sites).
- **KEPT unchanged**: `indexer.compressor.compressed` (128-dim scores ≠ 512-dim
  values — different data), `dsa_key_cache` (scoring side), `pending_kv`/`prev_overlap`
  (cross-chunk accumulators — irreducible).

## Difference (current → final)

| | Current | Final |
|---|---|---|
| VALUE stores | `sw_window_cache` bf16 + `compressor.compressed` bf16 + FP8 pool | FP8 pool only |
| block→(page,row) map | inline `sw_blocks + r/64` at ≥3 sites | one `Dsv4BlockMap` (landed) |
| attention read lanes | scalar (bf16) + FlashMLA (FP8) | FlashMLA only |
| value memory / slot | ~3× (bf16 shadows + FP8) | 1× (FP8) |
| L2/L3 tiering | whole-slot blobs (can't split a tangle) | per-page (flat FP8 pages) |
| indexer key / dsa_key_cache | separate stores | **unchanged** |

## Phases (P1/P2/P4 landed; P3 remaining)

- **P1 ✅** (611d18cd) — `Dsv4BlockMap` collapses the inline comp-row→(page,row) map. Byte-identical.
- **P2 ✅** (2dd9d07c) — single-source `sw_blocks`/`page_size` from the arena through the pack/index kernels (no host-transform move; the decode kernel already routes via page_table). Byte-identical.
- **P4 ✅** (2dd9d07c) — CSA-only select-boundary assert `indexer_rows == value_rows`, mode-gated off GLM/frozen.
- **P3 (remaining) — delete the bf16 value shadow + scalar lane.** Remove
  `compressor.compressed`'s VALUE-read role and `dsv4_swa/hybrid_attention_*`
  (kernels + 6 dispatch sites). Compressor packs straight into the FP8 pool via
  `Dsv4BlockMap`. Precondition: prove `flashmla_used` is unconditionally true for
  every supported DSv4 mode (else the scalar path is live — stop). Pure hygiene;
  gate on H20: decode tok/s A/B non-regression + needle ≤2K unchanged (NOT a #146 fix).
- **P5 (follow-on)** — page-granular L2/L3: value KV as flat FP8 pages → tier
  demote/promote per page not whole slot (`executor.rs:2469`) → unblocks >200K.

## #146 — tracked separately (content precision, not this refactor)

Signature: deep-position compressed-value content, trailing-digit loss,
deterministic. Candidates: (a) compressor bf16 compute at high abs_pos, (b) FP8
quant precision. Split with a value-dump A/B (dequant FP8 pool row vs bf16
`compressed.data` row vs high-precision recompute), NOT with a needle depth-sweep
(no control ⇒ confounded by model recall). A separate issue; this refactor neither
fixes nor blocks it.

Also surfaced by the audit: **`num_slots` hardcoded 256** caps TP=4 at max_seq≲8800
(32K needs TP=8) — the real blocker for long-context testing. Separate ticket.

## Non-goals

- Not unifying indexer key INTO the value pool (128-dim scores vs 512-dim values).
- Not touching the SW ring's fixed 128-token semantics.
- Not the FP4 lane (#137, fail-closed).
