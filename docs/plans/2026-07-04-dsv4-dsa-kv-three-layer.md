# DSv4 KV storage — audit + map single-sourcing

> Status: Shipped — 2026-07-05. The refactor is complete at P1/P2/P4. Audit
> conclusion: DSv4 KV storage access is correct/race-free [V], the one real debt
> (duplicated block map) is single-sourced, and there is **no remaining KV-storage
> architecture work** — the "collapse to one FP8 value pool" redesign was dropped
> (both premises wrong, §Retracted) and the budget layer is a fail-closed capacity
> gate, not a correctness or architecture issue. Two things remain but are NOT this
> refactor: #146 (a value-content bug) and long-context VRAM headroom (capacity). Every
> claim is tagged **[V]** verified (code/measured, file:line) or **[I]** inferred
> (follows from code, not measured); no untagged conclusions.

## Audit — storage access is race-free on every path  [V]

On HBM a KV access is correct iff **index correct + physical correct +
write-happens-before-read**. GPU adds the third term (async kernels; a cross-stream
write/read without a fence can race). Every path that touches the value pool:

| Path | Pack (write) | Read | Stream |
|---|---|---|---|
| decode-eager | `ctx.stream` | `try_flashmla_decode_attention` (flashmla.rs:238 `cu_stream()`) | single |
| prefill-eager | `flashmla_pack_sw_ring`/`_compressed_delta` `ctx.stream` (attention.rs:2660/2677) | `try_flashmla_prefill_attention` all `ctx.stream` (2091-2586) | single |
| decode-graph | single-row DEVICE pack, `start_pos_device`-derived, records into capture (attention.rs:1909) | capture-audited (attention.rs:4267) | capture-safe |

- `comm_stream` (NCCL) / MTP persistent-verify stream never appear in attention/flashmla → never touch the value pool.  [V]
- Read bounded: `start_row = fp8_kv_comp_packed_rows`, reads `[0, packed_rows)`; `packed_rows ≤ indexer_rows_before` (attention.rs:7935).  [V]
- Index/physical single-sourced: `Dsv4BlockMap` + kernel `page_table` routing masked by `flashmla_total_pages()`.  [V]

All three terms hold on every path ⇒ **storage access is not a #146 source.**  [V]

## Retracted premises (adversarial review, 2026-07-05)

1. **"#146 is content precision."** → **"#146 is not storage access [V]; cause open [I]."**
   The needle deep-position digit loss had no control (model recall confounds it) —
   it does not prove content corruption.
2. **"sw_window_cache bf16 is precision-necessary."** → decode reads the FP8 pool SW
   region (packed from `sw_window_cache`) [V]; near-window is already FP8 in the read
   path. The bf16 buffer exists as the scalar SW-prefill lane's input format, not as
   a precision requirement. Whether FP8 near-window loses accuracy is **not measured [I].**

Both fed the "one FP8 pool" collapse. Dropped: no measured benefit, and blocked (§Not pursued).

## Current state — 5 KV stores, 2 read lanes  [V]

| Store | Type | Role | Read by |
|---|---|---|---|
| `sw_window_cache` | `CudaSlice<bf16>` | SW ring (128 tok) | scalar SW-prefill **+** packed into FP8 pool SW region |
| `state.compressor.compressed` | `HiddenStates` bf16 [head_dim, rows] | VALUE compressor output | scalar hybrid **+** packed into FP8 pool comp region |
| `state.indexer.compressor.compressed` | `HiddenStates` bf16 [128, rows] | INDEX-KEY (scoring) | DSA selector |
| `dsa_key_cache` | `CudaSlice<u8>` | rotated keys for DSA | DSA selector |
| `flashmla_kv_pool` | `TokenKVPool` FP8, 584 B/tok, page=64 | paged pool: SW `[0,sw_blocks)` + comp `[sw_blocks,total)` | FlashMLA lane |

The scalar lane (`dsv4_swa_attention`, `if !flashmla_used`) is **live**: `try_flashmla_prefill_attention` returns `Ok(false)` for `mode == SlidingWindow && chain_verify.is_none()` (attention.rs:2121), so every DSv4-Flash SW-layer prefill runs it (DSv4-Flash interleaves SW + CSA layers; SW = `ratio==0`, v4.rs:492).  [V]

## Budget / pre-allocation layer — a fail-closed capacity gate, NOT correctness  [V]

Interdependent, but out of the refactor's scope: it decides *how long a
`max_seq_len` fits*, never *whether a served config is correct*. kv_layout.rs:1015
is `ensure!` — fits → build succeeds and runs the audited race-free path; doesn't
fit → engine build **aborts** (never silently wrong). Chain (all file:line-verified):

```
requested num_slots (default 256; NOT a serve flag — fully budget-derived)
   │  reserved_for_slots = per_slot × min(requested, affordable)      dsv4.rs:1810-1812
   ▼
pool_budget_total = total_budget − reserved_for_slots                 dsv4.rs:1813-1814
   │  / num_layers → pool_budget_bytes_per_layer
   ▼
pool.max_total_pages  must be ≥ flashmla_slot_pages (= one band)      kv_layout.rs:1015 (ensure, else build abort)
```

**Both `per_slot` (DSA key-cache + batched scratch) and `flashmla_slot_pages` (=sw_blocks+comp_blocks) scale with `max_seq_len`, drawn from one budget [V].** At long `max_seq_len` the remainder after reserving slots can fall below one band → engine build aborts (`FlashMLA pool page mismatch`). So the three — num_slots, per-slot cost, pool size — **cannot be changed in isolation** [V].

`requested` num_slots is NOT user-settable — serve exposes no `--num-slots` flag; it is fixed at the default and fully budget-derived. So there is no knob to trade concurrency for context; a config that doesn't fit needs more VRAM (TP=8 / bigger GPUs), not a flag.  [V]

## What shipped  [V]

- **P1** (611d18cd) — `Dsv4BlockMap` collapses the inline `sw_blocks + r/64` map to one source. Byte-identical.
- **P2** (2dd9d07c) — single-source `sw_blocks`/`page_size` through the pack/index kernels. Byte-identical.
- **P4** (2dd9d07c) — CSA-only `indexer_rows == value_rows` select-boundary guard, mode-gated off GLM/frozen. Catches Shape-layer *row-count* desync only — NOT physical block→page map drift (equal counts can still map to a wrong page). A cheap tripwire, not a map-drift proof.

## Not pursued

- **Delete scalar lane + one-pool collapse** — blocked (scalar SW-prefill live, attention.rs:2121 [V]) and unmotivated (both premises retracted). Prerequisite would be teaching `try_flashmla_prefill_attention` to handle SW-mode prefill (a feature). Deferred, no owner.
- **Page-granular L2/L3 tiering** — was justified only by the collapse. If ever wanted, tier the FP8 compressed-history region alone; no collapse needed.

## Open, separate from storage access

- **#146** — not storage access [V]; cause ∈ {compressor high-pos compute, FP8 quant, model}, undecided [I]. Split with a value-dump A/B (dequant FP8 pool row vs bf16 `compressed.data` row vs high-precision recompute). Own ticket.
- **Long-context cap** — the budget chain above is genuine VRAM exhaustion at long `max_seq_len` (both per-slot DSA cost and the FlashMLA band scale with it); fail-closed, no user knob. Reaching 32K needs more memory (TP=8 / bigger GPUs), not a code change. Own ticket.

## Non-goals

- Not unifying indexer key INTO the value pool (128-dim scores vs 512-dim values).
- Not touching the SW ring's fixed 128-token semantics.
- Not the FP4 lane (#137, fail-closed).
