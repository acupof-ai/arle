# DSv4 single-row CSA decode-graph capture — READ lane graph-safe (chunk 1) — pending-remote

`pending-remote`: implemented + Mac-cross-compiled (no nvcc/GPU locally). The
8×H20 / TP=4 pod is the bench + correctness gate (CUDA build, needle ladder, c=1
tok/s) — the human runs it via `bin/pod`. This stub lands per the §Benchmarks
rule (every runtime change → a wins/ entry or pending-remote stub).

## Context

The c=1 decode lever per
[`2026-06-27-dsv4-decode-launch-bound-nsys-roofline.md`](2026-06-27-dsv4-decode-launch-bound-nsys-roofline.md):
DSv4-Flash c=1 is launch-bound (nsys: ~1333 launches/step, 44 s launch + 39 s sync
vs ~12 s GPU compute → GPU ~79% idle; 30 tok/s is 5× off the roofline). The decode
CUDA graph already captures the compressor in-graph but **bails on the CSA SELECT**
(`attention.rs` `"DSv4 decode-graph does not support the official CSA select"`).

The single-row CSA select is two parts:
- **block (a) cache WRITE** (`csa_select_official` hadamard-rotate + fused-store of
  the rows committed this step): host-driven `newly_packed = rows_after -
  packed_rows`, slices `keys.data[..]` → **variable launch shape per step** (0 most
  steps, `ratio` rows on a compression boundary). NOT graph-capturable as-is.
- **blocks (b)-(f) READ** (paged-MQA logits + topk): the batched device-meta select
  (`csa_select_official_batched`, `ARLE_DSV4_DSA_DEVICE_META`) is already fully
  device-side EXCEPT two per-step `upload_i32` (slot_ids, key_counts) — H2D allocs
  the capture audit (`graph.rs` `audit_capturing_graph`, FATAL on host-memcpy nodes)
  rejects.

## What this chunk did (READ lane, eager, gated, needle-ready)

Made the **READ** graph-capturable and reachable for eager needle verification;
left the cache **WRITE** as the documented remaining blocker (see below).

1. `csa_select_official_batched`: added optional pre-staged device
   `slot_ids_dev`/`key_counts_dev`. Both `Some` (+device_meta) → skip the two
   `upload_i32` (read the persistent buffers) → capture-audit-clean. Eager batched
   lane passes `None` → `upload_i32` path **byte-identical**.
2. `Dsv4MlaDecodeGraphScratch`: persistent `csa_slot_id_dev`/`csa_key_count_dev`
   (1 i32 each) + `csa_meta_initialized`. Lazy-init H2D on the eager/warm run only
   (slot_id + key capacity are graph-lifetime constants; `start_pos_device` is
   already updated pre-replay outside capture).
3. `Dsv4LayerAttentionState`: `dsa_official_slot_idx()` +
   `indexer_compressed_capacity()` accessors.
4. `csa_select_decode_graph`: when persistent buffers supplied → cache-writes-only
   pass (block a) + n=1 batched device-meta READ (the `csa_q_i`/`csa_weights`
   scratch ARE the `[width,1]` n=1 batch, no gather). Per-tile READ fallback else.
5. `forward_tokens_decode_graph`: thread `dsa_shared` (was `None`) into the attn
   decode-graph; under `ARLE_DSV4_DECODE_GRAPH_CSA=1` bypass attn-graph capture so
   the read lane runs EAGER (block a not yet capturable) — MoE portion still
   captures.

Gating: `ARLE_DSV4_DECODE_GRAPH_CSA=1` (default OFF) enables the eager graph-safe
read lane. Default `ARLE_DSV4_DECODE_GRAPH` still bails on CSA. Non-CSA / read-lane-
off decode-graph is byte-identical.

## Correctness to verify on the pod (needle, A/B)

- n=1 batched device-meta READ ≡ per-tile single-row READ selection. The on-device
  `context_lens = min(start_pos/ratio, key_capacity)` and `positions = start_pos`
  match the single-row GPU fill byte-for-byte; the block_table band
  (`slot_id*num_pages + b`) reads the same physical DSA cache bytes block (a) wrote
  to `pool.dsa_slot_range(official.slot_idx)` (the #60 batched-lane equivalence).
  **Gate**: needle ladder ×3 same-config repeats with `ARLE_DSV4_DECODE_GRAPH=1
  ARLE_DSV4_DECODE_GRAPH_CSA=1 ARLE_DSV4_DSA_DEVICE_META=1` vs the eager baseline
  envelope (NOT byte-identity — MoE non-determinism).

## Remaining for FULL capture (next chunk — CUDA kernel work)

block (a) cache WRITE must become **device-shape driven**: a new
`dsv4_dsa_pack_index_k_start_pos_ptr`-style fixed-grid kernel that gates each row
on-device from `start_pos` (commit 0/`ratio` rows), mirroring
`dsv4_compressor_update_start_pos_ptr_cuda` (fixed `token_count=1` grid, device-
gated). Then route block (a) through it in the graph path, drop the attn-graph
bypass, and remove the bail — the full single-row CSA decode (compressor already
in-graph + the now-stable write + the now-stable read) captures as one replay.
Expected payoff (per the nsys roofline lever): collapse ~1333 launches/step →
1 replay, ~5× (30 → ~150 tok/s).

## Verify
- Mac: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda --release
  --no-default-features --features cuda,no-cuda --lib` — GREEN.
- `cargo check -p infer-api ... cuda,no-cuda --lib` — GREEN (pre-push gate).
- Commits: `98578f21` (device-meta inputs), `347a7af5` (read lane wiring).
