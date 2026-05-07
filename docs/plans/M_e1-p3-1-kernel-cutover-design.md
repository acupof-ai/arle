# M_e.1 P3.1 — Kernel cutover design (Qwen3.5 SDPA reads from MetalKVPool)

> Sub-plan of [`M_e1-metal-paged-kv-hot-path.md`](M_e1-metal-paged-kv-hot-path.md) §3 P3.1.
> Prerequisite: P2.2 dual-write working (commit `60a9b32`) — pool now
> carries the same K/V as the session.
> Acceptance: c=4 ITL p50 ≤ 9.3 ms (currently 19.11 ms),
> c=16 ITL p50 ≤ 12 ms, c=16 output tok/s ≥ 350.

## 0. The exact code to change

C++ SDPA call site at
[`crates/mlx-sys/src/mlx_qwen35_model.cpp:840-855`](../../crates/mlx-sys/src/mlx_qwen35_model.cpp):

```cpp
// Current (left-pad cache):
new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {B,nkv,end,hd});
new_v_cache = slice_update(v_cache, v, {0,0,cache_pos,0}, {B,nkv,end,hd});
k_full = slice(new_k_cache, {0,0,0,0}, {B,nkv,end,hd});
v_full = slice(new_v_cache, {0,0,0,0}, {B,nkv,end,hd});

attn_out = fast::scaled_dot_product_attention(q, k_full, v_full, ...);
```

After P3.1:

```cpp
// Paged: k_full / v_full come from MetalKVPool.gather_kv_rows on
// the Rust side and are passed as direct inputs to the forward graph.
// The compiled session no longer slice_updates a per-request cache;
// the pool is the single source of truth.
attn_out = fast::scaled_dot_product_attention(q, k_full, v_full, ...);
```

## 1. Graph topology change

Today's compiled `forward` graph for the step path (line 1901 area):

```
Inputs:  [token_ids, k_cache_0, v_cache_0, k_cache_1, v_cache_1, ..., gdr_states...]
         (kv_caches passed as graph inputs from the active session)

Forward body:
  for each full attention layer:
    1. q, k, v = qkv_proj(x)
    2. k_cache_new = slice_update(k_cache, k, ...)
    3. k_full = slice(k_cache_new, ..., end)
    4. attn_out = SDPA(q, k_full, v_full)
    5. (continue to MLP, residuals, etc.)

Outputs: [logits, k_cache_new_0, v_cache_new_0, ...]
         (cache deltas captured back into session)
```

After P3.1:

```
Inputs:  [token_ids, k_full_0, v_full_0, k_full_1, v_full_1, ...]
         (k_full / v_full are pre-gathered tensors from the pool;
          shape [B, nkv, current_seq_len, hd] per layer per K/V;
          NO leftover-cache_capacity dimension)

Forward body:
  for each full attention layer:
    1. q, k, v = qkv_proj(x)
    2. (still need to write new K/V somewhere — but pool already has it
        post-P2.2. So: compose k_full' = concat(k_full, k) along seq dim,
        OR: trust the pool already has the new column written and
        gather is shape-current. Architectural choice — see §3.)
    3. attn_out = SDPA(q, k_full', v_full')
    4. (continue)

Outputs: [logits]   (no cache deltas — pool owns state)
```

## 2. Two viable shapes for the cutover

### 2a — Pre-write to pool, gather, feed to graph

Order of ops per step:

1. Rust: compute new token's K/V (need access to qkv_proj output —
   currently lives only on C++ side). NOT TRACTABLE without bigger
   surgery; qkv_proj is a fused op inside the compiled graph.

2. **Skip — does not match the compiled-graph constraint.**

### 2b — Graph computes K/V, writes to pool inside graph, gathers, SDPA

Order of ops per step:

1. C++ graph receives gathered K/V from pool as input (covers indices
   `[0, cache_pos)`).
2. C++ graph computes new q/k/v from input embedding.
3. C++ graph appends new k/v to the gathered K/V (concat along seq
   axis) → k_full / v_full of length `cache_pos + 1`.
4. C++ graph writes new k/v to pool (or returns it to Rust which
   writes — but writing inside the graph keeps the FFI cost down).
5. SDPA on (q, k_full, v_full).

**This is the implementable shape.** The pool acts as the cross-step
storage; per-step graph extends it by one row in-flight; result
materialized back to pool by next step's gather.

### 2c — Hybrid: keep dual-write path, switch only SDPA input source

Lighter cutover:

1. P2.2 dual-write continues — every step writes new K/V to pool from
   Rust. This already works.
2. Graph still gets the legacy left-pad cache as input AND ALSO gets
   gathered K/V from pool (two parallel inputs). SDPA reads from the
   pool-gathered tensors; the legacy cache is updated inside the
   graph but NOT consumed (would be deletion-style refactor in P4.2).
3. Pool data must be byte-equal to legacy cache slice (the parity
   contract from P2.3).

This keeps the existing graph shape mostly intact — adds new inputs,
changes only SDPA arg sources. Effort: **M (smaller than 2b)**.

## 3. Recommended implementation order

Land these as separate atomic commits:

**P3.1a** — Add `qwen35_compiled_step_session_paged` C entry point that
takes pre-gathered `k_full / v_full` arrays as input (one per full
attention layer × 2 for K and V). For now, the new entry point is
WIRED but not called from Rust — landing the C++ surface lets the
parity-test path use it.
- Effort: M
- Files: `crates/mlx-sys/src/mlx_qwen35_model.cpp`,
  `crates/mlx-sys/src/lib.rs`, `infer/src/backend/metal/qwen35.rs`
  (Rust wrapper)
- Acceptance: cargo build green; new entry point silently fails-over
  to legacy when called with empty inputs.

**P3.1b** — Switch Qwen35StepDriver::run_step to gather K/V from pool
and call `step_session_paged`. The compiled graph still computes new
k/v inside; no SDPA read-source change yet. Verify forward correctness
by comparing logits to legacy step_session output across N tokens.
- Effort: M
- Files: `metal/request_state.rs`, possibly `metal/qwen35.rs`
- Acceptance: logits match legacy ±tolerance for the first 32 decode
  steps. Bench c=4 baseline regression-only (expect parity).

**P3.1c** — Inside the compiled graph, switch SDPA input from the
session left-pad cache to the gathered K/V appended-with-new-row.
This is the actual unlock — eliminates left-pad overhead.
- Effort: L (the real cutover, including possibly recompiling the
  graph variant)
- Acceptance: **c=4 ITL p50 ≤ 9.3 ms; c=16 ITL p50 ≤ 12 ms; c=16
  output tok/s ≥ 350**. ITL p95 c=16 ≤ 15 ms. c=1 long ITL p50
  stays ≤ 1.05× pre-cutover 4.37 ms.

**P3.1d** — Retire the slice_update + slice legacy cache code in the
graph. Pure deletion-style refactor (`feedback_no_half_states.md`).
- Effort: S
- Acceptance: graph still produces same logits as P3.1c.

## 4. Risks specific to P3.1

1. **Compiled graph re-compilation cost.** Adding a new step entry
   point may force a fresh `compile()` call at session begin, costing
   seconds the first time. The existing `prepare_session` call site
   needs to be inspected.
2. **MLX SDPA shape constraints.** The `fast::scaled_dot_product_attention`
   may have shape requirements (e.g. seq_len divisibility). Gathered
   K/V from the pool will have `current_seq_len = N` for arbitrary N.
   Verify the kernel handles N=1, N=4096, etc.
3. **GDR (gated delta rule) layers** — Qwen3.5 has 18 GDR layers and
   only 6 full-attention. P3.1 only changes full-attention layers;
   GDR state remains in `session_gdr_states`. Confirm the new C entry
   point keeps GDR plumbing identical to current `step_session`.
4. **Concurrency.** Changing the compiled graph for single-stream
   first; packed-decode (Qwen35PackedDecodeBatch) is the c≥2 path
   with shared-cache + left-padding. Cutover for packed decode is a
   separate workstream after P3.1c lands and validates the approach.

## 5. What stays the same

- DFlash speculative decode path (different graph already).
- Qwen3 plain decode (separate code path).
- The C++ session lifecycle (begin_session / end_session) — only the
  step entry point changes.
- All Rust-side scheduler logic.
- The MetalKVPool internals (already supports gather_kv).

## 6. Cross-references

- Prerequisites: commits `cb1fcc3` (P2.1 FFI) + `60a9b32` (P2.2
  dual-write).
- Plan parent: [`M_e1-metal-paged-kv-hot-path.md`](M_e1-metal-paged-kv-hot-path.md) §3 P3.1.
- Errata established that CPP path owns K/V opaquely:
  `M_e1-metal-paged-kv-hot-path.md` §7.4–7.5.
- C++ SDPA call site:
  `crates/mlx-sys/src/mlx_qwen35_model.cpp:840-855`.
- Master analysis decomposition (the 2.09× batching gap this commit
  closes):
  [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../projects/2026-05-07-metal-optimization-master-analysis.md).
