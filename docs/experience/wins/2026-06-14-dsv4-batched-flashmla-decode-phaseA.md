# DSv4 batched FlashMLA decode — Phase A scaffolding (gated, byte-identical default)

## Context
The #1 concurrency lever ([plan](../../plans/dsv4-batched-flashmla-decode.md)):
the batched decode lane's attention is a per-row `for r in 0..n` loop
(`dsv4.rs:1872`), 58% of the c=8 step and 7.27× ∝c (nsys
[step-profile](2026-06-14-dsv4-config-mechanism-classification.md)). Batching it →
~2× aggregate decode tok/s (130.9→65.3 ms/step at c=8). This entry lands **Phase A**
(the precondition infra) per §0.1 land-and-isolate-on-a-clean-baseline.

## What landed (Phase A, gated `ARLE_DSV4_FLASHMLA_DECODE_BATCHED=1`, default OFF)
Model-wide batched scratch `Dsv4FlashMlaDecodeBatchScratch` (all N-sized buffers +
`block_table`/`slot_block_offsets`), allocated once in `Dsv4KvAdapter::new` under
the existing FlashMLA-alloc gate (`max_batch = num_slots`). Per-forward, for the
n>1 non-CSA lane: build `block_table[N]` (`flashmla_slot_first_block` per row) →
**one batched `build_indices(b=N)`** (the orphaned
`dsv4_flashmla_decode_build_indices_batched_raw`) → **per-forward `sched_meta(b=N)`**
(the cached-meta-is-b=1-only pitfall fix). The per-row attention kernel call is
**KEPT** in Phase A — so flag-ON still produces the per-row result; Phase A only
exercises the batched meta-build path. Phase B (`dsv4.rs` `// PHASE B:` marker)
swaps the per-row loop for `gather → sparse_decode_fwd_batched(b=N) → scatter`
(that kernel call is written + stride-verified, `#[allow(dead_code)]` until wired).

## Buffer enumeration (§0.1 — every mutated batched buffer + precondition)
`indices[max_batch×max_topk_unified]` (build_indices, pool-absolute) ·
`topk_length[max_batch]` · `start_pos[max_batch]` + `slot_block_offsets[max_batch]`
(H2D per forward) · `lse_accum[(num_sm_parts+max_batch)×h_q]` +
`o_accum[(num_sm_parts+max_batch)×h_q×head_dim]` — **the split dim folds b via
num_splits per `arle_flashmla_decode_shim.cu:202`, NOT a `b×accum_rows` axis**
(corrected mid-impl; the 5 accum strides stay single-row) · `sched_meta` recomputed
per forward · `num_splits[max_batch+1]` · `q_batched`/`out_batched`/`tp_gathered_q`
(Phase B gather/scatter, sized but unused).

## Verify
- **Mac CUDARC typecheck** (`infer-api`, `cuda,no-cuda`, on main): clean. infer-cuda
  + examples clean; no new dead-code warnings beyond the marked Phase-B function.
- **Default byte-identical**: flag OFF → the batched scratch is allocated but never
  touched; the n>1 default per-row path is unchanged; N=1 never reaches
  `forward_decode_batch_stream_impl`. B=1 42.0 ms/step + needle hold by construction.
- **Pod — VERIFIED 2026-06-14** (synced 0b70c78c, `INCR_BUILD_EXIT=0`, Phase-A
  symbol present):
  - **Default byte-identical ✓** — flag-OFF + mtp B=1: needle 512/6000 exact ×3,
    **42.68 ms/forward-step** (on the ~42 target), accept 1.69 tok/forward.
  - **Phase-A meta path runs CLEAN + correct ✓** — flag-ON (BOTH gates, see below)
    at c=8: zero `illegal memory access`/CUDA-error/panic/assert, 0 WARN/ERROR; needle
    warm 6/6 @512 + 4/4 @6000 exact (one cold transient 738292 self-corrected, within
    the MoE/cold non-det floor). `build_layer_batch_meta`→`build_indices_batched`+
    `sched_meta_for_batch(b=N)` executed under real c=8 traffic, no fault.
  - **Locked c=8 flag-OFF baseline = 45.60 tok/s** (175.4 ms/step, per-row serial-cap)
    — the Phase B A/B reference.
- **TWO env gates (brief-omission caught in verify):** reaching `forward_decode_batch`
  needs `INFER_DSV4_BATCHED_DECODE=1` (`executor.rs:1563`
  `dsv4_batched_decode_enabled`); the Phase-A meta-build inside it needs
  `ARLE_DSV4_FLASHMLA_DECODE_BATCHED=1`. Phase B's A/B treatment arm must set BOTH.
- **The executor batched lane is NOT a no-op** — isolation run (`INFER_DSV4_BATCHED_DECODE=1`
  only, Phase-A OFF, no-mtp): **45.6 → 67.6 tok/s = +48% at c=8** (correctness-clean).
  This **corrects** the long-ctx campaign's "batched flag no-op / aggregate flat":
  that campaign used `--spec-type mtp`, which *disables* the batched lane, so both
  arms ran MTP-per-row and the flag was inert. The true batched lane (no-mtp) scales;
  Phase A's FlashMLA meta adds no perf change on top (per design — per-row kernel kept).
- **License-to-default-flip is Tier-2** (per understand-until-simple): not flipped;
  Phase B + a c=8 aggregate-rises A/B (treatment = both gates) is the next gate.

## Rule
- **Land the precondition infra gated + byte-identical-default first, verify on a
  clean baseline, then wire the behavior swap** (§0.1). Phase A is opt-in scaffolding
  with zero default-path change; the ~2× is Phase B + a pod perf license.
- The FlashMLA accum buffers are `[num_sm_parts+b, …]` (b folded via num_splits),
  not `[b×accum_rows, …]` — check the shim doc before sizing batched split-KV accums.
- **A "batched" flag that washes can be inert, not no-op** — the long-ctx campaign
  ran `--dsv4-batched-decode` *under* `--spec-type mtp`, which disables the batched
  lane (`executor.rs:1563`), so the flag never engaged. Always confirm the lane
  under test actually executes ([[feedback_verify_slo_lane_runs_before_optimizing]]):
  the no-mtp batched lane is +48% @c=8, not flat.
- **Phase B watch-item (sched_meta aliasing):** `sched_meta_for_batch` overwrites
  the shared `self.sched_meta` scratch with a b=N layout — the SAME buffer the kept
  per-row b=1 `mla_attention` reads. Phase A stayed correct (n matched), but Phase B
  must FULLY replace the per-row kernel with `sparse_decode_fwd_batched(b=N)` before
  this aliasing matters; watch the cold-run needle blip under the A/B as the canary.
