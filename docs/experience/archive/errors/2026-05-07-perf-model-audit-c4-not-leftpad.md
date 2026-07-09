# Perf model audit — c=4 ITL gap may NOT be left-padding — 2026-05-07

## Context

The morning's master analysis decomposed the c=4 ITL gap as
`2.70× = 1.29× per-token kernel × 2.09× ARLE batching multiplier`,
and assigned the 2.09× to "left-padding overhead" because
`Qwen35PackedDecodeBatch.packed_kv_flat` carries a shared cache + per-row
`left_padding`. The M_e.1 plan + 17 commits of paged-KV preparation
(P2.0 → P3.1c.3c) were predicated on flipping SDPA to read from
MetalKVPool to eliminate that left-pad overhead.

When preparing P3.1c.3d (the actual SDPA flip) I re-read the c=4 hot
path more carefully and found a hole in the original analysis.

## Finding

The c=4 bench dispatches through `decode_qwen35_batch`
(`request_state.rs:2976`), NOT `Qwen35PackedDecodeBatch`. Two
distinct c≥2 paths in the tree:

| Path | When dispatched | Cache shape per state | Left-pad? |
|---|---|---|---|
| `decode_qwen35_batch` | All states have **same cache_len** (line 3007-3010 invariant) | `[1, nkv, kv_capacity, hd]` per layer per K/V | **No** |
| `Qwen35PackedDecodeBatch` | Variable cache_len across rows | shared `packed_kv_flat` + per-row `left_padding` | **Yes** |

A sustained c=4 bench with identical prompt template starts all 4
requests at the same decode position, advances them together, and
stays in `decode_qwen35_batch`. `Qwen35PackedDecodeBatch` only fires
when requests join the batch at different positions (variable-length
in-flight), which our matched-c bench does not exercise.

Inside `decode_qwen35_batch` at the C++ side
(`mlx_qwen35_model.cpp:2313-2320`):

```cpp
for (int kv_idx = 0; kv_idx < n_kv_per_request; ++kv_idx) {
    std::vector<array> per_request;
    for (int b = 0; b < batch_size; ++b) {
        per_request.push_back(*to_arr(kv_caches[b * n_kv_per_request + kv_idx]));
    }
    inputs.push_back(concatenate(per_request, 0));  // batch dim
}
```

Each state's K/V is concatenated across batch_size = 4. The result is
`[4, nkv, kv_capacity, hd]`. Inside `attention_block` (line 829-833):

```cpp
new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {B,nkv,end,hd});
k_full = slice(new_k_cache, {0,0,0,0}, {B,nkv,end,hd});
```

`k_full` shape is `[4, nkv, end, hd]` where `end = cache_pos + 1` for
decode. There is **no left-padding** — every row has exactly
`end` valid keys.

mlx-lm at c=4 likely produces the same shape SDPA input via its own
batched-attention path. So both backends feed SDPA the same shape and
the gap **cannot be left-pad**.

## Where the 2.09× actually lives (hypotheses, not yet verified)

1. **Per-state cache concat-then-split overhead.** Every step,
   `decode_qwen35_batch` builds `flat_kv` from per-state `cpp.kv_flat`
   by extending across states (line 3049-3059), the C++ side then
   `concatenate`s across batch dim, and the post-step loop at line
   3111-3122 splits the batched output back into per-state arrays.
   At B=4 this is 4× more FFI traffic + 4 concat/split ops per step
   that B=1 does not pay. mlx-lm may keep K/V as a single batched
   tensor across requests rather than per-state arrays.
2. **Concat-then-slice churn inside the compiled graph.** Even though
   the per-step `slice_update + slice` produces an `[B, nkv, end, hd]`
   tensor — supposedly the same shape SDPA gets at B=1 — the path
   includes a `slice_update` on a `[B, nkv, kv_capacity, hd]` tensor,
   which on Metal may force a full-buffer write that takes O(kv_capacity)
   time even when only one column is being added.
3. **Per-step Rust ↔ C++ FFI cycles.** Each step sends 12 (= 6 layers
   × 2 K/V) `cpp.kv_flat` arrays and receives 12 updated arrays. At
   B=4 vs B=1 the Rust-side iteration overhead, while small per call,
   accumulates over 256 decode steps × 4 requests.

Without metal capture / instruments, none of these can be ranked.

## Implications for M_e.1 P3.1c.3d

The plan's stated acceptance — `c=4 ITL p50 ≤ 9.3 ms` from "paged-KV
eliminates left-padding" — is **not supported by the actual code
structure**. Specifically:

- Replacing `slice_update + slice` with `concatenate({history, k}, 2)`
  in `attention_block` will produce the **same `k_full` shape SDPA
  consumes today**. SDPA work doesn't shrink.
- The win COULD come from skipping `slice_update`'s O(kv_capacity)
  write (hypothesis 2 above), if Metal really treats `slice_update`
  as a full-buffer write. But this is not verified.
- The hypothesis 1 cost (per-state concat-then-split) is unchanged
  by the proposed P3.1c.3d — it still flat-extends per-state K/V,
  C++ still concats across batch.

So **P3.1c.3d as designed is NOT guaranteed to land near the 9.3 ms
target**. It may be a small win (close to noise), zero-impact, or even
a small regression depending on which hypothesis dominates.

## What this means for the 17 commits already shipped

P2.0 → P3.1c.3c are still **structurally correct**:

- MetalKVPool is allocated and populated correctly for both single-
  stream (Qwen35StepDriver) and batched (decode_qwen35_batch) paths.
- Pool data integrity verified by smoke tests across all bench cells.
- C++ FFI surfaces (step_session_paged, step_batch_paged) are wired
  and behaviorally identical to their non-paged counterparts.

But they do NOT close the bench gap. They are **infrastructure** for
a future change that needs additional perf-model work (profiling) to
identify the actual c=4 cost driver.

The 2.09× gap may be addressable by a different change entirely
(e.g. shared batched cache, kernel fusion, fewer FFI round-trips)
that the pool work doesn't directly enable.

## Action items

1. **Don't ship P3.1c.3d blind.** Revert/skip until the actual c=4
   bottleneck is profiled.
2. **Profile c=4 ARLE vs mlx-lm** with metal capture or `mlx
   instruments` — identify which of the three hypotheses dominates.
   This is M_e.0 (per-token / per-step kernel profile, originally
   demoted to follow-up) re-promoted given the audit hole.
3. **Validate hypothesis 1** by experimenting: replace per-state
   `flat_kv.extend(cpp.kv_flat...)` with a single batched cache
   maintained across all states. Bench c=4 to see if it moves
   the needle — even before any pool work.
4. **Update master analysis** with the corrected gap composition
   once profiling lands.

## Lesson

When a plan's acceptance criterion derives from "X explains Y%
of the gap" without direct profiling evidence, **verify X is the
real source of Y%** before shipping infrastructure for it. The
morning analysis identified a plausible cause (left-pad in
PackedDecodeBatch) but didn't trace WHICH path the c=4 bench
actually uses. The 17 commits remain valuable as paged-KV
substrate, but the headline perf claim was unverified.

Stack this with `feedback_substrate_audit_grep_full_tree.md` and
`feedback_ffi_session_owns_data.md` as a third audit lesson:
*before declaring a perf gap's cause, profile the actual hot
path; don't infer from code that may not be on it.*

## Cross-references

- Master analysis (now needs correction):
  [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../../projects/2026-05-07-metal-optimization-master-analysis.md)
- M_e.1 plan + commit ladder:
  [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md)
- P3.1c.3d impl note (the unshipped commit):
  [`docs/plans/M_e1-p3-1c-3d-cpp-sdpa-flip-impl.md`](../../plans/M_e1-p3-1c-3d-cpp-sdpa-flip-impl.md)
- c=1 isolation decomposition (the one verified piece of evidence):
  [`docs/experience/wins/2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md`](../wins/2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md)
- Earlier audit error (also missed the actual hot path):
  [`docs/experience/errors/2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md`](2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md)
