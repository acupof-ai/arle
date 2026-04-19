# FlashInfer planned-prefill — Plan

> **Superseded (2026-04-17):** the shared `BatchPrefillPagedPlan` landed in
> `infer/src/model/qwen3/prefill.rs` already realizes the plan-reuse win this
> doc proposed (one Plan() per model, not per forward). Any further
> planned-prefill work is tracked under
> [`p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md) §Phase 5.

## Status

Active plan as of 2026-04-17. Targets the 100ms single-request-TTFT gap
on Qwen3-4B at 4096-token prompts
(`docs/experience/wins/2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`)
and the per-token prefill cost gap (1.24× sglang past 2048 tokens).

## One-line thesis

Our prefill path calls `SinglePrefillWithKVCacheDispatched`, which has a
fixed static tile scheduler. sglang uses `BatchPrefillWith*` with
`PrefillPlan()` — a CPU-side tile scheduler that maps the work across
available SMs based on actual seq_len, KV size, and SM count. On L4
(58 SMs), a 4096×128 prefill under the static scheduler leaves SMs
idle or imbalanced; the planned scheduler spreads work evenly. We
already have the machinery for **decode** (see
`flashinfer_tc_decode_plan/run` in `flashinfer_tc_decode.cu`); we need
the same pair for **prefill**.

## Current state inventory

| purpose | op | wrapper | has plan? |
|---------|----|---------|-----------|
| Qwen3 prefill (HD=128) | attention | `SinglePrefillWithKVCacheDispatched` | **no** |
| Qwen3.5 full-attn prefill (HD=256) | attention | `SinglePrefillWithKVCacheDispatched` | **no** |
| Qwen3 batched decode (HD=128) | attention | `BatchPrefillWithPagedKVCacheDispatched`, qo=1 | yes (`flashinfer_tc_decode_plan`) |
| Qwen3.5 batched decode (HD=256) | attention | `BatchPrefillWithPagedKVCacheDispatched`, qo=1 | yes |

## What changes

### New C++ kernels (mirror tc_decode pair)

1. **`crates/cuda-kernels/csrc/attention/flashinfer_planned_prefill.cu`** — HD=128.
   - `flashinfer_planned_prefill_plan(...)` — wraps `flashinfer::PrefillPlan<IdType>`
     with `qo_indptr=[0, seq_len]` (single-request), `kv_indptr=[0, kv_len]`,
     `enable_cuda_graph=true`. Writes `PrefillPlanInfo` to an opaque buffer.
   - `flashinfer_planned_prefill_run(...)` — wraps
     `BatchPrefillWithPagedKVCacheDispatched<CTQ, 128, 128, ...>`. Same
     dispatch-on-cta_tile_q switch as tc_decode.

2. **`flashinfer_planned_prefill_hd256.cu`** — same, head_dim=256, for Qwen3.5
   full-attention prefill.

### New Rust FFI layer

`crates/cuda-kernels/src/ffi/attention.rs`: add `planned_prefill_plan`
and `planned_prefill_run` bindings + one `PrefillPlanInfo` opaque buffer
type (~200 bytes, carry as `[u8; 256]`).

### Plan cache in Rust

Cache key: `(head_dim, num_qo_heads, num_kv_heads, page_size, seq_len)`.
Value: opaque `PrefillPlanInfo` buffer + an invalidation flag if any pool
workspace is reallocated.

Where the cache lives:
- Per-slot inside `Qwen3Model::PrefillBuffers` (or Qwen35's equivalent),
  since seq_len changes per request and cache hit rate is highest when
  the same seq_len bucket repeats.
- Invalidated whenever `num_kv_heads`, `page_size`, or workspace ptrs
  change (startup + KV-tier migration).

### Prefill forward callers switch

In `infer/src/ops/attention.rs` (or wherever the single-prefill entry
point lives — confirm during Explore):

```rust
// old
unsafe {
    flashinfer_single_prefill(
        q.as_ptr(), k_cache, v_cache, out, ...
    );
}

// new
let plan = plan_cache.get_or_build(seq_len, kv_len, ...);
unsafe {
    flashinfer_planned_prefill_run(
        workspace, int_workspace, plan.as_ptr(),
        q.as_ptr(), q_indptr_device, k_data, v_data,
        kv_indptr_device, kv_indices, kv_last_page_len,
        out, lse, 1, num_qo_heads, num_kv_heads, page_size, sm_scale, stream
    );
}
```

### CUDA Graph friendliness

Key subtlety: `PrefillPlan()` runs CPU-side and writes to a pinned buffer
that's then uploaded to int_workspace. That cannot happen inside a graph
capture — but the **captured graph can reference a prebuilt plan buffer**,
since its contents are deterministic for a given shape.

This is why the plan cache is critical for the P1 full-forward graph:
graph capture at shape S uses the plan built for S. When S is replayed,
the plan buffer is already populated. Capture-time plan build is
one-off; replays reuse the same buffer.

## Files to touch

| File | Change |
|------|--------|
| `crates/cuda-kernels/csrc/attention/flashinfer_planned_prefill.cu` | **new** — HD128 variant |
| `crates/cuda-kernels/csrc/attention/flashinfer_planned_prefill_hd256.cu` | **new** — HD256 variant |
| `crates/cuda-kernels/src/ffi/attention.rs` | + 4 extern C bindings + opaque `PrefillPlanInfo` type |
| `infer/src/ops/attention.rs` | switch `single_prefill` callers to plan→run path |
| `infer/src/model/qwen3/prefill_buffers.rs` | + plan cache struct, workspace |
| `infer/src/model/qwen35/prefill_buffers.rs` | same |
| `crates/cuda-kernels/build.rs` | (no change — `flashinfer_*.cu` already matched by the stem prefix) |

## Acceptance criteria

1. `cargo build --release -p infer` clean; `cargo test --release --test e2e`
   passes baseline (prefill forward numerical output must match).
2. Single-request 4096-token TTFT **drops 50ms+** on Qwen3-4B
   (baseline 797ms; target ≤ 747ms). 100ms-class gap closure expected
   based on sglang delta.
3. Qwen3.5 4096-token TTFT drops similarly on the HD256 path.
4. Plan build is one-off per shape — cache hit rate > 90% across a
   `prompt_tokens=4096` guidellm sweep.
5. `cargo check -p infer --no-default-features --features cuda,no-cuda`
   still green (stubs unchanged — planned prefill adds new symbols only
   behind `cuda`).

## Risks

1. **PrefillPlanInfo layout drift.** If flashinfer version bumps change
   the struct, our opaque `[u8; N]` carry must keep up. Guard with a
   compile-time `sizeof(PrefillPlanInfo)` assert in the .cu file.
2. **Workspace sharing with tc_decode.** Both variants use workspace
   buffers. Need to confirm concurrent (or alternating) usage doesn't
   corrupt state. sglang allocates separate workspaces per-wrapper; do
   the same.
3. **Plan cache stale on KV workspace realloc.** If the paged pool
   resizes (it shouldn't during steady-state, but during KV-tier
   migration it might), cached plans point at stale offsets. Invalidate
   on workspace ptr change.
4. **SinglePrefill might already be near-optimal for small seq_len.**
   Crossover at our measured n ≈ 1024 suggests SinglePrefill is fine
   below 1024. Consider routing small prefills through SinglePrefill and
   only using planned prefill for seq_len > 1024. Keep both paths alive
   at first, pick one after benchmarks.

## Benchmark methodology

1. `scripts/bench_guidellm.sh qwen3-4b-infer-l4-planned-prefill` after.
2. TTFT scan (`/tmp/ttft_probe.py`) at 128/512/1024/2048/4096 before-after.
3. Graph-capture amortisation: measure first-call plan time (~µs) vs
   cached-call run time at seq_len ∈ {1024, 2048, 4096}.
4. Pair with Qwen3.5 measurement to confirm HD256 path closes similar %.

## Order of execution

1. Write HD128 plan+run .cu files first.
2. Wire Rust FFI + callsite switch.
3. Validate e2e test baseline.
4. Bench — if TTFT drops ≥50ms, proceed to HD256. Else root-cause
   before spreading the pattern.
5. Write HD256 variant.
6. Bench — publish paired wins entry citing this plan.

## Deferred / out of scope

- Multi-request batched prefill. Our scheduler issues one prefill chunk
  at a time today; batch support is orthogonal.
- Chunked-prefill with planned prefill. Future; the key win is
  single-request planned prefill first.

## Rule

When sglang wins a kernel-level latency comparison where we share the
same FlashInfer backend, the cause is almost always "they use the
planned variant; we use the single/static variant." The planned variant
is 1-2 days of glue code; the kernel work is already done by FlashInfer.
This should be the first place to look when a kernel-level gap exists.
