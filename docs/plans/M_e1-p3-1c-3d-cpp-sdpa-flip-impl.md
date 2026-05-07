# M_e.1 P3.1c.3d — C++ SDPA flip implementation note

> Final atomic commit of the M_e.1 paged-KV unlock. Reading the actual
> C++ structure (`Qwen35CompiledModel::forward_impl` →
> `attention_block` at `crates/mlx-sys/src/mlx_qwen35_model.cpp:718-855`)
> made the change much smaller than the original L estimate suggested.
> Captured here so the next session can land it without re-reading.

## 0. Smallest-viable design

**Reuse the existing `kv_caches` input slots** instead of adding new
`k_full_per_state / v_full_per_state` slots. The latter were added in
P3.1c.3b for symmetry with `step_session_paged`, but they create a
redundant input path.

Single-flag protocol:

- `ForwardContext.has_paged_kv: bool` — gates the attention compose.
- When `false` (default, legacy): `kv_caches[i]` shape is `[B, nkv,
  kv_capacity, hd]`; attention uses `slice_update + slice` to compose
  `k_full = [B, nkv, cache_pos+1, hd]`.
- When `true` (paged): `kv_caches[i]` shape is `[B, nkv, cache_pos, hd]`
  (Rust-side pool gather); attention uses
  `concatenate({k_cache, k}, axis=2)` to compose
  `k_full = [B, nkv, cache_pos+1, hd]`.

Output `new_k_cache` / `new_v_cache` in the paged branch are placeholders
(can be `k_full` / `v_full` themselves) — pool is the source of truth,
Rust-side dual-write owns state, the graph's "updated cache" outputs are
ignored when `has_paged_kv` is true.

The redundant `k_full_per_state / v_full_per_state` C arguments stay
in the FFI surface (P3.1c.3b) as a no-op; remove them in P3.1d (legacy
retire) or earlier as a follow-up.

## 1. Concrete C++ diff

In `attention_block` at line 829-833 (the single-batch fall-through
branch — the `cache_pos_arr` branch at 767-827 is for verify mode and
unaffected):

```cpp
// ---------- BEFORE ----------
} else {
    int end = cache_pos + S;
    new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {B,nkv,end,hd});
    new_v_cache = slice_update(v_cache, v, {0,0,cache_pos,0}, {B,nkv,end,hd});
    k_full = slice(new_k_cache, {0,0,0,0}, {B,nkv,end,hd});
    v_full = slice(new_v_cache, {0,0,0,0}, {B,nkv,end,hd});
}

// ---------- AFTER ----------
} else if (ctx.has_paged_kv) {
    // Paged-KV: kv_caches input is already-gathered history of shape
    // [B, nkv, cache_pos, hd]. Append new K/V along seq axis. Output
    // cache placeholders are ignored by Rust (pool owns state).
    k_full = concatenate({k_cache, k}, 2);
    v_full = concatenate({v_cache, v}, 2);
    new_k_cache = k_full;
    new_v_cache = v_full;
} else {
    int end = cache_pos + S;
    new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {B,nkv,end,hd});
    new_v_cache = slice_update(v_cache, v, {0,0,cache_pos,0}, {B,nkv,end,hd});
    k_full = slice(new_k_cache, {0,0,0,0}, {B,nkv,end,hd});
    v_full = slice(new_v_cache, {0,0,0,0}, {B,nkv,end,hd});
}
```

In `ForwardContext` at line 544-559:

```cpp
struct ForwardContext {
    int cache_pos = 0;
    // ... existing fields ...
    bool has_paged_kv = false;  // M_e.1 P3.1c.3d
};
```

In `forward()` at line 1316-1345:

```cpp
ctx.has_paged_kv = current_has_paged_kv;
```

Add a class field `current_has_paged_kv = false` next to the other
`current_*` fields, set by FFI entry points.

In `qwen35_compiled_step_batch_paged` at line 2372 (existing P3.1c.3b
body):

```cpp
// Replace the (void)... casts with:
m->current_has_paged_kv = (k_full_per_state != nullptr && n_full_layers > 0);
// kv_caches input is the gathered-history-style layout when paged;
// existing concatenation across batch_size in the loop produces
// [B, nkv, cache_pos, hd] shape, fed to the graph's k_cache slot.
```

Reset `current_has_paged_kv = false` at the bottom of step_batch_paged
(symmetric to existing `current_batch_size = 1` reset).

Same flag plumbing in `qwen35_compiled_step_session_paged`.

## 2. Rust-side change

`infer/src/backend/metal/request_state.rs::decode_qwen35_batch` —
when `all_pool` is true, replace `flat_kv` from `cpp.kv_flat` with
pool-gathered K/V (shape `[1, nkv, cache_pos, hd]` per state per
layer). The existing concat-across-batch logic in C++
`step_batch_paged` at line 2313-2320 then produces
`[B, nkv, cache_pos, hd]` correctly for the `attention_block` paged
branch.

```rust
// In decode_qwen35_batch when all_pool:
let n_full = states[0].driver.arch.num_full_attention_layers();
let mut flat_kv = Vec::with_capacity(states.len() * n_kv_per_request as usize);
for state in states.iter_mut() {
    let pool = state.driver.kv_pool.as_mut().expect("all_pool ⇒ Some");
    // For each full attention layer, push K then V (matching
    // cpp.kv_flat ordering [k0, v0, k1, v1, ...]).
    for layer_idx in 0..n_full {
        let (k, v) = pool.gather_kv(layer_idx, METAL_REQUEST_STATE_ID)?;
        flat_kv.push(k);
        flat_kv.push(v);
    }
}
// Pass flat_kv to step_batch_paged; C++ sees has_paged_kv=true and
// composes k_full via concatenate.
```

The `k_full_per_state / v_full_per_state` args from P3.1c.3b can stay
empty — the flag is what triggers the paged branch.

Same pattern for `Qwen35StepDriver::run_cpp_step_paged` (single-stream,
P3.1c.2 already gathers; just keep the gather and let the C++ flag
fire).

## 3. Acceptance gates

Per master analysis decomposition:

- c=4 long ITL p50 ≤ 9.3 ms (currently 19.20 ms after P3.1c.3c)
- c=16 long ITL p50 ≤ 12 ms (currently 82 ms)
- c=16 output tok/s ≥ 350 (currently 78)
- c=1 long ITL p50 ≤ 1.05× current 4.15 ms (no single-stream regression)
- Logits parity: same first 32 decode tokens as legacy (token-by-token
  match; the `concatenate` path is mathematically identical to
  `slice_update + slice` so this should hold by construction)

## 4. Risks

1. **MLX `concatenate` along axis 2** vs `slice_update + slice`:
   the legacy path produces a contiguous tensor with positions
   `[0..end)` filled. `concatenate` produces the same shape but
   different memory layout (depending on MLX's lazy graph). SDPA
   should not care about layout, but verify with metal capture if
   logits drift.
2. **`new_k_cache` / `new_v_cache` placeholders**: in the paged branch
   we set them to `k_full` / `v_full` which has shape `[B, nkv,
   cache_pos+1, hd]`, NOT the legacy `[B, nkv, kv_capacity, hd]`. The
   `out_kv_caches` slice-back loop at line 2345-2351 splits along
   axis 0 (batch dim), which works for any seq length. But the Rust
   side then puts these back into `cpp.kv_flat` at line 3107-3122 of
   `request_state.rs`. Subsequent legacy step_batch calls (if user
   toggles --kv-pool off mid-run) would see un-padded `cpp.kv_flat`
   and crash at slice_update. Fix: when paged, do NOT write back to
   `cpp.kv_flat`. Pool is the source of truth.
3. **`prev_outputs` lifetime**: the graph keeps intermediates for
   GPU buffer reuse; verify the paged path doesn't leak.
4. **MLX version**: the `concatenate` call at axis 2 of a
   `[B, nkv, cache_pos, hd]` tensor needs to compile cleanly. Should
   work on MLX 0.31.1 (MLX has full concatenate support); verify with
   smoke before benching.

## 5. Test path

1. Unit-style smoke: c=1 single-stream "Say hi" 5 tokens with
   --kv-pool ON. Logits should match --kv-pool OFF.
2. c=1 long bench: 4096-in/256-out, expect ITL ≤ 1.05× current
   4.15 ms.
3. c=4 long bench: expect ITL p50 ≤ 9.3 ms (the unlock).
4. c=16 long bench: expect output tok/s ≥ 350.

If c=1 logits drift, debug with metal capture: re-run with
INFER_CAPTURE_STEP=1 and inspect the gputrace SDPA inputs.

## 6. Effort estimate

- C++ changes: ~25 LOC across forward_impl, ForwardContext,
  qwen35_compiled_step_batch_paged, step_session_paged.
- Rust changes: replace flat_kv source in decode_qwen35_batch
  (~15 LOC) + same in run_cpp_step_paged (already done in P3.1c.2,
  just don't pass empty arrays).
- Build + bench cycle: 1 cargo build (~40s) + 4 bench cells (~2 min
  each) = ~10 min.

**Total: M effort, single tick if focused.** The original L estimate
was based on assuming a full graph rewrite; the smallest viable
design just adds 5 lines of conditional in attention_block.

## 7. Cross-references

- Plan parent: [`M_e1-metal-paged-kv-hot-path.md`](M_e1-metal-paged-kv-hot-path.md)
- C++ surface (P3.1a + P3.1c.3b): commits `23fa52d` + `75e3d76`
- Rust call-site switches (P3.1b + P3.1c.3c): commits `df90ff0` + `c97afc5`
- Rust pool population (P2.0–P3.1c.1, P3.1c.3a): commits e25d617,
  60a9b32, 77b5c2a, ee5ad4e
- Master decomposition: `docs/projects/2026-05-07-metal-optimization-master-analysis.md`
- Audit errata that found c=4 wasn't on the right path:
  `docs/experience/errors/2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md`
