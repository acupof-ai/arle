## Context
On 2026-04-14 (commit `d902090`, post route-A revert) we attempted to
implement Phase P0 of `docs/projects/tiered-kv-cache.md` — flip the
CUDA paged KV pool from `page_size = 1` to `page_size = 16` for the
BF16 path while keeping quantized formats at 1.

The execution plan
[`docs/plans/tiered-kv-cache-tasks.md`](../../plans/tiered-kv-cache-tasks.md)
§1 listed P0 as ~7 mostly-mechanical files: add a `page_size` field on
the pool, rewrite `build_indptr` / `build_last_page_lens` semantics,
drop the literal `1` in three model `batch_decode.rs` files plus
`scheduler/cuda/decode.rs`, expose a `--page-size` CLI flag.

Mid-implementation exploration (read of every touch point and every
CUDA kernel under `crates/cuda-kernels/csrc/` — at the time of
writing, `infer/csrc/cuda/`; extracted 2026-04-15 Route-A) surfaced an architectural
blocker that makes BF16 + `page_size > 1` produce **silently wrong
attention output**, not a build error. We stopped before writing any
code and recorded this entry.

## Root Cause
The plan doc audited `kv_cache_to_paged_kernel`
([`crates/cuda-kernels/csrc/kv/kv_cache_to_paged.cu:18-51`](../../../crates/cuda-kernels/csrc/kv/kv_cache_to_paged.cu#L18))
and concluded the BF16 migration path was page_size-parametric — the
non-range kernel correctly does `logical_page = pos / page_size` and
writes HND. That's true.

But the production prefill→pool migration does **not** call that
kernel. It calls
[`migrate_kv_range_to_paged`](../../../infer/src/model/kv_cache.rs)
from
[`infer/src/scheduler/cuda/prefill.rs:184`](../../../infer/src/scheduler/cuda/prefill.rs#L184)
and
[`infer/src/scheduler/cuda/prefill.rs:270`](../../../infer/src/scheduler/cuda/prefill.rs#L270),
which dispatches to **`kv_cache_to_paged_range_kernel`** at
[`kv_cache_to_paged.cu:53-82`](../../../crates/cuda-kernels/csrc/kv/kv_cache_to_paged.cu#L53):

```cpp
// Range variant for token-level (page_size=1) paged pools.
int dst_offset = pool_idx * kv_dim
               + kv_head * head_dim
               + dim;
```

This is **NHD per-token**: `kv_dim = num_kv_heads * head_dim`, no
`page_size` parameter, no `logical_page / offset_in_page` decomposition.
The header comment on line 53 explicitly says "for token-level
(page_size=1) paged pools".

FlashInfer reads the same pool buffer via
[`flashinfer_decode.cu:137`](../../../crates/cuda-kernels/csrc/attention/flashinfer_decode.cu#L137):

```cpp
/*layout=*/ flashinfer::QKVLayout::kHND,
```

— unconditionally HND. The HND addressing formula for a token at
sequence position `pos` is:

```
physical_page = page_indices[pos / page_size]
offset = physical_page * (num_kv_heads * page_size * head_dim)
       + kv_head * (page_size * head_dim)
       + (pos % page_size) * head_dim
       + dim
```

NHD-per-token (`pool_idx * kv_dim + kv_head * head_dim + dim`) and
HND-per-page collapse to the same byte offset **only** when
`page_size = 1`. At `page_size = 2`, `num_kv_heads = 2`, `head_dim = 4`,
token at `pos=1` of slot 0 gets:
- NHD write: `1 * 8 + 0 * 4 + d = 8 + d`
- HND read: `0 * 16 + 0 * 8 + 1 * 4 + d = 4 + d`

→ attention reads the value the migration kernel wrote into a
neighbour position. Output is silently garbage, no panic, no compile
error — exactly the kind of failure the
[2026-04-13 batched-decode regression](2026-04-13-batched-decode-high-concurrency.md)
shows is catastrophically hard to bisect after the fact.

### Two further blockers found while verifying the first one
1. **`stride_page` in `ops/attention.rs`** — at the two
   `decode_prep_paged*` call sites (`infer/src/ops/attention.rs:673`
   for HD128, `:1068` for HD256) we pass `stride_page = paged.kv_pool.kv_dim`.
   That formula is `num_kv_heads * head_dim` — correct only when
   `page_size = 1`. For HND with `page_size > 1` the stride must be
   `kv_dim * page_size`. Plan doc §1.2 line 104 mentions this as a
   caller responsibility, but the touch-point list in §1.1 omits
   `ops/attention.rs`.

2. **`alloc_tokens` callers — 2 of 4 actually USE the returned `Vec`,
   not discard it.** Plan doc §1.1 says "Keep `Result<Vec<u32>>`
   signature even though all 4 callers discard the Vec — return
   `Vec::new()` for now". Verified with `grep .alloc_tokens` across
   `infer/src/scheduler/`:
   - `core.rs:287` discards (decode tail extension)
   - `decode.rs:119` discards (decode tail extension)
   - **`prefill.rs:184` consumes** — passes `new_indices` into
     `state.migrate_kv_range_to_paged(ctx, &self.paged_kv_pool, 0, &new_indices)`
   - **`prefill.rs:270` consumes** — passes `new_indices` into the
     same migration call after slot-prefill alloc

   Returning `Vec::new()` would silently break prefix migration on
   every prefill. The plan-doc-recommended shortcut is wrong.

## Fix
Not implemented in this commit. **P0 is rescoped** so the next
attempt has accurate prerequisites:

### Required additions to P0 scope (was: ~7 files, now: ~12 files)
1. **New CUDA kernel** `kv_cache_to_paged_range_hnd_kernel` in
   `crates/cuda-kernels/csrc/kv/kv_cache_to_paged.cu` (additive, ~60 lines):
   takes `(start_pos, count, slot_page_table, page_size, stride_page,
   …)`, writes HND pages. Does **not** touch the existing range
   kernel or either of the forbidden quant kernels
   (`kv_cache_to_paged_int8_kernel`, `quantize_paged_kv_fp8_kernel`).
2. **FFI declaration** for the new kernel in
   `infer/src/backend/cuda/ffi.rs`, plus dispatch in
   `infer/src/model/kv_cache.rs::migrate_from_contiguous_range_bf16`
   based on `pool.page_size`.
3. **`migrate_kv_range_to_paged` API change** — the function takes
   `&new_token_indices` today (just the newly-allocated tail). The
   new HND kernel needs the slot's full page table. Either pass
   `slot: usize` and look up `pool.token_indices[slot]` inside, or
   pass the full table from the caller. Touches the two call sites
   in `prefill.rs` and the function definition.
4. **`stride_page` fix** at `ops/attention.rs:673,1068`:
   `kv_dim` → `kv_dim * page_size`. Two sites.
5. **`alloc_tokens` two-level allocator** must still return the real
   newly-allocated indices (NOT `Vec::new()`), now in **page units**
   convertible to per-token indices for the migration callers.

### Touch points the plan doc DID get right (no change)
- `paged_kv.rs` struct field, `build_indptr` page-cumulative,
  `build_last_page_lens` formula, empty-slot filter, DEFAULT_PAGE_SIZE.
- 3 model `batch_decode.rs` literal-`1` removals.
- `scheduler/cuda/decode.rs:193` literal-`1` removal.
- `--page-size` CLI flag in `infer/src/main.rs` + `ServerRuntimeConfig`
  plumbing in `bootstrap.rs`.

### Verification gate (P0 acceptance)
- `cargo test -p infer --lib` stays at `272/272 passed`.
- New unit test for the HND range kernel: round-trip a synthetic
  contiguous tensor through the kernel at `page_size ∈ {1, 2, 4, 8, 16}`
  and assert the HND addressing formula reads back the same values
  that go in. Required because **e2e tests have weight drift** (see
  `project_remote_cuda_box.md`) and cannot serve as the regression
  gate for a kernel rewrite.
- `bench_serving request --prompt-len 512 --output-len 128` Qwen3-4B:
  TPOT ≤ 32.75 ms (current baseline, see
  `project_l4_perf_baseline.md` in memory) ± 5%.

## Rule
**Audit the function the production code actually calls, not the
function that looks like the audit target.** The plan doc's §1.2
"Kernel readiness" check looked at `kv_cache_to_paged_kernel` and
declared the BF16 migration path ready. The production prefill path
calls `kv_cache_to_paged_range_kernel`, which lives in the same file
12 lines below and has different semantics. A `grep migrate_` from the
caller side (`prefill.rs`, `kv_cache.rs`) would have caught this in
two minutes.

**Never trust the "all 4 callers discard the Vec" shortcut without
running the grep.** Treat any plan-doc claim about caller behaviour
as an unverified assertion; verify with `grep \.fn_name(` from the
call-site direction before relying on the claim. Plan docs decay
fast; production code is authoritative.

**Silent correctness bugs in CUDA kernels need explicit kernel-level
unit tests, not e2e parity gates.** When the e2e parity gate is
already red for unrelated reasons (weight drift, here), it cannot
catch a new kernel correctness regression. Round-trip kernel tests
at `page_size ∈ {1, 2, 4, 8, 16}` are the minimum bar for any change
that touches paged KV addressing.
