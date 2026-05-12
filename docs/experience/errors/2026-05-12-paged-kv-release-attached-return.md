# Paged KV release returned attached pages

## Context

During the Qwen3.5 / DSV4 operator stability pass, full
`cuda-kernels` release tests exposed a stable failure:

```text
paged_kv::tests::retain_release_without_free_slot_does_not_move_pages
release must not free a page that is still attached to a live slot
```

The failing path models a scheduler shadow-observer sequence:

1. A live slot owns a page.
2. The scheduler temporarily retains that page for prefix/radix bookkeeping.
3. The retain is released before `free_slot`.

The page must stay attached to the live slot and must not be reported as
reclaimed.

## Root Cause

`recycle_page_if_unreferenced()` already had the correct allocator invariant:
only pages with both `page_attach_count == 0` and `page_ref_count == 0` are
pushed back to `free_pages`.

The bug was the return value of `release_pages()` and the test mock's
equivalent `release()`: both pushed a page into the returned `newly_freed` list
whenever `page_ref_count` dropped to zero, even if the page was still attached
to a live slot and therefore was not actually recycled.

So allocator state was safe, but caller-visible reclaimed-page accounting was
wrong. Callers that use `release_pages(...).len()` for metrics or reclaim
counts could over-report reclaimed pages.

## Fix

Make `recycle_page_if_unreferenced()` return `true` only when it actually
pushes the page to `free_pages`. `release_pages()` now includes a page in its
returned vector only when that helper returns `true`.

The mock implementation was updated the same way, preserving the test as a
contract for production semantics.

Verification:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda paged_kv::tests -- --nocapture

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda
```

Results: `paged_kv::tests` passed `18/18`; full `cuda-kernels` release tests
passed `38/38`.

## Rule

For pool lifecycle APIs, returned "freed/reclaimed" vectors must mean actual
allocator state transitions, not only refcount transitions. Refcount zero is
not enough when a page can still be attached to a live slot.

