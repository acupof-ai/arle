# DSv4 batched prepare cleanup — skip dummy selected alloc and full-flatten row copies

## Context

After the compact FP8 MoE decode lane, TP4 phase logs still showed `sw_attn` around
20-22ms at n=2, with `compidx/perrow` several milliseconds. The batched prepare path
still carried two pieces of row-local bookkeeping that were no longer needed in the
full-flatten path:

- `csa_select` allocated a zero-length `CudaSlice<i32>` only to mean "no selected";
- full-flatten copied `normed[r]` and `c_q_normed[r]` into row scratch even though
  compressor/indexer projection and state update had already run in batched prepasses.

## What Worked

- `csa_select` now returns `Option<CudaSlice<i32>>`; the batched DSA gather path
  returns `None` without a device allocation.
- The full-flatten prepare loop skips the `normed_row` and `c_q_normed_row` D2D
  copies. SparseIndexed / non-full-flatten keeps the original copies.

No CUDA kernel changes.

## Verification

Local:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
```

Result: passed. Existing `cli/src/eli.rs` unused import warning is unrelated.

Pod (`sglang-test`, TP=4, commit `1c2caa19`):

- build passed in `target-nccl-dsv4`,
- service started and `/v1/stats` was healthy,
- 4 concurrent 64-token completion smoke completed: `generated_tokens=256`,
  `requests_completed=4`, no CUDA/runtime errors.

Representative phase lines after:

```text
[decode-phase] n=2 sw_attn=20.2ms (prep=10.3 [proj=3.0 compidx=4.2 compidx_split=[perrow=3.3 read=0.9]] fwd=2.2 finish=6.3) moe=20.7ms
[decode-phase] n=2 sw_attn=20.4ms (prep=10.6 [proj=3.0 compidx=4.3 compidx_split=[perrow=3.4 read=0.9]] fwd=2.2 finish=6.3) moe=20.6ms
```

Compared to the prior post-MoE profile (`perrow≈4.0-4.2ms` in common n=2 lines),
this is a small but real prepare-path cleanup.

## Rule

Once a batched prepass owns the dataflow, delete stale per-row copies instead of
carrying them as shape-compatible ghosts. Device allocation must not be used as an
`Option::None` marker.
