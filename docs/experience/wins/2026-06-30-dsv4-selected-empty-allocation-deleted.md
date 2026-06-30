# DSv4 batched DSA selected empty allocation deleted

## Context

In the DSv4 batched decode prepare path, `csa_select` returned a zero-length
`CudaSlice<i32>` for rows whose selected top-k is produced later by the single
batched DSA select. The caller immediately mapped that dummy buffer back to
`None`. That was a per-row device allocation used only as a sentinel.

## What Worked

`csa_select` now returns `Result<Option<CudaSlice<i32>>>` directly:

- batched DSA gather path returns `Ok(None)` with no device allocation,
- non-batched single-row/prefill path returns `Ok(Some(selected))`,
- `Dsv4MlaPrepared.selected` keeps the same `Option<CudaSlice<i32>>` shape.

This is a deletion-style cleanup; no CUDA kernel or math path changes.

## Verification

Local:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
```

Result: passed. Existing `cli/src/eli.rs` unused import warning is unrelated.

Pod (`sglang-test`, TP=4, commit `5d0e159e`):

- build passed in `target-nccl-dsv4`,
- service started and `/v1/stats` was healthy,
- 4 concurrent 64-token completion smoke completed: `generated_tokens=256`,
  `requests_completed=4`, no CUDA/runtime errors in the log.

## Results

No throughput claim. The change removes sentinel allocation churn from the
batched DSA prepare path; the expected impact is small and should be read as
risk/cost cleanup, not a headline perf lever.

## Rule

Do not allocate device memory to encode `None`. Keep sentinel state in the Rust
return type when the downstream path already consumes `Option`.
