# DSv4 batched DSA reads prepass buffers directly

## Context

The batched DSv4 CSA path already computes `q_i_batch` and `weights_batch` in a
batched indexer-query prepass. The later batched DSA read still allocated another
pair of staging buffers and copied every row into them before calling
`csa_select_official_batched`.

## What Worked

CompressedSparse now feeds the batched DSA read directly from the prepass buffers.
Only lanes without that prepass, such as SparseIndexed, allocate row-staging and
copy into it. This deletes the redundant row-gather work from the common DSv4
batched path while preserving the fallback path.

## Verification

Local:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
```

Result: passed. Existing `cli/src/eli.rs` unused import warning is unrelated.

Pod (`sglang-test`, TP=4, commit `0597a3ba`):

- build passed in `target-nccl-dsv4`,
- service started and `/v1/stats` was healthy,
- 4 concurrent 96-token synthetic completions completed; later 4 concurrent
  64-token completion smoke completed with `generated_tokens=256`, no runtime errors.

## Results

No throughput claim yet. The HTTP smoke confirms the dataflow is live; a clean
phase A/B was not captured for this exact change. Treat it as a safe prepare-path
cleanup until a later profile quantifies the delta.

## Rule

If a batched prepass already owns the exact tensor a later batched kernel reads,
do not copy it row-by-row into another staging buffer. Keep staging only for lanes
that genuinely lack the prepass.
