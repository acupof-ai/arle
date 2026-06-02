# DSv4 FlashMLA Fixed Decode Topk

## Context

The target lane is DSv4-Flash, TP8, EAGLE, CUDA graph, 256K/1500, hot GPU
cache, with a reference target around TTFT 0.44s, TPOT 4.85ms, E2E 7.7s, and
196 output tok/s.

The TP8 FlashMLA decode path was reachable, but HCA decode still derived
`max_compressed_keys` from the current token position. That makes
`topk_unified` and FlashMLA sched metadata change during decode. A CUDA graph
cache keyed only by batch size cannot safely replay that.

## What Worked

Bind HCA FlashMLA decode shape to the fixed compressed FP8 KV pool capacity:

- HCA batch FlashMLA decode now uses `comp_blocks * page_block_size` as the
  fixed `max_compressed_keys`;
- single-token HCA FlashMLA decode uses the same fixed pool-capacity shape;
- FP8 KV pool sizing now pads compressed capacity to FlashMLA's 128-key topk
  invariant, so fixed topk never exceeds pool capacity;
- the indices builder still masks causally unavailable compressed rows from
  `start_pos`, so the visible attention set is unchanged.

Local checks passed:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

Remote build and decode correctness passed on the DSv4 pod at commit
`ab9c40397caa3f9ed022046e44d2859d4347632b`:

- build artifact: `/tmp/dsv4_fixed_topk_20260603_build/build.log`;
- build time: `release-fast` finished in 18.39s and used the prebuilt CUDA fast
  path;
- validation artifact: `/tmp/dsv4_fixed_topk_reach_20260603`;
- `scripts/dsv4_batched_decode_validate.py 18085` exited 0, printed
  `ANSWER_PASS`, and completed c8 with zero HTTP errors;
- operator trace proved both FlashMLA decode paths still executed:
  `attn_flashmla_decode` 16408 calls and
  `attn_hca_batch_flashmla_decode` 2240 calls;
- after cleanup, `nvidia-smi --query-compute-apps` reported no remaining
  compute apps.

This is a graph-shape prerequisite, not a target performance result. The
validation used operator trace and `--disable-cuda-graph`.

## Rule

CUDA graph keys must cover every shape-changing parameter. If the cache key is
batch size, DSv4 decode metadata such as FlashMLA topk must either be fixed to
capacity or moved into graph-external device-side data updates.
