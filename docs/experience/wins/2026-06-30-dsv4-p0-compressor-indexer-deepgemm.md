# DSv4 P0 — compressor/indexer batched projections use DeepGEMM

## Context

The TP=4 B=4 decode profile put projection GEMVs at the top of the kernel table.
Commit `c0e74741` had already quantized DSv4-Flash-FP8 compressor/indexer weights
at load time, moving them off BF16 `gemv_handwritten` and onto scalar FP8
`dsv4_fp8_gemv_batch_tiled_kernel`. The remaining P0 step was to reuse the proven
`wq_b` DeepGEMM projection lane for the batched decode pre-pass projections:
`compressor.wkv`, `compressor.wgate`, and `indexer.weights_proj` (with
`indexer.wq_b` sharing the same helper).

## What Worked

- Added FP8 DeepGEMM caches to `Dsv4Compressor` (`wkv_deepgemm`,
  `wgate_deepgemm`) and `Dsv4Indexer` (`weights_proj_deepgemm`).
- Collapsed loader cache construction into one canonical `decode_proj_cache` helper:
  build a cache only when the decode DeepGEMM allocation gate is on and the weight
  is raw DSv4 FP8 block-scaled; bf16/GLM dialects naturally stay scalar.
- Collapsed all compressor/indexer batch projection dispatch through one
  `proj_batched` helper: DeepGEMM when cache+scratch are present and `M>1`, else
  `dsv4_linear`.
- Threaded the shared prefill DeepGEMM scratch through the batched decode pre-pass
  with a single borrow covering the main compressor, CSA indexer compressor, and
  indexer query/weights projection.
- Fixed two unrelated build blockers in the in-flight KV-tier image path so the
  change can be typechecked and built: `copy_pages_from_host_on_copy_stream` now
  delegates to a stream-selecting shared body; DSv4 image serialization matches the
  current image structs; `slot_image_bytes` is computed before moving `kv_adapter`.

## Verification

Local typecheck:

```
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
```

Pod build: inside the modern-glibc `sglang-test` static pod container (CUDA 12.9,
g++-13, glibc 2.39), `cargo build --release --features cuda,nccl --bin arle`
completed successfully. DeepGEMM native bridge enabled; no DeepEP sidecar (allreduce
backend).

Runtime: H20 x4, TP=4, `DeepSeek-V4-Flash-FP8`, `ARLE_DSV4_MOE_BACKEND=allreduce`,
`ARLE_DSV4_EXPERT_BACKEND=deepgemm`, `INFER_DSV4_MAX_SEQ_LEN=16384`, warm
`DG_JIT_CACHE_DIR=/host/deepgemm-warm`. Serve log had **zero** DeepGEMM fallback
or preflight failures.

| Shape | Before (DeepGEMM-on baseline) | After P0 | Delta |
| --- | ---: | ---: | ---: |
| c=1, 256 tokens | 30.7 tok/s | 30.7 tok/s | neutral |
| c=1, 512 tokens | 31.3 tok/s | 31.6-31.7 tok/s | +1% |
| c=4, 256 tokens x4 | 53.8 tok/s aggregate | 60.1-60.2 tok/s aggregate | **+11.7%** |

P0 removes the remaining scalar FP8 projection path in the batched pre-pass. The
final controlled c=4 rerun was stable (60.2, 60.1 tok/s aggregate). With
`ARLE_DSV4_DECODE_PHASE_TIME=1`, the internal step logs were stable and NUMA was
correctly pinned:

```
[decode-phase] n=2 sw_attn≈21.5ms (prep≈11.6, proj≈3.0, compidx≈5.5,
  compidx_split=[perrow≈4.6 read≈0.9], fwd≈2.2, finish≈6.3) moe≈21.0ms
[numa-pin] gpu0..3 -> numa0, disjoint core ranges, 1/1 threads pinned
```

Next verdict should come from nsys / internal phase timing, not HTTP wall time.

## Rule

Do not generalize a projection DeepGEMM win from M=1. The useful boundary is the
batched pre-pass (`M=N`): cache+scratch present and `input.seq_len > 1`. Keep one
routing helper so new projection lanes either take the same tensor-core path or
fall back through `dsv4_linear` without parallel ad hoc branches.
