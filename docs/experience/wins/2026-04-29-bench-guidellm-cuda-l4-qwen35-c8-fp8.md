# cuda-l4 Qwen3.5-4B c=8 fixed 120s, FP8 KV — first canonical bench

> First successful canonical bench for Qwen3.5-4B (hybrid full/linear
> attention). Headline: ITL p50 **44.21 ms**, tok/s **121.48** at c=8
> conservative config. ITL is 38% faster than Qwen3-4B at the same
> shape — Qwen3.5's hybrid architecture has only **8 KV layers**
> (vs Qwen3's 36) so per-token KV-read bandwidth is ~4.5× lower.
>
> Path is **TileLang HD256** for both prefill and full-attention
> decode under default `--features cuda` (which implies
> `tilelang-attn`). The K2 problem is WORSE than Qwen3: Qwen3.5
> has **no `supports_mixed_batch` impl at all** (uses trait default
> `false` at `infer/src/model.rs:569`), so it always uses `Split`
> plan regardless of KV format.

## Goal

Establish a working canonical bench number for Qwen3.5-4B at the
post-fix-stack baseline so future iteration has a reference.
Goal type: **publication baseline**.

## Hypothesis

- ITL should be lower than Qwen3 (8 vs 36 KV layers → 4.5× less
  attention KV read per token).
- TTFT will likely be similar or worse (HD256 prefill is heavier
  than HD128 per token, mitigated by fewer layers).
- Same K2 stall pattern in Split plan; possibly worse since the
  HD256 FlashInfer workspace is bigger (640 MB hardcoded —
  triggered the OOM under c=16 at default fraction).

## Setup

- Hardware: NVIDIA L4 sm_89, 22 GB, CUDA 12.8
- Commit: `23cb9abe`
- Feature set: `cargo build --release --features cuda` (implies
  `tilelang-attn`)
- Server: `target/release/infer --model-path Qwen/Qwen3.5-4B
  --num-slots 8 --max-seq-len 4608 --kv-cache-dtype fp8
  --mem-fraction-static 0.70 --chunked-prefill-size 512`
- Bench: `bash scripts/bench_guidellm.sh qwen35-c8-fp8 --model
  Qwen3.5-4B --processor <hf-cache-snapshot>
  --concurrencies 8 --max-seconds 120`
- Pool: 354,624 tokens / 7.3 GB at fraction=0.70 (only 8 KV layers
  → ~4.5× more tokens than Qwen3 at same GB)

## Results

| Metric          | Value     |
|-----------------|----------:|
| TTFT p50 (ms)   |   6386.1  |
| TTFT p99 (ms)   |   6615.8  |
| ITL p50 (ms)    |    44.21  |
| ITL p99 (ms)    |    46.57  |
| out tok/s       | **121.48**|
| req/s actual    |    0.4    |

### Cross-comparison vs Qwen3-4B

| Metric | Qwen3-4B (c=16, slots=16, chunk=512) | Qwen3.5-4B (c=8, slots=8, chunk=512) |
|---|---:|---:|
| out tok/s | 145.30 | 121.48 |
| ITL p50 (ms) | 71.58 | **44.21** |
| TTFT p50 (ms) | 11884 | 6386 |
| KV layers | 36 | **8** |
| KV pool tokens | 148,256 | 354,624 |
| Concurrency | 16 | 8 |

Per-slot tok/s: Qwen3 = 9.08, Qwen3.5 = 15.18 (1.67× higher per
slot). The 8-layer KV cuts per-token decode work; per-slot
throughput is what matters for sequential workloads.

## Path & K2 status

Both prefill and full-attention decode use **TileLang HD256** AOT
cubins under default features:

- Prefill: `prefill_attention_paged_run_hd256` at
  `infer/src/model/qwen35/prefill.rs:595`. Under `tilelang-attn`,
  routes to `tilelang_batch_prefill_paged_hd256_q*_kv*_run_cuda`
  cubins; under no feature, falls back to FlashInfer's
  `BatchPrefillPagedPlan::new_hd256` (640 MB workspace).
- Decode: `infer/src/model/qwen35/batch_decode.rs:347`, similarly
  TileLang-gated for HD256 paged decode.
- Linear-attention layers: recurrent state, no KV pool involvement.

**Mixed-batch status (worse than Qwen3)**: Qwen3.5 has no
`supports_mixed_batch` override — falls through to the trait
default (`infer/src/model.rs:569`) which returns `false`
unconditionally. There's no `forward_mixed_batch` impl either.
**All KV formats always use `Split` plan**. Qwen3 at least has
the BF16-only Mixed path; Qwen3.5 has nothing.

To enable Mixed for Qwen3.5:
1. Need an HD**256** varlen attention kernel (the `4e4906f5`
   `decode_attention_varlen_fp8` kernel is HD128-only).
2. Need to coordinate Mixed across the hybrid architecture:
   full-attention layers go through varlen attention, linear-
   attention layers update recurrent state in the same packed
   step.
3. Implement `qwen35::supports_mixed_batch` + `forward_mixed_batch`.

Estimated effort: 2-3× the Qwen3 K2 wire-up because of (1) and
(2). Filed as task #29 follow-up.

## Problems

1. **OOM cliffs are unavoidable at typical config**. At c=16 with
   default `--num-slots auto` (picks 105) + default fraction 0.85,
   bench gets `Alloc chunk_state failed: OUT_OF_MEMORY` —
   recurrent-state buffers per slot ate the rest of GPU. Even at
   slots=16 + fraction=0.85 + chunk=512, FlashInfer
   `float_workspace alloc` fails. Had to drop to slots=8 +
   fraction=0.70 + chunk=512.

2. **Server enters a permanently-broken state after OOM**. After
   the failed c=16 bench, single curl `/v1/completions` requests
   returned `"text":"","completion_tokens":0` for every prompt —
   slot leak / prefill-failure path doesn't reset cleanly.
   Confirmed K7 in `docs/projects/2026-04-29-perf-bug-roundup.md`
   is real and blocking. Server restart recovers.

3. **bench wrapper K6 detector caught the silent OOM correctly** —
   the JSON had `successful=3494, errored=0, iter mean=1.0,
   out_tokens mean=256.0, ttft_p50=0.0, itl_p50=0.0` — bench
   refused to print the headline, surfaced the error.

## Cross-references

- Throughput gap analysis: `docs/projects/2026-04-29-throughput-gap-analysis.md`
- Pipeline map: `docs/projects/2026-04-29-scheduler-pipeline-map.md`
- Bug roundup K7 (slot leak): `docs/projects/2026-04-29-perf-bug-roundup.md`
- Qwen3 c=16 baseline: `docs/experience/wins/2026-04-29-bench-guidellm-c16fixed-fp8.md`
- KV-quant matrix: `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-kv-quant-matrix.md`

## Next steps

1. **Fix K7 slot leak**: server should free stuck slots when prefill
   OOMs; right now a single OOM permanently degrades to empty
   responses. Top-priority server-side robustness fix.
2. **Tune Qwen3.5 default config**: 22 GB L4 + 8-layer KV + 640 MB
   FlashInfer workspace + per-slot recurrent state means default
   fraction needs to be lower than 0.85 for safety. Should be
   model-aware: `auto_num_slots` should subtract the FlashInfer
   workspace + recurrent-state estimates.
3. **HD256 varlen kernel + Qwen3.5 Mixed-batch wire-up** (task #29
   extension) — would close the K2 stall on the 8 full-attention
   layers, biggest tok/s lever.
