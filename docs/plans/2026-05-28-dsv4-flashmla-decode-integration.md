---
title: DSv4 FlashMLA decode integration — make FlashMLA the default decode path
date: 2026-05-28
type: implementation plan
status: ready for codex pickup（next session）
owner: ckl
related:
  - docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md (A2 axis closure)
  - docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md (prefill done)
  - https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/api/sparse_decode.h
---

# DSv4 FlashMLA decode integration

## Why

V2.4 closed the prefill path (FlashMLA on, 12.4% win at 16K). **Decode
remains on the legacy `dsv4_hybrid_attention_cuda` kernel** at ~26 ms/token
on H20 / TP=8. That's 38 tok/s — close to SLO TPOT 30 ms but not below.

Upstream FlashMLA ships a separate **sparse decode kernel**
(split-KV MLA, `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`)
that targets the same DSv4 hybrid attention but for `token_count=1`. The
algorithm differs structurally from prefill: it splits KV across SMs
(SplitKV) and combines partial results via a separate combine kernel.

Switching ARLE decode to FlashMLA should give:
- ~5-10 ms/token (SGLang's published number on similar HW),
- 33–38 ms/token → 5-10 ms/token = **2-5× TPOT improvement**,
- Unlocks SLO TPOT ≤ 18 ms target.

## Scope

`ARLE_DSV4_FLASHMLA_DECODE` (default on after validation), single-binary
change in `infer/src/model/deepseek/weights.rs::finish_attention_gpu` for
the `token_count == 1` branch.

## Implementation phases

### Phase D-1 — Vendor decode kernels

Files missing from `crates/cuda-kernels/vendor/flashmla/`:

```
csrc/sm90/decode/sparse_fp8/
  splitkv_mla.h                 # entry: run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>
  splitkv_mla.cu                # kernel impl
  ...
csrc/smxx/decode/get_decoding_sched_meta/
  get_decoding_sched_meta.h     # CPU-side block scheduler
  get_decoding_sched_meta.cu
csrc/smxx/decode/combine/
  combine.h                     # combine partial split-KV results
  combine.cu
csrc/sm90/decode/instantiations/  # NUM_HEADS × MODEL_TYPE template instantiations
```

**Action:** vendor matching upstream commit `df022eb`, mirror dir layout
under `crates/cuda-kernels/vendor/flashmla/csrc/`. Update
`crates/cuda-kernels/build.rs` to walk these new sub-trees with nvcc
(same `gencode arch=compute_90a,code=sm_90a` flags as prefill).

Expected new .cu files: ~10-15. Expect ~3-5 minute incremental rebuild
delta on first pod cycle.

### Phase D-2 — Shim function `arle_flashmla_sm90_sparse_decode_fwd`

New extern "C" entry in `crates/cuda-kernels/csrc/misc/arle_flashmla_shim.cu`
(or new file `arle_flashmla_decode_shim.cu`):

```cpp
extern "C" cudaError_t arle_flashmla_sm90_sparse_decode_fwd(
    const bf16* q,            // [s_q=1, h_q, d_qk]
    const bf16* kv_blocks,    // [num_blocks, page_block_size, d_qk]
    const int* indices,       // [s_q=1, topk]
    const int* topk_length,   // [s_q=1]
    const float* attn_sink,   // [h_q]
    bf16* out,                // [s_q=1, h_q, d_v]
    float* lse,               // [s_q=1, h_q]
    // SplitKV scratch (caller allocates)
    float* lse_accum,         // [num_splits, s_q=1, h_q]
    float* o_accum,           // [num_splits, s_q=1, h_q, d_v]
    DecodingSchedMeta* sched_meta,  // [num_sm_parts]
    int* num_splits_ptr,      // [batch_size+1] = [2] for our b=1
    int num_sm_parts,
    // shape + strides + topology
    int s_q, int h_q, int h_kv, int d_qk, int d_v,
    int num_blocks, int page_block_size, int topk,
    int model_type,           // 1 = MODEL1
    float sm_scale,
    int stride_q_h_q,
    int stride_kv_block, int stride_kv_row,
    int stride_indices_s_q,
    int stride_o_h_q,
    int stride_lse_h_q,
    int stride_lse_accum_split, int stride_lse_accum_s_q,
    int stride_o_accum_split, int stride_o_accum_s_q, int stride_o_accum_h_q,
    cudaStream_t stream
);
```

Sets up `SparseAttnDecodeParams`, runs `Decode_Sm90_Impl::run()`. Same
exception-safety wrap (catch `std::exception&` + `...`) per V1 KU_ASSERT
escape lesson.

A second shim `arle_flashmla_sm90_decode_combine` wraps the split-KV
combine kernel.

### Phase D-3 — KV cache layout adapter

**Critical:** FlashMLA decode expects KV as block-paged
`[num_blocks, page_block_size=64, d_qk]`. ARLE DSv4 KV cache is a
contiguous ring per layer at `[max_seq_len, h_kv=1, d_qk]` with explicit
SW window + compressed pool.

Two options:

**D-3a (recommended)**: Build a thin block-table on-the-fly. ARLE's
contiguous KV addressed by token_id → FlashMLA expects (block_idx,
in_block_row). With page_block_size=64 and SW window=128 + compressed
N: emit indices in FlashMLA's block coords by computing
`block_idx = token_id / 64, row = token_id % 64` and tagging the SW
slots vs compressed slots in the same indices buffer.

**D-3b (alternative)**: Allocate a FlashMLA-shaped block-paged KV buffer
per layer per request and copy from ARLE's KV ring at decode-time. Simpler
but ~256 MB extra memory per request and a memcpy per step.

D-3a is what SGLang does — strongly preferred. Requires careful
verification that ARLE's KV pool block alignment is compatible
(page_block_size=64 vs ARLE's prior allocations).

### Phase D-4 — Decode dispatch wire-in

In `finish_attention_gpu`:

```rust
let use_flashmla_decode = sm_major == 9
    && (mode_int == 1 || mode_int == 2)
    && token_count == 1                                     // decode only
    && (head_dim == 512 || head_dim == 576)
    && dsv4_flashmla_decode_enabled()?;
if use_flashmla_decode {
    // Build per-token indices = [SW slots(128) | compressed selections(index_topk for CSA,
    //                            full compressed range for HCA)] in BLOCK-PAGED coordinates
    // Get DecodingSchedMeta via Decode_Sm90_Impl::get_meta() then upload to GPU
    // Alloc lse_accum / o_accum scratch [num_splits, h_q, ...]
    // Call decode shim + combine shim
} else {
    // legacy dsv4_hybrid_attention_cuda
}
```

The SchedMeta upload is a per-layer fixed-size CPU→GPU upload (~80 bytes).
Could amortize via a single per-step CPU→GPU upload that all 43 layers
share.

### Phase D-5 — Test + perf

Smoke: 4K prompt + 32 decode tokens, FlashMLA decode env on, compare
output byte-equality vs legacy. (Some FP-precision divergence expected;
allclose with abs_tol=8e-4 per upstream test convention.)

Bench:
- `bash dsv4_long_probe.sh 4 64 1` (large decode count to amortize startup) — measure decode TPOT
- Same at 16K and 24K
- Compare to V2.4-decode-legacy baseline (~26 ms/token)
- Target: ≤ 10 ms/token

### Phase D-6 — Default-on flip

After byte-equality (or paired t-test of accuracy on MMLU/GSM8K subset)
clears, flip `dsv4_flashmla_decode_enabled()` default to true.

## License-or-kill

- **PASS**: decode TPOT ≤ 12 ms/token at TP=8 H20 (≥ 2× over legacy 26 ms);
  greedy output stays within ARLE's existing FP-precision tolerance;
  no degradation in throughput at the c=8 qps=8 SLO shape.
- **KILL**: TPOT regresses, or output divergence breaks downstream model
  quality benchmarks. Roll back via the env knob (default off restoration).

## Estimated cost

- Phase D-1 (vendor): 1-2 hours (mostly mechanical mirror + build.rs).
- Phase D-2 (shim): 2-3 hours including exception-safety + alloc bookkeeping.
- Phase D-3 (KV adapter): 4-6 hours — this is the hardest part; needs
  careful audit of ARLE's KV pool indexing against FlashMLA's expectations.
- Phase D-4 (dispatch): 1-2 hours.
- Phase D-5 (test + perf): 2-4 hours per iteration.

**Total: 1-2 focused sessions** (8-15 hours wall-clock).

## What this unblocks

| Metric | Now (V2.4) | After decode (target) | SLO |
|---|---:|---:|---:|
| Prefill TTFT (16K) | 103.13s | 103.13s (unchanged) | 4.8s |
| Decode TPOT | 26 ms/tok | 5-10 ms/tok | 18 ms |
| End-to-end @ 32K in / 1.5K out | ~205s prefill + 39s decode = ~244s | ~205s + 15s = ~220s | ≤ 200s |

After decode lands, the binding constraint shifts back to prefill, where
A4 multi-stream overlap (TokenWeave-style per-tile compute-comm overlap)
becomes the right next axis.

## Refs

- [FlashMLA upstream sparse_decode.h](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/api/sparse_decode.h)
- [FlashMLA upstream sm90/decode/sparse_fp8/](https://github.com/deepseek-ai/FlashMLA/tree/main/csrc/sm90/decode/sparse_fp8)
- [SGLang DSv4 day-0 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) — uses the same kernel for production decode
- V2.4 wins entry: [`docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`](../experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md)
