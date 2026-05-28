---
title: DSv4 FlashMLA decode integration — make FlashMLA the default decode path
date: 2026-05-28
type: implementation plan
status: D-1 + D-2 + FFI landed; D-3 un-killed → D-3' FP8 pack + pool active
owner: ckl
related:
  - docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md (A2 axis closure)
  - docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md (prefill done)
  - docs/experience/wins/2026-05-28-dsv4-flashmla-decode-integration.md (D-3 kill — superseded by D-3')
  - https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/api/sparse_decode.h
---

# DSv4 FlashMLA decode integration

> **2026-05-28 update — D-3 UN-KILLED → D-3'.** The original D-3a/D-3b
> sketch assumed upstream decodes consume bf16 block-paged KV. Upstream
> source contradicts: the kernel reinterprets `params.kv` as `fp8*` and
> asserts `stride_kv_row == BYTES_PER_TOKEN` (MODEL1 = 584, V32 = 656)
> with a model-specific FP8-NoPE + bf16-RoPE + fp8_e8m0-scales layout.
>
> **Decision (per user directive 2026-05-28): build the FP8 KV pool +
> packing kernel.** SGLang already does this; ARLE's bf16 pool is not a
> blocker, it is a missing layer. See **Phase D-3' below** for the
> full SOLID contract + implementation plan. Vendor (D-1) + shim (D-2)
> + FFI landed (`5d18b624`, `3f7923a3`); D-3' + D-4..D-6 active.

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

### Phase D-3 — KV cache layout adapter [SUPERSEDED by D-3']

(Original sketch retained for reference: assumed bf16 block-paged KV.
Superseded — see D-3' below for the correct FP8 contract.)

### Phase D-3' — FP8 KV pool + packing kernel (SOLID contract, active path)

**Upstream contract (MODEL1 = DSv4-Flash), evidence-anchored:**

| Source                                         | Evidence                                                                                                                        |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `splitkv_mla.cuh:694`                          | `constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8 = 448 + 128 + 8 = 584`                                      |
| `splitkv_mla.cuh:695`                          | `KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN)` (per-token byte stride pin)                                                |
| `splitkv_mla.cuh:558`                          | `gK_base = k_ptr + block_idx*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16))`                    |
| `splitkv_mla.cuh:555`                          | Scales offset = `page_block_size*(NoPE+RoPE_bytes) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0)`                              |
| `config.h: NUM_SCALES = 8 / QUANT_TILE_SIZE=64`| 8 fp8_e8m0 scales per token (7 used for the 7×64 NoPE tiles + 1 pad)                                                            |
| `api/sparse_decode.h:295`                      | preflight: `bytes_per_token = 448 + 64*2 + (448/64)*1 + 1 = 584`                                                                |

**MODEL1 per-block layout (page_block_size=64, 37376 B/block/layer):**

```
offset 0       : [T0 NoPE 448 B][T0 RoPE 128 B]  (576 B per token AoS)
offset 576     : [T1 NoPE 448 B][T1 RoPE 128 B]
...
offset 36288   : [T63 NoPE 448 B][T63 RoPE 128 B]
offset 36864   : [T0 scales 8 B][T1 scales 8 B]...[T63 scales 8 B]   ← scales region appended
total          : 64 × 576 + 64 × 8 = 37376 B = 64 × 584
```

NoPE = fp8_e4m3 (HEAD_DIM_NOPE=448 elements). Quantized per **tile of 64**
NoPE dims (so 7 tiles per token) with one **fp8_e8m0** scale per tile.
RoPE = bf16 verbatim (HEAD_DIM_ROPE=64 elements → 128 bytes).

Dequant inside the kernel (`dequant.h`): pairs of fp8_e8m0 scales are
converted to `bf16x2` via `__nv_cvt_e8m0x2_to_bf162raw`, then applied
per fp8x8 chunk via `cvt_fp8x8_bf16x8`.

**Encoder side — what we need to build:**

```
arle_dsv4_fp8_kv_pack<MODEL1>(
    const bf16* nope,         // [n_tokens, 448]
    const bf16* rope,          // [n_tokens, 64]
    uint8_t* packed_kv,        // block-paged FP8 layout, MODEL1 = 584 B/token
    const int* block_table,    // [n_tokens] → block_id
    const int* in_block_row,   // [n_tokens] → 0..page_block_size-1
    int page_block_size,       // 64 for DSv4
    int n_tokens,
    cudaStream_t stream
)
```

Per-token per-tile (8 tiles × 64 dims = 448 NoPE elements per token... wait,
7 tiles × 64 = 448, the 8th scale slot is just padding):
1. amax = warp-reduce max(|nope[i]|) over the 64 NoPE dims of the tile.
2. scale_e8m0 = derive E8M0 exponent s.t. amax/2^scale_e8m0 ≤ FP8_E4M3_MAX (448).
3. quant: x_fp8 = __nv_cvt_float_to_fp8x4(x_bf16 * 2^(-scale_e8m0)).
4. Write 64 fp8 NoPE bytes to `block_id*block_stride + row*576`.
5. Write 128 bf16 RoPE bytes to `block_id*block_stride + row*576 + 448`.
6. Write 8 scale bytes (7 tile scales + 0 pad) to
   `block_id*block_stride + page_block_size*576 + row*8`.

**Pool design — written-through at compressor update:**

| Path                | Allocator                                                                                                              | Trigger                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| Prefill (V2.4 FlashMLA on) | Output still goes through ARLE's existing bf16 KV pool — prefill uses different kernel; **no FP8 pool dependency** | n/a                                              |
| Compressor update (DSv4 hybrid layers, on every decode step) | After the bf16 NoPE+RoPE compressor key/value lands, run `arle_dsv4_fp8_kv_pack` on the **same** stream      | Existing compressor-update hook in `weights.rs` |
| Decode (FlashMLA path)     | Reads the FP8 pool directly                                                                                  | `token_count == 1` branch                        |

Pool size per rank: `num_layers × max_blocks × page_block_size × 584`.
For DSv4-Flash @ TP=8, num-slots=4, max_seq_len=32768:
- max_blocks per rank ≈ ceil(32768 / 64) × num-slots = 2048 (worst-case across all slots).
- 43 layers × 2048 × 64 × 584 = **3.2 GB / rank**. H20 96 GB — comfortable.

Allocation strategy: per-request, allocated at session start, freed at
session end. Same lifetime as ARLE's existing bf16 KV pool.

**V32 follow-up:** scales become 4 × float32 (NUM_SCALES=4, scales are
fp32 inside the kernel — see splitkv_mla.cuh:545-549), BYTES_PER_TOKEN
= 656, layout AoS. Same encoder skeleton, different scale dtype + count.
Defer V32 implementation until MODEL1 lands.

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

- Phase D-1 (vendor): **DONE** (`5d18b624`).
- Phase D-2 (shim + FFI): **DONE** (`3f7923a3`).
- Phase D-3' (FP8 pack kernel + pool): 6–10 hours — central work.
- Phase D-4 (dispatch + block-paged indices): 2–3 hours.
- Phase D-5 (parity test vs bf16 reference): 2–4 hours.
- Phase D-6 (default-on flip after bench PASS): 30 min + bench.

**Total remaining: ~12–18 hours wall-clock.**

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
