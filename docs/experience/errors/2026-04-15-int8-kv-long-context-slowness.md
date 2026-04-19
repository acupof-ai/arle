# 2026-04-15 · INT8 KV cache ~2× slower than BF16 at 25k context — root cause

## Context

The 2026-04-15 long-sequence bench
([`../wins/2026-04-15-bench-longseq.md`](../wins/2026-04-15-bench-longseq.md))
measured Qwen3-4B L4 decode ITL at 4k / 8k / 16k / 25k input tokens:

```
              bf16 ITL p50        int8 ITL p50        delta
 4 000  tok   35.2 ms              37.7 ms            +2.5 ms   (+7%)
 8 000  tok   37.4 ms              42.1 ms            +4.7 ms  (+13%)
16 000  tok   41.8 ms              50.7 ms            +8.9 ms  (+21%)
25 000  tok   33.1 ms†             61.9 ms           +28.8 ms  (+87%)
```

(`†` — bf16 25k ITL dip is a separate repeatability anomaly, flagged
in the long-seq note; it does not affect this root cause.)

Throughput at 25 k: **bf16 = 12.2 tok/s, int8 = 9.0 tok/s, a −26 %
regression**. int8 is supposed to save HBM and leave throughput at
worst unchanged; instead it becomes **slower at long context**, not
just memory-efficient.

## Root Cause

Two independent factors compound. Corroborated by an independent
codex investigation launched in parallel — we both landed on the
same kernel + pool structure after tracing from different entry
points.

**Correction to my initial hypothesis**: I originally suspected the
slowness came from `KVCache::prepare_layer` / `commit_layer`
(contiguous cache, `infer/src/model/kv_cache.rs:206,267`) which
**does** dequantise the full prior KV span to a bf16 working buffer
per layer per decode step. Codex traced the live path and confirmed
that **this is not the path the bench uses**. For `--num-slots 1
--max-seq-len 32768`, `forward_decode_batch`
(`infer/src/model/qwen3/forward.rs:366`) unconditionally routes
through the paged pool path, not the contiguous path, even at
`batch_size=1`. The paged path calls
`infer/src/model/qwen3/batch_decode.rs:574 kv_quant::decode_attention_int8`
which reads INT8 K/V + per-token f32 scales directly from the pool
and dequantises in registers — **no bf16 working buffer
materialisation of the prior KV span anywhere on the hot path**.

The bf16 work buffer that `TokenKVPool` holds
(`crates/cuda-kernels/src/paged_kv.rs:31`) is only used as a
staging area for the *single new token* before
`quantize_paged_kv_single` at `batch_decode.rs:494` writes it into
the INT8 pool. Not the full cache. This rules out the "full
materialise every step" failure mode.

FlashInfer's own int8 paged decode wrapper does not exist in-tree
(`crates/cuda-kernels/csrc/attention/flashinfer_decode.cu` is
bf16-only, `BatchDecodeParams<__nv_bfloat16, ...>`), so the int8
path has no choice but to use the custom
`decode_attention_int8_partial_kernel` in
`decode_attention_quantized.cu`.

With that ruled out, the actual regression comes from:

### 1. `INT8` paged pool is hardcoded at `page_size = 1`

M0.3 lifted `BF16` paged pool to `page_size = 16` and deliberately
kept `INT8 / FP8E4M3 / TurboQuant` at `page_size = 1`. This was
documented at the time as **intentional but known-broken follow-up**:

- [`docs/plans/tiered-kv-cache-tasks.md:220`](../../plans/tiered-kv-cache-tasks.md)
  flags `kv_cache_to_paged_int8_kernel` at
  `kv/kv_cache_to_paged.cu:64-103` as **hardcoding `page_size = 1`**
  and missing the `page_size` parameter entirely.
- `docs/plans/tiered-kv-cache-tasks.md:226-227` flags the chosen fix
  path:
  > Recommended: gate `page_size` per-format. … Document the
  > divergence prominently. Schedule **P1.5 to either rewrite the
  > INT8 kernels with `pos / page_size` decomposition or move INT8
  > to a separate HND format**.
- [`docs/projects/tiered-kv-cache.md:107`](../../projects/tiered-kv-cache.md)
  Granularity-mismatch row: "Quantized tiers still need follow-up."
- [`infer/src/model/kv_types.rs:21-26`](../../../crates/cuda-kernels/src/kv_types.rs)
  (`KVFormat::default_page_size`) hardcodes `BF16 => 16`,
  `FP8E4M3 | INT8 | TurboQuant { .. } => 1`.

That P1.5 rewrite never landed. INT8 pool is still token-granular
in the 2026-04-15 tree.

### 2. `decode_attention_int8_partial_kernel` is a scalar per-token loop, not a tile-based pipeline

The CUDA kernel at
[`crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`](../../../crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu)
was clearly intended to use shared-memory tiling and `cp.async`
pipelining — the header carries:

```c
#include <cuda_pipeline.h>
#define TILE_TOKENS 16   // "Tokens per shared memory tile (loaded via cp.async pipeline)"
```

but the actual kernel body at lines 42-217 **never references either**:

- `TILE_TOKENS` is defined but unused in the kernel file (checked:
  `rg TILE_TOKENS crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
  returns 2 matches, both in the comment + `#define`).
- No `cp.async` / `__pipeline_*` invocations anywhere in the file.
- No `__shared__ smem_k` / `smem_v` tile for K/V data. The only
  shared memory is the 3 × NUM_WARPS + NUM_WARPS*HEAD_DIM cross-warp
  merge scratch at line 160-162, which is per-block softmax state,
  not KV data.

Instead, the inner loop at line 115 processes **one token per warp
per iteration**, with four global loads per token (line 117-147):

```c
for (int t = warp_id; t < my_tokens; t += NUM_WARPS) {
    int global_t = my_start + t;
    int pool_idx = kv_indices[tok_start_global + global_t];      // scalar i32 global read
    int base = pool_idx * kv_dim + kv_head * HEAD_DIM;            // scatter offset
    int scale_off = pool_idx * num_kv_heads + kv_head;

    float k_scale = K_scales[scale_off];                          // scalar f32 global read
    int32_t k_packed = *reinterpret_cast<const int32_t*>(         // coalesced 4 i8 read per lane
        &K_data[base + d_base]);
    // QK compute + warp reduce

    float v_scale = V_scales[scale_off];                          // scalar f32 global read
    int32_t v_packed = *reinterpret_cast<const int32_t*>(         // coalesced 4 i8 read per lane
        &V_data[base + d_base]);
    // V accumulate
}
```

At 25 k tokens per decode step with 8 KV heads × 36 layers, each
warp chases **~6 250 scattered page lookups** per layer per KV head.
Each `pool_idx = kv_indices[...]` read is serialised against the
next token's work because there is no load pipelining. The 32-thread
warp does coalesce within each token's KV row (128 bytes per warp
per token), but across tokens the page indices point at arbitrary
HBM offsets, so L2 cache locality is poor and the reads serialise on
memory latency rather than bandwidth.

### How the two factors compound

**Comparison per layer per decode step at 25 k tokens:**

| path | bytes read | access pattern | kernel optimisation |
|---|---|---|---|
| `BF16` pool, FlashInfer paged decode | 25 000 × 256 B = 6.4 MB | 1 563 page reads at `page_size = 16`; each page reads 16 contiguous tokens' worth of KV in coalesced / vectorised form | cp.async pipeline, shared-memory tile, FlashAttention split-KV, vectorised `hlf2` |
| `INT8` pool, `decode_attention_int8_partial_kernel` | 25 000 × 128 B + scales ≈ 3.3 MB | **25 000 page reads** at `page_size = 1`; one scatter-read per token | naïve per-token loop, no SMEM tile, no cp.async |

INT8 reads **half the KV bytes** but pays **16× more page lookups**
and loses every optimisation the FlashInfer BF16 kernel has. At low
context (4 k tokens) the page-lookup overhead is small relative to
the fixed kernel launch and softmax reduction cost, so the two paths
are within ~3 ms; at 25 k the scatter-read cost completely dominates.

At L4 HBM (300 GB/s) the bf16 path bandwidth-bounded theoretical for
a 6.4 MB read is ~21 µs per KV-head per layer → ~6 ms per step for
all 8 × 36 KV layers. Observed 33 ms decode ITL is within the
reasonable kernel-overhead + softmax-reduce envelope.

For int8, 3.3 MB theoretical bandwidth ≈ 11 µs per KV-head per
layer → ~3 ms per step if bandwidth-bound; but observed 62 ms ITL
is **20× that theoretical**, confirming the kernel is **latency-
bound on scattered small reads**, not bandwidth-bound.

## Fix

The project docs already picked the fix direction in M0.3 §1.3:

> **Schedule P1.5 to either rewrite the INT8 kernels with
> `pos / page_size` decomposition or move INT8 to a separate HND
> format.**

Three concrete fix options, ranked by cost:

### Option A — complete the in-file TILE_TOKENS optimisation (cheapest, keeps page_size=1)

The `decode_attention_quantized.cu` file already imports `cuda_pipeline.h`
and defines `TILE_TOKENS = 16` — wire them up. Load 16 tokens' worth
of K/V data + 16 K/V scales into `__shared__` tiles via `cp.async`,
overlap the next tile's prefetch with the current tile's QK/V compute,
and process the softmax across a tile instead of a token.

- Files touched: 1 (`decode_attention_quantized.cu`)
- Estimated LOC: ~200–400 in the INT8 partial kernel, similar again
  for FP8
- Risk: kernel correctness regression on an already-passing unit
  test surface — compare against `decode_attention_int8_cuda` unit
  tests before declaring victory.
- **Does not change the pool format or any Rust side.** Pure kernel
  rewrite inside one `.cu` file.
- Expected win: at 25 k, drop int8 ITL from 62 ms back toward
  bf16's 33 ms, since the main cost is the serialised scatter reads,
  not bandwidth.

### Option B — lift INT8 pool to `page_size = 16` (medium cost, matches M0.3 story)

Rewrite `kv_cache_to_paged_int8_kernel` at `kv/kv_cache_to_paged.cu`
and `decode_attention_int8_partial_kernel` together so both agree on
a 16-token page layout. This is the "INT8 with `pos / page_size`
decomposition" path from tasks.md §1.3.

- Files touched: `kv/kv_cache_to_paged.cu`, `attention/decode_attention_quantized.cu`,
  `paged_kv.rs` (pool field changes), `kv_types.rs` (dispatch table),
  plus tests.
- Estimated LOC: ~500–800
- Risk: bigger blast radius, crosses into pool allocator; breaks
  `scatter_kv.cu` and `quantize_paged_kv_fp8_kernel` in the same
  breath, which the tasks.md audit (§1.2) also flagged as hardcoded
  page_size=1.
- Expected win: same as Option A plus page-level allocator savings
  in the pool (fewer `free_pages` metadata entries at the same
  capacity), plus alignment with the M0.3 "per-format dispatch"
  design intent.

### Option C — document and park until a second consumer (zero cost, no fix)

Flip `KVFormat::INT8` / `FP8E4M3` to "memory lever, not perf lever"
in the CLI help text, the `2026-04-15-bench-kv-quant-sweep.md`
recommendation, and `docs/projects/tiered-kv-cache.md`. Leave the
kernel as-is until someone actually needs int8 at long context, and
treat that as the trigger to schedule Option A.

- Cost: a handful of doc edits.
- Risk: ships a known-slow long-context path and surprises users
  who pick int8 expecting "fast-and-memory-efficient" behaviour.

**Recommendation**: before committing to either option, **profile
first**. Codex's investigation flagged the same point:

> Before committing to [a page_size lift], run `ncu` on
> `decode_attention_int8` at 25 k tokens to measure achieved HBM BW
> and SM occupancy. If the kernel is memory-bound but underutilising
> BW due to warp stalls (likely), the page-size lift alone may not
> close the gap and the kernel itself needs restructuring.

Concretely:

1. **Profile first** (no code change):
   - `ncu --set full --kernel-name regex:decode_attention_int8_.*_kernel target/release/infer --model-path models/Qwen3-4B --num-slots 1 --max-seq-len 32768 --kv-cache-dtype int8 &` then drive a 25 k request at the server. Capture `dram__bytes_read.sum.per_second`, `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.sum`, and the stall reasons.
   - If DRAM BW is already saturated, the page_size lift is the
     only lever that helps.
   - If SM occupancy is low and the primary stall reason is long
     scoreboard (load-latency) waits, the cheap Option A fix
     (SMEM tile + cp.async, still at page_size=1) unblocks the
     kernel without touching the pool.
   - If compute-bound on softmax reduction, neither help as much
     and the rewrite needs a different structure.

2. **Land the cheapest fix that the profile justifies**:
   - If profile points at Option A: commit a focused
     `perf(kv-quant): SMEM-tile + cp.async int8 decode kernel` under
     `crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
     alone. The commit should attach a 4k / 8k / 16k / 25k rerun
     of `/tmp/longseq_bench.py` showing int8 ITL within 5 % of
     bf16 at 25 k.
   - If profile points at Option B: open a new tasks.md §6 ticket
     "M0.4 — lift INT8/FP8 paged pool to page_size = 16" with a
     brief mapping the kv_types → paged_kv → decode kernel blast
     radius codex surfaced
     (`batch_decode.rs:449,494`, `quantize_paged_kv_single`,
     `kv_cache_to_paged_int8_kernel`). Schedule under M3 or M4
     depending on whether tier promote/demote wants int8.

3. **Regardless of path**: add a long-seq smoke gate (e.g.,
   `bench_throughput_sweep --quick-longseq --kv-cache-dtype int8`
   at 4 k / 16 k / 25 k) to the tiered-kv remote acceptance matrix
   so the next deferral of a quantised-kernel rewrite ships with
   numbers, not just "token-granular for now".

## Rule

**A "dequantise in registers" quantised decode attention kernel is
not automatically fast.** The fused-dequant approach only pays off
if the rest of the kernel is tile-pipelined and coalesced to modern
standards. A naïve per-token inner loop with scalar scale loads
pays the full scatter penalty every token, and at long context that
penalty is ~2× worse than the bf16 FlashInfer path on the same
pool. "Fused dequant" in the kernel docstring is a necessary but
not sufficient condition for int8-faster-than-bf16; the rest of
the kernel has to actually be optimised.

**Corollary**: when deferring a kernel rewrite (here, the M0.3 P1.5
page_size lift for quantised pools), **measure the cost of the
deferral before treating "stays at page_size = 1" as a zero-cost
caveat**. M0.3 explicitly shipped `{ BF16 => 16, INT8 => 1 }` with
the rationale "token-granular kernels stay as-is for now" — but
"stay as-is for now" turned out to mean "25 k decode is 2× slower
than bf16", which would have disqualified int8 from any long-context
bench gate if anyone had measured it at M0.3 time. The long-seq
bench gate should be added to the M0.3 / M3 ongoing remote
acceptance matrix so the next deferral of a quantised-kernel
rewrite at least has numbers attached.

**Corollary 2**: `TILE_TOKENS` + `#include <cuda_pipeline.h>` in a
kernel file that does not actually use them is a code smell — it
means someone started an optimisation and never finished it. A lint
pass over `crates/cuda-kernels/csrc/` looking for this
pattern would turn up at least one real item (this one) and
probably more.
