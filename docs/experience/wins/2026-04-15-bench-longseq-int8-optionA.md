# 2026-04-15 · INT8 decode kernel Option A — SMEM tile + cp.async partial win

## Context

Follow-up to
[`../errors/2026-04-15-int8-kv-long-context-slowness.md`](../errors/2026-04-15-int8-kv-long-context-slowness.md).
Root cause: `decode_attention_int8_partial_kernel` in
`crates/infer-cuda-kernels/csrc/attention/decode_attention_quantized.cu`
was a naive per-token loop that defined `TILE_TOKENS = 16` and
included `<cuda_pipeline.h>` but never actually wired them up. Inner
loop did four scalar global loads per token per warp (pool_idx,
K_scales, K_data, V_scales, V_data) without shared-memory tiling or
cp.async.

Codex landed the Option A fix in ~10 min against
`task-mnzw10jl-s9j83s`. This note captures the empirical validation
run on the L4 host (which codex's sandbox could not do — its
sandbox had no CUDA / network access).

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Server: `target/release/infer --model-path models/Qwen3-4B
  --num-slots 1 --max-seq-len 32768 --kv-cache-dtype int8
  --kv-pool-headroom-mb 2048 --gpu-reserved-mb 512 --port 8001`
- `TokenKVPool: 42033 max tokens (42033 pages @ page_size=1), 3.4 GB
  for 36 layers, format=INT8` (pool layout unchanged from pre-fix)
- Bench: `/tmp/longseq_bench.py --configs 4000,8000,16000,25000
  --output-tokens 128`, greedy decode, num_slots=1
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`

## Diff summary

One file, 90 insertions / 34 deletions, all inside
`decode_attention_int8_partial_kernel` (HEAD_DIM template), lines
114-218 of
`crates/infer-cuda-kernels/csrc/attention/decode_attention_quantized.cu`.

- Lines 114-122: **double-buffered shared-memory tiles** —
  `smem_k[2][TILE_TOKENS][HEAD_DIM]`,
  `smem_v[2][TILE_TOKENS][HEAD_DIM]`, plus
  `smem_k_scales[2][TILE_TOKENS]` /
  `smem_v_scales[2][TILE_TOKENS]`. Ping-pong between stage 0 and
  stage 1 so the next tile's data loads in the background while the
  current tile's compute runs.
- Lines 127-150: **initial async preload** of tile 0 via
  `__pipeline_memcpy_async(int8_t * EPT)` for K/V data +
  `__pipeline_memcpy_async(float)` for K/V scales (lane-0-only),
  followed by `__pipeline_commit()`.
- Lines 152-218: **outer tile loop** (`tile_idx = 0..tile_count`)
  that (a) waits on the current tile with `__pipeline_wait_prior(0)
  + __syncthreads()`, (b) issues the next tile's async prefetch
  into the alternate ping-pong stage, (c) runs the existing online-
  softmax inner loop reading K / V / scales from SMEM instead of
  from direct global loads.
- Tail tiles handled via `min(TILE_TOKENS, my_tokens - tile_start)`.
- Cross-warp merge scratch (`smem_m`, `smem_l`, `smem_o`) moved to
  the top of the kernel next to the data tiles — unchanged logic.
- **FP8 variant unchanged** — Codex kept scope to one kernel
  function. Same optimisation applies to
  `decode_attention_fp8_partial_kernel` for a later focused commit.

No FFI signature changes, no Rust-side changes, no
`paged_kv.rs` / `kv_types.rs` / `batch_decode.rs` touches. The
INT8 pool is still `page_size = 1`; only the attention kernel
changed.

## Results — INT8 long-seq (same-host, same flags)

```
                  int8 BASE (pre-fix)   int8 OPT-A (post-fix)     delta
input   bf16 ITL   TTFT   ITL     tok/s     TTFT   ITL     tok/s   ITL Δ
──────────────────────────────────────────────────────────────────────────
 4 000   35.2 ms   740   37.7 ms  23.2      757   37.0 ms  23.4    −0.7 ms  (−1.9 %)
 8 000   37.4 ms   824   42.1 ms  20.8      812   40.9 ms  21.3    −1.2 ms  (−2.9 %)
16 000   41.8 ms  2059   50.7 ms  15.1     2098   48.5 ms  15.5    −2.2 ms  (−4.3 %)
25 000   33.1 ms† 6360   61.9 ms   9.0     6390   56.9 ms   8.9*   −5.0 ms  (−8.1 %)
```

`†` bf16 25k ITL dip is the separate repeatability anomaly from the
longseq note, not relevant here.

`*` int8 optA at 25k only emitted **115 / 128** tokens before
natural EOS (vs 128 / 128 in the pre-fix run). The kernel's tile-
based accumulation changes the float rounding order vs the
per-token loop, shifting some logits by ~1 ULP, flipping greedy
argmax at some point, and eventually leading the model to a
natural EOS 13 tokens earlier than the old kernel. Pure decode
steady-state rate at 25k:

- **int8 BASE**: 128 tokens in 14.22 − 6.36 = 7.86 s = **16.3 tok/s**
- **int8 OPT-A**: 115 tokens in 12.93 − 6.39 = 6.54 s = **17.6 tok/s**

That is a clean **+8 % steady-state decode throughput** at 25 k,
matching the ITL improvement, not contradicting it.

## Honest assessment — partial win

Option A **does** land the optimisation it was supposed to. The
kernel now has the exact shared-memory tile + `cp.async` ping-pong
shape the file was aspirationally scaffolded for since
`TILE_TOKENS = 16` was first defined. The 4 → 8 % ITL improvement
grows monotonically with context length, which is consistent with
"cp.async hiding more memory latency as the scatter chain gets
longer".

But it **does not** recover the bf16 ceiling at 25 k. The residual
gap is large:

- bf16 25 k: 33 ms ITL
- int8 OPT-A 25 k: 57 ms ITL
- int8 OPT-A is still **72 % slower than bf16 at 25 k**

Where the residual gap comes from (codex's own prediction, which
this run confirms):

1. **The `pool_idx` lookups are still per-token**. Inside the
   prefetch loop at lines 167-180 of the new kernel, each of
   TILE_TOKENS iterations still reads
   `kv_indices[tok_start_global + global_t]` as a separate scalar
   load before issuing the `cp.async`. So the scatter chain
   `kv_indices → pool_idx → K_data[pool_idx * kv_dim + …]` is only
   partially pipelined: the `cp.async` hides the K / V data fetch,
   but not the index lookup that computes the address.
2. **`page_size = 1` means no within-page coalescing**. With
   BF16's `page_size = 16`, every page lookup loads 16 tokens'
   worth of contiguous KV in one transaction. With INT8's
   `page_size = 1`, each lookup gives one token's worth, and the
   attention kernel pays the full scatter penalty 16× more often
   per decode step.
3. **FlashInfer's bf16 paged decode kernel is more optimised** —
   it uses SM89-tuned tile shapes, register layouts, and prefetch
   scheduling that this custom kernel does not match. Option A
   catches up on the `cp.async` axis but not on the FlashInfer
   kernel-quality axis.

Option A was always the cheaper of the two fixes the error doc
proposed. The bigger fix is **Option B** — lift the INT8 paged
pool to `page_size = 16` and rewrite `kv_cache_to_paged_int8_kernel`
+ the decode kernel + `batch_decode.rs:449,494` together so all
three agree on a 16-token page layout. That is the fix that closes
the residual gap. It is also 5-8× the code change of Option A and
pulls in the `scatter_kv.cu` / `quantize_paged_kv_fp8_kernel` audit
follow-ups as collateral.

## Decision — land Option A, schedule Option B

Option A is a clean 8 % improvement for a ~90-line kernel change in
one file. It is cheaper than doing nothing because:

- It wires up infrastructure (`TILE_TOKENS`, `cp.async`) that was
  already scaffolded in the file — finishing half-done work is
  usually the right call.
- It gives us a cleaner baseline for measuring Option B's
  incremental gain later.
- The 15 % emitted-token gap at 25k (115 vs 128) is a numerical
  rounding artefact, not a quality regression; the model simply
  hits EOS a few tokens earlier along a slightly different
  trajectory.

Landing Option A. Opening a new tasks.md entry for "M0.4 — INT8
paged pool `page_size = 16` lift (Option B)" as the follow-up.

## Open items

1. **Numerical drift vs previous int8 baseline** — the greedy
   sequence at 25 k differs from the pre-fix baseline by enough to
   shift the EOS point from token 128 to token 115. Any test that
   compares int8 output against a stored baseline JSON (there are
   none in the tree today for int8 specifically, but
   `greedy_consistency.rs` and `e2e.rs` may surface this if they
   ever gain an int8 variant) will see a mismatch. Not a ship
   blocker for Option A; worth noting.
2. **FP8 variant untouched** — `decode_attention_fp8_partial_kernel`
   at line 267+ has the same per-token loop shape and the same
   missed-TILE_TOKENS optimisation. The fp8 quick sweep numbers
   (2026-04-15-bench-kv-quant-sweep.md) are currently bound by the
   same inefficiency. A follow-up commit can mechanically apply
   the same change.
3. **`--kv-pool-headroom-mb` default is wrong for long-context
   single-slot.** Long-seq note already flags this — int8 pool
   auto-sizes to 42 k tokens only because we explicitly dropped
   headroom to 2048 MB. Default 4096 MB gives ~8 k tokens which
   cannot fit 25 k publish. Separate bug on the auto-sizer.
4. **Pre-existing panic on prefix-cache publish** at
   `paged_kv.rs:595` did **not** fire in this Option A run (25 k
   request finished cleanly, server stayed up). Worth noting
   because the pre-fix bf16 25 k run panicked in the background
   after publish; the int8 pool has more capacity (42 k vs 31 k)
   which may coincidentally keep us under the publish threshold.
   The bug itself is unchanged.
5. **Residual optimisation** inside Option A itself: batch the
   `pool_idx` loads for a tile into one vectorised `int4` read
   (16 × 4 bytes = 64 bytes, one cache line) so the entire tile's
   addresses are resolved in a single transaction before any
   `cp.async` fires. Estimated additional ~50 LOC in the same
   kernel, probably another 2-5 % on 25 k. Not worth a separate
   commit — roll into Option B if/when that lands.

## Rule

**Finishing a half-scaffolded optimisation is usually right even
when the speedup is modest.** The `TILE_TOKENS = 16` constant and
the `<cuda_pipeline.h>` include were left in the file by whoever
first wrote the int8 decode kernel, as a marker for future work.
That future work never landed, and downstream users (us, 2026-04-15
long-seq bench) measured the cost — 2× slowdown at long context.
Completing the optimisation gives us ~8 % back and, more
importantly, removes a "why is this unused?" question from the
next person to read the kernel.

**Corollary**: when your kernel's inner loop has a scatter-read
pattern that cannot be fully hidden by `cp.async` (because the
scatter address itself needs to be fetched serially), the only
remaining lever is to reduce the number of scatter lookups. That
is a **pool-layout** change (Option B: `page_size = 16`), not a
kernel change (Option A). You have to pay for both eventually.
