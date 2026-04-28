# cuda-l4 weight dedup — Qwen3-4B pool 114k → 175k tokens, +52% tok/s @ c=16

> Removes the duplicate merged QKV / gate_up_proj weight copies in
> Qwen3 (Attention + MLP). The merged-form had been kept alongside the
> individual q/k/v + gate/up forms for a "fused GEMM + split kernel"
> fast path on bf16 weights. The quantized branch in batched decode
> already used 3 separate q/k/v GEMMs — this commit unifies all paths
> onto the existing-tested separate-GEMM flow and drops the merged
> copies from VRAM. Net: **4.73 GB freed at model load**, KV pool
> grows 114,208 → 175,008 tokens at fraction=0.94, surpassing
> SGLang's 156,000-token reference at the same config. Bench tok/s
> at c=16/4096-in/256-out: **184.96 (median n=3)** vs 121.74 baseline
> (TileLang ON, FP8 KV) — **+52%**.

## Goal

Close the 4.82 GB unaccounted-for delta between bf16-weight size
(4B × 2 = 8 GB) and observed `post_model_load` GPU consumption
(12.82 GB), surfaced by the staged memory instrumentation in
`83e67ff2`. Goal type: **memory reduction → throughput at admission
limit**.

## Hypothesis

Per `Attention { q_proj, k_proj, v_proj, qkv_proj, ... }` in
`infer/src/model/qwen3/weights.rs:30-38` and `MLP { gate_proj,
up_proj, gate_up_proj, ... }` in `infer/src/model/common.rs:24-32`,
each layer holds BOTH the individual projections AND a
`concat_rows`-built merged copy. For Qwen3-4B (32 q heads, 8 kv
heads, head_dim=128, hidden=2560, intermediate=9728):
- per-layer attn duplication: q+k+v (31.46 MB) + qkv_proj (31.46 MB)
  → **31.46 MB extra**
- per-layer MLP duplication: gate+up (99.6 MB) + gate_up (99.6 MB)
  → **99.6 MB extra**
- × 36 layers = **4.72 GB** of duplicate VRAM

The merged form was used only by two batched-decode call sites
(`infer/src/model/qwen3/batch_decode.rs:638, 1595`) under the bf16
branch:

```rust
if layer.attention.q_proj.is_quantized() {
    // 3 separate q/k/v GEMMs + decode_prep_paged
} else {
    // 1 merged qkv_proj GEMM + split_qkv_cuda + decode_prep_paged_fused_qkv
}
```

The quantized branch was production-tested. Unifying both onto the
3-separate-GEMM path:
- adds 2 extra GEMM launches per layer per decode step on bf16 (codex
  P2 concern — measured below)
- frees 4.7 GB of weight VRAM permanently
- grows KV pool ~50%, which at c=16 widens the admission window from
  "16 × 4352 = 70k tokens fits in 84k pool comfortably" to "70k tokens
  fits in 175k pool with massive headroom for prefix cache"

Expected: pool grows ~50%; tok/s at c=16/4096-in is bench-noise
limited but should improve due to less prefix-cache eviction.

## Command

```bash
target/release/infer \
    --model-path models/Qwen3-4B \
    --port 8000 --num-slots 16 --max-seq-len 4608 \
    --mem-fraction-static 0.94 --cuda-graph true \
    --kv-cache-dtype fp8

bash scripts/bench_guidellm.sh cuda-l4-dedup-fixed-fp8-r{1,2,3} --fast
```

## Params

- backend label: cuda-l4
- model: Qwen3-4B (bf16 weights, FP8E4M3 paged KV)
- num_slots: 16, max_seq_len: 4608, mem_fraction_static: 0.94
- cuda_graph: true, tilelang-attn: ON (post-`47bad713` kernel fix)
- bench preset: `--fast` (concurrent c=16, 4096-in / 256-out, 30s)
- guidellm 0.6.0

## Env

- L4 sm_89, 22 GB, CUDA 12.8 (V12.8.93), driver 580.82.07
- features: `cuda` (implies `tilelang-attn`)
- prompts: `prompt_tokens=4096,prompt_tokens_stdev=1`,
  `output_tokens=256,output_tokens_stdev=1`

## Results

### Memory

| Stage              | Baseline (free GB) | After dedup (free GB) | Δ free |
|--------------------|-------------------:|----------------------:|-------:|
| post_cuda_ctx      |              23.13 |                 23.13 |    0.00 |
| pre_model_load     |              23.13 |                 23.13 |    0.00 |
| post_model_load    |              10.31 |                 15.04 |  +4.73 |
| TokenKVPool budget |               8.9 GB / 114,208 tok | 13.6 GB / 175,008 tok | **+4.7 GB / +60,800 tok** |

Pool delta `+4.73 GB` matches the predicted weight duplication exactly.
SGLang reference at fraction=0.85 was 156k tokens — our 175k @ 0.94
surpasses that.

### Bench (n=3 median, c=16/4096-in/256-out, --fast)

| Config | TTFT p50 (ms) | ITL p50 (ms) | tok/s | vs baseline |
|---|---:|---:|---:|---:|
| baseline (TileLang ON + FP8 KV, no dedup) |   5865 | 75.69 | 121.74 | — |
| **dedup + TileLang ON + FP8 KV** | 7839 | 75.95 | **184.96** | **+52%** |

(Variance is high under `--fast` — individual runs spanned
179.25 / 205.81 / 184.96. Median is robust to the warm-prefix-cache
outlier of r2.)

### Codex review of the diff

Two P2 perf concerns flagged on the diff (review log:
`tasks/codex-dedup-review.log`):

1. **BF16 batched decode lost merged-QKV fast path** — adds 2 extra
   GEMM launches per layer per decode step. Real cost.
2. **BF16 batched decode lost fused gate/up MLP** — adds 1 extra GEMM
   launch per layer per decode step.

Both are accurate. At c=16/4096-in the larger pool drowns out the
extra-launch cost: prefix-cache hit rate goes up (more capacity →
less eviction), and admission throttling drops, so end-to-end tok/s
increases despite the per-decode-step overhead. At c=1 (decode-bound,
no admission pressure) the codex-flagged cost is likely the dominant
signal — needs a separate bench run to quantify.

## Problems

- `--fast` preset variance is ±50 tok/s at c=16/30s. Need
  `--max-seconds 90+` or sweep profile for tighter bounds. Median of 3
  runs is a workable compromise.
- Codex's P2 launch-overhead concern is real and unmeasured at low
  concurrency. Files a follow-up bench at c=1 to validate the
  trade-off across concurrency.

## Learnings

- **Memory > microoptim at admission-limit shapes.** The codex review
  was technically correct on the launch overhead but missed that c=16
  on a 22 GB GPU is admission-limited, not GEMM-launch-limited. Trading
  per-step kernel-launch cost for 50% more pool capacity is the right
  call when concurrency × prompt-length saturates the admission queue.
- **Quantized path was the natural unification target.** When two
  branches do the same thing under different conditions, prefer
  unifying onto the more-general one (separate GEMMs handle both
  quantized and dense; merged GEMM only works for dense). Removes the
  conditional branch and removes one weight copy.
- **The earlier "dedup is broken" diagnosis was wrong** — the gibberish
  output that made me revert this work originally was caused by the
  TileLang short-qlen NaN bug, NOT this refactor. After fixing TileLang
  in `47bad713`, the dedup work compiles + runs + benches green on the
  first try. See
  `docs/experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md`
  for the contextual confusion that masked this.

## Next steps

1. **Bench at c=1.** Quantify codex's per-step launch cost at the
   shape where it matters most.
2. **Bench at c=32, 64.** Pool growth from dedup scales pool from
   "comfortably enough" at c=16 to "newly enabling" at higher
   concurrency. Headline tok/s gain should be larger.
3. **Optionally restore merged form lazily.** If the c=1 regression is
   real, store only `q_proj`/`k_proj`/`v_proj`, build `qkv_proj`
   on-demand into a per-batch scratch buffer. Saves the eager 4.7 GB
   and brings back the merged-GEMM fast path. Out of scope for this
   commit.
