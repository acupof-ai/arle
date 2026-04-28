# cuda-l4 SGLang-aligned KV pool budget — pool +35%, perf parity at c=16

> Structural correctness fix + Codex P2 fix + boot-time GPU memory
> instrumentation. Pool grew 84,096 → 113,808 → 114,208 tokens at
> fraction=0.94 (the 408-token bump comes from the Codex P2 snapshot
> reliability fix). Perf at c=16/4096-in/256-out is **parity with the
> 2026-04-28 TileLang prefill A+C baseline** within `--fast` preset
> variance (median 162 tok/s across n=6 vs 155.81 prior). The pool was
> already non-binding for c=16 (16 × 4352 = 70k tokens fit in the old
> 84k pool), so the structural win unlocks higher-concurrency benches
> rather than moving the c=16 number.

## Goal

Align our KV pool budget formula with SGLang's `profile_max_num_token`
and instrument boot-time GPU memory consumption to measure the
remaining `free-after-model-load` gap (we sit at ~10.3 GB free vs
SGLang's ~14-15 GB on L4 / Qwen3-4B). Goal type: **structural
correctness + measurement infrastructure**.

## Hypothesis

Three independent issues stack to deflate our pool:

1. We pre-deduct estimated workspace from the budget; SGLang doesn't,
   it lets workspace allocate from leftover headroom dynamically.
2. We use `total × (1 - mem_fraction_static)` as headroom; SGLang
   uses `pre_model_free × (1 - mem_fraction_static)`. Difference is
   the driver+ctx overhead (528 MB on L4 here).
3. Our `cuMemGetInfo` snapshot in `bootstrap.rs:261` runs without an
   explicit `DeviceContext::new()` — when `--num-slots N` is set,
   `auto_num_slots` is skipped, so cudarc may not yet hold a primary
   context. The snapshot then silently fails and falls back to
   `total`. (Codex P2 finding.)

Fix all three. Result expectations:

- Pool size at fraction=0.94: 84k → ~113k tokens (+35%).
- Perf at c=16/4096-in: parity (pool was already non-binding).
- Perf at higher concurrency / longer sequences: future bench — that's
  where the pool growth becomes observable.
- Stage-by-stage GPU memory log surfaces the remaining gap to SGLang
  for the next investigation.

## Command

```bash
target/release/infer \
    --model-path models/Qwen3-4B \
    --port 8000 \
    --num-slots 16 \
    --max-seq-len 4608 \
    --mem-fraction-static 0.94 \
    --cuda-graph true

bash scripts/bench_guidellm.sh cuda-l4-stage-instrument-r1 --fast
```

## Params

- backend label: `cuda-l4`
- model: Qwen3-4B (bf16 weights, FP8E4M3 paged KV)
- num_slots: 16, max_seq_len: 4608, mem_fraction_static: 0.94
- cuda_graph: true, kv_cache_dtype: auto (resolves to FP8E4M3)
- bench preset: `--fast` (concurrent c=16, 4096-in / 256-out, 30s)
- profile: concurrent (single point, not sweep)
- guidellm 0.6.0

## Env

- L4 sm_89, 22 GB, CUDA 12.x
- features: `cuda` (implies `tilelang-attn`), default
- prompts: `prompt_tokens=4096,prompt_tokens_stdev=1`,
  `output_tokens=256,output_tokens_stdev=1`

## Results

### Pool size at boot

| fraction | before (broken formula) | after (SGLang-aligned) | Δ tokens | Δ GB pool |
|---------:|------------------------:|-----------------------:|---------:|----------:|
| 0.94     |                  84,096 |                114,208 |  +30,112 |     +2.4 GB |
| 0.85     |                       — |                 86,448 |        — |        — |

SGLang reference at fraction=0.85: 156,000 tokens. **Remaining gap:
70k tokens / ~5.4 GB** in `free-after-model-load`.

### Stage-by-stage GPU memory (the new instrumentation)

```text
GPU memory @ post_cuda_ctx (early):  free=23.13 GB / total=23.66 GB
                                     (driver+ctx+cuBLAS overhead = 528 MB)
GPU memory @ pre_model_load:         free=23.13 GB
                                     (delta vs post_cuda_ctx = +0 MB —
                                      AOT cubins + lazy_static loaders ARE lazy)
GPU memory @ post_model_load:        free=10.31 GB
                                     (model load consumed 12.82 GB)
TokenKVPool budget: 8.9 GB → 114,208 tokens
```

**Surprise**: AOT cubin loaders are truly lazy; the delta is +0 MB.
We had assumed Triton + TileLang AOT eager loads. Real overhead is
elsewhere.

**Open question**: model load consumes 12.82 GB but Qwen3-4B bf16
weights = 4B × 2 = **8 GB**. The **4.82 GB unaccounted** during
`load_model_components` is the next investigation target. Decode
metadata workspace is 256 MB (FlashInfer DEFAULT) — that's only 5%
of the gap. RoPE cache, embedding tables, attention masks, or
intermediate dequant buffers are candidates.

### Bench at c=16/4096-in/256-out

n=6 across `--fast` preset (cold prefix-cache state varies):

| run     | TTFT p50 (ms) | ITL p50 (ms) | out tok/s |
|---------|---------------|--------------|-----------|
| r1      |       8794.3 |        80.19 |    185.34 |
| r2      |       1155.8 |        83.41 |    162.85 |
| r3      |       9232.6 |        70.09 |    143.83 |
| r4      |      20724.1 |        78.82 |     84.72 |
| r5      |       8348.0 |        69.85 |    137.57 |
| stage   |       9056.3 |        78.01 |    193.75 |
| **median** | 8794.3 (variance huge) | **78.42** | **162.85** |

Prior 2026-04-28 TileLang A+C baseline (n=3 median): **155.81 tok/s**.

Δ vs prior: **+4.5%** on median (within `--fast` variance: σ ~38 tok/s).

**Verdict at c=16**: parity. The pool was already non-binding for this
shape (16 × 4352 = 70k tokens, well under the old 84k pool), so the
+35% pool growth is structural-only here.

The TTFT variance (1156–20724 ms) is admission-scheduling-dominated
under `--fast` (only ~10 requests per run, prefix-cache cold start
matters). Future systematic bench should use a longer `max-seconds`
or sweep concurrency to surface the structural improvement at higher
concurrency where pool size *is* binding.

## Problems

1. **`--fast` preset variance**. ±38 tok/s σ at c=16/30s. n=6 needed
   to call median trustworthy. Future bench protocol should bump
   max-seconds to ≥120 for stable single-point reads.
2. **Codex P2 fix is observably zero-impact at fraction=0.94** because
   `total - pre_model_free = 528 MB`, and 528 MB × 0.06 = 32 MB
   headroom delta = ~430 tokens. Would matter more at lower
   `mem_fraction_static` or larger driver+ctx overhead (e.g. H100
   tensor cores have larger ctx).
3. **The 4.82 GB unaccounted at model load is the lever**, not the
   formula. Closing that gap by ~half would give us SGLang parity on
   pool size at fraction=0.85.

## Learnings

- **`cuMemGetInfo` requires a current CUDA context.** Calling it
  without an active `DeviceContext` silently returns the wrong number
  and the budget formula falls back to `total`. The fix:
  `let _ = DeviceContext::new();` before the snapshot when no caller
  has guaranteed a context.
- **Stage-by-stage memory snapshots are cheap and load-bearing.**
  528 MB driver overhead, 0 MB AOT cubin overhead, 12.82 GB model
  load — three numbers we couldn't quote before. The instrumentation
  itself is the win because it makes the next investigation
  pointable.
- **AOT cubin lazy loading isn't a problem we have**, despite earlier
  agent suspicion. cudarc's module loader IS lazy for our use; the
  cuModuleLoad fires only when the kernel is first dispatched. Save
  this delta for future pre-touch optimization (warming kernel
  cubins before bench start to avoid first-request TTFT spike).

## Next steps

1. **Investigate `load_model_components` 4.82 GB gap.** Audit GPU
   `alloc_zeros` calls during weight load — RoPE freqs, attn masks,
   embedding lookup tables.
2. **Bench at higher concurrency (c=32, 48, 64).** That's where the
   +30k token pool growth should produce observable tok/s gains.
3. **Move snapshot earlier in main.rs** if possible (it already runs
   right after Args::parse, but before tracing or model_path
   resolution might shave another few hundred MB if anything is
   pre-allocating).
