# Marlin stored the model twice: 18.7 GB back, 8.4x on the 32K chain — CUDA, 2026-08-20

> Status: Shipped, then superseded the same day. `free_quant_source_after_marlin`
> and `QWEN_MARLIN_MAX_M` named below no longer exist: the frees moved inline into
> the two repacks, and prefill moved off Marlin entirely. The 23.06 GB resident
> here is now 22.36 GB with the prefill arms on — see
> [2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md](2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md).
> The mechanism this entry established — a repack that keeps its source stores
> the model twice — held, and was violated again within hours.

## Context

Qwen3.8-27B-NVFP4 served a 32K long-agent workload at c=4 in ~7× the wall clock
of Qwen3.6-27B-FP8 — and the same NVFP4 arm had matched FP8 on an older build.
Decode was not the problem: ITL was ahead of FP8 at every
concurrency. The wall clock was flat at ~680 s whatever the concurrency, which is
not a slow kernel, it is work being redone.

## What the logs said

Two lines, same box, same run:

```
NVFP4: free 57371MB -> max_total_tokens 281577   |  24x "falling back to full recompute"
FP8:   free 68507MB -> max_total_tokens 593995   |   0x
```

The checkpoint with the *smaller* weights had *half* the KV capacity. Measured
from the safetensors headers rather than the docs:

| | file | resident | delta |
|---|---:|---:|---:|
| Qwen3.8-27B-NVFP4 | 23.42 GB | 42.08 GB | **+18.66 GB** |
| Qwen3.6-27B-FP8 | 30.87 GB | 30.41 GB | −0.46 GB |

The 18.66 GB is the Marlin layouts, held alongside the bytes they were built
from. Predicted from the shapes: NVFP4 packed + group scales `N*K/2 + N*K/16` =
8.43 GB, per-channel FP8 GPTQ `[K/4, N]` = `N*K` = 10.62 GB, total 19.05 GB —
2% above measured, the gap being the weights whose shape the repack declines.

The source was kept on purpose: `QWEN_MARLIN_MAX_M = 1024` sent prefill chunks
(2048) to dequant→BF16→cuBLAS, which reads the pre-repack bytes. So the model
was stored twice to keep one GEMM 12-21% faster.

The KV pool paid for it, and a 16-conversation x 8-turn workload does not fit in
half a pool: slots get reused, a cross-slot prefix restore lands between the
8192-token sidecar snapshots, and the whole 33K prefix is recomputed.

## What worked

1. **Free the pre-repack bytes at load** (`free_quant_source_after_marlin`,
   called from `marlin_repack_dense`). Rolling, per weight: load, repack, free,
   next — peak is one weight's two copies (2.5 GB at lm_head), not the model's.
2. **Marlin claims every M** for a weight it repacked (`fp4_route` / `fp8_route`
   lose their `m <= QWEN_MARLIN_MAX_M` gate), because the dequant arm it used to
   fall to has nothing to read.
3. **The single-row lane gets the same arms.** `quant_linear::gemv` is a second
   dispatch with no Marlin at all; `output_projection` reaches it with lm_head
   every single-row step. The Marlin calls moved into
   `marlin_fp4_gemm_raw` / `marlin_fp8_gemm_raw` so both lanes share them.
4. **The LoRA merge fails loudly** rather than skipping. `merge_base_fp8()`
   returns `None` both for "never was FP8" and "source released", and two of its
   three callers used `if let Some`. `quant_source_freed()` separates the cases.

## Result

1xH20, TP=1, FP8 KV, no spec. `bench-agent-32k-16x8.jsonl` (sha 8867f63e,
1,052,018 prompt / 6,848 output tokens per point, identical on both arms), 32
requests/point, `--max-tokens 214`. Both arms on the same binary, FP8
re-measured rather than reused.

ITL is ahead at every point: 20.44 /
40.04 / 70.81 against 24.80 / 47.38 / 78.96 (+21.3% / +18.4% / +11.5%).

Mechanism confirmed, not inferred:

```
free VRAM       57,371 MB -> 75,515 MB    (+19.0 GB, predicted 18.7-19.05)
resident weights   42.08 GB -> 23.06 GB    (file is 23.42 GB)
KV pool          281,577 -> 790,603 tokens (FP8: 593,995)
full recomputes       24 -> 2              (FP8: 0)
```

Correctness: needle ladder 3/3 exact at 512 / 4096 / 16384 / 32768,
`SERVER_ERRORS=0`, 32/32 requests complete on both arms.

c=1 still pays for prefill because this workload is 154:1 prefill-to-decode
and c=1 has no batching to amortise it, where Marlin now pays the 12-21% it
used to avoid. From c=4 the batching absorbs it.

## Rule

A repack that keeps its source stores the model twice, and the second copy is
invisible in every weights-only accounting. It cost more than it saved here in
two ways at once: the KV pool that turns into recompute, and deployability — a
23.4 GB 4-bit checkpoint did not fit a 32 GB card while the 30.9 GB FP8 one did.
Free the source, and make sure every dispatch lane can serve the layout that
replaced it: there were two lanes, and the second one was found by a crash, not
by reading.
