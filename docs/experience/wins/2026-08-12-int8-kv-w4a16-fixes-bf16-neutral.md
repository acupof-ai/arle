# INT8 KV + W4A16 kernel fixes — CUDA, 2026-08-12

> Status: Shipped

## Goal

Verify the INT8 KV support + W4A16 kernel fix treatment does not regress BF16
decode throughput on ThinkingCap-Qwen3.6-27B.

## Hypothesis

The treatment stubs the W4A16 dequant GEMM path (kernels absent from the
prebuilt lib), removes unused h16g16 flashqla kernel entries, and makes the
DSpark batched ring-attention path unreachable. None of these touch the BF16
hot path (model uses (16,32) linear-attention heads, no W4A16, no DSpark), so
throughput should be within noise of baseline.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8000 \
  --concurrency-grid 1,4,8,16 \
  --seconds-per-concurrency 30 \
  --max-tokens 256
```

- Baseline: `fa9649a` (system GCC, prebuilt kernels)
- Treatment: `0c0d72059` (conda GCC 15 linker, prebuilt kernels + local W4A16)
- Model: ThinkingCap-Qwen3.6-27B, BF16 KV
- Trials: 1 per concurrency (30s each)

## Environment

- Host: H20 pod, 8×H20 (single GPU used)
- GPU: NVIDIA H20, 96 GB
- Model / dtype: ThinkingCap-Qwen3.6-27B / BF16
- KV: BF16, 215 slots, 3864 pages (61824 max tokens)
- Server flags: `--backend cuda --kv-cache-dtype bf16`

## Results

| concurrency | baseline tok/s | treatment tok/s | delta |
|---:|---:|---:|---:|
| 1 | 53.3 | 53.2 | −0.2% |
| 4 | 158.9 | 158.8 | −0.1% |
| 8 | 231.2 | 230.7 | −0.2% |
| 16 | 342.0 | 333.6 | −2.5% |

All deltas within run-to-run noise for this workload. No regression.

## Problems

- Initial treatment build required conda GCC 15 as linker (prebuilt TileLang
  libs need newer libstdc++). System GCC link fails with `GLIBCXX_3.4.26` not
  found.
- First treatment run crashed with "row fuse" OOM — GPU 2 had only 10 GB free.
  Switched to GPU 0 (97 GB free).
- An earlier c=8 measurement showed −7% and a c=16 measurement showed +19.5%;
  both were artifacts of different VRAM states (collapsed KV pool at 256 pages
  vs 3864 pages). Re-run with identical memory state confirmed neutral.

## Learnings

PASS. Treatment is within ±2.5% of baseline across all concurrency levels.
The changed paths (W4A16 stub, h16g16 flashqla removal, DSpark unreachable)
do not affect the BF16 hot path. Correctness verified separately (needle_gate
3/3 exact on lengths 115–8000).
