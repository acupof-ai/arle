# NVFP4 TP8: per-channel FP8 scale replication under column sharding — CUDA, 2026-08-22

> Status: Shipped

## Goal

Serve `unsloth/Qwen3.8-27B-NVFP4` at `--tensor-parallel-size 8` and measure the
c=1..64 decode curve on the 8×H20 pod. TP>1 failed at engine build:
`FP8 block col shard 0..768 must align to block_k=6144 for cols=6144`.

## Root cause

The model's attention / linear-attention projections are FP8 per-channel
(F8_E4M3 + one BF16 scale per row, `block_k = cols`). The block-scale col
shard path required every shard boundary to align to `block_k`; with one scale
column per row, only TP1 satisfies that. The scale is per-row and a col shard
keeps every row, so each rank needs the whole scale column — replicate, don't
slice. The FP4 MLP group scales already align (group_size=128 divides
17408/8 and 5120/8); the weight bytes shard without alignment constraints.

## Fix

`crates/infer-cuda/src/loader.rs` — `shard_fp8_block_scales_cow`, Cols arm:
when `scale_cols == 1`, return the full scale tensor to every rank (9 lines).
`from_fp8_block_scaled` then indexes `scale[row][0]` for every local column —
the per-row scale, correct for any shard width.

TP8 also required `--spec-type none`: the checkpoint's MTP head is single-GPU
only ("TP-sharded MTP draft head not yet wired"). MTP is counterproductive on
this model anyway (2026-08-18 entry: 6.2 vs 9.3 tok/s).

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8080 \
  --model qwen \
  --concurrency-grid 1,2,4,8,16,32,64 \
  --seconds-per-concurrency 30 \
  --max-tokens 128 \
  --temperature 0 --seed 42 \
  --output /tmp/nvfp4_tp8_bench.json
```

- Baseline: TP>1 fails at load (assertion above); TP1 c=1 = 66.6 tok/s
  (2026-08-18 entry, kernel-ladder figure)
- Treatment: local working tree (loader fix + `26790bd4d`), TP8
- Prompt tokens: 8 (synthetic)
- Completion tokens: 128 per request
- Trials: 1

## Environment

- Host / GPU: 8×H20 pod, all 8 ranks
- Driver / CUDA: sm_90, CUDA 12.x
- Model: `/data00/Qwen3.8-27B-NVFP4` (22 GB safetensors + 811 MB MTP, MTP disabled)
- KV cache: FP8 E4M3 paged, 65,536 max total tokens
- Server flags: `--backend cuda --tensor-parallel-size 8 --kv-cache-dtype fp8 --spec-type none`

## Results

| concurrency | completed | errors | decode tok/s (aggregate) | req/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0 | 66.9 | 0.52 | 21 / 62 | 7.9 / 51 |
| 2 | 30 | 0 | 120.7 | 0.94 | 105 / 167 | 6.3 / 58 |
| 4 | 32 | 0 | 127.0 | 0.99 | 150 / 216 | 22 / 73 |
| 8 | 32 | 0 | 126.7 | 0.99 | 219 / 315 | 62 / 105 |
| 16 | 32 | 0 | 127.3 | 0.99 | 364 / 399 | 124 / 171 |
| 32 | 32 | 0 | 127.7 | 1.00 | 687 / 689 | 248 / 296 |
| 64 | 64 | 0 | 127.8 | 1.00 | 1266 / 1758 | 496 / 555 |

Aggregate decode saturates at ~127 tok/s from c=4 — the batching ceiling.
TP8 c=1 (66.9) matches TP1 (66.6): the FP4 GEMV is kernel-bound at 1.58% of
H20 bandwidth (2026-08-18 entry), so 8× the weight bandwidth cannot move the
single-request number. TP8's value here is capacity (headroom for the KV pool
and longer contexts), not c=1 latency.

Correctness: needle ladder 115/180/241/300/446/1000 ×3 on both TP8 and TP1 —
18/18 exact, deterministic on both (`NEEDLE_MAX_TOKENS=512`).

Raw artifacts: `/tmp/nvfp4_tp8_bench.json` (pod), run logs
`/root/arle-ops/runs/tp8-fix2/` and `tp1-base/`.

## Rule

Per-channel quantized weights (one scale per row) shard by replicating the
scale column under any column-parallel split — the alignment assertion only
applies when scales themselves are block-partitioned.
