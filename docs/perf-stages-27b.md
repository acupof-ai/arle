# 27B-FP8 DSpark per-stage timing reference

> Reference for debugging throughput regressions. All numbers from
> `ARLE_CUDA_PROFILE=1` on the A/B binary (abb3461ef + pre-budget/truncate
> disabled), c=4, 8 requests, 32K prompts, 214 max-tokens, greedy.
> Linked from `docs/baselines.md` — the SOTA row carries the throughput
> numbers; this carries the per-stage breakdown that explains them.

## Per-step wall-clock (decode, c=4)

| Stage | Count | Busy μs | μs/step | Share |
|---|---:|---:|---:|---:|
| decode_forward_steps | 152 | 9,813,339 | 64,561 | 18% |
| mixed_forward_steps | 54 | 44,320,847 | 820,756 | 82% |
| prefill_forward_steps | 34 | 47,529,625 | 1,397,930 | — |

Decode is 18% of wall; mixed (prefill+decode overlap) is 82%. The 64.6 ms
decode step breaks down below.

## Per-layer decode timing (c=4, 153 forward_hidden calls)

| Layer type | Layers | Calls/layer | ms/call | ms/step (all layers) |
|---|---:|---:|---:|---:|
| full_attention_layer | 16 | 359 | ~3.8 | 60.8 |
| full_paged/attention_layer | 16 | 359 | ~2.8 | 44.8 |
| dense_ffn_layer | 16 | 565 | ~1.2 | 19.2 |

forward_hidden total: 84,983,687 μs / 153 calls = 555,449 μs avg.

## Per-stage decode cost (c=4, 64.6 ms/step)

| Stage | μs | Share |
|---|---:|---:|
| full attention (16 layers) | 60,800 | 38% |
| paged attention (16 layers) | 44,800 | 28% |
| dense FFN (16 layers) | 19,200 | 12% |
| sampling + argmax | ~2,000 | 3% |
| KV alloc + scheduler | ~3,000 | 5% |
| recurrent (GDR+conv, 32 layers) | ~15,000 | 9% |
| other (sync, memcpy, etc.) | ~10,000 | 6% |

## Bench-level metrics (128 requests/concurrency, 32K prompts)

### Baseline (fad8f4d5b, 2026-08-14)

| c | output tok/s | ITL mean ms | TTFT p50 ms | accept |
|---:|---:|---:|---:|---:|
| 1 | 46.94 | 10.89 | 978.4 | 40.64% |
| 2 | 99.48 | 17.06 | 581.3 | 27.35% |
| 4 | 124.42 | 29.46 | 584.8 | 27.06% |
| 8 | 152.94 | 48.86 | 625.1 | 27.63% |
| 16 | 168.30 | 89.34 | 878.6 | 27.28% |

### arle-gate (a8150bc6b, 2026-08-18)

| c | output tok/s | ITL mean ms | TTFT p50 ms |
|---:|---:|---:|---:|
| 1 | 48.03 | 10.34 | 928.9 |
| 2 | 99.96 | 17.19 | 572.1 |
| 4 | 126.30 | 28.55 | 572.2 |
| 8 | 155.72 | 47.61 | 598.5 |
| 16 | 173.18 | 84.55 | 1223.5 |

### pre-CP (abb3461ef, 2026-08-17)

| c | output tok/s | ITL mean ms | TTFT p50 ms |
|---:|---:|---:|---:|
| 1 | 48.18 | 10.32 | 937.0 |
| 2 | 94.04 | 17.61 | 648.7 |
| 4 | 119.79 | 30.03 | 591.2 |
| 8 | 143.46 | 51.20 | 611.6 |
| 16 | 156.92 | 95.24 | 955.0 |

## Regression signature (pre-CP vs arle-gate)

- c=1: flat (within ±2%)
- c≥2: ITL +2.4% to +11.2%, throughput −5.9% to −9.4%
- Acceptance rate: within 2pp at every point
- TTFT: noisy, no clear pattern

The regression is in **chain rate** (per-step decode time), not speculation
quality. It scales with concurrency — the batched decode path is the primary
suspect.

## Prefix cache overhead (per request, cold → warm)

- recurrent sidecar serialize: 146.8 MiB in ~70-80 ms (D2H copy)
- prefix-attach: matched=32416 restored=32416 (32K prompt, ~100% hit after warmup)
- 5 serializes per request during prefill (chunked sidecar store)

## Key configuration

- KV pool: 51853 pages, 54.4 GB, BF16 format
- Recurrent: 16 slots × 195 MB = 3127 MB
- L2 DRAM tier: ~774 GB budget
- DSpark: block_size=6, taps=[1,16,31,46,61]
- Decode graph: ARMED (16 slots, lazy capture per slot)
