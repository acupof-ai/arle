# DSv4 DSpark block-drafter: +37% decode at c=1

## Context

DSv4-Flash (8×H20, TP=8/EP=8, FP8 MoE) plain decode baseline is 53 tok/s at
c=1. The DSpark block drafter (3 stages, block=5, target layers [40,41,42]) was
wired but never measured end-to-end on the 8-GPU serve path.

## What Worked

- Moved base model + DSpark draft from HDD to NVMe: base load 924s → 9.9s
  (29.8 GB/s). The HDD path was timing out the engine-ready barrier.
- DSpark draft index.json in the non-`-fp8` dir is truncated (163840 B, EOF at
  line 2138); the `-fp8` dir's index is complete (282026 B). Used the `-fp8`
  dir.
- DSpark runtime initialized: stages=3 block=5 target_layers=[40,41,42],
  +2592 MB weights per GPU.

## Numbers

| Concurrency | Decode tok/s | Total tok/s | Δ vs plain |
|---:|---:|---:|---:|
| 1 | 72.4 | 76.9 | +37% |
| 8 | 176.2 | 187.6 | — |
| 16 | 242.2 | 257.9 | — |

Acceptance rate: 58.7% (1390 accepted / 2367 drafted, 574 chains).
Average accepted tokens per chain: 2.42 (block size 5).

## Rule

DSv4 verify cost is higher than Qwen3.6-27B (larger model, FP8 MoE), so the
speedup is +37% vs Qwen3.6's 2.9×. The acceptance rate (58.7%) is actually
higher than Qwen3.6's (~30%), but the per-verify cost eats the gain.
