# The OPD recompute chunk belongs to the rank's sequence, not a constant — 2026-08-22

> Status: Shipped

`ThinkingCap-Qwen3.6-27B-FP8`, H20 (97,508 MiB), `--synthetic-writeback-seq`,
binary `sp21` (`fb5795599`).

## Context

With the sequence-parallel core in place, a 131,072 cp=2 backward profile put
63 % of the 383 s in `SeqChunkedRecompute` — the position-wise projections
replayed chunk by chunk — against 6.6 % in the linear-attention core. The chunk
was a 4096 constant, so a rank holding 65,536 rows paid 16 slice + alloc + tape
setups per projection, and one holding 131,072 rows paid 32.

## What worked

The chunk derives from the rank's sequence: `chunk * rank_seq <= 2^30`, capped
at 16,384 (nothing above it is measured), floored at 4,096. `--opd-seq-chunk`
overrides with a fixed value.

Two forces set the bound. Larger chunks cut a per-chunk overhead that is
independent of chunk size, so the saving is roughly proportional to the chunk
count removed. Against that, the replay transient grows with the chunk while
headroom shrinks as the rank's sequence grows — at rank 131,072 the card holds
~12 GB free and a 16,384-row replay needs more than the 37 GB free at the layer
scope where it died.

## Result

Global 262,144, one step:

| cp | rank seq | chunk | forward | backward | step | peak | loss |
|---|---|---|---|---|---|---|---|
| 4 | 65,536 | 4,096 | 153.9 s | 539.0 s | 692.9 s | 65.1 GB | 1.560897 |
| 4 | 65,536 | 16,384 | 155.4 s | 463.0 s | **618.4 s** | 64.8 GB | 1.560237 |
| 2 | 131,072 | 4,096 | 279.9 s | 1,082.9 s | 1,362.8 s | 85.7 GB | 1.561557 |
| 2 | 131,072 | 8,192 | 269.6 s | 893.8 s | **1,163.4 s** | 86.1 GB | 1.561359 |
| 2 | 131,072 | 16,384 | 269.1 s | OOM at layer 62 | — | — | — |

Global 131,072 cp=2 (rank 65,536), same build, four arms:

| arm | forward | backward | step | loss |
|---|---|---|---|---|
| chunk 4,096 | 127.0 s | 390.5 s | 517.5 s | 3.034898 |
| chunk 16,384 | 122.6 s | 320.2 s | **442.8 s** | 3.035551 |
| `--fp8-native-gemm` | 130.5 s | 358.5 s | 489.0 s | 3.041999 |
| both | 111.9 s | 303.2 s | 415.1 s | 3.040560 |

−11 % to −15 % on the step, peak flat, losses within the MoE non-determinism
envelope (0.01–0.04 %).

## Also measured

- **Frozen-weight dequant cache: no effect.** `matmul_bt` backward dequantized
  the whole FP8 weight per call; caching it by source buffer left the backward
  at 390.5 s vs 391.3 s. The 5.1 %-of-step dequant recorded on 2026-08-06 does
  not survive FlashQLA + chunked projections. Kept — it is correct and its cost
  is one ≤180 MiB buffer — but it is not a speed-up.
- **`--fp8-native-gemm` stays opt-in.** −5.5 % alone and −6 % on top of the
  chunk change, but the loss moves 0.23 %, six to ten times the envelope the
  other arms sit in, and its forward is 2.7 % slower than the bf16 baseline
  (cause unknown). It reaches the backward through the checkpoint replay's
  recompute, so the 2026-08-06 forward-only parity result does not cover it.
  A default flip needs the needle + lever gates first.

## Rule

- A recompute chunk is a memory-for-overhead trade, and the memory side scales
  with the rank's sequence. Size it from that, not from a constant.
- A saving measured at one rank sequence does not transfer to a longer one on
  the same card: the same 16,384-row chunk that ran with the peak flat at rank
  65,536 OOM'd at rank 131,072.
