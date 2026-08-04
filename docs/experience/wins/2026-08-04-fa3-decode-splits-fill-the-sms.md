# FA3 decode split ceiling derived from the SM count — CUDA, 2026-08-04

> Status: Shipped

## Goal

Decode step time at 33K context, Qwen3.6-27B W8A16, one H20.

## Hypothesis

`--qwen35-fa3-decode-splits` is an upper bound on FA3's own split-KV choice:
`flash_prepare_scheduler.cu:156-158` computes `num_splits_dynamic` from
`total_blocks`, `num_head` and `num_sm`, then clamps it by the value passed in.
The shipped constant 8 is therefore a ceiling that binds whenever FA3 wants
more, which is exactly the low-batch long-context case. `pack_gqa` folds the 24
query heads into their 4 kv heads, so at batch 1 a decode layer has `4 × 8 = 32`
work tiles for 78 SMs.

## Parameters

```bash
# per arm: serve with ARLE_STEP_PHASE=1, 33K prompt x 300 tokens, x2 rounds,
# read the decode-only `submit` phase at steps=500
arle serve --model-path <w8a16-27b> --max-running-requests 16 \
  --qwen35-fa3-decode-splits {8,20,40,78,0}
```

- Baseline: `--qwen35-fa3-decode-splits 8` (the shipped constant)
- Treatment: explicit 20/40/78, then `0` = derive
- Prompt tokens: 33000; completion 300; concurrency 1, 4, 8
- Trials: 2 interleaved reps per arm

## Environment

- Host / GPU: 8×H20 pod, GPU 6 (78 SMs)
- Model / dtype: Qwen3.6-27B W8A16, 64 layers (48 linear + 16 full attention)
- `head_dim 256`, `num_attention_heads 24`, `num_key_value_heads 4`
- TP 1, paged KV 57061 pages @ page_size 16, whole-step decode graph on

## Results

Split ceiling at batch 1 — `submit` ms, two reps:

| splits | tiles (kv_heads × splits) | r1 | r2 | mean | Δ |
|---:|---:|---:|---:|---:|---:|
| 8 (baseline) | 32 | 19.005 | 18.973 | 18.989 | — |
| **20** | 80 | 16.871 | 16.847 | **16.859** | **−2.130 (−11.2%)** |
| 40 | 160 | 16.872 | 16.875 | 16.874 | −2.115 |
| 78 | 312 | 16.842 | 16.837 | 16.840 | −2.149 |

The three arms at and above 20 sit inside a 0.035 ms band — within single-arm
repeat spread. The gain arrives entirely at one tile per SM and nothing follows
it, which is what a ceiling that has stopped binding looks like.

Concurrency, shipped code (`0` = derive), two reps:

| c | splits 8 | derived | Δ |
|---:|---:|---:|---:|
| 1 | 18.970 | 16.845 | **−2.125 (−11.2%)** |
| 4 | 31.642 | 31.675 | +0.033 |
| 8 | 42.927 | 42.837 | −0.090 |

At c ≥ 4 the derived value *is* 8, so those two rows are the same configuration
run twice: they measure the noise floor, 0.09 ms at c=8 (within-arm spread
across reps reaches 0.115 ms). Read them as a control, not as an effect.

Shipped ceiling:

```rust
sm_count().div_ceil(batch * kv_heads).max(FA3_DECODE_SPLITS_FLOOR).clamp(2, 256)
```

20 at batch 1, 8 from batch 4 up. `--qwen35-fa3-decode-splits 0` is the new
default and means derive; an explicit value still clamps to [2, 256].

## Correctness

`lever_gate.sh`, needle ladder 115/300/2000/8000/16000/32000 × 3 runs, thinking
model flags (`RAW=1 TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=256`). Both arms:
`exact=3 partial=0 miss=0 DET` at every length, envelope comparison PASS. The
32000 rung runs at 33691 prompt tokens, the regime the ceiling changes.

## Kernel geometry behind the number

From the champion nsys trace, decode window:

```
cutlass::device_kernel<flash::FlashAttnFwdSm90<...>>
n=34784  avg=309.0us  grid=(78,1,1)  block=(384,1,1)  smem=232448  reg=168
```

`smem 232448` is the whole SM shared-memory budget, so occupancy is 1 CTA/SM by
construction; the grid is persistent at 78 and the split tiles are what feed it.

Roofline for the same shape: `4 kv_heads × 256 head_dim × 2 (K,V) × 2 B` =
4096 B per token per layer; 33K tokens × 16 full-attn layers = **2.16 GB per
decode step**; at H20's 3.5 TB/s achievable read that is a **618 µs** floor
against 5.001 ms measured before this change.

## Problems

`ncu` produced no data on this kernel across two attempts. The first filtered
`--kernel-name regex:FlashAttnFwdSm90` under the default
`--kernel-name-base function`, which strips template arguments from
`cutlass::device_kernel<...>` so nothing ever matched. The second, with
`--kernel-name-base demangled`, stalled indefinitely after the decode graph was
captured. Every number here was obtained without it.

## Learnings

PASS. −11.2% of the whole decode step at batch 1, unchanged at batch ≥ 4,
correctness gate green to 33K.

The kernel is still far from its 618 µs roofline. The next question is its DRAM
traffic: whether it reads the required 2.16 GB at low efficiency or reads
several times that. `ncu` with the decode graph off is the probe.

**Rule: a parallelism default belongs to the device, not to the source.** A
split, tile, or block count written as a constant is right only on the GPU it
was tuned on, at the batch and context length it was tuned at. Derive it from
`sm_count()` and the model's own head counts and the same code is right on the
next card and the next model.

**Rule: when a derived value collapses onto the old constant in some regime,
that regime is a free noise control.** Comparing the two arms there costs
nothing and calibrates every other delta in the same table — here it showed a
0.15 ms effect elsewhere was at the edge of noise rather than 3× above it.
