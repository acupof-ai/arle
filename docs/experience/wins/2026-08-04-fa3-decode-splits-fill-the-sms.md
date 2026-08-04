# FA3 decode splits derived from the SM count — CUDA, 2026-08-04

> Status: Shipped

## Goal

Decode step time at 33K context, c=1, Qwen3.6-27B W8A16 on one H20.

## Hypothesis

A decode layer's only work tiles are `kv_heads × num_splits`: `pack_gqa` folds
the 24 query heads into their 4 kv heads, and split-KV is the sole remaining
parallel axis. The shipped default `--qwen35-fa3-decode-splits 8` therefore
launches 32 tiles onto 78 SMs. Raising the split count until tiles ≥ SMs should
cut FA3 decode time; raising it further should do nothing.

## Parameters

```bash
# per arm: serve with ARLE_STEP_PHASE=1, one 33K prompt x 300 tokens x2,
# read the decode-only `submit` phase at steps=500
arle serve --model-path <w8a16-27b> --max-running-requests 4 \
  --qwen35-fa3-decode-splits {8,20,40,78}
```

- Baseline: `--qwen35-fa3-decode-splits 8` (the shipped default)
- Treatment: 20, 40, 78 — same binary, one flag
- Prompt tokens: 33000; completion 300
- Metric: `ARLE_STEP_PHASE` `submit`, decode-only steps, in-process host timing

## Environment

- Host / GPU: 8×H20 pod, GPU 6 (78 SMs)
- Model / dtype: Qwen3.6-27B W8A16, 64 layers (48 linear + 16 full attention)
- `head_dim 256`, `num_attention_heads 24`, `num_key_value_heads 4`
- TP 1, 4 slots, paged KV 57061 pages @ page_size 16, whole-step decode graph on

## Results

| splits | tiles (4 kv-heads × splits) | submit ms | delta |
|---:|---:|---:|---:|
| 8 (baseline) | 32 | 19.005 | — |
| 20 | 80 | 16.871 | **−2.134 (−11.2%)** |
| 40 | 160 | 16.872 | −2.133 |

The curve flattens the moment tiles exceed the SM count and stays flat at 2×.
`78/4 = 20` is the knee, so the shipped value is derived, not tuned:

```rust
self.ctx.sm_count().div_ceil(self.local_kv_heads.max(1)).clamp(2, 256)
```

`--qwen35-fa3-decode-splits 0` is the new default and means "derive"; an
explicit value still clamps to [2, 256].

## Supporting measurement

Kernel geometry from the champion nsys trace, decode window:

```
cutlass::device_kernel<flash::FlashAttnFwdSm90<...>>
n=34784  avg=309.0us  grid=(78,1,1)  block=(384,1,1)  smem=232448  reg=168
```

`smem 232448` is the whole SM shared-memory budget, so occupancy is 1 CTA/SM by
construction — the grid is persistent at 78 and the tiles are what feed it.

Roofline for the same shape: `4 kv-heads × 256 head_dim × 2 (K,V) × 2 B` =
4096 B per token per layer; 33K tokens × 16 full-attn layers = **2.16 GB per
decode step**; at H20's 3.5 TB/s achievable read that is a **618 µs** floor
against a measured 5.001 ms.

## Problems

`ncu` on this kernel did not produce data. Two attempts: the first filtered on
`--kernel-name regex:FlashAttnFwdSm90` under the default
`--kernel-name-base function`, which strips template arguments from
`cutlass::device_kernel<...>` so nothing ever matched; the second, with
`--kernel-name-base demangled`, stalled indefinitely after the decode graph was
captured. The split-count result was obtained without it.

## Learnings

PASS, −11.2% of the whole decode step from one derived default.

The remaining wall is the same kernel: even filled, FA3 decode attention is far
from the 618 µs roofline. Next probe is its DRAM traffic under `ncu` with the
decode graph off, to separate "reads the right 2.16 GB at low efficiency" from
"reads several times that".

**Rule: a parallelism default belongs to the device, not to the source.** A
split/tile/block count that is written as a constant is correct only on the GPU
it was tuned on and only at the context length it was tuned at. Derive it from
`sm_count()` and the model's own head counts, and the same code is right on the
next card and the next model.
