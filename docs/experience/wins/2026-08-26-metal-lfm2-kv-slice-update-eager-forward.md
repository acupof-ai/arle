# LFM2 Metal decode: in-place KV write + eager forward — metal, 2026-08-26

> Status: Confirmed (local M4 Pro 48 GB)

## Context

LFM2.5-8B-A1B-MLX-4bit c=1 decode was 135 tok/s at 512 context and fell to
71 tok/s at 12k, while `mlx_lm` on the same machine held 134 → 118 tok/s.
The DSpark draft (`LiquidAI/LFM2.5-8B-A1B-DSpark`) made it worse (112 tok/s):
a verify block of 5 tokens reads 4.6× the expert bytes of a single token
(MoE, top-4 of 32), so the block costs 27 ms against 2.7 accepted tokens.

Root cause of the length scaling, from the Metal System Trace (GPU 86%
busy, 27 command buffers/step, 379 ops/step vs 216 for `mlx_lm`) and the
code: the LFM2 attention step grew the KV cache with `concatenate`, which
also left `kv_flat.shape[2] == cache_len`, so `ensure_kv_capacity` zero-padded
the cache to 2× and synchronously evaluated it on every step — about 900 MB
of traffic per token at 12k context.

## What worked

Adopt the Qwen3.5 C++ step pattern for LFM2 (`mlx_lfm2_model.cpp`):

- `slice_update` at `cache_pos` into the executor-reserved capacity; SDPA
  reads a `[0, cache_pos+S)` view. The cache never reallocates per step.
- One eager forward for prefill / decode / DSpark verify. The whole-graph
  shapeless `compile`, the S=5-specialised `compiled_moe`, the eager
  fallback FFI (`lfm2_eager_step_session`) and the `LFM2_EAGER_*` env knobs
  are deleted.
- conv `in_proj` runs as one `quantized_matmul` + `split` instead of three
  pre-split sub-weights (−34 kernels/step, measured −0.27 ms in isolation);
  the MoE router no longer `astype`s the router weight and expert bias every
  step (bias converted once at load).

Matched A/B, same binary tree, `/v1/completions`, `ignore_eos`, TPOT from
`(lat_400 − lat_1) / 399`, two repeats within 2%:

| context | before TPOT | after TPOT | decode tok/s |
|---|---|---|---|
| 512 | 7.41 ms | 6.58 ms | 135 → 152 |
| 4096 | 9.37 ms | 7.18 ms | 107 → 139 |
| 12288 | 14.09 ms | 8.92 ms | 71 → 112 |

`mlx_lm` reference (eval-per-step loop): 7.46 / 7.57 / 8.49 ms. TTFT at 12k
also dropped 10.4 s → 8.6 s. Needle gate (`NEEDLE_MAX_TOKENS=400`, lengths
512–12000, ×3): exact 3/3 at every length on both binaries, identical output
prefixes. DSpark verify path still runs (2.54 accepted/block) and remains a
loss on this model; leave it off for A1B.

Remaining per-step budget at 512 context (6.6 ms): 3.9 ms weight reads at
DRAM bandwidth (1.02 GB/token, 265 GB/s measured), 0.85 ms tied 6-bit
lm_head (128k vocab), ~1.5 ms in ~290 small ops at ~5 µs each. The next
lever is op count (router + top-k + weighted sum as one `fast::metal_kernel`,
residual-add + rms_norm fusion), not bandwidth.

## Rule

- On a MoE model with a small active set, block speculation is a loss by
  construction: verify cost scales with the number of distinct experts the
  block touches, so the weight-read amortisation that makes speculation pay
  on dense models does not exist. Measure `S=1` vs `S=block` layer cost
  before wiring a drafter.
- `bench_local_metal.py` derives TPOT from `N=64`; at ≥4k prompts the
  prefill jitter divided by 63 is larger than the effect under test. Use
  `N≥400` or a streaming ITL for long-context decode claims.
- A KV cache whose seq axis equals `cache_len` after every step defeats any
  capacity reservation above it; check `shape[2]` against `cache_len` when a
  Metal path shows O(n) per-step cost.
