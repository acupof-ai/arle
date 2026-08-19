# Marlin for per-channel FP8 — the only lever left on the NVFP4 concurrency gap

> Status: Analysed, entry point written, repack + parity not started

## Why

`Qwen3.8-27B-NVFP4` is a mixed checkpoint. 145 of ~200 quantised GEMMs per
forward are FP8 per-channel (all 48 linear-attn in/out_proj, all 16 self-attn
q/k/v/o, MLP on 8 of 64 layers, lm_head); only 56 layers' MLP is NVFP4. The
Marlin work landed on the NVFP4 minority; the FP8 majority is still on a scalar
batched GEMV.

That is the whole remaining asymmetry against Qwen3.6-27B-FP8:

| weight | Qwen3.6-27B-FP8 | Qwen3.8-27B-NVFP4 |
|---|---|---|
| MLP | 128x128 block -> DeepGEMM | FP4 -> Marlin |
| attention / linear-attn | 128x128 block -> DeepGEMM | per-channel -> **batched GEMV** |

Measured after the dequant-arm fix, aggregate tok/s, 1xH20:

| c | NVFP4 | FP8 | NVFP4 ITL | FP8 ITL |
|---:|---:|---:|---:|---:|
| 1 | 66.4 | 56.8 | 15.05 | 17.60 |
| 2 | 102.6 | 99.0 | 19.49 | 20.21 |
| 4 | 163.5 | 194.4 | 24.47 | 20.58 |
| 8 | 215.3 | 356.7 | 37.16 | 22.43 |
| 16 | 236.1 | 628.4 | 67.78 | 25.46 |

NVFP4's step cost grows 4.5x over 16x concurrency; FP8's grows 1.45x. The GEMV's
own comment names the ceiling: `TILE == B` means "register pressure == B", and
above B=8 it falls back to a fixed-8 tile with `grid.y = ceil(B/8)`, re-reading
the weight. Marlin is flat in M (68.9 us/call at both M=1 and M=3 on
34816x5120), so this moves c=1 and c>=2 together.

**The decode graph does not substitute for it.** That gate is symmetric — both
checkpoints lose the graph under `--kv-cache-dtype fp8` — so it moves absolute
numbers and not the ratio.

## What is already in the tree

- `scalar_type.hpp:308` — `kFE4M3fn`
- `dequant.h:321` — `dequant<nv_bfloat162, kFE4M3fn>` (bf16, both skip_flop arms)
- `gptq_marlin.cuh:416` — **`BIGGROUP_GET_IF(host::kFE4M3fn)` is already
  instantiated**, covering `group_blocks in [-1, 8]`. `-1` is channelwise
  (`group_size == -1` at `:598`), which is exactly per-channel.
- `gptq_marlin.cuh:577` — the weight-type check already admits `kFE4M3fn`.

So there is **no template growth and no extra nvcc cost** — only `marlin_gemm.cu`
recompiles. This is a smaller change than the `kFE2M1f` one that preceded it.

## What is missing

**1. `marlin_fp8_gemm_cuda` entry (written, parked out of `675d7f32a`).**
Mirrors `marlin_fp4_gemm_cuda` with `host::kFE4M3fn`, `num_groups = 1`,
`group_size = -1`, `s2 = nullptr`. Plus the `ARLE_DISABLE_MARLIN_SM70` stub and
the `ffi/gemm.rs` declaration.

**2. `repack_for_marlin_fp8` (not written).** Model on `repack_for_marlin_w8a16`
(`tensor.rs:2690`):
- Guard `weight_format == Fp8BlockScaled && quant_block_m == 1 && quant_block_k == cols`,
  sm >= 8, `K % 16 == 0`, `N % 64 == 0`.
- Weight: `qweight_u8` `[N, K]` raw E4M3 -> GPTQ `[K/4, N]` i32, element (n,k) at
  bit `(k%4)*8`. **No `+128`** — the W8A16 path adds it for `kU8B128`; the
  `kFE4M3fn` dequant reads the raw byte (`dequant.h:329` masks sign + shifts the
  exponent field, no bias subtraction).
- Reuse `marlin_gptq_repack_w8a16_cuda` — same 8-bit lane layout. **Unverified
  assumption**; the parity gate is what decides it.
- Keep `qweight_u8` / `scale_f32` resident: prefill above
  `QWEN_DEQUANT_GEMM_PREFILL_MIN_M` still reads them.

**3. The two places this will go wrong.**

*The 2^120 factor.* `marlin_template.h:328` sets `dequant_skip_flop = !is_int_type`,
and `is_int_type` covers only kU4/kU8/kU4B8/kU8B128 — so `kFE4M3fn` takes the
skip_flop arm and the bias multiply is skipped. That bias is
`BIAS = ((128-8) + 127) << 23` = `2^120`; the dequantised weight comes out as
`true_value * 2^-120`. `kFE4M3fn` has no `s2` channel (only `kFE2M1f` reads
`scale2_ptr`, `marlin_template.h:334`), so **`2^120` must be folded into the
per-channel BF16 scale**. Representable: FP8 weight scales are ~1e-3..1e-1, so
`scale * 2^120` lands near 2^113..2^117 against BF16's 2^128 ceiling. This is the
same class of bug that made the first NVFP4 Marlin wiring return `nonzero 0/256`.

*The scale permutation.* Channelwise uses vLLM's `scale_perm_single`, not the
length-64 `scale_perm` the W8A16 repack implements:

    for i in 0..4: for j in [0,1,8,9,16,17,24,25]: perm.push(2*i + j)

Reshape the `[1, N]` scales to `[-1, 32]`, permute columns, reshape back.
`N % 64 == 0` guarantees `N % 32 == 0`.

*Precision note.* The GEMV path reads f32 scales; Marlin reads BF16 (8 mantissa
bits, ~0.4% relative). The W8A16 path already accepts this, but the parity gate
should report the delta rather than assume it.

**4. Parity gate (not written).** Mirror
`crates/infer-cuda/examples/marlin_w8a16_parity.rs` (279 lines): f64 host
reference, Marlin lane, and the in-tree GEMV lane over the 27B shapes and an M
sweep. Its docstring already names the failure this catches — "a Marlin error
many× the fallback's error is the silent-wrong-repack / wrong-scale-perm signal
this gate exists to catch". Run it before any perf measurement.

**5. Routing.** An `Fp8Route` mirroring `Fp4Route`: Marlin below
`QWEN_FP4_MARLIN_MAX_M`, dequant+cuBLAS above, GEMV as the un-repacked fallback.
That also retires `QWEN_DEQUANT_GEMM_PREFILL_MIN_M` — with a Marlin arm claiming
M <= 1024, the dequant arm only ever sees prefill, same as every other format.

## Order

Parity first, on a standalone harness, before the serving path is touched. The
`kFE2M1f` work established why: three sessions of scalar optimisation were spent
before checking what the vendored kernel already supported, and the first wiring
returned all zeros for a scale-encoding reason no perf measurement would have
found.
