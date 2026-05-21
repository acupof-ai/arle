# Qwen3.5-9B TQ4 Quantized LM Head KILL

## Context

The prior 9B-TQ4 OPD attempt failed the 16 GiB memory gate after the untied
dense `lm_head.weight` fix. The direct memory axis was to quantize
`lm_head.weight` as TurboQuant 4-bit, saving the largest remaining dense BF16
teacher tensor before retrying full-model logits parity and OPD.

Experiment target:

- Source checkpoint:
  `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B`
- Trial checkpoint:
  `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4-lmh`
- Quant command:
  `PYTHONHASHSEED=0 .venv/bin/python scripts/turboquant_weights.py --bits 4 --group-size 128 --quantize-lm-head ...`
- Gate 1: dense module `lm_head` parity `rmse/ref_rms <= 5%`
- Gate 2: full-model top-64 dominant relerr `<= 0.20`
- Gate 3: 100-step OPD no-OOM/no-NaN with monotonic held-out KL

## Evidence

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-tq4-lmh/
```

The quantizer produced the intended tensor layout:

| Tensor | Result |
| --- | --- |
| `lm_head.weight` | absent from TQ4-lmh index |
| `lm_head.tq_packed` | present |
| `lm_head.tq_scales` | present |
| `lm_head.tq_signs` | present |
| `turboquant_config.quantize_lm_head` | `true` |

On disk, the trial checkpoint dropped from the previous `11G` TQ4 checkpoint
to `9.0G`.

Dense module parity failed the first license gate:

| Module | RMSE/ref-RMS | Max Abs | Gate |
| --- | ---: | ---: | --- |
| embedding | `0.000000%` | `0.000000` | pass |
| final_rmsnorm | `0.000000%` | `0.000000` | pass |
| lm_head | `9.765882%` | `0.572266` | **fail** |

Tensor-local attribution on `lm_head` rows `0..256` separated quantization
error from decoder error:

| Comparison | RMSE/ref-RMS | Max Abs |
| --- | ---: | ---: |
| Python faithful dequant vs BF16 source | `9.653763%` | `0.009186` |
| ARLE CUDA dequant vs BF16 source | `9.655076%` | `0.008789` |
| ARLE CUDA dequant vs Python faithful | `0.166153%` | `0.000418` |

The attempted full fused-GEMV vs bulk-dequant cuBLAS harness on the complete
`lm_head` failed before producing metrics:

```text
Error: bulk dequant + cuBLAS GEMM failed

Caused by:
    DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")
```

That failure is secondary; the tensor-local result already shows ARLE CUDA
dequant matches the faithful format closely, while the 4-bit approximation of
the source `lm_head` itself is about `9.65%` RMSE/ref-RMS on sampled rows.

## Root Cause

Quantizing Qwen3.5-9B `lm_head.weight` to the current TurboQuant 4-bit format
is too lossy for the required OPD teacher parity gate. The loader/kernel path is
not the primary issue: sampled ARLE CUDA dequant matches faithful Python dequant
within `0.166%` RMSE/ref-RMS, but both are roughly `9.65%` away from the BF16
source for the lm-head slice.

The failed dense-module output (`9.77%`) is consistent with the tensor-local
quantization error. This means the memory-saving axis works structurally but
does not preserve enough output-projection fidelity at 4 bits.

## Fix

Killed at Gate 1. Do not run full-model logits parity, do not run the
9B-TQ4-lmh -> 0.8B OPD bench, and do not switch README/web/manual headline docs
to 9B-TQ4.

No product code was shipped for quantized `lm_head` support in this tranche; the
trial `--quantize-lm-head` script/loader edits were reverted before commit.

Next viable axes:

1. Try TurboQuant 8-bit for `lm_head.weight` only while keeping the rest TQ4.
   This should recover output projection fidelity while still saving roughly
   half of the dense lm-head memory.
2. If memory still fails after TQ8 lm-head, reduce OPD `rollout_len` from `8`
   to `4` as an explicit activation-footprint control.
3. Keep the 4B BF16 InferTeacher path as the current honest headline until a
   9B teacher passes both parity and OPD runtime gates.

## Rule

Do not quantize the output projection with the same acceptance envelope as
hidden projections. `lm_head` directly defines logits, so tensor-local
compression error propagates one-to-one into teacher distributions; it needs its
own parity gate before any memory win can be licensed.
