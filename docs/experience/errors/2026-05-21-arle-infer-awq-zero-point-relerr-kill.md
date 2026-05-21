# Qwen3.5-9B AWQ Zero-Point Loader KILL

## Context

The `tclf90/Qwen3.5-9B-AWQ` ModelScope checkpoint is the desired 9B teacher for
OPD on a 16 GB RTX 4070 Ti SUPER. The checkpoint uses AutoAWQ-style grouped
INT4 tensors (`qweight`, `qzeros`, `scales`) for MLP weights, while layer 0,
`self_attn`, `linear_attn`, `visual`, and `mtp` remain dense via
`modules_to_not_convert`.

The attempted ARLE implementation repacked those grouped tensors into the
existing W4 GEMV path and added zero-point subtraction in the CUDA unpack loop.
The loader then passed the load/smoke gate:

```text
target/release/infer --model-path /home/ckl/.cache/modelscope/hub/tclf90/Qwen3___5-9B-AWQ \
  --port 8123 --num-slots 1 --max-seq-len 128 --chunked-prefill-size 128 \
  --max-num-batched-tokens 128

Server listening on 0.0.0.0:8123
GPU memory @ post_model_load: free=6.34 GB / total=16.72 GB
```

One-token `/v1/completions` also succeeded:

```json
{"model":"Qwen3___5-9B-AWQ","choices":[{"text":"非得","finish_reason":"length"}]}
```

## Gate

The correctness gate was first-8-logit relative error against a PyTorch
AWQ baseline for fixed token prompt `[9419]` (`"Hello"`), threshold `<= 1e-3`.

PyTorch baseline setup:

- `transformers==5.8.0`
- `gptqmodel==7.0.0`
- AWQ backend forced to `gemm`
- `modules_to_not_convert` expanded to regex patterns so dense
  `linear_attn` / `self_attn` modules were not incorrectly converted
- `lm_head` cast to BF16 after load to match the checkpoint's dense BF16 path

Observed logits:

```text
ARLE first8:
[0.0341796875, 0.419921875, -0.4453125, -0.3359375,
 1.515625, -1.5546875, -1.390625, -0.45703125]

PyTorch/gptqmodel first8:
[12.875, 6.25, 2.015625, 3.0625,
 2.65625, 4.0625, 5.65625, 3.859375]
```

Max relative error:

```text
1.3826923076923077
```

## Root Cause

The load/smoke success was not enough. The zero-point AWQ layout was likely
misinterpreted during repack or dequantization. The leading hypotheses are:

1. AutoAWQ's qweight bit packing order differs from the attempted row-major
   nibble extraction.
2. AutoAWQ's `qzeros + 1` convention or qzeros group indexing was applied with
   the wrong axis.
3. Scales were stored as FP16 bit patterns in a BF16-typed device buffer and
   reinterpreted in the kernel; this may be correct byte-wise but is too fragile
   without a smaller per-layer dequant parity test.

No conclusion is licensed beyond "the attempted end-to-end path is numerically
wrong".

## Fix

Killed and reverted before committing runtime code. The next attempt needs a
small controlled dequant parity harness before touching full-model inference:

1. Pick one tensor, e.g. `layers.1.mlp.gate_proj`.
2. Dequantize a small `[rows x cols]` slice with PyTorch/gptqmodel.
3. Dequantize the same slice through ARLE's repack + CUDA path.
4. License the qweight/qzero/scale axis mapping with max relerr `<= 1e-3`.
5. Only then re-enable full-model load and logits parity.

## Rule

For quantized model loaders, "model loads and decodes one token" is only a
smoke gate. The first license gate must be tensor-local dequant parity, then
layer-local matmul parity, then full-model logits parity.
