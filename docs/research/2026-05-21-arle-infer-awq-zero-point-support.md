# ARLE Infer AWQ Zero-Point Support

## Context

The Qwen3.5 9B -> 0.8B distillation plan originally preferred the ModelScope
teacher `tclf90/Qwen3.5-9B-AWQ` so the teacher could fit on a 16 GB RTX 4070
Ti SUPER. Phase 1 confirmed the checkpoint downloads and exposes AWQ metadata:

```text
quant_method=awq
bits=4
group_size=128
version=gemm
zero_point=true
```

The checkpoint is not currently runnable through ARLE infer. The demo path has
pivoted to `Qwen/Qwen3.5-4B` as the teacher so the 4B -> 0.8B OPD comparison can
continue on the same 16 GB hardware.

## Root Cause

The community AWQ checkpoint stores quantized later-layer linear weights as
separate grouped tensors:

```text
*.qweight
*.qzeros
*.scales
```

Example from `tclf90/Qwen3.5-9B-AWQ`:

```text
model.language_model.layers.1.mlp.gate_proj.qweight [4096, 1536] int32
model.language_model.layers.1.mlp.gate_proj.qzeros  [32, 1536] int32
model.language_model.layers.1.mlp.gate_proj.scales  [32, 12288] float16
```

ARLE's current CUDA Qwen3.5 infer loader still looks for a dense
`*.weight` tensor at that point in the model load, so it fails before it can
dispatch an AWQ kernel:

```text
Tensor 'model.language_model.layers.1.mlp.gate_proj.weight' not found in any shard
```

The lower-level quant metadata parser already identifies zero-point AWQ as an
unsupported layout. `infer/src/weight_loader.rs` records the current boundary:

```text
zero-point AWQ qzeros are not supported by the current CUDA W4 loader
```

So the blocker is not model availability. It is missing end-to-end support for
AutoAWQ-style zero-point grouped tensors in the infer weight loader and CUDA W4
decode path.

## What Is Needed

This should be a separate kernel/runtime axis, not part of the OPD distillation
critical path.

Work items:

1. Extend the Qwen3.5 infer weight loader to recognize the grouped AWQ tensor
   triplet for each linear weight: `qweight`, `qzeros`, and `scales`.
2. Define the exact in-memory packed representation passed to CUDA. The current
   W4 path cannot silently reinterpret zero-point groups as symmetric weights.
3. Extend the relevant `crates/cuda-kernels/csrc/quant/` kernels to apply the
   zero-point correction while unpacking W4 groups.
4. Add a real-checkpoint loader smoke for `tclf90/Qwen3.5-9B-AWQ`.
5. Add a 1-token infer smoke on the 16 GB 4070 Ti SUPER and record nvidia-smi
   before/during/after snapshots.

## Layout Cross-Reference

The canonical AutoAWQ family uses grouped W4 weights plus per-group scales and,
when `zero_point=true`, qzeros. The `tclf90/Qwen3.5-9B-AWQ` checkpoint follows
that broad contract, but it is a community quant with mixed dense exceptions:
layer 0, `self_attn`, `linear_attn`, `visual`, and `mtp` are excluded from
quantization per `modules_to_not_convert`.

That mixture is why ARLE gets past early dense tensors and then fails at
`layers.1.mlp.gate_proj`: the loader needs per-tensor dispatch based on actual
checkpoint tensor names, not a single dense-or-quantized assumption for the
whole model.

## Revisit Trigger

Revisit after the 4B -> 0.8B OPD demo is complete. The acceptance gate for this
axis should be:

- `tclf90/Qwen3.5-9B-AWQ` loads through `arle serve --backend cuda`
- one-token `/v1/completions` succeeds on the RTX 4070 Ti SUPER
- peak GPU memory leaves enough headroom for the 0.8B LoRA student in the OPD
  process
- output is numerically sanity-checked against a PyTorch/AutoAWQ decode for a
  fixed short prompt

Until then, the bench-honest README phrasing is:

> ARLE OPD at Qwen3.5-4B -> Qwen3.5-0.8B runs on 16 GB. Qwen3.5-9B teacher is
> pending zero-point AWQ support in ARLE infer.
