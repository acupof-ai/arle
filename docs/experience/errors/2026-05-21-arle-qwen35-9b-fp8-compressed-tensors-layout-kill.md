# Qwen3.5-9B FP8 Compressed-Tensors Loader Gate KILL

## Context

After the GPTQ-INT4 loader-layout kill, we tried the FP8 axis because it should
avoid the W4 physical-layout path entirely. ModelScope search found one 9B FP8
candidate:

```text
RedHatAI/Qwen3.5-9B-FP8-dynamic
/home/ckl/.cache/modelscope/hub/RedHatAI/Qwen3___5-9B-FP8-dynamic
```

Raw inspection artifacts:

```text
bench-output/2026-05-21-qwen35-9b-fp8-dynamic-layout/
```

Only config/tokenizer/index files were downloaded. The full 9 GB weight file
was not pulled because the config/index gate already failed the current ARLE
loader contract.

## Command

ModelScope search:

```bash
curl -sS -X PUT "https://modelscope.cn/api/v1/dolphin/models" \
  -H "Content-Type: application/json" \
  -d '{"PageSize":20,"PageNumber":1,"Name":"Qwen3.5-9B-FP8"}'
```

Metadata download:

```bash
.venv/bin/python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download(
    'RedHatAI/Qwen3.5-9B-FP8-dynamic',
    cache_dir='/home/ckl/.cache/modelscope/hub',
    allow_patterns=['*.json', '*.txt', 'tokenizer*'],
))
PY
```

## Result

KILL at config/layout gate. No `arle serve`, generation smoke, or OPD bench was
run on this candidate.

The checkpoint is compressed-tensors FP8:

```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "weights": {
    "num_bits": 8,
    "strategy": "channel",
    "symmetric": true,
    "type": "float"
  },
  "input_activations": {
    "num_bits": 8,
    "strategy": "token",
    "dynamic": true,
    "symmetric": true,
    "type": "float"
  }
}
```

The safetensors index uses `.weight_scale`, not `.weight_scale_inv`:

```text
weight_scale_inv 0
weight_scale 128
  model.language_model.layers.0.mlp.down_proj.weight_scale
  model.language_model.layers.0.mlp.gate_proj.weight_scale
  model.language_model.layers.0.mlp.up_proj.weight_scale
```

The current ARLE FP8 loader recognizes ModelOpt-style FP8 metadata and side
tensors:

```text
infer/src/quant.rs:482: "fp8" | "float8" | "fp8_e4m3"
infer/src/weight_loader.rs:813: name.replace(".weight", ".weight_scale_inv")
```

It does not parse `quant_method=compressed-tensors`, and its FP8 load path looks
for `.weight_scale_inv`. Downloading the full weight file would only defer the
same failure to serve time.

## Root Cause

The earlier statement "ARLE FP8 is production-tested" was true for the existing
ModelOpt-style FP8 loader, not for compressed-tensors W8A8 layout.

This RedHatAI checkpoint uses a different mature ecosystem format:

- `quant_method=compressed-tensors`
- `format=float-quantized`
- per-channel `.weight_scale`
- dynamic token activation scales

ARLE's current FP8 implementation is:

- `quant_method=fp8` / `quant_type=fp8`
- E4M3 weights
- block `weight_scale_inv`
- load-time dequantization to BF16 host/device matrix

Those are not the same physical format.

## Fix

Do not run the 9B FP8 generation smoke, OPD bench, or headline switch on this
candidate.

The next viable FP8 axis is a real compressed-tensors loader tranche:

1. Extend quant metadata parsing for `quant_method=compressed-tensors` and
   `format=float-quantized`.
2. Add tensor-local tests for `.weight` + `.weight_scale` dequantization.
3. Decide whether to dequantize to BF16 at load time, or keep W8A8 device
   kernels for dynamic token activation scales.
4. Run a one-layer parity gate before full-model serve.

## Rule

"FP8" is not one loader format. Gate 0 for quantized checkpoints must compare
the config and side-tensor names against the actual ARLE loader contract before
downloading multi-GB weights or starting serve-quality tests.
