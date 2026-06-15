# CUDA Qwen3.5 Dense Routing Loader Fix

## Context

`Qwen/Qwen3.5-4B` Colab one-click eval built successfully on L4, resolved the
HF snapshot path, then failed at serve startup:

```text
engine build failed: load Qwen3 config from .../Qwen3.5-4B
```

The CUDA classifier advertised `qwen3_5` checkpoints as servable, but dense
Qwen3.5 configs did not contain MoE fields, so they were classified as
`Qwen3Dense` and routed into the vanilla `Qwen3Config` loader.

## What Worked

- Classified `model_type=qwen3_5` / `Qwen3_5*` architectures as `Qwen35`, not
  `Qwen3Dense`.
- Renamed the CUDA Qwen35 constructor path away from MoE-only naming.
- Removed the Qwen35 loader's `num_experts > 0` guard; dense layers already load
  through the existing dense-MLP branch.
- Generalized the contiguous full-attention prep/gate kernel from fixed HD256
  indexing to `head_dim` 128/256 so Qwen3.5-4B's HD128 full-attention layers do
  not silently use HD256 strides.
- Kept FA3 prefill gated to HD256 only; HD128 falls back to the in-tree
  nonpaged attention kernel, which already accepts 128/256.

## Verification

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo test -p infer-api --release \
  --no-default-features --features cuda,no-cuda classify_tests --lib
CUDARC_CUDA_VERSION=12060 cargo test -p infer-cuda --release \
  --no-default-features --features cuda,no-cuda qwen35::tests --lib
```

Result: all three passed locally.

## SLO / Bench Status

Pending GPU verification. This entry is a correctness/routing fix record, not a
performance license. The next gate is a Colab L4 CUDA build + `Qwen/Qwen3.5-4B`
serve startup + small MMLU eval through `scripts/arle_capability_eval.py`.

## Rule

CUDA model classification must route by the actual config family, not just by
expert fields. A config family advertised as servable must land on a loader whose
shape guards cover that family before eval can be trusted.
