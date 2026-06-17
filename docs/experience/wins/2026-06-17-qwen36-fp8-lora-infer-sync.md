# Qwen3.6 FP8 LoRA sync into infer student

## Context

Track 1 OPD rollout generation is already routed through `InferStudent` on
current `main`: each OPD step syncs the live LoRA adapter into the inference
engine and samples rollout tokens through KV-cache decode; teacher scoring and
student KL/backward remain autograd.

The remaining FP8 gap was in the sync path itself. `Qwen35Model::remerge_student_lora`
only snapshotted dense BF16 resident weights, so a non-zero LoRA update against a
Qwen3.6 FP8 infer student could fail or update the wrong resident buffer. The
earlier A8 smoke used zero LoRA adapters and was reachability evidence only, not
non-zero train-to-infer sync evidence.

## What Changed

`crates/infer-cuda/src/qwen35.rs` now supports FP8 block-scaled resident
weights in the LoRA re-merge path:

- Snapshot dense BF16 or FP8 block-scaled base weights on first non-zero touch.
- For FP8, cache the original qweight bytes and f32 128x128 block scales,
  dequantize a BF16 host base for `base + scale * B * A`, then requantize back
  to FP8 in-place.
- Preserve resident storage and device addresses, so pointer tables and graph
  captures keep seeing the same allocation.
- Add a target-aware path for Qwen3.6 routed experts: per-expert matrices use
  the standalone `DeviceMatrix`; FP8 DeepGEMM grouped caches address the
  `[group, row_offset..row_offset+rows, cols]` qweight/scale slices directly.

## Verification

Local gates:

```text
cargo fmt --check
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_load_gate
PASS

cargo clippy -p infer-cuda --release --no-default-features --features no-cuda --lib -- -D warnings
PASS

CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib fp8_block_scaled_lora_quant_roundtrip_is_bounded -- --nocapture
PASS: 1 passed
```

Remote `.62`, GPU1, non-native/per-expert path:

```text
model=/data01/models/Qwen3.6-35B-A3B-FP8
binary=/data01/arle-target-track1-opd-rollout-infer-202606170646/release/examples/qwen36_fp8_lora_load_gate
CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1
ARLE_QWEN35_DEEPGEMM=0
```

Attention-qv non-zero sync:

```text
target_set=attention-qv
perturb_adapter=model.language_model.layers.3.self_attn.q_proj.weight.lora_b
load_seconds=8.206049
infer_load_seconds=13.014242
sync_seconds=1.232496
smoke_seconds=1.040884
contains_expect=true
needle=BLUE-73-MANGO
```

All-linear non-zero routed expert sync:

```text
target_set=all-linear
perturb_adapter=model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b
adapters=62220
trainable_elements=641121600
load_seconds=12.896774
infer_load_seconds=13.789293
sync_seconds=1.624925
smoke_seconds=1.031849
contains_expect=true
needle=BLUE-73-MANGO
```

Native DeepGEMM grouped-cache runtime gate was attempted with a binary built
using `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`. The binary built, but `.62` only has
gcc/g++ 8.3, and DeepGEMM JIT failed at runtime because nvcc ignored
`-std=c++20` with that host compiler, then CUTLASS/CUTE C++17 traits were not
available:

```text
DeepGEMM native contiguous bridge failed: NVCC DeepGEMM compile failed
nvcc warning : The -std=c++20 flag is not supported with the configured host compiler. Flag will be ignored.
... cute/util/type_traits.hpp: namespace "std" has no member "conjunction"
```

So grouped-cache sync is typechecked and code-covered, but the native runtime
gate is blocked by the existing DeepGEMM JIT host-compiler environment on `.62`;
it is not claimed as a runtime PASS here.

## Rule

Zero LoRA sync is not evidence for OPD train-to-infer correctness. Perturb a
live adapter and decode the actual generation. For FP8 Qwen3.6, also separate
the per-expert hand path from the DeepGEMM grouped-cache path; the latter needs
a working native JIT environment before it can be marked runtime-verified.
