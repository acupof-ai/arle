# A7 Qwen3.6 FP8 loader memory gate

## Context

Path A needs ARLE autograd-native Qwen3.6-35B-A3B FP8 LoRA training. A6 wired
the train-side Qwen3.6 MoE FP8 loader contract, but the real 35B CUDA
loader/memory gate was still open.

The load-bearing risk was constructor-time memory: checkpoint loading first
builds a `Qwen35Model` and then overwrites each tensor slot from safetensors. If
frozen base slots allocate full random f32 placeholders before the checkpoint
data is read, a 35B FP8 LoRA student can spend tens of GB of host memory on
values that are immediately discarded.

## What Worked

- Added an unmaterialized `Tensor` constructor for checkpoint loaders:
  metadata and shape exist, but no host buffer or device handle exists until
  the loader installs real checkpoint data.
- Added a guard so a `Dirty::Device` tensor with no handle fails loudly through
  the existing missing-handle path instead of silently uploading an empty host
  buffer.
- Kept public synthetic constructors unchanged. `Qwen35Model::new_for_eval` and
  `new_with_lora_targets` still materialize random frozen base weights, so the
  existing finite-diff tests keep their synthetic reference behavior.
- Switched only checkpoint loading to the unmaterialized frozen-base
  constructors. LoRA adapter tensors still allocate normally and remain
  trainable.
- Extended `qwen36_fp8_lora_load_gate` to report `live_host_mib`.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo test -p autograd --release --no-default-features \
  tensor::tests::unmaterialized_tensor_fails_loud_until_handle_is_installed -- --nocapture
PASS

cargo test -p train --release --no-default-features --features no-cuda --lib \
  qwen36_checkpoint_load_constructor_keeps_frozen_base_unmaterialized -- --nocapture
PASS

cargo test -p train --release --no-default-features --features no-cuda --lib \
  qwen36_moe_lora_gradient_matches_finite_difference -- --nocapture
PASS: rel_err=5.862e-3 on experts.0.up_proj.lora_b

cargo test -p train --release --no-default-features --features no-cuda --lib \
  qwen35_loader::tests:: -- --nocapture
PASS: 22 passed

cargo check -p train --release --no-default-features --features no-cuda --lib
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --lib -- -D warnings
PASS

cargo clippy -p autograd --release --no-default-features --lib -- -D warnings
PASS

cargo check -p train --release --no-default-features --features no-cuda \
  --example qwen36_fp8_lora_load_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release \
  --no-default-features --features cuda,no-cuda --lib
PASS
```

Remote `.62` real-checkpoint gate:

```text
Host: iv-ye8is8fbi8s6iplibbg7 / .62
GPU: H20 GPU7 via CUDA_VISIBLE_DEVICES=7, GPU3 avoided
Source: /data01/arle-patha-qwen36-fp8-hostless-20260617
Target: /data01/arle-target-patha-qwen36-fp8-hostless-20260617
Model: /data01/models/Qwen3.6-35B-A3B-FP8
Build env: CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
           ARLE_CUDA_DISABLE_FLASHMLA=1
           INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
Build: cargo build -p train --example qwen36_fp8_lora_load_gate --release
       --no-default-features --features cuda
Build result: PASS in 3m59.733s
```

Attention-qv target set:

```text
qwen36_fp8_lora_load_gate_result load_seconds=8.473214
total_vram_mib=97508.8 used_delta_mib=34080.0
free_before_mib=97117.6 free_after_mib=63037.6
live_host_mib=72.3
hidden=2048 layers=40 vocab=248320 experts=256 topk=8
moe_intermediate=512 shared_intermediate=512
all_param_tensors=31375 frozen_param_tensors=31335
trainable_param_tensors=40 trainable_elements=1024000 adapters=40
```

All-linear target set:

```text
qwen36_fp8_lora_load_gate_result load_seconds=14.039844
total_vram_mib=97508.8 used_delta_mib=34080.0
free_before_mib=97117.6 free_after_mib=63037.6
live_host_mib=2514.1
hidden=2048 layers=40 vocab=248320 experts=256 topk=8
moe_intermediate=512 shared_intermediate=512
all_param_tensors=93555 frozen_param_tensors=31335
trainable_param_tensors=62220 trainable_elements=641121600 adapters=62220
```

## Delta

| Gate | Before | After | Verdict |
|---|---:|---:|---|
| Constructor frozen-base host allocation | full f32 placeholder per frozen base slot | metadata-only slot until checkpoint load | fixed |
| 35B FP8 LoRA attention-qv load | not licensed | 8.47s, 34.1GiB VRAM delta, 72.3MiB live host | pass |
| 35B FP8 LoRA all-linear load | not licensed | 14.04s, 34.1GiB VRAM delta, 2.5GiB live host | pass |
| Existing Qwen3.6 MoE LoRA finite-diff | rel_err=5.862e-3 | rel_err=5.862e-3 | unchanged |

## Remaining Wall

This licenses the real 35B FP8 train-side loader/memory gate. It does not yet
license a full 35B OPD step. The next Path A gate is a real-checkpoint
quality/gradient gate: either a model-level finite-diff on a cheap adapter slice
or a needle/coherence gate after syncing the current LoRA into the FP8
`InferStudent`.

## Rule

For checkpoint loaders, do not allocate synthetic frozen base values that will
be overwritten by checkpoint data. Use metadata-only slots, and make accidental
pre-load execution fail loudly.
