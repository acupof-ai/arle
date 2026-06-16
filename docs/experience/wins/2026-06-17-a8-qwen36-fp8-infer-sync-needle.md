# A8 Qwen3.6 FP8 Infer-Student Sync Needle Gate

## Context

Path A needs ARLE autograd-native Qwen3.6-35B-A3B FP8 LoRA training for OPD
35B. A7 licensed the real FP8 checkpoint loader/memory gate, but left the next
runtime-quality wall open: after loading the train-side FP8 LoRA student, can
the current LoRA tensors sync into the FP8 inference engine and still decode
coherent text through the KV-cache path?

This tranche is a real-checkpoint needle gate. It is not a model-level
finite-diff gradient gate.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU7 via `CUDA_VISIBLE_DEVICES=7`; GPU3 avoided.
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Binary:
  `/data01/arle-target-track1-opd-rollout-infer-202606170646/release/examples/qwen36_fp8_lora_load_gate`.
- Source: train files matching the code commit that added the rollout smoke
  gate (`71ea0c70`); current `main` only added docs after that.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.

## Command

```bash
MODEL=/data01/models/Qwen3.6-35B-A3B-FP8
BIN=/data01/arle-target-track1-opd-rollout-infer-202606170646/release/examples/qwen36_fp8_lora_load_gate

CUDA_VISIBLE_DEVICES=7 "$BIN" \
  --model "$MODEL" \
  --infer-model "$MODEL" \
  --device 0 \
  --target-set attention-qv \
  --rank 8 \
  --alpha 16 \
  --sync-infer \
  --perturb-adapter none \
  --rollout-smoke-prompt "Context: The secret code is BLUE-73-MANGO. Question: What is 2 + 3? Also repeat the secret code exactly. Answer:" \
  --rollout-smoke-tokens 96 \
  --expect-substring BLUE-73-MANGO
```

The all-linear run used the same command with `--target-set all-linear`.

## Results

Attention-qv target set:

```text
qwen36_fp8_lora_load_gate_result load_seconds=8.186200
used_delta_mib=34080.0 live_host_mib=72.3
trainable_param_tensors=40 trainable_elements=1024000 adapters=40
qwen36_fp8_lora_sync_gate_result infer_load_seconds=13.369560 sync_seconds=0.002563 perturb_adapter=none
qwen36_fp8_lora_rollout_smoke_result prompt_tokens=32 generated_tokens=96 smoke_seconds=1.041092 expect=Some("BLUE-73-MANGO") contains_expect=true
```

All-linear target set:

```text
qwen36_fp8_lora_load_gate_result load_seconds=13.549135
used_delta_mib=34080.0 live_host_mib=2514.1
trainable_param_tensors=62220 trainable_elements=641121600 adapters=62220
qwen36_fp8_lora_sync_gate_result infer_load_seconds=12.820288 sync_seconds=1.587442 perturb_adapter=none
qwen36_fp8_lora_rollout_smoke_result prompt_tokens=32 generated_tokens=96 smoke_seconds=1.024405 expect=Some("BLUE-73-MANGO") contains_expect=true
```

Both generated outputs repeated the prompt context coherently and contained the
exact `BLUE-73-MANGO` needle.

## Delta

| Gate | Before | After | Verdict |
|---|---:|---:|---|
| 35B FP8 train-side load | A7 licensed load/memory only | unchanged | pass |
| 35B FP8 LoRA sync into infer engine, attention-qv | not measured | 2.6ms | pass |
| 35B FP8 LoRA sync into infer engine, all-linear | not measured | 1.59s for 62,220 adapters | pass |
| 35B FP8 post-sync KV decode quality | not measured | exact needle + coherent output | pass |

## Remaining Wall

The next Path A gate is still gradient evidence on the real checkpoint: a
cheap model-level finite-diff on a selected Qwen3.6 FP8 adapter tensor, then a
bounded OPD step using the 35B FP8 student path. This entry only licenses
train-to-infer sync reachability and post-sync generation quality.

## Rule

After a checkpoint loader/memory gate, validate the actual train-to-infer LoRA
sync path with a real checkpoint and decoded tokens. A successful load is not
evidence that the inference engine can consume the live adapter state.
