# Qwen3.5-9B TurboQuant Logits Parity KILL

## Context

Path B is the licensed architecture for cross-size OPD: the teacher runs
through `infer`, then `InferTeacher` bridges BF16 logits into train/autograd.
The Qwen3.5-4B BF16 teacher bench already passed. This axis tried to scale the
teacher to Qwen3.5-9B by quantizing the BF16 ModelScope checkpoint to ARLE's
native TurboQuant 4-bit format.

The target setup was:

- Teacher source: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B`
- Teacher quantized output: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Student target, if parity passed: Qwen3.5-0.8B-Base LoRA rank 16
- Gate before OPD bench: dominant-logit top-64 relative error `<= 5e-2`
  against PyTorch BF16 9B for fixed token prompt `[9419]`

## Evidence

Quantization completed:

```bash
PYTHONHASHSEED=0 .venv/bin/python scripts/turboquant_weights.py \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --output-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --bits 4 \
  --group-size 128
```

The script reported compressed linear tensors:

```text
Total: 11.02 GB -> 2.84 GB (3.9x compression)
Config: /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4/turboquant_config.json
```

The on-disk model directory is still 11 GB because non-linear tensors, dense
fallback tensors, embeddings, and the LM head remain unquantized.

`arle serve` smoke passed with one-token decode:

```json
{"model":"Qwen3___5-9B-TQ4","choices":[{"text":"缝隙","finish_reason":"length"}]}
```

The runtime loaded TurboQuant weights and had enough GPU memory:

```text
Loaded TurboQuant ... packed 4-bit on GPU, group_size=128
GPU memory @ post_model_load: free=7.32 GB / total=16.72 GB
```

Raw-logits dump through `LoadedInferenceEngine::forward_token_logits` also
ran:

```text
raw_logits_dump ... seq_len=1 vocab_size=248320
load_seconds=111.247646 forward_seconds=2.857958 host_readback_seconds=0.059612
```

The parity gate failed against PyTorch BF16:

```text
top64_max_abs: 14.546875
top64_mean_abs: 10.775253
top64_max_rel: 1.3515625
top64_mean_rel: 1.0082824
top64_ref_mean_abs: 10.701172
gate threshold: 0.05
gate_pass: false
```

Representative top-dominant logits:

```text
rank token  torch_bf16   arle_tq4    rel_err
0    175169 -15.125000   -1.937500   0.871901
1    11      14.250000    0.515625   0.963816
2    53983  -14.187500   -0.351562   0.975220
3    0       13.312500   -1.148438   1.086268
4    4858    12.250000    1.828125   0.850765
```

Raw artifacts:

- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/quantize.txt`
- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/serve-smoke.log`
- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/serve-smoke-response.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/arle-tq4-logits.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/parity-summary.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/parity-top64.csv`
- `bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/nvidia-smi-after-parity.txt`

## Root Cause

The 9B-TQ4 model is runnable but not numerically licensed as a teacher. The
dominant logits are compressed toward small values and often change sign versus
the BF16 baseline, so the quantized teacher distribution is not a faithful
replacement for the BF16 teacher.

The precise source of drift is not licensed by this full-model test. Leading
hypotheses are:

1. TurboQuant's Hadamard/sign dequant path does not match the offline packing
   semantics for this Qwen3.5-9B layout.
2. Group-scale or inverse-rotation application is wrong for one or more
   projection shapes.
3. The current generic TurboQuant recipe is too lossy for Qwen3.5-9B teacher
   logits at 4-bit without a calibration or layer-local correction pass.

No OPD bench was run. Running 100 steps with this teacher would only measure a
wrong target distribution.

## Fix

Killed before the 9B -> 0.8B OPD bench and before README updates.

Next attempt should not start with a full OPD run. It needs the same staged
parity ladder as the AWQ zero-point kill:

1. Tensor-local dequant parity for one quantized projection against the offline
   Python reconstruction.
2. Layer-local matmul parity for a fixed hidden vector.
3. Layer-0 forward parity.
4. Full-model raw-logits parity with top-64 dominant relerr `<= 5e-2`.
5. Only then rerun the 9B-TQ4 InferTeacher OPD bench.

## Rule

For quantized teachers, serving smoke is only a liveness gate. OPD teacher
eligibility requires downstream logits parity before any training benchmark or
README headline.
