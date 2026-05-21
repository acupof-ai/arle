# ARLE Qwen3.5-9B-TQ4 Dense Tensor And Dense Module Parity

## Goal

After projection-local TurboQuant GEMV checks passed, this tranche tests the
remaining dense-path suspects for Qwen3.5-9B-TQ4:

1. Did `scripts/turboquant_weights.py` corrupt any dense BF16 tensors while
   copying them into the TQ4 checkpoint?
2. If dense tensors are clean, do ARLE infer's embedding, final RMSNorm, and
   LM-head/output-projection module outputs match a PyTorch BF16 reference on
   the same inputs?

No OPD bench or headline switch is licensed by this tranche.

## Params

- Source BF16 checkpoint: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B`
- ARLE TQ4 checkpoint: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Token id for embedding: `9419`
- Dense tensor gate: every non-`*.tq_packed` / `*.tq_scales` / `*.tq_signs`
  tensor in the TQ4 checkpoint must be bit-identical to the source tensor with
  the same name.
- Dense module gate: `RMSE/reference-RMS <= 1%` per module.
- CUDA env:
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`,
  `CARGO_BUILD_JOBS=1`

Command:

```bash
.venv/bin/python scripts/qwen35_tq4_dense_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tq-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --token-id 9419 \
  --output-dir bench-output/2026-05-21-qwen35-9b-tq4-dense-parity
```

Raw artifacts:

- `bench-output/2026-05-21-qwen35-9b-tq4-dense-parity/run.txt`
- `bench-output/2026-05-21-qwen35-9b-tq4-dense-parity/dense-tensor-bitcompare.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-dense-parity/dense-module-parity.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-dense-parity/summary.json`

## Results

Dense tensor bit-compare:

| Metric | Value |
| --- | ---: |
| Dense tensors scanned | `640` |
| Non-identical dense tensors | `0` |
| Gate | PASS |

Critical dense tensors are bit-identical:

| Tensor | Shape | Dtype | Bit-identical |
| --- | ---: | --- | :---: |
| `model.language_model.embed_tokens.weight` | `248320x4096` | `torch.bfloat16` | PASS |
| `model.language_model.norm.weight` | `4096` | `torch.bfloat16` | PASS |
| `lm_head.weight` | `248320x4096` | `torch.bfloat16` | PASS |

Dense module parity:

| Module | Len | Max abs | RMSE/reference-RMS | Gate |
| --- | ---: | ---: | ---: | :---: |
| embedding | `4096` | `0.0` | `0.0%` | PASS |
| final RMSNorm | `4096` | `0.0` | `0.0%` | PASS |
| LM head / output projection | `248320` | `7.984375` | `130.502%` | FAIL |

First 8 LM-head outputs:

| Index | ARLE | PyTorch | Abs err | Rel err |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `1.2421875` | `-0.380859375` | `1.623046875` | `4.2615385` |
| 1 | `1.2578125` | `0.98828125` | `0.26953125` | `0.2727273` |
| 2 | `-0.310546875` | `1.2265625` | `1.537109375` | `1.2531847` |
| 3 | `1.2890625` | `1.0078125` | `0.28125` | `0.2790698` |
| 4 | `2.21875` | `-0.76171875` | `2.98046875` | `3.9128206` |
| 5 | `0.90234375` | `0.3671875` | `0.53515625` | `1.4574468` |
| 6 | `1.203125` | `0.1708984375` | `1.0322265625` | `6.04` |
| 7 | `-0.197265625` | `1.078125` | `1.275390625` | `1.1829710` |

## Decision

The quantization script did not corrupt dense BF16 tensors. All 640 dense
tensors in the TQ4 checkpoint are bit-identical to the BF16 source checkpoint.

Embedding and final RMSNorm also pass module parity exactly on this probe.
The remaining dense-path failure is the LM-head/output-projection path.

The likely code-level cause is that Qwen3.5 infer currently projects logits
with `embed_tokens`, while this checkpoint has a separate dense
`lm_head.weight`. The checkpoint copy is clean; the runtime is not using that
separate LM head in the Qwen3.5 output projection path.

Do not run the 9B-TQ4 OPD bench yet. The next tranche should fix Qwen3.5
loader/model state to load and use `lm_head.weight` when present, while falling
back to tied embeddings only when the checkpoint is actually tied. Then rerun
this dense module parity gate followed by full-model logits parity.

## Problems

- This tranche identifies the failing dense module but intentionally does not
  fix it.
- The ARLE module dump currently records the runtime's effective output
  projection, which is `embed_tokens` for Qwen3.5 today. That is exactly the
  behavior under test, but the JSON field is named `lm_head` for the parity
  contract.
- Full-model logits parity remains blocked until the LM-head path passes.

## Learnings

For quantized checkpoints with dense fallbacks, dense tensor copy parity and
dense module parity are separate gates. A checkpoint can copy `lm_head.weight`
bit-perfectly and still fail downstream if the runtime ignores the separate
head and silently uses tied embeddings.
