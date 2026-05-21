# Qwen3.5-9B GPTQModel Dense Module Continue Scan

## Context

The DavidWen GPTQModel 4-bit checkpoint still fails multi-token generation
quality, but prior attribution showed the sampled W4 GEMV projections are clean
against a faithful GPTQ dequant reference. The next question was whether the
dense tensor bit-compare failure (`linear_attn.A_log` / `linear_attn.norm.weight`
stored as BF16 instead of source FP32) also poisons the known dense modules:
embedding, final RMSNorm, and untied `lm_head`.

## Change

`scripts/qwen35_tq4_dense_parity.py` now has:

```text
--continue-after-dense-fail
```

This keeps the existing fail-closed default, but allows diagnostics to continue
into module parity after a dense tensor bit-identity failure. The script also
sets `INFER_EXPERIMENTAL_GPTQMODEL_W4=1` for the ARLE dump subprocess because
this checkpoint is intentionally behind the experimental loader gate.

## Command

```bash
.venv/bin/python scripts/qwen35_tq4_dense_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tq-model /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --token-id 9419 \
  --continue-after-dense-fail \
  --output-dir bench-output/2026-05-22-qwen35-9b-gptqmodel-dense-module-continue
```

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-dense-module-continue/
```

## Results

Dense tensor identity remains a fail:

```text
dense_tensor_count=576 gate_pass=False
```

But the dense module outputs pass against the BF16 source model:

| Module | RMSE / reference RMS | Max abs | Gate |
| --- | ---: | ---: | --- |
| embedding | `0.00000000e+00` | `0.00000000e+00` | PASS |
| final_rmsnorm | `0.00000000e+00` | `0.00000000e+00` | PASS |
| lm_head | `1.75861725e-05` | `7.81250000e-03` | PASS |

## Decision

The generation failure is not explained by embedding, final norm, or untied
`lm_head`. Those dense modules are functionally clean despite the dense tensor
bit-compare failure.

Next axis: module-local parity for the hybrid `linear_attn` block under the
GPTQModel checkpoint, including `A_log`, `norm.weight`, gated state update, and
conv/state handling. That is now the first plausible dense-path suspect.

