# Qwen3.5-9B GPTQModel Full-Logits Gate After F32-Load Fix

## Context

The 9B GPTQModel path previously failed multi-token generation after the
`linear_attn.A_log` / `linear_attn.norm.weight` dtype-load fix: greedy chat
prompts collapsed to repeated `!` tokens. That smoke is useful, but it is too
coarse to attribute whether the model forward is numerically unusable.

This gate compares a single-token full-model logits vector against the
unquantized PyTorch BF16 source for the same token. It is the next narrower
check after layer-0 `linear_attention` parity passed.

## Command

```bash
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example raw_token_logits_dump --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --input-ids 9419 \
  --cuda-graph false \
  --output bench-output/2026-05-22-qwen35-9b-gptqmodel-full-logits-after-f32load-fix/arle-logits.json
```

The reference was PyTorch `AutoModelForCausalLM` on the BF16 source checkpoint:

```text
/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B
```

## Results

Input token: `[9419]`

| Metric | Result |
| --- | ---: |
| vocab size | `248320` |
| finite pairs | `248320` |
| ARLE non-finite logits | `0` |
| PyTorch BF16 non-finite logits | `0` |
| top-64 dominant max relerr | `0.1242236` |
| top-64 dominant mean relerr | `0.0356395` |
| top-64 RMSE/reference-RMS | `0.0428670` |
| all-logits RMSE/reference-RMS | `0.1075743` |
| ARLE argmax | `11` |
| PyTorch BF16 argmax | `11` |
| ARLE argmax value | `14.25` |
| PyTorch BF16 argmax value | `14.25` |
| ARLE load seconds | `78.600089` |
| ARLE forward seconds | `1.512694` |
| PyTorch load seconds | `35.330610` |
| PyTorch CPU forward seconds | `2.192391` |

Top dominant entries:

| index | ARLE | PyTorch BF16 | abs err | rel err |
| ---: | ---: | ---: | ---: | ---: |
| `175169` | `-15.0625` | `-15.1250` | `0.0625` | `0.00413` |
| `11` | `14.2500` | `14.2500` | `0.0000` | `0.00000` |
| `53983` | `-13.6875` | `-14.1875` | `0.5000` | `0.03524` |
| `0` | `13.5625` | `13.3125` | `0.2500` | `0.01878` |
| `4858` | `12.0625` | `12.2500` | `0.1875` | `0.01531` |
| `158457` | `-11.9375` | `-12.1875` | `0.2500` | `0.02051` |
| `9332` | `-11.6875` | `-12.1250` | `0.4375` | `0.03608` |
| `13` | `12.0625` | `12.0000` | `0.0625` | `0.00521` |

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-full-logits-after-f32load-fix/
```

## Decision

The single-token full-logits gate passes the pragmatic 9B GPTQModel envelope:
top-64 dominant relerr is below `0.20`, logits are finite, and the dominant
argmax matches the PyTorch BF16 source exactly.

This does not erase the prior multi-token generation-quality kill. It narrows
the current state:

- the DavidWen GPTQModel checkpoint can load and run full-model inference;
- layer-0 `linear_attention` no longer NaNs after the dtype-load fix;
- the first-token dominant logits are close enough for an OPD-teacher bench;
- repeated-`!` generation is not explained by a first-token full-logits
  mismatch on `[9419]`.

The next licensed gate is therefore functional OPD: run the GPTQModel 9B
teacher through `InferTeacher` against the 0.8B LoRA student and require a
monotonic held-out KL trajectory before any user-facing headline switch.
