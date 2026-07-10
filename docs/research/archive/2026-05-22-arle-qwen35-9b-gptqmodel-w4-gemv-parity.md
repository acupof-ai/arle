# Qwen3.5-9B GPTQModel W4 Layer-Local GEMV Parity

## Context

`DavidWen2025/Qwen3.5-9B-GPTQ-4bit` is locally complete and the gated
loader path can serve, but the generation-quality smoke failed: three greedy
prompts collapsed into repeated punctuation. The full-model output was too
coarse to attribute, so this tranche isolates the quantized projection path.

Model paths:

- Quantized: `/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit`
- BF16 source: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B`

Harness:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p infer --example gptqmodel_w4_gemv_parity --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --source-model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tensor-base model.language_model.layers.0.mlp.gate_proj \
  --seed 1592594996 \
  --output bench-output/2026-05-22-qwen35-9b-gptqmodel-layerlocal/layer0-mlp-gate-proj.json
```

## Method

For each projection, the harness compares two separable questions:

1. **CUDA W4A16 GEMV vs faithful GPTQ reference**: convert GPTQModel physical
   layout (`qweight [K/8,N]`, `scales [K/group,N]`, symmetric qzeros nibble
   `7`) to ARLE's `[N,K/2]` runtime packing and compare `w4a16_gemv_cuda`
   against a Rust CPU reference using `q - 8`.
2. **Faithful GPTQ reference vs BF16 source**: compare the same quantized
   reference matvec against the original BF16 weight matvec on the same
   deterministic BF16 input vector.

Layer 0 is `linear_attention`, so the conventional q/k/v/o scan uses layer 3,
the first full-attention layer. Layer 10 `gate_proj` checks a mid-layer MLP
projection.

## Results

| tensor | shape | CUDA vs faithful rmse/ref | faithful vs BF16 rmse/ref |
|---|---:|---:|---:|
| layer0 mlp.gate_proj | 12288x4096 | 0.243% | 13.445% |
| layer0 mlp.up_proj | 12288x4096 | 0.242% | 13.392% |
| layer0 mlp.down_proj | 4096x12288 | 0.248% | 14.882% |
| layer0 linear_attn.in_proj_qkv | 8192x4096 | 0.250% | 14.767% |
| layer0 linear_attn.in_proj_z | 4096x4096 | 0.232% | 14.533% |
| layer0 linear_attn.out_proj | 4096x4096 | 0.235% | 13.112% |
| layer3 self_attn.q_proj | 8192x4096 | 0.238% | 12.837% |
| layer3 self_attn.k_proj | 1024x4096 | 0.222% | 13.450% |
| layer3 self_attn.v_proj | 1024x4096 | 0.245% | 13.384% |
| layer3 self_attn.o_proj | 4096x4096 | 0.218% | 12.681% |
| layer10 mlp.gate_proj | 12288x4096 | 0.245% | 13.233% |

Raw artifacts:
`bench-output/2026-05-22-qwen35-9b-gptqmodel-layerlocal/`.

## Attribution

The ARLE GPTQModel W4 packing and `w4a16_gemv_cuda` path are not the first
generation-quality culprit. Every sampled quantized projection reproduces the
faithful GPTQ reference within 0.25% RMSE/source-RMS, well under the 1% layer-local
gate.

The faithful GPTQ projection outputs differ from the BF16 source by roughly
12.7-14.9% RMSE/source-RMS on a deterministic random hidden vector. That is a
real quantization error budget, but it is not an ARLE kernel-layout bug. The
remaining full-model failure must be attributed above the isolated W4 GEMV
path: dense fallback modules, hybrid linear-attention semantics, embedding /
norm / lm_head handling, or accumulated GPTQ quantization sensitivity.

## Next Axis

Do not run OPD with this checkpoint until layer-local forward parity passes.
The next single-variable test should compare ARLE vs PyTorch BF16 for layer 0
hybrid `linear_attn` forward on the same hidden input, then dense `embedding`,
final norm, and untied `lm_head` if linear attention passes.

Kill criterion: if layer 0 linear-attention forward RMSE/reference-RMS exceeds
5%, hold the 9B GPTQModel teacher path and isolate the specific linear-attention
sub-op (`conv1d`, delta-rule state update, gating, norm, or output projection).
