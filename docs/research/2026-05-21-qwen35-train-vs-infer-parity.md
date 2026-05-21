# Qwen3.5 Train vs Infer Parity Attribution

## Goal

Localize the first numerical divergence between `train::Qwen35Model` and
`infer::Qwen35Model` before resuming Path B `InferTeacher` wiring.

## Setup

- Model: ModelScope `Qwen/Qwen3.5-0.8B-Base`
- Path: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Input: `input_ids=[9419]`, `positions=[0]`
- Compared stages:
  1. embedding output
  2. layer 0 RMSNorm output
  3. layer 0 attention output
  4. layer 0 FFN output
  5. layer 0 final residual sum
  6. final RMSNorm output
  7. lm_head output
- Bridge: infer-owned BF16 `DeviceVec` imported into autograd via the BF16 D2D
  bridge from `2026-05-21-arle-autograd-bf16-d2d-bridge.md`

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example qwen35_train_vs_infer_parity --release --features cuda
```

## Results

| stage | len | max_abs | max_rel | first train | first infer |
| --- | ---: | ---: | ---: | ---: | ---: |
| embedding | 1024 | 0.00000000e0 | 0.00000000e0 | 1.55029297e-2 | 1.55029297e-2 |
| layer0_rmsnorm | 1024 | 3.25469804e0 | 1.13204801e0 | 8.74701679e-1 | 1.75000000e0 |
| layer0_attention | 1024 | 5.09386063e-1 | 1.99678266e0 | -1.58582717e-1 | -6.67968750e-1 |
| layer0_ffn | 1024 | 4.81446832e-1 | 1.98535359e0 | 1.07406648e-2 | 4.92187500e-1 |
| layer0_residual | 1024 | 1.75127640e-1 | 1.99880493e0 | -1.32339120e-1 | -1.60156250e-1 |
| final_rmsnorm | 1024 | 3.42896118e1 | 1.99665082e0 | -9.08593082e0 | -1.77001953e-2 |
| lm_head | 248320 | 1.51531124e1 | 1.99995601e0 | 6.51607990e0 | 1.09375000e1 |

First divergence over `1e-4`: `layer0_rmsnorm`.

## Interpretation

Embedding is bit-equivalent after D2D import, so this is not a tokenizer ID,
embedding lookup, device bridge, or tied-embedding indexing issue.

The first measured break is layer 0 RMSNorm. The infer path intentionally uses
the Qwen3.5 `(1 + weight)` offset RMSNorm variant in prefill and final norm.
The train path currently records standard RMSNorm at the same stage. That
makes the next fix axis narrow: align train-side Qwen3.5 RMSNorm semantics with
infer, then rerun this harness before resuming Path B.

## Problems

- This is a diagnostic harness, not a model fix. Path B remains paused until a
  follow-up commit brings the stage table under `1e-4`.
- The comparison keeps infer tensors device-resident until the autograd D2D
  bridge import, but the final metric calculation downloads both tensors to
  host. That is acceptable for attribution and is not a perf claim.

## Cross-links

- `docs/experience/wins/2026-05-21-arle-infer-raw-token-logits-api.md`
- `docs/experience/wins/2026-05-21-arle-autograd-bf16-d2d-bridge.md`
- `docs/research/2026-05-21-arle-opd-infer-teacher-zero-copy-blocker.md`
