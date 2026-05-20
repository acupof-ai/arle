# ARLE CUDA OPD Rollout KV Cache

## Goal

Eliminate prefix re-encoding in the OPD greedy rollout path for Qwen3-0.6B
CUDA. The pre-license target was `<= 0.20 s/step`; the previous AdamW wins
entry measured `0.294321 s/step` with `rollout_student_forward = 0.125247 s`
as the largest remaining phase.

## Hypothesis

Caching per-layer K/V during rollout should turn the rollout from repeated
full-prefix forwards into prompt prefill plus one-token decode forwards.
For `prompt_len=4`, `rollout_len=8`, that reduces computed rollout positions
from `4+5+...+11 = 60` token positions to `4+7 = 11`.

## Command

Real-checkpoint profile:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda
```

Moderate non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Driver/runtime: CUDA 13.x path, `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Prompt: `[1, 872, 198, 3456]`
- Rollout length: 8, final rollout length 12
- LR: 5e-5, AdamW betas=(0.9, 0.999), eps=1e-8, wd=0

## Results

Final Qwen3-0.6B profile, n=3:

| Run | Step seconds | Rollout forward seconds | Grad clip seconds |
|---|---:|---:|---:|
| 1 | 0.253919 | 0.097539 | 0.035915 |
| 2 | 0.253510 | 0.097279 | 0.035967 |
| 3 | 0.253191 | 0.097001 | 0.036371 |
| mean | 0.253540 | 0.097273 | 0.036084 |
| median | 0.253510 | 0.097279 | 0.035967 |
| sigma / mean | 0.119 % | 0.226 % | 0.576 % |

Delta vs `2026-05-21-arle-cuda-fused-adamw-qwen3-06b.md`:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 0.294321 | 0.253540 | -13.86 % |
| rollout student forward | 0.125247 | 0.097273 | -22.34 % |
| rollout argmax readback | 0.021119 | 0.010280 | -51.32 % |
| speedup | 1.00x | 1.16x | +1.16x |

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=17.036237 median_steps_per_sec=17.066318 sigma_steps_per_sec=0.127022 sigma_pct=0.746 mean_step_seconds=0.058702 median_step_seconds=0.058595 max_loss_relative_error_vs_cpu=0.000001276
```

This keeps the moderate shape under the `80 ms` ceiling and improves it from
the previous `68.925 ms` AdamW entry to `58.702 ms`.

## Problems

This did not meet the `<= 0.20 s/step` pre-license target. The root-cause
hypothesis was only partially true: prefix re-encoding was measurable, but at
`seq_len <= 12` wall-clock is still dominated by many one-token decode launches
and per-token projection/MLP work, not quadratic attention work.

The KV cache implementation needed two follow-up trims before the final
number:

- Cache repeated full-head K/V so decode does not rerun `repeat_kv` over the
  entire cached prefix each step.
- Skip the causal mask add for decode rows where `q_len=1` and every cached key
  is visible.

## Learnings

KV cache is correctness-preserving and worth a stable `13.86 %` end-to-end
speedup, but it is not enough to make OPD rollout the next 200 ms class axis.
For short OPD rollouts, launch count is the more likely binding constraint than
attention's `O(N^2)` token count.

Next single-variable axis: CUDA graph capture or a fused one-token rollout
decode path for the cached rollout loop. Pre-license: total step `<= 0.20 s`
and `rollout_student_forward <= 0.055 s` with n=3, `sigma < 5 %`; kill if total
stays above `0.235 s` after graph/fusion evidence, because then grad clip /
backward / cleanup are the remaining wall-clock floor.

## Gates

- `cargo check --workspace`: passed
- `cargo test -p train qwen35_rollout_kv_cache_matches_full_forward_tokens --release`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p train --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda`: passed
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed,
  mean `0.058702 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed,
  mean `0.253540 s/step`

## Artefacts

- Final real-checkpoint repeats:
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/decode-mask-skip-run.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/final-repeat-2.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/final-repeat-3.txt`
- Moderate CUDA OPD:
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/moderate-final-run.txt`
- Intermediate probes:
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/run.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/last-logits-run.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-rollout-kvcache/full-head-cache-run.txt`
