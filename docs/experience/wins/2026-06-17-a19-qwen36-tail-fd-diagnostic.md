# A19 Qwen3.6 full-model FD tail diagnostic

## Context

A18 removed the dense `MatmulBT` backward wall, making the full-model finite
difference gate cheap enough to probe. The full-model single-element gate still
failed, for example:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_full_fd_eps_1e-3_gpu1_20260617_121235.log

target=model.language_model.layers.0.mlp.experts.182.up_proj.weight.lora_b
eps=1.0e-3 analytic=-5.421358943e-1 numeric=2.025604248e0 rel_err=1.268e0
```

Before treating that as a backward bug, we needed to isolate whether the
post-layer tail (`final_norm -> lm_head -> weighted logits loss`) was correct.

## What Changed

Added a hidden diagnostic-only path:

- `Qwen35Model::forward_lm_head_tail_for_diagnostics(hidden, ...)`;
- `qwen36_fp8_lora_fd_gate --mode tail`;
- tail mode marks the synthetic hidden tensor trainable and finite-diffs the
  hidden element with the largest analytic gradient.

No production forward path changes. `mlp-layer` and `full-model` behavior stay
unchanged.

## Evidence

Local gates:

```text
cargo fmt --check
cargo check -p train --release --no-default-features --features no-cuda --lib
cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
```

Remote `.62`, GPU0/GPU2/GPU4, GPU3 avoided:

```text
source=/data01/arle-track1-route-frozen-fd-fast-20260617095440
target=/data01/arle-target-track1-route-frozen-fd
model=/data01/models/Qwen3.6-35B-A3B-FP8
CUDA_HOME=/usr/local/cuda
CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1
INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
ARLE_QWEN35_DEEPGEMM=0
```

Build:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_tail_mode_rebuild_20260617_122556.log

cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
PASS: Finished release target in 8.33s
```

Tail diagnostic:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_tail_fd_after_eps_1e-2_gpu2_20260617_122622.log

mode=tail target=diagnostic.tail_hidden index=652 eps=1.0e-2
loss_base=5.581412315e-1 loss_minus=1.230585456e0 loss_plus=-6.710906327e-2
analytic=-6.454741669e1 numeric=-6.488472748e1 rel_err=5.199e-3
qwen36_fp8_lora_fd_gate PASS
RUN_EXIT=0
```

Tail at `eps=1e-3` was close but above the strict gate:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_tail_fd_after_eps_1e-3_gpu0_20260617_122622.log

mode=tail target=diagnostic.tail_hidden index=652 eps=1.0e-3
analytic=-6.454741669e1 numeric=-6.617498016e1 rel_err=2.459e-2
RUN_EXIT=1
```

Full-model layer-39 MoE target remained highly eps/probe sensitive:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_full_fd_layer39_eps_1e-3_gpu0_20260617_121426.log
eps=1.0e-3 analytic=-1.678310990e0 numeric=-2.579116631e1 rel_err=9.349e-1

log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_full_fd_layer39_eps_1e-2_gpu1_20260617_121426.log
eps=1.0e-2 analytic=-1.678310990e0 numeric=-4.873275757e-1 rel_err=7.096e-1

log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_full_l39_sparse_eps_1e-1_gpu2_20260617_122753.log
eps=1.0e-1 analytic=-1.678310990e0 numeric=-1.578960419e0 rel_err=5.920e-2

log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_full_l39_dense_eps_1e-2_gpu4_20260617_122753.log
eps=1.0e-2 sparse_logit_probe=false analytic=-1.796462774e0 numeric=-9.593963623e-1 rel_err=4.660e-1
```

Layer-local MoE remained licensed:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a19_mlp_layer39_fd_gpu0_20260617_121523.log

mode=mlp-layer layer=39
target=model.language_model.layers.39.mlp.experts.96.up_proj.weight.lora_b
analytic=-5.701754162e-7 numeric=-5.699121175e-7 rel_err=4.618e-4
qwen36_fp8_lora_fd_gate PASS
RUN_EXIT=0
```

## Verdict

The full-model single-element FD failure is not explained by a broken
`final_norm -> lm_head` tail: the isolated tail passes at `eps=1e-2`, and the
same layer's local MoE FD passes. The full-model single-element probe is
ill-conditioned for this real FP8 checkpoint/loss shape: numeric estimates move
from 25.8 to 0.49 to 1.58 as eps/probe changes.

Do not use the current full-model single-element FD as a hard correctness gate.
Keep the licensed gates component-wise for now: layer-local MoE FD, attention FD,
tail FD, and e2e coherence. A future full-model gradient gate should use a
better-conditioned directional derivative or a loss/target designed for the
real checkpoint's quantized forward.

## Rule

Full-model finite difference is a gate only after its numerical conditioning is
licensed. If eps/probe sweeps change the numeric derivative by orders of
magnitude while component gates pass, the single-element full-model probe is the
bug, not automatically the backward implementation.
