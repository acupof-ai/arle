# Qwen3.5-0.8B InferTeacher OPD Phase Attribution

## Goal

Track A measurement for the relaxed Path B gate: attribute a single
Qwen3.5-0.8B-Base LoRA OPD step that uses the `InferTeacher` path. The user
relaxed the self-teach wall-clock gate from 0.4 s to 5 s because the
cross-runtime path intentionally includes scheduler, paged-KV, and D2D bridge
overhead that the in-runtime teacher does not.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --steps 1 \
  --eval-steps 0 \
  --max-step-seconds 5 \
  | tee bench-output/2026-05-21-qwen35-08b-infer-teacher-phase-attribution/run.txt
```

Raw artefact:

- `bench-output/2026-05-21-qwen35-08b-infer-teacher-phase-attribution/run.txt`

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student mode: LoRA, rank 16, alpha 32, target set `attention-qv`
- Prompt: `[9419]`
- Rollout length: 8 requested, 9 scored tokens after prompt + rollout
- LR: `1e-5`
- CUDA graph: enabled for the infer teacher runtime

## Results

The measured self-teach step passed the revised 5 s gate:

| Metric | Seconds |
|---|---:|
| Total OPD step | 2.151332 |
| Student rollout | 0.712337 |
| Infer `forward_token_logits` call | 0.002702 |
| Infer sync | 0.029969 |
| D2D bridge import | 0.000037 |
| Teacher forward total as seen by OPD | 0.032710 |
| Student KL forward | 0.158496 |
| KL loss compute + readback | 0.007076 |
| Optimizer zero grad | 0.000001 |
| Backward | 1.227058 |
| Grad clip | 0.000513 |
| Optimizer step | 0.000624 |
| Post-step cleanup | 0.012501 |

Shares of the step:

| Phase | Share |
|---|---:|
| Backward | 57.04% |
| Student rollout | 33.11% |
| Student KL forward | 7.37% |
| Teacher forward total | 1.52% |
| Post-step cleanup | 0.58% |
| KL loss compute + readback | 0.33% |
| Optimizer step | 0.03% |
| Grad clip | 0.02% |

The `forward_token_logits` call itself is mostly an enqueue/API frame. The
explicit `sync` immediately after it is the conservative CUDA wall-clock frame
for teacher work in this harness. The D2D BF16-to-train import is 37 us and is
not a current bottleneck at the 0.8B self-teach shape.

## Interpretation

The relaxed Path B gate is licensed for 0.8B self-teach: 2.151 s is below the
new 5 s ceiling. The earlier 3.27 s LoRA measurement remains consistent with
the same order of magnitude; this run uses the new attribution harness and
should be treated as the measured Track A datapoint.

The cross-runtime teacher bridge is not the binding cost in this self-teach
step. `InferTeacher` total time is 32.7 ms, only 1.52% of the step, and the D2D
import is effectively noise. The dominant phases are student backward
(1.227 s) and student rollout (0.712 s), together 90.2% of wall-clock.

## Next Axis

For Track B, proceed with the real Qwen3.5-4B teacher to Qwen3.5-0.8B LoRA
bench. The bridge overhead is small enough at 0.8B self-teach that the useful
question is now whether the 4B teacher run completes without OOM and produces
a monotonic KL trajectory over 200 steps.

Future optimization should not start with the D2D bridge. A SOLID next
performance axis would first profile the 0.8B LoRA backward and rollout phases
under the cross-runtime harness, then pick one single-variable target from
those measurements.
