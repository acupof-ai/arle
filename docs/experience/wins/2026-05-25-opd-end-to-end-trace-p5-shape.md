# OPD End-to-End Trace, P5 Shape

Related:
`docs/projects/2026-05-24-opd-mainline-task-backlog.md` T2 and
`docs/research/2026-05-24-arle-opd-end-to-end-trace.md`.

## Context

After P5 and T14 released the GPU, T2 profiled the same mainline OPD shape:
Qwen3.5-4B teacher, Qwen3.5-0.8B student LoRA, `prompt_max_tokens=16`,
`rollout_len=8`, pure OPD, and `examples/opd/sample-prompts.jsonl`.

## What Worked

The trace used a clean 5-step wall-clock anchor plus a same-shape Nsight
Systems pass. The anchor is the license source; nsys is diagnosis context.

Headline wall-clock split over 5 anchor train steps:

| phase | share |
| --- | ---: |
| backward | 42.83% |
| student_rollout | 41.91% |
| teacher_forward_total | 8.48% |
| student_forward | 6.33% |
| kl_loss | 0.17% |

The top bottleneck is student-side train work. Teacher scoring is secondary,
and KL/checkpoint/optimizer are not P5-shape performance targets.

## Rule

License OPD optimizations from per-step wall-clock, not narrow nsys kernel
windows. For P5 shape, a future optimization must reduce
`student_rollout + backward` or show a new phase table where another bucket
exceeds 10% of wall-clock.

## Artifacts

- Research: `docs/research/2026-05-24-arle-opd-end-to-end-trace.md`
- Raw: `bench-output/2026-05-25-t2-opd-end-to-end-trace/`

## Verdict

PASS. T2 identifies the next licensed P5 optimization axis as student
rollout/backward, not KL, checkpoint, optimizer, or teacher logits micro-work.
