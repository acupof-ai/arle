# Agent-OPD: masked-CE writeback + sandbox hang fix

`pending-remote` — the writeback path is CUDA-only (`forward_logits_window` +
LoRA backward on the 27B); can't bench on this Mac. Remote ticket: run a full
agent-OPD round on the 8×H20 pod and record TTFT/throughput vs the prior
per-pair writeback.

## Context

Full agent-OPD loop on Qwen3.6-27B-FP8: in-process student rolls out the
read/write/replace/bash agent against a SWE-bench-Pro sandbox; `git diff` is the
candidate patch, hidden `fail_to_pass` tests are the reward; passing trajectories
are written back as masked next-token CE targets.

## What Worked

- **Reproducible cold-start**: the 27B solves the ansible FQCN task (adds
  `import keyword` + a keyword check in `is_valid_collection_name`), passes the
  hidden tests — `passed=true` on 2/2 runs at `max-turns 24`.
- **qwen3_coder native XML tool format** (治本): `<function=NAME><parameter=…>`
  — 12/12 valid tool calls, eliminated the JSON-escaping breakage.
- **Sandbox `wait4` hang fixed**: `run_captured` runs bash/pytest in their own
  process group + captures to a temp file (no pipe a backgrounded grandchild can
  hold) + kills the group on exit/timeout. Regression test
  `bash_does_not_hang_on_backgrounded_child`.
- **Masked single-trajectory CE writeback**: one windowed forward over
  `prompt ++ response`, loss only on `response_mask==1` tokens — replaces the
  O(N²) per-pair explosion that OOM'd (`cuda alloc_zeros failed`). Grad scaled by
  `1/total_targets` so the effective LR is trajectory-length-independent.

## Rule

Long agentic trajectories crush the writeback two ways: dense `[seq,vocab]` logits
(fixed by per-window lm_head) and `[heads,seq,seq]` attention scores (fixed by
head-chunking — heads are independent, so exact). Both must be bounded, not just
the logits.
