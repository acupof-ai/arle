# Qwen3.6 full-model finite-diff still CPU-bound after MoE backward fixes

## Context

A10-A12 reduced the real-checkpoint Qwen3.6-35B-A3B FP8 single-layer MoE
backward gate from 7.103s in `MoeGroupedLinear` to 0.198s, with the routed
expert finite-diff still passing. That made it tempting to assume the earlier
full-model finite-diff blocker was mostly MoE backward padding.

This entry records the control run after A12: the full-model gate is still not
licensed.

## Evidence

Remote command on `.62`, GPU7, model
`/data01/models/Qwen3.6-35B-A3B-FP8`:

```text
CUDA_VISIBLE_DEVICES=7 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --device 0 \
  --target-set all-linear \
  --target-adapter auto:routed-up \
  --mode full-model \
  --layer 0 \
  --eps 1e-3 \
  --profile-backward
```

Observed after 150-180s:

```text
qwen36_fp8_lora_fd_gate_start model=/data01/models/Qwen3.6-35B-A3B-FP8 device=0 rank=8 alpha=16.000000 target_set=all-linear target_adapter=auto:routed-up eps=1.0e-3 tokens=[1, 3, 8] mode=full-model layer=0 profile_backward=true

nvidia-smi:
GPU7 memory.used=34471 MiB utilization.gpu=0 %

ps:
qwen36_fp8_lora_fd_gate pcpu=99.8 rss=23935732 KiB
```

No backward profile or finite-diff result had printed by 180s, so the run was
killed with SIGTERM. GPU memory residency plus 0% GPU util and one saturated CPU
thread matches the earlier A9 full-model failure mode.

## Root Cause

Not fully rooted. The measurement proves the A12 MoE backward fix does not by
itself unlock full-model finite-diff. The active wall is still a full-model
host path before the finite-diff gate reaches a profiled backward result.

The likely suspect remains Qwen3.6 full-model train forward/backward host
coverage outside the isolated MLP gate, especially linear attention or another
CPU-only op in the full forward. That is a hypothesis, not a conclusion; this
run did not include per-layer/per-op forward timers.

## Fix

Do not keep running full-model finite-diff as a blind gate. Add a bounded
full-model phase timer first: layer, attention/linear-attention, MLP, norm, and
host/device sync attribution. Then move the next optimization to the measured
full-model host path.

## Rule

A fast licensed micro-gate does not prove the full-model gate. Re-run the full
shape after each wall falls, but if the process is GPU-resident, GPU-idle, and
single-thread CPU-bound, stop and instrument the full-model phase boundary
instead of waiting for a finite-diff result.
