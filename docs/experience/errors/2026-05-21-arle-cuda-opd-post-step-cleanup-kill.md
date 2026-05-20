# ARLE CUDA OPD post_step_cleanup KILL

## Context

Axis: reduce Qwen3-0.6B OPD `post_step_cleanup`, measured after
`36606d5` at about `0.019 s` / `8.9%` of the step. Moderate shape showed
cleanup at only about `0.25%`, so the suspicion was a host-resident sweep
similar to the earlier host-mirror retention bug.

No runtime code is retained from this tranche.

## Probe

Temporary instrumentation split `post_step_cleanup` into:

- `tape.entries.clear()` + `tape.set_enabled(true)`
- `keep_extra.clone()`
- `extend_keep_with_params_and_grads(...)`
- `store.retain_ids(...)`

The same probe counted freed TensorStore slots, device handles, total logical
elements, and host-resident elements. A second leak-control kept the same
iteration and slot clearing but intentionally `mem::forget`-ed dead tensors to
skip `CudaSlice::Drop` / `cuMemFreeAsync`. The leak-control was process-local
diagnostic code only and was reverted.

## Evidence

Instrumented cleanup profile:

```text
cleanup_profile slots_visited=22670 tensors_freed=21736 device_tensors_freed=21716 elements_freed=1578036935 host_elements_freed=5242563
phase_summary rank=4 phase=post_step_cleanup seconds=0.018202 pct_total=8.817
phase_summary rank=5 phase=post_step_cleanup_retain_ids seconds=0.018133 pct_total=8.784
phase_summary rank=12 phase=post_step_cleanup_tape_clear seconds=0.000047 pct_total=0.023
phase_summary rank=13 phase=post_step_cleanup_keep_extend seconds=0.000020 pct_total=0.010
phase_summary rank=15 phase=post_step_cleanup_keep_clone seconds=0.000001 pct_total=0.001
```

Leak-control profile:

```text
cleanup_profile slots_visited=22670 tensors_freed=21736 device_tensors_freed=21716 elements_freed=1578036935 host_elements_freed=5242563
phase_summary rank=9 phase=post_step_cleanup seconds=0.000435 pct_total=0.228
phase_summary rank=10 phase=post_step_cleanup_retain_ids seconds=0.000373 pct_total=0.196
step_summary loss=1.788745430531e-5 rollout_len=12 total_step_seconds=0.190994
```

The host-resident part is small: `5,242,563` f32 elements, about 20 MiB.
The cleanup wall-clock is almost entirely the destruction of `21,716` CUDA
device tensors that together represent `1,578,036,935` logical f32 elements.

## Root Cause

The hypothesis was half right but the mechanism was different. Cleanup is not
building the keep set slowly and is not sweeping a large host mirror. It is
issuing thousands of device allocation frees at the end of the step.

`TensorStore::retain_ids` legitimately prunes dead activations and temporary
device tensors. Dropping each dead `Tensor` drops its `CudaStorage`, and
cudarc's `CudaSlice::Drop` enqueues a `cuMemFreeAsync` for each allocation.
At Qwen3-0.6B one OPD step creates over 21k temporary device tensors, so the
cleanup phase is a 21k-call allocator/free storm.

The leak-control proves the floor: if the frees are skipped, cleanup falls
from `18.2 ms` to `0.435 ms`, but that is not a valid implementation because
it leaks the entire 6+ GiB temporary device working set.

## Fix

Killed this small-fix axis. There is no safe local patch in
`cleanup_after_backward` or `retain_ids` that removes the cost while preserving
memory correctness. Moving drops to a background thread would only hide the
cost from the phase timer and risks allocator/stream-order contention across
steps; leaking handles passes the one-step profile but is invalid for training.

The real fix is an allocation strategy change, not a cleanup helper tweak:

- a CUDA tensor arena / memory pool for ephemeral forward/backward outputs;
- or model/op fusion that reduces the number of intermediate TensorStore
  allocations before cleanup sees them;
- or a rollout-specific preallocated decode runner for the short-sequence path.

## Rule

Do not treat Qwen3-0.6B `post_step_cleanup` as a host-mirror bug. It is a
device allocator/free-count problem. License any future cleanup axis only with
a multi-step wall-clock benchmark, because shifting frees out of the named
cleanup phase is not a win unless full step time improves.

Recommended next axis: CUDA ephemeral tensor arena. Pre-license:

- Qwen3-0.6B OPD mean step `<= 0.195 s` over n=3 with sigma `< 5%`;
- moderate shape `<= 56 ms`;
- no increase in peak GPU memory beyond the current profile's budget;
- 50-step `lr=1e-7` convergence non-regression;
- kill if the arena only moves time from `post_step_cleanup` into forward,
  backward, or allocator contention and total step remains `> 0.205 s`.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-cleanup-profile/instrumented-profile.txt`
- `bench-output/2026-05-21-arle-cuda-opd-cleanup-profile/leak-control-profile.txt`
