# ARLE CUDA OPD SwiGLU Fusion KILL

## Context

Axis: fuse the OPD rollout decode MLP `silu(gate) * up` pair into one CUDA
kernel. The hypothesis was that removing one launch from the SwiGLU activation
per layer would save about 6 ms at Qwen3-0.6B:

```text
gate_proj -> up_proj -> silu(gate) -> mul(silu_gate, up) -> down_proj
```

The implementation added a backend `silu_multiply` method, a CUDA
`silu_multiply_f32` kernel, and a rollout-only Qwen MLP fast path. It passed
the local CUDA parity test and rollout token equivalence test, then was
reverted before commit because whole-step wall-clock did not move.

No runtime code is retained from this tranche.

## Evidence

Baseline from
[`../wins/2026-05-21-arle-cuda-opd-decode-prepare-fusion.md`](../wins/2026-05-21-arle-cuda-opd-decode-prepare-fusion.md):

| Metric | Baseline |
|---|---:|
| Qwen3-0.6B step mean | 0.164387 s |
| Qwen3-0.6B step sigma / mean | 0.42% |
| rollout MLP mean | 0.010365 s |

Fused `silu_multiply` probe, serial n=3:

| Run | Step seconds | Rollout MLP seconds | Rollout attention seconds |
|---|---:|---:|---:|
| 1 | 0.163811 | 0.009109 | 0.027815 |
| 2 | 0.164245 | 0.009205 | 0.028242 |
| 3 | 0.165135 | 0.009147 | 0.028148 |
| mean | 0.164397 | 0.009154 | 0.028068 |
| sigma / mean | 0.335% | n/a | n/a |

Wall-clock acceptance frame:

| Metric | Baseline | Fused probe | Delta | Status |
|---|---:|---:|---:|---|
| step mean | 0.164387 s | 0.164397 s | +0.006% | KILL |
| rollout MLP mean | 0.010365 s | 0.009154 s | -11.69% | narrow win only |
| rollout token sequence | matched | matched | unchanged | correctness pass |

The pre-license gate for this axis was:

- license: Qwen3-0.6B step `<= 0.158 s`;
- license-with-investigation: `0.158-0.162 s`;
- kill: `> 0.162 s`.

The measured mean was `0.164397 s`, so the axis is killed even though the
narrow MLP timer improved.

## Root Cause

The launch-count hypothesis was too small for the current frontier. At this
shape, the `silu + mul` pair is a sub-10 ms rollout phase and behaves like a
bandwidth / data-movement surface: fusing the math into one elementwise kernel
reduced the local MLP timer by about 1.2 ms, but the end-to-end OPD step did
not improve beyond noise.

No `ncu` memory-throughput trace was captured for this killed probe, so the
SOLID conclusion is intentionally conservative:

- **licensed:** fusing this elementwise pair does not move Qwen3-0.6B OPD
  wall-clock at the current code state;
- **working explanation:** the remaining cost is not the extra launch alone,
  but data movement / surrounding one-token decode work that this kernel does
  not reduce.

This matches the earlier CPU-side lesson in
[`../../projects/2026-05-20-opd-cpu-perf-cycle-wrap.md`](../../projects/2026-05-20-opd-cpu-perf-cycle-wrap.md):
the Axis F attempt to add CPU parallelism to an already bandwidth-bound
MatmulBT backward kernel regressed by about 1.1%. In both cases, adding
parallelism or launch fusion to a bandwidth-bound kernel surface did not move
the wall-clock gate; the next viable axis must reduce bytes, allocations, or a
larger fused region.

## Fix

Killed and reverted the implementation before commit.

No `Backend::silu_multiply` method, CUDA kernel, Qwen MLP fast path, or parity
test remains in the tree. The only retained output is the bench artifact set.

## Rule

For sub-10 ms OPD phases, check whether the surface is bandwidth/data-movement
bound before proposing a kernel-fusion axis. A launch-count argument is not
enough. Pre-license future micro-fusion only if one of these is true:

- the target cluster is at least 10 ms of wall-clock step time;
- the fused kernel removes a materialization or allocation, not only one
  elementwise launch;
- `ncu` / profiler evidence shows launch overhead rather than memory traffic is
  the dominant cost.

Otherwise prefer larger decode-block fusions such as `append_kv + merge +
o_proj`, allocator pooling, or preallocated decode runners.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-swiglu-fused/realckpt-profile-run1.txt`
- `bench-output/2026-05-21-arle-cuda-opd-swiglu-fused/realckpt-profile-run2.txt`
- `bench-output/2026-05-21-arle-cuda-opd-swiglu-fused/realckpt-profile-run3.txt`
