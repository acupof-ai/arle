# G3 CUDA Graph Noop On P5 Shape

## Context

G3 was revised after the original gap-analysis premise turned stale. ARLE
already has batch-size CUDA Graph warmup/cache:

- `infer/src/scheduler/cuda/core/warmup.rs:26,418` warms batch-size lists.
- `infer/src/model/qwen3/batch_decode.rs:171` caches `graph_cache[batch_size - 1]`.
- `infer/src/model/qwen35/batch_decode.rs:168` caches Qwen3.5 piecewise
  `graph_cache[group_idx][batch_size - 1]`.

The revised question was not "should ARLE add a bucket pool?" but "does the
existing graph path still move P5-shaped low-c decode wall-clock enough to
justify more graph work?"

Measurement shape:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER.
- Model: Qwen3.5 0.8B Base at
  `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`.
- Workload: P5 serve proxy, synthetic `prompt_tokens=16`, `output_tokens=8`,
  c=1/2/4/8, one sequential run per mode.
- Modes: `--cuda-graph` vs `--disable-cuda-graph`.
- Tooling: `scripts/profile_nsys_signal.sh` with GuideLLM and nsys CUDA API
  stats.
- GPU pre/post check: every valid run started and ended at
  `NVIDIA GeForce RTX 4070 Ti SUPER, 1093, 16376, 0`.

Two initial c=1 graph attempts were invalid and excluded: one used the wrong
OpenAI model id, and one used GuideLLM `*_stdev=0`, which the synthetic loader
rejects. Valid artifacts are the `v3` c=1 graph run plus the seven paired
graph/eager runs listed below.

## Root Cause

The existing graph path reduces launch API overhead, but request wall-clock
does not move by the 5% license threshold. This matches T2's P5 trace: the
dominant OPD wall-clock buckets are student rollout and backward, with
`cuStreamSynchronize`/sync behavior showing up as diagnosis context rather than
an isolated launch-overhead bottleneck.

GuideLLM request wall-clock:

| c | graph mean request ms | eager mean request ms | graph delta | graph output tok/s | eager output tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 33.838 | 34.952 | +3.19% | 233.52 | 226.15 |
| 2 | 45.562 | 46.645 | +2.32% | 347.80 | 339.76 |
| 4 | 62.550 | 63.535 | +1.55% | 507.53 | 499.74 |
| 8 | 97.626 | 98.041 | +0.42% | 651.58 | 648.89 |

nsys CUDA API summary:

| c | `cudaLaunchKernel` delta | `cuLaunchKernel` delta | combined launch delta incl. `cuGraphLaunch` | `cuStreamSynchronize` delta |
|---:|---:|---:|---:|---:|
| 1 | +62.52% | -13.92% | +37.15% | +4.98% |
| 2 | +47.88% | +48.84% | +31.15% | -0.04% |
| 4 | +39.68% | +37.38% | +24.92% | +0.30% |
| 8 | +28.47% | +23.20% | +16.59% | +2.05% |

Interpretation:

- The graph path is real: runtime `cudaLaunchKernel` time drops by 28-63%.
- It is not a licensed next axis: mean request latency improves by only
  0.42-3.19%, below the 5% wall-clock threshold at every concurrency.
- `cuStreamSynchronize` is effectively unchanged except c=1 noise, so this
  does not contradict T2's sync-bound diagnosis.

## Fix

KILL G3 implementation work for P5-shaped low-c decode. Keep the existing
graph warmup/cache, but do not spend engineering budget on another CUDA Graph
bucket/eager-launch axis until a future trace shows wall-clock launch overhead
above threshold.

Move G3 budget toward the T2-licensed bottleneck: student rollout/backward.

## Rule

For G3-class graph work, API launch-time reduction is diagnostic only. The
license decision must use wall-clock/request framing first, then use nsys API
stats to explain why.

## Artifacts

- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c1-graph-v3/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c1-graph-v3-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c1-eager/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c1-eager-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c2-graph/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c2-graph-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c2-eager/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c2-eager-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c4-graph/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c4-graph-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c4-eager/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c4-eager-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c8-graph/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c8-graph-profile-nsys-signal/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c8-eager/`
- `bench-output/2026-05-25-g3-qwen35-08b-p5shape-c8-eager-profile-nsys-signal/`

Related: `docs/experience/wins/2026-05-25-opd-end-to-end-trace-p5-shape.md`.
