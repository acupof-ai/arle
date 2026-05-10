# Task #35 — cap=8 prefill startup warmup

## Goal

Close the cap=8 cold-start first-burst hole by warming paged prefill kernels,
Marlin prefill GEMM paths, cuBLASLt heuristics, and scratch allocation during
server startup rather than on the first live request burst.

## Hypothesis

For W4A16 Marlin with `max_concurrent_prefill_requests=Some(8)`, first-burst
TTFT regressions come from deferred prefill first-touch work. Moving that work
to startup should trade roughly one prefill-warmup window of startup time for
lower cold-request variance. Expected startup cost was 1-3s from the original
M_warmup plan. After review, Pass 3 now warms production-sized rows instead of
the initial 64-token scout shape; measured startup overhead is about 8.2s for
the W4A16 cap=8 server, with an automatic B=8 row-size backoff to avoid Marlin
scratch OOM.

## Implementation

- Added Pass 3 in `infer/src/scheduler/cuda/core/warmup.rs`.
- Pass 3 drives the persistent async prefill context via `launch_prefill_batch`
  / `complete_prefill_batch` for `B=1..=prefill_cap`.
- Warmup row tokens are derived from the configured prefill chunk, step token
  budgets, and `effective_max_seq_len`. If a production-sized row OOMs during
  warmup, the pass clears the slot state and retries that batch size with a
  half-sized row.
- Added `GenerationState::reset_for_warmup_clear()` and Qwen3, Qwen3.5, and
  DeepSeek implementations so dummy warmup KV/logits are cleared before serving.
- Added diagnostic escape hatch `INFER_PREFILL_WARMUP=0` for matched cold-start
  A/B only. Default remains enabled.

## Environment

- Backend: CUDA
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- CUDA: 13.2, driver 595.71.05
- Model: `infer/models/Qwen3-4B-W4A16-sym-g128-marlin`
- Base commit before this change: `940c7cc`
- Feature set: `cargo build --release -p infer --features cuda`
- Server flags: `--num-slots 8 --max-seq-len 5120`

## Verification

```bash
cargo fmt --all --check
git diff --check
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release -p infer --features cuda
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo clippy --release -p infer --features cuda -- -D warnings
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_solo_vs_concurrent -- --test-threads=1
```

`test_greedy_solo_vs_concurrent` passed. The full `greedy_consistency` binary
still includes the existing W4A8-vs-BF16 accuracy gate failure and is not a
Task #35 regression signal.

## Bench — sustained-load smoke

Matched controls:

- Baseline: `INFER_PREFILL_WARMUP=0`
- Treatment: default Pass 3 enabled
- Command shape: `scripts/bench_guidellm.sh 35-warmup-{off,on}-rN`
- Workload: `512-in / 128-out`, `--concurrencies 1,2,4`, `--max-seconds 30`,
  `--warmup 0`
- N=3 fresh server restarts per arm.

| arm | conc | TTFT p50 mean | TTFT p50 CV | ITL p50 mean | out tok/s mean |
|---|---:|---:|---:|---:|---:|
| Pass 3 off | 1 | 66.0 ms | 0.09% | 5.80 ms | 159.76 |
| Pass 3 on | 1 | 66.4 ms | 0.23% | 5.80 ms | 159.53 |
| Pass 3 off | 2 | 79.0 ms | 3.85% | 7.44 ms | 245.96 |
| Pass 3 on | 2 | 80.8 ms | 3.36% | 7.44 ms | 245.92 |
| Pass 3 off | 4 | 157.2 ms | 0.28% | 8.31 ms | 423.82 |
| Pass 3 on | 4 | 157.7 ms | 0.19% | 8.03 ms | 430.58 |

Short sustained-load p50 is effectively unchanged, which is expected: this
smoke validates no steady-state regression, not the full W4 8k first-burst
turn-success claim.

These sustained-load rows were collected before the review-driven production
shape correction. They remain a steady-state regression guard because the live
request path is unchanged, but the startup-cost section below is the current
post-review warmup behavior.

## Startup Cost

| arm | warmup total N=3 | mean | CV |
|---|---|---:|---:|
| Pass 3 off | 1043 / 1051 / 1046 ms | 1046.7 ms | 0.39% |
| Pass 3 on | 9221 / 9244 / 9235 ms | 9233.3 ms | 0.13% |

Measured startup overhead after production-shape correction: `+8186.6 ms`.
The Pass 3 log itself reported `8179 / 8199 / 8193 ms`.

At `B=8`, the first attempt at the configured 2048 tokens/row exceeded the
Marlin prefill scratch envelope and backed off to 1024 tokens/row:

```text
Pass 3 prefill warmup for B=8 at 2048 tokens/row failed (... CUDA_ERROR_OUT_OF_MEMORY ...); retrying at 1024 tokens/row
Pass 3 prefill warmup done in 8179ms (8 batch sizes, max 8)
```

The initial scout implementation used a fixed 64-token row and measured only
`+282.7 ms` startup overhead:

```text
Pass 3 prefill warmup done in 308ms (8 batch sizes, max 8)
```

`codex review --uncommitted` correctly identified that this was not warming the
production packed-token shapes and that graph-prefill mode would warm a
throwaway context. The shipped implementation uses the configured row shape and
the persistent scheduler prefill context.

## Problems

- A first W4 agent-trace attempt used `--model default` and was invalid: the
  server returned 404 for every request and `/v1/stats` remained at
  `requests=0`.
- A corrected W4 agent-trace run was valid but long-running rather than a short
  smoke: after several minutes it had `requests=127`, `active=8`, and
  `kv_util=100.0%`. I stopped it to avoid holding the GPU and did not count it
  as license data.

## Learnings

- Production-shape warmup is not cheap on this model: it adds about 8.2s to
  startup on the 4070 Ti SUPER. That cost is acceptable only if the first-burst
  cap=8 workload is the priority.
- The correct acceptance workload for the original cap=8 bimodal investigation
  is the full W4 8k trace; the 512-token GuideLLM smoke is only a regression
  guard for conc 1/2/4.
- `bench_agent_trace.py` must use the served model id
  `Qwen3-4B-W4A16-sym-g128-marlin`; `default` is no longer accepted by the
  OpenAI model validator.
- Review value-add was load-bearing: fixed both the under-sized 64-token warmup
  and the temporary prefill-context path before commit.

## Artefacts

- `bench-output/2026-05-10-35-warmup-off-r1/`
- `bench-output/2026-05-10-35-warmup-off-r2/`
- `bench-output/2026-05-10-35-warmup-off-r3/`
- `bench-output/2026-05-10-35-warmup-on-r1/`
- `bench-output/2026-05-10-35-warmup-on-r2/`
- `bench-output/2026-05-10-35-warmup-on-r3/`
- Server logs: `/tmp/infer-35-{off,on}-r{1,2,3}.log`
- Post-review startup logs: `/tmp/infer-35-backoff-on-r{1,2,3}.log`

## Rule

Startup warmup changes need two gates: a short sustained-load regression smoke
for conc 1/2/4, and a separate full first-burst workload for the workload that
originally exposed the bimodal failure. Do not substitute one for the other.
