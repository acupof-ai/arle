# guidellm sweep cuda-graph-tilelang-total-pages — pending, 2026-05-06

## Goal
Regression check for the Qwen3/Qwen3.5 CUDA Graph TileLang decode fix.

## Hypothesis
Using the static KV-pool page capacity for TileLang decode attention should
preserve CUDA Graph replay correctness after decode crosses page boundaries,
with no intended throughput regression beyond normal noise.

## Command
Pending canonical GuideLLM run:

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 4 \
  --max-seq-len 4096

scripts/bench_guidellm.sh cuda-graph-tilelang-total-pages \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment
- **Status:** pending-full-sweep
- **GPU:** RTX 4070 Ti SUPER (`sm_89`, 16 GiB)
- **CUDA:** 13.2
- **Compiler override:** `NVCC_CCBIN=/usr/bin/g++-14`
- **TileLang Python:** `/home/ckl/projects/arle/.venv/bin/python`
- **Model:** Qwen3-4B BF16, `infer/models/Qwen3-4B`
- **Feature set:** `cargo --release -p infer --features cuda`

## Results
Correctness gate completed before the full sweep:

| Check | Result |
|---|---|
| `cargo test --release -p infer --features cuda --test e2e -- --nocapture` | pass |
| `cargo test --release -p infer --features cuda --test greedy_consistency -- --nocapture` | still fails on coherent B=1/B=3 trajectory divergence; graph gibberish gone |

GuideLLM throughput table pending.

## Problems
The remaining `greedy_consistency` exact-equality failure is tracked in
`docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md`.
The graph-specific failure mode is fixed; B=3 exact greedy equality remains a
separate numerical-consistency/test-contract issue.

## Learnings
CUDA Graph captures host launch scalars by value. TileLang decode kernels
should receive graph-stable shape bounds when replay is expected across decode
steps; per-request validity should come from device metadata such as
`kv_indptr`.

## Δ vs baseline
Pending. Compare against the latest CUDA Qwen3-4B GuideLLM baseline for this
host/backend after the full sweep runs.
