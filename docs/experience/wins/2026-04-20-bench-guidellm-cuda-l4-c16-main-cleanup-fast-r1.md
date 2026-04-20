# main cleanup fast run — guidellm, cuda-l4-c16-main-cleanup-fast-r1, 2026-04-20

## Goal

- Regression check: verify that the CUDA scheduler shutdown cleanup on `main` does not break c16 serving throughput, and confirm that process exit releases CUDA resources cleanly.

## Hypothesis

- Joining the scheduler thread on shutdown is teardown-only and should not materially change steady-state c16 throughput.
- After `Ctrl+C`, the server should drop the last `SchedulerHandle`, let the scheduler exit, and release the `infer` GPU allocation.

## Command

```bash
CARGO_TARGET_DIR=/tmp/agent-infer-main-target cargo run -p infer --release -- \
  --model-path models/Qwen3-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --cuda-graph=false

./scripts/bench_guidellm.sh cuda-l4-c16-main-cleanup-fast-r1 \
  --fast \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `NVIDIA L4`, `23034 MiB`, driver `580.82.07`, CUDA `13.0`
- **Commit:** `d73549b`
- **Feature set:** `cargo build -p infer --release --offline`
- **Non-default flags / env vars:** `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --cuda-graph=false`
- **Server launch:** `cargo run -p infer --release -- ...`

## Results

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc16 | 0 | 0 | 0 | 0 | 418.62 | 1.4 |

## Problems

- `guidellm` reported `TTFT/ITL=0` for this `main` run, so the throughput numbers are usable but the latency fields are not trustworthy for comparison.
- `main` still captured CUDA graphs during warmup even with `--cuda-graph=false`; the log showed batched decode graph capture for `B=1..16`.
- The wrapper's old default tokenizer path (`models/Qwen3-4B`) was stale on `main`; this run used the explicit correct path `infer/models/Qwen3-4B`.

## Learnings

- The shutdown cleanup is correctly isolated from the request path: the server sustained `conc16` generation while the new teardown path only ran after signal handling.
- The robust way to verify CUDA cleanup is end-to-end: `Ctrl+C` the server, wait for the scheduler thread join log, then confirm `nvidia-smi` no longer shows an `infer` allocation.
- `guidellm` synthetic-text runs on `main` need a valid local tokenizer path; if the processor path is wrong, the benchmark fails before any requests are sent.

## Δ vs baseline

- First `main` fast c16 cleanup run after porting the shutdown guard. Nearest branch-side c16 reference exists, but the `TTFT/ITL=0` reporting bug on this run makes latency deltas non-comparable.

## Artefacts

- Raw: `bench-output/2026-04-20-cuda-l4-c16-main-cleanup-fast-r1-run3/benchmarks.json`
- CSV: `bench-output/2026-04-20-cuda-l4-c16-main-cleanup-fast-r1-run3/benchmarks.csv`
- HTML: `bench-output/2026-04-20-cuda-l4-c16-main-cleanup-fast-r1-run3/benchmarks.html`

## Notes

- Shutdown verification logs:
  - `Shutdown signal received`
  - `Scheduler shutting down: all handles dropped`
  - `Waiting for scheduler thread to shut down cleanly (model=Qwen3-4B)`
  - `Scheduler thread shut down cleanly (model=Qwen3-4B)`
- Post-shutdown `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` showed only `/usr/bin/python3, 220 MiB`; no `infer` process remained on the GPU.
