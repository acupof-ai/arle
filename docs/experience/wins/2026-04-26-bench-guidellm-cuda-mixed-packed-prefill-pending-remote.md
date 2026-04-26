# CUDA Mixed Packed Prefill — guidellm sweep, cuda-qwen3, 2026-04-26

## Goal

- Regression-check the CUDA Qwen3 scheduler/model refactor that changes mixed decode+prefill from one prefill row to packed prefill rows and routes paged prefill appends through shared-tail COW with launch-time budget refresh.

## Hypothesis

- BF16 mixed batches should keep decode behavior stable while reducing split launches when multiple prefill chunks are admissible with active decode rows, without corrupting radix-shared prefix tail pages.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen3-mixed-packed-prefill
```

Invoked via: pending remote CUDA host.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote NVIDIA host
- **Commit:** pending
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** none expected
- **Server launch:** `scripts/start_infer.sh models/Qwen3-4B 8000` or equivalent

## Canonical Params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-qwen3-mixed-packed-prefill`

## Results

- Status: pending-remote. Local machine cannot execute CUDA kernels.

## Problems

- CUDA runtime validation is pending because this workspace is on macOS without NVIDIA/CUDA execution.

## Learnings

- Keep mixed-batch support format-gated by `KVFormat`; unsupported quantized KV formats must fall back to split prefill+decode instead of silently entering a BF16-only mixed lowering.

## Delta vs Baseline

- **Baseline:** docs/experience/wins/2026-04-24-bench-guidellm-cuda-mixed-retract-page-budget-pending-remote.md
- **Delta table:** pending remote CUDA run.

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in the code since baseline: mixed scheduler/model contract now accepts packed prefill rows in one BF16 FlashInfer mixed batch; paged prefill allocation and page budgeting now account for shared hot-tail COW before append; launch paths re-check current pool budget before issuing prefill work and fall back to split prefill+decode if a packed mixed batch is rejected.
- Suspected cause of any regression: mixed prefill row packing, per-row KV page-table metadata, or COW detach overhead under prefix attach.
- Follow-ups: run CUDA guidellm sweep and compare TTFT/ITL/output tok/s against the mixed-retract baseline.
