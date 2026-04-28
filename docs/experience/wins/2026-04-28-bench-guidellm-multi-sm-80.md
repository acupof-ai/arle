# Multi-SM Phase D — guidellm sweep on sm_80 (A100), 2026-04-28 (pending-remote)

> **Status: pending-remote.** Commit A landed the tier policy +
> `TORCH_CUDA_ARCH_LIST` migration; this entry retires when the multi-SM
> binary (commits B + C) is bench-validated on a real A100 box per
> [`docs/plans/sm-coverage.md`](../../plans/sm-coverage.md) §5.

## Goal

- Validate that the T1 fat binary built with
  `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` runs the canonical
  guidellm sweep on **sm_80 (A100)** with TTFT and out_tok/s within
  ±5 % of the most recent same-SM baseline.

## Hypothesis

- A100 baseline does not exist yet; first multi-SM run on sm_80 *is* the
  baseline. Subsequent runs delta against this.

## Command

```bash
scripts/bench_guidellm.sh cuda-multi-sm-80 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --processor models/Qwen3-8B
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-8B
- **Hardware:** A100 (40 or 80 GB; record exact SKU when filled)
- **Commit:** TBD — multi-SM commit B + C merge sha
- **Feature set:** `cargo build --release --features cuda` (no `tilelang-attn` for first run; add as Phase D.2 if time)
- **Non-default flags / env vars:** `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
- **Server launch:** `scripts/start_infer.sh Qwen3-8B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — sweep headline table

_Pending. Expected first run on remote A100._

## Δ vs baseline

- **Baseline:** none yet (this is the first sm_80 multi-SM run; subsequent
  same-SM entries delta against this).
- Cross-SM comparison only as sanity check; T1 ship gate is per-SM ±5 %
  not cross-SM equality.

## Notes

- Bench gate context: this is one of four ship-blocking entries (sm_80 +
  sm_86 + sm_89 + sm_90) that must land before commits B + C are
  considered shipped per
  [`docs/plans/sm-coverage.md`](../../plans/sm-coverage.md) §5.
- If A100 access is delayed, leaving this stub `pending-remote` is
  acceptable; the other three SMs can ship independently and this row
  retires last.
