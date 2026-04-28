# Multi-SM Phase D — guidellm sweep on sm_90 (H100), 2026-04-28 (pending-remote)

> **Status: pending-remote.** Commit A landed the tier policy +
> `TORCH_CUDA_ARCH_LIST` migration; this entry retires when the multi-SM
> binary (commits B + C) is bench-validated on a real H100 box per
> [`docs/plans/sm-coverage.md`](../../plans/sm-coverage.md) §5.

## Goal

- Validate that the T1 fat binary built with
  `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` runs the canonical
  guidellm sweep on **sm_90 (H100)** with TTFT and out_tok/s within
  ±5 % of the most recent same-SM baseline. H100 is the SM where
  TileLang's TMA / WGMMA / warp-spec leverage fires hardest, so cubin
  selection correctness matters most here.

## Hypothesis

- H100 baseline lives in
  [`docs/plans/tilelang-integration-verification.md`](../../plans/tilelang-integration-verification.md)
  §4 / §5 (Phase 0 wins entries). Multi-SM build should match within
  ±5 % since dispatch picks the same per-SM cubin that single-SM build
  produces.

## Command

```bash
scripts/bench_guidellm.sh cuda-multi-sm-90 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3.5-4B (matches Phase 0 H100 reference workload)
- **Hardware:** H100 80 GB SXM or PCIe (record exact SKU when filled)
- **Commit:** TBD — multi-SM commit B + C merge sha
- **Feature set:** `cargo build --release --features cuda,tilelang-attn`
  (TileLang is calibrated for sm_90; this is the canonical TileLang bench host)
- **Non-default flags / env vars:** `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
- **Server launch:** `scripts/start_infer.sh Qwen3.5-4B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — sweep headline table

_Pending._

## Δ vs baseline

- **Baseline:** Phase 0 H100 entry (link to be added once Phase 0 lands a
  canonical H100 wins file; today the closest reference is the floor
  L4 entry plus Phase 0 plan §5 thresholds).
- **Pass criterion:** TTFT p50 / p99 and out_tok/s saturation within ±5 %
  of single-SM `TORCH_CUDA_ARCH_LIST="9.0"` rebuild on the same host.

## Notes

- Tilelang Phase 0 §5 ship/revert thresholds **continue to apply** on top
  of this gate — multi-SM build must not regress single-SM Phase 0 H100
  numbers, and Phase 0 ship/revert decisions remain H100-driven (per
  `tilelang-integration-verification.md` §0).
- A/B: rebuild same binary with `TORCH_CUDA_ARCH_LIST="9.0"` and re-run
  if the multi-SM number is suspect, per
  `feedback_matched_ab_for_small_bench_effects.md`.
