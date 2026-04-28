# Multi-SM Phase D — guidellm sweep on sm_89 (L4/4090), 2026-04-28 (pending-remote)

> **Status: pending-remote.** Commit A landed the tier policy +
> `TORCH_CUDA_ARCH_LIST` migration; this entry retires when the multi-SM
> binary (commits B + C) is bench-validated on a real L4 / 4090 box per
> [`docs/plans/sm-coverage.md`](../../plans/sm-coverage.md) §5.

## Goal

- Validate that the T1 fat binary built with
  `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` runs the canonical
  guidellm sweep on **sm_89 (L4 or RTX 4090)** with TTFT and out_tok/s
  within ±5 % of the most recent same-SM baseline.

## Hypothesis

- L4 has a recent bench baseline:
  [`2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md`](2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md).
  Multi-SM build should match within ±5 % since dispatch picks the same
  per-SM cubin that single-SM build produces.

## Command

```bash
scripts/bench_guidellm.sh cuda-multi-sm-89 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-0.8B \
  --processor models/Qwen3.5-0.8B
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3.5-0.8B Q4_K_M (matching the existing baseline) +
  optional Qwen3-8B repeat for cross-model sanity
- **Hardware:** L4 24 GB or RTX 4090 24 GB (record exact SKU when filled)
- **Commit:** TBD — multi-SM commit B + C merge sha
- **Feature set:** `cargo build --release --features cuda,tilelang-attn`
  (L4 has the most TileLang validation evidence; toggle on for first run)
- **Non-default flags / env vars:** `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
- **Server launch:** `scripts/start_infer.sh Qwen3.5-0.8B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — sweep headline table

_Pending._

## Δ vs baseline

- **Baseline:** [`2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md`](2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md)
  (CUDA L4 Qwen3.5-0.8B GGUF Q4_K_M: 183.3 out tok/s at c=1 and
  222.2 out tok/s at c=2; c>=4 invalid due to allocator/OOM behavior).
- **Pass criterion:** out_tok/s saturation and TTFT p50 within ±5 % of
  the baseline numbers. Multi-SM cubin dispatch should be neutral.

## Notes

- This is the highest-priority bench in the gate because L4 already has
  a baseline and is the SM with the most multi-month bench history; a
  regression here is the strongest signal that multi-SM cubin dispatch
  introduced overhead or wrong-cubin selection.
- Re-run the same bench against single-SM build (`TORCH_CUDA_ARCH_LIST="8.9"`)
  for an A/B if the multi-SM number is suspect — same binary same env,
  per `feedback_matched_ab_for_small_bench_effects.md`.
