# Qwen3-4B regression bench — paged-prefill reverted + HD128/HD256 CUDA fixes

> Canonical guidellm sweep. Purpose of this run: confirm that commits
> `190baf4` / `f08d265` / `927c390` (HD256 workspace bump, `total_num_rows`
> device-pointer wires for HD128+HD256, FFI try/catch, workspace size
> per-path, Qwen3+Qwen3.5 paged-prefill flag reverted to `false`) do
> **not** regress the stable Qwen3-4B contiguous path vs the
> 2026-04-17 baseline.

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B (bf16)
- **Hardware:** NVIDIA L4 24GB, CUDA 13.0, driver 580.82.07, SM 8.9
- **Commit:** `927c390`
- **Feature set:** `cargo build --release --features cuda -p infer --bin infer`
- **Non-default flags / env vars:** `--num-slots 10 --max-seq-len 5120
  --mem-fraction-static 0.88`. Server log at `RUST_LOG=warn`.
- **Server launch:**
  ```bash
  ./target/release/infer --model-path models/Qwen3-4B --port 8000 \
      --num-slots 10 --max-seq-len 5120 --mem-fraction-static 0.88
  ```
- **Prior snapshot:**
  [`2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md`](2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md)
  (commit `cae7d38`, same hardware, same canonical params).

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --profile sweep \
  --data  prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-18-paged-phase1-qwen3-run4/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh paged-phase1-qwen3`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 878.7 | 897.8 | 35.38 | 35.44 | 26.22 | 0.1 |
| throughput | 13139.2 | 37757.5 | 78.62 | 101.68 | 97.43 | 0.383 |
| 0.135r/s | 1212.6 | 1228 | 41.02 | 41.12 | 32.47 | 0.117 |
| 0.171r/s | 1232.8 | 1244.5 | 45.16 | 45.63 | 39.13 | 0.15 |
| 0.206r/s | 1233.5 | 1250.3 | 47.34 | 47.39 | 46.23 | 0.167 |
| 0.241r/s | 1245 | 1267.3 | 52.16 | 52.23 | 51.89 | 0.183 |
| 0.277r/s | 1261.6 | 1285.2 | 57.41 | 57.56 | 56.97 | 0.217 |
| 0.313r/s | 1273.4 | 1322.9 | 64.29 | 64.45 | 61.27 | 0.233 |
| 0.348r/s | 1297.2 | 1324.2 | 69 | 69.35 | 65.78 | 0.25 |
| 0.383r/s | 1302.1 | 1367.6 | 74.69 | 76.03 | 69.39 | 0.267 |

## Artefacts

- Raw: `bench-output/2026-04-18-paged-phase1-qwen3-run4/benchmarks.json`
- CSV:  `bench-output/2026-04-18-paged-phase1-qwen3-run4/benchmarks.csv`
- HTML: `bench-output/2026-04-18-paged-phase1-qwen3-run4/benchmarks.html`

## Delta vs 2026-04-17 baseline (same hardware, same canonical params)

| metric | 2026-04-17 | 2026-04-18 | Δ% |
|---|---:|---:|---:|
| TTFT p50 @ sync | 850.1 ms | 878.7 ms | **+3.4%** |
| TTFT p99 @ sync | 871.4 ms | 897.8 ms | +3.0% |
| TTFT p50 @ 0.135 r/s | 1205.5 ms | 1212.6 ms | +0.6% |
| TTFT p50 @ 0.241 r/s | 1242.6 ms | 1245 ms | +0.2% |
| TTFT p50 @ 0.383 r/s | 1292.6 ms | 1302.1 ms | +0.7% |
| TTFT p99 @ 0.241 r/s | 1266.8 ms | 1267.3 ms | +0.04% |
| ITL p99 @ sync | 35.32 ms | 35.44 ms | +0.3% |
| ITL p99 @ 0.241 r/s | 52.08 ms | 52.23 ms | +0.3% |
| ITL p99 @ throughput | 101.81 ms | 101.68 ms | -0.1% |
| peak out tok/s | 97.91 | 97.43 | -0.5% |

**Verdict:** at parity with 2026-04-17 (all deltas ≤3.4%, well inside
run-to-run noise). The CUDA fixes and paged-prefill flag revert do not
regress the stable path.

## What changed in the code since the baseline

Between `cae7d38` (Apr 17) and `927c390` (this run), the relevant set of
commits on the Qwen3 path is:

- `1551893` / `9821cb2` / `fb8f3a4` (Phase 2 + 3a): migrated Qwen3 prefill
  onto the paged-KV pool path. Introduced the `prefill_uses_paged_pool
  → true` flag.
- `859c3d2` (Phase 1A): same migration for Qwen3.5 HD256 layers.
- `3702434` / `7a5a962` (fix): HD256 FlashInfer workspace bump to 512 MiB
  + C++ try/catch + `total_num_rows` device-pointer wire.
- `927c390` (fix): same two fixes applied to HD128; `FlashInferWorkspace`
  size made caller-configurable; Qwen3.5 paged flag reverted to `false`.
- **Qwen3 `prefill_uses_paged_pool` set back to `false` in `927c390`**
  as well — this is what brings the code path back to the Apr 17 shape.
  The paged path is still kept in the tree (re-enabling is a one-line
  flip) but disabled until the scheduler-level slot-reuse bug is fixed.

## Notes

- **Paged prefill stays disabled for now.** Even with the three kernel
  fixes landed, HD128/HD256 `PrefillPlan` overflows `batch_prefill_tmp_s`
  under concurrent-sweep load at 10 slots × 4096 tokens, and there is a
  separate scheduler bug on the slot-reuse + radix-hit path for
  `supports_partial_prefix=false` models. Both are tracked in
  `docs/experience/errors/2026-04-18-paged-prefill-workspace-and-plan-bugs.md`
  and `docs/plans/p99-unified-mixed-batch.md` §Phase 1C.
- **What did ship:** the two-fix pattern (try/catch + `total_num_rows`)
  is now applied to both HD128 and HD256 FlashInfer paged-prefill FFI
  boundaries, and `FlashInferWorkspace` can be sized per call site.
  Those are general-purpose correctness improvements that remain live
  when Phase 1A/3a is eventually re-enabled.
- **Follow-ups:**
  - Phase 1C scheduler fix: pool/seq_len reset on slot reuse when
    `supports_partial_prefix=false` and the admission path is
    MISS-despite-radix-hit.
  - Revisit FlashInfer workspace sizing: why does `batch_prefill_tmp_s`
    need more than 512 MiB when sglang uses 512 MiB at similar scale?
    Likely `padded_batch_size × num_splits × num_qo_heads × head_dim`
    sizing interacts with our 10-slot × 4096-row configuration
    differently than sglang's.
