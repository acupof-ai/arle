# 2026-04-13 · Qwen3-4B L4-24GB — `baseline-main-2026-04-13` regression baseline

## Context
General regression baseline for `main` as of 2026-04-13 (§7.1 of
`docs/plans/tiered-kv-cache-tasks.md`). Becomes the reference snapshot
for comparing every subsequent main commit on L4 for throughput / TTFT /
ITL. Run on a fresh server process (separate from the `page1` run in
`2026-04-13-bench-page1.md`) so the two snapshots are independent.

Paired artifact: `2026-04-13-bench-baseline-main.json` (raw snapshot in
the same directory).

## Environment
- GPU: NVIDIA L4 24GB (driver 580.82.07, CUDA 13.0)
- Model: Qwen3-4B BF16, HuggingFace `Qwen/Qwen3-4B` (Instruct variant)
- Commit: `876b986` (after `git pull`, tree at 2026-04-13 local batch head)
- Server: `target/release/infer --model-path models/Qwen3-4B --num-slots 4 --port 8000`
  - `cuda_graph=true`, warm batch sizes {1,2,4}
  - `max_seq_len=4096` (auto), num_slots=4 (explicit)
  - `kv_cache_dtype=bf16`, TokenKVPool 24,888 tokens / 3.7 GB BF16
- Bench tool: `scripts/bench_throughput_sweep.py --label baseline-main-2026-04-13`
- Feature set: default (`cuda`)
- Build: `cargo build --release` with FlashInfer 0.6.3 + Triton 3.5.1 AOT

## Results

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err |   Wall
--------------------------------------------------------------------------------------------
  128 |    64 |  1 |     29.9 t/s |      37ms |      46ms |   33.2ms |   33.2ms |   0 |  15.4s
  128 |   128 |  1 |     30.1 t/s |      37ms |      44ms |   33.2ms |   33.3ms |   0 |  34.1s
  128 |   256 |  1 |     29.9 t/s |      37ms |      44ms |   33.3ms |   33.4ms |   0 |  54.0s
  128 |   512 |  1 |     29.7 t/s |      37ms |      44ms |   33.7ms |   33.7ms |   0 |  98.8s
  512 |   128 |  1 |     29.4 t/s |      39ms |      89ms |   33.9ms |   34.0ms |   0 |  34.9s
  512 |   256 |  1 |     29.3 t/s |      39ms |      88ms |   34.1ms |   34.1ms |   0 |  61.8s
  512 |   512 |  1 |     29.1 t/s |      39ms |      71ms |   34.4ms |   34.4ms |   0 | 123.3s
 1024 |   128 |  1 |     28.5 t/s |      41ms |     146ms |   34.8ms |   34.8ms |   0 |  31.4s
 1024 |   256 |  1 |     28.5 t/s |      41ms |     149ms |   35.0ms |   35.0ms |   0 |  63.0s
 1024 |   512 |  1 |     28.3 t/s |      41ms |     143ms |   35.2ms |   35.3ms |   0 | 126.5s
 2048 |   256 |  1 |     27.1 t/s |      46ms |     278ms |   36.6ms |   36.7ms |   0 |  57.0s
  512 |   256 |  2 |     57.1 t/s |     110ms |     198ms |   34.6ms |   34.6ms |   0 |  35.8s
  512 |   256 |  4 |    111.5 t/s |     184ms |     418ms |   34.8ms |   34.9ms |   0 |  18.4s
  128 |   128 |  2 |     54.8 t/s |     106ms |     120ms |   33.9ms |   33.9ms |   0 |  17.6s
  128 |   128 |  4 |    105.4 t/s |     165ms |     208ms |   34.0ms |   34.0ms |   0 |   3.4s
  128 |   256 |  8 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.0s
  512 |   256 |  8 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.0s
  128 |   256 | 16 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.1s
  512 |   256 | 16 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.1s
  128 |   256 | 32 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.1s
  512 |   256 | 32 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.2s
  128 |   256 | 64 |      0.0 t/s |       0ms |       0ms |    0.0ms |    0.0ms |   0 |   0.2s
```

Peak throughput: **111.5 tok/s** (512 in / 256 out, C=4)
Peak (C=1): **30.1 tok/s** (128 in / 128 out)
ITL p50 range (C=1–4 valid rows): **33.2 ms – 36.6 ms**
TTFT p50 (C=1, 128 in): **37 ms**

## Delta vs. `2026-04-13-bench-page1.md` (same main, 15 min earlier)
The `page1` and `baseline-main` snapshots measure the same commit on a
cold-started server. Differences are pure run-to-run noise.

| Metric | page1 | baseline-main | Δ |
|---|---|---|---|
| Peak C=4 throughput (512/256) | 111.4 | 111.5 | +0.1 % |
| Peak C=1 throughput (128/128) | 30.1 | 30.1 | 0 |
| ITL p50 floor | 33.0 ms | 33.2 ms | +0.6 % |
| TTFT p50 (128 in C=1) | 37 ms | 37 ms | 0 |
| 2048/256 C=1 TTFT p50 | 46 ms | 46 ms | 0 |

Both snapshots are interchangeable for regression comparison on L4.
Use this one (`baseline-main-2026-04-13`) as the canonical reference
going forward; `page1` stays the explicit P0 delta target.

## Known issue (same as `2026-04-13-bench-page1.md`)
C≥8 rows all 0 tok/s because the server's CUDA context dies once a
batched-decode step hits `CUDA_ERROR_ILLEGAL_ADDRESS` during batched
sampling. First crash in this run at the transition into the
`128/256 C=8` config (~775 s wall time). Not caused by A/B/C/D/E/F/G/H/I
— same failure reproduces at the pre-A commit `37a8a82`. Tracked in
`docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md`.

## Rule
- L4 24GB `main` baseline (2026-04-13): 30 tok/s C=1, ~111 tok/s C=4 at
  BF16 KV, num_slots=4. Matches pre-P0 baseline within noise.
- Any subsequent main commit dropping these numbers by >2 % on C=1
  configs or >5 % on C=4 is a regression — investigate before merging.
- Valid data rows on this server are C=1 through C=4. C≥8 is
  structurally broken until the batched-decode regression is fixed.
