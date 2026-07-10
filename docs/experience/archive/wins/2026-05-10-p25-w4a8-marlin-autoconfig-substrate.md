# P2.5 — W4A8 Marlin Auto-Config Substrate

## Context

P2.5/M'' targets the QQQ `thread_config_t` schedule-selection pattern for
ARLE's W4A8 Marlin GEMM. The source audit found a tradeoff: QQQ has a cleaner
runtime selector, while ARLE already has sm_89-specific L2 cache-hint
`cp.async` and broader compile-time config coverage.

## What Worked

- Added a QQQ-style `thread_config_t` selector to
  `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`.
- Preserved ARLE's `cp_async4_stream` / `cp_async1_stream` cache-hint paths.
- Preserved the historical default by having Rust pass explicit legacy
  `(thread_k, thread_n)` values unless `INFER_MARLIN_W4A8_AUTOCONFIG=1`.
- Added `INFER_MARLIN_W4A8_AUTOCONFIG=1` as an opt-in A/B switch for the
  QQQ-style selector.

## Correctness

Both arms passed the W4A8-vs-BF16 accuracy gate with the qzeros-fixed W4A8
fixture:

| Arm | Env | Result |
|---|---|---|
| Legacy explicit config | unset | 32/32 token match, 0.0% diff |
| Auto-config selector | `INFER_MARLIN_W4A8_AUTOCONFIG=1` | 32/32 token match, 0.0% diff |

## Bench Status

Status: `landed-bench`.

Matched-control A/B used the qzeros-fixed W4A8 fixture
`infer/models/Qwen3-4B-GPTQ-W4A8-zpfix`, `prompt_tokens=4096`,
`output_tokens=256`, `--max-seconds 120`, `--warmup 10`, and seed
`20260416`. GPU was checked idle before each shot. The wrapper ran in
fixed-concurrency exploration mode, so snapshots were filled manually from raw
artefacts.

Primary pair:

| Shape | TTFT mean | TTFT p99 | ITL mean | ITL p99 | out tok/s | total tok/s | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| c=1 | 415.4 -> 416.1 ms (+0.17%) | 747.2 -> 756.8 ms (+1.28%) | 13.57 -> 13.56 ms (-0.07%) | 13.59 -> 13.58 ms (-0.07%) | 66.29 -> 66.29 (+0.00%) | 1127.12 -> 1127.18 (+0.01%) | PASS |
| c=4 | 1738.0 -> 1760.6 ms (+1.30%) | 1864.6 -> 2034.3 ms (+9.10%) | 17.30 -> 17.33 ms (+0.17%) | 17.46 -> 17.48 ms (+0.11%) | 168.75 -> 168.62 (-0.08%) | 2869.46 -> 2867.21 (-0.08%) | RETEST: TTFT p99 tail outlier |

c=4 matched repeat:

| Shape | TTFT mean | TTFT p99 | ITL mean | ITL p99 | out tok/s | total tok/s | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| c=4 repeat | 1707.7 -> 1706.6 ms (-0.06%) | 1889.7 -> 1883.9 ms (-0.31%) | 17.33 -> 17.33 ms (+0.00%) | 17.48 -> 17.48 ms (+0.00%) | 169.38 -> 169.48 (+0.06%) | 2880.11 -> 2881.84 (+0.06%) | PASS |

Conservative c=4 run-level average across the primary and repeat pairs:

| Metric | Legacy avg | Auto-config avg | Delta |
|---|---:|---:|---:|
| TTFT mean | 1722.9 ms | 1733.6 ms | +0.62% |
| TTFT p99 | 1877.2 ms | 1959.1 ms | +4.37% |
| ITL mean | 17.32 ms | 17.33 ms | +0.09% |
| ITL p99 | 17.47 ms | 17.48 ms | +0.06% |
| out tok/s | 169.07 | 169.05 | -0.01% |
| total tok/s | 2874.79 | 2874.53 | -0.01% |

Bench snapshots:

- [`2026-05-10-bench-guidellm-p25-w4a8-baseline.md`](2026-05-10-bench-guidellm-p25-w4a8-baseline.md)
- [`2026-05-10-bench-guidellm-p25-w4a8-autoconfig.md`](2026-05-10-bench-guidellm-p25-w4a8-autoconfig.md)
- [`2026-05-10-bench-guidellm-p25-w4a8-baseline-c4-rerun.md`](2026-05-10-bench-guidellm-p25-w4a8-baseline-c4-rerun.md)
- [`2026-05-10-bench-guidellm-p25-w4a8-autoconfig-c4-rerun.md`](2026-05-10-bench-guidellm-p25-w4a8-autoconfig-c4-rerun.md)

License decision: LICENSE the auto-config selector as opt-in only. The stable
evidence is neutral within +/-5%, but there is no >=2% latency improvement, so
do not promote it to default. Production default remains the legacy explicit
tile choice; `INFER_MARLIN_W4A8_AUTOCONFIG=1` stays as the A/B switch.

## Rule

When upstream schedule selection is a tradeoff, land it behind an explicit
A/B switch and keep the local architecture-specific fast path as the default
until bench data proves the new selector.
