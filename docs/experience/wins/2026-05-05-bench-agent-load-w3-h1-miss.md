# W3 H1 Gate Miss - agent-load benchmark, agent-w3-short-multiturn, arle-cuda, 2026-05-05

## Goal

Validate the guarded W3 H1 page-size-16 decode metadata fast path against the
canonical `agent-w3-short-multiturn` trace on the L4 CUDA box.

Goal type: optimization / regression check.

## Hypothesis

With `INFER_DECODE_METADATA_FAST_PAGE16=1`, page-size-16 FP8 decode should avoid
steady-state full metadata rebuilds and improve W3 output throughput by at
least `1.10x` while not regressing warm TTFT p99.

## Command

Build and code verification before the bench:

```bash
source /tmp/arle-env.sh
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda --bin infer
CUDA_HOME=/usr/local/cuda cargo clippy --release -p infer --features cuda -- -D warnings
CUDA_HOME=/usr/local/cuda cargo test --release -p infer --features cuda flashinfer::
codex review --uncommitted
codex review --uncommitted
codex review --uncommitted
```

Baseline server:

```bash
source /tmp/arle-env.sh
INFER_DECODE_METADATA_FAST_PAGE16=0 RUST_LOG=info CUDA_HOME=/usr/local/cuda \
  ./target/release/infer --model-path models/default --port 8000 \
  --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85 \
  --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 \
  --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096 \
  > bench-output/2026-05-05-w3-h1-baseline/server.log 2>&1
```

Baseline client:

```bash
source /tmp/arle-env.sh
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://127.0.0.1:8000 \
  --label w3-h1-baseline-a9944b5c \
  --out bench-output/2026-05-05-w3-h1-baseline/results.json \
  --trace-out bench-output/2026-05-05-w3-h1-baseline/trace.jsonl \
  > bench-output/2026-05-05-w3-h1-baseline/client.log 2>&1
curl -s http://127.0.0.1:8000/v1/stats \
  > bench-output/2026-05-05-w3-h1-baseline/stats_after.json
```

Fast-path server:

```bash
source /tmp/arle-env.sh
INFER_DECODE_METADATA_FAST_PAGE16=1 RUST_LOG=info CUDA_HOME=/usr/local/cuda \
  ./target/release/infer --model-path models/default --port 8000 \
  --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85 \
  --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 \
  --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096 \
  > bench-output/2026-05-05-w3-h1-fastpath/server.log 2>&1
```

Fast-path client:

```bash
source /tmp/arle-env.sh
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://127.0.0.1:8000 \
  --label w3-h1-fastpath-a9944b5c \
  --out bench-output/2026-05-05-w3-h1-fastpath/results.json \
  --trace-out bench-output/2026-05-05-w3-h1-fastpath/trace.jsonl \
  > bench-output/2026-05-05-w3-h1-fastpath/client.log 2>&1
curl -s http://127.0.0.1:8000/v1/stats \
  > bench-output/2026-05-05-w3-h1-fastpath/stats_after.json
```

## Environment

- **Workload:** `agent-w3-short-multiturn`.
- **Backend / engine:** `arle-cuda`.
- **Model:** `models/default -> Qwen3-4B`.
- **Tokenizer / processor:** local Qwen3 tokenizer under `models/default`.
- **Hardware:** NVIDIA L4, 23,034 MiB VRAM.
- **Driver / CUDA:** NVIDIA driver `580.82.07`, `nvcc` CUDA `12.8`.
- **Runtime commit:** `a9944b5c`.
- **Feature set:** `cargo build --release -p infer --features cuda --bin infer`.
- **KV dtype / cache mode:** paged KV `FP8E4M3`, contiguous cache `BF16`.
- **Session / prefix flags:** prefix cache on, `--num-slots 8`,
  `--max-seq-len 12288`, T1 host pinned capacity `32768 MiB`, high/low water
  `0.98 / 0.95`, min T1 prompt tokens `4096`.
- **Non-default env vars:** baseline `INFER_DECODE_METADATA_FAST_PAGE16=0`,
  candidate `INFER_DECODE_METADATA_FAST_PAGE16=1`, `RUST_LOG=info`.

The bench ran from runtime commit `a9944b5c`. During the run, unrelated
DeepSeek worktree changes appeared in `Cargo.lock`, `infer/Cargo.toml`, and
`infer/src/model/deepseek*`; they were not staged and were not part of the
built W3 runtime commit.

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `16` |
| sessions | `128` |
| turns | `384` |
| scored turns | `320` |
| prompt shape | W3 short multi-turn, base about `1024 +/- 32` tokens with `64 +/- 8` token tail |
| max output tokens | `64` |
| warm/cold mix | `64` warm sessions with one unscored warmup plus four scored turns; `64` cold distractor turns |
| run cap | full trace completion |

## Results

Both runs completed all turns, but the fast-path gate missed. The generated
baseline and fast-path traces are byte-identical:
`97d90c9a251b736b8e8fe2db924e5fa24931b88a73ec6c90ecab7fe8ba2363ee`.

### A/B Headline

| metric | baseline, fast path off | fast path on | delta |
|---|---:|---:|---:|
| turns OK | 384 / 384 | 384 / 384 | no change |
| scored turns OK | 320 | 320 | no change |
| scored output tokens | 17,872 | 17,669 | -1.1% |
| scored summed wall | 1,576.50 s | 1,881.80 s | +19.4% |
| scored output tok/s, summed wall | 11.336 | 9.389 | -17.2% |
| scored output tok/s, client window | 111.411 | 110.637 | -0.7% |
| TTFT p50 / p99 | 661.9 / 62,475.6 ms | 662.4 / 83,213.4 ms | +0.1% / +33.2% |
| ITL p50 / p99 | 42.8 / 44.1 ms | 42.8 / 45.0 ms | +0.2% / +1.8% |
| E2E p50 / p99 | 3,713.9 / 66,425.6 ms | 3,728.8 / 85,703.8 ms | +0.4% / +29.0% |

### W3 Warm/Cold

| metric | baseline warm | fast warm | delta |
|---|---:|---:|---:|
| scored turns | 256 | 256 | no change |
| TTFT p50 | 531.5 ms | 540.6 ms | +1.7% |
| TTFT p99 | 16,666.8 ms | 81,637.8 ms | +389.8% |
| output tok/s, summed wall | 12.505 | 10.682 | -14.6% |

| metric | baseline cold | fast cold | delta |
|---|---:|---:|---:|
| scored turns | 64 | 64 | no change |
| TTFT p50 | 843.1 ms | 838.9 ms | -0.5% |
| TTFT p99 | 60,686.6 ms | 67,717.1 ms | +11.6% |
| output tok/s, summed wall | 8.540 | 6.530 | -23.5% |

### Service-Side Cache / Scheduler

| metric | baseline, fast path off | fast path on |
|---|---:|---:|
| requests | 384 | 384 |
| final active / waiting / scheduled | 0 / 0 / 0 | 0 / 0 / 0 |
| `tokens_out` | 21,797 | 21,610 |
| `kv_util` | 74.9% | 74.9% |
| `prefix_hit_rate` | 97.9% | 97.9% |
| `prefix_skip_rate` | 62.8% | 62.4% |
| `prefix_request_hit_rate` | 100.0% | 100.0% |
| `prefix_request_skip_rate` | 92.6% | 92.9% |
| `session_affinity_hit` | 376 | 376 |
| `session_affinity_miss` | 8 | 8 |
| `matched_prefix_tokens` | 1,392 | 1,392 |
| `resume_prefill_tokens` | 111 | 107 |
| `plan_label` | idle:1,decode:2542,prefill:1,split:0,mixed:374 | idle:1,decode:2510,prefill:1,split:0,mixed:370 |
| `step_phase_us` | adm:727,prefill:16,decode:36885,emit:9,total:37636,cleanup:33,loop_total:33913 | adm:833,prefill:20,decode:37429,emit:9,total:38291,cleanup:43,loop_total:34514 |
| `kv_fetch_q` / `kv_fetch_waiters` | 0/16 / 0 | 0/16 / 0 |
| `kv_store_q` / `kv_store` / `kv_bp` | 0/16 / sub:0,done:0,fail:0,rej:0 / fetch:0,store:0 | 0/16 / sub:0,done:0,fail:0,rej:0 / fetch:0,store:0 |

### Acceptance Gates

| gate | target | actual | status |
|---|---:|---:|---|
| Output tok/s | fast path `>= 1.10x` baseline | `0.83x` by summed-wall tok/s; `0.99x` by client-window tok/s | MISS |
| Warm TTFT p99 | no regression | `16,666.8 -> 81,637.8 ms` | MISS |
| Default safety | fast path remains opt-in | env default off in code and baseline run | PASS |

## Problems

- The guarded fast path did not produce the expected throughput win and
  regressed W3 warm TTFT p99 in this run. Do not flip
  `INFER_DECODE_METADATA_FAST_PAGE16` on by default.
- The service-side counters stayed nearly identical while tail TTFT moved
  substantially, so the next useful step is diagnosis/profiling of the
  metadata-update and CUDA graph replay path rather than another tuning pass.
- The worktree was not clean by the end of the bench because unrelated
  DeepSeek files appeared during the run; the W3 runtime commit and staged
  bench entry exclude those paths.

## Learnings

- A lower metadata-upload volume does not automatically translate into W3
  throughput when the benchmark tail is dominated by decode-loop scheduling
  and graph replay variance.
- Keep the page-size-16 metadata shortcut behind the env guard until a paired
  profile shows a stable tail-latency reduction.

## Delta Vs Baseline

- **Baseline:** `bench-output/2026-05-05-w3-h1-baseline/results.json`.
- **Candidate:** `bench-output/2026-05-05-w3-h1-fastpath/results.json`.
- **Prior W3 reference:** [`2026-05-02-bench-agent-load-a1-w3-warm-p99.md`](2026-05-02-bench-agent-load-a1-w3-warm-p99.md).

| metric | 2026-05-02 A1 W3 | 2026-05-05 baseline | 2026-05-05 fast path |
|---|---:|---:|---:|
| successful scored turns | 320 / 320 | 320 / 320 | 320 / 320 |
| scored output tok/s, summed wall | 159.1 wall-clock reference | 11.336 | 9.389 |
| warm TTFT p99 | 718.2 ms | 16,666.8 ms | 81,637.8 ms |
| cold TTFT p99 | 6,684.4 ms | 60,686.6 ms | 67,717.1 ms |

## Artefacts

- Baseline result:
  `bench-output/2026-05-05-w3-h1-baseline/results.json`
  (`sha256 be104383570bfb231b3cc8df236af6caa32590ffcbfaf16a62f1942ab680289a`).
- Baseline trace:
  `bench-output/2026-05-05-w3-h1-baseline/trace.jsonl`
  (`sha256 97d90c9a251b736b8e8fe2db924e5fa24931b88a73ec6c90ecab7fe8ba2363ee`).
- Baseline client log:
  `bench-output/2026-05-05-w3-h1-baseline/client.log`.
- Baseline server log:
  `bench-output/2026-05-05-w3-h1-baseline/server.log`.
- Baseline stats:
  `bench-output/2026-05-05-w3-h1-baseline/stats_before.json`,
  `bench-output/2026-05-05-w3-h1-baseline/stats_after.json`.
- Fast-path result:
  `bench-output/2026-05-05-w3-h1-fastpath/results.json`
  (`sha256 a5b80b623b4fb0235db3d95b0a09c5b8a21582aca27bad76db4c66135ccad3b8`).
- Fast-path trace:
  `bench-output/2026-05-05-w3-h1-fastpath/trace.jsonl`
  (`sha256 97d90c9a251b736b8e8fe2db924e5fa24931b88a73ec6c90ecab7fe8ba2363ee`).
- Fast-path client log:
  `bench-output/2026-05-05-w3-h1-fastpath/client.log`.
- Fast-path server log:
  `bench-output/2026-05-05-w3-h1-fastpath/server.log`.
- Fast-path stats:
  `bench-output/2026-05-05-w3-h1-fastpath/stats_before.json`,
  `bench-output/2026-05-05-w3-h1-fastpath/stats_after.json`.

## Notes

- Code since baseline: `a9944b5c` adds the guarded `DecodeMetaUpdate`
  fast-path modes and leaves the feature default-off.
- Verdict: `miss`. The code is safe to keep behind the env guard, but this
  bench does not justify enabling it.
- Follow-up: profile the decode metadata update path before attempting another
  W3 H1 optimization tranche.
