# Phase 1 close attempt: evictable-prefix c4 margin secured, c1 guard blocked

## Goal

- Close the longctx Phase 1 entrance gate for Qwen3-4B FP8 KV on L4.
- Primary c=4 row: prompt=32768, output=256, concurrency=4, 300s.
- Required S5 guard: c=1, 360s.
- Gate source: `docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
  §2.4 requires `ARLE.tok/s >= 1.30 x max(SGLang, vLLM, TRT-LLM, Mooncake)`.

## Hypothesis

- Counting evictable prefix-cache pages in admission should remove the prior
  KV-pool edge deadlock mode and lift effective c=4 output throughput above
  the SGLang Phase 1 S5 target.

## Command

```bash
/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs

for run in r1 r2 r3; do
  WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 \
    scripts/bench_guidellm.sh phase15-evictable-c4-"$run" \
    --target http://127.0.0.1:8000 \
    --model Qwen3-4B \
    --processor infer/models/Qwen3-4B
done

LONGCTX_SECONDARY_C1_ONLY=1 WORKLOAD=longctx-32k LONGCTX_MAX_SECONDS=360 \
  scripts/bench_guidellm.sh phase15-evictable-c1-guard \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Successful-only recompute:

```bash
total_output_tokens / 300
median(first_request_iteration - request_start)
median(inter_token_latency_ms)
```

## Environment

- **Backend:** CUDA
- **Hardware:** NVIDIA L4, 24 GB class VRAM
- **Model:** Qwen3-4B
- **Weights:** `infer/models/Qwen3-4B`
- **Runtime patch commit:** `051b1081`
- **Repo head during close:** `9e002fa9` before this doc
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Build env:** `CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=8.9`,
  `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **KV pool:** FP8E4M3, 136976 max tokens, 8561 pages, page_size=16,
  11.0 GB

## Canonical params

- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--rate 4`
- `--max-seconds 300`
- `--random-seed 20260416`
- Wrapper: `scripts/bench_guidellm.sh`

## Results - c4 primary row

SGLang reference:
`docs/experience/wins/2026-04-30-bench-sglang-longctx-longctx-32k-phase1-s5.md`
reports c=4 `16.27 out tok/s`, so the leadership §2.4 1.30x threshold is
`21.151 out tok/s`.

| run | successful requests | total output tokens | effective out tok/s | vs SGLang | successful-only TTFT p50 | successful-only ITL p50 |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 32 | 8192 | 27.307 | 1.678x | 32225.9 ms | 178.4 ms |
| r2 | 28 | 7168 | 23.893 | 1.469x | 33888.9 ms | 116.2 ms |
| r3 | 32 | 8192 | 27.307 | 1.678x | 33879.3 ms | 117.5 ms |
| mean | - | - | 26.169 | 1.608x | - | - |

Variance: output-throughput stdev `1.971 tok/s`, CV `7.53%`. All three
runs exceed the `21.151 out tok/s` 1.30x threshold.

## Results - c1 S5 guard

SGLang reference reports c=1 `11.67 out tok/s` primary and `11.57 out tok/s`
secondary. The c=1 guard did not reach the Phase 1 S5 `>=0.95x` acceptance
line.

| run | successful requests | total output tokens | GuideLLM out tok/s | effective out tok/s | vs SGLang primary | successful-only TTFT p50 | successful-only ITL p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 guard | 13 | 3328 | 9.83 | 9.244 | 0.792x effective / 0.842x GuideLLM | 12540.6 ms | 56.84 ms |

## Results - service-side KV / scheduler metrics

| run | peak active | peak waiting | peak running_batch | peak prefill_queue | plan labels | peak kv_util |
|---|---:|---:|---:|---:|---|---:|
| r1 | 4 | 0 | 4 | 1 | `idle=6,decode=262,prefill=115,split=0,mixed=2` | 100.0% |
| r2 | 4 | 0 | 4 | 2 | `idle=6,decode=262,prefill=106,split=0,mixed=2` | 100.0% |
| r3 | 4 | 0 | 4 | 0 | `idle=6,decode=262,prefill=122,split=0,mixed=1` | 100.0% |

## Problems

- GuideLLM aggregate validation marked each ARLE run invalid because the
  completed-request summary reported TTFT p50 and ITL p50 as `0.0` despite
  successful requests with non-zero output tokens.
- Per user directive, this close uses raw successful request records only:
  successful-only TTFT/ITL p50 and `total_output_tokens / 300` effective
  throughput.
- The c=1 S5 guard completed successfully but failed the SGLang parity gate:
  GuideLLM out tok/s was `9.83`, effective `total_output_tokens / 360` was
  `9.244`, versus SGLang c=1 `11.57-11.67`.

## Learnings

- c=4 entrance row **PASSED**.
- The 1.30x margin target from leadership §2.4 is **SECURED** on this local
  L4 W1 c=4 longctx row: the weakest run is `1.469x` SGLang and the mean is
  `1.608x`.
- The evictable-prefix admission patch eliminated the earlier c=4 bimodal
  deadlock signature for these three runs: no run produced the previous
  near-zero output mode.
- Full Phase 1 S5 close is **BLOCKED** by c=1 guard throughput. This is not a
  c=4 admission regression; it is the single-concurrency long-prompt prefill
  gap that remains below SGLang.

## Delta vs baseline

- **SGLang mission target:**
  `docs/experience/wins/2026-04-30-bench-sglang-longctx-longctx-32k-phase1-s5.md`
- **ARLE pre-patch anchor:**
  `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- **Runtime patch:**
  `051b1081 fix(scheduler): count evictable prefix pages in admission`

| metric | baseline / target | now | delta |
|---|---:|---:|---:|
| SGLang c=4 target | 16.27 out tok/s | 26.169 mean out tok/s | +60.8% |
| 1.30x threshold | 21.151 out tok/s | 23.893 worst run | +13.0% over threshold |
| pre-patch ARLE c=4 mean | 2.99 out tok/s | 26.169 mean out tok/s | +775.2% |
| pre-patch ARLE best run | 8.07 out tok/s | 26.169 mean out tok/s | +224.3% |
| pre-patch ARLE c1 mean | 9.77 out tok/s | 9.83 GuideLLM out tok/s | +0.6% |
| SGLang c1 secondary | 11.57 out tok/s | 9.83 GuideLLM out tok/s | -15.0% |

## Artefacts

- r1 raw: `bench-output/2026-05-01-phase15-evictable-c4-r1/benchmarks.json`
- r1 service trace:
  `bench-output/2026-05-01-phase15-evictable-c4-r1/service_stats_trace_summary.md`
- r2 raw: `bench-output/2026-05-01-phase15-evictable-c4-r2/benchmarks.json`
- r2 service trace:
  `bench-output/2026-05-01-phase15-evictable-c4-r2/service_stats_trace_summary.md`
- r3 raw: `bench-output/2026-05-01-phase15-evictable-c4-r3/benchmarks.json`
- r3 service trace:
  `bench-output/2026-05-01-phase15-evictable-c4-r3/service_stats_trace_summary.md`
- c1 guard raw:
  `bench-output/2026-05-01-phase15-evictable-c1-guard/benchmarks.json`
- c1 guard service trace:
  `bench-output/2026-05-01-phase15-evictable-c1-guard/service_stats_trace_summary.md`

## Notes

- Phase 2 draft is opened in
  `docs/plans/2026-05-01-longctx-spec-decode-phase2.md`, but implementation
  is gated on resolving the c=1 S5 guard blocker.
