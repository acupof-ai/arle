# Agent-load W3-H1 SGLang gate miss

## Context

A1 session-affinity admission was already committed as `b1716819` and the
local ARLE W3 trace was recorded in
`docs/experience/wins/2026-05-02-bench-agent-load-a1-w3-warm-p99.md`.

That entry compared ARLE W3 output rate against the older
`project_l4_perf_baseline` c=1 Qwen3-4B throughput numbers. This was the wrong
entrance criterion for the agent-load mission. The active benchmark contract is
`docs/plans/2026-05-02-agent-load-bench-spec.md` §3.3:

```text
ARLE successful output tok/s >= 1.00x best pinned competitor
ARLE warm TTFT p99 <= 1.05x best pinned competitor
```

This rerun installed and launched the pinned SGLang competitor locally on the
same NVIDIA L4 box. No `pending-remote` result was used.

## Commands

Pinned SGLang checkout:

```bash
git clone https://github.com/sgl-project/sglang.git \
  /tmp/sglang-arle-214c35b03184c354acf1f86f99746799e1c9b3a9
git -C /tmp/sglang-arle-214c35b03184c354acf1f86f99746799e1c9b3a9 \
  checkout --detach 214c35b03184c354acf1f86f99746799e1c9b3a9
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
  python3 -m pip install -e \
  /tmp/sglang-arle-214c35b03184c354acf1f86f99746799e1c9b3a9/python
```

Server:

```bash
CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
PYTHONPATH=/tmp/sglang-arle-214c35b03184c354acf1f86f99746799e1c9b3a9/python \
python3 -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30000 \
  --model-path /content/workspace/agent-infer/models/Qwen3-4B \
  --served-model-name default \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 16 \
  --mem-fraction-static 0.85 \
  --max-total-tokens 120000 \
  --enable-cache-report \
  --enable-metrics \
  --random-seed 20260502
```

Client:

```bash
START_MS=$(date +%s%3N)
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://127.0.0.1:30000 \
  --label sglang-w3-h1-clean \
  --out bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/results.json \
  --trace-out bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/trace.jsonl \
  --no-probe-stats
END_MS=$(date +%s%3N)
echo "ELAPSED_MS=$((END_MS - START_MS))"
```

## Environment

- **Run commit:** `6d951d35`, clean tree before run.
- **GPU:** NVIDIA L4, 23,034 MiB, driver 580.82.07.
- **CUDA compiler:** `/usr/local/cuda/bin/nvcc`, `cuda_12.8.r12.8`.
- **Model / tokenizer:** `models/Qwen3-4B`, served as OpenAI model `default`.
- **SGLang commit:** `214c35b03184c354acf1f86f99746799e1c9b3a9`.
- **SGLang version:** `0.5.10.post2.dev613+g214c35b03`.
- **Python runtime:** `torch 2.9.1+cu128`, `flashinfer 0.6.7.post3`,
  `transformers 5.5.4`.
- **Cache flags:** radix cache enabled (`disable_radix_cache=False`),
  `--enable-cache-report`, FP8 E4M3 KV.
- **Request limit:** `--max-running-requests 16`.

SGLang printed:

```text
Skipping import of cpp extensions due to incompatible torch version.
Please upgrade to torch >= 2.11.0 (found 2.9.1+cu128).
```

The server still launched and completed the W3 trace. The warning is recorded
because it may affect SGLang's absolute number, but the pinned run remains the
local competitor row for this check.

## Results

Clean SGLang W3 run:

| metric | value |
|---|---:|
| turns OK | 384 / 384 |
| scored turns OK | 320 / 320 |
| scored tokens | 20,480 |
| elapsed wall-clock | 93.740 s |
| successful output tok/s | 218.477 |
| TTFT p50 | 419.1 ms |
| TTFT p99 | 2642.7 ms |
| ITL p50 | 41.4 ms |
| ITL p99 | 42.5 ms |

Warm/cold split:

| metric | warm | cold |
|---|---:|---:|
| scored turns | 256 | 64 |
| TTFT p50 | 405.4 ms | 1640.0 ms |
| TTFT p99 | 2516.3 ms | 2641.9 ms |

ARLE comparison from
`bench-output/2026-05-02-agent-load-a1-w3-warm-p99/results.json`:

| engine | commit/tag | successful output tok/s | warm TTFT p99 | cache report |
|---|---|---:|---:|---|
| ARLE | `b1716819` runtime, docs head `6d951d35` | 159.129 | 718.2 ms | `/v1/stats`, prefix request hit 100.0%, session hits 368/16 |
| SGLang | `214c35b03184c354acf1f86f99746799e1c9b3a9` | 218.477 | 2516.3 ms | `/metrics`, `cached_tokens_total{device}=296892` |

Gate result:

```text
tok/s ratio = 159.129 / 218.477 = 0.728  -> FAIL (< 1.00)
warm p99 ratio = 718.2 / 2516.3 = 0.285 -> PASS (<= 1.05)
W3_margin = min(0.728, 2516.3 / 718.2) = 0.728
W3-H1 entrance = red
```

Service-side SGLang counters after the clean W3 run:

| metric | value |
|---|---:|
| `/v1/chat/completions` responses | 384 |
| `prompt_tokens_total` | 477,424 |
| `generation_tokens_total` | 24,584 |
| `cached_tokens_total{device}` | 296,892 |
| `realtime_tokens_total{prefill_compute}` | 180,532 |
| `realtime_tokens_total{prefill_cache}` | 296,892 |
| `evicted_tokens_total{RadixCache}` | 85,563 |
| `kv_evictable_tokens` after run | 95,019 |
| `kv_used_tokens` after run | 24,097 |

## Root Cause

The previous A1 report used the wrong comparison target. A c=1 ARLE baseline is
not the W3-H1 entrance gate; the plan requires best pinned competitor
comparison under the canonical W3 trace.

The corrected local SGLang row shows ARLE already wins warm-turn tail latency,
but not successful output throughput. ARLE also produced fewer scored output
tokens on the same `max_tokens=64` trace (`18,987` vs SGLang's `20,480`) and
had slower ITL (`50.4 ms` vs `41.4 ms`), so the tok/s miss is not just an
elapsed-time artifact.

## Fix

Do not claim W3-H1 entrance green from c=1 baselines. The next A1/A5 decision
must close the W3 successful-output throughput gap against the pinned SGLang
row, or deliberately revise the benchmark contract in the plan before any
mission claim.

Concrete next checks:

- Keep the pinned SGLang row as the current W3-H1 local competitor floor.
- Investigate ARLE's W3 output-token shortfall and decode ITL before tuning
  more admission policy.
- Re-run the same `agent-w3-short-multiturn` trace after any scheduler/decode
  change; entrance is green only when ARLE tok/s is at least `218.477` and
  warm TTFT p99 remains no worse than `2516.3 * 1.05 = 2642.1 ms`.

## Rule

For agent-load W3, compare against the plan §3.3 pinned competitor gate first.
Project-local c=1 baselines are useful sanity checks, but they are not evidence
for world-#1 entrance.

## Artifacts

- ARLE result: `bench-output/2026-05-02-agent-load-a1-w3-warm-p99/results.json`
- SGLang clean result:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/results.json`
- SGLang trace:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/trace.jsonl`
- SGLang client log:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/client.log`
- SGLang server log:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/sglang_server.log`
- SGLang metrics before:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/metrics_before.prom`
- SGLang metrics after:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/metrics_after.prom`
- Comparison JSON:
  `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/comparison.json`
