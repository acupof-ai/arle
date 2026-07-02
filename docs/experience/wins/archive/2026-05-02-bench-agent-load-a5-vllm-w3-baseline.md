# A5 vLLM W3 baseline - agent-load benchmark, agent-w3-short-multiturn, vLLM, 2026-05-02

## Goal

- Baseline vLLM on the canonical W3 short-prompt multi-turn trace for the A5 four-engine panel.

## Hypothesis

- vLLM with automatic prefix caching should beat the pinned SGLang row on throughput and expose a content-prefix cache hit counter, but it has no session-affinity routing signal equivalent to ARLE A1.

## Command

Server:

```bash
/tmp/arle-vllm-0.20.0/bin/vllm serve /content/workspace/agent-infer/models/Qwen3-4B \
  --host 127.0.0.1 \
  --port 30100 \
  --served-model-name default \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --seed 20260502
```

Client:

```bash
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://127.0.0.1:30100 \
  --label vllm-w3-h1 \
  --out bench-output/2026-05-02-agent-load-vllm-w3-h1/results.json \
  --trace-out bench-output/2026-05-02-agent-load-vllm-w3-h1/trace.jsonl \
  --no-probe-stats
```

## Environment

- **Workload:** `agent-w3-short-multiturn`.
- **Backend / engine:** `vllm`.
- **Model:** Qwen3-4B, `/content/workspace/agent-infer/models/Qwen3-4B`.
- **Tokenizer / processor:** same local model path.
- **Hardware:** NVIDIA L4, 23,034 MiB VRAM, driver 580.82.07.
- **Commit:** ARLE `fbb39407`, clean tree before run.
- **Feature set:** vLLM `0.20.0` in isolated `/tmp/arle-vllm-0.20.0` venv; `torch 2.11.0+cu130`, `transformers 5.7.0`, `flashinfer 0.6.8.post1`.
- **KV dtype / cache mode:** FP8 KV, automatic prefix caching enabled.
- **Session / prefix flags:** request `session_id` is sent by the trace client, but vLLM has no session-affinity routing equivalent; prefix reuse is content-prefix cache only.
- **Non-default flags / env vars:** flags shown above; `/metrics` captured before and after.

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `16` |
| sessions | `64` warm + `64` cold |
| scored turns | `256` warm + `64` cold |
| prompt shape | W3 canonical `1024 +/- 32` base, `64 +/- 8` appended tail |
| max output tokens | `64` |
| warm/cold mix | `80%` warm, `20%` cold |
| tool output tokens | n/a |
| run cap | full trace completion |

## Results - Headline

Elapsed wall-clock uses the client run timestamp to `results.json` mtime because the current harness only persists per-request wall times.

| metric | value |
|---|---:|
| successful scored turns | 320 / 320 |
| incomplete scored turns | 0 |
| scored output tokens | 20,480 |
| elapsed wall-clock | 86.941 s |
| successful output tok/s | 235.561 |
| TTFT p50 (ms) | 176.2 |
| TTFT p99 (ms) | 2307.4 |
| ITL p50 (ms) | 41.3 |
| ITL p99 (ms) | 42.5 |
| E2E p50 (ms) | 2973.2 |
| E2E p99 (ms) | 6092.2 |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | 256 | 64 |
| TTFT p50 (ms) | 172.8 | 929.6 |
| TTFT p99 (ms) | 212.5 | 2960.9 |
| E2E p50 (ms) | 2938.3 | 4884.1 |
| E2E p99 (ms) | 3017.9 | 6704.6 |
| output tokens | 16,384 | 4,096 |

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| `prompt_tokens_total` | 477,418 |
| `prompt_tokens_cached_total` | 313,712 |
| `prompt_tokens_by_source{local_compute}` | 163,706 |
| `prompt_tokens_by_source{local_cache_hit}` | 313,712 |
| `generation_tokens_total` | 24,576 |
| `prefix_cache_queries_total` | 477,418 |
| `prefix_cache_hits_total` | 313,712 |
| prefix token hit rate | 65.7% |
| session-affinity hit/miss | unavailable |
| peak waiting | unavailable |
| peak kv_util | log sampled 13.5%, final gauge 0.0% |

## Four-Engine Comparison

| engine | commit/tag | output tok/s | warm TTFT p99 (ms) | cold TTFT p99 (ms) | cache report | raw artefact |
|---|---|---:|---:|---:|---|---|
| ARLE | `b1716819` runtime, docs head `6d951d35` | 159.129 | 718.2 | 6684.4 | `/v1/stats`, prefix request hit 100.0%, session hits 368/16 | `bench-output/2026-05-02-agent-load-a1-w3-warm-p99/results.json` |
| SGLang | `214c35b03184c354acf1f86f99746799e1c9b3a9` | 218.477 | 2516.3 | 2641.9 | `/metrics`, `cached_tokens_total{device}=296892` | `bench-output/2026-05-02-agent-load-sglang-w3-h1-clean/results.json` |
| vLLM | `0.20.0` | 235.561 | 212.5 | 2960.9 | `/metrics`, `prefix_cache_hits_total=313712` | `bench-output/2026-05-02-agent-load-vllm-w3-h1/results.json` |
| TensorRT-LLM | pending A5 row | n/a | n/a | n/a | n/a | n/a |

Interim W3-H1 gate after adding vLLM:

```text
best competitor throughput = 235.561 tok/s (vLLM)
best competitor warm p99 = 212.5 ms (vLLM)
ARLE throughput ratio = 159.129 / 235.561 = 0.676
ARLE warm p99 ratio = 718.2 / 212.5 = 3.38
W3-H1 entrance remains red until ARLE beats the best competitor on both axes.
```

## Problems

- vLLM `0.16.0` was rejected for this row because a system-site install saw an incomplete global `flash_attn` namespace and failed Qwen3 rotary initialization. The published row uses a fully isolated `vllm==0.20.0` venv instead.
- vLLM prints no session-affinity routing metric; the row records content-prefix reuse only.
- The harness does not persist run start/end timestamps, so output tok/s uses `results.json` timestamp-to-mtime elapsed wall-clock.

## Learnings

- vLLM's content-prefix cache is effective on W3 and moves warm TTFT p99 below both ARLE and SGLang on this L4 run; A5 ranking cannot defend an ARLE warm-tail leadership claim after this row.
- For competitor rows, isolate Python environments per engine; system-site package mixing can produce false setup failures before the engine is actually tested.

## Delta Vs Baseline

- **Baseline:** [`2026-05-02-agent-load-w3-h1-sglang-gate-miss.md`](../errors/2026-05-02-agent-load-w3-h1-sglang-gate-miss.md).

| metric | SGLang baseline | vLLM now | delta |
|---|---:|---:|---:|
| output tok/s | 218.477 | 235.561 | +7.8% |
| warm TTFT p99 | 2516.3 ms | 212.5 ms | -91.6% |
| cold TTFT p99 | 2641.9 ms | 2960.9 ms | +12.1% |
| prefix cached tokens | 296,892 | 313,712 | +5.7% |

## Artefacts

- Raw result: `bench-output/2026-05-02-agent-load-vllm-w3-h1/results.json`
- Trace: `bench-output/2026-05-02-agent-load-vllm-w3-h1/trace.jsonl`
- Client log: `bench-output/2026-05-02-agent-load-vllm-w3-h1/client.log`
- Server log: `bench-output/2026-05-02-agent-load-vllm-w3-h1/vllm020_server.log`
- Metrics before: `bench-output/2026-05-02-agent-load-vllm-w3-h1/metrics_before.prom`
- Metrics after: `bench-output/2026-05-02-agent-load-vllm-w3-h1/metrics_after.prom`
- Key sha256:
  - `results.json`: `d0ee5494eed7d73203d1cd2c6caa7230efb48e8e660be6f452bfcd6634904a25`
  - `trace.jsonl`: `97d90c9a251b736b8e8fe2db924e5fa24931b88a73ec6c90ecab7fe8ba2363ee`
  - `client.log`: `3ee4612ce594c1a2193fe88d5bef972b3a6156a7dd43fa202e9e88f6d4836df7`
  - `vllm020_server.log`: `213b2ac18fe95d647a1c2c9ce0395576961a01afecbc84cce2e8e18110c30865`
  - `metrics_after.prom`: `4483f69afab519e52ef3654844005b8e9c9a854c764299a767b6845752e2ad80`

## Notes

- What changed in the code since baseline: none; A5 is measurement-only.
- Suspected cause of any regression: n/a.
- Follow-ups: add TensorRT-LLM W3 row next; after vLLM, the current ARLE warm-TTFT leadership claim is falsified on this four-engine panel.
