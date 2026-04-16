# KV Cache Baseline — NVIDIA L4 24GB, Qwen3-4B, 2026-04-16

First canonical baseline for the KV tiered cache project. All benchmarks run
serially (one at a time) on a single L4 GPU with exclusive access.

## Hardware & Config

- **GPU:** NVIDIA L4 24GB, CUDA 12.8, driver 580.82.07, SM89
- **Model:** Qwen/Qwen3-4B (BF16 weights, BF16 paged KV, page_size=16)
- **Commit:** c1956da (feat: wire ServerMetrics + optimize bench scripts)
- **Build:** `cargo build --release -p infer` (default CUDA features)
- **Server:** `./target/release/infer --model-path models/Qwen3-4B --port 8000`
- **Env:** `PEGAINFER_CUDA_SM=89`

## 1. guidellm sweep (canonical, prompt=4096 tokens, output=256 tokens)

Source: `bench-output/2026-04-16-cuda-l4-kv-baseline/benchmarks.json`
Wins file: `2026-04-16-bench-guidellm-cuda-l4-kv-baseline.md`

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s |
|---|---|---|---|---|---|---|
| sync | 805 | 836.3 | 35.56 | 35.61 | **26.26** | 0.1 |
| throughput | 21347.8 | 51390.6 | 53.77 | 68.25 | **89.56** | 0.35 |
| 0.131 r/s | 1270.2 | 1306.4 | 40.8 | 40.97 | 31.72 | 0.117 |
| 0.350 r/s | 1253.5 | 1290.2 | 50.52 | 54.63 | **73.64** | 0.283 |

**Key numbers:** sync decode 26.26 tok/s (ITL 35.6ms), saturation throughput 89.56 tok/s.

## 2. ShareGPT (200 prompts, concurrency=4, max_tokens=256)

Source: `2026-04-16-bench-sharegpt-kv-baseline.json`

| Metric | Value |
|---|---|
| Total requests | 200 |
| Wall time | 460.52s |
| **Throughput** | **110.9 tok/s** |
| Request rate | 0.43 req/s |
| Prompt tokens | 35,465 |
| Output tokens | 51,067 |
| TTFT p50 / p90 / p99 | 110.5 / 312.5 / 652.7 ms |
| ITL p50 / p90 / p99 | 34.8 / 35.4 / 41.8 ms |
| E2E p50 | 9,087.6 ms |

## 3. Agent trace (6 sessions, 14 turns, concurrency=4)

Source: `2026-04-16-bench-kv-baseline-agent-trace.json`

| Metric | Value |
|---|---|
| Turns OK | 14 / 14 |
| Total tokens | 3,472 |
| Wall total | 123.44s |
| **TTFT p50 / p99** | **126.0 / 219.9 ms** |
| **ITL p50 / p99** | **34.7 / 34.9 ms** |
| prefix_hit_rate | 0.0% (sessions have unique system prompts) |
| kv_util (after) | 0.5% |

## Snapshot locations (single source of truth)

All bench snapshots converge to `docs/experience/wins/`:

| File | Type | Dataset |
|---|---|---|
| `2026-04-16-bench-guidellm-cuda-l4-kv-baseline.md` | guidellm sweep (canonical) | synthetic 4096/256 |
| `2026-04-16-bench-sharegpt-kv-baseline.json` | ShareGPT throughput | ShareGPT V3 (200) |
| `2026-04-16-bench-kv-baseline-agent-trace.json` | Agent trace replayer | agent_trace_default.jsonl |
| `2026-04-16-bench-kv-baseline-all.md` | **This file** — consolidated baseline | all |

Raw guidellm artefacts (JSON/CSV/HTML): `bench-output/2026-04-16-cuda-l4-kv-baseline/` (gitignored).

## Baseline declaration

These numbers are the **first official KV cache baseline** for the L4 CUDA backend.
All future bench runs MUST cite this file as the baseline and show deltas.

**Exit criteria for KV tiered cache project (P1):**
- Cross-session prefix hit rate ≥ 70% (currently 0% — sessions don't share prompts in default trace)
- No throughput regression vs this baseline at any concurrency level
- ITL p99 ≤ 50ms under concurrent load
