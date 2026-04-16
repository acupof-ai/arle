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

Source: `bench-output/2026-04-16-cuda-l4-kv-baseline-v2/benchmarks.json`
Wins file: `2026-04-16-bench-guidellm-cuda-l4-kv-baseline-v2.md`

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s |
|---|---|---|---|---|---|---|
| **sync** | 766.3 | 784.3 | 35.46 | 35.53 | **26.43** | 0.1 |
| **throughput** | 21290.7 | 51425.3 | 53.71 | 69.94 | **89.39** | 0.35 |
| 0.131 r/s | 1244.8 | 1304.9 | 40.86 | 41.06 | 31.66 | 0.117 |
| 0.350 r/s | 1243.1 | 1316.0 | 50.55 | 54.23 | **73.67** | 0.283 |

**Key numbers:** sync decode **26.43 tok/s** (ITL 35.5ms), saturation throughput **89.39 tok/s**.

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
- Cross-session prefix hit rate ≥ 70% (currently 0% — see caveat below)
- No throughput regression vs this baseline at any concurrency level
- ITL p99 ≤ 50ms under concurrent load

## Known test-behaviour limitations

**Agent trace prefix_hit_rate=0% is a test artefact, not a system bug.**

Root cause: the default trace (`scripts/data/agent_trace_default.jsonl`) has
6 sessions with 4 distinct system prompts. With num_slots=7 and concurrency=4,
every session occupies a different slot. After a request completes, the slot
is freed and its materialized KV is overwritten by the next session's request.
The radix tree records the prefix, but `best_reusable_slot_for_radix_hit()`
cannot find a free slot with matching materialized state — so it falls back
to MISS.

Only agent-001 turn 2 hits a PARTIAL (48/77 tokens) because it arrives
immediately after agent-001 turn 0 completes on the same slot.

**What would make the test meaningful:**
1. Multiple requests in the SAME session sent sequentially (multi-turn within
   one session) — each turn extends the prefix, same slot reuses it.
2. A shared system prompt across ALL sessions (not just 2 of 6) AND enough
   slots that at least one retains the materialized prefix between sessions.
3. Higher slot count (> session count) so freed slots survive long enough
   for cross-session reuse.

**Underlying system limitation (documented in `tiered-kv-cache.md` §3):**
Cross-slot page aliasing is intentionally unsupported. Prefix reuse is limited
to same-slot resurrection: the radix tree knows the prefix exists, but the
scheduler can only reuse it if the slot's contiguous KV state still materialises
the matched tokens. This is a Phase-2 (M3 → M5) upgrade target.
