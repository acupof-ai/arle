# A1 W3 Warm-P99 - local L4 trace run, 2026-05-02

> Status: local NVIDIA L4 run complete. This is the focused A1 W3 entrance
> check after the A2 stats-surface run was accepted as accounting-only.

## Goal

- Measure `agent-w3-short-multiturn` warm TTFT p99 on the A1
  session-affinity admission build.
- Compare W3 output throughput against `project_l4_perf_baseline`
  Qwen3-4B c=1 baselines: 30.5 tok/s in-process, 29.3 tok/s HTTP.

## Hypothesis

- A1 should keep W3 warm-turn TTFT p99 bounded while retaining prefix/session
  accounting through `/v1/stats?format=json`.

## Command

Server, first with the requested model path:

```bash
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
./target/release/infer --model-path models/Qwen3-4B --port 8000 --num-slots 16
```

Preflight result:

```text
GET /health -> 404 route_not_found
GET /v1/models -> {"id":"Qwen3-4B"}
```

The W3 harness currently hard-codes request `model: "default"`, so the W3
HTTP replay was run against the same weights through the existing symlink:

```bash
ln -sfn Qwen3-4B models/default
./target/release/infer --model-path models/default --port 8000 --num-slots 16
```

Client:

```bash
START_MS=$(date +%s%3N)
python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://localhost:8000 \
  --label a1-w3-warm-p99 \
  --out bench-output/2026-05-02-agent-load-a1-w3-warm-p99/results.json \
  --trace-out bench-output/2026-05-02-agent-load-a1-w3-warm-p99/trace.jsonl
END_MS=$(date +%s%3N)
echo "ELAPSED_MS=$((END_MS - START_MS))"
```

## Environment

- **Backend / engine:** arle-cuda
- **Model:** Qwen3-4B Instruct (`models/default -> models/Qwen3-4B`)
- **Hardware:** NVIDIA L4 24GB, driver 580.82.07, CUDA 13.0, SM 89
- **Commit:** A1 runtime `b1716819`; docs head `c667fb86`
- **Feature set:** release CUDA binary, `--no-default-features --features cuda`
- **Server shape:** `--num-slots 16`, auto `max_seq_len=4096`
- **KV dtype / cache mode:** auto FP8E4M3 paged KV, RadixCache on
- **Python tools:** `tilelang 0.1.9`

## Results - W3

| metric | value |
|---|---:|
| turns OK | 384 / 384 |
| scored turns OK | 320 / 320 |
| scored tokens | 18987 |
| elapsed wall-clock | 119.318 s |
| scored wall-clock tok/s | 159.1 |
| server total tokens_out | 22991 |
| server total tok/s | 192.7 |
| summed scored wall | 1457.52 s |
| summed scored tok/s | 13.0 |
| TTFT p50 | 233.3 ms |
| TTFT p99 | 5600.7 ms |
| ITL p50 | 50.4 ms |
| ITL p99 | 53.5 ms |

## Results - Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | 256 | 64 |
| TTFT p50 | 226.7 ms | 637.2 ms |
| TTFT p99 | 718.2 ms | 6684.4 ms |

## Results - Service Stats

| metric | value |
|---|---:|
| final active | 0 |
| final waiting | 0 |
| final kv_util | 69.2% |
| final prefix_hit_rate | 95.8% |
| final prefix_skip_rate | 57.9% |
| final prefix_request_hit_rate | 100.0% |
| final prefix_request_skip_rate | 92.5% |
| session_affinity_hit | 368 |
| session_affinity_miss | 16 |
| matched_prefix_tokens | final request 1392 |
| resume_prefill_tokens | final request 113 |

## Baseline Comparison

| baseline | value | W3 scored wall-clock delta | W3 server-total delta |
|---|---:|---:|---:|
| Qwen3-4B in-process c=1 | 30.5 tok/s | 5.22x / +421.7% | 6.32x / +531.8% |
| Qwen3-4B HTTP c=1 | 29.3 tok/s | 5.43x / +443.1% | 6.58x / +557.6% |

The c=1 baseline is a single-stream throughput reference. W3 is a 16-concurrent
multi-turn session trace, so the wall-clock aggregate rates above are the
closest direct comparison; summed scored tok/s is kept separately and should
not be used as a c=1 throughput comparison.

## Problems

- `/health` is not registered on this router and returns 404; `/v1/models`
  and `/v1/stats?format=json` were used for readiness.
- `scripts/bench_guidellm.sh` does not support W3 trace replay. The correct
  W3 harness is `scripts/bench_agent_trace.py --workload
  agent-w3-short-multiturn`.
- `bench_agent_trace.py` still hard-codes request model `"default"`, requiring
  the `models/default -> Qwen3-4B` symlink for strict model validation.

## Learnings

- A1's W3 warm TTFT p99 entrance signal is 718.2 ms on the direct release
  binary with `--num-slots 16`.
- The W3 aggregate output rate is well above the c=1 project baseline, while
  cold p99 remains the dominant tail contributor.

## Artefacts

- Raw turns / summary: `bench-output/2026-05-02-agent-load-a1-w3-warm-p99/results.json`
- Generated trace: `bench-output/2026-05-02-agent-load-a1-w3-warm-p99/trace.jsonl`
- Server log: `bench-output/server-logs/2026-05-02T07-56-12-port8000-a1-w3-warm-p99-direct-default.log`
