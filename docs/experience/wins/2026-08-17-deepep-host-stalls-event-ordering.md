# DeepEP host stalls → on-device event ordering — CUDA, 2026-08-17

> Status: pending-remote

## Goal

Eliminate per-MoE-layer host blocks on the DeepEP transport path. Each
dispatch/combine call issued `cudaStreamSynchronize` on the compute stream
(or passed `compute_stream=0` which made the C++ wrapper host-sync), blocking
the CPU for the full NVLink transfer duration on every MoE layer.

## Hypothesis

Passing the real compute stream handle to the DeepEP C++ wrapper activates
the existing (but dead) event-based `cudaEventRecord` + `cudaStreamWaitEvent`
path, replacing host sync with on-device stream ordering. Additionally,
replacing the `recv_topk_idx` D2H→host-i64→i32→H2D roundtrip with an
on-device cast kernel eliminates a second per-layer host roundtrip.

## Parameters

```bash
# A/B: baseline = parent of 142b959d4, treatment = 142b959d4
# DSv4 MoE model, ARLE_DSV4_MOE_TRANSPORT=deepep, TP>=2
python3 scripts/bench_throughput.py \
  --url <url> \
  --model <dsv4-moe-model> \
  --prompts-jsonl bench-agent-119k-16x8.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 128 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/deepep-event-order/bench
```

- Baseline: `142b959d4^` (compute_stream=0, host sync, D2H/host/H2D roundtrip)
- Treatment: `142b959d4` (event ordering, device cast kernel)
- Trials: 3 (matched A/B, simultaneous)

## Environment

- Host / GPU: 8×H20 pod (sm_90)
- Driver / CUDA: TBD
- Model / dtype: DSv4 MoE (GLM-5.2 family), BF16/FP8
- TP / EP / slots / KV: TP=8 (or ≥2), DeepEP transport
- Server flags: `ARLE_DSV4_MOE_TRANSPORT=deepep`

## Results

| concurrency | arm | completed | errors | output tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | | | | | | | — |
| 1 | treatment | | | | | | | |

Raw artifacts: TBD.

## Problems

None yet.

## Learnings

pending-remote. The C++ wrapper had event-based ordering for combine and LL
paths but every Rust call site passed `compute_stream=0`, silently falling
back to host sync. The intranode dispatch path had no event support at all.
The `recv_topk_idx` roundtrip was a separate host stall on the same path.
