# TP/CP NCCL collectives → comm_stream — CUDA, 2026-08-17

> Status: Shipped (smoke + needle gate PASS)

## Goal

Move NCCL collectives off the compute stream so communication can overlap
compute. The strictly-dependent decode chain (AR output → residual add →
next layer) has no slack, so this is a wash on decode by construction; the
value is the stream plumbing that enables the T3 CP-decode merge path
(attn_tp all-reduce + cp row-gather per layer) to overlap compute.

## Hypothesis

Bracketing each NCCL collective with `comm_waits_for_compute` /
`compute_waits_for_comm` fences and running the NCCL enqueue on
`comm_stream` produces identical results with no decode regression (the
fences add event create/destroy overhead but no host stall).

## Parameters

```bash
# A/B: baseline = a59c6c661^, treatment = a59c6c661
# ThinkingCap-27B-FP8, TP>=2 (NCCL arm, not one-shot)
python3 scripts/bench_throughput.py \
  --url <url> \
  --model ThinkingCap-27B-FP8 \
  --prompts-jsonl bench-agent-119k-16x8.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 128 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/comm-stream/bench
```

- Baseline: `a59c6c661^` (NCCL on compute stream, no fences)
- Treatment: `a59c6c661` (NCCL on comm_stream, fenced)
- Trials: 3 (matched A/B, simultaneous)

## Environment

- Host / GPU: 8×H20 pod (sm_90)
- Driver / CUDA: TBD
- Model / dtype: ThinkingCap-27B-FP8
- TP / EP / slots / KV: TP=8 (or ≥2)
- Server flags: default

## Results

Smoke test (2026-08-17, pod 8×H20, TP=8, ThinkingCap-Qwen3.6-27B-FP8):
serve ready, all 8 workers loaded, coherent generation (thinking model),
no NCCL errors. Needle gate ×3 (LENGTHS=8000, RUNS=3): exact=3/3,
correctness PASS.

| concurrency | arm | completed | errors | output tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | | | | | | | — |
| 1 | treatment | | | | | | | |

A/B bench: pending (decode is a wash by construction — the strictly-dependent
chain has no slack; the value is the stream plumbing for T3.2 CP-decode merge).

Raw artifacts: `/host/arle-runs/212-comm/` on pod.

## Problems

None.

## Learnings

PASS (smoke + needle). The fence machinery (`comm_waits_for_compute` /
`compute_waits_for_comm`) already existed in tensor.rs and was used by the
dsv4 shared-expert overlap path; this change generalizes it to all NCCL
collectives. The dsv4 prefill shared-expert overlap is preserved by keeping
the moe all-reduce on the compute stream via `all_reduce_sum_on(Compute)`.
One-shot all-reduce stays on the compute stream (small-message fast path;
moving it is a follow-up — `arle_car_allreduce_bf16_into` already takes a
stream parameter).
