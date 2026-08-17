# CUDA event pool for pipeline fences — CUDA, 2026-08-17

> Status: pending-remote (pod offline; build + bench queued)

## Goal

Recover TP=8 decode throughput to the pre-comm-stream baseline (78.7 tok/s
at 128K) without reverting the comm-stream architecture.

## Hypothesis

The comm-stream regression (78.7 → 55–59 tok/s) is caused by per-fence
`cuEventCreate` / `cuEventDestroy` pairs: 80 all-reduces × 2 fences = 160
event allocations per decode step at TP=8. Reusing events via a pool
eliminates the host-side allocation cost and recovers the baseline.

## Parameters

```bash
# A/B: baseline = 9a82dbe4d^ (comm-stream, per-fence alloc),
#      treatment = 9a82dbe4d (comm-stream, event pool)
# TP=8 decode-rate probe at 128K
python3 scripts/decode_rate_probe.py \
  --url <url> \
  --model ThinkingCap-27B-FP8 \
  --target-tokens 128000 \
  --max-tokens 128
```

- Baseline: `9a82dbe4d^` (comm-stream, per-fence `new_event(None)`)
- Treatment: `9a82dbe4d` (comm-stream, `Arc<Mutex<Vec<CudaEvent>>>` pool)
- Trials: 3 (matched A/B)

## Environment

- Host / GPU: 8×H20 pod (sm_90)
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8
- TP / EP / slots / KV: TP=8, CP=1
- Server flags: default

## Results

| build | decode tok/s @ 128K | Δ vs pre-comm-stream |
|---|---:|---:|
| pre-comm-stream (4bcefcb57) | 78.7 | — |
| comm-stream, per-fence alloc (a59c6c661) | 55–59 | −25–30% |
| + all-reduce → compute stream (partial revert) | 63 | −20% |
| + event pool (9a82dbe4d) | pending | pending |

## Problems

None yet.

## Learnings

pending-remote.
