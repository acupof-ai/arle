# CUDA event pool for pipeline fences — CUDA, 2026-08-17

> Status: **Shipped; root cause was deeper than event allocation.** The event
> pool eliminated per-fence `cuEventCreate`/`cuEventDestroy`, but the decode
> regression persisted because the event pool commit (`9a82dbe4d`) accidentally
> reverted `a8fb1047b`'s fix that kept `all_reduce_sum` on the compute stream.
> The AR is on the layer's critical path — comm-stream overlap cannot hide it,
> and the fence bracket (record+wait per collective) adds overhead regardless
> of event reuse. Fix: keep `all_reduce_sum` on the compute stream
> (`df9b6a15c`). Measured TP=8 @100K: 70–80 tok/s (matching the 78.7 baseline).
> One-shot path measured slower (52 tok/s) — NCCL stays the default.

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

TP=8, ThinkingCap-Qwen3.6-27B-FP8, 100K context, `decode_rate_probe.py --runs 5`:

| build | decode tok/s @100K | notes |
|---|---:|---|
| pre-comm-stream (4bcefcb57) | 78.7 @128K | baseline |
| comm-stream, per-fence alloc (a59c6c661) | 55–59 | −25–30% |
| + all-reduce → compute stream (a8fb1047b) | 63 | partial recovery |
| + event pool, AR back on Comm (9a82dbe4d) | 48–64 | event pool alone insufficient |
| + one-shot enabled (--comm-backend auto) | 51–53 | one-shot slower than NCCL |
| **+ AR on Compute stream (df9b6a15c)** | **70–80** | **recovered** |

Needle gate (512/4096/16384/32768 ×3 runs): 12/12 exact, PASS.

The event pool is still useful for prefill/CP collectives that stay on the
comm stream, but it is not the decode fix. The decode fix is keeping
`all_reduce_sum` on the compute stream (no fences on the critical path).

## Problems

**Event pool alone did not fix the regression.** The pool eliminated per-fence
allocation, but the record+wait CUDA API calls per collective (128+ per decode
step at TP=8) still added overhead. The AR is on the layer's critical path —
the next layer reads its output — so comm-stream overlap cannot hide the fence
cost. The correct fix is keeping the AR on the compute stream (unfenced),
which `a8fb1047b` already did but `9a82dbe4d` accidentally reverted.

**One-shot path is slower on Qwen3.5/3.6.** Measured 52 tok/s with one-shot
vs 75 tok/s with NCCL on the compute stream. The previous DSv4 measurement
(769920038) showed one-shot as wall-neutral, but Qwen3.5/3.6's 64-layer
decode path with 1-2 ARs per layer amplifies the one-shot signal/staging
overhead. NCCL stays the default; one-shot is opt-in.

## Learnings

The fence overhead has two components: event allocation (fixed by the pool)
and stream synchronization (record+wait per collective, NOT fixed by the
pool). On a strictly-dependent decode chain, the only way to eliminate the
sync overhead is to keep the collective on the compute stream — the comm
stream is for prefill overlap, not decode.

**"Wall-neutral" on one model does not transfer.** The one-shot path was
wall-neutral on DSv4 but 30% slower on Qwen3.5/3.6. Per-op latency
improvements don't predict wall-clock when the chain is skew-bound.
