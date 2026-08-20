# Stats timeout fix: /metrics returns non-zero spec chains — CUDA, 2026-08-20

> Status: Verified on pod (DSpark DSv4-Flash, TP=2, H20)

## Context

The `/metrics` endpoint returned all-zero spec decode counters
(`arle_spec_chains_total` = 0) even when DSpark speculative decoding was
actively running. Root cause: the coordinator queried worker stats with a 2s
timeout and discarded the entire snapshot if any rank missed it. Under load
(long prefill, busy decode), workers routinely missed the 2s window.

## What Worked

Two-part fix in `coordinator.rs` (`c7eb23420`):

1. Return partial ranks on timeout instead of discarding the entire snapshot.
   Counters are monotonic — a partial snapshot underestimates but never
   overestimates. Gauges may overestimate, but all-zero is strictly worse.
2. Increase `/metrics` timeout from 2s to 30s.

## Result

DSpark serve on DSv4-Flash-0731 (NVFP4, TP=2, 6 KV slots, H20 GPUs 1,7):

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| `arle_spec_chains_total` | 0 | 333 |
| `arle_spec_accept_rate` | 0 | 46% |
| `arle_spec_drafted_total` | 0 | 780+ |
| `arle_requests_completed_total` | 0 | 16 |

Also verified: 10 concurrent requests against 6 KV slots — admission control
throttles correctly, no crash, no OOM. Long-prefill (≥64 token prompt)
produces coherent output.

## Environment

- Host / GPU: H20 pod, GPUs 1,7 (96GB each)
- Model: DeepSeek-V4-Flash-0731 (NVFP4, 43 layers, 256 experts)
- Draft: DeepSeek-V4-Flash-DSpark-draft-fp8 (3 stages, block=5, target=[40,41,42])
- TP: 2, slots: 6, max_seq_len: 131072
- Binary: `stats-fix2` (commits `c7eb23420` + `a1b5dffe2`)

## Rule

A metrics endpoint that returns all-zero under load is worse than no metrics
at all — it looks like the system is idle when it is not. Partial snapshots
with monotonic counters are always safe; the timeout window must exceed the
longest expected worker stall.
