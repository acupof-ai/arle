# CP prefill snapshotted blind-tail fix — CUDA, 2026-08-18

> Status: Shipped

## Goal

Fix TP=2 CP=2 needle loss at len=2000 and server crash at len=8000+ on
the ThinkingCap-Qwen3.6-27B-FP8 model. CP=1 control passed 21/21; CP=2
failed both.

## Hypothesis

`prefill_row_snapshotted` splits the prefill at L* (and periodic stride
boundaries) for recurrent-state snapshots. Under 2D, each segment's ring
prefill attends only to the current segment's rotating KV — it cannot
read prior segments' pool KV. The tail segment's last-token hidden state
is blind to the prefix, losing the needle. Skipping the snapshotted path
under 2D (`!two_d_engaged()` guard) lets the single-pass
`prefill_row_paged_default` cover the entire prompt in one ring pass.

The same root cause drives the len=8000+ liveness bug: the snapshotted
path's per-segment `snapshot_recurrent` D2H copies (36 serializations,
18 requests × 2) block the host between forward passes, stalling the
lockstep coordinator past its 120s timeout.

## Parameters

```bash
TEMPLATE=qwen3_nonthink RAW=1 PORT=18189 \
  python3 scripts/needle_gate.py 115,300,446,2000,8000,8192,16384 3
```

- Baseline: `cpfix3` build (pre-fix), TP=2 CP=2
- Treatment: `e767bd5ac` (fix), TP=2 CP=2
- Needle: `738291`, depth=0.0 (position 0)
- Trials: 3 per length, 7 lengths = 21 requests

## Environment

- Host / GPU: H20 pod, 8× H20 (sm_90, 97 GB each)
- Driver / CUDA: CUDA 12.x
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8, BF16 KV pool
- TP / CP: TP=2 × CP=2 = world 4, GPUs 0,2,5,7
- Server flags: `--tensor-parallel-size 2 --context-parallel-size 2 --port 18189`

## Results

| len | baseline (cpfix3) | treatment (e767bd5ac) |
|---:|---|---|
| 115 | 3/3 exact | 3/3 exact |
| 300 | 3/3 exact | 3/3 exact |
| 446 | 3/3 exact | 3/3 exact |
| 2000 | 0/3 (needle lost) | 3/3 exact |
| 8000 | server crash | 3/3 exact |
| 8192 | server crash | 3/3 exact |
| 16384 | server crash | 3/3 exact |

Treatment: 21/21 exact, zero lockstep stalls, zero server errors.
Sidecar serializations dropped from 36 (18 req × 2) to 18 (18 req × 1,
the `save_recurrent_sidecar` fresh snapshot for prefix-matched requests).

Wall-clock per request (treatment): len=2000 1.1-1.3s, len=8000 3.0s,
len=16384 5.8s.

Raw artifacts: `/root/arle-ops/runs/tp2cp2-fix/needle.log`,
`/root/arle-ops/runs/tp2cp2-fix/log`.

## Problems

Build failed on first attempt: `rust-lld: undefined symbol: nvfp4_to_w4afp8`.
The `.cu` file was pushed to the pod mid-build (timestamp 04:48 vs build
start 04:41), racing the archive creation. Clean rebuild resolved it.

## Learnings

PASS. The 2D ring prefill's "attends only to rotating KV" contract means
any prefill split (snapshotted segments, chunked prefill) breaks causal
attention across the split. The planner already prevents chunked prefill
under 2D (`planner.rs:74-86`, `kv_shard_spec().is_some()` → one full-prompt
row); the executor's `prefill_row_snapshotted` was the remaining split
point. The fix is a one-line guard; the structural invariant is "under 2D,
the entire prompt lands in one ring pass, no exceptions."

The liveness bug was the same root cause compounded: each snapshotted
segment adds a host-blocking D2H snapshot, and the multi-segment forward
passes desync the collective stream. Eliminating the segments eliminates
both the correctness blind spot and the serialization-driven stall.
