# c1 single-stream decode gap parallel track

## Context

After the evictable-prefix admission patch (`051b1081`) closed the W1/c4
SGLang row, the supplementary c=1/360s measurement was kept as no-regression
evidence and as an independent optimization track.

Command:

```bash
LONGCTX_SECONDARY_C1_ONLY=1 WORKLOAD=longctx-32k LONGCTX_MAX_SECONDS=360 \
  scripts/bench_guidellm.sh phase15-evictable-c1-guard \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Root Cause

The c=4 admission fix removes the KV-pool edge deadlock but does not optimize
the single-stream c=1 decode path. The measured c=1 row is close to the
pre-patch ARLE c1 anchor, so this is not a regression from the evictable
admission patch.

Observed c=1 guard:

| metric | ARLE | SGLang reference | delta / ratio |
|---|---:|---:|---:|
| GuideLLM out tok/s | 9.83 | 11.57 secondary | 0.850x |
| effective `total_output_tokens / 360` | 9.244 | 11.57 secondary | 0.799x |
| TTFT p50 | 12540.6 ms | 11862.86 ms secondary | +5.7% slower |
| ITL p50 | 56.84 ms | 43.10 ms secondary | +31.9% slower |

## Fix

- Do not block the W1/c4 Phase 1 close on this supplementary row.
- Track c=1 as a parallel single-stream decode profile/optimization task.
- Keep future Phase 2 W2 work guarded by the W1/c4 no-regression row.

## Rule

When a supplementary benchmark row is outside a watch line, keep an explicit
follow-up record even if the mission-critical row is closed.
