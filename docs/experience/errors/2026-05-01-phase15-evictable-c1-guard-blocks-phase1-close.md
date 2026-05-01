# Phase 1 close blocked by c1 S5 guard

## Context

After the evictable-prefix admission patch (`051b1081`) cleared the c=4
longctx row by successful-only accounting, `codex review --uncommitted`
correctly flagged that the mission doc's Phase 1 S5 gate also requires a
c=1/360s guard.

The guard was run on 2026-05-01:

```bash
LONGCTX_SECONDARY_C1_ONLY=1 WORKLOAD=longctx-32k LONGCTX_MAX_SECONDS=360 \
  scripts/bench_guidellm.sh phase15-evictable-c1-guard \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Root Cause

The c=4 admission fix removes the KV-pool edge deadlock but does not improve
single-concurrency long-prompt prefill enough to match SGLang.

Observed c=1 guard:

| metric | ARLE | SGLang reference | ratio |
|---|---:|---:|---:|
| GuideLLM out tok/s | 9.83 | 11.57 secondary | 0.850x |
| effective `total_output_tokens / 360` | 9.244 | 11.57 secondary | 0.799x |
| TTFT p50 | 12540.6 ms | 11862.86 ms secondary | +5.7% slower |
| ITL p50 | 56.84 ms | 43.10 ms secondary | +31.9% slower |

## Fix

- Do not declare full Phase 1 S5 closed from the c=4 row alone.
- Treat the c=4 row as "1.30x margin secured" and the c=1 row as the current
  Phase 1 blocker.
- Next engineering work should profile the c=1 long-prompt path before Phase 2
  implementation starts.

## Rule

When a phase gate cites a multi-row acceptance criterion, the closeout wins
entry must include every required row or explicitly mark the missing/failing
row as a blocker.
