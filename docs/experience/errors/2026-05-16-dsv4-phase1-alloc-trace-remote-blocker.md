# DSv4 Phase 1 allocation trace remote blocker - 2026-05-16

## Context

The DSv4 decode memory-access brief is in Phase 1: identify the current-main
top allocation callers before licensing any preallocation fix. The required
evidence is a single-in-flight 8xH20 `/root/DeepSeek-V4-Flash` profile with
`ARLE_CUDA_ALLOC_TRACE=1`.

Completed setup before this blocker:

- `8c8b90a` landed the FP4 batch GEMV scale-row hoist tranche.
- `ccd5f72` landed the DSv4 decode memory-access binding constraints audit.
- `a27ccff` added the default-off CUDA allocation trace substrate.
- `36f0143` taught the profile wrapper to emit `request-traces.json` and
  `cuda-alloc-trace-process-delta.{json,csv}` artifacts.

## Root Cause

The current local machine cannot produce Phase 1 ground truth:

- `nvidia-smi --query-gpu=name,count,memory.total --format=csv,noheader`
  reports one `NVIDIA GeForce RTX 4070 Ti SUPER, 1, 16376 MiB`, not the
  8xH20 topology used by the 2026-05-14 reference request.
- `ls -ld /root /root/DeepSeek-V4-Flash` cannot access the checkpoint path:
  `/root/DeepSeek-V4-Flash: Permission denied`.
- No allocation trace output exists yet under `docs/trace-artifacts/`:
  `cuda-alloc-trace-process-delta.{json,csv}` is absent.

Without that remote profile, the top allocation caller table is still
hypothesis-grade. Any preallocation fix would violate the brief's SOLID gate
and single-variable A/B rule.

## Fix

Run the current pushed `main` on the 8xH20 host:

```bash
ARLE_CUDA_ALLOC_TRACE=1 ./scripts/profile_dsv4_single_decode_nsys.sh \
  --out docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-alloc-trace
```

Then inspect:

- `cuda-alloc-trace-process-delta.csv` for top callers by calls and bytes.
- `summary.json` for `cuda_alloc_trace_found=true` and decode wave wall time.
- `request-traces.json` for request-level decode EMA and e2e token framing.

Only after that evidence exists should Phase 1 license one preallocation site
at a time.

## Rule

Hardware/data blocker is an explicit exit criterion for this brief. Stop here
rather than guessing from local SM89 behavior or stale 2026-05-14 aggregate API
totals. Remote 8xH20 caller evidence is the next required artifact.
