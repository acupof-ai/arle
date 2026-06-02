# DSv4 Startup Request Ownership Contract

## Context

The DSv4-Flash target remains the single-node TP8 + EAGLE hot-cache shape:
256K/1500, TTFT about 0.44 s, TPOT about 4.85 ms, E2E about 7.7 s, and output
throughput about 196 tok/s.

The existing DSv4 best-practice startup gate still inferred request ownership
from TP/EP world size alone. That made the gate keep reporting missing
token-owned request routing even after the HTTP scheduler had a
`token-owned-dp-ep` owner-group route for the SGLang profile.

## What Worked

- Added a typed `SchedulerStartupContract` that carries the scheduler-planned
  request ownership, effective world size, token-owner group count, and CUDA
  graph cap into `ModelForward::validate_scheduler_contract`.
- Populated that contract from the DSv4 serving bootstrap before scheduler
  construction. Under `sglang-best-practice`, ARLE now derives the SGLang owner
  groups before model startup validation; debug fallback remains
  `replicated-token`.
- Updated the DSv4 startup log to print `request_ownership`,
  `request_effective_world_size`, and `token_owner_groups`.
- Kept the high-performance path fail-closed for the real remaining blockers:
  full-decode CUDA graph capture/replay, graph-captured MTP/EAGLE draft,
  DeepEP/NCCL collective graph replay, FlashMLA/SWA/C4/C128 metadata replay, and
  batched attention without host start-position loops.

## Verification

Local:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote DSv4 fast-build and startup-contract verification are pending for this
tranche.

## Rule

Do not infer DSv4 request ownership from TP/EP size. Startup validation must
consume the exact scheduler-planned ownership contract, or the fail-closed list
mixes stale blockers with real graph-path gaps.
