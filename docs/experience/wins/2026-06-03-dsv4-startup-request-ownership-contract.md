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

Follow-up remote verification on pod `/data01/build/arle`, commit `cb1fb328`:

- `scripts/dsv4_fast_build.sh` used the prebuilt fast path and completed in
  28.61 s without harvesting stale `OUT_DIR` artifacts.
  Artifact: `/tmp/dsv4_fast_build_cb1fb328_20260602_163334.log`.
- High-performance TP8 + EAGLE startup with
  `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`, FP8 KV,
  `--cuda-graph-max-bs 16`, native DeepEP, DeepGEMM, shared KV pool, and
  `--spec-draft-model eagle` still failed closed before serving opened.
  Artifact: `/tmp/dsv4_eagle_contract_cb1fb328_20260602_163421.log`.
- The startup contract now reports the scheduler-planned route:
  `request_ownership=token-owned-dp-ep`,
  `request_effective_world_size=8`, and `token_owner_groups=1`
  (`groups=[[0,1,2,3,4,5,6,7]]` for the TP8/no-attention-DP target).
- The stale token-owned/owner-group blockers are gone from the fail-closed
  list. The remaining blockers are the real executable gaps:
  full-decode CUDA graph capture/replay, DeepEP/NCCL collective graph replay,
  frozen-KV EAGLE graph replay, graph-captured FlashMLA/SWA/C4/C128 metadata
  replay, and batched decode attention without host `start_pos` loops.

## Rule

Do not infer DSv4 request ownership from TP/EP size. Startup validation must
consume the exact scheduler-planned ownership contract, or the fail-closed list
mixes stale blockers with real graph-path gaps.
