# G5 Coordinator T2 Disk Wireframe

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T10 and
`docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md` G5.

## Context

T10 was narrowed to G5 after the G4 Metal sampler gate was found invalid on
Linux: `cargo check --features cuda,no-cuda` does not typecheck Metal cfg code.
G5 is the CUDA scheduler-side Coordinator consumer stub for T2 disk
fetch/store. It must stay code-only and default-off until T4b runs the >=4k
SERVE workload.

## What Worked

- Added `SchedulerConfig::t2_disk_tier_enabled`, default `false`.
- Reused the existing `--disk-store-root` CLI override as the explicit opt-in:
  passing it sets the disk root and enables scheduler-owned T2 disk tiering.
- The Coordinator builder now attaches `DiskStore` only when T2 disk tiering is
  enabled. The scheduler still keeps its `session_disk_store()` API unchanged.
- Store consumer gate: T1 spill/drain returns immediately when both T2 disk and
  T3 remote are disabled, and skips `StoreTarget::Disk` submissions when T2 is
  off.
- Fetch consumer gate: staged readmission/prefetch plans containing T2 disk
  blocks do not submit disk fetches while T2 is off; a live waiting request
  falls back to cold prefill.

## Semantics

| Field / branch | Meaning |
| --- | --- |
| `SchedulerConfig::t2_disk_tier_enabled` | Scheduler-owned T2 disk fetch/store is allowed. Default false. |
| `--disk-store-root <path>` | Explicitly opts into T2 disk tiering and supplies the node-local root. |
| Coordinator `DiskStore` attachment | Defense-in-depth: default-off means disk store/fetch requests are not executable by the Coordinator. |
| T1 spill/drain gate | Prevents host-pinned pressure ticks from submitting T2 disk store work unless enabled. |
| staged fetch gate | Prevents staged readmission from turning a disk-only hit into disk I/O unless enabled. |

## Verification

```bash
NVCC_CCBIN=g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python cargo check -p infer --features cuda --no-default-features
NVCC_CCBIN=g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python cargo check -p infer --features cuda,no-cuda --no-default-features
cargo test -p infer --lib
```

- `cargo check -p infer --features cuda --no-default-features`: exit 0.
- `cargo check -p infer --features cuda,no-cuda --no-default-features`: exit 0.
- `cargo test -p infer --lib`: 588 passed, 0 failed, 14 ignored.
- The two CUDA checks emitted existing DeepSeek/main warning noise; no new
  error gate was introduced by the G5 wireframe.
- No SERVE bench was run. G5 is a default-off wireframe; bench is deferred to
  T4b's >=4k-token SERVE workload after P5 PID 28950 releases the GPU.

## Rule

Code-only tier wireframes must be default-off at both policy and Coordinator
construction boundaries. A benchmark-dependent path may land only with an
explicit deferred bench gate and a wins entry that says which workload licenses
activation.
