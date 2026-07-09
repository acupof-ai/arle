# Runtime env→CLI-flag sweep — behavior-preserving, remote A/B smoke pending-remote

## Context

`feedback_runtime_config_cli_flags_not_env` final cleanup: ~55 runtime env
vars deleted in favor of serve/train CLI flags riding
`EngineLoadConfig.{cuda,metal}` runtime-flags structs (multiproc workers
inherit via the engine-config transport). Commits 4d8c8c827, 0318109e9,
4adeb4656, f7b2467cb. Deferred: reads inside frozen `dsv4.rs`
(spec-decode / decode-graph / whole-step-graph / moe-transport /
lm-head-shard) — convert when that file unfreezes.

## What Worked

All defaults byte-identical by construction (each flag default equals the old
env-unset behavior); gates green locally: cuda,no-cuda + metal + vulkan
typechecks, cpu test lane, new `metal_speculative_flags_ride_engine_config`
test, clippy clean on changed files.

**Bench: pending-remote.** One `scripts/bench_guidellm.sh` smoke vs the
latest CUDA baseline is owed at the next pod session to confirm the
flag-plumbing added no load-path regression (expected Δ ≈ 0%).

## Rule

Runtime knobs are CLI flags from birth; env vars are for build/toolchain,
IPC transport, ecosystem standards (`SGLANG_*`, `HF_*`), and profiling
instrumentation only.
