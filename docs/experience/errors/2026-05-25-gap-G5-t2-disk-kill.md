# G5 T2 Disk License Kill

## Context

G5 asked whether ARLE should spend implementation budget on the CUDA HiCache
T2 disk hot path. The license gate in
`docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md` requires a
wall-clock/metric measure before implementation:

- PASS: real workload keeps T0 full for at least 20% of the measured time and
  evicted prefixes later hit T1/T2 at least 90%.
- KILL: T0 is rarely full, or eviction does not turn into useful tier hits.

T4b had already shown one useful T1 staged-readmission control but no T2 store:

- `bench-output/2026-05-25-t4b-kv-tier-observability-readmission-4k/`
- T0->T1 demote bytes: 214,106,112.
- Staged readmission: one T1 fetch, 58.492 ms raw wait.
- T1->T2 store bytes: 0.

G5 then ran two direct pressure controls with T2 explicitly enabled through
`--disk-store-root`.

## Root Cause

The current hot path does not produce enough useful T2 work. Under normal 4k
session pressure, T0 does not stay full long enough. Under stronger pressure,
the scheduler hits retain high-water/hard-cap behavior and skips publishing
new prefixes rather than producing broad T1/T2 evict-and-readmit traffic.

Measurement 1: five 4k sessions.

- Artifact: `bench-output/2026-05-25-gap-G5-t2-license-measure/`
- Server: Qwen3-4B, 4 slots, 4096-token prompts, 128 MiB T1, T2 disk enabled.
- Peak `kv_util`: 62.57%.
- `kv_util >= 0.80`: 0%.
- `kv_util >= 0.95`: 0%.
- T0->T1 demote bytes: 0.
- T1->T2 store bytes: 0.

Measurement 2: ten 4k sessions.

- Artifact: `bench-output/2026-05-25-gap-G5-t2-license-measure-10session/`
- Same server shape, ten session-tagged 4096-token prompts, then A readmission.
- Peak `kv_util`: 99.95%.
- `kv_util >= 0.80`: 22%.
- `kv_util >= 0.95`: 12%.
- T0->T1 demote bytes: 2,433,024.
- T1->T2 store bytes: 0.
- Readmission fetch wait: 1.180 ms, source `h:2/d:0/r:0`.
- Final cumulative prefix hit rate: 9.09%, far below the 90% gate.

Server log evidence from the 10-session run:

```text
prefix cache demotion: released 2 pool pages back to free list
prefix cache publish skipped ... retain hard cap hit or high-water pressure would start synchronous eviction
Request 10: staged prefix ready in 1.2ms src=h:2/d:0/r:0 waiters=1
Request 10: paged prefix ATTACH 4095/4096 tokens
```

Interpretation:

- T0 can briefly touch full, but not for the required 20% of samples at the
  `>=95%` full threshold.
- The only measured readmission was a tiny T1 readmission, not T2.
- Disk store remained zero despite T2 being enabled, so the T2 transport is not
  the licensed next optimization axis.
- This is consistent with T4b: T1 observability/readmission is real; T2 disk
  value is not demonstrated by current workloads.

## Fix

KILL G5 implementation work for now. Keep the default-off T10 Coordinator T2
wireframe, but do not turn it into a scheduler policy/hot-path project until a
future real workload demonstrates both sustained T0 pressure and high tier-hit
readmission.

If this area is reopened, the next license should target publish-under-pressure
policy separately from disk transport. The current failure mode is that high
pressure skips new prefix publication before T2 can become useful.

## Rule

Do not license a storage tier because the transport exists. Tier work needs
wall-clock pressure and a readmission hit-rate denominator that proves evicted
prefixes come back often enough to beat recompute/cold prefill.

## Artifacts

- `bench-output/2026-05-25-gap-G5-t2-license-measure/summary.json`
- `bench-output/2026-05-25-gap-G5-t2-license-measure/request-log.jsonl`
- `bench-output/2026-05-25-gap-G5-t2-license-measure/stats-trace.jsonl`
- `bench-output/2026-05-25-gap-G5-t2-license-measure-10session/summary.json`
- `bench-output/2026-05-25-gap-G5-t2-license-measure-10session/request-log.jsonl`
- `bench-output/2026-05-25-gap-G5-t2-license-measure-10session/stats-trace.jsonl`
- `docs/experience/wins/2026-05-25-kv-tier-observability-serve-baseline.md`
- `docs/experience/wins/2026-05-25-gap-G5-coordinator-stub.md`
