# Gate KV residency gauges on an explicit tier-presence signal — 2026-09-13

> Status: Shipped, CPU-only gates plus the Mac CUDA-stub CI-lint lane. No
> device run and no pod build (GPU embargo). Real-nvcc compile pending-remote.
> Closes findings #6 and #7 from the false-zero audit.

Rule: a surface publishes a number only when something measured it.

## Context

Two KV gauges were emitted unconditionally with a value of 0 on backends where
the quantity does not exist:

- #6 `kv_system_host_demoted_pages` / `kv_system_disk_pages`: backends with no
  page-tier view (HIP, Vulkan, OCR) and the CUDA *placeholder* executor
  substituted hardcoded zeros. On real CUDA the same series carry live
  whole-slot-store samples, so a zero meant two opposite things — "store empty"
  on CUDA, "no store here" on the others.
- #7 `kv_tier_resident_blocks`: structurally pinned to 0 on CUDA, which pins
  page-tier capacity to 0 and parks whole slots through the slot tier instead.
  The gauge read "page tiering idle" while slots were actively being spilled.
  Only Metal `--kv-disk` has a capacity-bearing page tier.

## What changed

Two explicit presence signals, carried in the core stats structs and the
multiproc wire DTO:

- `KvSystemMetrics.residency_measured` — a real tier store backs the host/disk
  residency gauges. Set from a new narrow seam predicate
  `KvPageTier::kv_tier_residency_measured()` (default false): real CUDA and
  constructed Metal return true; the CUDA placeholder and backends with no tier
  view return false. The default is false on purpose: a backend that has not
  declared the gauges measured gets them omitted, so adding a backend cannot
  silently publish a fabricated zero. The predicate is independent of page
  capacity, because CUDA's whole-slot store reports residency with page
  capacity pinned to 0.
- `KvTierStats.page_blocks_measured` — a capacity-bearing page tier backs the
  block gauge. Set to `kv_page_tier_view().capacity_pages() > 0`, so it is true
  only on Metal-with-disk and false on CUDA.

`render_prometheus` emits the three gauges only when their signal is true.
Reuse-hit counters in the same block stay unconditional: they are monotonic
event counters whose zero is honest. The wire DTO gains both bools with
`#[serde(default)]` (older workers read false → omitted, never zero-present);
presence is OR-aggregated across TP ranks and DP groups. OR (rather than
omitting when any rank is unmeasured) is correct because ranks in one
deployment are homogeneous: one measuring rank proves the quantity exists, and
the accompanying gauges are already min/sum aggregates of the ranks that
reported.

Gates: a metrics test renders the default snapshot and asserts all three series
are absent, then flips both flags and asserts the measured-zero series export.
38 infer-server and 20 infer-core tests pass; clippy `-D warnings` clean on the
default, Metal, and `cuda,no-cuda,nccl,deepep` CI-lint lanes (the last compiles
the new `#[cfg(feature = "cuda")]` arm without a toolkit). A real-nvcc pod
compile is the remaining check, deferred under the GPU embargo.

Scope: Prometheus, the audited surface. The `/v1/stats` JSON already carries
`kv_tier.available` and keeps a stable field shape; its host/disk fields were
not changed in this lane.

## Rule

Give every gauge that only some backends can measure an explicit presence
signal sourced from that backend's capability, and omit the series where the
capability is absent. A gauge with no signal cannot distinguish "measured zero"
from "quantity does not exist here".
