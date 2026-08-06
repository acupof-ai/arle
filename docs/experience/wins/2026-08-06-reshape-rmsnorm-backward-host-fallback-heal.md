# reshape/rmsnorm backward host-fallback contagion — heal the upstream grad

**Date:** 2026-08-06 · **Commit:** `7da312d0d` · **Status:** pending-remote
(no local GPU — CpuBackend never takes the `device() != Cpu` branch, so the heal
is inert locally; pod parity + re-profile is the gate) ·
Ref: [[reference_opd_backward_is_72pct_host_idle_launch_bound]]

## Context

Probe C flamegraph of the 80K OPD backward (292,949 IP samples, 100%
threadState=Running) put **21% of on-CPU self-time in `reshape`/`rmsnorm`
backward → libc `memcpy`** — a host-side copy inside a CUDA-backend backward that
moves 327 GB HtoD + 101 GB DtoH. Suspicious: reshape backward is metadata-only,
rmsnorm backward has a device kernel.

## Root cause

Both ops gate device-vs-host on per-tensor residency (`layout.rs:498`,
`norm.rs:190`) but — unlike `matmul_backward` (`matmul.rs:150-154`) — never call
`ensure_device` on their operands first. So one `Dirty::Host` upstream grad drops
them to the host fallback: `tensor_host` (`tensor.rs:1023`) deep-clones the full
grad `Vec<f32>` (the memcpy) + a DtoH readback via `ensure_host`, and the
host-built grad forces a downstream HtoD re-upload. The demotion is contagious —
each host-fallback grad demotes its consumers, so one host grad upstream (e.g.
from a host concat/accumulate backward) cascades through every reshape/rmsnorm
below it.

## Fix

Mirror `matmul_backward`'s heal: when `device() != Cpu`, `ensure_device` the
upstream grad (reshape) / upstream+x+weight (rmsnorm) before the gate, so the
device kernel path is taken and no host round-trip occurs. +16/−2 lines, two
files, in-tree pattern copy — no new kernel, no dep change.

## Result

Pending-remote. Pod gate (via devops):
1. **Parity** — `needle_gate` ×3 + `lever_gate.sh`: the healed device path must be
   f32-consistent with the prior host fallback (this is the correct-inference
   license; the fix routes a numerically-equivalent path, so parity must hold).
2. **Re-profile** — one 80K backward, confirm the `reshape`/`rmsnorm`→memcpy
   self-time (was 21%) drops and the associated DtoH/HtoD contagion shrinks.
3. **Step wall** — matched A/B vs the 315.6 s champion backward; this is the
   first of the two-lever offload attack, so record the delta before Lever 2
   (pinned+async offload pool) so its baseline is uncontaminated.

## Rule

A device op that gates on operand residency without first healing host-resident
operands is a latent host-fallback trap: one host grad upstream cascades. When
adding a device/host gate, copy the sibling op's `ensure_device` heal — a gate
without a heal silently demotes the whole downstream chain.
