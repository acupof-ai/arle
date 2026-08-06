# Lever 2 — checkpoint reload-to-device + pinned async offload (80K OPD backward)

**Date:** 2026-08-06 · **Pod:** 8×H20, ThinkingCap-Qwen3.6-27B-FP8, LoRA, cp=2 seq=81920

> Status: pending-remote. Two arms land behind two default-off flags; the pod A/B
> against the 315.7 s backward baseline licenses each flip separately.

## Context

The 80K OPD backward is 315.7 s, of which **131.7 s is on-CPU host work not inside
any CUDA API** ([memory](../../../CHANGELOG.md), Probe A/B/C/D, 2026-08-06). The
flamegraph bucket split puts ~85% of that in the checkpoint offload/reload round
trip, and the caller-frame probe resolved the two biggest rows:

| Row | Share of 131.7 s | Cause |
|---|---:|---|
| `upload_slice` → `cuMemcpyHtoDAsync` | 36% | pageable HtoD; the host thread blocks inside the driver's staging copy |
| `reshape`/`rmsnorm` → libc `memcpy` | 21% | recompute-forward on a **host-resident** activation takes the host `Vec` path |
| `offload_checkpoint_to_host` → `cuStreamSynchronize` | 17% | the DtoH is synchronous; the host spins |
| `cuda_download` → DtoH | 9.4% | same DtoH, attributed to the callee |

Graph-capture (13% / 20 s) and the `OPD_SEQ_CHUNK` knob (null) are already
rejected. Copy-engine traffic in-window: HtoD 327.8 GB / 38.16 s, DtoH 101.5 GB /
32.04 s, against 150.2 s of pure GPU idle.

## Arm 1 — `--checkpoint-reload-device` (reload the offloaded hidden to device)

`ensure_checkpoint_device` (tensor.rs) restored **only** `CheckpointResidency::L3`.
It was added by the L3 commit `ac348032c` for the case where the payload is not in
`tensor.data` at all; the `Host` case was left to lazy per-op `ensure_device`. But
the replay's forward ops gate on device residency (`reshape` at ops/layout.rs:22,
`slice`, `silu`, … each fall to a host `Vec` path when `dirty == Host`), so a
host-offloaded hidden makes the whole recompute chain repack host-side and then
re-upload the repacked copy. `ops/checkpoint.rs:44-47` already documents the
intended contract as "re-fetches via ensure_device" — the implementation never
matched it for the Host case.

The flag makes `Host` reload too. Predicted mechanism: the 21% `reshape` host
`memcpy` row disappears (its input is now device-resident) and redundant HtoD bytes
drop. `seq_chunked_recompute_backward` re-parks the full-seq input after its chunk
loop, mirroring `checkpoint_backward`, so the device high-water stays at one hidden.

**Cost:** one resident hidden of VRAM during the replay (`[1, seq/cp, dim]`). Peak
is 92/97 GB with ~5 GB free, so this is the flag's only real risk.

## Arm 2 — `--checkpoint-pinned-offload` (pinned host storage, async copies)

Pending. Design note: the copies do **not** need a second stream. The backward is
72% GPU-idle, so the constraint is the *host* blocking inside pageable copies, not
a lack of GPU overlap. Pinned host memory on the existing default stream makes
`cuMemcpyHtoDAsync`/`DtoHAsync` return to the host immediately and removes the
`synchronize`; the copy then serializes with compute on a stream that is idle
150 s. That drops the cross-stream event/fence machinery and its three
silent-corruption modes entirely.

Second design note: cudarc's `alloc_pinned` sets `CU_MEMHOSTALLOC_WRITECOMBINED`,
so host **reads** from a pinned buffer are uncached and slow. The pinned buffer
therefore *owns* the offloaded activation for its whole host residency — there is
no `pinned → Vec` copy on either leg. A pass-through staging design would have
added ~100 GB of write-combined host reads and could have been slower than the
pageable path it replaced.

## Results — pending-remote

| arm | backward | step | loss | grad_norm | peak VRAM |
|---|---:|---:|---:|---:|---:|
| baseline (`7da312d0d`, both flags off) | 315.7 s | — | 4.537510 | 7.965–7.985 | 92/97 GB |
| `--checkpoint-reload-device true` | | | | | |
| + `--checkpoint-pinned-offload true` | | | | | |

Gate: parity is the license, not the serve gates. `needle_gate.py` / `lever_gate.sh`
boot `arle serve` (inference forward) and never run the training backward, so the
correctness license here is the OPD step's loss/grad_norm against 4.537510 /
7.965–7.985. Both arms reorder *when* bytes move, never *what* is computed, so
parity must hold exactly.

Re-profile check: `upload_slice` 35% and `reshape` 16.2% must both fall. If the
backward drops but those rows do not, the mechanism is not the one claimed.

## Problems

None yet — nothing measured.

## Learnings

pending-remote.
