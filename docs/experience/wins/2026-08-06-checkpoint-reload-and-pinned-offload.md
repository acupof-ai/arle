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

## Arm 2 — `--checkpoint-pinned-offload-bytes` (pinned host storage, async copies)

`CheckpointResidency::Pinned(slot)` parks the activation in a backend-owned pinned
buffer: `offload_checkpoint_to_host` issues one async DtoH into it,
`ensure_device` issues one async HtoD out of it. Both are unconditional in
`ensure_checkpoint_device` — pinning an activation commits to reloading it. The
flag is a byte budget (0 = off); above it the pageable path takes over, so the
pinned footprint is a hard ceiling, which matters because the pages are
unswappable and one rank already carries ~171 GB host RSS.

**No second stream.** The plan called for a copy stream with event fences. It is
not needed and was dropped: the backward is 72% GPU-idle, so what has to stop
blocking is the *calling thread*, not the copy engine. `cuMemcpyDtoHAsync` blocks
the host on pageable memory and returns immediately on pinned; that alone recovers
the 36% + 17% + 9.4% rows. Keeping both legs on the existing default stream makes
all three of the plan's silent-corruption modes disappear by construction:

| Risk | Guard |
|---|---|
| pinned buffer reused while its copy is in flight | reuse re-enqueues on the same stream, ordered behind the previous copy; the one host read (`checkpoint_pin_readback`) drains the stream |
| device buffer freed under an in-flight DtoH | `CudaSlice::drop` frees via `cuMemFreeAsync` on the same stream, so the free is ordered behind the copy |
| consumer reads a reloaded handle before the HtoD lands | same stream, so stream order is the ordering |
| pool dropped under an in-flight DtoH | copies pass plain slices, which record no event on the buffer, so `PinnedCheckpointPool::drop` drains the stream before `cuMemFreeHost` |

**No `pinned → Vec` leg.** cudarc's `alloc_pinned` sets
`CU_MEMHOSTALLOC_WRITECOMBINED`, where host reads are uncached. The pinned buffer
owns the activation for its whole host residency; nothing copies out of it except
the rare `ensure_host` fallback. A pass-through staging design (the plan's shape)
would have added ~100 GB of write-combined host reads and could have come out
slower than the pageable path it replaced.

Slot capacities round to 64 MiB with best-fit reuse and an exact-fit fallback
(`aa5b1f820`, `d1870526f`): exact-length reuse plus varying OPD trajectory
lengths exhausted the budget on the first size classes and silently disabled the
pinned path. Every copy names its own length, since cudarc's typed memcpy copies
the whole destination.

`d1870526f` also removes a full-source clone in `slice_host_eager` (it cloned
the whole tensor to read one chunk), which cuts the baseline arm's replay memcpy
too — all arms must be measured on the same tree.

Precedent for DtoH into cudarc write-combined pinned memory:
`infer-cuda/src/qwen35.rs:1396` (recurrent-state snapshot), in production.

## Results — measured 2026-08-06, pod GPUs 4+5, tree `d7ecbbcee`, one rep/arm

| arm | backward | step | loss | grad_norm | peak VRAM (cp0/cp1) | host RSS |
|---|---:|---:|---:|---:|---:|---:|
| A baseline (both flags off) | 304.6 s | 372.1 s | 4.537510 | 7.981584 | 91.5 / 89.9 GB | 52.9 GiB |
| B `--checkpoint-reload-device true` | **121.8 s (−60.0%)** | 192.1 s | 4.537510 | 7.970384 | 88.7 / 89.1 GB | 49.0 GiB |
| C = B + `--checkpoint-pinned-offload-bytes 8589934592` | 115.3 s | 180.1 s | 4.537510 | 7.967986 | 87.4 / **97.4** GB | 49.3 GiB |

Verdict:
- **B accepted, default flipped to on** (this entry's commit). −182.8 s is 19× the
  9.8 s run-to-run spread; loss bit-identical, grad_norm in the 7.965–7.985
  envelope. The predicted "one extra resident hidden" VRAM cost never
  materialized — B's peak is *lower* than A's on both ranks (−2.9 GiB cp0),
  and host RSS fell 3.9 GiB with the pageable round-trip gone.
- **C rejected, pinned pool stays 0 (off).** −6.5 s over B is inside the 9.8 s
  spread, and its cp1 peak reached 97.4 of 97.9 GB — 449 MiB headroom
  (rank-asymmetric, cause unknown). The pool emits no engagement probe, so
  there is no positive evidence it fired; adjudicating it needs an engagement
  counter plus 3 reps, and the upside on the table is a wash.
- A reproduces the `7da312d0d` baseline (304.6 vs 315.7 / 307.7 prior reps,
  loss exact), so the `slice_host_eager` borrow did not move the baseline arm.

Three arms, one binary, one flag changed per step. The pinned arm rides on top of
the reload arm because pinning implies reloading; that ordering is why the reload
arm is measured first rather than bundled.

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
