# Pinned snapshot staging: prefill D2H 0.577 s → 0.062 s — 2026-08-05

> Status: **Confirmed, kept, default.** The transfer is 9× faster and the code
> is one path. TTFT did not move — the prediction that it would was wrong, and
> the reason is recorded below.

## Context

The 33K cold prefill copies the recurrent state to host every 2048 tokens
(`SIDECAR_SNAPSHOT_STRIDE_PAGES = 128` × page 16) so a later conversation can
restore a prefix instead of re-prefilling it. CUPTI measured 2.771 GB of D2H
costing 0.577 s over the prefill.

`cudarc`'s `impl<T> HostSlice<T> for Vec<T>` returns `SyncOnDrop::Sync(None)`:
every `memcpy_dtoh` into a `Vec` ends in a full stream synchronize. The
snapshot issues 96 of them (48 GDN + 48 conv layers).

## What worked

`Qwen35SlotState` now owns reusable `PinnedHostSlice` staging, allocated once
per occupant in `ensure_snapshot_staging` and freed with the slot. Pinned
slices sync on their own event, so the 96 stream synchronizes are gone and the
copies run at DMA speed.

GPU 6, `iso-tc-huihui-w8a16`, `bench-agent-32k-64.jsonl`, c=1, 16 requests ×
256 tokens, seed 20260416, two reps.

| | before (`0ac780495`) | after |
|---|---:|---:|
| D2H bytes | 2.771 GB | 2.771 GB |
| **D2H time** | **0.577 s** | **0.062 s** |
| effective bandwidth | ~4.8 GB/s | ~44 GB/s |
| TTFT p50 | 25.01 s | 25.29 / 25.35 s |
| ITL p50 | 16.69 ms | 16.70 / 16.68 ms |

44 GB/s is pinned bandwidth on this box; pageable is ~5. The treatment
engaged. TTFT and ITL are inside the ±3% drift band — unchanged.

One path, no flag: the `Vec` destination is deleted, not kept alongside. The
`pinned -> Vec` copy at the end stays deliberately — the snapshot outlives the
step and pinned memory is too scarce to hold for a cache entry.

## Why the TTFT prediction was wrong

The prediction was that the 18 prefill stalls of ~90 ms would collapse, because
0.577 s of D2H and 1.605 s of stall looked like part and whole. They are not.

Measured after the fact, on the same trace: **94% of all memcpy time sits
inside the big gaps** (0.0585 s of 0.062 s). The copies really are in the
stall. But removing 0.515 s of copy time shrank the stall by **zero**
(1.605 s → 1.672 s across 18 → 19 gaps). The copies were overlapped by a
longer host block, not serialized ahead of it.

Root cause of the misjudgment: I compared two aggregate totals and treated the
smaller as a serial component of the larger. **Containment is not
contribution.** The falsifying query was one SQL statement against a sqlite
file already on disk — intersect the memcpy intervals with the gap intervals
and check whether the gap has a residual once every overlapping activity is
subtracted. It was skipped because the mechanism story (`Vec` →
`SyncOnDrop::Sync` → 96 syncs) was correct and satisfying. A correct mechanism
is not an attribution.

The stall itself remains unattributed: 19 gaps, evenly spaced at the ~1.45 s
chunk cadence, each bounded by `argmax_kernel_fast` → `embedding_batched_native_kernel`
with no GPU work between them. Host time. The 151 MB `pinned -> Vec` copy
accounts for ~15 ms of the ~90; the rest is unknown.

## Rule

Attribute a stall by subtraction, not by size match. Before optimizing an
activity that sits near a gap, compute the gap's residual after removing every
activity that overlaps it — if the residual is the whole gap, the activity is
not the cause however plausible its mechanism.
