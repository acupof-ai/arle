# DSv4-Flash-FP8 TP=4 num_slots recovery: 4 → 52 (KV-budget trim + default + true-per-slot sizing)

> Status: verified 2026-07-04, 8×H20 pod, DeepSeek-V4-Flash-FP8, TP=4/EP=4,
> MTP-on (`--spec-type mtp`), greedy T=0, `INFER_DSV4_MAX_SEQ_LEN=5120`.
> Pinned to **GPUs 1,2,4,5** (non-contiguous — NCCL takes any 4 ordinals;
> 0/3/6/7 held a foreign 4-GPU job, untouched). Measured via `/v1/stats` +
> `dsv4_measure.py` stdlib client (guidellm pip is network-blocked on the pod).
> Anchors the num_slots-wall fix stack; supersedes the 4-slot wall in
> [2026-07-03-dsv4-fp8-tp4-138fixed-perf.md](2026-07-03-dsv4-fp8-tp4-138fixed-perf.md).

## Goal

Verify the three-commit KV-budget fix stack recovers DSv4 `num_slots` from the
hard 4-wall, boots without OOM, and lifts concurrent-decode throughput past the
old ~50 tok/s ceiling.

- `1ad322f9` — trim the cuMemAllocAsync pool before the budget's `mem_get_info`.
- `1ada98e0` — `num_slots` default 4 → 256 (the removed `--num-slots` flag's
  residual cap; the budget clamps to what VRAM affords).
- `8e77a9d9` — size the budget's `per_slot` by the **true** `Dsv4SlotState::device_bytes`
  (was a hand-rolled ~43 MB that missed the MTP `spec_verify` scratch).

## Result — num_slots 4 → 52, no OOM

**Verbatim budget log** (rank0, `dsv4.rs:1819`, post-fix):

```
DSv4 KV budget: free 22539MB, per_slot 385MB (slot-state 382MB + DSA key-cache 3MB + DSA batched 0MB; FP8 arena in shared pool), shared DSA 22MB, shared MoE decode 0MB, shared expert scratch 2MB, shared MLA decode 2MB, pool_per_layer 4MB, affordable 52
```
```
WARN DSv4 KV budget: requested 256 slots × ~385MB/slot ... exceeds the cross-rank-min affordable 52 ...; clamping num_slots to 52.
WARN CUDA engine: executor clamped slots 256 -> 52; scheduler follows
```

| quantity | before fix | after fix |
|---|---:|---:|
| requested num_slots | 4 (hard default) | 256 (default) |
| budget `per_slot` | 43 MB (under-count) | **385 MB** (= real slot0 382 MB) |
| `affordable` | ~471 (9× over) | **52** |
| **effective num_slots** | **4** | **52** |
| boot | (256 → OOM at ~slot 31) | **engine-ready, no OOM** |
| drift-guard (per_slot vs slot0 device_bytes) | n/a | **SILENT** (385 vs 382 = 0.8% < 5%) |

VRAM ledger (per rank, all 4 identical): after weights `used 76377MB free 21131MB`;
post-trim budget `free 22539MB` (**trim recovered only 1408 MB**); 52 slots ×
385 MB fit within 0.9 × free → final ~95.7 GB/GPU, engine ready.

## Two hypotheses falsified

1. **Retained-scratch (~27 GB) was wrong.** The trim recovered **1.4 GB**, not
   27 GB. Weights+context genuinely use ~74.6 GB/rank; free-after-weights ~21 GB.
   The old 4-slot wall was **purely the `num_slots` default of 4**, never scratch
   starvation — the trim fix (`1ad322f9`) is ≈ a no-op here (+1.4 GB ≈ +3 slots).
2. **`256` default alone OOMs.** Raising the default without correcting the
   budget's `per_slot` (43 MB, missing `spec_verify` 282 MB/slot) left
   `affordable` ~9× high → 256 never clamped → the slot-state loop OOMed at
   ~slot 31 (decoded from the boot log, pre-`8e77a9d9`). `8e77a9d9` sizing
   `per_slot` by the real 382 MB (`spec_verify` 282 + attention 78 + spec_normed
   21) makes the clamp bind correctly (256 → 52).

## Sanity — #138 holds

>128-tok greedy completion: 200 tokens, `finish=length`, coherent on-topic prose
(tensor vs expert parallelism), no token-0/NaN collapse.

## Throughput — old ~50 tok/s wall broken

Same 2048-in/128-out shape, `dsv4_measure.py` stats-trace method (peak windowed
2 s decode tok/s), same client as the 07-03 baseline.

| lane | peak windowed decode tok/s | vs old wall |
|---|---:|---:|
| c=1 single-stream (1 slot) | **48.1** (TPOT 21.9 ms, MTP ~2.0 tok/step) | — |
| **N=8** (8 slots live) | **75.4** | **1.57× c1** |
| **N=16** (16 slots live) | **68.2** | 1.42× c1 |
| old baseline (num_slots=4) | c1 34.6 · c8 45.8 · c16 49.9 | — |

The ~50 tok/s peak-decode wall is broken (N=8 75.4, N=16 68.2). Attribution is
**two-part**: (a) single-stream decode is already 48.1 tok/s on this binary
(34.6 → 48.1 = the landed #141-143 per-token speedups); (b) the num_slots
recovery (4 → 52) adds the batch multiplier (1.57× at N=8) on top. N=16 < N=8
and full-wall aggregate stays ~35–40 tok/s because the 2048/128 shape is
**prefill-bound** (every request pays a cold 2048-tok prefill that serializes;
MoE decode-batch scaling is sublinear) — the decode-concurrency sweet spot is
~N=8 here. A decode-heavy shape (short prompt / long output) would expose more
of the 52-slot headroom.

## Environment

- DeepSeek-V4-Flash-FP8 (294 GB FP8, 46 shards), TP=4/EP=4, allreduce MoE +
  native DeepGEMM experts, MTP `mtp_draft_tokens=2 topk=1`. Binary built from
  `8e77a9d9` (`cuda,nccl`), snapshot at `/host/arle-kvbudget-snap/arle`.
- 8×H20 (97,871 MiB), CUDA 12.9, driver 535.161.08. GPUs **1,2,4,5** (physical),
  remapped to logical 0-3 via `CUDA_VISIBLE_DEVICES`. **Topology caveat**: the 4
  picked GPUs may span NVLink domains → allreduce latency a touch higher;
  irrelevant for this slot-count/throughput-scaling verify (slot-count-bound,
  not collective-bound), but a later per-token-TPOT A/B should pin the same 4.

## Problems

- guidellm unavailable (pod pip network-blocked) — `/v1/stats` + stdlib client.
- MTP accept counters read 0 (client's `SERVE_LOG` points at the 07-03 log path,
  not `/host/kvbudget-serve.log`) — cosmetic; peak tok/s uses `/v1/stats`.
- Cross-binary vs the 07-03 baseline confounds a pure num_slots isolation — hence
  the same-binary c=1 anchor (48.1) is the honest batch-scaling reference.

## Δ vs baseline

- num_slots **4 → 52** (13×), budget `per_slot` **43 → 385 MB** (now correct),
  boot OOM → engine-ready. Peak decode **~50 wall → 75.4 (N=8)**. First DSv4
  serve that runs >4 concurrent slots.
