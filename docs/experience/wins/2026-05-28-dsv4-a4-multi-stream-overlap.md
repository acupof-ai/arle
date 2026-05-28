# DSv4 A4 multi-stream TP comm/compute overlap — Phase 1 (FlashMLA prefill AllGather Q)

## SLO-shape probed? — pending-remote (build in flight at 2026-05-28T17:xx UTC)

Pod build `tn -H arle ... cargo build --release --features cuda,nccl
-p infer --bin infer` is running. Once the binary lands, the probe at
`/sgl-workspace/dsv4_long_probe.sh {4,16,24} 32 1` will be re-run with
`ARLE_DSV4_FLASHMLA_TP_OVERLAP=0` (baseline) and `=1` (Phase 1).

## TL;DR (target, to be replaced with measured numbers)

V2.4 closed FlashMLA prefill correctness. The 24K probe is still a wash
because AllGather Q serializes per-layer with `arle_flashmla_csa_pack_kv`
and `arle_flashmla_csa_build_indices` on the compute stream. Phase 1
hoists AllGather Q onto `ctx.comm_stream` via a secondary TP NCCL group
and uses `CudaPipelineFence` to order it with the compute stream so
those kernels run concurrently with the in-flight collective.

Targeted gain: ≥ 5% wall-clock TTFT on the 24K probe (190.94 s →
≤ 181.4 s). KILL if any probe regresses or produces garbage tokens.

## Roofline check

Wall-clock framing per CLAUDE.md §0:

| Workload | V2.4 baseline | Phase 1 (overlap on) | Δ% | Verdict |
|---|---:|---:|---:|---|
| 4K  (4017 tok)   | 16.96 s  | pending-remote | pending | pending |
| 16K (16017 tok)  | 103.13 s | pending-remote | pending | pending |
| 24K (24016 tok)  | 190.94 s | pending-remote | pending | **target case** |

Window-percent framing (informative only): nsys may show AllGather Q
% of NVTX window collapse to near-zero on `comm_stream`. This is
expected — the kill criterion is **wall-clock**, not window %.

## What changed

Two commits:

- `7e67a1c8` — `feat(layer-comm): add TP overlap NCCL group surface for A4`
  Adds `tp_overlap_nccl` field + `with_tp_overlap_nccl` builder +
  `tp_overlap_nccl` accessor + `tp_overlap_can_all_gather_bf16` capability
  to `LayerCommunicator`. Mirrors `ep_overlap_nccl` exactly.

- `91d5c077` — `perf(dsv4): A4 multi-stream TP comm/compute overlap (FlashMLA prefill)`
  In `DeepseekV4Model::layer_communicator_from_config`, boots a secondary
  TP NCCL group on `ctx.comm_stream` via
  `dsv4_nccl_env_bootstrap_with_port_offset(2)` when
  `ARLE_DSV4_FLASHMLA_TP_OVERLAP=1`. In `finish_attention_gpu` /
  `use_flashmla` branch, hoists the AllGather Q launch to
  immediately after the `padded_send_count` setup (before
  `arle_flashmla_csa_pack_kv` is enqueued on compute). The fence
  `ctx.comm_waits_for_compute()` records the compute-stream HEAD
  post-`dsv4_prepare_qk`, pre-`kv_pack`. Then before the repack kernel
  reads `gathered`, `ctx.compute_waits_for_comm()` orders the comm
  output back into compute.

Phase 1 is per-layer overlap (overlap layer N's AllGather Q with
layer N's `kv_pack`+`build_indices`, on the same forward call). Phase 2
target is per-NCCL-chunk pipelining à la TokenWeave (arxiv 2505.11329),
deferred to a follow-up.

## Why per-layer overlap should help

Per layer at TP=8 / 16384 tokens / head_dim=192:
- `padded_send_count` = 16384 × 8 × 192 / TP = 25 MiB per rank
- AllGather Q at 8×H20 / NVLink: O(few hundred µs) per layer
- `arle_flashmla_csa_pack_kv` + `arle_flashmla_csa_build_indices`:
  hundreds of µs combined (pre-V2.4 nsys trace, not re-measured under
  V2.4)

If those two compute kernels are within 2× of AllGather Q latency,
per-layer wall-clock per layer drops to max(AllGather, kv_pack +
build_indices) instead of sum. Over 61 layers, ARLE's binding-constraint
profile (`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`)
shows the AllGather contribution is a measurable fraction of prefill
wall-clock — exact % deferred to nsys cross-check during this bench.

## Fences — sync correctness (CRITICAL)

The forward-path code path under `ARLE_DSV4_FLASHMLA_TP_OVERLAP=1`:

```
compute stream                              comm stream
  dsv4_prepare_qk_fused      (writes q_prepared)
  ─── ctx.comm_waits_for_compute() ────────→
                                              overlap.all_gather_bf16_device
                                                (reads q_prepared,
                                                 writes gathered)
  arle_flashmla_csa_pack_kv
  arle_flashmla_csa_build_indices
  arle_flashmla_fill_pad_rows (no-op @ V2.4)
  ─── ctx.compute_waits_for_comm() ←─────────
  dsv4_tp_q_repack_cuda      (reads gathered, writes packed)
  arle_flashmla_sm90_sparse_prefill_fwd
  dsv4_tp_out_slice_cuda
```

cudarc auto-event-tracking is disabled in `DeviceContext::new()`
(`tensor.rs:339-341`), so `device_ptr()` guards on `CudaSlice` are
inert. Correctness rides 100% on our explicit
`CudaPipelineFence` calls, matching the
`moe_reduce_scatter_bf16_overlap` consumer in `mlp.rs:4583-4592`.

Failure modes if a fence is wrong:
- Missing `comm_waits_for_compute`: AllGather reads stale q_prepared
  → garbage tokens (not crash).
- Missing `compute_waits_for_comm`: repack reads pre-AllGather
  `gathered` → garbage tokens.
- `comm_waits_for_compute` recorded AFTER kv_pack/build_indices: comm
  waits for them on the comm stream → overlap collapses to zero
  (perf regression to V2.4 wall-clock, not a correctness bug).

## NCCL port assignment

- Main TP NCCL group: `NcclInitMethod::EnvBootstrap` (offset 0)
- EP combine overlap group: offset +1 (existing)
- **TP overlap group (this commit): offset +2**

`dsv4_nccl_env_bootstrap_with_port_offset(2)` resolves `MASTER_ADDR` and
`MASTER_PORT+2`. Pod's k8s network must have that port open and
unbinded; if both EP combine overlap and TP overlap are enabled at the
same time, three NCCL bootstrap rendezvous happen at boot.

## Refs

- Backlog axis: `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` (A4)
- Phase 1 design: `docs/plans/2026-05-28-dsv4-a4-multi-stream-overlap.md`
- V2.4 prefill state: `docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`
- Prior art (EP combine overlap):
  - Boot: `infer/src/model/deepseek/weights.rs:230-238`
  - Consumer + fence: `infer/src/model/deepseek/mlp.rs:4583-4592`
- Pipeline fence primitives: `crates/cuda-kernels/src/tensor.rs:170-540`
- TokenWeave reference (Phase 2 target): https://arxiv.org/abs/2505.11329

## Status: pending-remote

Build kicked off on `sglang-eic-test`. This entry will be amended once
the 4K/16K/24K probes complete under both `ARLE_DSV4_FLASHMLA_TP_OVERLAP=0`
(baseline reproduction) and `=1` (Phase 1). The expected next commit
either:

1. **PASS** — fills the result table, declares default flip to `=1`,
   updates the `Why now` section in
   `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` A4 to
   point at this wins entry, and lists Phase 2 (TokenWeave) as next.
2. **KILL** — moves this entry under `docs/experience/errors/`, captures
   the regression mode (correctness vs perf vs hang), and leaves the
   env-var defaulted off.

## Rule (preliminary, to lock in on PASS)

When a TP collective has a stable producer→consumer fence pattern and
already-routed comm-stream infrastructure, **hoist the collective
launch to immediately after the producer kernel completes (event
record point), not after subsequent unrelated compute kernels**.
Putting the fence record AFTER unrelated compute kernels forces the
collective to wait for them, collapsing overlap. Mirror the EP overlap
pattern's fence placement (`mlp.rs:4583-4592`): producer kernel →
`comm_waits_for_compute()` → collective launch → unrelated compute
→ `compute_waits_for_comm()` → consumer kernel.
