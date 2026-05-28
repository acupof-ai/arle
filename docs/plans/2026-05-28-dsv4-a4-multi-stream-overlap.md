---
title: DSv4 A4 multi-stream TP comm/compute overlap (FlashMLA prefill)
date: 2026-05-28
type: implementation plan
status: Phase 1 ready for codex pickup
owner: ckl
related:
  - docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md (A4 axis)
  - docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md (V2.4 prefill baseline)
  - docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md (sister axis, decode side)
  - https://arxiv.org/abs/2505.11329 (TokenWeave, target architecture for Phase 2)
---

# DSv4 A4 multi-stream TP comm/compute overlap (FlashMLA prefill)

## Why now

V2.4 closed FlashMLA prefill correctness — `nullptr` `max_logits` / `lse`
was the bug, not s_q alignment. Measured wins:

| Workload | Legacy | V2.4 | Δ% |
|---|---:|---:|---:|
| 4K  | 17.60 s   | 16.96 s | −3.6%  |
| 16K | 117.75 s  | 103.13 s | **−12.4%** |
| 24K | ~190 s    | 190.94 s | **≈ 0% (wash)** |

The 24K wash is the binding constraint. Both 24K chunks (chunk-1
16384 + chunk-2 7632) run FlashMLA at TP=8 (V2.4 gate is `tp_world>1
&& token_count>1`), and **AllGather Q + repack still serializes
per-layer** with compute. Chunk-2's smaller s_q amortizes
the collective worse, so wall-clock stays at the legacy ceiling.

Industry references (do not re-research):

- Megatron-LM TP overlap — separate NCCL stream + cuda events, ~10%
- TokenWeave (arxiv 2505.11329) — per-chunk pipelining within
  AllGather, compute starts as soon as the first NCCL chunk arrives.
  ~20–30%. **Architecture target for Phase 2.**
- NCCL Symmetric Memory + Multimem — Phase 3, optional.
- DeepEP — DSv4 MoE comm only, not TP attention, out of scope here.

## Scope

`ARLE_DSV4_FLASHMLA_TP_OVERLAP` (default off in this commit, default on
after empirical PASS on pod). Single-binary change in
`infer/src/model/deepseek/weights.rs::finish_attention_gpu` —
**prefill branch only** (sibling subagent owns the decode branch).

## State surface already in place

This is the lever that makes Phase 1 cheap: ARLE's plumbing is already
multi-stream-shaped — we only need to wire the AllGather Q dispatch
into it.

- `DeviceContext` in `crates/cuda-kernels/src/tensor.rs` already
  carries `comm_stream: Arc<CudaStream>` alongside `stream` (compute)
  and `copy_stream`. Streams are created with cudarc auto-event-tracking
  **disabled**, so explicit fence ordering is the only correctness
  story (no hidden cross-stream waits).
- `CudaPipelineFence` + `record_pipeline_fence(Compute/Comm/Copy)` +
  `wait_on_pipeline_fence(consumer, fence)` exist and are tested
  (`tensor.rs::pipeline_fence_tests`). Helper shorthands:
  `comm_waits_for_compute()` / `compute_waits_for_comm()` /
  `copy_waits_for_compute()` / `compute_waits_for_copy()`.
- Same overlap pattern is **already in production** on the EP axis for
  the MoE reduce-scatter combine path:
  - `weights.rs:230-238` builds an "overlap" EP NcclGroup on
    `ctx.comm_stream` via `dsv4_nccl_env_bootstrap_with_port_offset(1)`,
    keyed off `dsv4_combine_overlap_enabled()`.
  - `LayerCommunicator::with_ep_overlap_nccl` (and
    `moe_reduce_scatter_bf16_overlap`,
    `moe_reduce_scatter_bf16_can_overlap`) at
    `layer_communicator.rs:184-200, 446-471`.
  - Consumer at `mlp.rs:4583-4592` fences with
    `ctx.comm_waits_for_compute()` before launching the collective on
    the overlap group, then records a `CudaPipelineFence` on `Comm`.

This Phase 1 design mirrors that exact pattern for the **TP** axis,
applied to the prefill AllGather Q.

## Phase 1 design — naive per-layer multi-stream

### Stream-level data flow per layer (8×H20 TP=8)

```
compute stream (ctx.stream):
  qk_prep -> kv_pack -> build_indices -> [WAIT comm_done] -> repack
            \                                          /
             [fence comm_waits_for_compute]   ^       [fence compute_waits_for_comm]
                                              \     /
comm stream (ctx.comm_stream):                 AllGather Q on q_prepared
                                                                      ^
                                                          fully overlapped with
                                                          kv_pack + build_indices
                                                          on compute stream
```

Key observations on dependency edges:

1. **q_prepared** is produced by `dsv4_prepare_qk[_fused]` on the
   compute stream. AllGather Q reads it. → `comm_waits_for_compute`
   fence before issuing AllGather on the overlap NCCL group.
2. **kv_pack** (`arle_flashmla_csa_pack_kv`) reads `window_cache`,
   `k_prepared`, and `compressed`. None of those are touched by
   AllGather. → can run concurrently with AllGather on compute.
3. **build_indices** (`arle_flashmla_csa_build_indices` /
   `_hca_build_indices`) reads `selected` (already on device) and
   produces `indices_unified` + `topk_length`. No AllGather
   dependency. → also concurrent with AllGather.
4. **fill_pad_rows** — touches `indices_unified`/`topk_length` after
   build. Stays on compute, before repack. (No-op at V2.4 since
   `padded_s_q == token_count`.)
5. **repack** (`dsv4_tp_q_repack_cuda`) reads the AllGather output
   `gathered`. → `compute_waits_for_comm` fence before issuing the
   repack on compute.
6. FlashMLA kernel (`arle_flashmla_sm90_sparse_prefill_fwd`) consumes
   both `packed_q` (from repack) and `kv_unified`/`indices_unified` —
   all already on compute, no extra fence needed.

Expected per-layer overlap = max(AllGather latency, kv_pack + build_indices
latency) instead of sum. From V2.4 24K probe wash + Phase B-3
DeepEP combine experience, AllGather Q at `padded_send_count` ≈
`16384 * 8 * 192 = 25 MiB` per layer per rank at TP=8 has measurable
latency on NVLink — overlap is real, not symbolic.

### Concrete code edits in `weights.rs::finish_attention_gpu`

A. **Boot a TP overlap NCCL group at model construction**
   (around line 200–260, alongside the existing
   `dsv4_combine_overlap_enabled()` block):

   ```rust
   if dsv4_tp_overlap_enabled() && config.tp.world_size > 1 {
       let tp_overlap = Arc::new(NcclGroup::new_on_stream(
           config.tp.rank,
           config.tp.world_size,
           dsv4_nccl_env_bootstrap_with_port_offset(2)?, // 0 = main, 1 = EP combine, 2 = TP overlap
           ctx.comm_stream.clone(),
       )?);
       comm = comm.with_tp_overlap_nccl(tp_overlap)?;
   }
   ```

B. **Extend `LayerCommunicator`** with the symmetric surface to
   the EP overlap pattern:
   - field `tp_overlap_nccl: Option<Arc<NcclGroup>>`,
   - builder `with_tp_overlap_nccl`,
   - accessor `tp_overlap_nccl()`,
   - convenience `tp_overlap_can_all_gather_bf16(&self) -> bool`.

C. **In the prefill FlashMLA dispatch (around line 1985)** — replace
   the existing `tp_nccl.all_gather_bf16_device(...)` call with a
   guarded fast path:

   ```rust
   let overlap = self.layer_communicator.tp_overlap_nccl();
   if let Some(overlap_nccl) = overlap.as_ref() {
       // Fence: comm waits for q_prepared (produced on compute by qk_prep).
       self.ctx.comm_waits_for_compute()?;
       overlap_nccl.all_gather_bf16_device(
           &q_prepared.data, padded_send_count, &mut gathered,
       )?;
       // Compute will wait before reading `gathered` (see below).
   } else {
       // Legacy path: all on compute.
       tp_nccl.all_gather_bf16_device(
           &q_prepared.data, padded_send_count, &mut gathered,
       )?;
   }
   ```

   And before the repack kernel:

   ```rust
   if overlap.is_some() {
       self.ctx.compute_waits_for_comm()?;
   }
   // dsv4_tp_q_repack_cuda(..., self.ctx.stream.cu_stream())
   ```

D. **Reorder** the `arle_flashmla_csa_pack_kv` and
   `arle_flashmla_csa_build_indices` blocks so they're enqueued
   **between the AllGather launch and the
   `compute_waits_for_comm()` fence**. Currently they sit before the
   `if tp_world > 1` AllGather block — physically still correct (they
   don't read `gathered`), but moving them after the AllGather launch
   maximizes the overlap window without changing kernel semantics.

E. **Env-gate** with `ARLE_DSV4_FLASHMLA_TP_OVERLAP` defaulting to
   `false` for landing → flip to `true` after pod PASS.

### Sync correctness (CRITICAL)

`CudaSlice::device_ptr_mut(&self.ctx.stream)` in cudarc returns a
`SyncOnDrop` guard that records the stream on the slice's drop-tracking.
With `disable_event_tracking()` already called in `DeviceContext::new()`
(`tensor.rs:339-341`), these guards are inert no-ops — they don't issue
cross-stream waits. Correctness rides 100% on **our** explicit
`CudaPipelineFence` calls, exactly like the existing
`moe_reduce_scatter_bf16_overlap` consumer site.

Three places where the fence must be exactly right:

1. **Before AllGather launch** on overlap group:
   `ctx.comm_waits_for_compute()`. Reason: `q_prepared` was written on
   compute by `dsv4_prepare_qk_fused`, AllGather reads it on the
   `comm_stream`-bound overlap NCCL group.
2. **Before the repack kernel** on compute:
   `ctx.compute_waits_for_comm()`. Reason: repack reads `gathered`
   which AllGather wrote on comm.
3. **After repack** (no fence needed) — repack writes `packed` on
   compute, FlashMLA reads `packed` on compute, in-order with
   compute-stream semantics.

The `gathered` and `packed` `CudaSlice`s themselves are allocated
on the compute stream (`ctx.stream.alloc_traced::<bf16>`). With
event-tracking disabled, the allocator does not enforce stream
ordering; cudarc's memory pool returns memory whose only safe-use
discipline is **our** fence ordering. We honor that.

### Failure modes to watch on pod

- **Stale `q_prepared` read on comm stream**: produces a corrupted
  full-out tensor → garbage tokens, no crash. Mitigation: the fence
  in step 1 is mandatory.
- **Repack runs before AllGather completes**: produces
  garbage-or-mixed-rank Q for FlashMLA → garbage tokens. Mitigation:
  fence in step 2.
- **Stream contention with EP overlap NCCL group**: both
  `dsv4_combine_overlap_enabled()` (EP) and the new TP overlap share
  `ctx.comm_stream`. NCCL queues serialize on the same stream — this
  is fine, no race, but it caps potential overlap when both axes are
  active. Phase 2 may want separate comm streams; not in scope here.
- **NCCL port collision**: TP overlap uses
  `dsv4_nccl_env_bootstrap_with_port_offset(2)`, EP overlap uses
  offset 1, main group uses no offset. Verify the bootstrap is
  reachable on `MASTER_PORT + 2` (open port range or unbinded socket
  on the pod's k8s network).

## Phase 2 — TokenWeave per-NCCL-chunk overlap (deferred)

Goal: pipeline compute against partial AllGather arrivals (not
per-layer). Inside one layer's AllGather, start the repack-then-FlashMLA
pipeline on the FIRST rank's worth of Q as soon as it arrives.

Sketch (do not implement in this PR):

- Split AllGather Q into `N = world_size` per-source chunks, each
  posted as a separate `ncclAllGather` with `send_count =
  local_q_count`. Round-robin completion is best-effort —
  NCCL doesn't expose per-source completion fences.
- Workaround: split the **destination** into `world_size` separate
  AllGather calls, one per source rank, each in a NCCL group. Each
  individual call posts a separate event on the comm stream → fine
  granularity for compute to consume.
- Repack kernel becomes streaming: given `gathered[0..k]` rank-tiles
  filled, repack those rank tiles into `packed` (k * h_local heads in
  the global-head slot). FlashMLA waits for full `packed` before
  launch.

Estimated gain over Phase 1 at TP=8: another 10–20% wall-clock on the
24K wash case (TokenWeave reports 20–30% on a single optimization;
half on Phase 1, half on Phase 2 is a reasonable split).

Defer to a follow-up PR. Phase 1 alone is the goal of this commit.

## Phase 3 — NCCL Symmetric Memory (further deferred)

Use `ncclCommSymmetricRegister` / `ncclMemAllocSymmetric` to enable
multimem AllGather hardware path on H100/H20. Requires NCCL ≥ 2.22
and is incompatible with the current cudarc-pool allocator pattern
(buffers must come from NCCL-symmetric allocator).

Out of scope. Document only.

## License-or-kill (Phase 1)

Per `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` A4
gate, with the 24K wash case as the binding constraint:

**Wall-clock framing (ground truth, per CLAUDE.md §0):**

- **PASS**: 24K probe TTFT improves ≥ **5%** vs V2.4 baseline (190.94 s
  → ≤ 181.4 s); 16K probe does not regress > 2%; output is qualitatively
  sensible (no garbage tokens; finish_reason="length" preserved).
- **STRETCH (Phase 2 territory)**: 24K ≥ 12%; 16K ≥ 0%.
- **KILL**: 24K wall-clock regresses, or any probe produces garbage
  output. Roll back via `ARLE_DSV4_FLASHMLA_TP_OVERLAP=0`.

Window-percent framing (informative only, do not gate on this):
nsys may show AllGather % of NVTX window collapses to near-zero on
the comm stream — that's expected but is not the kill criterion.

## Test plan (pod)

```bash
# 1. Sync + build (sister subagent owns decode-branch files; we own
#    weights.rs prefill block + layer_communicator.rs + new env helper).
tn exec -H arle -- 'kubectl exec sglang-eic-test -- bash -c "cd /sgl-workspace/arle-fresh && git fetch origin main && git reset --hard origin/main && touch infer/src/model/deepseek/weights.rs && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 cargo build --release --features cuda,nccl -p infer --bin infer"'

# 2. Baseline (TP overlap off) — repro V2.4 24K wash.
ARLE_DSV4_FLASHMLA_TP_OVERLAP=0 /sgl-workspace/dsv4_long_probe.sh 16 32 1
ARLE_DSV4_FLASHMLA_TP_OVERLAP=0 /sgl-workspace/dsv4_long_probe.sh 24 32 1

# 3. Phase 1 (TP overlap on).
ARLE_DSV4_FLASHMLA_TP_OVERLAP=1 /sgl-workspace/dsv4_long_probe.sh 16 32 1
ARLE_DSV4_FLASHMLA_TP_OVERLAP=1 /sgl-workspace/dsv4_long_probe.sh 24 32 1

# 4. 4K smoke (sanity, gate fires on tp_world>1 even at 4K).
ARLE_DSV4_FLASHMLA_TP_OVERLAP=1 /sgl-workspace/dsv4_long_probe.sh 4 32 1
```

## Win-entry skeleton

`docs/experience/wins/2026-05-28-dsv4-a4-multi-stream-overlap.md`:

```
| Workload | V2.4 baseline | Phase 1 overlap | Δ% |
|---|---:|---:|---:|
| 4K  | 16.96 s  | ??? | ??? |
| 16K | 103.13 s | ??? | ??? |
| 24K | 190.94 s | ??? | ??? |
```

Roofline note: if 24K hits −5% to −12%, declare PASS. Cite
TokenWeave deferred to Phase 2.

## Refs

- V2.4 prefill state: docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md
- Backlog axis A4: docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md
- Prior art (EP combine overlap):
  infer/src/model/deepseek/weights.rs:200-260 + mlp.rs:4583-4592
- Pipeline fence primitives: crates/cuda-kernels/src/tensor.rs:170-540
