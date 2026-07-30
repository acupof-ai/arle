# N-D parallel OPD training — design (DP · PP · CP · TP · EP)

> Status: **all five axes have a landed, CPU-verified core (2026-07-29); the
> multi-rank NCCL data-planes are pending-remote.** "Core" = the correctness-
> load-bearing math/config (adjoints, coordinate derivation, shard tiling), gated
> by local unit tests. "Pending-remote" = the wire transport + model-level parity,
> which need a pod (≥2 GPU + NCCL) and are not locally verifiable.
> Scope: >3 files + architectural → approach-first per the agent contract.

## What landed (2026-07-29), per axis

Train reads all parallelism coordinates from the one device mesh
(`infer_topo::MultiAxisConfig` / `RankCoord`) — the same mesh serving reads — not
private duplicate configs.

| Axis | Core landed + CPU-gated | Pending-remote (pod NCCL) |
|---|---|---|
| **Mesh** | `train_mesh()` → `MultiAxisConfig`+`RankCoord`; `CpContext`/`Qwen35TensorParallelConfig`/`DpContext`/`PpContext` are derived views, one source of truth | — |
| **CP** | option B live + converged; ring option-A is a **live tape op** (`cp_causal_sdpa`, `BackwardOp::RingAttention`) — world==1 taped grad matches `causal_sdpa_recompute`, multi-block merge+backward matches full-softmax reference | **ladder measured (2026-07-30): option B fits 256K, no OOM** — ring NOT needed for memory. The real 256K blocker is a post-backward CP-collective hang (§8), not the KV wall. Default stays option B. |
| **TP** | attention-TP live; **MoE-TP** built (column/row-parallel experts+shared, `maybe_tp_all_reduce`) | MoE finite-diff on ≥2 GPU |
| **EP** | **live tape op** — `ep_dispatch_op`/`ep_combine_op` (`BackwardOp::EpDispatch`/`EpCombine`), backward = the transpose; dropped token gets zero grad; gated through the real tape | NCCL all-to-all transport; capacity + router aux loss; qwen35 routing hook |
| **DP** | **wired end-to-end** — `DpContext` threaded into `masked_writeback_step`; global count all-reduce for `inv_n`; grad-reduce gate `(cp‖dp)`; `--dp-size` launcher; world==1 byte-identical | multi-rank correctness (≥2 GPU); combined CP×DP (`ncclCommSplit` subgroups) |
| **PP** | `PpContext` layer-partition (`pipeline_parallel.rs`); 1F1B documented as wrong-fit for single-pass writeback | cross-stage activation send/recv; layer-loop split |

Local gate (all pass): `cargo test -p train -p autograd --no-default-features
--features no-cuda` (193 + 50 lib tests) + clippy + Mac CUDA typecheck. The
model-level parity gate `crates/train/examples/nd_parallel_parity.rs`
(`cuda,nccl`) and every wire transport above are **pending-remote**.

Nothing here is marked "shipped end-to-end": each axis's math/config is verified,
its NCCL data-plane and default-flip await the pod. Option B stays CP's default
until the ring passes pod parity (no half-states; unverified path never default).


## 0. The one correction that reshapes everything

The device mesh **already exists** and already carries all five axes — do NOT
build a new `DeviceMesh`. Converge the train side onto it.

`crates/infer-topo/src/topology.rs`:
- `struct MultiAxisConfig` (L12): `tp_size` L13, `pp_size` L14, `ep_size` L15,
  `attn_dp_size` L16, `attn_cp_size` L17, `moe_dp_size` L18. `world_size()=tp*pp`
  (L242); `validate()` (L201) enforces `tp % (attn_dp*attn_cp) == 0` (L219) and
  `tp % (ep*moe_dp) == 0` (L228).
- `struct RankCoord` (L275): per-rank `tp_rank`/`pp_rank`/`attn_tp_rank`/
  `attn_dp_rank`/`attn_cp_rank`/`moe_tp_rank`/`moe_ep_rank`/`moe_dp_rank`
  (L277-284). CP rank math at L303.
- Group builders (rank-list `Vec<Vec<usize>>`, pure math, no NCCL types):
  `build_tp_groups` L325, `build_pp_groups` L334, `build_attn_cp_groups` L342,
  `build_attn_tp_groups` L363, `build_attn_dp_groups` L382, `build_moe_dp_groups`
  L426, `build_moe_ep_groups` L448, `build_moe_tp_groups` L468.

This is Megatron's actual shape, and it is already correct: **attention** shards
on `attn_{dp,cp,tp}`; **MoE FFN** shards on `moe_{dp,ep,tp}` — two sub-meshes over
the same `world = tp*pp` cards. EP is not a physical card axis; it is the MoE
sub-mesh's expert split. It is pure coordinate math (no backend types), so `train`
may depend on it without breaching backend isolation.

## 1. Ground truth — what each axis actually reaches today

| Axis | Train-side state (file:line) | Gap to "supported" |
|---|---|---|
| **CP** | option B implemented + converged onto the mesh: `cp.is_enabled()` branch qwen35.rs L1674 all-gathers full KV → `causal_sdpa_recompute_with_q_start(q,k_full,v_full,q_start)` L1717. Local shard-parity unit-tested; N=2 loss parity PASSES on pod (rel_err 4.19e-5) | KV **not sharded** — every rank gathers full KV (O(full_seq), ~1 GB/layer bf16 @256K). No "full scores": the fused forward is a flash-2 kernel (`nonpaged_prefill_attention.cu`), no `[seq,seq]` transient. Ladder (2026-07-30) proved option B fits 256K, so the ring is a memory optimization that isn't currently needed — not a correctness gap. |
| **TP** | attention-TP proven `a2_qwen35_tp_lora_fd.rs` L181/L191; `maybe_tp_all_reduce` L313 | **MoE MLP rejects TP** ("requires single-rank TP" L1256/L1282/L1319). MoE-TP unbuilt. |
| **EP** | **train side has none.** DeepEP dispatch/combine exist only in *serving* (`infer-cuda/moe.rs` `dsv4_moe_forward_deepep` L3781); train MoE uses grouped-linear on token rows (qwen35.rs L1355-1401), no all-to-all | Bring differentiable all-to-all into train MoE + its backward. **Real work, not wiring.** |
| **DP** | CP's post-backward weight all-reduce (`all_reduce_cp_grads`, opd.rs L3238) is already DP-semantics | Batch-shard dataloader on `attn_dp_size>1`. Near-free once mesh drives it. |
| **PP** | none | 1F1B over layers. Worst fit for single-pass OPD writeback (no throughput loop to amortize the bubble). Last. |

Reusable autograd primitives that already exist (differentiable, with adjoints):
`all_gather_seq` (collective.rs L94), `reduce_scatter_sum` (L170),
`all_reduce_sum` (L14); `causal_sdpa_with_q_start` (attention.rs L240) + chunked
backward `causal_sdpa_recompute_backward_device_chunked` (L552);
`CudaBackend::new_with_nccl` (backend_cuda.rs L208); NCCL peer
`send`/`recv`/`group_start`/`group_end` (collective.rs L470/491/512/516,
**0 production callers** — ring attention is the first).

## 2. Convergence (delete-style — the structural cost we pay once)

Train side today has a parallel, simpler parallelism config (`CpContext` in
opd.rs L3024; `Qwen35TensorParallelConfig` in a2). **Delete that duplication:**
train reads its coordinates from `RankCoord`/`MultiAxisConfig`, same as serving.
`CpContext` becomes a thin view `(attn_cp_rank, attn_cp_size)` derived from the
coord, not a second source of truth. One mesh, two consumers (serve + train).

Hard invariant carried from the CP-wedge bug: every per-rank list fed to a
positional collective — including `ncclCommSplit` color/key and the group-builder
outputs — must be **deterministically ordered (Vec, never HashMap)**, or ranks
pair mismatched shapes and NCCL wedges silently. The group builders already return
`Vec`; the guard is: never route a mesh coordinate through a HashMap.

## 3. Landing order (binding-constraint driven — mesh once, impls incremental)

Seam converges once; axes land only as they become the binding constraint. "Five
axes supported" = **mesh-pluggable**, not five impls written at once.

- **P0 — CP ring attention.** The only 256K activation wall. Highest risk.
- **P1 — MoE-TP + DP.** TP unblocks the rejected MoE path (L1256); DP is near-free
  batch sharding on the same mesh.
- **P2 — EP (train all-to-all) + PP.** Both are real builds, not wiring; neither
  is a 256K wall for a 27 GB-weight LoRA run, so they follow.

Per-axis "done =": N=1 degenerates bit-identical (identity collective) **and** an
N≥2 parity within the correct-inference envelope (not byte-identity — MoE is
nondeterministic) **and** a pod c-sweep showing throughput scaling.

## 4. P0 in detail — ring attention (pinned)

**New** `crates/autograd/src/ops/ring_attention.rs`: `cp_causal_sdpa(q,k,v,coord)`
on LOCAL shards `[1,heads,seq/N,head_dim]`.

**Current (option B, to be replaced), qwen35.rs L1717:** all-gather full KV, one
rank ends up with O(full_seq) KV + O(q_chunk·full_seq) scores → does not shard
attention → OOM at 256K.

**Target (option A):** ring the KV block-by-block; each step attends the local q
against one remote KV block via `causal_sdpa_with_q_start` (L240) with the block's
absolute `q_start`; merge partial outputs with **online softmax** (running
max/denominator). Per-rank attention is O(seq/N · seq/N). Transport =
`send`/`recv` inside `group_start/end` (collective.rs L470/491/512/516). Backward
rings grad_k/grad_v partials back to block owners via `reduce_scatter_sum` (L170).

**The U2 risk (why P0 is highest-risk):** the online-softmax merge needs per-row
LSE (max + denom), which no existing op returns — the chunked backward (L552)
recomputes a full softmax per q-chunk, it does not expose block stats. So option A
needs one new `sdpa_block_with_stats` forward + a **hand-written merge and its
backward adjoint**. A wrong merge silently corrupts every gradient — this is the
piece that must be parity-gated hardest.

RoPE hazard: q/k are RoPE'd at absolute positions (qwen35.rs L1671-1672) *before*
attention; each ring block must attend with its block's absolute `q_start`, or
attention is silently wrong. The parity test must vary `q_start`.

**Three more P0 correctness/perf items:**
- **checkpoint × collective lockstep (was plan U3).** The layer forward runs under
  activation checkpointing, so the ring `send`/`recv` also fire during the backward
  *recompute*. The recompute must replay the identical ring schedule (same block
  order, same step count) on every rank, or the group desyncs and NCCL wedges — the
  same silent-hang class as the CP-wedge bug. The ring loop must be deterministic
  and take no data-dependent branch.
- **`seq % N != 0` → last-rank-remainder.** Pad the short shard; padding rows still
  enter every ring step (or the collective desyncs) but are masked to zero
  attention contribution and zero grad.
- **comm/compute overlap.** A naive ring is communication-bound and won't scale
  near-linearly. Overlap step k's KV send/recv with step k-1's block attention
  (double-buffer the KV). Ship correctness-first (blocking ring, parity-gated),
  then add overlap as a measured perf step — never claim scaling before the c-sweep.

### Per-card memory ledger at 256K (to be measured, not asserted)

The go/no-go for "CP alone fits 256K" is this table — currently **incomplete**
because the linear-attn CP peak is unmeasured (hazard §6.3). Fill it on a pod
seq-ladder before any 256K claim.

| Term | Class | Scales with |
|---|---|---|
| FP8 weights (27 GB) + LoRA adapter grad/AdamW | replicated | fixed |
| embedding · RMSNorm · MLP intermediate · residual · CE hidden | sharded | O(seq/N) |
| checkpoint boundary (one hidden × 64 layers) | sharded | O(seq/N) |
| full-attn transients | **option B: unsharded** → ring: sharded | O(seq/N) after ring |
| **linear-attn CP recurrence activation** | **UNMEASURED** | O(full_seq)? |

The last row is the open risk: if it's O(full_seq) and large, ring full-attn alone
does not fit 256K and Stage-2 (linear-attn boundary gradient) becomes mandatory.
No ungrounded extrapolation — measure the row before deciding.

## 5. Parity gates (new `crates/train/examples/nd_parallel_parity.rs`)

- **Tier 1 (hard):** every axis at size=1 → loss **and** every trainable grad
  bit-identical (raw f32) to the single-card `masked_writeback_ce_step`, seq=2048.
  Proves zero code-path divergence.
- **Tier 2:** CP N=2 (and later CP×TP, CP×EP) → `rel_err` within the
  correct-inference envelope, not byte-identity.
- **Pod:** seq-ladder to the new ceiling + c-sweep (multiproc timing change ⇒ TP=N
  c8/c16 required, not just an N=2 loss check).

## 6. Hazards carried forward (xp)

1. **A rank that HANGS without exiting wedges the group silently.** The launcher
   (`train_multiproc.rs:92`) already tears the group down on a rank that
   crash-*exits* — a 100 ms `try_wait` poll that kills survivors on the first exit.
   The residual gap is the rank that deadlocks *inside* a collective without
   exiting: `try_wait` never fires and there is no NCCL-level timeout / heartbeat /
   `ncclCommAbort`, so a 256K ladder OOM that hangs (rather than exits) is a silent
   wait. Add a per-step deadline + `ncclCommAbort` for production N-D training.
2. **Mesh construction must not reintroduce HashMap-order collective deadlock**
   (§2). First assert when wiring `ncclCommSplit`.
3. **linear-attn may be the real 256K wall, not full-attn.** Qwen3.6 is hybrid;
   most layers are gated-delta linear-attn. Under CP they currently all-gather the
   hidden and run the **full** sequence recurrence → their activation is O(full_seq),
   unsharded. U1 measured only the full-attn segment. **The CP-mode linear-attn
   activation peak at 256K is unmeasured** — it could be the next wall after ring
   attention, or not. Measure before claiming CP-alone fits 256K.

## 7. Deferred workstreams (real builds, tracked, post-P0)

Named so "five axes supported" doesn't silently omit them; none blocks P0.

- **Failure recovery / checkpoint-resume** under an N-D mesh: store + restore
  optimizer state and mesh topology so a long 256K run survives a killed rank.
  Pairs with the watchdog (§6.1).
- **Rollout nondeterminism → rank-0 broadcast** (was plan U6): real (non-synthetic)
  training must broadcast the trajectory + response_mask from rank 0 so every CP
  rank shards the identical sequence. Synthetic-writeback doesn't hit this.
- **EP correctness beyond wiring:** capacity factor, token-drop/overflow policy,
  router aux load-balance loss, and the differentiable all-to-all backward.
- **Multi-axis loss normalization:** `inv_n_override` (opd.rs L3073) divides by
  global targets; with DP×CP both sharding targets, the global count must all-reduce
  across *both* axes, not CP alone. Verify the invariant per axis combination.

## 8. Status and next binding constraint

**Landed (2026-07-29):** the mesh convergence (decision a) plus a CPU-verified
core for all five axes — see the per-axis table at the top. Every axis now has
its correctness-load-bearing piece in-tree and unit-gated: mesh coordinate
derivation, CP ring flash-2 merge+backward, MoE-TP, EP dispatch/combine adjoint,
DP global-mean, PP layer partition.

**Pending-remote (the gating work now):** each axis's NCCL data-plane and
model-level parity — ring device kernels + `ring_send_recv_kv`, EP all-to-all
transport + capacity/aux-loss, DP launcher + count reduce, PP activation
send/recv, MoE-TP finite-diff. All need a pod (≥2 GPU + NCCL); none is locally
verifiable, so none is a default flip yet. Option B stays CP's default until the
ring passes pod parity.

**Next binding constraint (MEASURED 2026-07-30):** the pod seq-ladder ran — cp=4,
27B FP8, 4×H20. **Option B fits 256K: every rung 65536→262144 completed forward
AND backward with no OOM** (262144 peak 96.4/97.9 GB, `[ckpt-gate]` auto-engaging
checkpointing). Peak is checkpoint-recompute-bounded, not O(full_seq) — 131072→
196608 peak *dropped* (81→79 GB). So the CP ring / EP / linear-attn sharding are
NOT required for 256K on memory grounds; the §6.3 linear-attn peak did not bind.
See `docs/experience/wins/2026-07-30-cp-ladder-option-b-fits-256k.md`.

The ladder surfaced the ACTUAL next wall, which the memory argument was blind to:
262144 completed both memory-heavy phases then **hung post-backward in the CP
collective** (3 of 4 ranks flushed `phase=backward`, all parked in NCCL
busy-wait). A seq-scale-specific collective desync, not a memory wall. Ruled out:
ckpt-gate (engage byte-identical across ranks), empty-shard `sum·0` (no empty
shard at 65536 local), generic checkpoint-recompute-under-CP (seq=8192 + forced
checkpoint completes lockstep). Surviving suspect: the >65535 chunked-SDPA branch
(`attention.rs:171`) — 262144/cp=4 is the only rung whose local seq (65536)
crosses it. This desync — not the ring — is what blocks a completing 256K step.

