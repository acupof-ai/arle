# N-D parallel OPD training — design (DP · PP · CP · TP · EP)

> Status: **CP/TP/DP have a landed, CPU-verified core; the multi-rank NCCL
> data-planes are pending-remote.** "Core" = the correctness-load-bearing
> math/config (adjoints, coordinate derivation, shard tiling, the ring
> collectives, zigzag `SeqShard`, the linear-attn CP wrapper), gated by local
> unit tests. "Pending-remote" = the wire transport + model-level parity, which
> need a pod (≥2 GPU + NCCL) and are not locally verifiable.
> Scope: >3 files + architectural → approach-first per the agent contract.
>
> CP was calibrated against Megatron-Core (`NVIDIA/Megatron-LM`, `megatron/core`)
> on 2026-07-30: a source review found our first cut diverged on two algorithmic
> points — contiguous (not zigzag load-balanced) attention sharding, and a planned
> serial carry-ring (not all-to-all-to-head) for linear attention. Both are now
> corrected and landed (§4): zigzag `SeqShard` (§4.2) and the linear-attn CP
> chunked zigzag ring carry (§4.3), all CPU-gated.
> A second source review confirmed our fused-qkv + packed-conv1d
> `linear_attention_core` already *is* Megatron's gated-delta-net CP contract — no
> core-interface refactor needed. See §4.

## What landed, per axis

Train reads all parallelism coordinates from the one device mesh
(`infer_topo::MultiAxisConfig` / `RankCoord`) — the same mesh serving reads — not
private duplicate configs.

| Axis | Core landed + CPU-gated | Pending-remote (pod NCCL) |
|---|---|---|
| **Mesh** | `train_mesh()` → `MultiAxisConfig`+`RankCoord`; `CpContext`/`TpContext`/`DpContext` are derived views, one source of truth | — |
| **CP** | ring attention is the CP full-attn path (`cp_causal_sdpa`, `BackwardOp::RingAttention`) — device fwd-merge + finalize + bwd kernels (`ring_block_attention.cu`), wired in `qwen35.rs`, replacing the option-B all-gather (deleted). Sequence sharded **zigzag** load-balanced (`SeqShard`, front+back chunk pair); the ring masks causally by per-row absolute position so the two chunks attend the right prefix. Linear-attn CP is a **chunked zigzag ring carry** (`linear_attention_core_cp`, `cp_chunked_forward`, `CpChunkGeometry`): the recurrent state is carried chunk-by-chunk over a ring, exact, no cross-rank dependency in the steady state. world==1 ring taped grad matches `causal_sdpa_recompute`; multi-block merge+bwd matches the full-seq reference; head-split linear-attn reconstructs the full-seq recurrence on CPU | multi-rank ring transport (`ring_send_recv_kv`); >65535 local-seq parity; 256K liveness; zigzag load-balance c-sweep. Device per-row-position ring kernel (zigzag on GPU) is pending-remote — the device path errors loudly on `positions.is_some()`, never silently mis-attends |
| **TP** | attention-TP and **MoE-TP** ops built (column/row-parallel experts+shared); model-agnostic core is `train::tensor_parallel` (`TpContext` + `divide` + `maybe_all_reduce`, mirror of `CpContext`/`DpContext`); qwen35 shard dims are a `Qwen35TpDims` impl. Production construct uses `TpContext::single()` — no model runs TP-sharded yet | MoE finite-diff on ≥2 GPU |
| **DP** | **wired end-to-end** — `DpContext` threaded into `masked_writeback_step`; global count all-reduce for `inv_n`; grad-reduce gate `(cp‖dp)`; `--dp-size` launcher; world==1 byte-identical | multi-rank correctness (≥2 GPU); combined CP×DP (`ncclCommSplit` subgroups) |

PP was deleted (`pipeline_parallel.rs`, `PpContext` — 1F1B is a wrong fit for single-pass writeback); it is not a live axis.

Local gate (all pass): `cargo test -p train -p autograd --no-default-features
--features no-cuda` + clippy + Mac CUDA typecheck. The model-level parity gate
`crates/train/examples/nd_parallel_parity.rs` (`cuda,nccl`) and every wire
transport above are **pending-remote**.

Nothing here is marked "shipped end-to-end": each axis's math/config is verified,
its NCCL data-plane and default-flip await the pod (no half-states; an unverified
path never becomes the default).


## 0. The one correction that reshapes everything

The device mesh **already exists** and already carries all five axes — do NOT
build a new `DeviceMesh`. Converge the train side onto it.

`crates/infer-topo/src/topology.rs`:
- `struct MultiAxisConfig` (L11): `tp_size`, `pp_size`, `attn_dp_size`, `attn_cp_size`. `world_size()=tp*pp`; `validate()` enforces `tp % (attn_dp*attn_cp) == 0`.
- `struct RankCoord` (L116): per-rank `attn_tp_rank`/`attn_dp_rank`/`attn_cp_rank`.
- Group builders (rank-list `Vec<Vec<usize>>`, pure math, no NCCL types): `build_tp_groups` L145, `build_attn_cp_groups` L154, `build_attn_tp_groups` L175.

The EP/moe_dp axis was removed from the mesh in the 2026-08 sweep (expert
placement follows the TP worker set); the MoE sub-mesh language in this doc is
historical.

This is Megatron's actual shape, and it is already correct: **attention** shards
on `attn_{dp,cp,tp}`; **MoE FFN** shards on `moe_{dp,ep,tp}` — two sub-meshes over
the same `world = tp*pp` cards. EP is not a physical card axis; it is the MoE
sub-mesh's expert split. It is pure coordinate math (no backend types), so `train`
may depend on it without breaching backend isolation.

## 1. Ground truth — what each axis actually reaches today

| Axis | Train-side state (file:line) | Gap to "supported" |
|---|---|---|
| **CP** | ring attention on LOCAL shards: `cp.is_enabled()` branch qwen35.rs → `cp_causal_sdpa(q,k,v,cp.size,cp.rank,Some(positions))` on `[b,heads,seq/N,hd]`, never materializing full KV. Sequence sharded **zigzag** load-balanced (`SeqShard.shard`, front+back chunk pair); `positions` are threaded from opd.rs (the shard's absolute rows, the same slice that builds RoPE cos/sin), so the ring masks by true absolute position with one source of truth — not re-derived from `(seq_len, cp)`. Linear-attn CP is a **chunked zigzag ring carry** (`linear_attention_core_cp`, `forward_linear_attention` `cp` param): the recurrent state is carried chunk-by-chunk over a ring, exact | multi-rank ring transport; device per-row-position ring kernel (zigzag on GPU); >65535 local-seq parity; load-balance c-sweep |
| **TP** | attention-TP proven `a2_qwen35_tp_lora_fd.rs` L181/L191; `tensor_parallel::maybe_all_reduce` | **MoE MLP rejects TP** ("requires single-rank TP" L1256/L1282/L1319). MoE-TP unbuilt. |
| **EP** | **train side has none.** DeepEP dispatch/combine exist only in *serving* (`infer-cuda/moe.rs` `dsv4_moe_forward_deepep` L3781); train MoE uses grouped-linear on token rows (qwen35.rs L1355-1401), no all-to-all | Bring differentiable all-to-all into train MoE + its backward. **Real work, not wiring.** |
| **DP** | CP's post-backward weight all-reduce (`all_reduce_cp_grads`, opd.rs L3238) is already DP-semantics | Batch-shard dataloader on `attn_dp_size>1`. Near-free once mesh drives it. |
| **PP** | none | 1F1B over layers. Worst fit for single-pass OPD writeback (no throughput loop to amortize the bubble). Last. |

Reusable autograd primitives that already exist (differentiable, with adjoints):
`all_gather_seq` (collective.rs L94), `all_reduce_sum` (L14); the ring
collectives live in `ring_attention.rs` / `ring_block_attention.cu`; NCCL peer
`send`/`recv`/`group_start`/`group_end` (collective.rs) feeding `ring_send_recv_kv`.

## 2. Convergence (delete-style — the structural cost we pay once)

Train side today has a parallel, simpler parallelism config (`CpContext` in
opd.rs L3024; `TpContext` in a2). **Delete that duplication:**
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

- **P0 — CP ring + zigzag + linear-attn ring carry.** Not a memory wall (the
  ladder disproved that) but the correctness + load-balance core, and the fix for
  the >65535 local-seq boundary. Ring, zigzag `SeqShard`, and the linear-attn CP
  chunked ring carry are all built + CPU-gated; multi-rank pod parity pending
  throughout.
- **P1 — MoE-TP + DP.** TP unblocks the rejected MoE path (L1256); DP is near-free
  batch sharding on the same mesh.
- **P2 — PP.** A real build, not wiring; not a 256K wall for a
  27 GB-weight LoRA run, so it follows.

Per-axis "done =": N=1 degenerates bit-identical (identity collective) **and** an
N≥2 parity within the correct-inference envelope (not byte-identity — MoE is
nondeterministic) **and** a pod c-sweep showing throughput scaling.

## 4. CP in detail — calibrated against Megatron-Core

CP shards the sequence over N ranks. Two attention families need different
treatment, and Megatron-Core (`megatron/core`) settled both; our first cut
diverged on both and is now corrected.

### 4.1 Full attention — ring, on the local shard

`cp_causal_sdpa(q,k,v,cp_size,cp_rank)` (`ring_attention.rs`) attends this rank's
local q `[b,heads,seq/N,hd]` and rings the KV block-by-block: step j feeds the
block owned by rank `(cp_rank−j) mod N` into a one-block flash-2 device kernel
(`ring_block_attention.cu`: fwd-merge → finalize → bwd) that merges partial
outputs with an on-device online softmax (running max/denom). Peak is O(seq/N·hd),
never the O(full_seq) gathered KV — this is why it fits where the old all-gather
(deleted) OOM'd at local seq > 65535. The kernel saves per-row LSE so the backward
reconstructs `P = exp(S − lse)` directly (flash-2 adjoint); grad_k/grad_v ring back
to each block's owner in reverse. GQA repeat happens per-block inside the kernel,
so k/v ship at kv-head width. The host `ring_forward_tile`/`ring_backward_tile`
stay as the CPU reference and the correctness gate (world==1 taped grad ==
`causal_sdpa_recompute`; multi-block merge+bwd == full-softmax).

**Absolute causal masking:** each block attends with its absolute `q_abs`/`k_abs`,
because q/k are RoPE'd at absolute positions before attention. The ring loop is
deterministic (fixed order, fixed step count) so it replays identically under
activation-checkpoint recompute — a data-dependent branch would desync the group
(the CP-wedge hang class).

### 4.2 Zigzag load balancing (Megatron `get_batch_on_this_cp_rank`)

Contiguous shards are *correct but imbalanced*: under a causal mask, the rank
owning the tail attends ~N× the keys the head rank does, and the ring stalls on the
slowest rank every step. Megatron-Core splits the sequence into `2N` chunks and
gives rank r the pair `{r, 2N−1−r}` — one from the front, one from the back — so
every rank carries the same causal work. `SeqShard` (`context_parallel.rs`) is now
a chunk-list: `CpContext::shard` returns rank r's two chunks, `local_rows()` is the
gather index into the global sequence, and `local_of(pos)` is its inverse
(position→local row) that `opd.rs` rebases loss targets through. Because the two
chunks are non-contiguous, the ring can't assume `q_abs = cp_rank*s`: it masks by
per-row absolute position (`cp_causal_sdpa` takes `positions`, `ring_forward_tile`/
`ring_backward_tile` mask `k_pos[c] > q_pos[r]`). The requirement is
`seq % (2N) == 0` (pad up). `DpContext::batch_shard` stays contiguous — batch items
are independent, no causal imbalance to balance.

### 4.3 Linear attention — all-to-all to the head axis (Megatron gated-delta-net)

> **Superseded (2026-08).** The `all_to_all` op and the all-to-all-to-head
> wrapper were deleted in the dead-code sweep. The live linear-attn CP path is
> `linear_attention_core_cp` as a chunked zigzag ring carry (`cp_chunked_forward`,
> `CpChunkGeometry` with `recv_from`/`send_to` peers,
> `BackwardOp::LinearAttentionCpChunked`). The section below is the original
> design rationale, kept for the Megatron calibration record.

The gated-delta recurrence is Markovian along the sequence: a contiguous shard
would need rank r's state seeded by rank r−1's, which serializes the ranks and
kills the parallelism. The wrong fix (a serial "carry ring") was planned and
rejected. Megatron-Core's answer, landed here as `linear_attention_core_cp`
(`linear_attention.rs`, wired via `forward_linear_attention`'s `cp` param):
**all-to-all the sequence axis into the head axis.** Before the conv+recurrence,
`all_to_all` turns each rank's `[b, seq/N, hidden]` into `[b, seq, hidden/N]` —
every rank now holds the *full* sequence for 1/N of the value-heads. The recurrence
never crosses value-heads (state, conv taps, `a_log[h]`, `dt_bias[h]`, `beta[h]`,
per-head rmsnorm are all head-local), so each rank runs the complete conv1d +
recurrence locally with no cross-rank dependency and no approximation; a second
all-to-all restores the sequence shard afterward. Memory is unchanged (hidden/N per
rank). `all_to_all` is self-adjoint with the two axes swapped, so the backward is
the same op; `cp_size==1` is `linear_attention_core` verbatim (byte-identical).

**Calibrated to Megatron's gated-delta-net (`megatron/core/ssm/gated_delta_net`),
not its Mamba path.** GDN — our exact arch — carries the fused `qkv` through
*one* unsectioned all-to-all (a cheap head-dim `index_select` permute beforehand
makes one fused shuffle equal a per-section one) and keeps the conv1d weight
*packed*, merely section-slicing `[q|k|v]` per rank. Mamba splits into five
per-projection all-to-alls only because its B/C groups are CP-replicated (a
heterogeneous shard axis we don't have) — and even it re-fuses before the kernel.
So our fused-qkv + packed-conv1d `linear_attention_core` *is* the GDN contract:
CP is additive (pre-permute + one fused a2a in + per-rank conv slice + one fused
a2a out), touching only the CP wrapper — the core's 8-arg interface and its
callers are untouched.

### Per-card memory at 256K (measured 2026-07-30, option-B ladder)

The pod seq-ladder ran cp=4, 27B FP8, 4×H20: every rung 65536→262144 completed
forward AND backward, peak 96.4/97.9 GB, checkpoint-recompute-bounded (131072→
196608 peak *dropped* 81→79 GB — not O(full_seq)). So even the pre-ring all-gather
fit 256K on memory; the linear-attn peak did not bind. Ring + zigzag + all-to-all
are correctness/load-balance improvements and the fix for the >65535 slice_bwd OOM
on the crossing rung — not a memory prerequisite. `docs/experience/wins/
2026-07-30-cp-ladder-option-b-fits-256k.md`.

## 5. Parity gates (new `crates/train/examples/nd_parallel_parity.rs`)

- **Tier 1 (hard):** every axis at size=1 → loss **and** every trainable grad
  bit-identical (raw f32) to the single-card `masked_writeback_ce_step`, seq=2048.
  Proves zero code-path divergence.
- **Tier 2:** CP N=2 (and later CP×TP, CP×EP) → `rel_err` within the
  correct-inference envelope, not byte-identity. Zigzag chunking, multi-block ring
  merge+bwd, and head-split linear-attn each parity against the full-seq reference
  on CPU.
- **Pod:** the >65535 local-seq gate (cp=2 global 131072 → local 65536): a full
  optimizer step completes AND CP loss-sum matches single-card within REL_TOL. Then
  the seq-ladder to 256K + a load-balance c-sweep (zigzag vs contiguous wall-clock;
  multiproc timing change ⇒ TP=N c8/c16, not just an N=2 loss check).

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
3. **linear-attn CP: settled + landed — all-to-all-to-head (§4.3), not a wall.**
   Qwen3.6 is hybrid — most layers are gated-delta linear-attn, whose recurrence is
   Markovian along the sequence. A contiguous shard would serialize the ranks; the
   fix is Megatron-Core's all-to-all into the head axis (full seq × 1/N heads per
   rank, no cross-rank dependency), validated against Megatron's gated-delta-net so
   our fused-qkv core needs no interface change. Measured: the CP linear-attn
   activation peak did *not* bind at 256K (the ladder ran to completion). Landed as
   `linear_attention_core_cp` + the `all_to_all`/`cat` ops, CPU-gated by a head-split
   reconstruction test — kept here so the hybrid subtlety isn't re-litigated.

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

**Landed + CPU-gated:** mesh convergence plus the correctness core for all five
axes (per-axis table up top). CP's ring flash-2 fwd-merge/finalize/bwd
(`ring_block_attention.cu`) is wired in `qwen35.rs` with the old all-gather
deleted; the sequence shards **zigzag** load-balanced (`SeqShard`, §4.2) with the
ring masking by per-row absolute position; linear-attn CP is a **chunked zigzag
ring carry** (`linear_attention_core_cp`, §4.3) over the ring transport. Plus
MoE-TP, DP global-mean. (EP dispatch/combine adjoint and PP layer partition
were removed with the EP axis in the 2026-08 sweep.)

**Pending-remote (the gating work now):** the multi-rank NCCL data-planes and
model-level parity — ring `ring_send_recv_kv` + all-to-all transport, the device
per-row-position ring kernel (zigzag on GPU; the device path errors loudly on
`positions.is_some()` rather than silently mis-attending), EP capacity/aux-loss, DP
launcher + count reduce, PP activation send/recv, MoE-TP finite-diff. All need a pod
(≥2 GPU + NCCL); none is a default flip until it passes pod parity (no half-states).

**The one binding pod gate:** local seq > 65535. The 2026-07-30 ladder proved
memory is not the wall (cp=4 27B FP8 ran every rung to 262144, peak 96.4 GB,
checkpoint-bounded — `docs/experience/wins/2026-07-30-cp-ladder-option-b-fits-256k.md`).
It also surfaced the real wall: 262144/cp=4 is the only rung whose local seq
(65536) crosses the 65535 CUDA grid-dim / chunked-SDPA boundary
(`attention.rs:171`), where the old path desynced. The ring kernel launches the
large dimension as `grid.x` (up to 2³¹−1) specifically to clear that boundary; the
pod gate is the >65535 CP parity run that proves a full optimizer step completes
and the CP loss-sum matches single-card. Until it runs, that rung is `cite
pending-remote`.

