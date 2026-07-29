# N-D parallel OPD training — design (DP · PP · CP · TP · EP)

> Status: **design proposal, awaiting sign-off. Nothing here is implemented.**
> Scope: >3 files + architectural → approach-first per the agent contract.

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
| **CP** | option B live + N=2 verified: `cp.is_enabled()` branch qwen35.rs L1674 all-gathers full KV → `causal_sdpa_recompute_with_q_start(q,k_full,v_full,q_start)` L1717 | attention **not sharded** — rank N-1 holds full KV + full scores → OOM at 256K. Needs ring (option A). |
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

1. **A single crashed rank hangs the whole group silently** — the exact CP-wedge
   failure mode. Production N-D training needs an NCCL watchdog / timeout, or every
   256K ladder OOM is a 20-min silent hang. Not yet present.
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

## 8. Decision needed

Sign off on: (a) converge train onto `MultiAxisConfig`/`RankCoord`, delete
`CpContext`/`Qwen35TensorParallelConfig` duplication; (b) build **P0 ring
attention only** now, other four axes stay identity in the mesh; (c) landing order
P0→P1→P2. On sign-off I implement P0; I do not touch TP/EP/DP/PP impls.
