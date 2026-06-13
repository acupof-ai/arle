# DSv4 DP-attention — design (request-level data-parallel attention)

Status: **design** (2026-06-13). Tracks GitHub #89. Licensed by the measured
serial-cap baseline
([wins](../experience/wins/2026-06-13-dsv4-concurrency-baseline-serial-capped.md)),
not yet implemented. Effort estimate: **3-4 weeks**, scheduler is the crux.

## Why (the measured problem)

The lockstep single-scheduler serve processes concurrent decode requests
serially: aggregate throughput is flat at ~53 tok/s for c=1..16, wall-clock =
C × single-stream (baseline above). The ceiling is the **admission model**,
not the GPU.

Two distinct levers sit on top of this:

- **Batched-TP decode (#60)** — one TP=8 group co-batches B requests in one
  forward. Amortizes weight reads, but attention still does the per-token
  TP collectives (Q all-gather + O all-reduce) whose *latency* does not
  amortize with B. Helps, but the collective floor caps the scaling.
- **DP-attention (this doc, #89)** — each rank runs attention **data-parallel**:
  it computes the *full* attention (all 64 heads) for its **own** subset of
  requests, with **zero per-token attention collectives**. The MoE stays
  EP-sharded; the only cross-rank exchange is the all-to-all at the attn→MoE
  boundary (which DSv4 already runs for EP). At concurrency, the per-token
  attention-collective floor vanishes — this is the structural throughput
  lever. (SGLang's `enable_dp_attention`.)

DP-attn is **not** replicated-attn (killed,
[here](../experience/errors/2026-06-13-dsv4-replicated-attn-bandwidth-kill.md)):
replicated had every rank redundantly compute *all* requests (8× waste). DP-attn
has each rank compute *different* requests (1× work, partitioned). It is a
**throughput (c>1) lever, not a B=1 latency win** — likely neutral-or-loss at
c=1 (the boundary all-to-all is pure overhead with one request); gate on the
c-sweep wall-clock, never c=1.

## What already exists (re-usable substrate)

- **KV is per-rank-full.** DSv4 MLA compresses KV to a latent that is *not*
  head-sharded; each rank already holds full KV for its slots. DP-attn needs
  exactly this — zero work.
- **MoE all-to-all exists.** DeepEP (`deepep_ll`) dispatch/combine is
  implemented + debugged (`ARLE_DSV4_MOE_TRANSPORT=deepep_ll`). The boundary
  exchange DP-attn needs is the same primitive.
- **Full-width attention compute** is re-derivable from the replicated-attn
  work (full-width Q projection + grouped o-projection) — but re-derive the
  o-projection **vectorized** (uint4 FP8, the decode-lane trick), not the
  scalar version that killed replicated-attn.

## The blocker: lockstep `serve_multiproc`

Today every rank receives the **identical** `TickAdmissions` and runs the
**identical** ForwardPlan, because the NCCL collective sequence must match
across ranks or it deadlocks (observed: the boot-time barrier deadlock when
rank0 stalls). DP-attn requires each rank to process a **different** request
set — directly at odds with lockstep.

The key realization that makes it tractable: **the only cross-rank collective
in a DP-attn step is the boundary all-to-all, and it is fixed-shape/symmetric**
(every rank dispatches/combines the same total token budget). So ranks can run
divergent attention locally and still stay NCCL-aligned *as long as the
boundary all-to-all is invoked in lockstep with a fixed token budget*. Break
lockstep on attention; keep lockstep on the boundary.

## Implementation phases (the multi-week plan)

**Phase A — scheduler / admission (the crux, ~40% of effort).**
- Replace "broadcast identical batch" with "broadcast a DP-partition": the
  coordinator assigns each admitted request to a DP rank (round-robin over
  active slots), and broadcasts the *partition map* (rank → request-ids), not
  one shared batch. Each rank builds its own local batch from its slice.
- Padding contract: every rank must drive the boundary all-to-all every step
  with a fixed per-step token budget `T_step` (the max local batch across
  ranks, agreed via a 1-int all-reduce-max at step start). Ranks with fewer
  local tokens pad to `T_step`; padded rows dispatch to a null expert and are
  discarded on combine. This keeps the all-to-all shape rank-identical →
  NCCL-aligned despite divergent local work.
- Determinism: the partition map + `T_step` are the only shared state; derive
  them from data already on every rank (the admission queue is broadcast) so
  no rank can compute a divergent plan.

**Phase B — forward: DP attention + boundary exchange (~30%).**
- Attention: each rank runs `mla_attention` over its local batch with
  `tp_world` treated as 1 for the attention math — full heads, **no**
  Q all-gather, **no** O all-reduce. (The collectives deleted with
  replicated-attn are exactly the ones that must NOT run here.)
- Boundary: after attention+norm, all-to-all(dispatch) the local tokens to
  their EP-expert owners (DeepEP), run the grouped expert GEMM, all-to-all
  (combine) back to the originating DP rank. Reuse the existing DeepEP LL path
  — it is already `[N]`-batched and token-owned.

**Phase C — attn↔MoE layout redistribution (~20%).**
- The token layout differs: attention is **DP** (rank owns its requests'
  tokens, contiguous), MoE is **EP** (rank owns its experts' tokens). The
  dispatch all-to-all *is* the DP→EP reshuffle; combine is EP→DP. Verify the
  index math (owned-slice prefix sums) against the inverted-naming foot-guns
  (`combine num_recv_tokens` = original tokens; `recv_channel_prefix`).

**Phase D — correctness + perf gate (~10%).**
- Needle ladder ×3 per DP rank (each rank must retrieve for *its* requests) +
  same-config-twice floor — not byte-identity (MoE non-determinism).
- c-sweep wall-clock (c=1/4/8/16): license requires aggregate throughput to
  beat the 53-flat baseline at c≥4 AND not regress c=1 TTFT beyond the
  boundary-all-to-all cost. Per the bench spec, multi-shape (short + long
  prompt) before any default flip.

## Risks / open unknowns (hypotheses, not evidence)

- **Planner determinism under divergent per-rank state** is the hard part: any
  rank computing a different `T_step` or partition → NCCL deadlock. Must be
  derived from broadcast-identical inputs only.
- **Boundary all-to-all latency** may make DP-attn lose below some
  concurrency threshold; the crossover is the licensing question.
- **Padded-token correctness**: null-expert dispatch + combine-discard must be
  proven not to perturb real tokens (enumerate every buffer the pad rows
  touch).
- Interaction with d2 spec decode: the draft/verify forwards must also run
  DP-partitioned, or spec is disabled under DP-attn for v1.

## v1 scope cut (to land something testable fast)

DP-attn for **decode-only, no spec, fixed DP=TP=8** first: prove the
serial-cap is broken on a c-sweep, then layer spec + prefill + dynamic
partitioning. Do not co-develop with batched-TP-decode (#60) — pick DP-attn as
the single throughput axis (it subsumes the attention-collective problem #60
cannot).
