# Smooth In-Process Actor-Learner RL Loop — Implementation Plan

> Status: Active — 2026-07-07 · Driver: ckl · plan-only (read-only architecture
> pass). No code written yet. Anchors verified against HEAD.
> **Prerequisite (gating):** the unified/co-resident agentic-OPD mode's *effect*
> is NOT yet validated — the only capability measurement
> ([12-round](../experience/wins/2026-07-03-agent-opd-27b-loop-stability-12rounds.md))
> was on a **ceilinged** held-out set (base 23/24), so +1/24 is within noise,
> single seed. Same disease as the STaR 6→7→7 plateau: the eval substrate is too
> easy to measure a lift. The difficulty-calibrated pool (c3 gen +
> `filter_inband.py`) is the missing prerequisite for BOTH effect-validation and
> the default flip. **Sequence: validate effect on the calibrated substrate
> first → then build 1-5.** Performance IS validated (37.3×, `--share-frozen-base`
> default-on).

**Scope:** decompose the 5 workstreams turning the batched STaR/OPD round loop
into a continuous actor-learner loop.
**Verify gate legend:** `MAC` = `cargo check -p <crate> --features cuda,no-cuda`
typecheckable on Mac (struct/enum/signature/CPU-logic only). `POD` = needs the
H20 (real CUDA context / NCCL / kernel / VRAM behavior); cannot be observed on Mac.

---

## 0. The barrier being removed

The current loop is a hard sequential barrier per round:

- `crates/cli/src/train_cli.rs:2637` `for round in 0..args.rounds { … }` — one
  round = rollout all tasks → collect accepted → writeback all → sync LoRA →
  eval → save, strictly serial.
- Rollout + accept: `crates/train/src/agent_opd.rs:375` `run_agentic_opd_round`;
  accepted trajectories buffer into a local `Vec` at `:399`, pushed `:567`,
  drained by the `train_on_accepted` callback `:594-599`.
- That callback is the writeback closure `train_cli.rs:2661-2701`
  (`masked_writeback_ce_step_dispatch`), which **first frees the KV pool +
  inference scratch** (`:2668`,`:2681`) — writeback and rollout cannot co-reside
  in VRAM today.
- Weight handoff back to the actor: `train_cli.rs:2721` `sync_lora_from_store` →
  `infer_student.rs:315` → `serve_engine.rs:342` `remerge_student_lora` →
  `executor.rs:5352` → `qwen35.rs:4101`.

WS1-3 break the "free-KV → writeback → remerge → refill-KV" swing into an
overlapped pipeline; WS4 removes the VRAM/stream fragility forcing the swing;
WS5 lifts the single-GPU restriction on the handoff.

---

## WS1 — Streaming replay buffer + continuous learner

**Goal:** accepted trajectories flow into a bounded, depth-limited buffer as each
rollout completes; a learner consumes continuously; hot-swap every K updates.

### Steps
1. `crates/train/src/replay_buffer.rs` (new): `ReplayBuffer { deque:
   VecDeque<Trajectory>, cap, max_staleness, produced_epoch }`; `Trajectory =
   (prompt, resp, mask, rollout_lora_epoch)`. Push evicts oldest at `cap`;
   `pop_batch` drops `current_epoch - rollout_lora_epoch > max_staleness`. Mirror
   the tuple at `agent_opd.rs:399`. `MAC`.
2. Producer seam: `run_agentic_opd_round` accept path `agent_opd.rs:567` calls a
   `FnMut(Trajectory)` sink instead of `.push`; keep the Vec sink as default so
   the batched path stays byte-identical. `MAC`.
3. Learner consumer: `run_agent_opd_streaming` (peer of `run_agent_opd_impl`) —
   rollout on the engine-owning thread, writeback on main thread pulling
   `pop_batch()`, reusing `masked_writeback_ce_step_dispatch`. `MAC` wiring / `POD` run.
4. K-update hot-swap: replace per-round `sync_lora_from_store` (`train_cli.rs:2721`)
   with a counter — sync + bump `produced_epoch` every K. `MAC`.
5. CLI: `--buffer-cap`, `--max-staleness`, `--sync-every-k`. `MAC`.

**DAG:** spine. WS2 depends on it; WS3/WS4 make it physically concurrent; WS5 independent.
**Effort/risk:** L. Cross-thread engine `Mutex` contention; smoothness payoff gated on WS4 (VRAM).
**Verify:** MAC — buffer eviction+staleness unit test. POD — A/B streaming vs round, same seed: lever = held-out pass-rate parity; needle = trajectories-in == accepts − staleness-drops.

## WS2 — One-step-off pipelining

**Goal:** overlap rollout(N+1) with update(N); rollout serves under LoRA_{N-1}
while learner computes LoRA_N (verl one-step-off, staleness=1, rollout-owned logprobs for IS).

### Steps
1. Epoch tag already carried by WS1's `rollout_lora_epoch`. `MAC`.
2. Non-blocking sync: learner does NOT join on the `run_on_engine` remerge closure
   (`serve_engine.rs:343`) — fire-and-forget, epoch bump on completion. `MAC`/`POD` races.
3. IS hook: plumb rollout-time logprobs (`SlotToken.logprob`, `executor.rs:4845`)
   into the `Trajectory`; keep masked-CE loss for now. `MAC`.
4. Remove the KV-free/refill swing (`train_cli.rs:2668-2685`,`2642`) — **hard dep
   on WS4**; gate behind `--pipeline`. `MAC` guard.

**DAG:** depends on WS1 + WS4. Without WS4, "overlap" = time-share (no wall-clock win).
**Effort/risk:** M plumbing, L after WS4. Cross-context, staleness correctness.
**Verify:** MAC epoch/staleness invariant. POD A/B pipeline on/off: lever = pass-rate parity + wall-clock/round drop; needle = every `rollout_lora_epoch ∈ {cur, cur-1}`. Pod-only.

## WS3 — Additive-at-compute LoRA hot-swap

**Goal:** replace D2H-of-A/B + `W ← pristine + scale·BA` merge with additive
`+scale·B(Ax)` in the projection GEMM epilogue; swap B,A pointers atomically.
Kills the per-swap W-mutation and the FP8→BF16 2× promotion.

### Replaces
`qwen35.rs:4101` `remerge_student_lora` → `:4231` `merge_lora_proj` (recompute W
from `lora_base_dev` `:1649`); promote `:4148` (permanent 2× VRAM).

### Steps
1. Resident A/B: `lora_a_dev, lora_b_dev, lora_scale` next to `lora_base_dev`
   (`qwen35.rs:1649`); base stays FP8 pristine → no BF16 promotion. `MAC`.
2. Additive forward: attention/MLP proj GEMM sites — epilogue `y += scale·B·(A·x)`
   (two rank-r GEMMs), gated per-matrix `Option<LoraDelta>`; `None` = byte-identical. `MAC`/`POD` kernel.
3. `apply_student_lora_inplace(update)`: H2D or zero-copy-borrow A/B, swap the
   `Option<LoraDelta>` under the control seam. Sub-ms, no W recompute. Opt-in
   `--additive-lora`. `MAC`.
4. Keep `invalidate_prefix_cache` on swap (`serve_engine.rs:345`).
5. Decode-graph: additive epilogue vs B=1 graph capture (`executor.rs:4838`) —
   scope graph-off (OPD loop already graph-off, `train_cli.rs:2409-2427`). `POD`.

**DAG:** independent CUDA track. WS1/WS2 benefit (cheaper swap → higher K) but don't require it.
**Effort/risk:** XL. Hot serving path; numerical parity; decode-graph. Highest-risk.
**Verify:** MAC plumbing + `None`-path. POD — numerical parity (merge vs additive logits, max abs diff < tol = needle); decode tok/s non-regression + additive overhead < merge cost. Pod-only.

## WS4 — Co-resident CUDA context/stream robustness

**Goal:** remove the hand-tuned fences + VRAM time-share so rollout + writeback
physically co-reside (enables WS2 overlap).

### Fragility (verified)
- Engine parks streams with event-tracking DISABLED → context-wide sync
  deadlocks → stream-scoped fence (`train_cli.rs:2489-2511`, measured deadlock).
- VRAM: `mem_fraction_static 0.2` + no decode graph (`train_cli.rs:2409-2427`);
  per-step KV free (`:2668-2685`); `EngineOffloadMode` time-share
  (`opd.rs:90-155`); `All` corrupts W4A8 Marlin across 3 contexts (`opd.rs:100-105`).

### Steps
1. Enable event-tracking on the engine's parked streams (infer-cuda ctx setup) —
   root cause of the deadlock. `POD`.
2. Single owned stream registry (rollout/writeback/sync, each event-tracked);
   replace ad-hoc `stream_synchronize` (`train_cli.rs:2505`). `MAC` shape / `POD`.
3. VRAM budget: replace `0.2` magic + per-step KV free with an explicit budget so
   rollout-KV + writeback-activations (~55 GB no-graph) co-reside. `MAC` / `POD`.
4. Retire `EngineOffloadMode` time-share for the co-resident path; keep the 16 GB
   fallback. `MAC`.

**DAG:** enabler for WS2 overlap; removes WS1's swing. Independent of WS3/WS5. **Highest leverage.**
**Effort/risk:** L-XL. Cross-context deadlock — every change can hang the pod.
Bisect one variable at a time (§0 confounder rule: don't change event-tracking + budget + offload together).
**Verify:** MAC structs. POD — needle: context-wide `cuCtxSynchronize` after handoff does NOT deadlock; lever: round completes with KV+writeback co-resident, pass-rate unchanged. Land event-tracking → budget → offload as separate A/Bs. Pod-only.

## WS5 — Multi-GPU remerge across TP

**Goal:** `remerge_student_lora` (and WS3's additive swap) apply on every TP rank.
Today hard-bails (`qwen35.rs:4103` `is_single`; `executor.rs:5356`
`ensure_not_collective` def `:4812`).

### Steps
1. `RelayEnvelope::RemergeLora { seq, payload }` (`multiproc_relay.rs:439`),
   broadcast at a tick boundary like `CancelRequest` (`:456-463`) so every rank
   applies at the identical step. `payload` = shared adapter ref, not BF16 over TCP. `MAC`.
2. `shard_lora_for_rank(update, tp_config)` — slice A/B by the projection shard
   axis, matching the base shard math (`loader.rs:2958`,`3110`) per proj type
   (q/k/v/o, gate/up/down). **The real work.** `MAC` (CPU-testable).
3. Drop guards behind the relay path: single-GPU keeps direct; multi-rank routes
   through the envelope + per-rank shard. `MAC`.
4. Worker apply: consume `RemergeLora` in `execution.rs` (peer to admission
   handling ~`:211-235`), call local remerge/additive on the shard. `MAC` / `POD` NCCL.

**DAG:** fully independent (multi-GPU pod). Composes with WS3. Lowest priority for single-node.
**Effort/risk:** L. Multiproc lockstep desync = silent divergence or hang
([ref](../experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md)).
Sharding math is the correctness core.
**Verify:** MAC — `shard_lora_for_rank` reconstruction unit tests (full A/B from
shards == original, per proj type) — the bulk of correctness, fully MAC. POD —
TP≥2: gather each rank's resident proj == single-GPU merge (needle); TP2
pass-rate == TP1 (lever). Lockstep/no-hang pod-only.

---

## Recommended landing sequence

1. **WS1** buffer + producer/consumer split (MAC-heavy, L). Default path
   byte-identical. Low risk.
2. **WS4 event-tracking ONLY** (POD, isolated A/B) — the single highest-leverage
   unblock; kills the deadlock forcing every fence + the free/refill swing. Then
   VRAM budget, then offload retirement — one variable each.
3. **WS2 plumbing** (steps 1-3 MAC), gate step 4 on WS4; flip `--pipeline` and A/B.
4. **WS3 additive-LoRA** (XL, POD parity gate) — parallel CUDA track; MAC plumbing
   early, pays off after WS4. Numerical-parity A/B before trusting.
5. **WS5 last** — multi-node only. Do the sharding-math unit tests on Mac now.

**Critical path to first smooth loop:** WS1 → WS4(event-tracking) → WS4(budget) →
WS2(step4). WS3/WS5 are parallel tracks not gating the single-GPU smooth loop.

## Cannot verify without the H20 pod
- WS4 entirely (deadlock removal, VRAM co-residency, offload retirement).
- WS2 physical overlap + staleness-under-load; wall-clock win.
- WS3 additive-vs-merge numerical parity + decode tok/s.
- WS5 NCCL lockstep / no-hang + cross-rank resident-weight identity.
- WS1 runtime behavior (engine `Mutex` contention, real accept counts).

**Fully MAC-verifiable now:** all struct/enum/signature; `ReplayBuffer`
eviction+staleness (WS1); epoch/staleness invariants (WS2); additive epilogue
plumbing + `None`-path (WS3); `shard_lora_for_rank` reconstruction (WS5).
