# Agent-OPD multi-rank rollout merge

**Status: deferred — trigger below.** 2026-08-07.

## Current state (shipped)

`9da8ff777` fixed the cp>1 deadlock (see
[errors/2026-08-07](../experience/errors/2026-08-07-agent-opd-cp2-rollout-divergence-deadlock.md)):
cp rank 0 owns the whole lane (serve, rollouts, filtering, saves) and streams
every update's batch to follower ranks via `MeshUpdateChannel`
(`crates/cli/src/train_cli.rs`) — write-then-rename JSON files under the
coordinator-minted `ARLE_TRAIN_MESH_DIR` (`crates/cli/src/train_multiproc.rs`).
Followers load only the autograd student + optimizer and mirror the update
calls, so the writeback's cp collectives see identical call sequences by
construction. dp>1 in this lane bails fast.

Cost accepted: follower GPUs idle through the rollout phase (rollout wall
85–600 s/sample vs tens of seconds per writeback), measured 70.9 GB leader /
28.9 GB follower residency during rollouts.

## Complete design (this plan)

Standard disaggregated rollout/train shape (verl/OpenRLHF), specialized to the
single-box cp group. Correctness invariant is unchanged: **rank 0 stays the
only scheduler/filter/orderer; every rank executes the identical update
sequence.** Only rollout production becomes multi-rank.

1. **Every rank runs a serve + harness during the rollout phase.** Follower
   ranks reuse the leader's existing load path (rollout engine +
   `--share-frozen-base` student + cc harness); the per-rank serve port offset
   already exists (`serve_port + world_rank`, train_cli.rs).
2. **Task sharding.** Rank 0 assigns whole groups (not samples) round-robin to
   ranks at round start — group granularity keeps GRESO/zero-variance
   accounting per-task and needs no cross-rank reward merge.
3. **Reverse channel: followers publish finished groups.** Same file protocol
   as `MeshUpdateChannel`, direction reversed: `grp-r<rank>-<seq>.json`
   carrying the serialized `CcGroup` records + rewards. Rank 0 consumes them
   into its existing per-group loop (filter → cap → publish update batch →
   collective), interleaved with its own local groups.
4. **Weight sync per rank.** After each policy update, every rank re-merges the
   LoRA into its own engine (`sync_lora_from_store`) before rolling the next
   assigned group — same `policy_version`/staleness tagging as today; a
   follower's in-flight group is simply tagged with the behavior version it
   launched under.
5. **Update stream unchanged.** The existing `MeshUpdateChannel` publish/recv
   is untouched; followers time-slice between "roll my assigned groups" and
   "join the update collective when a batch file appears". The natural
   implementation is the leader's existing boot-ahead/staleness machinery
   generalized per rank, not a new scheduler.

Expected gain: rollout throughput ~N× on an N-rank group (rollout dominates
round wall), follower idle eliminated.

New surface (why this is deferred, not free): per-rank engine lifecycle around
each writeback (quiesce + KV-pool release on every rank, not just rank 0),
reverse-channel backpressure, and GRESO state fed from remote groups.

## Prioritization rationale

Build-ahead is justified only when both hold: the final requirement is known,
and the interim structure creates path dependence. Here neither holds.

- **The final requirement is not yet specified.** Multi-node or multi-rank
  rollout scale has no scheduled workload; when it arrives, its numbers (rank
  count, message volume, fault-tolerance semantics) decide the transport. A
  design built now would target an unmeasured workload and likely be rebuilt.
- **Path dependence is near zero.** The one-time structural cost was paid in
  `9da8ff777`: the transport is confined to `MeshUpdateChannel` with two call
  sites (leader publish in `run_update`, follower recv loop). Replacing files
  with an NCCL host broadcast or a distributed queue is a localized swap; the
  training loop does not change. The same seam is where the reverse group
  channel (step 3) plugs in.

The rule: pay for the narrow seam now; defer the implementation until the
workload that selects it exists. This also rejected the alternative of a
general shared object store (Ray-style): the need is ordered broadcast of
small batches on one host — single producer, lockstep consumers — and a store
adds a service process, a failure domain, and a dependency that this access
pattern never uses.

## Trigger

Build this when **cp>1 is the production training config** — i.e. when 256K-seq
OPD training is scheduled (`max_update_seq` beyond one card). Until then the
23K-capped lane trains single-GPU and the merge is speculative throughput work.
Decision input: the follower idle fraction from the cp=2 validation report
(fulltrain5).
