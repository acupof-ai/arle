# LoRA consolidation cadence + base-weight snapshot — train, 2026-08-25

> Status: pending-remote

## Goal

SOPD Phase-3 host portion (#96): the consolidation cadence controller (K-step
/ adapter-norm trigger) and the pre-merge base-weight snapshot for rollback.
The weight-requant primitive itself is cuda-kernels (TurboQuant/Marlin/Q4_K)
— out of scope for this tranche.

## What landed

- `ConsolidationCadence` (lora.rs): fires every K steps or when the adapter
  norm crosses a threshold. Pure host logic, 3 unit tests.
- `BaseWeightSnapshot` (lora.rs): captures the base weight's host data before
  merge; `restore` rewinds it (dirty=Host, device mirror dropped) so a failed
  post-merge gate can revert. Roundtrip unit test on the cpu lane.
- The {adapter, AdamW} snapshot already exists (`EmaTrainSnapshot`,
  ema_self_teacher.rs:188) — the consolidation snapshot composes with it.

## Parameters

```bash
# pending-remote: OPD loop with consolidation cadence on H20
# - needle + same-config-twice floor across the requant boundary
# - self-consistency (correct-inference, not byte-identity)
# - failed gate reverts to pre-upgrade base + adapter
```

- Baseline: `27ab564e2` (no consolidation)
- Treatment: this commit (cadence + snapshot)
- Trials: pending-remote

## Environment

- Host / GPU: H20 pod (pending-remote)

## Rule

A cadence controller is pure logic (step counter + threshold); the snapshot
is the only stateful piece, and it composes with the existing EMA/adapter
snapshot rather than duplicating it.
