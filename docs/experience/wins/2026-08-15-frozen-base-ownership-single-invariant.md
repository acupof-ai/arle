# 2026-08-15 — frozen-base ownership collapses to one invariant

## Context

Adversarial review (16 findings, 10 surviving 3-refuter verification) showed
the frozen-base sharing path held a half-state between two designs:
`free_retired_fp8_buffers` freed the engine's retired FP8 qweight/scales every
sync, while the promote path's keepalive comment, the `a1a3fda92` offload
guard, and the `7c4c9082f` trainer-owns-pristine skip all assumed those bytes
stay resident. Consequences: a dangling trainer FP8 alias for LoRA-targeted
shared projections (the `a1a3fda92` UAF class, one lane over), delta
accumulation across syncs (`lora_dirty`/`lora_base_dev` cleared, so re-merge
treated merged bytes as base), and a fused-sibling cache key that could
restore the other projection's window bytes.

## What Worked

Delete the free chain entirely (`8c0ac637c`: model → executor → lib → serve
engine → loaded API → sync call site) and key the pristine window cache per
real projection. The retired FP8 bytes are simultaneously the pristine source
for the idempotent re-merge and the storage behind the trainer's shared
alias; keeping them resident is what the sharing design already required, so
the fix is a deletion, not a mechanism. Ownership is now one invariant:
exported or promoted base bytes are never freed while the model lives.

Verification: regression arm (0.8B, attention-qv, offload=student, 5 steps)
finite losses 26.59 / 26.09 / 27.31 / 28.65 / 25.18. The "row fuse" load
failure first blamed here was a misdiagnosis: the card held a foreign 4x84 GB
job and the flattened error chain hid the OOM root cause (fixed by printing
`{err:#}` in the engine-build error). On a free GPU the FP8-shared all-linear
lane ran two full rubric-opd rounds clean: 23.2 GB of shared FP8 base stayed
resident through both engine offload/reload cycles, no UAF, named errors
only. The non-zero-delta merge on the FP8 lane remains unexercised because
the acceptance path needs the Flash judge (8 GPUs); the BF16 merge lane with
real deltas is covered by a 5-step all-linear run.

## Rule

When one buffer serves two owners, the free must be owned by an invariant,
not by call sites: any per-site free of shared bytes will eventually be
called from a site that does not know about the other owner. Prefer deleting
the free over guarding it.
