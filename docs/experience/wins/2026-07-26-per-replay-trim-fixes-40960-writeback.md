# Per-replay pool trim is the 40960 writeback fix — CUDA, 2026-07-26

> Status: Shipped (per-replay trim was the passing arm; made unconditional +
> flags deleted). Default-path re-verify at 40960 pending-remote.

## Goal

Single-GPU H20-96GB agent-OPD masked writeback: complete the backward at
seq=40960 (was OOM at concat_axis2), and collapse three trim flags to the one
that works, on by default.

## Four-arm sweep (seq=40960, offload ON, GPU 1, sequential)

| arm | flags | result | peak / free-at-failure |
|-----|-------|--------|------------------------|
| A baseline | none | **OOM** | peak 97099, backward layer 47, concat_axis2 |
| B | `--trim-after-checkpoint-replay true` | **PASS** | peak 96651, **857 MiB free**, loss 8.686 |
| C | `--trim-before-backward true` | OOM | peak 95595, layer 51 |
| D | `--writeback-window 512` | OOM | peak 97099, layer 47 |
| E | all trims removed | OOM | peak 95019, **3961 MiB free**, layer 59 |

## The finding: per-replay trim wins, low-peak-still-OOM is fragmentation

Only arm B completes. The decisive comparison is B vs E: **B passes with 857 MiB
free; E (zero trims) OOMs with 3961 MiB free** — more free memory yet failure.
That is external fragmentation: the failing `concat_row_chunks` alloc
(`matmul_bt_lora_backward_tiled`, a contiguous `[40960,5120] f32`, ~2.5 GiB
transient) can't be placed even though the bytes exist.

Per-replay `trim_to(0)` after each checkpoint replay's re-offload is what keeps
the arena packable: it returns the just-freed hidden's pages so the next
replay's large contiguous grad has a clean block to land in. A one-shot
pre-backward trim (C) or no trim (E) both let the arena fragment past the
concat point. This **inverts** an earlier wrong hypothesis ("trims cause the
fragmentation, delete them") — deletion (E) reproduced the two-trim OOM at the
same 3961 MiB free, disproving it.

## What changed

Keep the one trim that works, on by default; delete the two that don't and all
three flags (single flow, no A/B knobs — CLAUDE.md no-half-states):

- `crates/autograd/src/tape.rs`: `trim_after_checkpoint_replay` is now a `Tape`
  method gated on `self.offload_checkpoints` (not a flag). All three backward
  call sites unchanged — arm B passed with all three firing, so all stay.
- `crates/train/src/opd.rs`: deleted `trim_before_backward` (arm C, insufficient)
  and `trim_after_writeback` (dormant, post-cleanup) helpers + the ledger's
  `post_forward_trim`/`post_trim` columns.
- Deleted CLI flags `--trim-before-backward`, `--trim-after-writeback`,
  `--trim-after-checkpoint-replay` and their runtime_flags plumbing.

Gate rationale: arm B ran offload ON; without offload nothing is re-offloaded
per replay, so there are no freed pages to reclaim — the trim is a no-op there
and skipping it avoids the syscall.

## Rule

Low-peak-still-OOM is a fragmentation tell, not a capacity one: compare
free-at-failure across arms, never peak-used alone (B 857 free PASS vs E 3961
free OOM). And a single A/B can't prove a mechanism — the "trims fragment,
delete them" hypothesis felt right and was refuted by actually deleting them.
Extends [[feedback_vram_attribution_needs_ab_not_arithmetic]].

## Pending-remote

Re-verify the default path (no flags) at 40960 now that per-replay trim is
unconditional-under-offload: expect PASS matching arm B (peak ~96.6 GiB, loss
~8.69). Ticket: this session's devops lane.
