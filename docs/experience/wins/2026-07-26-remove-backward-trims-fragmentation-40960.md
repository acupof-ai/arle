# Removing backward-path pool trims clears the 40960 writeback wall — CUDA, 2026-07-26

> Status: pending-remote (default-path verify at 40960 on H20)

## Goal

Single-GPU H20-96GB agent-OPD masked writeback: complete the backward at
seq=40960 (was OOM at concat_axis2), by fixing allocator fragmentation rather
than adding memory-reclaim calls.

## The finding: low peak still OOMs → fragmentation, and the trims caused it

Three measured runs on the same card (GPU 1, seq=40960, offload ON):

| config | backward peak used | free at failure/finish | result |
|--------|--------------------|-----------------------|--------|
| per-replay trim only (arm B) | 96651 MiB | ~857 MiB | **PASS** |
| trim-before-backward only (arm C) | 95595 MiB | — | OOM, layer 51 |
| both trims (default after 1st refactor) | 93547 MiB | **3961 MiB** | OOM, layer 59 |

**arm B allocated successfully with 857 MiB free; the two-trim path failed with
3961 MiB free.** More free memory + failure vs less free + success is the
signature of *external fragmentation*: total is sufficient, but no single
contiguous block large enough for the failing `alloc_zeros`.

The failing alloc is `concat_row_chunks` in `matmul_bt_lora_backward_tiled`
(`layers.N.self_attn.q_proj`) — a contiguous `[40960, 5120] f32 = 838 MiB`
grad_a (concat transient peak ~2.5 GiB). At OOM, `used=93547 free=3961`
(93547+2500 ≈ 96 GiB < 97508 ceiling) — the bytes fit; a contiguous block
doesn't.

**Root cause: `trim_to(0)` fights the pool's own design.** `DeviceContext::new`
sets `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = u64::MAX` (cuda-kernels tensor.rs) —
"never release to OS, keep coalesced blocks for reuse." Each backward-path
`trim_to(0)` forces the opposite: it hands the pool's large coalesced free
blocks back to the OS mid-backward, and the subsequent reallocs (interleaved
with growing live grads) fragment. arm B trims less than the two-trim path, so
it fragments less and completes at a *higher* used.

The CUDA async mempool (`cuMemAllocAsync`) is already the global paged
allocator: page-level sub-allocation, coalescing, contiguous VA over reused
pages. With threshold=MAX it is correctly tuned to minimize fragmentation.
Trimming during active growth is the bug, not the cure.

## What changed

Deleted every backward-path trim; rely on the threshold=MAX pool to reuse pages
(single flow, no flags — CLAUDE.md no-half-states):

- `crates/autograd/src/tape.rs`: removed all three `trim_after_checkpoint_replay`
  calls and the method (checkpoint replay no longer trims).
- `crates/train/src/opd.rs`: removed `trim_before_backward` and both call sites;
  dropped the ledger's `post_forward_trim` column.
- Deleted CLI flags `--trim-before-backward`, `--trim-after-checkpoint-replay`,
  `--trim-after-writeback` and their runtime_flags plumbing (train + autograd).

Kept the legitimate phase-boundary trims (weight offload, pre-KV-budget,
release_kv_pool) — those trim *after* a bulk free with no interleaved growth, so
they return whole blocks without fragmenting.

## Rule

`cuMemGetInfo` counts pool-reserved-but-freed pages as "used", which reads like
a leak and tempts a `trim_to(0)`. With RELEASE_THRESHOLD=MAX those pages are the
allocator's coalesced reuse reserve — trimming them mid-allocation-growth
*causes* the fragmentation OOM it looks like it should prevent. Low-peak OOM is a
fragmentation tell; attribute it with a free-at-failure comparison, never with a
used-MiB peak alone. Extends
[[feedback_vram_attribution_needs_ab_not_arithmetic]].

## Pending-remote

Verify: rerun the seq=40960 synthetic writeback with the built binary (no trim
flags exist) and confirm it completes (rc=0, `loss=` line). Ticket: this
session's devops lane.
