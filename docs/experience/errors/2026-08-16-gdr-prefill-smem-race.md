# 2026-08-16 — GDR prefill recurrent kernel smem race, exposed by CP's NCCL interleaving

## Context

T2.b gates (replicated-KV CP prefill, `docs/plans/2026-08-16-cp-ideal-state.md`)
failed on the H20 pod: at world=4 (attn_tp=2 × attn_cp=2) the needle ladder
produced deterministic `!` garble once a prefill chunk reached ≥512 rows, while
world=2 (attn_tp=1, attn_cp=2) was clean at every length. The garble was
output-text corruption, not a crash — the model ran, the recurrent state had
silently drifted.

Evidence ladder that isolated it (each step ruled a suspect out or in):

- Full-attention CP arm disabled alone → still red; linear (GDN) CP arm disabled
  alone → green. Conviction: the GDN relay path, not KV all-gather / FA3 slices.
- Five-quantity differential probe (GEMM inputs, GEMM outputs, conv output,
  GDR output, post-step state, CP arm vs collective-free full-chunk reference):
  GEMM and conv quantities bit-identical; drift appeared only inside the GDR
  advance and grew with row count, row 0 clean.
- Path probe: the drift only appeared on the scalar
  `gated_delta_rule_prefill_recurrent_kernel` fallback. The fq chunked kernels
  have AOT instances only for geometries (16,32) and (16,48); attn_tp=2 shrinks
  the local geometry to (8,24), which falls through to the scalar kernel. This
  explains the attn_tp>1 conjunction.
- Run-to-run drift magnitude varied 0.16 ↔ 24.5 — a scheduling-dependent race,
  not a deterministic math error. Two same-weight ranks running the "reference"
  pass disagreed with each other: the reference itself was racy.

## Root cause

`crates/cuda-kernels/csrc/recurrent/gated_delta_rule.cu`, prefill recurrent
kernel token loop: after the norm sync, each thread writes its own token's
`smem_q[val_idx]` / `smem_k[val_idx]`, and all threads then read those slots
cross-thread (`smem_k[j_base+jj]`, `smem_q[...]`) in the decay/update loops —
with no barrier between the per-thread writes and the cross-thread reads. A
lagging warp read the previous token's q/k; the stale values polluted the
recurrent state and accumulated token by token (row 0 clean, drift growing with
rows).

The race was latent: under cp=1 the warp schedule is quiet enough that the
window never produced a wrong value above the needle threshold. CP interleaves
NCCL collectives with the compute, perturbing warp scheduling enough to make
the race fire frequently.

## Fix

One `__syncthreads()` after the smem_q/smem_k writes, before the cross-thread
reads (1f7948070). The decode siblings (`gdr_decode_batch.cu`, the decode path
in `gated_delta_rule.cu`) already sync after their smem writes — audited, no
sibling bug.

Verified on the pod: self-check diffs bit-zero at layer 0 across 48 layers;
needle 12/12 exact at world=4 and 4/4 at world=2; cp=1 control unchanged;
128K cold-prefill TTFT 54.14s → 30.93s (1.75×).

## Rule

Cross-thread shared-memory reads after per-thread writes require an explicit
barrier — a kernel that is correct in isolation can ship a latent race that
only fires once collective traffic perturbs the schedule, so "passes at cp=1"
is not evidence of kernel correctness under CP. When a new parallel axis
changes the kernel mix (here: attn_tp>1 selecting the scalar fallback), audit
the selected kernel's smem discipline before debugging the new axis's
collectives.
