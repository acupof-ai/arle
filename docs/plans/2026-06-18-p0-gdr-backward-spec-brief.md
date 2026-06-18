# P0 brief — extract upstream GDR backward so Claude can write the line-level spec

**Owner**: codex (pane 2). **Reviewer/integrator**: Claude (writes the final
line-level spec from your extraction). Per
[`2026-06-18-gated-delta-kernel-and-org-gap.md`](../research/2026-06-18-gated-delta-kernel-and-org-gap.md).

**This is the 闭门造车 fix — so the iron rule: NOTHING is hand-derived.** Every
stage/formula in your output must trace to an upstream `file:line`. No
hand-rolled math, no "should be", no inventing. If upstream is unclear, quote it
and flag it — do not fill the gap yourself.

## Constraints

- **Read-only. No code edits. No H20 / no pod / no bench / no training.**
- Do **not** touch the 4 WIP files (they are what we will delete).
- **Source is ALREADY CLONED** (Claude did it) — read in place, do not re-clone,
  do not vendor: `/tmp/fla` (FLA) and `/tmp/flashqla` (FlashQLA).

## Task

1. **Read the FLA backward** (primary — portable; FlashQLA bwd is Hopper-TMA,
   reference only), line by line:
   - `/tmp/fla/fla/ops/gated_delta_rule/chunk.py` — the
     `torch.autograd.Function` backward and the bwd helpers it calls.
   - `/tmp/fla/fla/ops/common/chunk_delta_h.py` (`chunk_gated_delta_rule_bwd_dhu`
     / dh), `/tmp/fla/fla/ops/common/chunk_o.py` (chunk_o bwd → dq/dk/dv/dg),
     `/tmp/fla/fla/ops/common/chunk_scaled_dot_kkt.py`,
     `/tmp/fla/fla/ops/gated_delta_rule/wy_fast.py` (the WY / A-inverse bwd).
   - Reference only (do not port; Hopper-specific):
     `/tmp/flashqla/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py`,
     `kkt_solve.py`.
2. **Map to our existing forward AOT** (already in-repo, forward-only) —
   `crates/cuda-kernels/tools/tilelang/gated_delta_rule.py`, 7 stages that
   ALREADY SAVE the backward's inputs: `gdr_chunk_prepare` (q/k/v/g/beta),
   `gdr_chunk_cumsum` (g_cumsum), `gdr_chunk_a` + `gdr_chunk_solve` (a_inv = the
   WY inverse `A`), `gdr_chunk_recompute` (w, u), `gdr_chunk_state`
   (**chunk_state = per-chunk-boundary h `[num_chunks,hv,K,V]`** + final_state +
   v_new), `gdr_chunk_o` (output). For EACH FLA bwd stage, name which of these
   saved tensors it consumes and which NEW AOT stage we must add.
3. **Map our autograd wiring**: `crates/autograd/src/backend_cuda.rs`
   `cuda_linear_attention_scan_backward` (FFI signature, tensors it
   passes/expects) and `crates/autograd/src/ops/linear_attention.rs` CPU forward
   + the (WIP-deleted) `state_history` — the exactness oracle for the
   gradient-check gate.

## Output — `docs/research/2026-06-18-gdr-backward-upstream-extract.md`

- **Upstream backward stage decomposition** (chunk=64): each stage = kernel
  name + the exact formula + which forward intermediates it consumes + how
  chunk-boundary state is checkpointed-and-recomputed (the stable replacement
  for the hand-roll's `s_decay/exp_g` reverse-division). Cite upstream
  `file:line` per stage.
- **Delta vs our forward AOT**: which new AOT stages we must add to
  `gated_delta_rule.py` to get the backward, reusing the licensed forward.
- **Delta vs the hand-roll**: exactly where the reverse-division kernel diverges
  from the upstream chunked backward (the correctness-relevant points).
- **Open questions** for Claude (e.g. autograd→cuda-kernels FFI direction; any
  upstream step we can't map cleanly).
