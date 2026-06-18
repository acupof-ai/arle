# P2 brief — extract Liger fused-linear distillation loss for the line-level spec

**Owner**: codex (tmux1). **Reviewer/integrator**: Claude. Per Track 1
[`2026-06-18-opd-memory-infra-gap.md`](../research/2026-06-18-opd-memory-infra-gap.md):
OPD still materializes dense `[window, vocab]` teacher/student logits; the
industry fix is a **fused final-linear + distill-loss** that never materializes
the big logits tensor. This extracts the upstream algorithm so Claude can spec
the adopt.

## Iron rule (same as P0)

**Nothing hand-derived.** Every formula/step traces to upstream `file:line`.
ARLE's autograd is from-scratch (CPU + Metal + CUDA, host-authoritative
`Vec<f32>`), **NOT** PyTorch/Triton — so we adopt the **algorithm**, not the
Triton kernels. Say where the algorithm maps to our autograd; do not pretend a
Triton drop-in.

## Constraints

- **Read-only. No code edits. No H20 / no pod / no bench. Do NOT use tmux2.**
- Source already cloned: read `/tmp/liger` in place (do not re-clone/vendor).

## Task

1. **Read the fused-linear distillation algorithm** (local, line by line):
   - `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py`
     — the base class: how it **chunks over tokens**, applies `lm_head` per
     chunk, computes the distill loss + gradient in the same pass, and
     accumulates grad to hidden + lm_head **without** ever holding the full
     `[T, vocab]` logits.
   - `/tmp/liger/src/liger_kernel/chunked_loss/jsd_loss.py` (JSD distill),
     `/tmp/liger/src/liger_kernel/ops/fused_linear_jsd.py`,
     `/tmp/liger/src/liger_kernel/ops/fused_linear_cross_entropy.py`,
     `/tmp/liger/src/liger_kernel/ops/{jsd.py,kl_div.py}`.
2. **Map to our current dense path** (the sites to replace, from Track 1):
   `crates/train/src/opd.rs:2048-2155` (windowed: hidden →
   `logits_from_hidden_window` → `kl_distill_loss_for_config`),
   `opd.rs:1446-1476` (chunked KL window),
   `crates/train/src/loss.rs:37-195` (dense softmax/log-softmax KL/JSD chunks).
   State exactly what each currently materializes and at what `[shape]`.
3. **Secondary** (note, don't deep-dive): verl async OPD's **sparse top-k
   teacher targets** (`teacher_topk_logps`/`indices`) as the complementary
   teacher-memory lever — flag whether it composes with the fused loss.

## Output — `docs/research/2026-06-18-fused-linear-loss-extract.md`

- The upstream **chunked fused-linear-distill algorithm**, stage by stage, with
  the exact peak-memory it avoids (`chunk × vocab` vs full `T × vocab`) and the
  gradient flow (dloss → dhidden + d(lm_head)). Cite `file:line`.
- **Map to ARLE autograd**: which new op(s) we'd add (forward + backward), what
  the chunk loop looks like in our `Vec<f32>` host-authoritative backends, and
  where it plugs into the opd.rs windowed path (replacing
  `logits_from_hidden_window` + `kl_distill_loss`).
- **Open questions** for Claude (autograd op surface, CUDA vs CPU-first, how it
  interacts with the existing window scheme).
