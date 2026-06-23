# Multi-arm OPD death (dense 27B) is the GPU memory ceiling — not JIT-cache, not session

## Context

[`2026-06-19-opd-multiarm-shared-resource-collision.md`](2026-06-19-opd-multiarm-shared-resource-collision.md)
left a single-variable test owed: 2 concurrent OPD arms, unique tmux session,
**shared vs unique `DG_JIT_CACHE_DIR`** — does shared-JIT kill them (DeepGEMM
JIT-cache race, the prime suspect) or is it the session? Run on the new H20 box
with Qwen3.6-27B-FP8, the actual current OPD student.

## Root Cause

**The death is the per-GPU memory ceiling, reproduced and measured — not the JIT
cache and not the session.** Evidence, in order:

1. **DeepGEMM does not engage for the dense 27B.** With a binary built
   `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`, a single OPD step wrote **0 files** to
   `DG_JIT_CACHE_DIR` and emitted no DeepGEMM/JIT messages. Qwen3.6-27B-FP8 is
   DENSE; DeepGEMM is the MoE grouped-GEMM (FP8) path — the dense FP8 GEMM goes
   through TileLang/cuBLAS. So there is no DeepGEMM JIT to race. (No FP8-MoE model
   is on the box to test the JIT hypothesis directly: `Qwen3-30B-A3B` and
   `Qwen3.5-122B-A10B` are both bf16 `quant: none`; the 122B is 234 GB.)
2. **Shared vs unique `DG_JIT_CACHE_DIR` made no difference.** Both conditions
   failed identically and fast (~16 s, at load), `ARM_EXIT=1` (graceful error, not
   the SIGKILL 137 of the original report) — so the JIT dir is not the variable.
3. **Per-GPU memory timeline (the decode).** Two concurrent arms, GPU 0 and 1,
   1 s sampling:
   ```
   GPU0 (arm-A): 0 → 7.4 → 26.9 → 53.3 → 76.9 → 90.4 → 95.2 GB  → OOM, crash
   GPU1 (arm-B): 0 → 7.2 → 24.5 → 43.6 → 54.9 → 57.4 GB         → D2D error, crash
   ```
   arm-A hit **95,156 / 97,871 MiB (97%)** during the **infer-teacher** upload and
   OOM'd on `layers.19.mlp.gate_proj` (`engine build failed: upload FP8
   block-scaled tensor`). `INFER_CUDA_DEVICE` is honored (A→GPU0, B→GPU1, cleanly
   separated — no device pile-up).

Each arm loads **~3×27B on one GPU** — infer-rollout student + infer-teacher
(`--teacher-runtime infer`) + autograd training student — peaking ~86 GB solo
(which succeeds) but ~95 GB under concurrent load (the FP8 upload's transient
staging rises when the upload is slowed by host/PCIe contention from the other
arm), tipping over the 97 GB ceiling. The original "single stable / any 2
concurrent die fast" signature is exactly an OOM at ~97% with <3 GB headroom.

arm-B's `cuda D2D copy failed (bf16 bridge)` at step 1 is a secondary concurrent
symptom (it had not OOM'd at 57 GB); whether it reproduces in isolation needs its
own decode and is not the primary cause.

## Fix

Isolation done — the lever is **per-arm GPU footprint**, not JIT/session:
- **Reduce footprint** (best): share weights between the infer-rollout student
  and the autograd training student (they are the same student) so an arm loads
  ~2×27B not 3×27B; or place the infer-teacher on a separate GPU from the student.
- **Operationally now:** the dense-27B OPD arm with `--teacher-runtime infer`
  self-distill sits at ~86–95 GB on one GPU — at the edge of a 97 GB H20. Do not
  stack it, and stagger arm *loads* (start the next arm after the previous
  finishes uploading) so transient peaks don't overlap. A single arm is stable.
- The 2026-06-19 unique-`DG_JIT_CACHE_DIR` advice remains a correct zero-cost
  default for any *FP8-MoE* workload (which does JIT-warm DeepGEMM), but it is not
  what was killing the dense-27B arms.

## Rule

"Single stable / concurrent dies" is a shared-resource signature, but **which**
resource must be measured, not assumed — the 2026-06-19 entry's DeepGEMM-JIT
"prime suspect" was right to flag a shared resource and right to demand the
single-variable test, but for the dense workload the resource is plain **GPU
memory**, and the model never even engages DeepGEMM. Sample **per-GPU memory at
1 s during the concurrent run**; an OOM at ~97% on the model whose upload errors
is the proof. Don't carry a hypothesis (JIT cache) across a model-class change
(MoE→dense) without re-measuring. See
[[reference_dsv4_deepgemm_jit_cache_persist_62]] and
[[feedback_vram_accounting_bit_exact]].
