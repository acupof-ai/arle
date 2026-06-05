# DSv4 perf endgame — architecture roadmap (adopt-best-first)

**Date:** 2026-06-05. Principle: `先用最好的再自己写,不要闭门造车`
([[feedback_no_closed_door_solutions]]). Every lever below leads with **what to
adopt** (vendored / upstream / proven) and writes custom only for the genuine gap.
Hypothesis-grade sequencing; each lever is license-or-kill on a wall-clock A/B at
the B=1 SLO shape.

## Where we are

Decode **39.5 ms/token (6× structural)**, GPU-kernel-bound. SGLang on the same
H20: **15.89 ms no-spec / 8.24 ms +EAGLE** (V3.2 proxy). Gap = **~2.5× kernels +
1.93× EAGLE**, plus serving-architecture levers ARLE lacks. The structural-overhead
arc (host-route / alloc / D2D / launch) is closed and measured; everything below is
kernel + serving architecture, and the recurring finding is **the best-practice
piece is often already vendored or config-scaffolded, just unwired.**

## The adopt-best-first endgame (sequenced)

| # | Lever | Adopt (best existing) | Write (the gap) | Δ target | State |
|---|---|---|---|---|---|
| **1** | **MLA attention** | the **vendored** FlashMLA fused sparse-decode kernel (`vendor/flashmla/.../splitkv_mla.cuh`) + shim + FP8-KV pack — all already in-tree | runtime wire-up in `attention.rs` (un-gate `alloc_fp8_arena`, dispatch, delete the 3 scalar kernels) | SM 1-3% → full occupancy; the ~10 ms decode-attn slice toward 2 ms | **in progress** (spec `2026-06-05-flashmla-sparse-decode-already-vendored-wireup-spec.md`) |
| **2** | **FP8 attention linear** | SGLang's **`fp8_gemm_nt` call *form*** (qkv-fused, activation quantized once, batched) — the DeepGEMM bridge `d41bb189` is already the entry | the fused call structure in the MLA-linear forward (per-projection swap already proved a kill) | the FP8-linear share of decode (SGLang 4.94 ms equiv) | after #1 |
| **3** | **EAGLE / spec decode** | the **vendored MTP draft head** (`mtp.0.*`, `num_nextn_predict_layers=1`, confirmed in the checkpoint — no training) + SGLang's MTP/EAGLE verify-loop structure | the draft-loop + tree-verify in ARLE's scheduler (`Engine<E,K>`), reusing the Medusa substrate | 1.93× (compounds on the kernel base: ×fast not ×slow) | banked (after kernels) |
| **4** | **DP-attention** | SGLang's `--enable-dp-attention` design (decode attention runs data-parallel → **no attention all-reduce**) | wire ARLE's existing-but-unwired `attn_dp_size` topo axis into the attention path | the `attn_allreduce` decode slice + better scaling | config axis exists (`infer-topo`), **unwired** |
| **5** | **DeepEP low-latency** | SGLang's `--deepep-mode low_latency` (pure-RDMA GPU dispatch/combine, hook overlap, ~20 SM) | replace the combine `ctx.sync` (the #24 bridge) with the LL-mode path | the MoE all-to-all decode cost | **not present** (#24 left a sync) |
| 6 | PD-disaggregation | SGLang's prefill/decode-node split + mooncake KV transfer + router | multi-node serving wiring | throughput/interference (not single-req latency) | multi-node, **deferred** (off single-pod scope) |

**Trajectory (hypothesis):** 39.5 ms → [#1 occupancy] → [#2 FP8 fused] → ~16 ms
(kernel parity with SGLang no-spec) → [#3 EAGLE ×1.93] → **~8 ms**. #4/#5 trim the
all-reduce + MoE-a2a slices on top. 5-6 ms remains H100-class.

## Architecture notes

- **#1 is the template.** The kernel-registry's "library-present but unwired" rows
  (`arle_flashmla_*`, the `attn_dp_size` axis) are the tell for adopt-first wins —
  the hard work was done in an earlier session and left unwired. Audit those rows
  before authoring anything.
- **#2's lesson is "match the call structure, not just the kernel"** — a correct
  kernel called 344×/forward ships ~0% ([[errors/2026-06-05-fp8-linear-per-projection-deepgemm-no-win]]).
  Adopt SGLang's *fused call form*, not just `fp8_gemm_nt`.
- **#3 compounds, so order matters** — EAGLE on a 16 ms kernel base ≈ 8 ms; on a
  39.5 ms base ≈ 20 ms. Kernels first (ckl: "kernel 是所有的一切的基础"), spec second.
- **#4/#5 are serving-architecture, not kernels** — they need engine/scheduler +
  topology changes (ARLE's strength), and DP-attention also unblocks better
  multi-request scaling beyond the single-token-latency goal.
- **Operator-library hygiene** (parallel, non-blocking): the `misc/` junk-drawer
  reorg + the post-#18 dead-code prune ([[2026-06-05-cuda-kernels-deadcode-scan]])
  land after the DSv4 kernel arc settles, since several dead rows are entangled
  with the in-flight FP8/route files.

## Verify-locally gates (per lever)

KV-precision-parity vs the bf16 reference (#1), wall-clock A/B at the B=1 SLO shape
(all), `strings | grep <symbol>` that the pod built the change before trusting
parity (the `2026-05-28-...precond-fail` trap), nsys/ncu before/after for the
occupancy/all-reduce deltas. No default flip without the matched A/B.
