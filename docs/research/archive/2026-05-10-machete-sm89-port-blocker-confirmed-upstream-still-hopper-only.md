---
title: Machete sm_89 port BLOCKER — upstream vLLM Machete still Hopper-only as of 2026-05-10
date: 2026-05-10
type: research
status: solid-blocker-needs-user-clarification
---

# Machete sm_89 port BLOCKER — upstream vLLM Machete still Hopper-only as of 2026-05-10

> User directive this tick: "Machete W4 kernel 移植 from vLLM —
> port machete from vllm-project/vllm to ARLE crates/cuda-kernels for
> sm_89 W4A8 优化 (预估 -20-40% ITL vs current Marlin)".
>
> **SOLID-critical contradiction**: prior ARLE survey killed Machete
> as Hopper-only on 2026-05-09. Fresh check this tick confirms
> upstream is still Sm90-only. Surfacing before kicking off a port
> that would KILL on first sm_89 build.

## §0 SOLID gate — primary evidence

### Evidence 1 — prior ARLE survey (2026-05-09)

`docs/research/2026-05-09-w4a8-industry-kernel-survey.md` §"❌ Machete (vLLM)
— Hopper-only,sm_89 不能用":

> - `vllm-project/vllm:csrc/quantization/machete/`(~3000 LOC)
> - Spiritual successor to Marlin,**基于 Cutlass + WGMMA + TMA**
> - 明确 `using ArchTag = cutlass::arch::Sm90`(只支持 sm_90+)
> - **结论:sm_89 RTX 4070 Ti SUPER 完全不能用**,跳过

Survey's actual P0 W4 recommendation (still valid):
> 🥇 **P0 推荐 — 移植 vLLM 当前 marlin(分阶段)**
> Phase 1 (Claude 1-2 天 OR codex 1 天): **抄 dequant.h + atomic_add option**
> ... LOC ~700, 风险低, 预估 gain ITL -3-8% + TTFT -2-5%

### Evidence 2 — fresh 2026-05-10 upstream check

`gh api repos/vllm-project/vllm/contents/csrc/quantization/machete/Readme.md`:

> "Machete is a spiritual successor to the Marlin kernel but
> **optimized for Hopper architectures** and based on Cutlass."

`gh api .../machete_mainloop.cuh` grep for `Sm[0-9]+`:

```
92:  using ArchTag = arch::Sm90;
157:    using ArchTag = arch::Sm90;
```

Both occurrences on the only place an SM tag is set in the mainloop —
**no Sm89 path, no fallback path, no conditional dispatch**.

Current Machete csrc inventory (11 files, all Sm90-targeted):
```
Readme.md
generate.py
machete_collective_builder.cuh
machete_interleaving_utils.cuh
machete_mainloop.cuh        ← Sm90-only
machete_mm_kernel.cuh
machete_mm_launcher.cuh
machete_prepack_kernel.cuh
machete_prepack_launcher.cuh
machete_prepacked_layout.cuh
machete_pytorch.cu
```

### Evidence 3 — architectural blockers for sm_89 backport

Even if a backport were attempted, three Hopper-specific dependencies
would each require non-trivial replacement:

1. **WGMMA (Warp Group MMA)** — Sm90 wmma.m64nNk16 instructions. sm_89
   has only `mma.m16n8k16` (older Ampere/Ada). All WGMMA call sites in
   `machete_mainloop.cuh` would need rewriting to mma.sync semantics +
   per-warp tile decomposition.
2. **TMA (Tensor Memory Accelerator)** — async global→shared copies via
   `cp.async.bulk.tensor`. sm_89 only has `cp.async` (Ampere-style).
   Loss of TMA means losing the prepacked-layout fast-path that is
   Machete's main perf differentiator over Marlin.
3. **Cutlass 3.x mainloop API** — Hopper collective builder
   (`machete_collective_builder.cuh`) targets `cutlass::arch::Sm90` as
   compile-time tag throughout. ARLE's current cutlass dependency
   (per `crates/cuda-kernels/build.rs`) would need verification + the
   Sm89 collective specializations from cutlass-itself would need to
   be wired (cutlass does have Sm89 GEMM specializations, but Machete
   does not use them).

Realistic backport magnitude: ~5000+ LOC architectural surgery,
high risk that the Sm89-backport result performs **worse** than the
current Marlin path because the prepacked-layout / TMA win is gone.

## §1 What the user might mean — disambiguation paths

### Path A — Machete Sm89 backport as named

User accepts the ~5000 LOC + multi-week + high-KILL-risk scope. Phase 0
spike: try compiling Machete with `arch::Sm89` substituted, fail fast
on WGMMA missing, document as KILL or pivot.

**Recommendation**: NOT P0. The KILL is near-certain at the WGMMA
substitution step, and even successful backport loses the prepack/TMA
win that is Machete's reason for existing.

### Path B — vLLM-current Marlin port (the prior survey's actual P0)

What the user **likely meant** given the "-20-40% ITL vs current Marlin"
target: port the **upstream evolved Marlin** (~5000 LOC across 6 files,
sm_75/80/89-compatible, what the 2026-05-09 survey actually
recommended). High-confidence sm_89 viability, multi-shape
specialization is the main perf delta vs ARLE's PR #31 cherry-pick.

**Recommendation**: **THIS IS P0.** Mirrors prior survey. Magnitude
similar to what user's directive estimates. ITL -3-8% per Phase 1
(dequant.h + atomic_add) is conservative; multi-shape specialization
in Phase 2 could plausibly hit -20-40% on certain N×K combos that
ARLE's single-template kernel mishandles.

### Path C — Different kernel — user knows something we don't

User may be referring to:
- **Machete-FP8** — a hypothetical W4A8 (FP8 activation) variant of
  Machete. **Searched**: no such variant in upstream Machete csrc;
  the FP8 path in vLLM's W4 kernels lives at
  `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu` (covered
  by Path B).
- **Marlin-Machete hybrid** — some Marlin-evolved kernel branded
  "machete" in a paper or fork. Not found in upstream.
- **A different repo's machete** — third-party port with sm_89 path?
  No such project surfaced via web search.

If user has a specific repo/branch in mind, please cite it and the
SM constraint claim re-evaluates. Otherwise default = Path B.

## §2 Decision request — needs user clarification

| Path | Magnitude | sm_89 viable | Expected ITL Δ | Risk |
|------|-----------|--------------|----------------|------|
| A — Machete sm_89 backport | ~5000 LOC + multi-week | KILL near-certain | unknown (likely worse than Marlin) | very high |
| B — vLLM-current Marlin port | ~700-2000 LOC × 2 phases | YES (sm_75+) | Phase 1: -3-8%; Phase 2: plausibly -20-40% | low-medium |
| C — Different "machete" user knows about | unknown until cited | unknown | unknown | unknown |

**Default Claude action absent user clarification**: pivot to Path B
(prior survey's documented P0). Wait one tick for user response, then
brief codex on Phase 1 of Path B.

## §3 What this tick produced

- This research entry (commit pending)
- PushNotification to user surfacing the SOLID contradiction
- No code changes. Codex's parallel #36 PrefixAware bench work
  (`infer/src/metrics.rs` + `metrics/render.rs` + `admission.rs` WIP
  this tick) continues unaffected.

## Cross-references

- Prior survey: `docs/research/2026-05-09-w4a8-industry-kernel-survey.md`
- Prior survey companion: `docs/research/2026-05-09-w4a8-upstream-qqq-survey.md`
- Current ARLE Marlin substrate:
  - `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (844 LOC)
  - `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC)
  - `crates/cuda-kernels/csrc/gemm/marlin_repack.cu` (151 LOC)
  - Total current = 1982 LOC
- Existing W4 dispatch: `infer/src/ops/linear.rs` (Marlin / Hybrid /
  W4A16Gemv / W4A16BatchGemv enum + select_w4_path)
- Upstream Machete Readme:
  https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/Readme.md
- Upstream Marlin (the right port target):
  https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin
- Memory: `feedback_first_principle_solid_or_deeper.md` (推断≠evidence;
  混淆变量必须隔离;root cause 假设也 license-or-kill)
- Memory: `feedback_p0_survey_before_plan_body.md` (P0 survey
  before plan body — applied this tick to catch the contradiction)

## 状态

**SOLID blocker confirmed.** Both prior survey and fresh upstream
check agree: Machete is Hopper-only. Pivoting to Path B (vLLM-current
Marlin port) is the recommended action absent user clarification.
PushNotification dispatched. Awaiting user decision — until then,
codex continues #36 PrefixAware bench (orthogonal, productive).
