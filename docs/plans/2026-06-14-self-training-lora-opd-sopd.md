# Plan: Self-training LoRA OPD (SOPD) — on-device weight self-update

> ⚠️ **DRAFT — ONE CANDIDATE, AWAITING ckl's DECISION.** Per ckl 2026-06-14:
> survey peer technical options first (apples-to-apples), clarify all
> current-state detail, size/hardware budgets, the extreme achievable state +
> 训推一体, and only start once the simplest viable approach is clear. **The
> survey has now landed:**
> [`../research/2026-06-14-self-training-lora-options-survey.md`](../research/2026-06-14-self-training-lora-options-survey.md).
> Its §6 recommends the *simplest-viable* composition = **A1 EMA self-teacher +
> B1 vanilla LoRA + C1 fp hot-swap (自更新-first)**, with this doc's verifier-
> selected best-of-N as the **A2 booster** and merge-then-requant (升级) as a
> **separable later milestone** — NOT the bring-up keystone. Do not implement
> until ckl selects an approach; if A1-first is chosen, demote the best-of-N
> keystone here to a Phase-2 booster.

**Date**: 2026-06-14 · **Status**: draft candidate (superseded-pending-survey) · **Driver**: ckl

> **One line**: turn the OPD LoRA loop into a *teacher-free, self-improving,
> on-device* loop — the model distills from itself (verifier-selected best-of-N
> + EMA-anchor), updates a LoRA adapter, hot-swaps it into its own serving
> engine, and periodically consolidates the adapter into a re-quantized base.
> No external teacher checkpoint; runs on AIPC unified-memory silicon.

## Strategic placement (honest)

This is a **Phase-3+ axis** under
[`2026-06-10-arle-master-strategy-v2.md`](../projects/2026-06-10-arle-master-strategy-v2.md):
it fuses Phase 3 #3 (OPD reclaims GPU) + #5 (AIPC route, #71) and **re-opens
the retired [`agent-rl-self-evolving`](../projects/agent-rl-self-evolving.md)
doctrine**. It is licensed to exist *only as distillation* (KL/CE self-distill),
**never GRPO** — that is the line the [2026-05-18 OPD-only
pivot](../projects/2026-05-18-opd-only-pivot.md) drew (the 322× pretrain gap and
the "RL duplicates verl/TRL" argument both still hold). Under D4 (three lines
strictly serial) this does **not** pre-empt the Phase-1 batched-lane keystone;
the Phase-0 keystone below is a *cheap* CUDA premise-test that can run on the
pod off the critical path. Everything after Phase 0 is gated on it.

## Why LoRA-only makes this tractable (the structural unlock)

base frozen + quantized → backward flows **only through the rank-r adapter**,
never through Marlin/quantized base. This single property solves four problems
at once:

1. **AIPC-trainable** — no backward through the quantized base (the "pure-Rust
   train-infer is feasible" premise, retired doctrine §1.2).
2. **Teacher-free memory** — student and self-teacher *share the same frozen
   base*; the only per-role state is a ~10 MB adapter. The
   teacher+student co-residency OOM that
   [blocked the 9B→0.8B plan](2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md)
   (15871/16384 MiB) disappears.
3. **Trivial rollback** — base never mutates; a bad self-update is reverted by
   restoring the adapter + AdamW moments (§Mutated state).
4. **Quant only on the base, only on "upgrade"** — `更新` (update) touches the
   fp adapter; `升级` (upgrade) is the only event that re-quantizes (§Two
   cadences).

## Two cadences (= the user's "自更新和升级,流式自动量化")

| Term | Cadence | Mechanism | Re-quant? |
|---|---|---|---|
| **自更新** (update) | fast / continuous | LoRA self-distill step → fp adapter hot-swap into serving engine | no |
| **升级** (upgrade) | slow / periodic | `merged_tensor()` folds accumulated adapter into base → **re-quantize base** → new frozen quantized base, adapter reset to zero | **yes — this is `流式自动量化`** |

Existing primitives: `lora.rs:218 merged_tensor()` already folds A·B into a
dense base; `infer-cuda/src/qwen35.rs::remerge_student_lora` already hot-swaps a
fresh adapter into the **bf16** serving base each step. The two net-new gaps are
both *on the quantized base*: (a) a quantized-base + fp-adapter separate
low-rank serve path (today `merge_lora_proj` `bail!`s on a non-bf16 base), and
(b) a base re-quantize-after-merge primitive (the tree has only MoE *activation*
requant, no *weight* requant).

## Existing substrate (reuse, do not reinvent)

| Capability | Status | Anchor |
|---|---|---|
| OPD LoRA step (forward-KL, GKD λ-blend, KL-mask, chunked-KL, windowed-logits) | ✓ | `train/src/opd.rs::opd_step`, `backward_chunked_kl_rollout` |
| `TeacherForward` trait + in-process teacher | ✓ | `train/src/teacher_infer.rs` |
| LoRA on q/v (hybrid Qwen3.5: 6 full-attn layers), config, merge, name-map | ✓ | `train/src/lora.rs` (`LoraTargetSet::AttentionQv`, `merged_tensor`, `adapter_name_map`) |
| Fast rollout via infer engine (CUDA-graph + paged KV, 4.99×, **default**) | ✓ | `train/src/infer_student.rs`, `2026-05-29-opd-infer-rollout-default-p4.md` |
| Per-step LoRA hot-swap (cached-base re-merge, bf16) | ✓ | `infer-cuda/src/qwen35.rs::remerge_student_lora` |
| GKD SFT anchor on student rollout + λ blend (the best-of-N distill primitive) | ✓ | `opd.rs::GkdSftAnchor::StudentRollout`, `mix_gkd_losses` |
| Correct-inference gate (needle ladder, not byte-identity) | ✓ | `scripts/needle_gate.py`, `scripts/lever_gate.sh` |
| Math exact-match verifier (the Phase-0 reward) | retired (was M3) | revive from `agent-rl-self-evolving` §3, `examples/opd/gsm8k-train.jsonl` |

Net-new is small and concentrated: an EMA self-teacher, a best-of-N+verifier
selection wrapper, a Metal device port (adapter path only), a quantized-base
serve path, and the upgrade/requant step.

---

## Phase 0 — KEYSTONE: does teacher-free self-distillation lift capability?

**This gate decides whether the entire AIPC self-training line is alive.** Run
on the pod (CUDA), cheap, off the Phase-1 critical path. **No external teacher.**

### The self-distillation engine (net-new, but reuses GKD machinery)

Per prompt, per step:
1. **Sample best-of-N** — student samples `N∈{4,8}` completions at `T>0` via the
   infer engine (rollout exists greedy; add temperature). A **verifier**
   (GSM8K exact-match, revived M3) scores each; pick the best trajectory `τ*`.
   If none verify, skip the prompt (no signal) — log the skip rate.
2. **Distill toward `τ*`** — set the OPD rollout = `τ*`, then the existing GKD
   loss: `λ`·CE(student ‖ `τ*` tokens, `GkdSftAnchor::StudentRollout`) +
   `(1−λ)`·KL(student ‖ **EMA-self-teacher** on `τ*`). The EMA term is the
   anti-collapse regularizer; the CE-to-best-trajectory is the capability
   signal (self-improvement / ReST/STaR-style, but distillation not RL).
3. **EMA update** — after the AdamW step:
   `θ_ema ← α·θ_ema + (1−α)·θ_student` over **adapter tensors only**
   (`α=0.999`). Tiny elementwise op on rank-r A/B.

### Net-new code (small)

- `train/src/ema_self_teacher.rs` — `EmaSelfTeacher: TeacherForward`. Mirrors
  `InProcessTeacher` but forwards `base(frozen) + EMA-adapter` with tape
  **disabled**. Shares the student's frozen base tensors in the same
  `TensorStore` (zero extra base memory).
- EMA-update helper (~30 lines) over `adapter_name_map()` ids.
- Best-of-N + verifier wrapper around the rollout call in `opd.rs`; verifier
  trait + GSM8K exact-match impl (revive from retired `reward.rs` history).
- Wire `EmaSelfTeacher` as the `teacher` arg into `backward_chunked_kl_rollout`;
  no change to the loss/backward code itself.

### Memory budget (any consumer card; the OOM blocker is gone)

| Component | Peak |
|---|---:|
| Qwen3.5-0.8B base (bf16, frozen, **shared** by student + EMA-teacher) | ~1.6 GB |
| Student adapter (r=16, q/v) + AdamW moments | ~50 MB |
| EMA-teacher adapter | ~10 MB |
| Best-of-N rollout KV (infer engine, N parallel) | ~0.5 GB |
| OPD tape / activations | ~2 GB |
| **Total** | **~4.2 GB** (fits 8 GB; one base, not two models) |

### License-or-kill (explicit)

- **PASS**: held-out GSM8K lift over the un-trained base, **multi-seed ≥5,
  mean ± σ + Wilson 95% CI** (the 2026-05-28 rule — magnitude likely <5pp on a
  small eval, so CI is mandatory), with the U-curve caveat (eval *every* saved
  checkpoint; valley-then-recovery is the literature-default trajectory,
  `wins/2026-05-22-distill-trajectory-valley-then-recovery.md`). Add a 2nd
  capability dim (IFEval or a code-unit-test task) — single-task is not enough
  to verdict (`wins/2026-05-22-opd-task-divergent-impact.md`).
- **KILL**: no CI-separated lift on ≥2 dims after a recipe sweep (lr, N, λ, α)
  → teacher-free self-distillation does not inject capability here. **Stop —
  do not port to Metal / build quant-base serve / build upgrade.** The entire
  AIPC self-training vision is dead without this.
- **Reward-hack tripwire**: held-out verifier + manual trajectory spot-check;
  if the in-loop verifier score rises while held-out falls, the model is gaming
  its own judge — revert (drop adapter) and tighten the verifier.

---

## Phase 1 — AIPC device port (Metal first; gated on Phase 0 PASS)

LoRA-only collapses the device cost: backward needs only **adapter A/B matmul
bwd + KL bwd + frozen base forward** (the base forward already runs on the Metal
infer executor). This is far less than the full M5.3b autograd op coverage.

- Metal `EmaSelfTeacher` + Metal `InferStudent` (mirror the CUDA ones on the
  existing Metal autograd backend + Metal infer executor).
- Acceptance: Phase-0 loop converges on Metal (KL down + same GSM8K lift shape),
  step time within a stated factor of CUDA. Canary: step-1 token agreement
  vs CUDA path (BF16 rounding tolerance, per the rollout-via-infer canary).
- HIP/Vulkan training is a **later** P1 extension (net-new autograd `Backend`
  impl — the *autograd* seam, not the inference `BackendExecutor`/`KvPool` seam
  that master-strategy-v2 #71 covers). Defer until Metal proves the loop.

## Phase 2 — quantized-base + fp-adapter serve path (the AIPC serving shape)

On AIPC the served base is quantized (W4A8 / Q4_K). `merge_lora_proj` refuses a
non-bf16 base, so the per-step hot-swap must become a **separate low-rank
path**: quant-base GEMM ‖ fp-adapter low-rank GEMM (standard quant+LoRA
serving). Net-new in `infer-cuda` / `infer-metal` student forward. Gate:
correct-inference (`needle_gate.py`) holds vs the bf16-merge reference.

## Phase 3 — 升级 / streaming auto-quant (the `流式自动量化`)

Periodic consolidation: `merged_tensor()` folds the accumulated adapter into the
base → **re-quantize the merged base** (net-new weight-requant primitive, reuse
TurboQuant/Marlin/Q4_K kernels) → new frozen quantized base, adapter reset to
zero, AdamW moments reset. Cadence = every K updates *or* when adapter norm
crosses a threshold. Gate: needle + same-config-twice floor + self-consistency
across the requant boundary (NOT byte-identity — MoE non-determinism,
`feedback_correct_inference_not_baseline_identity`). A failed gate reverts to
the pre-upgrade quantized base + adapter.

## Phase 4 — skills as the on-policy data engine (closes the self-evolving loop)

Replace static GSM8K prompts with **live agent skill/tool trajectories**: the
agent does real tasks using its skills → `(prompt, trajectory, outcome)` →
verifier / self-judge → best-of-N selection (Phase 0 engine) → adapter
self-distill. Net-new: a trajectory-emit channel from the agent/tools crate into
the OPD data buffer (the retired doctrine's §0 data flow, minus GRPO). The
"upgrade" gate (Phase 3) becomes the skill-acquisition checkpoint: the agent's
weights consolidate only after a demonstrable, held-out skill improvement.

---

## Mutated state (rollback enumeration — every cadence, per §0.1)

- **Per update step**: student adapter A/B; AdamW moment buffers; **EMA-teacher
  adapter** (advanced by the EMA update — itself mutated state, not read-only);
  **served prefix-KV pages** (now computed under the *old* adapter epoch —
  `RadixCache` is token-keyed with no version, `enable_prefix_cache` default-on);
  (read-only, safe) the re-merge pristine base cache.
  → rollback = restore student adapter + AdamW moments **and the EMA-teacher
  adapter** from the last-good snapshot, **and invalidate/version the prefix cache
  for the rejected epoch** (epoch-tag → mismatch-is-miss, or flush). Restoring only
  the student adapter (as a naive first cut would) leaves the EMA teacher trained
  against rejected state and serves stale-epoch KV — both silent correctness bugs.
- **Per upgrade**: base weights + quant scales; **AdamW moments are reset to zero
  (Phase 3)** — so they are mutated state on this cadence too.
  → rollback = restore pre-upgrade quantized base + the adapter that produced it
  **+ the EMA-teacher adapter + the pre-reset AdamW moments** + invalidate the
  prefix cache (base weights changed). Snapshot before merge; never overwrite in
  place until the gate passes.
- Base weights are **immutable** within the 自更新 cadence, so no update can corrupt
  the base — the strongest safety property LoRA-only buys. (The 升级 cadence is the
  one place the base mutates; that is why it needs the full snapshot above.)

## DAG / critical path

```
Phase 0 (CUDA, EMA self-teacher + best-of-N verifier)   ← KEYSTONE, blocks all
   │  PASS ↓                                                KILL ⇒ line dead
   ├─▶ Phase 1 (Metal device port; HIP/Vulkan later)
   ├─▶ Phase 2 (quant-base + fp-adapter serve)  ──┐
   │                                              ├─▶ Phase 3 (升级 / requant)
   └──────────────────────────────────────────────┘
                                                   └─▶ Phase 4 (skills data engine)
```

## Cross-links

- Strategy: [`2026-06-10-arle-master-strategy-v2.md`](../projects/2026-06-10-arle-master-strategy-v2.md) §3 Phase 3 #3/#5, D4, §5 ROCm/AIPC
- OPD-only pivot (the distillation-not-RL line): [`2026-05-18-opd-only-pivot.md`](../projects/2026-05-18-opd-only-pivot.md)
- Retired doctrine (data flow + verifier + reward-hack risk): [`agent-rl-self-evolving.md`](../projects/agent-rl-self-evolving.md)
- Rollout-via-infer (LoRA-sync mechanism, AttentionQv lock): [`2026-05-29-opd-student-rollout-via-infer.md`](2026-05-29-opd-student-rollout-via-infer.md)
- Teacher+student OOM (why teacher-free matters): [`2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md`](2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md)
- Eval discipline: `errors/2026-05-28-mmlu-cross-base-was-noise.md`, `wins/2026-05-22-distill-trajectory-valley-then-recovery.md`, `wins/2026-05-22-opd-task-divergent-impact.md`
