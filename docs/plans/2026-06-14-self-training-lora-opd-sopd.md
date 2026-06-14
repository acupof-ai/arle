# Plan: Self-training LoRA OPD (SOPD) — on-device weight self-update

> ✅ **APPROACH DECIDED (ckl 2026-06-14).** After the apples-to-apples
> [survey](../research/2026-06-14-self-training-lora-options-survey.md) + three
> Codex-review passes (9 findings, all fixed), ckl locked the first cut:
> - **Timing = G2** (整句后更新 / per-sequence inline) — the update fires after each
>   rollout *sequence* on the rollout path; **not** per-token G3, **not** offline batch.
> - **Device/scope = CUDA + Qwen3.5-0.8B**, **自更新-only** (A1 EMA self-teacher) —
>   prove the loop first. Metal is a **separate later port** (today the OPD driver is
>   CUDA-gated, survey §1.3). best-of-N (A2) and 升级 (Phase 3) are **deferred
>   boosters/milestones**, not the bring-up keystone.
> - **Composition = A1 EMA (Axis A) + B1 LoRA q/v + C1 fp hot-swap + D1
>   single-process + G2 inline + H1 base-neutral design**, brought up on the
>   AttentionQv-wired Qwen3.5.
> **Gate before code**: this spans >5 files (infer-core, infer-seam, train/opd,
> infer-cuda, cli) — Claude delivers the line-level spec below; **await ckl's go**.

**Date**: 2026-06-14 · **Status**: approach-approved → awaiting plan sign-off → impl · **Driver**: ckl

> **One line**: turn the OPD LoRA loop into a *teacher-free, self-improving* loop
> that fires **at rollout time** — as the engine serves, each rollout *sequence*
> distills the student toward its own slow-EMA adapter (A1), updates the rank-r
> LoRA, and hot-swaps it live; periodically the adapter consolidates into a
> re-quantized base (升级). No external teacher; one process, one weight store
> (训推一体). First cut on CUDA+Qwen3.5; AIPC/Metal is a later port.

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

## Phase 0 — KEYSTONE: does the inline (G2) A1-EMA self-update loop run correctly and hold no-regression?

**ckl's locked first cut.** CUDA + Qwen3.5-0.8B, 自更新-only, EMA self-teacher,
update fires **after each rollout sequence** (G2) on the rollout path. This phase
proves the **mechanism** — inline loop + prefix-cache correctness + rollback
completeness + EMA anti-divergence + no-regression. The **capability-lift**
question is Phase 0.5 (best-of-N booster), kept separate because best-of-N is
batch-y and does not fit strict G2-inline. Run on the pod (CUDA), cheap, off the
Phase-1 batched-lane critical path.

### The inline self-update step (net-new, reuses GKD machinery)

Per served rollout sequence (G2):
1. **Rollout** — student decodes the sequence via the infer engine (exists,
   default-on, 4.99×; tape disabled, CUDA-graph + paged-KV).
2. **Score + step (two-pass-inline — the bring-up choice)** — keep the fast
   no-tape rollout, then recompute+step: set the OPD rollout = the just-served
   tokens; the existing `backward_chunked_kl_rollout` computes
   KL(student ‖ **EMA-self-teacher**) and steps AdamW on the **adapter only**.
   *(Fused-single-pass — reuse the rollout's own logits/activations in one pass —
   is a later perf option; it loses CUDA-graph speed, so bench before adopting.
   Survey §7 Q4.)*
3. **EMA update** — after the step: `θ_ema ← α·θ_ema + (1−α)·θ_student` over
   **adapter tensors only** (`α≈0.999`). Tiny elementwise op on rank-r A/B. The
   EMA *lag* makes the KL target non-degenerate (identical weights ⇒ KL=0 —
   survey §5 / Codex pass-2) and is the online-update anti-divergence anchor
   (mean-teacher).
4. **Accepted-update bookkeeping (REQUIRED, not optional)** — bump the adapter
   epoch and **invalidate/flush the prefix cache**. `RadixCache` is token-keyed
   with no version and `enable_prefix_cache` is default-on, so a later request
   sharing a token prefix would otherwise reuse KV computed under the *old*
   adapter epoch (Codex pass-2/3). Bring-up = full flush on each accepted update;
   production = epoch-tag pages, mismatch-is-miss (survey §7 Q10).

### Net-new code (small, concentrated)

- `train/src/ema_self_teacher.rs` — `EmaSelfTeacher: TeacherForward`. Mirrors
  `InProcessTeacher` but forwards `base(frozen) + EMA-adapter` with tape
  **disabled**. Shares the student's frozen base in the same `TensorStore` —
  **adapter-only EMA, ~10 MB, zero extra base memory**.
- EMA-update helper (~30 lines) over `adapter_name_map()` ids.
- **Adapter-epoch counter + prefix-cache invalidation hook** on accepted update —
  designed base-neutral (H1: an `infer-seam`/`infer-core` hook, **not** a Qwen3.5
  special-case), brought up on the wired Qwen3.5 path.
- Wire `EmaSelfTeacher` as the `teacher` arg into `backward_chunked_kl_rollout`;
  **no change to the loss/backward code itself.**
- A **windowed needle-gate driver** — snapshot/revert on regress; cadence =
  every N rollouts / wall-clock (survey §7 Q6).

### Memory budget (CUDA, any consumer card; the OOM blocker is gone)

| Component | Peak |
|---|---:|
| Qwen3.5-0.8B base (bf16, frozen, **shared** by student + EMA-teacher) | ~1.6 GB |
| Student adapter (r=16, q/v) + AdamW moments | ~50 MB |
| EMA-teacher adapter (adapter-only) | ~10 MB |
| Rollout KV (infer engine) | ~0.4 GB |
| OPD tape / activations | ~2 GB |
| **Total** | **~4.1 GB** (one base, not two models) |

### License-or-kill (explicit — mechanism keystone)

- **PASS**: the inline G2 loop runs end-to-end on CUDA+Qwen3.5; KL decreases;
  **no regression** vs the un-trained base on the needle ladder + ≥1 held-out
  capability dim (same-config-twice floor; NOT byte-identity — MoE
  non-determinism); the **prefix-cache epoch invalidation is verified** (a
  prefix-sharing request issued *after* an accepted update does NOT serve
  stale-epoch KV — assert via a targeted test); rollback restores adapter +
  AdamW + EMA cleanly. A measurable consistency/calibration gain is a **bonus,
  not the bar** — A1-EMA's honest effect is consistency / skill-shaping, not new
  knowledge (survey §5).
- **KILL**: the inline loop cannot hold no-regression after a recipe sweep
  (lr, α, λ, window N), or the prefix-cache/rollback machinery cannot be made
  correct at acceptable cost → the rollout-time *inline* shape is not viable;
  fall back to a periodic (near-online) cadence before porting anything.
- **Reward-hack tripwire**: n/a for pure A1 EMA (no verifier/reward) — becomes
  live in Phase 0.5.

## Phase 0.5 — capability booster: best-of-N rejection self-distill (A2, gated on Phase 0)

A1-EMA proves the loop and shapes consistency, but the *strong capability lift*
(GSM8K / code) comes from injecting "which trajectory is actually correct" — that
is **A2 best-of-N + verifier**, which is **batch-y** (N completions then a filter)
and so runs as a **periodic booster** alongside the continuous A1 loop, **not**
strictly inline. Per prompt batch: sample `N∈{4,8}` at `T>0`, a verifier (GSM8K
exact-match, revived M3) picks `τ*`, distill via GKD `λ·CE(student ‖ τ*) +
(1−λ)·KL(student ‖ EMA)`. This is the original capability keystone, now correctly
placed **after** the inline mechanism is proven. Net-new: best-of-N + verifier
wrapper around the rollout in `opd.rs`; verifier trait + GSM8K exact-match impl
(revive from retired `reward.rs` history).

- **PASS**: held-out lift over the base on **≥2 dims** (e.g. GSM8K + IFEval /
  code-unit-test — single-task is not enough,
  `wins/2026-05-22-opd-task-divergent-impact.md`), **multi-seed ≥5, mean ± σ +
  Wilson 95% CI** (2026-05-28 rule; <5pp likely → CI mandatory), U-curve caveat
  (eval *every* saved ckpt, `wins/2026-05-22-distill-trajectory-valley-then-recovery.md`).
- **KILL**: no CI-separated lift on ≥2 dims after a sweep (lr, N, λ, α) →
  teacher-free capability injection does not work here; the line is A1-consistency
  only, not capability growth.
- **Reward-hack tripwire**: held-out verifier + manual trajectory spot-check; if
  in-loop verifier score rises while held-out falls, the model is gaming its own
  judge — revert (drop adapter) and tighten the verifier.

---

## Phase 1 — AIPC device port (Metal; LATER, gated on Phase 0 PASS)

The AIPC/Mac payoff, but explicitly **after** the CUDA loop is proven (ckl's
"先证 loop"). Today the OPD *driver* is CUDA-gated (survey §1.3), so this is a
real port, not just speed work: `build_opd_store` (add a Metal arm),
`InferStudent`/LoRA hot-swap (`#[cfg(metal)]`), and CLI `--backend metal`. LoRA-
only then collapses the device cost: backward needs only **adapter A/B matmul
bwd + KL bwd + frozen base forward** (the base forward already runs on the Metal
infer executor) — far less than full M5.3b autograd op coverage.

- OPD-driver Metal port (`build_opd_store` / `InferStudent` / CLI `--backend metal`).
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
  → **on every accepted update** (gate pass): bump the adapter epoch and
  invalidate/version the prefix cache — pre-update pages are stale *even when no
  rollback occurs*, because the weights changed. Unconditional, not a rollback step.
  → **on rollback** (gate fail): restore student adapter + AdamW moments **and the
  EMA-teacher adapter** from the last-good snapshot, and invalidate the prefix cache
  for the rejected epoch (epoch-tag → mismatch-is-miss, or flush). Restoring only the
  student adapter (as a naive first cut would) leaves the EMA teacher trained against
  rejected state and serves stale-epoch KV — both silent correctness bugs.
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
Phase 0 (CUDA, inline G2 A1-EMA loop)   ← MECHANISM KEYSTONE, blocks all
   │  PASS ↓                               KILL ⇒ rollout-time inline shape not viable
   ├─▶ Phase 0.5 (best-of-N A2 capability booster, periodic — the lift gate)
   ├─▶ Phase 1 (Metal device port: OPD-driver port + EmaSelfTeacher; HIP/Vulkan later)
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
