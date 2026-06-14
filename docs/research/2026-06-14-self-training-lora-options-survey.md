# Research: Self-Training LoRA OPD (SOPD) — apples-to-apples options, budget, ceiling

**Date**: 2026-06-14
**Status**: research / survey — NOT a plan. Commissioned by ckl: "先去调研技术方案有哪些
选择,方案之间要对等,搞清楚了认为简单了咱们再开工;所有的现状细节要梳理清楚;不同尺寸
模型的预算和硬件需求,以及我们能做到的极致状态和最终效果;训推一体。"
**Deliverable**: option tables (peers comparable on the same dimensions) per design axis,
the full current-state ledger, a per-model-size budget × hardware matrix, the train-infer-
unified framing, and the achievable ceiling — so a *simple* composition can be chosen before
any implementation plan is written.
**Supersedes-pending-decision**: [`docs/plans/2026-06-14-self-training-lora-opd-sopd.md`](../plans/2026-06-14-self-training-lora-opd-sopd.md)
(that draft committed to ONE composition — EMA self-teacher + best-of-N — before this survey;
treat it as one candidate in §2, not the chosen approach).

---

## 0. The question, stated precisely

ckl's vision decomposes into one runtime capability with two cadences:

> A single ARLE process on an AIPC serves the user **and** improves its own weights from its
> own work ("skills"), **teacher-free**, *as it serves* — **the update logic runs at rollout
> time**, not in a separate batch/idle phase. The better adapter is already live (**自更新 /
> update**); periodically it folds into the base and re-quantizes (**升级 / upgrade =
> 流式自动量化**). One runtime, one weight store: **训推一体**. The loop is **base-model-agnostic**
> — you size it by **hardware budget**, then pick whatever base fits.

> **Corrections from ckl (2026-06-14), now binding on this doc:**
> 1. **Update fires AT rollout time** (online, inline) — NOT an offline/idle batch loop. The act
>    of rolling out *is* the update. (Reframes the cadence; see new **Axis G**. This is actually
>    truer to the existing `opd_step`, which already does rollout→backward→step in one tight loop.)
> 2. **Base-model-agnostic; gate on hardware budget.** The loop is a *unified engine capability*
>    (like paged-KV — model-neutral in the seam, per `feedback_unified_abstraction_not_per_model`),
>    not a per-model path. The budget question is therefore **budget-first, base-second** (§3
>    reoriented; see new **Axis H**).

Seven things must be true for that to be *simple and real*, and each is a design axis with
peer options (§2):

1. **Where does the learning signal come from with no external teacher?** (signal source — Axis A)
2. **What is the trainable surface?** (PEFT variant / target / rank — Axis B)
3. **How does a trained adapter become a quantized base?** (streaming auto-quant / 升级 — Axis C)
4. **What is the process shape that makes train == infer?** (训推一体 boundary — Axis D)
5. **Which device runs the loop, and what generates the prompts?** (backend + skills data — Axes E/F)
6. **When does the update fire relative to rollout?** (online coupling — **Axis G**, ckl correction 1)
7. **Where does the loop live — per-model or seam-level?** (base coupling — **Axis H**, ckl correction 2)

This doc does NOT pick the composition. It lays out the peers, the costs, and the ceiling.

---

## 1. Current-state ledger (evidence vs gap — §0 SOLID discipline)

Everything below is **read from the tree at HEAD** (evidence) unless tagged *(gap)* or
*(hypothesis)*. The point of the ledger is that SOPD is mostly *re-wiring existing parts*, not
new substrate — that is the argument for "simple".

### 1.1 What exists today (evidence)

| Capability | Where | State |
|---|---|---|
| One in-tree training axis: **OPD** (frozen teacher → student, forward-KL) | `crates/train/src/opd.rs` `opd_step` / `backward_chunked_kl_rollout` | mature; CLI `arle train opd` |
| GKD λ-blend, KL-mask (`Full`/`CompletionOnly`), chunked-KL, windowed-logits, SFT-anchor (`StudentRollout`/`CorpusTruth`) | `opd.rs` `GkdLossConfig`, `mix_gkd_losses` | mature |
| **On-policy rollout through the infer engine** (CUDA-graph + paged-KV), default ON | `infer_student.rs`, `opd.rs` `infer_rollout_flag_enabled` (`ARLE_OPD_INFER_ROLLOUT`) | default; 4.99× vs hand-decode ([2026-05-29 win](../experience/wins/2026-05-29-opd-infer-rollout-default-p4.md)) |
| **Teacher = serving path, in-process** (the 训推一体 seed already exists) | `teacher_infer.rs` `InProcessTeacher` / `TeacherForward`, `engine.forward_token_logits` | mature; teacher consumes only `rollout: Vec<u32>` |
| Per-step LoRA sync into a live infer engine (cached-base in-memory re-merge, 6×(q,v)) | `infer_student.rs:148` `sync_lora_from_store`; `infer-cuda/qwen35.rs` `remerge_student_lora`/`merge_lora_proj` | mature, AttentionQv-locked |
| LoRA substrate: A·B, scale=α/r, `merged_tensor`, target sets `AttentionQv`/`AllLinear` | `crates/train/src/lora.rs` | mature |
| Engine offload (student/teacher/all) to fit co-residency | `opd.rs` `EngineOffloadMode` | mature |
| Autograd: CPU (ref) + Metal (MLX) + CUDA, AdamW + checkpoint codec | `crates/autograd/`, `adamw_state.rs` `export_state`/`import_state` | M5.3a device-resident done |
| Correct-inference gate (needle ladder + same-config-twice floor + self-consistency) | `scripts/needle_gate.py`, `scripts/lever_gate.sh` | the gate, NOT byte-identity |
| Memory pre-flight: params / trainable / grad / adam / activation-floor | `arle train estimate-memory` (`crates/cli/src/train_cli.rs:149`) | the SOLID validator for §3 below |
| Static quant at load: TurboQuant, Marlin W4A8, KV-quant, GGUF Q-types (HIP) | `cuda-kernels/csrc/quant/`, `infer-hip` | mature, load-time only |

### 1.2 The structural unlock that makes self-training cheap (evidence)

**LoRA-only training means backward never flows through the quantized base.** Base weights stay
frozen; the tape only carries the rank-r adapter (`lora.rs` `LinearWithLora` applies
`delta*scale`, base is `&self` immutable per the project weight-sharing rule). Consequences,
each load-bearing for AIPC feasibility:

- **Optimizer + gradient memory is governed by adapter size (MB-scale), not model size.** AdamW
  m/v + grads are allocated for trainable params only (`estimate-memory` separates
  `trainable_param_count`). This is why a 30B base can self-train where a full-FT 30B cannot.
- **Teacher-free self-distillation shares the base.** The teacher signal comes from the
  *same frozen base* plus an **adapter-only EMA snapshot** (≈ MB; never a second base copy — the
  EMA tracks the rank-r adapter, not the merged weights), so there is **no second model copy** —
  this is exactly what killed the classic 9B→0.8B OPD plan (teacher+student co-residency OOM,
  15871/16384 MiB; [plan](../plans/2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md)).
  Self-teacher removes that line item entirely.
- **Quant only ever touches the base.** Adapter trains in fp; 自更新 hot-swaps the fp adapter
  (no requant); 升级 is the *only* operation that re-quantizes, and it is rare.

### 1.3 The gaps (what SOPD actually needs to build)

| Gap | Why it's a gap | Severity |
|---|---|---|
| **Teacher-free signal** — every current path needs a *separate* teacher dir (`teacher_dir` in `run_opd_from_dirs`); `--teacher-model` defaults to student dir but the loss still distills toward a frozen copy, not a self-improvement target | `opd.rs` has no self-teacher / no rejection-filter loop | **core** — this is the SOPD idea |
| **Quantized-base serving of the student** | `merge_lora_proj` "requires dense BF16 base weights"; student base is bf16 today (only teacher is W4A8). 升级 into a 4-bit base has no path | **core** (升级 only; 自更新 works on bf16 base) |
| **Streaming requant (流式自动量化)** | quant is load-time only; no merge-then-requant cadence; no QA-LoRA-style zero-point merge | **core** (升级 only) |
| **Skills → training prompts** | OPD eats static jsonl (`examples/opd/*.jsonl`); no capture of real agent tool-use traces | **feature** (can start with static, add later) |
| **Metal/CUDA autograd op coverage** for the full student backward | M5.3b pending; many ops still CPU-readback per step on Metal | **perf** (works, slow on Metal today) |
| **OPD driver is CUDA-only** — `build_opd_store` falls back to CPU off-CUDA; the inline-rollout student (`InferStudent` / LoRA hot-swap) is `#[cfg(feature="cuda")]`; the CLI `--backend` enum is `auto\|cpu\|cuda` (no `metal`) | the *autograd* backend supports Metal, but no path drives an *inline rollout-time* loop on Metal today | **scope** — Metal SOPD is an unstarted port, NOT runnable today |
| **No HIP/Vulkan training** | autograd backends are CPU+Metal+CUDA only | **scope** — AIPC train target is Metal (M-series) or CUDA, NOT the HIP/Vulkan inference lanes |

---

## 2. Design axes — apples-to-apples option tables

Each table compares peers on the **same** dimensions, with a "simplest-viable" flag. The
recommended composition (§6) just reads off the flags — but the alternatives are real and the
choice is ckl's.

### Axis A — Self-improvement signal source (teacher-free)

The literature splits cleanly into **soft (per-token KL)** vs **hard (filter-then-CE)** families.

| Option | Signal | Needs verifier? | Needs 2nd model? | On-policy? | Reward-hack risk | OPD-license fit | Impl cost |
|---|---|---|---|---|---|---|---|
| **A1. EMA / mean-teacher self-distill** | soft KL: student ← EMA(adapter) per-token | no | no (EMA = adapter-only snapshot, ~MB; base shared) | yes | low (no reward) | ✅ pure distillation | low — reuse `backward_chunked_kl_rollout`, teacher = EMA-adapter snapshot |
| **A2. Best-of-N rejection self-distill** (ReST-EM / STaR / RFT) | hard CE on verified-correct rollouts only | **yes** (answer/schema/tool verifier) | no | yes | medium (verifier gaming) | ✅ CE/SFT-anchor, NOT GRPO | medium — N-sample + filter + SFT-anchor path exists |
| **A3. Self-consistency distill** | hard CE/soft KL toward majority-vote pseudo-label | no (uses agreement) | no | yes | medium (mode collapse) | ✅ distillation | medium — N-sample + vote |
| **A4. External small peer teacher** | soft KL from a *different* small model (not self) | no | **yes** (2nd base resident) | yes | low | ✅ classic OPD | low (existing path) but **breaks "teacher-free"** + costs memory |

Dimensions that decide it: A1 is the only one that is **teacher-free, verifier-free, pure-soft-KL,
and reuses the existing KL backward unchanged** — lowest new surface. A2/A3 add a sampling+filter
loop and (A2) a verifier, but give a *stronger* improvement signal on tasks with a checkable
answer (math, code, tool-call schema) — and A2 is the closest match to ckl's word "skills"
(verified task completion). **Apples-to-apples verdict**: A1 = simplest correct; A2 = strongest
on verifiable skills. They compose (A1 KL as the always-on floor, A2 CE as a periodic booster).
Literature anchors: ReST-EM/ReST (ICLR'24), "Self-Distilled Reasoner: On-Policy Self-Distillation"
(2026), "A Model Can Help Itself: Reward-Free Self-Training" (2510.18814).

**License constraint (hard):** all four are distillation/CE — none is GRPO. The 2026-05-18
OPD-only pivot retired GRPO/multi-turn-RL on an economic argument; A-options stay on the licensed
side of that line precisely because the update is KL or CE, never a policy-gradient objective.
([agent-rl-self-evolving.md](../projects/agent-rl-self-evolving.md) is RETIRED; its data-flow
shape — rollout→reward→loss→hot-swap — is reusable ONLY with a distillation loss.)

### Axis B — PEFT structural variant

| Option | Extra params | Merge-into-quant? | Accuracy vs full-FT | Autograd impl cost | AIPC fit |
|---|---|---|---|---|---|
| **B1. Vanilla LoRA (q/v)** — current | tiny (§3) | dense-bf16 only (current `merge_lora_proj`) | good for narrow adaptation | **zero** (exists) | ✅ |
| **B2. DoRA** (weight-decomposed) | tiny + magnitude vec | needs decompose-aware merge | better than LoRA at low rank | medium (new op) | ✅ but new |
| **B3. QA-LoRA** (quantization-aware) | tiny, group-aligned | ✅ **lossless into quantized base** (zero-point update) | matches QLoRA, no PTQ loss | medium (group-aligned A/B + zero-point merge) | ✅ — **the 升级 enabler** |
| **B4. Full fine-tune** | = model size | n/a | best | n/a | ❌ memory |

**Verdict**: B1 for the 自更新 cadence today (it already works). B3 (QA-LoRA, ICLR'24) is the
*correct* answer for the 升级 cadence — it merges the adapter into the 4-bit base by updating the
group zero-points, **no dequant→merge→requant round-trip and no bf16 base cache required**. B2
(DoRA) is a quality upgrade orthogonal to the quant story; defer unless low-rank quality is the
bottleneck. Related 2025 work: CLoQ (calibrated SVD init), IntLoRA (integer-only adapters),
LoTA-QAF (lossless ternary), LoraQuant (PTQ of the adapter).

### Axis C — Quant-in-the-loop / streaming auto-quant (升级 path)

| Option | Requires bf16 base cache? | Accuracy delta | Upgrade latency | Reuses ARLE kernels? | When |
|---|---|---|---|---|---|
| **C1. fp adapter hot-swap, no requant** | n/a (adapter stays fp) | 0 (no merge) | ~ms (D2D copy) | yes (`remerge_student_lora`) | **自更新** — every cycle |
| **C2. QA-LoRA zero-point merge into 4-bit** | **no** | ~0 (lossless by construction) | seconds (per-layer zero-point update) | partial (new merge kernel) | **升级** — periodic |
| **C3. dense-merge + TurboQuant/Marlin requant** | **yes** (current path needs it) | small (PTQ error each cycle) | tens of s (full requant pass) | ✅ (TurboQuant exists) | **升级** fallback |
| **C4. QLoRA NF4 dequant-merge-requant** | yes | small, compounds per cycle | tens of s | no (NF4 not in tree) | not recommended |

**Verdict**: C1 is the whole 自更新 cadence and needs *nothing new* on a bf16 base. The 升级
cadence is C2 (QA-LoRA, clean) vs C3 (dense-merge + existing TurboQuant, pragmatic-first). C3
reuses kernels that already exist and is the obvious bring-up; C2 is the end-state that avoids
the bf16-base cache and per-cycle PTQ drift. **流式自动量化 = C2 or C3 run on a schedule, gated
by the correct-inference gate (needle ladder) each cycle.**

### Axis D — 训推一体 process shape

| Option | Adapter swap latency | Memory (shared base?) | Hot-swap safety | Matches current code | Complexity |
|---|---|---|---|---|---|
| **D1. Single-process shared TensorStore** — current | µs–ms (in-mem re-merge) | ✅ one base | needs swap-at-step-boundary guard | ✅ (`InProcessTeacher`+`InferStudent` already this) | low |
| **D2. Async worker boundary** (serve thread + train thread, double-buffered adapter) | ms (pointer flip) | ✅ one base | ✅ lock-free double-buffer | partial | medium |
| **D3. Subprocess train + IPC adapter push** | 10–100 ms (disk/IPC) | ❌ duplicate base | ✅ isolated | no | high |

**Verdict**: D1 is already the shape — teacher, student-rollout, and KL backward all run in one
process over one `TensorStore`. SOPD adds the self-teacher and the loss; it does **not** need a
new process boundary. D2 is the natural evolution once serving and training must run *concurrently*
(serve foreground, learn in idle) — but that is an optimization, not a bring-up requirement. D3 is
rejected (duplicates the base, the exact memory we're protecting). **This axis is where ARLE's
moat lives** — see §4.

### Axis E — Device / backend for the train loop

| Option | Op coverage | Device-resident | AIPC target? | Notes |
|---|---|---|---|---|
| **E1. CUDA autograd** | full | yes | cloud/pod, dGPU laptop | most mature; the pod/upgrade tier |
| **E2. Metal (MLX) autograd** | most ops; M5.3b gaps | M5.3a done, some CPU-readback | **the M-series AIPC target** | **target, not wired today** — OPD CLI / `build_opd_store` are CUDA/CPU only, `InferStudent`/LoRA hot-swap are CUDA-gated; the Metal inline self-train loop is an unstarted port (§1.3) |
| **E3. CPU autograd** | full (reference) | n/a | ❌ too slow | correctness reference only |

**Hard scope fact**: there is **no HIP/Vulkan autograd backend**. `infer-hip`/`infer-vulkan` are
inference-only AIPC lanes. So the AIPC *self-training* device is **Metal (Apple Silicon) or CUDA
(NVIDIA laptop/pod)** — a Ryzen-AI / Radeon AIPC can *serve* (HIP/Vulkan) but cannot *self-train*
in-tree today. This is a real boundary for the budget matrix (§3).

### Axis F — Skills data engine (what generates training prompts)

| Option | Data freshness | On-policy? | Reward-hack risk | Infra cost | OPD-license fit |
|---|---|---|---|---|---|
| **F1. Static curated jsonl** — current | stale | no | none | zero (exists) | ✅ |
| **F2. Skill-replay**: capture real agent tool-use traces → prompts | fresh | partial | low | medium (trace capture) | ✅ |
| **F3. Self-generated curriculum** (model proposes tasks) | fresh | yes | medium (drift) | medium | ✅ if distilled |
| **F4. Verifier-gated skill rollouts** (math/code/tool verifiers) | fresh | yes | medium (verifier gaming) | high (verifiers) | ✅ with A2 |

**Verdict**: F1 is the bring-up (the loop is the hard part, not the data). F2 ("skills" =
replaying what the agent actually did) is the truest match to ckl's vision and the natural second
step. F4 pairs with A2 (best-of-N) when a verifier exists. Start F1, design the loop so F2 is a
data-source swap, not a rewrite.

### Axis G — Update timing / rollout coupling (ckl correction 1: "update fires at rollout time")

When does the optimizer step fire relative to the rollout? This is the axis ckl's first
correction added — and it changes which **signal source (Axis A)** is even compatible.

| Option | When the update fires | On-policy staleness | KV-cache validity | Speed cost | Stability | Impl cost |
|---|---|---|---|---|---|---|
| G1. Offline batch *(rejected by ckl)* | after N rollouts, separate trainer run | stale (data ≠ current policy) | n/a | — | easy | — |
| **G2. Inline-per-rollout** | right after each rollout sequence, same loop | minimal | fresh per sequence **only if the prefix cache is epoch-invalidated** (⚠ verdict) | + recompute *or* fused pass | good (with EMA target) | **low — ≈ `opd_step` + prefix-cache versioning** |
| G3. Streaming per-token (test-time training) | after each token/chunk, *mid-sequence* | zero | **stale within sequence** (early KV computed with pre-update adapter) | highest | needs slow EMA + small lr | high |

**Verdict**: ckl's "rollout 时就走更新" = **G2** (with G3 as the aggressive limit). G2 is
structurally what `opd_step` already is (rollout → backward → step, one tight loop); the
correction is that there is **no separate idle/batch phase** — the serving rollout *drives* the
update inline, so 自更新 is *continuous*, not a discrete cadence. **This reshapes Axis A**:
best-of-N (A2) is inherently batch-y (needs N complete sequences then a filter) and does **not**
fit G2/G3 cleanly, whereas **A1 EMA soft-KL is computable per-token inside the rollout forward** —
so the rollout-time constraint *promotes A1 from "simplest" to "the spine"*, and the EMA's
slow-moving target is precisely the stability mechanism that keeps an online per-rollout update
from diverging (the classic mean-teacher / online-distillation reason for EMA). Two sub-forks fall
out (→ §7): **fused-single-pass** (use the rollout's own logits/activations, kill the recompute,
but lose the no-tape CUDA-graph speed) vs **two-pass-inline** (keep the fast rollout kernel,
recompute+step immediately after each sequence); and, for G3, the **KV-staleness** handling (accept
frozen stale KV à la `frozen_kv_mtp`, refresh naturally as new tokens use the new adapter).

**⚠ Prefix-cache staleness (correctness, not perf) — surfaced by Codex review.** "Fresh per
sequence" holds *only* if served prefix KV is invalidated on each accepted adapter update.
`enable_prefix_cache` is **default-on** (`infer-core/src/lib.rs:107`) and `RadixCache` is keyed by
**token blocks only** — there is **no adapter-epoch/version key** (`infer-core/src/radix.rs`).
`sync_lora_from_store` / `remerge_student_lora` mutate q/v weights **in place**, so after an inline
update a *later* request sharing a token prefix would reuse KV computed under the *previous* adapter
epoch — silently serving stale-policy KV. The rollout-time loop therefore **must** either (a)
tag/version prefix pages with an adapter epoch and treat epoch-mismatch as a cache miss, or (b)
disable prefix reuse across adapter epochs. This is a **required** line item for G2/G3, not an
optimization — it joins the rollback/mutated-state enumeration (plan doc §Mutated state). It is the
strongest reason "inline at rollout time" is *not* free: every accepted update has a KV-cache cost.

### Axis H — Base coupling (ckl correction 2: "base-agnostic; gate on hardware budget")

| Option | Where the loop lives | Per-model work | Matches unified-abstraction principle | Today |
|---|---|---|---|---|
| **H1. Seam-level unified loop** (base-agnostic) | `infer-core`/`infer-seam`; each model plugs an adapter-sync trait | one small trait impl per base | ✅ (mirrors paged-KV / batched-decode being model-neutral) | aspiration |
| H2. Per-model special path | duplicated per base | high | ❌ (the DSv4-special anti-pattern) | current sync is **Qwen3.5-only** |

**Verdict**: base-agnostic per ckl → **H1**. The rollout-time update loop is an *engine capability*
that sits above the seam, model-neutral; each base plugs an adapter-sync (mirroring how models plug
paged-KV adapters; `feedback_unified_abstraction_not_per_model`). Today only Qwen3.5 has the
cached-base re-merge wired (`remerge_student_lora`, AttentionQv-locked) — so "base-agnostic" =
**generalize that sync into a seam-level capability** (the real gap H1 names). Consequence for the
budget question: it becomes **budget-first, base-second** — pick the largest base the hardware
budget allows; the loop is identical regardless of base (§3 reoriented).

---

## 3. Budget × hardware — base-agnostic, sized by hardware budget

**Orientation (ckl correction 2): budget-first, base-second.** The loop is base-independent
(Axis H), so the real input is the hardware budget — you pick the largest base that fits and the
*same* rollout-time self-update runs. **§3.2 is the budget-first, base-agnostic answer to
"只看硬件预算"; §3.1 is the per-model backing detail** behind those param-class rows.

**Memory model (matches `arle train estimate-memory`):**

```
peak ≈ base_weights(quant)              # frozen, dominates
     + adapter_fp + grads + AdamW(m,v)  # trainable-only → MB-scale for LoRA
     + activation_floor(batch,seq)      # tape; recompute keeps it modest
     + rollout_KV(ctx)                  # on-policy generation, shared engine
     [+ EMA/self-teacher buffer]        # A1 only: 0 (reuse base) … 1× base (fp EMA)
```

The trainable block is tiny. **Adapter param count** (LoRA on q,v; A·B with rank r):
per layer ≈ `r·(H + q_out) + r·(H + kv_out)`. Worked examples (r=16):

- Qwen3-0.6B (H=1024, q_out=2048, kv_out=1024, 28 layers, all-dense q/v): ≈ **2.3M params** →
  adapter+grad+AdamW(m,v) in fp32 ≈ **37 MB**.
- Qwen3.5-0.8B **hybrid** (only 6 full-attn layers carry q/v, AttentionQv-locked): ≈ **0.5M
  params** → ≈ **8 MB**. (This is why the trainable block is a rounding error.)

So **self-training peak ≈ inference peak × ~1.1–1.5**, dominated by the quantized base + KV.
That is the headline: *if a device can serve the model, it can almost-certainly self-train its
adapter on it.*

### 3.1 Per-model backing detail (analytical estimates — validate with `arle train estimate-memory`)

> ⚠️ **These are back-of-envelope estimates (hypothesis per §0), not measured.** 4-bit base ≈
> params×0.55 B (incl. scales/zeros). KV @4k ctx, bf16. Activation floor for batch=1, seq≈512
> with recompute. Run `arle train estimate-memory --model <id> --lora-rank 16 --batch 1`
> to convert any row to evidence before committing. *(The estimator reports a built-in LoRA count
> from `--lora-rank` and an activation floor from `--batch`; there is **no** `--target`/`--seq`
> flag today — a Q/V-only vs all-linear split needs that flag added first. Validation pre-req.)*

| Model | Params | 4-bit base | KV @4k | Trainable blk (Qv r16) | Activation (b1,s512) | **Self-train peak** | Min HW (self-train) | Min HW (serve only) |
|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen3-0.6B | 0.6B | ~0.35 GB | ~0.45 GB | ~0.04 GB | ~0.3 GB | **~1.2 GB** | 8 GB (phone-class) | 4 GB |
| Qwen3.5-0.8B (hybrid) | 0.8B | ~0.45 GB | ~0.4 GB | ~0.01 GB | ~0.3 GB | **~1.3 GB** | 8 GB | 4 GB |
| Qwen3-4B | 4B | ~2.2 GB | ~0.9 GB | ~0.1 GB | ~0.6 GB | **~4.0 GB** | 8 GB (tight) / 16 GB | 8 GB |
| Qwen3-8B | 8B | ~4.5 GB | ~1.2 GB | ~0.15 GB | ~0.8 GB | **~7.0 GB** | 16 GB | 12 GB |
| Qwen3.5-30B-A3B (MoE) | 30B tot / 3B act | ~16.5 GB | ~0.6 GB (A3B) | ~0.1 GB | ~0.6 GB | **~18–20 GB** | 24 GB (tight) / 36–64 GB | 24 GB |
| Qwen3.6-35B-A3B (MoE, Metal canonical) | 35B tot / 3B act | ~19 GB | ~0.7 GB | ~0.1 GB | ~0.7 GB | **~22–24 GB** | 36–64 GB unified | 24–32 GB |
| DSv4-Flash | huge MoE | multi-GPU | — | — | — | **8×H20 (TP8/EP8)** | cloud only | 8×H20 |

### 3.2 Budget-first, base-agnostic (the answer to "只看硬件预算")

The loop is base-independent (Axis H), so read the table by **budget**, not by model. Rows are a
**param-class** ("the largest base that self-trains @ rollout"), not a specific model — pick any
base in that class. Same rollout-time self-update (§Axis G) regardless.

| HW budget | Train device | Largest base class self-training @ rollout | Cadence available |
|---|---|---|---|
| **8 GB** unified | Metal | ≤4B-class (0.6 / 0.8 / 4B-tight) | 自更新 inline |
| **16 GB** unified | Metal / CUDA | ≤8B-class | 自更新 inline |
| **24 GB** | CUDA / Metal | ≤30B-A3B MoE-class (tight) | 自更新 inline; 升级 tight |
| **36–128 GB** unified | Metal (Apple Silicon) | 35B-A3B-class comfortably | 自更新 inline + local 升级 |
| **8×H20** pod | CUDA TP8/EP8 | DSv4-class | cloud teacher / 升级 (requant) tier |

**Budget reading**: the whole small-to-mid class (≤8B) self-trains on commodity 8–16 GB AIPC
hardware *because LoRA-only makes the trainable block ~MB and the self-teacher (A1) adds no second
base* — so **self-train fits wherever serve fits** (peak ≈ inference × ~1.1–1.5). The MoE 30–35B
class needs a 24 GB dGPU or a 36 GB+ unified-memory Mac.

**Base-agnostic caveat (Axis E):** a Ryzen-AI / Radeon AIPC can *serve* any class (HIP/Vulkan) but
cannot *self-train* in-tree — no HIP/Vulkan autograd backend exists. Base-agnostic on the train
side still means a **Metal or CUDA** train device (out of scope to change; master-strategy
DEFER-until-Phase-3). **And today only CUDA actually runs the inline loop**: the Metal rows above
are the *target*, not a runnable lane — the OPD driver (`build_opd_store`, `InferStudent`) is
CUDA-gated (§1.3), so M-series self-train is an unstarted port. The budget numbers hold; the Metal
*wiring* does not exist at this commit.

---

## 4. 训推一体 (train-infer unified) — the framing and the moat

**It already half-exists, and that is the differentiator.** ARLE's OPD teacher *is* the serving
engine (`InProcessTeacher` calls `engine.forward_token_logits`); the student rollout *is* the
serving engine (infer-rollout default, 4.99×). One process, one `TensorStore`, one KV pool, one
seam (`BackendExecutor`/`KvPool`). SOPD's self-teacher collapses even the teacher into the same
weights.

What competitors do, and why the unified shape is a moat:

| System | Train | Infer | Unified? | On-device self-train? |
|---|---|---|---|---|
| **Apple AFM (iOS 27)** | on-device LoRA on quantized 3B base, adapter swap, data stays local | CoreML/AFM runtime | adapters swap into infer | ✅ (but no self-distillation loop; Apple's signal is supervised) |
| HF TRL / GKD | separate training stack | separate (vLLM) | ❌ | ❌ |
| Unsloth / MLX-LM LoRA | fast LoRA FT | separate inference | ❌ (export then load) | partial (manual) |
| verl / SGLang-RL | RL trainer | bolts inference on for rollouts | heavy multi-process | ❌ (datacenter) |
| **ARLE SOPD (target)** | OPD/self-distill on adapter | same engine produces rollouts AND teacher logits | ✅ **single process, single weight store** | ✅ **+ self-distillation loop + 升级 requant** |

**The moat is not "we can fine-tune on-device" (Apple ships that).** It is: *one device-neutral
runtime that serves, generates its own on-policy training data, computes the teacher signal from
the same weights, trains the adapter, hot-swaps it live, and periodically re-quantizes the base —
with no second model and no external service.* The serving path's speed (CUDA-graph + paged-KV +
the fast teacher forward) is what makes the rollout/teacher cost negligible, which is precisely
why master-strategy-v2 lists OPD as the one training axis where ARLE's runtime authority is
structurally differentiating.

**ckl's two corrections sharpen the moat into something none of the above can state:**
- **Update at rollout time (Axis G)** makes 训推一体 *literal*: the rollout pass and the update are
  the same act, not a serve-phase followed by a train-phase. The model improves *as it serves*,
  token-stream by token-stream — maximally on-policy by construction (zero data↔update staleness).
  Apple's on-device pipeline is supervised + offline-trained adapters; verl/TRL collect rollouts
  then train in a separate process. "Update fires on the rollout path" is the line none of them
  cross.
- **Base-agnostic seam-level loop (Axis H)** makes it a *runtime capability*, not a model feature:
  the same online-update loop runs on whatever base fits the budget (§3.2), each base plugging one
  adapter-sync trait — exactly how paged-KV/batched-decode are model-neutral in the seam today.
  The moat is the *runtime that does this for any base*, not a single fine-tuned model.

---

## 5. The ceiling — extreme state and final effect

**Extreme state (what "world-top-tier AIPC self-training" looks like at the limit):**

> A laptop/Mac runs one ARLE process on whatever base fits its memory budget (§3.2 — base-
> agnostic). As it **serves**, every rollout *is* an update: the engine produces an on-policy
> rollout, computes the EMA self-teacher signal (A1) from a separate slow-EMA **adapter** snapshot
> scored right after the rollout (identical weights would give KL=0 — the teacher must *lag* the
> student), and distills the delta into the rank-r adapter **right there on the rollout path**
> (Axis G) — the
> better adapter is already live, no separate train run, no idle batch. The model gets measurably
> better at the user's own skills *while being used*, with the needle gate running periodically to
> snap the adapter back to the last-good snapshot if a window regresses. Periodically the accepted
> adapter **merges into the quantized base and re-quantizes (升级, C2/C3 = 流式自动量化)**, resetting
> the adapter for the next window. **No cloud, no teacher model, no data leaves the device — and
> the same loop runs on a 0.6B phone-class base or a 35B-A3B MoE, sized only by hardware budget.**

**Final effect, concretely:**
- **自更新 (now continuous, at rollout time — Axis G)**: no discrete cycle; the live adapter
  improves with each served rollout, maximally on-policy, trivially reversible (drop the adapter /
  revert to the EMA-anchored last-good snapshot).
- **升级 cadence**: periodic; base re-quantized; permanent capability gain folded in; gated by
  correct-inference each cycle.
- **Budget (base-agnostic, §3.2)**: ≤8B-class self-improves-while-serving on 8–16 GB consumer
  hardware; 35B-A3B-class on a 36 GB+ Mac — pick the base by budget, the loop is identical.
- **Versus the SOTA anchor (Apple AFM)**: matches on-device LoRA-on-quantized-base + adapter swap
  + privacy; **adds** the self-distillation improvement loop and the merge-then-requant upgrade
  cadence in a single train-infer-unified runtime — the parts Apple's supervised, adapter-only
  pipeline does not have.

**Honest ceiling caveats (§0 — don't oversell):**
- Self-distillation improves *calibration, consistency, and skill-specific behavior*; it does
  **not** add knowledge the base never had (the 322× pretrain-gap argument still holds — this is
  distillation, not pretraining).
- Quality claims < 5pp on small evals need multi-seed ≥5 + Wilson 95% CI (2026-05-28 rule); the
  U-curve (valley-then-recovery) trajectory applies.
- Reward-hack / mode-collapse risk rises from A1 → A4/F4; the correct-inference gate is the
  backstop, but a verifier (A2/F4) needs its own anti-gaming audit.

---

## 6. Simplest viable composition (the convergence — for ckl to accept/amend, NOT yet a plan)

Reading the "simplest-viable" flag off each axis gives one coherent, low-new-surface composition.
This is a **recommendation to decide on**, not an implementation plan:

| Axis | Simplest-viable pick | New surface |
|---|---|---|
| A. Signal | **A1 EMA self-teacher** (soft KL), reuse `backward_chunked_kl_rollout` — *promoted to spine by Axis G* | EMA snapshot buffer + teacher=snapshot wiring |
| B. PEFT | **B1 vanilla LoRA q/v** (exists) | none for 自更新 |
| C. Quant | **C1 fp hot-swap** for 自更新 (exists); **C3 dense-merge+TurboQuant** for first 升级 | requant-on-schedule wiring (升级 only) |
| D. Process | **D1 single-process** (exists) | none |
| E. Device | **E2 Metal** (AIPC) / **E1 CUDA** (pod) | M5.3b op coverage for speed (works today, slow) |
| F. Data | **F1 static jsonl** to bring up, design for **F2 skill-replay** | none for bring-up |
| **G. Timing** | **G2 inline-per-rollout** (≈ existing `opd_step` shape); two-pass-inline first, fused later | loss+step fire on the rollout path, not a batch run |
| **H. Base coupling** | **H1 seam-level** design; bring up on the wired model, keep the loop base-neutral | adapter-sync as a seam capability (today Qwen3.5-only) |

**Net new surface for a first 自更新-only SOPD**: (1) an EMA snapshot of the adapter (NOT full
merged weights — keeps the base shared, §7 Q2) as the self-teacher, (2) the loss wiring to distill
student → EMA **inline on each rollout** (G2, reuse `backward_chunked_kl_rollout`), (3) a periodic
needle-gate snapshot/revert. Everything else is existing code. **升级/流式自动量化 is a strictly
later, separable milestone** (C3 first, C2 = QA-LoRA as the clean end-state) and does NOT block the
自更新 bring-up.

**Why this is "simple"**: it adds *one EMA buffer and one loss target* to a loop that already does
on-policy rollout + KL backward + live adapter swap in one process — and ckl's correction 1 means
it reuses the `opd_step` rollout→backward→step shape *as-is* (inline), rather than inventing a
batch/idle scheduler. The hard, novel parts (升级 requant, skill-replay data, verifier-gated A2,
QA-LoRA, Metal op coverage, base-agnostic seam generalization, Ryzen self-train) are all
**deferrable and independently licensable** — none is on the critical path to a working 自更新 demo.
Bring-up uses the model that already has the LoRA-sync wired (Qwen3.5-0.8B, AttentionQv); the
*design* stays base-neutral (H1) so other bases plug in via one trait, not a fork.

---

## 7. Open questions to resolve before any implementation plan

1. **A1 self-teacher target**: EMA of the *base* (pure mean-teacher) vs EMA of the *merged
   student* vs the frozen base itself? Decides whether there's any improvement signal at all at
   step 0 (frozen-base self-distill = zero gradient until the adapter moves).
2. **Does the EMA buffer cost a 2nd base copy?** If the self-teacher is the frozen base + a
   *separate* EMA of the adapter, the base is shared (cheap). If it's an EMA of full merged
   weights, that's a 2nd base — re-introduces the memory the unlock was meant to remove. **This
   is the single most important design decision for the budget story.** *(With Axis G, the EMA-
   of-adapter answer is also the stability anchor for online updates — strong reason to pick it.)*
3. **Axis G granularity — per-sequence (G2) vs per-token (G3)?** G2 (update after each rollout
   sequence) is the safe default and matches `opd_step`. G3 (test-time training, adapter changes
   mid-sequence) is the aggressive limit and needs the KV-staleness decision (Q4). **Which does
   ckl's "rollout 时就走更新" mean?** — recommend G2 first, G3 as a gated research follow-on.
4. **Axis G pass structure — fused vs two-pass-inline?** Fused = use the rollout forward's own
   logits/activations for the gradient (one pass, but the infer rollout runs no autograd tape +
   CUDA-graph, so this needs the engine to expose activations / a custom backward → loses graph
   speed). Two-pass-inline = keep the fast no-tape rollout, then recompute+step immediately
   (reuses today's path). **Bench which wins; recommend two-pass-inline for bring-up.**
   *(If G3 per-token: also decide KV-staleness — accept frozen stale KV à la `frozen_kv_mtp` and
   let it refresh as new tokens use the new adapter, vs recompute KV (expensive).)*
5. **升级 first cut**: C3 (dense-merge + TurboQuant, reuses kernels, needs bf16 base cache) vs C2
   (QA-LoRA, no bf16 cache, new merge kernel). Pragmatic-first says C3; clean end-state says C2.
6. **Correct-inference gate cadence**: with G2/G3 the update is continuous, so the needle ladder
   can't run every step — run it on a *window* (every N rollouts / wall-clock) and snapshot/revert
   the adapter to the last-good on regress. What N? (Cost vs safety trade.)
7. **Axis H — generalize the adapter-sync to a seam capability?** Base-agnostic (H1) needs the
   Qwen3.5-only `remerge_student_lora`/cached-base re-merge lifted into an `infer-seam` trait each
   base implements. Bring up on Qwen3.5-0.8B (already wired) but scope the trait now so it's not a
   later fork. *(Also: should the runtime auto-select the largest base that fits the detected
   hardware budget — §3.2 — as a first-class feature?)*
8. **Metal op coverage (M5.3b)**: which ops still CPU-readback in the student backward, and is the
   per-rollout latency acceptable for an *inline* (Axis G) loop on M-series? (Bench, don't assume.)
9. **Validate §3 numbers**: run `arle train estimate-memory` per param-class to convert the
   budget-first matrix (§3.2) from estimate to evidence before committing hardware claims.
10. **Prefix-cache epoch invalidation (Axis G correctness, NOT optional)**: an inline adapter
    update makes any served prefix KV from the previous epoch stale, but `RadixCache` is token-keyed
    with no version (§Axis G ⚠). Decide the mechanism: (a) tag prefix pages with an adapter epoch
    and treat epoch-mismatch as a miss (fine-grained, keeps cross-request reuse within an epoch), vs
    (b) flush the whole prefix cache on each accepted update (simplest, but throws away reuse), vs
    (c) disable prefix reuse entirely while the inline loop is active. Recommend (a) for production,
    (b) for bring-up. This is on the critical path for G2 *correctness*, alongside Q1/Q3.

**Recommendation**: resolve Q1/Q2/Q3 **and Q10** (they define the memory + signal + online-
granularity + cache-correctness story),
then a small plan for **inline (G2) 自更新-only on Qwen3.5-0.8B** (the model with existing
AttentionQv LoRA sync), designed base-neutral (H1), is enough to prove the loop. Defer 升级,
per-token G3, and the seam generalization to their own milestones.

---

## References (literature anchors)

- ReST / ReST-EM — iterative generate→filter→fine-tune self-training (ICLR 2024 "ReST meets ReAct").
- "Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs" (arXiv 2601.18734, 2026).
- "A Model Can Help Itself: Reward-Free Self-Training for LLM Reasoning" (arXiv 2510.18814).
- QA-LoRA: Quantization-Aware Low-Rank Adaptation (ICLR 2024, arXiv 2309.14717) — lossless
  adapter→quantized-base merge via zero-point update. CLoQ (2025), IntLoRA (2024), LoTA-QAF
  (2505.18724), LoraQuant (2510.26690).
- Apple Intelligence Foundation Language Models Tech Report 2025 (arXiv 2507.13575) — on-device
  3B, native LoRA, dynamic adapter load/cache/swap, on-device LoRA training (iOS 27), accuracy-
  recovery adapters on the quantized base.
- MLX-LM / Unsloth / mlx-tune — on-device LoRA/QLoRA on Apple Silicon unified memory.

### Internal cross-refs
- [`docs/projects/2026-05-18-opd-only-pivot.md`](../projects/2026-05-18-opd-only-pivot.md) — OPD-only license, GRPO retired.
- [`docs/projects/2026-06-10-arle-master-strategy-v2.md`](../projects/2026-06-10-arle-master-strategy-v2.md) — Phase 3 #3 (OPD-GPU) + #5 (AIPC); D4 strict serialization.
- [`docs/plans/2026-05-29-opd-student-rollout-via-infer.md`](../plans/2026-05-29-opd-student-rollout-via-infer.md) — infer-rollout default; AttentionQv lock; B1.5 cached-base re-merge.
- [`docs/plans/2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md`](../plans/2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md) — teacher+student co-residency OOM (the line item self-teacher removes).
- [`crates/autograd/AGENTS.md`](../../crates/autograd/AGENTS.md) — CPU+Metal+CUDA only; M5.3b op coverage.
