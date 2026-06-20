# OPD ceiling + elicitation on a 27B dense student — long-range plan

**Goal (ckl, 2026-06-21):** verify the *ceiling* of On-Policy Distillation and
whether it can *elicit* (激发) latent capability in a student. Headline target:
**DeepSeek-V4-Flash teacher (TP4) → Qwen3.6-27B-FP8 dense student (LoRA)** on 8×H20.
Dense 27B chosen to dodge the MoE-router autograd problems that dogged the 35B-A3B
student. Hard sequencing constraint: **complete plan + code first, GPU only after.**

Success = a capability curve (base → OPD) on a held-out reasoning/agentic benchmark
with CI-separated lift, plus a measured answer to "can the student approach — or
exceed — the teacher, and what gates it."

---

## 1. Industry practice (surveyed 2026-06-21)

| Finding | Source | Consequence for us |
|---|---|---|
| OPD = dense KL-constrained RL; teacher per-token log-ratio is an **implicit reward**; **scaling it >1× pushes the student past the teacher** | Rethinking-OPD (arxiv 2604.13016) | "上限/激发" is real and tunable → add a `--kl-reward-scale` knob |
| **Thinking-pattern alignment DOMINATES teacher strength.** A 75% and a 50% teacher had ~identical effect when reasoning patterns mismatched; mismatch weakens distillation *regardless of the teacher's benchmark advantage* | Rethinking-OPD | **The DSv4→Qwen cross-family gap is the #1 risk, bigger than cross-vocab** |
| RL-post-trained teachers transfer "new knowledge beyond family"; same-pipeline teachers transfer little | Rethinking-OPD | DSv4-Flash (RL-reasoning) is a *good* teacher on this axis; a vanilla Qwen base would be a poor one |
| Recipe: reverse-KL, temp 1.0, **sampled-token OPD** (≈ top-k, cheaper), avoid top-1, ~150 steps, mix teacher-aligned + OOD prompts (else entropy collapse) | Rethinking-OPD + Thinking-Machines (2025) | Matches our existing reverse-KL/`--kl-mask completion`; add reward-scale + prompt-mix |
| OPD is **7–10× fewer steps / 50–100× less compute than RL**, ~70% AIME in ~150 steps | Thinking-Machines | 150-step runs are cheap → many A/B arms feasible |
| Cold-start (SFT on teacher rollouts) raises thinking-pattern overlap when patterns mismatch | Rethinking-OPD | The bridge for the cross-family DSv4→Qwen path |
| **Cross-vocab is an open problem.** ULD = optimal-transport over *sorted* logits (on-policy-compatible). BLD byte-interface = **off-policy only**, inconsistent. MultiLevelOT similar. "No method consistently wins." | ULD 2402.12030 · BLD 2604.07466 · MultiLevelOT 2412.14528 | Cross-vocab token-KL is research-grade; ULD is the only on-policy option; don't expect it to beat same-vocab |

**Net:** the headline DSv4-Flash→Qwen-27B path is doubly hard — cross-vocab *and*
cross-family pattern mismatch (the dominant failure mode). It must be attempted
*after* a same-vocab control establishes the ceiling, or a null result is
unattributable (CLAUDE.md §0 confounder rule).

---

## 2. Current codebase state (Explore map, 2026-06-21)

- **Student archs:** `crates/train/src/qwen35.rs` (Qwen3.5/3.6 only). Dense support is
  free — `qwen35.rs:2552 if cfg.is_moe_layer(layer_idx)` and
  `qwen35_loader.rs:675-690` key on `num_experts`; `num_experts==0` → no MoE layers.
  **Qwen3.6-27B-FP8 (dense) loads with zero new code** (verify with a load smoke).
- **KL loss:** `loss.rs:58-100 kl_distill_loss()` (forward/reverse), `loss.rs:451-456`
  **hard-asserts `student.shape == teacher.shape` incl. vocab dim** → no cross-vocab.
  `--kl-beta` already gives Generalized-JSD; `--kl-mask completion` slices
  `[prompt_len-1, seq_len-1]` (`opd.rs:1329-1358`).
- **Teacher:** in-process Qwen35 (`opd.rs:2722`) or **infer-api** generic
  (`train_cli.rs:1803-1836` → `InferTeacher`, `teacher_infer.rs:740-769` extracts
  `forward_token_logits` *and validates vocab match*). DSv4-Flash is **not** wired as
  a teacher and would fail the vocab-match check.
- **On-policy flow:** student rollout `opd.rs:846 forward_rollout_cached` → same tokens
  to teacher `opd.rs:1436` → KL on the masked range. Correct OPD shape.
- **Vocab fact (evidence, not inference):** the validated 4B math OPD
  (0.518→0.792) ran Qwen3.6-35B-A3B teacher → Qwen3.5-4B student under token-KL,
  which *requires* matching vocab → **Qwen3.5 and Qwen3.6 share vocab 248320.**
- **Reward-scale knob:** does not exist; needed for the "exceed-teacher" experiment.

---

## 3. Strategy — two phases, confounder-isolated

Phase 0 reuses ~all existing code and isolates the *OPD-elicitation* question from the
two confounders (cross-vocab, cross-family). Phase 1 is the headline DSv4-Flash path,
interpreted *against* the Phase-0 baseline.

### Phase 0 — same-vocab ceiling (the clean experiment)
**Qwen3.5-122B-A10B-FP8 teacher → Qwen3.6-27B-FP8 dense student**, token-KL OPD.
Same vocab (248320), same family (consistent thinking patterns = the #1 success
predictor), strong frontier teacher (≫27B dense), RL-post-trained (new knowledge).
Optionally scale the teacher to Qwen3.5-397B-A17B-FP8.

Answers directly: *how far does OPD lift a 27B dense student toward a much stronger
same-family teacher, and can reward-scaling push it past?*

**Tasks (file:line):**
- P0.1 Download `Qwen3.6-27B-FP8` + `Qwen3.5-122B-A10B-FP8` via `oniond` to `/data01/models`.
- P0.2 Load smoke: `qwen35_loader.rs` loads the dense 27B (assert `num_experts==0`,
  no MoE layers built); single forward on a fixed prompt prints finite logits.
- P0.3 Teacher smoke: `infer-api` loads the 122B-A10B MoE at TP2; `teacher_infer.rs`
  `forward_token_logits` returns vocab=248320, vocab-match passes vs the 27B student.
- P0.4 New knob `--kl-reward-scale f32` (default 1.0): scale the teacher log-ratio /
  KL gradient in `loss.rs` reverse-KL path; behind the flag, default byte-identical.
  Unit test: scale=1.0 ≡ current loss; scale=2.0 doubles the grad magnitude.
- P0.5 Prompt corpus: a reasoning set (math + agentic) with teacher-aligned + OOD mix
  (anti entropy-collapse) as `--prompts-file`.
- P0.6 Recipe per research: reverse-KL, `--rollout-temperature 1.0`, sampled-token
  (existing `--teacher-topk`/window), `--kl-mask completion`, ~150 steps, LoRA r32/a64.
- P0.7 Eval harness (reuse `scripts/arle_capability_eval.py` + BFCL/MATH gates) — clean,
  timeout-free, multi-seed per the §0 case-as-fact rule.

### Phase 1 — cross-vocab DSv4-Flash (the headline, research-grade)
**DeepSeek-V4-Flash teacher (TP4) → Qwen3.6-27B-FP8 student**, ULD on-policy +
off-policy cold-start. Tests whether a cross-family frontier reasoning teacher beats
the same-family Phase-0 teacher, or whether the thinking-pattern mismatch kills it.

**Tasks (file:line) — net-new machinery:**
- P1.1 DSv4-Flash as `infer-api` teacher at TP4 (`train_cli.rs` teacher load path);
  relax the vocab-match check to a cross-vocab branch.
- P1.2 **Cross-tokenizer position alignment**: student rollout is *student* tokens;
  teacher must score the same *text* under its own tokenization → decode student
  tokens → re-encode with teacher tokenizer → build a char/byte-span alignment map
  between student positions and teacher positions. New module `train/src/cross_vocab.rs`.
- P1.3 **ULD loss**: `loss.rs` new `uld_loss()` — Wasserstein/OT over *sorted* teacher
  and student logits per aligned position, no vocab-match assert. Behind `--kl-loss uld`.
- P1.4 **Off-policy cold-start**: SFT the student on DSv4-Flash rollouts (decoded →
  re-encoded into Qwen) to raise thinking-pattern overlap before OPD (the research
  bridge). Reuse the existing rollout + an SFT-CE path.
- P1.5 Thinking-pattern overlap metric (token-overlap ratio, the research's success
  predictor) logged per run, so we can attribute a null result to mismatch vs code.
- P1.6 Run vs the Phase-0 baseline; attribute (case-as-fact, decoded outputs).

---

## 4. DAG, critical path, GPU budget

```
P0.1 download ─┬─ P0.2 student smoke ─┐
               └─ P0.3 teacher smoke ─┼─ P0.4 reward-scale ─ P0.6 recipe run ─ P0.7 eval ─► Phase-0 ceiling result
               P0.5 corpus ───────────┘                                                          │
                                                                                                 ▼ (baseline locked)
P1.1 dsv4 teacher ─ P1.2 align ─ P1.3 ULD ─ P1.4 cold-start ─ P1.5 overlap metric ─ P1.6 run ─► Phase-1 headline
```

Critical path = P0.1→P0.3→P0.4→P0.6→P0.7 (Phase-0), then the P1 chain. P0.4 (reward
scale) and P0.5 (corpus) parallelize. Phase-1 P1.2/P1.3 are the long poles.

**GPU budget (8×H20, 96 GB):**
- Phase 0: teacher 122B-A10B-FP8 ≈ TP2 (≈122 GB), student 27B-FP8 autograd ≈ 1–2 GPUs
  (35B-A3B used ~75 GB single-GPU at seq-2048; 27B dense comparable). Total 3–4 GPUs.
  397B teacher → TP4–5 if we scale the teacher.
- Phase 1: DSv4-Flash TP4 (4 GPUs) + student 1–2 + cold-start. Fits 8.
- Dev loop: `scripts/pod_pipeline.sh` (sync→build→verify, GPUs 4-7); ckl granted all 8.

**Build/run:** the proven CUDA env in `pod_pipeline.sh`; teacher via `--teacher-runtime
infer`; profile via `ARLE_OPD_STEP_PROFILE=1` (no `--json`); suffix-detach
`--lora-layer-start` available if the 27B backward is the wall (measured 8.1× at seq-2048).

---

## 5. Decision points for ckl (before code lands)

1. **Phase 0 first?** Research says cross-family DSv4→Qwen may null out on pattern
   mismatch *regardless of DSv4's strength*; the same-vocab Qwen-mega→27B control
   isolates that and reuses all code. Recommend **yes — Phase 0 then Phase 1.**
2. **Phase-0 teacher size:** 122B-A10B (tractable, TP2) vs 397B-A17B (max gap, TP4–5).
   Recommend **122B first**, scale to 397B if the 122B ceiling is clean.
3. **Cross-vocab method for Phase 1:** ULD (on-policy, OT over sorted logits) is the
   only on-policy option; BLD is off-policy. Recommend **ULD**, with cold-start as the
   pattern bridge. Accept that "no method consistently wins" — Phase 1 is exploratory.

Status: plan only. No GPU runs, no code yet — awaiting ckl's call on the three forks.
