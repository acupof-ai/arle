# Plan — Agentic OPD showcase: a real 27B capability curve

> Status: Active — 2026-07-03 (phase 1 closed same day: toy-capability lane
> KILLED, harness + stability run SHIPPED; phase 2 = teacher-rescue on real
> SWE-Pro, below) · Driver: ckl ("真实的能提升 27B 模型的能力，拿到数据和曲线")

**Phase-1 verdict (2026-07-03, measured).** The synthetic-corpus capability
lane is dead: five escalations (easy → hard-v1 → hard-v2 neutral-docstrings +
gold scenery → turn budgets 3/2/1) all landed outside the 10–40% band —
baselines 8/8, 24/24, 24/24, 22/24, 20/24, 0/24. Root cause + rule:
[errors/2026-07-03-agent-opd-toy-corpus-saturation-kill.md](../experience/errors/2026-07-03-agent-opd-toy-corpus-saturation-kill.md).
What shipped anyway: the full harness (corpus gen + self-check, run script,
curve plotter, eval channel), two real runtime fixes found under load (tape
footprint 2.54× underestimate → OOM at seq≈1350, sandbox `__pycache__`
staleness), the accept-wall nudges, and a 16-round stability run (wins entry).
The capability curve moves to phase 2.

**Goal.** Turn the agent-OPD infra win (438s → 11.75s/round, 37.3×,
[wins/2026-07-03-opd-full20-curve.md](../experience/wins/2026-07-03-opd-full20-curve.md))
into a **capability** result: the 27B Qwen3.6-FP8 student measurably improves at
**held-out** agentic coding tasks via execution-rewarded on-policy training, on
ONE H20 GPU, in one overnight run. Headline artifact = a baseline→round-N
held-out pass-rate curve for the README OPD section, produced by one script.

Supersedes the eval-surface portion of
[2026-06-16-agentic-ropd-35b.md](2026-06-16-agentic-ropd-35b.md) (the E1–E3
"net-new eval surface" is now built: `--eval-dataset` held-out pass in
`crates/cli/src/train_cli.rs` `run_agent_opd_eval_pass`).

---

## What exists (verified in code, 2026-07-03 HEAD)

- **Loop**: `arle train agent-opd` — rollout (read/write/replace/bash agent in a
  per-task git sandbox) → execution reward (`git diff` non-empty AND hidden
  `test_patch` + `fail_to_pass` pytest exit-0, `sandbox.rs:score_workdir`) →
  masked single-trajectory CE writeback on accepted trajectories → per-round
  LoRA sync into the rollout engine. 11.75s/round steady-state on the toy shape.
- **Eval channel**: `--eval-dataset` runs a greedy eval-only pass at round 0
  (baseline) and every `--eval-every` rounds; dumps
  `eval_round_{label}.jsonl` (per-task + aggregate pass-rate) to
  `--eval-out-dir`; logs `held-out pass_rate=… (baseline=…, Δ=…)`.
- **Task schema**: SWE-bench-Pro JSONL (`swe_dataset.rs::SweTask`) + staged
  plain trees at `<staged-root>/<instance_id>/` (git-init happens in
  `boot_workdir`; `test_patch` is `git apply`-ed at scoring time).
- **Licensed defaults** from the optimization campaign: `--share-frozen-base`
  (default on), LoRA attention-qv r16/α32, adaptive checkpointing,
  chunk-parallel LA backward.

## What is missing (the gaps this plan fills or defers)

| # | Gap | Disposition |
|---|---|---|
| G1 | **No task corpus in-repo** — the 20-round curve ran on 1 ad-hoc toy task; no generator, no staging tool | **Build now**: `scripts/gen_agent_opd_tasks.py` |
| G2 | **Accept-rate bootstrap wall on real SWE-Pro** — decoded 0-accept: explore-forever (0 edit calls in 30 turns), instant-stop, tool-schema misuse (`read` called with `command`) ([errors/2026-06-29](../experience/errors/2026-06-29-agent-opd-accept-wall-is-no-edit-exploration-not-wrong-dir.md)) | **Mitigate now** (G3+G4+corpus difficulty); teacher-rescue deferred (G6) |
| G3 | **Tool-schema errors don't name the schema** — `read` with no `path` returns "ERROR: … is a directory"; unknown tool doesn't list valid tools (`sandbox.rs` `execute()`) | **Fix now** (flagged-not-fixed in the 06-29 entry) |
| G4 | Edit-pressure: system prompt says "must edit" but the decoded rollout wandered 30 turns | **Prompt-only nudge now** (edit-by-turn-N line in `agent_system_prompt`); mid-rollout injected reminder in `agent::run_turn` deferred |
| G5 | `--writeback-cap` truncates head-of-list — over-trains early tasks, no per-task dedup of the best-of-N accepts | Defer; note in wins entry if it binds |
| G6 | **No teacher in the agentic loop** — current writeback is execution-filtered self-CE (RFT). The "OPD" completion = think-on teacher (or DSv4-Flash via `InferTeacher`) rescue on 0-accept tasks + masked CE/KL on teacher trajectories. Solves bootstrap when baseline = 0 | **Phase 2 — now the licensed path** (see below) |
| G7 | Train-side metrics are stderr lines only (eval side already dumps JSONL) | Parse the stable log lines in the plot script; no Rust change |

---

## Corpus design (G1) — synthetic bug-fix suite, SWE-Pro schema

Real SWE-Pro is the wrong showcase substrate today: baseline ≈ 0 accepts
(G2 ⇒ no RFT gradient), 100MB+ sandbox copies per sample (buries the 12s/round
win), uncontrollable difficulty. Synthetic tasks in the SAME schema keep the
whole harness honest (same loader, sandbox, scorer) and scale up to real
SWE-Pro later with zero harness change.

`scripts/gen_agent_opd_tasks.py --out <dir> --seed 0`:

- **Repos**: ~18 small self-contained Python packages (3–5 modules, 60–150
  lines each; inventory/date/text/graph/matrix/config-parser… domains), each
  with ONE injected bug from an archetype pool: inverted comparison, off-by-one,
  wrong default, missing None-guard, swapped args, wrong operator, early
  return, mutable default, boundary condition, string-format slot.
- **problem_statement**: issue-style symptom with a concrete repro
  (input → expected vs actual), names the module, NOT the function/fix.
- **test_patch**: unified new-file diff adding `tests/test_hidden_<slug>.py`;
  `fail_to_pass` = its pytest node ids. No PYTHONPATH needed (`python -m
  pytest` puts cwd on `sys.path`).
- **gold_patch** column (ignored by the Rust loader) + **`--self-check`**: for
  every task, staged tree must FAIL the hidden tests and gold-patched tree must
  PASS. Runs anywhere with python3+pytest — the corpus correctness gate.
- **Split**: disjoint task sets — train 12 / eval 24 (eval n=24 ⇒ ~4pp
  resolution; the claim gate needs a big Δ, not fine resolution).
- **Difficulty knob** (`--difficulty easy|medium`): distractor-file count +
  symptom explicitness. Calibrated from the smoke baseline (target 10–40%).

## Run design

Student 27B Qwen3.6-FP8, single H20, `--share-frozen-base`, LoRA
`attention-qv` r16/α32, `--lora-skip-experts` semantics already covered by the
qv target set. Rollout temp 1.0, eval greedy (temp 0).

**Smoke-then-size** (no wall-clock promises before measurement —
per-rollout cost on multi-turn small-repo tasks is unmeasured; the toy round's
5.08s rollout doesn't transfer): smoke = 2 rounds, `--task-limit 4`,
`--eval-n 8`, `--eval-every 1`. Measures per-rollout wall, per-eval-task wall,
baseline pass-rate. Then size the full run to ≤ ~8h; initial full shape:
rounds=16, 12 train tasks × `--samples-per-prompt 2..4`, `--max-turns 8`,
`--max-tokens 768`, `--eval-every 2`, `--writeback-cap 8`, fixed
`--rollout-seed`.

**Non-determinism envelope**: 3× same-config greedy baseline evals before
training (MoE non-determinism; correct-inference framing, not byte-identity).
The envelope width is the noise floor the curve must clear.

**License-or-kill gate** for the README claim: final held-out pass-rate −
baseline ≥ **+15pp** AND above the baseline envelope max. Baseline stuck at 0
⇒ bootstrap gap confirmed ⇒ KILL the RFT-only claim, G6 becomes the licensed
next step (that is itself a publishable finding, documented in errors/).

## Phase 2 — cc-as-harness (ckl pivot 2026-07-03) + teacher-rescue on real SWE-Pro

**Pivot: Claude Code is the rollout harness; ARLE serves the student behind
an Anthropic `/v1/messages` adapter** (shipped 2026-07-03, `crates/
infer-server/src/anthropic.rs`; live CC verified locally and on the pod).
First pod probe (n=3 real ansible instances, 27B FP8, 1×H20): the same model
that got 0 edits in the in-house loop **edited 2/3 repos under CC**, pass@1
0/3 —
[wins/2026-07-04](../experience/wins/2026-07-04-cc-as-harness-pod-e2e-baseline-probe.md).
Harness = capability multiplier; the remaining correctness gap is the OPD
target. Next: ① stage a runnable real corpus (≥12+12, staging gate:
base-FAILS-not-ERRORS / gold-PASSES on the pod's python3.12); ② curve =
CC-driven pass@1 baseline → distill CC trajectories (re-render + tokenize →
existing masked-CE writeback) → CC-driven pass@1 again; ③ the in-house-loop
rescue features below remain the fallback lane.

Why this is the curve: at real-repo scale the 27B has a decoded 0-accept wall
(explore-forever, never edits) — the regime where RFT starves (0 accepts ⇒ no
gradient) and where a runtime-served teacher is the ARLE-differentiated move.
This is also what makes the loop *OPD* rather than plain rejection sampling.

1. **Corpus staging** (prerequisite): `scripts/stage_swe_pro.py` — pick ≥12
   train + 12 eval python SWE-Pro instances (pytest-runnable, bounded repo
   size), clone at `base_commit` under `staged/<instance_id>/`, run
   `before_repo_set_cmd`, verify each with the same base-FAILS/gold-PASSES
   self-check. Pod already holds 4 ansible instances (269 MB) from 06-27..30.
2. **Teacher = the same 27B engine, thinking ON** (zero extra VRAM; precedent:
   the 2026-06-20 4B think-on BFCL win). Rescue pass in
   `agent_opd.rs::run_agentic_opd_round`: tasks with 0 student accepts get M=2
   teacher rollouts (prompts without the `/no_think` soft-switch, temp ~0.7,
   `--teacher-max-tokens` ≥10K for thinking budget); passing teacher
   trajectories enter the SAME masked-CE writeback. Open decision to settle by
   reading the 06-20 entry first: train the student think-on too (no mask
   mismatch) vs mask `<think>` spans out of the CE targets.
3. **Curve**: held-out real-instance pass-rate, baseline → round N, with the
   same envelope protocol. Secondary read: student accept-rate on train tasks
   (does the rescued student start accepting on-policy? — the flywheel
   igniting is the story).
4. Escalation if the same-27B teacher also 0-accepts everything: DSv4-Flash
   teacher via `InferTeacher` (time-share the device per
   `offload_engine_weights`), at real extra cost — own license.

## Deliverables

1. This plan.
2. `scripts/gen_agent_opd_tasks.py` (+ self-check).
3. `scripts/agent_opd_curve.sh <label>` — gen → train+eval → plot, one command,
   env-overridable shape (SMOKE=1 for the smoke shape).
4. `scripts/plot_agent_opd_curve.py` — eval dumps + train log → `curve.json` +
   PNG (pass-rate + train accept-rate + train loss vs round).
5. G3 fix + unit test in `crates/train/src/sandbox.rs`; G4 prompt line in
   `crates/train/src/swe_dataset.rs`.
6. Pod runs (smoke → full) → wins entry (bench-spec sections) → README +
   README.zh-CN OPD section update with the curve → CHANGELOG line.
