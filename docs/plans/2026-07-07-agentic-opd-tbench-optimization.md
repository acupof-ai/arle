# Plan — Agentic-OPD on Terminal-Bench: break the plateau (paper-driven)

> Status: Active — 2026-07-07 · Driver: ckl · Grounds the TB-OPD optimization in
> five Agent-RL deep-reads (ARPO / AEPO / Tmax / OpenThoughts-Agent / Agent-World).
> Baseline: [wins/2026-07-07-terminal-bench-opd-format-distill-lift](../experience/wins/2026-07-07-terminal-bench-opd-format-distill-lift.md).

## Problem

We got a first real agentic-OPD lift — Terminal-Bench pass@1 **20.5% → 25.6%**
by distilling execution-passing terminus trajectories; the gradient was
**output-format conformance** (the 27B could act but couldn't emit parseable
terminus-JSON). But the automated STaR loop **plateaued at 6→7→7** across 3
rounds while the corpus grew 69→216. The format-conformance gradient is
one-shot; something structural caps further self-distillation lift.

## Diagnosis (Tmax, confirmed)

The plateau is a **data problem, not an algorithm problem**. On our fixed
13-task set, once the model learns the format, every remaining task is either
always-pass (reward 1 → zero gradient) or always-fail (reward 0 → no trajectory).
RL/OPD needs a **sweet-spot band** — tasks the model *sometimes* solves — to
carry gradient. Our substrate has no such band by construction. This is exactly
Tmax's thesis: *"the bottleneck is data difficulty calibration, not a smarter
algorithm."*

## Five-paper synthesis

| Paper | One-line | Actionable insight for us |
|---|---|---|
| **Tmax** (2606.23321) | Simple recipe + difficulty-calibrated data → 9B hits 27.2% on TB-2.0 | Compositional data gen over difficulty axes; **soft-filter** all-0/all-1 batches; **FP32 LM head** for train-infer logprob parity (= our train-infer-unified); outcome-only reward, no shaping/KL/SFT-warmup |
| **OpenThoughts-Agent** (2606.24855) | 100+ ablations pricing every data choice | **Task source is the highest-variance stage** (30pp); **Top-4 source mix** optimal; **task-description augmentation HURTS** (negative result — don't rewrite); teacher ≈5pt, traj-filter ≈3pt |
| **ARPO** (2507.19849) | Entropy spikes after tool calls → branch there | LLM is **most uncertain right after tool output** — branch sampling at high-entropy tool-call rounds for step-level exploration; +4-6%, fewer tool calls |
| **AEPO** (2510.14545) | Balance entropy in rollout AND update | (a) high-entropy rollout collapses to 1-3 trajectories → consecutive-branch penalty; (b) clipping kills the high-entropy gradients you want → **stop-gradient clip preserves+rescales**; (c) entropy-aware advantage weighting |
| **Agent-World** (2604.18292) | The environment is the bottleneck; co-evolve it | Self-evolving arena: **diagnose the agent's weakest environments → generate more of those**; build verifiable tasks by tool-dependency graph walks + reverse-engineering task-from-solution |

## Workstreams

### ① Soft-filter the distill corpus — SHIPPED (`c59aab9ca`)
Distil only sweet-spot tasks (`0 < passes < attempts`); drop always-pass
zero-gradient trajectories that flat-lined the STaR curve. One-block change in
`scripts/tbench_opd_loop.sh`. *Leverage: high · Cost: trivial.*

### ② Wider task spread — SHIPPED (`c59aab9ca`)
13 → 28 light/medium TB tasks for a calibrated difficulty range so a sweet-spot
band exists (Tmax difficulty calibration + OpenThoughts source-diversity, minus
augmentation). Prewarmed base images already cover the deps. *Leverage: high ·
Cost: low.* **Running now** (`/host/tbench_opd_v2`).

### ③ Curriculum — diagnose-then-target (Agent-World) — NEXT
After each round, diagnose the model's weakest *sweet-spot* tasks (barely-pass /
just-fail band) and (a) oversample them in the distill corpus, (b) request more
generated tasks of that category (needs ⑤). Turns the flat STaR into a curriculum
that keeps producing gradient. *Depends on ⑤ for the "generate more" half; the
"oversample" half is a loop-side reweight, doable now.*

### ④ Compositional task generator (Tmax §5.1) — SHIPPED (`2d05f33fc`)
`scripts/gen_terminal_tasks.py`: difficulty-calibrated, self-verifying,
TB-compatible tasks over axes {domain × command-complexity × verifier × n-steps},
each with a reference `solution.sh` + `tests/test_outputs.py`, gated by a
Docker-free self-check (solution passes / un-solved fails). Feeds `tb run
--dataset-path` (wired into the loop via `DATASET_PATH`). This is the **root
fix** — an unbounded difficulty-calibrated pool so the sweet-spot band never runs
dry — and the supply side ③ consumes. 40-task pool validated on the pod
(easy/medium/hard 13/14/13, 6 domains, 24/24 self-check). *Leverage: highest.*
**Next: run a loop on `DATASET_PATH=/host/gen_tasks` once ② reports (GPU-serial).**

### ⑤ Entropy-aware rollout + update (ARPO/AEPO) — DESIGN
Our failure taxonomy is entropy-shaped: ~22/31 baseline fails were runaway
reasoning / parse errors — the model over-reasons at the high-entropy moment
after tool output. Two moves, both needing the on-policy/GKD path (`loss.rs::
kl_distill` + `teacher_infer::ApiTeacher`, already built) rather than the current
masked-CE replay:
- **Rollout**: branch sampling at high-entropy tool-call rounds (ARPO); cap
  over-branching with a consecutive-branch penalty (AEPO).
- **Update**: stop-gradient clipping to keep high-entropy token gradients;
  entropy-aware advantage weighting.
*Leverage: high on the exact failure mode · Cost: high (moves OPD from
masked-CE-RFT toward GKD/on-policy).* **Blocked on the GKD wiring + a teacher.**

## Sequencing & success criteria

1. **Now**: ①②③(oversample half) running → does the plateau break? Success =
   pass@1 curve keeps climbing across rounds (not 6→7→7), sweet-spot count > 0
   each round.
2. **Then ④**: generator online → loop uses generated + real tasks → the
   sweet-spot band is unbounded. Success = per-round sweet-spot task count stays
   high as the model improves (curriculum keeps up).
3. **Then ③(generate half)**: diagnose-weakest → gen-targeted → curriculum.
4. **Then ⑤**: GKD + entropy-aware for the runaway-reasoning residual — the last
   lever once self-distill on a good substrate is exhausted (needs a teacher
   stronger per-step than the student: think-on self or DSv4-Flash).

## Aligned assets (our structural edge)

- **train-infer-unified = Tmax's FP32-LM-head fix** — the vLLM-inference vs
  HF-training logprob mismatch Tmax calls out; ARLE shares one FP8 base weight
  set zero-copy, so there is no mismatch. This is the differentiator the papers
  keep hitting.
- `scripts/gen_agent_opd_tasks.py` (synthetic bug-fix gen), `terminus_to_records.py`
  (traj → records), `tbench_opd_loop.sh` (STaR loop) — the substrate ④/③ extend.
- Prewarmed TB base images (uv+pytest+deps baked) — clean offline scoring.

## Non-goals (per OpenThoughts negative results)

- No LLM task-description augmentation (measured to hurt).
- No reward shaping / KL penalty / SFT warmup on the RL recipe (Tmax: outcome-only
  is enough with good data).
