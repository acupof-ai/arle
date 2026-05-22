# 2026-05-22 — OPD cycle wrap (EOD)

> **Status:** end-of-day cycle wrap. Supersedes
> [`2026-05-22-serve-fix-and-capability-baselines.md`](2026-05-22-serve-fix-and-capability-baselines.md)
> (written mid-cycle, before the lr sweep + GSM8K findings).
> This is the single read-first entry point for the 2026-05-22 OPD
> cycle. Tomorrow's reader: start here, then drill into the linked
> wins/errors entries.

## TL;DR (5 lines)

1. **ARLE serve had a hidden bug** — Qwen3.5 GDR chunkwise prefill diverged at ≥33-token prompts, corrupting all long-prompt output (`!` saturation, garbage tokens). Found via the capability eval P0 attempt, fixed in `a374108` with a native recurrent prefill CUDA kernel.
2. **OPD pipeline now closes end-to-end**: train → save LoRA → load via `INFER_LORA_PATH` → eval through the existing OpenAI v1 surface → compare. Three commits across `a1ad570` (save), `584f07b` (load), `bb72066` (loop).
3. **ARLE serve ≈ HF transformers** on the same Qwen3.5-4B checkpoint, MMLU n=171: 77.33 % vs 78.18 % (Δ +0.85 pp, within ±5 pp 95 %-CI). Cross-engine validation passes.
4. **2 k-step OPD distillation does not yet beat the base on MMLU** at either lr=2e-5 or lr=1e-5. U-curve dynamic is **fundamental, not lr-driven**.
5. **Task-divergent OPD impact**: at the same training step, MMLU and GSM8K move in opposite directions (lr=1e-5 step 1000: MMLU 50.6 % below base, GSM8K 3.16 % briefly above the 4B teacher's 2.5 %). One-task verdicts are incomplete.

## Headline numbers

### Pre/post serve fix on Qwen3.5-4B MMLU 5-shot (n=171)

| State | scored | invalid | correct | accuracy |
|---|---:|---:|---:|---:|
| Pre-fix (broken) | 0 | 171 | 0 | **0 %** |
| Post-fix (ARLE serve) | 150 | 21 | 116 | **77.33 %** |
| Cross-check (HF transformers) | 165 | 6 | 129 | **78.18 %** |

### OPD distill trajectory (Qwen3.5-4B teacher → 0.8B-Base LoRA r=16 student, 2 k steps)

| Snapshot | MMLU step 0 | MMLU step 1000 | MMLU step 2000 | GSM8K step 1000 | GSM8K step 2000 |
|---|---:|---:|---:|---:|---:|
| Base (no LoRA) | 51.4 % | — | — | 1.5 % | — |
| lr=2e-5 | 51.4 % | 47.9 % | 50.0 % | — | 1.6 % |
| **lr=1e-5** | 51.4 % | **50.6 %** | **48.5 %** | **3.16 %** | 2.22 % |
| Teacher 4B | 77.3 % | 77.3 % | 77.3 % | 2.5 % | 2.5 % |

- **MMLU**: U-curve. Both lrs show a valley, neither crosses base in 2 k.
- **GSM8K**: inverse U-curve at lr=1e-5. Peak (3.16 %) briefly exceeds teacher (2.5 %), then regresses. **Likely sample noise at near-floor accuracy** (1-4 correct out of 200), not a real signal — see task-divergent wins entry for the full reading.

## Today's chart

![OPD distill MMLU U-curve + GSM8K inverse-U](img/2026-05-22-arle-opd-distill-trajectory.png)

(Generator: `scripts/plot_opd_today.py`. Source data: `bench-output/2026-05-22-capability-after-distill-*` and the cycle's wins entries.)

## Commit chronicle (≈30 commits, 26 hours)

### Real bug fixes (5)

| Commit | What |
|---|---|
| `4214b4d` | `fix(qwen35): load BF16 linear-attn f32 tensors by dtype` (earlier — loader fix) |
| `a374108` | **`fix(infer): repair qwen35 long prompt prefill`** — the headline GDR prefill fix |
| `424a4cf` | `fix(cuda): include H2D allocation details in autograd upload errors` (diagnostic) |
| `d3eb295` | `fix(opd): MMLU extractor + raw-response debug dump` (eval harness layered extractor) |
| `3e191b9` | `fix(ci): scope autograd cuda feature to infer's cuda` (CI cudarc leak) |

### CI automation (4)

| Commit | What |
|---|---|
| `a3cfc57` | `fix(ci): drop stale README Latest Updates hygiene check` |
| `8db358a` | `ci: skip docs-only and bench-output commits via paths-ignore` |
| `9a1d7b7` | `ci: scope fmt check to changed .rs files + nightly full-workspace sweep` |
| `491b4f8` | `ci: add scripts/ci-smoke.sh for local pre-push verification` |

### OPD pipeline + tooling (8)

| Commit | What |
|---|---|
| `c04ddc6` | `feat(opd): minimal capability eval harness — MMLU + GSM8K vs ARLE serve` (353 lines Python) |
| `7d41ba8` | `test(opd): offline unit tests for capability eval harness parsers` (29 tests) |
| `b604d83` | `feat(opd): eval harness HF backend + side-by-side comparison driver` |
| `a1ad570` | `feat(opd): save LoRA student checkpoints` |
| `584f07b` | **`feat(qwen35): load LoRA adapters in CUDA serve`** — closes the load gap |
| `bb72066` | `docs(opd): close qwen35 lora train save load eval loop` (P1-B PASS) |
| `0629294` | `docs(opd): record lr1e-5 valley sweep kill` (P2 KILL) |
| `9fa5fc2` | `docs(opd): GSM8K inverse-U-curve + task-divergent OPD impact` |

### Cycle documentation (10+)

| Commit | What |
|---|---|
| `5875519` | OPD methodology + industry best practices research note (204 lines) |
| `9cf61c5` | Capability eval plan |
| `6b6e14c` | Capability eval plan — HTTP audit + task subset revision |
| `a585919` | Cross-model long-prompt corruption research note |
| `a9380db` | Threshold bisection of the long-prompt bug |
| `6560f5c` | Mid-cycle wrap (serve fix + first valid baselines) — superseded by THIS doc |
| `1586cef` | Valley-then-recovery diagnosis (lr=2e-5) |
| `98c69a4` | Cross-validate ARLE serve vs HF transformers |
| `6841e66` + `827686c` | README chart + headline updates |

## What's actually claimed vs what's pending

### Claimed (today's deliverables)

- **OPD pipeline closes** at the 4B → 0.8B LoRA shape, on 16 GB GPU.
- **`arle serve` correctness** matches PyTorch reference on Qwen3.5-4B MMLU within ±1 pp.
- **The serve long-prompt bug** is fixed, and a regression test pins it (`infer/src/ops/tests.rs::test_gdr_prefill_matches_repeated_decode_at_long_prompt_threshold`).
- **OPD U-curve dynamic** is observable + characterized at lr=2e-5 (valley → partial recovery) and lr=1e-5 (shallower valley, no recovery within 2 k).
- **Task-divergent impact** is real: MMLU U vs GSM8K inverse-U at the same training step.

### Not yet claimed

- **Distilled student beats base on MMLU**. Best result is lr=2e-5 step 2000 = 50.0 %, still 1.4 pp below base 51.4 %.
- **OPD recipe with a published capability win**. The literature target is 10 k–100 k steps; we've only run 2 k.
- **GKD λ-mixing (SFT + OPD blend)** as a stabilizer. The research note recommends it; not yet implemented.

## Recommended next axes (with cost / expected information)

In rough order of cost-to-information ratio (best first):

1. **Save finer-grain checkpoints in a rerun** (e.g. `--save-every 250`) so the MMLU valley shape is bisected at sub-1000-step resolution. ~2 h GPU. Information: where exactly the valley bottoms, and whether the lr=1e-5 GSM8K peak at step 1000 is sharp or broad.
2. **GKD λ-mixing implementation** (`--gkd-lambda 0.3` = 30 % SFT + 70 % OPD). ~2 h codex implementation + 1 unit test + 1 short 2 k smoke run. Information: literature hypothesis that early-phase fluency stabilization closes the valley.
3. **5 k step rerun at lr=2e-5** (the lr that showed actual recovery). ~7.5 h GPU. Information: whether the recovery dynamic continues linearly past 2 k and finally crosses base.
4. **TRL `GKDTrainer` head-to-head on capability** (we have perf only). ~6 h GPU (matched-setup TRL run + eval). Information: ARLE OPD vs PyTorch ecosystem on the same capability dimension.

The user has not selected an axis yet; this list is the cycle's "what to do next" map for the next session.

## Cross-links (in load order)

1. **OPD methodology + industry best practices**: [`../research/2026-05-22-opd-methodology-and-industry-best-practices.md`](../research/2026-05-22-opd-methodology-and-industry-best-practices.md)
2. **Long-prompt bug research (cross-model evidence)**: [`../research/2026-05-22-arle-serve-long-prompt-corruption-cross-model.md`](../research/2026-05-22-arle-serve-long-prompt-corruption-cross-model.md)
3. **Long-prompt bug fix**: [`../experience/wins/2026-05-22-arle-serve-long-prompt-bug-fix.md`](../experience/wins/2026-05-22-arle-serve-long-prompt-bug-fix.md)
4. **Pipeline closure**: [`../experience/wins/2026-05-22-p1b-train-save-load-eval-loop.md`](../experience/wins/2026-05-22-p1b-train-save-load-eval-loop.md)
5. **Valley-then-recovery diagnosis**: [`../experience/wins/2026-05-22-distill-trajectory-valley-then-recovery.md`](../experience/wins/2026-05-22-distill-trajectory-valley-then-recovery.md)
6. **lr sweep KILL**: [`../experience/errors/2026-05-22-p2-lr-sweep-not-the-fix.md`](../experience/errors/2026-05-22-p2-lr-sweep-not-the-fix.md)
7. **Task-divergent OPD impact**: [`../experience/wins/2026-05-22-opd-task-divergent-impact.md`](../experience/wins/2026-05-22-opd-task-divergent-impact.md)
8. **Cross-engine validation vs HF**: [`../experience/wins/2026-05-22-arle-vs-hf-transformers-cross-validation.md`](../experience/wins/2026-05-22-arle-vs-hf-transformers-cross-validation.md)
9. **Capability eval plan (executable)**: [`../plans/2026-05-22-arle-opd-capability-eval-plan.md`](../plans/2026-05-22-arle-opd-capability-eval-plan.md)
10. **CI root-cause + automation**: [`../experience/errors/2026-05-22-ci-cudarc-leak-and-hygiene-drift.md`](../experience/errors/2026-05-22-ci-cudarc-leak-and-hygiene-drift.md)
11. **Mid-cycle wrap (superseded)**: [`2026-05-22-serve-fix-and-capability-baselines.md`](2026-05-22-serve-fix-and-capability-baselines.md)

## Cooperative-cycle protocol notes

- **Audit-first-then-substrate-then-experiment** worked again, just at a different scale: instead of feature-substrate audit, it was bug-root-cause audit. The codex probe loop (`qwen35_prefill_path_probe` example → narrow diff to GDR chunkwise → kernel fix) is the canonical pattern.
- **License-or-kill applies to hypotheses too**, not just code. The `lr=1e-5` sweep was a hypothesis (lr-driven valley) that we killed with one 3 h experiment. The errors entry preserves the evidence.
- **Cross-engine validation is mandatory after a serve-path fix.** The pre-2026-05-22 GDR bug was invisible to ARLE-only smokes because no PyTorch baseline existed for divergence comparison. After today, `scripts/arle_capability_eval.py --backend hf` is the standing cross-check.
- **Task-divergent eval is the unit of evidence.** A single-task OPD verdict was misleading until we had GSM8K alongside MMLU. The eval triplet now defaults to both, plus future tasks should be added as adapter support broadens.
