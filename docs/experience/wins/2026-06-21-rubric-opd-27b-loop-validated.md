# Rubric-OPD 27B-dense loop validated end-to-end (capability curve pending)

## Context
New `arle train rubric-opd` subcommand (rubric-OPD / RFT): the Qwen3.6-27B dense student
samples N on-policy rollouts per prompt, a strong judge (Qwen3.6-35B-A3B-FP8; DSv4-Flash
deferred) grades each against a **text-level** rubric (vocab-agnostic — the cross-vocab
sidestep), and accepted rollouts are written back as completion-masked CE (RFT). Plan:
[2026-06-21-opd-ceiling-27b-dense.md](../../plans/2026-06-21-opd-ceiling-27b-dense.md).
Built solo (codex rate-limited); pod cuda,nccl; single GPU (GPU7).

## What Worked — full loop validated at 27B
Final smoke (1 prompt, 35B-A3B judge, GPU7, `--lora-layer-start 48`, max-new/verdict 1024):
```
round 0: prompts=1 accepted=2 distinct=1 parse_err=0 trained=2 mean_loss=0.1359  RUN_EXIT=0
```
Decoded cases (§0 case-as-fact): the 27B student solves the problem correctly; the judge
emits a clean JSON verdict (`finish=Stop parse_err=0`) and accepts; CE backward fits; loss
finite; GPU freed clean. The rollout→judge→select→writeback-CE pipeline is correct.

## Infra gaps the GPU surfaced — each root-caused + fixed
1. **LoRA-sync between rounds** (`4612c3cc`): the rollout engine kept sampling base weights
   every round → multi-round RFT never iterated. Fixed via `sync_lora_from_store` after each
   round (round 0 correctly samples base).
2. **VRAM — 3 models co-resident** (`gap #2`): 27B autograd + 27B rollout + 35B-A3B judge.
   Phase-batched `run_rubric_rounds` (sample+judge all → **offload both inference engines**
   (~65 GB freed) → CE all accepted → reload). `FlashJudge` gained offload/reload.
3. **FP8 dense-MLP loader** (`f71bf31e`): `is_fp8_cuda_frozen_base_tensor` whitelisted attn +
   MoE projections but never plain dense `mlp.{gate,up,down}_proj` → a dense 27B-FP8 student
   hit "unsupported dtype F8_E4M3". Added the 3 dense projections (scales already present).
4. **Auto-TP/NCCL**: `INFER_CUDA_DEVICES=4,5,6,7` made the rollout engine attempt TP4 (needs
   `INFER_NCCL_UNIQUE_ID`). A 27B-FP8 fits one GPU → single-GPU placement avoids TP entirely.
5. **CE-OOM at 27B dense / seq ~1k** (`alloc_zeros (transpose)`): even with 65 GB freed, the
   27B dense autograd backward OOMs. Fixed with the proven 35B levers: `--lora-layer-start 48`
   (suffix-detach, top-16 of 64 layers) + gradient checkpointing (`forward_batch_indices`
   honors both). Loaded via `load_qwen35_lora_from_hf_dir_with_layer_start`.

Also surfaced: the only registry 27B Qwen3.x models are **VLMs** (`model.language_model.*`
nesting) — but the loader already handles that nesting; the real blocker was just the FP8
dense-MLP whitelist (#3). And **full-materialized bf16 checkpointing is host-loop pathological
for 27B** (100% CPU, 53 min, no file) — the capability curve will use AdapterOnly save or
in-process eval via the rollout engine, never per-round full-materialize.

## Recipe (validated)
reverse-judge text rubric (Factual gates acceptance), `--max-verdict-tokens 1024` (a
thinking judge truncates its JSON at 200 → parse_err), `--max-new-tokens 1024`,
`--samples-per-prompt 4`, `--lora-layer-start 48` + gradient checkpointing, single GPU,
35B-A3B judge. Engines offloaded during CE.

## Rule
A new train path's correctness is validated by the GPU run decoding actual cases, not by
typecheck. Each OOM/dtype/TP failure is a case to root-cause at file:line, not a structural
KILL — five sequential gaps each fell to a small fix. Full-materialize checkpointing does
not scale to 27B (host-loop); eval the trained student via the in-process rollout engine.

## Capability curve attempt — eval is CONFOUNDED, not a real delta (§0 case-as-fact)
First 3-round curve (n=12 MATH-500, cap=8, `--lora-layer-start 48`, think-on eval
`--eval-max-new-tokens 4096`) scored **base 6/12 (0.500) → round0 3/12 (0.250)** — a −25pp
"regression". Decoding every case (NOT trusting the aggregate) shows it is a **measurement
artifact, not a capability loss**:
- **~40% of base answers truncate before `\boxed`.** Cases 1,7,9,10,11 all end mid-sentence
  (think-on rambles 8–12k chars past the budget) → base capability is itself budget-masked.
- **2 of the 3 round0 flips are pure truncation.** Case 4 (base boxed → r0 truncated mid
  "Step 2"), case 6 (base boxed 27 → r0 truncated mid-deliberation). The metric measures
  "did think-on box within budget," not correctness.
- **n=12 swing = 3 problems = 25pp.** Exactly the small-n + harness-artifact trap; the
  capability-claim rule wants multi-seed ≥5 / larger n for a <5pp effect.
- **Think-off does NOT fix it.** `render_jinja` (infer-server tokenizer.rs:198) already renders
  without `enable_thinking` (lenient-undefined → falsy), yet the 27B produces long CoT and
  truncates anyway. The fix is **larger budget + larger n**, not a think toggle.

**Verdict:** rubric-OPD infra validated at 27B (above); the capability delta is **unmeasured**
— the n=12/4096-budget eval cannot resolve it. No regression claim shipped; no README muscle-flex
on a confounded number. The cosmetic round-label bug (inner `run_rubric_rounds(rounds:1)` logs
"round 0" every outer round; eval files correctly use the outer index `eval_round{0,1,2}.jsonl`)
is harmless to data.

## Pending — infra-first (throughput is the wall to a trustworthy curve)
A trustworthy curve needs larger n + larger think-on budget → **throughput-bound** on one GPU
(CE ~4.5 min/step; eval 12×4096-tok ≈ 30 min). So the next step is the §0 GPU-CE measurement
(`ARLE_OPD_STEP_PROFILE` forward/backward attribution on a 27B-dense CE step) → pick the #1
speedup lever (L3b completion-only-logits confirmed: `forward_batch_tokens` at opd.rs:2838
materializes full `[B,max_len,vocab=248320]` logits + wastes lm_head FLOPs on prompt positions;
fix = lm_head on completion positions only via `forward_batch_hidden_indices` qwen35.rs:3371).
Plan: [2026-06-21-autograd-gpu-ce-speedup.md](../../plans/2026-06-21-autograd-gpu-ce-speedup.md).
DSv4-Flash judge (needs multi-engine-TP) deferred.
