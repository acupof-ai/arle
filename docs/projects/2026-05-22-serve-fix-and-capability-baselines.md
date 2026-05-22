# 2026-05-22 — Serve fix + capability baselines: today's wrap

> **Status:** project wrap. Records the unanticipated 90-minute pivot
> from "rerun 4B OPD with BF16" → ARLE serve long-prompt bug discovery
> → root-cause + fix → first valid capability baseline triplet. Lists
> the next concrete tranches to fulfill the user's
> "对比训练前后的变化 + 对比 pytorch 生态" request now that the path
> is unblocked.

## Headline

- ARLE CUDA serve had a **gated-delta-rule chunkwise batched prefill
  bug** that silently corrupted output for prompts ≥ 33 tokens on
  every Qwen3.5 hybrid model (4B BF16, 0.8B-Base, 9B-GPTQ-4bit). OPD's
  `forward_token_logits` direct path bypassed it, so it had been
  latent under every test surface.
- **Fix shipped:** `a374108 fix(infer): repair qwen35 long prompt prefill`
  — a native CUDA recurrent prefill kernel that replays the
  decode-equivalent recurrence over the prefill sequence, routed for
  `seq_len > 32`. Plus a secondary fix in the scheduler warmup
  buffer-reallocation path that was leaving the old large buffer
  resident on shape changes.
- **First valid capability baselines** now exist for the 4B teacher
  and 0.8B student that the OPD pipeline targets.

## Capability baseline triplet (post-fix)

| Label | Backend | Model | MMLU 5-shot | GSM8K 3-shot |
|---|---|---|---|---|
| Student (HF base) | ARLE serve | Qwen3.5-0.8B-Base | **51.4%** (73/142, inv 29) | 1.5% (3/194, inv 6) |
| Teacher (HF base) | ARLE serve | Qwen3.5-4B | **77.3%** (116/150, inv 21) | 2.5% (5/198, inv 2) |
| **Teacher − Student** | | | **+25.92 pp** | +0.98 pp |

This is the OPD distillation ceiling on the current hardware/model
shape: ~26 pp on MMLU between 0.8B and 4B teachers. GSM8K is not a
useful distillation target at this shape because the 4B teacher
itself only solves 2.5% — `Qwen3.5-4B-Instruct` would be a more
realistic teacher for math, but it isn't on disk.

Generated with the new comparison driver:

```bash
python scripts/arle_capability_compare.py \
  --pair "Student=bench-output/...08b-retry-after-longprompt-fix/summary.json" \
  --pair "Teacher=bench-output/...4b-retry-after-longprompt-fix/summary.json"
```

Stored: `bench-output/2026-05-22-capability-baseline-after-longprompt-fix-compare.md`.

## How today's session traced this

| Tick | Commit | What |
|---|---|---|
| pre-arc | `aba67bf` | P0 first run KILL — harness output all invalid |
| | `d3eb295` | Layered MMLU extractor + raw-response debug dump |
| | `2010207` | P0 retry KILL — even with layered extractor, still all invalid → not a harness bug |
| 47 | `a585919` | Cross-model corruption research note — reframes 9B `!` as universal serve bug |
| 48 | `a9380db` | Threshold bisection → **~30 tokens** → H1 (RoPE) strongest, H2 ruled out |
| 49–54 | codex probe (1h35m) | New `qwen35_prefill_path_probe` example + `test_gdr_prefill_matches_repeated_decode_at_long_prompt_threshold` test → bisected divergence to GDR chunkwise prefill at `seq_len=33` |
| 54 | `a374108` | **FIX**: native recurrent prefill CUDA kernel + buffer-realloc sync drop |
| 54 | `b604d83` | Comparison tooling: `HfTransformersClient` (PyTorch backend) + `arle_capability_compare.py` (Δpp driver) |

Cycle-wide: 15+ commits, real bug surfaced + fixed end-to-end in ~2.5 h cooperative cycle.

## What this unblocks

1. **OPD with remote `ApiTeacher`** — was the only practical 9B-teacher
   path on 16 GB per the 2026-05-22 BF16 cycle wrap, but routed
   through the broken serve. Now viable.
2. **Capability eval** — P0 PASS at last. P1 (10k-step distill) and
   P2 (post-distill eval) per the capability-eval plan are unblocked.
3. **End-user `arle serve` chat** — anyone with >30-token prompts was
   getting garbage. Now works.

## "对比训练前后的变化" — framework + open work

Comparison driver is ready (`scripts/arle_capability_compare.py`). The
missing piece for before/after distillation is **student checkpoint
save** in the OPD training example:

- File: `crates/train/examples/opd_step_cuda_infer_teacher_train.rs`
  (currently dirty — codex's territory)
- Needed flag: `--save-student-checkpoint <dir>` saving the LoRA
  adapter (≈50 MB per checkpoint) every `--save-every` steps.
- Wiring: ARLE serve already supports LoRA adapter via the existing
  loader; eval would target the merged or adapter-applied model.

Then the actual comparison becomes:

```bash
# Before
python scripts/arle_capability_eval.py --backend arle --base-url ... \
  --model-id Qwen3___5-0___8B-Base \
  --output bench-output/distill-before/

# Train 10k steps with --save-student-checkpoint
arle train opd --teacher ... --student ... --save-every 2000 \
  --save-student-checkpoint runs/distill-08b/

# After (each checkpoint)
for ck in 2000 4000 6000 8000 10000; do
  # spin up arle serve with the saved LoRA + same base
  python scripts/arle_capability_eval.py --backend arle ... \
    --output bench-output/distill-${ck}/
done

python scripts/arle_capability_compare.py \
  --pair "base=bench-output/distill-before/summary.json" \
  $(for ck in 2000 4000 6000 8000 10000; do
      printf -- '--pair "step%d=bench-output/distill-%d/summary.json" ' "$ck" "$ck"
    done) \
  --pair "teacher=bench-output/baseline-4b/summary.json"
```

Recommended next codex tranche: add `--save-student-checkpoint`.

## "对比 PyTorch 生态" — framework + open work

`HfTransformersClient` is ready (this session, in `b604d83`). One
command to get the PyTorch reference for cross-validation:

```bash
python scripts/arle_capability_eval.py --backend hf \
  --model-id Qwen3___5-4B \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --tasks mmlu --n-samples 200 \
  --output bench-output/2026-05-22-hf-baseline-4b/
```

Then:

```bash
python scripts/arle_capability_compare.py \
  --pair "ARLE serve=bench-output/.../4b-retry-after-longprompt-fix/summary.json" \
  --pair "HF transformers=bench-output/2026-05-22-hf-baseline-4b/summary.json"
```

Expected outcome: agreement within ~1 pp validates the harness + the
serve fix. Disagreement > 2 pp is evidence of another tokenizer / BOS
/ format issue worth filing.

The TRL `GKDTrainer` head-to-head exists from
`docs/experience/wins/2026-05-21-arle-vs-trl-gkd-head-to-head.md`
(2.04× perf win at matched setup). The capability-side comparison
requires P1 distillation completing, then evaluating both ARLE's and
TRL's distilled students on the same harness.

## Cross-links

- Fix details: `docs/experience/wins/2026-05-22-arle-serve-long-prompt-bug-fix.md`
- Bug research: `docs/research/2026-05-22-arle-serve-long-prompt-corruption-cross-model.md`
- Capability eval plan (now executable): `docs/plans/2026-05-22-arle-opd-capability-eval-plan.md`
- OPD methodology: `docs/research/2026-05-22-opd-methodology-and-industry-best-practices.md`
- BF16 cycle wrap (yesterday's substrate): `docs/projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md`
- Comparison driver: `scripts/arle_capability_compare.py`
- Eval harness: `scripts/arle_capability_eval.py` + tests in `scripts/tests/`
