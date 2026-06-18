# OPD recipe gap — world best practice vs ARLE current source

**Date**: 2026-06-18  
**Track**: Track 2 — on-policy 蒸馏配方 recipe  
**Scope**: research only. No code, no H20, no pod, no training, no bench.

## Read-first ledger

Read before source audit:

- `docs/research/2026-05-25-opd-methodology-audit.md` — old four-gap audit:
  temperature, stochastic rollout, completion-only masking, LR schedule
  (`:16-23`, `:251-309`).
- `docs/research/2026-06-14-rubric-opd.md` — Path A boundary: use rubric
  machinery in distillation form, never GRPO (`:18-26`, `:75-88`).
- `docs/research/2026-06-14-self-training-lora-options-survey.md` — current
  SOPD ledger and Axis A/G framing (`:58-69`, `:117-151`, `:226-261`).
- `docs/research/2026-05-28-opd-effect-axis-next.md` — eval-noise and
  reasoning-corpus gaps (`:29-82`, `:106-145`).

Evidence labels below:

- **Evidence** = current source `file:line` plus primary external source.
- **Hypothesis** = expected capability impact; no ARLE A/B was run in this pass.

## Gap table

| 世界最佳做法 (引用来源) | 我们当前做法 (file:line) | gap | adopt 清单 (删什么/换什么/留什么) |
|---|---|---|---|
| **GKD uses sampled self-generated outputs** to close train/inference distribution mismatch; TRL `GKDTrainer` generates with `do_sample=True`, `temperature=args.temperature`, `top_k=0` ([GKD paper](https://arxiv.org/html/2306.13649v3) §abstract/method; `huggingface/trl@8027c6e:trl/experimental/gkd/gkd_trainer.py:204-214`, `:427-449`). | OPD has sampling, but default remains greedy: `--rollout-temperature` default `0.0` in `crates/cli/src/args.rs:679-686`; `rollout_sampling_params` returns `None` at temperature 0 in `crates/cli/src/train_cli.rs:1213-1230`; rollout falls back to argmax in `crates/train/src/opd.rs:427-455` and `:497-518`. | **Evidence: partially fixed.** The 5/25 “no stochastic rollout” gap is fixed as opt-in, but the default recipe still violates on-policy best practice. Impact is **hypothesis** until same-config A/B. | **换** default recipe/presets from greedy to `rollout_temperature=0.9, top_k=0, top_p=1.0`. **留** greedy for smoke/deterministic gates. **删** docs/help wording that says OPD “samples greedily” as the normal recipe. |
| **TRL GKD defaults**: `temperature=0.9`, `lmbda=0.5`, `beta=0.5`, `max_new_tokens=128` (`huggingface/trl@8027c6e:trl/experimental/gkd/gkd_config.py:29-39`, `:55-77`). | ARLE regular OPD defaults: rollout len 8 (`crates/cli/src/args.rs:675-677`), rollout temp 0.0 (`:679-686`), KL direction forward (`:700-706`), KL mask completion (`:708-710`), LR schedule fixed (`:724-730`), `gkd_lambda=0.0` (`:732-739`). | **Evidence: default recipe gap.** ARLE now exposes many knobs, but the default CLI still describes a smoke recipe, not a GKD recipe. | **换** defaults only after approval: rollout 128 for generic GKD, stochastic temp 0.9, completion mask kept, cosine schedule on. **留** `--rollout-len 8` only in smoke examples. |
| **Completion-only loss**: TRL masks prompt labels to `-100` after generation and loss uses the label mask (`huggingface/trl@8027c6e:trl/experimental/gkd/gkd_trainer.py:287-304`, `:421-425`). | ARLE default is completion-only: `OpdKlMaskArg::Completion` in `crates/cli/src/args.rs:140-146`, default at `:708-710`, tested at `:1263-1280`. Implementation maps completion KL to causal positions `prompt_len-1..sequence_len-1` in `crates/train/src/opd.rs:1264-1295`, then slices logits at `:3118-3137` / windowed paths `:1968-2167`. | **Evidence: fixed.** The 5/25 prompt-KL gap is closed for current defaults. | **留** completion-only as default. **删** any old research/plan claim that current default trains over prompt tokens. **换** nothing unless a future full-sequence ablation is explicitly requested. |
| **Temperature-scaled distillation loss** is part of TRL JSD implementation (`huggingface/trl@8027c6e:trl/experimental/gkd/gkd_trainer.py:225-260`), and TRL config exposes sampling temperature 0.9 (`gkd_config.py:55-58`). | ARLE loss now accepts `kl_temperature` and applies `logits / T` plus `T^2` scaling in `crates/train/src/loss.rs:52-91`; CLI exposes `--kl-temperature` in `crates/cli/src/args.rs:704-706`; it is passed into OPD at `crates/cli/src/train_cli.rs:948-956`. But validation forbids `kl_temperature != 1.0` when `gkd_lambda > 0` in `crates/train/src/opd.rs:1122-1140`. | **Evidence: partially fixed.** Pure KL temperature exists. A TRL-style GKD blend cannot combine temp with SFT anchor today. Default remains `1.0`, not TRL’s 0.9 sampling/default recipe. | **留** current pure-KL temperature path. **换** recipe docs to say temperature is pure-KL only today. **Add later only if approved**: correct scaling for blended JSD/SFT or keep temperature off whenever CE anchor is on. |
| **Loss family best practice is not plain forward KL only.** GKD/TRL uses generalized JSD with beta interpolation (`huggingface/trl@8027c6e:trl/experimental/gkd/gkd_config.py:35-37`, `:66-73`; trainer `:266-285`). MiniLLM argues reverse KL is better for LLM generation ([MiniLLM](https://arxiv.org/html/2306.08543v3) lines 62-64, 337). Thinking Machines uses per-token reverse KL for OPD ([TML blog](https://thinkingmachines.ai/blog/on-policy-distillation/) lines 78-87). | ARLE exposes only `Forward`/`Reverse` in `crates/cli/src/args.rs:134-138`; `KlDirection` only has `Forward`/`Reverse` in `crates/train/src/loss.rs:8-13`; implementation has forward/reverse branches but no beta-JSD mixture in `crates/train/src/loss.rs:68-92`; default is forward in `crates/cli/src/args.rs:700-702`. | **Evidence: loss-family gap.** Reverse-KL endpoint exists, but TRL parity beta-JSD is absent and default remains forward KL. Impact is **hypothesis** until A/B on our eval gates. | **换** first recipe run to use existing `--kl-direction reverse` for OPD/reasoning A/B, because it costs no code. **Add later if approved**: `--kl-beta` generalized JSD. **留** forward KL as baseline/control. |
| **GKD lambda means on-policy data fraction**: TRL `lmbda` controls probability of using student-generated outputs (`huggingface/trl@8027c6e:trl/experimental/gkd/gkd_config.py:32-34`, `:59-65`; trainer `:426-449`). | ARLE `gkd_lambda` means CE/SFT anchor blend weight, not on-policy fraction: CLI text at `crates/cli/src/args.rs:732-739`; `validate_train_opd_gkd_args` rejects `student-rollout` with lambda > 0 and only allows corpus-truth when lambda > 0 in `crates/cli/src/train_cli.rs:246-273`; `mix_gkd_losses` blends scalar KL/SFT in `crates/train/src/opd.rs:3138-3152` and tests endpoints at `:3640-3662`. | **Evidence: semantic gap.** Current regular OPD is effectively always on-policy for KL, while `gkd_lambda` is a hard-token CE mix. Calling it GKD lambda obscures recipe parity. | **换** docs/recipe language: call current knob `sft_anchor_weight`. **留** CLI name until compatibility decision. **Add later if approved**: separate `on_policy_fraction` only if we actually mix offline rows and sampled rows. |
| **LR schedule/warmup should be available in trainer recipes**; TRL inherits Trainer schedule fields, default API exposes linear scheduler and warmup knobs ([TRL docs](https://huggingface.co/docs/trl/en/gkd_trainer) config surface line 242). | ARLE now has optional cosine schedule: `LrScheduleArg::{Fixed,Cosine}` in `crates/cli/src/args.rs:168-174`; OPD args expose schedule/warmup at `:724-730`; `OpdLrSchedule` implements warmup+cosine at `crates/cli/src/train_cli.rs:399-469` and applies per step at `:923-928` / SOPD `:1425-1429`. Default remains `Fixed`; AdamW weight decay is hard-coded `0.0` in `crates/cli/src/train_cli.rs:885` and SOPD `:1390`. | **Evidence: mostly fixed but not default.** The 5/25 “unwired warmup” gap is closed; default recipe still uses fixed LR and no weight-decay knob. | **换** approved recipe preset to `--lr-schedule cosine` with default warmup. **留** fixed LR for ablation. **Add later if approved**: `--weight-decay`; lower priority than sampling/loss. |
| **Reasoning distillation uses long generated reasoning traces / high max generation.** DeepSeek-R1 distills Qwen/Llama from DeepSeek-R1 reasoning data and reports strong math/code results (`deepseek-ai/DeepSeek-R1@0cf7856:README.md:60-97`, `:136-152`); Qwen3 report says smaller models are built by leveraging flagship knowledge ([Qwen3 report](https://arxiv.org/html/2505.09388v1) lines 45-48). | ARLE source can accept `--prompts-file` completions (`crates/cli/src/train_cli.rs:511-587`) and corpus-truth SFT anchor (`:868-883`, `:949-956`), but default OPD has one prompt path when no file is supplied (`:570-586`) and rollout len default 8 (`crates/cli/src/args.rs:675-677`). Existing research already found current corpora lack CoT (`docs/research/2026-05-28-opd-effect-axis-next.md:106-145`). | **Evidence: data/length recipe gap.** Code can consume better data, but the default recipe does not encode long reasoning traces or generated reasoning corpora. | **换** recipe runbooks to require task-matched generated completions for reasoning. **留** single prompt only for smoke. **删** any claim that loss tweaks alone can fix GSM8K/R1 without reasoning-shaped data. |
| **Inline SOPD needs stable self-training signal inside the OPD-only boundary.** Existing SOPD survey promotes A1 EMA soft KL as the rollout-time spine and records A2/A5 as CE selection boosters, not GRPO (`docs/research/2026-06-14-self-training-lora-options-survey.md:117-151`, `:226-249`; rubric Path A in `docs/research/2026-06-14-rubric-opd.md:18-26`). | `self-opd` defaults `gkd_lambda=0.5` because pure KL cold-start is zero-gradient (`crates/cli/src/args.rs:868-872`; guard at `crates/cli/src/train_cli.rs:1246-1258`). It uses EMA teacher + student-rollout CE anchor, completion-only KL, no corpus truth (`crates/cli/src/train_cli.rs:1437-1457`). | **Evidence: recipe gap remains for capability.** SOPD bootstrap has a practical nonzero gradient, but it is not the same as GKD/TRL beta-JSD or Thinking Machines reverse-KL recipe. | **留** EMA+CE bootstrap for cold start. **换** evaluation recipe to compare `reverse KL`, `forward KL`, and CE weight, one variable at a time. **Do not add GRPO**; Path A boundary stands. |

## Status of the 2026-05-25 four gaps

| Old gap | Current status |
|---|---|
| Distillation temperature | **Surface fixed, default/combination gap remains.** `--kl-temperature` exists, but only pure-KL accepts `T != 1.0`. |
| Stochastic student rollout | **Surface fixed, default gap remains.** Sampling exists; default is still greedy. |
| Completion-only token masking | **Fixed.** Current default is completion-only and implementation slices causal completion logits. |
| LR schedule / warmup | **Surface fixed, default gap remains.** Cosine+warmup exists; default remains fixed LR; weight decay still fixed at 0.0. |

## Ranked adopt list

1. **Adopt an existing-knob “GKD-ish” recipe before adding code.** Use stochastic rollout (`temp=0.9, top_k=0, top_p=1.0`), `rollout_len >= 128` for generic tasks, completion-only KL, and cosine warmup. This uses current source except default/runbook changes.
2. **Run loss-family A/B with existing knobs first.** Forward KL vs reverse KL is already implemented. Do not add beta-JSD until reverse-vs-forward fails to settle the question.
3. **Separate lambda semantics.** Treat current `gkd_lambda` as CE anchor weight, not TRL on-policy fraction. Rename in docs first; code rename only after approval.
4. **Reasoning data is mandatory for reasoning gains.** For R1/GSM8K-style work, use long teacher-generated reasoning completions; the default one-prompt/8-token smoke recipe is not evidence.
5. **Keep the closed fixes.** Completion-only masking and batchmean KL scale are now correct enough to leave alone unless a single-variable ablation asks otherwise.
6. **Low-priority cleanup after recipe proof.** Add `--weight-decay` and maybe beta-JSD only after the existing knobs show a real effect outside eval noise.

## Sources

- GKD paper: <https://arxiv.org/html/2306.13649v3>
- TRL GKD config: <https://github.com/huggingface/trl/blob/8027c6e23edf9c762042ae52c99f63905e49f29c/trl/experimental/gkd/gkd_config.py>
- TRL GKD trainer: <https://github.com/huggingface/trl/blob/8027c6e23edf9c762042ae52c99f63905e49f29c/trl/experimental/gkd/gkd_trainer.py>
- MiniLLM: <https://arxiv.org/html/2306.08543v3>
- Thinking Machines Lab OPD: <https://thinkingmachines.ai/blog/on-policy-distillation/>
- DeepSeek-R1 repo: <https://github.com/deepseek-ai/DeepSeek-R1/blob/0cf78561f1d51c84a21b2190626b21116d5c68bb/README.md>
- Qwen3 Technical Report: <https://arxiv.org/html/2505.09388v1>
