# BBuf Skill Import Distilled Into ARLE Skills

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` session
artifact ledger and `.claude/skills/arle-upstream-runtime-scan/SKILL.md`.

## Context

ckl asked to fetch and scan three BBuf repositories, distill only the parts
that are useful for ARLE into `.claude/skills/`, and avoid clone-then-link.

Local constraints:

- ARLE is a Rust-native inference runtime plus OPD workflow.
- Existing `kernel-optimization` already owns kernel/operator methodology.
- New skills must be ARLE-specific, not generic upstream skill copies.
- No GPU, `infer/src/kv_tier/`, `infer/src/scheduler/`, or P5 training process
  was touched.

## Sources

| Repository | Fetched commit | Scan result |
| --- | --- | --- |
| <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS> | `7340e5c7915c6ae79636429612613e8e47005ff0` | Relevant pieces: model PR history, serving benchmark checklist, capacity/KV budget triage, torch-profiler stage split, model compute simulation. |
| <https://github.com/BBuf/KernelWiki> | `76d27b56f804e7e7295d4c570e1e5d7eef4b0a75` | Relevant pieces: Hopper/Blackwell kernel prior-art lookup, symptom-to-pattern map, PR/source confidence discipline. High overlap with existing `kernel-optimization`. |
| <https://github.com/BBuf/kernel-pilot> | `1b477c0031ecc1ddb4ad7e980ab14f4bb6de048b` | Relevant pieces: K/R/W kernel campaign framing, correctness/benchmark ledger discipline, use prior-art and ncu only when they affect next edit. High overlap with existing `kernel-optimization`. |

## Verdicts

| Candidate | Trigger | Source | Verdict | Overlap / reason |
| --- | --- | --- | --- | --- |
| `arle-upstream-runtime-scan` | Use when ckl asks to ground ARLE serving/runtime/model/benchmark/capacity/Qwen/DeepSeek/OPD decisions in upstream framework evidence. | AI-Infra Auto Driven SKILLS | Kept as new skill | More than 30% ARLE-specific after anchoring to `docs/index.md`, ARLE bench spec, `/v1/stats`, local model paths, and ARLE source modules. |
| KernelWiki prior-art lookup | Use when ckl asks to optimize ARLE kernels and needs upstream Hopper/Blackwell, FlashAttention, DeepGEMM, MoE, FP8/FP4, or ncu-guided precedent. | KernelWiki | Kept by merging into `kernel-optimization` | Sibling skill killed because kernel catalog is already owned by `kernel-optimization`; distilled source-survey rule retained. |
| KernelPilot K/R/W loop | Use when ckl asks for a focused ARLE kernel campaign needing exact semantics, correctness oracle, and workload distribution. | kernel-pilot | Kept by merging into `kernel-optimization` | Sibling skill killed because loop discipline overlaps existing license-or-kill methodology. |
| `llm-serving-auto-benchmark` | Compare SGLang/vLLM/TensorRT-LLM commands under fixed workload/SLA. | AI-Infra Auto Driven SKILLS | Kept inside `arle-upstream-runtime-scan` | Useful only after mapping to ARLE's `bench_guidellm` protocol; not enough ARLE-specific value as a standalone skill. |
| `llm-serving-capacity-planner` | Explain KV/cache/capacity/OOM from serving logs. | AI-Infra Auto Driven SKILLS | Kept inside `arle-upstream-runtime-scan` | Useful only when translated to ARLE knobs and logs. |
| `llm-torch-profiler-analysis` / `llm-pipeline-analysis` | Split external traces by prefill/decode/layer/kernel. | AI-Infra Auto Driven SKILLS | Kept inside `arle-upstream-runtime-scan` | ARLE uses nsys/ncu first; torch-profiler guidance is external-framework context only. |
| `model-compute-simulation` | Estimate model FLOPs/MFU and tensor shapes. | AI-Infra Auto Driven SKILLS | Kept inside `arle-upstream-runtime-scan` | Useful as a checklist for Qwen/DeepSeek shape reasoning; no script copied. |
| `sglang-humanize-review` | Review SGLang PRs like SGLang maintainers. | AI-Infra Auto Driven SKILLS | Killed | SGLang-specific review corpus; less than 30% ARLE-specific. |
| `sglang-prod-incident-triage` | Debug live SGLang incidents. | AI-Infra Auto Driven SKILLS | Killed | Production SGLang incident workflow, not ARLE module guidance. |
| `sglang-sota-humanize-loop` | Run autonomous SGLang SOTA loop. | AI-Infra Auto Driven SKILLS | Killed | Plugin/RLCR workflow and SGLang patch loop; conflicts with ARLE direct-tranche process. |
| `model-architecture-diagram` | Return public model architecture images. | AI-Infra Auto Driven SKILLS | Killed | Mostly diagram lookup, not ARLE runtime work. |

## What Changed

- Added `.claude/skills/arle-upstream-runtime-scan/SKILL.md`.
- Updated `.claude/skills/kernel-optimization/SKILL.md` with:
  - ARLE-specific trigger wording;
  - KernelPilot-style `K/R/W` gate;
  - BBuf KernelWiki / KernelPilot source-survey rule in the catalog.
- Updated `.claude/skills/tmux-agent-control/SKILL.md` with:
  - ARLE-specific trigger wording;
  - removal of one worked example and one stale related-memory line.

## SOLID Check

This is a source-scan and prompt distillation change only. It does not claim
runtime performance, correctness, or capacity behavior.

Evidence level:

- Solid for repository fetch state, file existence, and source-scan verdicts.
- Hypothesis-grade for whether these skills improve future agent behavior; that
  can only be verified by future task outcomes.

Bench status: exempt. This is a pure docs/skill change and does not touch
runtime code, scripts, feature flags, kernels, scheduler, or KV tier code.

## Rule

When importing external skills into ARLE, do not symlink or vendor the upstream
repo. Distill only ARLE-specific triggers and workflows, cite the source URL,
merge overlapping kernel content into `kernel-optimization`, and kill any
candidate whose useful content is less than 30% ARLE-specific.
