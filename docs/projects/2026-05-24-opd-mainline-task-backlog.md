---
title: OPD mainline + runtime optimization backlog
date: 2026-05-24
type: backlog + execution-order index
status: live — codex executes top→bottom; Claude maintains
owner: ckl
related:
  - docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md
  - docs/research/2026-05-24-bf16-frozen-base-impl-path.md
  - docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md
  - docs/projects/2026-05-18-opd-only-pivot.md
---

# OPD mainline + runtime optimization backlog

## Live state (refreshed each /loop tick)

- **Mainline**: optimize OPD effect + perf. Per CLAUDE.md + 2026-05-18 OPD-only pivot.
- **Concurrent local GPU**: P5 pure-OPD 5k run (PID 28950), step 1007/5000
  (eval@1000: train_kl 1.51e-5→1.36e-5 −10%, heldout_kl 1.74e-5→1.60e-5 −8%
  vs step 0). ETA remainder ~10h. ~4.7 GB GPU headroom (11.7 GB used).
  GPU is OK to use for sub-4 GB peak jobs (user 2026-05-24 22:05 "gpu 可以
  用啊没问题的"). Do not OOM-kill it.
- **Codex active task**: auto-pulling from §Queue per standing instruction
  (sent 2026-05-24 23:30). Just shipped T1 (14c3be9 code + fc65d4f wins,
  11m46s). Expected next pickup: T2 trace analysis.
- **Recent commits this session**: fdb021c (codex bbuf skills),
  d436dfa (Claude backlog + gkd-OOM errors), 14c3be9 (codex T1 code),
  fc65d4f (codex T1 wins). All pushed origin/main.
- **Codex hard-stop conditions** (only times it should idle):
  1. license-or-kill threshold needs user/Claude call
  2. change would touch P5 PID 28950
  3. file collision with other in-flight editor
  4. single task >2h impl with no license-or-kill in backlog
  5. SOLID self-check finds the gate itself is bugged
- **Codex authority**: reorder by measurement, record reason in wins/errors
  entry.

## Rules

- One commit per gap. Wins entry per PASS, errors entry per KILL.
  License-or-kill thresholds are in source plan §5 (for G-series) or this
  doc's queue items.
- Mainline = OPD. Items that improve teacher-infer wall-clock (G1/G2/G3,
  chunked-KL, kv_tier observability) flow back to OPD step time and are
  in-scope. Items that only improve serve-side without an OPD link can be
  deprioritized.
- `不要限制多` — codex has wide latitude inside each task. The brief states
  the goal + acceptance gate, not the implementation. If codex finds a better
  axis, follow it and document the pivot.
- Cooperative discipline: explicit-path stage, no `git stash`, no `git add .`,
  no force push to main without confirmation. Don't touch P5 PID 28950 or
  `infer/src/{kv_tier,scheduler}/` outside the kv_tier observability task.

## Queue (execution order — codex top→bottom unless license disagrees)

| # | Task | Owner | Status | Gate | Source |
|---|---|---|---|---|---|
| T1 | Ship `run_opd_from_dirs` CLI + wins entry | codex | in_progress | compile + clippy clean on standalone diff | This session 2026-05-24 |
| T2 | End-to-end OPD trace, max-split (per-phase wall-clock) | codex | **deferred until P5 finishes** | every phase has a measured number, not file:line citation only | User 2026-05-24 22:00 |
| T3 | Delete non-mainline / dead code audit | codex | **completed** (8ca4403, 81842cc, 2f975cb; 4th-cluster grep clean) | each removal cites zero grep usage; one commit per cluster | User 2026-05-24 22:00 |
| T4a | kv_tier observability metrics — **code-only** (no bench) | codex | queued | new metric fields landed + unit tests pass; audit-first to avoid duplicating existing infrastructure | Split 2026-05-25 — code-only part is CPU-safe |
| T4b | kv_tier observability — ≥4k SERVE baseline bench | codex | **deferred until P5 finishes** | baseline numbers recorded before any PrefetchPolicy::Timeout work | Split from T4 |
| T5 | Chunked-logits KL implementation | codex | queued | real-corpus GKD reaches eval_summary step=0 + 1 train_step on 16GB | bf16 research mit. 2 |
| T6 | gap-analysis §6 G1→G7 ordered execution | codex | queued | each Gn passes its §5 license-or-kill threshold (PASS→wins, KILL→errors) | User 2026-05-24 23:xx |
| T7 | SGLang docs deep-mine — surface gaps not yet in T6 | codex | queued | docs/research/2026-05-24-sglang-deep-mine-gaps.md with kill thresholds | User 2026-05-24 23:xx |

Detail per task:

### T1 — `run_opd_from_dirs` CLI ship

- Brief: `/tmp/codex-task1-shipopd-cli.txt`.
- Why: 101-line WIP in `crates/cli/src/train_cli.rs` wires actual `arle train
  opd --student-model <dir>` end-to-end (autograd Tape + AdamW + qwen35_loader
  + opd_step). Previously printed "pending". Major user-facing surface; never
  committed; never documented.
- Acceptance: standalone commit on `crates/cli/src/train_cli.rs` + wins entry
  `docs/experience/wins/2026-05-24-arle-train-opd-from-dirs.md`. Don't drag
  other dirty files; if compile depends on them, STOP and report.

### T2 — End-to-end OPD trace, max-split

- Goal: every phase from CPU scheduling → tokenize → admission → prefill
  stages → decode → KL → backward → optimizer step → checkpoint, with measured
  wall-clock. Max-split — break each stage until no further sub-phase.
- Baseline shape: 4B teacher + 0.8B student LoRA, prompt_max_tokens=16,
  rollout_len=8 (current P5 shape). Then a second pass at prompt_max_tokens=256
  if T5 lands.
- Tools: NVTX-annotated bench + `nsys profile` (canonical per
  `docs/bench-and-trace-spec.md`); phase_summary log from
  `opd_step_cuda_infer_teacher_train` as the in-process counter.
- Deliverable: `docs/research/2026-05-24-arle-opd-end-to-end-trace.md` —
  per-phase wall-clock table + identified bottleneck rank + license-or-kill
  thresholds for the top 3.
- Constraint: don't OOM P5. If GPU full, queue this for after P5 finishes
  (ETA ~11h from session start).

### T3 — Delete non-mainline / dead code

- Mainline = OPD per 2026-05-18 pivot. Already deleted: scratch pretrain,
  SFT, GRPO, multi-turn RL.
- Codex audit pass: find experimental examples, deprecated paths, dead code
  surviving prior cleanups. Each deletion must cite zero grep usage in the
  current workspace (excluding examples/tests for the same module).
- One commit per cluster of related deletions, not a mega-commit.
- Record in `docs/experience/wins/2026-05-24-nonmainline-prune-<cluster>.md`
  per cluster; cite which 2026-05-18 pivot exclusion or 2026-05-22 OPD-only
  EOD wrap motivates each.

### T4 — kv_tier observability metrics patch

- Per codex 2026-05-24 kv deep audit, the architectural direction is sound;
  the real gap is metrics-driven autotune of static knobs. **Add metrics
  FIRST, change algorithms LATER.**
- Counters to add to `ServerMetrics`:
  - Per-tier hit rate (T0 / T1 / T2 / T3)
  - T0→T1 demote latency histogram + bytes
  - T1→T2/T3 store latency histogram + bytes
  - Staged-readmission fetch-wait p50/p99
  - Queue-saturated fallback count
  - Recompute-advised fallback count
  - Host pool high/low pressure tick count
- Baseline: SERVE workload with ≥4k-token prompts (T1 gate is 4096) to
  actually fire the path. Record baseline before any
  `PrefetchPolicy::Timeout` work (T6 G5 dependency).
- Out of scope: policy changes, algorithm tweaks, NIXL T3 hookup.

### T5 — Chunked-logits KL

- Per `docs/research/2026-05-24-bf16-frozen-base-impl-path.md` mitigation 2.
- Effect axis only — unblocks real-corpus GKD at `prompt_max_tokens=512+`
  (currently OOM-kills before step 0 per the gkd-real-corpus-tape-oom-kill
  errors entry). NOT a 16-tok perf win (KL is 0.13% of step at current shape).
- Touches `crates/train/src/loss.rs:89-115` + eval path + autograd.
- Acceptance: real-corpus GKD reaches `eval_summary step=0` + ≥1
  `train_step` on 16 GB GPU at prompt_max_tokens=512, rollout_len=8.

### T6 — gap-analysis §6 G1→G7

- Direct execution of `docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md`
  §6 ordered table (10 sub-items including license-or-kill experiments).
- Each Gn has its own PASS/KILL threshold in §5. Honor it.
- Each PASS → `docs/experience/wins/2026-05-24-gap-G<n>-<short>.md`.
- Each KILL → `docs/experience/errors/2026-05-24-gap-G<n>-kill.md`.
- Codex can interleave with T2/T3/T4/T5 when a gap depends on those.

### T7 — SGLang docs deep-mine

- Read `https://docs.sglang.ai/` end-to-end + key blog posts.
- Cross-reference against current ARLE state.
- Surface optimizations NOT in T6's gap-analysis. Likely candidates:
  speculative decoding (Eagle-2/3/MTP), structured output (xgrammar/outlines),
  FP8 W8A8, prefix-aware scheduling, chunked-prefill heuristics, request-level
  mixed precision, KV cache compression.
- Output: `docs/research/2026-05-24-sglang-deep-mine-gaps.md` — each candidate
  with ARLE relevance score + license-or-kill threshold.

## What changed in this session that updated this backlog

- 2026-05-24 22:00 — User flagged docs gap; recorded uncommitted `train_cli`
  WIP as T1, queued trace + delete-nonmainline + chunked-KL + observability.
- 2026-05-24 22:05 — User: GPU OK to use. Removed the "skip GPU" constraint
  from T2 (still don't OOM P5).
- 2026-05-24 23:xx — User: execute gap-analysis G1→G7, deep-mine SGLang.
  Added T6 + T7.
- 2026-05-24 23:xx — User: "不要限制多 引导 多看方向". Loosened brief
  prescriptiveness — codex has authority to reorder by measurement.

## How Claude maintains this

- Every /loop tick: refresh §"Live state" (P5 progress, GPU usage, codex
  current task).
- On new user directive: add to §"Queue" with cite-line in §"What changed".
- On codex task ship: move row to "completed" with commit hash + wins/errors
  link.
- On license-or-kill: record verdict against the gate column.
- Do NOT write code from this file. This file is the index; codex writes the
  code; Claude writes wins/errors/research.
