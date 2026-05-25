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

## Live state (refreshed by T13 docs audit, 2026-05-25)

- **Mainline**: optimize OPD effect + perf. Per CLAUDE.md + 2026-05-18 OPD-only pivot.
- **Concurrent local GPU**: P5 pure-OPD 5k run has exited. T14 swept all five
  saved checkpoints; MMLU did not beat the no-LoRA base, and GSM8K remains
  near-floor. T5b then KILLed the 512-token chunked-KL acceptance. T2 traced
  the P5 shape and ranked student rollout/backward as the wall-clock bottleneck.
  GPU is free for the next serialized GPU task.
- **Codex active task**: auto-pulling from §Queue per standing instruction
  (sent 2026-05-24 23:30). CPU-only tasks through T13 are now linked in
  §Session artifact ledger. GPU-blocked work resumes only after P5 exits.
- **Recent commits this session**: see §Session artifact ledger for the
  fdb021c→HEAD cross-reference map.
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
| T1 | Ship `run_opd_from_dirs` CLI + wins entry | codex | **completed** (14c3be9 code, fc65d4f wins) | compile + clippy clean on standalone diff | This session 2026-05-24 |
| T2 | End-to-end OPD trace, max-split (per-phase wall-clock) | codex | **completed** (trace docs) | every phase has a measured number, not file:line citation only | User 2026-05-24 22:00 |
| T3 | Delete non-mainline / dead code audit | codex | **completed** (8ca4403, 81842cc, 2f975cb; 4th-cluster grep clean) | each removal cites zero grep usage; one commit per cluster | User 2026-05-24 22:00 |
| T4a | kv_tier observability metrics — **code-only** (no bench) | codex | **completed** (375f09f audit, 83b9710 impl, a696fb4 tests; 588 unit tests pass) | new metric fields landed + unit tests pass; audit-first to avoid duplicating existing infrastructure | Split 2026-05-25 — code-only part is CPU-safe |
| T4b | kv_tier observability — ≥4k SERVE baseline bench | codex | **completed** (GuideLLM baseline + controlled 4k session replay) | baseline numbers recorded before any PrefetchPolicy::Timeout work; T0→T1 demote and T1 readmission metrics validated, T1→T2 store stayed zero under session-ref protection | Split from T4 |
| T5a | Chunked-logits KL — **code-only** (forward + backward + unit tests) | codex | **completed** (dae29d0 audit, 1d7cd5b impl, ab2d0f6 tests, 61980ef wins) | parity test against existing KL on a small shape passes within ε; tape memory drops vs current shape (synthetic check, no real-corpus bench) | Split 2026-05-25 — code-only part is CPU-safe |
| T5b | Chunked-logits KL — real-corpus 512-tok acceptance bench | codex | **KILL** (`291ec53` default-off wiring; errors entry documents c64/c8/c1 controls) | real-corpus GKD reaches eval_summary step=0 + 1 train_step on 16GB at prompt_max_tokens=512 | bf16 research mit. 2 |
| T6 | gap-analysis §6 G1→G7 ordered execution | codex | **partial** (G1 deferred for architecture license; G6 completed 9dcc166; GPU/Metal items remain gated) | each Gn passes its §5 license-or-kill threshold (PASS→wins, KILL→errors) | User 2026-05-24 23:xx |
| T7 | SGLang docs deep-mine — surface gaps not yet in T6 | codex | **completed** (c05e055) | docs/research/2026-05-24-sglang-deep-mine-gaps.md with kill thresholds | User 2026-05-24 23:xx |
| T11 | Storage + transport library — **design exploration** (HIGH PRIORITY, runs after T7) | codex (design only, no impl) | **completed** (ce17782) | output: docs/plans/2026-05-25-kv-storage-transport-library-design.md per §"T11" detail block | User 2026-05-25 — "存储层 + 传输层 高效库,尤其 SSD↔HBM / DRAM↔HBM" |
| T8 | M-state dirty file audit — decide ship-vs-revert per file | codex | **completed** (3bc1ea9 audit cleanup, 08a2cca corpus ship, 3b86586 fmt alignment) | each of the still-dirty M files (lora.rs, weights.rs, bootstrap.rs, qwen35_checkpoint.rs, teacher_infer.rs, train_cli.rs leftover, 3 train+infer examples, autograd test) has a verdict: ship as standalone commit, revert if abandoned, or merge into a related landed feature | Continuous-cleanup discipline |
| T9 | Audit `cargo test -p infer` / `-p train` "existing unrelated blockers" called out in T4a wins | codex | **completed** (9669212, 3103a0c, bca1f31) | wins entries 2026-05-25 cite test failures unrelated to the changed code — codex audits whether those are real flakes, env-specific, or hidden bugs; fix or document each | T4a wins entry surfaced this |
| T10 | G-series code-only wireframes — **scope narrowed to G5 only** (G2/G4 defer to Mac) | codex | **completed** (8b595a6 feat(kv-tier) gate coordinator t2 disk wireframe; cargo check + 588 lib tests + codex review --uncommitted all green) | G5 only — Coordinator stub for T2 disk fetch/store, gated behind existing config flag default-off | Codex caught bugged gate 2026-05-25 — Linux can't typecheck Metal cfg |
| T12 | **Capability eval harness preflight** (Task 14 prep) | codex | **completed** (ccb5ce9) | Verify `scripts/arle_capability_eval.py` runs end-to-end with `--dry-run` or equivalent: paths resolve, env vars match P5's checkpoint layout (runs/2026-05-24-p5-pure-opd-5k/), MMLU/GSM8K loader hits cache, INFER_LORA_PATH expects the right adapter file shape from `crates/train/src/qwen35_checkpoint.rs:save_lora_only`. Output: dry-run log + any path bugs fixed, so the moment P5 hits step 5000 we just hit play with no setup delay. **Industry-result enabler.** | User 2026-05-25: "目标是要拿到结果 / 业界有成就的成果"; P5 ETA ~3h |
| T13 | Session docs cross-reference audit | codex | **completed** (docs-only cross-reference audit) | Audit this session's commits (fdb021c onwards) — every wins/errors/research entry should link to and from the related plan/project doc. Update CHANGELOG.md (if exists) and ROADMAP.md (if stale). Fix broken links. Continuous-cleanup discipline. | Continuous-cleanup |
| T14 | P5 pure-OPD 5k capability sweep | codex | **completed** (capability sweep docs) | Sequential MMLU/GSM8K eval over step_001000..step_005000; compare capability winner against heldout-KL winner and base | User 2026-05-25 after P5 exit |
| T18 | **OPD recipe variant** — lr=1e-5 or LR-decay-from-step-1000, save every 250 | codex | queued (after T17) | bench shape same as P5: Qwen3.5-4B teacher → 0.8B-Base LoRA, rollout 8, prompt_max 16, 5000 steps. Knob: `--lr 1e-5` OR `--lr-decay-after-step 1000`. PASS = a checkpoint heldout_kl < P5 best (1.598e-5) AND its MMLU >= 50% (i.e., recovery exceeds P5 step-2000 best). KILL = no checkpoint beats P5 step-2000 (50.00% MMLU). T14 wins entry §"new research direction" pre-licensed this; T2 trace says backward+rollout dominate so this recipe change doesn't shift wall-clock. Output: docs/experience/wins/2026-05-25-opd-recipe-variant.md (PASS) or errors (KILL). | User 2026-05-25 "opd 也继续推进" + T14 wins recommendation |
| T19 | **Stale doc + dead code cleanup pass** ("删除陈旧的文档和无用的代码 需要的时候再加") | codex | queued (after T18) | See §"T19" detail block. Scan-classify-prune cycle: identify stale docs / dead code → classify SAFE vs needs-ckl-review → ship safe deletions one cluster per commit → list borderline items in a wins/errors entry for ckl decision. Respect bench-spec §9 immutability (`wins/`, `errors/` entries are historical record, not deletable). | User 2026-05-25 "需要的时候再加" philosophy |

### T19 — Stale doc + dead code cleanup pass

User 2026-05-25 directive: "删除所有的陈旧的文档和无用的代码 需要的时候
再加" (delete all stale docs and unused code; add back when needed).

**What counts as stale / dead**:

- Docs claiming things that aren't true today (e.g., `docs/projects/
  2026-05-18-opd-only-pivot.md:86,150` says "OPD substrate landing next
  milestone" but it landed via `crates/cli/src/train_cli.rs::run_opd_from_dirs`
  on 2026-05-24 `14c3be9`).
- Plan docs whose subject was executed and superseded (e.g., a plan that
  said "we will do X" where X is now done and tracked elsewhere).
- Duplicate plan files (e.g., `docs/plans/dsv4-small-repro.md` vs
  `docs/plans/2026-05-25-dsv4-from-scratch-feasibility.md` — verify
  duplication, kill if redundant).
- Code paths with zero grep usage across the workspace (T3 did one pass;
  the cycle continues — re-grep after T17/T18 changes).
- `TODO`/`FIXME` comments without actionable item (either action them or
  delete the comment).
- Dead feature gates (`#[cfg(feature = "x")]` where `x` has no callers).
- Old `bench-output/` artifacts older than 30 days that aren't referenced
  by any current wins/errors entry.

**What is NEVER deletable** (per `docs/bench-and-trace-spec.md` §9 + this
project's experience discipline):

- Any file under `docs/experience/wins/` or `docs/experience/errors/`
  — historical SOLID record, immutable.
- `CHANGELOG.md` entries — append-only.
- Git commits — already-shipped history (Co-Authored-By trailer cleanup
  is a separate user decision, pending B-option call).
- Plan docs that capture design rationale that future work depends on
  (even if the plan is "done", the rationale may still inform follow-ups).

**Classification rule**:

For each candidate, codex outputs one of:

- **SAFE_DELETE** — zero current refs, no historical value, ship in this
  task's commits.
- **EDIT_NOT_DELETE** — stale claim within an otherwise-valuable doc
  (e.g., 2026-05-18 pivot doc); update the claim, don't delete the doc.
- **NEEDS_CKL_REVIEW** — codex is unsure if deletion loses real value;
  list in T19 errors entry with the question for ckl.

**Output**:

- One commit per cluster of related SAFE_DELETE items
  (`chore(cleanup): prune <cluster-name>` or `docs(cleanup): drop stale
  <topic> docs`).
- One commit per EDIT_NOT_DELETE batch (`docs(...): update stale
  <claim>`).
- Wins entry `docs/experience/wins/2026-05-25-stale-doc-dead-code-cleanup.md`
  with the full classification table (kept / deleted / edited / pending-ckl).
- Total touched lines should be measurable but conservative — better to
  underdelete and surface borderline calls than overdelete.

**Anti-patterns to avoid**:

- Deleting a doc because it's "old". Date alone doesn't make a doc stale.
- Deleting code because it's "complex". Complexity alone doesn't make
  code dead.
- Mass-renaming or restructuring as part of cleanup — separate concern.
- Touching `wins/` / `errors/` entries (immutable).
- Reverting any commits to "clean history" — see CLAUDE.md "Never destructive
  without explicit ask".

## Session artifact ledger (fdb021c onwards)

| Task / commit range | Plan / project anchor | Artifact |
| --- | --- | --- |
| BBuf skills import (`fdb021c`) | This backlog; `.claude/skills/arle-upstream-runtime-scan/SKILL.md` | [wins/2026-05-24-bbuf-skills-import.md](../experience/wins/2026-05-24-bbuf-skills-import.md) |
| T1 OPD CLI main path (`14c3be9`, `fc65d4f`) | [2026-05-18 OPD-only pivot](2026-05-18-opd-only-pivot.md) | [wins/2026-05-24-arle-train-opd-from-dirs.md](../experience/wins/2026-05-24-arle-train-opd-from-dirs.md) |
| T2 OPD end-to-end trace | This backlog | [research/2026-05-24-arle-opd-end-to-end-trace.md](../research/2026-05-24-arle-opd-end-to-end-trace.md), [wins/2026-05-25-opd-end-to-end-trace-p5-shape.md](../experience/wins/2026-05-25-opd-end-to-end-trace-p5-shape.md) |
| T3 non-mainline prune (`8ca4403`, `81842cc`, `2f975cb`, `e049787`) | [2026-05-18 OPD-only pivot](2026-05-18-opd-only-pivot.md) | [train-test](../experience/wins/2026-05-24-nonmainline-prune-train-test.md), [empty train commands](../experience/wins/2026-05-24-nonmainline-prune-empty-train-commands.md), [sample corpus](../experience/wins/2026-05-24-nonmainline-prune-train-sample-corpus.md) |
| T4a kv-tier observability (`375f09f`, `83b9710`, `a696fb4`) | [tiered-kv-runtime-flow.md](tiered-kv-runtime-flow.md) | [wins/2026-05-25-kv-tier-observability-code-patch.md](../experience/wins/2026-05-25-kv-tier-observability-code-patch.md) |
| T4b kv-tier SERVE baseline | [tiered-kv-runtime-flow.md](tiered-kv-runtime-flow.md) | [wins/2026-05-25-kv-tier-observability-serve-baseline.md](../experience/wins/2026-05-25-kv-tier-observability-serve-baseline.md) |
| T5a chunked KL (`dae29d0`, `1d7cd5b`, `ab2d0f6`, `61980ef`) | [bf16 frozen-base impl path](../research/2026-05-24-bf16-frozen-base-impl-path.md) | [wins/2026-05-25-chunked-logits-kl-code-patch.md](../experience/wins/2026-05-25-chunked-logits-kl-code-patch.md), [errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md](../experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md) |
| T5b real-corpus chunked-KL acceptance (`291ec53`) | [bf16 frozen-base impl path](../research/2026-05-24-bf16-frozen-base-impl-path.md) | [errors/2026-05-25-chunked-kl-real-corpus-512-kill.md](../experience/errors/2026-05-25-chunked-kl-real-corpus-512-kill.md) |
| T6/G6 radix insert validation (`9dcc166`) | [SGLang gap analysis](../plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md) | [wins/2026-05-25-gap-G6-radix-insert-noop.md](../experience/wins/2026-05-25-gap-G6-radix-insert-noop.md) |
| T7 SGLang deep mine (`c05e055`) | [SGLang gap analysis](../plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md) | [research/2026-05-24-sglang-deep-mine-gaps.md](../research/2026-05-24-sglang-deep-mine-gaps.md) |
| T11 storage + transport design (`ce17782`) | [tiered-kv-runtime-flow.md](tiered-kv-runtime-flow.md) | [plans/2026-05-25-kv-storage-transport-library-design.md](../plans/2026-05-25-kv-storage-transport-library-design.md) |
| T8 M-state audit (`3bc1ea9`, `08a2cca`, `3b86586`) | [2026-05-18 OPD-only pivot](2026-05-18-opd-only-pivot.md) | [wins/2026-05-25-m-state-audit.md](../experience/wins/2026-05-25-m-state-audit.md) |
| T9 test cleanup (`9669212`, `3103a0c`, `bca1f31`) | T4a follow-up | [errors/2026-05-25-test-suite-cleanup.md](../experience/errors/2026-05-25-test-suite-cleanup.md) |
| T10/G5 coordinator stub (`8b595a6`) | [SGLang gap analysis](../plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md), [T11 transport plan](../plans/2026-05-25-kv-storage-transport-library-design.md) | [wins/2026-05-25-gap-G5-coordinator-stub.md](../experience/wins/2026-05-25-gap-G5-coordinator-stub.md) |
| T12 eval harness preflight (`ccb5ce9`) | [2026-05-22 EOD OPD cycle wrap](2026-05-22-eod-opd-cycle-wrap.md) | [wins/2026-05-25-capability-eval-preflight.md](../experience/wins/2026-05-25-capability-eval-preflight.md) |
| T13 docs cross-reference audit | This backlog | [wins/2026-05-25-session-docs-cross-reference-audit.md](../experience/wins/2026-05-25-session-docs-cross-reference-audit.md) |
| T14 P5 capability sweep | [2026-05-22 EOD OPD cycle wrap](2026-05-22-eod-opd-cycle-wrap.md), [distill trajectory](../experience/wins/2026-05-22-distill-trajectory-valley-then-recovery.md), [task-divergent impact](../experience/wins/2026-05-22-opd-task-divergent-impact.md) | [wins/2026-05-25-p5-pure-opd-5k-capability-sweep.md](../experience/wins/2026-05-25-p5-pure-opd-5k-capability-sweep.md) |

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
- T5b verdict 2026-05-25: KILL. `--kl-chunk-size 64`, `8`, and `1` all
  failed before `eval_summary step=0`; current chunking happens after full
  teacher/student logits are already materialized.

### T6 — gap-analysis §6 G1→G7

- Direct execution of `docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md`
  §6 ordered table (10 sub-items including license-or-kill experiments).
- Each Gn has its own PASS/KILL threshold in §5. Honor it.
- Each PASS → `docs/experience/wins/2026-05-24-gap-G<n>-<short>.md`.
- Each KILL → `docs/experience/errors/2026-05-24-gap-G<n>-kill.md`.
- Codex can interleave with T2/T3/T4/T5 when a gap depends on those.

### T8 — M-state dirty file audit

The repo has these uncommitted M-state files that have lived through multiple
codex tasks without being shipped (or reverted):

- `crates/autograd/tests/test_cuda_lazy_ops.rs`
- `crates/train/src/qwen35_checkpoint.rs`
- `crates/train/src/teacher_infer.rs`
- `crates/train/examples/opd_step_cuda_{convergence_bench,realckpt_diag,realckpt_profile}.rs`
- `infer/examples/{gptqmodel_w4_gemv_parity,qwen35_dense_module_dump,qwen35_linear_attn_parity}.rs`
- `infer/src/backend/cuda/bootstrap.rs`
- `infer/src/model/qwen35/{lora,weights}.rs`
- `examples/opd/sft-anchor-mmlu-gsm8k.jsonl` (untracked)
- `bench-output/2026-05-22-h3-max-seq-len-4096-08b/serve.log` (output noise — likely git-ignore-worthy)

For each file, codex runs `git diff <file>`, decides:
1. **Ship as standalone commit** — diff is meaningful + self-contained + tests
   pass → commit with appropriate scope.
2. **Merge into a landed feature** — diff is a follow-up to an already-shipped
   commit → cherry-pick into that line if reasonable, otherwise standalone
   commit citing the parent.
3. **Revert** — diff is abandoned experiment with no value → `git checkout
   <file>`. Cite the reason.
4. **Add to .gitignore** — output / log noise → ship .gitignore update.

One commit per cluster of related verdicts. Document in
`docs/experience/wins/2026-05-25-m-state-audit.md`.

### T9 — Unrelated test blockers audit

T4a wins entry (`2026-05-25-kv-tier-observability-code-patch.md`) noted
that `cargo test -p infer` and `cargo test -p infer --tests` hit existing
example/Metal audit blockers that are unrelated to the T4a change. Codex:

1. Runs `cargo test -p infer 2>&1 | tee /tmp/infer-test-output.log` and
   `cargo test -p train 2>&1 | tee /tmp/train-test-output.log` on a clean
   target dir.
2. Classifies each failure: env-specific (Metal-only on Linux), flake
   (intermittent), real-bug (deterministic, unrelated to recent changes).
3. Fix the real bugs (≤3 files per fix, separate commit per).
4. Skip with `#[ignore]` + comment for env-specific (CI matrix already
   covers via the right runner).
5. Mark flakes with `#[ignore = "flaky — see docs/...."]` and open errors
   entry for each.

Output: `docs/experience/errors/2026-05-25-test-suite-cleanup.md` with
the classification table.

### T10 — G-series code-only wireframes (low priority)

For G-series gaps that ARE hardware-blocked but have a code-only
"skeleton" that could land without bench verification, codex can stage:

- **G4 Metal GPU sampler**: port CUDA `gpu_sample_cuda` logic to MLX
  primitives in Rust; gate with the existing `metal` feature; `cargo check
  -p infer --no-default-features --features cuda,no-cuda` passes (Linux
  Mac-equivalent typecheck). Bench (sampling KS test ≤ 0.05) deferred until
  ckl runs on Mac.
- **G2 spike skeleton**: experiment harness file under
  `crates/mlx-sys/examples/encode_replay_spike.rs` ready to run when
  ckl/codex have Mac access; not licensed yet.
- **G5 wiring stub**: extend the `Coordinator` consumer in
  `infer/src/scheduler/cuda/core.rs` for the T2 disk fetch/store path,
  gated behind a config flag default-off; bench deferred.

ONLY do these after T7/T8/T9 complete. They're skeleton-shipping with
deferred verification, so the value is "doesn't block when hardware
appears" — not "delivers value today".

### T11 — Storage + transport library design exploration

User 2026-05-25 request: efficient library for storage layer + transport layer,
especially **SSD ↔ HBM** and **DRAM ↔ HBM**. Core: efficient organization +
transport.

**Current state (Claude pre-survey, codex to verify file:line)**:

| Layer | Existing | Where | Status |
|---|---|---|---|
| T0 GPU HBM page pool | `TokenKVPool` | `crates/cuda-kernels/src/paged_kv.rs` | Owns T0; not in kv_tier |
| T1 DRAM pinned arena | `HostPinnedPool` | `infer/src/kv_tier/host_pool.rs` | Live |
| T2 SSD persistence | disk transport | `infer/src/kv_tier/transport/disk.rs` | Live, no-fsync proven |
| T3 shared-fs / NIXL | `shared_fs.rs` + `nixl.rs` stub | `infer/src/kv_tier/transport/` | Real = shared-fs; NIXL stub |
| Local D↔H | `local_cuda.rs` | `infer/src/kv_tier/transport/` | Live |
| Native persistence | `kv-native-sys` | `crates/kv-native-sys/` | Substrate, partially exposed |

**Codex design pass — output `docs/plans/2026-05-25-kv-storage-transport-library-design.md`**:

1. **Inventory** — every storage + transport surface today, with file:line +
   API shape + who calls it. No omissions.
2. **Bottleneck map** — for each existing path, measured or hypothesized
   wall-clock cost per byte + cost per op + sync points. Mark "measured" vs
   "hypothesis" per §0 SOLID.
3. **Upstream survey** — what do these projects do for SSD↔HBM / DRAM↔HBM that
   ARLE doesn't:
   - NVIDIA GPUDirect Storage (GDS) — bypasses CPU bounce buffer for SSD→HBM
   - NVIDIA NVLink/NVSwitch — HBM↔HBM peer-to-peer
   - SGLang HiCache backends — storage/distributed-storage tiers
   - MoonCake transfer engine — KV pool migration across nodes
   - NIXL spec — RDMA-class remote tier abstraction
   - NCCL DMA paths — multi-GPU KV sharing
4. **Proposed crate / API shape** — choose ONE:
   - extend `crates/kv-native-sys` (substrate already exists)
   - extend `infer/src/kv_tier/transport/` (live, but tied to scheduler)
   - new `crates/kv-transport` (clean break, swappable backends)
   Give a recommendation with reason; do not pre-commit.
5. **ROI per proposed sub-component** — must cite measurable wall-clock win or
   correctness gap. Items without ROI → KILL.
6. **License-or-kill thresholds** — each sub-component gets PASS/KILL gate
   tied to a bench or test, NOT to "we should have this".
7. **Constraint**: don't propose anything that requires Mac access (Metal
   unified memory makes T0/T1 boundary moot); CUDA-lane focus.
8. **Constraint**: don't propose anything that needs Coordinator rewrite —
   the boundary discipline from the 2026-05-24 kv audit must hold (RadixCache
   = metadata truth, scheduler = lifecycle, coordinator = bytes).

**Design-only this tranche.** No code, no architectural refactor commits. After
ckl reviews the design doc → license one sub-component tranche at a time.

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
