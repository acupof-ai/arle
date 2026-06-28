# Agent-OPD first CLEAN end-to-end run on the H20 box — baseline held-out pass_rate=0.3333 captured, but `trained_pairs=0` (config/data artifact, NOT capability) → held-out Δ=+0.0000 is a no-train null, not a value signal

## Context

Mainline brief: capture the agent-OPD value signal (`trained_pairs`/`mean_loss`
+ held-out eval Δ) under strict process hygiene on the shared 8×H20 pod, using a
FAST-CAPTURE config (`--samples-per-prompt 2 --writeback-cap 1`,
`--max-turns 16 --max-tokens 768`) to minimize the ~26-min/trajectory writeback.
Marker discipline: run ONLY via `/host/arleCKL` (symlink →
`/host/arle-ckl-aopd/target/release/arle`), `exec -a arleCKL` so argv[0]=arleCKL,
own-tag-only cleanup, pinned to a free GPU. Binary HEAD `7ae42221` (built Jun 28
12:36); the brief's expected `8bfbd615` was not what was on the pod — the `7ae42221`
binary has `--writeback-cap` and the full `agent-opd-eval` path, so it was used.

Continuation of the node-governance crash-loop blocker documented in
[full-loop-killed-by-node-governance](../errors/2026-06-27-agent-opd-full-loop-killed-by-node-governance-not-code.md),
[writeback-gpu-loss-captured-crashloop-blocks-full-signal](2026-06-28-agent-opd-writeback-gpu-loss-captured-host-op-pinned-crashloop-blocks-full-signal.md),
and [train-task-validated-6of6-accept-crashloop-block](2026-06-28-agent-opd-train-task-validated-6of6-accept-pod-crashloop-block.md).

## What Worked (case-as-fact, measured)

**FIRST clean end-to-end agent-OPD completion on this box — `[wrapper] exit_code=0`,
`done (1 rounds)`.** The full loop ran: engine+student load → round-0 baseline
held-out eval → round-1 rollout → score → writeback-attempt → LoRA save → round-1
held-out eval → clean exit. Every prior session (and attempts 1-5 this session)
died by silent process-group SIGKILL during load/eval. Attempt 6 (GPU 7) caught a
stable ~10-min window (the "37-min stable phase" the blocker entries predicted)
and ran the whole loop. Marker hygiene held: `ps -ef` showed `argv[0]=arleCKL`
across the run; own-tag-only cleanup freed all GPUs to 0 MiB, no foreign process
touched.

**Baseline (round-0) held-out pass_rate = 0.3333 (1/3) — the first time the full
3-task baseline eval has completed on this box.** Per-task, IDENTICAL in base and
round-1 eval (`eval_round_base.jsonl` == `eval_round_1.jsonl`):

| instance_id | base | round-1 | note |
|---|---|---|---|
| ansible__ansible-0ea40e0 | **PASS** | **PASS** | `[exit 0]` — `combine_vars` `TypeError` fix, hidden test `test_combine_vars_replace` passes |
| ansible__ansible-12734fa | fail | fail | `[exit 4]` (edited, tests fail) |
| ansible__ansible-5e36960 | fail | fail | `[exit 1]` (edited, tests fail) |
| **aggregate** | **0.3333** | **0.3333** | **Δ = +0.0000** |

The base Qwen3.6-27B-FP8 student has real agentic SWE capability: it edited all 3
held-out tasks and genuinely solved 1/3 (the hidden test gates the pass).

**The round-1 rollout decode shows strong bug-localization (case-as-fact).** On the
TRAIN task `ansible__ansible-f327e65` (Python-keyword collection-name validation),
sample 0 ran 15+ turns of systematic `find`/`grep`/`read` and correctly localized
the bug at turn 12-15: `VALID_COLLECTION_NAME_RE = re.compile(r'^(\w+)\.(\w+)$')`
+ the `keyword.iskeyword` gap in `is_valid_collection_name` /
`collection_loader/_collection_finder.py`. This is real reasoning, not blind
exploration.

## Root Cause of `trained_pairs=0` — config/data artifact, NOT a capability ceiling or a real null

```
round 0: tasks=1 rollouts=2 passed=0 distinct=0 no_token_record=0 trained_pairs=0 mean_loss=0.0000
round 0: held-out pass_rate=0.3333 (baseline=0.3333, Δ=+0.0000) train_mean_loss=0.0000
```

Decoded per-sample (the gate that distinguishes (b) from (b)-by-artifact):
- **sample 0: `score error: git apply of test_patch failed: patch does not apply`.**
  The model edited, but the hidden `test_patch` targets `test/units/cli/test_galaxy.py:460`
  and `test/units/utils/collection_loader/test_collection_loader.py:718`, which do
  NOT line up with the staged repo at `base_commit f533d465` → `git apply` fails →
  the rollout is **unscorable** (not failed — never validated). This is a
  data-staging mismatch (staged base_commit vs test_patch offsets), not a model
  miss.
- **sample 1: `no edits (turns=16, MaxTurns)`.** Hit the FAST-CAPTURE `--max-turns 16`
  cap mid-exploration. The prior 6/6-accept run used the default 30-turn/2048-token
  budget on this SAME task — the 16-turn cap truncated it before the edit.

So `trained_pairs=0` here is the **dual product of (1) a staged-repo/test_patch
offset mismatch making sample 0 unscorable and (2) the 16-turn cap truncating
sample 1** — NOT "the model can't do it" (it solves this task 6/6 at the default
budget, prior entry) and NOT a real on-policy null. Because zero pairs were
written back, the LoRA never updated (`adapters_round1/adapter_model.safetensors`
is the zero-init identity adapter), so the held-out **Δ=+0.0000 is trivially flat
— a no-train null, not a train-then-flat signal.**

## Verdict

**(c)→partially cleared, then (b)-by-artifact.** The infra blocker (node-governance
SIGKILL) was hit 5× early but a stable window opened and the loop completed cleanly
for the first time — clearing the "is the loop even runnable e2e" wall. The
training-gate `trained_pairs>0` was NOT met (0 pairs), so there is **no real value
signal this round**: held-out Δ=+0.0000 reflects zero writeback, not a measured
flat. This is NOT verdict (a) (no training happened) and NOT a clean (b) (the null
is a staging/turn-cap artifact, not on-policy evidence).

To get a real signal next run (cheap, file:line):
1. **Re-stage the TRAIN task so its `test_patch` applies** at the staged
   `base_commit` (the gate that makes an accepted rollout scorable), OR pick a
   train task whose patch is verified to apply — mirror the eval-task staging that
   already works (all 3 eval tasks scored cleanly).
2. **Raise `--max-turns` to the default 30** (the FAST-CAPTURE 16-turn cap is the
   thing truncating the rollout before the edit); keep `--max-tokens` adequate.
3. Then `trained_pairs>0` → `mean_loss>0` → a real round-1 Δ. A rigorous capability
   claim still needs multi-seed (≥5) + Wilson CI per the small-n-eval rule
   (3-task eval is far below that bar; this run is a plumbing+baseline capture,
   not a capability verdict).

## Run facts

- Marker: `exec -a arleCKL /host/arle-ckl-aopd/target/release/arle` (HEAD `7ae42221`),
  argv[0]=arleCKL confirmed in `ps -ef`. nvidia-smi `--query-compute-apps`
  `process_name` shows `[Not Found]` (the documented H20 PID-namespace trap:
  nvidia-smi reads host `/proc`, the proc is container-namespaced) — process
  ownership was proven instead via the PGID-rooted tree (`arleCKL` marker → worker
  → DeepGEMM nvcc/ptxas children) and the GPU-mem allocation appearing exactly at
  our engine load.
- GPU: attempt 6 on physical GPU 7, peak ~40 GB (well under 97 GB — NOT a memory
  OOM; corroborates the node-governance kill is residency-time-correlated, not
  footprint).
- 5 early attempts (GPU 1/2/3/4/6) died <90 s by silent group SIGKILL at the
  load/eval step — same signature as the prior crash-loop entries (no panic, no
  CUDA error, no wrapper exit line). Attempt 6 survived the window.
- Eval out: `/host/aopd_evalout_ckl/eval_round_{base,1}.jsonl`. LoRA:
  `/host/agentopd_ckl/adapters_round1` (zero-init identity, trained_pairs=0).

## Bench

Exempt: agent-OPD training path, not a serving hot path. No code change this run
(config + pod-data staging only); default serve/CLI byte-identical. Per the
mandatory-bench rule this is the training-axis capture, not a guidellm serving
delta.

## Rule

- **`trained_pairs=0` is a case to decode per-sample, never a capability verdict.**
  Here it was (sample 0) an unscorable rollout — the hidden `test_patch` didn't
  `git apply` to the staged repo — plus (sample 1) the `--max-turns 16` cap
  truncating before the edit. The same student solves this task 6/6 at the default
  30-turn budget. A `Δ=+0.0000` on top of `trained_pairs=0` is a **no-train null**
  (zero writeback → unchanged LoRA), NOT an on-policy flat — do not report it as a
  value signal.
- **Validate that the TRAIN task's `test_patch` applies at the staged
  `base_commit` before a writeback run, exactly as the eval tasks are.** A scorable
  rollout is the precondition for `trained_pairs>0`; a patch-offset mismatch makes
  an otherwise-correct edit unscorable and silently zeros the writeback.
- **FAST-CAPTURE turn caps trade away the writeback.** `--max-turns 16` minimizes
  wall-clock but can truncate the agentic rollout before it edits → no accepted
  trajectory. For a value-signal run, keep the turn budget at the level where the
  task is known to be solvable (30 here), and minimize elsewhere.
- **On this ELKEID/node-governance H20 box the kill is residency-time-correlated,
  not footprint:** a single engine survives 88 GB for minutes; the dual-copy train
  path is SIGKILL'd at ~40 GB. Stable ~10-37-min windows DO open — relaunch into
  them rather than concluding "unrunnable"; a clean `[wrapper] exit_code=0` is
  reachable.
- **nvidia-smi `process_name=[Not Found]` on a container-namespaced proc is the
  PID-namespace trap, not a tagging failure.** Prove ownership via the PGID tree +
  the GPU-mem-appears-at-our-load timing, with `exec -a arleCKL` giving argv[0] in
  `ps`.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
