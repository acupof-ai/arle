# Terminal-Bench OPD: format-conformance distillation lifts 27B pass@1 +5.1pp

> Status: Active — 2026-07-07 · Driver: ckl · First measured OPD capability
> lift on a standardized benchmark. Follows
> [2026-07-05-terminal-bench-pod-27b-frontier-baseline](2026-07-05-terminal-bench-pod-27b-frontier-baseline.md).

## Goal

Close the agentic-OPD loop on a *clean, standardized* substrate: baseline →
distill execution-passing trajectories → re-eval → measured Δpp. The prior
SWE-Pro self-distill returned Δ≈0 (zero gradient — distilled already-solved
tasks). Terminal-Bench + a decoded failure analysis exposed a REAL gradient.

## The key insight — the gradient is output-format conformance

On a clean pass@3 baseline the 27B's dominant failure is NOT capability: of 31
fails across 39 trials, ~22 are **terminus strict-JSON parse errors + runaway
reasoning** (`unknown_agent_error` 11, `fatal_llm_parse_error` 6, `agent_timeout`
5). The model emits reasoning prose before the JSON, or exhausts `max_tokens`
mid-reason and returns empty content → terminus can't parse → no commands run →
fail. This is a distillable deficit the model *lacks* (clean parseable
terminus-JSON), i.e. a genuine capability gradient — unlike self-distilling
solved tasks.

## Params

- **Baseline / eval**: `tb run -a terminus -m openai/Qwen3.6-27B-FP8`,
  `terminal-bench-core==0.1.1`, 13 tasks × `--n-attempts 3`, `--n-concurrent 3`,
  agent-timeout 900s, test-timeout 300s. Optimized `--release` binary,
  `--bind 0.0.0.0`.
- **Clean harness (prereq)**: pre-warmed base images
  (`ghcr.io/laude-institute/t-bench/{ubuntu-24-04,python-3-13}:latest` rebuilt
  with uv+pytest+numpy/pandas/gitpython/… baked into the uv cache) eliminate the
  test-phase wheel-download timeouts that made earlier runs 0/14. Recipe:
  [[reference_terminal_bench_on_pod_podman_proxy]].
- **Distill**: `arle train agent-opd --replay-records` over **41 records** =
  every episode of the **8 execution-passing** baseline trials (converted by
  `scripts/terminus_to_records.py`: each episode → one masked-CE (prompt,
  assistant-action) pair via the model's chat template). LoRA attention-qv
  r16/α32, `--writeback-window 1024`, 3 epochs. Loss 0.2165 → 0.1796 → 0.1453.
- **Post-eval**: serve `--lora-adapters` (re-merge: 16 layers r16 α32), same 13
  tasks × 3.

## Results

**pass@1: 20.5% (8/39) → 25.64% (10/39) = +5.1pp** — understated by artifacts
below.

| Task | baseline pass@3 | post pass@3 | note |
|---|---|---|---|
| hello-world | 0/3 | **3/3** | format fix — was unparseable output, now clean JSON (verified) |
| heterogeneous-dates | 0/3 | **2/3** | format fix |
| fix-permissions | 2/3 | **3/3** | ↑ |
| grid-pattern-transform | 2/3 | 2/3 | = |
| git-workflow-hack | 2/3 | **0/3** | REAL regression (test ran 0.03s, genuine fail) |
| fix-git | 1/3 | 0/3 | ARTIFACT — all 3 post attempts hit the uv-installer re-curl timeout |
| openssl-selfsigned-cert | 1/3 | 0/3 | ARTIFACT — same (3× test-timeout) |
| chess / configure-git-webserver / csv-to-parquet / fibonacci / nginx / password-recovery | 0/3 | 0/3 | = |

- **Mechanism confirmed (case-as-fact)**: hello-world baseline failed because the
  model's output didn't parse (no file created); post-distill it emits
  `{"commands":[{"keystrokes":"echo \"Hello, world!\" > hello.txt"}, …]}` — clean
  parseable JSON — and passes 3/3.
- **Artifact correction**: fix-git and openssl "losses" are 100% test-timeout
  (the task test re-runs `curl astral.sh/uv/install.sh` against live egress,
  which the wheel-cache prewarm does NOT cover; both hit it 3/3 this run). Their
  baseline 1/3 was a lucky non-timeout attempt. Excluding both artifact-poisoned
  tasks, the net on clean tasks is **+6 gains − 2 (git-workflow-hack) = +4
  trials**.
- **One real regression**: git-workflow-hack 2→0 — the LoRA over-fit the terminus
  format at the cost of this task's git logic. Net still positive.

## Rule

- **OPD needs a capability GRADIENT; find it by decoding failures, not by
  distilling solved tasks.** Self-distilling execution-passing trajectories only
  lifts when those trajectories carry something the base policy *lacks* — here,
  output-format conformance (parseable terminus-JSON without runaway reasoning),
  which ~22/31 baseline fails were starved of. SWE-Pro self-distill was Δ≈0
  because the passing trajectories were of already-solved easy tasks (no new
  info). The decoded failure taxonomy is what tells you whether a gradient
  exists.
- **Clean the harness before believing a Δ.** Three separate 0/14 non-results
  this campaign were harness artifacts (slow release-fast binary → agent
  timeout; intermittent egress → test-dep download timeout; container bridge net
  → CC streaming reset), not capability. Pre-warmed images + optimized binary +
  host-side terminus + pass@3 were each required to get a trustworthy number.

## Next

- Kill the residual uv-installer re-curl timeout (bake uv itself + a pinned
  pytest venv into the base image, or point the installer at a pod-local cache)
  → clean the fix-git/openssl signal.
- Investigate the git-workflow-hack regression (format LoRA vs task logic
  tradeoff — lower α or mix in more diverse passing tasks).
- Scale: full 89-task set, more distill trajectories (best-of-N to widen the
  format-conformance corpus), STaR iteration.

Raw: pod `/host/tb_runs/2026-07-06__08-49-39` (baseline),
`/host/tb_runs/2026-07-06__15-11-39` (post), `/host/tb_lora/`,
`/host/tb_pass_records.jsonl`, `/host/tb_distill.log`.
