# Agent-OPD held-out execution eval — wired + committed; e2e ATTEMPTED on H20, blocked by a node-governance kill

> **UPDATE 2026-06-27 (e2e attempt):** the held-out eval data was built + locally
> validated as real tasks and pushed to the H20 box, the binary built green, and
> every loop component (both 27B loads, the fast GPU chunked-CE writeback fwd/bwd,
> AdamW, bare engine decode, sandbox cp/git/pytest) verified in isolation — BUT the
> full agent-OPD loop is **killed by a silent external SIGKILL at the dual-27B
> ~58.8 GB resident floor** (5/5 reproductions, no RUN_EXIT / no panic / no CUDA
> error even under `CUDA_LAUNCH_BLOCKING=1`). Root-caused to **node-level GPU
> governance (ELKEID HIDS box), NOT an ARLE code bug** — only the dual-model
> co-residency dies; every single-engine workload survives. Full write-up +
> isolation table + footprint-reduction options:
> [errors/2026-06-27-agent-opd-full-loop-killed-by-node-governance-not-code.md](../errors/2026-06-27-agent-opd-full-loop-killed-by-node-governance-not-code.md).
> The held-out pass-rate trend (baseline → round-N) is therefore **still not
> measured** — needs an un-governed box or a footprint-reduction lever.

Status: the missing measurement component is wired + every local gate green
(cuda,no-cuda check clean; `cargo test -p cli` train_cli 10/10; new
`pass_rate_aggregates_held_out_pass_fail` unit test green). The end-to-end eval
RUN (does the held-out pass-rate climb baseline → round-N?) is BLOCKED — first on
GPU + staged repos (now provisioned + validated), then on the node-governance kill
documented above.

## Why this exists
A systematic audit found agent-OPD had **zero** held-out eval: `TrainAgentOpdArgs`
had no eval fields and `run_agent_opd_impl` logged only `mean_train_loss`. Train
loss alone cannot tell you the model got better at coding — the reward is hidden-
test execution, not loss. Rubric-OPD already had per-round held-out eval
(`rubric_eval_pass`, train_cli.rs); this is the execution-scored agent-OPD analogue.

## What landed (crates/cli + crates/train only)
- **Args** (`args.rs`, `TrainAgentOpdArgs`): `--eval-dataset <jsonl>` (held-out,
  separate from `--dataset`), `--eval-staged-root <dir>` (falls back to
  `--staged-root`), `--eval-n <N>`, `--eval-every <rounds>` (default 1; 0 = off),
  `--eval-out-dir <dir>`, `--eval-temperature` (default 0.0 greedy).
- **Eval harness** (`agent_opd.rs`, `run_agentic_opd_eval`): mirrors
  `run_agentic_opd_round`'s rollout + `score_workdir` loop, but EVAL-ONLY — ONE
  greedy sample/task, NO `train_on_accepted`, NO `masked_writeback_ce_step`, NO
  `optimizer.step`. Returns `AgentEvalReport` (per-task pass/fail/edited +
  `pass_rate()`). Pure `pass_rate(passed,total)` helper is backend-independent +
  unit-tested (`pass_rate_aggregates_held_out_pass_fail`).
- **Wiring** (`train_cli.rs`, `run_agent_opd_eval_pass` + `run_agent_opd_impl`):
  loads held-out tasks (bails on any instance_id overlap with `--dataset`), runs
  a **round-0 baseline BEFORE any training**, then evals every `--eval-every`
  rounds (and always on the final round) AFTER the LoRA sync, re-acquiring the KV
  pool the writeback freed. Writes `eval_round_{base,N}.jsonl` (per-task lines +
  one `"aggregate"` line with `pass_rate`) and LOGS the pass-rate + Δ-vs-baseline
  next to `mean_train_loss`.

Default agent-OPD path is byte-unchanged when `--eval-dataset` is unset (eval
block is fully gated on a non-empty held-out set). The CE writeback / optimizer /
autograd / cuda-kernels were NOT touched (a parallel agent owns the CE port there).

## Why the run is pending-remote
The eval drives the real agent rollout (in-process student engine) + pytest on
the hidden `test_patch` — needs a GPU-resident Qwen3.5 student + the SWE-bench-Pro
repos staged under `--eval-staged-root`. Not runnable on the Mac dev box. Verify
remotely: point `--eval-dataset` at a 1-2 task held-out slice, confirm
`eval_round_base.jsonl` is written before round 0 and the logged
`pass_rate` + `Δ` advances across rounds.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
