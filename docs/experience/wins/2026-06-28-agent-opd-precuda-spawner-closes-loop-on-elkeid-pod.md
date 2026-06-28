# Pre-CUDA sandbox-spawner closes the agent-OPD loop on the ELKEID H20 pod — first end-to-end held-out pass-rate

## Context

The agent-OPD full loop (rollout → reward → masked-CE writeback → LoRA-sync →
held-out eval) was the last unblocked OPD value-run. After the prior fixes
(`mem_fraction_static` clamp 0.05 → co-resident 27B FP8 OOM cleared `999f5339`;
`cuStreamSynchronize` self-fence → share-frozen-base load deadlock cleared
`748b082e`), the loop died at the **rollout transition** when the first sandbox
subprocess (`cp -a` / `git` / `bash`) forked: the 8×H20 pod's `[ELKEID]` kernel
HIDS aborts (`do_coredump` / `tgkill SIGABRT`) any `fork()` from a CUDA-resident
process. Node-side `dmesg` named the mechanism (`forktest` + `elkeid` +
`do_coredump`) after 5 black-box repros could not — a kernel fork-hook abort looks
identical to a footprint SIGKILL from inside the container
([errors entry](../errors/2026-06-27-agent-opd-full-loop-killed-by-node-governance-not-code.md)).
A libc-level setsid fork-safety fix cannot dodge a kernel hook; the structural fix
that doc prescribes is a **pre-CUDA fork-server**.

## What Worked

Wired the (previously built-but-unwired) `crates/train/src/spawner.rs` fork-server:
fork ONE non-CUDA helper BEFORE the first CUDA context, and route every agent-OPD
rollout subprocess spawn (bash/cp/git/pytest) through it over a unix socket. The
helper never touches CUDA, so its forks are ELKEID-safe; the CUDA process itself
never forks.

- `cli::run()`: when `ARLE_SPAWNER_LISTEN` is set (only by `SpawnerHandle::launch`,
  which re-exec's this binary), run `spawner::serve_loop()` and exit — the FIRST
  thing in `run()`, before logger/clap/threads, so the helper stays a plain
  single-threaded non-CUDA process.
- `run_agent_opd_impl`: `SpawnerHandle::launch()` right before `build_opd_store()`
  (first `CudaBackend::new`); its `Drop` reaps the helper at function exit.
- `sandbox.rs`: `run_captured` (bash, combined-timeout) + `run_checked` /
  `diff_workdir` / `score_workdir` git-apply (plain `.output()`) route through
  `SpawnClient::from_env()` when set; gated → byte-identical default when unset.

**DE-RISK (the load-bearing precondition, measured on the pod first):** a
standalone multithreaded + zero-CUDA program doing `Command::spawn(echo)`
(`launch()`'s exact syscall) exits 0 with **no new dmesg coredump** → `launch()`
itself is ELKEID-safe. (A minimal `cudaFree(0)`-resident fork did NOT reproduce the
abort either — the hook keys on the instrumented full-engine executable, not a
light context; the non-CUDA helper fork is the safe path regardless.)

## Result (measured, H20 GPU4, `RUN_EXIT=0`)

Value-run: `arle train agent-opd --student-model /host/Qwen3.6-27B-FP8
--rounds 1 --samples-per-prompt 1 --max-turns 4 --max-tokens 256
--lora-layer-start 32 --rollout-num-slots 1`, 1 train task + 3 held-out eval tasks.

- **THE CHECKPOINT PASSED**: helper bound its socket, parent confirmed ready, then
  the first `cp -a`/`git`/`bash` and every agent tool turn (find/grep/read/bash)
  **SURVIVED** across both the training rollout and the held-out eval. No SIGABRT,
  **no new dmesg `do_coredump`** in the run window (the only coredumps in dmesg are
  stale, from the prior day). The fork-from-CUDA wall is defeated.
- **Loop closed end-to-end**: rollout → reward → writeback → LoRA-sync → eval →
  `RUN_EXIT=0`. Held-out `pass_rate`: baseline `0.0000` → round-0 `0.3333`
  (1/3 tasks). The passing task (`ansible__ansible-0ea40e0`) actually edited the
  repo (`edited=true`) and the hidden tests passed (`[exit 0]`) — real agentic work.

**Honest attribution (case-as-fact):** `train_mean_loss=0.0000`,
`trained_pairs=0` — the single train task yielded no accepted rollout, so the LoRA
weights were **never updated**. The 0/3 → 1/3 is therefore **rollout sampling
non-determinism, NOT a training-attributable capability gain**. The win is
**infrastructure**: the agent-OPD loop now runs end-to-end on the ELKEID pod for
the first time. A capability claim needs a config that actually trains (≥1
accepted rollout → non-zero `trained_pairs`) and multi-seed eval per the
small-n-eval rule.

## Bench

Exempt: agent-OPD training path, not a serving hot path. The default serve/CLI
path is byte-identical (all routing gated on `ARLE_SPAWNER_SOCKET`, set only by
`launch()`); no `bench_guidellm` delta applies. Train lib sandbox+spawner tests
green on the pod (Linux) incl. new `spawner_routing_matches_direct` (helper-routed
bash/cp/git output byte-identical to direct).

## Rule

- **A pre-CUDA fork-server is the structural dodge for a kernel HIDS fork-hook on
  a CUDA-resident process.** No libc-level fork-safety (setsid/process_group) can
  escape a kernel `fork()` hook; fork the helper BEFORE any CUDA context (parent
  still non-CUDA-resident → that one fork is safe), then route all subprocess
  spawns through it. Gate on an env var so the default path is byte-identical.
- **`mean_loss=0.0000` + `trained_pairs=0` ⇒ no weight update ⇒ any eval delta is
  sampling noise, not learning.** Read the training counters before crediting a
  pass-rate move to the model. Infrastructure-closes-the-loop and
  capability-improves are different claims.
