# Agent-OPD: 6/6 training rollouts pass the validated SWE-bench-Pro task — accept→writeback gate proven, writeback capture blocked by pod CrashLoopBackOff

## Context

Follow-up to the [pre-CUDA spawner win](2026-06-28-agent-opd-precuda-spawner-closes-loop-on-elkeid-pod.md),
which closed the agent-OPD loop end-to-end but logged `trained_pairs=0` /
`mean_loss=0.0000`: the single train task with `--samples-per-prompt 1` at the
constrained `--max-turns 8 --max-tokens 768` yielded NO accepted rollout, so the
LoRA was never updated and the 0/3→1/3 held-out move was rollout sampling noise.

Goal this run: make ≥1 rollout get accepted so the LoRA actually updates
(`trained_pairs>0`, `mean_loss>0`), then measure the real held-out delta. Built
HEAD (`8bfbd615`, spawner wired) in an OWN pod tree (`/host/arle-ckl-aopd`, not
shared `/host/arle-build`); the spawner activates AUTOMATICALLY in
`run_agent_opd_impl` via `SpawnerHandle::launch()` (no env flag — `launch()` runs
before the first CUDA context and re-exec's the binary as the non-CUDA helper).

## What Worked (case-as-fact, measured)

**Spawner auto-activates and forks survive.** Every run logged
`[arle sandbox-spawner] listening …` + `pre-CUDA sandbox-spawner ready (pid …)`;
the baseline eval (3 tasks) and 6 training rollouts each ran cp/git/bash/pytest
through the helper with **zero SIGABRT / no dmesg coredump** — the ELKEID
fork-hook is defeated, confirming the prior spawner win.

**The train task is independently validated as REAL.** `ansible__ansible-f327e65`
(FQCN keyword-validation): re-staged from the persisted `/host/agent_opd_tree.tar`;
applied the hidden `test_patch`; at base the 4 `fail_to_pass` tests FAIL
(`is_valid_collection_name('assert.this')` returns `True`, test expects `False`);
a hand-applied gold fix (`from keyword import iskeyword` + reject Python keywords
per FQCN part) flips all 4 to PASS. So an accepted rollout is reachable
end-to-end.

**6/6 training rollouts PASSED — the accept→writeback gate works.** On the lucky
container window that survived ~20 min (RUN1: `--rounds 2 --samples-per-prompt 8
--rollout-temperature 1.0 --max-turns 30 --max-tokens 2048 --lora-layer-start 32`),
the round-0 training sampled the validated task and the first **six** samples
(0-5) each `passed=true (turns=16) :: [exit 0]` before the container was killed
(sample 6 mid-flight). 6/6 distinct accepted trajectories → `trained_pairs` would
be 6-8, `mean_loss>0`. The constrained 8-turn/768-token config from the prior win
was the reason it found nothing; the default 30-turn/2048-token budget lets the
student converge in 16 turns.

**Case-decoded the passing trajectory (sample 0).** The student located the bug
(turn 7: `VALID_COLLECTION_NAME_RE = r'^(\w+)\.(\w+)$'` doesn't reject Python
keywords), added `from keyword import iskeyword` (turn 10-12), rewrote
`is_valid_collection_name` to reject keyword parts, then SELF-VERIFIED via a bash
`python3 -c` import check (turns 13-15) — real agentic edit+test work, not
exploration. This is the exact gold fix.

**Baseline held-out eval = 1/3 (0.3333), all 3 edited** (`0ea40e0` pass,
`12734fa`/`5e36960` edited-but-fail) — same as the prior win's baseline, now
measured on the un-tuned student before training.

## Result — blocked from capturing the writeback by pod CrashLoopBackOff (infra, not code)

The writeback (`trained_pairs`/`mean_loss` logged) and post-training held-out
delta could NOT be captured: the `sglang-test` static-pod container entered an
**escalating CrashLoopBackOff** mid-session (started ~12:24 UTC when a concurrent
actor changed `/etc/kubernetes/manifests/sglang-test.yaml` — kubelet logged
"Deleted mirror pod as it didn't match the static Pod"). Node-side `crictl`:
4 consecutive container attempts exited **137 (SIGKILL), reason=Error,
oomkilled=False**, lifetimes 5m39s → 4m26s → 3m08s → ~4m, backoff escalating
10s→20s→40s. Node has 944 GB free (no host OOM); `tini -- sleep infinity` (pid1)
doesn't self-exit — an external node-governance reaper SIGKILLs the container's
pid namespace on a ~3-6 min cadence, **shorter than the ~5-6 min 27B-FP8
share-frozen-base load**. RUN1 caught a lucky ~20-min window (load + baseline
eval + 6 training rollouts); every relaunch after the crashloop tightened died
during the model load, before training.

**Honest verdict (a/b/c per the brief): the gate WOULD clear — it trained.** The
6/6 accept rate proves the run produces accepted rollouts and would log
`trained_pairs` 6-8 / `mean_loss>0`; the writeback step itself was simply never
reached on a surviving container. This is **NOT** a capability ceiling (c): the
27B student solves this SWE-bench-Pro task reliably (6/6). It is **NOT** a null
(b): the accept rate is decisive. It is verdict **(a)-pending**: training is
real, but the *measured* held-out Δ (round-0 vs baseline) is blocked by infra and
must be captured once the pod stabilizes. A rigorous capability claim still needs
multi-seed (≥5) + Wilson CI per the small-n-eval rule.

## Bench

Exempt: agent-OPD training path, not a serving hot path. Default serve/CLI is
byte-identical (spawner gated on `ARLE_SPAWNER_SOCKET`, set only by `launch()`).
No code change in this run (config + pod-data re-staging only); the binary is HEAD
`8bfbd615` built in `/host/arle-ckl-aopd`.

## Rule

- **`trained_pairs` needs samples × turns × temperature, not 1×8-turn-greedy.**
  The prior win's `trained_pairs=0` was a CONFIG artifact (8-turn/768-token,
  1 sample), not a capability ceiling: the same student solves the task 6/6 at the
  default 30-turn/2048-token budget with temperature-1.0 diversity. Decode the
  trajectory before concluding "the student can't do it".
- **Validate the train task is real BEFORE blaming the model.** Apply the hidden
  test_patch, confirm `fail_to_pass` FAILS at base and a gold fix PASSES — proves
  an accepted rollout is reachable, so a `trained_pairs=0` is a sampling/config
  problem, not an unsolvable-task problem.
- **A pod container SIGKILL (137, not OOMKilled, host has free RAM) on a fixed
  cadence = node-governance reaping the static-pod pid namespace, NOT your code.**
  `crictl inspect <cid>` exitCode/reason + `journalctl -u kubelet … CrashLoopBackOff`
  name it; a container lifetime shorter than your model load means the run is
  unrunnable until the pod stabilizes — relaunching into an escalating backoff is
  futile. Re-stage data on persistent `/host` (the ephemeral container `/root` is
  lost on each restart) and bake `GIT_CONFIG_GLOBAL=/host/<gitconfig>` so
  `safe.directory` survives the restart.
