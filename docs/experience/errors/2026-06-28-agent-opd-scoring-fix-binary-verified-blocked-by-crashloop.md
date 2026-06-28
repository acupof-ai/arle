# agent-OPD scoring fix (`ccedd788`) verified IN BINARY, but the value run is re-blocked by the node-governance crash-loop at dual-27B load — checkpoint not reached

## Context

Mainline task: verify the just-committed scoring fix (`ccedd788`,
`crates/train/src/sandbox.rs` — reset hidden `test_patch` target paths to base
before `git apply`, so a rollout that dirties a hidden test file no longer breaks
`score_workdir` with "patch does not apply") makes agent-OPD actually train
(`trained_pairs>0` / `mean_loss>0`), then capture the first value signal
(held-out pass-rate baseline→round-1).

Build + launch were clean and the fix is provably in the running binary:

- Synced local HEAD `ccedd788`'s `sandbox.rs` (only file the fix touches; hash
  `2e340e32`, matches local) onto the pod build tree `/host/arle-ckl-aopd` (was at
  divergent `7ae42221`, pre-fix `sandbox.rs` hash `0c03c9b9`) via `tn push` to the
  node path `/root/arle-ckl-aopd/...` (container `/host` == node `/root`).
- Incremental rebuild `cargo build --release --features cuda --bin arle` under its
  own flock → **BUILD_EXIT=0**, binary mtime fresh (`15:50:53 UTC`), and
  `strings target/release/arle | grep -c "git checkout test path"` = **1** (the
  fix's unique string literal; the stale binary had **0**).
- Marker `/host/arleCKL` → `…/target/release/arle` resolves to the new binary.
- Launched via `exec -a arleCKL …` on **free GPU 1** (`CUDA_VISIBLE_DEVICES=1`,
  verified 0 MiB). `ps -ef | grep arleCKL` showed the tagged process; the run log
  printed `fix_string_in_binary=1`. Config exactly as briefed:
  `--samples-per-prompt 4 --writeback-cap 1 --rounds 1 --eval-every 1
  --eval-temperature 0.0 --rollout-temperature 1.0 --max-turns 16 --max-tokens 768
  --lora-layer-start 32 --rollout-num-slots 1 --bash-timeout-secs 120
  --test-timeout-secs 240`, data + outputs on `/host` (`…_ckl2`).

## Root Cause

**The scoring checkpoint was never reachable** — the run dies during model
loading, ~47–80 s after launch, long before any rollout / scoring / writeback.
3 fresh reproductions (attempts on free GPU 1, each setsid-detached), all with the
identical signature:

- last log line is always `loading student from /host/Qwen3.6-27B-FP8`
- **GPU 1 never allocates** (0 MiB at death — so NOT an htod OOM, which would log
  `CUDA_ERROR_OUT_OF_MEMORY` after partial allocation, as `agentopd_run.log` did)
- **no `RUN_EXIT`, no Rust panic, no CUDA error** — abrupt silent termination
- the viewing `crictl exec` returned 137 in one poll (exec-teardown reap), but the
  `arleCKL` process itself was independently gone from a fresh `ps`, i.e. the
  detached process was reaped, not just the tunnel.

This is the **same node-governance / container crash-loop wall** documented in
`2026-06-27-agent-opd-full-loop-killed-by-node-governance-not-code.md` (5 prior
reproductions: silent external SIGKILL at high GPU residency, kill point drifts
with load speed, correlates with wall-clock not a code line). My 3 attempts are
reproductions 6–8. **NOT the scoring fix and NOT ARLE code** — the binary carries
the fix; the loop simply cannot survive the dual-27B (~58.8 GB resident) load in
the current sustained bad phase. No ~50-min good window appeared across the
~5-minute retry window.

## Fix

None to ARLE code — the fix under test is correct and present. To actually capture
the value signal, the dual-27B load must clear the crash-loop:

- Retry during a **good window** (the box alternates 2–3 min bad / 30+ min good per
  the crash-loop entries) — same marker + free-GPU procedure; the run wrapper is at
  `/host/run_aopd_ckl2.sh`.
- Or run on a **non-ELKEID box** (the governance kill is node-level), per the
  `reference_h20_pod_elkeid_kills_cuda_forks` memory.

The scoring fix itself remains unverified end-to-end **only because the harness
can't reach the scoring path** — its unit regression
(`score_resets_student_dirtied_test_file_before_applying_patch`) and the diagnostic
(staged base correct + patch applies to a clean tree) already cover the mechanism.

## Rule

A binary-verified fix (`strings | grep <unique-literal>` = 1, fresh mtime) is the
SOLID proof the build took — separate that from the run reaching its checkpoint.
When the same silent-SIGKILL-at-load signature recurs (no RUN_EXIT / panic / CUDA
error, GPU never allocates), it is the node-governance crash-loop, not your change;
do not re-debug code — confirm the binary, then either wait for a good window or
move to a non-ELKEID box. Verdict here = **(c) still blocked at the exact same
pre-scoring wall**, fix confirmed present but its `trained_pairs>0` checkpoint
unreached.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
