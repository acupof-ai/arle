# agent-OPD full-loop e2e on the H20 box dies by SILENT external SIGKILL at the dual-27B ~58.8 GB resident floor — node governance, NOT an ARLE code bug

## Context

Mainline task: run the agent-OPD (train-infer-unified) FULL loop end-to-end on
the 8×H20 box with the new held-out eval (commit `7dff5484`) + the fast GPU CE
writeback (commit `65a46817`) — verify the loop integrates (rollout → reward →
writeback → LoRA-sync → eval → repeat) and measure held-out pass-rate trend.

Build green on the pod (`BUILD_EXIT=0`, binary carries the `agent-opd-eval`
symbol; pod build tree had been behind — synced 172-file tarball from local HEAD
`65a46817` onto ancestor `1dd0f46b` before building). Data set up + **locally
validated as real tasks** (each: test fails at `base_commit`, passes with the gold
`patch`): TRAIN = the pod's existing 1 task `ansible__ansible-f327e65`; EVAL = 3
held-out ansible tasks (`0ea40e0`, `12734fa`, `5e36960`) staged from
`ScaleAI/SWE-bench_Pro`, fetched + git-archived at each `base_commit` locally and
pushed to the pod. Train/eval `instance_id` overlap = NONE.

`arle train agent-opd --student-model /host/Qwen3.6-27B-FP8 ... --share-frozen-base
(default) --lora-layer-start 32 --rollout-num-slots 1` on a free H20 GPU.

## Root Cause

**The run is killed by an external SIGKILL — a node-level governance kill, not an
ARLE code defect.** 5 reproductions, all identical signature: the process dies
with **no `RUN_EXIT`, no Rust panic, no CUDA error**, somewhere in the window from
mid-student-load through the round-0 baseline-eval start. The kill point drifts
with load speed (mid-load under `CUDA_LAUNCH_BLOCKING=1`, at the resident-floor log
otherwise) — i.e. it correlates with **wall-clock at high GPU residency**, not a
specific code line.

Isolation (single-variable, each in a persistent tmux so the kill is not exec
teardown):

| Workload (footprint) | Result |
|---|---|
| Rollout engine load (48.7 GB) + autograd student load (→ **58.8 GB resident, 38 GB free**) | load **succeeds**, then killed |
| `--synthetic-writeback-seq` (release engine, ~40 GB, chunked-CE fwd/bwd) | **survives** 60+ layers (collaterally killed only when it shared a tmux server with a separately-killed run) |
| bare `arle run` single engine (~87 GB) | **survives** minutes |
| `arle run` + **bash tool subprocess** at GPU residency (`echo HELLO_FROM_TOOL`) | **clean RUN_EXIT=0** — falsifies the "HIDS kills on subprocess-spawn" hypothesis |
| sandbox `cp -a` + `git init` + `pytest` standalone (no GPU process) | **survives** |
| **agent-OPD full** (dual-27B 58.8 GB resident + eval forward) | **killed**, silent, 5/5 |

Ruled out, with evidence:
- **CUDA OOM** — 38 GB free at the resident floor; `CUDA_LAUNCH_BLOCKING=1`
  surfaced **no** CUDA error (a real OOM prints `engine build failed: upload FP8
  block-scaled tensor`, cf.
  [2026-06-23-opd-multiarm-dense27b-gpu-memory-ceiling](2026-06-23-opd-multiarm-dense27b-gpu-memory-ceiling.md)).
- **Host RAM OOM** — 1838 GB free, cgroup `memory.max` unlimited,
  `memory.events` empty.
- **`~/bin/pod` exec-teardown reaping** (the known 137/143 hazard) — reproduced
  the kill INSIDE a persistent, separately-socketed tmux server that survived for
  every single-engine case.
- **Subprocess-spawn-under-GPU** — falsified by the clean `arle run` + bash-tool
  case above.

What's left, and unique to the killed runs: a **sustained dual-27B ~58.8 GB GPU
residency** (rollout engine + autograd student co-resident) on a box whose kernel
log shows **`[ELKEID]`** (redacted HIDS) actively instrumenting executables, and
on which a foreign 87 GB GPU process vanished at the same time as the first kill.
The signature (silent SIGKILL of the whole process tree, uncatchable, no CUDA/host
OOM, footprint/wall-clock-correlated) is consistent with a **node-level GPU-memory
/ process governance policy**, not anything ARLE controls.

## Fix

Not an ARLE code fix — an infra/operational matter. Options, none landed here:
- Run on a box WITHOUT the governance kill (a plain H20 without the ELKEID/governor
  policy), or get the policy threshold raised for this workload.
- Shrink the resident footprint below whatever the governor tolerates: the
  `--share-frozen-base` zero-copy is already on (student aliases the engine's FP8
  base for layers < `lora_layer_start`); the next lever is **don't co-resident the
  full rollout engine** — e.g. tear the engine down to a smaller KV/static
  reservation during eval, or place engine vs autograd student on separate GPUs.
- A small-model loop-logic proof (e.g. a qwen35-arch ~4 B FP8 student) would close
  the loop end-to-end at a footprint under the governor — blocked here because the
  only small model on the box (`/host/Qwen3-4B`) is `Qwen3ForCausalLM`, NOT the
  `Qwen3_5ForConditionalGeneration` arch the OPD `load_qwen35_*` loader accepts.

## What DID verify (so the integration is not unknown, just not closed)

- Held-out eval data path is real + correct: `boot_workdir` (`cp -a` staged tree
  → `git init`/commit) + `score_workdir` (`git apply test_patch` → `pytest
  fail_to_pass`) reproduce the fail-at-base / pass-with-fix semantics for all 3
  eval tasks AND the train task (validated locally in a venv).
- Model load (both copies), the fast GPU chunked-CE writeback fwd/bwd, the AdamW
  step path, and bare engine decode each run on the box in isolation.
- The held-out guard (bails on train/eval `instance_id` overlap) and the
  round-0-baseline-before-training wiring are present in the binary.

The ONE step never observed running: the round-0 baseline **eval forward** at the
dual-27B resident floor — because the process is killed at that boundary.

## Rule

**A silent SIGKILL (no RUN_EXIT, no Rust panic, no CUDA error even under
`CUDA_LAUNCH_BLOCKING=1`) of a high-GPU-footprint process on a governed
(ELKEID/HIDS) box is an infra kill, not a code bug — STOP attributing it to the
code.** Discriminate with single-variable isolation (single-engine vs dual-model;
subprocess-spawn vs not; persistent-tmux vs exec-teardown) before writing any
"OPD loader OOMs" conclusion. Pre-flight any pod e2e by checking `dmesg | grep
ELKEID`; if the box governs the process, the OPD run needs a different box BEFORE
burning loop attempts.

---

## CORRECTION 2026-06-28 — mechanism is `fork()` from the CUDA process → ELKEID `tgkill(SIGABRT)`, NOT the "~58.8 GB dual-residency" footprint

Re-ran on the **fresh HEAD `748b082e`** binary (rebuilt from local HEAD in an
isolated tree `/host/arle-opd-final`, `BUILD_EXIT=0`, `strings | grep -x setsid`
present — the FIRST run that actually carried the setsid fork-safety fix; all
prior runs used a stale pre-setsid binary). Captured **node-side `dmesg`** (via
`tn exec`, not the container) at the kill — the prior entry never had this.

**Direct kernel evidence overturns the footprint theory:**
- `dmesg`: `Comm: forktest2 … Tainted: P  IOEL` + `Modules linked in: elkeid(OE)
  … [last unloaded: elkeid]`; live node procs `[elkeid - antiro]`, `[elkeid -
  dput]`, `/etc/elkeid/plugin/driver/driver`.
- The fatal path is `do_signal → get_signal → do_coredump → elf_core_dump →
  dump_user_range`, with `ORIG_RAX=0xea` (`tgkill`) and `RDX=R12=0x6` = **SIGABRT
  (signal 6)** — i.e. `tgkill(self, SIGABRT)`, an `abort()` driven by the elkeid
  fork hook, NOT an external SIGKILL.
- The 23 s `watchdog: BUG: soft lockup … [forktest2]` is the kernel **core-dumping
  the multi-GB CUDA process** (`ulimit -c unlimited`); `ulimit -c 0` removes the
  lockup but the process still dies — the abort is the killer, the dump is just
  collateral.

**Where it dies, exactly (4/4 runs, share AND no-share):** right after
`[opd-load-trace] post stream-sync OK; building InferStudent` — i.e. the load +
the `748b082e` stream-sync deadlock fix **both succeed** ("post stream-sync OK"
proves the fix works), then the round-0 **baseline eval**'s very first
subprocess — `boot_workdir`'s `Command::new("cp").output()`
(`crates/train/src/sandbox.rs:367`, a plain `run_checked` fork, BEFORE any bash
tool or model forward) — `fork()`s the multithreaded CUDA-resident process and
ELKEID aborts it. A `[kill] <defunct>` zombie in the pgid is arle's own
`kill_group` firing as the doomed parent unwinds.

**Two prior claims falsified:**
1. **"~58.8 GB dual-27B co-residency is the trigger"** — WRONG. The
   `--no-share-frozen-base` control (loads the student first, single-engine
   residency, NO co-resident dual model) died at the SAME first-fork boundary
   ("loading rollout engine", 9 s in). The trigger is **`fork()` from a
   CUDA-resident process**, independent of footprint and of share/no-share.
2. **"arle run + bash tool subprocess → RUN_EXIT=0 falsifies subprocess-spawn"**
   (the prior entry's table) — that control was confounded: it did not reproduce
   the same fork-from-fully-initialized-multithreaded-CUDA-state, so its clean
   exit did not actually clear the fork hypothesis. The node `dmesg` is ground
   truth: the abort is on `fork`.

**The setsid fix is correct but insufficient here.** It made the *bash-tool*
spawn fork-safe at the libc atfork level (no in-child `setpgid` window). But (a)
the FIRST fork in the loop is `boot_workdir`'s `cp`/`git`/`bash` via `run_checked`
(plain `Command::output()`), not the setsid `run_captured` path; and (b) ELKEID
aborts at the **kernel** fork hook regardless of libc-level fork-safety. No
source-level spawn change dodges a kernel HIDS fork hook.

**Residual blocker = the box's ELKEID policy aborts any `fork()` from this
CUDA-resident process.** Genuinely infra/node-governance (the prior entry's
top-line verdict stands), but the mechanism is fork-hook-abort, not
footprint-governor-SIGKILL. Levers, none landed: run on a box without the ELKEID
fork policy; get arle's `argv0` allowlisted in the ELKEID driver config; or make
the agent loop **never fork from the CUDA process** — pre-fork a subprocess
helper/agent BEFORE any CUDA init and drive all `cp`/`git`/`bash`/`pytest` through
it over a pipe (a fork-server), so the only forks happen from a CUDA-free child.

### Rule (correction)
- **Get node-side `dmesg`/`tn exec` before calling a silent kill "footprint
  governance".** The kernel log named the mechanism in one read (`forktest` +
  `elkeid` + `tgkill SIGABRT` + `do_coredump`) that 5 prior black-box repros could
  not — a HIDS fork-hook abort looks identical to a footprint SIGKILL from inside
  the container.
- **`fork()` from a multithreaded CUDA-resident process is the hazard on an
  ELKEID box — not the bash tool specifically, not the footprint.** The first
  plain `Command::output()` (`cp`) trips it. A libc-level fork-safety fix (setsid)
  cannot dodge a kernel fork hook; the structural fix is a pre-CUDA fork-server.
- **"Clearing one blocker exposes the next" cuts both ways — re-decode, AND
  re-test that the new binary actually changed the failure point.** Here the
  setsid + stream-sync fixes verifiably advanced the death from mid-load (deadlock)
  to first-fork (abort); confirm the advance with the trace, don't assume the bundle
  fixed the end-to-end.
