# Pod DSv4 verify pipeline — recipe + hard-won lessons (kept as experience, NOT shipped)

## Context
Built a local pod verify pipeline (sync→build→serve→probe) for DSv4-Flash B=1/perf
iteration. **Not committed as a script** (hardcoded pod paths = personal tool; violates
no-absolute-pod-paths-in-repo). Per ckl, the value is the recipe + the bugs hit — recorded
here as experience, the script stays local.

## What Worked (the recipe)
sync (`git stash create` → `git bundle` → `tn push` → pod `git fetch`/`reset --hard`) →
build (`dsv4_fast_build`, gate on the script's own `BUILD_EXIT` marker, NOT a wrapper echo)
→ serve TP4 GPUs 0-3 via `setsid` (kill-group-safe by unique port, never `pkill`) → probe
(a PUSHED python script does ready-retry + needle + tok/s).

## Hard-won lessons (each cost real time this session)
1. **Don't inline complex commands through `tn→kubectl→bash -lc`.** A curl whose JSON
   payload has escaped quotes breaks across the multi-shell layers → the ready-check looped
   ~7.5 min on an already-up serve. Push a SCRIPT instead.
2. **ready-wait MUST check serve-death** (ps the serve pid) or it polls a crashed serve ~10 min.
3. **After `kill -9` a CUDA serve, wait ~10 s before relaunch.** SIGKILL leaks GPU memory
   briefly (zombie allocation, `[Not Found]` in nvidia-smi); a new TP serve OOMs on the
   not-yet-reclaimed memory. The driver reclaims after a few seconds.
4. **nsys: profile the SERVE under nsys, not the curl** (profiling the curl captures no GPU
   work). Multiproc TP workers are fork+exec → `--inherit` misses them (per-rank wrap needed);
   the stats export is flaky → explicit `nsys export` + verify a FRESH rep (it silently
   re-reads a stale rep → byte-identical "results" that are last run's).
5. **High-concurrency: an ad-hoc threaded client can't verify real engine batching and is
   unstable** (0-token runs). Use the canonical `scripts/bench_guidellm.sh`, not a hand driver.
6. **Warm THOROUGHLY before any cold-vs-warm delta.** Cold-start/JIT swamps the signal — it
   produced a phantom `+666%` decode-TPOT artifact AND the prefix-reuse "no-reuse" flip-flop;
   both vanished with proper warmup.

## Rule
A pod iteration tool stays LOCAL — ship the LESSONS as experience, not the path-baked script.
For committed perf/throughput numbers the canonical tool is `scripts/bench_guidellm.sh`.
