# `pod.sh run` auto-fires the serve model pin — no more forgotten first-command

> Measured 2026-08-03 on the 8×H20 pod. `pod.sh run ... serve` fires an
> idempotent `mlock` of the model dir at launch; verified +30.2 GB `Mlocked`
> == full Qwen3.6-27B-FP8, and a second serve of the same model forks **0** new
> pin processes. NOT measured: the cold-boot wall-clock after automation — the
> pin runs detached and the first serve races the 274 GB pre-read, so this
> guarantees the *trigger* fires, not that the first boot is warm.

## Context

[2026-07-25](2026-07-25-pin-model-weights-in-ram.md) proved `mlock` kills the
25-min cold boot, but left a manual step: *"re-run it after any pod restart — it
is the first command of a session."* Manual first-commands rot. A forgotten pin =
a silent 25-min cold boot, indistinguishable from a hang. The residency win was
real but its trigger was a habit, not the runtime.

## What Worked

Move the trigger into the one path that always precedes a weight load —
`scripts/pod-remote-run.sh`, the `run` op, right after `pod-build-env.sh` sources
and before `reap_run.py` launches. Three guards, no new dependency:

- **Serve-only**: a new `serve_model_dir()` parses `--model-path` from the
  recorded `argv.nul` (reusing the existing `serve_port()` NUL-parse pattern) and
  returns empty for any non-`serve` command — `train`/eval runs never pin.
- **Local-dir only**: `[ -d "$model_dir" ]` skips HF-ID model-paths (nothing to
  mlock; they resolve elsewhere).
- **Idempotent**: `pgrep -f "pin_model_cache.py $model_dir"` — an already-running
  pin for that dir means skip the fork, so re-serving never double-locks the RAM.

The pin runs `setsid nohup` detached and holds; the serve then hits the warm
page cache. `pin_model_cache.py` is unchanged and git-tracked, so `pod.sh sync`
ships it.

## Evidence (pod, isolated tree, run `autopin_r5`, GPU 3)

Validated on `Qwen3.6-27B-FP8` (29 GB, currently un-pinned) rather than the
274 GB DSv4 which was already pinned by two standing processes and needs
multi-GPU to serve — the auto-pin code is model-agnostic, so the smaller model is
an equivalent test of the trigger.

1. **Fires**: run log — `auto-pin: launched for /host/Qwen3.6-27B-FP8`.
2. **Locks**: `Mlocked` 317903060 → 348109540 kB, **Δ 30.2 GB == whole model**;
   pin log tail `resident: 30.9 GB, VmLck 30.9 GB`.
3. **Serve reaches ready over the warm cache**: `curl /v1/models` →
   `{"id":"Qwen3.6-27B-FP8",...}`.
4. **Idempotent**: a second serve (`autopin_r6`, port 18458) of the same model
   logged `auto-pin: launched` **0** times; exactly **1** pin process for that
   dir survived (PID 2907722). The standing DSv4 / ThinkingCap pins were never
   re-forked either — idempotent in production, not just in the smoke.

## Rule

- Automate the residency trigger at the path that always precedes the cost, not
  as a session habit — a "remember to run X first" step is a latent 25-min
  regression waiting for one forgetful boot.
- Idempotency is the whole safety of an auto-fork on a hot path: guard on
  `pgrep -f "<script> <arg>"` so re-entry is a no-op, never a second 274 GiB
  `mlock`. Verified by re-running, not by inspection.
- Shared-tree hazard, re-confirmed: `pod.sh`'s `source_digest` hashes tracked +
  untracked files, so a collaborator's live untracked script in the tree root
  drifts the digest and trips every build/run guard. Use an isolated
  `POD_TREE=/host/arle-build-<tag>` for a parallel session; never `pod.sh sync`
  the shared tree out from under a running job.
