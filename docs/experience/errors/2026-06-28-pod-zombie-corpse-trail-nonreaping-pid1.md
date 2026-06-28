# Pod zombie corpse-trail: 1809 defunct PIDs from a non-reaping pid 1

## Context

The 8×H20 sglang-test container accrued **1809 zombie (`Z`) processes**, 5+ days
old. Concern was GPU/capacity impact. Measured on-box (`~/bin/pod`):

- `pid_max = 4194303`, current procs+threads `1815` → zombies are **0.04% of the
  PID space. Zero PID pressure.**
- All 8 GPUs `0 MiB / 0%`; no live `arle`/`cargo`/build; container otherwise idle
  (the `loadavg 11.7` is **host-global** — `/proc/loadavg` is not cgroup-scoped —
  i.e. other tenants on the shared box, not us).
- Zombies hold no memory and no GPU — a zombie is only a PID-table slot until
  reaped. **Harmless to capacity.**

## Root Cause

`ps -p 1` → **`sleep infinity`**. The container's pid 1 is a bare keepalive, not
an init. **All 1809 zombies have `ppid=1`.**

When a process dies without its real parent `wait()`-ing on it, the kernel
reparents it to pid 1. A real init reaps reparented children continuously;
`sleep` never calls `wait()`, so every reparented child becomes a **permanent**
zombie until pid 1 exits (container restart).

The zombie `comm` breakdown is our own corpse-trail, not a leak in any one tool:
`bash` 578, `arle` 471, `sccache` 237, `tmux` 93, `kill` 92, `sleep` 41,
`python3` 38, `pkill` 35, plus the build chain (`rustc`/`nvcc`/`cargo`/
`cudafe++`/`cc1plus`). Two production sources:

1. **agent-OPD rollouts SIGABRT'd by ELKEID** on `fork()` (the fork-from-CUDA
   HIDS kill), leaving orphaned `bash`/`setsid`/`kill` children.
2. **Detached builds/runs killed by `pod.sh kill` (`kill -- -<pgid>`)** — the
   `setsid` group leader dies with the group, so any surviving child reparents to
   pid 1. Unavoidable for a killed detached job whenever pid 1 doesn't reap.

## Fix attempted 2026-06-28 — tini init REVERTED (ELKEID kills foreign pid 1)

The box is a **k8s static pod** (`/etc/kubernetes/manifests/sglang-test.yaml`,
kubelet-managed; `/host` = hostPath `/root`, persistent). pid 1 was
`command: ["sleep","infinity"]`. Attempt:

1. Staged static `tini` (v0.19.0) at `/host/bin/tini`; backed up the manifest
   (`/root/sglang-test.yaml.bak-pretini`) and `/work` → `/host/work-backup-pretini`.
2. Changed the command to `["/host/bin/tini","--","sleep","infinity"]`; kubelet
   recreated the pod (~35 s). **Initial result looked good:** pid 1 = tini,
   **zombies 1809 → 0**, orphan-spawn test reaped to 0, `/host` + GPUs intact.

**But it CRASHLOOPED.** The container was then **SIGKILL'd (exit 137) every
~3.5 min** (`crictl` ATTEMPT climbed to 8) — NOT OOM (node had 1.8 TB free, no
dmesg/kubelet OOM). The box had been stable 5 days on `sleep infinity`; the only
change was the command. Cause (high-confidence): the **ELKEID kernel HIDS kills
the unrecognized static `/host/bin/tini` running as pid 1** — same class as the
fork-hook that SIGABRTs CUDA forks on this box ([[reference_h20_pod_elkeid_kills_cuda_forks]]).
A foreign binary as pid 1 is exactly what a HIDS flags.

**REVERTED** to `["sleep","infinity"]` (restored the backup manifest) → fresh pod,
ATTEMPT 0, stable again. A crashlooping shared GPU box is far worse than the
harmless zombies, so stability wins.

**Side effect of the recreate:** the rust toolchain lived in the container's
**ephemeral** root (`~/.rustup`/`~/.cargo`), so each recreate wiped the pinned
1.95.0 (only the image's 1.96.0 stable survives). With the SOCKS build-proxy also
down, rebuilds are blocked until the proxy returns or the toolchain is local-fed.
Lesson: **put the toolchain on `/host`** (persistent) so a recreate never wipes it.

## Fix — STILL OPEN (needs devops / HIDS-trusted init)

A reaping init is the right permanent fix, but on this ELKEID box it must be a
**HIDS-trusted** init, which is outside container reach:
- devops whitelists `tini` in ELKEID, then the manifest-command swap above works; OR
- the container runtime is configured with a runtime-provided init (`--init`-style,
  a trusted path) instead of a foreign `/host` binary; OR
- bake a trusted init into the image.

Until then: `sleep infinity` (stable, harmless zombies slowly re-accrue — verified
zero capacity impact: 0.04 % of pid_max, no GPU/mem held).

**Prevention we own (no restart):** the pre-CUDA sandbox-spawner
([`crates/train/src/spawner.rs`]) removes source (1): the helper is a non-CUDA
process that owns all rollout spawns and reaps each child via `wait()` /
`kill_group`, so an ELKEID-free, leak-free agent-OPD loop no longer orphans
`bash`/`setsid`/`kill` to pid 1.

## Rule

A container whose **pid 1 is `sleep infinity`** reaps nothing — every orphaned
descendant becomes a permanent zombie. Before blaming a tool for "leaking
processes", check `ps -p 1 -o comm` and `ppid` of the zombies: `ppid=1` + a
non-init pid 1 means the tool is fine and the **init is the bug**. Zombies cost
only a PID slot (check `pid_max`), so triage capacity impact before urgency. The
durable fix is a reaping init (`tini`/`--init`); userspace can only stop *future*
orphaning in code it controls (reap your own children), never reap what already
reparented to a non-reaping pid 1.
