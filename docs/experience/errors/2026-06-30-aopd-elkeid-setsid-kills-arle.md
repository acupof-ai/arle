# Agent-OPD: ELKEID kills arle via setsid() ancestry hook

## Context

Agent-OPD training loop on the H20 pod. After `--no-share-frozen-base` model
loading completes (both models loaded, 72875 MiB used, warmup done), arle exits
with code 137 (SIGKILL) before any request is processed.

## Root Cause

ELKEID's eBPF hook kills arle when any process in the syscall's ancestor chain
is CUDA-resident AND the `setsid()` syscall fires. The pre-CUDA sandbox spawner
helper's `run_captured` was calling:

```rust
let mut command = Command::new("setsid");
command.arg(&req.program).args(&req.args);
// ...
let mut child = command.spawn()?;
```

This spawns the `setsid` binary. `setsid` calls `setsid()` syscall to become a
new session leader, then exec's the target program. ELKEID hooks `setsid()`,
walks the ancestor chain of the calling process (setsid → spawner → arle),
finds arle is CUDA-resident, and sends SIGKILL to arle.

The spawner itself is not CUDA-resident (launched before CUDA init), but ELKEID
checks the ANCESTOR chain, not just the immediate parent.

## Diagnosis Path

**Evidence 1 — synthetic writeback succeeds (exit 0):**
`--synthetic-writeback-seq 1024` skips all subprocess calls → completes with
`loss=7.380744`. This proved both models load fine and the writeback works.

**Evidence 2 — bad staged path exits 1 (not 137):**
`--staged-root /host/STAGED_DOES_NOT_EXIST` → spawner runs `cp -a` → cp returns
exit code 1 (no such file) → arle gets Rust error → exits 1. No SIGKILL.
This proved the spawner's fork of `cp` (plain `cmd.output()`, no setsid) is safe.

**Evidence 3 — timing delta isolates setsid:**
- bad staged (cp fails before setsid): 14970ms total
- valid staged (exit 137): 16342ms total
- delta = 1372ms ≈ time for cp + git init/add/commit + first `setsid bash`

The 1372ms gap covers the boot_workdir subprocess sequence. The exit 137
happens at the first `setsid bash` call (overview command in
`run_agentic_opd_round`), not at the plain cp/git calls.

**No dmesg evidence:** Exit 137 leaves no kernel log → rules out Linux OOM killer
and GPU Xid faults. ELKEID uses `bpf_send_signal(SIGKILL)` which bypasses dmesg.

## Fix

`crates/train/src/spawner.rs` — replace `setsid` binary usage with
`process_group(0)` (calls `setpgid`, not `setsid`), and replace `kill` binary
fork with `libc::kill(-pgid, SIGKILL)`:

```rust
// Before: two-fork chain that triggers ELKEID
let mut command = Command::new("setsid");
command.arg(&req.program).args(&req.args);
// ...
kill_group with Command::new("kill").arg("-KILL")...

// After: direct spawn in new process group
let mut command = build_command(req);
command.process_group(0);  // setpgid, not setsid
// kill_group:
unsafe { libc::kill(-pgid, libc::SIGKILL) };
```

Commit: `ea2e6133`

## Rule

When routing subprocess spawning through a pre-CUDA helper to avoid ELKEID's
fork hook, also ensure the helper does NOT call `setsid()` itself (via the
`setsid` binary or libc call). ELKEID's ancestry check on `setsid()` reaches
back through the helper to the CUDA-resident parent.

Use `process_group(0)` (`setpgid`) instead of the `setsid` binary; use
`libc::kill(-pgid, SIGKILL)` instead of forking an external `kill` binary.
