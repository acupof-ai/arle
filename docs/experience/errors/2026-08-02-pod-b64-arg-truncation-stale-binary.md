# Oversized base64 arg silently truncated → stale pod binary → phantom CUDA error — 2026-08-02

## Context

Debugging a `CUDA_ERROR_NOT_SUPPORTED` on the FlashQLA chunked serve path.
Standalone kernel probes passed; the serve kept failing; three rounds of
`eprintln` instrumentation "shipped and rebuilt" yet never fired.

## Root Cause

The file-ship channel `~/bin/pod "echo '$B64' | base64 -d > file"` breaks at
large sizes: a 536 KB base64 payload exceeds the crictl-exec arg limit and is
**silently truncated**, so the pod's `qwen35.rs` no longer matched local, and
successive "rebuilds" compiled stale or corrupt trees while the identical
`Finished in 7m 11s` line (from an old log section) masked that some builds
never ran. Every observation after the first bad ship was made on a binary
that did not contain the code being reasoned about. The original error was
never a kernel problem — on the first verified-md5 clean build, the chunked
path served correctly.

## Fix

Chunked transfer + verify: split the base64 into <100 KB pieces appended on
the pod, then **`md5sum` on the pod against the local hash before building**,
and give every build log a unique marker (`BUILD2_EXIT=$?` in a fresh file)
so "finished" provably belongs to this run.

## Rule

After any pod file ship, md5-verify before building; a build gate is a unique
exit marker in a fresh log, never a familiar-looking tail. (Extends
[feedback_build_exit_marker_not_wrapper_echo] and the ship-channel notes in
[feedback_podsh_sync_only_modified_use_tarball].)
