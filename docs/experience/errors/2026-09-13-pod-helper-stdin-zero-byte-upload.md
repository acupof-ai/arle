# $HOME/bin/pod does not forward stdin: a zero-byte script uploads, forks, and reports as started

Date: 2026-09-13. Surfaced firsthand on the gauge-presence-gates pod build;
entry only, not fixed in that lane.

## Context

To run a compile-only build detached on the pod, I wrote a shell script locally
and uploaded it through the remote helper with redirected stdin:

```
$HOME/bin/pod "cat > <remote>/g.sh && chmod +x <remote>/g.sh && echo WRITTEN" < /tmp/g.sh
# -> WRITTEN
```

The helper reported `WRITTEN`. I then launched it under `setsid`, which also
reported `LAUNCHED`. No build existed: each poll showed zero `rustc`/`nvcc`
processes and an empty log. The remote file was zero bytes:

```
-rwxr-xr-x 1 root root 0 <remote>/g.sh
```

`cat > file` on empty stdin creates the file and exits 0, so every stage
succeeded: the transfer "succeeded", the chmod "succeeded", the fork
"succeeded", and the zero-byte script exited instantly. Two launch attempts
produced nothing and looked fine at every step except in the result.

Uploading the same script via a base64 command-line argument instead of stdin
made the bytes arrive, and the build ran.

## Root Cause

The `pod`/`tn` transport does not attach local stdin to the remote command, so a
`cat > file <stdin` shape receives no data. Nothing along the chain treats
"received no bytes" as an error, and the launcher cannot distinguish "the child
finished instantly" from "the child is compiling" — both are a successful fork.

## Rule

A build marker is claimable only from a run you can show produced output. A
fork returning a PID and a launcher printing "started" prove a process was
created, not that any work happened; the evidence of execution is a non-empty,
advancing log (or an exit marker), and the evidence of a transfer is the remote
byte count, not a successful status. Prefer a non-empty-input check at the
boundary (`test -s file || exit`) over relying on the transport or the fork to
fail when zero bytes flow.
