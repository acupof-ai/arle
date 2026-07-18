#!/usr/bin/env python3
"""Subreaper wrapper for pods whose PID 1 never reaps orphans.

The H20 pod's PID 1 is `sleep infinity` (a keep-alive, not a reaping init), so any
arle TP worker that is orphaned — e.g. when a hung-at-startup coordinator is
SIGKILLed — reparents to PID 1 and leaks as a PERMANENT zombie (only the parent
can wait() it, and `sleep` never does). Hundreds accumulate across serve attempts.

This wrapper runs <cmd> in its own process group with PR_SET_CHILD_SUBREAPER set,
so orphaned grandchildren reparent HERE and are reaped, never leaked. Signals:
1st SIGTERM/SIGINT -> SIGTERM the cmd's group (graceful shutdown reaps workers);
2nd -> SIGKILL the group (the wrapper, still alive, reaps the orphans). The
wrapper exits only once the whole tree is gone, propagating the cmd's exit code.
"""
import ctypes
import os
import signal
import sys

PR_SET_CHILD_SUBREAPER = 36

if len(sys.argv) < 2:
    sys.exit("usage: reap_run.py [<operation-id> --argv-file <path>] <cmd> [args...]")

if len(sys.argv) >= 5 and sys.argv[2] == "--argv-file":
    operation_id = sys.argv[1]
    with open(sys.argv[3], "rb") as argv_file:
        raw = argv_file.read()
    args = raw.split(b"\0")
    if args and not args[-1]:
        args.pop()
    command = [sys.argv[4], *(os.fsdecode(arg) for arg in args)]
else:
    operation_id = "legacy"
    command = sys.argv[1:]

libc = ctypes.CDLL("libc.so.6", use_errno=True)
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    sys.exit("prctl(PR_SET_CHILD_SUBREAPER) failed: " + os.strerror(ctypes.get_errno()))

child = os.fork()
if child == 0:
    os.setpgid(0, 0)  # own process group so we can signal the whole tree
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        sys.stderr.write(f"reap_run[{operation_id}]: exec {command[0]} failed: {exc}\n")
        os._exit(127)
os.setpgid(child, child)  # set in the parent too (race-free with the child)

_terms = 0


def _on_term(_signum, _frame):
    global _terms
    _terms += 1
    sig = signal.SIGTERM if _terms == 1 else signal.SIGKILL
    try:
        os.killpg(child, sig)
    except ProcessLookupError:
        pass


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)

main_status = 0
while True:
    try:
        pid, status = os.waitpid(-1, 0)  # reap ANY descendant (incl. orphans)
    except ChildProcessError:
        break  # whole tree reaped
    except InterruptedError:
        continue  # woken by a signal; re-wait
    if pid == child:
        main_status = status

sys.exit(os.waitstatus_to_exitcode(main_status))
