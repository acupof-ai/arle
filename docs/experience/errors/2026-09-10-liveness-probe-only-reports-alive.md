# A liveness probe that can only report alive is worse than no probe

## Context

Two probes, two projects, same failure shape on the same night (2026-09-09/10,
H20 pod `iv-yeozpb5g5cbw80bls64e`):

1. A monitor polling a pod benchmark used `pgrep -f bench_throughput.py` inside
   the pod wrapper. The wrapper shell's own command line contains the pattern
   text, so `pgrep -f` matched the wrapper itself. The monitor reported
   `BENCH_RUN` for 60 minutes while the bench had died in one second (its script
   file was absent — a push had landed on the node filesystem, invisible inside
   the container). The GPU sat idle the whole night, claimed and believed busy.
2. A tileRL collector checked process liveness with `os.kill(pid, 0)` and
   `/proc/<pid>`. Both report a zombie as alive; only `ps -o stat=` shows the
   `Z` state. Finished runs were counted as running.

## Root Cause

Each probe encoded the answer "alive" into its construction: the `pgrep -f`
pattern matched the probe's own shell, and the existence checks matched a
process table entry that no longer executes. A probe that cannot return its
negative result does not measure liveness — it manufactures confidence. The
60-minute idle GPU was worse than an unmonitored one: with no monitor, the
empty benchmark log would have been checked directly.

## Fix

- `pgrep -f` patterns use a character class so the regex does not match the
  pattern text in the wrapper's own command line: `pgrep -f
  'bench_throughput[.]py'` matches the process name, not the literal
  `bench_throughput[.]py` the wrapper carries.
- Liveness asserts the artifact, not the process: the benchmark's output JSON
  gaining points, the serve answering `/v1/models`. A process that produces no
  artifact is dead by the only definition that matters.
- Where a process check is needed, `ps -o stat=` and reject `Z`.

## Rule

A probe must be able to report its negative result. A liveness check that can
only say alive differs from no probe by making you more certain; before
trusting one, ask what it reports when the target is dead, and test that branch
(kill the target, watch the probe fire).
