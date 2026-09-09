# The pod container has no /dev/fd, so process substitution reads nothing

## Context

`pod.sh run <build> <label> auto` reported "no free GPU" while five of the
eight H20 cards were idle. Requesting a card by index worked; only the `auto`
path failed.

## Root cause

`scripts/pick-gpu.sh` fed its selection loop with `done < <(load_gpus)`. Bash
implements process substitution through `/dev/fd`, and the pod container does
not have it. The redirection fails, the loop body never executes, and the
script reaches its final `echo NONE` — the same output it prints when every
card is genuinely busy. Nothing distinguishes "no card is free" from "the
enumeration never ran".

`scripts/pod-remote-run.sh` had the same construct at its `status|log|kill`
loop, where an empty read is reported as "no operation: <label>".

## Fix

Both loops read a here-string produced by command substitution, which uses a
temporary file rather than `/dev/fd`:

    gpu_rows="$(load_gpus)" || { echo NONE; exit 1; }
    while IFS=',' read -r idx uuid used _; do ... done <<< "$gpu_rows"

The producer's failure is now separable from an empty result. Both loops
already skip empty lines, so a here-string over an empty variable keeps the
existing "found nothing" behaviour.

A `ln -s /proc/self/fd /dev/fd` in the container is the equivalent runtime
patch and was used to unblock a run, but it is lost on container restart and
leaves the next person with the same silent failure.

## Rule

An environment gap gets closed in the code that runs there, not by patching the
environment. And a selection loop whose "found nothing" branch is also its
"could not look" branch will report a full machine as busy — separate the
producer's failure from an empty result.
