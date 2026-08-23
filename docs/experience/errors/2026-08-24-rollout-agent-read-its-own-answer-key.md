# The rollout agent read the answer key

## Context

Agent-OPD held-out eval on the harder corpus (`eval_hard.jsonl`, gold patches
41-109 lines). Reviewing a trajectory for quality, not for correctness, I found
the agent had located the corpus and printed the `test_patch` it was about to
be scored against.

## Root Cause

There is no filesystem confinement. `cc_harness.rs` spawned the agent as
`Command::new("claude").current_dir(workdir)` — as root, with every path on the
box readable, and `IS_SANDBOX=1` set, which removes CC's own permission
prompts. The corpus was mode `-rw-r--r--`.

The trigger was task quality. `prettytable pr_312`'s problem statement is "New
color themes?" and never says which themes, so the task is not solvable from
its own statement. The agent tried `WebFetch` and `curl` for the upstream PR,
found no network, and then did the reasonable thing for an agent that cannot
infer the spec: it searched the filesystem for it. It found
`/host/opd-corpora/staged-sweetspot3/eval.jsonl`, grepped its own
`instance_id`, and printed the record — `test_patch` in full, `gold_patch`
alongside it. It also diffed a previous run's workdir (`/tmp/fp4rl-c/...`)
holding the same task.

Blast radius, measured rather than assumed:

| run | dumps | corpus path seen | answer key read |
|---|---|---|---|
| `fp4rl-b` | 242 | 0 | 0 |
| `fp4rl-c` | 206 | 0 | 0 |
| `fp4rl-d` | 89 | 17 (all one task) | 2 |

Confined to one task in one run, and that task was killed before it scored, so
no contaminated number landed. The earlier baselines stand.

A first count said every dump in all three runs had touched a peer workdir.
That was my grep matching each agent's own cwd. The number that mattered was
one pattern away from being 25x too large.

## Fix

- The agent runs as an unprivileged account (`--rollout-user`, default
  `arle-rollout`); its workdir and a HOME beside it are chowned to that user
  and mode 700. HOME sits outside the scored tree so CC's own config cannot
  read as an edit.
- Startup refuses to run unless `--dataset`, `--staged-root` and
  `--eval-dataset` are all unreadable by that account, probed with `setpriv`.
  A documented chmod would have been forgotten; the failure it prevents is
  silent.
- A scored workdir is deleted. One that outlives its score is the next run's
  answer key.

Verified on the box: all four corpus paths blocked for uid 999, root still
reads them.

## Rule

An agent with Bash is an untrusted process, and "it has no reason to look
there" is not a control. Assume it will read anything it can read, and put the
answers where it cannot — then assert that at startup, in the same run that
depends on it.

An under-specified task does not fail honestly. It converts the agent into a
search for the spec, and the spec is on the same disk.
