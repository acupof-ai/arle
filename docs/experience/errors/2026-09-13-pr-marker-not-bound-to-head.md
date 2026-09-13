# The PR precheck verifies the marker line exists, not that the measurement was taken at the head

Date: 2026-09-13. Surfaced on #422; not fixed in that lane.

## Context

`scripts/lane_pr_precheck.py` requires a line such as `CUDA_CHECK_EXIT=0`
(check_cuda_check_exit) or `CLIPPY_EXIT=0` (check_clippy_exit) in the PR
body when a diff touches a no-cuda-gated crate. The check is

```python
if CUDA_CHECK_EXIT_OK.search(pr_body) or not cuda_gate_triggered(...):
    return []   # pass
```

It greps for the line and nothing else. A body carrying a truthful number
measured at an earlier commit, and a body carrying a placeholder the
author intends to replace after the build, are indistinguishable to it.
On #417 a verification section was headed with `fd17552f2` while the PR
head was `444e0878` — the cited pod run predated the delta and therefore
said nothing about it; the precheck passed because the marker text was
present. The mismatch was caught only because a reviewer read the sha by
eye, which is not a mechanism.

## Root Cause

The check validates the *form* of the evidence (a recognized
`KEY=0` line) but not the two things that make it evidence: that the
measurement was taken at the head commit, and that it was taken at all.
This is the same class as the rest of this week's audit — a check whose
mechanism runs but whose input is not the quantity under test. The
shell-script analog was `grep -q needle` with no guarantee the command
producing the input ran; here the analog is `regex.search(body)` with no
tie between the body and the head sha.

The precheck is also run locally before push against a body file the
author writes, so even a correct sha check would be honest only to the
author's claim; it cannot independently know the build ran. The sha tie
narrows the lie (a stale measurement no longer satisfies) but does not
prove execution.

## Confirmed instance

2026-09-13, #434, merged. Its `BUILD_EXIT=0` was hand-written. The build ran
through a pipe (`cargo build ... 2>&1 | grep -vE ... | tail -20`) and the
status was never read; success was inferred from the output binary existing,
and the marker line typed to satisfy the rule. The two other markers in the
same body, `CUDA_CHECK_EXIT=0` and `CLIPPY_EXIT=0`, were captured correctly
from unpiped commands, so a single body carried both a measured marker and a
manufactured one, indistinguishable to the precheck and to the reviewer.

Found by asking the author how each marker was captured, not by any check.
That question is the only thing today that separated the two.

A second capture trap sits next to this one and produces the same false zero
without anyone typing it: `cmd 2>&1 | tee log; echo "BUILD_EXIT=$?"` reports
tee's status, so it is 0 whenever tee succeeds, in every shell. And on the
local mac the tool shell is zsh, where bash's `${PIPESTATUS[0]}` expands to
empty, so a guard built on it neither gates nor errors.

## The rule also runs on exactly one path, which the author chooses

Measured 2026-09-13, after the confirmed instance above. The three marker
rules are the only rules that need a PR body, and `--pr-body` is passed from
exactly one place: `scripts/lane.sh` at the `pr` subcommand. No GitHub
workflow invokes `lane_pr_precheck.py` at all, and the pre-push hook runs it
in `--push-content` mode, which carries the five body-less rules only.

A pull request opened any other way — `gh pr create`, the web UI, a push to
an existing branch — never runs them. This is not theoretical either. Run
against the lane for #436 with that PR's real body, the precheck refuses on
all three:

```
- build-exit: examples/ or benches/ changed but the PR body has no BUILD_EXIT=0 line
- cuda-check: diff touches Rust in a no-cuda-gated crate ...
- clippy-exit: diff touches Rust in a no-cuda-gated crate ...
```

`CRATE_RUST_PATH` is `^crates/([^/]+)/.*\.rs$`, so an examples-only diff in
`crates/infer-cuda/` matches, and `crates/infer-cuda/Cargo.toml` declares
`no-cuda`, so `cuda_gate_triggered` is true. The rule applies and did not run.

#433 was merged in this state, with no marker of any kind. #431 and #436 are
open in the same state. The controller who merged them is the same person who
wrote #428/#432 to stop direct-to-main pushes bypassing the other five rules,
and then bypassed these three by opening pull requests with `gh`.

The bounded version of the lesson: moving five rules onto the push left three
on a path chosen by the author, and "pull-request-only" was read as "runs on
every pull request" when it meant "runs if you use one script".

## Rule 5's marker cannot cover an examples-only diff; rule 7's can

Measured 2026-09-13 on the lane for #436, whose diff is one file under
`crates/infer-cuda/examples/`. Both markers were demanded, because
`CRATE_RUST_PATH` matches any `.rs` under a crate and `infer-cuda` declares
`no-cuda`.

The two rules name different commands, and only one of them reaches an example:

- Rule 5 wants `cargo check --features cuda,nccl`. `cargo check` selects
  library and binary targets; examples are built only under `--examples`,
  `--example <name>` or `--all-targets`. So a green `CUDA_CHECK_EXIT` on an
  examples-only diff reports on code the diff does not touch.
- Rule 7 wants `cargo clippy --workspace --all-targets --features cuda,nccl`.
  `--all-targets` does include examples, so that marker covers the diff.

The author error underneath is worth recording separately, because it is the
one that actually happened. Both markers were first supplied from the macOS
lint mirror in the agent contract,
`-p infer-api --release --no-default-features --features cuda,no-cuda,nccl,deepep --lib`.
Both exited 0 and both were honestly measured, but that command is neither of
the two the rules name: it is scoped to one package, restricted to `--lib`, and
run with `no-cuda`, the feature the rules explicitly exclude. The shared target
directory holds 701 fingerprint units, 16 of them `infer-cuda-*` and 0 for any
parity example — nothing on that machine had ever compiled the changed file.
The precheck accepted it, since it greps for `KEY=0` and cannot see which
command produced the number.

So the marker rules are stronger than the substitution made for them, and the
gap that remains is narrower than it first looked: rule 7 is sound as written,
rule 5 is blind on an examples-only diff, and a marker sourced from a
convenient local command is indistinguishable from one sourced from the
prescribed remote one.

## Fix

Entry only. Options, smallest first:
1. Require the marker line to include the head sha it was measured at
   (e.g. `CUDA_CHECK_EXIT=0 @<sha>`), parsed and compared against the PR
   head the hook/CI knows. A stale or placeholder sha fails.
2. Have the pod run emit a machine-readable artifact keyed by the source
   digest (the sync already computes one) and have the precheck key the
   marker to that digest, so the measurement is bound to the exact tree.
3. Longer-term, move the marker from a free-text PR body into a
   generated record the runner writes and CI reads, removing the
   author-asserted text entirely.

## Rule

A marker is evidence only when it is bound to the exact artifact under
review. Verifying that a status line is present verifies the line, not
the run; the line must carry the head sha (or source digest) and the
check must compare it. A mechanism that cannot tell a measured value
from a placeholder has the same hole whether the language is bash or
Python.
