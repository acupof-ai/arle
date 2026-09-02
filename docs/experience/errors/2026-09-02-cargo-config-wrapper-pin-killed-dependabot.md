# A repo-level sccache pin silently killed every Dependabot cargo run

## Context

Triaging two open Dependabot PRs (#245 pip, #246 github_actions). The pip and
github_actions lanes were green; the cargo lane showed `failure`. Checking its
history, every cargo run had failed:

    2026-09-02 failure cargo    2026-09-01 failure cargo
    2026-08-21 failure cargo    2026-08-01 failure cargo

The last cargo Dependabot PR to reach the repo was #51 on 2026-06-03. Nothing
surfaced this: the pip and github_actions lanes kept opening PRs on schedule, so
the ecosystem looked alive from the PR list alone.

## Root Cause

`.cargo/config.toml` carried a repo-level wrapper pin, added 2026-06-30 in
`358335265` to speed up local rebuilds (infer-util re-check 14.6s -> 0.35s):

    [build]
    rustc-wrapper = "sccache"

Dependabot's `dependabot-updater-cargo` container has no sccache, and the pin is
repo-level, so it applies to every cargo invocation on every host. Resolving the
dependency graph dies before it starts:

    Handled error whilst updating serde: dependency_file_not_resolvable
    {message: "error: could not execute process `sccache .../bin/rustc -vV`
    (never executed)\n\nCaused by:\n  No such file or directory (os error 2)"}

Matched A/B on the same tree, only PATH varied — `cargo metadata
--format-version 1`:

| sccache on PATH | exit |
|---|---|
| yes | 0 |
| no  | 101 |

`--no-deps` exits 0 in both arms, which is why a casual check misses it: the
wrapper is only invoked once the full graph is resolved.

The reach was wider than the local-dev speedup it bought. Four CI lanes had
already each papered over it with `RUSTC_WRAPPER: ""`, and
`scripts/pod-build-env.sh` plus `setup.sh:573` carry their own detect-and-clear
branches — five workarounds for one pin, none of which covers a container the
repo does not control. The cost was not slow builds; it was that RUSTSEC
security-update PRs, which arrive through this same lane regardless of schedule,
had nowhere to land for two months.

## Fix

Delete the pin; make sccache an opt-in per environment that has it.

- `.cargo/config.toml`: no `[build] rustc-wrapper`, with a comment naming this
  failure so it does not come back.
- `.github/workflows/metal-ci.yml`: the one CI lane that wants sccache now sets
  `RUSTC_WRAPPER: "sccache"` itself. `mozilla-actions/sccache-action` installs
  the binary but does not export the variable, so the lane was relying on the
  pin.
- The pod (`scripts/pod-build-env.sh`) already exported `RUSTC_WRAPPER=sccache`
  when sccache is present, so it needed no change beyond a corrected comment.
- The `RUSTC_WRAPPER: ""` overrides in ci / cargo-deny / release stay as
  belt-and-braces; their comments no longer describe a pin that is gone.

Verified by the same A/B: the sccache-absent arm goes 101 -> 0, and the
opt-in arm stays 0.

## Rule

Never put a `rustc-wrapper` (or any tool-path pin) in a committed
`.cargo/config.toml`. It applies on hosts the repo does not control — CI
containers, Dependabot updaters, a fresh clone — and turns a missing optional
binary into a hard failure in `cargo metadata`. Developer-machine
accelerators belong in `~/.cargo/config.toml` or the shell rc; a lane that
wants one sets its own env var.

Corollary: when the same pin accumulates per-environment workarounds
(`RUSTC_WRAPPER: ""` in four workflows, two detect-and-clear branches in
scripts), the workarounds are the signal. Each one covers a host someone
noticed; the failure lands wherever nobody was looking.

Second corollary: a scheduled bot lane that fails produces no PR and no
notification, which is indistinguishable from "no updates available". Check the
lane's run history, not its PR list.
