# The pre-push hook gave every lane its own target tree: 18.6 GB

## Context

Disk sat at 84% with 67 GB free. `crates/` in a lane worktree is 82 MB, but
each of the four lane worktrees measured 4.0–5.2 GB.

    ../arle-lanes/a  4.9G   (target/ 4.8G)
    ../arle-lanes/b  4.0G   (target/ 3.8G)
    ../arle-lanes/c  5.2G   (target/ 5.1G)
    ../arle-lanes/d  4.8G   (target/ 4.7G)

The lane-worktree design states the opposite: one shared cargo target
directory, "a lane costs ~97 MB of tracked files and no build artifacts",
enforced by `../arle-lanes/.cargo/config.toml` setting
`target-dir = <main checkout>/target`. `cargo metadata` run inside a lane does
resolve `target_directory` to the shared path, so the config is working.

## Root cause

`scripts/pre_push_checks.sh` exported

    CARGO_TARGET_DIR="${REPO_ROOT}/target/pre-push-quick"

and `REPO_ROOT` is the worktree root the hook fired in, not the main checkout.
An environment variable outranks `target-dir` from any config file, so every
lane's pre-push run built a complete dependency tree inside its own worktree —
the one place the design says holds no build artifacts. Ordinary interactive
builds in the same worktree kept using the shared directory, which is why
`cargo metadata` reported the intended answer and the disk said otherwise.

Four lanes plus the main checkout meant five copies of the same dependency
graph, none of them shared.

## Fix

Anchor the hook's target directory to the main checkout, found from the common
git directory rather than from the worktree:

    MAIN_ROOT="$(dirname "$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)")"
    export CARGO_TARGET_DIR="${MAIN_ROOT}/target/pre-push-quick"

The separate `pre-push-quick` tree stays — the hook builds a HEAD snapshot from
a different source path than the working tree, so sharing one tree with
interactive builds would thrash fingerprints — but there is now one of it.

## Rule

A shared-resource guarantee stated in a config file is not in force until it is
measured on disk. `CARGO_TARGET_DIR` outranks `target-dir`, so any script that
exports it silently opts out of the sharing scheme, and `cargo metadata` will
still report the shared path because it does not see the script's environment.
