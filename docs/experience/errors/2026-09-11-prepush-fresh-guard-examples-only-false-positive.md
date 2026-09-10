# Pre-push Fresh guard blocked examples-only CUDA pushes

## Context

The local pre-push hook (`scripts/pre_push_checks.sh`) runs a verbose CUDA
clippy and asserts every crate the pushed range changes was actually rebuilt:
a shared `target/pre-push-quick` dir across lanes can otherwise let a stale or
foreign-lane artifact make cargo report a content-changed crate Fresh, and a
real compile error then passes the gate green (the 2026-09-10 incident).

The parity-gate lanes (#312, fd's #313) pushed commits that changed only
`crates/infer-cuda/examples/*.rs`. The lib-fresh loop matched the crate from
the push range with `grep ^crates/infer-cuda/` but inspected the LIB clippy
output for `Checking infer-cuda`. An examples-only commit leaves the lib
fingerprint unchanged, so cargo legitimately reported the lib Fresh, and the
guard failed every such push with
"CUDA lint reported infer-cuda Fresh … cross-contaminated". The separate
examples-fresh assertion — the one that actually covers example rebuilds — was
healthy throughout. Recovery required `cargo clean -p infer-cuda`, turning a
warm sub-minute push into a full rebuild.

## Root Cause

The assertion keyed "this crate needs a lib Checking line" on the pushed file
RANGE, which includes files that never affect a lib fingerprint
(`examples/`, `tests/`, `benches/`). The key should be what rsync actually
changed in the snapshot's lib-affecting files.

## Fix

- The snapshot rsync now records updated and deleted paths
  (`--out-format='%n'`, `deleting <p>` lines stripped) into a temp list.
- `rsync_touched_lib <crate>` is true only when that list contains
  `crates/<c>/(src/**|build.rs|Cargo.toml)`; both the
  `assert_step_rebuilt` test-step guard and the CUDA lib-fresh loop use it.
- The examples-fresh guard keys on any rsync update under
  `crates/infer-cuda/`, so an example delta still demands a rebuilt example.
- The strict-CUDA-lint branch gate stays on the push RANGE: it decides
  whether the verbose lint runs at all; only the assertions use rsync state.

`test_hook_cuda_lint_freshness.sh` adds an examples-only control (lib
legitimately Fresh, example Checking, push accepted) and runs every red world
from a snapshot pre-seeded at the parent commit, which is the real
rsync-delta condition the hook sees. All 11 `scripts/tests/test_*.sh` pass.

## Rule

A freshness/rebuild assertion must key on the inputs that actually change the
build fingerprint, observed after the source sync — not on the set of files a
push happens to carry under a crate directory. `examples/` and `tests/` are
not lib inputs.

## Follow-up (S18d)

This guard was deleted entirely the same day: under the post-#302 rsync
(mtime=now) plus the machine lock, cargo's own mtime fingerprint guarantees a
changed dep-info input rebuilds, so the output-scraping guard was redundant
and could only false-positive. See
`2026-09-11-prepush-fresh-guard-was-redundant.md`.
