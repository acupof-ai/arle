# Hand-rolled pod check script silently resolves a different TileLang than the canonical build

## Context

Relaying a lane that edited `crates/cuda-kernels/build.rs`, the pod
`cargo clippy` / `cargo check` failed with exit 101:

```
TileLang is required to (re)generate a missing or source-changed AOT kernel,
but it is unavailable ... Probe results: python3: tilelang 0.1.14, pinned
0.1.13 | python: AssertionError: tilelang 0.1.14, pinned 0.1.13.
```

It read as a code defect in the lane. The lane did not touch TileLang, so the
first instinct was to chase the diff; the actual cause was the check harness.

## Root cause

The check was a hand-rolled script on the pod that ran `cargo` directly after
only `source ~/.cargo/env`. It never sourced `scripts/pod-build-env.sh`, so it
got no `INFER_TILELANG_PYTHON`. The pod's default `python3` has TileLang
0.1.14; the tree pins 0.1.13 in `requirements-build.txt`. build.rs's probe
then rejected 0.1.14 and aborted.

The canonical pair avoids this by construction: `scripts/pod-remote-build.sh:4`
sets `TREE` from `POD_TREE` (defaulting to the standard build tree) and
`:344` sources
`"$TREE/scripts/pod-build-env.sh"`, whose resolver walks a candidate
interpreter list and accepts one only when its TileLang version equals the
tree's pin. Bypassing it silently swaps the codegen version.

A second latent trap sat in the resolver itself: when `TREE` was unset it read
the pin from the hardcoded default-tree `requirements-build.txt` fallback
rather than the tree being built, harmless only because the fallback literal
happened to equal the pin; and the resolved interpreter was never printed, so
the mismatch was invisible until build.rs failed.

Diagnosis cost: one full `cuda-kernels` rebuild.

## Fix

- Run pod cargo checks through the canonical env: set `TREE` to the synced
  tree and `source "$TREE/scripts/pod-build-env.sh"`; never inline a one-off
  environment (the file's header already says so).
- The resolver now fails loudly when the tracked `requirements-build.txt`
  exists but has no `tilelang==` pin, keeps the literal fallback only when the
  file is genuinely absent (partial sync / sourced outside a tree), prints the
  resolved interpreter and matched pin on every run, and drops the hardcoded
  default-tree venv candidate that duplicated the `$TREE`-keyed one.

## Rule

A pod build/check must source `scripts/pod-build-env.sh` with `TREE` pointing
at the tree under test. Do not hand-roll a `source ~/.cargo/env; cargo …`
script: it resolves a different toolchain than the build that will ship and
the failure surfaces deep in a build.rs probe, looking like a code defect.
