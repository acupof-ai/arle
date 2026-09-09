# A fresh lane tree fell through to the system TileLang and JIT'd against 0.1.8

## Context

A CUDA build in a fresh per-lane pod tree failed inside TileLang codegen with
`14 stages vs 15 pipeline stages` on `batch_prefill_paged_hd64_q16_kv1`,
sm_90, `BUILD_EXIT=1`. A second lane building on the same pod, from the shared
tree, succeeded on the same commit and the same pin.

## Root cause

Two independent things had to line up.

**The bundle was absent.** A fresh lane tree starts with an empty
`crates/cuda-kernels/generated/`. `scripts/kernel_artifacts.sh sync` finds no
published bundle for that source identity, so `build.rs` regenerates the
kernels through the TileLang JIT instead of unpacking 105 prebuilt cubins. The
shared tree `/host/arle-build` already held a fetched bundle and never entered
the JIT path at all — which is why one lane passed and the other did not.

**The interpreter was not the pinned one.** `find_tilelang_python()` walks
`INFER_TILELANG_PYTHON`, then `tools/tilelang/.venv/bin/python`, then
`.venv/bin/python`, then `python3` and `python` on PATH. A fresh tree has no
venv, so the walk reached the system interpreter, which carried TileLang 0.1.8
against a `requirements-build.txt` pin of 0.1.13. `probe_tilelang_python()`
accepted it, because the probe was `import tilelang` and nothing more.

`scripts/pod-tilelang-env.sh` does assert the version equals the pin, but it
provisions the venv; it cannot constrain an interpreter that `build.rs` picks
up from PATH after deciding the venv is absent.

The reported error names neither the interpreter nor the version, so the
failure reads as a kernel-source bug on the SM the lane happens to target.

## Fix

`probe_tilelang_python()` now runs `importlib.metadata.version('tilelang')` and
requires it to equal the pin parsed from `requirements-build.txt`, reporting
the last line of the failure:

    /usr/bin/python3: AssertionError: tilelang 0.1.8, pinned 0.1.13

An unreadable or pin-less `requirements-build.txt` falls back to the old
import-only probe rather than becoming a build failure of its own.

## Consequence for per-lane pod trees

One tree per lane multiplies both halves of this: each new tree starts with an
empty `generated/` and no venv, so each one takes the JIT path once and
provisions its own venv. The bundle is content-addressed by source identity
over tracked files only, so trees on the same commit can share one — copying
the fetched bundle between trees is sound, and was what unblocked this build.

## Rule

A probe that asks "does it import" licenses any version. When a pin exists,
the probe checks the pin, and its failure names the interpreter and both
versions — otherwise the mismatch resurfaces as an error message from three
layers down, about something else.
