# The pre-push Fresh guards were redundant after rsync mtime=now

## Context

After the 2026-09-10 incident (a real CUDA compile error passed the pre-push
gate green because a foreign lane's newer artifact in the shared
`target/pre-push-quick` made cargo report a changed crate Fresh), the hook
grew output-scraping guards: `assert_step_rebuilt` and CUDA lib/examples
loops grepped verbose cargo output for `Checking <crate>` whenever the pushed
range touched the crate.

Those guards then false-positived legitimate pushes:
- examples-only commits (#312 twice, #313) leave the lib fingerprint
  unchanged, so the lib lint is correctly Fresh, but the range grep
  `^crates/infer-cuda/` demanded a lib `Checking` line.
- a `#[cfg(test)]`-only file under `src/` (e.g. #310's
  `crates/cuda-kernels/src/ffi/gemm_tests.rs`) is not in the non-test unit's
  dep-info; the non-test clippy never reads it.

## Derivation and experiment

Cargo fingerprints a unit by the mtimes of the files in its dep-info. The
hook (post-#302) already guarantees:
1. rsync runs WITHOUT `-t`, so every file it updates gets mtime=now;
2. a machine-global lock serializes rsync → every cargo step, so only the
   current snapshot's build writes into the shared target.

Therefore an updated dep-info input is strictly newer than the previous
artifact and the unit MUST rebuild; a Fresh unit necessarily had no dep-info
input updated. Under the lock "input changed / cargo Fresh" is unreachable.

Verified with REAL cargo (`scripts/tests/test_prepush_fresh_invariant.sh`,
scratch target, one snapshot dir, mkdir-lock):
- changed lib.rs content under mtime=now rsync → rebuilds 100% of the time
  across ping-pong content changes; identical re-sync → Fresh;
- cfg(test)-only file change → non-test `cargo check` correctly Fresh;
- the same changed input back-dated and copied with `rsync -t` (the
  pre-#302 behavior) → reported Fresh, reproducing the 2026-09-10 false
  negative. That is the real defect, and rsync mtime=now already fixes it.
- un-locked builds from two source dirs into one target within the same mtime
  second can collide; the lock removes exactly that race.

## Fix (net deletion)

- Deleted `assert_step_rebuilt`, the CUDA lib/examples Fresh loops, the
  verbose `-v` lint captures, `CUDA_CRATES_CHANGED`, and the rsync
  `--out-format` SYNCED_FILES machinery. The CUDA lints are plain
  `run cargo clippy` again (same coverage, no scraping).
- Replaced the mocked-cargo freshness test with
  `test_prepush_fresh_invariant.sh`, which drives real cargo and proves the
  mtime invariant, including the rsync `-t` control that would break if a
  future change silently restored old-mtime preservation.

## Rule

Do not scrape build-tool stdout to enforce an invariant the build tool
already enforces, when the inputs feeding it are serialized and stamped.
Pin the invariant itself (mtime=now sync + lock → changed dep-info input
rebuilds) with a real-tool test; mock-output tests pin the scraper, not the
guarantee.
