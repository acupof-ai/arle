# Phase DX-1 follow-up #2 — publish-last helper + canonicalize-once resume

**Commit (pending):** closes the two codex findings on `8bde810` that
landed after the first DX-1 follow-up landed the atomic symlink swap +
deferred Trainer publish. Same day, one helper + one canonicalization
pattern repeated at both binary entry points.

## Context

`8bde810` fixed three codex findings from the initial `0da212f` DX-1
land (Medium: remove-then-create window, Trainer publishing `latest`
before weights, bare `file_name()` on the symlink literal). It left two
open findings behind that `codex review --commit 8bde810` flagged:

> **High** — `pretrain_qwen3::resume_from_checkpoint` and
> `train_sft::run` each call `resume_dir.join(...)` / pass
> `args.resume_from` into `TrainerConfig` without canonicalizing first.
> A concurrent trainer repointing `<out>/latest` between our opens
> (weights, config, `trainer_state.json`, `optimizer.safetensors`) can
> mix step N weights with step N+1 metadata. The DX-1 doc asserts a
> "single snapshot" read; the code does not enforce it.

> **Low** — the test change at `tests/test_trainer_loop.rs:278` removed
> the only positive end-to-end assertion that any training save path
> still publishes `<out>/latest`. The unit tests prove
> `write_latest_symlink` works, but the binary hook calls at
> `pretrain_qwen3.rs:910` and `train_sft.rs:676` are now untested. A
> future regression that drops or reorders them would pass CI.

(Findings transcribed from `/tmp/codex-review/8bde810.log` lines
5411–5460 — manually excerpted because the review sandbox can't read
`~/.codex/sessions` directly.)

## What Worked

**One canonicalize line per binary, one helper that encodes the
publish-last contract, three targeted unit tests.**

1. **Canonicalize-once-at-entry** in both binaries:

   - `pretrain_qwen3::resume_from_checkpoint`
     (`crates/train/src/bin/pretrain_qwen3.rs`) — calls
     `resume_dir.canonicalize()?` as its first statement. Error surface
     names the symlink literal (`--resume <path>`) + hints at missing
     target, so a user typo is debuggable without grepping the diff.
     The later `canonical = resume_dir.canonicalize().unwrap_or_else(...)`
     fallback around the `file_name()` step-derivation is now dead and
     removed — the already-canonical `resume_dir` flows through.

   - `train_sft::run` (`crates/train/src/bin/train_sft.rs`) —
     canonicalizes `args.resume_from` once into a local
     `resume_dir_canonical: Option<PathBuf>` **before** the
     `TrainerConfig` struct literal. `trainer_cfg.resume_from` uses the
     canonical path, so `Trainer::resume_if_configured` reads
     `trainer_state.json + optimizer.safetensors` from the same snapshot
     that `validate_resume_config` and the weight-existence check
     verified. The later `if let Some(resume_dir) = &args.resume_from`
     now reads from `resume_dir_canonical.as_deref()` — same ref, same
     `.join("model.safetensors")`, same canonical target.

   The single-snapshot invariant is now **enforced by construction**:
   every subsequent reference to the resume dir in both binaries goes
   through the already-canonicalized value. A concurrent trainer
   repointing `latest` after our canonicalize can't reach us.

2. **`checkpoint::publish_latest_after_weights(parent, target_basename)`**
   (`crates/train/src/checkpoint.rs`) — a ≤10-line helper that asserts
   `<parent>/<target_basename>/model.safetensors` exists as a regular
   file, then delegates to the existing `write_latest_symlink`. If the
   weight file is missing (or is a directory), it returns
   `io::ErrorKind::NotFound` with a message citing the publish-last
   contract by name. Both binary save hooks now call this instead of
   raw `write_latest_symlink`:

   - `pretrain_qwen3.rs:923` — in `save_checkpoint`, directly after
     `registry.save_from(...)` writes `model.safetensors`.
   - `train_sft.rs:699` — in `save_checkpoint_via_registry`, same
     position.

   The import lists in both binaries swap
   `checkpoint::write_latest_symlink` for
   `checkpoint::publish_latest_after_weights` — no half-state where
   both names are in scope.

3. **Three regression tests** (`checkpoint::latest_symlink_tests`):

   - `publish_after_weights_writes_symlink_when_weights_present` —
     happy path: stage `step_000001/config.json` +
     `step_000001/model.safetensors`, assert symlink gets published and
     resolves to the step dir.
   - `publish_after_weights_refuses_when_weights_missing` — simulate a
     refactor that publishes before `registry.save_from`:
     `config.json` present but no `model.safetensors`. Assert
     `ErrorKind::NotFound`, that the error message names
     `model.safetensors` and "publish-last", and that no `latest`
     symlink exists after the refused call.
   - `publish_after_weights_refuses_when_weights_is_directory` —
     guard the `.is_file()` check against a future loosening to
     `.exists()`: create a *directory* at the expected weights path,
     assert the refusal still fires.

   Together they pin the exact finding: a future refactor that moves
   the publish call above the weight write (or drops it) fails these
   tests at the train-crate-lib level, not at CI integration time.

4. **Verified**: `cargo test -p train --release --lib
   latest_symlink_tests` → 7/7 green (4 pre-existing + 3 new).
   `cargo test -p train --release --lib` → 17/17 green (full train lib
   test surface). `cargo build -p train --release --bin pretrain_qwen3
   --bin train_sft` → clean.

## Why canonicalize AND publish-last both?

They address **different** failure modes that the codex review conflated
at first read:

- **Canonicalize-once-at-entry** fixes *reader* drift: the DX-1 resume
  path reads 4+ files from the step dir; without canonicalization, a
  concurrent writer can repoint `latest` between any two of those reads.
- **Publish-last helper** fixes *writer* drift: the DX-1 save path
  writes 3+ files into the step dir (trainer_state, optimizer, weights)
  in two stages (Trainer then binary); without the gate, a future
  refactor that flips the order exposes an incomplete dir to readers.

Either one alone leaves a correctness hole. Landing them together in
one commit keeps the DX-1 "single-snapshot" invariant whole on both
sides.

## Rule

**Canonicalize symlinked path inputs once at the function entry where
they enter the computation.** Downstream code should handle
`PathBuf`s, not `Option<PathBuf>` that may-or-may-not have been
canonicalized. The canonicalize call is the hand-off point where "the
user gave us a name" becomes "we've committed to this snapshot".

**When two writers cooperate on a publish contract, encode the
ordering invariant in code, not comments.** The Trainer comment at
`trainer.rs:625` documenting "publishing here would expose an
incomplete dir" is useful *context* but it doesn't prevent the
regression — a helper that refuses to publish without the final
artifact does. If the contract is load-bearing, turn it into a unit
test that fails on violation.

**After a "refactor away end-to-end coverage" move, add three unit
tests at the refactored layer.** Losing an integration test because
the mock setup got expensive is usually fine; losing coverage of the
behavior entirely is not. The three new `publish_after_weights_*`
tests cost ~40 lines and run in 0ms, vs. the
`test_trainer_loop.rs:278` test that needed a full trainer setup.

## Bench Policy

**Bench-exempt.** Both changes are path-handling + file-existence
checks on the save/resume paths — `canonicalize()` is a single
`realpath(3)` + allocation per resume, called once per `--resume`;
`publish_latest_after_weights` adds one `stat(2)` per checkpoint save.
Neither is on the token-generation hot path. Runtime impact: immeasurable.

Not producing a `bench_guidellm.sh` run. Stating the exemption in the
commit body per CLAUDE.md §Benchmarks → "Exempt".

## Cross-refs

- [`docs/experience/wins/2026-04-20-phase-dx1-latest-symlink.md`](2026-04-20-phase-dx1-latest-symlink.md) —
  original DX-1 land + first-follow-up (8bde810) writeup; §Follow-up
  there describes the three-fix landing that motivated this entry.
- [`crates/train/src/checkpoint.rs`](../../../crates/train/src/checkpoint.rs) —
  `publish_latest_after_weights` helper + three new unit tests.
- [`crates/train/src/bin/pretrain_qwen3.rs`](../../../crates/train/src/bin/pretrain_qwen3.rs) —
  canonicalize at `resume_from_checkpoint` entry; call-site swap in
  `save_checkpoint`.
- [`crates/train/src/bin/train_sft.rs`](../../../crates/train/src/bin/train_sft.rs) —
  canonicalize before `TrainerConfig`; call-site swap in
  `save_checkpoint_via_registry`.
- `/tmp/codex-review/8bde810.log` lines 5411–5460 — the two findings
  this entry closes.
