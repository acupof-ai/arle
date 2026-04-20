# Phase DX-1 — `latest` symlink + uniform step padding across save sites

**Commit:** 0da212f (feat(train): DX-1 — `latest` symlink + uniform step
padding across save sites)

## Context

2026-04-20 directive: *"现在训练出来的模型能够直接自动化评测和推理吗；这
部分的 dx 也做好；做好 cli 的易用性，可理解"* — can trained models be
directly auto-evaluated and served? polish the DX; make CLI usable and
understandable.

Pre-DX-1 reality (from `docs/plans/train-eval-infer-dx-v1.md` status
table):

| Question | Answer |
|----------|--------|
| Can I serve a trained checkpoint without hand-assembling paths? | ⚠️ FRAGILE — no `latest` marker; user guesses step number. |

Trainer saved to `step_{:06}` (padded), `train_sft` saved to
`step_{:06}` (padded), `pretrain_qwen3` saved to `step_{N}` (**un**padded).
Downstream tooling either hard-coded a specific step number
(`train_and_chat.sh` — `$OUT_DIR/step_2`) or had to enumerate + sort
`step_*` entries. Plus, across padded-vs-unpadded producers, a single
glob pattern wouldn't work.

## What Worked

**One helper, three wires, one contract test.**

1. `checkpoint::write_latest_symlink(parent, basename)` writes a *relative*
   symlink. Relative (not absolute) is deliberate: copying / rsync'ing the
   checkpoint root to a new machine preserves the pointer instead of
   dangling at the original path. Atomic (remove-then-create); refuses to
   overwrite a regular file or directory at `<parent>/latest` (guards
   against a user accidentally stashing a final checkpoint under that
   exact name). Rejects `basename` containing `/` or `\` to keep the
   pointer scoped to the root. Unix-only; non-unix is a documented no-op
   with the lex-max `step_*` fallback called out in the doc comment.

2. Wired into all three save sites in a single commit so producers don't
   disagree about the pointer's presence:
   - `Trainer::save_checkpoint` (runs on `save_every`)
   - `pretrain_qwen3::save_checkpoint` (hand-written post-step hook)
   - `train_sft::save_checkpoint_via_registry` (registry-driven save)

3. Normalized `pretrain_qwen3`'s `format!("step_{step}")` →
   `format!("step_{step:06}")`. Resume-path lookup now collapses to a
   single `glob("step_??????/")` pattern across all three binaries;
   `--resume-from <out>/latest` roundtrips without the caller caring
   which producer wrote the checkpoint.

4. Extended `trainer_save_then_resume_roundtrip` (tests/test_trainer_loop.rs)
   to assert the symlink exists, **is** a symlink (not a file/dir), and
   resolves to the just-written basename. This turns the DX-1 contract
   into an automated regression check — any future producer that skips
   `write_latest_symlink` will fail the test, not silently ship broken
   DX.

5. `scripts/train_and_chat.sh` updated in the same commit to address
   `$OUT_DIR/latest` instead of `$OUT_DIR/step_2`. The script is now
   step-number-agnostic — it keeps working at `--steps 20` or `--steps
   2000` without edits.

**Why relative symlink, not an absolute path or a `LATEST` text file:**

- Absolute path: breaks on copy/rsync to a different mount point. The
  user's most common next step after training is "scp to the inference
  box" — an absolute symlink dangles there.
- `LATEST` text file: every consumer needs custom parse logic. With a
  symlink, every UNIX tool (`ls -L`, `readlink`, shell glob, Rust's
  `std::fs::read_dir`) already handles it uniformly.
- Tradeoff: non-unix targets (Windows) no-op. Acceptable — workspace
  support matrix is Linux + macOS; Windows users would hit larger gaps
  first.

## Rule

**Producers of step-versioned artifacts must refresh a `latest`
pointer.** If three save sites each decided independently whether to
write `latest`, users would see inconsistent DX depending on which
binary they ran. Ship the helper, wire every producer, lock it in with
a roundtrip test that asserts the pointer's contents.

**Format changes ride on top.** The `step_{step}` → `step_{step:06}`
normalization was a one-character change but it couldn't ship
independently: if Trainer writes padded and `pretrain_qwen3` writes
unpadded, `latest` pointing at `step_7` vs `step_000007` still leaks
into downstream glob patterns. Same-commit cutover.

## Bench Policy

Bench-exempt per CLAUDE.md §Benchmarks — CLI/DX only, no hot-path
impact. The helper runs at most once per `save_every` (typically every
N=100+ steps), and the bulk is a `symlink_metadata` + `remove_file` +
`symlink` triple — ~tens of microseconds on any sane filesystem,
dwarfed by the actual `model.safetensors` serialize.

## Follow-up: codex review on 0da212f → 8bde810

`codex review --commit 0da212f` flagged three issues. Fixed in
`8bde810 fix(train): DX-1 follow-up`:

1. **High — `--resume <out>/latest` restarted from step 0.**
   `pretrain_qwen3::load_resume_checkpoint` derived `start_step` from
   `resume_dir.file_name()`, which is the string `"latest"` when the
   caller passes the symlink. Integer parse fell through to 0 and the
   next save clobbered `step_000001`. Fixed with
   `resume_dir.canonicalize()` before parsing.

2. **Medium — Trainer published `latest` before weights landed.**
   `Trainer::save_checkpoint` wrote `trainer_state.json` +
   `optimizer.safetensors`; the binary's `on_step_end` then wrote
   `model.safetensors`. Publishing `latest` from Trainer meant readers
   could briefly see a dir without weights. Fixed by removing the
   symlink call from Trainer — only the binaries publish, AFTER their
   weight write completes. Test updated to assert the inverse.

3. **Medium — symlink update was not atomic.**
   The original `remove_file(link)` + `symlink(target, link)` sequence
   left a window where `latest` did not exist. Replaced with
   `symlink(target, .latest.tmp)` + `rename(.latest.tmp, latest)` —
   POSIX rename on same directory is atomic. Readers see either the
   old target or the new one, never nothing.

### Updated rule

**Publish the pointer *last*, swap it atomically.** A "latest" marker
is a contract: its existence promises that the target is complete. If
you publish before the payload lands, or if you replace non-atomically,
you break that contract during a window users will hit. The
`symlink(tmp) + rename(tmp, final)` idiom is a two-liner and ships with
every POSIX filesystem — there is no excuse for the remove-then-create
pattern in durable tooling.

## Cross-refs

- [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)
  §Phase DX-1 — closes the phase. DX-2 (standalone `eval_lm` binary) +
  DX-3 (clap + flag normalization) + DX-4 (chat/agent history
  unification, blocked on user decision) still open.
- [`docs/experience/wins/2026-04-20-codex-reviews-60f7183-76ea6ce.md`](2026-04-20-codex-reviews-60f7183-76ea6ce.md)
  — 76ea6ce Medium #2 (chat/agent history loss) is tracked as DX-4
  above; explicitly blocked on user design decision.
