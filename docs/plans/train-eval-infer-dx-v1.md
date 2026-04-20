# Train → Eval → Infer DX (v1)

**Driver:** user directive 2026-04-20 — "现在训练出来的模型能够直接自动化评测和推理吗；这部分的 dx 也做好；做好 cli 的易用性，可理解"
(Can trained models be directly auto-evaluated and served? Polish the
DX; make CLI usable and understandable.)

**Status today** (from Explore punch-list 2026-04-20):

| Question | Answer |
|----------|--------|
| Does the checkpoint format match what `infer` expects? | ✅ (exact) |
| Can I serve a trained checkpoint without hand-assembling paths? | ✅ `latest` marker is landed; serve from `train_multi_turn --serve` output or `infer --model-path <out>/latest` without guessing the step number. |
| Is there a standalone eval binary? | ❌ NO — eval is only inline in `pretrain_qwen3` during training. |
| Do train binaries have `--help`? | ❌ NO — all hand-roll arg parsing. |
| Are flag names consistent across binaries? | ❌ NO — `--seq` vs `--seq-len`, `--model` vs `--model-path`, `--resume` vs `--resume-from`. |
| Does `infer` fail early on a malformed `config.json`? | ❌ NO — `is_model_dir` only checks file existence; field schema validated late. |

## Phases

Phased so each ships independently with its own bench/wins entry.
Cut order chosen to maximize DX per commit.

### Phase DX-1 — End-to-end "train a model then chat with it" works from one path

**Acceptance:** after any training run writes a checkpoint, the user
can run `infer --model-path <out>/latest` (or REPL) without knowing
the step count or reading directory listings.

**Current state:** DX-1 is already landed in-tree. The remaining work in
this plan is DX-2 / DX-3 follow-through, not the `latest` marker flow.

**Work:**
1. Every trainer save_checkpoint hook (and the hand-written save
   paths in `pretrain_qwen3`) writes a `latest` symlink in the
   parent dir pointing at the just-written `step_N` dir. Atomic via
   `symlink_metadata()` unlink + `symlink()` (or `LATEST` text file
   containing the step dir name if the filesystem refuses symlinks).
2. Normalize checkpoint dir padding to `step_{N:06}` across all
   producers (`pretrain_qwen3` currently unpadded; Trainer
   `save_every` already pads to 6). Resume path lookup becomes a
   single glob pattern instead of per-binary branches.
3. `scripts/train_and_chat.sh` walkthrough: 3-line data.jsonl →
   train_sft 10 steps → REPL, with inline comments explaining each
   flag.

**Out of scope:** broader flag rename (in DX-3).

**Estimate:** 1 commit, ~50 LOC + 1 script + 1 wins entry.

**Shipped:**
- `0da212f` — initial `latest` symlink + step padding + wrapper script.
- `8bde810` — DX-1 follow-up: atomic swap, deferred Trainer publish,
  symlink-aware resume (3 codex findings on `0da212f`).
- `d700a24` — DX-1 follow-up #2: canonicalize `--resume`/`--resume-from`
  once at entry, `publish_latest_after_weights(parent, basename)` helper
  + 3 unit tests pinning the publish-last contract. Closed both remaining
  codex findings on `8bde810`. See
  [`wins/2026-04-20-dx1-publish-last-helper-and-canonicalize.md`](../experience/wins/2026-04-20-dx1-publish-last-helper-and-canonicalize.md).

### Phase DX-2 — Standalone eval binary

**Acceptance:** `cargo run --release -p train --bin eval_lm --
--model-path <ckpt> --data <held_out.jsonl>` prints
`{"loss": ..., "ppl": ..., "tokens": ...}` and optionally writes
`--metrics-jsonl <path>` with one record.

**Work:**
1. Extract the held-out eval loop from `pretrain_qwen3.rs` into a
   `train::eval_lm` helper: takes `&Weights`, `&State`, tokenized
   eval windows → `EvalOutcome { loss, token_count }` (same type
   the Trainer already emits).
2. New binary `crates/train/src/bin/eval_lm.rs`: load weights via
   existing Qwen3 loader, tokenize `--data` via existing jsonl
   dataloader, call the helper, emit one `MetricSample` with
   `eval_loss`, `eval_ppl = exp(eval_loss)`, `eval_tokens`.
3. Shares `cli_args::trainer_args()` helper for consistency with
   training binaries.

**Acceptance tests:** reproduce the `eval_loss` that
`pretrain_qwen3.rs` prints on the same data + checkpoint to within
f32 round-trip. Catches "extracted helper diverged from inline
path" regressions.

**Out of scope:** non-CE loss surfaces (e.g. reward models); add
only if a real caller needs them.

**Estimate:** 1-2 commits, ~200 LOC + tests + wins entry.

### Phase DX-3 — CLI flag normalization + clap adoption

**Acceptance:** `cargo run -- --help` works on every training +
eval binary; flag names match across binaries for the same concept.

**Proposed canonical names:**

| Concept | Flag | Type |
|---------|------|------|
| base model directory | `--model-path` | path |
| training data | `--data` | path |
| checkpoint output root | `--out` | path |
| sequence length | `--seq-len` | usize |
| resume from checkpoint | `--resume-from` | path (accepts `<out>/latest`) |
| metrics JSONL | `--metrics-jsonl` | path |
| grad clip | `--grad-clip` / `--no-grad-clip` | f32 / flag |

**Work:**
1. Adopt `clap` derive across all train binaries (replacing hand-
   rolled parsers in `pretrain.rs`, `pretrain_qwen3.rs`,
   `train_sft.rs`, `train_grpo.rs`, `train_multi_turn.rs`).
2. Migrate to `cli_args::trainer_args()` shared helper (already
   exists per 15ed922 refactor).
3. Keep old flag names as deprecated aliases for one release cycle
   (emit a stderr warning + honor the flag).
4. `infer`'s `is_model_dir()` adds a config-field schema check:
   require `architectures`, `model_type`, `hidden_size`, `vocab_size`,
   `eos_token_id`. Fail-early with a precise error pointing at the
   missing field.

**Out of scope:** REPL (already uses rustyline + clap-style state
in `crates/cli/src/repl.rs`).

**Estimate:** 2-3 commits, ~500 LOC churn across 5 bins + wins
entry. Biggest phase; may split.

### Phase DX-4 — Chat + agent history unification (76ea6ce Medium #2)

**Driver:** 76ea6ce codex review flagged `/chat` ↔ `/agent` mode
switch loses prior conversation context (`chat_history` and
`AgentSession` diverge).

**Design question (needs user decision before implementation):** on
`/chat` after `/agent`, should the REPL:
- (a) **merge** agent turns → chat history, dropping tool-call
  metadata; or
- (b) **reset** to a fresh history, explicit "you switched modes,
  context is gone" banner; or
- (c) **preserve both** in parallel, switch only flips the active
  pointer — existing implementation + doc fix.

**Out of scope until user picks.**

## Dependencies / blockers

- Phase DX-1 touches every save site. Currently `train_grpo` and
  `train_multi_turn` have hand-written save paths (if any) — need
  to audit per Explore punch-list P2.
- Phase DX-2's helper extraction must not regress
  `pretrain_qwen3`'s inline eval (same tokens, same loss).
- Phase DX-3 backward-compat: existing `scripts/` and any user
  muscle-memory break on rename; deprecated aliases buy a release.

## Bench policy

DX-1 and DX-3 are CLI-only, no hot-path numbers move → bench-exempt
per CLAUDE.md (document in commit body). DX-2 runs a forward pass;
bench-exempt for the helper itself (no new kernels) but should
produce a "before/after consistency" number in the wins entry.

## Cross-refs

- [Explore punch-list 2026-04-20] mapped the state → found above.
- [Codex review 76ea6ce] — High (fixed in 97c1a95), Medium #1
  (fixed in 97c1a95), Medium #2 (DX-4 above).
- [`docs/plans/train-runtime-architecture-v1.md`] — Phase 4
  landed; this plan is the observability/DX follow-up layer.
