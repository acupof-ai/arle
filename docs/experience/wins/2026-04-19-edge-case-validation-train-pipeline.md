# Edge-case validation of the train pipeline (+ codex-found P1/P2 fixed)

## Context

User (2026-04-19) asked: "同时端到端验证一下本地能验证的你说的这些
功能是否都好用，多种边界case测试好". After the earlier
`2026-04-19-end-to-end-training-flows.md` happy-path validation, run
the same spine through realistic edge inputs and fix anything that
surfaces. A `codex review --base origin/main~4` ran in parallel on
the training-related commits and reported two issues; both are fixed
here.

## What Worked

### Edge cases that passed unchanged

- `data_adapter` — five unit tests cover dolly ± context, alpaca ±
  input, sharegpt unknown-role drop, chat passthrough. All green.
- `train_sft` on a 2-line JSONL where one line is missing
  `assistant` → trainer warns and skips, completes the step without
  abort.
- `train_grpo --sft-steps 0` → RL-only loop runs: loss drifts to
  `-0.0010`, `mean_kl -0.68`. Reference copy is frozen correctly.
- `agent-infer` with empty stdin → prints banner and waits (no crash).
- Bad `--format` / missing `--input` / missing file → exit 2 with
  actionable message.
- 40 autograd CPU tests + 38 Metal parity tests still green.

### Issues found + fixed

**P1 (codex-found, data loss): `convert_dataset --input X --output X`
truncated the source.** `File::create(output)` truncates the already-
open inode on Unix, so the read loop sees EOF and the original
dataset is lost. Fixed by canonicalizing both paths up front and
refusing the run:

```rust
fn paths_alias(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}
```

Now also catches the `./file` vs `file` aliasing case:

```
$ convert_dataset --input a.jsonl --format dolly --output ./a.jsonl
error: --input and --output resolve to the same file (a.jsonl);
       in-place conversion would truncate the source before reading
exit 1
```

**P2 (codex-found, silent pipeline failure): `0 written` exited 0.**
A wrong `--format` produced an empty output file and a success exit,
so shell pipelines moved on and `train_sft` blew up much later with
a less actionable error. Fixed by treating `written == 0` as failure
in `crates/train/src/bin/convert_dataset.rs`:

```
$ convert_dataset --input wrong-schema.jsonl --format dolly --output /tmp/out.jsonl
[data_adapter] skipping line 1 (wrong-schema.jsonl): missing field `instruction` ...
[convert_dataset] 1 lines · 0 written · 1 skipped
[convert_dataset] error: produced 0 records (wrong --format, or input didn't match the expected schema)
exit 1
```

**BytesDataset ignored `--vocab-size`**: `pretrain --dataset bytes
--vocab-size 64` crashed mid-step with
`IndexOutOfBounds { index: 104, upper: 64 }`. `BytesDataset::sample`
emits raw byte ids 0..=255 and never consulted the vocab argument,
so an undersized embedding table overflowed. Fixed in
`build_dataset` in `crates/train/src/bin/pretrain.rs` to reject the
config before construction:

```
$ pretrain --dataset bytes --vocab-size 64 ...
Error: "--dataset bytes emits byte ids 0..=255; --vocab-size 64 < 256
        would overflow the embedding table"
```

## Not yet validated

Two rough edges noted but left as UX follow-ups, not bugs:

- `agent-infer --model-path /does/not/exist` falls through to the HF
  hub download path before reporting "not found". A local-path
  existence check up front would short-circuit the network hop.
- `train_sft` surfaces IO errors with Debug formatting (`Os { code:
  2, kind: NotFound, ... }`). Switching to `{:#}` / `Display` would
  be friendlier. Neither blocks the pipeline; opened as tickets in a
  future cleanup pass.

## Rule

Edge-case coverage for a multi-stage pipeline (download → convert →
train → infer) has three failure modes worth pinning tests on:

1. **Aliased paths**. In-place normalization is a tempting UX shortcut
   that silently destroys data. Always canonicalize input/output and
   reject equality before opening the writer.
2. **Zero-work success**. A stage that produces 0 outputs should exit
   non-zero, not pass. Silent emptiness converts an actionable stage
   error into a confusing downstream error.
3. **Config-data mismatch that surfaces late**. When a CLI arg
   (`--vocab-size`) and a data property (byte-range tokenization)
   disagree, the check belongs at config-build time, not inside the
   step loop.

All three patterns showed up in one pass through this pipeline;
assume they live in every multi-stage CLI tool until proven otherwise.
