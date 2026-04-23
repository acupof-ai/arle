# Train CLI Model Command Cleanup

## Context

- Goal: keep `crates/cli/src/train_cli.rs` clean after the earlier invocation
  wrapper unification by removing the next layer of internal duplication.
- Review focus:
  - `train sft` and `train eval` were both "model-backed" commands, but they
    each reimplemented the same `inspect model -> choose backend -> decide
    model arg -> carry notes forward` flow
  - `train env` still rebuilt a static command list into a freshly allocated
    `Vec<String>` on every call even though the list is fixed at compile time

## What Worked

- Added `resolve_model_command(...)` so `train sft` and `train eval` share one
  canonical model-backed resolve path.
- Kept the command-specific behavior local:
  - `sft` still owns `--out` defaulting and SFT-specific notes
  - `eval` still owns tokenizer-override handling and eval-specific notes
- Replaced the dynamically built `train env` command list with one static
  `TRAIN_ENV_COMMANDS` slice wired straight into the serialized report.
- Verified the external CLI contract stayed stable:
  - `agent-infer train env --json`
  - `agent-infer train sft --model models/Qwen3-0.6B --data models/tiny_sft.jsonl --dry-run --json`
  - `agent-infer train eval --model models/Qwen3-0.6B --data models/tiny_sft.jsonl --dry-run --json`
  - `agent-infer train test --backend metal --json`

## Rule

- When multiple CLI commands are all "model-backed", keep one shared helper for
  model inspection, backend resolution, and resolved model-arg selection.
  Command-specific flags should layer on top of that base, not reimplement it.
- For static CLI metadata, keep it static. Do not allocate and rebuild the same
  command list on every call if the source of truth is compile-time constant.
- CLI / operator-DX only; no runtime hot path changed, so no performance
  benchmark entry was required.
