# CLI live tests assumed stable tool prompts and home-path rendering

## Context

- Change under test: CLI DX closure for `arle` help, one-shot mode, doctor
  output, and real-model live validation on Apple Silicon.
- Validation included:
  - `cargo test -p cli --release --no-default-features --features metal,no-cuda`
  - `cargo test -p agent-infer --release --no-default-features --features metal,no-cuda,cli --test cli_smoke`
  - `ARLE_MODEL=models/Qwen3-0.6B cargo test -p agent-infer --release --no-default-features --features metal,no-cuda,cli --test cli_agent_live -- --ignored --nocapture`

Two failures were not product regressions in the CLI surface itself, but test
assumptions that were too brittle for real-model execution.

## Root Cause

- The live one-shot test assumed a small local model would reliably obey an
  English instruction like "Use the python tool ..." and actually execute the
  tool within a tight `--max-tokens 96` budget. In practice Qwen3-0.6B often
  spent the budget narrating intent instead of reaching the tool call.
- The model-picker path abbreviation test assumed the rendered path would start
  with `/path/to/...`, but `abbreviate_path()` intentionally collapses
  the active `$HOME` prefix to `~/...` when it matches the current environment.
  That made the assertion host-dependent.

## Fix

- Reworked the live tests to use prompts that were observed to stably trigger
  real shell execution on the local model:
  - `run --prompt "本地有哪些文件" --json`
  - `run --stdin --json` with the same query
- Kept the REPL/reset live test focused on actual multi-turn/reset behavior
  instead of requiring a tool marker on every arithmetic turn.
- Made the live-test mutex resilient to prior test panics by recovering the
  poisoned lock state.
- Changed the model-picker abbreviation test to use a non-home path
  (`/opt/huggingface/...`) so it validates the head/tail abbreviation logic
  without depending on `$HOME` rendering.

## Rule

- Real-model CLI tests should validate stable observable behavior, not assume a
  small model will always follow a verbose tool-use instruction literally.
- Path-rendering tests must not hardcode a home-directory prefix when the
  production function intentionally rewrites `$HOME` to `~`.
