# Train CLI DX Defaults And Auto Resolution

## Context

- Goal: make `agent-infer train ...` and `agent-infer data ...` feel like one
  coherent CLI instead of a thin passthrough wrapper with hidden underlying
  semantics.
- Regression to fix: the first unified wrapper incorrectly treated empty argv
  as help and rewrote dataset exit codes, so `agent-infer data convert` could
  exit `0` when the underlying command would exit `2`.
- DX gaps to close:
  - standard `--help` only showed `[ARGS]...`
  - `sft` / `eval` model-dir auto-loading existed in train binaries but was not
    visible in the top-level CLI
  - common defaults like output paths and backend choice were not surfaced
  - there was no train-side `env`, `test`, or `estimate-memory` command

## What Worked

- Replaced the generic passthrough leaf commands with typed clap subcommands for
  `train pretrain|sft|grpo|multi-turn|eval|env|test|estimate-memory` and
  `data download|convert`.
- Preserved advanced escape hatches by keeping `extra_args` after `--`, so the
  top-level CLI exposes the common flags directly without blocking lower-level
  experimentation.
- Restored correct command semantics:
  - missing required args now fail through clap with usage exit `2`
  - dataset runtime failures keep their original `ExitCode`
  - standard help now lists real required/defaulted flags instead of only
    `[ARGS]...`
- Added wrapper-side defaults and resolution:
  - `data convert` auto-defaults `--output` to `<input-stem>.chat.jsonl`
  - `train pretrain` auto-defaults `--out` under `runs/pretrain/...`
  - `train sft` auto-defaults `--out` under `runs/sft/...`
  - `train pretrain --tokenizer <model-dir>` resolves to
    `<model-dir>/tokenizer.json`
  - `train sft` / `train eval` inspect `--model` and surface auto-loaded
    `config.json`, `tokenizer.json`, and `generation_config.json`
  - `--backend auto` now resolves to the compiled training backend
- Added train-side operator tooling:
  - `agent-infer train env --json`
  - `agent-infer train estimate-memory --json`
  - `agent-infer train test --backend metal --json`
- Real validation covered both parser correctness and actual CLI execution:
  - `cargo test -p cli --release --no-default-features --features no-cuda`
  - `cargo clippy -p cli --release --no-default-features --features no-cuda -- -D warnings`
  - `cargo build -p agent-infer --release --no-default-features --features cli,no-cuda`
  - `cargo build -p agent-infer --release --no-default-features --features cli,metal,no-cuda`
  - `cargo fmt --manifest-path infer/Cargo.toml --all -- --check`
  - `agent-infer data convert` now exits `2` on missing required args
  - `agent-infer data convert --help` now shows `--input/--format/--output`
  - `agent-infer train pretrain --help` now shows the real curated flag surface
  - `agent-infer data convert --dry-run --json` shows resolved defaults
  - `agent-infer train pretrain --dry-run --json` shows resolved tokenizer path
    and output dir
  - `agent-infer train sft --dry-run --json` shows resolved model metadata
  - `agent-infer train estimate-memory --model models/Qwen3-0.6B --json`
  - `agent-infer train test --backend metal --json` ran
    `convert -> pretrain -> sft -> eval` end-to-end

## Rule

- For user-facing training DX, prefer typed top-level commands with explicit
  defaults, resolved-plan visibility, and preserved exit-code semantics over a
  raw passthrough wrapper. Keep advanced overrides behind `--`, not hidden from
  the normal help path.
- This change is CLI / operator-DX only; no runtime hot path changed, so no
  performance benchmark entry was required.
