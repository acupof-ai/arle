# UX polish: fail-fast model paths + Display errors in train_sft

## Context

Two user-reported papercuts:

1. `agent-infer --model-path /does/not/exist` fell through to a HuggingFace
   Hub lookup and errored from deep inside the network path — user expected
   an immediate "path not found".
2. `train_sft` surfaced `std::io::Error` as
   `Os { code: 2, kind: NotFound, message: "..." }` — the default `Debug`
   format of `fn main() -> Result<(), E>`.

## What Worked

- Added `looks_like_local_path` heuristic in `infer/src/hf_hub.rs`
  (absolute / `./` / `../` / `~` prefixes, backslash, or multiple `/`) and
  short-circuited `resolve_model_path` when the input shape is path-like
  but no local candidate exists. `Qwen/Qwen3-0.6B`-style repo ids still
  fall through to the Hub.
- Converted `train_sft`'s `fn main() -> Result<(), CliError>` into
  `fn main() -> ExitCode` + `fn run()`. Errors now print via `Display`
  (`{err}`), which `thiserror`'s `#[error(transparent)]` already delegates
  to the inner `std::io::Error` Display — no more `Os { ... }` blob.
- Manual check:
  `agent-infer --model-path /does/not/exist ...` → exit 1, message
  `model path does not exist: '/does/not/exist' (looks like a filesystem
  path, skipping HuggingFace Hub lookup)`, no Hub traffic.
  `train_sft --data /does/not/exist.jsonl ...` →
  `[train_sft] error: failed to open SFT JSONL /does/not/exist.jsonl:
  No such file or directory (os error 2)`.

## Rule

Pure-UX / error-reporting diffs in the binaries do not move runtime numbers
— bench-exempt per `CLAUDE.md` §Benchmarks (commit body states so). When a
`fn main() -> Result<(), E>` surfaces io errors to the user, wrap it in a
runner that prints `{err}` / `{err:#}` and returns `ExitCode`; the default
`Debug` path leaks `Os { ... }` internals.
