# KV SSD default root

## Goal

Make local SSD tier usage ergonomic: users should not need to spell an
absolute cache directory for the common local path.

## Hypothesis

The operator still needs an explicit opt-in for SSD tiering, but that opt-in can
resolve to a stable OS cache root when no path is supplied. This keeps default
serving unchanged while allowing `--kv-ssd` or `--kv-ssd-max-bytes` to use local
SSD immediately.

## Params

- Added `arle serve --kv-ssd` as an explicit default-root opt-in.
- `--kv-ssd-max-bytes` no longer requires `--kv-ssd-path`; when supplied alone
  it also enables the tier at the default root.
- Default root resolution:
  - `ARLE_KV_SSD_PATH` when set.
  - macOS: `~/Library/Caches/arle/kv-ssd`.
  - Linux/Unix: `$XDG_CACHE_HOME/arle/kv-ssd` or `~/.cache/arle/kv-ssd`.
  - Windows: `%LOCALAPPDATA%\\arle\\kv-ssd`.
- The serve boundary creates the root directory if missing, then still requires
  an absolute directory.

## Env

Local Apple Silicon macOS. Metal smoke used
`mlx-community/Qwen3.5-0.8B-MLX-4bit`.

## Results

- `cargo fmt --check`: passed.
- `cargo test -p cli --release --no-default-features --features metal,no-cuda kv_ssd -- --nocapture`: 3 passed.
- `cargo test -p infer-api --release --no-default-features --features metal,no-cuda kv_ssd -- --nocapture`: 3 passed, adapter 0 tests.
- `cargo clippy -p cli --release --no-default-features --features metal,no-cuda -- -D warnings`: passed.
- `cargo clippy -p infer-api --release --no-default-features --features metal,no-cuda -- -D warnings`: passed.
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`: passed.

Local smoke:

```bash
/opt/homebrew/bin/timeout 45s cargo run --release \
  --no-default-features --features metal,no-cuda,cli -- serve \
  --backend metal \
  --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --kv-ssd \
  --kv-ssd-max-bytes 1073741824 \
  --port 0
```

Observed:

- Serve mounted the SSD tier under
  `/Users/bytedance/Library/Caches/arle/kv-ssd/arle-metal-kv-tier-...`.
- Budget was `1073741824` bytes, `capacity_pages=10591`.
- Timeout exit code 124 was the expected hard stop after mount verification.

## Problems

This does not make SSD tiering default-on. It only removes the path spelling
requirement after explicit opt-in. Performance impact remains covered by the
separate Metal T2 work item.

## Learnings

The path should be an implementation detail for local single-process SSD cache
use. Operators can still override it, but the common path should be one flag
away.
