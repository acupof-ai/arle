# Contributing to agent-infer

Thanks for your interest in contributing! This document covers the essentials.

## Getting Started

```bash
git clone https://github.com/cklxx/agent-infer && cd agent-infer
./setup.sh --full          # Installs Rust, Python venv, builds, downloads model
```

Or manually:

1. **Rust 1.85+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. **CUDA 12.x** (for GPU builds)
3. **Python 3.10+** with `flashinfer-python` and `triton` (build-time only)

## Development Workflow

```bash
# Build (CPU-only, fast iteration)
cargo build --no-default-features --features no-cuda

# Build (GPU)
cargo build -p infer --release

# Test
cargo test --no-default-features --features no-cuda   # Unit tests (~9s)
cargo test --release --test e2e                         # E2E (GPU required)

# Lint + format
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Follow [Commitizen](https://www.conventionalcommits.org/) format: `<type>(<scope>): <subject>`
   - Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`
3. Ensure CI passes: `cargo test`, `cargo clippy`, `cargo fmt --check`
4. One logical change per PR. Keep diffs focused.

## Code Conventions

- **Flat module layout**: `src/ops.rs` + `src/ops/` (no `mod.rs`)
- **GPU/CPU separation**: GPU-only code behind `#[cfg(feature = "cuda")]`
- **Error handling**: `anyhow::Result` for internal, structured `ApiError` for HTTP
- **Always `--release`**: Debug builds are unusably slow for GPU work

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.

Key entry points:
- `infer/src/model.rs` — `ModelForward` trait (start here for new models)
- `infer/src/scheduler/` — Continuous batching scheduler
- `infer/src/ops/` — GPU operations (attention, norm, sampling)

## Adding a New Model

1. Create `infer/src/model/<name>/` with `config.rs`, `weights.rs`, `forward.rs`
2. Implement `ModelForward` trait
3. Register architecture in `infer/src/model_registry.rs`
4. Add E2E test baseline in `infer/test_data/`

## Reporting Issues

Use the [issue templates](.github/ISSUE_TEMPLATE/) for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
