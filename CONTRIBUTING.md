# Contributing to ARLE

Thanks for your interest in contributing! This document is the main entry point
for contribution workflow.

If your change is user-visible, compatibility-sensitive, performance-sensitive,
or release-related, also read:

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)
- [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [docs/release-checklist.md](docs/release-checklist.md)
- [docs/environment.md](docs/environment.md)

## Getting Started

```bash
git clone https://github.com/cklxx/arle && cd arle
./setup.sh                 # Installs Rust, Python venv, builds, downloads model
./setup.sh --check         # Linux/CUDA workstation check
make hygiene               # Public docs/templates/link guardrails
make pre-push              # CI-aligned snapshot validation
make check-metal           # Apple Silicon quick check
```

Or manually:

1. **Rust**: use the pinned toolchain from [`rust-toolchain.toml`](rust-toolchain.toml)
2. **CUDA 12.x** (for GPU builds)
3. **Python 3.10+** with `flashinfer-python` and `triton` (build-time only)

For first-time contributor setup, install the repo-managed hook path:

```bash
make install-hooks
make pre-push
```

## Development Workflow

```bash
# Build (CPU-only, fast iteration)
cargo build --no-default-features --features no-cuda

# Build the CLI smoke path
cargo build -p agent-infer --release --no-default-features --features cpu,no-cuda,cli --bin arle

# Build (GPU). The cuda feature is no longer the default — pass it explicitly:
cargo build -p infer --release --features cuda

# Test
cargo test --no-default-features --features no-cuda   # Unit tests (~9s)
cargo test --release --features cuda --test e2e        # E2E (GPU required)
cargo test -p train --release --features no-cuda --lib
cargo test -p autograd --release --features no-cuda --lib
cargo test -p agent-infer --release --no-default-features --features metal,no-cuda,cli --test cli_tiny_fixture_live

# Lint + format
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
cargo deny check advisories bans licenses sources
```

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Follow [Commitizen](https://www.conventionalcommits.org/) format: `<type>(<scope>): <subject>`
   - Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`
3. Ensure CI passes: `make hygiene`, `cargo test`, `cargo clippy`, `cargo fmt --check`, `cargo deny`
4. One logical change per PR. Keep diffs focused.
5. If the change affects a documented API, CLI behavior, environment variable,
   benchmark claim, or migration-sensitive workflow, include the relevant docs
   updates in the same PR.

## PR Readiness Checklist

Before opening a PR, make sure you can answer these clearly:

1. **What surface changed?**
   - internal implementation
   - documented API / CLI
   - backend/runtime path
   - benchmark / tooling / docs
2. **What stability level does it belong to?**
   - stable, beta, experimental, or internal
3. **What validation did you run?**
   - unit / contract / integration / e2e / benchmark / profiling
4. **Does it change support expectations or compatibility?**
   - if yes, update the relevant docs and changelog
5. **Does it claim a performance win?**
   - if yes, include before/after evidence and command context

If you cannot answer those questions yet, the PR is not ready.

## Code Conventions

- **Flat module layout**: `src/ops.rs` + `src/ops/` (no `mod.rs`)
- **GPU/CPU separation**: GPU-only code behind `#[cfg(feature = "cuda")]`
- **Error handling**: `anyhow::Result` for internal, structured `ApiError` for HTTP
- **Always `--release`**: Debug builds are unusably slow for GPU work

## Compatibility and Support Rules

- Treat documented HTTP APIs, documented CLI behavior, and documented
  environment variables as compatibility-sensitive.
- Do not treat internal modules as public extension points unless the docs say
  they are stable.
- If support status changes for a backend, model family, quantization path, or
  platform combination, update `README.md`, `docs/support-matrix.md`, and
  `CHANGELOG.md` together.
- If you deprecate or replace a documented surface, include migration guidance.

See [docs/stability-policy.md](docs/stability-policy.md) for the full policy.

## Validation Expectations

Use the lightest meaningful validation first, then broaden based on risk.

- **CPU-side logic / pure Rust helpers**: run targeted tests plus the CPU-only
  test path.
- **HTTP / protocol / CLI behavior**: run targeted tests and update docs when
  behavior changes.
- **CUDA / Metal / scheduler runtime / quantization changes**: run targeted
  correctness checks and include benchmark evidence when claiming a performance
  improvement.

The CLI now has one canonical tiny-fixture path for real backend validation:
`arle train test --out-dir <tmp>` leaves a checkpoint at `<tmp>/sft/latest`,
and the Metal/CUDA CI lanes reuse that artifact for `arle run --json` coverage.
GitHub-hosted macOS runners cover Metal; CUDA CI lives in the dedicated
`.github/workflows/cuda-ci.yml` self-hosted GPU lane.

See [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
for the detailed matrix.

## Environment Variables

Use [docs/environment.md](docs/environment.md) as the source of truth for:

- CLI runtime variables
- build/toolchain variables
- test and integration variables
- setup-related variables

If you add, rename, or deprecate an environment variable, update that document
in the same PR.

## Dependency Hygiene

- Rust and GitHub Actions dependencies are updated via `dependabot.yml`.
- Supply-chain policy is checked with `cargo deny` using [`deny.toml`](deny.toml).
- If you add a dependency with a new license or a new registry source, update
  `deny.toml` in the same PR.

## Release Work

If you are preparing a release, use
[docs/release-checklist.md](docs/release-checklist.md).

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

If reporting a performance regression, include hardware, model, command, before
result, after result, and whether the regression is correctness-related or only
throughput/latency-related.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
