# Troubleshooting

Common errors and how to resolve them. If your problem is not here, open a
[GitHub issue](https://github.com/acupof-ai/arle/issues/new) with the output of
`arle --doctor --json` and the exact command you ran.

---

## Build / install

### `nvcc not found` / cudarc fails to build (any OS)

ARLE no longer enables the `cuda` feature by default — the previous default
forced macOS users to type `--no-default-features --features metal,no-cuda,cli`
on every command. After the 2026-04-26 default-features cleanup, pick a backend
explicitly:

```bash
# Linux + NVIDIA
cargo build --release --features cuda

# Apple Silicon
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle

# CPU-only smoke (no GPU)
cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
```

`cargo build` with no flags builds a backend-less `arle` binary; `arle --doctor`
will report `bare` and `arle serve --backend auto` will refuse to start with an
actionable message.

### `error: could not execute process 'sccache .../rustc -vV' (never executed)`

`RUSTC_WRAPPER` (or `~/.cargo/config.toml` `[build] rustc-wrapper`) points to
`sccache`, but it is not installed on this host — the wrapper binary itself
can't be found, so no compile ever starts. Either install it
(`cargo install sccache` or the system package) or clear it for this build:

```bash
RUSTC_WRAPPER= cargo build --release --features cuda --bin arle
```

### `error: linker 'cc' not found` on Linux

Install build essentials: `apt install -y build-essential pkg-config` (Debian /
Ubuntu) or the equivalent. Several `-sys` crates also need `clang` /
`libclang-dev` (bindgen) and `cmake`; `setup.sh` installs the full native-dep
set.

### TileLang AOT build dep fails to install

TileLang is the one build-time-only Python dep on the CUDA AOT path (attention /
GDR kernel codegen); it is pinned in
[`requirements-build.txt`](../requirements-build.txt) to the version the 8×H20
pod runs. FlashInfer needs no pip step — its kernels are vendored as C++ headers
under `crates/cuda-kernels/csrc/vendor/flashinfer/` and compiled by nvcc (the
Triton-AOT lane and its wheel were deleted in #88, `23d6a0b8`). If the TileLang
install fails, verify `nvidia-smi` reports a GPU and that
`$CUDA_HOME/bin/nvcc --version` matches the pinned CUDA major (12.x).

### `pip install -e ".[bench|dev|observe|serve]"` fails with "no such package"

Run from the repo root. The `.` resolves to the local
[`pyproject.toml`](../pyproject.toml), which is a private deps bundle (renamed
to `arle-pytools` to avoid being confused with a publishable PyPI package).

---

## Runtime

### `arle --doctor` reports `Compiled backend: bare`

You built without selecting a backend feature. Rebuild with one of `cuda` /
`metal,no-cuda` / `cpu,no-cuda` (see the build section above).

### `arle serve` exits with `serve requires a backend build; rebuild with cuda, metal/no-cuda, or cpu/no-cuda`

Same root cause as the `bare` doctor message — backend feature was not
compiled in. Pass the matching `--features` flag at build time.

### Which `model` name do I send?

Any string. The server routes every request to the single served model
reported by `GET /v1/models`, so Claude Code's `claude-*` ids and the OpenAI
SDK's `model="default"` both work. Omitting `model` is accepted too.

### Server is up but the first request hangs or returns 503

`GET /health` answers as soon as the HTTP layer is bound; weight loading and
warm-up happen before that, so a bound port means the model is loaded. If a
request still stalls, `GET /v1/stats` shows the scheduler state, and stderr
carries the underlying error for a failed load (typically a missing tokenizer
or an incompatible weight format).

### `bind: address already in use` on `:8000`

Another process is bound to port 8000 (often a previous `arle serve` that did
not exit cleanly). `lsof -i :8000` will show it; `kill <pid>` or pick a new
port: `arle serve --port 8010 ...`.

### `arle serve` says the requested backend is unavailable

Serving is **in-process** — the workspace ships a single `arle` binary and
`arle serve` loads the model in the same process (`crates/cli/src/serve.rs`).
There are no standalone `infer` / `metal_serve` / `cpu_serve` binaries to
find on `PATH`; if the requested `--backend` is not compiled into the binary
you are running, rebuild with the matching feature:

```bash
# Apple Silicon
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle

# Linux + NVIDIA
cargo build --release --features cuda --bin arle
```

`arle --doctor` reports which backend the current binary was compiled with.
The release tarballs at
[GitHub Releases](https://github.com/acupof-ai/arle/releases) ship the `arle`
binary built for the platform's backend. `--bind` is honored in-process by
every backend (the old "Metal-only `--bind`" limitation died with the
monolith's spawn-a-binary front door).

---

## Tests / CI

### `cargo test --release --test e2e` fails with `cuda` not found

E2E tests now require explicit `--features cuda`:

```bash
cargo test --release --features cuda --test e2e
```

### `pytest tests/` finds nothing

Python tests live under [`tests/python/`](../tests/python/) (split from the
Rust integration tests at `tests/*.rs`). Run `pytest tests/python/` or
`make test-py`.

---

## Reporting a problem

When opening an issue, please include:

- Output of `arle --doctor --json`
- The exact command you ran
- The full stderr (not just the last line)
- OS + GPU + driver version (Linux) or chip + macOS version (Apple Silicon)

For runtime crashes that look like a kernel / scheduler bug, attach a
[`scripts/bench_throughput.py`](../scripts/bench_throughput.py) repro at the
smallest concurrency that reproduces the failure.
