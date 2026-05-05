# Troubleshooting

Common errors and how to resolve them. If your problem is not here, open a
[GitHub issue](https://github.com/cklxx/arle/issues/new) with the output of
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

### `error: linker 'cc' not found` on Linux

Install build essentials: `apt install -y build-essential pkg-config` (Debian /
Ubuntu) or the equivalent. CUDA users also need `clang` for some FlashInfer
kernels.

### `flashinfer-python` install fails

FlashInfer is a build-time-only Python dep used by the CUDA AOT path. It needs
CUDA 12.x and a matching Triton wheel. The repo pins both in
[`requirements-build.txt`](../requirements-build.txt). If `pip install` fails,
verify `nvidia-smi` reports a GPU and that `$CUDA_HOME/bin/nvcc --version`
matches the pinned major (12.8 today).

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

### `Model 'qwen3-4b' is not available on this server; loaded model is 'Qwen3-0.6B-4bit'`

The server matches the request body's `model` field (case-insensitive, last
path segment) against the loaded model's id. Either:

1. Match the loaded model name in your client request, or
2. Use [`examples/openai_chat.py`](../examples/openai_chat.py), which lists
   `/v1/models` first and uses whatever the server reports, or
3. Set `ARLE_MODEL=<id>` to override the default in the example.

Note: omitting the `model` field entirely is also accepted — the server uses
the loaded model.

### Server starts but `/healthz` is fine and `/readyz` is 503

The scheduler reports not-ready while it is loading weights or warming up. Wait
a few seconds and retry; for prebuilt CUDA images the warm-up usually finishes
within ~10s of the binary launching. Persistent 503s indicate the model failed
to load — see stderr for the underlying error (typically a missing tokenizer
or an incompatible weight format).

### `bind: address already in use` on `:8000`

Another process is bound to port 8000 (often a previous `arle serve` that did
not exit cleanly). `lsof -i :8000` will show it; `kill <pid>` or pick a new
port: `arle serve --port 8010 ...`.

### Apple Silicon `arle serve` runs but bind warning says "metal-only flag"

`--bind 0.0.0.0` (or any value other than `127.0.0.1`) is currently only
supported on the Metal serving binary. CUDA / CPU dispatch silently falls back
to the backend's default bind. This is a documented Beta gap; use
`127.0.0.1` for those backends or run behind a reverse proxy.

### `metal_serve` / `cpu_serve` / `infer` not found by `arle serve`

`arle serve` looks for the matching binary next to the `arle` executable
first, then on `PATH`. If you built only `arle` you also need to build the
serving binary:

```bash
# Apple Silicon
cargo build --release --no-default-features --features metal,no-cuda --bin metal_serve -p infer

# Linux + NVIDIA
cargo build --release --features cuda -p infer
```

The release tarballs at
[GitHub Releases](https://github.com/cklxx/arle/releases) ship `arle` next to
the matching backend binary, so `arle serve` "just works" for the prebuilt
artifact.

---

## Tests / CI

### `cargo test --release --test e2e` fails with `cuda` not found

E2E tests now require explicit `--features cuda`:

```bash
cargo test --release --features cuda --test e2e
```

### Hygiene check fails: missing marker `## 📰 Latest Updates`

`scripts/check_repo_hygiene.py` enforces that README.md keeps the
"Latest Updates" emoji marker (and zh-CN keeps `## 📰 最新动态`). If you slim
the README, keep the marker even if the section is short — the hygiene script
parses on it.

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
[`scripts/bench_guidellm.sh`](../scripts/bench_guidellm.sh) repro at the
smallest concurrency that reproduces the failure.
