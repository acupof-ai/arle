# Installing ARLE

ARLE ships pre-built binaries on every `v*` tag. Three install paths cover
the supported platforms; pick the one that matches your environment.

## Support matrix

| Platform | Backend | Binaries shipped |
|---|---|---|
| macOS arm64 (Apple Silicon) | Metal / MLX | `arle`, `metal_serve` |
| Linux x86_64 | CUDA 12.x (driver required on host) | `arle`, `infer`, `bench_serving` |
| Other (macOS x86_64, Linux aarch64, Windows) | — | Build from source |

CUDA binaries are linked against `cudart` 12.x and need a matching NVIDIA
driver / CUDA runtime present on the host. Metal binaries need macOS 14+.

## 1. Homebrew (macOS arm64)

```bash
brew install cklxx/tap/arle
arle --doctor
```

Tap source: <https://github.com/cklxx/homebrew-tap>. The formula is bumped
automatically on every `v*` tag from this repo's
[release workflow](../.github/workflows/release.yml).

To upgrade:

```bash
brew update && brew upgrade arle
```

To uninstall:

```bash
brew uninstall arle
brew untap cklxx/tap   # optional: remove the tap entirely
```

## 2. One-line installer (macOS arm64 / Linux x86_64)

```bash
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh
```

What it does:

1. Detects platform via `uname -s` / `uname -m`.
2. Resolves the `latest` tag through GitHub's redirect.
3. Downloads `arle-<tag>-<platform>.tar.gz` and `SHA256SUMS.txt`.
4. Verifies the SHA256 (uses `sha256sum` or `shasum -a 256`).
5. Extracts and `install -m 0755`s the binaries into `$INSTALL_DIR`
   (default `~/.local/bin`).
6. Prints a `PATH` hint if `$INSTALL_DIR` is not on `PATH`.

### Environment overrides

| Variable | Default | Effect |
|---|---|---|
| `ARLE_VERSION` | `latest` | Pin to a specific tag, e.g. `v0.1.0`. |
| `INSTALL_DIR` | `$HOME/.local/bin` | Where binaries land. Use `/usr/local/bin` for system-wide (needs `sudo`). |
| `ARLE_NO_VERIFY` | unset | If set, skip SHA256 verification (not recommended). |

Examples:

```bash
# Pin a version, system-wide:
curl -fsSL https://github.com/cklxx/arle/releases/download/v0.1.0/install.sh \
  | INSTALL_DIR=/usr/local/bin sudo sh

# Inspect the script before running:
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh -o install.sh
less install.sh
sh install.sh
```

To uninstall, just delete the binaries:

```bash
rm -f ~/.local/bin/arle ~/.local/bin/metal_serve ~/.local/bin/infer ~/.local/bin/bench_serving
```

## 3. Docker (Linux + NVIDIA)

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-4B:/model:ro \
  ghcr.io/cklxx/arle:latest \
  serve --backend cuda --model-path /model --port 8000
```

The `:latest` tag tracks `main`. Tagged releases are published as
`ghcr.io/cklxx/arle:X.Y.Z` (no `v` prefix — docker metadata-action strips it).

## 4. From source

Required for the `cpu` backend, CUDA/TileLang, or hacking on the runtime.
See the [README Quick Start](../README.md#quick-start) for the canonical
`cargo build` invocations per backend, and [environment.md](environment.md)
for the env-var knobs that affect the build.

## Verifying an install

```bash
arle --doctor          # human-readable
arle --doctor --json   # machine-readable, suitable for CI gates
```

`--doctor` prints the compiled backend, runtime feature flags, and a
self-check of the model-loading path. If it errors out, the most common
causes are documented in [troubleshooting.md](troubleshooting.md).
