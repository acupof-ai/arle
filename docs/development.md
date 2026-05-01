# Development Container

ARLE's `dev` Docker target is a CUDA/Rust/Zig/Python toolchain container for
building and checking the CUDA backend. The default final Dockerfile stage
remains the release runtime image used by `ghcr.io/cklxx/arle:latest`.

Build the local image:

```bash
./scripts/docker_build_dev.sh
```

The script tags the image as `arle-dev:<git-sha>`. To run an interactive shell:

```bash
docker run --rm -it --gpus all -v "$PWD:/workspace" arle-dev:$(git rev-parse --short=8 HEAD)
```

The image contains:

- CUDA 12.8 devel toolkit and `nvcc`
- Rust `1.95.0` with `rustfmt` and `clippy`
- Zig `0.16.0` at `/usr/local/bin/zig`
- Python packages for CUDA build and benchmark workflows: Torch, FlashInfer,
  TileLang, GuideLLM, and Hugging Face Hub

Useful checks inside the image:

```bash
cargo --version
nvcc --version
zig version
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda
```

Push the image after `docker login ghcr.io`:

```bash
./scripts/docker_push.sh
```

The pushed tag is `ghcr.io/cklxx/arle:dev-<git-sha>`.
