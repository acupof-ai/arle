# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
ARG RUST_TOOLCHAIN=1.98.0

FROM ${CUDA_IMAGE} AS base
ARG RUST_TOOLCHAIN

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV ARLE_CUDA_ENABLE_FA3=1
# FlashMLA's sm90 sparse_fp8 kernels use thread-block clusters that require the
# sm_90a arch variant. build.rs compiles the FlashMLA TUs sm_90a-ONLY, decoupled
# from TORCH_CUDA_ARCH_LIST above, so they build cleanly inside the T1 image and
# the binary carries FlashMLA-sm_90a (dispatched only on sm_90 hardware by the
# runtime gate dsv4_flashmla_decode_enabled, dormant elsewhere). No disable.

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    git \
    build-essential \
    libffi-dev \
    libssl-dev \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rustup component add rustfmt clippy

FROM base AS python-deps

# Build deps (tilelang pin) come from requirements-build.txt — single source
# shared with release.yml and setup.sh. The rest are dev-image extras.
COPY requirements-build.txt /tmp/requirements-build.txt

# requirements-build.txt pins the cu12 stack (torch 2.9.1+cu129 + tilelang
# 0.1.11 + apache-tvm-ffi 0.1.11); it is the only build-critical Python set —
# build.rs imports tilelang for AOT codegen, setup.sh uses huggingface_hub.
# flashinfer-python is intentionally NOT installed here: 0.6.9 pulls
# nvidia-cutlass-dsl → cuda-python 13.x, whose
# ffi double-registers against tilelang's apache-tvm-ffi (`__ffi_repr__ already
# registered`, build.rs:1037) and breaks the cargo build. ARLE's runtime is
# Rust-native (no Python on the hot path) and the published `runtime` stage
# ships only the arle binary, so flashinfer is unused — keep the build green.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-build.txt

FROM python-deps AS dev

WORKDIR /workspace

ENV INFER_TILELANG_PYTHON=/usr/bin/python3

CMD ["bash"]

FROM dev AS builder

# `.cargo/config.toml` pins rustc-wrapper=sccache, but sccache isn't installed
# in the image — cargo would fail on the missing wrapper. Disable it: a one-shot
# container build gets nothing from sccache. Empty value overrides config.toml.
ENV RUSTC_WRAPPER=""

COPY . .

# build.rs runs TileLang AOT, which imports tilelang — vanilla apache-tvm-ffi
# 0.1.12 (from requirements-build.txt) hard-aborts the import (tilelang#2367).
# Rebuild tvm-ffi with the repo's first-wins patch, same as the CI lanes.
RUN scripts/ci-patch-tvm-ffi.sh

# Cache mounts persist the cargo registry and target dir across rebuilds;
# the binary is copied out because cache mounts are not part of the image.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/workspace/target \
    cargo build --release --features cuda,cli -p arle --bin arle \
    && cp target/release/arle /usr/local/bin/arle

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/arle /usr/local/bin/arle

ENV LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64

EXPOSE 8000

ENTRYPOINT ["arle"]
CMD ["--help"]
