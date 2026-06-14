# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
ARG RUST_TOOLCHAIN=1.95.0

FROM ${CUDA_IMAGE} AS base
ARG RUST_TOOLCHAIN

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# FlashMLA's sm90 sparse_fp8 kernels use thread-block clusters (launch_bounds
# CLUSTER_SIZE) that require sm_90a, but the T1 arch set above uses plain 9.0 →
# nvcc hard-fails (build.rs:1783). FlashMLA is a pod-only sm_90a artifact per
# docs/plans/sm-coverage.md (DSv4-Flash needs the 8xH20 pod toolchain anyway),
# so disable it in the release image. build.rs swaps in a clean stub; the
# runtime gate dsv4_flashmla_decode_enabled defaults OFF — no half-state.
ENV ARLE_CUDA_DISABLE_FLASHMLA=1

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
# guidellm is the dev-image bench client. flashinfer-python is intentionally
# NOT installed here: 0.6.9 pulls nvidia-cutlass-dsl → cuda-python 13.x, whose
# ffi double-registers against tilelang's apache-tvm-ffi (`__ffi_repr__ already
# registered`, build.rs:1037) and breaks the cargo build. ARLE's runtime is
# Rust-native (no Python on the hot path) and the published `runtime` stage
# ships only the arle binary, so flashinfer is unused — keep the build green.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-build.txt \
    && python3 -m pip install --no-cache-dir \
      "guidellm[recommended]==0.6.0"

FROM python-deps AS dev

WORKDIR /workspace

ENV INFER_TILELANG_PYTHON=/usr/bin/python3

CMD ["bash"]

FROM dev AS builder

COPY . .

# Cache mounts persist the cargo registry and target dir across rebuilds;
# the binary is copied out because cache mounts are not part of the image.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/workspace/target \
    cargo build --release --features cuda,cli -p agent-infer --bin arle \
    && cp target/release/arle /usr/local/bin/arle

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/arle /usr/local/bin/arle
RUN ln -s /usr/local/bin/arle /usr/local/bin/agent-infer

ENV LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64

EXPOSE 8000

ENTRYPOINT ["arle"]
CMD ["--help"]
