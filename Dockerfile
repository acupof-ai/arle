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

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
      torch \
      flashinfer-python==0.6.9 \
      tilelang \
      "guidellm[recommended]==0.6.0" \
      huggingface_hub==0.36.2

FROM python-deps AS dev

WORKDIR /workspace

ENV INFER_TRITON_PYTHON=/usr/bin/python3
ENV INFER_TILELANG_PYTHON=/usr/bin/python3

CMD ["bash"]

FROM dev AS builder

COPY . .

RUN cargo build -p infer --release --features cuda \
    && cargo build --release --features cuda,cli -p agent-infer --bin arle

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/target/release/infer /usr/local/bin/infer
COPY --from=builder /workspace/target/release/arle /usr/local/bin/arle
RUN ln -s /usr/local/bin/arle /usr/local/bin/agent-infer

ENV LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64

EXPOSE 8000

ENTRYPOINT ["arle"]
CMD ["--help"]
