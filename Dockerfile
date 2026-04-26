# ============================================================================
# ARLE: multi-stage Docker build
# ============================================================================
# Build:  docker build -t arle .
# Run:    docker run --gpus all -v /path/to/model:/model ghcr.io/cklxx/arle:latest serve --backend cuda --model-path /model
# ============================================================================

# --- Stage 1: Build ---
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# Install Python build deps (Triton AOT + FlashInfer)
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir flashinfer-python==0.6.3 triton==3.5.1

ENV INFER_TRITON_PYTHON=/opt/venv/bin/python3
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /build
COPY . .

RUN cargo build -p infer --release --features cuda && \
    cargo build --release --features cuda,cli -p agent-infer --bin arle

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/infer /usr/local/bin/infer
COPY --from=builder /build/target/release/arle /usr/local/bin/arle
RUN ln -s /usr/local/bin/arle /usr/local/bin/agent-infer

ENV LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64

EXPOSE 8000

ENTRYPOINT ["arle"]
CMD ["--help"]
