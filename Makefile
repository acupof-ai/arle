# agent-infer — local development shortcuts
#
# Usage:
#   make build-metal               # macOS / Apple Silicon
#   make build-agent-metal         # macOS / Apple Silicon CLI
#   make check-metal               # verify infer + agent-infer Metal surfaces
#   make test-metal                # run Metal lib tests serially
#   make bench-metal METAL_MODEL=models/Qwen3-0.6B-4bit
#   make build-cuda                # Linux / NVIDIA GPU
#   make bench-cuda
#   make test                      # any platform (CPU-only)
#   make test-py

METAL_MODEL ?= models/Qwen3-0.6B-4bit
CUDA_MODEL ?= models/Qwen3-4B

.PHONY: build-metal build-agent-metal check-metal test-metal bench-metal bench-metal-compare build-cuda bench-cuda test test-py pre-push install-hooks

# ── Metal (macOS / Apple Silicon) ────────────────────────────────────────────
build-metal:
	cargo build --release --no-default-features --features metal,no-cuda -p infer

build-agent-metal:
	cargo build --release --no-default-features --features metal,no-cuda,cli -p agent-infer

check-metal:
	cargo check --no-default-features --features metal,no-cuda -p infer --lib
	cargo check --no-default-features --features metal,no-cuda,cli -p agent-infer

test-metal:
	cargo test --no-default-features --features metal,no-cuda -p infer --lib -- --test-threads 1

bench-metal:
	cargo run --release -p infer --no-default-features --features metal,no-cuda \
		--bin metal_bench -- --model $(METAL_MODEL)

bench-metal-compare:
	cargo run --release -p infer --no-default-features --features metal,no-cuda \
		--bin metal_bench -- --model $(METAL_MODEL) --compare-baseline benchmarks/metal_baseline.json

# ── CUDA (Linux / NVIDIA GPU) ─────────────────────────────────────────────────
build-cuda:
	cargo build --release --features cuda -p infer

bench-cuda:
	cargo run --release --features cuda --bin bench_serving

# ── Platform-agnostic ─────────────────────────────────────────────────────────
test:
	cargo test --no-default-features --features no-cuda -p infer --lib

test-py:
	pytest tests/ -x

pre-push:
	./scripts/pre_push_checks.sh

install-hooks:
	./scripts/install_git_hooks.sh
