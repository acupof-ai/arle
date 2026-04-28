# ARLE — local development shortcuts
#
# Usage:
#   make hygiene                  # public docs/templates/link guardrails
#   make build-metal               # macOS / Apple Silicon
#   make build-agent-metal         # macOS / Apple Silicon CLI
#   make check-metal               # verify infer + arle Metal surfaces
#   make test-metal                # run Metal lib tests serially
#   make bench-metal METAL_MODEL=models/Qwen3-0.6B-4bit
#   make build-cuda                # Linux / NVIDIA GPU
#   make bench-cuda
#   make test                      # any platform (CPU-only)
#   make test-py
#   make web-install               # bun install for the web/ landing
#   make web-dev                   # dev server with HMR (Astro+Vite)
#   make web-build                 # production build to web/dist/
#   make web-check                 # type-check the web/ frontend
#   make web-clean                 # remove web/dist + web/.astro + web/node_modules

METAL_MODEL ?= models/Qwen3-0.6B-4bit
CUDA_MODEL ?= models/Qwen3-4B

.PHONY: hygiene build-metal build-agent-metal check-metal test-metal bench-metal bench-metal-compare build-cuda bench-cuda test test-py pre-push install-hooks web-install web-dev web-build web-check web-clean

hygiene:
	python3 scripts/check_repo_hygiene.py

# ── Metal (macOS / Apple Silicon) ────────────────────────────────────────────
build-metal:
	cargo build --release --no-default-features --features metal,no-cuda -p infer

build-agent-metal:
	cargo build --release --no-default-features --features metal,no-cuda,cli -p agent-infer --bin arle

check-metal:
	cargo check --no-default-features --features metal,no-cuda -p infer --lib
	cargo check --no-default-features --features metal,no-cuda,cli -p agent-infer --bin arle

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
	pytest tests/python/ -x

pre-push:
	./scripts/pre_push_checks.sh

install-hooks:
	git config core.hooksPath .githooks
	@echo "[install-hooks] configured core.hooksPath=.githooks"

# ── Web frontend (web/ — Astro 5 + Vite + bun) ───────────────────────────────
# Drives the public landing at https://cklxx.github.io/arle/. Requires bun on
# PATH; `./setup.sh --web-only` will bootstrap it if missing.
web-install:
	cd web && bun install --frozen-lockfile

web-dev:
	cd web && bun run dev

web-build:
	cd web && bun install --frozen-lockfile && bun run build

web-check:
	cd web && bun run check

web-clean:
	rm -rf web/dist web/.astro web/node_modules
