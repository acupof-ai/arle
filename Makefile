# ARLE — local development shortcuts
#
# Targets mirror the CI matrix (.github/workflows/{ci,metal-ci}.yml).
#
# Usage:
#   make hygiene                  # public docs/templates/link guardrails
#   make build-metal              # macOS / Apple Silicon → target/release/arle
#   make check-metal              # CI-mirrored Metal checks (infer-api lib + CLI surface)
#   make test-metal               # CI-mirrored Metal tests (cli crate + smoke)
#   make bench-metal              # native throughput bench (Qwen3.6 default)
#   make build-cuda               # Linux / NVIDIA GPU → target/release/arle
#   make kernels-sync             # pull source-matched AOT bundle → generated/ (auto before build-cuda)
#   make check-cuda               # Mac-safe CUDA-Rust typecheck (no nvcc needed)
#   make test                     # any platform (CPU-only, CI-mirrored)
#   make test-py
#   make web-install              # bun install for the web/ landing
#   make web-dev                  # dev server with HMR (Astro+Vite)
#   make web-build                # production build to web/dist/
#   make web-check                # type-check the web/ frontend
#   make web-clean                # remove web/dist + web/.astro + web/node_modules

METAL_MODEL ?= mlx-community/Qwen3.6-35B-A3B-4bit

.PHONY: hygiene build-metal check-metal test-metal bench-metal build-cuda check-cuda kernels-sync test test-py pre-push install-hooks web-install web-dev web-build web-check web-clean

hygiene:
	python3 scripts/check_repo_hygiene.py

# ── Metal (macOS / Apple Silicon) ────────────────────────────────────────────
build-metal:
	cargo build --release --no-default-features --features metal,no-cuda,cli -p arle --bin arle

check-metal:
	cargo check -p infer-api --no-default-features --features metal,no-cuda --lib
	cargo check --no-default-features --features metal,no-cuda,cli -p arle --bin arle

test-metal:
	cargo test -p cli --release --no-default-features --features metal,no-cuda
	cargo test -p arle --release --no-default-features --features metal,no-cuda,cli --test cli_smoke

bench-metal:
	python3 scripts/bench_local_metal.py http://localhost:8000 $(METAL_MODEL)

# ── CUDA (Linux / NVIDIA GPU) ─────────────────────────────────────────────────
# Pull the source-matched AOT bundle (content hash, not git) into generated/ so
# build.rs skips ~1h TileLang codegen; no-op offline / on a miss.
kernels-sync:
	scripts/kernel_artifacts.sh sync

# sccache (when installed) wraps both rustc and nvcc — including the TileLang
# AOT cubins — so kernel rebuilds after a csrc/.cuh touch are cache hits.
build-cuda: kernels-sync
	@if command -v sccache >/dev/null 2>&1; then \
		echo "[build-cuda] sccache detected: wrapping rustc + nvcc"; \
		CUDA_HOME=$${CUDA_HOME:-/usr/local/cuda} \
		RUSTC_WRAPPER=$${RUSTC_WRAPPER:-sccache} \
		ARLE_NVCC_WRAPPER=$${ARLE_NVCC_WRAPPER:-sccache} \
		cargo build --release --features cuda; \
	else \
		CUDA_HOME=$${CUDA_HOME:-/usr/local/cuda} cargo build --release --features cuda; \
	fi

# Mac-safe CUDA-Rust typecheck (CI-mirrored; no nvcc required)
check-cuda:
	cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib

# ── Platform-agnostic ─────────────────────────────────────────────────────────
test:
	cargo test -p arle --release --no-default-features --features cpu,no-cuda,cli
	cargo test --release \
		-p infer-core -p infer-server -p infer-plan -p infer-seam \
		-p infer-moe -p infer-topo -p infer-util -p deepseek-spec -p agent
	cargo test -p chat -p tools -p qwen3-spec -p qwen35-spec -p kv-native-sys --release

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
