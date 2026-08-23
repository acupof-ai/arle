# ARLE dev commands. Run `just` (or `just --list`) to see recipes.
#
# Feature sets follow CLAUDE.md: Metal and CPU builds are
# --no-default-features; CUDA builds need CUDA_HOME.

scripts_dir := "scripts"
arle_bin := "target/release/arle"
# Pin cudarc's CUDA probe so the cuda,no-cuda typecheck runs without nvcc.
cudarc_version := "12080"

# List available recipes.
default:
    @just --list

# ── Build ────────────────────────────────────────────────────────────────────

# Workspace default build (CUDA dev box).
build:
    cargo build --release

# Metal backend (Apple Silicon).
build-metal:
    cargo build --release --no-default-features --features metal,no-cuda

# CUDA backend (Linux + NVIDIA).
build-cuda:
    CUDA_HOME=/usr/local/cuda cargo build --release --features cuda

# Portable / CI smoke build.
build-cpu:
    cargo build --release --no-default-features --features cpu,no-cuda

# ── Test ─────────────────────────────────────────────────────────────────────

# Full workspace test suite.
test:
    cargo test --workspace

# Metal CLI tests (Apple Silicon).
test-metal:
    cargo test -p cli --release --no-default-features --features metal,no-cuda

# CPU smoke lane (CI test-backend feature set, release-fast profile).
test-cpu:
    cargo test -p arle --profile release-fast --no-default-features --features cpu,no-cuda,cli

# Fast unit lane: device-neutral spec/tools crates, no GPU required.
test-unit:
    cargo test --release -p chat -p tools -p qwen3-spec -p qwen35-spec -p kv-native-sys

# ── Clippy ───────────────────────────────────────────────────────────────────

# Workspace lint.
clippy:
    cargo clippy --workspace -- -D warnings

# CUDA-Rust surface typecheck without a GPU toolchain (Mac pre-push lint).
clippy-cuda:
    CUDARC_CUDA_VERSION={{cudarc_version}} cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda,nccl,deepep --lib -- -D warnings

# CPU-only surfaces (CI lint lane).
clippy-cpu:
    cargo clippy -p infer-api --no-default-features --features cpu,no-cuda --lib -- -D warnings
    cargo clippy -p cli --no-default-features --features no-cuda -- -D warnings
    cargo clippy -p arle --no-default-features --features cpu,no-cuda,cli --bin arle -- -D warnings

# ── Format ───────────────────────────────────────────────────────────────────

# Format the workspace.
fmt:
    cargo fmt

# Check formatting without writing (pre-push / CI).
fmt-check:
    cargo fmt --all -- --check

# ── Bench ────────────────────────────────────────────────────────────────────

# Streaming throughput runner against a running serve; forwards args (e.g. `just bench-throughput --seconds 60`).
bench-throughput *ARGS:
    python3 {{scripts_dir}}/bench_throughput.py {{ARGS}}

# Matched A/B driver (forwards args; see scripts/bench_ab.sh header).
bench-ab *ARGS:
    bash {{scripts_dir}}/bench_ab.sh {{ARGS}}

# ── Correctness gates ────────────────────────────────────────────────────────

# Metal lever gate: needle ladder + temp + concurrent arms.
# Usage: `just gate-metal <label>`
gate-metal LABEL:
    GATE_PROFILE=metal bash {{scripts_dir}}/lever_gate.sh {{LABEL}}

# DSv4 lever gate (8xH20, TP=8; reserves eight free SM90 GPUs).
# Usage: `just gate-dsv4 <label>`
gate-dsv4 LABEL:
    GATE_PROFILE=dsv4 bash {{scripts_dir}}/lever_gate.sh {{LABEL}}

# Standalone needle gate (needs a running serve on $PORT, default 18189).
needle:
    python3 {{scripts_dir}}/needle_gate.py --check

# ── Misc ─────────────────────────────────────────────────────────────────────

# Remove build artifacts.
clean:
    cargo clean

# Environment self-check.
doctor:
    {{arle_bin}} --doctor

# Repo hygiene checks (wins cap, etc.).
hygiene:
    python3 {{scripts_dir}}/check_repo_hygiene.py

# Full CI-aligned pre-push validation (Metal arms: ARLE_PRE_PUSH_METAL=1).
prepush:
    bash {{scripts_dir}}/pre_push_checks.sh
