# Multi-SM coverage policy

**Status:** Phase A (tier policy + env var migration) landing 2026-04-28.
Owner: ckl. Verification: T1 four-card bench gate (sm_80/86/89/90).
Cross-link: [`tilelang-integration-verification.md`](tilelang-integration-verification.md)
for SM-specific bench thresholds, [`cuda-kernel-crate-extraction.md`](cuda-kernel-crate-extraction.md)
for the kernel-crate boundary this policy enforces.

---

## 1 · Why this plan exists

ARLE compiles a **single CUDA binary per host** today: `nvcc` SASS is one
SM, Triton AOT cubin is one SM (highest detected), TileLang AOT cubin is
one SM. Cross-SM cubin loading fails with `cuModuleLoadData →
CUDA_ERROR_INVALID_SOURCE` (see [`tilelang-integration-verification.md`](tilelang-integration-verification.md) §0).
Default fallback is `sm_80`, which silently accepts T3 (Volta/Turing) and
emits an unrunnable artifact.

The shipped target is one release binary that runs natively on the four
mainstream T1 SMs without rebuild, with a clear opt-in path for Blackwell
(T2) and an explicit reject for older hardware (T3).

---

## 2 · Tier policy

| Tier | SM | GPU 代表                           | 默认接入 | AOT 要求                                                   |
|------|----|-----------------------------------|---------|-----------------------------------------------------------|
| T1   | 80 | A100                              | yes     | every AOT kernel must emit cubin; build fails otherwise   |
| T1   | 86 | A10 / RTX 3090                    | yes     | same                                                      |
| T1   | 89 | L4 / RTX 4090                     | yes     | same                                                      |
| T1   | 90 | H100                              | yes     | same                                                      |
| T2   | 100| B100 / B200                       | no — opt-in via `TORCH_CUDA_ARCH_LIST` | same as T1 once opted in |
| T2   | 120| RTX 5090 / RTX PRO 6000           | no — opt-in via `TORCH_CUDA_ARCH_LIST` | same as T1 once opted in |
| T3   | < 80 | V100 / T4 / Pascal / older      | rejected at build time | n/a — `panic!` with hint |

**Why T2 is opt-in.** Triton 3.6 has working evidence on sm_120 (vLLM
[PR #31089](https://github.com/vllm-project/vllm/pull/31089)) but not all
12 Triton kernels have been validated. TileLang has zero upstream issues
for sm_120 in `tile-ai/tilelang`. Defaulting to T2 would make build hard
fails the common case for users on T1-only hardware.

**Why T3 is rejected.** Marlin (`csrc/gemm/marlin_kernel.cu`) needs 96 KB
shared memory which exceeds the 64 KB per-block cap on sm_75 (Turing).
Most kernels assume native bf16, only available from sm_80. Emitting a
binary that segfaults on older hardware is worse than refusing to build.

---

## 3 · Environment variable contract

ARLE uses the **PyTorch / vLLM / SGLang / FlashInfer standard env var**
[`TORCH_CUDA_ARCH_LIST`](https://pytorch.org/docs/stable/cpp_extension.html).
`CMAKE_CUDA_ARCHITECTURES` is accepted as alias.

The legacy `INFER_CUDA_SM` / `CUDA_SM` were removed in this rollout. No
backwards-compat shim — see `feedback_no_half_states.md`.

**Resolution order** (`crates/cuda-kernels/build.rs::detect_sm_targets`):

1. `TORCH_CUDA_ARCH_LIST` — semicolon / comma / space separated, PyTorch format
2. `CMAKE_CUDA_ARCHITECTURES` — same parser
3. `nvidia-smi --query-gpu=compute_cap` — auto-detect the local GPU
4. Default to **T1 set** `{80, 86, 89, 90}` — never includes T2

**Accepted formats** (any combination per-token):

| Form              | Example          | Source convention |
|-------------------|------------------|-------------------|
| Major.minor       | `8.0`            | PyTorch native    |
| Packed integer    | `80`             | CMake / nvcc      |
| nvcc style        | `sm_80`          | NVCC `-arch=`     |
| nvcc PTX style    | `compute_80`     | NVCC `-arch=`     |
| `+PTX` suffix     | `9.0+PTX`        | PyTorch native    |

**Examples:**

```bash
# Default (no env): T1 fat binary
cargo build --release --features cuda

# Explicit T1 only (CI default)
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" cargo build --release --features cuda

# T1 + B100 (datacenter Blackwell)
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0" cargo build --release --features cuda

# RTX 5090-only build (Blackwell consumer)
TORCH_CUDA_ARCH_LIST="12.0" cargo build --release --features cuda

# CMake-style integer alias works too
CMAKE_CUDA_ARCHITECTURES="80;86;89;90" cargo build --release --features cuda

# Force PTX for forward compat to unknown hardware
TORCH_CUDA_ARCH_LIST="9.0+PTX" cargo build --release --features cuda
```

**Difference from PyTorch.** PyTorch is *best-effort* — if a kernel can't
compile for a target SM, it warns and skips. ARLE is **hard-fail**: any
target SM that can't emit cubin for any kernel panics the build, with a
suggested `TORCH_CUDA_ARCH_LIST` value to exclude that SM. This matches
the production-binary contract: every supported SM must work.

---

## 4 · AOT failure policy

When Triton AOT or TileLang AOT can't emit cubin for an opt-in SM:

1. `build.rs` panics immediately with the failing (SM, kernel) tuple.
2. Error message includes a suggested `TORCH_CUDA_ARCH_LIST=...` excluding
   that SM, plus a pointer to upstream version requirements
   (`requirements-build.txt::triton`, `pyproject.toml::tilelang`).
3. User options:
   - Bump the upstream toolchain (commit the bump, re-run).
   - Exclude the SM from `TORCH_CUDA_ARCH_LIST` and accept reduced coverage.
   - File an upstream issue (the canonical Blackwell scoreboard sit at
     [Triton issue #5950](https://github.com/triton-lang/triton/issues/5950)
     and [tile-ai/tilelang issues](https://github.com/tile-ai/tilelang/issues)).

There is no warn-skip path. A binary that silently lacks cubin for a
declared target SM is a footgun.

---

## 5 · Bench gate (commit D ship criteria)

Before declaring multi-SM "done" and removing the `pending-remote` stubs:

| GPU                  | SM | Test                                       | Bench                                         |
|----------------------|----|---------------------------------------------|-----------------------------------------------|
| A100 (40/80 GB)      | 80 | `cargo test --release --test e2e`           | `scripts/bench_guidellm.sh cuda-multi-sm-80`  |
| A10 or RTX 3090      | 86 | `cargo test --release --test e2e`           | `scripts/bench_guidellm.sh cuda-multi-sm-86`  |
| L4 or RTX 4090       | 89 | `cargo test --release --test e2e_qwen35`    | `scripts/bench_guidellm.sh cuda-multi-sm-89`  |
| H100                 | 90 | `cargo test --release --test e2e_qwen35`    | `scripts/bench_guidellm.sh cuda-multi-sm-90`  |

**Pass criteria:**
- e2e tests green on all four cards.
- `bench_guidellm.sh` TTFT and out_tok/s within **±5 %** of the most recent
  same-SM baseline in `docs/experience/wins/` (sm_89 baseline already exists
  via `2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md`; sm_90
  baseline from the TileLang Phase-0 H100 run; sm_80 / sm_86 baselines TBD,
  the first bench *is* the baseline).

**Fail recovery:**
- Per-SM regression > 5 % → record in `docs/experience/errors/` with root
  cause; do not ship the multi-SM binary on that SM until resolved.
- The other three SMs can ship independently if their wins/ entries land
  cleanly.

---

## 6 · Implementation phases

| Commit | Scope | Local verify | Remote verify |
|--------|-------|--------------|---------------|
| **A** | Tier policy + `TORCH_CUDA_ARCH_LIST` parsing + delete `INFER_CUDA_SM` + docs + bench stubs | Mac `cargo check --features cuda,no-cuda` | none |
| **B** | Triton AOT multi-cubin + dispatch wrapper for 12 kernels | none | Linux + nvcc + ≥1 T1 GPU |
| **C** | TileLang AOT multi-cubin + dispatch wrapper for 3 head families | none | Linux + nvcc + ≥1 T1 GPU |
| **D** | T1 four-card bench validation; replace `pending-remote` stubs | none | A100 + A10/3090 + L4 + H100 |

Commits B and C are independent (no order dependency); both require
Commit A's tier policy to be in place.

---

## 7 · Cross-references

- [`sm-coverage-verification.md`](sm-coverage-verification.md) — full
  step-by-step verification runbook for Phases B/C/D (build, parity, bench
  gate, fail-recovery, T2 opt-in, rollback).
- [`tilelang-integration-verification.md`](tilelang-integration-verification.md)
  — Phase 0 H100/L4 runbook; SM-specific bench thresholds. The §5
  ship/revert thresholds for TileLang prefill HD128 still apply per-SM
  inside this plan's bench gate.
- [`cuda-kernel-crate-extraction.md`](cuda-kernel-crate-extraction.md) —
  the kernel-crate boundary this policy enforces (`infer → cuda-kernels`,
  never the reverse).
- [`../environment.md`](../environment.md) — `TORCH_CUDA_ARCH_LIST` user
  reference.
- [`../support-matrix.md`](../support-matrix.md) §2 — GPU/SM support row.
- [`../../crates/cuda-kernels/AGENTS.md`](../../crates/cuda-kernels/AGENTS.md)
  §"build.rs rules" — agent-facing summary of the tier policy.
