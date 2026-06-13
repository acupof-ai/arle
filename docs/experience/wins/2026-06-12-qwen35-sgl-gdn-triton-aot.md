# Qwen3.6-35B GDN decode → SGLang Triton-AOT lane (U1+U2) — opt-in, validated WASH

**Date:** 2026-06-12 (impl); **2026-06-13** validation.
**Backend:** CUDA, Qwen3.6-35B-A3B, H20, TP=1.
**Verdict (2026-06-13):** lane RUNS (probe fired) + correct (= off GDN math);
perf is a **WASH** (c=1/2/4 within ±~2.4% of off). The validate2 "+22% @ c=8"
was a false signal from c=8 admission bistability, killed by a c=8 ×12 high-rep
re-run (off itself is ~75% fast / ~25% slow). Stays opt-in, no default flip.
See the [validation entry](2026-06-13-qwen36-sgl-kernel-align-validate-bistability.md).

## Context

First two tranches of the Qwen-lane SGLang kernel alignment
([plan](../../plans/2026-06-12-qwen-lane-kernel-alignment-sglang.md), ckl
directive "kernel 全对齐 sglang"). The 35B B=1 decode sits at 93.5 tok/s ≈ 14%
of the weight-bytes roofline — the structural small-kernel GEMV tax. Adopting
SGLang's GDN decode kernels (30 of 40 layers are linear-attn) is T1 of that
plan; this commit lands the adoption substrate (U1) + the GDN decode swap (U2).

## What landed

- **U1 — Triton AOT lane** (`crates/cuda-kernels/`): `tools/triton/`
  (`gen_triton_aot.py` programmatic `triton.compile` → cubin + launch metadata,
  3 vendored `@triton.jit` kernels under `kernels/`, README), `build.rs` AOT
  compile step (mirrors the TileLang AOT lane; `INFER_TRITON_PYTHON` honored,
  absence non-fatal → links `CUDA_ERROR_NOT_SUPPORTED` stubs), `src/ffi/triton.rs`
  FFI decls. Grid (gX,gY,gZ) first / stream last; each kernel exports a
  `*_load_cuda` idempotent module-load symbol called once at executor init so no
  module load happens inside a CUDA-graph capture.
- **U2 — GDN decode swap** (`crates/infer-cuda/src/qwen35.rs`): the three GDN
  decode kernels (conv1d_update → fused_recurrent_decode → rms_norm_gated) routed
  through the AOT trio behind opt-in `ARLE_QWEN35_SGL_GDN`. Wired in BOTH the
  single-row decode path (`seq_len==1`) AND the batched decode path (the c=1
  default route) — shape-guarded to the baked Qwen3.6 shard (H=16/HV=32/128/128),
  decode-only (prefill chunks untouched). Default OFF → byte-for-byte the current
  hand-kernel path; the hand kernels stay the baseline arm (no half-states).

## Verification (Mac, no nvcc)

- `cargo check -p infer-api --features cuda,no-cuda --lib` green;
  `cargo check -p agent-infer --features cpu,no-cuda,cli` green;
  `cargo clippy` infer-cuda 0 warnings. build.rs no-cuda path skips kernel
  compile (links stubs) — confirmed.

## Pending (one-shot pod pass, #88)

1. Build on 8×H20 pod with `INFER_TRITON_PYTHON` set (`CARGO_NET_OFFLINE=1`);
   confirm the AOT cubins compile (not stubs) via the runtime loud-fail probe.
2. Needle gate ×3 DET vs the locked 2026-06-12 envelope (len 2000/8000 exact).
3. Same-binary same-shell A/B `ARLE_QWEN35_SGL_GDN` OFF vs ON, c=1/2/4/8, vs the
   locked baseline 93.5/152.3/207.5/255.6 tok/s. Δ% per c. License-or-kill on
   wall-clock per shape; a losing A/B keeps the lane opt-in with the verdict
   recorded. GDN is 30/40 layers so this tranche carries most of the per-layer
   decode time — the load-bearing A/B of the alignment.

## Rule

A multi-tranche kernel-alignment campaign commits each verified-compiling,
opt-in-OFF tranche immediately (zero risk to the default path) even when the
on-device A/B is batched into a later one-shot pass — the `pending-remote` stub
records exactly which gate is outstanding so the validation pass can't silently
skip a tranche.
