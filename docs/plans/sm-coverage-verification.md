# Multi-SM Coverage — Verification Runbook

**Status:** Companion to [`sm-coverage.md`](sm-coverage.md). Owner: ckl.
Mirrors the structure of [`tilelang-integration-verification.md`](tilelang-integration-verification.md).
Phase A (env var + tier policy) and Phase B+C (multi-SM AOT + dispatch)
are landed; this doc is the runbook to retire the four `pending-remote`
bench stubs in `docs/experience/wins/` and ship Phase D.

---

## 0 · Hardware modes

| Mode | SM    | GPU                           | Use                                                        | Decision authority |
|------|-------|-------------------------------|------------------------------------------------------------|--------------------|
| A100 | 8.0   | NVIDIA A100 40 / 80 GB        | T1 floor; first multi-SM bench on this card sets baseline. | Yes — Phase D ship gate row 1. |
| A10  | 8.6   | NVIDIA A10 24 GB *or* RTX 3090 24 GB | sm_86 coverage; cheapest T1 row to schedule.        | Yes — Phase D ship gate row 2. |
| L4   | 8.9   | NVIDIA L4 24 GB *or* RTX 4090 24 GB  | sm_89 coverage; existing baseline `2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md`. | Yes — Phase D ship gate row 3. |
| H100 | 9.0   | NVIDIA H100 80 GB SXM/PCIe    | sm_90 coverage; TileLang TMA / WGMMA leverage.             | Yes — Phase D ship gate row 4. **Also drives** the existing TileLang Phase 0 §5 ship/revert thresholds (those still apply). |
| B100 | 10.0  | NVIDIA B100 / B200            | T2 opt-in; not in ship gate. Optional.                     | No — pending-remote, T2 column. |
| 5090 | 12.0  | NVIDIA RTX 5090               | T2 opt-in; not in ship gate. Optional.                     | No — pending-remote, T2 column. |

**Per-SM cubin** is the contract: every binary the user runs on a given
T1 SM dispatches into the cubin compiled for that exact SM, not via PTX-JIT.
Cross-SM cubin loading fails with `cuModuleLoadData →
CUDA_ERROR_INVALID_SOURCE`; the dispatch wrapper enforces correct selection
via `cuDeviceGetAttribute(COMPUTE_CAPABILITY_*)`.

---

## 1 · Pre-flight (don't skip)

```bash
# 1.1 GPU is what we expect; record `compute_cap`. T1 expects {8.0, 8.6, 8.9, 9.0}.
nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader

# 1.2 CUDA toolchain ≥ 12.8 (required for sm_120 / opt-in T2; safe even for T1 only).
nvcc --version | grep release
echo "$CUDA_HOME"   # expect /usr/local/cuda or equivalent

# 1.3 Triton + TileLang Python deps available.
python3 -c "import triton; print('triton', triton.__version__)"
python3 -c "import tilelang; print('tilelang', tilelang.__version__)"

# 1.4 Workspace clean.
git status --short    # only your in-flight diffs
git log --oneline -3  # confirm Phase A + Phase B+C commits are in
```

If any of these fail, **stop**. Don't paper over — fix the toolchain or
the env first.

---

## 2 · Build the multi-SM fat binary

The default (no `TORCH_CUDA_ARCH_LIST`) compiles for T1 = `{8.0, 8.6, 8.9, 9.0}`
on this host:

```bash
# 2.1 T1 fat binary, no TileLang (faster build).
unset TORCH_CUDA_ARCH_LIST CMAKE_CUDA_ARCHITECTURES
cargo build --release -p infer --features cuda

# 2.2 T1 fat binary with TileLang (canonical configuration).
cargo build --release -p infer --features cuda,tilelang-attn
```

Successful build emits:

```
warning: cuda-kernels@0.1.x: Compiling CUDA kernels for targets: sm_80,sm_86,sm_89,sm_90
warning: cuda-kernels@0.1.x: Triton AOT: built per-SM cubins for 4 target(s); SM dispatch via pthread_once + cuDeviceGetAttribute.
warning: cuda-kernels@0.1.x: TileLang AOT: built per-SM cubins for 4 target(s) across HD128/HD256 prefill + HD256 decode; ...
```

### 2.x Symbol verification

The dispatch wrappers should expose **single** public-symbol entries
even though there are now 4× cubins per kernel:

```bash
# Triton: each public _aot_cuda symbol has 4 per-SM siblings (one extern per SM).
nm target/release/infer | grep -E '\b(silu_mul_triton_aot_cuda|add_cuda)\b' | wc -l   # expect 2
nm target/release/infer | grep silu_mul_triton_sm | sort                              # expect 4: sm80/86/89/90
nm target/release/infer | grep add_triton_sm | sort                                   # expect 4
nm target/release/infer | grep flash_attention_prefill_hd256_sm | sort                # expect 4

# TileLang: 10 head-config families × 4 SMs each.
nm target/release/infer | grep tilelang_batch_prefill_paged_hd128_q16_kv8_sm | sort   # expect 4
nm target/release/infer | grep -c tilelang_batch_                                     # expect 50: 10 dispatch + 40 per-SM
```

If any count is wrong, the multi-SM expansion didn't fire — re-run
`cargo build --release -v` and check for `cargo:warning=Compiling CUDA
kernels for targets: ...` listing all 4 SMs.

### 2.y Single-SM rebuild for A/B comparison

When a multi-SM bench number looks off, rebuild with one SM only on the
same host and compare. Same binary same env per
`feedback_matched_ab_for_small_bench_effects.md`:

```bash
# Match the host SM exactly. L4 example:
TORCH_CUDA_ARCH_LIST="8.9" cargo build --release -p infer --features cuda,tilelang-attn
```

---

## 3 · Numerical parity (per-SM, before bench)

Run the e2e tests on each T1 host. They drive the same dispatch path
the server uses, so passing here is the lower bound for "the cubin is
correct on this SM."

All four tests are gated by `#![cfg(feature = "cuda")]`; `infer/Cargo.toml`
declares `default = []`, so without `-p infer --features cuda` the test
files compile to no-ops and the gate would pass silently. The GGUF
smoke is also `#[ignore]`, so `-- --ignored` is required to actually
run it.

**Feature set must match §4.1.** Cargo features are per-invocation —
running parity with `--features cuda` on a host that boots the server
with `cuda,tilelang-attn` (sm_89, sm_90) compiles a *different* test
binary that doesn't exercise the TileLang wrappers / cubins the bench
will hit. Set `FEATURES` per host so parity covers the same code path
as §4.1.

```bash
# Pick the same feature set as the §4.1 server build for this host:
#   sm_80 / sm_86 → FEATURES=cuda
#   sm_89 / sm_90 → FEATURES=cuda,tilelang-attn   (TileLang validation hosts)
FEATURES="cuda,tilelang-attn"

# 3.1 Qwen3 e2e (silu_mul / add / embedding / FlashInfer prefill).
cargo test --release -p infer --features "$FEATURES" --test e2e

# 3.2 Qwen3.5 e2e (HD256 prefill + GDR chunkwise + TileLang decode HD256).
cargo test --release -p infer --features "$FEATURES" --test e2e_qwen35

# 3.3 GGUF + Qwen3.5 smoke. The test reads $INFER_Q35_PATH; without it
#     it defaults to models/Qwen3.5-4B-GGUF-Q6_K (not provisioned on the
#     L4/4090 row). Point it at the §4.1 model for this host.
INFER_Q35_PATH=models/Qwen3.5-0.8B-GGUF \
  cargo test --release -p infer --features "$FEATURES" \
    --test smoke_qwen35_gguf -- --ignored

# 3.4 Q4_K kernel correctness (CUDA-specific quantized embedding).
cargo test --release -p infer --features "$FEATURES" --test q4k_kernel_correctness
```

JSON baselines under `infer/test_data/`:
- `Qwen3-4B.json`
- `Qwen3-8B.json`
- `Qwen3.5-4B.json`

A failure here means **stop**. The cubin for this SM is wrong; do not
proceed to bench. Re-check that `nvidia-smi --query-gpu=compute_cap`
matches the cubin actually loaded (`tracing::info!("CUDA device: sm_X.Y")`
line in the server start log).

---

## 4 · Bench — the four-card gate

For each T1 SM, run the canonical guidellm sweep. **Order is irrelevant**;
each row retires its own pending-remote stub independently.

### 4.1 Boot the server

The model and `--features` set differ per SM by design — they match what
the corresponding stub at
`docs/experience/wins/2026-04-28-bench-guidellm-multi-sm-<sm>.md` declares.
**Always launch the model named in the stub for that SM**, otherwise the
bench results can't be directly compared against the row's baseline.

```bash
# A100 (sm_80, 40/80 GB): Qwen3-8B full bf16.
# Stub declares `cargo build --release --features cuda` (no tilelang-attn:
# sm_80 has no Phase-0 TileLang validation; first run is the baseline).
./target/release/infer --model-path models/Qwen3-8B --port 8000 \
  --num-slots 16 --max-seq-len 8192 &

# A10 / RTX 3090 (sm_86, 24 GB): Qwen3-8B fits at this VRAM.
# Stub declares `cargo build --release --features cuda` (no tilelang-attn,
# same reason as sm_80).
./target/release/infer --model-path models/Qwen3-8B --port 8000 \
  --num-slots 8 --max-seq-len 4096 &

# L4 / RTX 4090 (sm_89, 24 GB): Qwen3.5-0.8B Q4_K_M to match the existing
# 2026-04-27 L4 baseline. Stub declares `cargo build --release --features
# cuda,tilelang-attn` (sm_89 is the canonical TileLang validation host).
# Mirror the baseline launch in
# docs/experience/wins/2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md
# exactly: GGUF dir holds tokenizer + config so `--model-path` alone is
# enough; `infer` has no `--gguf-quant` flag (quant is detected from the
# .gguf header). `--max-seq-len 4608` is required for the canonical
# guidellm `prompt=4096+output=256` workload to admit on 24 GB.
./target/release/infer --model-path models/Qwen3.5-0.8B-GGUF \
  --port 8000 --num-slots 8 --max-seq-len 4608 --mem-fraction-static 0.85 &

# H100 (sm_90, 80 GB): Qwen3.5-4B (matches Phase-0 H100 reference workload).
# Stub declares `cargo build --release --features cuda,tilelang-attn`
# (sm_90 is where TileLang TMA/WGMMA leverage fires; tilelang-attn must be on).
./target/release/infer --model-path models/Qwen3.5-4B --port 8000 \
  --num-slots 16 --max-seq-len 8192 &
```

**Why the per-SM `tilelang-attn` divergence.** sm_89 and sm_90 carry the
TileLang Phase-0 validation history (see
[`tilelang-integration-verification.md`](tilelang-integration-verification.md)
§0). sm_80 and sm_86 had no TileLang Phase-0 run, so the stubs build
without `tilelang-attn` for the first run; once those rows are retired
without regression, a follow-up bench can re-test with TileLang on.

### 4.2 Run guidellm sweep

`--model` MUST match the model name the server registered for itself
(visible at `GET /v1/models`); the OpenAI-compat preflight rejects any
mismatch as `model_not_found`. `--processor` MUST point at a directory
with the safetensors `tokenizer.json` + `config.json` for that model
(GGUF rows still need the safetensors tokenizer for guidellm's
client-side tokenisation).

```bash
# A100 (sm_80) and A10/3090 (sm_86) — Qwen3-8B safetensors.
SM=80    # or 86
scripts/bench_guidellm.sh cuda-multi-sm-${SM} \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --processor models/Qwen3-8B

# L4 / RTX 4090 (sm_89) — GGUF row matching the 2026-04-27 baseline.
SM=89
scripts/bench_guidellm.sh cuda-multi-sm-${SM} \
  --target http://localhost:8000 \
  --model Qwen3.5-0.8B-GGUF \
  --processor models/Qwen3.5-0.8B

# H100 (sm_90) — Qwen3.5-4B, matches the Phase-0 H100 reference workload.
SM=90
scripts/bench_guidellm.sh cuda-multi-sm-${SM} \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B
```

The wrapper writes `bench-output/<date>-cuda-multi-sm-${SM}/` with
`benchmarks.{json,csv,html}`, plus service trace files.

### 4.3 Pass criteria (the gate)

The gate compares **multi-SM build vs single-SM build on the same
host, same commit, same nvcc, same TileLang version, back-to-back**
(matched A/B per `feedback_matched_ab_for_small_bench_effects.md`).
We are validating that the new pthread_once + cuDeviceGetAttribute
dispatch overhead is non-measurable, NOT that absolute throughput
matches an older wins entry: scheduler/KV defaults have shifted since
the 2026-04-27 L4 entry (auto FP8 KV `0af5769`, HBM-tier
chunked_prefill `62f44a0`), so a cross-commit absolute compare would
fold in unrelated changes.

| Metric                              | sm_80 | sm_86 | sm_89                | sm_90                |
|-------------------------------------|-------|-------|----------------------|----------------------|
| Baseline                            | first run = baseline | first run = baseline | single-SM rebuild on this host (§4.3.x) | single-SM rebuild on this host (§4.3.x) |
| TTFT p50 @ synchronous, max delta   | n/a   | n/a   | ±2 % vs single-SM    | ±2 % vs single-SM    |
| out tok/s @ saturation, max delta   | n/a   | n/a   | ±2 % vs single-SM    | ±2 % vs single-SM    |

For sm_80 and sm_86 the first run *is* the baseline; record the
absolute numbers in the wins entry. For sm_89 and sm_90, delta against
the §4.3.x same-host single-SM rebuild.

**Cross-commit sanity floor (informational, not gating).** The
2026-04-27 L4 entry (`2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md`,
c=1: TTFT p50 247.4 ms / 183.3 out tok/s · c=2 saturation: 222.2 out
tok/s) is a useful order-of-magnitude check for sm_89 — if the
multi-SM run on HEAD is dramatically below that floor (e.g. 50 % out
tok/s drop), something else regressed and the multi-SM A/B is
ill-defined. Do not gate on this comparison; defaults have moved.

### 4.3.x Single-SM rebuild for the matched A/B

```bash
# On the same host as 4.2, rebuild single-SM with the same feature
# set as §4.1 for this host (cuda or cuda,tilelang-attn).
TORCH_CUDA_ARCH_LIST="${SM_DOTTED}" cargo build --release -p infer --features "$FEATURES"
# Kill + restart the server with the rebuilt binary, then:
scripts/bench_guidellm.sh cuda-single-sm-${SM} \
  --target http://localhost:8000 \
  --model "$MODEL" --processor "$PROCESSOR"
```

Use the same `--model` / `--processor` arguments as §4.2 for this SM.
The multi-SM and single-SM TTFT/throughput should match within ±2 %.
If multi-SM is consistently slower by >2 % across 3+ runs, file
`docs/experience/errors/...` with the diff and fall back to single-SM
build for that platform.

---

## 5 · Retire the pending-remote stubs

Each T1 SM has a stub at
`docs/experience/wins/2026-04-28-bench-guidellm-multi-sm-<sm>.md`.
After bench passes:

```bash
# 5.1 Copy bench output into the artefact paths the stub claims.
cp bench-output/<date>-cuda-multi-sm-<SM>/benchmarks.json \
   bench-output/<date>-cuda-multi-sm-<SM>/benchmarks.csv \
   bench-output/<date>-cuda-multi-sm-<SM>/benchmarks.html \
   /tmp/  # or wherever you stage uploads

# 5.2 Edit the stub:
#   - Remove the `pending-remote` banner.
#   - Fill `Hardware`, `Commit` (sha of HEAD on the bench host),
#     `Results — sweep headline table`, `Results — service-side`,
#     `Δ vs baseline`, `Notes` (any deviation from the run-book).
#   - Keep `Goal`, `Hypothesis`, `Command`, `Environment`, `Canonical
#     params` exactly as written; if the bench actually used different
#     params, that's a bench-spec violation, not a stub fix.

# 5.3 Commit per-SM:
git add docs/experience/wins/2026-04-28-bench-guidellm-multi-sm-<SM>.md
git commit -m "docs(wins): cuda-multi-sm-<SM> bench results retire pending-remote stub"
```

**Ship rule**: when all four T1 stubs are retired without `> 5 %`
regressions, multi-SM is shipped. Update `docs/support-matrix.md` §2
GPU/SM row from "Beta" to "Supported" if any T1 SM was Beta-only before.

---

## 6 · Fail-recovery flowchart

```
Bench failed on this SM?
├── Build failed: AOT panic?
│   ├── Triton failure → bump triton in requirements-build.txt, OR exclude this SM via
│   │   TORCH_CUDA_ARCH_LIST="<remaining T1>" and document in support-matrix.md.
│   └── TileLang failure → bump tilelang in pyproject.toml (now `tilelang>=0.1`), OR
│       exclude via TORCH_CUDA_ARCH_LIST. Errors here often originate from upstream
│       (see `docs/experience/errors/2026-04-26-tilelang-aot-tilelang-0p1p9-blocker.md`
│       for prior art).
│
├── e2e test failure (numerical parity)?
│   ├── First, log the actual sm_X.Y the server saw on startup
│   │   (`tracing::info!("CUDA device: sm_X.Y", ...)` in DeviceContext::new()).
│   ├── If logged SM matches host but cubin loaded was for another SM →
│   │   the dispatch wrapper switch is broken; investigate
│   │   `crates/cuda-kernels/build.rs::format_dispatch_wrapper`
│   │   (a missing case arm = unreachable).
│   └── If logged SM is correct but parity fails → kernel correctness regression
│       on this SM. File `docs/experience/errors/<date>-multi-sm-<SM>-parity.md`
│       and revert that SM's per-SM cubin via `TORCH_CUDA_ARCH_LIST` until fix.
│
└── Bench regression > 5 %?
    ├── Repeat 3+ times to rule out thermal noise (ref `feedback_matched_ab_for_small_bench_effects.md`).
    ├── Run single-SM A/B per §4.3 on same host. If both regress vs single-SM build,
    │   the multi-SM dispatch is the root cause. Profile with `nvprof --print-gpu-trace`
    │   and check the first-call pthread_once latency.
    └── If only multi-SM regresses, file `docs/experience/errors/<date>-multi-sm-dispatch-overhead-<SM>.md`
        and either accept (if ≤2 %) or revert to single-SM build for that platform.
```

---

## 7 · T2 opt-in verification (B100 / 5090)

Outside the ship gate. Only run when you actually have access to the hardware.

```bash
# 7.1 B100 / B200 — datacenter Blackwell.
TORCH_CUDA_ARCH_LIST="9.0;10.0" cargo build --release -p infer --features cuda,tilelang-attn

# 7.2 RTX 5090 — consumer Blackwell.
TORCH_CUDA_ARCH_LIST="8.9;12.0" cargo build --release -p infer --features cuda
# Note: TileLang on sm_120 has zero upstream evidence (per docs/plans/sm-coverage.md §3);
# omit `tilelang-attn` for the first 5090 attempt and let the Triton path handle attention.
```

If any AOT step panics, follow the fail-recovery flowchart §6 — the
panic message includes the exact `TORCH_CUDA_ARCH_LIST` value to retry
with that SM excluded.

T2 results land in *separate* wins entries (e.g.
`2026-MM-DD-bench-guidellm-multi-sm-100-b200.md`), not in the four T1
ship-gate stubs.

---

## 8 · Rollback

If multi-SM ships but a regression surfaces post-launch:

```bash
# 8.1 Per-host single-SM build (no code change).
TORCH_CUDA_ARCH_LIST="<host SM>" cargo build --release -p infer --features cuda

# 8.2 Or revert the multi-SM commits and rebuild.
git revert <Phase-B+C-sha> <Phase-A-sha>      # double revert in commit order
cargo build --release -p infer --features cuda
```

The Phase A revert is **only** needed if the env var migration itself
broke a downstream tool (release.yml, setup.sh). The Phase B+C revert
alone restores single-SM AOT behavior.

---

## 9 · Cross-references

- [`sm-coverage.md`](sm-coverage.md) — tier policy + env var contract.
- [`tilelang-integration-verification.md`](tilelang-integration-verification.md)
  §5 — the H100-driven Phase 0 thresholds that **continue to apply**
  on top of this gate.
- [`../experience/wins/2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md`](../experience/wins/2026-04-27-bench-guidellm-cuda-l4-qwen35-0p8b-packed-gguf.md)
  — sm_89 baseline.
- [`../experience/wins/2026-04-28-bench-guidellm-multi-sm-{80,86,89,90}.md`](../experience/wins/)
  — the four pending-remote stubs this runbook retires.
- [`../environment.md`](../environment.md) §`TORCH_CUDA_ARCH_LIST` — env var reference.
