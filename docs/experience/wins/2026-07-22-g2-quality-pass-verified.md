# G2 sm_120 CUTLASS grouped FP8 MoE — reviewed quality pass (pending-remote)

> Status: pending-remote — sm_120 needle + bench must run on a warm G2 session.

## Context

Post-ship review pass on the sm_120 CUTLASS grouped FP8 MoE GEMM (G2, shipped
`2026-07-22-bench-sm120-fp8-moe-cutlass-grouped.md`, c=1 prefill TTFT
84.6s→760ms, 111×, needle exact/DET). All changes are behavior-preserving on the
shipped single-GPU path; goal is to keep 111× while removing real hazards +
duplication.

## What landed (commits `112792b59`, `6720a1689`, `96c9f002e`, `55458ab06`)

- **#1 device-scratch data race (real hazard):** `static GroupedScratch[16]` +
  unguarded `cudaGetDevice` → per-device `unordered_map<int,unique_ptr<>>` +
  `g_scratch_mutex` + TLS fast-path (ports `gemv.cu` `g_per_device_state`).
  Mutex now guards every `ensure_arrays`/`ensure_workspace` grow. Kills the
  malloc/free race, the dev≥16 aliasing, and the hardcoded 16.
- **#3 load↔exec layout contract (silent-corruption risk):** added
  `MoeFp8ExpertGroup::sfb_n_contiguous`, set once at load (survives
  offload/reload snapshot); executor dispatches on it (`w13==down` `ensure!` +
  `debug_assert` == runtime SM) instead of re-deriving `sm120`.
- **#5** `DeviceContext::is_sm120()` replaces 4 re-derived `== 12` checks.
- **#6** one `run_grouped_gemm` closure for the w13 + down GEMMs.
- **#7** `m_indices` fill + neg1 buffer gated behind `!sm120` (dead on the
  CUTLASS path — it consumes offsets+counts).
- **#8** `free_arrays` → one `void**` loop; `max_m` → `bool any_rows`.
- **#9** build.rs: hoisted the twice-computed sm_120 grouped-FP8 guard.

## Deliberately not done

- **#2 per-call `cudaStreamSynchronize` hoist (highest perf target):** NOT
  landed. The correct form is a caller-driven once-per-layer host D2H of
  offsets+counts + an FFI ABI change; the CudaSlice owners are already mutably
  borrowed inside `deepgemm_routed_tail`, and a pointer-memoized in-kernel skip
  is *incorrect* (the scratch offset buffers are reused across layers with
  changing contents). Task authorized leaving it; the shipped path is untouched
  → no regression risk. Left for a dedicated, VM-verified change.
- **#4 dead C operand drop:** skipped — the only Tier-2 item with real compile
  risk (CUTLASS epilogue wants `const ElementC**`; passing `ElementD**` is not a
  valid 2nd-level-const conversion). Cosmetic; not worth gambling the build.

## Verification done

- Rust (`#3/#5/#6/#7/#9`): `cargo check` + `cargo clippy` `-p infer-cuda`
  `--features cuda,no-cuda` — clean.
- `.cu` (`#1/#8`): the authored scratch/mutex/TLS/`free_arrays` logic compiles
  under `nvcc 12.8 -gencode arch=compute_120a,code=sm_120a` (isolated TU, RC 0)
  on a real RTX PRO 6000 (cc 12.0).

## Pending-remote (blocks acceptance)

Full runtime gate NOT run — a bare Colab G4/sm_120 session lacks the repo, the
`Qwen/Qwen3.6-35B-A3B-FP8` weights, and the Rust toolchain; no warm
`moehang`/`g2dev`/`g2wire` session existed. On a warm G2 session:

1. `RUSTC_WRAPPER= CUDA_HOME=/usr/local/cuda cargo build --release --features cuda`
2. `scripts/needle_gate.py` on `Qwen/Qwen3.6-35B-A3B-FP8` (RAW=1
   TEMPLATE=qwen3_nonthink) — MUST stay exact/DET/miss=0.
3. `bench_throughput.py` c=1,8 vs `2026-07-22-bench-sm120-fp8-moe-cutlass-grouped.md`
   — prefill TTFT must not regress from 111×.

## Rule

Behavior-preserving hazard/cleanup passes on a shipped GPU path still gate on the
same-model needle + non-regression bench; when the verify env is unavailable,
land + stub pending-remote, never silent-pass.
