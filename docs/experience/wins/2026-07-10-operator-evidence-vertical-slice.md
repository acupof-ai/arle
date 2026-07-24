# Operator evidence vertical slice — Qwen FP8 dense, 2026-07-10

> Status: Active — H20 probe evidence and native benchmark engagement gate pending.
> P1 update (2026-07-11): backend dispatch counters wired in `quant_linear.rs` — all 4 FP8 dense
> paths (DeepGEMM, dequant GEMM, GEMV in gemm_batch, GEMV in gemv) increment global atomics.
> `/v1/stats` now returns real `implementation_hits` + `fallback_count`.
> Truth-boundary correction (2026-07-11): the old runner's automatic E2E pass,
> probe-binary E2E artifact, and git-commit bundle identity are invalid. No run
> from that path qualifies. Qualification now requires a verified full CUDA
> manifest and an identity-bound actual-model E2E artifact.

## SLO-shape probed? N

No dispatch default changes in this tranche. The generated policy preserves the
existing M=1 GEMV / M>=2 pack+DeepGEMM fallback until qualified H20 evidence is
committed.

## Roofline check

Deferred to the H20 probe. This tranche establishes the evidence path; it makes
no operator-performance claim.

## Goal

Make Qwen FP8 dense dispatch decisions reproducible from machine-checkable
numeric, timing, identity, E2E, and engagement evidence.

## Hypothesis

The current fallback remains unchanged, while exact measured cells can later
override it without affecting unknown hardware or shapes.

## Parameters

- Operator: `qwen.fp8_dense_projection`
- Current fallback: M=1 GEMV; M>=2 pack+DeepGEMM
- Evidence status: no qualified exact cells
- Numeric gate: per element `abs <= 1.0 + 0.02 * abs(reference)`
- Stats fields: policy, product, bundle, implementation hits, fallback count

## Environment

- Local: Apple Silicon, Rust 1.95, CUDA type-check via
  `CUDARC_CUDA_VERSION=12080` and `cuda,no-cuda`
- Remote control: 8x H20, CUDA 12.9; aligned source build passed in 1m53s
- H20 evidence run: pending on an isolated pod tree

## Results

| Gate | Result |
|---|---|
| seam/core/server tests | 201 run; 200 passed, 1 ignored |
| Native benchmark stats regression | 1 passed |
| invalid stats JSON | rejected and recorded as `ok:false` |
| multiprocess stats relay | policy/identity/hit/fallback preserved |
| generated selector | no exact cells; fallback unchanged |

## Problems

- The shared `/host/arle-build` tree had concurrent users. Final H20 evidence
  must use an isolated `POD_TREE`.
- **FIXED 2026-07-11:** Backend dispatch counters wired in `quant_linear.rs`.
  `DEEPGEMM_HITS`, `GEMV_HITS`, `DEQUANT_GEMM_HITS`, `FALLBACK_COUNT` global
  atomics incremented at all 4 call entry points (`gemm_batch` × 3 paths,
  `gemv` × FP8 path). Reachable via `/v1/stats → operator_dispatch`.
- **FIXED 2026-07-11:** The probe runner no longer manufactures qualification.
  Runs without a verified kernel manifest are stored as `unverified` with no
  bundle ID and cannot produce exact selector cells.

## Learnings

- Historical prose is not selector evidence. A cell qualifies only with clean
  source/product identity, numeric parity, E2E correctness, timing, and
  independent engagement proof.
- Unknown cells retain the existing correct fallback.

## Pending H20 gates

- numeric candidate/reference probe with the committed mixed tolerance;
- same-binary, same-bundle actual-model E2E artifact and independent launch engagement;
- canonical native SLO run and wall-clock delta;
- exact policy/product/bundle identity in `/v1/stats`.

## Artefacts

- Schema: `benchmarks/operators/schema.json`
- Registry: `operators/registry.toml`
- Policy: `benchmarks/operators/optimal.json`
- Reducer: `scripts/reduce_operator_evidence.py`
- Probe: `crates/infer-cuda/examples/fp8_smallm_gemm_probe.rs`
- Probe runner: `scripts/run_fp8_probe.sh`
- Generated selector: `crates/infer-cuda/src/ops/generated/qwen_fp8_dense_projection.rs`
