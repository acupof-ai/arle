# Operator evidence vertical slice — Qwen FP8 dense, 2026-07-10

> Status: Active — H20 component, engagement, and GuideLLM gates pending.

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
- Stats fields: policy, product, bundle, implementation hits, fallback count

## Environment

- Local: Apple Silicon, Rust 1.95, CUDA type-check via
  `CUDARC_CUDA_VERSION=12080` and `cuda,no-cuda`
- Remote control: 8x H20, CUDA 12.9; aligned source build passed in 1m53s
- H20 evidence run: pending on an isolated pod tree

## Results

| Gate | Result |
|---|---|
| seam/core/server tests | 200 run; 199 passed, 1 ignored |
| GuideLLM stats regression | 1 passed |
| invalid stats JSON | rejected and recorded as `ok:false` |
| multiprocess stats relay | policy/identity/hit/fallback preserved |
| generated selector | no exact cells; fallback unchanged |

## Problems

- The shared `/host/arle-build` tree had concurrent users. Final H20 evidence
  must use an isolated `POD_TREE`.
- Backend counters remain empty until the family-local fixed-counter path is
  wired without per-token heap allocation.

## Learnings

- Historical prose is not selector evidence. A cell qualifies only with clean
  source/product identity, numeric parity, E2E correctness, timing, and
  independent engagement proof.
- Unknown cells retain the existing correct fallback.

## Pending H20 gates

- numeric candidate/reference probe with predeclared mixed tolerance;
- same-binary E2E A/B and independent launch engagement;
- canonical GuideLLM SLO run and wall-clock delta;
- exact policy/product/bundle identity in `/v1/stats`.

## Artefacts

- Schema: `benchmarks/operators/schema.json`
- Policy: `benchmarks/operators/optimal.json`
- Reducer: `scripts/reduce_operator_evidence.py`
- Plan: `docs/plans/2026-07-10-operator-artifact-dev-release-system.md`
