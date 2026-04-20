# Project Constitution and Refactor Plan

**Status:** Active  
**Started:** 2026-04-20  
**Owner:** ckl  
**Purpose:** turn `agent-infer` from a technically strong repository into a top-tier project with a clear identity, one authoritative documentation system, one coherent runtime boundary, and a smoother toolchain.

---

## 0. Why this plan exists

The repository already has unusual technical strengths:

- strong CUDA serving path
- meaningful Metal work, not just a toy port
- serious KV-cache and scheduler work
- from-scratch Rust autograd / local RL experimentation

But the project shape is currently weaker than the underlying engineering:

- multiple docs disagree about current truth
- service boundaries are broader than they need to be
- model structure is duplicated between inference and training stacks
- benchmark and release workflows are not fully aligned with the public story

The goal of this plan is to fix that shape without flattening the ambition.

---

## 1. Project identity

`agent-infer` is:

> **A Rust-native agent inference runtime with integrated local self-evolution.**

This wording is deliberate.

- The **runtime** stays primary.
- Training / GRPO / self-evolution are strategic multipliers, not a second unrelated product.
- The project is **not** to be presented as a general training framework.
- The project is **not** to drift into two equal main lines with two equal architectures.

### Locked identity rule

When there is tension between:

1. making the inference runtime cleaner, more credible, and more production-grade
2. adding more standalone training surface area

the runtime wins unless a concrete user-facing loop becomes stronger because of the training work.

---

## 2. Canonical truth surfaces

Top-tier projects do not let every document describe reality independently.

From this plan forward, the authoritative sources are:

| Concern | Canonical source | Non-canonical docs must do |
| --- | --- | --- |
| Support status of backends / APIs / features | `docs/support-matrix.md` | mirror only |
| Meaning of stable / beta / experimental / internal | `docs/stability-policy.md` | link, not reinterpret |
| Workspace topology and crate/module map | `docs/codebase-map.md` | mirror only |
| Architectural ownership and boundaries | `docs/architecture.md` | mirror only |
| Benchmark process and evidence rules | `docs/bench-and-trace-spec.md` | link, not restate |
| Canonical end-to-end benchmark tool + params | `scripts/bench_guidellm.sh` + `docs/plans/guidellm-integration.md` | use exactly |
| Contributor operating contract | `AGENTS.md` | point to the sources above |

### Derived-surface rule

`README.md`, `ROADMAP.md`, `docs/index.md`, crate-local READMEs, and release notes are **derived surfaces**.

They may summarize.  
They may not define conflicting reality.

If a derived surface conflicts with a canonical source, the derived surface is wrong and must be updated immediately.

---

## 3. Architecture rules

### 3.1 One service boundary

Inside `infer`, the request lifecycle must collapse toward one internal service boundary.

Target shape:

- HTTP / CLI / agent code do wire-format adaptation only
- runtime request execution goes through one internal contract
- optional capabilities must not leak concrete storage/runtime types unless that is the intended public contract

This means:

- avoid parallel abstractions that both mean "submit a generation request"
- avoid engine traits that also act as storage/session internals dumping grounds

### 3.2 One model schema

Model structure must have one authority.

Training-specific execution details may differ, but the following should not be authored twice long-term:

- config field names
- hidden/head/KV shape constraints
- tie-weight rules
- layer-count/layout invariants

Short-term, duplication may exist during migration.  
Long-term, the project should converge on a shared model-spec/config layer consumed by both `infer` and `train`.

### 3.3 Runtime first

If a training-side design increases divergence from the runtime's model or execution semantics, it must justify that divergence explicitly.

The default assumption is:

- `infer` owns serving/runtime truth
- `train` integrates with that truth

not the other way around.

---

## 4. Toolchain rules

### 4.1 Benchmarks

The project keeps one canonical end-to-end benchmark path.

Rules:

- `guidellm` is the canonical e2e serving benchmark
- raw outputs live in `bench-output/`
- committed narrative stays in `docs/experience/wins/`
- committed small baselines live in `benchmarks/`
- active docs must not keep pointing to deprecated benchmark entrypoints

### 4.2 Release credibility

Release automation must build and package the binaries that the public docs actually present as the supported operator path.

If the README says a binary/script is canonical, release and CI must not ignore it by accident.

### 4.3 Supply-chain maturity

The medium-term governance target should align with:

- Diataxis for documentation structure
- MLPerf-style explicit benchmark process discipline
- OpenSSF Scorecard for open-source hygiene
- SLSA provenance for builds
- CycloneDX SBOM / ML-BOM for release transparency

These are not all day-one tasks, but they are the standards bar for "top-tier project" claims.

---

## 5. Immediate refactor tranches

### Tranche T0 — start now

Goal: remove contradictions and fix verified runtime-contract bugs.

- fix paged-prefill single-token correctness in CUDA Qwen3 path
- fix CUDA graph warmup to use the real paged-pool geometry
- align AGENTS workspace shape with the actual workspace
- align DFlash status across README/support/stability docs
- align canonical `guidellm` parameters across script/plan/template
- remove active references to deprecated throughput-sweep tooling where the canonical path is already established
- bench evidence for the runtime slice: [`docs/experience/wins/2026-04-20-bench-guidellm-cuda-paged-prefill-contract-fix.md`](../experience/wins/2026-04-20-bench-guidellm-cuda-paged-prefill-contract-fix.md)

### Tranche T1 — boundary tightening

Goal: reduce abstraction drift.

- narrow the internal request-execution boundary inside `infer`
- audit `InferenceEngine` optional methods and remove concrete-type leakage where possible
- define the intended contract between `infer` and `train`

### Tranche T2 — model-schema unification

Goal: reduce long-term drift between training and inference model definitions.

- introduce a shared model-spec/config layer
- migrate Qwen3 / Qwen3.5 shape/config invariants to that layer
- make divergence explicit where truly necessary

### Tranche T3 — docs and toolchain maturity

Goal: make the repo feel complete to a serious external user.

- make `docs/index.md` current and mechanically maintainable
- trim or clearly scope crate-local READMEs
- align CI/release with the documented operator path
- add supply-chain/security/release metadata improvements

---

## 6. Acceptance criteria

This plan is working only if all of the following become true:

1. A new contributor can identify the current feature status without reading multiple conflicting docs.
2. A maintainer can answer "what is authoritative for this topic?" in one sentence.
3. A new model/backend/feature cannot silently compile while violating a core runtime contract.
4. The repository story reads as one project with one spine, not multiple adjacent projects.
5. CI, release, benchmark, and docs present the same operator reality.

---

## 7. Non-goals

This plan does **not** mean:

- shrinking the ambition of local RL / self-evolution work
- removing the training crates
- forcing a monolith back into one crate
- stopping performance work

It means the project must earn its ambition structurally, not only technically.

---

## 8. Change management rule

Any future architectural proposal that adds:

- another truth surface
- another request boundary
- another model definition authority
- another benchmark authority

must first explain why the existing authority is insufficient.

Default answer: do not add a second one.
