# Stability Policy

This document defines the stability levels used by `agent-infer` and how they
affect compatibility, review, and release decisions.

The purpose is simple: users and contributors should know what they can depend
on, and maintainers should know what must be treated carefully.

---

## 1. Stability Levels

`agent-infer` uses four levels.

### Stable

For documented, intended-for-use surfaces.

Rules:

- avoid breaking changes
- document user-visible behavior changes
- provide migration guidance when users must act

### Beta

For documented and usable surfaces that are still evolving.

Rules:

- keep behavior reasonably consistent
- document meaningful user-visible changes
- prefer transition over abrupt removal

### Experimental

For early-stage or fast-moving surfaces.

Rules:

- may change faster
- must still be explicit about limitations
- should not be presented as long-term stable contracts

### Internal

For implementation details.

Rules:

- may be refactored freely
- must not be treated as public extension points unless promoted explicitly

---

## 2. Current Classification

This reflects repository state as of 2026-04-10.

### Stable

- documented HTTP APIs
  - `/v1/completions`
  - `/v1/chat/completions`
- `GET /metrics`
- `GET /v1/stats`
- documented build, test, lint, and format workflows in `README.md` and
  `CONTRIBUTING.md`

### Beta

- CLI agent behavior
- Metal serving path
- GGUF loading path
- benchmark tooling in `scripts/`

### Experimental

- fast-moving quantization paths
- speculative decoding
- tensor-parallel scaffolding
- undocumented flags or undocumented environment variables

### Internal

- internal modules under `infer/src/` unless explicitly documented otherwise
- scheduler internals
- backend dispatch internals
- model-specific buffer/layout details

---

## 3. What Counts as Breaking

For stable surfaces, these count as breaking changes:

- removing or renaming a documented CLI flag or command
- changing a documented HTTP request/response contract
- changing documented environment-variable behavior
- changing documented file or session formats without migration guidance

These usually do not count as breaking:

- internal refactors
- undocumented helper renames
- experimental-surface redesigns
- additive optional fields

When in doubt, treat the change as breaking and document it.

---

## 4. Review Expectations by Level

### Stable

Require:

- tests or targeted verification
- docs update if behavior changed
- changelog coverage when user-visible
- migration note when needed

### Beta

Require:

- explicit verification path
- docs update when behavior changes materially

### Experimental

Require:

- clear scope
- minimal validation
- explicit limitation framing

---

## 5. Promotion Rule

A surface should move toward stable only when it has all of the following:

1. a clear user story
2. durable documentation
3. repeatable verification
4. no major semantic ambiguity
5. at least one release cycle without immediate redesign pressure

---

## 6. Maintainer Rule of Thumb

If users are likely to depend on a behavior, classify it explicitly.

If the project is not ready to support that behavior long-term, mark it beta or
experimental instead of letting it become an accidental stable contract.

Related docs:

- [support-matrix.md](support-matrix.md)
- [compatibility.md](compatibility.md)
- [perf-and-correctness-gates.md](perf-and-correctness-gates.md)
