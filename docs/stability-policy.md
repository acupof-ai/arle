# Stability and Compatibility Policy

This document is the single source of truth for `ARLE` stability levels,
compatibility-sensitive surfaces, breaking-change rules, and the deprecation
process.

The rule is simple: if users can reasonably build around a documented surface,
maintainers must treat it carefully. Users and contributors should know what
they can depend on, and maintainers should know what must be reviewed
carefully.

---

## 1. Stability Levels

`ARLE` uses four levels.

### Stable

For documented, intended-for-use surfaces.

- avoid breaking changes
- document user-visible behavior changes
- provide migration guidance when users must act

### Beta

For documented and usable surfaces that are still evolving.

- keep behavior reasonably consistent
- document meaningful user-visible changes
- prefer transition over abrupt removal

### Experimental

For early-stage or fast-moving surfaces.

- may change faster
- must still be explicit about limitations
- should not be presented as long-term stable contracts

### Internal

For implementation details.

- may be refactored freely
- must not be treated as public extension points unless promoted explicitly

---

## 2. Current Classification

This reflects repository state as of 2026-04-21. Per-surface support status
lives in [support-matrix.md](support-matrix.md); this section names which
surfaces fall under each tier.

### Stable

- documented HTTP APIs
  - `POST /v1/completions`
  - `POST /v1/chat/completions`
  - `GET /v1/models`
  - `GET /healthz`
  - `GET /readyz`
- `GET /metrics`
- `GET /v1/stats`
- documented build, test, lint, and format workflows in `README.md` and
  `CONTRIBUTING.md`

### Beta

- `POST /v1/responses` (current text/tool-call subset with non-streaming and SSE forms; structured outputs still pending)
- CLI agent behavior
- train-side `/v1/train/status|events|stop|save` control plane exposed by the active training binaries' `--serve` flag
- Metal serving path
- Metal DFlash speculative decode path
- GGUF loading path
- benchmark tooling in `scripts/`

### Experimental

- fast-moving quantization paths (TurboQuant KV, W2, new quant formats)
- speculative decoding framework on CUDA (CPU stubs only today)
- tensor-parallel scaffolding (sharding math shipped, NCCL comm not wired)
- undocumented flags or undocumented environment variables

### Internal

- internal modules under `infer/src/` unless explicitly documented otherwise
- scheduler internals
- backend dispatch internals
- model-specific buffer/layout details

---

## 3. Compatibility-Sensitive Surfaces

The following are compatibility-sensitive when documented:

- HTTP APIs
- CLI commands and flags
- environment variables (see [environment.md](environment.md) for the
  reference list)
- session or output formats
- operational workflows documented in `README.md` or `CONTRIBUTING.md`

Undocumented internal modules are not covered by this policy.

### What Counts as Breaking

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

### Environment Variable Rule

Environment variables are part of the external surface once documented.

- documented variables are compatibility-sensitive
- undocumented variables are not guaranteed stable
- naming should converge toward one primary prefix over time
- legacy variables may remain temporarily, but should be called out clearly

---

## 4. Before Breaking Something

Before changing a documented surface, answer:

1. Is this breaking?
2. Who is affected?
3. Why is the change necessary?
4. What is the migration path?
5. What docs and tests must change?

If those answers are unclear, the change is not ready.

---

## 5. Deprecation Process

When practical, use this order:

1. document the replacement
2. mark the old surface deprecated
3. keep compatibility for at least one release cycle
4. remove the old surface with migration notes

If fast removal is required for correctness or security, say so explicitly in
the changelog or release notes.

---

## 6. Review Expectations by Level

| Level | Required for review |
| --- | --- |
| Stable | Tests or targeted verification; docs update on behavior change; changelog coverage when user-visible; migration note when needed. |
| Beta | Explicit verification path; docs update when behavior changes materially. |
| Experimental | Clear scope, minimal validation, explicit limitation framing. |

---

## 7. Changelog Rule

If a change affects compatibility, the changelog must say so.

At minimum, record:

- breaking changes
- deprecated surfaces
- removed surfaces
- migration notes when users must act
- support changes when they affect user expectations

---

## 8. Promotion Rule

A surface should move toward stable only when it has all of the following:

1. a clear user story
2. durable documentation
3. repeatable verification
4. no major semantic ambiguity
5. at least one release cycle without immediate redesign pressure

---

## 9. Maintainer Rule of Thumb

If users are likely to depend on a behavior, classify it explicitly. If a user
could reasonably learn a behavior from the docs and rely on it, treat that
behavior as compatibility-sensitive.

If the project is not ready to support that behavior long-term, mark it beta or
experimental instead of letting it become an accidental stable contract.

Related docs:

- [support-matrix.md](support-matrix.md)
- [environment.md](environment.md)
- [perf-and-correctness-gates.md](perf-and-correctness-gates.md)
