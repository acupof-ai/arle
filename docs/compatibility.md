# Compatibility and Deprecation Policy

This document defines how `agent-infer` handles breaking changes,
compatibility-sensitive behavior, and deprecations.

The rule is simple: if users can reasonably build around a documented surface,
maintainers must treat it carefully.

---

## 1. Compatibility-Sensitive Surfaces

The following are compatibility-sensitive when documented:

- HTTP APIs
- CLI commands and flags
- environment variables
- session or output formats
- operational workflows documented in `README.md` or `CONTRIBUTING.md`

Undocumented internal modules are not covered by this policy.

---

## 2. Stable vs Beta vs Experimental

- **Stable**: avoid breaking changes unless necessary, and always document them.
- **Beta**: changes are allowed, but meaningful user-visible shifts must still
  be documented.
- **Experimental**: can move faster, but should still not change silently if
  documentation exposes them.

See [stability-policy.md](stability-policy.md) for the level definitions.

---

## 3. What to Do Before Breaking Something

Before changing a documented surface, answer:

1. Is this breaking?
2. Who is affected?
3. Why is the change necessary?
4. What is the migration path?
5. What docs and tests must change?

If those answers are unclear, the change is not ready.

---

## 4. Deprecation Process

When practical, use this order:

1. document the replacement
2. mark the old surface deprecated
3. keep compatibility for at least one release cycle
4. remove the old surface with migration notes

If fast removal is required for correctness or security, say so explicitly in
the changelog or release notes.

---

## 5. Environment Variable Rule

Environment variables are part of the external surface once documented.

Rules:

- documented variables are compatibility-sensitive
- undocumented variables are not guaranteed stable
- naming should converge toward one primary prefix over time
- legacy variables may remain temporarily, but should be called out clearly

See [environment.md](environment.md) for the current reference list.

---

## 6. Changelog Rule

If a change affects compatibility, the changelog must say so.

At minimum, record:

- breaking changes
- deprecated surfaces
- removed surfaces
- migration notes when users must act
- support changes when they affect user expectations

---

## 7. Maintainer Rule of Thumb

If a user could reasonably learn a behavior from the docs and rely on it, treat
that behavior as compatibility-sensitive.

Related docs:

- [stability-policy.md](stability-policy.md)
- [support-matrix.md](support-matrix.md)
- [environment.md](environment.md)
