Relevant policies:
- [docs/support-matrix.md](../docs/support-matrix.md)
- [docs/stability-policy.md](../docs/stability-policy.md)
- [docs/compatibility.md](../docs/compatibility.md)
- [docs/perf-and-correctness-gates.md](../docs/perf-and-correctness-gates.md)
- [docs/release-checklist.md](../docs/release-checklist.md)

## Summary

Brief description of the change.

## Why

What problem does this solve?

## Surface Area

- [ ] Internal-only implementation
- [ ] Public CLI / API / environment variable
- [ ] Backend / runtime / scheduler / model behavior
- [ ] Benchmark / profiling / docs / release process

## Stability / Support / Compatibility

- Stability tier impacted:
- Support matrix impact:
- Compatibility / deprecation / migration notes:

## Docs Updated

- [ ] `README.md` / `README.zh-CN.md`
- [ ] `docs/support-matrix.md`
- [ ] `docs/stability-policy.md`
- [ ] `docs/compatibility.md`
- [ ] `docs/environment.md`
- [ ] `CHANGELOG.md`
- [ ] No doc updates required

## Validation

- [ ] `python3 scripts/check_repo_hygiene.py`
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace -- -D warnings`
- [ ] Targeted `cargo test` / integration / e2e coverage
- [ ] Other:

## Benchmark / Profiling Evidence

- [ ] Bench-exempt; reason:
- [ ] Benchmark / profile attached; commands and before/after:

## Migration Notes

None / not applicable, or describe the operator-facing migration here.
