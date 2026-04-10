# Release Checklist

This checklist defines the minimum steps for cutting a release of
`agent-infer`.

It is meant to keep releases repeatable, readable, and safe.

---

## 1. Confirm the Release Scope

Before tagging, confirm:

- what changed since the last release
- whether any breaking change is included
- whether support expectations changed
- whether any beta or experimental surface changed status

---

## 2. Update Required Documents

Review and update, as needed:

1. `README.md`
2. `CONTRIBUTING.md`
3. `CHANGELOG.md`
4. `docs/support-matrix.md`
5. `docs/compatibility.md`
6. `docs/stability-policy.md`

At minimum, make sure:

- new user-facing features are documented
- changed behavior is documented
- deprecated or removed surfaces are documented
- migration guidance exists when required

---

## 3. Run Validation

Typical baseline:

```bash
cargo test --no-default-features --features no-cuda
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
python -m pytest tests/ -v
```

Then add targeted validation depending on what changed.

Use [perf-and-correctness-gates.md](perf-and-correctness-gates.md) as the rule
for deciding what else must run.

---

## 4. Verify Release Artifacts

Current release automation publishes:

- Linux x86_64 CUDA artifacts
- macOS arm64 Metal artifacts

Before release, verify:

- `.github/workflows/release.yml` still matches intended support
- artifact names are correct
- packaged binaries are the intended ones

---

## 5. Review Compatibility

Before tagging, answer:

1. Did any documented CLI behavior change?
2. Did any documented HTTP behavior change?
3. Did any documented environment variable change?
4. Did the support matrix change?
5. Does upgrading require user action?

If yes, reflect that in:

- `CHANGELOG.md`
- `docs/compatibility.md`
- `docs/support-matrix.md`
- README examples when relevant

---

## 6. Tag and Publish

Recommended sequence:

1. merge release-prep changes
2. verify default branch is green
3. create and push tag `vX.Y.Z`
4. confirm release workflow succeeds
5. inspect generated release notes and uploaded artifacts

---

## 7. Post-Release Check

After release:

- verify the GitHub Release page is readable
- verify users can identify the correct artifact
- open follow-up issues for any known deferred work

---

## 8. Short Checklist

- [ ] Docs updated
- [ ] Changelog updated
- [ ] Compatibility reviewed
- [ ] Support matrix updated
- [ ] Validation run
- [ ] Artifacts verified
- [ ] Tag created and workflow passed

Related docs:

- [compatibility.md](compatibility.md)
- [support-matrix.md](support-matrix.md)
- [perf-and-correctness-gates.md](perf-and-correctness-gates.md)
