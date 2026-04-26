# Release Checklist

This checklist defines the minimum steps for cutting a release of
`ARLE`.

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
2. `README.zh-CN.md`
3. `ROADMAP.md`
4. `docs/index.md`
5. `CONTRIBUTING.md`
6. `CHANGELOG.md`
7. `docs/support-matrix.md`
8. `docs/stability-policy.md`

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

When the release surface changes:

- treat `.github/workflows/release.yml` as the packaging authority
- keep `.github/workflows/metal-ci.yml` in lockstep for the macOS
  packaging script so default-branch CI exercises the same tarball
  shape that release packaging will publish
- the shared macOS packaging script is
  `scripts/package_macos_metal_artifact.sh`
- for Metal-facing changes, make sure branch validation still covers the
  exact `cargo build --no-default-features --features metal,no-cuda --bin metal_serve --release`
  path and the matching `cargo build --no-default-features --features metal,no-cuda,cli -p agent-infer --bin arle --release`
  path, not just library checks

---

## 4. Verify Release Artifacts

Current release automation publishes:

- Linux x86_64 CUDA artifacts named `arle-<version>-linux-x86_64.tar.gz`
  containing `arle`, `infer`, and `bench_serving`
- macOS arm64 Metal artifacts named `arle-<version>-macos-arm64.tar.gz`
  containing `arle` and `metal_serve`
- branch CI uploads the same tarball layout for validation

Before release, verify:

- `.github/workflows/release.yml` still matches intended support and
  remains the artifact-packaging authority
- `.github/workflows/metal-ci.yml` still mirrors the same macOS
  packaging script used by release packaging
- both workflows still call `scripts/package_macos_metal_artifact.sh`
- artifact names are correct
- packaged binaries are the intended ones (`arle`, `infer`, `bench_serving` on
  Linux; `arle`, `metal_serve` on macOS)
- unpacked artifacts pass the local smoke check:
  `./arle --doctor --json` and `./arle serve --help`

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
- `docs/stability-policy.md`
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

- [stability-policy.md](stability-policy.md)
- [support-matrix.md](support-matrix.md)
- [perf-and-correctness-gates.md](perf-and-correctness-gates.md)
