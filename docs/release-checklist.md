# Release Checklist

Release automation fails closed unless the tag commit, workspace version, explicit blocker registry, and passed kernel evidence agree.

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
3. `docs/index.md`
4. `CONTRIBUTING.md`
5. `CHANGELOG.md`
6. `docs/support-matrix.md`
7. `docs/stability-policy.md`

At minimum, make sure:

- new user-facing features are documented
- changed behavior is documented
- deprecated or removed surfaces are documented
- migration guidance exists when required

### 2a. Bump the workspace version

Cargo's `workspace.package.version` lives in the root `Cargo.toml`. Before
tagging `vX.Y.Z`, sync it to that version so anyone reading `Cargo.lock` or
`cargo metadata` can correlate a build to the release. The simplest path:

```bash
cargo install cargo-edit            # one-time, if not already installed
cargo set-version --workspace X.Y.Z # then commit the change
```

If you skip this step, the binaries you ship will still report the *previous*
version via `arle --version`, which has confused operators and bisecting
contributors before. The hygiene check does not catch this — that's why the
bump lives in this checklist.

---

## 3. Run Validation

Before tagging, `release-blockers.json` must contain exactly the current explicit blockers. Release automation never scans historical docs. An empty release-ready registry is:

```json
{"schema":1,"blockers":[]}
```

Validate the candidate tag against a local qualified kernel bundle directory or the `kernel-artifacts` release:

```bash
scripts/validate_release.sh vX.Y.Z /path/to/bundle-assets
scripts/validate_release.sh vX.Y.Z
```

The validator requires tag commit = checkout commit, tag base version = workspace product version, zero blockers, and a fetched qualified archive whose aggregate qualification sidecar binds its SHA-256 and current bundle ID. The tested commit must exist locally and be an ancestor of the tag; fetch or unshallow history when Git cannot prove ancestry. Descendant commits remain qualified only while `scripts/kernel_artifacts.sh id` is unchanged.

Typical baseline:

```bash
cargo test --no-default-features --features no-cuda
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
python -m pytest tests/python/ -v
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
  exact `cargo build --no-default-features --features metal,no-cuda,cli -p arle --bin arle --release`
  path (the Metal release artifact), not just library checks

---

## 4. Verify Release Artifacts

Current release automation publishes:

- Linux x86_64 CUDA artifacts named `arle-<version>-linux-x86_64.tar.gz`
  containing `arle` (CUDA serving runs via `arle serve`)
- macOS arm64 Metal artifacts named `arle-<version>-macos-arm64.tar.gz`
  containing `arle` (Metal serving runs via `arle serve --backend metal`)
- the `install.sh` shell installer (uploaded as a top-level release asset
  so `curl -fsSL .../releases/latest/download/install.sh | sh` resolves)
- `SHA256SUMS.txt` (consumed by `install.sh` for verification)
- branch CI uploads the same tarball layout for validation

Before release, verify:

- `.github/workflows/release.yml` still matches intended support and
  remains the artifact-packaging authority
- `.github/workflows/metal-ci.yml` still mirrors the same macOS
  packaging script used by release packaging
- both workflows still call `scripts/package_macos_metal_artifact.sh`
- `scripts/install.sh` still matches the artifact naming used by
  release packaging (platform string, binary list)
- artifact names are correct
- packaged binaries are the intended ones (`arle` on both Linux and macOS)
- unpacked artifacts pass the local smoke check:
  `./arle --doctor --json` and `./arle serve --help`

### 4a. Homebrew tap

The `bump-homebrew` job in `release.yml` updates
[`cklxx/homebrew-tap`](https://github.com/cklxx/homebrew-tap)'s
`Formula/arle.rb` after `create-release` succeeds. It needs:

- repository secret `HOMEBREW_TAP_TOKEN` — a fine-grained PAT scoped to
  `cklxx/homebrew-tap` with `Contents: Read and write`. Without it the
  job fails (the rest of the release still ships).
- the tag must not contain `-` (pre-releases like `v0.2.0-rc1` are skipped).

After the bump PR/commit lands, validate:

```bash
brew update
brew install cklxx/tap/arle
arle --doctor
```

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
- [ ] Workspace version bumped (`cargo set-version --workspace X.Y.Z`)
- [ ] Changelog updated
- [ ] Compatibility reviewed
- [ ] Support matrix updated
- [ ] `release-blockers.json` has zero blockers
- [ ] Qualified kernel evidence matches tag commit and bundle ID
- [ ] Product identity matches tag/workspace version
- [ ] Validation run
- [ ] Artifacts verified
- [ ] Tag created and workflow passed

Related docs:

- [stability-policy.md](stability-policy.md)
- [support-matrix.md](support-matrix.md)
- [perf-and-correctness-gates.md](perf-and-correctness-gates.md)
