# ARLE installers: curl|sh + Homebrew tap (Mac arm64 + Linux x86_64)

## Context

v0.1.0 already shipped `arle-v0.1.0-{macos-arm64,linux-x86_64}.tar.gz`
on the GitHub Release, but users had no first-class install path —
only `git clone && cargo build`. Two missing layers:

1. A platform-aware installer that downloads + SHA-verifies the right
   tarball without making the user think.
2. A Homebrew tap so Mac users get the `brew install` muscle memory.

## What Worked

Stayed on the existing `release.yml` (incremental add) instead of
swapping the whole pipeline to `cargo-dist`. The existing workflow
already handles the multi-SM CUDA build and Docker push to ghcr.io;
cargo-dist wouldn't have improved either. Three additions on top:

- **`scripts/install.sh`** — POSIX `sh`, `uname` platform detection,
  `curl` + SHA256 (via `sha256sum` or `shasum`) + `tar`, install to
  `~/.local/bin` with `INSTALL_DIR` override. ~80 lines, no deps.
  End-to-end smoke against the live v0.1.0 release confirmed download
  + SHA verify + extract + Mach-O arm64 install.
- **`cklxx/homebrew-tap`** with `Formula/arle.rb` pinned to v0.1.0
  via the actual tarball SHAs from `SHA256SUMS.txt`. Real
  `brew install cklxx/tap/arle` → `arle --help` worked first try.
- **`bump-homebrew` job in `release.yml`** using
  `mislav/bump-homebrew-formula-action@v3`. Needs a one-time
  `HOMEBREW_TAP_TOKEN` repo secret (fine-grained PAT on the tap
  repo with `Contents: Read and write`); after that, every `v*` tag
  bumps the formula automatically.

The validation loop was unusually tight: build the formula, push to
the tap, `brew install` from a clean tap state, verify the binary
runs — full feedback in <2 minutes per iteration because the
artifacts already existed on the release. Skip-the-CI-roundtrip
strategy worth reusing for any release-tooling change.

## Rule

- **Don't replace working release infra to add an install path.**
  Layer the new ergonomics (installer.sh, brew formula) on top.
  cargo-dist is the right call for *new* projects without a release
  pipeline; for projects with a working `release.yml`, the cost of
  swapping it usually exceeds the benefit.
- **Validate Homebrew formulae against the live tap** with the actual
  release tarball before bumping. `brew audit` catches syntax;
  only `brew install` catches "the formula installs the wrong files
  for this platform." The macOS install test ran in 2s and would
  have caught any bin-name typo.
- **Pin the tap-bump action behind a fine-grained PAT** (not a
  classic `GITHUB_TOKEN`) so the tap repo isn't writable by every
  job in `cklxx/arle`. Document the secret in the release checklist
  so it's discoverable at the next release cut.
