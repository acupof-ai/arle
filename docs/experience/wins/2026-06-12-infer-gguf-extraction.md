# infer-gguf extraction — lateral backend dep killed by a pure move

## Context

Roadmap tranche R1
([`2026-06-12-architecture-refactor-roadmap.md`](../../plans/2026-06-12-architecture-refactor-roadmap.md) §4):
`infer-vulkan` depended on sibling backend `infer-hip` solely for the GGUF
host substrate (`gguf`/`dequant`/`config` modules) — a lateral backend
coupling (problem P2). Fix: extract those three host-only modules to a
neutral leaf crate `crates/infer-gguf` and point both backends at it.
Structural refactor only; no behavior change, no perf claim.

## What Worked

- **Pure move via `git mv`**: `infer-hip/src/gguf.rs` →
  `infer-gguf/src/gguf.rs`, `dequant.rs` → `dequant.rs`, `config.rs` →
  `deepseek4.rs` (renamed — the module is the deepseek4-arch
  GGUF→`DeepSeekV4Config` mapper, not generic config; file naming must
  match content semantics). `dequant.rs` and `deepseek4.rs` are
  byte-identical to their sources; the only content edits anywhere are
  module-path fixes (`crate::{config,dequant,gguf}` →
  `infer_gguf::{deepseek4,dequant,gguf}`) plus rustfmt rewrap.
- **One forced visibility change**: `gguf::test_writer` was
  `#[cfg(test)] pub(crate)`; infer-hip's executor/loader unit tests consume
  it cross-crate after the move, which `cfg(test)` cannot serve (dependency
  builds compile without it). It is now `pub` and always compiled — the
  minimal change the move itself forces, documented in the module doc.
- **No re-export shim left in `infer-hip`**; `infer-vulkan`'s re-export
  switched to `pub use infer_gguf::{deepseek4, dequant, gguf};` (public
  name `config` → `deepseek4`). `memmap2` moved out of `infer-hip`'s deps
  (its only consumer was `gguf.rs`). `infer-vulkan` ends with zero
  `infer-hip`/`infer_hip` references (Cargo.toml, code, and doc comments).
- **Docs rode along in the same tranche** (the R0.2 CI gate enforces
  codebase-map membership): codebase-map §1/§3.9/§4 + dependency diagram,
  architecture.md Package Boundaries rows + Dependency Direction block —
  the `infer-vulkan → infer-hip` DEBT markers from R0.1 are gone.

Verification matrix (all PASS, 2026-06-12 local host):

| Command | Result |
| --- | --- |
| `cargo check -p infer-gguf -p infer-hip -p infer-vulkan` | PASS |
| `cargo test -p infer-gguf -p infer-hip -p infer-vulkan` | PASS — 15 + 17 + 22 unit tests, 0 failed |
| `cargo clippy -p infer-gguf -p infer-hip -p infer-vulkan -- -D warnings` | PASS, clean |
| `cargo check -p infer-api --no-default-features --features hip` | PASS |
| `cargo check -p infer-api --no-default-features --features vulkan` | PASS |
| `python3 scripts/check_repo_hygiene.py` | PASS (`[repo-hygiene] OK`) |
| `grep -rn "infer-hip" crates/infer-vulkan/ --include=*.toml --include=*.rs` | empty |
| `grep -rn "infer_hip" crates/infer-vulkan/src` | empty |
| `grep -rn "memmap2" crates/infer-hip/` | empty |

Host-only structural move; no GPU lane touched, so no bench delta is
applicable (bench-spec exemption: no hot-path behavior change — the diff is
paths + Cargo plumbing).

## Rule

Shared host substrate between sibling backends lives in a neutral leaf
crate, never in one backend re-exported by another. Per-arch GGUF→config
mappers belong in the substrate crate (depending on `*-spec`); `*-spec`
crates stay dependency-pure and never depend back. When a pure move strands
a `#[cfg(test)]` helper behind a crate boundary, promote it to an
always-compiled `pub` module in the new crate — the one deviation a move
legitimately forces — and record it.
