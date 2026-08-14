# ARLE — Agent Contract

Assisting **ckl**. Project caveats and hard gates only — generic Rust / CUDA /
Metal / git knowledge is intentionally absent, and so is anything you can read
off the file tree. Match the surrounding code's idiom, naming, and comment
density rather than a style rulebook.

**Load on demand, not upfront:**

| When | Read |
|------|------|
| Evidence bar, decomposition, distilled lessons | [`docs/agent-method.md`](docs/agent-method.md) |
| Any bench or trace | [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md) |
| Where code lives / execution paths | [`docs/codebase-map.md`](docs/codebase-map.md), [`docs/architecture.md`](docs/architecture.md) |
| Editing `crates/{autograd,cuda-kernels,mlx-sys}/` | that crate's `AGENTS.md` |
| Backend / model / quant support level | [`docs/support-matrix.md`](docs/support-matrix.md) |
| Env vars, SM tier policy | [`docs/environment.md`](docs/environment.md) |
| Session start | [`docs/index.md`](docs/index.md) (PARA index) |

`AGENTS.md` is canonical; `CLAUDE.md` is a symlink to it.

---

## Project shape

`ARLE` is a Rust-native, device-neutral inference runtime with an integrated
local agent and **On-Policy Distillation (OPD)** workflows. No PyTorch, no Python
on the hot path.

Two backends plug into one seam (`infer_seam::{BackendExecutor, KvPool}`, two
host-only traits): CUDA continuous batching (`cudarc` + vendored FlashMLA /
DeepGEMM / DeepEP + TileLang AOT + native CUDA C) and Metal (`crates/mlx-sys`
C++ bridge, packed varlen decode). One `infer_core::Engine<E, K>` drives both —
**the seam is the engine's cost contract (submit/poll + `StepLimits` +
explicit capability accessors): a new backend implements the core loop and
declares its capabilities, and a family whose decode is not submit-poll
shaped forks the loop (precedent: `diffusion_executor.rs`) rather than
bending the trait.**

Non-obvious ownership:
- **`infer-*` owns serving/runtime truth.** The monolithic `infer/` crate was
  deleted 2026-06-04 (`e81b98fb`, ~167k LOC) — any doc or command referencing
  `infer/` or `-p infer` is stale.
- `infer-api` (`LoadedInferenceEngine`) is the single programmatic entry point;
  `arle` is the CLI entry point.
- **`train` is OPD-only** (no second product line). Scratch pretrain / SFT /
  GRPO / multi-turn RL were deleted in the 2026-05-18 pivot (pretrain unwinnable
  at a 322× gap; the rest duplicate vLLM+verl / TRL / axolotl). OPD is the one
  axis where ARLE's runtime authority differentiates.
- CUDA kernels: adopt-official-first (`vendor/`), hand-rolled at
  `crates/cuda-kernels/csrc/` only for the genuine gap.

**Metal canonical model — globally unified:
`mlx-community/Qwen3.6-35B-A3B-4bit`** (MoE, ~19 GB, HF-cached) — the default for
every Metal serve, `scripts/bench_*.sh`, smoke, and Metal wins/errors; detects
MoE regressions a dense model cannot. Unit-test opt-out:
`INFER_TEST_MODEL_PATH=models/Qwen3.5-0.8B-MLX-4bit` (document why). CUDA benches
keep their own defaults.
- **Auto-wired-limit** (always-on): the Metal executor pins weights via
  `mlx::set_wired_limit` at construction (`infer-metal/src/wired_limit.rs`) —
  c=1 p99 86→15 ms on Qwen3.6.
- **`MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` are not defaults** — a
  Qwen3.5-dense tune, wash-or-loss on Qwen3.6 MoE. Per-workload matched-A/B only.

---

## Hard gates

**Backend isolation (CRITICAL).** Never `cfg`-leak backend types into
cross-backend modules — everything above the seam (`infer-core` / `-server` /
`-api`) stays device-neutral; backend types live only in `infer-cuda` /
`infer-metal`. CUDA stubs on other targets: `todo!("GPU required: ...")`. Mac
pre-push typecheck without nvcc:
`cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`.

**Every runtime change produces a bench entry.** A dated entry under
`docs/experience/wins/` (or `errors/` on regression) — no entry, not shipped. In
scope: `crates/infer-*/src/`, `crates/cuda-kernels/csrc/`, `crates/mlx-sys/src/`,
`src/`, `scripts/bench_*` param changes, feature-flag default flips, hot-path dep
bumps. Exempt: docs / agent files / memory / dev-only tooling — say so in the
commit body. Can't run locally (CUDA on a Mac) → stub `pending-remote` and cite
the remote ticket; no silent skips. Minimum, params, and the A/B contract live in
the bench spec.

**GPU kernel work** ships a measured before/after — `ncu` (CUDA) or Xcode Metal
capture / MLX instruments (Metal).

**Fast path only.** If it only works on the eager/un-captured path,
it isn't done — no per-step readback/sync in the hot loop; capture at sync
points, rebuild transient state on restore.

**Correctness parity = the correct-inference gate** (byte-identity is not
required, due to MoE non-determinism): `scripts/needle_gate.py` + `scripts/lever_gate.sh`, needle
ladder ×3 same-config vs the baseline envelope. Default flips additionally need a
wall-clock perf license.

**No half-states.** Finish a refactor unit or revert it; never leave parallel
old+new paths in the tree.

**Use plain language.** Say the finding in plain words first, numbers second.
No dense jargon, no hedging, no restating the question. If a sentence needs a
glossary, rewrite it.

**Approach-first for >3 files or architectural decisions** — outline, then
execute. Wait for the user ONLY when there is a real tradeoff to adjudicate
(two viable paths with different costs). No tradeoff → nothing to decide →
don't ask. Adopting the SOTA/industry-standard approach is never a decision
point — just execute it (2026-08-04).
Never delete content outside the stated scope; inside it, prefer deletion-style
refactors (collapse duplicates, converge on one flow) over layering adapters.

---

## Working rules

**Phases** (non-trivial tasks): Explore until you can name every file you will
touch → Plan (accepted in writing; >5 files or irreversible → stop and flag) →
Implement (compiles, simplify pass on the diff) → Verify (`cargo test
--workspace`, `cargo clippy -- -D warnings`, bench entry) → Reflect (bug that
took >1 attempt → `docs/experience/errors/`; user correction → feedback memory).
Trivial → Implement + Verify.

**Tests: minimal and end-to-end.** Default is no new test. Add one only when the
change carries logic that can silently break (branch, parser, quant, rollback,
sampling, security), and then the smallest end-to-end gate that fails when it
breaks — not a per-function suite.

**Delegation.** Claude does direction, docs, planning, and integration;
`general-purpose` subagents execute; `Explore` maps; `Plan` handles >5-file
plans; review is `codex review --uncommitted` at Bash, backgrounded and tee'd.
Independent tasks go out in one message, in parallel. Two failed subagent
attempts → hand-write the diff or re-brief a fresh agent with what was tried.
**Never** `codex:codex-rescue` / `mcp__openmax__execute_with_codex` for
execution — both hang (2026-04-19).

**Git.** Commitizen `<type>(<scope>): <subject>`, scopes `metal` `cuda`
`scheduler` `qwen3` `qwen35` `http` `kv-tier` `docs`. Commit directly to `main`
from the current workspace — no feature branches, no alternate worktree. Small
tranches, each self-contained, simplify pass first. Never `git stash` others'
work; commit only your own files by explicit path. After `git mv` + edits,
re-check `git status` — the fmt hook de-stages renames.

**CHANGELOG is the central progress record.** Three event classes land a line the same
day, linking the wins/errors entry: **phase exit · default flip ·
accept-or-reject verdict**. Phase exits also cut a release tag. Weekly (~30 min):
CHANGELOG catch-up; promote patterns recurring ≥3× into `docs/agent-method.md`;
archive the oldest zero-inbound-reference wins entries before the
`check_repo_hygiene` cap blocks a push; drift-probe
`git log --since='7 days ago' -- 'crates/infer-*/src'` against `docs/experience/`.

**Code layout gotchas.** Flat modules, no `mod.rs` — `src/ops.rs` declares a
sibling `#[path = "ops/attention.rs"] mod attention;`. Weights are `&self`
(immutable, pool-shared); per-request mutable state lives in the `State`
associated type. Comments carry the non-obvious *why* in ≤1 line, in English,
never the *what* and never which task added it; issue numbers only when naming a
specific bug. If the code already reads clearly, leave it bare — no comment.

**Memory.** Always-loaded: the auto-memory index + the latest 3 of
`docs/experience/{errors,wins}/`; full entries on demand. Skeletons:
`errors/YYYY-MM-DD-slug.md` = Context / Root Cause / Fix / Rule;
`wins/…` = Context / What Worked / Rule. Bench snapshots use
[`TEMPLATE-bench.md`](docs/experience/wins/TEMPLATE-bench.md), never overwritten.

---

## Writing (reports & experiment notes)

1. **Standard terms.** Use the established term for a concept; do not coin a
   new one.
2. **No metaphors.** Name the referent directly; any phrase that needs the
   reader to infer what it stands for gets rewritten.
3. **Neutral headers.** Table headers and category names are neutral nouns
   (Problem, Phenomenon, Impact, Result). Keep section status labels
   consistent (e.g. Completed / Confirmed / Below baseline / Pending).
4. **No "X, not Y" sentences.** State the fact directly without contrastive
   framing.
5. **Table cells may be descriptive.** Not every entry needs a number or
   conclusion — an entry may describe only what happened.
6. **Unknown cause.** Write "cause unknown"; do not append unverified
   explanations. Once a mechanism is confirmed, that content may stay.
7. **No colloquialisms.** Formal, precise wording only.
8. **No personification.** Systems, processes, and data do not act or feel.

---

## Build & run

Always `--release` — debug GPU builds are unusably slow.

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda        # CUDA (Linux+NVIDIA)
cargo build --release --no-default-features --features metal,no-cuda   # Metal (Apple Silicon)
cargo build --release --no-default-features --features cpu,no-cuda     # portable / CI smoke
# Multi-GPU features stack: nccl (implies cuda) → deepep (implies nccl)

# CI test lanes use --profile release-fast (cu=16, no LTO); the release profile's
# cu=1 cold build OOMs 16 GB CI runners:
cargo test -p arle --profile release-fast --no-default-features --features cpu,no-cuda,cli
cargo test -p cli --release --no-default-features --features metal,no-cuda
cargo test -p kv-native-sys --profile release-fast
```

Weekly: `cargo sweep --time 30`.
