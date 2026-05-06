# 2026-05-07 · Session arc — deferred bug closed, M1+M2+M3 unification landed, full combo plan spec'd

## Goal
Capture the multi-hour 2026-05-06→07 session in one wins entry as a
durable record. Two parallel agents (Claude + Codex) coordinated through
git + tmux, **31 commits** landed on `origin/main` from `61e9835` to
`60fd7d3`.

## Hypothesis
A user-driven push of the form "代码全部跑起来吧 → 真bug就修复呀 →
推进世界第一... 顶级管理者" can be sustained over a multi-hour cron-loop
session if the agents agree on a delegation contract (Claude = plans /
direction; Codex = code / kernel work; both peer-review via
`codex review --uncommitted`).

## What worked

### Phase 1 — Bring-up + the deferred 04-13 bug (commits a2428af → 9833068)

| Commit | What | Owner |
|---|---|---|
| `a2428af` | `complete_stream` blocking contract on CUDA | Claude |
| `3f4c1b7` | `NVCC_CCBIN` for GCC≥16 hosts (Arch / CachyOS) | Claude |
| `4ddfa5d` | TileLang decode `total_pages` graph-stable scalar | Codex |
| `9833068` | `INFER_DETERMINISTIC=1` skips cublasLt autotune | Claude |

The 04-13 deferred bug
([`2026-04-13-batched-decode-high-concurrency.md`](../errors/2026-04-13-batched-decode-high-concurrency.md))
peeled into three layered sub-bugs in this order:

1. CUDA Graph capture freezes scalar kernel arguments — TileLang paged
   attention had `total_pages` baked in at warmup → kernel rejected
   reads past warmup-time bound → gibberish after first decode page
   boundary.
2. Autotune algo cache keyed by `(M, N, K)` → B=1 vs B=3 GEMMs land on
   different fp accumulation paths → greedy argmax flips.
3. Even with autotune off, `cublasLtMatmulAlgoGetHeuristic` and
   `cublasGemmEx` both pick different internal kernels for M=1 vs M=3.

`bb648fe` (Codex, Phase 2) closed this with the per-row N=1 graph-safe
GEMM trick: in deterministic mode, route all BF16 dense GEMMs through
M=1 calls so the kernel sees the same M regardless of the actual
batch size. **Solo (B=1) and Concurrent (B=3) decode now produce
byte-exact identical output, matching the HF baseline in
`infer/test_data/Qwen3-4B.json`.**

### Phase 2 — Strategic plans materialized (commits ec54340 → 4001681)

Three peer plans drafted within 90 minutes by both agents:

| Plan | Owner | Scope |
|---|---|---|
| [`backend-unification.md`](../../plans/backend-unification.md) | Codex | M1-M7 CUDA↔Metal收敛主线, 12 weeks to world-#1 |
| [`dsv4-small-repro.md`](../../plans/dsv4-small-repro.md) | Codex | DeepSeek-V4 1B from-scratch pretrain on 1×4070 Ti SUPER 16GB |
| [`longctx-spec-tilelang-combo.md`](../../plans/longctx-spec-tilelang-combo.md) | Claude | Cross-vendor combinatorial pipeline (spec×tilelang×hybrid×tier-KV) |

These are **complementary, not redundant**:
- Codex's two cover inference convergence + training substrate.
- Claude's covers cross-vendor combinatorial innovation that none of
  vLLM / TGI / SGLang / TRT-LLM / mlx-lm ships today.

### Phase 3 — M1 + M2 + M3 implementation (commits 92b5ba9 → a9f0327)

- **M1** unified backend telemetry across CUDA + Metal (`92b5ba9`).
- **M2** Metal kv-tier T2 adapter (`f8f063d`).
- **M3** unified scheduler decision IR — six-step migration:
  - `e289520` S1: shared `LogicalServePlan` IR + CPU round-trip tests
  - `c463dd2` S2: Metal aliases the unified IR
  - `dd48e5f` S3: CUDA shadow-mode emit
  - `99fae49` S4: CUDA happy-path lowering (default-on `unified_scheduler` feature)
  - `620eddb` S5: Metal lowering through the IR
  - `a9f0327` S6: retire legacy `ScheduleDecision` enum
  - `b197de4` S7: wins entry + verification gates

  M3 wins
  ([`2026-05-07-m3-unified-scheduler-ir.md`](2026-05-07-m3-unified-scheduler-ir.md))
  honestly notes the **+388 line delta vs the original `-800` deletion
  target** — IR is unified, but the two production scheduler loops are
  not yet collapsed into one shared CPU policy. M3.5 / M4 land that
  delta.

### Phase 4 — Combo plan completed top-to-bottom (commits 8b2c958 → 1e181cc)

Five sub-plans, each with a P0-grounded survey before the plan body:

| Sub-plan | Status | P0 finding |
|---|---|---|
| M_a (`d58e274`) | **landed** | `infer/src/main.rs` already exposes `--spec-enabled / --spec-draft-k / --spec-draft-model {none,self,external:<path>}`; `bench_ab.sh` already a generic A/B harness; only residual was plumbing `spec_acceptance_rate` into `EngineTelemetry`/`/v1/stats`, which `d58e274` does. |
| M_b (`8b2c958` + `38d1c47`) | brief done | `DraftEngine` is a full Qwen3Model not a shmem-resident tiny head; verify is already a single K+1-token prefill launch. M_b split into M_b.1 (tiny-head fusion, requires a new EAGLE-Qwen3 head) and M_b.2 (sparse-self-spec shmem sharing, no train). Recommend M_b.2 first. |
| M_c (`d4c2aa5`) | brief done | `RecurrentState::save_snapshot/restore_snapshot` already implemented for prefix-cache full-hit. M_c reuses the snapshot for spec-verify rollback; ~49 MB / spec step memcpy ≈ 0.14 ms on 4070 Ti SUPER, < 1.5% spec-step overhead. Workscope shrank from 1 week to 3 days. |
| M_d (`a39f414`) | brief done | Q1 (radix pollution repro) is the gating test — confirms-or-rules-out the suspect spec-tentative publish path before scratch-page infra investment. |
| M_e (`1e181cc`) | brief done | 8 workloads × 5 baselines × 4 ARLE configurations = 160 cells; reuses unification M6 matrix skeleton, adds the cross-vendor spec-on/spec-off dimension. |

## Final test state

RTX 4070 Ti SUPER (sm_89), CUDA 13.2, gcc-14 ccbin:

- `cargo test --release --workspace` (CPU): 53 passed
- `cargo test --release -p infer --features cuda --test e2e`: all 4 phases pass
- `INFER_DETERMINISTIC=1 cargo test --release -p infer --features cuda --test greedy_consistency`: B=1 ≡ B=3 byte-exact, matches HF baseline
- `cargo test --release -p infer --features cuda --test spec_decode_correctness`: 4 ok
- `cargo test --release -p infer --no-default-features --features no-cuda scheduler::`: 91 ok
- `cargo clippy -p infer --features cuda -- -D warnings`: clean
- `cargo check -p infer --no-default-features --features metal,no-cuda`: clean

## What we learned

### About the codebase

- **CUDA Graph capture freezes scalar kernel arguments by value.** Any
  scalar that varies per call must be either device-side (read from a
  pointer) or made invariant by passing a static upper bound. The
  TileLang `total_pages` fix is now the project's reference example.
- **`cublasLt` algo selection is M-dependent** at every layer of its
  internal heuristic — autotune cache, `AlgoGetHeuristic`, even the
  fallback `cublasGemmEx`. The only kernel-determinism guarantee across
  batch sizes is "always launch with the same M" (the per-row N=1
  trick).
- **Plan before code, P0 survey before plan.** Three of the four
  combo sub-plans had their scope cut by ≥50% after the P0 file survey
  exposed pre-existing infrastructure (M_a's CLI flags, M_c's snapshot
  API, even bench harness `bench_ab.sh`). Plans without P0 surveys
  consistently over-estimate.
- **Honest line deltas matter.** Codex's M3 wins entry called out the
  `+388` vs `-800` gap rather than hiding it. That signals the next
  milestone (loop collapse) without requiring a separate post-mortem.

### About the workflow

- **Cooperative multi-agent works on a shared `main` branch when
  the contract is explicit.** Claude wrote plans ahead, Codex
  implemented behind, both peer-reviewed via `codex review --uncommitted`
  before push. Conflicts limited to clippy-warning-fix turf; no
  rebase storms.
- **`tmux paste-buffer` of long briefs is the only reliable way** to
  hand Codex a multi-paragraph directive. Inline `tmux send-keys` of
  long input drops keys silently. Documented in the
  `tmux-agent-control` skill.
- **Cron-driven self-pacing keeps the loop moving without human
  ticks.** Each `*/12 * * *` firing of the same prompt prevents the
  manager track from stalling on "what's next?" gaps. The user
  intervened only twice (B=3 fix push, "review 两轮以上" gate) over a
  multi-hour run.

## Rule

- **Peel layered bugs in dependency order**: graph-stable scalars →
  M-keyed algo cache / autotune → cuBLAS internal kernel selection.
  Each layer hides the next. Don't declare a bug fixed until the
  *next* failure mode also resolves.
- **Production fast-path stays untouched in correctness fixes**:
  `INFER_DETERMINISTIC=1` is opt-in; the per-row N=1 path is
  diagnostic / parity-proof only.
- **Plan files include `## P0 finding` block at the top whenever the
  P0 survey changed scope**. Future readers (and future Claude /
  Codex) need that delta context to trust the plan body.
- **31-commit sessions are sustainable** when (a) each commit is
  small + atomic, (b) wins / errors entries land alongside the
  feature commit, and (c) the agent split is stable (one writes
  plans, one writes code, neither tries to do both in the same turn).

## Bench
**pending-remote.** Three remote-host bench runs gated on this
session's commits:
- `2026-05-06-bench-guidellm-cuda-stream-blocking-fix-pending-remote.md`
- `2026-05-07-cuda-greedy-consistency-deterministic-gemm.md` (Codex's bb648fe)
- `2026-05-07-m2-metal-kv-tier-adapter.md` (Codex's f8f063d, Metal-only)

The local 4070 Ti SUPER finished `--quick` runs but stalled at c=8 on
the canonical sweep (separate stuck issue tracked at
[`2026-05-07-m3-guidellm-bench-stuck.md`](../errors/2026-05-07-m3-guidellm-bench-stuck.md)).
H100 / L4 / M3 Max cells of M_e gauntlet remain pending-remote.

## Next manager moves

- **Codex track**: M4 of `backend-unification.md` (Unified `OpsBackend` trait + Metal `crate::ops::*` implementor), Week 5-6 budget. Backend-unification.md §M4 already has the design sketch.
- **Claude track**: M_b.2 implementation (sparse-self-spec shmem sharing) once M_d Q1 (radix pollution repro) lands. The Q1 test guards spec-tentative path correctness; M_b.2 layers fusion on top.
- **Joint**: M_e gauntlet data collection cannot start until M4 + M_b.2 + M_c land — that's still 3-4 weeks out. M_e brief is ready and waiting.
