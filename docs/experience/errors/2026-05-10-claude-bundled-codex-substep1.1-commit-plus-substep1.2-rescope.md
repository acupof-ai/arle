# Title: Claude accidentally bundled codex's Substep 1.1 commit + Substep 1.2 rescope

## Context

Two findings from one tick this session:

1. **Cooperative discipline violation**: my `09ae5a5` commit accidentally bundled codex's Substep 1.1 implementation (`marlin_dequant.h` + `marlin_kernel.cu` mod + wins entry) with my unrelated `docs(research): REVISION — no immediately-actionable -50%+ ITL path` research entry. Wrong commit attribution.

2. **Substep 1.2 atomic_add scope wrong** (caught by codex in the wins entry): my pre-drafted `43bda9c` brief targeted W4A16 marlin_kernel.cu but the `max_par × 64 × n` reduce buffer is on the W4A8 kernel (`marlin_w4a8_kernel.cu`), not W4A16. Substep 1.2 as drafted should be SKIPPED.

## Direct evidence (raw `git show` + `grep` this tick, NOT memory recall per skill v1.10.0 #28)

### Bundling proof

```bash
$ git show --stat --pretty=oneline 09ae5a5
09ae5a5d347bd719c5998086cad7c0d31fd0176c docs(research): REVISION — no immediately-actionable -50%+ ITL path on sm_89 W4 decode (M_spec classical KILLED at 4k)
 crates/cuda-kernels/csrc/gemm/marlin_dequant.h     | 651 +++++++++++++++++++++
 crates/cuda-kernels/csrc/gemm/marlin_kernel.cu     |  22 +-
 ...-05-10-path-b-phase1-substep1.1-dequant-port.md |  61 ++
 ...no-immediate-50pct-itl-path-revised-priority.md | 189 ++++++
 4 files changed, 904 insertions(+), 19 deletions(-)
```

Four files in one commit. My intended commit was the research entry only (1 file). Three other files (codex's WIP) were staged when my `git commit` ran.

### Substep 1.2 rescope proof — codex was right

```bash
$ grep -nE "alloc_zeros|max_par|workspace|lock|atomic" \
    crates/cuda-kernels/csrc/gemm/marlin_kernel.cu | head
145: __device__ inline void barrier_acquire(int* lock, int count) {
158: __device__ inline void barrier_release(int* lock, bool reset = false) {
168: asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
189: int* locks // extra global storage for barrier synchronization
192: // Each threadblock processes one "stripe" of the B matrix...
```

**marlin_kernel.cu (W4A16) uses `int* locks` only — NO `alloc_zeros` or extra reduce buffer.** Reduction is via inter-CTA atomic-add into the output buffer + barrier synchronization.

```bash
$ grep -nE "alloc_zeros|max_par|workspace|lock|reduce" \
    crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu | head
258: int4* __restrict__ C, // int32 global_reduce buffer of shape (max_par*16*4)xn
266: int* locks // extra global storage for barrier synchronization
```

**marlin_w4a8_kernel.cu (W4A8) line 258**: `int32 global_reduce buffer of shape (max_par*16*4)xn`. THIS is the buffer atomic_add could eliminate — but it's the W4A8 kernel path, not W4A16.

## Root Cause

### Bundling violation

Cooperative-worktree git discipline (per `feedback_git_status_before_commit_in_cooperative.md`):

> "git status before commit; git add <path> (NOT -A) limits scope; in cooperative-worktree, NEVER use git commit -a"

I followed `git add <path>` for ONE file, but skipped the `git status --short` BEFORE-commit check. Between my `git add` (which staged 1 file) and my `git commit -m` (which committed all staged files), codex's parallel git process staged its own WIP via what was likely `git add -A` or explicit multi-add. My `git commit` then captured the union of staged files.

### Substep 1.2 scope error

`e59beb5` Phase 1 survey was based on the upstream vLLM marlin_template.h pattern (which has `use_atomic_add` to eliminate a reduce buffer). I assumed the same buffer existed in ARLE's Marlin path. Direct grep this tick shows:

- W4A16 (marlin_kernel.cu): NO reduce buffer to eliminate. Already uses lock-based inter-CTA atomic. atomic_add wouldn't help.
- W4A8 (marlin_w4a8_kernel.cu): Has the int32 global_reduce buffer. atomic_add COULD apply, but only saves prefill TTFT (per `09ae5a5` revised priority, prefill-only FP8 is the better TTFT-axis pickup since codex's P0.A 5.21× evidence is real).

So Substep 1.2 atomic_add as drafted (43bda9c) does not deliver value:
- Doesn't apply to W4A16 (no buffer to eliminate)
- For W4A8 it eliminates a prefill alloc, but TTFT axis better served by prefill-only FP8 directive

## Fix

### Bundling — sediment + apologize

Cannot undo published commit. Going forward:

1. Send PushNotification user with the situation transparency — codex's Substep 1.1 work is correctly landed but commit attribution is mine. Codex's actual contribution: marlin_dequant.h port + W4A16 greedy_consistency PASS verification.
2. Sediment as skill v1.10.0+ anti-pattern candidate (anti-pattern #30): "Commit-time worktree race in cooperative session — `git status` BEFORE commit, not just before add".
3. Update `feedback_git_status_before_commit_in_cooperative.md` (memory file) with the additional rule: "git status BEFORE commit, not just before add. Cooperative process may stage files between your add and commit."

### Substep 1.2 — KILL the pre-drafted brief

`43bda9c` brief is now obsolete. Phase 1 closes at Substep 1.1 only.

Revised Phase 1 outcome (supersedes 43bda9c §2 decision matrix):

| Substep | Status | Outcome |
|---------|--------|---------|
| 1.1 dequant.h port | LANDED (`09ae5a5` accidentally) | W4A16 greedy_consistency PASS, build/fmt/clippy CLEAN |
| 1.2 atomic_add | KILLED in design phase | Doesn't apply to W4A16 (no buffer); W4A8 alternative deferred to prefill-only FP8 directive |
| 1.3 fp32_reduce | DEFERRED with 1.2 | Not on near-term roadmap |

Phase 1 wins entry (codex `09ae5a5` bundle) is the FINAL Phase 1 deliverable. No throughput license claimed (Substep 1.1 was correctness substrate only). Bench A/B for ITL/TTFT impact NOT run because:
- Substep 1.1 is dequant inlining; expected ITL Δ is small (≤ ±2% per FasterTransformer reference equivalence)
- Substep 1.2 is KILLED; no atomic_add comparison to make
- Path B Phase 2 (multi-shape spec) is the next layer if performance is desired; out of current Phase 1 scope

## Rule (sediment for skill v1.10.0+ anti-pattern #30 candidate)

**"Commit-time worktree race in cooperative session — `git status` BEFORE commit, not just before add"**

In a cooperative worktree where multiple agents/processes operate, the staged set at `git commit` time may differ from the staged set at `git add` time. The other process may have staged additional files in the interim.

Mitigation:
1. ALWAYS `git status --short` immediately BEFORE `git commit` (within the same shell turn)
2. If status shows files you didn't intend to commit: `git restore --staged <path>` to unstage, then commit
3. In a cron-loop session, treat the `git add` → `git commit` sequence as atomic from your perspective: fetch fresh `git status` immediately before committing to verify scope

Companion to existing memory rule `feedback_git_status_before_commit_in_cooperative.md` (which already says check status before commit, but was about user dirty paths, not codex parallel staging).

## Cross-references

- The bundled commit: `09ae5a5` (4 files, intended 1)
- Wins entry (codex's actual deliverable): `docs/experience/wins/2026-05-10-path-b-phase1-substep1.1-dequant-port.md`
- Substep 1.2 brief (now obsolete): `docs/research/2026-05-10-phase1-substep1.2-atomic-add-brief-pre-draft.md` (43bda9c)
- e59beb5 Phase 1 survey: still valid for Substep 1.1 (verbatim port intent), now-obsolete for 1.2 atomic_add framing
- `61c9666` revised priority + `09ae5a5` further revision (Phase 1.2 KILL added)
- Memory rule: `feedback_git_status_before_commit_in_cooperative.md`
- Skill v1.10.0 anti-pattern #28: source rule that catches both findings here

## Status

Both findings sedimented. Phase 1 closes at Substep 1.1 (correctness substrate landed via `09ae5a5` bundle). Substep 1.2 KILLED. Next pickup per `09ae5a5` revised priority: prefill-only FP8 directive (P1, codex P0.A 5.21× evidence) OR #34 CLI surface (P2, ~30-50 LOC). Per skill v1.10.0 #28: every claim verified by raw `git show`/`grep` output, NOT memory recall.
