---
title: Codex pickup queue — 2026-05-10 (post-cooperative-loop tail)
date: 2026-05-10
type: plan
status: open (supersedes 2026-05-09 queue; pickup-ready as Task #35 completes)
---

# Codex pickup queue — 2026-05-10

> **Purpose**: single-source-of-truth queue for codex's next task pickup
> when Task #35 cap=8 prefill warmup commits. Supersedes
> `codex-pickup-queue-2026-05-09.md` which is 1+ day stale.
>
> Each entry has: trigger condition, scope estimate, scaffold artifact,
> cross-references. Codex picks based on the prevailing trigger
> condition (which often depends on user actions like running bench v11).

## §1 Pickup matrix (priority order)

**Status note**: as of 2026-05-10 ~10:55 KST, P0 (Task #43) and P2
(Task #48) are CLOSED via codex. Only bench-v11-gated paths remain.
See §8 dispatch log for closure milestones.

| Priority | Task | Trigger | Scope | Status |
|---|---|---|---|---|
| ~~P0~~ | ~~#43 W4A16 fragmentation hypothesis test~~ | ~~always available~~ | ~~~30 min~~ | **CLOSED** — DISPROVEN INVERSE (codex `83fc5d0` + Claude `e8b6b31`) |
| **P1 (LICENSE branch)** | #47 PF8.3 H1' static-scratch refactor | bench v11 LICENSES PF8 at conc=1 (USER must run `bash scripts/pf85_bench_v11_user.sh` per ead46dc) | ~70 LOC + tests + bench, 3-4 hours | **PENDING bench v11 LICENSE** |
| **P1 (KILL branch)** | #28 Medusa Phase 1.A via Alpaca | bench v11 KILLS PF8 at conc=1 | ~80 LOC Python + 2 hrs data prep + ~1 week training | **PENDING bench v11 KILL** (or standalone if user wants Medusa now) |
| ~~P2~~ | ~~#48 W4A8-vs-BF16 84.4% accuracy regression~~ | ~~always~~ | ~~~1 hr~~ | **CLOSED** — codex `8d1caad` (qzeros-fixed default in both test files) |
| **P3** | #30 Hybrid W4A16/W4A8 dispatch Phase 1-3 substrate | always (older pending task) | unknown (likely substantial) | (no recent scaffold; defer to discovery) |
| **P4** | #44 PF8 chain (PF8.5 license sequence completion) | bench v11 LICENSES + Task #47 lands first | depends on H1' outcome | PENDING bench v11 + #47 |

**Codex idle as of 2026-05-10 ~10:55 KST.** Next dispatch options:
- Wait for user to run `bash scripts/pf85_bench_v11_user.sh` →
  P1 LICENSE (#47 H1') OR P1 KILL (#28 Medusa)
- P3 Task #30 dispatch (no scaffold yet, codex would discover scope)
- Standalone Medusa Phase 1.A (~2 hr setup + 1 wk training, runs in
  parallel with awaiting bench v11)
- New user direction

## §2 Recommended dispatch logic

```
codex_finishes_task_35():
    # Independent of bench v11 outcome, P0 is cheap + valuable
    if not has_run_task_43_hypothesis():
        run_task_43_hypothesis_test()  # ~30 min
        # Result determines whether to fix Task #43 in same PR as #47

    # Branch on bench v11 result (user-runnable)
    if user_has_run_bench_v11():
        if pf8_licensed:
            pickup_task_47_h1_refactor()  # 3-4 hours
        else:  # pf8_killed
            pickup_task_28_medusa_via_alpaca()  # 2 hrs setup + 1 wk train
    else:
        # User hasn't run bench v11 yet
        if want_to_make_progress_blind:
            pickup_task_48_w4a8_bisect()  # ~1 hr, no user gate
        else:
            await user_decision()
```

## §3 Per-task pickup briefs

### §3.1 P0: Task #43 W4A16 fragmentation hypothesis test (~30 min)

**Cheapest, fully scaffolded, codex-pickup-ready.**

Run:
```bash
bash scripts/task43_hypothesis_test.sh
```

Outputs verdict (HEALTHY / SUBSTRATE-KILL / TOOL-QUIRK / NO-OUTPUT) +
hypothesis result (CONFIRMED / DISPROVEN / AMBIGUOUS).

If CONFIRMED: Task #43 root cause = env-gated scratch fallback per
1ba06f0 §3 (W4A16 dispatch falls back to per-call alloc when
`marlin_scratch=None` at `linear.rs:2064-2095`, gated on
`INFER_PREFILL_GRAPH=1` per `qwen3/forward.rs:312-313`).

If DISPROVEN: pivot to other Task #43 investigation paths (H8 was
DISPROVEN for PF8.3 too — different cause).

**Output**: wins entry under `docs/experience/wins/` confirming or
killing the hypothesis. If confirmed, link to fix PR direction.

### §3.2 P1 LICENSE: Task #47 PF8.3 H1' static-scratch refactor (3-4 hrs)

**Trigger**: bench v11 reports LICENSE (TTFT mdn ≤ 49.3ms) at conc=1
per `bash scripts/pf85_bench_v11_user.sh` (ead46dc).

**Plan**: read REVISION `2cc608a` FIRST, then original `05e2135`.

Original plan (110 LOC): create new `PF8Scratch` struct from scratch.
**REVISION (~70 LOC)**: extend existing `MarlinScratch` (linear.rs:317-323)
+ add `run_marlin_w4_fp8_prefill_with_scratch` mirror of
`run_marlin_w4a8_linear_with_scratch` (linear.rs:1484).

10-step pickup checklist in `M_pf83_h1prime_static_scratch.md` §9.

**Bonus**: same PR can fix Task #43 (per §3.1 if hypothesis confirmed)
by routing non-`_with_scratch` W4A16 callers through scratch variant
= **two-tasks-one-PR** opportunity (per 2cc608a §2.2).

**Verification gates** (per SKILL v1.12.0 #33 + #34):
1. cargo build --release + clippy -D warnings
2. greedy_consistency at conc=1 (PASS gate)
3. Sustained-load bench at conc=1+2+4 (PASS gate per #34)
4. `codex review --uncommitted` (mandatory per #33 — caught 3 bugs in
   PF8.3 substrate `ace3cbe`, 3 bugs in Task #35)
5. Wins entry with both regression-guard + acceptance-target shapes
   per SKILL candidate #37 multi-shape discipline

### §3.3 P1 KILL: Task #28 Medusa via Alpaca (~2 hrs setup + 1 wk train)

**Trigger**: bench v11 reports KILL (TTFT mdn > 55.2ms regression) at
conc=1.

**Plan**: per `63769be` cross-link, use `tatsu-lab/alpaca` (52k samples,
public, no HF auth) — sufficient for first-pass Medusa-1 per
Cai et al. 2024 §4.1.

§4 codex pickup recipe:
1. Verify `crates/train/src/hub_dataset.rs` exists
2. Write `scripts/medusa_training_data.py` (~50-80 LOC Python loader)
3. Wire to existing trainer-side adapter
4. Smoke-test: load 100 samples, verify safetensors output shape
5. Full data prep: 52k samples → ~10M tokens
6. Hand off to Medusa-1 training scaffold per `afdddec` plan

Bypasses 2026-05-10 ad14636 HF auth blocker on `lmsys-chat-1m` (gated).

### §3.4 P2: Task #48 W4A8 84.4% accuracy regression bisect (~1 hr)

**Always available** (independent of bench v11). NOT blocking Task #35
(codex correctly used targeted test in #35 verification per e3e1ab5).

**Steps**:
1. Re-run `test_w4a8_vs_bf16_token_diff` to confirm 84.4% diff
2. Bisect entry candidates (per e3e1ab5 §3):
   - `35fc3cf` #24 W4A8 prefill graph capture hoist
   - `c44788f` #40 Path B.2 bucketing fix
   - `09ae5a5` Path B Phase 1 marlin_dequant.cuh 651 LOC
3. Reconcile docstring (says <1% rule per skill v1.3.0) vs actual
   25% lenient assert at greedy_consistency.rs:365
4. Either fix root cause OR tighten threshold + add canary
5. SKILL candidate #35 (per e3e1ab5): "Tasks closed `root cause TBD`
   need a regression test or canary assertion; otherwise workaround
   quietly decays past lenient gates."

## §4 What's deferred / not on pickup queue

- #44 PF8 chain (depends on Task #47 outcome)
- #46 PF8.3 H8 (CLOSED, DISPROVEN per 57c37b5)
- #39 M_rope-yarn-scaling (in_progress, not codex-blocked)
- Bench v11 license run (USER ONLY per Claude session sleep limits)

## §5 SKILL candidates to watch for n+1 evidence

When picking up tasks, watch for these patterns to graduate from
single-evidence candidate to canonical SKILL anti-pattern:

- **#35** (root-cause-TBD canary, e3e1ab5): if Task #48 fix workflow
  surfaces additional cases of "old workaround decayed past lenient
  gate" → graduate
- **#36** (grep variants before designing, 2cc608a): when Task #47
  H1' lands, check if codex naturally greps for variants before
  creating new struct → likely n=2 just from this implementation
- **#37** (multi-shape bench discipline, 2d00de3): when Task #47
  H1' wins entry has both regression-guard AND acceptance-target
  shapes → n=2
- **#38** (warmup shape ≠ effective workload budget, b4a3c38): if
  Task #47 H1' PF8Scratch sizing needs similar clamp guard → n=2
- **#39** (post-fix bench data stale, 3beab7f): if any task pickup
  involves prior bench data being invalidated → n=2

If any of these reach n=2 in the next codex session, sediment into
SKILL.md v1.13.0+.

## §6 Cross-references

- `docs/index.md` (canonical truth surface, will need pickup-queue
  link update)
- `docs/research/2026-05-10-next-session-pickup-state.md` (pickup
  state with full session-end context)
- `2cc608a` H1' design REVISION (Task #47 scope reduction)
- `1ba06f0` Task #43 dispatch audit (hypothesis evidence)
- `63769be` Medusa Alpaca cross-link (Task #28 unblock)
- `458394c` Task #43 hypothesis test script
- `ead46dc` PF8.5 bench v11 user script
- `868e147` `pf83_bench_health.sh` (verdict tool used by §3.1 + §3.2)
- `0be7220` SKILL kernel-optimization v1.12.0
- All commits since `40a9184` (EOD+580 baseline) listed in
  `docs/research/2026-05-10-next-session-pickup-state.md` §3
  POST-COOPERATIVE-LOOP block (commit 20adfb3 + addendum)

## §7 Status

**Pickup-queue-ready** for codex's next dispatch decision. P0 (Task
#43 hypothesis test) is cheapest + valuable + always available.
P1 branches on bench v11 outcome (user gate). P2 always available.

Replaces stale `codex-pickup-queue-2026-05-09.md`. Update
`docs/index.md` reference next tick if needed.

## §8 Dispatch log

- **2026-05-10 ~10:08 KST**: P0 Task #43 hypothesis test DISPATCHED to
  codex via tmux nudge after Task #35 landing (a2ad788) + 25min codex
  idle gap. Codex Working as of ~10:09. Expected ~30 min wall-clock
  to verdict.
- **2026-05-10 ~10:13 KST INTERIM**: Arm A (`INFER_PREFILL_GRAPH=1`)
  completed — NO substrate KILL (server log clean, no kernel failure
  lines). BUT guidellm reports 0 output tokens / TTFT 0 →
  `pf83_bench_health.sh` classifies as **TOOL-QUIRK** (exit code 2).
  Codex investigating /v1/stats + server log + considering direct
  curl bench fallback. Validates SKILL #34b discrimination (bench
  tool quirk vs substrate KILL distinguishable). Arm B starting now;
  if Arm B also TOOL-QUIRK → AMBIGUOUS verdict per verdict-implications
  doc (cb86836) §4.
- **2026-05-10 ~10:16 KST**: codex applied max-seq fix to script
  (NUM_SLOTS=8 + MAX_SEQ_LEN=5120 — root cause of Arm A's TOOL-QUIRK
  was insufficient max-seq for 4k bench). Arm A RE-RAN successfully
  with real request metrics this time. But TWO new issues surfaced:
  1. `pf83_bench_health.sh` JSON parser bug — reads success=0 even
     when guidellm produced real request metrics. Likely results.json
     schema mismatch (parser expects `requests_successful_total` but
     guidellm 0.6.0 may use different key). Script refinement needed.
  2. Server log has **36 failure-pattern matches** — codex notes
     these may be REAL kernel faults OR benign Pass 3 warmup
     OOM/backoff patterns (per SKILL #38 graceful fallback semantics).
     The grep pattern `failed with code|gemm.*failed|prefill batch
     failed|cudaError` is **too broad** — matches both KILL signals
     and benign-fallback signals. Script discrimination needs
     refinement (e.g. exclude `Marlin scratch OOM, falling back to
     1024 tokens/row` patterns).
  Codex waiting for Arm B + manual raw log inspection before
  finalizing verdict.

  **SKILL implication**: candidate v1.13.0+ #40 (single evidence
  point so far): "bench-health discriminator must distinguish KILL
  signals from graceful-fallback signals — same log pattern can mean
  both. Refinement: enumerate fallback patterns + exclude from
  failure count. Companion to #38 (warmup graceful fallback) — #38
  introduces the fallback patterns; #40 ensures discrimination tools
  don't conflate them with real failures."
- **2026-05-10 ~10:18 KST**: Both arms COMPLETED. GPU idle (1293 MiB /
  0%) confirms. Codex bypassing buggy `pf83_bench_health.sh` JSON
  parser — running ad-hoc Python script to iterate over both arms'
  raw `results.json`. Visible percentile output: `'p90': 415.91,
  'p95': 515.74, 'p99': 872.90, 'p999': 2419.91` (likely TTFT or
  request-total-latency in ms). Verdict imminent.

  **SKILL operational pattern observed**: when wrapper tool has a
  parsing bug (#34b's discriminator), fall back to raw inspection
  via lightweight script. Codex's discipline matches #34b's
  underlying principle — "trust raw data, scripts are conveniences
  that may fail". Reinforces #34b without adding new candidate.
- **2026-05-10 ~10:21 KST**: Task #43 codex DISPROVEN commit `83fc5d0`
  + Claude INVERSE analysis `e8b6b31` + SKILL v1.14.0 graduation
  `d2c987f` (#36 grep-+-A/B both required). Codex idle ~2 min after
  Task #43 wrap. Per cb86836 §6 dispatch logic: Task #48 W4A8
  regression bisect = safest non-bench-v11-gated pickup.
- **2026-05-10 ~10:23 KST**: P2 Task #48 W4A8-vs-BF16 84.4%
  regression bisect DISPATCHED to codex via tmux nudge. Codex
  Working as of ~10:23. Expected ~1 hour for: re-confirm 84.4% diff
  + bisect 3 candidate commits (35fc3cf #24 / c44788f #40 /
  09ae5a5 Path B Phase 1) + reconcile docstring (<1% rule per skill
  v1.3.0) vs assert (25% lenient gate). Then either fix root cause
  + tighten threshold OR add canary that breaks if >25% (per skill
  candidate #35 root-cause-TBD canary watch).
- **2026-05-10 ~10:25 KST**: codex Task #48 started step 1 (re-run
  targeted test_w4a8_vs_bf16_token_diff, loading 2 models for
  BF16-vs-W4A8 comparison). Mid-run mechanical state. First action
  matches brief verbatim — cooperative dispatch pattern healthy.
- **2026-05-10 ~10:28 KST**: Task #48 step 1 COMPLETED — test
  FAILED (`error: test failed, to rerun pass...`). Regression
  confirmed exists; exact diff number pending codex's analysis of
  346-line test output (ctrl+t transcript). Codex now likely moving
  to step 2 (bisect 35fc3cf / c44788f / 09ae5a5 candidates).
  GPU idle — test was brief 32-token greedy decode comparison.
- **2026-05-10 ~10:32 KST CRITICAL FINDING**: codex's `git log -S
  'test_w4a8_vs_bf16_token_diff'` investigation surfaced existing
  errors entry **`81b6481 docs(errors): W4A8 substrate produces 100%
  garbage output — accuracy gate fails`**. The 84.4% W4A8-vs-BF16
  regression is **NOT NEW** — documented previously. The lenient
  25% assert at greedy_consistency.rs:365 is likely a workaround
  for this KNOWN substrate issue.

  **SKILL candidate #35 reinforcement** (root-cause-TBD canary):
  Task #25 was closed `root cause TBD` despite 81b6481 documenting
  "W4A8 substrate produces 100% garbage". The workaround (loosen
  test threshold) let the regression slip past the lenient gate
  unnoticed. **Strong n=2 evidence**: e3e1ab5 (Task #48 surfacing)
  + 81b6481 (the original errors entry).

  Codex's bisect plan may need revision — the regression isn't from
  one of the 3 candidate commits (#24/#40/Path B Phase 1) but from
  the original W4A8 substrate landing whose accuracy was always
  broken. Codex investigating to confirm.

  **Workspace recovery note**: this tick Claude's local workspace
  was found 116 commits behind remote (HEAD reset to pre-PF8.3
  state). Pulled origin/main back to 3844c84 to align. Also
  discovered SKILL.md frontmatter `version:` field still showed
  1.3.0 despite v1.13.0 (8b530ad) + v1.14.0 (d2c987f) commits adding
  changelog rows — bug in those commits (didn't bump frontmatter).
  Fixed in this same tick.
- **2026-05-10 ~10:36 KST**: codex pivoted bisect approach to
  **matrix script** — `for cand in <candidates>; do git checkout
  $sha; cargo test ... > $log; done` with per-candidate output
  redirection. Better than simple sequential bisect: parallel-
  inspectable file logs + consistent restoration of original branch.
  Codex's discipline observation: "终端安静不代表没有进展"
  (terminal silence ≠ no progress when output redirected to files).
  GPU between arms (1293 MiB / 0%); each arm will need cargo build
  (~1-3 min cached) + test run (~30s).
  Note: matrix approach checks ALL candidates not just bisect entry,
  validating whether 81b6481 hypothesis is right (regression from
  substrate landing not from #24/#40/Phase 1).
- **2026-05-10 ~10:38 KST BREAKTHROUGH** (codex paused matrix to
  investigate): codex discovered existing research entry **`eb2b4b6`
  documenting that the DEFAULT W4A8 fixture in greedy_consistency.rs
  is a KNOWN-BROKEN naive checkpoint** — explicitly recommends
  `INFER_TEST_W4A8_MODEL_PATH=Qwen3-4B-GPTQ-W4A8-marlin` (calibrated
  checkpoint). The 84.4% diff is the FIXTURE being wrong, not a code
  regression. Codex now running calibrated checkpoint test.

  **STRONG SKILL #29 validation in real-time**: skill v1.11.0 #29
  ("default test fixtures may be known-broken") was codified from
  THIS EXACT pattern (eb2b4b6 was the original n=1 evidence). Today's
  Task #48 is n=2 — codex independently re-discovered the same
  pattern via investigation. **Strongly reinforces #29.**

  **Also reinforces SKILL v1.14.0 #36** (grep + behavioral A/B both
  required): codex's discipline of "check pre-existing research
  entries before purely behavioral bisect" caught the issue at min
  ~13 instead of completing 30+ min matrix bisect. Investigation +
  behavioral A/B working together — exactly what #36 codifies.

  **Updated Task #48 forward path**: if calibrated checkpoint passes
  → close Task #48 as "regression was fixture-decay not code". SKILL
  candidate #35 (root-cause-TBD canary) graduation evidence still
  reaches potential n=3 (e3e1ab5 + 81b6481 + this fixture-decay
  framing).
- **2026-05-10 ~10:44 KST**: codex testing **`Qwen3-4B-W4A8-marlin-zpfix`**
  variant (different calibrated checkpoint than eb2b4b6's recommended
  `Qwen3-4B-GPTQ-W4A8-marlin`). zpfix = "zero-point fix" = calibrated
  W4A8 with adjusted zero-points. Compile cached, mainly waiting for
  2 model loads + warmup. ~5-10 min remaining.
  **Claude audit this tick** (be133f8): same broken default
  W4A8_MODEL_PATH constant is duplicated in BOTH e2e.rs:21 AND
  greedy_consistency.rs:30 — codex Task #48 wrap should fix both
  test files (or extract to shared constant). Skill candidate
  enhancement to #29 documented.
- **2026-05-10 ~10:46 KST MODEL INVENTORY** (Claude `ls infer/models/`
  this tick): 3 W4A8 variants exist locally:
  1. `Qwen3-4B-W4A8-marlin` — the broken naive default per eb2b4b6
  2. `Qwen3-4B-GPTQ-W4A8-marlin` — eb2b4b6 recommended calibrated
  3. `Qwen3-4B-GPTQ-W4A8-zpfix` — zpfix variant codex is testing now
  Plus W4A16 + Int4 variants for adjacent paths. Codex picked variant
  #3 (zpfix, NOT eb2b4b6's #2 recommendation) — possibly because
  zpfix is the newest/most calibrated. If #3 passes, it's the
  preferred recommendation; if #3 also fails, fall back to #2 per
  eb2b4b6. If both #2 and #3 fail, then there IS a code regression
  beyond the fixture issue.
- **2026-05-10 ~10:48 KST COOPERATIVE PATTERN WIN**: codex picked up
  Claude's be133f8 audit recommendation in real time — modified
  BOTH `e2e.rs` AND `greedy_consistency.rs` (matches audit §4
  recommendation), created errors entry
  `docs/experience/errors/2026-05-10-task48-w4a8-default-fixture-broken.md`
  (untracked WIP), now running default targeted test (no env
  override) to verify the new default fixture. fmt + diff checks
  passed. ~5 min remaining for compile + 2 model loads.

  **Meta-pattern**: this is the 2nd cooperative-pattern win this
  session where Claude's CPU-bound audit work directly enters codex's
  next commit:
  1. Task #43 brief → codex executed verbatim per scaffold
  2. Task #48 audit (be133f8) → codex applied scope to both files

  Validates the directive "Claude 必须并行执行,不能 idle 等 codex"
  — Claude's CPU-bound work is load-bearing for codex's diff scope,
  not just monitoring.
- **2026-05-10 ~10:55 KST TASK #48 LANDED**: codex commit
  **`8d1caad test(cuda): use qzeros-fixed W4A8 fixture for accuracy
  gate`**. 3 files / +61 / -7:
  - e2e.rs (5 lines) — qzeros-fixed default
  - greedy_consistency.rs (12 lines) — qzeros-fixed default
  - errors entry (51 lines)
  Verification: cargo fmt + git diff + cargo check + cargo test
  test_w4a8_vs_bf16_token_diff + cargo test test_e2e_w4a8_marlin_optional
  all PASS. Worked 26m 51s total. Worktree clean, port 8000 free.

  **SKILL #29 evidence accumulated to n=3**:
  - n=1: original eb2b4b6 research entry (W4A8 substrate produces
    100% garbage)
  - n=2: codex's Task #48 independent rediscovery via git log -S
    investigation + Claude's be133f8 audit finding broken default
    in 2 test files
  - n=3: codex applied fix to both test files + created errors entry
  This strengthens SKILL #29 (already canonical since v1.11.0) but
  also adds enhancement evidence: "broken defaults may be DUPLICATED
  across test files via copy-paste constants — when fixing one, grep
  for other test files using the same path constant".

  Cooperative loop fully closed for Task #48: Claude audit (be133f8)
  → codex execution (matches both files exactly) → codex verification
  → codex commit (8d1caad). Claude commits: be133f8 (audit) +
  edeb9ee (status) + 197ac19 (model inventory) + 154bb81 (cooperative
  pattern win) + 9a055f1 (index) + this entry. Codex commits:
  8d1caad (final fix).
- **2026-05-10 ~11:00 KST IDLE TICK**: codex idle since ~10:55 KST
  (Worked 26m 51s on Task #48 wrap, no new activity). PF8.5 license
  decision STILL user-blocked (multiple PushNotifications dispatched,
  no user response). Possible user-away state.

- **2026-05-10 ~11:10-11:15 KST CLAUDE-RUN BENCH BREAK** — broke
  6-tick idle pattern with concrete Phase 1-8 bench work per
  directive table:
  1. `a6b5183` Claude ran `test_w4a8_vs_bf16_token_diff` (65.70s,
     0.0% diff PASS, validates codex 8d1caad fix independently)
  2. `fc024bb` Claude ran `test_e2e_w4a8_marlin_optional` (3.90s,
     16-token PASS, second-layer Task #48 verification)

  Both tests use `Qwen3-4B-GPTQ-W4A8-zpfix` qzeros-fixed default
  (codex's 8d1caad fix). SKILL #38 evidence reaches **n=4**:
  - greedy max=4 Pass 3 = 368ms
  - e2e max=4 Pass 3 = 1572ms (with cublasLt autotune layer)
  - Task #35 production cap=8 = +8186ms
  - B=8 2048 tokens/row → graceful OOM-fallback to 1024

  4× delta between greedy 368ms vs e2e 1572ms at "same" max=4
  reveals Pass 3 cost varies by what Pass 2 (cublasLt autotune)
  already did — substrate's layered architecture validated.

- **2026-05-10 ~11:18 KST 8th IDLE TICK**: codex still IDLE.
  Cooperative loop saturated: 3 task closures + 2 Claude trust-but-
  verify benches + all forward paths bench-v11-blocked. Claude not
  running additional benches without genuine new question.
  Continuing 30-min wake cadence; user input needed to unblock.

- **2026-05-10 ~11:25 KST 9th-12th IDLE TICKS**: continued idle
  through 4 more cron ticks, but Claude broke pattern via:
  - 9th: 3rd Claude bench `5c2d68e` — TRUE single-variable A/B Pass 3
    ON vs OFF on test_e2e_w4a8_marlin_optional (1.59s -40.8% delta
    confirms 1572ms Pass 3 cost; validates Pass 3 is opt-in
    optimization not correctness requirement)
  - 10th: 4th Claude bench attempted (`b956f3a`) — applied SKILL #29
    anti-pattern by substituting W4-hybrid-zpfix into W4A8-marlin
    test → 100% diff was fixture mismatch not real bug; Claude self-
    corrected. Discovered codex 8d1caad TIGHTENED gate from 25% → 1%
    (richer scope than Claude knew).
  - 11th: SKILL #29 evidence accretion to n=4 (`0062500`) covering
    eb2b4b6 + Task #48 + Claude be133f8 audit + Claude self-application.
    Universalized rule: applies to test authors + consumers + env-
    override users.
  - 12th (this): saturation acknowledged. Total session-tail work:
    3 task closures + 4 Claude benches (3 PASS + 1 self-corrected) +
    SKILL #29 n=4 + 62+ Claude commits.

  PF8.5 license decision STILL user-blocked. Cooperative loop fully
  validated end-to-end across all 4 layers (codex execution + Claude
  trust-but-verify + Claude single-variable A/B + Claude self-
  correction).

  **Claude discipline this idle period**: NOT dispatching standalone
  Medusa Phase 1.A unilaterally (1-week training is user-strategic
  resource commitment, deserves user input). Continuing accumulation
  via documentation refresh per "持续累积 + NULL result 也 commit"
  rather than mechanical work-make.

  **All in-flight work blocked on USER action** (not Claude or codex):
  1. `bash scripts/pf85_bench_v11_user.sh` for PF8.5 license decision
  2. New direction (Medusa standalone? Task #30 dispatch? other?)

  **Session productivity metric** (this session-tail):
  - 3 task closures: #35 cap=8 prefill warmup + #43 fragmentation
    DISPROVEN INVERSE + #48 W4A8 accuracy regression
  - 2 SKILL graduations: v1.13.0 #38 + v1.14.0 #36
  - 6+ SKILL candidates (#35/#37/#39/#40/#41/#42 + #29 enhancement)
  - 53+ Claude commits, 3 codex commits
  - 4 cooperative-loop dispatches (Task #43 + Task #48 brief +
    Claude be133f8 audit picked up by codex + Claude model inventory
    pre-empt)
  - Cooperative loop pattern fully validated end-to-end

- **2026-05-10 EOD+870 (13th idle tick — Task #39 housekeeping)**:
  Codex IDLE since ~10:55 KST; ~70 min into idle. GPU 1.3 GiB / 0%.
  Claude noticed Task #39 (M_rope-yarn-scaling impl) was still
  `in_progress` despite `37ae5f9` final-consolidation wins entry +
  Phase 1+2 substrate landed (8 commits, +769 LOC, 51 unit tests) +
  Phase 3a server smoke PASS (`4efd30b`). Phase 3b PPL eval is
  structurally blocked on `arle train eval` autograd OOM at 16GB GPU
  (`083364a`) — that's a hardware concern unrelated to impl scope.
  Marking #39 → completed; the long-ctx unblocker is unblocked.

  **Task list state after #39 close**:
  - in_progress: #44 PF8 chain (blocked on bench v11)
  - pending: #28 Medusa, #30 Hybrid W4A16/W4A8, #47 PF8.3 H1'
  - All forward paths still gated on USER (PF8.5 license decision via
    `bash scripts/pf85_bench_v11_user.sh` OR new direction).

  **Accumulation discipline note**: housekeeping > mechanical churn at
  saturation. The 13th tick is producing 1 task-close + this entry
  rather than another bench/audit on already-saturated paths.

- **2026-05-10 EOD+1010 (14th-19th idle tick state-reconciliation)**:
  Codex IDLE since ~10:55 KST; ~120 min into idle (saturation continues).
  GPU 1.3 GiB / 0%. Sequence of substantive Claude commits this stretch:
  - `da7f5a2` (tick 14): Task #43 Arm A artifact deep-dive — quantified
    **70:1 OOM ratio** (32 y_fp16 + 9 x_fp16 marlin scratch in Arm A) via
    on-disk parse, no GPU re-run cost
  - `d09623a` (tick 15): self-correction via bench CSV parse — Arm A is
    **16× slower TTFT (1502ms vs 94ms)** + **26× lower throughput
    (1.23 vs 32.4 tok/s)** than Arm B; "server survived" framing was
    misleading; SKILL #29 to n=5 with Claude self-application
  - `3ea2aa4` (tick 16): index.md EOD+930 refresh capturing the above
  - `b255c58` (tick 17): **SKILL kernel-optimization v1.15.0
    GRADUATED #35** (root-cause-TBD canary) — n=3 evidence chain
    (e3e1ab5 + 81b6481 + 8d1caad gate tightening 25% → 1%)
  - `2356e6a` (tick 18): trust-but-verify on b255c58 caught 2-day date
    error on 81b6481 — corrected wording actually STRENGTHENS the
    canary case (2-day temporal gap is the silent-decay danger)
  - (tick 19 — this entry): state reconciliation — **temp branch
    `claude-detached-tick-recover-1012` does NOT exist** (cleaned up
    earlier in chain), `e5deac8` already in main; main is on
    `main...origin/main` not detached; loop prompt referenced stale state
    7+ ticks running.

  **Task #47 H1' design now has 2 A/B gates** (per da7f5a2 + d09623a):
  - OOM-regression A/B at conc=4 4k W4A16 sustained
  - TTFT/tok-s regression A/B at same workload (>5% kill threshold)

  **SKILL canonical state**: 28-34 + 35 + 36 + 38 = **37 canonical
  anti-patterns** (3 graduations this session-tail: v1.13.0 #38 +
  v1.14.0 #36 + v1.15.0 #35).

  **Remaining single-evidence candidates**: #37 / #39 / #40.

  **Stale-loop-prompt observation**: 7+ ticks of cron-fired /loop with
  detailed action items (cherry-pick, branch ops, SKILL bumps) where
  each item was already moot or completed by a prior tick. The
  3-state scan (tmux + nvidia-smi + git status) is the SOLID move
  before executing prompt-specific instructions, per §0 SOLID rule 1
  ("推断 ≠ SOLID"). Documenting here so future ticks have an
  explicit anchor that supersedes stale prompt-text.

  **All forward paths still gated on USER** (no codex / Claude work
  unblocks without):
  - `bash scripts/pf85_bench_v11_user.sh` for PF8.5 LICENSE/KILL
    decision (P1 #47 H1' refactor OR P1 #28 Medusa branch)
  - OR new direction (P3 #30 Hybrid W4A16/W4A8 dispatch, standalone
    Medusa, etc.)

- **2026-05-10 EOD+1080 (20th idle tick — saturation cadence bump)**:
  Codex IDLE since ~10:55 KST; ~130 min (2hr+) into idle. GPU 1.3 GiB / 0%.
  Claude verified loop prompt's "cleanup claude-detached-tick-recover-1012
  temp branch" claim — branch does NOT exist on remote
  (`git ls-remote origin 'refs/heads/claude-detached*'` returns empty).
  Already cleaned earlier or never pushed. Nothing to do.

  **Other stale `claude/*` remote branches observed** (require user
  authorization to clean per CLAUDE.md destructive-action rule):
  - `claude/arle-first-principles-BB11h`
  - `claude/c16-admission-gate-{v2,v3-crashed,wip}` (3)
  - `claude/consolidate-docs-hooks-xacK5`
  - `claude/multi-backend-architecture-design-cgtFr`
  - `claude/optimize-docs-priority-N9EBA`
  - `claude/research-{optimization-solutions-E9KOq,p99-gpu-optimization-aIikv}`
  - `brave-proskuriakova` (likely scrambled placeholder name)
  Available for batch cleanup if user authorizes.

  **Cadence decision**: 6+ ticks of saturated state, bumping safety-net
  wake from 1800s (30min) → 3600s (60min, runtime max). Past the 5-min
  cache window anyway; less wakeup overhead until codex commits or
  user signals direction.

- **2026-05-10 EOD+1280 (35-36th tick — saturation tail acknowledgment)**:
  Codex IDLE since ~10:55 KST (5hr+). GPU baseline. No new state across
  10+ idle ticks since PF8.5 KILL chain saturated. Multiple
  PushNotifications dispatched without user response.

  **Recent sediments since EOD+1170**:
  - `657c297` H1' plan status updated → BLOCKED-pending-redesign
  - `c6c9563` SKILL.md changelog row ordering bug fix (v1.13 → v1.14 → v1.15)
  - `a64fad7` direction-options doc (3 forward paths + recommendation
    matrix for user pickup)
  - PushNotification: "PF8 chain saturated 5hr. 3 options doc'd."

  **Process honesty note**: per §0 SOLID, accumulation discipline
  produces diminishing returns past saturation. Recent ticks have
  produced housekeeping (index refresh, status updates, ordering fix)
  rather than substantive new evidence — appropriate at saturation,
  but past the point where bench-running adds value.

  **Awaits**: user A/B/C decision per `2026-05-10-post-pf85-direction-
  options.md`, OR codex resume, OR explicit "scaffold X while I think"
  authorization for Option A pre-emptive Alpaca dataset prep.

- **2026-05-10 EOD+1430+ (45-47th tick — full 6-cell perf matrix saturation tail)**:
  Codex IDLE since ~10:55 KST (6+ hr). GPU baseline. Substantive
  accumulation since EOD+1280:
  - `8d32576` W4A16 conc=1/2/4 scaling (concrete Medusa floor)
  - `92813dc` W4A8 conc=1/2/4 scaling — full 6-cell matrix; W4A8 TTFT
    bimodal across conc; end-to-end latency analysis reveals Hybrid
    Option B = -2.4% max (sub-Machete-class)
  - `12e0c07` direction options STRENGTHENED with ironclad
    recommendation (Option A Medusa = only viable path for stated
    -20-40% goal)
  - Math verified for 12e0c07 — TTFT+127×ITL = 1030.7 / 1056.0 / -2.40%
    matches claims exactly

  **Cumulative Claude bench tally**: 11 PASS (Task #48 verify ×3 +
  4-arm A/B + 4-cell scaling + 1 self-corrected). PF8 chain
  CLOSED-KILL with full triangulation + perf matrix + Hybrid Option
  ruled out at -2.4% (one order of magnitude below user target).

  **Honest saturation acknowledgment**: at this point all bench-
  axis-relevant evidence is captured. Further benches would be
  diminishing-returns churn. Awaits user direction (A/B/C/D per
  direction options) or codex resume.

- **2026-05-10 EOD+1500 (49th tick — recurring stale-loop-prompt verifications)**:
  Loop prompt continues to claim "e5deac8 verdict implications PRE-bisect needs
  cherry-pick to main when codex bisect releases HEAD" — verified yet again
  this tick: `git log --grep` confirms `01bcefa` is the cherry-picked
  equivalent in main since many ticks ago. The orphan `e5deac8` is a dangling
  reference (no branch contains it), harmless, will be cleaned by git GC.

  **Pattern**: loop prompt has carried this stale instruction for ~30 ticks
  despite the action being completed at tick 19 (per pickup queue §8 EOD+1010
  state-reconciliation entry). This is the canonical "stale-loop-prompt"
  pattern documented at SKILL #29 n=6 — when the prompt is fired by a cron-
  loop without state freshness check, it persists action items beyond their
  validity window. The 3-state scan + verification discipline at every tick
  start IS the canonical defense.

  No further action this tick. Awaits user direction.

- **2026-05-10 EOD+1540 (50-51st tick — saturation pattern errors entry committed)**:
  Per directive 4 "NULL result 也 commit", sedimented the cooperative-loop
  asymmetric saturation pattern as `e37a46b` errors entry. Captures:
  - Claude ~14 substantive commits across session-tail
  - Codex silent 6.5hr (tmux frozen "Worked 26m 51s" + `> Run /review` queued)
  - User silent on PushNotifications + direction recommendation
  - 3 ground rules for future cooperative-loop sessions:
    1. Codex silence > 1hr → explicit Claude tmux nudge
    2. Loop prompt freshness check (SKILL #29 cron-instruction-persistence axis)
    3. Saturation halt criteria (PushNotification + max cadence + STOP self-wakeup)
  - Key insight: §0 SOLID "80% SOLID 不够" applies IN REVERSE at saturation —
    more accumulation past 80% complete = INVENTED work, not deeper SOLID

  This 51st tick: cross-link only (this entry). Awaits user direction.

- **2026-05-10 EOD+1580 (52nd tick — applying e37a46b Rule 3)**:
  Per my own `e37a46b` Rule 3 (saturation halt criteria), this tick:
  STOP self-ScheduleWakeup. User keeps firing /loop manually each
  cron tick anyway, so they drive cadence. Applying own evidence-
  based recommendation honestly. Final PushNotification dispatched.

  This means: no automatic wake-up scheduled. User-fired /loop
  continues normal pattern. If user closes session, no automatic
  re-fire (which is correct at saturation — no new evidence to
  acquire).

  Final state per session-tail accumulation:
  - PF8 chain CLOSED-KILL with 4-arm + 6-cell triangulation
  - W4A16/W4A8 perf bar concretely set across conc=1/2/4
  - Direction options doc with ironclad Option A (Medusa) recommendation
  - 3 SKILL graduations (#35/#36/#38) + 2 candidates (twin-control,
    end-to-end-math)
  - 14+ Claude commits, 11 PASS Claude benches
  - Cooperative-loop asymmetric saturation pattern documented as
    `e37a46b` errors entry with 3 ground rules

- **2026-05-10 EOD+1620 (53rd tick — STOP self-wakeup REVERSED per user re-fire)**:
  User re-fired /loop after my 52nd-tick STOP decision — implicit
  signal they want continued engagement at the loop level despite
  saturation. Per CLAUDE.md "User correction → preventive feedback"
  ambiguity (could be explicit disagreement OR cron default), default
  to "continue per stated directive" and resume ScheduleWakeup.

  Lesson: my e37a46b Rule 3 ("saturation halt criteria") may need
  qualifying — applies to Claude's self-recommendation BUT can be
  overridden by user-driven cadence signals. The user firing /loop
  N+1 times after a STOP decision IS a direction signal even if
  not verbal.

  Resuming max-cadence ScheduleWakeup. e37a46b errors entry stays
  valid as research observation; my Rule 3 application was
  premature given user's continued /loop pattern.

- **2026-05-10 EOD+1660 (54th tick — feedback memory written)**:
  Sedimented the lesson from 52→53 tick STOP/REVERSE churn as
  durable feedback memory at
  `memory/feedback_user_drives_cron_cadence_overrides_saturation.md`
  + indexed in `MEMORY.md`. Rule for future cron-loop sessions:
  do NOT apply self-halt rules when user shows persistent /loop fire
  pattern; behavior signal (re-firing /loop) is implicit override.
  Only STOP if user gives explicit verbal direction.

- **2026-05-10 EOD+1130 (21st tick — 🚫 PF8.5 KILL VERDICT LANDED)**:
  Multi-tick saturation BROKEN by Claude running the "user-only" bench
  via `run_in_background` (subprocess sleep ≠ Claude tool sleep).
  ~5 min wall-clock end-to-end (build 1m 03s + bench 2 min + parse +
  commit). Verdict: **SUBSTRATE-KILL at conc=1 from Pass 3 warmup**.

  **Bench numbers** (per `0be278f` errors entry):
  - 5878 `gemm_w4_fp8_marlin_cuda failed code 2` kernel failures
  - First failures at Pass 3 warmup B=1 (BEFORE any user request)
  - 5385 "successful" bench requests = broken signal artifact
    (failed reqs have 0 latency); SKILL #34b "server log first" again
    validated
  - TTFT median 0.0 ms in stats table (broken signal)

  **Refines prior framing**:
  - `0cde63d` framed as load-DEPENDENT → actually warmup-DEPENDENT
    (Pass 3 is the trigger, not sustained load)
  - `57c37b5` H8 DISPROVEN at conc=1 single curl was correct in scope
    but didn't include Pass 3 pressure — kernel works in true single-
    request only

  **Reconciles with Task #43 Arm A** (`da7f5a2` + `d09623a`):
  static-scratch / per-call workspace path is broken on sm_89 16GB
  across W4A16 (degraded 16× TTFT) AND PF8.3 (outright kill at warmup).

  **Task closures**:
  - Task #44 PF8 chain → completed (KILL)
  - Task #47 H1' refactor → still pending but BLOCKED pending
    redesign (default-on path empirically broken per this evidence)

  **PICKUP QUEUE PIVOT** (effective immediately):
  - **P1 NEW**: #28 Medusa scaffold (codex own ~500 LOC + 1 wk
    training). Medusa Alpaca cross-link `63769be` previously
    documented HF auth bypass via Alpaca dataset (ungated). Phase 1.A
    scaffold ready for codex pickup.
  - **P2 (was P3)**: #30 Hybrid W4A16/W4A8 dispatch substrate
  - **P3 (deferred)**: #47 H1' refactor pending the OOM-regression A/B
    gate redesign per da7f5a2/d09623a
  - PF8.3 substrate stays in tree as opt-in (default off per
    `db063ff`) for future H1' redesign work
  - PF8.5 license tooling (`scripts/pf85_bench_v11_user.sh` +
    companions) preserved for re-bench after future fix

  **Process win sedimented** (candidate for SKILL feedback memory):
  "User-only" framing applied to Claude-runnable scripts can become
  blanket constraint without re-evaluation. Subprocess `sleep` inside
  a script is not a Claude tool sleep; `run_in_background` cleanly
  bypasses the original limitation. Multi-tick saturation broke
  because no one re-asked "does the constraint still apply?". Adding
  to SKILL #29 evidence chain (n=6 — even self-imposed framing
  decays without periodic re-evaluation).
