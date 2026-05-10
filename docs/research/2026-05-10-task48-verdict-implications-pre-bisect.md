---
title: Task #48 verdict implications pre-computed (matrix bisect in flight)
date: 2026-05-10
type: research
status: open (codex matrix bisect running, ~30 min to verdict; pre-computed pivot plan)
related_tasks: [#48 (codex in_progress matrix bisect), #25 (W4A8 accuracy fix closed root-cause-TBD)]
---

# Task #48 verdict implications — pre-computed conditional plan

> **Purpose**: Task #48 codex matrix bisect running NOW. Per cron-loop
> directive "准备下 round 假设" (prepare next round hypothesis), pre-
> compute conditional plan so codex can pivot immediately when verdict
> lands. Companion to `cb86836` for Task #43.

## §1 The matrix bisect setup

Codex pivoted from sequential bisect to a matrix script per b1a9c1e:
- Checks out each of 3 candidates (`35fc3cf` #24 / `c44788f` #40 /
  `09ae5a5` Path B Phase 1)
- Per-candidate: cargo build + cargo test
  test_w4a8_vs_bf16_token_diff
- Output redirected to per-candidate log files
- Restores original branch at end

Cross-cutting context: codex's `git log -S` already surfaced existing
errors entry **`81b6481` "W4A8 substrate produces 100% garbage output —
accuracy gate fails"** — strongly suggests the regression is from the
ORIGINAL W4A8 substrate landing, NOT from any of the 3 newer candidates.

## §2 Possible verdicts + Claude-side pre-computed actions

### §2.1 ALL 3 candidates have regression (most likely per 81b6481)

**Verdict**: Confirms 81b6481 hypothesis — regression is from the
original W4A8 substrate landing, not from any of #24/#40/Path B Phase 1.

**Forward path**:
1. Codex commits Task #48 errors entry: "regression NOT from any
   bisect candidate; pre-existing issue per 81b6481, lenient 25%
   gate is the workaround"
2. SKILL candidate #35 (root-cause-TBD canary) reaches **n=2 → n=3
   evidence** (e3e1ab5 + 81b6481 + this Task #48 result) → graduate
   to canonical SKILL anti-pattern in v1.15.0
3. Decision: tighten threshold (would fail in CI immediately) OR
   document as known limitation OR fix the W4A8 substrate properly
4. Task #48 closes; pivot to next pickup (Task #47 H1' if user runs
   bench v11 LICENSE OR Task #28 Medusa via Alpaca per 63769be)

### §2.2 Some candidates have regression, others don't

**Verdict**: Bisect found entry commit. e.g. 35fc3cf clean, c44788f
fails → c44788f introduced the regression.

**Forward path**:
1. Codex investigates the diff at the entry commit
2. Either revert that change OR fix the W4A8 path it broke
3. Re-run targeted test to verify fix
4. Update test threshold to original <1% rule per skill v1.3.0
   (assuming fix is good)
5. Task #48 closes with proper fix; remove the lenient 25% gate

### §2.3 ALL 3 candidates HEALTHY at their respective HEADs

**Verdict**: Regression entered AFTER all 3 candidate commits (i.e.
in commits between 09ae5a5 (newest of 3) and current HEAD).

**Forward path**:
1. Bisect needs expansion to include later commits (e.g. a2ad788
   Task #35, ace3cbe PF8.3 substrate)
2. Codex re-runs matrix with expanded candidate set
3. Lower-priority — original W4A8 substrate works, regression is
   from a recent change

### §2.4 Build/test environmental failures on some candidates

**Verdict**: Matrix script can't fully execute on some candidates
(missing dependency / API change / etc.).

**Forward path**:
1. Per skill v1.14.0 #36 (grep + behavioral A/B both required):
   build failure IS evidence — means the candidate isn't bisectable
   in its current form
2. Either revert candidate's build dependency at bisect time OR
   skip + flag the candidate as "untestable historically"
3. Document the bisect gap; partial verdict OK

### §2.5 Pass 3 contamination (current concern from tmux observation)

**Note from tmux**: Visible log line during first arm: `Pass 3:
warming prefill code paths for 4 batch sizes`. Pass 3 was added in
a2ad788 (Task #35), AFTER all 3 bisect candidates. This MIGHT mean:
- (a) Matrix script not isolating cleanly — running current-HEAD
  code despite checkout
- (b) Pass 3 logging string was ALWAYS in code (just renamed by Task
  #35), so log message at older commits is misleading
- (c) Codex is running a baseline pass at current HEAD before the
  matrix arms (entirely plausible)

**Forward path if (a) is true**: codex re-checks matrix script
isolation before trusting verdict. Re-run if needed.

**Forward path if (b)/(c) is true**: not a concern, normal log noise.

## §3 SKILL accumulation watch

| Candidate | Current evidence | Task #48 outcome could push to |
|---|---|---|
| #35 root-cause-TBD canary | n=2 (e3e1ab5 + 81b6481) | n=3 if §2.1 verdict (graduate to canonical v1.15.0) |
| #37 multi-shape bench | n=1 (Task #35 codex §Rule) | unchanged unless Task #48 surfaces shape concerns |
| #39 post-fix bench data stale | n=1 (codex on Task #35) | unchanged |
| #40 KILL vs graceful-fallback discriminator | n=1 (Task #43 evidence) | unchanged |
| (new candidate #41?) terminal silence ≠ no progress | n=1 (codex matrix bisect this session) | unchanged unless reinforced |

If §2.1 verdict lands → SKILL v1.15.0 graduation of #35.

## §4 Cross-references

- `b1a9c1e` Task #48 matrix bisect dispatch + pivot
- `62e8295` 81b6481 finding cross-link + SKILL version fix
- `e3e1ab5` original W4A8 regression flag + skill candidate #35
- `81b6481` original errors entry "W4A8 substrate produces 100% garbage"
- `cb86836` Task #43 verdict implications (template for this doc)
- `f63838b` codex pickup queue 2026-05-10 (P2 Task #48 spec)
- SKILL `kernel-optimization` v1.14.0 (8b530ad + d2c987f + 62e8295
  for version field fix)

## §5 Status

Pre-computed conditional plan. Codex's matrix bisect verdict expected
~30 min from start (~10:25 KST → ~10:55 KST). At verdict, dispatch
logic per §2.x is ready — zero-discovery-time pivot.
