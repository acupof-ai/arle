---
title: #36 brief gap — bench script doesn't restart server, ambiguous "-- --foo" suffix in brief
date: 2026-05-10
type: research
status: course-correction-sent-to-codex
---

# #36 brief gap — bench script doesn't restart server, ambiguous "-- --foo" suffix in brief

> Tick deliverable: document the brief-gap → codex-misread →
> course-correction chain so the bench server-restart protocol is
> encoded explicitly for future briefs.

## What happened

### Step 1 — Original brief (this loop tick, /tmp/codex-brief-36-bench.txt)

Claude brief included this bench command:

```bash
scripts/bench_guidellm.sh 36-bench-A-queuebound --concurrencies 8 ... \
    -- --admission-policy queue-bound --max-waiting-requests 4
```

The trailing `-- --admission-policy queue-bound --max-waiting-requests 4`
was Claude's intent to mean "args to the running server, not to
bench_guidellm.sh". This was implicit and ambiguous.

### Step 2 — Codex investigation (8m 36s into Working)

Codex's terminal showed:

> "还有一个实际阻塞：当前 infer CLI 没有 --max-waiting-requests，
>  用户给的 A/B 命令无法真正降低 cold soft-cap。我要补这个 CLI override，
>  否则 PrefixAware gate 仍可能在默认 256 队列上不触发。"

Translation: codex concluded `--max-waiting-requests` doesn't exist
as a CLI flag and decided to add it.

### Step 3 — Direct verification (Claude this tick)

Direct grep on `infer/src/main.rs`:

```rust
infer/src/main.rs:124:    admission_policy: String,            // --admission-policy
infer/src/main.rs:127:    /// Defaults to max_waiting / 4.
infer/src/main.rs:129:    cold_headroom: Option<usize>,        // --cold-headroom
infer/src/main.rs:133:    max_waiting_requests: Option<usize>, // --max-waiting-requests
infer/src/scheduler/types.rs:226: max_waiting_requests: 256,    // default when omitted
```

**All three flags ALREADY EXIST**. Codex's conclusion was wrong —
likely a misread during the 8min build/explore window where codex
searched bench_guidellm.sh and start_infer.sh first (those don't
pass through the flag) and conflated "bench script doesn't pass
through" with "CLI flag doesn't exist".

If codex had committed the duplicate clap arg, the build would have
failed with a clap conflict.

### Step 4 — Course correction (this tick)

Claude paste-buffered `/tmp/codex-correction-36.txt` to codex tmux:

- Cited main.rs:124,129,133 + types.rs:226 as direct evidence flags exist
- Identified the **real gap**: `bench_guidellm.sh` line 57 comment says
  "infer HTTP server is already running at --target" — bench script
  does NOT spawn or restart server. The trailing `-- --foo` suffix in
  the original brief had no effect on the running server's behavior.
- Provided the correct A/B bench protocol:
  1. Stop existing server
  2. Restart with `--admission-policy queue-bound --max-waiting-requests 4 ...`
  3. Wait for /health
  4. Run bench arm A
  5. Stop, restart with `--admission-policy prefix-aware --max-waiting-requests 4 ...`
  6. Wait for /health
  7. Run bench arm B
  8. Capture `/v1/stats prefix_aware_admit_deferrals` to prove gate fired
- Suggested **NOT** adding bench-script restart logic (over-engineering for one A/B);
  document the manual restart-between-arms protocol in the wins/errors entry.

Codex acknowledged correction (now `Working (3s)`) and proceeds.

## Anti-pattern caught (skill candidate for v1.10.0?)

**"Brief ambiguity in CLI passthrough syntax"** — the `--` separator
in shell convention can mean either:

1. End-of-options for the *current* command (POSIX convention)
2. Forward-args-to-downstream-process (less universal; depends on
   whether the consuming script implements `--` as a passthrough)

`bench_guidellm.sh` does NOT implement (2); it parses everything as
its own args. So `bench_guidellm.sh --foo bench-name -- --admission-policy x`
treats `--admission-policy x` as bench_guidellm.sh args (which it
rejects).

**Fix for future briefs**: when the desired effect is "change server
config between bench arms", explicitly say:

```
1. kill any existing infer server (pid via pgrep -f infer)
2. start new server: cargo run --release ... -- --admission-policy queue-bound ...
3. wait for /health, then run bench
4. repeat for arm B
```

Don't rely on `-- --foo` suffix shorthand.

## Companion finding — codex search-then-conclude trap

Codex's 8min Explored window searched 5+ files and read main.rs twice
but still concluded the wrong thing. Possible cause:

- Search 1: `max-waiting|admission-policy|model-path|num-slots|...` in
  `bench_guidellm.sh` and `start_infer.sh` first → no match (these
  scripts don't pass through)
- Search 2: re-read main.rs but kept the prior "no match" frame → missed
  the `Option<usize>` definition

This is similar to skill anti-pattern #25 ("hypothesis-context vs
implementation-context mismatch") in spirit: codex's hypothesis
context was "bench script + server start scripts" but the answer
lived in the server CLI definition.

**Mitigation for future codex briefs**: when asking codex to verify
flag existence, give the **exact file:line** if known. Claude already
had `main.rs:124,129,133` in hand from the 5e902da survey but didn't
include them in the brief — including would have prevented the
misread.

## Action items

1. ✅ Course correction sent (this tick)
2. ⏳ Codex resumes #36 bench with correct protocol
3. ⏳ Wins or errors entry per A/B outcome
4. 🔄 Skill v1.10.0 candidate: "brief ambiguity in CLI passthrough"
   anti-pattern (queue for next skill update tick)
5. 🔄 Brief-template improvement: include exact file:line for any
   flag/symbol existence claim in future codex briefs

## Cross-references

- Original brief: `/tmp/codex-brief-36-bench.txt`
- Course correction: `/tmp/codex-correction-36.txt`
- Substrate survey: `docs/research/2026-05-10-36-prefix-aware-admission-substrate-complete-bench-pending.md`
- Counter audit: `docs/research/2026-05-10-36-codex-counter-audit-clean.md`
- M_b3 plan (SUPERSEDED): `docs/plans/M_b3-prefix-aware-admission-step1-directive.md`
- Skill anti-pattern #25 (hypothesis-context vs implementation-context):
  `.claude/skills/kernel-optimization/SKILL.md`

## 状态

Brief gap surfaced + codex course-corrected + bench protocol clarified.
Codex `Working (3s)` on the corrected path. Counter instrumentation
(audited CLEAN 60ffa41) still good — only the duplicate CLI flag work
was canceled. Next tick: check codex's bench arm A/B progress.
