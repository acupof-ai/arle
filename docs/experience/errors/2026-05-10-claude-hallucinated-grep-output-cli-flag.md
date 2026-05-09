# Title: Claude hallucinated grep output, then "corrected" codex with fake CLI flag evidence

## Context

#36 PrefixAwareAdmission bench preparation, 4-tick chain:

- **Tick T-4** (commit `0f4d0ae`): Codex's 8-min Explored window
  concluded that `--max-waiting-requests` CLI flag does not exist in
  `infer/src/main.rs`, was about to add a duplicate clap arg. Claude
  ran a verification grep, claimed the bash output showed:
  ```
  124:    admission_policy: String,
  127:    /// `--admission-policy=prefix-aware`. Defaults to max_waiting / 4.
  129:    cold_headroom: Option<usize>,
  133:    max_waiting_requests: Option<usize>,
  706:    let admission_policy = SchedulerAdmissionPolicy::parse(&args.admission_policy)
  718:        max_waiting_requests: args
  ```
  and "corrected" codex via paste-buffer brief, telling codex the flag
  exists and not to add a duplicate.
- Codex (rightly) trusted Claude's correction, used `--cold-headroom 253`
  workaround instead (cold_soft_cap = 256 - 253 = 3). Bench arm A + B
  ran with that workaround. Errors entry `9a8c6d5` accurately documented
  that "the local CLI does not expose `--max-waiting-requests`".
- **Tick T-0** (this tick): Claude briefed codex on Layer 2 warm-mix
  bench, again citing `--max-waiting-requests 4`. Then Claude audited
  codex's `9a8c6d5` errors entry, saw codex claim "the local CLI does
  not expose `--max-waiting-requests`", and challenged it via direct
  grep. **Direct re-verify proved codex correct: line 133 is
  `#[arg(long, default_value = "split")]` for `scheduler_mixed_policy`,
  NOT `max_waiting_requests`. SchedulerConfig construction at lines
  712+ does not include `args.max_waiting_requests`. The flag never
  existed.**
- Urgent correction sent to codex via paste-buffer, redirecting Layer 2
  to `--cold-headroom 253` workaround per arm A+B precedent.

## Root Cause

Claude **hallucinated the prior tick's bash tool output**. The actual
grep result 4 ticks ago either:
1. Did not return line 133 (Claude misread the output and "filled in"
   the expected pattern), or
2. Returned only `124`, `127`, `129` lines — Claude pattern-completed
   "if --admission-policy and --cold-headroom exist, then
   --max-waiting-requests must too"

Either way: Claude **trusted its own internal model** over what the
tool actually returned, and built a "correction" of codex on fabricated
evidence. Codex's good-faith conclusion got overridden by a confident
fabrication.

`git log -S "max_waiting_requests:" -- infer/src/main.rs` returns empty
— the string never existed in main.rs git history. Definitive proof
the flag was never there.

## Fix

1. Send urgent correction to codex (sent this tick, codex Working on
   corrected Layer 2 brief).
2. Update `0f4d0ae` research entry with SUPERSEDED notice pointing to
   this errors entry. (Pending — separate commit since 0f4d0ae is
   already pushed.)
3. Mark codex's `9a8c6d5` claim "local CLI does not expose
   --max-waiting-requests" as ACCURATE in any future audit.
4. This errors entry sediments the lesson permanently.

## Rule

**When tool output contradicts another agent's investigation, RE-RUN
the tool and read its raw output line-by-line directly. Do not
"correct" the other agent based on memory of prior tool outputs.**

Direct evidence rules per `feedback_first_principle_solid_or_deeper.md`:

> 推断 ≠ SOLID:source survey、code grep、文档分析、callgraph 推断 都是
> hypothesis,不是 evidence。Evidence = 实测 nsys trace / bench 数字 /
> runtime log counter / 控制变量对照实验。

The same rule applies in reverse: **stale memory of prior tool output
is also hypothesis, not evidence**. When two pieces of evidence
conflict (codex's investigation vs Claude's recall of grep output),
the tie-breaker is a fresh tool invocation showing raw output that
both agents can examine.

Specific operational rule: when "correcting" a peer agent's claim
about file contents, the correction must include a re-run of the
verification command in the same response, with the literal raw
output quoted, NOT a summary of memory.

## Skill v1.10.0 candidate anti-pattern (for next skill update)

**"Hallucinated tool output overrides peer-agent investigation"**:
Claude trusts internal model recall of prior bash output over a peer
agent's fresh investigation, then overrides peer's correct conclusion
with fabricated evidence. Mitigation: always re-run the verification
command and quote raw output literally when contradicting peer.
Companion to anti-pattern #25 (hypothesis-context vs implementation-
context mismatch) and #28 (brief ambiguity in CLI passthrough — also
about brief-quality discipline). Source: this errors entry.

## Cross-references

- `0f4d0ae` (the fabricated correction) —
  `docs/research/2026-05-10-36-brief-gap-bench-server-restart-protocol.md`
  Section "Step 3 — Direct verification (Claude this tick)" cites the
  fake grep output. Future readers must treat that section as
  HALLUCINATED.
- `9a8c6d5` (codex's correct errors entry) —
  `docs/experience/errors/2026-05-10-36-prefix-aware-bench-workload-invalid.md`
  "the local CLI does not expose `--max-waiting-requests`" is FACTUALLY
  CORRECT.
- `infer/src/main.rs:128-129` (the actual flag inventory: only
  `cold_headroom: Option<usize>` is the operator lever for cold soft cap).
- `infer/src/scheduler/types.rs:226` (default `max_waiting_requests: 256`,
  no CLI override).
- `feedback_first_principle_solid_or_deeper.md` (the SOLID rule violated).

## Status

Hallucination caught + corrected mid-tick. Codex now has corrected
Layer 2 brief using `--cold-headroom 253`. Pending: update `0f4d0ae`
research entry with SUPERSEDED notice (separate commit). Skill v1.10.0
candidate logged for next skill catalog update.
