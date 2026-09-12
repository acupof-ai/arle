# Thinking-model needle: which channel does the gate judge?

Status: pending-remote. The mechanism landed in the gate (diagnostics only,
criterion unchanged); the criterion decision waits for one real thinking-model
serve. This document takes a position and names the run that overturns it.

## The disagreement

`needle_gate.py` has two arms that read different channels of an
OpenAI-shaped chat response:

- **Greedy needle arm** classifies `message.content` only. The needle
  `738291` in `reasoning_content` with empty `content` is a **miss**.
- **temp coherence arm** (`needle_gate.py temp`) concatenates
  `reasoning_content + content` before its empty/glued-repeat checks
  (added 2026-07-24 after an observed empty-`content` thinking response).

The arms are not inconsistent by accident — they answer different
questions. The needle arm asks whether the model *retrieved the fact and
returned it to the caller*; an API caller reads `content`, so a needle that
never leaves `reasoning_content` was never returned. The temp arm asks
whether *any* coherent long-form generation happened, and reasoning text is
valid evidence for that.

The new mechanism makes the split visible without judging it: every greedy
run prints `loc=content|reasoning_only|both|neither`, a reasoning-only run
prints `NEEDLE_REASONING_ONLY`, and the per-length SUMMARY carries
`reasoning_only=N`. The parser accepts both the enriched and the legacy
SUMMARY shape. No verdict or exit code changes; `reasoning_only` rows are
still `miss`. Coverage:
`scripts/tests/test_needle_gate_reasoning_location.sh`.

## Position: do not widen. The content-only verdict is correct.

The gate licenses the inference "this backend returns retrieved facts to
callers." Widening it to accept a needle found only in reasoning_content
would license that inference from a response in which the caller received no
answer. That is the expensive error on a correctness gate: a suppressed true
negative. Nothing in the synthetic evidence shows a healthy model producing
that shape, and the code already contains a fail-safe against the obvious
cause — `split_reasoning` (`crates/infer-server/src/schema.rs:1144`) returns
an unclosed thinking block (max_tokens cutoff mid-reasoning) **entirely as
content**, precisely so a truncated answer is not delivered as an empty
response. A `reasoning_only` marker therefore cannot come from a plain
token-budget cutoff; it requires the model to have *closed* the thinking
block and then produced no answer.

The likely explanations, in order:

1. **Gate/serve budget mismatch (leading, not one of the three options).**
   The gate requests `max_tokens=16` (`NEEDLE_MAX_TOKENS`). A thinking model
   that spends the budget, emits `</think>`, and stops at the cap yields
   reasoning-with-needle + empty content with `finish_reason=length`. This
   is a gate configuration defect for thinking serves, not a criterion
   question and not a model defect. The run below tests this first because,
   if it reproduces, the criterion is never in dispute.
2. **Option C — serving bug.** The model produced a post-thinking answer
   that the splitter or template path dropped, or the streaming
   (`sse_util`) and non-streaming (`finalize_chat_content`) split policies
   disagree. The two policies are documented as needing to stay in lockstep.
3. **Option A — widen the greedy arm.** Only if a healthy, adequately
   budgeted model stops with `finish_reason=stop`, puts the needle in
   reasoning, and leaves content empty *and* real callers of this serve
   consume `reasoning_content` as the answer channel. Then the caller-facing
   contract differs from the gate's and the gate must follow it.
4. **Option B — narrow the temp arm to content.** Reject on current
   evidence: its join was added for a measured empty-content case and it
   tests coherence, a property reasoning text genuinely has. Narrowing would
   make two different questions share one channel for symmetry, and would
   reintroduce a vacuous empty-output pass. Revisit only if real joined
   responses pass coherence while content is unusable.

## The run that settles it

One serve window on one sm90 card; folds into P7's window (same serve,
same model class).

- **Model:** `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (canonical CUDA
  thinking model), greedy `temperature=0`.
- **Serve:** `arle serve --max-thinking-tokens 512` (default 0 disables the
  split; `enable_thinking` then tracks the template/request default).
- **Prereg:** `needle-reasoning-channel-<ts>`.
- **Prompt:** standard needle ladder, short subset `115,241,446`, ×3.

Three configs, each with the raw response JSON archived
(`content`, `reasoning_content`, `finish_reason`, `completion_tokens` —
finish_reason is the discriminating field and is not in the gate log
today; capture it with a direct request alongside the gate):

| Config | Request | What it establishes |
|---|---|---|
| A current budget | `NEEDLE_MAX_TOKENS=16` | Reproduce `reasoning_only` cheaply; expect `finish_reason=length` |
| B adequate budget | `NEEDLE_MAX_TOKENS=512` | Does a completed answer reach content? |
| C thinking off | `RAW=1 TEMPLATE=qwen3_nonthink`, same ladder | Control: retrieval itself is intact and lands in the judged text |

Decision table — what `content` / `reasoning_content` must contain:

| Observed in B (adequate budget) | Conclusion | Action |
|---|---|---|
| `content` exact, `reasoning` may also contain it (`loc=content`/`both`, all ×3); A alone showed reasoning-only with `finish_reason=length` | Budget artifact | Keep criterion; set thinking-serve budget guidance (raise `NEEDLE_MAX_TOKENS` under thinking) |
| Needle in `reasoning`, `content` empty/no-needle, `finish_reason=stop`, tokens well under cap; same prompt on a reference serve (vLLM/SGLang) returns the answer in `content`, or SSE vs non-stream differ | Option C: serving bug | Fix `split_reasoning`/template/SSE path; gate stays content-only; add a parity-style regression |
| Same shape as above but the reference serve also returns empty content, and our consumers read `reasoning_content` as the answer | Option A: contract differs | Widen the greedy arm to judge both channels, explicitly and with a caller-side justification; temp arm unchanged |
| Joined reasoning passes the temp arm on responses whose content is unusable, systematically | Option B | Narrow temp to content; record the 2026-07-24 observation as the bug it was |

Config C must show exact content hits in every row; if it does not, the
model is missing the needle regardless of channel and the criterion
question is moot — the failure is retrieval.

## What is deliberately not in this lane

No change to the greedy verdict, the miss definition, exit codes, or the
temp arm. A synthetic transcript can build the diagnostic and prove both
behaviors; it cannot show what a healthy thinking model emits, which is the
only fact that distinguishes the options. Synthetic coverage now pins both
arms' present behavior, so whichever way the run goes, the change is a
visible criterion change with a failing-or-passing test rather than a
silent drift.
