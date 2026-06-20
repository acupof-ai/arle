# Agentic OPD "regression" was a confounded experiment: the gate was a teacher-timeout artifact

## Context
Gate-zero reported teacher 35B beats base 4B by +42pp on BFCL live_irrelevance
(teacher 0.88 vs base 0.46) → "room exists" → we ran greedy reverse-KL OPD on a
2045-row held-out BFCL **live** corpus (student Qwen3.5-4B, teacher
Qwen3.6-35B-A3B-FP8, no-think, rollout-128, 50 steps). Aggregate BFCL-live went
**0.6505 → 0.5097 (−14pp)**; live_irrelevance collapsed 0.46 → 0.00. The first
write-up of this entry concluded a **structural** cause ("on-policy distillation
can't teach abstention") and KILLed the arm. **That conclusion was wrong** — it
generalized a bug-result without decoding the cases (ckl: "看 case 再说,
做算法以 case 为事实"). See §0 *Case-as-fact*.

## Root Cause (decoded cases — the real attribution)
Decoding the actual outputs (base / step25 / step50 / teacher) on live_irrelevance:

| live_irrelevance | abstain (prose) | call | timeout/error |
|---|---|---|---|
| base (n=50) | 23 (46%) | 27 | 0 |
| **teacher (n=17)** | **1** | **2** | **14** |
| student step25/50 (n=50) | 0 | 50 | 0 |

1. **The gate's "+42pp teacher abstention" was a TIMEOUT ARTIFACT.** The slow
   no-think 35B teacher *timed out* on 14/17 cases; those error strings don't start
   with `[`, so the eval bucketed them as "correctly declined" → fake 0.88. Of the
   3 cases the teacher actually answered: **1 abstain, 2 call = 33% abstention —
   BELOW the base's 46%.** There was never +42pp room; the no-think teacher is
   *worse* at abstention than the base.
2. **OPD then faithfully distilled the no-think teacher's over-calling.** The
   student's abstention dropped toward the (bad) target — working as designed on a
   confounded target, not a structural failure of OPD. (Reverse-KL + the
   tool-call-heavy corpus amplify it past the teacher's own 33%.)
3. **Why no-think is the wrong regime for abstention:** thinking helps the model
   reason "no tool applies" — base think-on irrelevance was 0.65 vs no-think 0.28.
   The teacher was forced into no-think only because its FP8/DeepGEMM-disabled
   generation is too slow to finish a thinking trace (the same slowness caused the
   timeouts).

The hypothesis ("OPD lifts agentic capability") is **NOT overturned** — the
experiment was confounded: a fake gate (timeouts-as-abstention) + a wrong teacher
mode (no-think over-calls). Tool-use categories *did* lift (live_simple 0.70→0.84,
relevance 0.75→0.81), consistent with OPD working where the target is good.

## Fix
1. **Timeout-clean gate** — exclude `Error during inference` / request-errors from
   the denominator AND never count them as "abstention/decline"; re-measure the
   *true* teacher abstention before claiming room.
2. **A teacher mode that actually abstains** — bounded thinking (the runtime
   `enable_thinking`/`chat_template_kwargs` fix, task #9, with a token budget so it
   finishes), or a faster teacher build (DeepGEMM-native) so think-on doesn't time
   out. Then re-gate: is there real room on irrelevance?
3. Only after a clean gate + a good target: re-run OPD and re-decode the cases.

## Rule
A passed *gate* and an aggregate metric are both worthless until you **decode the
cases and audit the harness for artifacts** (here: timeouts silently scored as
abstention). 做算法以 case 为事实; 先归因清楚再推翻 — never KILL a hypothesis on a
confounded experiment. See §0 *Case-as-fact* in AGENTS.md. (Sibling infra note:
BFCL prompts are long → the MATH recipe OOM'd; `ARLE_OPD_GRADIENT_CHECKPOINTING=1`
fits it, ~60 GB peak.)
