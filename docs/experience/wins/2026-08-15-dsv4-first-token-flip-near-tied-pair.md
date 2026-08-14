# DSv4 first-token flip is a near-tied logit pair, not a batching defect

## Context

Issue #202 item 2: `Explain bloom filters and their use cases.` over raw
`/v1/completions` at temperature 0 continued as answer form (` A:`) at c=1 and
document form (`.`) at c=8/16 on the TP=8 DSpark baseline
(errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md).
Near-tied numerics and a batching defect carry opposite remedies; nothing
measured separated them.

## What Worked

Logit-bias threshold measurement — zero code change. The DSv4 sampling path
has no logprobs capture (501 by design), so the margin was measured through
the existing `logit_bias` parameter: bias the losing first token upward until
the continuation flips; the threshold estimates the top-2 margin. Serve:
DSv4-Flash-FP8 + DSpark draft, TP=4 on 4×H20 (build lp202n, a8150bc6b),
GPUs 4–7.

- Positive control: `logit_bias +100` on token 334 (` A`) and 16 (`.`) each
  force that token — the bias path is live on DSv4, no silent no-op.
- c=1: baseline `.` 10/10; bias +0.125 on ` A` flips 3/3. **Top-2 margin
  < 0.125 logit units.**
- c=16 (bloom + 15 filler prompts, concurrent): unstable at zero bias —
  6/10 `.`, 4/10 ` A`. The margin is inside the batch-composition noise.
- Cross-config: TP=4 c=1 picks `.` where the incident's TP=8 c=1 picked
  ` A` — the winner flips across TP reduction orders too.

Verdict: the pair is tied to within FP8 MoE + reduction-order noise at every
measured configuration. Concurrency changes batch composition per step, so
the winner varies run-to-run. This is the near-tied-prompt arm of the #202
dichotomy; no runtime defect is indicated.

## Rule

When two remedies hinge on a logit margin and the backend has no logprobs
capture, `logit_bias` thresholding measures the margin through the public API:
positive control first (a large bias must force the token), then raise the
bias on the losing token until the output flips. A flip threshold below the
run-to-run flip rate at zero bias means the pair is tied within noise.
