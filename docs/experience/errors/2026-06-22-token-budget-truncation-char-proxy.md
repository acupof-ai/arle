# Char-length proxy for "token budget" → shipped a wrong root cause (truncation called format-confound)

## Context

Validating rubric-OPD capability on Qwen3.6-27B-FP8. Base MATH-500 pass@1 was
0.160 (8/50); 33/50 answers had no `\boxed` and scored as wrong. To decide
whether 16% was a real ceiling, I checked whether the no-`\boxed` answers were
**truncated at the 1536-token eval cap**. I measured answer length in **characters**
(`>5000 chars ≈ near the 1536-tok cap`) and found 0/50 near the cap, max 5093
chars. I concluded "**NOT a 1536-token-cap artifact**; it's a `\boxed`-format-
compliance floor" and **committed that into the wins doc + PNG + commit message**
(0f4f5253).

## Root Cause

The char→token conversion was wrong. I assumed ~4 chars/token (English-prose
rate). But these answers are **LaTeX-dense math**, which tokenizes at ~2–3
chars/token (every `\frac`, `{`, `}`, `^`, digit is its own token). So 1536
tokens of this text is only ~3–5k **chars** — my `>5000 char` threshold for
"near the cap" never fired, even though the generations were sitting **exactly**
at the cap.

Counting with the **real tokenizer** (`Qwen3.6-27B-FP8/tokenizer.json`):

| group | n | token min/median/max | ≥1500 tok |
|-------|---|----------------------|-----------|
| has `\boxed` | 17 | 303 / 1536 / 1536 | 14/17 |
| no `\boxed`  | 33 | **1536 / 1536 / 1536** | **33/33** |

**All 33 no-`\boxed` answers are at exactly 1536 tokens** — truncated mid-equation,
mid-sentence. The decoded tails confirm it ("`= \frac{-6`", "`We need three`",
"`Carla:`"). It was a token-budget truncation all along. The true root cause:
Qwen3.6-27B is a heavy-CoT thinker that overruns the 1024 (rollout) / 1536 (eval)
budgets before emitting `\boxed`. An earlier n=12 analysis had *correctly* called
this a truncation confound; my char-proxy "corrected" it the wrong way.

## Fix

- Re-measured with the model tokenizer; corrected the wins doc, the PNG footnote,
  and re-committed with the true root cause (token-budget truncation).
- Next run: re-eval the saved per-round adapters at eval budget ≥8192 (training-
  free) for a clean curve; retrain with rollout budget ≥4096 + ≫16 prompts.

## Rule

**To test a token-budget hypothesis, count tokens with the actual tokenizer —
never a char-length proxy.** chars/token is content-specific (English ~4, LaTeX
~2–3, code in between); a fixed char threshold silently mis-converts and can flip
the conclusion. This is the §0 trap: the aggregate (char proxy) **and** a plausible
mechanism (format non-compliance) both lied; the decoded cases + real token counts
were the only ground truth. Decode the cases **and** measure in the native unit
before shipping a root cause — especially before committing it.
