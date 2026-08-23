# Fixed a real config bug that was not the cause

> Status: fix shipped (`7130f0b8b`); the symptom it was credited with is unexplained

## Context

`RadixArk/Qwen3.8-27B-DSpark` drafting for `Qwen3.8-27B-FP8` accepted 13% of
its drafts at c=1 and 0% at c=8 — no speedup, and c=1 ran at half the MTP rate.

Reading the configs turned up a genuine defect. `DsparkConfig` read
`rope_theta` from the top level only, defaulting silently when absent, and had
**no field at all** for RoPE scaling, so `dspark.rs` passed `None` to
`precompute_rope` unconditionally. `transformers >= 5.12` nests both under
`rope_parameters`, and this drafter declares `rope_type: "yarn", factor: 32`.
So a YaRN drafter was running vanilla RoPE.

That is a real bug, it is fixed, and the reasoning for why it would destroy
acceptance was sound: wrong positions from layer one, verify rejects everything.

## What the measurement said

| | before the fix | after |
|---|---|---|
| accept | 13% | **13%** |
| tok/step | 1.64 | 1.58 |
| tok/s | 31.5 | 29.8 |

Unchanged. The fix corrected a real defect and moved nothing.

## Root Cause

Unknown for the acceptance collapse. The remaining structural mismatch is that
the Qwen3.8 target uses mRoPE — `partial_rotary_factor: 0.25`, interleaved,
sections `[11,11,10]` — while the drafter uses full standard RoPE. That is not
a config field; matching it needs mRoPE support in the draft path. It is a
candidate, not a conclusion: it has not been tested either.

## What went wrong in the reasoning

The commit message credited the fix with the 13%→0% symptom before measuring.
The defect was real and the causal story was plausible, and neither of those
makes it the cause. Two separate claims got merged into one:

1. this config field is dropped — verified by reading the code and the config;
2. that is why acceptance collapsed — never tested until after the commit.

## Rule

A fix earns the symptom only by moving the number. State the defect and the
symptom as separate claims until a measurement joins them, and keep the commit
message to whichever one is actually established.
