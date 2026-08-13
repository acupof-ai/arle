# w2s step budget: the four KL terms cost more than forward plus backward

> Status: Confirmed

## Context

First per-stage timing of `arle train w2s` after the FP8 offload round-trip was
deleted ([errors](../errors/2026-08-13-w2s-fp8-base-offload-roundtrip-was-lossy.md)).
Six steps, `--confidence-threshold 0.99` so the confidence gate skips nothing;
ThinkingCap-Qwen3.6-27B-FP8 base/student, four 0.8B aux, GSM8K prompts, one free
H20 (95.2 GB). Build `w2s-refactor1` at `18096ec7f`, `RUN_EXIT=0`.

`StageTimer` (`crates/train/src/w2s.rs`) fences the device before stopping each
stage, so these are GPU times. Stage sum 4.418 s vs measured total 4.415 s — the
breakdown is complete, nothing hides between stages.

## What Worked

Steady state (steps 1-5) is 4.105 s/step. Mean over all six:

| Stage | Mean s | Share | Range |
|-------|--------|-------|-------|
| backward | 0.973 | 22.0% | 0.635-1.959 |
| global_kl | 0.820 | 18.6% | 0.498-2.104 |
| consistency | 0.652 | 14.8% | 0.347-1.027 |
| local_kl | 0.576 | 13.0% | 0.478-0.669 |
| student_fwd | 0.567 | 12.8% | 0.485-0.753 |
| confidence | 0.302 | 6.8% | 0.161-0.471 |
| cleanup | 0.283 | 6.4% | 0.072-1.107 |
| aux_delta | 0.208 | 4.7% | 0.141-0.408 |
| optimizer | 0.027 | 0.6% | 0.022-0.042 |
| kd_loss | 0.010 | 0.2% | 0.006-0.013 |
| total | 4.415 | | 2.871-5.964 |

The step runs three 27B forwards (student, π_old, π_base) plus four 0.8B aux
forwards, so forward count was the expected cost center. It is not.
`student_fwd` is 12.8%. The divergence work — `local_kl + global_kl +
consistency + kd_loss` — is 2.058 s, 46.6% of the step, more than forward
(0.567 s) and backward (0.973 s) together.

Two stages are host round-trips rather than model work:

- `consistency` (0.652 s, 14.8%) reads both `[seq, vocab]` ΔT tensors to host
  and computes the cosine similarity on the CPU (`cosine_similarity`, w2s.rs).
- `confidence` (0.302 s, 6.8%) softmaxes the full sequence, copies all of
  `probs` to host, then uses only the last position's max.

`kd_loss` at 0.010 s is the control: the same KL machinery costs ~60x less when
it goes through the chunked path, which is why the two regularizer terms at
0.576/0.820 s are worth investigating rather than accepting.

Two outliers, cause unknown, single occurrence each: step 3 `global_kl=2.104`
(other five 0.498-0.669) and step 2 `cleanup=1.107` (others 0.072-0.165). Step
0's `backward=1.959` is cold start; steady-state backward is 0.775.

VRAM unchanged by the refactor: 27.9 GB after base+student, 33.5 GB after aux.
Loss fell 25.158342 -> 19.483383 over the six steps.

## Rule

Count the forwards, then measure them anyway. Three 27B forwards looked like the
budget and were an eighth of it; the loss terms that each walk the logits
separately were half. A stage that copies a `[seq, vocab]` tensor to host to
extract one scalar costs more than the 27B forward that produced it.
