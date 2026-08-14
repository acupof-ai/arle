# Sampling penalties and logit_bias verified on Qwen3.6 and DSv4 DSpark

## Context

`repetition_penalty` / `frequency_penalty` / `presence_penalty` were accepted at
the HTTP layer and dropped before sampling; `logit_bias` was accepted and also
kept the device-argmax fast path. Both landed for `fad8f4d5b`, verified on the
H20 pod against the shipping serve, not a unit harness.

## What worked

**The fast-path veto is observable from outside.** `is_raw_argmax()` gates the
device argmax, and a `+100` bias on one token makes the veto visible in the
decoded text: on Qwen3.6-27B DSpark the biased arm diverges from the
zero-penalty reference at character 0 and opens with the biased token, while the
three penalty arms diverge at character 61. A veto that failed would leave the
biased arm identical to the reference, because the bias would never reach the
argmax.

| arm | diverges at | first characters |
|---|---:|---|
| reference (no penalty) | — | `Here's a thinking process:` |
| `presence_penalty` 1.0 | 61 | `Here's a thinking process:` |
| `frequency_penalty` 1.0 | 61 | `Here's a thinking process:` |
| `repetition_penalty` 1.5 | 61 | `Here's a thinking process:` |
| `logit_bias` +100 | 0 | `Banana\nApple\nOrange\n…` |

**Speculation was proved engaged rather than assumed.** Two independent
readings: the throughput identity over the same window (448 steps produced
640 tokens = 1.43 tokens/step on short prompts, 2.86 on the 32K workload) and
`/v1/stats` (`chains 9066`, `drafted 45330`, `accepted 18416`). Either alone
would leave open that the penalties had silently pushed every request off the
spec path.

**`/v1/stats` carries the binary identity**, so the path probe is free:
`product_binary_sha256` on the live serve matched the build receipt byte for
byte, ruling out a stale process serving the measurement.

**The TP>1 relay gate needs a liveness follow-up, not a status code.** The
failure this gate exists for returns 200 and then kills every rank. At TP=8:
biased request 200 with the biased token dominating, two ordinary requests then
answer correctly, zero `relay deserialize` lines in the serve log.

## Rule

For a sampling-path change, the gate is a decoded-text arm that must differ from
an unpenalized reference on the real serve, plus a spec-engagement counter from
the same window. A unit test on the sampler proves the arithmetic; it cannot
prove the parameter survived the multiproc relay, reached the executor, and
vetoed the fast path.
