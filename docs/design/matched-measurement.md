# Matched measurement, and the gate's positive control

Design note 5 of 5 ([plan](../plans/2026-09-02-design-theses.md)). Date:
2026-09-09.

## Problem

Performance work on this runtime produces one claim per change: faster,
slower, wash. The serving workload is a 32K-token agent context at
concurrency — 119K prefix, 875 append, 214 output at the TraceLab medians,
95.7% prefix-hit rate — and on that workload three things go wrong routinely.
A c=1 win is a c=16 loss when the feature serializes. MoE decode is
non-deterministic, so token-exact parity kills correct changes and passes
vacuously when the gate's arm errors. Effects worth having are often inside
10%, where one run cannot sign the answer. The problem is distinguishing a
real effect from noise, harness artifact, and a gate that passes without
testing anything.

## Standard practice and where it fails

"Run once, merge if faster": a before/after on the box, one end-to-end
`out tok/s` number, token-exact output against the baseline. It fails here at
three named points. End-to-end tok/s blends time-to-first-token and
inter-token latency, which have opposite treatments — prefill work moves TTFT,
decode work moves ITL, and one number hides which moved. Token identity against
the baseline is the wrong bar: MoE decode is non-deterministic (expert routing
and reduction order vary run to run), so a correct kernel swap diffs tokens and
a broken one can match them. And a single run cannot sign a ≤10% effect on a
shared GPU box: in the decode-graph A/B the between-GPU spread (87.6 vs 97.8
tok/s on the same arm) exceeded the treatment spread.

## The design

**Two SLOs, reported separately.** TTFT and ITL are separate goals with
separate shapes ([bench spec](../bench-and-trace-spec.md) §2, §3.4).
`out tok/s` is never reported; decode throughput is `1000 / ITL mean`. A claim
names the SLO it moves and the concurrency it moves it at.

**Matched A/B, three trials.** Same binary, shell, machine, GPU clocks, server
lifecycle, prompts, and seed; one treatment variable; baseline and treatment
side by side from the same shell (§3.1). Below 5% or inside run variance: at
least three trials per arm, median plus range. The default iteration path is
one arm against a rolling champion row with a measured ±3% drift band (§3.0);
a fingerprint change (flags, GPUs, slots, dataset, driver) re-anchors the
champion first.

**Correct inference is an envelope, not an identity.** The gate is the needle
ladder: `scripts/needle_gate.py` retrieves a fact at 115–8000 tokens, three
runs per length, requiring every length exact and deterministic, plus
`scripts/lever_gate.sh` against a same-config baseline envelope. MoE
non-determinism relaxes token identity at the model layer only; the operator
contract stays bit-exact, and kernel changes carry the §4.1 numerical bounds
against an FP32/FP64 reference.

**The gate needs a positive control.** A gate arm that errors stops gating:
the failure is loud, never silent. `lever_gate.sh` echoes every skipped arm and
fails loudly when no baseline envelope exists ("any miss count would pass").
The needle gate's temp arm glues `reasoning_content + content` because a
thinking model's empty `content` made the coherence check vacuous (observed
2026-07-24). A default flip proves the off-arm is really off: the DSv4
decode-graph flip verified the eager arm at 0 captures.

**Deletion is a verdict too.** A flag is a promise that both arms are viable.
Once one arm has a dated losing verdict, the flag is dead A/B wiring: three
deletion waves removed 17 serve flags (66 → 49) and 5 env aliases, hardcoding
the measured winner
([wins 2026-08-22](../experience/wins/2026-08-22-flag-deletion-wave.md)).

## The failure

The decode-graph flag that logged ARMED and captured nothing
([errors 2026-08-01](../experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md)).
`--qwen35-decode-graph` printed an ARMED line from the warmup path while the
dispatch path returned before reaching the graph lane under paged KV — the
serving default. A four-arm serve A/B produced only noise; one nsys capture
settled it: zero `cuGraph*` calls with the flag on. The rule it wrote: a
flag's log line is not evidence the feature ran — count the API calls the
treatment should produce. The flag was deleted in the first deletion wave once
the paged path shipped. The ten failures worth carrying out of this codebase
are collected in [what-breaks.md](what-breaks.md).

## The number

Three numbers, each from a matched measurement. DSpark spec decode: 2.24× at
c=1, −39% at c=16 (8×H20, 27B, 8 requests per concurrency level; the no-spec
control scaled 2.9× over the same range) — the measurement that re-scoped the
feature to low concurrency and wrote the "measure against not using it at the
target operating point" rule
([errors 2026-07-26](../experience/errors/2026-07-26-dspark-spec-decode-serializes-and-loses-above-c1.md)).
The drift band: ±3%, measured 2026-07-16, the threshold below which a one-arm
comparison escalates to a three-trial matched A/B. The deletion waves: serve
flags 66 → 56 → 53 → 49, env names 38 → 33, every deleted flag's off-arm
carrying a dated losing verdict.

## What would be done differently

Build the positive control into the gate harness from the first arm, not
per-arm after a vacuous pass ships. The needle gate's thinking-model fix
(2026-07-24) and lever_gate's loud skips (2026-08-24) both landed after silent
passes had been trusted. A gate that cannot distinguish "passed" from "did not
run" should not exist, and the harness should enforce that once, centrally,
instead of each arm rediscovering it.
