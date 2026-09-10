# Speculative decoding is a correctness-preserving transform

Design note 2 of 5 ([plan](../plans/2026-09-02-design-theses.md)). Metal and
CUDA, Qwen3.5 / Qwen3.6. Date: 2026-09-09.

## Problem

A coding-agent server on an Apple Silicon Mac serves one large model at
concurrency 1. Decode is bandwidth-bound: the 27B 4-bit checkpoint reads
~18 GB per step, the M4 Pro moves 273 GB/s, so the physical ceiling is
~15.2 tok/s and the framework sat at 12.3. No kernel or framework tuning
crosses that ceiling — the only lever is the algorithm: read the weights once
and emit several tokens per forward. Speculative decoding is that lever, but
it is conventionally sold as an *approximate* speedup, and the concurrency at
which it stops paying is almost never published. The workload that exposes
both halves: c=1 agentic decode on Metal, and the H20 serving regime where the
same model serves 8–16 concurrent agent streams.

## Standard practice and where it fails

The standard treatment: a small draft model proposes a span of tokens; the
target verifies them in one forward; the longest matching prefix is accepted.
Accepted output is described as "close to" the target — an approximation you
trade for speed. Two failures follow on this workload.

First, approximation is the wrong contract. A coding agent's correctness gate
is greedy parity with the non-speculative server: the same prompt must produce
the same tokens, or the feature is a silent quality regression. Without a
token-exact parity gate, a broken verify or rollback corrupts output
invisibly. The claim here is stronger and machine-checked: the accepted
tokens are target tokens by construction, so speculative output equals greedy
output token for token, and `scripts/spec_parity.py` fails loudly if that ever
stops holding.

Second, the published number is a c=1 number. The verify forward is free only
while the GPU has idle compute to spend on it. Once batching starts paying —
the regime a serving engine actually runs in — per-request draft generation
serializes and the sign flips. The standard papers do not publish that
concurrency; it has to be measured, and here it was measured twice.

## The design

**The draft proposes; the verify owns correctness.**
`qwen35_speculative_block`
([`dflash.rs:960`](../../crates/infer-metal/src/dflash.rs), called from
[`executor.rs:1116`](../../crates/infer-metal/src/executor.rs)) runs one draft
block, verifies it on the target, accepts the longest matching prefix, and
rolls the KV and recurrent state back past the first mismatch. Every emitted
token is a target token; the draft can only change how many tokens a forward
produces, never which tokens.

**The block drafter rides the prefix cache — it stores nothing across a
prefix boundary.** The draft KV is reset at the end of every block
(`DFlashDraftState::reset`,
[`dflash.rs:853`](../../crates/infer-metal/src/dflash.rs)), so a restored
prompt has no draft state to recover. The target hidden state the draft needs
is re-seeded by the tail prefill: every prefix match is capped at
`prompt_len - 1` ([`prefix.rs:114`](../../crates/infer-core/src/prefix.rs)),
so a matched prompt always re-prefills a non-empty tail, and the draft state
is seeded on first sight of the restored prompt
([`executor.rs:935`](../../crates/infer-metal/src/executor.rs)). A shorter
first-block draft context costs acceptance, never correctness.

**The regime is a measured boundary, not a default.** On H20, DSpark measured
2.24× at c=1 and −39% at c=16 against the same engine's plain decode
([errors 2026-07-26](../experience/errors/2026-07-26-dspark-spec-decode-serializes-and-loses-above-c1.md)):
draft generation ran per request, so concurrency added latency without adding
throughput. The lever that finding named — batched draft generation across
slots — landed at [`dspark.rs:174`](../../crates/infer-cuda/src/qwen35/dspark.rs),
and the dispatch ladder now routes batched greedy DSpark on BF16 pools and
falls to plain decode above `--spec-max-batch`
([`qwen35.rs:2511`](../../crates/infer-cuda/src/executor/qwen35.rs)). A
KV-format gate is a statement about the verify kernel, not the draft: lifting
the BF16-only gate to run batched DSpark over quantized KV lost from c=8 and
was rejected twice on measurement, so the gate stays
([`qwen35.rs:2493`](../../crates/infer-cuda/src/executor/qwen35.rs),
[errors 2026-08-22](../experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md)).

## The failure

The c=1 win shipped as a latency feature with no concurrency axis. Three
block sizes — 16, 8, 6 — were swept at c=1/8/16, and all three converged to
one decode line at c=16. That convergence was the symptom: the axis under
test had stopped being connected to the cost, because the draft path
serialized while plain decode scaled 2.9× from c=1 to c=16. The 2.24× said
nothing about c=8, and re-scoping the feature cost a second measurement
cycle. The rule that entry left: before tuning a knob inside a feature,
measure the feature against *not using it* at the target operating point.

## The number

Metal, M4 Pro, `mlx-community/Qwen3.6-27B-MTP-4bit` head, depth 3, temp 0,
same-session A/B
([wins 2026-06-21](../experience/wins/2026-06-21-metal-qwen36-mtp-spec-decode.md)):

| Config | decode tok/s |
|---|---:|
| Baseline (no draft) | 12.30 |
| Spec (MTP head, d3) | 17.75 (+44%) |

The 15.2 tok/s bandwidth ceiling breaks because the weights are read once per
three emitted tokens. Acceptance 68.8% (2.375 of 3 per block). H20, DSpark
block drafter, c=1
([wins 2026-07-11](../experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md)):
2.39–3.14× (45.8 → 109.5 tok/s at ~50-token context; 32.1 → 100.9 at ~3K).
The boundary: net loss from roughly c≥4, −39% at c=16; batched over quantized
KV, −10% to −19% at c≥8. Correctness gate: `scripts/spec_parity.py` — N
prompts greedy with and without `--draft-model`, bar is zero token-id
mismatches, with a negative control (baseline at temperature 0.3) that proves
the gate can go red. The gate's only run (2026-09-09) went red on the default
Metal pairing (Qwen3.5-0.8B + r3lax DSpark): 7 of 8 prompts diverged, first
divergence at token 13-50. That is the gate working — it caught a real DSpark
Metal parity drift, still open, in which the target's batched verify forward
and per-token decode forward disagree on the GDR recurrent state
([errors 2026-09-09](../experience/errors/2026-09-09-dspark-draft-metal-parity-drift.md)).
No Metal draft pairing has passed zero-mismatch; the gate is the admission
test a pairing must pass before it is used in a benchmark or a default.

The gain region has a second boundary, independent of concurrency: drafter and
target mismatch. On Qwen3.8-27B-NVFP4, same card, both without speculation,
tileRL decodes 92.4 tok/s against our 84.5
([baselines](../baselines.md); tileRL
`docs/experience/wins/2026-08-28-decode-split-by-occupancy.md`). We cannot run
speculation on that checkpoint at all: DSpark draft acceptance is 13% at c=1
and 0% at c=8
([roadmap Goal 1](../plans/2026-08-24-roadmap.md), item 2; suspected mRoPE
mismatch, untested). The comparison case is tileRL's own spec suite on the
same model — net loss at every measurement point, 0.58× at B=1 depth 2 and
0.76× at its best, B=8 depth 4
(`tileRL/docs/experience/wins/2026-08-29-spec-decode-net-win.md`). A drafter
the target does not agree with makes the verify step pure overhead; the regime
ends there regardless of idle compute.

## What would be done differently

Publish the c-sweep in the same table as the c=1 license, from the start. The
2.24× was real and the feature ships at c=1, but the missing concurrency axis
is what made the batched-quantized-KV attempt worth trying at all — the two
rejections in
[errors 2026-08-22](../experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md)
were spent rediscovering a boundary the first sweep could have drawn. The
parity gate should also be a merge gate, not a script: a spec-decode change
without a green `spec_parity.py` run does not land.
