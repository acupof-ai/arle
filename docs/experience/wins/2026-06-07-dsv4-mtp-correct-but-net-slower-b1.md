# DSv4 MTP/EAGLE: correct (greedy-identity, 82–88% accept) but NET SLOWER at B=1 — verify doesn't amortize

## Context

MTP/EAGLE spec decode (`ARLE_DSV4_SPEC_DECODE`, default off) is the structural lever
toward ~6ms/token. Measuring it required a spec-aware decode driver — the example
harnesses drove the model 1-token/step with hand-computed `kv_seq_len`, incompatible
with spec's variable-token advance (executor returns accepted base_next + bonus and
sets `pending=bonus`). Fixed the `dsv4_resident_ab` driver to consume all emitted
tokens, materialize the host pool before submit (`materialize_plan_kv`), and size the
host pool to the design max (length-agnostic). Also threaded `max_seq_len` as a
parameter (not an env the executor reads — runtime-config-as-CLI-flag rule).

## What worked / what was measured

8×H20 TP=8, `flashmla_fused_wqkv` variant, same binary, two prompts, spec off vs on:

| prompt | SPEC=0 tok/s | SPEC=1 tok/s | accept/total | Δ |
|---|---:|---:|---:|---:|
| needle (passcode 73914) | 39.81 | 35.33 | 14/17 (82%) | **−11%** |
| capital-of-France | 38.39 | 33.80 | 15/17 (88%) | **−12%** |

**Correctness: PASS.** SPEC=1 output is **byte-identical** to SPEC=0 on BOTH prompts
(greedy-identity — the multi-prompt gate per `feedback_spec_decode_gate_needs_multi_prompt`).
MTP draft+verify+accept is logically correct (`[dsv4-mtp] accept_total=14 reject_total=3`).

**Speed: NET SLOWER at B=1**, despite 82–88% accept. Cost model: each spec step =
`mtp_forward` (draft) + `forward_tokens_verify([pending, draft])` (a 2-token forward).
At B=1 the verify over 2 tokens costs ~2× a 1-token decode (per-token processing, no
weight-read amortization), so 1.82 accepted-tokens/step ÷ ~2×-cost step ≈ 0.9× — a
net loss. The per-token-greedy ms/token rose 25.1 → 28.3 (needle).

## Rule

- **MTP's win is gated on a BATCHED verify.** The K+1-token verify forward must
  amortize the 149GB weight read (memory-bound: a 2-token forward should ≈ a 1-token
  forward), exactly like batched decode. Until `forward_tokens_verify` batches the
  draft+verify tokens (the same Phase 5/6 batched-attention/MoE levers), MTP at B=1 is
  net slower — **do not default-on `ARLE_DSV4_SPEC_DECODE`**. MTP is not a standalone
  6ms lever; the batched multi-token forward is its prerequisite.
- The earlier "#33 MTP 1.9× decode lever" claim does not hold under this clean
  same-binary A/B; license MTP only after the batched verify lands and re-measures > 1×.
- Measure spec decode with an EXECUTOR-driven loop (consume accepted+bonus, let the
  executor manage KV) — the 1-token/step harness can't drive it and silently stops
  after one spec step.
