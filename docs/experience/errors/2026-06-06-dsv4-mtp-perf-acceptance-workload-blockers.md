# DSv4 frozen-KV MTP — CORRECT but no perf win yet: low draft acceptance + degenerate greedy workloads are the real blockers (not the verify kernel)

## Context

frozen-KV MTP spec-decode reached **correct + wired** (hybrid: per-token scalar
verify attention + batched MoE/comm; Gate-A mismatches=0, scoped scalar-verify with
FlashMLA-base). ckl relaxed the goal to an **opt-in flag** (not default-on); the bar
is "verify performance + correct + wired". Correct + wired: done. **Performance: not
a win** on any measurement so far.

## What the perf runs actually showed

| run | result |
|---|---|
| hybrid scalar-verify, num_draft=4, "Natalia" GSM8K | spec-OFF 39.0 → spec-ON **20.9 tok/s (slower)**, accept_length **~1.40** |
| A1 per-token, earlier needles | accept/reject ~2/3 → **~40%** on coherent-repeating text |
| SGLang DSA MTP (reference) | accept_length **~2.7** (~68% per-token) |

## Root causes (the perf blockers — ordered)

1. **Draft acceptance is too low (~20-40% vs SGLang's ~68%).** This is the
   fundamental blocker — amortization is `~(1+accept)/cost`; at ~40% the verify
   overhead isn't covered. The low acceptance is NOT just the workload: even on
   coherent repeating text ("the capital of France is Paris …") it's ~40%. So the
   **MTP draft quality** (the `mtp.0` head forward, the depth-4 chain drafting,
   num_draft/topk tuning) needs investigation — fixing the verify kernel won't help
   if the draft is wrong.
   - **UPDATE 2026-06-06 (env-gated draft-vs-actual dump, executor.rs):** the
     **depth-1 head is GOOD — 96.875% (31/32)** on a deterministic 64-token greedy
     loop (only mismatch step=3). So the `mtp.0` head forward + the basic
     `mtp_forward` formula are NOT the blocker. The ~20-40% acceptance collapse is
     in the **multi-draft chain feedback (num_draft=4)**, not depth-1 — if MTP perf
     is ever un-parked, look at the chain drafting / KV-feedback path between draft
     steps, not the head. (MTP is parked per the single-request-fundamentals reset;
     this just closes the dangling "draft quality" question.)
2. **Every test workload is degenerate under greedy.** The dsv4_parity harness feeds
   raw token IDs (no chat template); greedy decoding loops/degenerates on real
   prompts (Natalia GSM8K loops even spec-OFF), so the acceptance numbers are
   confounded. A coherent workload (chat template + a prompt the model answers
   without looping) is needed to measure real acceptance — but greedy degeneration
   makes this hard, and spec-decode requires greedy.
3. **Verify-attention speed is marginal.** Scalar per-row verify amortizes only at
   break-even even at high acceptance; the faster FlashMLA per-row verify has a
   causal bug (row-0 sees future rows' SW window in the batched context — the
   per-row FlashMLA must use causal SW ≤ row r). The s_q=K FlashMLA path is dead
   (call-param-invariant divergence; kernel = upstream byte-identical).

## Status / Decision

The spec-decode VERIFY is correct and cleanly wired (a flag). The perf win is a
multi-layer effort with uncertain payoff: (draft quality) + (coherent greedy
workload) + (FlashMLA per-row causal fix). Meanwhile the **certain** perf levers are
designed and waiting: residual-wo→DeepGEMM (decode 14%), prefill MLA-LoRA→DeepGEMM
(prefill 30.2%), comm one-shot (16%), the throughput sweep (#34).

## Rule

**Spec-decode amortization is gated by DRAFT ACCEPTANCE first, verify-kernel speed
second.** Measure acceptance on a COHERENT workload (chat template, non-degenerate
greedy) before optimizing the verify path — a correct verify over a poor draft is a
slowdown. And greedy-degenerate prompts give uninformative acceptance: confirm the
base (spec-OFF) output is coherent before trusting any accept_length number
([[feedback_correct_inference_not_baseline_identity]] — needle + coherence gate).
