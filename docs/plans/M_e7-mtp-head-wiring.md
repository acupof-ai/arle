# M_e.7 — Native Multi-Token Prediction (MTP) head wiring for Qwen3.5/3.6

**Owner:** ckl · **Status:** ⚠️ blocked on canonical model 2026-05-07
**Track:** Metal scheduler / qwen35 spec · **Predecessor:** M_e.4 SwiGLU compile-fusion

## ⚠️ Blocker — current Metal canonical model has no MTP tensors

Inspected `mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint (the canonical
Metal model per `AGENTS.md`):
- 2090 tensors total
- **0 tensors named `mtp.*`**
- `model_type=qwen3_5_moe`, architecture `Qwen3_5MoeForConditionalGeneration`
- Multimodal: `vision_tower.*` keys present, `image_token_id`,
  `video_token_id` set
- `text_config.mtp_layers` and `num_mtp_layers` both **absent**

→ **The MTP head wiring plan can't run against this model.** Either:

1. **Switch canonical model** to a Qwen3.5-MoE text-only checkpoint
   that ships MTP heads (e.g. the upstream `Qwen/Qwen3.5-30B-A3B`
   or whatever mlx-community quantizes that ships `mtp.*`).
2. **Add a second canonical model** for "text-only with MTP" alongside
   the multimodal Qwen3.6-VL canonical.
3. **Defer M_e.7** until checkpoint availability is sorted.

This is a **prerequisite check** — landing the implementation against
the current canonical would have failed at weight load. Catching it
during planning prevents an L-effort waste.

Original design retained below for the eventual implementation tick.

## Goal

Wire the **native MTP heads** that Qwen3.5/3.6 already ship in their
HuggingFace checkpoints (`mtp.*` tensors) so each forward generates
multiple tokens. Predicted ~40% c=1 ITL reduction (~23 ms → ~14 ms) at
acceptance ≥0.95 on chat workloads. **The single biggest expected
Metal ITL win** in the post-M_e.4 / post-oMLX-C v3 lever stack.

## Why this is the biggest miss

Per the parallel research subagent (this date), 2026 H1's bleeding-edge
Apple Silicon LLM serving projects (`vllm-mlx` waybarrios, `omlx`
jundot) have **all** wired native MTP — and they cite it as their
biggest decode-ITL win. ARLE's `qwen35-spec` strips the `mtp.*` tensors
during weight load today, so ARLE pays the full per-step encoder cost
for every single token. With MTP, one forward emits K tokens (K=2-3 is
typical), amortizing the 23 ms encode over multiple committed tokens.

Reference implementations:
- `omlx` commit `696d90a` (2026-05-06): runtime monkey-patches
  `mlx-lm#990` (AirRunner) into the model __call__.
- `vllm-mlx` (waybarrios): `--mtp` runtime flag.
- mlx-lm PR #990: native AirRunner — verify head sits on top of the
  base model, draft heads are extra Linear layers fed by the base
  hidden state.

## Architecture sketch (from research)

Qwen3.5/3.6 checkpoints contain:

- Base transformer body (already loaded by ARLE).
- **`mtp.*` tensors**: a small set of Linear projections that take the
  base model's last hidden state and project to additional speculative
  draft logits. Typically 1-2 extra heads.

Inference pattern (per mlx-lm AirRunner):

```
hidden = base.forward(input_ids)     # gives logits[:, -1, :] AND mtp_input
y0     = sample(logits[:, -1, :])    # standard next-token
y1     = sample(mtp_head_0(mtp_input))  # +1 speculative token
y2     = sample(mtp_head_1(mtp_input)) # +1 speculative token (if 2 heads)
# Verify y1, y2 against subsequent forwards; accept while contiguous matches.
```

**Verify pattern is the simple concat-verify mlx-lm uses for spec
decode** (see `mlx-lm/generate.py:685-700`): no
`forward_with_hidden_states` needed. The base model just runs forward
on `[committed, y1, y2]`, and we accept the prefix where logits match
the speculative samples.

## Implementation steps

1. **Stop stripping `mtp.*` tensors.** Find ARLE's `qwen35-spec` weight
   loader (likely `crates/qwen35-spec/`); the strip lives there. Allow
   `mtp.*` keys through the contract.
2. **Add MTP head load to `Qwen35MetalWeights`** at
   `crates/mlx-sys/src/mlx_qwen35_model.cpp` — load the additional
   Linear projections.
3. **Extend `qwen35_compiled_step` / `qwen35_compiled_step_batch_packed`**
   to optionally emit (logits, mtp_logits[]) — gated on whether MTP
   heads are loaded.
4. **Scheduler-side verify path**: after each step, if MTP samples are
   present, queue the *next* forward over `[committed, mtp_y1,
   mtp_y2]`. Compare `argmax(committed_logits[i])` against
   `mtp_y[i]`. Accept matched prefix; advance KV cache; resample on
   first mismatch.
5. **Env gate**: `INFER_METAL_MTP=1` for opt-in; default off until
   matched-A/B confirms the win.
6. **Path probe**: `M_E7_MTP_PROBE` once-fire log on first MTP-aware
   step.

## Composition with shipped levers

- **oMLX-C v3** (host pipelining): synergistic. MTP doesn't change
  the per-step shape; the pipelining still works. The `prev_sampled`
  array becomes a vector of MTP samples.
- **M_e.4 SwiGLU compile-fusion**: synergistic. MTP head's MLP can
  use the same compiled-shapeless SwiGLU helper.
- **Auto-wired-limit**: orthogonal. MTP heads are tiny (extra ~50 MB
  on a 35B model); doesn't change the wired-limit calculation.
- **DFlash with tiny draft (M_e.6 viability validated)**: arguably
  redundant. MTP gives ~40% per-step amortization; tiny-draft DFlash
  predicted ~50% (2× speedup). They could STACK if MTP heads' draft
  quality is lower than tiny-draft's, but unlikely worth the
  complexity. **Pick one; MTP first since the heads are already in
  the checkpoint.**

## Acceptance bench

`scripts/bench_guidellm.sh qwen36-mtp-c1` — Qwen3.6 35B-A3B-4bit at
c=1, max_tokens=256, chat-style prompts (HumanEval or
chat_template_qwen workload).

Expected:
- c=1 ITL p50 ≤14 ms (currently ~23 ms; ~40% reduction at ≥0.95
  acceptance which is typical for natural language).
- TTFT unchanged (MTP only affects decode).
- Acceptance rate logged separately; if <0.7 on representative
  workload, the win shrinks proportionally.

Matched A/B per
`feedback_matched_ab_for_small_bench_effects.md` at 2 sessions before
flipping default ON.

## Risks

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | `mtp.*` tensors don't actually exist in mlx-community/Qwen3.6-35B-A3B-4bit (might be stripped during quantization) | First step: `huggingface-cli scan-cache` + grep tensor manifest. If absent, use a vanilla checkpoint or skip M_e.7 in favor of M_e.6 dense-draft DFlash. |
| R2 | mlx-lm AirRunner's monkey-patch approach is fragile; ARLE wants a clean Rust + C++ FFI integration instead | Don't port AirRunner verbatim — read its semantics, implement directly in `mlx_qwen35_model.cpp`. |
| R3 | Acceptance rate <0.7 on tool-use workloads (where outputs are structured JSON/code, less predictable) | Document as a workload-specific win; default OFF for code/tool routing, default ON for natural-language chat. Possibly a per-request hint. |
| R4 | MTP heads pre-trained for older Qwen3.5 may not match Qwen3.6 distribution exactly | Verify against ground-truth: run with MTP off, capture token sequence; run with MTP on; assert generated tokens match (greedy mode). |

## References

- mlx-lm PR #990 (AirRunner native MTP):
  https://github.com/ml-explore/mlx-lm/pull/990
- omlx commit `696d90a` (2026-05-06): native MTP wiring
- vllm-mlx README: https://github.com/waybarrios/vllm-mlx (`--mtp` flag)
- Verify pattern source: `mlx-lm/generate.py:685-700`
- Predecessor lever stack:
  [`docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md`](../experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md)
