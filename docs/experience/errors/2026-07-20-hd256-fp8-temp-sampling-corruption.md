# hd256 q/k RMSNorm convention flip (b4b293f0c) — one kernel bug, an eight-hypothesis false chase

> Status: ROOT-CAUSED + FIXED (`e4d5580ca`). The agentic-rollout / long-context
> degeneration is a **single kernel bug**: `b4b293f0c` flipped hd256 q/k RMSNorm
> from OFFSET `x·rms·(1+w)` to STANDARD `x·rms·w`, shrinking 27B q/k ~3× and
> collapsing attention at length. Binary bisect on the agentic greedy rollout is a
> clean adjacent flip. Fix restores `(1+w)` at all 5 hd256-only sites; verified:
> base greedy agentic rollout emits proper tool calls, 7 turns, fixes the real
> bug, reward 1.0. **A separate temp>0 sampling defect survives the fix — see the
> tail + [[project or #59]].**

## Context

Agentic-OPD rollouts on Qwen3.6-27B (ThinkingCap-FP8 student) produced no edits /
degenerate output. A prior session's relay (#48) said "b4b293f0c breaks temp>0" —
**correct from the start.** The chase below spent eight hypotheses re-discovering
that, because b4b293f0c passed its own smoke (short greedy needle/Fibonacci
coherent — the damage is length-dependent) and the symptom shape (temp-graded,
multilingual salad) pointed everywhere but the kernel.

## Root Cause

`b4b293f0c` ("hd256 q/k RMSNorm convention mismatch — OFFSET→STANDARD") changed
5 hd256 kernel sites from `x·rms·(1+weight)` to `x·rms·weight`, arguing the 27B
q/k_norm weights are STANDARD (mean 0.2–0.75, <1) and citing the 4B hd128 model as
verification. Both premises are wrong:

1. **The weights are OFFSET.** Measured on Qwen3.6-27B-FP8: q/k_norm `mean|w| =
   0.490` (min 0.225, max 0.761), all 34 tensors < 0.75. A weight centered at 0.49
   (not 1.0) is stored as a delta from 1 — the norm multiplier is `(1+w) ≈ 1.49`,
   not `w ≈ 0.49`. The Metal reference (`mean|w| < 0.75 → w+1`) agrees exactly.
2. **"Verified against 4B" is a red herring.** The hd256 kernels are 27B-only —
   the fused batched kernel is guarded by `head_dim == BATCHED_DECODE_HEAD_DIM ==
   256`, and the `*_hd256` helpers are hd256-only. The 4B (hd128) model never
   executes these kernels, so it can't validate them.

Dropping the `+1` shrank each q/k component ~3× → attention scores collapsed.
**Length-dependent:** short prompts survive (greedy needle + Fibonacci coherent →
b4b293f0c's smoke passed), long context degenerates (agentic drift, temp>0 salad).

**Binary bisect — clean adjacent flip** (base Qwen3.6-27B-FP8, greedy agentic
cc-harness rollout; pre-`fed715dc3` the rollout is greedy by default):

| commit | rollout |
|---|---|
| `67e15b0a6` (= b4b293f0c^, still `(1+w)`) | GOOD — 5 turns, `<tool_call>` Grep/Read/Bash, fixed the sqlparse bug, reward 1.0 |
| **`b4b293f0c`** (`w`) | BAD — degenerate salad, zero tool calls |
| `3bdcbfa84` (= a41827b75^) | BAD (so a41827b75 is exonerated for THIS bug) |

## The eight-hypothesis false chase (each killed by measurement)

The symptom (temp-graded multilingual salad, only on hd256/FP8) misdirected the
investigation through, in order: (1) sampler plumbing → refuted (hd128 coherent
same binary); (2) MoE router FP8-quantized → **no routers** (hybrid linear-attn;
loose grep of `.mlp.gate_proj` as `.mlp.gate`); (3) FP8 scales → bit-identical to
base; (4) FP8 values/clipping → faithful 2.65% floor; (5) config/rope/template →
identical to base; (6) ThinkingCap weights → base salads identically; (7)
temperature default (`fed715dc3` greedy→temp>0) → **greedy salads too** at length;
(8) prompt/render regression → `prompt_token_ids` byte-identical old vs current.
Only after all eight did a binary bisect land on the kernel. The norm mis-fix
`9851ced6b` (input/post layernorm `w−1`) was a *different* wrong turn, reverted
`485eefe0d`.

## Fix

`e4d5580ca` — restore OFFSET `(1+weight)` at all 5 hd256 q/k sites
(`decode_prep_paged_hd256.cu`, `prefill_attention_hd256.cu`,
`prefill_attention_paged_prep.cu`, and the 2 inline fused-batched sites in
`fused_attention.cu`). 27B-only; 4B/hd128 untouched; keeps b4b293f0c's separate
(correct) MTP `pre_fc_norm` load fix.

**Verified on pod** (isolated build, sm_90 nvcc rebuild, HEAD e4d5580ca): base
Qwen3.6-27B-FP8, greedy agentic rollout → proper `<tool_call>` Glob/Grep, 7 turns,
located + fixed the real `lexer.py is_keyword` bug, hidden tests pass, reward 1.0.
Matches the GOOD bisect parent. **Agentic-OPD unblocked at greedy.**

## Open — a SEPARATE temp>0 defect (#59): CONFIRMED, two independent causes

The fix restores the GREEDY/argmax path but temp>0 still degenerates AFTER it —
confirmed by a **sha-verified** probe on HEAD `9edfcb234` (product binary sha ==
on-disk build; source has OFFSET `(1+w)` at all 4 sites; temp=1.0 → SCRAMBLED).
This **refutes** the tempting "temp>0 collapses into b4b293f0c / earlier report was
a stale binary" hypothesis. There are **two independent causes**:

1. **Type-A = `b4b293f0c`** (hd256 q/k RMSNorm OFFSET→STANDARD) — broke greedy-at-
   length AND temp>0; **fixed at HEAD**. Garbage `funciton/Fibonaacci/
   _selection_selection_`. Isolated cleanly (`67e15b0a6` OFFSET=COHERENT vs
   `b4b293f0c` STANDARD=SCRAMBLED, byte-identical through `a41827b75`).
2. **Type-B = a second, temp>0-specific tail bug that PERSISTS at HEAD** (different
   garbage `fkk fkk`, early-stop ~117 tok). Proven by two controls that flip to the
   SAME garbage-B: `a41827b75`+OFFSET-overlaid, and sha-verified HEAD. Localized by
   a zero-rebuild param sweep: **top_k=1/greedy → COHERENT** (bug lives in the
   sampled low-prob **tail**, not attention/argmax); **COHERENT at temp≤0.7,
   SCRAMBLED at temp=1.0** (temp<1 sharpens away from the bad tail); top_p 1.0 vs
   0.95 no diff → not a top_p renorm bug. Regression window `67e15b0a6..a41827b75`,
   survives to HEAD; **not** the norm and **not** `a41827b75`'s `sample_token →
   sample_token_logprob` rewrite alone (byte-identical under STANDARD). Suspects:
   `qwen35.rs`/`prefix_state.rs` attention-prep, `d94cf4b80` (rejection-sampling).
   Second bisect (OFFSET held fixed, one variable) in flight.

**Actionable:** on-policy grpo runs at **temp≤0.7 (coherent)** as the interim;
temp=1.0 recovered after the Type-B fix. Key new fact: the tail was clean at
temp=1.0 in `67e15b0a6` and got poisoned by one of the 10 following commits — a
real **regression**, not an immutable FP8-logit-tail property.

## Rule

- **When the same symptom class has bitten before, TEST the prior claim first.**
  #48's relay named `b4b293f0c` on day one; eight hypotheses later a bisect
  confirmed it. A cheap `git revert <named-commit> + rebuild + A/B` would have
  cost one build, not a multi-probe forensic tower. The named prior suspect earns
  the first experiment, not the last.
- **Binary bisect beats forensic inference for a "worked before, broken now" +
  deterministic symptom.** CPU forensics (scales/values/config/prompt) can only
  find a smoking gun; they never *clear* a hypothesis. One same-prompt greedy A/B
  across the commit window localized in 3 builds what 8 static probes could not.
- **A "fix" is a suspect.** b4b293f0c and `9851ced6b` were both *fixes* that
  regressed; `b4b293f0c` even had a passing smoke. Length-dependent damage escapes
  a short-prompt gate — gate kernel-numerics changes on the SLO shape (long
  agentic / long generation), never greedy-short alone.
- **Convention from the stored weights, not from a sibling model.** mean|w|=0.49
  < 0.75 = OFFSET; the 4B hd128 "verification" never ran the hd256 kernels. Verify
  a convention against the tensors the kernel actually consumes.
- Prior rules still hold: reproduce the premise on a clean binary before a chain
  ([[feedback_reverify_premise_on_clean_binary_before_chain]]); tensor-name/arch
  ground truth over loose grep ([[feedback_validate_comparison_inputs_before_bug]]).
