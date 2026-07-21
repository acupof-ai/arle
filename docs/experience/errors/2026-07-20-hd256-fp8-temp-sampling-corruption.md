# RMSNorm convention flip (b4b293f0c) — TWO bugs in one commit, both fixed

> Status: ROOT-CAUSED + FIXED. `b4b293f0c` carried **two independent** wrong
> changes; the fix landed in two commits. **Type-A** (`e4d5580ca`): the kernel half
> — flipped hd256 q/k RMSNorm from `(1+w)` to `w`, shrinking 27B q/k ~3× and
> collapsing attention at length (greedy-at-length + temp>0). Fix restores `(1+w)`
> at the 5 hd256 `.cu` sites. **Type-B** (this session): the load half — the same
> commit added a `w-1` transform (`load_final_norm_offset`) on the final RMSNorm
> weight in `qwen35.rs`; the `(1+w)` kernel is correct and untouched, but feeding it
> `w-1` sign-corrupts the STANDARD final-norm's negative channels → flattened logits
> → temp=1.0 sampled-tail garbage (greedy survives). `e4d5580ca` fixed only the
> kernel half, so Type-B persisted to HEAD. Fix = revert both sites to `load_vec`.
> Both verified on pod (temp=1.0 + greedy COHERENT). **temp=1.0 grpo unblocked.**

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

## Type-B — a SEPARATE temp>0 defect (#167): ROOT-CAUSED + FIXED

`b4b293f0c` carried **two** wrong changes; `e4d5580ca` reverted only the kernel
one. The Rust half survived at HEAD and is a distinct temp>0 corruption:

1. **Type-A = `b4b293f0c`'s 4 `.cu` sites** (hd256 q/k RMSNorm OFFSET→STANDARD) —
   broke greedy-at-length AND temp>0; **fixed by `e4d5580ca`**. Garbage
   `funciton/Fibonaacci/_selection_selection_`.
2. **Type-B = `b4b293f0c`'s one-line `qwen35.rs` change** — the **main-model +
   MTP-head final RMSNorm** load switched from `loader.load_vec(...)` (raw) to
   `load_final_norm_offset(...)` (`w → w-1`), on the premise "the trunk kernel
   applies `(1+w)`, Qwen3.5 norms are STANDARD, raw would double." **Both premises
   wrong for the final norm.** `e4d5580ca` never touched `qwen35.rs` → persists to
   HEAD. Garbage `MemoizacionGMEMIZATION… fkk fkk`, early-stop ~117 tok.

**Bisect (OFFSET held fixed, one variable, sha-verified each build):** every
overlaid commit `67e15b0a6..HEAD` and genuine HEAD `9edfcb234` emit the *identical*
`fkk`-117 output → the overlay faithfully reproduces HEAD. `67e15b0a6+OFFSET =
COHERENT` vs `b4b293f0c+OFFSET = SCRAMBLED` is the adjacent flip — same commit as
Type-A, but the Rust half.

**Mechanism, proven by reading the kernel + measuring the weights (not the
comment):** the final norm is applied by the **shared `rms_norm_offset` `(1+w)`
kernel** (not head-dim-gated, one path for all models). Its weight
`model.language_model.norm.weight` is **STANDARD (centered ~1)** everywhere — 27B
`mean|w|=0.962` (min −0.27), 35B-A3B `1.628`, 122B `1.292`. Feeding the raw
STANDARD weight through `(1+w)` is what the model expects (`load_vec` = COHERENT at
temp=1.0 + greedy, sha 1b24b8e3). `load_final_norm_offset` subtracts 1, so the
**negative channels flip sign** (min −0.27 → −1.27 → kernel `(1+(−1.27)) = −0.27`)
and the hidden is sign-corrupted → logits flattened. **Argmax ordering survives**
(greedy coherent); the flattened softmax poisons the sampled tail at temp=1.0.
Param sweep confirms: `top_k=1` COHERENT, `temp≤0.7` COHERENT, `temp=1.0
top_p={0.95,1.0}` SCRAMBLED.

**Fix (`d703b5240`):** revert both `load_final_norm_offset` sites to
`loader.load_vec(...)` (= pre-`b4b293f0c` behavior) and delete the helper.
STANDARD across all models → blanket revert is safe (no 4B/27B convention split;
the hd256 kernels were 27B-only, but the final-norm convention is uniform).
**Both sites verified on pod:**
- **Main norm (3326)** — base serve, 27B temp=1.0 + greedy both COHERENT, full
  600 tok, matches `67e15b0a6`.
- **MTP-head norm (8373)** — `--spec-type mtp` active: coherent output + **48.3%
  draft acceptance (~2.45 tok/step @draft=3)**; a corrupted MTP norm would give
  near-zero acceptance. `mtp.norm.weight mean|w|=1.27` = STANDARD, same as the
  main norm (0.96) → identical `load_vec` logic.

**temp=1.0 on-policy grpo unblocked** — no more temp≤0.7 interim.

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
- **One commit can carry two bugs; a partial revert leaves the quiet one.**
  `b4b293f0c` touched both a `.cu` kernel (Type-A, loud — killed greedy) and a
  `.rs` loader (Type-B, quiet — greedy survived, only temp>0 broke). `e4d5580ca`
  reverted only the `.cu` and shipped; Type-B rode to HEAD. When reverting a
  named regressor, diff its **full** change surface, not just the file matching
  the loudest symptom — `git show <sha> --stat` before a targeted revert.
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
