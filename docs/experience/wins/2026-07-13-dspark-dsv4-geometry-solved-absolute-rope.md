# DSpark-for-DSv4 draft geometry SOLVED — absolute-position RoPE (accept 0 → 0.14)

> **SUPERSEDED — OVERTURNED 2026-07-13** by
> [errors/2026-07-13-dspark-dsv4-draft-context-blind-at-gating-pos](../errors/2026-07-13-dspark-dsv4-draft-context-blind-at-gating-pos.md).
> The `0.143` was noise from ONE France case (token 11111). Full-block
> multi-prompt attribution shows 0/300 accept: the draft is context-blind at the
> gating position. Geometry is NOT solved; the "context window" here is NOT the
> lever. Read the errors entry, not the conclusion below.

## Context

DSpark speculative decode for DeepSeek-V4-Flash ran correct + lockstep under TP=4
([lockstep win](2026-07-13-dspark-dsv4-tp4-lockstep-fix.md)) but every draft was
rejected — **accept_rate 0.0**, drafts semantically UNRELATED to the target argmax
(`[dspark-dbg]` dump: `anchor=603 drafts=[68745]` vs `target=[671]`). Two cheap
hypotheses were disproven by pod A/B, both no-ops on the drafts:
1. output post-attention inverse-RoPE (`f36675d85`, reverted `88360c888`) — no
   counterpart in the SGLang DFlash reference.
2. absolute-position shift alone — pure relative RoPE, a constant base shift
   doesn't change scores.

## Root cause

The draft RoPE'd the context K **and** the noise block Q/K at **draft-local
`latent_kv` buffer positions** (`ctx_end` cursor + `hc_mult` stride), so the
context→block RELATIVE offset the draft attended was the buffer stride
(`hc_mult` + accumulated cursor), not the true `1`. Every attention score was
scrambled → the draft hidden was garbage → tied-head argmax unrelated to the
target. The SGLang DFlash reference (`dflash_worker.py`) is explicit: the
draft-cache length is for KV-slot allocation ONLY, **never** RoPE — RoPE runs in
the absolute target-position frame.

## What Worked

Decouple the RoPE position frame from the `latent_kv` write offset (commit
`b350b0f90`). Thread `block_abs` (= executor `verify_pos = start_pos + 1`) through
`dspark_forward_block → dspark_stage_forward / dspark_append_latent`:
- noise block Q/K RoPE at absolute `block_abs + [0..block)`;
- each step's committed-token context at absolute `block_abs - 1` (all `hc_mult`
  rows share it — the HC-mapping choice, empirically validated here);
- `latent_kv` write offset `(slot - ctx_base)` unchanged.

The context→block relative offset is now the true `1`.

### Result (GPUs 3-6, TP=4, `The capital of France is`, greedy, max_tokens=16)

`[dspark-dbg]` (the decisive line): `drafts=[11111, 84941]`,
`target_argmax=[11111, 1, 978]` → **drafts[0]=11111 == target[0] → accepted=1**.
The draft now proposes tokens the base verify agrees with.

- `/v1/stats spec_decode`: **`accepted:1, drafted:7, accept_rate:0.143`** (was 0.0).
- Output coherent (`" Paris."`, finish_reason `stop`). Geometry SOLVED.

Eliminated non-bugs along the way (all pod- or Explore-verified): output de-RoPE
(none in ref), tied-head tensor (draft reuses `self.lm_head`, proven by correct
MAIN output), wide-tap capture (`main_proj [hidden, 3*hidden]` folds wide taps
per-HC-row exactly like the proven MTP `h_proj`), non-causal kernel (already).

## Open — accept 0.14 is geometry-correct but context-limited

The remaining rejected blocks are the **single-token-context limit**, NOT a
geometry bug: only the anchor-forward token seeds a context entry per step; the
prompt prefix and within-step accepted-draft tokens don't. Lifting accept toward
the reference's 60–85% needs the **per-committed-token context window** — a trunk
forward-path change to capture per-token 3-layer taps at prefill + verify (the
shipped qwen35 DSpark track has the pattern: multi-row `taps` target threaded
through the stream forward + a post-verify `dspark_append_ctx`). That is the
follow-on lever; geometry is no longer the blocker.

## Rule

Spec-decode draft attention over a cache: **RoPE positions are the ABSOLUTE model
frame; the draft-cache buffer index is a SEPARATE concern (slot allocation only).**
Conflating them corrupts the relative offset RoPE encodes → garbage drafts that
look like a model/weight bug but are pure geometry. Under relative RoPE a constant
absolute shift is a no-op — only the context↔block RELATIVE offset moves scores;
verify the fix changes that offset, not just the base. Attribute at token level
(draft id vs target argmax) — a single aligned match (accept≥1) proves geometry
before chasing accept magnitude.
