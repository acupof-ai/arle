# DSpark-for-DSv4 full-committed-sequence context seed — landed, but accept still ~0 (second fuse bug)

> Status: **MEASURED 2026-07-14** — the seed WORKS (context accumulates the whole
> prompt) but accept is still ~0: `1/145 = 0.0069` on stable `/v1/completions`
> targets (was 0/170). A correct, necessary structural step — NOT a regression
> (opt-in `--spec-type dspark`; DSpark-DSv4 was already 0). A SECOND bug remains:
> the context fusion produces near-uniform per-position context. See "Measured" below.

## Context

DSpark draft for DeepSeek-V4-Flash had accept ≈ 0 on stable targets even after the
[value-tail inverse-RoPE fix](2026-07-13-dspark-dsv4-value-tail-inverse-rope.md):
`base_argmax` (pure forward) collapsed to a fixed attractor-token set, never
tracking the target. Two authoritative angles converged on the cause:
- **Internal audit**: every sub-step inside `dspark_stage_forward` is faithful to
  the validated `mtp_forward_level`; the divergence is a MISSING path — the draft
  captured only the per-block anchor tap; the prompt prefix + within-block
  committed tokens are never in the context.
- **DeepSeek DeepSpec reference** (Qwen3DSpark; DSv4 not open-sourced) + the paper
  (HyperDFlash, arXiv 2606.26744): the drafter attends the **FULL committed
  sequence** (per-position depth-fused taps, KV-cached across blocks). Ours
  attended ~1 sequence position → context-starved → attractor collapse.

The earlier "window is not the lever" KILL was measured BEFORE the inverse-RoPE
fix (a doubly-broken forward) → confounded/void (§0.1). With the value-tail bug
gone, the full context IS the lever.

## What was implemented (deletion-style refactor)

ONE canonical `dspark_append_context(draft, df, taps, rows, start_abs)` path
(`dsv4/dspark.rs`): fuses `rows` committed tokens' `n_taps` taps
(`main_proj`+`main_norm`), then appends `hc_mult` compressed-latent rows per token
to every stage's `latent_kv`, **each token RoPE'd at its own absolute position
`start_abs+j`**, storage offset `(start-ctx_base)` fully decoupled (the b350b0f90
invariant that regressed a prior attempt — held here, verified line-by-line).
- **Prefill** captures the whole chunk's multi-row taps (`[stream_dim, seq_len]`
  per target layer, transient) → seeds the prompt-prefix context.
- **Decode** appends the anchor tap (`rows=1`, positions all `start_pos`) through
  the same path — reduces to the exact prior single-token geometry.
- `dspark_forward_block` stripped of its inline context fuse + `taps` param
  (now embeds noise + attends the accumulated `latent_kv` + exit only).
- VRAM: `latent_cap = max_seq_len*hc_mult + block` (each token caches `hc_mult`
  lanes); transient prompt-tap buffer excluded from the ledger.

## Measured (2026-07-14, TP=4 GPUs 3-6, conf-threshold 0, raw /v1/completions)

- The seed FIRES and accumulates: `[dspark-stat] context rows=20/24/40` (prompt_len ×
  hc_mult=4) confirmed in the log — the whole committed sequence is now the context.
- **But accept ≈ 0 still: `1/145 = 0.0069`** (was 0/170). `base_argmax` still
  attractor-collapses ({24132, 112434, …}); outputs still degenerate.
- **Root cause of the residual failure — near-uniform context.** Every context row
  has `row_l2 ≈ 5.64` (the `main_norm` output magnitude) and `row_spread ≈ 0.25`
  across 40 rows — only ~2.5× the within-token spread (0.10). The context barely
  distinguishes positions, so the (now-present) full context is INEFFECTIVE.

## Open — SECOND fuse bug (the wide-HC-tap reduction)

Our fuse LANE-SPLITS the wide HC tap: per committed token it makes `hc_mult` context
vectors, each from ONE HC lane (`fuse_in` column `c=j*hc_mult+r`, main_proj over one
lane's tap-concat) → each context vector sees only 1/hc_mult of the pre-collapse HC
residual → weak, near-uniform. The DeepSpec/HyperDFlash reference REDUCES the wide
pre-collapse HC residual to ONE vector per position via the target's `hc_head` gate
(Eq 1: `α=σ(W_f·RMSNorm(vec(H))+b)`, `y=Σ αⱼ Hⱼ`) BEFORE the drafter — encoding the
FULL stream, one context vector per token (main_proj `[hidden, n_taps*hidden]` then
takes the n_taps reduced vectors). Next: replace the lane-split with an
`hc_head`-gate reduce (reuse `head_hidden_from_stream`) per tap per position →
`n_taps` reduced vectors → main_proj → ONE context latent per committed token (not
`hc_mult`). This also shrinks the latent cache back to `max_seq_len + block`
(the `×hc_mult` sizing was a consequence of the lane-split).

## Rule

A DSpark/DFlash-family draft must attend the FULL committed sequence's context
(prompt prefix + every committed token), not a single tap position — context
starvation degenerates the frozen-head draft to a fixed attractor set (accept≈0)
that mimics a weight bug. Confirm the draft's context KV holds ≈ sequence length
positions, not a handful of taps.
