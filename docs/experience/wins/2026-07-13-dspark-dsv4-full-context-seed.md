# DSpark-for-DSv4 full-committed-sequence context seed — `pending-remote` (built, UNMEASURED)

> Status: **UNVERIFIED** — built clean (BUILD_EXIT=0) but accept_rate NOT measured;
> the pod serve never reached engine-ready (274GB cold load exceeded the 57-min
> barrier under co-tenant disk-IO contention, 2026-07-13). Re-measure when the box
> is idle before trusting any accept number. Opt-in path (`--spec-type dspark`),
> production defaults untouched.

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

## Verification status

- Local typecheck clean (`cargo check -p infer-api … cuda,no-cuda`).
- Geometry reviewed end-to-end: stream-copy layout ↔ fuse index `c=j*hc_mult+r`
  ↔ per-row RoPE positions — consistent; decode reduces to the validated baseline.
- **Accept_rate: NOT measured** (pod infra block above). Decisive probe when
  unblocked: raw `/v1/completions` (stable targets), watch `[dspark-dbg]`
  `base_argmax` start varying + tracking `target_argmax`, `[dspark-stat]` `context
  rows=` grow to prompt-len×hc_mult, and spec_decode delta accept > 0.

## Rule

A DSpark/DFlash-family draft must attend the FULL committed sequence's context
(prompt prefix + every committed token), not a single tap position — context
starvation degenerates the frozen-head draft to a fixed attractor set (accept≈0)
that mimics a weight bug. Confirm the draft's context KV holds ≈ sequence length
positions, not a handful of taps.
