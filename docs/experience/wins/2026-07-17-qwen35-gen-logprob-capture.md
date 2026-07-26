# Qwen3.5/3.6 generation-time behavior-logprob capture → authoritative ratio denominator

> Status: Shipped — capture/F.6 measured on H20 (2026-07-18); the
> denominator flip and fail-closed replay contract were verified on H20
> offline replay and a real online stochastic update on 2026-07-26; see
> [2026-07-26-agent-rft-sidecar-denominator](2026-07-26-agent-rft-sidecar-denominator.md).

## F.6 verdict (H20 f6d, 2026-07-18)

First real sampled ratio-floor — `exp(logp_recompute − logp_sidecar)` over 1406
tokens where both the V0 bf16 recompute and the generation-time sidecar logp
exist:

```
ratio_floor_mean=1.0011  ratio_floor_max=5.2769  (1406 tokens)
kl_rollout=1.46e-2  clip_frac=0.025  trained=10
```

**FP8-serve vs bf16-train per-token numerics gap is ~0.1% in the mean, heavy
tail to 5.28× (exp 1.66).** The sidecar behavior-logp is safe to use as-is: the
importance ratio needs no numerics correction, only the existing clip
(clip_frac=0.025 already absorbs the tail). This LICENSES the IS-ratio flip from
V0 recompute onto the sidecar logp (kills the recompute forward — see the
release/ensure round-boundary crash below that the recompute path exposed).

**Precondition that made this non-zero:** the serve default temperature must be
> 0 for the cc lane (`--rollout-temperature`, default 1.0 →
`set_default_temperature`). Claude Code omits `temperature`; the old serve
default 0.0 made every rollout greedy → `logprob:None` → empty sidecar →
`ratio_floor_tokens=0`. See is_greedy root-cause (fed715dc3).

**VRAM guard held:** 34 SKIP (seq>23K `--max-update-seq` wall, 9ed5143e1), 10
trained, no OOM; backward survived seq 22256 (461s). But 34 skip / 10 train =
77% of the cc corpus discarded by the 23K cap — data efficiency, not safety, is
now the binding constraint (raise-cap lane / #50).

## Context

Train-side `capture_rollout_logprobs` RE-COMPUTES π_behavior at V0 under bf16
train numerics and assumes θ unchanged since generation — both wrong in general
(FP8 serve vs bf16 train; staleness>0 later). P6 captures log p_filtered
(committed token) at generation time on every Qwen3.5/3.6 commit path and rides
it through the serve into the `.tokens.json` sidecar.

## What Worked

- **Zero new kernels, zero new syncs.** Every sampled commit path already
  crossed D2H: plain decode host-samples over full logits
  (`infer_plan::sample_token_logprob`, new twin returning the filtered logprob);
  MTP/DSpark chain accepts materialize the filtered `p` rows on device and D2H
  the verdict with a sync — the logprobs are k+1 four-byte reads after that sync
  (`qwen35.rs chain_commit_logprobs`). Greedy emits `None` (delta policy) — the
  sidecar skips greedy requests. No per-slot device ring was needed: the ring
  design assumed on-device sampling, but the actual sampled tail is host-side
  and never graph-captured.
- **Plumbing rides the existing token path exactly:** `SlotToken.logprob`
  (already in the seam) → engine token observer → `StreamItem`/`PendingTokens`
  → `RelayCompletionDelta.logprobs` (serde-default, both local and multiproc
  lanes) → `TokensSidecar.gen_logprobs` (all-or-nothing vs `gen_token_ids`).
- **Consumer:** cc-convert threads sidecar logprobs into
  `CcRecord.gen_logprobs` (one per masked token, mask order = target order).
  The 2026-07-26 contract binds this vector directly to every ratio-weighted
  denominator. The earlier recompute-vs-sidecar `ratio_floor_*` diagnostic and
  its redundant forward were removed after F.6 licensed the flip.

## Gate

Capture and the F.6 comparison are complete. The authoritative-denominator
acceptance evidence is recorded in the 2026-07-26 follow-up; no throughput
claim is attached to deleting the redundant recompute forward.

## Rule

Before designing a capture mechanism, locate where the value already crosses
the D2H boundary — a sampled path that host-samples full logits gives the
behavior logprob for free; only device-resident distributions need a gather at
the existing sync point.
