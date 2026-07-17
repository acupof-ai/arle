# Qwen3.5/3.6 generation-time behavior-logprob capture → sidecar → F.6 ratio-floor

> Status: pending-remote — GPU gate: F.6 ratio-floor numbers on one collected
> group + needle unaffected, per plan §P6
> ([2026-07-16-agent-rl-unified-infra](../../plans/2026-07-16-agent-rl-unified-infra.md)).
> Sibling of [2026-07-17-qwen35-mtp-rejection-sampling](2026-07-17-qwen35-mtp-rejection-sampling.md).

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
- **Consumer + F.6:** cc-convert threads sidecar logprobs into
  `CcRecord.gen_logprobs` (one per masked token, mask order = recompute target
  order); the PG update emits `ratio_floor_mean/max/tokens` =
  `exp(logp_recompute − logp_sidecar)` stats whenever both sources exist. The
  IS ratio stays on the V0 recompute until F.6 licenses the flip.

## Gate (pending-remote)

F.6 ratio-floor row on one collected group (temp=1 rollout → cc-convert →
`--replay-records` PG update) + needle ×3 unaffected + baseline tok/s A/B
(logprob reads add ≤ (depth+1)·4 B D2H per spec step on an already-synced
stream; expected wash).

## Rule

Before designing a capture mechanism, locate where the value already crosses
the D2H boundary — a sampled path that host-samples full logits gives the
behavior logprob for free; only device-resident distributions need a gather at
the existing sync point.
