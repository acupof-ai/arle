# DSpark train sidecar — end-to-end verified (Phase 1 shipped)

## Context

The DSpark train sidecar plan Phase 1 wired an acceptance-weighted trainer into the `arle serve --spec-type dspark`
path: the inference hot path captures (draft_tokens, draft_logits, accepted)
tuples into a global buffer; a background thread drains it, runs a policy
gradient step on the Markov head, and hot-swaps the updated weights back into
the running engine.

## What worked

End-to-end verification on H20 (8×H20, CUDA), model pair:
- **Target**: `/host/Qwen3.6-27B-FP8` (64 layers, hidden 5120, vocab 248320)
- **Draft**: `/host/dspark-aeon` (DSpark draft with Markov head,
  `target_layer_ids=[1,16,31,46,61]`, `mode=dspark-sp+markov`)

| Check | Result |
|-------|--------|
| "DSpark train sidecar started" | Yes |
| `dspark_train: loss=` training steps | 6 |
| `train step failed` | 0 |
| `weight update failed` | 0 |
| Loss trend | −4.04 → −3.49 → −3.09 → −2.95 → −3.36 → −3.18 (decreasing) |
| Baseline EMA | 0.495 → 0.471 (adapting) |
| Batch sizes | n=1, then n=63–64 (buffer draining) |

Pipeline: experience capture → buffer drain → acceptance-weighted step → weight
hot-swap — all functioning, zero errors.

## Bugs found & fixed

1. **Hardcoded `vocab_size`** (`crates/train/src/dspark_train.rs`):
   `DsparkTrainConfig::default()` had `vocab_size: 151936` but Qwen3.6-27B has
   `vocab_size: 248320`. Every training step failed with index-out-of-bounds /
   shape mismatch. **Fix**: removed `vocab_size` from `DsparkTrainConfig`; the
   trainer lazily initializes Markov params from the first experience's actual
   `vocab_size`, making it model-agnostic at construction.

2. **Wrong draft model**: initially used `/host/Qwen3.6-27B-DFlash` which lacks
   `markov_w1`/`markov_w2` tensors (`mode=dflash-backbone`), causing "weight
   update failed: dspark head has no Markov head to update". **Fix**: switched
   to `/host/dspark-aeon` which has the Markov head (`mode=dspark-sp+markov`).

## Rule

- The trainer must be model-agnostic at construction: vocab size, hidden size,
  and layer count are all inferred from the first drained experience, never
  hardcoded.
- `--spec-type dspark` requires a draft checkpoint that actually contains the
  Markov head (`dspark-sp+markov`), not a backbone-only DFlash checkpoint.

## Post-verification fixes (2026-07-21)

A code review after the E2E run found two correctness bugs that silently
degraded training quality; both fixed before any acceptance-rate benchmark:

1. **L1 loss was dimensionally wrong** (`crates/train/src/dspark_train.rs`):
   the supervised loss computed `softmax(draft) − raw_target_logits` — mixing
   [0,1] probabilities with unbounded logits. The `target_probs_id` tensor
   (the actual `softmax(target)`) was computed but never used. With L1 weight
   0.9, 90% of the gradient signal was garbage. **Fix**: negate
   `target_probs_id` in-graph (`ops::mul_scalar(target_probs_id, -1.0)`) so the
   loss is `softmax(draft) − softmax(target)`. Also removed the dead
   host-side `neg_target_probs` Vec allocation.

2. **Trainer started from random init, discarding checkpoint weights**:
   `init_params` used sin/cos pseudo-random init, overwriting the engine's
   pre-trained Markov head on the first weight hot-swap. Acceptance would
   regress before recovering. **Fix**: added `get_dspark_markov_weights()`
   (mirrors `update_dspark_markov_weights` across the 6-file dispatch chain)
   to read the loaded checkpoint weights; the sidecar seeds the trainer from
   them as its first action (inside the spawned thread, so serve startup is
   not blocked by the D2H copy + sync).

Also added: gradient clipping (`max_grad_norm`, default 1.0, reuses
`crate::grad_clip::clip_grad_norm`), configurable `baseline_init`, and
replaced the 22-item manual tensor-free list with `free_new_except` snapshot.
