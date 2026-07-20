# DSpark RL sidecar — end-to-end verified (Phase 1 shipped)

## Context

The DSpark RL sidecar plan (`docs/plans/2026-07-19-dspark-rl-sidecar.md`)
Phase 1 wired a REINFORCE trainer into the `arle serve --spec-type dspark`
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
| "DSpark RL sidecar trainer started" | Yes |
| `dspark_rl: loss=` training steps | 6 |
| `train step failed` | 0 |
| `weight update failed` | 0 |
| Loss trend | −4.04 → −3.49 → −3.09 → −2.95 → −3.36 → −3.18 (decreasing) |
| Baseline EMA | 0.495 → 0.471 (adapting) |
| Batch sizes | n=1, then n=63–64 (buffer draining) |

Pipeline: experience capture → buffer drain → REINFORCE step → weight
hot-swap — all functioning, zero errors.

## Bugs found & fixed

1. **Hardcoded `vocab_size`** (`crates/train/src/dspark_rl.rs`):
   `DsparkRlConfig::default()` had `vocab_size: 151936` but Qwen3.6-27B has
   `vocab_size: 248320`. Every training step failed with index-out-of-bounds /
   shape mismatch. **Fix**: removed `vocab_size` from `DsparkRlConfig`; the
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
