# DSpark prob-match loss normalized per-token, not per-vocab-class, 2026-07-28

> Status: correctness fix (the term now produces a live gradient); default `α=0.5`
> unchanged; acceptance A/B is 7b → pending-remote. This changes the effective
> objective balance, so it is NOT a default-safe perf claim — it needs the A/B.

## Goal

Make the DSpark probability-matching term contribute a real gradient at
production vocab scale, instead of underflowing f32 to zero.

## Context

The DSpark head objective is `loss = (1−α)·PG + α·prob_match`. The H20 ISO
premise sweep found `prob_match` inert: at `α=1` (pure PM) loss was 0.0000 with no
learning; at `α=0.5` the PM half contributed ~0. Root cause (errors entry
2026-07-27): the loss was `Σ_{rows,vocab}(Δsoftmax)² / (weight_sum · vocab_size)`
— a squared-L2 distribution distance (a **sum** over vocab, since the intended
surrogate is TV = ½Σ|p−q|) mis-normalized as a per-class **mean** over 248,320
classes, driving the term and its gradient under f32 precision.

## What changed

`dspark_train.rs`: drop the `1/vocab_size` factor, normalize per token only
(`1/weight_sum`). The loss now equals the mean-over-tokens `‖softmax(draft) −
softmax(target)‖²` it was documented to be (line 504), whose natural scale is
O(TV) ~ 0.01–1 — comparable to PG's ~0.4, so `α` is a live mix. The serve-frame
oracle test (`dspark_trainer_serve_frame_and_alignment`) is updated to the same
normalization (it had encoded the old `/(wsum·V)` reference — it correctly caught
the change).

## Results

```text
test_dspark_train: 8 passed (oracle re-pinned to per-token PM)
train lib: 180 passed ; CUDA/no-CUDA typecheck + fmt: clean
```

## Problems

This is an objective change: `α=0.5` was effectively pure-PG (PM ≈ 0) and is now
a genuine 50/50 mix. Whether the live PM term **improves speculative acceptance**
is unmeasured — 7b's gate ("a loss is licensed only if acceptance improves")
applies. Pending-remote: matched acceptance-rate A/B at `α = 0 / 0.5 / 1` with the
corrected loss. It also unblocks the ISO premise sweep's `α=1` arm (previously a
null discriminator). Until the A/B runs, the default stays `α=0.5` because that is
the historical default, not because the new balance is licensed.

## Learnings

**A distance defined as a sum over classes must not be normalized as a mean over
classes.** TV / squared-L2 between distributions is O(1) by construction; dividing
by the vocabulary size reinterprets it as a per-class average and, at a 250k
vocab, buries it under f32 precision. The knob (`prob_match_alpha`) looked live
but weighted a dead term — the same silent-no-op class as an unwired CLI flag.
Measure the term's gradient norm, not just its coefficient in the loss.
