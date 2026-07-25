# DSpark train sidecar: the trained head now survives the process

> Status: code landed, **run pending-remote** (TC-27B on the H20 box). Bench
> impact is confined to `--dspark-train` serves (default off).

## Context

Auditing the DSpark training loop before running it on ThinkingCap-27B
(`bottlecapai/ThinkingCap-Qwen3.6-27B-FP8`, qwen35 executor lane). The loop was
closed in-process — capture (`executor/dspark_train.rs`) → CPU autograd on the
Markov head (`train/src/dspark_train.rs`) → hot-swap
(`qwen35/dspark.rs::update_markov_weights`) — but had two defects that make a
real training run worthless:

1. **Nothing was ever written to disk.** `run_loop` hot-swapped into the live
   engine and dropped the weights at shutdown. A 6-hour run produced no artifact.
2. **Swap fired every step.** Each swap is `2 × vocab × rank` bf16 H2D plus a
   full `ctx.sync()` on the serve stream, issued from the sidecar thread. At
   vocab 152k / rank 256 that is ~160 MB and one hard serve stall per gradient
   step — and `drain()` returned whatever was buffered, so on an idle serve it
   stepped on 3-row batches and stalled decode for pure gradient noise.

## What Worked

- `DsparkTrainer::save_weights` writes bf16 safetensors under the **draft
  loader's own names** (`markov_head.markov_w{1,2}.weight`, `[vocab, rank]`), so
  the file overlays a draft checkpoint dir with no conversion step. Path comes
  from `--dspark-train-out`.
- `run_loop` accumulates into a full `batch_size` before stepping, and
  publishes (swap + checkpoint) every `swap_every` = 8 steps, plus once on exit.
  Default-off path is untouched; the train path trades 8× less serve stall for
  8 steps of weight staleness in the drafter, which only costs acceptance, never
  correctness (the trunk verifies every token).
- Gate: `dspark_trainer_saves_loadable_markov_head` asserts the names, `BF16`
  dtype, `[vocab, rank]` shape, and an f32→bf16 value round-trip — a silently
  renamed or transposed save is the failure that wastes a whole run.

## Rule

- **A training loop without persistence is a profiler, not a trainer.** Check
  for the write path before scheduling GPU hours.
- **Save in the consumer's frame.** The loader's tensor names and layout are the
  contract; a conversion script between trainer and loader is a second thing to
  keep in sync.
- **A background thread that syncs the serve stream is a hot-path cost.** Same
  class as [#183](../errors/2026-07-25-dspark-verdict-contaminated-by-train-sync.md):
  cadence it, don't do it per step.

## Known ceiling (not fixed here)

The sidecar trains **only** the Markov head — the rank-256 previous-token bias
on the draft's logits. The backbone (`mtp.0`, the 3-tap `main_proj` fusion, the
confidence head) stays frozen. For TC-27B the off-distribution part is exactly
that backbone: the DFlash draft was trained against *base* Qwen3.6-27B hidden
states, and TC-27B is an RL finetune that moved them. Expect the sidecar to
recover part of the acceptance gap, not all of it; closing the rest means
training the fusion, which this path cannot reach.
