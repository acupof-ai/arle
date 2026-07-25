# DSpark self-RL: first closed loop on TC-27B, and the three things that were broken

> Status: loop **runs end-to-end and is measured**. Acceptance effect: **neutral**
> at 13–18 steps. Not licensed, not killed — the run is trainer-throughput bound,
> see Open.

## Context

`--dspark-train` is self-RL in the strict sense: the experiences are the serve's
own draft blocks, the reward is its own verified acceptance
(`accepted / block_size`), and no external data or labels enter anywhere. Asked
to actually run it and lock the algorithm, on ThinkingCap-27B
(`/host/ThinkingCap-Qwen3.6-27B-FP8`) + the Qwen3.6-27B DFlash draft, one H20,
greedy, 8 concurrent prompts.

Three separate defects had to be fixed before a single number meant anything.
Each one would have produced a *plausible-looking* run — falling loss, saved
checkpoints, no errors — that taught nothing.

## What Worked

**1. The trained head was never persisted.** `run_loop` hot-swapped into the live
engine and dropped everything at shutdown. Fixed with `--dspark-train-out` (bf16
safetensors in the draft loader's own tensor frame, write-then-rename) plus
`--dspark-markov-init` + `load_markov_head` to install one back. A loose file
copied into an indexed draft dir would *not* have loaded: `loader.rs:864` opens
only the shards `model.safetensors.index.json` lists.

**2. There was no head to train.** The pod's Qwen3.6-27B draft is a DFlash
backbone — `fc` fusion, `hidden_norm`, `norm`, 58 tensors, zero `markov_head.*`
— so `head.markov` was `None` and `update_markov_weights` would have failed on
*every* publish while the loss curve fell convincingly. `--dspark-train` now
materializes the head (`[248320, 256]`, `w2 = 0` so the bias is identically zero
and an untrained head is an exact no-op; `∂bias/∂w2 = w1[c] ≠ 0` keeps it
trainable). Verified in the log: `mode=dflash-backbone` → `mode=dspark-sp+markov`,
and publishes now succeed — 254 MB checkpoint at step 8, no failed hot-swap.

**3. The policy gradient had the wrong sign for hundreds of steps.** With
`baseline_init = 0.5` against a true mean reward of ~0.33, `r - baseline` is
negative for *every* sample until the EMA walks down — at `alpha` 0.01, ~340
steps. The log proves it without any modelling: `accept_ema` fell monotonically
on all 13 steps, and a monotonically falling EMA means the mean reward was below
the baseline at every step. So the PG term spent the entire run pushing down the
log-prob of every drafted token, accepted or not. `baseline_init` is now
`Option<f32>`, `None` = seed from the first batch's mean reward.

### Three arms, interval acceptance (Δaccepted/Δdrafted, not the cumulative counter)

| arm | head bias absmax | interval accept_rate | verdict |
|---|---:|---|---|
| lr 1e-4, baseline 0.5 | 0.0039 | 0.2039 … 0.2039, flat | head 250× too small to flip an argmax |
| lr 1e-2, baseline 0.5 | 0.42 | 0.2028 → **0.1985** | head bites; wrong-signed advantage costs acceptance |
| lr 1e-2, baseline seeded | 0.47 | 0.2016 … 0.2062, flat | decline gone; no gain either |

The first arm is the reason a null result needs a magnitude check: nothing moved
because the trained bias was ~3.9e-3 against top-2 logit gaps of O(1). Raising
the lr made the *pre-existing* baseline defect visible — the middle arm is the
only one that showed a regression, and fixing the baseline removed it. That
sequence is the A/B that identifies the bug.

## Rule

- **A falling loss curve is not evidence the loop is closed.** Two of the three
  defects here produced falling loss with zero effect on the serve. Verify the
  write reaches the consumer — grep the publish path for failures, and check the
  artifact exists — before reading any curve.
- **Null result? Check the magnitude before the mechanism.** "Acceptance didn't
  move" and "the parameter is 250× too small to move it" are the same
  observation, and only the second one tells you what to do next. Read the
  trained weight and compute the quantity it must reach.
- **A monotone EMA is a sign test.** A baseline that only ever falls is proof its
  advantage was one-sided throughout. Cheaper than instrumenting the gradient.
- **Seed a baseline from data, never from a midpoint guess.** `0.5` looks neutral
  for a ratio in `[0, 1]` and is not: the true mean was 0.33, and the gap
  becomes a uniform push away from whatever the policy currently does.

## Open

Acceptance is unimproved, and the run cannot yet say whether that is the method
or the budget. The budget looks decisive: 13 steps × 64 = 832 experiences
consumed against 17299 draft chains produced — **the sidecar sees ~5% of its own
data**, because CPU autograd on a `[248320, 256]` head costs ~1 s/step while the
serve produces ~20 chains/s. The ring buffer drops the rest. A 127M-parameter
head trained on 832 samples is not a tested method.

Two ways forward, in order: (a) run hours not minutes, which is cheap to start
and answers it directly; (b) cut the per-step cost — the trainer densely updates
all 248320 rows of `w1` when a batch touches at most a few hundred condition
tokens, so a sparse-row update is the obvious ~1000× on that factor. Do (a)
first; only (b) if the long run says the signal is real but slow.

ISO's fixed-spectrum retraction does **not** apply to this configuration:
`w2 = 0` has no base spectrum to preserve, so it belongs to adapting DSv4's real
DSpark head, not growing one here. `iso_spectrum::matrix_roots` fails loudly on a
zero factor rather than silently producing garbage.

Also fixed en route: `scripts/pick-gpu.sh` treated a **zombie** claim holder as
live. A killed run reparents to pid 1, which in this container does not reap, so
its `/proc` entry outlives it and the GPU claim leaked permanently — two launches
died on "no free GPU" against eight idle GPUs. State `Z` now counts as stale.
