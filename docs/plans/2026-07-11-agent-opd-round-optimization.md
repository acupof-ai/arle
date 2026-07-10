# Agent-OPD training round — speed-up plan

> Status: Active

One anchored workload, one pattern, levers ranked by return-over-cost. Do them
top to bottom; each row links its wins/errors entry when done.

## Anchored workload

Production run = `scripts/agent_opd_curve.sh` full-run:

- corpus `staged-run1` (real swe_smith + synthetic), student Qwen3.6-27B-FP8
- rollout is serial, one stream at a time (`agent_opd.rs:473`)
- 16 rounds, 2 samples/prompt, 8 turns, 768 max-tokens, eval-n 24

All measurements below are one H20 GPU, this config (shorter A/B variant where
noted). Numbers are per round; round wall = 122.4 s after the two shipped wins.

Round split: rollout 42% · writeback 33% (backward 24.5%) · eval 20% · rest 5%.

## Pattern: trade memory for speed on short sequences

When a trajectory is short, its activations fit in VRAM, so the runtime can drop
two memory-savers that cost time:

- host offload of gradient checkpoints (a host round-trip that stalls the GPU)
- recomputing the forward pass during backward

Long trajectories keep both and stay safe. The switch is the sequence length.
This is the whole family of cheap wins here; apply it, don't hand-tune per run.

## Shipped

| lever | win | how |
|---|---|---|
| DSpark on serial decode | rollout −29%, eval −30% | already default on; drafts land (net speed-up + draft logs) |
| seq-adaptive checkpoint offload | backward −36% at short seq | `writeback_offload_for_seq()`: offload only when seq ≥ 4096 |

Together: **round −30.1%**, quality unchanged (eval 4/4).
`wins/2026-07-11-agent-opd-dspark-decode-and-seq-adaptive-offload.md`.

## To do, best return first

1. **Skip recompute on short sequences** — −7.5% round, no quality cost.
   Backward spends 30.7% (2.4 s/call) recomputing the forward. `should_checkpoint`
   (`qwen35.rs:2601`) already skips checkpointing when activations fit free VRAM,
   but the writeback forward still checkpointed at seq≈1300. Find why (was
   gradient-checkpointing forced on, or is the writeback path unconditional?),
   make it honor the memory check. Same seq switch as the offload win.

2. **Eval less often** — −5% round. `--eval-every 2 → 4`. Pure config, no code.
   Cost: half as many points on the capability curve. Confirm that's acceptable
   for the run, then set it.

3. **Train only the top half of layers** — −12% round, but trades capability.
   `--lora-layer-start 32` detaches the tape at layer 32 (`detach_before_lora_layer`,
   `qwen35.rs:3453`), so backward runs 32 layers instead of 64 — mechanism
   confirmed. Gate: does training half the layers still learn as well? Needs a
   pass-rate/needle A/B before shipping. Not free.

## Killed / deferred

- **Bigger checkpoint groups** — killed. Backward is not recompute-bound; the
  group size only changes offload count, and offload is already off. The
  backward's real cost is the LinearAttention scan (47%, a sequential recurrence).
  `errors/2026-07-11-agent-opd-backward-is-linearattention-scan-bound-group_size-dead.md`.
- **Faster LinearAttention backward** — deferred. It's already a device kernel;
  the cost is the delta-rule recurrence, which can't parallelize along the
  sequence. Ceiling ~6% round for a 2× kernel, high effort.
- **Concurrent rollout** — killed. Serial DSpark already wins at 2 samples.
- **Train a stronger drafter head (P3)** — biggest single win (−16%, helps both
  rollout and eval) but project-scale, out of this round's scope.
