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

No backward lever is risk-free. The runtime already picks checkpoint-vs-resident
by a memory estimate, and training fewer layers trades quality — each needs one
check before shipping.

1. **Eval less often** — −5% round, zero risk. `--eval-every 2 → 4`. Config only.
   Cost: half as many points on the capability curve. Set it if that's fine for
   the run. This is the only change with no measurement gate.

2. **Skip recompute on short sequences** — −7.5% round, no quality cost, but needs
   an OOM check first. Backward spends 30.7% (2.4 s/call) recomputing the forward.
   The CLI forces checkpointing on (`train_cli.rs:2550`); `should_checkpoint`
   (`qwen35.rs:2601`) then keeps it because its estimate (a ×3 headroom) says the
   activations won't fit at seq≈1300. Real residency is ~1/3 of that, so the
   margin is likely too conservative — but the ×3 may cover the backward's own
   peak. Gate: cut the margin at short seq, sweep production trajectory lengths,
   confirm no OOM. Then it's the offload win's twin.

3. **Train only the top half of layers** — −12% round, trades quality.
   `--lora-layer-start 32` detaches the tape at layer 32 (`detach_before_lora_layer`,
   `qwen35.rs:3453`), so backward runs 32 layers instead of 64 — confirmed. Gate:
   does training half the layers learn as well? Needs a pass-rate A/B first.

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
