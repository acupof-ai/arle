# The rollout engine re-profiles its KV pool after the student is resident

## Context

Re-running the agent-OPD rollout A/B after `4850d0dd7` raised
`mem_fraction_static` from a hardcoded 0.2 to a `--rollout-mem-fraction` flag
defaulting to 0.5.

The first task ran correctly — 2 samples, 10 and 13 agent turns, 26–29K
sequences, one clean group row:

```
gpu_busy_secs 96.995  rollout_secs 105.261  busy_frac 0.921
prompt_tokens 278892  completion_tokens 3622  rollout_tok_per_sec 34.41
```

Task 2 onward produced nothing. Each burned its full 183 s wall on
`API Error: 500 engine thread closed; cannot submit`.

## What actually happened

The rollout engine profiles its KV pool **twice**, and the two profiles see
very different cards.

| profile | when | free VRAM | reserve at F=0.5 | pool |
|---|---|---:|---:|---|
| 1 | engine construction, before the student loads | ~68 GB | 48.9 GB | real |
| 2 | prefix warm-up after round 0, student resident | **45.1 GB** | 48.9 GB | **floor** |

`reserve = total × (1 − F)` is a constant — 0.5 × 97871 MiB = 48935 MiB on the
H20. Free at the second profile is 46180 MiB (measured: 51691 MiB used of
97871). `46180 − 48935 < 0`, so the profile lands on
`PROFILE_KV_TOKENS_FLOOR = 4096` by construction, for the same reason 0.2 did
at the first profile. Raising the default moved the cliff; it did not remove
it.

The formula double-counts for a co-resident caller. `free_bytes` already
excludes everything resident — the student, its optimizer, the engine's own
weights. Subtracting a fixed share of **total** on top charges the engine for
memory it is not using, and the charge grows as its co-tenant grows.

Three defects chain from there:

1. **The re-profile is taken at the structurally worst moment.** Free VRAM is
   at its minimum right after the student lands, which is exactly when the pool
   gets resized.
2. **The floor path builds an inconsistent pool.** The device pool shrinks to
   the floor while the host pool keeps its requested size:
   `TokenKVPool::mirror_slot page 256 out of range 256 (host pool total_pages
   exceeds device pool budget?)`. That is a panic, not a degraded mode.
3. **agent-OPD treats a dead engine as recoverable.** It logs `prefix warm-up
   failed (continuing)` and continues into a state where no request can ever
   succeed. Three of four tasks ran to their full timeout producing zero
   tokens, and the run had to be killed by hand.

## Fix

Not yet written. The shape:

- A co-resident profile should bound against **free**, not against `total × (1
  − F)` — the reserve exists to leave activation/scratch headroom, and that is
  a function of the engine's own working set, not of the card.
- A profile that lands on `PROFILE_KV_TOKENS_FLOOR` must resize the host pool
  to match, or refuse to resize at all and keep the previous pool.
- `engine thread closed` is terminal. The rollout loop should abort the round,
  not spend the remaining task budget on requests that cannot be served.

Interim workaround for measurement: `--task-limit 1`, which finishes the group
before the second profile is taken.

## Rule

**A VRAM profile is only valid at the moment it is taken, and re-profiling a
long-lived pool re-runs the sizing arithmetic against a card that has changed
underneath it.** Any sizing formula that mixes a `free` reading with a `total`
constant will drift as co-tenants grow. Check what else is resident at every
point the profile runs, not just the first one.

**A warning that names a knob without printing the measurement cannot be acted
on.** The `KV pool collapsed to the 4096-token floor at mem_fraction_static
0.5` warn added in `4850d0dd7` fires at WARN while the free/total/cell numbers
sit in the adjacent INFO line, which is suppressed at the default log level.
The diagnosis needed `nvidia-smi` and arithmetic to recover numbers the process
already had. Put the operands in the message that fires.
