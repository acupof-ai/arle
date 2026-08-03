# T6 GDN decode kernel: −2.8% decode ITL — CUDA, 2026-08-03

> Status: **Shipped, default path** (#196 T6). c=1 W8A16 decode ITL p50
> **20.77 → 20.19 ms**; cumulative vs pre-#196 baseline **26.88 → 20.19
> (−24.9%)**. Greedy byte-identical (accumulation order unchanged); graphed
> lane intact (17 captures / 4100+ replays).

## What shipped

Two changes to `gated_delta_rule_decode_kernel`, both aimed at the same
measured fact — the kernel took 17 µs/layer against SGLang's 6 µs for the
identical recurrence:

- **State stays in registers between the two passes.** Pass 1 decayed the
  `[32, val]` tile and wrote it back; pass 2 read it again, added the rank-1
  update, and wrote again. The tile now lives in a `float s_reg[32]` across
  both passes: **one state read + one write per step instead of two of
  each.** State traffic is essentially the whole kernel (48 heads ×
  128×128 f32 = 3.1 MB read+written per step per layer).
- **Grid 48 → 96 blocks.** Each head's 128 value columns split across 2
  blocks (256 threads each), so a 78-SM H20 gets ≥1 block per SM instead of
  leaving 30 idle. Q/K loads move to the first `key_dim` threads (the split
  makes the old `j_slice == 0` mapping cover only half the row).

Arithmetic is unchanged — same order of the same f32 operations — so greedy
output is byte-identical.

## Learnings

**A prior baseline note said "widening the grid is not the lever" for this
kernel; that was wrong at c=1 decode.** It was reasoned from the prefill
varlen form, whose `grid(heads, batch)` already fills the GPU, and from the
recurrence being a latency chain along the *token* axis. But the decode step
has no token axis to shorten — a single token per step — so the only axes
left are exactly the ones the note dismissed: grid width and memory traffic.
Together they returned 0.58 ms. The two changes shipped as one tranche and
were not A/B'd apart; the register-caching half is the one with a clean
mechanism (halved DRAM traffic on the dominant term).
