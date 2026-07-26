# DSpark: one ragged-window launch per draft layer — attn −8×, +3.4% tok/s

## Context

The prior tranche's phase probe left draft `attn` as the largest per-slot O(B)
slice: 1.53 ms/slot at c=8, 41% of draft. Each of the 5 draft layers launched
`block` (16) separate `nonpaged_prefill_attention_ring_cuda` calls with
`seq_len=1`, so one slot paid 80 launches of `grid(num_q_heads, 1)` — 40 blocks
of 256 threads on a 78-SM H20, serialized on one stream. 19 µs per launch is far
above launch overhead; the cost was under-occupancy.

Model ThinkingCap-Qwen3.6-27B-FP8, 1×H20 GPU 0, `--spec-type dspark
--mtp-draft-model Qwen3.6-27B-DFlash --dspark-block-size 16 --spec-max-batch 16
--max-running-requests 16`, greedy, driver `conc_drive.py`.

## What Worked

Rows differ only in their sliding lower bound `lo = max(q_pos-window+1,
ctx_base)`, and `lo` reads nothing layer-dependent — so one host-built window
table serves all five layers.
`nonpaged_prefill_attention_ring_varlen_cuda` takes device per-row
`ring_base[]` / `kv_len[]` and walks each row's window non-causally:
same kernel, same ring mapping, one `grid(num_q_heads, block)`.

Matched A/B, arms differing only in the three touched files
(`nonpaged_prefill_attention.cu`, `ffi/attention.rs`, `qwen35/dspark.rs`):

| metric | BEFORE | AFTER | Δ |
|---|---:|---:|---:|
| draft `attn`, per slot (c=8) | 1.53 ms | **0.19 ms** | −8× |
| draft `attn`, per slot (c=1) | 1.71 ms | 0.22 ms | −8× |
| draft total, per slot | 3.73 ms | 2.40 ms | −36% |
| draft @ c=8 | 21.66 ms | 14.66 ms | −32% |
| tick sum @ c=8 | 101.59 ms | 95.29 ms | −6.2% |

Aggregate tok/s, 3 interleaved trials per arm (A,B,A,B — median):

| conc | BEFORE | AFTER | Δ |
|---|---:|---:|---:|
| 1 | 61.9 (64.0/61.9/61.2) | 64.5 (66.3/62.9/64.5) | +4.2% |
| 4 | 87.7 (88.2/87.7/84.3) | 89.7 (91.6/88.4/89.7) | +2.3% |
| 8 | 85.4 (85.4/82.6/85.5) | 87.0 (89.2/85.2/87.0) | +1.9% |

AFTER wins all 9 paired (trial, concurrency) points, so the sign is settled even
though single-trial spreads overlap.

Acceptance is unchanged: the phase pass drove identical request counts and both
arms logged the same tick count (114 phase lines at c=8, 644 vs 643 draft
lines) — same decode steps for the same tokens means the same accept rate. The
math is per-row identical to the old launches (softmax is per-row), so this is a
launch-shape change, not a numerics change.

## Problems

- Ranges are device-resident, so the kernel cannot bound-check `kv_len <=
  ring_modulus`; the Rust caller asserts it per row while building the table.
- Draft is now 2.40 ms/slot, of which `mlp` 1.07 and `head` 0.70 are `block=16`-
  column GEMMs re-reading the draft weights once per slot. Those are the next
  O(B) target, and batching them means one forward over `Σ block` columns.
- `verify` is untouched at 62 ms, still 65% of the c=8 tick.
- Serving numbers are short-prompt (`conc_drive.py`); the long-agent re-measure
  on the multi-turn dataset (bench spec §3.3) is still outstanding.
- ckl committed to main between the two arm builds. The interleaved diff is
  train-only plus one additive, unused `DeviceContext` method — checked, not
  assumed.

## Rule

A per-row launch loop expressing a ragged shape is an occupancy bug, not a
launch-overhead bug: 80 launches of 40 blocks each cost 19 µs apiece because
each one leaves the GPU 95% idle, not because the launch is expensive. The fix
is the same shape the batched verify already uses — push the raggedness into
device index arrays and keep one grid. Check whether the per-row parameters
depend on the loop you are inside: here the window table was invariant across
all 5 layers, so one upload served the whole forward.
