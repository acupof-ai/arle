# Sequence-parallel linear-attention core: global 262,144 trains on 4×H20 — 2026-08-22

> Status: Shipped

Measured on `ThinkingCap-Qwen3.6-27B-FP8`, H20 (97,508 MiB), `--synthetic-writeback-seq`,
binaries `sp12` … `sp16` (`b3fef817d` … `e632993f9`).

## Context

The previous entry closed with the a2a linear-attention core as the ceiling: under CP
it all-to-alls the sequence into the head axis and runs the recurrence on the **global**
sequence per rank, so its transient is O(global seq) and does not shrink with CP.
cp=2 could not reach 131,072 on 97 GB.

## What worked

Three changes, in the order they were needed.

**1. Sequence-parallel core with cross-rank state carry.** Rank `r` runs the recurrence
on its own rows only and hands `(final_state, conv window)` to the rank owning the next
global zigzag chunk. Transient becomes O(rank seq). Requires `d_initial_state` in the
gated-delta backward (the FlashQLA `fq` route already produces it) and a taped carry so
the state gradient crosses back.

**2. Layer param grads parked on host.** The top-level checkpoint arm exit offloads the
grads of that layer's leaf params (`tape.rs`, `offload_to_host`). LoRA f32 grads had
accumulated 8.5 GB across 64 layers — 133 MiB/layer, live until the optimizer step —
stacking on top of the core region's transient. The optimizer and grad clip already read
host-resident grads, so nothing else changed.

**3. The CP core became one tape entry.** The per-chunk-checkpoint form held O(rank seq)
duplicates inside the core region: row-slice copies of qkv/z, a packed `out‖state` per
chunk, the `cat` over chunk outputs, and an f32 grad for each — +13.4 GB at 65,536
rows/rank. It is now an untaped chunk loop writing rows into one `SeqAccum`, raw p2p for
the carry, and a reverse-chunk replay backward that exchanges `d_state` with the
neighbouring ranks explicitly. `CpRecv` / `CpSendAttach` deleted.

## Result

| seq | cp | GPUs | loss | forward | backward | peak used |
|---|---|---|---|---|---|---|
| 4,096 | 2 | 2 | 9.857565 | — | 9.0 s | — |
| 16,384 | 2 | 2 | 11.229959 | — | 35.0 s | 44.5 GB |
| 262,144 | 4 | 4 | 1.560897 | 158 s | 538 s | 78.1 GB |

Both cp=2 rungs are bit-identical to the pre-rewrite SP core, and 16,384 matches the
a2a-era value (11.229878, Δ 0.0007%). The 262,144 step is ~12 min end to end.

Backward memory is flat across layers once grads are parked: `used` held 52.5 GB with
45 GB free from layer 63 down to layer 0, where before it climbed 52.6 → 61.2 GB.

## Rule

- A linear-attention core under CP must run on the rank's own rows with a state carry.
  The all-to-all form's transient is O(global seq) and caps the global sequence at what
  one rank's memory holds, whatever the CP degree.
- Param grads that live until the optimizer step are O(layers) resident memory. Park
  them as each layer's backward completes; the optimizer reads them back.
- A chunked replay that slices its inputs per chunk pays for the slice copies **and**
  their f32 grads. Slice inside the replay against the full-seq input instead, and write
  outputs into one accumulator rather than concatenating chunk outputs.
