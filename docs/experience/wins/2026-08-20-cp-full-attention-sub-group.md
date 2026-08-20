# A checkpoint boundary around the CP full-attention core: −7,942 MiB resident, −1,280 MiB at the peak — 2026-08-20

Commit: `0206208a5` (`perf(train): give the CP full-attention core its own
checkpoint boundary`).

## Context

Target is global sequence 262,144 on 2 GPUs. `--synthetic-writeback-seq N
--cp-size 2`, ThinkingCap-Qwen3.6-27B-FP8, LoRA r16 α32 attention-qv, 2×H20
(97,508 MiB). The ceiling has been 131,072 since `28a1a79ef`.

`ARLE_OPD_OP_MEM_CHECKPOINT_FN=60` measures one layer's checkpoint replay from
inside. At local 65,536, layer 63 (a full-attention layer) replays 45 ops:
`RingAttention` ×1, `MatmulBT` ×8, `Transpose` ×5, `RMSNorm` ×4, `RoPE` ×2,
`SeqChunkedRecompute` ×1 (the MLP), plus reshapes.

The MLP has been sequence-chunked all along, and the non-CP attention path has
its own chunking in `forward_full_attention_chunked`. The CP attention path had
neither.

## Result

Matched arms, same `scope_enter` anchor to the digit:

| local 65,536 | before | after |
|---|---:|---:|
| `scope_enter` `pool_used` | 46,345 | 46,345 |
| `post_replay` (outer layer) | 83,223 | **75,281** |
| replay's resident set | +36,878 | **+28,936** |
| tensors taped by the replay | 56 | **44** |
| peak inside the group | 83,223 | **81,943** |
| loss | 3.036179 | 3.036179 |
| backward | 216.9 s | 223.3 s (+2.9%) |
| step | 328.1 s | 329.5 s (+0.4%) |

**−7,942 MiB resident, −1,280 MiB at the peak, for +0.4% step time.**

Global 163,840 still fails, but the failure moved: from
`zeros [1, 81920, 5120]` at the first allocation of the backward to
`matmul_bt_bf16 [81920, 5120]` later in it.

## What worked

A checkpoint group frees once at its boundary, so a layer's peak is the SUM
over its stages; a nested boundary makes it the MAX. The ring, its gate, the
head merge and out_proj now replay under their own boundary and free when their
own backward completes, rather than staying resident through the projection
backwards. Same shape as `28a1a79ef` used for the linear-attention core, and
inert in the forward — `checkpoint` passes straight through while the outer
group has the tape disabled.

## Why the peak moved less than the resident set

The nested replay runs while the outer stage's intermediates are still live, so
the two overlap. The resident set fell 7,942 MiB but the in-group high-water
fell only 1,280.

This bounds the axis: **more boundaries will not pay.** The replay forward still
costs +28,936 MiB on its own, and every sub-group has to hand its output to the
rest of the layer, so no boundary placement removes that. Against a gap of about
20 GB, roughly 1 GB per boundary is the wrong instrument.

The instrument that does work on the replay forward is sequence chunking —
what `SeqChunkedRecompute` already does for the MLP, and what
`forward_full_attention_chunked` does for non-CP attention. Chunking CP ring
attention is the open problem; a previous attempt failed on the pod with
`ring FA3: q/k runs partially overlap`.

## Rule

A nested checkpoint boundary converts a stage's cost from resident to
transient, and the peak only moves by whatever does not overlap the stages
still live around it. Measure the boundary's effect on the in-group high-water,
not on the resident set — they differ by a factor of six here.
