# cp=2 global 131,072: five stacked faults, and the a2a linear-attention core is the ceiling — 2026-08-21

Measured on `ThinkingCap-Qwen3.6-27B-FP8`, 2×H20 (97,508 MiB), `--cp-size 2`,
`--synthetic-writeback-seq`, binaries `seqchunk` … `seqchunk19` (`9aa1ed8a2` …
`962304caa`).

## Context

Target is global 262,144 on 2 GPUs. The previous entry put the 131,072 wall at
the replay of one checkpoint group. A layer-60 per-op ledger at 114,688 showed
the attention segment of a linear-attention layer saving +20,279 MiB across 36
tensors during the replay forward, of which ~3.6 GB is the irreducible
full-seq set (h, qkv, z, core out, attn out); the rest is matmul/LoRA/transport
intermediates.

## Phenomenon

Every rung of this tranche exposed a different fault behind the same symptom
(alloc failure at 131,072). In order:

| Fault | Evidence | Fix |
|---|---|---|
| Rank-local error invisible; peer wedged | rank1 1 h in `ncclCommDestroy`, rank0 100% CPU in `cublasSgemm` module load, GPU0 100% util; no error line | `NcclBackend::drop` uses `ncclCommAbort` (`77a734de3`) |
| Per-chunk pool trim | `trim_after_checkpoint_replay` ran per seq chunk under offload | once per region (`6c6db1586`) |
| Pre-backward hoard | pre-backward `hoarded=29,387 MiB`, first backward alloc (1.28 GB) OOM at `free=9 MiB` | trim before masked-writeback backward (`6912771fa`) |
| Full-attention layer replay | layer 63 (full attn) replay `post_input_norm→post_attention` **+25,880 MiB, +39 tensors**; linear layers had been chunked, full-attn projections had not | CP full attention chunked end-to-end over q tiles (`cf3fb1500`); ring FA3 accepts a q tile inside a longer k run (`6ff5ee708`, coverage test); ring backward derives k-side extents from k (`73a012a9b`) |
| Linear core full-seq backward | after the above, OOM at `la dqkv` ~35 layers deep; core runs the a2a'd **global** sequence in one call | core over sequential head groups (`8ab4d7b18`), G limited by the built flashqla geometry set (`948deb459`) — G=2 for 24/8 |

With all five in place the layer-30 (linear) ledger at 131,072 reads:

```
post_input_norm → post_attention   pool_used +6,936 MiB   (was +20,279 / +25,880)
outer replay total                 +12,063 MiB            (was +25,885)
layer backward peak pool_used      74,953 MiB
pool_reserved at that instant      91,808 MiB
```

Live memory peaks 22 GB under the card. The run still dies ~40 layers in,
alternately at `mul_backward grad_a` / `transpose [1,65536,6144]` /
`sum_backward [1,65536,6144]`, 1.6 GB each.

## Root cause (remaining)

Two allocator effects on top of a near-full card:

1. `pool_reserved − pool_used` sits at 12–33 GB inside a layer scope. The
   per-chunk pool trace (`ARLE_OPD_CHUNK_POOL_TRACE`) shows the seq-chunk
   loops themselves are clean at 32,768, forward and replay (reserved flat
   across chunks). The hoard is built by the **core sub-scope replay**: the
   a2a transport chain and the full-seq recurrence allocate ~14–30 GB of
   transients per linear layer, freed after, that the pool does not re-cut
   for the backward's grad shapes.
2. Trimming per chunk makes it worse: the driver then fails a 1.6 GB
   allocation with **10.5 GB free** (`seqchunk18`). Region/scope-level trims
   (hoard > 2 GB) and sync+trim+retry at every backend `alloc_zeros`
   (`7585dafa5`) reclaim what is reclaimable; the rest is live.

Underneath both: under CP the linear-attention core all-to-alls the sequence
into the head axis and runs the recurrence on the **global** sequence per rank.
Its transient is O(global seq) and does not shrink with CP. At 131,072 that
leaves a ~2–5 GB deficit; at 262,144 it is structurally impossible on 97 GB.

## Rule

- Chunk every position-wise stage of a layer replay (projections, LoRA,
  norm, RoPE, o_proj); measured saving per linear layer 20.3 → 6.9 GB, per
  full-attn layer 25.9 → ~4 GB. Exact, no kernel work.
- A failing rank must print before it tears down: teardown through
  `ncclCommAbort`, never `ncclCommDestroy`.
- Read `free` at the failure instant from the error line before reasoning
  about the pool; an OOM with GB free is allocator fragmentation, an OOM at
  `free≈0` is live memory.
- The a2a linear-attention core caps cp=2 below 131,072 on 97 GB. The next
  structural step is a sequence-parallel core with cross-rank state carry
  (rank r hands its boundary state to rank r+1), which needs
  `d_initial_state` in the gated-delta backward; the forward primitives
  (`linear_attention_core_with_carry`) already exist.

## Numerics

NVFP4 frozen base (converter now accepts FP8 block-scaled input,
`fc3947d0e`; `/data00/ThinkingCap-Qwen3.6-27B-NVFP4`, 21 GB) matches FP8 at
seq 16,384: loss 11.228516 vs 11.229878 (Δ 0.012%). Sharing the NVFP4 base
between rollout and student is not wired (serve Marlin-repacks and releases the
source); unshared it costs more than shared FP8, so it stays parked until the
share path exists.
