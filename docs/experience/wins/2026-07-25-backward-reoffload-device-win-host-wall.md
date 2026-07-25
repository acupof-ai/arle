# Backward re-offload lifts the device VRAM wall to 24576 — but 256K needs LA-chunk, not more offload

## Context

`e4be96108` closed the backward half of grad-checkpoint offload: `checkpoint_backward`
re-fetched every offloaded hidden per-layer on replay but never put it back, so device
residency climbed monotonically and all N ended co-resident — offload netted almost
nothing. The fix re-offloads the replayed hidden after `free_new_except`, mirroring the
forward path: one checkpoint device-resident, not N.

Verified on the H20 pod against ThinkingCap-Qwen3.6-27B-FP8 (r16 α32 attention-qv,
`--synthetic-writeback-seq N`, one masked-CE writeback, no rollout). Offload engages at
`seq ≥ 16384` (the `writeback_offload_for_seq` gate). GPU 1, `ARLE_OPD_VRAM_TRACE=1`.

## What Worked

**The device VRAM wall moved 24576→OK.** Matched A/B, offload ON vs OFF:

| seq   | OFF (`--writeback-offload false`) | ON (default) |
|-------|-----------------------------------|--------------|
| 24576 | **CUDA OOM** (`alloc_zeros concat_axis2`, dev 96523) | **OK** (rc=0, dev 97424) |
| 28672 | CUDA OOM                          | SIGKILL (host) |

At 24576 offload is the difference between OOM and success — the mechanism genuinely
frees device VRAM. The backward asymmetry was a real leak: before the fix, offload at
24576 would also have co-resident-accumulated and OOM'd like the OFF arm.

## Rule

**Offload trades a device wall for a host wall — it does not reach 256K.** Full ON sweep:

| seq   | rc  | verdict     | device peak |
|-------|-----|-------------|-------------|
| 24576 | 0   | OK          | 97424 |
| 28672 | 137 | **host OOM-kill** | 77099 (20 GB dev free!) |
| 32768 | 137 | host OOM-kill | 97424 |
| 40960 | 1   | **device CUDA OOM** | 97099 |
| 49152 | 1   | device CUDA OOM | 97483 |

Two walls disqualify offload as the 256K lever:

- **28672–32768 = host memcg SIGKILL** (rc=137), device still had 20 GB free. Offload
  pushes ~48 layers × `[1,seq,5120]` f32 hidden to host; at 28672 that host RSS crosses
  the besteffort pod's memcg limit. The kill is non-monotonic (40960 ran *deeper* into
  backward than 28672 before dying), confirming the ceiling is host-side timing, not
  device accumulation.
- **40960+ = device CUDA OOM again** (rc=1, dev 97483 full). The peak is set by a single
  GDN layer's saved backward context — `qkv/q/k/v/g/g_cumsum/beta/chunk_state/raw_output/
  preact`, each `[heads, seq, dim]`, O(seq). Checkpoint already holds only ONE layer
  resident, yet one layer's context alone fills 97 GB at 40960. Offloading *retained*
  buffers can't touch this — it's the recompute's own working set.

Net gain: trainable ceiling ~24576→~28672 — one step, not an order of magnitude. This is
the S1a lesson again: **offload/bf16 act on retained buffers, but the peak is set by the
per-layer LA recompute's O(seq) working set.** The 256K lever is chunking that recompute
(bound single-layer device peak — like head-chunked SDPA `d2477c720`) or sequence
parallelism (TP8 splits seq → transient/8), NOT more host offload. Keep the fix (it's a
correct net-positive at the default) but stop scaling offload; redirect to LA-chunk.
