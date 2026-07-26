# Backward re-offload lifts the OPD-writeback device wall 24576→32768 — 256K still needs LA-chunk

## Context

`e4be96108` closed the backward half of grad-checkpoint offload: `checkpoint_backward`
re-fetched every offloaded hidden per-layer on replay but never put it back, so device
residency climbed monotonically and all N ended co-resident — offload netted almost
nothing. The fix re-offloads the replayed hidden after `free_new_except`, mirroring the
forward path: one checkpoint device-resident, not N.

Verified on the H20 pod against ThinkingCap-Qwen3.6-27B-FP8 (r16 α32 attention-qv,
`--synthetic-writeback-seq N`, one masked-CE writeback, no rollout). Offload engages at
`seq ≥ 16384` (`writeback_offload_for_seq`). GPU 1, `ARLE_OPD_VRAM_TRACE=1`.

## What Worked

**The device VRAM wall moved 24576→32768.** Clean sweep (28672/32768 re-measured after a
concurrent foreign `arle` was excluded — see Rule):

| seq   | OFF (`--writeback-offload false`) | ON (default) | ON loss | ON peak MiB |
|-------|-----------------------------------|--------------|---------|-------------|
| 24576 | **CUDA OOM** (`concat_axis2`, dev 96523) | **OK** | 11.57 | 97424 |
| 28672 | CUDA OOM                          | **OK**       | 11.69   | 77099 |
| 32768 | —                                 | **OK**       | 10.87   | 84971 |
| 40960 | —                                 | **CUDA OOM** (`concat_axis2`, dev 97099, 409 MiB free) | — | 97483 |
| 49152 | —                                 | CUDA OOM (`mul_backward grad_b`) | — | 97483 |

At 24576 offload is the difference between OOM and success. The backward asymmetry was a
real leak: before the fix, offload would have co-resident-accumulated and OOM'd like OFF.
Pre-fix baseline walled at 24576; post-fix runs clean through 32768 — a 1.33× lift in
trainable length at 27B. (28672's peak dips below 24576's — allocator fragmentation /
trim timing, non-monotonic and expected; the wall is what matters.)

## Rule

**Offload is a net-positive device lever but not the 256K path — the wall at 40960+ is a
per-layer O(seq) recompute working set, not a retained buffer.** At 40960 the device
CUDA-OOMs with 409 MiB free (`concat_axis2`): one GDN layer's saved backward context —
`qkv/q/k/v/g/g_cumsum/beta/chunk_state/raw_output/preact`, each `[heads,seq,dim]`, O(seq)
— fills 97 GB alone, with checkpoint already holding only ONE layer resident. Offloading
*retained* buffers can't touch this; it's the recompute's own working set. Same lesson as
S1a and bf16: retained-buffer levers (offload, store-bf16) act on the wrong term. The 256K
lever is chunking the LA backward recompute (bound single-layer device peak — like
head-chunked SDPA `d2477c720`) or sequence parallelism (TP8 splits seq → transient/8).

**Attribution discipline (self-correction):** an earlier probe of this same sweep mis-read
two `rc=137` SIGKILLs at 28672/32768 as "our offloaded hidden overran host memcg" and
committed that as root cause. Wrong twice over: (1) a clean re-measure shows both seqs
pass at rc=0 — the kill was a *concurrent foreign* `arle` (pid 2058667, 343 GB host RSS)
tripping the shared container memcg, nothing to do with our job; (2) our own host RSS at
40960 was only ~53 GB, an order of magnitude below any memcg limit — the arithmetic alone
refuted the host-wall story. The tell was there and ignored: a SIGKILL with **20 GB device
free** is not our device OOM, and an unattributed kill is not a root cause. Trace the
failing PID before writing "Root Cause"; on a shared box, re-measure a SIGKILL clean before
attributing it to your own memory.
