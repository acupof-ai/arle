# agent-OPD writeback: KV-pool lever works (−19.8 GB) but the binding wall is a per-group checkpoint-forward activation accumulation, not the ~0.3 GB the premise assumed

> **RESOLVED 2026-06-27** — root-caused to TWO checkpoint-forward leaks and fixed;
> loop closes. See
> [wins/2026-06-27-agent-opd-writeback-oom-closed-frozen-hidden-and-sdpa-chunk-leaks.md](../wins/2026-06-27-agent-opd-writeback-oom-closed-frozen-hidden-and-sdpa-chunk-leaks.md).
> The `+200 MiB/group` accumulation attributed below was Leak 1 (frozen-group input
> hidden pinned in VRAM because `checkpoint()` skips offload/free for
> `requires_grad=false` groups) — real but NOT binding. The binding wall was Leak 2,
> NOT decoded below: per-chunk SDPA transients (`scores/scaled/masked/probs`,
> 4×`[chunk,seq,seq]`) piling up across `head_chunked_sdpa_recompute`'s head loop
> under tape-disabled recompute (~100 GiB at seq=16000) — the "+15 GB final-group
> spike" was actually the first `full_attention` layer's un-freed chunk pileup.

## Context

Mainline task: close the agent-OPD (train-infer-unified) loop end-to-end; the
blocker is the masked-CE writeback forward OOM at seq ~16–20K. The framing handed
in was "the OOM is BARELY over — ~64 GB resident + ~34 GB writeback = ~98 GB vs
97.8 GB H20, over by ~0.3 GB; free ~3–5 GB and it fits", with the strongest lever
being "release the DEAD rollout KV pool before the writeback".

Verified on the 8×H20 box (GPU 5, Qwen3.6-27B-FP8, `--rollout-num-slots 1
--max-turns 24 --max-tokens 1024 --lora-layer-start 32`, `student_seq=32768`),
attributed with a new `ARLE_OPD_VRAM_TRACE` milestone log + a per-checkpoint-group
`device_mem_info` probe.

## Root Cause

**The KV-pool lever is correct and works, but is NOT the binding constraint.**
Measured resident breakdown (no `--share-frozen-base`, the verify config):

```
after rollout engine load : used=48715 MiB  (27B FP8 weights ~28.9 GB + KV pool 19.4 GB)
after autograd student load: used=58795 MiB  (+10 GB student)   ← the "~64 GB" floor
```

The rollout KV pool is **19830 pages = 19.4 GB**, NOT ~16 GB: `EngineLoadConfig.
total_pages` (derived from `student_seq`) is only the HOST admission floor; the
CUDA Qwen3.6 device pool is **profile-sized = `mem_fraction_static(0.2) × free
VRAM`** (`executor.rs` `build_full_attn_kv_pool` → `profile_kv_pool_tokens`), which
profiled 19830 pages (317K tokens) over the 2048 requested floor. So shrinking
`student_seq` would NOT shrink the device pool — only `mem_fraction_static` or an
explicit release does.

The lever (drop `Qwen35CudaExecutor::full_attn_kv: Option<PagedKVPool>` — already an
`Option` "so the OPD-offload path can drop it") **works once the async free is
ordered correctly**: the first attempt freed nothing (`mem_get_info` used
unchanged) because cudarc `CudaSlice::drop` enqueues `cuMemFreeAsync` on the stream
— the trim ran before the frees executed. Fix: `sync` → `take()` → `sync` AGAIN →
`trim_memory_pool`. Then:

```
synthetic-writeback pre-release : used=58795 MiB  free=38713 MiB
released full-attn KV pool      : freed 19830 MB
synthetic-writeback pre-writeback: used=38987 MiB  free=58521 MiB   ← −19.8 GB, verified
```

**But the writeback forward still OOMs with 58.5 GB free.** Per-checkpoint-group
`device_mem_info` probe (seq=16000, grad-checkpointing on, group≈2 layers):

```
ckpt-group 1 post: used=54415 MiB  (+15 GB over pre)
ckpt-group 2 post: used=97487 MiB  (+43 GB) → OOM on group 3
```

At seq=8000 the shape is clearer: steady **+~200 MiB / +23 device-tensors per
group** across ~32 groups, then a **+15 GB spike on the FINAL group** → OOM. So the
forward activations accumulate faster than `checkpoint`'s `free_new_except`
reclaims them, and a large final-group transient tips it over. `--lora-layer-start
60` (train only top 4 layers) OOMs even EARLIER — proving the wall is the
**forward of the frozen prefix (all 64 layers forward regardless of the
suffix-detach, which only bounds the backward)**, not the trainable suffix.

The "~34 GB writeback / over by 0.3 GB" premise was wrong: the real writeback
forward needs **>58.5 GB** on top of a 39 GB post-release floor — a 15–40 GB
accumulation gap, not 0.3 GB. (The prior `2026-06-26-...forward-activation-wall`
entry's "~34 GB" was the `--share-frozen-base` 63.6 GB-floor config; the
head-chunked SDPA scores are already capped to ~6 GiB and the ckpt group to ~8 GiB,
so neither is the unbounded term — the accumulation across groups is.)

## Fix

KV-pool release lever landed in the working tree (NOT committed — the commit gate
is loop-closure, which this does not achieve): seam `release_kv_pool`/
`ensure_kv_pool` (default no-op) threaded executor→engine→server→api→InferStudent,
called in the agent-OPD writeback closure + re-acquired at the next round top;
`build_full_attn_kv_pool` extracted so the pool re-profiles identically on
re-acquire. Default serve path byte-identical (new methods are agent-OPD-only).
Plus gated `ARLE_OPD_VRAM_TRACE` attribution logging (off by default).

**Next wall (separate root-cause, owns the loop-closure):** the per-group
checkpoint-forward accumulation. Candidate leads, each needs its own measured A/B:
1. The steady +23 device-tensors/group under grad-checkpointing — `checkpoint`'s
   `offload_to_host` of the saved inputs is not freeing device for the
   grad-requiring groups, OR `free_new_except` carries tensors forward via
   `live_before`. Decode WHICH tensors survive (the `live_device_bytes` probe
   over-counts FP8 weights as f32 — needs a per-tensor dump, not the aggregate).
2. The +15 GB final-group spike — instrument the last group's per-op allocs.
3. `mem_fraction_static=0.2` profiling a 19.4 GB pool (10× the 2048-page need at
   1 slot) inflates the at-load floor; capping the pool to the requested floor at
   load (not just freeing it pre-writeback) buys headroom for the rollout phase too.

## Rule

A "barely over, free 3–5 GB and it fits" premise is a HYPOTHESIS — attribute the
actual transient before sizing the lever. Here the designated lever (free the dead
KV pool) was correct and freed exactly its 19.8 GB, but the real gap was 15–40 GB
of forward accumulation, so the loop did not close on the lever alone. Two
async-pool gotchas confirmed by measurement, not inference: (1) a dropped
`CudaSlice` frees via `cuMemFreeAsync` — you MUST sync AFTER the drop, before
`trim_memory_pool`, or the trim reclaims nothing (the first attempt freed 0 GB);
(2) `mem_get_info` "used" counts async-pool-cached blocks, so cross-check genuine
retention with a store-side live-tensor count (`MEMPOOL_RETAIN=0` A/B showed the
accumulation is REAL, not a caching artifact). The device KV pool is
`mem_fraction_static × free VRAM`, NOT `config.total_pages` (the host admission
floor) — shrinking `student_seq` would not have helped. Commit gate held: loop did
not close → not committed.
