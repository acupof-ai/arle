# ARLE Vulkan dense decode 0.41 → 0.167 s/token on the AMD Radeon 8060S

## Context

After the 27B (`qwen35`, dense hybrid: 48 gated-delta linear + 16 full-attention
layers) ran coherently on the 8060S, decode was ~0.41 s/token — **2.9× slower
than llama.cpp's 0.139** (7.2 tok/s). Goal: close the gap.

## The decisive part — validating the bottleneck before optimizing

The first hypothesis (the obvious one) was that the **65 `submit_and_wait`/token**
(one per layer, each a full GPU-idle fence wait) were the cost. It was wrong, and
ablation proved it before any time was wasted on it:

1. **Submit count.** Batched the whole token into ONE command buffer (begin before
   the layer loop, one submit after; descriptor rings grown to a whole token's
   dispatch depth so no ring slot aliases an in-flight set). Submits 65 → 2.
   Result: 0.41 → **0.37** — only +10%. So submit serialization was a small cost.
2. **Barriers.** `ARLE_NO_BARRIER` probe (no-op every `vkCmdPipelineBarrier`;
   output goes garbage, timing is the signal). 0.37 → **0.34** — only +7%. So
   barrier/pipeline-drain serialization was not it either.
3. **Per-layer-type timing** (`ARLE_PROFILE_LAYERS`: submit + bucket per layer).
   **Full-attention = 2.1 ms/layer (16 → 33 ms). Linear = 6.25 ms/layer
   (48 → 300 ms).** The 48 gated-delta layers were **90 % of decode** and **3× a
   full-attention layer** — despite full-attn doing KV-cache flash-attention plus
   more projections.

The root cause was then one `grep` away: both gated-delta shaders
(`qwen35_ssm_conv.comp`, `qwen35_gated_delta_net.comp`) ran **`local_size_x = 1`**
— one GPU thread per workgroup. They were written as *"serial fallbacks
oracle-gated against the host f32 routine, not throughput kernels"* and **never
parallelized**. The q8_1 GEMV tuning a prior pass had chased was a dead end
because the GEMVs were never the dominant cost.

## What Worked

Parallelize both, oracle-gated byte-for-byte against the host f32 routines
(`crates/vulkan-kernels/tests/device_linear_attention.rs`, tol 1e-4/1e-5):

- **conv1d**: already independent per channel via `gl_GlobalInvocationID.x` —
  just widen `local_size_x` 1 → 256 and dispatch `ceil(channels/256)`. Body
  unchanged (was 10240 single-thread workgroups).
- **gated-delta recurrence**: one **workgroup per value head** (was one thread)
  with `local_size_x = 128` threads over the `val_dim` state columns. The key
  enabling fact: **state element `(j, val)` is touched only by the thread owning
  `val`, for every token** — so the recurrence needs **no shared memory and no
  barriers**. Each thread recomputes the per-head scalars (q/k l2-norm, decay,
  beta) in the host's serial reduction order, keeping the result byte-identical.

## Result (27B Q8_0, 8060S)

- Linear layer **6.25 → 2.20 ms/layer** (2.85×), now ≈ a full-attn layer (2.04 ms).
- Dense decode **0.41 → 0.167 s/token** (5.99 tok/s) = **1.20× of llama.cpp's
  0.139** (was 2.9×). Byte-identical " Paris." + identical gen ids throughout.
- 35B-A3B MoE coherence unchanged (shares the shaders; stays host-bridged so its
  perf — bound by host expert-gather — is unaffected).

After the fix the gated-delta block is only **0.16 ms/layer more** than
flash-attention, i.e. it is no longer the bottleneck; the residual gap to
llama.cpp is now FFN/GEMV bandwidth (already at llama.cpp's AMD config-parity) +
per-token host overhead.

## Rule

When a from-GGUF forward is correct but slow, **don't optimize the obvious
suspect — ablate to the binding constraint first.** Collapse submits, no-op
barriers, and bucket time per sub-component; each ablation that moves the needle
<10 % rules a suspect OUT. Here the real cost was two recurrence shaders left at
`local_size_x = 1` (correctness-first serial fallbacks that were never made
throughput kernels) — invisible to GEMV tuning, obvious once the time was
attributed per layer-type. A serial shader that passes its oracle is correct, not
fast; `local_size_x = 1` on a 64-wide-wave GPU wastes 98 % of every wave.
