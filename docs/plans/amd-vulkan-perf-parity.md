# ARLE Vulkan decode perf parity — Strix Halo 8060S (27B Q8 → ≥ llama.cpp)

Systematic design (researched before coding). Goal: ARLE `infer-vulkan` 27B decode
from ~5 s/token to **≥ llama.cpp parity (7.2 tok/s)** on the Radeon 8060S. Grounded
in: ARLE static bottleneck diagnosis, the on-box llama.cpp `ggml-vulkan.cpp`
(841 KB, the authoritative fast reference on this exact GPU), and Vulkan-compute
best practice. Phasing is phase + CC-mode (no calendar, per
`memory/feedback_no_calendar_in_plans.md`).

## Headline

ARLE 27B decode is **~98% host/driver overhead, not kernel math.** The fix is to
record **one barrier-chained command buffer per token** against a **device-resident
activation arena**, dispatch from **compile-once cached pipelines**, and **submit
once on a reused fence** — the exact `ggml-vulkan` structure. The GEMV math is
already proven (`device_gemv.rs`, ≤2e-2). This is overhead elimination, not a
kernel rewrite.

## Roofline (the real target)

- 27B stored Q8_0 = 1.0625 B/weight → **~28.7 GB weight traffic/token** (load test
  prints ~26 GiB resident; +KV/act ≈ 25–29 GB read/decode-step).
- 8060S = LPDDR5X-8000, 256 GB/s theoretical, **~215 GB/s effective**.
- **Ceiling: 215/28.7 ≈ 7.5 tok/s** (theoretical 256/28.7 ≈ 8.9). llama.cpp = 7.2
  (~95% of effective roofline). **ARLE today ≈ 0.2 tok/s = ~2.6% of roofline.**
- **Target: ≥ 7.2 tok/s** (~0.139 s/token). The 35× gap is *entirely* host-side
  (per-dispatch SPIR-V re-read + pipeline rebuild + `queue_wait_idle`), not FLOPs.
- (Q8 caps near single digits regardless of code; Q4_K → ~14 tok/s, A3B-MoE → ~80.
  Parity-at-Q8 is the contracted goal and is purely overhead removal.)

## Bottleneck ranking (measured from the code)

1. **Per-op `queue_wait_idle`** — `vulkan-sys/src/lib.rs:563-607` `one_shot_submit`
   submits with a NULL fence then drains the *whole queue*, per dispatch. ~900–994
   full GPU drains/token. **Dominant.** Fix: a `CommandRecorder` that records the
   token's dispatches with in-cmdbuffer barriers and submits **once**, one fence
   wait. (Mirror `ggml-vulkan.cpp:13474-13485`, `2278-2355`.)
2. **Per-dispatch pipeline rebuild from disk** — `vulkan-kernels/src/lib.rs:742-805`
   per call does `fs::read(.spv)` → ShaderModule → DescriptorSetLayout →
   DescriptorSet (own pool) → ComputePipeline with `PipelineCache::null()` → CommandPool,
   all dropped at end. ~900/token. **Second-dominant.** Fix: a `KernelCache`
   (pipelines + shared DSL built **once** at model load) + a real `vk::PipelineCache`.
   (Mirror `ggml-vulkan.cpp:2074-2096`, `2209-2255`, `6303-6332`.)
3. **Per-GEMV host round-trip + scratch alloc** — `forward.rs:587-675` allocs
   x_in/f0/f1 fresh + `copy_from_host`/`copy_to_host` per GEMV; host does
   norm/rope/etc. in `Vec<f32>` and re-uploads. Forces a CPU/GPU serialization point
   per op and blocks batching. Fix: a `DeviceArena` of named sub-ranges
   (residual/normed/q/k/v/gate/up/attn/logits) allocated once; offset-aware
   descriptors; UMA `DeviceLocal|HostVisible|HostCoherent` so CPU writes the embed
   row / reads logits with zero staging. (Mirror `ggml-vulkan.cpp:6239-6263`, `1840`.)
4. **No barrier/submit-batch model** — no `vkCmdPipelineBarrier` anywhere; sync is
   only the full drain. Fix: `CommandRecorder.barrier()` = one
   `vkCmdPipelineBarrier(COMPUTE→COMPUTE, SHADER_WRITE→SHADER_READ)`; batch-submit on
   a ~100-node / ~100-MB cadence (APU-TDR safe). (Mirror `ggml-vulkan.cpp:2717-2737`,
   `13994-14143`.)
5. **Host-resident elementwise/norm/attention** (`forward.rs:693-746,328-369,471-519`)
   — correct and the **numeric oracle**, but each forces a device→host→device hop.
   **NOT on the parity critical path** (after 1–3 the GEMVs dominate, bandwidth-bound).
   Port to device kernels **last, op-by-op, oracle-gated.**

## llama.cpp patterns to mirror (file:line into the on-box `ggml-vulkan.cpp`)

- `2037-2067` `ggml_vk_wait_for_fence` — `almost_ready_fence` (CPU sleeps ~80% of
  graph) then `getFenceStatus`+YIELD spin on the final fence (tail latency).
- `2278-2355` `ggml_vk_submit` — batches all queued seqs into **one** `vkQueueSubmit`.
- `13474-13485` `ggml_vk_synchronize` — exactly **one fence per token** (one cgraph).
- `13994-14143` graph batch loop — accumulate into one open cmd buffer; submit when
  `submitted_nodes>=100` OR `mul_mat_bytes>=min(100MB,last/40)` (doubled first 3) OR
  last/almost-ready. The cadence to copy.
- `2074-2096` create pipeline + single shared DSL **once** (async). `2209-2255`
  descriptor pool grow-by-50%, round-robin `descriptor_set_idx`. `6303-6332` per
  dispatch = 1 `updateDescriptorSets` + bind + dispatch (no object creation).
- `2717-2737` `ggml_vk_sync_buffers` — the exact one-barrier body.
- `6239-6263` `ggml_vk_tensor_subbuffer` — every tensor = (buffer,offset,size); UMA
  host-ptr fast path. `2664-2699` UMA heap flags `{DeviceLocal, HostVisible|HostCoherent}`.
- `7469-7706` `ggml_vk_mul_mat_vec_q_f16` — decode GEMV = 5 buffers [X,Y,D,F0,F1] +
  13-uint push, 1 workgroup/row = **ARLE's already-proven `gemv_params`/`device_gemv.rs` ABI.**
- `7403-7467` — on RDNA, Q8_0 does **not** take integer-dot mmvq; keep ARLE's float
  Q8_0 GEMV. `8617-8687` — N==1 decode forces **scalar** flash-attn (coopmat is NV/prefill).

## Implementation plan (surgical, smallest-shippable-first, oracle-gated)

**Step 0 — GATE: certify the host-f32 forward as the numeric oracle.** Run
`qwen35_27b_generates_coherent_text` (model_qwen35.rs, `#[ignore]`) on-box; confirm
coherent EN+ZH continuations; dump `forward_token` logits for a fixed prompt to a
**golden f32 blob**. **If not coherent → STOP and fix numerics first** — every later
step diffs against this oracle. Record the ~5 s/token "before" and re-confirm the
llama.cpp 7.2 tok/s bar (`llama-bench -m Qwen3.6-27B-Q8_0.gguf -p 0 -n 128`).

**Step 1 — `CommandRecorder` in vulkan-sys** (the one genuinely-missing primitive).
RAII: one PRIMARY cmd buffer (from the existing RESET pool) + one `vk::Fence`.
`begin()`/`dispatch(pipeline,&set,push,d)` (= the body of `one_shot_submit`'s closure)
/`barrier()` (single `vkCmdPipelineBarrier`)/`submit_and_wait()` (submit+fence, **no**
`queue_wait_idle`). Verify: 3 chained `add` dispatches w/ barriers == 3 sequential
`one_shot_submit`s, and exactly **one** `queue_submit` (submit counter). Keep
`one_shot_submit` **only** for cold weight upload; never on the forward path again.

**Step 2 — `KernelCache` + real `vk::PipelineCache` + `record_dispatch`.** A
`HashMap<(Kernel,spec,push_bytes,binding_count) → (ComputePipeline,DescriptorSetLayout)>`
that reads each `.spv` once and builds on miss; swap `PipelineCache::null()`
(vulkan-sys:870) for a device-wide cache; `record_dispatch` only records
bind+push+dispatch. Move the per-dispatch object creation out of
`launch_with_params_and_specialization` into the cache-miss builder. Keep
`Kernel`/`shader_name`/`specialization_u32`/`gemv_params`/`q8_1_quantize_params`
verbatim (pure data). Verify: `device_gemv.rs` byte-identical through the cached path;
a Kernel compiles exactly **once** across 100 `record_dispatch` calls.

**Step 3 — `DeviceArena` + offset-aware descriptors; `gemv_device` records, no
round-trip.** One wide `DeviceLocal|HostVisible|HostCoherent` buffer of named
(offset,len) slots (align to `minStorageBufferOffsetAlignment`), allocated once on
the model; `DescriptorSet::storage_buffers_ranged(buffer,offset,range)`; rewrite
`gemv_device` to write x into an arena slot and dispatch quantize+GEMV into arena
slots via `record_dispatch`. **Delete** `GemvScratch` + the per-GEMV allocs +
`copy_to_host`. Verify: one FFN-down GEMV from the arena == old result (within q8_1
tol); forward_token logits still match the step-0 golden blob.

**Step 4 — record the WHOLE token graph; submit once/token; read back only logits.
THE parity win.** forward_token uploads embed → records, per layer, the
quantize+GEMV pairs + barriers into the single recorder threading arena sub-buffers
→ **one** `submit_and_wait` per token (batch cadence ~100 MB/~100 nodes, APU-TDR
safe) → `copy_to_host` only `[vocab]` logits. (Host SDPA/gated-delta/rope stay
between batches for now.) Verify: logits == golden blob within tol; **measure tok/s,
target ≥7.2**; submit-counter shows a few submits + one fence/token; compare to
`llama.cpp -ngl 999` tg on the same GGUF.

**Step 5 — (stretch, post-parity) port host elementwise → device, op-by-op, fused.**
Move rms_norm/rope_neox/swiglu/add/SDPA/gated-delta onto the mapped device kernels
(`model_qwen35.rs:157-169`); fuse RMSNorm + bias into the next GEMV (`fusion_flags`);
scalar flash-attn for N==1. Each op replaces its host fn entirely once it matches the
oracle (delete the host fn). Diminishing returns past parity (bandwidth-bound) —
stop when host hops are gone or returns flatten.

## Clean-code principles (一针见血)

- **One canonical decode flow:** exactly one `CommandRecorder`, one `KernelCache`,
  one `DeviceArena`, one `record_dispatch` — the old per-op submit/launch path is
  **deleted, not wrapped** (no adapter layering).
- **Delete, don't add:** `GemvScratch` + its `copy_to_host`, per-GEMV allocs, and the
  per-dispatch object creation all go; `one_shot_submit` survives only for cold upload.
- **Keep proven data contracts verbatim:** `Kernel`/`shader_name`/`specialization_u32`/
  `gemv_params` (13-uint)/`q8_1_quantize_params` are pure data — near-zero risk
  (device_gemv ABI untouched).
- **The seam is sacred:** `BackendExecutor`, `kv_pool`, `executor.rs` submit/poll,
  loader residency tiers do **not** change — all work lands below `forward_token`.
- **Oracle-gated ports:** correct host-f32 is ground truth; no device op replaces its
  host counterpart until it matches within tolerance, then the host fn is removed.
- RAII wrappers in the existing thin-Ash Drop style; no new traits.

## Risks

- **APU TDR / DeviceLost** on an over-large single submit (~26 GB touched) — cap to
  the ~100 MB / ~100-node cadence, not literally one submit/token (cf. llama.cpp #21724).
- **Offset alignment** — arena slots must honor `minStorageBufferOffsetAlignment` or
  silently corrupt / VVL-error.
- **Fence/lifetime** — never re-record a cmd buffer before its fence signals
  (`wait_for_fences` before `begin()`); bugs look like numeric errors.
- **DeviceLocal readback** — request `DeviceLocal|HostVisible|HostCoherent` on this
  UMA box and verify the heap is host-mappable (`ctx.memory_types()`).
- **Step-5 numeric drift** — device bf16/fp16 vs host f32; SDPA/gated-delta/rope are
  highest-risk (stateful recurrence). Port last; parity is reachable at Step 4 with
  GEMVs alone.
- **AMDVLK ICD hijack** — an inactive AMDVLK ICD silently halves throughput; verify
  RADV is the active ICD before trusting any tok/s.

## Gating dependency

The whole plan is **gated on Step 0**: the host-f32 forward must be confirmed
*coherent* (not just finite-logit) as the oracle. The coherence test is the
prerequisite; if it is not green, fix the forward numerics before any perf step.
