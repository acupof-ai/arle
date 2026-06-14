# ARLE Vulkan MoE decode 1.97 → 0.085 s/token on the AMD Radeon 8060S

## Context

The 35B-A3B MoE (`qwen35moe`, Q4_K_M, 40 layers, 256 experts top-8) ran
coherently but decoded at **1.97 s/token** — **93× slower than llama.cpp's
0.021 s/token** (47.3 tok/s). The dense 27B sharing the same substrate was at
1.2× of llama.cpp, so the MoE number was clearly anomalous, not a hardware wall.

## See it first, then confirm — the discipline this win is about

The root cause was reached by **reasoning over the code + the roofline**, with
measurement used only to CONFIRM, not to discover:

1. **Roofline says it is NOT bandwidth.** Top-8/256 experts + shared + attention
   read only **~2.62 GB/token**. At this box's ~220 GB/s that floors at **~12
   ms/token**. 1.97 s is **~165×** the floor — the GPU is idle, not reading
   memory. So the cost is host-side, by elimination.
2. **Read the code to localize it.** `moe_ffn` does the router projection
   (`ffn_gate_inp`, `[hidden→n_expert]`) on host: `gemv_f32_host → dequant_f32`.
   `dequant_f32` calls `t.buffer.copy_to_host(...)` — and per `loader.rs`
   `plan_residency`, F32 tensors are `Residency::DequantF32`, which `upload_plan`
   puts in a **HOST_VISIBLE (write-combined)** buffer. CPU reads of write-
   combined memory are uncached, ~50–300 MB/s. The router is 2 MB, re-read
   **every layer every token**: 40 × 2 MB = **80 MB/token** of WC read-back
   ⇒ on the order of **~1.5 s**. That number is derivable before running anything.
3. **Confirm with one probe.** `ARLE_PROFILE_LAYERS` timing around the router
   GEMV: **1554 ms of the 1970 ms/token (79 %)** — matching the reasoned
   estimate. The measurement verified the understanding; it did not produce it.

## What Worked

`upload_plan` **already** dequantizes each `DequantF32` tensor to a `Vec<f32>`
to fill the device buffer — then drops it, forcing the read-back. Keep it:

- `DeviceTensor::host_f32: Option<Vec<f32>>` caches the dequantized values in
  **cached** RAM at upload (the bytes we already computed).
- `host_f32_values()` borrows that slice (**zero-copy** for the router GEMV);
  `dequant_f32()` is now a thin owned-copy wrapper, so the SSM
  conv/A_log/dt_bias/norm read-backs ride the same cache.
- The device buffer is untouched — norms and SSM params still bind it on-device.

One field + one helper; no kernel, no submit-graph change.

## Result (35B-A3B Q4_K_M, 8060S; `ARLE_PROFILE_LAYERS=1`)

- HOST router gemv **1554 → 15 ms/token** (~100×, as predicted).
- MoE decode **1.97 → 0.085 s/token** (11.8 tok/s) = **93× → 4.0×** of
  llama.cpp's 47.3 tok/s. Still coherent.
- Dense 27B unchanged: 0.167 s/token, byte-identical " Paris." — its hot path is
  on-device and never read these tensors back per token.

The prediction (~0.4 s) was conservative: the zero-copy borrow plus already-fast
device expert GEMVs landed it at 0.085 s.

## What is left (the remaining 4×, decomposed — not yet a wall)

Per-token after the fix: attn ~26 ms, moe_ffn ~39 ms (router 15 ms still on
host + device experts ~24 ms), lm_head/sample/embed ~20 ms, **121 submits/token**
(~37 ms in fence waits). Real levers, each now architecture work rather than a
bug: (a) move the `[hidden→256]` router GEMV on-device (~12 ms), (b) collapse
the 121 submits toward the dense path's single-submit-per-token batching.

## Rule

A host `copy_to_host` of a **HOST_VISIBLE / write-combined** Vulkan buffer in a
per-token loop is a latent ~100× trap: WC CPU reads are ~50–300 MB/s. If you
dequantized it to fill the buffer, **keep the host `Vec` and read that**, never
the buffer. And: you can *locate* this class of bug by reading the residency
plan + the hot loop and doing the roofline — measure to confirm the magnitude,
not to find it. See [the gated-delta win](2026-06-14-vulkan-gated-delta-parallel.md)
for the complementary case (the cost there was on-device, found by per-layer
ablation).
