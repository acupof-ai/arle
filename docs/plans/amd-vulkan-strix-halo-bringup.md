# AMD Vulkan bring-up — Strix Halo (Radeon 8060S) serving Qwen3.5-122B-A10B

Status as of 2026-06-13. Goal: serve a ~122B 4-bit model on `arle serve --backend
vulkan` on ckl's AMD box. Phasing uses phase + CC-mode (no calendar weeks, per
`memory/feedback_no_calendar_in_plans.md`).

## Target hardware (fixed)

- **AMD Ryzen AI MAX+ 395 "Strix Halo"**, iGPU **Radeon 8060S** (RDNA 3.5,
  **gfx1151**, 40 CU), unified-memory APU.
- **128 GB** LPDDR5X-8533; BIOS Variable Graphics Memory = **96 GB VRAM**
  (registry `qwMemorySize`), ~32 GB left for the OS.
- **Windows 11**, Vulkan loader + AMD Adrenalin driver present, `glslc` (MSYS2).
  **No ROCm/HIP** — and ROCm-on-Windows does not target gfx1151, so the
  `infer-hip` (ROCm/Linux) lane is a dead end here.

**Backend verdict: Vulkan only.** `infer-vulkan` + `vulkan-sys` + `vulkan-kernels`.

## Target model (user already has it)

`Qwen3.5-122B-A10B`, GGUF **UD-Q4_K_XL** (~77 GB, 3 shards). GGUF metadata:

- `general.architecture = qwen35moe`, `general.name = Qwen3.5-122B-A10B`
- 48 layers, **256 experts / top-8** (`expert_feed_forward_length=1024`, shared
  expert 1024), embedding 3072, GQA **head_count 32 / kv 2**, key/value length 256.
- **Hybrid: SSM/gated-delta linear attention** (`ssm.conv_kernel=4`,
  `state_size=128`, `inner_size=8192`, `group_count=16`, `time_step_rank=64`)
  interleaved with periodic full attention.
- 256K context, mRoPE `dimension_sections=[11,11,10,0]`, `rope.freq_base=1e7`.
- Tokenizer **gpt2 BPE + chat_template embedded in the GGUF** (no sibling
  `tokenizer.json`).

This maps to the **already-supported** infer-vulkan family: `model_qwen35.rs`
(gated-delta linear + full-attn) + `model_qwen36.rs` (sparse MoE). **No
gpt-oss / no new architecture** is required (the earlier gpt-oss idea is dropped:
this is a true 122B inside a supported family).

Source: `unsloth/Qwen3.5-122B-A10B-GGUF`. Quant choice **Q4_K_XL** (not
`MXFP4_MOE`) because the GGUF host path can't slice MXFP4 (ggml type 39,
`type_size=None`).

## Download method (learned the hard way)

- ModelScope anonymous download throttles to ~30–130 kB/s — unusable.
- `hf` 1.19 CLI ignores `HF_ENDPOINT` (Xet backend dials huggingface.co → blocked
  without VPN).
- **Working path:** `aria2c -x16 -s16 -c` against **hf-mirror.com resolve URLs**
  (`https://hf-mirror.com/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/UD-Q4_K_XL/<shard>`),
  which 302-redirect to the HF Xet AWS CDN. Sustained ~3.5–18 MB/s.
- The user's old `~/Downloads` copy is **stale/incomplete** (shard 2 is 100 MB
  *larger* than current source = older repo revision; shard 3 is 8.8 GB short).
  Mixed-revision shards won't load → clean re-download to
  `D:\models\Qwen3.5-122B-A10B-Q4_K_XL\`.

## Reference runtime

User has a built **llama.cpp Vulkan / Ryzen-AI** tree at
`~/Downloads/llama.cpp/llama.cpp-claude-vulkan-ryzen-ai-optimization-Sl23N/build-baseline/bin/`
(`llama-cli/llama-server/llama-bench.exe`). Use it for (a) the immediate "usable"
result once the download finishes and (b) the numeric/throughput reference for the
ARLE bring-up. ARLE's `vendor/llama.cpp/vulkan-shaders/` is the same shader family.

## Current state of infer-vulkan (the gap)

`infer-vulkan` is a **seam-correct skeleton with zero numeric execution**:

- `executor.rs` implements the `infer-seam` `BackendExecutor` (submit/poll) + host
  sampling, but `self.model` is never constructed; `load_qwen3_gguf` parses then
  `bail!`s; `forward_tokens` `bail!`s "no model loaded".
- All five `model_*.rs` pin the exact CUDA-authoritative op order / launcher→kernel
  maps / attention specs / MoE routing, but every `forward_token` `bail!`s.
- `kv_pool.rs` is the one real file (host page allocator, tested) — host page ids
  only, no device arenas.
- `vulkan-sys` runtime is complete for single-queue compute (device pick gated on
  int8/fp16/16-bit-storage/integer-dot-product — matches RDNA3.5), host-visible
  STORAGE buffers + H2D/D2H, one-shot synchronous submit, pipeline w/ push +
  specialization. **Missing:** device-local staging, fences/timeline (only
  `queue_wait_idle`), pipeline barriers, pipeline/descriptor caching (rebuilt per
  launch).
- `vulkan-kernels` registers: get_rows(F32), rms_norm, rope norm/neox, soft_max,
  flash_attn (scalar), silu/gelu/geglu/swiglu(+clamped), add, argmax, q8_1
  quantize, **GEMV** mul_mat_vecq Q4_K/Q5_K/Q6_K + Q2_K/IQ2_XXS, and the DSv4 /
  Qwen3.5 model-specific serial shaders. **Missing for a generic 4-bit MoE
  forward:** general dense GEMM (`mul_mm`), quantized mat-mat GEMM (`mul_mmq`),
  MoE router/top-k (argsort), expert GEMM (`mul_mat_id`). `mul_mm.comp` /
  `mul_mmq.comp` / `mul_mm_id_funcs.glsl` exist in `vendor/llama.cpp/vulkan-shaders`
  but are unwired in `build.rs` + the `Kernel` enum.

## Phases

### Phase 0 — Toolchain unblock — DONE
- Installed VS Build Tools (Desktop C++) → MSVC linker.
- Fixed `vulkan-kernels/build.rs::find_glslc` to also try `glslc.exe` on Windows;
  set `ARLE_VULKAN_GLSLC` (User scope).
- `cargo build --no-default-features --features cli,vulkan,no-cuda` is **green**
  and emits **28 SPIR-V** shaders; `arle.exe` builds. Exit criterion met.

### Phase 1 — Substrate kernels + runtime (CC-execution, heavy)
Register the missing GEMMs in `vulkan-kernels/build.rs` + `Kernel` enum:
`mul_mm` (dense prefill), `mul_mmq` (quantized prefill W4A8), a MoE router/top-k
shader, and `mul_mat_id` (expert GEMM). In `vulkan-sys`: pipeline barriers between
dispatches + persistent pipeline/descriptor caching (drop rebuild-per-launch);
device-local staging is a fast-follow. **Exit:** a host micro-bench dispatches
`mul_mmq` + `mul_mat_vecq` + `soft_max` + `flash_attn` against CUDA/llama.cpp
reference vectors within tolerance.

### Phase 2 — Residency loader + first tokens (CC-execution)
Build `VulkanLoadedModel` construction in `executor.rs`: parse GGUF, dequant/upload
Q4_K weights into `DeviceBuffer`s, set `self.model = Some(..)`. Use the **embedded
GGUF tokenizer** (loader currently wants a sibling `tokenizer.json` — teach it to
read the embedded one, or extract it). Prove the residency+forward path on a small
dense model first (Qwen3 dense) for the first greedy token, then a small MoE.
**Exit:** greedy decode emits coherent tokens on the 8060S.

### Phase 3 — Serve Qwen3.5-122B-A10B (qwen35moe) (CC-execution)
Wire `model_qwen35.rs` + `model_qwen36.rs` residency binding + dispatch
(gated-delta + ssm_conv + MoE router/`mul_mat_id`). Verify `infer-gguf` classifies
arch string `qwen35moe` and the loader/config covers the Gated-DeltaNet layout.
Validate prefill+decode on the ~77 GB Q4_K_XL checkpoint within 96 GB VRAM.
**Exit:** `arle serve --backend vulkan` end-to-end stable decode; compare tok/s
against the llama.cpp Vulkan reference.

### Cross-cutting (lower priority)
- `cli/src/hardware.rs detect_gpu` has no AMD/Vulkan probe (nvidia-smi + macOS
  only) → 8060S shows "none detected" in `--doctor`; add an AMD/Vulkan variant.
- Model download is gated to cuda/metal/cpu in `cli/src/lib.rs` → vulkan-only
  binary can't `arle model download`; add `vulkan` to the gate or download
  out-of-band (current approach).
- Sampling is argmax-only on device; host sampling via `infer_plan::sample_token`
  already works; add temperature/top-k/top-p later.

## Risks
- Phase 1 is the real cost: residency upload + prefill-capable dense/quantized GEMM
  + the entire MoE path (router/top-k + `mul_mat_id`) — none exist yet. All MoE
  targets block on this.
- `vulkan-sys` is single-queue, host-visible-only, synchronous — fine for a
  correctness bring-up, a perf risk at 77 GB on the iGPU (staging/persistent
  pipelines/barriers are non-trivial follow-on).
- gfx1151 Vulkan on Windows is less-trodden; coopmat flash-attn variants are
  unregistered (scalar only), capping attention throughput.
- Unified memory: 96 GB VRAM holds ~64 GB weights, but host-visible buffers mean
  weights + KV + activations contend in the same 128 GB pool; 256K-ctx KV without a
  KV-quant shader could pressure the 96 GB carve-out.

## Validation ladder (on-box models, avoids the MXFP4 blocker)

ckl already has these GGUFs locally (verified quant types, all loadable by the
`infer-gguf` K-quant path — **no download needed**):

| Order | Model | arch | quant types | Size | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | `Qwen3.6-27B-Q8_0` (`models\qwen3.6\`) | `qwen35` (**dense**) | Q8_0 + F32 | 26.6 GB | Simplest forward (no MoE) — validate core + linear-attn first |
| 2 | `Qwen3.6-35B-A3B-UD-Q4_K_M` (`models\qwen3.6\`) | `qwen35moe` | Q4_K/Q5_K/Q6_K/Q8_0/F32 | 20.6 GB | MoE path; **same arch as the 122B**, K-quants ARLE can dequant |
| 3 | `Qwen3.5-122B-A10B-UD-Q4_K_XL` (`Downloads\`) | `qwen35moe` | **MXFP4** experts + K-quants | 63.7 GB | Final target; needs MXFP4 (ggml type 39) dequant added to `infer-gguf` |

llama.cpp baseline to beat (record per goal): see
[`docs/experience/wins/2026-06-13-llama-cpp-vulkan-8060s-baseline.md`](../experience/wins/2026-06-13-llama-cpp-vulkan-8060s-baseline.md)
— 27B dense 141 pp / 7.2 tg; 35B MoE Q4 822 pp / 47.3 tg; 122B 205 pp / 23.4 tg.

## Decision: GGUF→safetensors conversion — DEFERRED (押后)

Asked: convert GGUF→safetensors to validate ARLE functionality? **Deferred**, because:
- `infer-vulkan` has **no safetensors loader** (GGUF-only by design; `model_qwen3.rs:5`
  is a doc-comment stub) — conversion would target a format the Vulkan path can't load.
- This box has **only Vulkan + CPU** (no CUDA/Metal — the backends that load
  safetensors), so a converted checkpoint has no working ARLE consumer here.
- Conversion explodes size (4-bit→16-bit ≈ 4×: 35B 20.6→~70 GB, 122B 64→~244 GB).
- The validation targets (#1/#2 above) are **standard K-quant GGUFs that load
  directly** — GGUF *is* the working input format, so no conversion is needed to
  validate functionality. Revisit only if a future need (e.g. CUDA cross-check)
  arises.
