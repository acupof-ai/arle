# ARLE Support Matrix

This document is the canonical support-status truth for `ARLE`.

It states what the repository currently supports, what is still limited, and
what validation exists for each area. If something is not listed as supported
here, do not assume it is supported just because it compiled locally.

State reflected here is based on repository evidence as of 2026-06-14.
**The device-neutral rewrite (`crates/infer-*`) IS the product** — PR #53 merged
to `main` 2026-06-04, and the legacy monolithic `infer/` crate is **deleted**.
Sections 1–7 below were written against the legacy runtime; those capabilities
are now served by the rewrite, and a per-capability re-verification pass is
ongoing — read §0 for the **current** new-stack status, and treat any §1–§7
"Supported" as the *capability intent*, verified on the rewrite only where §0 or
a dated `wins/` entry says so. Project framing:
[index.md §Current Positioning](index.md#current-positioning).

---

## 0. Rewrite-stack support (the shipped `crates/infer-*` graph)

> **Merged to `main` (PR #53, 2026-06-04); legacy `infer/` deleted. This is the
> product.** The stack is `infer-plan` (IR) → `infer-seam` (host-only traits) →
> `infer-core` (`Engine<E,K>`) → `infer-cuda` / `infer-metal` (executors) →
> `infer-server` / `infer-api` (serving front door). Per-capability
> re-verification on the new stack is ongoing; the rows below are current status.
> Source of truth: the per-capability rows in this file, cross-checked against
> `docs/baselines.md` (champion benches) and `CHANGELOG.md`.

| Backend | New-stack status | What is verified | Open |
| --- | --- | --- | --- |
| Metal (`infer-metal`) | **Verified** | Real MLX Qwen3.5/3.6 forward, **bit-identical greedy parity** vs legacy across 4 configs (Qwen3.5-0.8B single-token / full 16-tok / chunked prefill, and canonical Qwen3.6-35B-A3B-4bit MoE). Cross-step decode pipeline recovered the c=1 perf regression. Target-only Metal is single-flight. With a loaded DFlash/NextN runtime, the executor advertises `INFER_METAL_DFLASH_MAX_ROWS` (default 4) and supports DFlash-specific multi-row and mixed scheduling. | Full packed batched-decode parity with CUDA; FP8/4-bit Metal MoE quant swap points. |
| CUDA (`infer-cuda`) | **Verified (serve: prefill + decode)** | Qwen3 dense **16/16** vs HF gold; DSv4-Flash **TP=8/EP=8** (MLA+CSA/HCA+FP8 DeepGEMM MoE, FlashMLA, DeepEP/allreduce) serves in-process multi-rank (`63d814a4`); per-layer RoPE theta fix (`fa355315`) → needle exact at 32K; KV-precision parity gate re-ported + FlashMLA decode licensed (#58, 2026-06-10); INT8/FP8 paged quant-KV dispatch landed opt-in, correctness licensed (#68); DSv4 B>1 decode now takes the batched lane by default, and MTP D2/T2 verifies with chain-shaped rows (`draft_rows=2`, `verify_rows=3`, not full-tree 7) when `--spec-type mtp --mtp-draft-tokens 2 --mtp-draft-topk 2` is enabled. | DSv4 c>1 throughput still needs DP-attn (#89); lm_head vocab-parallel shard (#99). (#56/#57 closed.) |
| Vulkan (`infer-vulkan`) | **Experimental (bring-up)** | Qwen3.5 27B dense (`Qwen3.6-27B-Q8_0`) GGUF forward is now **coherent** on Radeon 8060S (Strix Halo, gfx1151) — three HF→GGUF convention bugs fixed (plain-`w` RMSNorm, gated-delta decay sign, partial rotary `rope.dimension_count=64`) (`324dbaff`, wins); the 35B-A3B MoE FFN now runs coherent too (`79598456`). A record-many/submit-once recorder + real `vk::PipelineCache` + arena take the dense forward **5.0→3.0 s/tok** (perf Steps 1–4). Device residency upload, per-row dequant, GGUF→`Qwen35Config` mapper, and a GGUF-embedded BPE tokenizer landed. | Still ~0.33 tok/s vs llama.cpp's 7.2 tg reference bar for the same shape (wins); no serving path yet; perf-parity is the bring-up goal (plan, #71). |
| HIP/ROCm (`infer-hip`) | **Experimental (substrate)** | Stage A (GGUF loader, CPU dequant, DSv4 config map, slot pool) + Stage B (DSv4 forward orchestration over the HIP kernel surface + `BackendExecutor`) landed; GGUF host substrate extracted to `infer-gguf`. | AIPC executor MVP + perf license (#77/#78); not a serving target yet. |

**New-stack model coverage:** Qwen3.5 / Qwen3.6 on Metal (verified) **and now CUDA**
— FP8 DeepGEMM MoE + batched paged decode landed (Qwen3.6-27B-FP8 1×H20 tok/s
scales c=1→8: 21→26; [wins](experience/wins/2026-06-29-cuda-qwen36-paged-batched-decode.md));
vanilla Qwen3-MoE loads + runs the full CUDA forward but is **numerically wrong —
not shipped**, decode-level debug tracked in
[#120](https://github.com/cklxx/arle/issues/120)
(wins).
Qwen3 dense + DeepSeek-V4-Flash (TP=8/EP=8) on CUDA (prefill + decode verified,
needle-exact to 32K, ~53 tok/s c=1). GLM-5.2 (`glm_moe_dsa`) is wired onto the
DSv4 CUDA path via an adapter — forward tranches landed but verification is
[pending-remote](experience/wins/2026-06-19-glm52-tranche-d-forward-pending-remote.md), not
production-verified. On Metal, three VLMs are bring-up: DiffusionGemma (block
diffusion, smoke + 60 tok/s fast path), Gemma4 (VLM smoke + image bench, 4bit),
and DeepSeek-OCR (DeepEncoder + MXFP8 MoE decoder, experimental). DiffusionGemma
has the backend-neutral block-diffusion generate-loop substrate wired through
`infer-plan`/`infer-seam`, Engine completion and repeated-prompt tests, Gemma4 /
DiffusionGemma config parsing, and a first Metal `MetalDiffusionGemmaModel`
bridge routed through `infer-api`. CUDA/Vulkan DiffusionGemma/Gemma4 forward
paths still fail closed. Target 26B completions/chat smoke passed locally after
the checkpoint download; throughput and long-generation validation remain pending.

**Now in the new stack (was legacy-only):** TP / EP / DeepEP multi-GPU, DeepGEMM
(FP8 grouped GEMM), DSv4 (MLA + FP8 KV) with incremental decode + MTP
speculative decode (**shipped, explicit opt-in** via `--spec-type mtp` / `--spec-type dspark`), INT8/FP8 paged quant-KV
dispatch (opt-in, #68), tiered KV **T1 default-on + T2 opt-in disk spill**
(#82–#84), and the HTTP/serving surface (`infer-server` + `infer-api`, both
executors wired). **Still pending re-port / verification:** PP (pipeline
parallel), the full **weight** quantization Rust dispatch matrix, T3
cluster-shared KV (stub only, #87), and the Qwen3.5/3.6 sidecar prefix-reuse
pod verification (#85 landed `52e2fdb4`, verify pending-remote; DSv4 reuses via
the position-0 whole-slot store instead) — tracked in §1–§7 + the active tasks. The capability detail in §1–§7
below predates the rewrite; verify against §0 + dated `wins/` entries.

---

## 1. Runtime Backends

| Backend | Status | Meaning |
| --- | --- | --- |
| CUDA | Supported | Primary serving path. Main runtime, scheduler, and benchmark focus. |
| Metal | Beta | Usable for local validation and live scheduler-backed serving. Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots; `arle serve --backend metal` is the canonical Apple bring-up path (in-process serve). The `arle serve` Metal path is measured across the model ladder on M4 Pro (512-in/128-out, c=1, decode = single-stream): Qwen3.5-0.8B **318 tok/s**, 4B 84, 9B 50, Qwen3.6-35B-A3B MoE **85 tok/s** ([snapshots](../benchmarks/README.md), wins). Metal is still missing full batched-decode parity with CUDA, especially on variable-length Qwen3.5 decode. |
| Metal DFlash/NextN | **Shipped for compatible models** | Qwen3.6-27B auto-enables its default MTP draft unless `--no-speculative`; `--draft-model` selects an explicit compatible draft. CUDA-oriented `--spec-type` and `--mtp-draft-model` remain unsupported and are rejected on Metal. |
| no-cuda / CPU-only | Development-oriented CPU backend | Build, test, and smoke-validation path for non-GPU logic. Not a production inference target. |

---

## 2. Platform Matrix

| Platform | Backend | Status | Validation |
| --- | --- | --- | --- |
| Linux x86_64 + NVIDIA GPU | CUDA | Supported | Release workflow builds CUDA artifacts; primary target. |
| macOS Apple Silicon | Metal | Beta | CI checks and tests Metal/no-cuda surfaces. |
| Linux/macOS without GPU | no-cuda | Development-oriented CPU backend | Unit tests, compile checks, and CPU backend smoke validation. |

### CUDA GPU / SM Matrix

Tier policy and rationale: see [`environment.md`](environment.md).
Env var contract: see [`environment.md`](environment.md) §`TORCH_CUDA_ARCH_LIST`.

| Tier | SM | Representative GPUs | Status | Default-built |
| --- | --- | --- | --- | --- |
| T1 | sm_80 | A100 40/80GB | Supported | yes |
| T1 | sm_86 | A10, RTX 3090, A40, A6000 | Supported | yes |
| T1 | sm_89 | L4, RTX 4090, L40 | Supported | yes |
| T1 | sm_90 | H100, H200 | Supported | yes |
| T2 | sm_100 | B100, B200 | Beta — opt-in via `TORCH_CUDA_ARCH_LIST` | no |
| T2 | sm_120 | RTX 5090, RTX PRO 6000 | Beta — opt-in via `TORCH_CUDA_ARCH_LIST` | no |
| T0-legacy | sm_70 | V100 | Legacy — SM-pinned Qwen3.5 BF16 attention + GDR lane | no |
| T3 | other sm < 80 | T4, Pascal, older | Unsupported — build rejects | n/a |

Notes:

- Hosted CI does not provide full CUDA runtime correctness coverage.
- CUDA correctness and performance still require dedicated GPU validation.
- T1 ship gate requires four-card bench validation (sm_80 + sm_86 + sm_89 + sm_90); see [`environment.md`](environment.md).
- sm_70 builds must be SM-pinned (`TORCH_CUDA_ARCH_LIST=7.0`) and are limited
 to the V100 Qwen3.5 BF16 attention + GDR path while Volta fallbacks are
 validated.

---

## 3. Model Family Matrix

| Model family | Status | Notes |
| --- | --- | --- |
| Qwen3.5 | Supported | Primary supported family. Supported on normal runtime paths; Metal live runtime has a narrow same-length decode batch path with packed-batch concurrent decode (2026-04-16 fix). Qwen3.5-0.8B serves at **318 tok/s** decode on M4 Pro (512-in/128-out, c=1; [snapshot](../benchmarks/README.md)). RoPE scaling (YARN / Linear / NtkAware) wired through `Qwen35Config::rope_scaling` for long-ctx extend (Phase 1+2 closed; Phase 3 bench pending). Metal DFlash/NextN serving is shipped for compatible models; see §4a. |
| Qwen3.6 / Qwen3.5-MoE | Supported (Metal + CUDA) | `mlx-community/Qwen3.6-35B-A3B-4bit` is the **canonical Metal production model** (globally unified 2026-05-07) — every Metal serve/bench/test defaults to it. **CUDA serving landed**: FP8 DeepGEMM MoE + batched paged decode (Qwen3.6-27B-FP8 1×H20 tok/s scales c=1→8: 21→26, [wins](experience/wins/2026-06-29-cuda-qwen36-paged-batched-decode.md)); vanilla Qwen3-MoE runs the full CUDA forward but is numerically wrong — not shipped, tracked in [#120](https://github.com/cklxx/arle/issues/120) (wins). Qwen3.5-122B-A10B serves at TP4 via GQA KV-head replication (TP > num_kv_heads, all 4 worker engines ready) — numerical-completion gate pending a clean re-run ([wins](experience/wins/2026-06-29-cuda-gqa-replication-122b-tp4.md)). |
| DeepSeek V4 | Serving (CUDA 8×H20 TP=8/EP=8) | DSv4-Flash serves via `arle serve --backend cuda` in-process multi-rank: FlashMLA + DSA/CSA/HCA hybrid attention, FP8 KV, DeepGEMM FP8 MoE, DeepEP/allreduce transports. Needle-exact to 32K after the per-layer RoPE theta fix (`fa355315`). Current MTP path is explicit (`--spec-type mtp`): D2/T2/topk=2 uses the B>1 batched spec lane and keeps verifier cost chain-shaped (`verify_rows=3`; wins). DP-attn is the remaining throughput lever for c>1 (#89); lm_head shard is #99. Phase-0 debt (#56/#57) closed; umbrella #55. **GLM-5.2** (`glm_moe_dsa`, DeepSeek-V3.2-DSA family, 256 experts) rides this same DSv4 CUDA path via an adapter — forward tranches wired, but verification is [pending-remote](experience/wins/2026-06-19-glm52-tranche-d-forward-pending-remote.md), not production-verified. `crates/deepseek-spec` now carries V4 + GLM-5.2; DSv4 scratch pretrain stays retired (2026-05-18 OPD-only pivot). |
| DiffusionGemma | Metal smoke + 60 tok/s fast path; quality pending | `infer-plan` contains a backend-neutral block-diffusion generate loop matching the public DiffusionGemma generation contract shape: fixed canvas, denoise passes, entropy-bound acceptance, stability/confidence convergence, and whole-canvas commit hook. `infer-seam::BufferedDiffusionExecutor` adapts that loop into the normal `BackendExecutor`/Engine path, disables cross-request prefix reuse, and honors Engine-normalized `max_tokens`. `gemma-spec` parses the top-level DiffusionGemma config, nested Gemma4 RoPE map, and MoE fields. `infer-metal` now loads `model.decoder.*` with per-weight MLX quantization overrides, registers a dedicated C++ Gemma4/DiffusionGemma forward bridge, handles full-attention K=V layers, self-conditioning, tied embedding logits, and canvas-sized device sampling summaries, then `infer-api` routes it as `MetalDiffusionGemma`. The OpenAI facade reads both inline `tokenizer_config.json` templates and external `chat_template.jinja`, matching the downloaded checkpoint layout. Local target smoke passed for `/v1/completions` and `/v1/chat/completions` on `mlx-community/diffusiongemma-26B-A4B-it-4bit`, and a 64-token chat completion reaches **60 generated tok/s** on the Metal fast path (`ARLE_DIFFUSION_MAX_DENOISING_STEPS=4`, [wins](experience/wins/2026-06-12-diffusiongemma-metal-fast-path-60tps.md)); long-generation quality and memory-pressure validation remain pending. CUDA/Vulkan classification fails closed instead of falling through to Qwen/Qwen3 dense. |
| Gemma4 | Metal VLM smoke + image bench; quality/throughput pending | Gemma4 (SWA + full attn, optional MoE, image-capable VLM) loads and serves on Metal at 4bit: text smoke, image VLM smoke, and a CLI image VLM bench all pass ([wins](experience/wins/2026-06-15-gemma4-metal-e2b-4bit-smoke.md), [vlm-cat](experience/wins/2026-06-15-gemma4-metal-vlm-cat-smoke.md), [cli-image-bench](experience/wins/2026-06-15-gemma4-cli-image-vlm-bench.md)). Quality and throughput validation remain pending. CUDA classifies Gemma4 but fails closed (no autoregressive forward). |
| DeepSeek-OCR | Metal VLM, experimental (vision numerics pending) | DeepSeek-OCR (`UnlimitedOCRForCausalLM`: SAM+CLIP DeepEncoder + MXFP8 MoE text decoder) loads and serves OpenAI v1 on Metal; the MXFP8 decoder is verified correct, but the DeepEncoder vision numerics are **not yet faithful** (OCR output not yet correct) — bring-up only ([wins](experience/wins/2026-06-24-deepseek-ocr-metal-bos-fix.md), [decoder](experience/wins/2026-06-25-deepseek-ocr-bos-loop-kv-ring-cli-agent.md)). |
| Llama 3/4 | Planned | Not yet supported. |
| DeepSeek-V3/R1 | Not carried | Deleted from the current registry/spec/train surface; reintroduction would require a new explicit project, not a compatibility branch inside DSv4. |
| Mistral / Mixtral / Gemma / Phi | Planned | Not yet supported. |

**Next-model roadmap priority:**

1. **DeepSeek V4 (DS4)** — serves at TP=8/EP=8 on CUDA; GLM-5.2 (`glm_moe_dsa`) rides the same path (verification pending-remote). Long-ctx residual + DP-attn for c>1 are the remaining runtime levers.
2. **Qwen 3.6** — **shipped on CUDA** (FP8 DeepGEMM MoE + batched paged decode) and Metal; the 122B-A10B TP4 numerical-completion gate is the remaining follow-up.

Other "Planned" families above sit behind these and are not actively scheduled.

---

## 4. Quantization Matrix

**Canonical map**: [`docs/quantization.md`](quantization.md). That doc is
the source of truth for KV-cache and weight quantization status, code
locations, test-harness semantics, and the active TileLang HD128
batched paged-prefill investigation (2026-05-27). The summary table
below is the one-glance view — for any change, edit
`quantization.md` first and re-sync here.

| Capability | Status | One-line |
| --- | --- | --- |
| BF16 KV cache | production | Explicit reference via `--kv-cache-dtype bf16`; correctness-safe fallback. |
| INT8 KV cache (Metal + CUDA) | production (Metal default; CUDA production) | Metal `--kv-cache-dtype auto` resolves to `int8` and stores full-attention K/V as MLX affine 8-bit packed triples; `bf16` remains the explicit fallback. CUDA `--kv-cache-dtype int8` uses KIVI per-channel K + per-row V; +57–113% throughput vs BF16 on A100 (`wins/2026-05-26-bench-int8-vs-bf16-kv-a100`). |
| FP8 E4M3 KV cache (CUDA, +KIVI) | opt-in | `--kv-cache-dtype fp8`; KIVI per-channel K + per-token V scaffolding (`8c6d92db`/`73a72615`/`25c7d409`); quality verdict deferred pending §5 paged-prefill investigation. |
| TurboQuant KV 2/3/4-bit (CUDA) | experimental | `--kv-cache-dtype tq{2,3,4}`; FWHT + packed indices; page_size=1 bypasses the HD128 paged prefill — the only KV format that matches the HF first token on the 2026-05-27 chat audit. |
| Weights — W4A16 / W8A16 / W2A16 | production / experimental (W2) | Native GEMV + Marlin W4 prefill; safetensors auto-detect. |
| Weights — MarlinW4A8 prefill-graph | production, **Tier-1 wins** | `INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1` → engine TTFT p50 –92.5%, +632% throughput (`a56b7a9`/`c44788f`). |
| Weights — GGUF Q3/Q4/Q5/Q6_K | production (CUDA & Metal) | Packed superblock kernels; `.gguf` auto-detect. Metal-native-q4 opt-in via `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all`. |
| Weights — TurboQuant | experimental | Tensor-local gate only (`errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill`). |
| Weights — DSv4 FP8/FP4 block-scaled | in progress | `Dsv4Fp8BlockScaled` / `Dsv4Fp4BlockScaled`; pending CUDA V4 attention/MoE/MTP kernels. |

Backend reach:
- Quantized KV cache is supported on Metal for INT8 only. Metal `auto`
 defaults to INT8 full-attention KV (MLX affine 8-bit groups) with
 `--kv-cache-dtype bf16` as the reference fallback. Metal does not support
 FP8/TurboQuant KV. Metal weight-quantized MLX models are unaffected.
- **Rewrite-stack CUDA `--kv-cache-dtype` dispatch landed (#68, 2026-06-12).**
 Seam-level INT8/FP8 paged quant-KV on the dense-Qwen3 CUDA path is wired
 end-to-end (refill / KIVI calibrate / row-quantize / fused-dequant decode) —
 **correctness LICENSED** (needle exact = BF16 envelope), **opt-in only, no
 default flip** (default stays BF16; post-fix −27% vs bf16+graph, −7% vs eager
 bf16; the initial −77% was an uncached `cudaGetDeviceProperties`, fixed same
 day). TQ4 deferred on the page_size=1 vs TileLang PAGE_SIZE=16 mismatch.
 [wins](experience/wins/2026-06-12-cuda-quant-kv-dispatch-int8-fp8.md).

---

## 4b. Multi-turn KV Reuse / Tiered KV Matrix

Rewrite-stack truth as of 2026-06-11 (#80 audit; tier re-port landed the same
day: #82 `4409491c`+`e2cc27ba`, #83 `b7458707`, #84 `9cadad3f`). The host
radix cache (`crates/infer-core/src/{prefix,radix}.rs`) shares page-aligned
prompt prefixes between slots; under page pressure, evicted blocks **demote
into the executor's host tier store** (when one exists) and **promote back**
on the next prefix match instead of re-prefilling. Engine-side correctness is
mock-tested; **CUDA-traffic verification (needle gate + multi-turn TTFT A/B)
is pending pod time** — see the dated `wins/` entries per row.

| Capability | Status | Notes |
| --- | --- | --- |
| Slot-sticky multi-turn KV reuse | Supported (CUDA), Beta (Metal) | Prior-turn KV stays in slot for the next turn so only new user tokens prefill. CUDA is the primary path; Metal Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots (see §1). |
| Radix-backed prefix cache (T0 GPU) | Supported (CUDA, **Qwen3-dense only**), Beta (Metal, GDR-clamped) | Page-aligned full-block radix (`RadixCache`) with retain/release refcounts and LRU evict. Qwen3.5/3.6 hybrid re-enabled via the recurrent sidecar snapshot protocol (#85, `52e2fdb4`; pod verify pending-remote); DSv4 stays off the page radix by design — it reuses via the position-0 whole-slot store (§4b). No tail-page CoW (no n>1 consumer, #86 closed as backlog). |
| T1 DRAM tier (host store) | **Default-on (all CUDA arms), verified on H20 pod** | One `CudaKvTierStore` per arm (Qwen3.6 adds a lazily-built `--kv-recall` tier). Qwen3-dense: page-granular radix demote/promote. Qwen3.5/3.6: whole-slot G3 capacity spill, stored as 16 MiB chunked blobs (same path as DSv4). The page-granular `KvPageTier` path is stubbed for Qwen3.5/3.6 by design — recurrent sidecars carry the restorable state, full-attn KV pages stay device-resident under radix reuse; a prefix-cache eviction under VRAM pressure loses full-attn pages (recompute on next hit) where dense Qwen3 demotes them. DSv4: whole-slot spill stored as 16 MiB chunks under a per-key manifest. Budgeted by `--kv-dram` (bytes / `%` of MemAvailable / `0`=off; default `50%`) — a DEPLOYMENT-TOTAL cap split across TP ranks at engine build. Read-hit telemetry (`kv_tier_read_hits`) and L3→L2 promote-on-read are wired on all arms ([wins](experience/wins/2026-08-17-kv-tier-read-hits-and-promote-on-read.md)). [wins](experience/wins/2026-06-30-kv-mmap-tier-e2e.md). |
| T2 NVMe disk spill | **Opt-in (Qwen3-dense + DSv4 TP=4 multiproc verified on H20 pod)** | `--kv-disk [DIR]` + `--kv-disk-limit` (bytes / `%` of free disk; default 50% of free disk at the root) ride `EngineLoadConfig` (`kv_ssd_root`/`kv_disk_limit`). The L3 cap is DEPLOYMENT-TOTAL, split across TP ranks at engine build; every rank attaches the tier inside the engine constructor — single-proc and multiproc TP workers alike; each process namespaces its own `KvMmapStore` (sparse mmap page-slot store) under the root. Per-arm granularity matches T1 (dense page-granular batch-H2D Manifest V2; Qwen3.5/3.6 whole-slot; DSv4 16 MiB chunks). Metal attaches at construction; non-consuming backends fail closed. [wins](experience/wins/2026-06-30-kv-mmap-tier-e2e.md), [wins](experience/wins/2026-07-02-dsv4-kvssd-multiproc-attach.md). |
| T3 cluster-shared backend | Stub only | **NIXL transport remains stub-only** (`nixl-sys` activates the stub feature, no real link); the rewrite has no KV export/import surface. P/D-disagg backlog in #87. |
| DSv4 position-0 prefix cache | **Default-on, rides THE tier store — pod-verified through NVMe** | Token index (`PrefixIndex`) over snapshot blobs stored in the shared `CudaKvTierStore` under `NS_PREFIX*` key namespaces (16 MiB chunks) — budgeted by the ONE DRAM budget, spills to NVMe when `--kv-disk` is set. LRU under store pressure (prefixes yield, parked slots never touched). The private BTreeMap cache and its `ARLE_DSV4_PREFIX_CACHE_BYTES` knob are deleted. TP=4 verified: 3375-tok prompt cold 26.6s → restored **0.47s** from disk, hit_rate 1.0, zero restore errors ([wins](experience/wins/2026-07-02-kv-unified-tier-flags-pod-verify.md)). Blob accounting attributed and bounded (supersede-in-place verified via capture logs; distinct suffixes bounded by the per-rank cap + coldest eviction); store write 1.20 GiB/s / warm read 2.22 GiB/s vs 197 MB/s device QD1 ([wins](experience/wins/2026-07-02-kv-tier-round4-close.md)). |
| Swap-style preemption | **Shipped with the tier (tier-gated)** | `retract_decode_to_fit`'s victim seals its prompt blocks into the radix and demotes exactly those pages; re-admission promotes instead of re-prefilling (decode is still recomputed). Without a tier store the recompute path is byte-for-byte unchanged. DSv4 whole-slot variant tracked under #84/#85 Route B. |

---

## 4a. Speculative Decoding Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Metal DFlash (Qwen3.5 dense/MoE) | **Shipped for compatible models** | `--draft-model` selects a compatible draft; model/checkpoint compatibility fails closed, and CUDA-oriented `--spec-type` and `--mtp-draft-model` are rejected on Metal. |
| Metal NextN/MTP (Qwen3.6) | **Shipped** | Qwen3.6-27B-MTP-4bit: 12.3 → 17.75 tok/s (+44%), 68.8% draft acceptance, bit-identical output. Default-on, `--no-speculative` to disable. ([wins](experience/wins/2026-06-21-metal-qwen36-mtp-spec-decode.md)) |
| CUDA speculative decoding | DSv4 MTP (explicit opt-in) | `arle serve --backend cuda --spec-type mtp --mtp-draft-tokens N --mtp-draft-topk K` lowers into the DSv4 checkpoint-native MTP head. `N` is clamped `[1, 8]`; `K` is the per-level draft candidate width. The verifier stays on the chain-shaped path, so D2/T2 uses 3 verifier rows while top-k only widens candidate matching (`1c41c4a8`, wins). Classical/self/external draft routes remain not shipped; Qwen3.5 Medusa is blocked on recurrent-state accepted-length rollback. |
| CUDA DSpark block-drafter (Qwen3.6 + DSv4-Flash) | **Shipped** | `arle serve --backend cuda --spec-type dspark --mtp-draft-model <DSpark/DFlash dir>` loads the external block drafter. Each Qwen request verifies its block in one target forward, and a greedy Qwen DSpark batch shares one draft forward, one snapshot, one capture and one rollback replay across slots (`--spec-max-batch`, default 16). MTP still loops per row and stays gated at 1; so do sampled batches and quant-KV pools. DSv4 has a separate cross-slot batched speculative path. |
| DSpark train sidecar | **Shipped (opt-in)** | `--dspark-train` (requires `--spec-type dspark`, CUDA-only): spawns a background thread that drains the experience buffer the hot path populates and runs acceptance-weighted policy gradient + probability matching on the Markov head, hot-swapping updated weights back into the running engine. Seeds from the loaded checkpoint so acceptance never regresses at startup. E2E verified on H20 ([wins](experience/wins/2026-07-20-dspark-train-sidecar-e2e-verified.md)). No-op without `--spec-type dspark` or on non-CUDA backends. |

---

## 5. Public API Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `/v1/completions` | Stable | Documented public API. |
| `/v1/chat/completions` | Stable | Documented public API. |
| `/v1/models` | Stable | Loaded-model discovery endpoint. |
| `/v1/responses` | Beta | Non-streaming and SSE forms shipped. Streaming emits `response.created`, `response.output_text.delta`, and terminal `response.completed`; structured outputs are still missing. |
| SSE streaming | Stable at high level | Intended to remain OpenAI-style; edge behavior may improve. |
| `/metrics` | Supported (re-ported, #81) | Prometheus text exposition of the same engine-tick counter snapshot `/v1/stats` serves: scheduler gauges (`arle_active_requests`, `arle_queue_depth`, `arle_kv_free_pages`) + prefix-cache counters, labelled `model_name=...`. Host-side and backend-neutral; off the request hot path. Smoke-verified on CPU + Metal serve (wins entry). |
| `/v1/stats` | Minimal rewrite surface | JSON scheduler counters plus prefix-cache hit counters. SSD KV recall reports `available=false` because the rewrite serve path has no active SSD recall tier. |
| Train-side `/v1/train/status|events|stop|save` | Removed 2026-07-24 | Parked control-plane substrate (`server.rs`/`control.rs`/`metrics.rs`) deleted as zero-production-caller dead code (never wired after the 2026-05-18 pivot; `--train-control-url` was rejected by the in-process serve stack). Recover from git history if `arle train opd --serve` is ever licensed. |
| Metal runtime memory knobs | Auto (no flags) | The rewrite Metal executor auto-pins model weights via `mlx::set_wired_limit` (model size + 1 GiB headroom, `infer-metal/src/wired_limit.rs`) at construction. The old `--memory-limit-bytes` / `--cache-limit-bytes` / `--wired-limit-bytes` flags died with the monolith; no CLI override currently exists. |
| CLI agent slash commands | Beta | Usable and documented, but not yet treated like the HTTP API for compatibility. |
| `arle serve` front door | Stable | **In-process** serving (`crates/cli/src/serve.rs`): the single `arle` binary loads the model and serves OpenAI v1 directly — no standalone backend binaries are spawned or searched. `--bind` is honored by every backend. |
| CLI built-in shell/python tools | Beta | Enabled by default for local trusted agent use. `--no-tools` disables them, and `arle --doctor` reports the detected sandbox backend (`nsjail`, `sandbox-exec`, or `bare`). Do not expose tool-enabled local agent prompts to untrusted users. |
| Structured-output grammar (xgrammar FFI) | Scaffold (Phase 1) | `crates/xgrammar-sys` Rust safe wrapper over upstream `mlc-ai/xgrammar` v0.1.34 (codex's #26 WIP, FFI substrate landed; default build = stub, `--features real` builds C++ shim via `cc` + pinned upstream checkout). No HTTP, scheduler, sampler, or GPU sampling integration yet. |

## 5a. Training Surface Matrix

> **2026-05-18 pivot — OPD only.** Scratch pretrain, SFT, GRPO, and
> multi-turn RL surfaces were retired in commit `bd94c09`.
> Rationale: the nanochat-d12 industry baseline measured 56 291 tok/s
> single-GPU on this hardware vs ARLE 174.7 tok/s = 322× gap, making
> from-scratch pretrain not a winnable axis; SFT/GRPO/multi-turn
> duplicate mature OSS (vLLM+verl, TRL, axolotl). OPD is the one
> training surface where ARLE's pure-Rust runtime authority is
> structurally differentiating — teacher hosted in `infer`, student
> LoRA on the same backend, no Python on the hot path. Historical
> validation evidence for the retired surfaces lives in
> `docs/experience/wins/` (immutable per bench-spec §9) and is not
> removed.

Two objective families ride one substrate (teacher/rollout on `infer`, student
LoRA + autograd `Tape` on the same backend): **OPD** (KL distillation) and **RFT**
(reward-selected masked-CE / policy-gradient). Source of truth: `TrainCommand` /
`UpdateStrategyArg` in `crates/cli/src/args.rs`, `UpdatePreset` in
`crates/train/src/update_strategy.rs`.

| `arle train <cmd>` | Family | Mechanism | Reward / signal | Backend | Status |
| --- | --- | --- | --- | --- | --- |
| `opd` | OPD (distill) | teacher forward → student rollout → KL + optional GKD hard-token CE anchor (`--sft-anchor student-rollout\|corpus-truth`, `--kl-direction`, `--kl-mask`) | teacher logits (KL) | CPU / CUDA | Supported (Beta). Wins: `2026-05-24-arle-train-opd-from-dirs`. |
| `self-opd` | OPD (self-distill) | EMA self-teacher; GKD = λ·CE + (1−λ)·KL(student‖EMA); every N steps a held-out NLL gate reverts {student, EMA, AdamW} on regress | self EMA teacher (`--gkd-lambda>0` required — pure KL cold-starts at zero gradient) | CPU / CUDA | Supported |
| `rubric-opd` | RFT | N rollouts/prompt → a Flash judge grades a text-level rubric **or** `--self-consistency` majority-votes → accepted rollouts written back as CE (`--writeback accepted\|correction`) | rubric (`--rubric-task math\|agentic`) or \boxed self-consistency | CUDA-only | Supported |
| `agent-opd` | RFT | student drives read/write/replace/bash tool loop against a per-task repo sandbox (SWE-bench-Pro, CC-harness) → run hidden tests → passing trajectories → CE / PG (`--update-strategy`, see below) | execution (pytest), no text judge | CUDA-only | Supported. Errors: `2026-07-22-agent-opd-dapo-null-gradient-trajectory-vram-wall`. |
| `cc-convert` / `ppl` | Utility | `/v1/messages` dump → verl token records (`agent-opd --replay-records`); teacher-forced perplexity for FP8/quant calibration | — | ppl: CUDA-only | Supported |
| `env` / `estimate-memory` | Diagnostic | env probe; param-count + rough memory estimate | — | — | Supported |
| DSpark train sidecar (`serve --dspark-train`) | Online spec-decode | in-process trainer alongside `arle serve --spec-type dspark`: drains the hot-path experience buffer → updates the Markov head, hot-swaps weights | PG + probability matching | CUDA-only | Shipped (opt-in). §4a; wins `2026-07-20-dspark-train-sidecar-e2e-verified`. |
| Infer-side `/v1/train/*` bridge | Control plane | Removed 2026-07-24 with the train control plane (`serve` had always rejected `--train-control-url`) | — | — | Removed |

**`agent-opd --update-strategy` presets** (each maps 1:1 to a `UpdatePreset`
constructor; `TrainAgentOpdArgs::update_preset`):

| Value | Algorithm | Filter / advantage |
| --- | --- | --- |
| `rejection-ce` *(default)* | reject failures, masked CE on full passes | `PassOnly` / `None` |
| `grpo` | group-normalized advantage, clamped per-token ratio | `KeepAll` / `Mean{group, std_norm}` |
| `dapo` | clip-higher + dynamic sampling + token-level (batch-mean) loss | `DropZeroAdvGroup` / `Mean{group}` |
| `dr-grpo` | GRPO minus length/std biases (fixed-constant normalizer) | `KeepAll` / `Mean{fixed norm}` |
| `gspo` | sequence-level clipped importance ratio | sequence-level IS |
| `cispo` | detached one-sided clamped-IS weight, batch-mean loss | one-sided IS |
| `sao-dis` | SAO Phase 1: hard-gated per-token PG, batch-centered advantage | `KeepAll` / `Mean` |
| `sao-value` | SAO Phase 2: per-token Skip-Obs GAE from a learned value critic | `ValueGae{γ, λ}`. Wins: `2026-07-13-sao-phase2-value-critic` |

`arle train test` was retired in the 2026-05-24 T3 prune (`81842cc`). The generic
Phase-2 `Trainer`/`GradAccumulator` and `MoeWithLora` surfaces were deleted
2026-07-23 (dead post-pivot; wins `2026-07-23-train-crate-systematic-simplification`).

---

## 6. CI Coverage Matrix

| Area | Coverage |
| --- | --- |
| Rust CPU-only compile/test | Yes |
| Python tests | Yes |
| Metal compile/test | Yes |
| CUDA compile | Partial |
| CUDA runtime correctness | No full hosted CI |
| Performance regression gating | Not yet standardized |

---

## 7. Update Rule

If support changes for a backend, model family, platform, or quantization path,
update all of the following together:

1. `README.md`
2. `docs/index.md` if the active-doc listing changed
3. this file
4. `CHANGELOG.md` when user-visible

Related docs:

- [stability-policy.md](stability-policy.md)
