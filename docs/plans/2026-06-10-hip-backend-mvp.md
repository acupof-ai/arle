# HIP backend MVP — Ryzen AI Max+ 395 (Strix Halo, gfx1151) op survey + wiring plan

**Status:** survey complete, verdict **GO** (H1 = #75). Tracks: #71 (AIPC umbrella),
chain #75→#76→#77→#78. Written 2026-06-10 from the source/web survey below; everything
marked *hypothesis* needs on-box confirmation during H2/H3.

## §0 Pinned target (exact, per execution-hygiene rule)

| Axis | Pin |
| --- | --- |
| SKU | Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S iGPU, 40 CU RDNA 3.5 |
| gfx arch | `gfx1151` (llama.cpp `common.cuh:80` names it "AI 370, AI Max 395 laptops"; `RDNA3_5` macro) |
| OS | Linux first (ROCm APU support is Preview and scoped to Linux; Windows out of MVP) |
| ROCm | 7.x known-good stack per [llama.cpp discussion #20856](https://github.com/ggml-org/llama.cpp/discussions/20856); TheRock nightlies ship native gfx1151 binaries — pin the exact version from the box once provisioned (pin-from-proven-env rule) |
| Memory | UMA/GTT: BIOS UMA minimum + GTT sizing via kernel params so the iGPU can map most of the 128 GB; ~212 GB/s measured GPU bandwidth ([llm-tracker](https://llm-tracker.info/_TOORG/Strix-Halo)) |
| Mission model | **DSv4-Flash at 2-bit weights** (ckl directive 2026-06-10) — the model only ARLE can serve, on a consumer 128 GB box. Weight format: IQ2-class i-quants (llama.cpp 2.06–2.5 bpw family, kernels vendored). Bring-up smoke: Qwen3-0.6B BF16 (in `models/`, proves the substrate before the big model) |
| Reference engine | llama.cpp HIP build on the same box (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON`, `ROCBLAS_USE_HIPBLASLT=1`, `GGML_HIP_NO_VMM=ON`) — bar set with an IQ2-class MoE it can run (it cannot load DSv4-Flash; nearest-proxy protocol per the 2026-06-04 SGLang A/B precedent) |

**Model-shape notes:** DSv4-Flash FP8 checkpoint is 149 GB (#69) ≈ ~149B params →
~39–47 GB at 2.06–2.5 bpw: fits the 128 GB UMA with room for compressed-attention KV
(DSA/CSA KV is small by construction — the long-ctx story survives quantization of
weights). Qwen3.5 is hybrid (3:1 gated-delta : hd256 full attn), NOT the dense
stepping stone — Qwen3-0.6B (plain GQA hd128) is. Per-tensor quant mix follows
llama.cpp dynamic-quant practice: attention/dense/shared-expert tensors stay at
higher bits, routed-expert FFNs take the 2-bit floor — exact recipe is an on-box
needle-gated experiment, not a guess.

## §1 Decode floor (formula before measurement)

B=1 decode is weight-bandwidth-bound: `bytes(weights touched per token) / 212 GB/s`.
For the MoE mission model the bytes are the ACTIVE subset, not total weights.

| Model | Bytes/token | Floor | Note |
| --- | --- | --- | --- |
| **DSv4-Flash IQ2-class** | active params × ~2.1–2.5 bpw + higher-bit attn/shared tensors — *pin active-param count from the checkpoint `config.json` on first session (hypothesis range: ~5–10 GB/tok → ~25–45 ms/tok ≈ 22–40 tok/s)* | TBD on-box | the mission number; compressed-attention KV adds little |
| Qwen3-0.6B BF16 | ~1.4 GB | ~6.6 ms/tok | substrate smoke only |

Total-weight residency: ~39–47 GB at 2-bit → GTT must map ≥64 GB (kernel-param
checklist item). Gap-to-floor on the box = engineering overhead, not physics
(measured-floor rule).

## §2 Op→kernel tables

llama.cpp refs at commit `d2462f8f7ac6` — **vendored into
[`vendor/llama.cpp/`](../../vendor/llama.cpp/README.md)** (MIT, pinned, op map in
its README).

### §2.0 Mission op set — DSv4-Flash 2-bit

| Op family | Source | Evidence / note |
| --- | --- | --- |
| 2-bit weight matmul (dense projections + lm_head) | vendored `vecdotq.cuh`+`mmvq/mmq` (HIP) / `mul_mat_vec_iq2_*`+`mul_mmq` (Vulkan) | llama.cpp's IQ2 kernels are the production 2-bit corpus; MMQ MFMA is CDNA-only but the dp4a/WMMA paths cover RDNA3.5 |
| MoE routed experts (per-expert matmul + top-k router) | vendored `mmid.*` + `topk-moe.*` / `mul_mm_id_funcs.glsl` | indirect matmul over expert ids = llama.cpp's MoE shape; our scheduler keeps ARLE routing semantics |
| DSA/CSA compressed attention, indexer, compressor, SW window | **our `crates/cuda-kernels/csrc/` DSv4 kernels, ported via the `vendors/hip.h` shim** (plain CUDA C, no SM90 asm on the fallback path) | the non-FlashMLA path (`dsv4_hybrid_attention`, `dsv4_csa_select`, compressor/indexer) predates official-kernel adoption and is shim-portable; FlashMLA/DeepGEMM/DeepEP = SM90/datacenter, excluded |
| Offline quantizer (FP8 safetensors → IQ2-class) | new tool, reference = vendored `ggml-quants.c` block codecs | per-tensor mix: attn/dense/shared-expert higher-bit, routed FFN 2-bit; needle-gate per recipe |
| Norm / rope / softmax / elementwise / sampling | same as §2 table below | shared with the substrate smoke |

### Substrate smoke op set — Qwen3 dense (proves the lane before the big model)

ARLE op names from `infer-cuda/src/{model,ops,attention}.rs` (the dense Qwen3 CUDA
forward).

| ARLE op | Role | HIP MVP source | llama.cpp reference / evidence |
| --- | --- | --- | --- |
| `gemm_cuda`/`gemv_cuda`/`gemm_batch` BF16 | QKV/O, MLP up·gate/down, lm_head | **hipBLASLt** (primary), rocBLAS fallback | ggml links `roc::hipblas+rocblas`; BF16 explicitly supported on AMD (`ggml-cuda.cu:1691`). rocBLAS has a 2.5–6× gfx1151 regression vs gfx1100 kernels; `ROCBLAS_USE_HIPBLASLT=1` lifts pp ~349→~986 t/s ([llm-tracker](https://llm-tracker.info/_TOORG/Strix-Halo)) |
| `embedding_batched_cuda` | token gather | hand HIP port (trivial) | `getrows.cu` |
| `rms_norm_cuda`/`_batched` | pre-attn/pre-MLP + QK-norm | hand HIP port | `norm.cu` |
| `decode_prep_paged_cuda`, `prefill_attention_paged_prep_cuda` | QK-norm+RoPE+paged-KV append | hand HIP port of our kernels (layout is ours) | `rope.cu` for rope math |
| `tilelang_batch_prefill_paged_hd128_q{16,32,40,64}_kv8` | paged prefill attention | **NEW HIP kernel** on our paged layout; pattern = AMD WMMA fattn (`AMD_WMMA_AVAILABLE` = HIP+RDNA3/4, so gfx1151 yes). Alt to evaluate: TileLang ROCm codegen (*hypothesis — TileLang AMD backend coverage unverified*) | `fattn-wmma-f16.cu`, `fattn-tile.cu`; dispatch logic `fattn.cu:497-525` |
| `tilelang_batch_decode_paged_hd128_q*_kv8` | paged decode attention (B small) | **NEW HIP kernel**, vec pattern | `fattn-vec.cuh` (chosen on AMD when `Q·gqa_ratio ≤ 2`, `fattn.cu:488`) |
| `silu_mul_cuda` | SwiGLU | hand HIP port | `unary.cu` |
| `add_cuda` | residual | hand HIP port | `binbcast.cu` |
| `argmax_cuda` | greedy token | hand HIP port | `argmax.cu` |
| sampling | host | `infer_plan::sample_token` unchanged | — |

Tier-2 (Qwen3.5 hybrid) additions, all with upstream HIP-compilable references:
`GGML_OP_GATED_DELTA_NET` kernel exists in ggml-cuda (`gated_delta_net.cuh`, HIP build
compiles it); `ssm-conv.cu` for the conv ring; hd256 attention is inside AMD's
`DKQ ≤ 256` envelope (`fattn.cu:29` rejects only `DKQ > 256`).

### §2.1 Borrow map is dual-lane (ckl: 很多算子都可以抄 llama.cpp)

Every op above has a liftable llama.cpp implementation in BOTH lanes (MIT, attribute
in-file; re-expressed on our paged-KV layout, not vendored wholesale):

| Op | HIP lane source (`ggml-cuda`, hipcc-compiled) | Vulkan lane source (`ggml-vulkan/vulkan-shaders`, GLSL→SPIR-V) |
| --- | --- | --- |
| decode attention | `fattn-vec.cuh` | `flash_attn.comp` (scalar) / `flash_attn_cm1.comp` (KHR coopmat — works on RDNA3.5 via radv) |
| prefill attention | `fattn-wmma-f16.cu` | `flash_attn_cm1.comp` + `flash_attn_split_k_reduce.comp` |
| GEMM/GEMV | hipBLASLt library | `mul_mm*.comp` (coopmat) / `mul_mat_vec*.comp` — no BLAS dependency at all |
| rms_norm / rope / silu / add / embedding / argmax | `norm.cu` / `rope.cu` / `unary.cu` / `binbcast.cu` / `getrows.cu` / `argmax.cu` | `rms_norm.comp` / `rope_*.comp` / unary comps / `add.comp` / `get_rows.comp` / `argmax.comp` |

The Vulkan column is self-contained shaders (no CUDA→HIP source porting, no ROCm
install, runs on AMD/Intel/NVIDIA + Windows); the HIP column is closer to our
existing CUDA kernel idioms and gets GEMM from a vendor library instead of shaders.

## §3 Kept / killed kernel-source candidates

| Candidate | Verdict | Why |
| --- | --- | --- |
| hipBLASLt | **KEEP (primary GEMM)** | community-proven on gfx1151 (pp ~986 t/s); ships in ROCm 7.x |
| rocBLAS | KEEP (fallback only) | works but 2.5–6× gfx1151 kernel regression |
| llama.cpp kernel *patterns* (fattn-vec/wmma, elementwise) | **KEEP (borrow patterns, MIT)** | the only proven RDNA3.5 attention corpus; we re-express on our paged-KV layout — no ggml runtime/graph adoption |
| rocWMMA | KEEP (inside our attention kernels) | llama.cpp's `GGML_HIP_ROCWMMA_FATTN` is the known-good FA path on gfx1151 |
| AITER | **KILL for MVP** | CDNA/MI-series kernel library; no RDNA3.5 coverage (*hypothesis-grade, one re-check at H3 if AMD ships RDNA support*) |
| composable_kernel direct | KILL for MVP | heavy dependency; hipBLASLt already fronts the CK GEMMs we need |
| Vulkan executor lane | **KEEP (co-candidate — ckl 2026-06-10 "vulkan 也可以,很多算子都可以抄 llama.cpp")** | llama.cpp Vulkan is competitive on this box (pp512 881 vs HIP+hipBLASLt ~986 t/s) with zero ROCm fragility; one Vulkan executor covers AMD/Intel/NVIDIA consumer AIPCs + Windows; the full op corpus is directly liftable GLSL (§2.1) incl. `flash_attn_cm1.comp` on KHR coopmat, and `mul_mm` shaders remove the BLAS dependency entirely. Runtime = `ash` (maintained bindings — no hand FFI needed, unlike HIP). Costs: descriptor/pipeline boilerplate, farther from our CUDA idioms, BF16 storage support device-dependent (fp16 path is the proven one). **Lane decision = day-1 on-box A/B** (llama.cpp ROCm vs Vulkan, same box/model: pp, tg, long-ctx FA, stability); HIP is default-first only because it's the shortest port from `infer-cuda` — a Vulkan win or ROCm fragility flips the lane, sanctioned |
| ggml/llama.cpp as a vendored execution engine | KILL | violates the seam architecture (we need paged-KV + continuous batching under `infer-core`, not ggml's graph) |

## §4 Wiring plan refinement (H2 #76, H3 #77)

**H2 — substrate (start now, typecheckable off-box):**

1. `crates/hip-sys`: thin hand-declared `extern "C"` FFI over the HIP runtime
   (init/device/props/malloc/memcpy/stream/module-launch) + hipBLASLt handle/matmul
   subset. No bindgen, no headers needed to typecheck — link `amdhip64`/`hipblaslt`
   only when the `hip` feature is on (mirror of the Mac `cuda,no-cuda` trick).
   Survey note: no maintained official Rust HIP bindings exist (*checked crates.io
   2026-06: cudarc is CUDA-only; community rocm crates unmaintained*) — thin own FFI
   follows the `deepep-sys`/`kv-native-sys` convention.
2. `hip` cargo feature stacked like `cuda`; cfg-isolation per CLAUDE.md (no leak above
   the seam). `ROCM_PATH`/`HIP_PATH` industry env for detection (build.rs).
3. Device-arch probe (`hipGetDeviceProperties.gcnArchName == "gfx1151"`) exposed for
   doctor + gate scripts.
4. Kernel build path: our HIP kernels live as `.hip.cpp` compiled by `hipcc` at build
   time (offline, `--offload-arch=gfx1151`), packaged like cuda-kernels cubins.
   Single-source CUDA/HIP via a vendors/hip.h-style shim is the llama.cpp-proven
   pattern **for the elementwise ports** — evaluate at first kernel, don't pre-commit.

**H3 — executor:** `infer-hip` implements `BackendExecutor`+`KvPool`; forward mirrors
`infer-cuda/src/model.rs` (dense Qwen3); paged pool layout reuses the seam contract.
Acceptance unchanged: zero `infer-core` edits; #68 needle gate; same-config-twice floor.
The executor design above the kernel layer is **lane-agnostic** — the seam traits and
the forward structure are identical whether kernels launch via HIP or Vulkan; only the
device-runtime substrate (hip-sys vs `ash`) and the kernel artifacts (.hsaco vs SPIR-V)
swap. Don't entangle forward logic with either runtime.

**On-box install checklist (first session on the 395):** ROCm 7.x per known-good
discussion; verify `rocminfo` shows gfx1151; GTT sizing kernel params; build llama.cpp
reference with the §0 flags; capture `ROCBLAS_USE_HIPBLASLT=1` A/B as the first
local evidence entry; **lane license A/B**: llama.cpp ROCm-vs-Vulkan on the same
box/model (pp + tg + stability) — keeps the HIP-vs-Vulkan executor choice
evidence-based instead of web-sourced.

## §5 Go/no-go

**GO.** Every Qwen3-dense op family has either an official library (GEMM) or an
upstream-proven RDNA3.5 kernel pattern (attention, elementwise); llama.cpp serves
real workloads on this exact SKU with the 2026 ROCm stack. The genuine gap — paged
attention on RDNA — is hand-rolled by necessity (AITER/CK don't cover it), which is
the adopt-official-first definition of a licensed gap.

Top risks, in order: hipBLASLt instability on gfx1151 ("no solution found" class
errors — pin versions, keep rocBLAS fallback switchable); ROCm-Preview driver churn
(pin the known-good stack, record exact versions in the first wins entry); WMMA
prefill kernel effort (fattn-wmma is the most complex borrow — schedule it after the
vec decode kernel proves the layout).

## Sources

- llama.cpp `d2462f8f7ac6` sparse checkout: `ggml/src/ggml-cuda/{common.cuh,fattn.cu,vendors/hip.h,ggml-cuda.cu}`, `ggml/src/ggml-hip/CMakeLists.txt`, `docs/build.md`
- [llm-tracker Strix Halo](https://llm-tracker.info/_TOORG/Strix-Halo) — ROCm/Vulkan/hipBLASLt numbers
- [llama.cpp #20856 Known-Good Strix Halo ROCm stack](https://github.com/ggml-org/llama.cpp/discussions/20856) · [#15021 HIP perf](https://github.com/ggml-org/llama.cpp/discussions/15021) · [lemonade-sdk/llamacpp-rocm#7 rocWMMA flag](https://github.com/lemonade-sdk/llamacpp-rocm/issues/7) · [#21284 gfx1151 prefill defaults](https://github.com/ggml-org/llama.cpp/issues/21284)
- [ROCm RDNA3.5 system optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) · [ROCm/ROCm#5339 gfx1151 support confusion](https://github.com/ROCm/ROCm/issues/5339)
