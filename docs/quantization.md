# ARLE Quantization Reference

Canonical map of every quantization path the runtime ships, the code that
implements it, and what the verification status is. Updated on
real findings from the 2026-05-26/27 KV chain. Replaces the per-row
"Beta, benchmarked" claims in [`support-matrix.md`](support-matrix.md) §4
with concrete evidence.

> **Format conventions**
> - **dtype** = how K/V or weight bits are laid out in memory.
> - **scale** = per-tensor / per-channel / per-group / per-(token, head),
> plus what numeric range it normalizes to (e.g. FP8 E4M3 absmax = 448,
> INT8 = 127).
> - **status** uses the same vocabulary as [`support-matrix.md`](support-matrix.md):
> `production` (default-safe, optionally with a parenthetical scope), `opt-in`
> (verified but enabled by an explicit flag, not the default), `experimental`
> (works but quality not gated, with a scope), `deferred` (accepted by config
> but no runtime arm), or `not implemented` (no code path).

---

## 0. At a glance

> Where a support level here disagrees with [`support-matrix.md`](support-matrix.md),
> the support matrix is canonical; this page gives the mechanism, the matrix
> gives the shipped readiness.

| Axis | Format | Status | Enable | Notes |
|---|---|---|---|---|
| **KV cache** | BF16 | production | `--kv-cache-dtype bf16` | Reference fallback. CUDA-paged + Metal. The only value DSv4 accepts. |
| KV cache | INT8 | production (Metal default + CUDA) | `--kv-cache-dtype int8`; Metal `auto` resolves to int8 | Metal stores full-attention K/V as MLX affine 8-bit packed triples (`uint32 data + bf16 scale/bias`, group 128/64/32 by head_dim). CUDA uses per-(token, head) scales for K and V (/127); decode on `paged_attention_quantized_fa3.cu`. **CUDA: Qwen3.5/3.6 family only** — a non-BF16 value bails at engine construction for any non-Qwen35 kind (`infer-api/src/loaded.rs:1020`); DSv4 MLA KV is already FP8-packed at 584 B/token regardless of the flag (`infer-cuda/src/dsv4/budget.rs:39-88`). |
| KV cache | FP8 E4M3 | opt-in (CUDA) | `--kv-cache-dtype fp8` | Per-(token, head) scales for K and V (/448). Same code shape as INT8 modulo quant range. **CUDA: Qwen3.5/3.6 family only** — a non-BF16 value bails at engine construction for any non-Qwen35 kind (`infer-api/src/loaded.rs:1020`); DSv4 MLA KV is already FP8-packed at 584 B/token regardless of the flag (`infer-cuda/src/dsv4/budget.rs:39-88`). |
| KV cache | TurboQuant TQ4 | deferred (CUDA) | `--kv-cache-dtype tq4` (the clap enum accepts `auto\|bf16\|int8\|fp8\|tq4` — there is no `tq2`/`tq3`, `args.rs:941`) | No runtime arm: engine construction bails with an explicit-deferral message (`infer-cuda/src/executor.rs:108`). |
| **Weights** | DenseBF16 | production | default | No quantization. |
| Weights | W4A16 (uniform-group packed INT4) | production (CUDA) | safetensors metadata | Native `w4a16_gemv` + Marlin W4 prefill. |
| Weights | W8A16 (per-group INT8) | production (CUDA) | safetensors metadata | GEMV + GEMM path. |
| Weights | W2A16 (per-group packed INT2) | not implemented | safetensors metadata | Enum variant `WeightFormat::W2A16` exists; no load or kernel scaffolding is wired; not gate-validated. |
| Weights | GGUF Q4_K / Q5_K / Q6_K | experimental (Vulkan & HIP; not served on CUDA or Metal) | `.gguf` extension | Packed superblock kernels live in `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`, but they are consumed only by the HIP path (`infer-hip/src/model.rs` calls `q4k/q5k/q6k_gemv_cuda`); Vulkan CPU-dequants in `infer-vulkan/src/loader.rs`. `infer-cuda` and `infer-metal` have no `infer-gguf` dependency and no GGUF loader branch, so there is no CUDA/Metal edge. |
| Weights | GGUF Q3_K | not implemented (no host launcher on any backend) | `.gguf` extension | Parsed by `infer-gguf/src/gguf.rs`, but there is no host launcher or gemv kernel on Vulkan, HIP, CUDA, or Metal. |
| Weights | DSv4 FP8 E4M3 block-scaled | production (CUDA, TP=8/EP=8) | DSv4 checkpoints | `Dsv4Fp8BlockScaled` dispatched on the serving path — attention/route GEMV `crates/infer-cuda/src/attention.rs:341,4943`, grouped MoE in `moe/dsv4.rs`, loader `dsv4/load.rs:928`. |
| Weights | DSv4 FP4 E2M1 block-scaled | production (CUDA, TP=8/EP=8) | DSv4 checkpoints | `Dsv4Fp4BlockScaled` dispatched one sibling line down — `attention.rs:353,4951`, `dsv4/load.rs:929`, FP4 grouped MoE + `marlin_fp4`. |

> **Default policy** (`--kv-cache-dtype auto`): Metal resolves `auto` to INT8
> full-attention KV after the 2026-06-11 long-context gate. CUDA keeps its
> backend-specific default policy; BF16 remains the explicit correctness
> fallback via `--kv-cache-dtype bf16`.

---

## 1. KV-cache quantization (CUDA-paged)

The BF16/INT8/FP8 KV formats live in the same Rust enum: see
`crates/cuda-kernels/src/kv_types.rs::KVFormat` (its fourth variant is the
DSv4 MLA opaque record; TQ4 has no runtime arm — §1.4). The underlying CUDA kernels
are in `crates/cuda-kernels/csrc/{kv,attention}/`; the runtime dispatch is the
Qwen3.5/3.6 full-attention path in `crates/infer-cuda/src/qwen35_attention.rs`.

### 1.1 BF16 (reference)

- **Storage**: `__nv_bfloat16` rows in the paged pool, no scale.
- **Quantize kernels**: none (direct write from the BF16 work buffer).
- **Decode-attn kernel**: FA3 (`arle_fa3_shim.cu`) or TileLang HD256 BF16
 paged attention.
- **Status**: production. Reference for all audits.
- **Memory cost**: 2 bytes / element (baseline).
- **Limitation**: No KV compression; cache size scales as
 `num_layers · 2 · num_kv_heads · head_dim · max_total_tokens · 2 B`.

### 1.2 INT8 (per-(token, head) scales)

- **Storage**: `i8` rows (NHD `[max_tokens, kv_dim]`) + `f32` scales,
 one scale per (token, kv_head) for K and for V (`absmax / 127`).
- **Quantize kernel** (`csrc/kv/kv_quant.cu`):
 `quantize_paged_kv_single_kernel` — symmetric per-(row, head) quantize
 of the new token rows, driven by `quantize_paged_kv_per_token`
 (`cuda-kernels/src/kv_quant.rs`).
- **Decode-attn kernel**: `paged_attention_quantized_fa3.cu` — split-KV
 over the 1-byte pool with inline dequant, no bf16 temp; graph-capturable.
 Prefill rows over a quantized pool run the FA3 quant shim
 (`arle_fa3_shim.cu` + `dequant_paged_kv.cu` page compaction).
- **Status**: production (CUDA). Correctness gate: `scripts/needle_gate.py`
 ×3 same-config vs the BF16 envelope.
- **Memory cost**: ~1.03 byte / element (i8 + two f32 scales per
 (token, head)).

### 1.3 FP8 E4M3 (per-(token, head) scales)

Identical code shape to INT8 modulo the quant range (`absmax / 448`) and the
hardware FP8 conversion.

- **Storage**: `__nv_fp8_e4m3` rows + `f32` scales, per (token, kv_head)
 for both K and V.
- **Quantize kernel** (`csrc/kv/kv_quant.cu`): `quantize_paged_kv_fp8_kernel`
 via the same `quantize_paged_kv_per_token` wrapper with
 `KVFormat::FP8E4M3`.
- **Decode-attn kernel**: `paged_attention_quantized_fa3.cu` (shared with
 INT8; format selects the dequant idiom).
- **Status**: opt-in (CUDA); same gate as INT8.

### 1.4 TurboQuant TQ4

Deferred. `--kv-cache-dtype tq4` is accepted by the CLI but fails loud at
engine construction (`infer-cuda/src/executor.rs:108`); the pack/unpack and
decode-attention kernels were removed with the TurboQuant weight format.

---

## 2. Weight quantization (CUDA)

All weight formats live in `crates/cuda-kernels/src/tensor/weight_format.rs::WeightFormat`
(re-exported from `tensor.rs`); kernels live in `crates/cuda-kernels/csrc/gemm/`.
Format detection at safetensors load runs in the CUDA weight loader
(`crates/infer-cuda/src/loader.rs`).

| Format | Bits | Scale | Kernel | Status |
|---|---|---|---|---|
| `DenseBf16` | 16 | n/a | `cublasLt` / cublasGemmEx | production |
| `W8A16` | 8 | per-group BF16 | `w8a16_gemv_cuda` | production |
| `W4A16` | 4 packed | per-group BF16 | `w4a16_gemv_cuda` + Marlin W4 prefill | production |
| `W2A16` | 2 packed | per-group BF16 | none (enum variant only) | not implemented |
| `GgufQ3K` | 3 packed (superblock) | embedded | none (no host launcher) | not implemented (no backend) |
| `GgufQ4K` | 4 packed (superblock) | embedded | `q4k_gemv_cuda` (HIP call only) | experimental (HIP; Vulkan CPU-dequant); not CUDA/Metal |
| `GgufQ5K` | 5 packed (superblock) | embedded | `q5k_gemv_cuda` (HIP call only) | experimental (HIP; Vulkan CPU-dequant); not CUDA/Metal |
| `GgufQ6K` | 6 packed (superblock) | embedded | `q6k_gemv_cuda` (HIP call only) | experimental (HIP; Vulkan CPU-dequant); not CUDA/Metal |
| `Dsv4Fp8BlockScaled` | 8 (E4M3) | per-block FP8 E8M0 | DSv4-specific GEMV + grouped MoE | production (CUDA, TP=8/EP=8) |
| `Dsv4Fp4BlockScaled` | 4 packed (E2M1) | per-block FP8 E8M0 | DSv4-specific GEMV + FP4 grouped MoE / Marlin FP4 | production (CUDA, TP=8/EP=8) |

### 1.5 Metal INT8 (MLX affine groups)

- **Storage**: one packed affine triple per full-attention K or V cache:
 `uint32` packed 8-bit data with last dim `head_dim / 4`, plus BF16
 `scale` and `bias` arrays with last dim `head_dim / group_size`.
- **Group size**: largest supported divisor among 128, 64, 32. Qwen3.6
 (`head_dim=256`) uses group 128.
- **Write path**: C++ session quantizes only the newly written K/V chunk, then
 `slice_update`s the packed data/scale/bias cache at `cache_pos`. It does not
 re-quantize the whole cache every token.
- **Read path**: the active prefix is dequantized to BF16 before MLX SDPA.
 This keeps correctness close to the existing BF16 attention path while making
 the persistent session KV about half-size.
- **Scope**: full-attention KV only. Qwen3.5/3.6 linear-attention recurrent and
 convolution state keep their existing FP32/BF16 dtypes.
- **Evidence**: Qwen3.6 16K serial probe on local Apple Silicon:
 BF16 after-clear active 24.203 GB vs INT8 23.691 GB, a 512 MB reduction;
 8K probe reduced 244 MB. See
 [`experience/wins/2026-06-11-metal-int8-kv-default.md`](experience/wins/2026-06-11-metal-int8-kv-default.md).

## 3. CLI quick reference

```bash
# KV cache
--kv-cache-dtype <auto|bf16|int8|fp8|tq4>
 # Metal: auto → int8, bf16 → reference fallback, int8 → explicit default path.
 # CUDA: int8/fp8 on the Qwen3.5-family paged path; tq4 accepted but fails loud
 # at engine construction (deferred); DSv4 accepts bf16 only (MLA KV is FP8-packed).

# Weight quantization
# Format is autodetected from safetensors metadata. No CLI flag needed.
# GGUF detected from .gguf extension.
```

Source: the `--kv-cache-dtype` CLI parser in `crates/cli`, carried through
`infer_api::EngineLoadConfig`. Metal resolves the neutral enum below
`infer-api` and the service/scheduler layers remain backend-neutral.

---

## 4. Test harness — what each one proves

| Test | What it runs | What it proves | What it does NOT prove |
|---|---|---|---|
| `cargo run -p infer-cuda --features cuda --example paged_quant_attn_parity` (`crates/infer-cuda/examples/paged_quant_attn_parity.rs`) | Boots the engine per KV precision, sends the needle ladder through the serving path, and compares the quantized paged-attention result against the BF16 envelope; carries a `--negative-control` arm. | Quantized INT8/FP8 paged decode answers the same needles as BF16 at the served geometry. | Anything about generation *quality* on open-ended prompts; a needle ladder is retrieval, not a distributional comparison. |
| `scripts/bench_throughput.py` | OpenAI-compatible streaming requests over a checked JSONL workload. Measures throughput, TTFT, and ITL. | Throughput and latency under load. Kernels run. | Independent output quality; use decoded cases and the model-specific correctness gate. |
| HuggingFace transformers reference | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16) + greedy generate` on the same prompt + chat template. | Independent ground truth for what greedy *should* generate. On Qwen3-4B chat + Eiffel Tower ChatML prompt: first 8 tokens `[151667, 198, 32313, 11, 279, 1196, 3855, 448]` = `"<think>\nOkay, the user started with"`. | Anything about ARLE's runtime kernels — it's a different stack entirely. |

**Reading the matrix**: passing the paged-quant attention parity example
means the quantized pool answers BF16's needles at the served shape; it does
not imply "matches the HF reference on an open chat prompt", which is a
separate quality comparison.

---

## 5. Retired paths

The dense Qwen3 CUDA path (HD128 TileLang paged prefill, per-channel K
KV quantization, fused-dequant decode) was removed 2026-08-22; the record is
[`wins/2026-08-22-delete-qwen3-dense-cuda-and-kivi.md`](experience/wins/2026-08-22-delete-qwen3-dense-cuda-and-kivi.md).

---

## 6. Update rule

If the status of any quantization scheme changes (new format, fix
lands, kill decision):
1. Update the row in [§0](#0-at-a-glance).
2. Update the detailed section (§1 for KV, §2 for weights).
3. Add a dated `wins/` or `errors/` entry per `bench-and-trace-spec.md`.
4. Re-link from [`support-matrix.md`](support-matrix.md) §4.
5. Touch `README.md` only if the user-visible support level changes.
