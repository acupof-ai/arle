# ARLE Quantization Reference

Canonical map of every quantization path the runtime ships, the code that
implements it, and what the verification status actually is. Updated on
real findings from the 2026-05-26/27 KV chain. Replaces the per-row
"Beta, benchmarked" claims in [`support-matrix.md`](support-matrix.md) §4
with concrete evidence.

> **Format conventions**
> - **dtype** = how K/V or weight bits are laid out in memory.
> - **scale** = per-tensor / per-channel / per-group / per-(token, head),
>   plus what numeric range it normalizes to (e.g. FP8 E4M3 absmax = 448,
>   INT8 = 127).
> - **status** uses one of: `production` (default-safe), `opt-in`
>   (verified for known use cases, not auto-default), `experimental`
>   (works but quality not gated), `known-broken` (reproduces a logged
>   failure today), `not-shipped` (planned).

---

## 0. At a glance

| Axis | Format | Status | Enable | Notes |
|---|---|---|---|---|
| **KV cache** | BF16 | production | `--kv-cache-dtype bf16` *(default via `auto`)* | Reference. CUDA-paged + Metal. |
| KV cache | INT8 | production (CUDA) | `--kv-cache-dtype int8` | Per-(token, head) scale, /127. +57–113% throughput vs BF16 on A100 4096/256 (`bench-int8-vs-bf16-kv-a100`). Verified token-level identical to BF16 on the audit's paged-prefill regime. |
| KV cache | FP8 E4M3 | opt-in (CUDA) | `--kv-cache-dtype fp8` | Per-(token, head) scale, /448. **+ KIVI per-channel K scaffolding** built but does not change the user-visible metric today — see §1.3. Quality re-audit pending on the paged-prefill investigation (§5). |
| KV cache | TurboQuant TQ2/3/4 | experimental (CUDA) | `--kv-cache-dtype tq{2,3,4}` | FWHT-rotated packed indices + FP16 group norms. Page-size-1 path, bypasses the HD128 batched prefill kernel — the **only** KV format that produced the HF-reference first token in the 2026-05-27 chat-prompt audit. |
| **Weights** | DenseBF16 | production | default | No quantization. |
| Weights | W4A16 (uniform-group packed INT4) | production (CUDA) | safetensors metadata | Native `w4_gemv` + Marlin W4 prefill. |
| Weights | MarlinW4A8 | production (CUDA), Tier-1 | env `INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1` for the prefill-graph win path (–92.5% TTFT p50). |
| Weights | W8A16 (per-group INT8) | production (CUDA) | safetensors metadata | GEMV + GEMM path. |
| Weights | W2A16 (per-group packed INT2) | experimental (CUDA) | safetensors metadata | Scaffolding lives in `tensor.rs::from_quantized_int2`; not gate-validated. |
| Weights | GGUF Q3_K / Q4_K / Q5_K / Q6_K | production (CUDA & Metal) | `.gguf` extension | Packed superblock kernels in `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`. Q4_K_M Metal-Q4-native opt-in via `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all`. |
| Weights | TurboQuant (packed + FP16 norms + Hadamard) | experimental (CUDA) | safetensors with TQ metadata | Tensor-local correctness only — full-model logits parity is **not** gated (`2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill`). |
| Weights | DSv4 FP8 E4M3 block-scaled | in progress (CUDA) | DSv4 checkpoints | `Dsv4Fp8BlockScaled` format; CUDA V4 attention/MoE/MTP kernels are the runtime blocker. |
| Weights | DSv4 FP4 E2M1 block-scaled | in progress (CUDA) | DSv4 checkpoints | `Dsv4Fp4BlockScaled`; same DSv4 dependency chain. |

> **Default policy** (`--kv-cache-dtype auto`): BF16 paged pool. FP8 was
> historically the auto default but the 2026-05-25 audit reproduced
> first-token divergence — auto is now correctness-safe BF16.

---

## 1. KV-cache quantization (CUDA-paged)

All four KV formats live in the same Rust enum: see
`crates/cuda-kernels/src/kv_types.rs::KVFormat`. Dispatch fans out at
`infer/src/model/qwen3/{prefill,batch_decode}.rs` and the underlying
CUDA kernels are in `crates/cuda-kernels/csrc/{kv,attention}/`.

### 1.1 BF16 (reference)

- **Storage**: `__nv_bfloat16` rows in the paged pool, no scale.
- **Quantize kernels**: none (direct write from the BF16 work buffer).
- **Decode-attn kernel**: TileLang HD128 BF16 attention (paged) — same
  kernel family as INT8/FP8, just without dequant.
- **Status**: production. Reference for all audits.
- **Memory cost**: 2 bytes / element (baseline).
- **Limitation**: No KV compression; cache size scales as
  `num_layers · 2 · num_kv_heads · head_dim · max_total_tokens · 2 B`.

### 1.2 INT8

- **Storage**: `i8` rows + `f32` scale per `(token, kv_head)`.
- **Scale**: `absmax / 127`, no numerical floor (only `(absmax > 0) ?
  absmax/127 : 1.0` guard against divide-by-zero).
- **Quantize kernels**: `quantize_paged_kv_int8_*` family in
  `crates/cuda-kernels/csrc/kv/kv_quant.cu`. Decode single-token quant
  via `quantize_paged_kv_single_kernel` (line 553).
- **Decode-attn kernel**:
  `decode_attention_int8_partial_kernel` (`csrc/attention/decode_attention_quantized.cu`
  line 46) — cp.async smem tiling with per-(token, kv_head) dequant.
- **Status**: production (CUDA). +57–113% throughput / –39–55% ITL p50
  vs BF16 on A100 4096/256 (`docs/experience/wins/2026-05-26-bench-int8-vs-bf16-kv-a100.md`).
  Verified token-level identical to BF16 on the paged-prefill regime
  used by the audit harness.
- **Memory cost**: ~1.05 byte / element (i8 + per-(token, head) f32
  scale, amortized).

### 1.3 FP8 E4M3 (+ KIVI per-channel K scaffolding)

- **Storage**: `__nv_fp8_e4m3` rows + `f32` scale.
- **Scale**: two paths, asymmetric per KIVI:
  - **V (and legacy K)**: per-(token, kv_head), `absmax / 448`. The
    1e-6 floor that masked deep-layer K activations was removed
    2026-05-26 (`25c7d409`); the guard now matches INT8 —
    `(absmax > 0) ? absmax/448 : 1.0`.
  - **K (KIVI mode, enabled whenever the FP8 pool is allocated)**:
    per-(kv_head, head_dim) static table calibrated from the first
    prefill batch via `compute_k_per_channel_absmax_cuda` +
    `finalize_k_per_channel_scales_cuda`. Floor at 1e-30 (essentially
    only catches truly-zero channels).
- **Quantize kernels** (`csrc/kv/kv_quant.cu`):
  - `quantize_paged_kv_fp8_kernel` (line 171) — legacy per-token K,
    used by V on every step.
  - `quantize_paged_kv_fp8_per_channel_kernel` (line 670) — KIVI
    per-channel K, used when `k_static_scales` is allocated.
  - `compute_k_per_channel_absmax_kernel` (line 722) +
    `finalize_k_per_channel_scales_kernel` — KIVI calibration pair.
- **Decode-attn kernels**:
  - `decode_attention_fp8_partial_kernel` (line 307) — legacy
    per-(token, head) K scale.
  - `decode_attention_fp8_per_channel_k_partial_kernel` (line 606,
    HEAD_DIM templated) — pre-loads `k_scale_reg[EPT]` once per warp
    from the per-channel table and multiplies per-dim during QK dot.
    Fix `73a72615` ensures it writes the *normalized* per-split
    average (`final_o * inv_final_l`) like the legacy kernel — earlier
    drafts dropped the divide and produced O(l_s)-scale-off
    attention.
- **Status**: opt-in (CUDA). KIVI implementation is unit-test clean
  and dispatches as expected; the bench shows real throughput. The
  end-to-end audit (`kv_precision_parity`) currently reports
  `mean_match=0.0156` vs BF16, but the 2026-05-27 chain showed that
  metric is **not a quality signal** under the audit's regime — see
  §4.
- **Open question**: shared paged-prefill regime affects BF16/INT8/FP8
  identically (§5). Quality verdict on FP8 specifically is deferred
  until the paged-prefill investigation resolves.

### 1.4 TurboQuant TQ2 / TQ3 / TQ4

- **Storage**: packed indices (2/3/4 bits per element) + FP16 per-group
  norms + Hadamard sign bits.
- **Scale**: FWHT-rotated absmax norm per group; bit-pair-combined
  during dequant. Pool allocates `page_size = 1` (one token per page).
- **Quantize kernels**:
  `crates/cuda-kernels/csrc/quant/turboquant_*` — pack/unpack pair.
- **Decode-attn kernel**:
  `decode_attention_turboquant_*` — fused dequant inline.
- **Prefill path**: `page_size = 1` triggers the contiguous BF16
  prefill path (`forward.rs` §1 dispatch at line 446-470, see also
  `infer/src/scheduler/cuda/prefill.rs::prepare` §537) rather than the
  HD128 batched paged kernel that BF16/INT8/FP8 use.
- **Status**: experimental. **As of 2026-05-27, TQ4 is the only KV
  format that produced the HF-reference first token (`151667 = <think>`)
  in the audit on Qwen3-4B chat**; BF16/INT8/FP8 produced token 0
  (`!`). This is the smoking gun that pointed the investigation at the
  shared HD128 batched paged prefill kernel — see §5.
- **Quality gate**: greedy token-trajectory match against BF16 is *not*
  a meaningful gate for TQ (tensor-local fixes license only their own
  gate, not full-model logits parity —
  `2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill`).

---

## 2. Weight quantization (CUDA)

All weight formats live in `crates/cuda-kernels/src/tensor.rs::WeightFormat`
(line 818). Format detection runs at safetensors load
(`infer/src/model/qwen3/weights.rs`); kernels live in
`crates/cuda-kernels/csrc/gemm/`.

| Format | Bits | Scale | Kernel | Status |
|---|---|---|---|---|
| `DenseBf16` | 16 | n/a | `cublasLt` / cublasGemmEx | production |
| `W8A16` | 8 | per-group BF16 | `gemv_w8a16` | production |
| `W4A16` | 4 packed | per-group BF16 | `w4_gemv_kernel` + Marlin W4 prefill | production |
| `MarlinW4A8` | 4 packed + dyn INT8 act | per-group BF16 | Marlin W4 + INT8 act prefill | production, **Tier-1 wins via prefill-graph capture** |
| `W2A16` | 2 packed | per-group BF16 | `gemv_w2a16` | experimental |
| `GgufQ3K` | 3 packed (superblock) | embedded | `gguf_q3k_gemv` | production (CUDA + Metal) |
| `GgufQ4K` | 4 packed (superblock) | embedded | `q4k_gemv_kernel` + packed fast path | production (CUDA + Metal) |
| `GgufQ5K` | 5 packed (superblock) | embedded | `gguf_q5k_gemv` | production (CUDA + Metal) |
| `GgufQ6K` | 6 packed (superblock) | embedded | `gguf_q6k_gemv` | production (CUDA + Metal) |
| `TurboQuant` | 2/3/4 packed + Hadamard | per-group FP16 | `turboquant_gemv` | experimental (tensor-local gate only) |
| `Dsv4Fp8BlockScaled` | 8 (E4M3) | per-block FP8 E8M0 | DSv4-specific | in progress (DSv4 dependency) |
| `Dsv4Fp4BlockScaled` | 4 packed (E2M1) | per-block FP8 E8M0 | DSv4-specific | in progress (DSv4 dependency) |

### 2.1 W4-hybrid prefill CUDA Graph capture — Tier 1 wins detail

Opt-in via:
```bash
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1
```

Path B.2 bucketing fix (`a56b7a9` / `c44788f`) delivers on matched
4k/c=4 60s on Qwen3.5 paged prefill:
- engine TTFT p50: 2000 ms → 150 ms (**–92.5%**)
- 7 unique capture keys, 98.5% LRU reuse
- +632% throughput, closes the +76.6% SGLang gap

Default behavior unchanged when env unset.

### 2.2 GGUF + Metal opt-in native-Q4

Default GGUF Q4_K_M on Metal goes through the exact GGUF affine/packed
kernel. Opt-in lossy conversion to MLX-native q4 group64 via
`AGENT_INFER_METAL_GGUF_NATIVE_Q4=all`. Faster, lossy, off by default.

---

## 3. CLI quick reference

```bash
# KV cache (CUDA only — Metal does not ship quantized KV today)
--kv-cache-dtype <auto|bf16|fp8|int8|tq2|tq3|tq4>
  # auto → bf16 paged pool (correctness-safe default since 2026-05-25)
  # fp8  → KVCacheDtype::BF16 + KVFormat::FP8E4M3 + KIVI per-channel K
  # int8 → KVCacheDtype::INT8 + KVFormat::INT8
  # tq{2,3,4} → KVCacheDtype::BF16 + KVFormat::TurboQuant { key_bits, val_bits }

# Weight quantization
# Format is autodetected from safetensors metadata. No CLI flag needed.
# GGUF detected from .gguf extension. MarlinW4A8 prefill-graph opt-in:
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1
```

Source: `infer/src/main.rs::parse_kv_cache_mode` (line 2000),
`kv_mode_candidates` (line 2043), `Args.kv_cache_dtype` (line 240).

---

## 4. Test harness — what each one actually proves

| Test | What it runs | What it proves | What it does NOT prove |
|---|---|---|---|
| `cargo test --test kv_precision_parity` | Boots scheduler per precision, sends string prompts via the IncomingRequest path, greedy decode, compares token trajectories vs the BF16 result. | The audit dispatch path produces the same/different token-IDs across precisions. Includes a **degenerate-baseline guard** (added 2026-05-27) that warns when the BF16 reference is a single-token repetition — that condition makes `mean_match` a noise-fidelity metric, not a quality metric. | Anything about generation *quality*. Greedy + base/chat LM + long prompts collapse to `!`-loops and INT8 reads as "perfect" because it faithfully reproduces the junk. |
| `cargo test --test kv_fp8_prefill_logit_parity` | BF16 vs FP8 raw logit deltas via the scheduler's `forward_raw_logits` (token-by-token decode loop, **not** batched paged prefill). | Single-token decode kernels produce sensible per-vocab logits. Last A100 run: `max_abs=0.000000, argmax_bf16=16, argmax_fp8=16, argmax_match=true, top1_val=17.625`. | Batched paged prefill correctness — the path the production scheduler uses for real prompts is *not* exercised here. |
| `scripts/bench_guidellm.sh <label>` | guidellm 0.6.0 + synthetic random-token prompts + sampled decode (temperature 0.6, top_p 0.95). Measures throughput / TTFT / ITL. | Throughput and latency under load. Kernels run. | Output quality — guidellm doesn't inspect generated text; synthetic random-token prompts produce semi-random outputs by design. |
| HuggingFace transformers reference | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16) + greedy generate` on the same prompt + chat template. | Independent ground truth for what greedy *should* generate. On Qwen3-4B chat + Eiffel Tower ChatML prompt: first 8 tokens `[151667, 198, 32313, 11, 279, 1196, 3855, 448]` = `"<think>\nOkay, the user started with"`. | Anything about ARLE's runtime kernels — it's a different stack entirely. |

**Reading the matrix**: a precision passing
`kv_precision_parity` means the bytes match BF16. A precision passing
`kv_fp8_prefill_logit_parity` means single-token decode logits are
clean. Neither implies "matches the HF reference on a chat prompt"
— that comparison is what the 2026-05-27 chain exposed as missing.

---

## 5. Active investigation: TileLang HD128 batched paged prefill

**Localized 2026-05-27. Not fixed.**

| Path | Prefill regime | Audit first-token on Qwen3-4B chat (ChatML wrap, greedy) |
|---|---|---|
| HF transformers (reference) | HF's own | `151667 (<think>)` ✓ |
| `kv_fp8_prefill_logit_parity` | token-by-token decode loop via `forward_with_logits(&[t], state)` | argmax=16, top_val=17.625 ✓ |
| `kv_precision_parity` BF16/INT8/FP8 | TileLang HD128 batched paged prefill (page_size=16) | `0 (!)` ✗ |
| `kv_precision_parity` TQ4 | Contiguous BF16 prefill (page_size=1 bypass) | `151667 (<think>)` ✓ |

The shared variable across the failing cases is the **TileLang HD128
batched paged prefill kernel**, dispatched at
`infer/src/model/qwen3/forward.rs:454` when
`pool.page_size == 16 && self.prefill_uses_paged_pool() && pool.is_active()`.
TQ4 bypasses by allocating `page_size = 1` and falling through to
`forward_prefill_batch` (contiguous BF16 prefill).

Ruled out across the chain (`2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`
+ `2026-05-26-kivi-per-channel-k-insufficient-for-qwen3-4b-fp8.md`
RETRACTED):
- KV-side numerics (KIVI scale dump shows non-degenerate, decode
  kernel math correct after `73a72615`).
- FP8 quant scale floor (fixed `25c7d409`).
- Model load / tokenizer / RMSNorm / RoPE / LM head (HF reference uses
  same weights and works; `forward_raw_logits` produces sensible
  argmax).
- `INFER_DETERMINISTIC` autotune-bypass / cublasGemmEx fallback
  (`228d6eb8` verified: turning it off doesn't change the
  result).
- ChatML special-token tokenization (HF tokenizer is shared via
  `tokenizers` crate; the same ChatML wrap that works in HF reaches the
  scheduler).

Bench numbers from `2026-05-26-bench-int8-vs-bf16-kv-a100.md`
(throughput +57–113% INT8 vs BF16 on synthetic random-token prompts)
are independent of this — guidellm doesn't inspect output text and
sampled decode with random prompts produces semi-random output by
construction.

**Next step (not yet executed — needs go-ahead before touching shared
kernel code)**: read `launch_prefill_paged_batch` →
`process_all_layers_batch_paged` → the TileLang HD128 paged prefill
attention kernel, then either dump intermediate tensors per-layer to
isolate where logits drift from token-by-token, or compare the K/V
write layout against the decode-time read layout for an off-by-one.

---

## 6. Cross-references

**Recent errors (chronological)**:
- `errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md` — the
  retract chain; `mean_match` under a degenerate `!`-loop reference is
  noise-fidelity, not quality.
- `errors/2026-05-26-kivi-per-channel-k-insufficient-for-qwen3-4b-fp8.md`
  — **RETRACTED**. Conclusion invalid because metric was bad.
- `errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md` — the
  "precision-floor compounding" hypothesis from f50dd674. Needs
  re-evaluation under a non-degenerate reference once the paged-prefill
  investigation closes.
- `errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md` —
  TurboQuant tensor-local fixes don't gate full-model logits parity.
- `errors/2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md`,
  `2026-05-05-fp8-kv-tier1-still-fail.md` — earlier FP8 KV failure
  characterizations also relied on `mean_match`; the retract chain
  applies.

**Recent wins**:
- `wins/2026-05-26-bench-int8-vs-bf16-kv-a100.md` — throughput numbers,
  independent of the quality investigation.

**Commits central to this matrix**:
- `25c7d409` fix(cuda): remove FP8 quant `s_scale` 1e-6 floor.
- `73a72615` fix(cuda): KIVI partial kernel must write normalized `o_s`.
- `e0c283d1` fix(cuda): recursive `rerun-if-changed` for csrc/ subdirs
  (so .cu edits actually trigger rebuilds).
- `0ef57994` debug(kv-tier): KIVI `k_static_scales` post-finalize dump.
- `8c6d92db` feat(kv-tier): KIVI per-channel K + per-token V FP8 KV
  implementation (V1, gated).
- `228d6eb8` test(kv-tier): don't force `INFER_DETERMINISTIC` + document
  scheduler-path bug.
- `9259fe13` test(kv-tier): natural-continuation prompt 0 + document
  repetition-penalty wiring gap.

---

**Background reading (industry landscape, not project status)**:
- [`resources/kv-cache-quantization.md`](resources/kv-cache-quantization.md)
  — methods survey (uniform, heterogeneous K/V, KIVI, KVQuant,
  TurboQuant), frameworks (LMDeploy, TensorRT-LLM, llm-compressor),
  evaluation methodology.

---

## 7. Update rule

If the status of any quantization scheme changes (new format, fix
lands, kill decision):
1. Update the row in [§0](#0-at-a-glance).
2. Update the detailed section (§1 for KV, §2 for weights).
3. Add a dated `wins/` or `errors/` entry per `bench-and-trace-spec.md`.
4. Re-link from [`support-matrix.md`](support-matrix.md) §4.
5. Touch `README.md` only if the user-visible support level changes.
