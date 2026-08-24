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
> - **status** uses one of: `production` (default-safe), `opt-in`
> (verified for known use cases, not auto-default), `experimental`
> (works but quality not gated), `known-broken` (reproduces a logged
> failure today), `not-shipped` (planned).

---

## 0. At a glance

| Axis | Format | Status | Enable | Notes |
|---|---|---|---|---|
| **KV cache** | BF16 | production | `--kv-cache-dtype bf16` | Reference fallback. CUDA-paged + Metal. The only value DSv4 accepts. |
| KV cache | INT8 | production (Metal default + CUDA) | `--kv-cache-dtype int8`; Metal `auto` resolves to int8 | Metal stores full-attention K/V as MLX affine 8-bit packed triples (`uint32 data + bf16 scale/bias`, group 128/64/32 by head_dim). CUDA uses per-(token, head) scales for K and V (/127); decode on `paged_attention_quantized_fa3.cu`. **CUDA: Qwen3.5/3.6 family only** — DSv4 rejects any non-BF16 value at engine construction (`infer-api/src/loaded.rs:2054`); its MLA KV is already FP8-packed at 584 B/token regardless of the flag (`infer-cuda/src/dsv4/budget.rs:39-88`). |
| KV cache | FP8 E4M3 | production (CUDA, opt-in) | `--kv-cache-dtype fp8` | Per-(token, head) scales for K and V (/448). Same code shape as INT8 modulo quant range. **CUDA: Qwen3.5/3.6 family only** — DSv4 rejects any non-BF16 value at engine construction (`infer-api/src/loaded.rs:2054`); its MLA KV is already FP8-packed at 584 B/token regardless of the flag (`infer-cuda/src/dsv4/budget.rs:39-88`). |
| KV cache | TurboQuant TQ4 | deferred (CUDA) | `--kv-cache-dtype tq4` (the clap enum accepts `auto\|bf16\|int8\|fp8\|tq4` — there is no `tq2`/`tq3`, `args.rs:927`) | No runtime arm: engine construction bails with an explicit-deferral message (`infer-cuda/src/executor.rs:100`). |
| **Weights** | DenseBF16 | production | default | No quantization. |
| Weights | W4A16 (uniform-group packed INT4) | production (CUDA) | safetensors metadata | Native `w4_gemv` + Marlin W4 prefill. |
| Weights | MarlinW4A8 | production (CUDA), Tier-1 | env `INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1` for the prefill-graph win path (–92.5% TTFT p50). |
| Weights | W8A16 (per-group INT8) | production (CUDA) | safetensors metadata | GEMV + GEMM path. |
| Weights | W2A16 (per-group packed INT2) | experimental (CUDA) | safetensors metadata | Enum variant `WeightFormat::W2A16` exists; no load or kernel scaffolding is wired; not gate-validated. |
| Weights | GGUF Q3_K / Q4_K / Q5_K / Q6_K | production (CUDA & Metal) | `.gguf` extension | Packed superblock kernels in `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`. |
| Weights | DSv4 FP8 E4M3 block-scaled | in progress (CUDA) | DSv4 checkpoints | `Dsv4Fp8BlockScaled` format; CUDA V4 attention/MoE/MTP kernels are the runtime blocker. |
| Weights | DSv4 FP4 E2M1 block-scaled | in progress (CUDA) | DSv4 checkpoints | `Dsv4Fp4BlockScaled`; same DSv4 dependency chain. |

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
- **Status**: production (CUDA); same gate as INT8.

### 1.4 TurboQuant TQ4

Deferred. `--kv-cache-dtype tq4` is accepted by the CLI but fails loud at
engine construction (`infer-cuda/src/executor.rs:100`); the pack/unpack and
decode-attention kernels were removed with the TurboQuant weight format.

---

## 2. Weight quantization (CUDA)

All weight formats live in `crates/cuda-kernels/src/tensor.rs::WeightFormat`;
kernels live in `crates/cuda-kernels/csrc/gemm/`. Format detection at
safetensors load runs in the CUDA weight loader (`crates/infer-cuda/src/loader.rs`).

| Format | Bits | Scale | Kernel | Status |
|---|---|---|---|---|
| `DenseBf16` | 16 | n/a | `cublasLt` / cublasGemmEx | production |
| `W8A16` | 8 | per-group BF16 | `gemv_w8a16` | production |
| `W4A16` | 4 packed | per-group BF16 | `w4_gemv_kernel` + Marlin W4 prefill | production |
| `MarlinW4A8` | 4 packed + dyn INT8 act | per-group BF16 | Marlin W4 + INT8 act prefill | production, **Tier-1 wins via prefill-graph capture** |
| `W2A16` | 2 packed | per-group BF16 | none (enum variant only) | experimental |
| `GgufQ3K` | 3 packed (superblock) | embedded | `gguf_q3k_gemv` | production (CUDA + Metal) |
| `GgufQ4K` | 4 packed (superblock) | embedded | `q4k_gemv_kernel` + packed fast path | production (CUDA + Metal) |
| `GgufQ5K` | 5 packed (superblock) | embedded | `gguf_q5k_gemv` | production (CUDA + Metal) |
| `GgufQ6K` | 6 packed (superblock) | embedded | `gguf_q6k_gemv` | production (CUDA + Metal) |
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

---

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
# GGUF detected from .gguf extension. MarlinW4A8 prefill-graph opt-in:
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1
```

Source: the `--kv-cache-dtype` CLI parser in `crates/cli`, carried through
`infer_api::EngineLoadConfig`. Metal resolves the neutral enum below
`infer-api` and the service/scheduler layers remain backend-neutral.

---

## 4. Test harness — what each one proves

| Test | What it runs | What it proves | What it does NOT prove |
|---|---|---|---|
| `cargo test --test kv_precision_parity` | Boots scheduler per precision, sends string prompts via the IncomingRequest path, greedy decode, compares token trajectories vs the BF16 result. | The audit dispatch path produces the same/different token-IDs across precisions. Includes a **degenerate-baseline guard** (added 2026-05-27) that warns when the BF16 reference is a single-token repetition — that condition makes `mean_match` a noise-fidelity metric, not a quality metric. | Anything about generation *quality*. Greedy + base/chat LM + long prompts collapse to `!`-loops and INT8 reads as "perfect" because it faithfully reproduces the junk. |
| `cargo test --test kv_fp8_prefill_logit_parity` | BF16 vs FP8 raw logit deltas via the scheduler's `forward_raw_logits` (token-by-token decode loop, **not** batched paged prefill). | Single-token decode kernels produce sensible per-vocab logits. Last A100 run: `max_abs=0.000000, argmax_bf16=16, argmax_fp8=16, argmax_match=true, top1_val=17.625`. | Batched paged prefill correctness — the path the production scheduler uses for real prompts is *not* exercised here. |
| `scripts/bench_throughput.py` | OpenAI-compatible streaming requests over a checked JSONL workload. Measures throughput, TTFT, and ITL. | Throughput and latency under load. Kernels run. | Independent output quality; use decoded cases and the model-specific correctness gate. |
| HuggingFace transformers reference | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16) + greedy generate` on the same prompt + chat template. | Independent ground truth for what greedy *should* generate. On Qwen3-4B chat + Eiffel Tower ChatML prompt: first 8 tokens `[151667, 198, 32313, 11, 279, 1196, 3855, 448]` = `"<think>\nOkay, the user started with"`. | Anything about ARLE's runtime kernels — it's a different stack entirely. |

**Reading the matrix**: a precision passing
`kv_precision_parity` means the bytes match BF16. A precision passing
`kv_fp8_prefill_logit_parity` means single-token decode logits are
clean. Neither implies "matches the HF reference on a chat prompt"
— that comparison is what the 2026-05-27 chain exposed as missing.

---

## 5. Retired paths

The dense Qwen3 CUDA path (HD128 TileLang paged prefill, per-channel K
KV quantization, fused-dequant decode) was removed 2026-08-22; the
investigation record lives under `docs/experience/errors/2026-05-26-*`.

---

## 6. Cross-references

**Recent errors (chronological)**:
- `errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md` — the
 retract chain; `mean_match` under a degenerate `!`-loop reference is
 noise-fidelity, not quality.
- `errors/2026-05-26-kivi-per-channel-k-insufficient-for-qwen3-4b-fp8.md`
 — retracted; the metric was invalid.
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
- `e0c283d1` fix(cuda): recursive `rerun-if-changed` for csrc/ subdirs
 (so .cu edits actually trigger rebuilds).
- `228d6eb8` test(kv-tier): don't force `INFER_DETERMINISTIC` + document
 scheduler-path bug.
- `9259fe13` test(kv-tier): natural-continuation prompt 0 + document
 repetition-penalty wiring gap.

---

## 7. Update rule

If the status of any quantization scheme changes (new format, fix
lands, kill decision):
1. Update the row in [§0](#0-at-a-glance).
2. Update the detailed section (§1 for KV, §2 for weights).
3. Add a dated `wins/` or `errors/` entry per `bench-and-trace-spec.md`.
4. Re-link from [`support-matrix.md`](support-matrix.md) §4.
5. Touch `README.md` only if the user-visible support level changes.
