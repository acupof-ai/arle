# M_b — TileLang HD128 paged decode (currently a kernel-coverage gap)

> Sister track to [`M3.6-F4-eliminate-per-step-sync.md`](M3.6-F4-eliminate-per-step-sync.md).
> F4 closes the **scheduling axis** (CPU/GPU overlap, 65 ms/tick
> sync wait). M_b closes the **kernel axis** (per-row work):
> ARLE 2.16 ms/row vs vLLM 1.43 ms/row = vLLM 51% faster per row.
> The two are independent and should land in parallel — combining
> them gets ARLE past vLLM, not just to parity.

## P0 finding (kernel-coverage gap)

`ls crates/cuda-kernels/tools/tilelang/` shows TileLang scripts for
**HD128 prefill, HD256 prefill, HD256 decode** but NOT
`batch_decode_paged_hd128.py`. Qwen3-4B (HD128, the production model
in every bench so far) decodes via **hand-written CUDA** in
`crates/cuda-kernels/csrc/attention/`:

| Variant | Implementation | Where |
|---|---|---|
| HD128 BF16 decode | hand-CUDA | `decode_attention_varlen_quantized.cu` |
| HD128 FP8 decode (paged) | hand-CUDA, FlashDecoding split-KV | `decode_attention_varlen_fp8.cu` |
| HD128 INT8 decode (paged) | hand-CUDA | `decode_attention_quantized.cu` |
| HD128 prefill | **TileLang** | `tools/tilelang/batch_prefill_paged_hd128.py` |
| HD256 decode (Qwen3.5) | **TileLang** | `tools/tilelang/batch_decode_paged_hd256.py` |
| HD256 prefill (Qwen3.5) | **TileLang** | `tools/tilelang/batch_prefill_paged_hd256.py` |

Phase 1 nsys trace showed
`decode_attention_varlen_quantized_partial_kernel<128, 1, 0>` =
**41.6% of all GPU time, 4.55 ms / call avg, 4,104 calls in 60 s**.
This is the hand-CUDA HD128 FP8 decode kernel. **It is the largest
single optimization target on the kernel axis.**

vLLM at the same workload runs decode through Triton/CUTLASS attention
that benefits from upstream tuning. ARLE's hand-CUDA is from 2025-Q4
and has not been auto-tuned for SM 8.9 (RTX 4080S) since.

## Two sub-tracks

The right move is to start with the **simpler shape that validates
the TileLang adaptation** (HD128 BF16 decode), then once that is
proven, attack the **production hot path** (HD128 FP8 paged
decode). FP8 is harder because TileLang HD128 decode templates do
not yet handle on-the-fly FP8 dequantization.

### M_b.1 — TileLang HD128 BF16 decode (~1 day, validation)

Adapt `batch_decode_paged_hd256.py` into
`batch_decode_paged_hd128.py`:

- `HEAD_DIM = 128` (vs 256)
- `SM_SCALE = 1.0 / sqrt(128)`
- Q layout: `[batch_size, num_q_heads * 128]`
- Supported (q_heads, kv_heads) for Qwen3-4B: **(32, 8)** (4× GQA ratio)
- BLOCK_M = 64 unchanged, BLOCK_N = PAGE_SIZE = 16 unchanged
- Tile/pipeline tunables: NUM_STAGES=2, NUM_THREADS=128 (mirror HD256)
- Add to `gen_tilelang_aot.py` discovery (the existing
  `--kernel-key` mechanism handles this automatically once the .py
  exists)

**Replaces**: hand-CUDA BF16 decode kernel (the BF16 variant of
`decode_attention_varlen_quantized.cu`).

**Expected gain**: ~10-15% on per-row decode time when KV is BF16
(based on TileLang HD128 prefill out-performing hand-CUDA HD128
prefill by similar margin per project history). BF16 is NOT the
production hot path (Phase 1 used FP8 KV), so this is primarily a
**validation step** for the TileLang HD128 decode IR.

### M_b.2 — TileLang HD128 FP8 paged decode (~3-4 days, the real win)

Extend M_b.1 with FP8 dequantization at the K/V load stage.
Reference TileLang upstream `examples/fp8_dequant_*` and
`examples/fp8_attention_*` for the on-the-fly dequant pattern.

Key design:
- K_pool / V_pool: `[max_pages, page_size, num_kv_heads, HEAD_DIM]`
  in FP8 E4M3
- Per-page scales: `[max_pages * page_size, num_kv_heads]` in
  bf16/fp16
- TileLang loads FP8 + scale, dequants to bf16 inside the warp's
  shared memory tile, then runs the same matmul as M_b.1
- FlashDecoding split-KV pattern (M_b.1 doesn't need split because
  HD128 BF16 decode contexts are short; FP8 production contexts
  reach 2k-32k tokens and need split for shared-memory headroom)

**Replaces**: `decode_attention_varlen_fp8.cu` (the 41.6%-of-GPU
hot path).

**Expected gain**: closes per-row gap to vLLM at FP8 production
shape. **Bench math (corrected 2026-05-07)**:
- ARLE current high-conc 1k/256/c=64: 843 out tok/s vs vLLM 647 =
  **+30.3% ALREADY** (post F4-Small + Phase 1A v3 multi-slot ring)
- High-conc shape is no longer the bottleneck; M_b.2 ROI here is
  diminishing returns on a leading shape
- Real M_b.2 ROI candidate: long-ctx 4k/c=4 where ARLE -3.4% vs
  vLLM out tok/s and TTFT 1.68× slower — but that is a prefill
  TTFT problem, not decode. M_b.2 (decode kernel) does NOT help
  TTFT.
- **Conclusion**: M_b.2 demoted from "world-first parity blocker"
  to **conditional optimization** post-Phase 0 baseline. Run only
  if SGLang/TRT-LLM beat ARLE high-conc by more than the F4-Small
  + Phase 1A v3 substrate margin.

(Original projection of "730 tok/s" assumed 30% F4-Big + 10% kernel
multiplicative. F4-Small + Phase 1A v3 already captured most of the
F4-Big budget and ARLE is now leading vLLM at this shape; M_b.2's
incremental value depends on actual #2 (likely SGLang/TRT-LLM) gap.)

## Tasks

| # | Task | File | LOC est. | Owner |
|---|---|---|---|---|
| M_b.1.1 | Author `batch_decode_paged_hd128.py` (adapt HD256 template) | `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py` (NEW) | ~150 | Codex (after F4-Small commits) |
| M_b.1.2 | Verify AOT generator picks up the new kernel; build + check kernel cubin emission | `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py` (no change expected) + `cargo build --features cuda` | ~10 | Codex |
| M_b.1.3 | Wire the new kernel in `crates/cuda-kernels/src/` to be selected when KV is BF16 + HD=128 | `crates/cuda-kernels/src/attention.rs` (or equivalent dispatcher) | ~50 | Codex |
| M_b.1.4 | Bench BF16 KV decode: must match or beat hand-CUDA baseline within 5% noise | `scripts/bench_guidellm.sh m_b1-bf16-baseline` (with `--kv-cache-dtype bf16`) + comparison entry | 0 | Claude |
| M_b.2.1 | FP8 dequant TileLang IR for HD128 decode | `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128_fp8.py` (NEW) | ~250 | Codex |
| M_b.2.2 | FlashDecoding split-KV in TileLang for HD128 (mirror hand-CUDA `kMaxSplits = 16`) | same file | (incl. above) | Codex |
| M_b.2.3 | Wire FP8 path: dispatch when KV is FP8 + HD=128 | `crates/cuda-kernels/src/attention.rs` | ~30 | Codex |
| M_b.2.4 | Bench at the s48 high-conc workload + nsys re-trace | `scripts/bench_guidellm.sh m_b2-fp8-prod` | 0 | Claude |

**M_b.1 total: ~210 LOC, ~1-1.5 days. M_b.2 additional: ~280 LOC,
~3-4 days. Both stages independent of F4 — can run in parallel.**

## Acceptance

### M_b.1

- New `batch_decode_paged_hd128.py` compiles via `gen_tilelang_aot.py`.
- Output cubin matches the C wrapper FFI (`ATTENTION_PUBLIC_PARAMS`).
- `cargo test --release -p infer --features cuda --test e2e` passes
  (correctness — argmax tokens identical to hand-CUDA baseline).
- Bench at BF16 KV: ≥ 95% of hand-CUDA baseline throughput
  (regression gate). If TileLang underperforms, it's an autotune
  problem (next step is `tilelang.tune` not abandon).

### M_b.2

- All M_b.1 tests + numerical correctness on FP8 KV
  (`test_decode_fp8_correctness.rs`).
- Bench at the M3.6 Phase 1 workload (1024 in / 256 out, c=64,
  s48): ≥ 580 out tok/s (≥ 25% above Phase 1 baseline 462)
  WITHOUT F4. Combined with F4-Big: ≥ 700 out tok/s (parity with
  vLLM 647).

## Out of scope

- TileLang HD256 decode (already exists for Qwen3.5).
- TileLang INT8 paged decode (`decode_attention_quantized.cu`).
  Pre-Q4-2025 INT8 path is rarely used in current bench; FP8 is
  the production target.
- Replacing hand-CUDA prefill paths. Prefill is already TileLang
  for HD128 + HD256.
- MLA decode (`mla_decode.cu`). DeepSeek-specific; not in M3.6
  scope.

## Risks + retreat

- **R1 — TileLang 0.1.9 cannot lower the FP8 dequant pattern**:
  upstream FP8 examples target newer tilelang versions. Mitigation:
  pin TileLang to a known-good version OR fall back to a hybrid
  approach (TileLang for the matmul, hand-CUDA for the
  FP8→bf16 dequant prologue).
- **R2 — Per-page scales layout mismatch**: hand-CUDA stores
  scales as `[max_pages * page_size, num_kv_heads]` (flat).
  TileLang needs the same layout — adapt the IR or change the pool
  layout (latter is a wider change, not in M_b scope).
- **R3 — TileLang autotune doesn't beat hand-CUDA**: the hand-CUDA
  kernel has 1+ year of micro-tuning. If naive TileLang loses,
  apply `tilelang.tune` with the autotune candidates from the
  HD128 prefill script as priors. Only abandon TileLang HD128
  decode if autotune still loses by > 10% — and document that as
  a hard reason in an errors entry.

## Cross-references

- Phase 1 trace evidence (the 41.6% kernel time):
  [`2026-05-07-m3.6-phase1-nsys-arle-s48-highconc.md`](../experience/wins/2026-05-07-m3.6-phase1-nsys-arle-s48-highconc.md)
- Sister plan (scheduling axis):
  [`M3.6-F4-eliminate-per-step-sync.md`](M3.6-F4-eliminate-per-step-sync.md)
- TileLang HD256 decode template (the source to adapt):
  `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd256.py`
- Hand-CUDA FP8 implementation (the kernel to replace):
  `crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`
- AOT generator (no change expected for M_b.1):
  `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py`
- Six-principles kernel review (heat map for tile/stage choices):
  `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md`
