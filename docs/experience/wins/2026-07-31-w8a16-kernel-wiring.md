# W8A16 support: the kernel existed, only the wiring was missing — 2026-07-31

> Status: Quantizer + numerics validated locally (self-check + end-to-end logit
> probe). Detect/validate/load/dispatch wired, Mac CUDA typecheck + clippy green.
> Pod serve parity (needle gate vs BF16) is the gate — pending-remote (nvcc).

## Context

Adding "a mainstream quant kernel" looked like a from-scratch job. It wasn't. An
audit of the tree showed the W8A16 path (per-group signed INT8 weights, BF16
activations — the Marlin/Machete weight-only shape) was ~90% built and dark:

- ✅ CUDA `w8a16_gemv_cuda` / `w8a16_gemv_batch_cuda` (1 byte/weight, per-group
  BF16 scale, dequant folded into the GEMV inner loop)
- ✅ `WeightFormat::W8A16` enum + `validate_shape` + `WeightKernelAlignment` +
  Display, all present
- ✅ `DeviceMatrix::from_quantized_int8(...)`, complete
- ✅ FFI decls

What was missing: nothing knew how to *recognize* a W8A16 checkpoint or *route*
to the kernel. Zero call-sites. The kernel had been written and never connected.

## What worked — mirror the W4A16 path, flip packed→non-packed

Five wiring points, each a sibling of the live W4A16 arm:

1. `QuantFormat::W8A16 { group_size }` enum variant (`quant_format.rs`).
2. Detect branch: `Dtype::I8` weight + BF16 `weight_scale`. **The one real
   difference from W4A16** — I8 is non-packed (`[rows, cols]`, 1 byte/elem),
   where W4A16 is U8 packed nibbles (`[rows, cols/2]`). Distinct dtype ⇒ no
   ambiguity with FP4/W4A16 (all U8).
3. Validate branch: rank-2, K group-aligned, BF16 scale `[rows, cols/gs]`.
4. `load_w8a16_view`: copy of `load_w4a16_view` with every `/2` and `*2`
   dropped — `shard_raw_2d_cow(.., elem=1)`, cast bytes `&[i8]`, call
   `from_quantized_int8`. Scale sharding reuses `shard_w4a16_scales_cow` verbatim
   (identical BF16 group-scale layout).
5. Four dispatch arms — `quant_linear.rs` gemv + gemv_batch — cast `qw as *const
   i8` (W4A16 casts `as *const u8`), otherwise identical.

**Scope call: dense linear only, no MoE.** No `moe_w8a16_grouped_*` kernel
exists, and the intended target (Qwen3.6-27B, the ISO-merge output) is dense. The
MoE match's existing `other => bail!` fail-closes with the format name — no
silent path, no speculative grouped-kernel work (YAGNI).

## Numerics — INT8 weights beat FP8 by ~2× at the same 8 bits

`scripts/w8a16_quant.py` (BF16 → per-group INT8, sibling of `fp8_block_cast.py`).
Measured on real Qwen3-0.6B weights, quantize→dequantize round-trip then
end-to-end forward vs the clean BF16 model:

| | worst weight rel-L2 | logit cos | logit rel-L2 | top-1 |
|---|---|---|---|---|
| **W8A16 (per-group INT8)** | **0.78%** | **0.99943** | **3.35%** | **5/5** |
| FP8 block-cast (128×128) | 2.65% | 0.99710 | 7.61% | 5/5 |

Both are behaviorally lossless (greedy top-1 unchanged on all probes); INT8 is
numerically ~2× tighter. Why: once 128-wide grouping isolates outliers, the
values inside a group have a narrow range, and INT8's *uniform* grid beats E4M3's
*exponential* grid there — E4M3 spends precision on near-zero values a tight group
doesn't have. (SVDQuant-style low-rank residual compensation was rejected earlier
for the same reason FP8 block already wins: the quant residual is full-rank
rounding noise, not low-rank outlier structure — `errors/2026-08-01`.)

## Rules

- **Audit the tree before "adding" a kernel.** Grep the FFI + enum + constructor
  before assuming from-scratch. W8A16 was one dark path away from working; the job
  was 5 wiring edits + a quantizer, not a kernel. `feedback_substrate_audit_grep_full_tree`.
- **Non-packed INT8 ≠ packed INT4 — don't copy the `/2`.** The only place the
  W4A16 mirror breaks: I8 stores 1 byte/weight, U8-nibble stores 2 weights/byte.
  Every shard/shape `/2`·`*2` must go.
- **INT8-weight is the accuracy pick, FP8 is the systems pick.** INT8 weights are
  ~2× tighter, but FP8 keeps native-format weight reads (no dequant→requant),
  train/serve format parity (Qwen/DSv4 ship FP8), and Hopper Tensor-Core
  activation-quant. W8A16 earns its place for BF16-native checkpoints (like an ISO
  merge) served where weight-bandwidth is the decode bottleneck.
