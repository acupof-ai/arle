# W8A16 support: the kernel existed, only the wiring was missing — 2026-07-31

> Status: **Validated end-to-end on H20, up to the 27B target.** The ISO-merged
> Qwen3.6-27B (`iso-tc-huihui`) quantized to W8A16 (492 tensors) serves on one H20
> and is byte-identical to its BF16 source on greedy probes; a dense 0.8B proved
> the path first. Quantizer + numerics validated locally; detect/validate/load/
> dispatch wired, Mac CUDA typecheck + clippy green.

## Context

Adding "a mainstream quant kernel" looked like a from-scratch job. It wasn't. An
audit of the tree showed the W8A16 path (per-group signed INT8 weights, BF16
activations — the Marlin/Machete weight-only shape) was ~90% built and dark:

- CUDA `w8a16_gemv_cuda` / `w8a16_gemv_batch_cuda` (1 byte/weight, per-group
  BF16 scale, dequant folded into the GEMV inner loop)
- `WeightFormat::W8A16` enum + `validate_shape` + `WeightKernelAlignment` +
  Display, all present
- `DeviceMatrix::from_quantized_int8(...)`, complete
- FFI decls

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

## Pod validation (H20, GPU 2) — and the scope-vs-coverage bug it caught

Served a dense Qwen3.5-0.8B (`Qwen3_5ForConditionalGeneration`, 24 layers)
quantized to W8A16 (208 tensors, group 128) against its BF16 source, same GPU:

| prompt | W8A16 | BF16 | match |
|---|---|---|---|
| "The capital of France is" | " Paris.\n…" | " Paris.\n…" | byte-identical |
| "def fibonacci(n):" | valid `if n<=0…` | same | byte-identical |
| "2+2=" | "4, …" | "4\n…" | answer "4" identical, diverges only on trailing separator |

Loaded + engine-ready in ~5s. The INT8 kernel path fires and produces correct
inference. Two real load-time errors surfaced first — both signal, not noise:

1. **Legacy `Qwen3ForCausalLM` builder is not W8A16-aware.** A Qwen3-4B W8A16
   crashed at `q_proj.weight: clean CUDA path accepts BF16 only, got I8`. W8A16
   dispatch is wired only into the `qwen35` builder — the arch limit is real,
   document it, don't paper over it.
2. **Quant scope exceeded loader coverage.** `--all-linear` quantized
   `linear_attn.in_proj_a`/`in_proj_b` (`[16,1024]`, one scalar per v-head —
   gates, not GEMMs), which `qwen35.rs:3296-3297` loads BF16-only by design.
   Serve read I8 through the BF16 path and bailed. Fix: the quantizer's
   `ALL_LINEAR_SKIP` now carries `in_proj_a`/`in_proj_b`/`conv1d` — the complete
   BF16-only `.weight` set in the builder (`195ba2e5d`).

This is why the pod gate exists: local numerics were all green (self-check, logit
probe), but the quantizer↔loader scope contract can only fail on a real serve.

## 27B validation — the real target, at the final code

The 0.8B proved the path; the 27B is what it was for. `iso-tc-huihui` (the ISO
merge output — BF16 dense Qwen3.6-27B, 55.5 GB) → W8A16 with the final
`e739a1105` script (520-tensor all-linear scope, **492 quantized**, group 128) →
29 GB. Served on one H20 (68 GB free → 337K max KV tokens), loaded clean, W8A16
kernel dispatched. All three greedy probes coherent (Paris / a correct recursive
fibonacci / "40 mph"), and **byte-identical to the BF16 source** on every prompt.
The merge's capability survives an 8-bit weight cast with no visible loss.

## Build acceleration — measured, and two of my assumptions were wrong

Tried to speed up the pod CUDA build (255 s clean). The honest result:

- **Arch pinning was already on.** `scripts/pod-build-env.sh:12` already exports
  `TORCH_CUDA_ARCH_LIST=9.0`; every sanctioned pod build already compiles sm_90
  only (confirmed in the nvcc lines: `-gencode arch=compute_90,code=sm_90` + PTX,
  no sm_80/86/89). The "÷4 from 4-arch default" I expected did not exist — a
  predecessor had already pinned it. `build.rs:195`'s 4-arch default only fires
  when no GPU is visible at build time, which the build-env overrides.
- **nvcc-sccache is the right knob but couldn't be measured this run** — the tn
  proxy (`127.0.0.1:1080`) was down, so sccache couldn't be installed and both
  `RUSTC_WRAPPER` and `ARLE_NVCC_WRAPPER` were empty. The wiring is correct; its
  payoff (warm-cache relink) awaits a run with the proxy up.
- **`--profile release-fast` vs `--release`: 1.22×** (209 s vs 255 s clean). Modest
  because this box's build is nvcc + arle-link bound, not Rust codegen-units, and
  LTO-link savings were partly masked by a concurrent foreign build sharing cores.

Net: the `ARLE_NVCC_WRAPPER=sccache` wiring now lives beside `RUSTC_WRAPPER` in
`pod-build-env.sh`'s sccache block (same guard) — the canonical place, not the
one-off spot it was first put. No hard-coded arch anywhere; the build-env owns it.

## Rules

- **Audit the tree before "adding" a kernel.** Grep the FFI + enum + constructor
  before assuming from-scratch. W8A16 was one dark path away from working; the job
  was 5 wiring edits + a quantizer, not a kernel. `feedback_substrate_audit_grep_full_tree`.
- **Non-packed INT8 ≠ packed INT4 — don't copy the `/2`.** The only place the
  W4A16 mirror breaks: I8 stores 1 byte/weight, U8-nibble stores 2 weights/byte.
  Every shard/shape `/2`·`*2` must go.
- **Quant scope must be a subset of the loader's quant-aware coverage.** A
  quantizer that quantizes more tensors than the loader routes through the
  quant-aware path produces a checkpoint that serve reads through the BF16 path
  and crashes on the I8 bytes. The scope list is defined by the loader, not by
  ".weight" name-matching — enumerate every plain-`load_matrix` call in the
  builder and skip exactly those (`embed`, `lm_head`, `in_proj_a/b`, `conv1d`).
  Local numerics can't catch this; only a real serve can.
- **INT8-weight is the accuracy pick, FP8 is the systems pick.** INT8 weights are
  ~2× tighter, but FP8 keeps native-format weight reads (no dequant→requant),
  train/serve format parity (Qwen/DSv4 ship FP8), and Hopper Tensor-Core
  activation-quant. W8A16 earns its place for BF16-native checkpoints (like an ISO
  merge) served where weight-bandwidth is the decode bottleneck.
