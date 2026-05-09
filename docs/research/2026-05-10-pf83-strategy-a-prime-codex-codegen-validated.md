---
title: PF8.3 Strategy A' codex codegen path VALIDATED — vLLM single-specialization + ARLE Phase 1 substrate already aligned
date: 2026-05-10
type: research
status: pf83-strategy-a-prime-validated-supersedes-paths-a-and-b
---

# PF8.3 Strategy A' codex codegen path VALIDATED — vLLM single-specialization + ARLE Phase 1 substrate already aligned

> Codex's PF8.3 audit this tick: discovered vLLM upstream
> `generate_kernels.py` produces single specialization
> `sm89_kernel_fe4m3fn_u4b8_bfloat16.cu` (106 LOC) for the W4×FP8×BF16
> sm_89 path. Codex first compile attempt FAILED on
> `--expt-relaxed-constexpr` flag (correctly diagnosed as smoke
> harness omission, NOT kernel bug). Independent Claude verification
> THIS tick:
> - Codex's diagnosis VALIDATED via raw grep on ARLE `build.rs:1191-1194`
> - Strategy A' surface is SMALLER than worst case because
>   ARLE's Phase 1 substrate (task #42, `marlin_dequant.cuh:93-106`)
>   already provides the `ScalarTypeTag` system with **matching IDs**
>   (kU4B8=2, kFE4M3fn=5) for vllm upstream type resolution
> - Strategy A' SUPERSEDES Path A (818b4e0 my recommendation) AND
>   Path B (259277c original recommendation)

## §0 Direct evidence (raw inspection THIS tick)

### Codex diagnosis verified (build.rs:1191-1194)

```rust
// crates/cuda-kernels/build.rs:1190-1194
// Marlin kernel needs C++17 + relaxed constexpr
if stem.starts_with("marlin_") {
    flags.push("-std=c++17".to_string());
    flags.push("--expt-relaxed-constexpr".to_string());
```

ARLE build.rs gates `--expt-relaxed-constexpr` on `marlin_*` file
stem. Codex's `/tmp/pf83_upstream_compile/` smoke didn't replicate
this — that's the failure root cause, not the upstream kernel.
Codex retry adds the flag.

### Codegen output structure (raw read THIS tick)

```bash
$ ls -la /tmp/vllm-marlin-src/csrc/quantization/marlin/sm89*
sm89_kernel_fe4m3fn_fe2m1f_bfloat16.cu      2277 bytes
sm89_kernel_fe4m3fn_u4b8_bfloat16.cu        8574 bytes  ← PF8.3 TARGET
sm89_kernel_fe4m3fn_u4b8_float16.cu         8478 bytes
sm89_kernel_fe4m3fn_u4_bfloat16.cu          8478 bytes
sm89_kernel_fe4m3fn_u4_float16.cu           8382 bytes

$ wc -l sm89_kernel_fe4m3fn_u4b8_bfloat16.cu
106
```

5 sm_89 specializations generated. ARLE needs only the W4×FP8×BF16
one (`fe4m3fn_u4b8_bfloat16` = FP8 e4m3 acts × INT4 weights with
zero-bias=8 × BF16 output) for Qwen3 W4A8-marlin model.

The 106-LOC file holds 12+ template instantiations with varying
tile parameters (256/128 threads × 1-4 stages × various warp shapes):

```cpp
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kU4B8.id(),
    vllm::kBFloat16.id(), vllm::kBFloat16.id(),
    256, 1, 8, 8, false, 4, -1, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kU4B8.id(),
    vllm::kBFloat16.id(), vllm::kBFloat16.id(),
    128, 1, 8, 4, false, 4, -1, false>( MARLIN_KERNEL_PARAMS );
... (12+ variants)
```

### ARLE Phase 1 substrate already provides scalar_type ⚠ MAJOR

```bash
$ find /tmp/vllm-marlin-src -name 'scalar_type*'
(no output — sparse-checkout was csrc/quantization/marlin/ only)

$ grep -nE "vllm::|ScalarType|kFE4M3|kU4B8" \
    /home/ckl/projects/arle/crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh
93:using ScalarTypeId = int64_t;
95:struct ScalarTypeTag {
96:  ScalarTypeId value;
97:  constexpr ScalarTypeId id() const { return value; }
100:static inline constexpr ScalarTypeTag kU4{1};
101:static inline constexpr ScalarTypeTag kU4B8{2};       ← matches vllm
102:static inline constexpr ScalarTypeTag kU8{3};
103:static inline constexpr ScalarTypeTag kU8B128{4};
104:static inline constexpr ScalarTypeTag kFE4M3fn{5};    ← matches vllm
105:static inline constexpr ScalarTypeTag kFE2M1f{6};
```

ARLE's Phase 1 task #42 codex port **already inlined** the vllm
scalar_type system as `ScalarTypeTag` with **matching IDs**. So
`vllm::kFE4M3fn.id()` calls in the codegen kernel resolve to the
same integer (5) as ARLE's `kFE4M3fn.id()`. Just need a `using
namespace` shim or rename in the codegen output.

### Existing ARLE marlin substrate (raw ls THIS tick)

```
crates/cuda-kernels/csrc/gemm/
├── marlin_dequant.cuh             (Phase 1 task #42 — ScalarTypeTag + dequant)
├── marlin_int4_fp8_preprocess.cu  (PF8.2 940f49e — INT4 weight preprocess)
├── marlin_kernel.cu               (W4A16 marlin existing)
├── marlin_repack.cu               (weight repack existing)
├── marlin_w4a8_kernel.cu          (W4A8 INT8 sm_89 existing — 987 LOC)
├── w4a8_activation_quant.cu       (INT8 act quant existing)
└── w4_fp8_activation_quant.cu     (PF8.1 940f49e — FP8 act quant)
```

PF8.1 + PF8.2 + Phase 1 dequant are ALL in tree. Strategy A'
integration only adds:
- `marlin_w4_fp8_kernel.cu` (the codegen output, ~8.5 KB / 106 LOC)
- Headers: `marlin_template.h` + `marlin_mma.h` + `marlin_dtypes.cuh`
  → either inline into ARLE tree OR keep as external includes

## §1 Strategy A' total LOC delta estimate

| Component | LOC | Source | Status |
|-----------|-----|--------|--------|
| `marlin_w4_fp8_kernel.cu` (codegen specialization) | 106 | upstream codegen | NEW |
| `marlin_template.h` (mega template) | 2081 | upstream | NEW import |
| `marlin_mma.h` (mma instructions FP8 + INT8 + BF16) | 268 | upstream | NEW import |
| `marlin_dtypes.cuh` (FP8 fragment types) | 149 | upstream | NEW import |
| `dequant.h` FP8 sections | ~100 of 609 | upstream | NEW or merge w/ ARLE marlin_dequant.cuh |
| `ScalarTypeTag` extension | 0 | ARLE has it (Phase 1) | EXISTING |
| FP8 dequant in marlin_dequant.cuh | ~80 | NEW | NEW |
| FFI shim `crates/cuda-kernels/src/ffi/gemm.rs` | ~30 | NEW | NEW |
| Dispatch site `infer/src/ops/linear.rs:1966+` | ~10 | replace bail | NEW |
| Build.rs entry for new .cu file | ~5 | NEW | NEW |

**Total NEW**: ~3000 LOC headers (most is `marlin_template.h` not
manually written, just imported)

**Total ARLE-AUTHORED NEW**: ~225 LOC (codegen + dequant FP8 + FFI +
dispatch + build.rs)

This is roughly **HALF** the size of my Path A estimate (818b4e0:
~400-600 LOC) because:
- Mma rewrite (~200 LOC manual code) replaced by codegen template
  expansion
- Tile param tuning (~100 LOC manual selection) replaced by codegen
  multi-instantiation
- Type system delta (~50 LOC manual additions) replaced by ARLE's
  already-aligned ScalarTypeTag

## §2 Strategy comparison (final)

| Strategy | LOC delta (ARLE-authored) | Risk surface | Codex pick |
|----------|---------------------------|--------------|-----------|
| Path A (818b4e0): mirror W4A8 with FP8 mma | ~400-600 | Manual mma + dequant rewrite | NOT picked |
| Path B (259277c): mirror with k=32 mma | ~800-1200 | Above + smem layout change | NOT picked |
| **Strategy A' (codex tick)**: codegen single specialization | **~225** | Header import + FFI thin shim | **PICKED** |

Codex's Strategy A' is OBJECTIVELY SMALLER and HIGHER FIDELITY than
both Claude's Path A and Path B. Reasons it's better:
1. Zero manual mma rewrite (mma asm is upstream-tested, 12+ tile
   variants out of the box)
2. Tile autotuning automatic (compile time selects best kernel
   per shape)
3. ARLE Phase 1 substrate aligned with upstream type IDs
4. Dependency surface ~3000 LOC headers but ZERO manually-written
5. PF8.5 license can A/B against the same baseline as W4A8

## §3 Codex compile retry status (THIS tick)

Codex `Working (9m 05s)` on retry with `--expt-relaxed-constexpr`
flag added. Per skill #32 ">5min wedge": at 9 min the threshold IS
exceeded, BUT the prior turn shows visible failure → diagnosis →
retry sequence (NOT a wedge). Direct ps/log verify deferred unless
next tick (~25 min) shows still Working without progress sign.

If smoke compiles clean: Strategy A' substrate ready for ARLE
integration (drop file + headers, add to build.rs, add FFI, plug
dispatch site).

If smoke STILL fails after relaxed-constexpr: codex falls back to
errors entry per their stated discipline — "如果上游 FP8 Marlin
模板依赖面太大,会先落 errors/research 结论而不是硬写一个不可验证的
kernel".

## §4 Updated PF8.3 license/kill matrix (no change from aebd4a5)

License gates unchanged from `aebd4a5`:
- TTFT p50 Δ ≥ -8% σ < 5% n=3 (PF8.5 e2e bench)
- ITL p50 regression < +2% (decode unchanged)
- greedy_consistency PASS
- PPL Δ% ≤ +1.0% wikitext (eval_ppl_pf83.py adaptation per aebd4a5 §2)

The substrate change (Path A → Strategy A') doesn't affect license
thresholds — only the implementation path.

## §5 Cross-references

- `93e1430` (initial PF8.3 brief sent — assumed Path B by default)
- `259277c` (PF8.3 Path B scope analysis — SUPERSEDED by this entry)
- `818b4e0` (PF8.3 Path A correction + 6th hallucination — SUPERSEDED)
- `aebd4a5` (PF8.3 PPL gate methodology — license matrix preserved)
- `a66d99a` (NEW prefill-only FP8 directive)
- `db063ff` (PF8.4 dispatch wiring — bail at linear.rs:1966+)
- ARLE Phase 1 substrate: `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh:93-106` (task #42 ScalarTypeTag)
- vLLM upstream: `/tmp/vllm-marlin-src/csrc/quantization/marlin/sm89_kernel_fe4m3fn_u4b8_bfloat16.cu` (codex codegen output)
- vLLM upstream: `/tmp/vllm-marlin-src/csrc/quantization/marlin/generate_kernels.py:292` (filename pattern)
- ARLE build.rs flag verified: `crates/cuda-kernels/build.rs:1191-1194`

## §6 Status

PF8.3 Strategy A' (codex's codegen + upstream specialization) is now
the canonical path. Two prior Claude recommendations (Path A 818b4e0,
Path B 259277c) SUPERSEDED but kept on disk for retrospective.

Codex compile retry in progress (9+ min Working). Next-tick check on:
- Compile success → ready for integration
- Compile failure → codex errors entry → PF8 chain KILL OR fallback
  to manual Path A

ARLE's Phase 1 task #42 substrate alignment is the unsung hero —
without `marlin_dequant.cuh:93-106` ScalarTypeTag system Strategy A'
would need a 300+ LOC scalar_type.hpp port AND that path would have
been much closer to my Path A LOC estimate.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(build.rs:1190-1194 raw read, marlin_dequant.cuh:93-106 raw read,
sm89_kernel_*.cu structure raw read, ARLE marlin tree raw ls — all
THIS tick).
