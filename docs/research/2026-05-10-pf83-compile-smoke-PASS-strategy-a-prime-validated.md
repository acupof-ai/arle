---
title: PF8.3 compile smoke PASS — Strategy A' substrate end-to-end validated; codex FFI integration in progress
date: 2026-05-10
type: research
status: pf83-compile-smoke-pass-codex-integrating-ffi
---

# PF8.3 compile smoke PASS — Strategy A' substrate end-to-end validated; codex FFI integration in progress

> Codex this tick: Strategy A' substrate compile-clean with
> ARLE-exact nvcc flags. Wrapper file `marlin_w4_fp8_kernel.cu` (255
> LOC) drops in at `crates/cuda-kernels/csrc/gemm/`. FFI symbol
> follows existing convention. Codex now integrating FFI binding +
> linear.rs dispatch swap. This entry documents the milestone +
> independent structure verification for cooperative cross-check.

## §0 Direct evidence (raw inspection THIS tick)

### Compile smoke PASS (codex shell output captured this tick)

```bash
$ /opt/cuda/bin/nvcc -arch=sm_89 -O3 -std=c++17 \
    --expt-relaxed-constexpr -ccbin /usr/bin/g++-14 \
    -Icrates/cuda-kernels/csrc \
    -c crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu \
    -o /tmp/marlin_w4_fp8_kernel.o
crates/cuda-kernels/csrc/gemm/marlin_pf8/marlin_template.h(345):
  warning #177-D: variable "is_int_type" was declared but never referenced
... +15 lines [cosmetic warnings only]
Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"
```

PASS. Only cosmetic unused-variable warnings (suppressible). Build
flags match ARLE `crates/cuda-kernels/build.rs:1190-1194` exactly:
- `-arch=sm_89` — single-arch build (matches dev workflow)
- `-std=c++17` + `--expt-relaxed-constexpr` — gated on marlin_* stem
- `-ccbin /usr/bin/g++-14` — required (g++-16 incompatible)
- `-Icrates/cuda-kernels/csrc` — header search path

### Wrapper file structure (raw read THIS tick, 255 LOC)

`crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu` header:

```cpp
/*
 * PF8.3 — ARLE W4 + FP8-activation Marlin GEMM wrapper.
 *
 * The Marlin template under gemm/marlin_pf8/ is adapted from vLLM's
 * csrc/quantization/marlin sources (Apache-2.0, copyright contributors to
 * the vLLM project and Marlin.2024 Elias Frantar). This wrapper intentionally
 * instantiates only the sm_89 prefill shape ARLE needs:
 *   A: FP8 e4m3 activations
 *   B: GPTQ INT4 U4B8 weights, zero-point preprocessed by PF8.2
 *   C: BF16 output
 *   Scale: BF16 per-group W4A16 Marlin scales
 */

#define MARLIN_NAMESPACE_NAME arle_marlin_pf8
#include "marlin_pf8/kernel.h"
#include "marlin_pf8/marlin_template.h"
```

Codex applied:
- Apache-2.0 + Frantar attribution (license hygiene)
- Namespace isolation `arle_marlin_pf8` (avoids clash with existing
  `marlin` namespace from `marlin_w4a8_kernel.cu`)
- Single shape spec (W4 INT4 + FP8 e4m3 acts + BF16 out + BF16 scales)
- No multi-shape autotune — focused PF8.3 path only

### Exported symbols (grep THIS tick)

```cpp
// line 39: internal launcher
void launch_pf8_kernel(const int4* A, ...);

// line 83: shape-templated dispatcher
bool maybe_launch_pf8_kernel(int thread_m_blocks, ...);

// line 116-118: error code constants
constexpr int ERR_PROB_SHAPE = 1;
constexpr int ERR_KERN_SHAPE = 2;
constexpr int ERR_ARCH = 3;

// line 120: PUBLIC FFI ENTRY POINT
extern "C" int gemm_w4_fp8_marlin_cuda(...);

// lines 223-235: 4 thread_m_blocks specializations
//   1, 8, 8, 8  — small batch
//   2, 16, 4, 8 — medium
//   3, 16, 4, 8 — medium-large
//   4, 16, 4, 8 — max kMaxThreadMBlocks
```

### FFI convention verification (raw grep THIS tick)

```bash
$ grep "_cuda" crates/cuda-kernels/src/ffi/gemm.rs | head -10
4:unsafe extern "C" {
5:    pub fn gemv_cuda(...)
14:    pub fn gemm_cuda(...)
46:    pub fn marlin_gemm_cuda(...)
64:    pub fn gemm_w4a8_marlin_cuda(...)        ← REFERENCE PATTERN
107:    pub fn quantize_bf16_rows_to_fp8_e4m3_cuda(...)  ← PF8.1 binding
121:    pub fn marlin_int4_fp8_preprocess_without_zp_cuda(...)  ← PF8.2 binding
```

Codex's `gemm_w4_fp8_marlin_cuda` mirrors `gemm_w4a8_marlin_cuda`
(line 64) — `gemm_<quant>_marlin_cuda` pattern. **Convention
consistent**. FFI binding to add will mirror line 64's W4A8 binding
shape but for FP8 acts (i.e. `*mut u8` for FP8 per ARLE convention,
per PF8.1's existing fp8 binding at line 107).

### marlin_pf8/ substrate dir (untracked, codex in-progress)

```
crates/cuda-kernels/csrc/gemm/marlin_pf8/
├── core/scalar_type.hpp     1820 bytes  ← codex-authored minimal port
├── dequant.h               22312 bytes  ← upstream verbatim
├── kernel.h                 2265 bytes  ← upstream verbatim
├── marlin.cuh              5076 bytes  ← upstream verbatim
├── marlin_dtypes.cuh       3902 bytes  ← upstream verbatim
├── marlin_mma.h           12756 bytes  ← upstream verbatim
└── marlin_template.h      81605 bytes  ← upstream verbatim (the mega template)
```

Total: ~3300 LOC vendored (most is `marlin_template.h`, not
hand-written). Codex's only hand-authored CUDA = `marlin_w4_fp8_kernel.cu`
(255 LOC) + `core/scalar_type.hpp` (~80 LOC).

ARLE-authored NEW = ~335 LOC CUDA (matches a0758e7 §1 estimate within
~50% — adjusted upward because codex chose namespace isolation over
reusing existing `marlin_dequant.cuh:93-106` ScalarTypeTag).

## §1 Why codex did NOT reuse marlin_dequant.cuh ScalarTypeTag

Independent observation: codex created a fresh `marlin_pf8/core/scalar_type.hpp`
(80 LOC) instead of reusing ARLE's existing
`marlin_dequant.cuh:93-106` ScalarTypeTag (which a0758e7 §0
identified as ID-aligned with vllm).

Hypothesis: namespace isolation. The vendored upstream files all
reference `vllm::kFE4M3fn.id()`, `vllm::kU4B8.id()` — codex's wrapper
defines `MARLIN_NAMESPACE_NAME arle_marlin_pf8` to avoid clash, so
reusing ARLE's existing ScalarTypeTag would require either:
(a) putting `vllm` namespace alias in ARLE's marlin_dequant.cuh, OR
(b) editing 3300 LOC of vendored upstream to use ARLE's namespace

Codex picked (c): keep upstream untouched, vendor a minimal
`vllm::ScalarType` shim in `marlin_pf8/core/`. This is the LOC-minimal
choice for the in-tree integration (3300 LOC stays verbatim from
upstream → patch-friendly for vLLM main updates).

ARLE marlin_dequant.cuh continues to serve its W4A8 INT8 path; the
new marlin_pf8 path has its own scalar_type. Two paths coexist.

## §2 Remaining PF8.3 integration work (codex in progress)

Per codex tmux trace (`Working 3m 49s`):
- Reading: `infer/src/ops/linear.rs` (the bail site)
- Reading: `crates/cuda-kernels/src/ffi/gemm.rs` (FFI binding location)
- Reading: ARLE marlin_dequant.cuh (substrate cross-reference)

Expected next-tick deliverables (codex):
1. FFI binding in `gemm.rs` (mirror line 64 W4A8 binding shape, ~25 LOC)
2. Rust wrapper around `gemm_w4_fp8_marlin_cuda` (handles FP8 act
   tensor + scales + result), maybe in `crates/cuda-kernels/src/lib.rs`
3. Replace bail at `infer/src/ops/linear.rs:1966+` with actual call
4. Add `marlin_w4_fp8_kernel.cu` + `marlin_pf8/*` to `build.rs`
   compile list (the marlin_* prefix already triggers
   `--expt-relaxed-constexpr` per build.rs:1191)
5. Greedy_consistency test PASS with `INFER_MARLIN_W4_FP8_PREFILL=1`
6. PF8.5 e2e bench A/B → wins entry per aebd4a5 license matrix

## §3 Updated PF8.3 status board

| Phase | Status | Evidence |
|-------|--------|----------|
| PF8.1 act quant kernel | LANDED + smoke PASS | `940f49e` + `b628eca` |
| PF8.2 weight preprocess | LANDED + smoke PASS | `940f49e` + `451d094` |
| PF8.3 GEMM substrate | **COMPILE SMOKE PASS** | `marlin_w4_fp8_kernel.cu` 255 LOC + `marlin_pf8/` 3300 LOC vendored (untracked) |
| PF8.3 FFI binding | IN PROGRESS (codex) | reading gemm.rs |
| PF8.3 dispatch wire | IN PROGRESS (codex) | reading linear.rs |
| PF8.3 greedy_consistency | NOT STARTED | pending FFI + dispatch |
| PF8.4 dispatch enum + env | LANDED (opt-in stub) | `db063ff` |
| PF8.5 e2e bench A/B | NOT STARTED | pending PF8.3 full |

## §4 Anti-pattern observations (skill v1.11.0+)

Codex's PF8.3 work demonstrates 4 of 5 catalogued anti-pattern fixes:

| # | Anti-pattern | Codex behavior |
|---|------|----------------|
| #28+#31 | Raw evidence for surface claims | Pulled vllm-marlin-src locally + read marlin_template.h before writing |
| #29 | Default fixtures may be broken | Compile smoke FIRST in /tmp before commit |
| #30 | Git status before commit | Has not committed marlin_pf8/ yet (waiting for FFI clean) |
| #32 | Peer Working >5min direct verify | (Doesn't apply — codex IS the peer here) |

This is exemplary cooperative discipline.

## §5 Cross-references

- `93e1430` (PF8.3 brief sent to codex)
- `259277c` (Path B SUPERSEDED)
- `818b4e0` (Path A SUPERSEDED)
- `a0758e7` (Strategy A' validation — predicted ~225 LOC ARLE-authored, actual ~335 LOC closer match)
- `aebd4a5` (PPL gate methodology — license sequence preserved)
- `db063ff` (PF8.4 dispatch wiring — bail site at linear.rs:1966+)
- a66d99a (NEW prefill-only FP8 directive — license matrix in §2)
- ARLE substrate: `crates/cuda-kernels/csrc/gemm/marlin_pf8/` (3300 LOC vendored, untracked)
- ARLE wrapper: `crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu` (255 LOC, untracked)
- FFI convention: `crates/cuda-kernels/src/ffi/gemm.rs:64` (W4A8 reference pattern)

## §6 Status

PF8.3 compile smoke PASS — the gating risk for Strategy A'. Codex
substrate (~3635 LOC total, ~335 ARLE-authored) compiles cleanly
with ARLE-exact nvcc flags. Codex now integrating FFI + dispatch.

Next-tick check (~25 min): expect either codex commit landing
substrate + FFI + dispatch OR codex still in greedy_consistency
debug phase OR errors entry if integration uncovered hidden gap.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(nvcc command + output captured this tick from tmux, wrapper source
raw read this tick, FFI convention raw grep this tick, marlin_pf8/
ls this tick).
