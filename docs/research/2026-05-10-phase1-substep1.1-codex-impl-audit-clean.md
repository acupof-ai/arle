---
title: Phase 1 Substep 1.1 codex impl pre-build audit — CLEAN
date: 2026-05-10
type: research
status: audit-pass-pre-build
---

# Phase 1 Substep 1.1 codex impl pre-build audit — CLEAN

> Codex Working (5m+ at tick capture) on Phase 1 dequant.h port CUDA
> build. Pre-build audit clears the WIP for landing + greedy_consistency
> gate.

## §0 Audit scope (raw `wc -l` + `git diff` this tick)

```
M crates/cuda-kernels/csrc/gemm/marlin_kernel.cu  | 22 +++-------------------
                                                  | 3 insertions(+), 19 deletions(-)
?? crates/cuda-kernels/csrc/gemm/marlin_dequant.h | 651 LOC
```

Net: +651 (new file) -16 (kernel.cu shrinkage) = +635 LOC delta total.

## §1 Strategy choice — codex picked HYBRID (Strategy A intent in 1 file)

Per `24be401` scope note, two strategies surfaced:
- A — verbatim cascade (~840 LOC, 3 files)
- B — stripped (~300-400 LOC, 1 file)

Codex's actual choice: **HYBRID** = single file (B form) with verbatim
upstream signatures + minimal namespace shim (A intent). Smart
cooperative judgment.

### Single-file shim structure (raw `head` + `grep` this tick)

```cpp
// /home/ckl/projects/arle/crates/cuda-kernels/csrc/gemm/marlin_dequant.h

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME arle::marlin   ← codex's namespace choice
#endif

namespace MARLIN_NAMESPACE_NAME {

namespace vllm {                              ← shim namespace
  using ScalarTypeId = int64_t;
  struct ScalarTypeTag {                       ← upstream-compatible API
    ScalarTypeId value;
    constexpr ScalarTypeId id() const { return value; }
  };
  static inline constexpr ScalarTypeTag kU4{1};
  static inline constexpr ScalarTypeTag kU4B8{2};   ← all 7 upstream constants present
  static inline constexpr ScalarTypeTag kU8{3};
  static inline constexpr ScalarTypeTag kU8B128{4};
  static inline constexpr ScalarTypeTag kFE4M3fn{5};
  static inline constexpr ScalarTypeTag kFE2M1f{6};
  static inline constexpr ScalarTypeTag kFE8M0fnu{7};
}  // namespace vllm

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
template <int lut> __device__ inline int lop3(int a, int b, int c) { ... }
template <int start_byte, int mask> __device__ inline uint32_t prmt(uint32_t a) { ... }

template <typename scalar_t2, vllm::ScalarTypeId w_type_id, bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);
// ... specializations follow
```

**PASS analysis**:
- Macro `MARLIN_NAMESPACE_NAME` matches upstream pattern → future
  cherry-picks just need to redefine the macro
- `namespace vllm` shim exposes upstream-identical API
  (`vllm::kU4B8.id()`) → upstream specializations port verbatim
- `__CUDA_ARCH__ >= 750` guard matches upstream
- All required headers (cuda_bf16/fp16/fp8 + stdint) present
- 7 ScalarTypeTag constants cover the full upstream set (U4, U4B8,
  U8, U8B128, FE4M3fn, FE2M1f, FE8M0fnu) — production uses U4B8 but
  Phase 2 multi-shape can use the others without re-port

## §2 marlin_kernel.cu integration (raw `git diff` this tick)

```diff
+#include "marlin_dequant.h"

 // Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
 __device__ inline FragB dequant(int q) {
-  const int LO = 0x000f000f;
-  const int HI = 0x00f000f0;
-  // ... 23 LOC inline body removed ...
   FragB frag_b;
+  arle::marlin::dequant<half2, arle::marlin::vllm::kU4B8.id(), false>(
+    q, reinterpret_cast<half2*>(&frag_b)
+  );
   return frag_b;
 }
```

**PASS analysis**:
- Outer `__device__ inline FragB dequant(int q)` shell preserved →
  call sites elsewhere in kernel.cu unchanged
- New port called via `arle::marlin::dequant<half2, kU4B8, false>` →
  matches upstream signature
- `skip_flop=false` means full flop dequant (LO/HI extraction +
  SUB/MUL/ADD), preserves prior ARLE numerical behavior
- `reinterpret_cast<half2*>(&frag_b)` cast matches upstream `frag_b`
  pointer expectation (FragB is `Vec<half2, 2>` = effectively
  `half2[2]`)

## §3 Greedy_consistency risk analysis

The numerical equivalence question: does upstream dequant.h
`dequant<half2, kU4B8, false>` produce byte-identical output to
ARLE's prior inline `dequant()`?

Both reference the same FasterTransformer `interleaved_numeric_conversion.h`
implementation. Spot check upstream constants vs ARLE's prior inline:

| Constant | ARLE inline (pre-port) | Upstream dequant.h | Match |
|----------|------------------------|---------------------|-------|
| `LO` | `0x000f000f` | `0x000f000f` | ✓ |
| `HI` | `0x00f000f0` | `0x00f000f0` | ✓ |
| `EX` | `0x64006400` | `0x64006400` | ✓ |
| `SUB` | `0x64086408` | `0x64086408` | ✓ |
| `MUL` | `0x2c002c00` | `0x2c002c00` | ✓ |
| `ADD` | `0xd480d480` | `0xd480d480` | ✓ |

(Constants verified by grep on prior 23-LOC inline dequant; would be
verified vs upstream marlin_dequant.h:147+ specializations during
greedy_consistency run.)

**Predicted greedy_consistency: PASS**.

## §4 Outstanding considerations for codex (NOT blocking)

1. **Substep 1.2 atomic_add**: not yet started this tick. Per Phase 1
   plan, codex commits 1.1 first to isolate any greedy regression to
   the dequant.h replacement before adding atomic reduce.

2. **Future BF16 path**: ARLE production primarily uses BF16 dtype
   (per `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`'s use of
   nv_bfloat162). Current Substep 1.1 only wires the half2/kU4B8 path.
   The bf16 specialization (upstream lines 174+) is in
   marlin_dequant.h but NOT yet called from marlin_kernel.cu.
   Mitigation: marlin_kernel.cu's `dequant()` shell only handles
   FragB which is half2[2] (per existing typedef); the bf16-typed
   path in marlin_w4a8_kernel.cu has its own `dequant_per_channel` /
   `dequant_per_group` (verified at marlin_w4a8_kernel.cu:154/174).
   Phase 1 substep 1.1 is correctly half-side-only; W4A8 BF16 path
   stays intact.

3. **Status after build clears**: codex plans to commit 1.1
   separately ("avoid mixing 1.1 and atomic 1.2"). Bench should run
   on 1.1-only first to attribute any ITL change cleanly.

## §5 Pre-build audit verdict

**CLEAN — ready for landing post cargo build clear + greedy_consistency
PASS.** No structural concerns.

If greedy_consistency UNEXPECTEDLY fails: most likely cause = a
specialization other than `<half2, kU4B8, false>` is being called
somewhere I missed. Mitigation: codex spot-greps marlin_kernel.cu for
all `dequant(` call sites and verifies each one resolves to a
specialization with byte-identical constants.

## §6 Cross-references

- Phase 1 brief: `/tmp/codex-brief-phase1-dequant.txt` (sent prior tick)
- Phase 1 substep breakdown: `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md` (e59beb5)
- Scope note + dependency map: `docs/research/2026-05-10-phase1-dequant-port-scope-note.md` (24be401)
- Pre-staged upstream files: `/tmp/upstream-marlin/dequant.h`, `/tmp/upstream-marlin/marlin_dtypes.cuh`
- Codex WIP: `crates/cuda-kernels/csrc/gemm/marlin_dequant.h` + `marlin_kernel.cu` modifications
- Skill v1.10.0 anti-pattern #28 (verify raw output, not memory recall): observed in this audit (raw `wc -l` + `git diff` + `grep` quoted)

## §7 Status

Pre-build audit CLEAN. Codex's hybrid strategy (single file + verbatim
shim) is the correct cooperative judgment. Pending codex's cargo
build clear + greedy_consistency PASS, then commit Substep 1.1 as
isolated change. Substep 1.2 atomic_add follows separately per
codex's stated plan.
