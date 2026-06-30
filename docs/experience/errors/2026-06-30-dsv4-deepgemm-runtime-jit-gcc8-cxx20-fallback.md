# DSv4 DeepGEMM silently falls back to BF16 — runtime JIT needs g++≥10 (C++20), not a build flag

## Context

DSv4-Flash-FP8 TP=4 decode nsys showed `gemv_handwritten` (BF16 GEMV) at **53.4%
of GPU time** — the FP8 projections (and MoE) were running on the scalar/BF16
hand path instead of DeepGEMM tensor cores. Initial hypotheses (wo unwired,
projections not batched) were all **falsified by reading the code** — those paths
already exist and are default-on. The real cause was found only by driving the
binary on the pod.

## Root Cause (pod-verified, H20, 2026-06-30)

DeepGEMM is a **runtime JIT**: it `nvcc -cubin -std=c++20`-compiles its SM90 FP8
kernels into `DG_JIT_CACHE_DIR` on first use (`deepgemm_native.cu` `get_or_build_runtime`).
There are **two independent gates**, and the failure is at the second:

1. **Build-time** (`-DARLE_ENABLE_DEEPGEMM_NATIVE`): already default-on via
   `build.rs` auto-detect (`deepgemm_buildable = sm_90 + vendored source + cutlass`).
   The pod binary has 0 stub markers and 5 sm90 symbols — **`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`
   is redundant.** The old `native_bridge=not_compiled reason=build_with_ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`
   logs were from a **stub build** — misleading; it points at a rebuild when the
   real issue is runtime.

2. **Runtime JIT preflight + compile** (the actual failure). On the serve pod:
   - **Preflight** (`deepgemm_preflight_report`) hard-checks `CUDA_HOME/bin/{nvcc,cuobjdump}`
     + the vendored `deepgemm_header` + `cutlass_barrier`. The serve container had
     **no CUDA toolkit** and the binary baked `cutlass_include=/host/arle-build/...`
     (build path), which **doesn't exist on the serve pod** (source is at
     `/root/arle-build/...`). Fix: `CUDA_HOME=/usr/local/cuda` (toolkit available)
     + `ARLE_DEEPGEMM_CUTLASS_INCLUDE=/root/.../flashmla/csrc/cutlass/include`.
   - **Compile** then runs and **fails**: `DSv4 fused wqkv prefill DeepGEMM dense
     failed: CUDA_ERROR_UNKNOWN`, **0 fresh cubins**. Reproduced directly:
     ```
     nvcc -cubin -std=c++20 -ccbin=/usr/bin/gcc-8 ... sm90_fp8_gemm_1d2d.cuh
     → nvcc warning: -std=c++20 not supported with the configured host compiler. Flag ignored.
     → cute/util/type_traits.hpp: error: namespace "std" has no member "conjunction"
     ```
     **The host compiler is gcc-8**, which has no C++20; nvcc silently drops the
     flag and CUTLASS's `std::conjunction`/`disjunction`/`negation` (C++17) fail to
     parse. Every cache-miss kernel fails → `CUDA_ERROR_UNKNOWN` → **silent BF16
     fallback** (`mla_linear` / hand grouped GEMM). This is exactly the failure
     `deepgemm_native.cu:1264-1277` documents (the c++17+`-fconcepts` → c++20 fix
     assumed a c++20-capable host compiler).

The serve container is **Debian 10 / glibc 2.28**; modern g++ (11/13) from
sibling container rootfs need glibc ≥2.32/2.35 and **won't run here**. The arle
binary itself needs GLIBC_2.39 — it runs via a foreign-container `ld-linux` loader
(`/opt/m/ld` interp patch + rootfs `LD_LIBRARY_PATH`), but that doesn't give the
JIT a usable C++20 host compiler.

**Definitive wall (proven 2026-06-30):** I wired g++-13 from a sibling rootfs to
run in this glibc-2.28 container by patching the `.interp` of every toolchain ELF
(`cc1plus`, `cc1`, `as`, `ld`/`ld.bfd`/`ld.gold`, `collect2`, `lto1`, `lto-wrapper`)
to the short `/opt/m/ld` modern loader + a `-B/opt/m/gcc13` ccbin wrapper. A trivial
`std::conjunction_v` C++20 program then compiled+linked **successfully**. But the
real DeepGEMM kernel compile still fails one level deeper:
```
nvcc -ccbin=g++-13 -std=c++20 ... sm90_fp8_gemm_1d2d.cuh
→ /usr/include/c++/13/bits/std_mutex.h: error: identifier "pthread_cond_clockwait" is undefined
→ /usr/include/c++/13/mutex: error: identifier "pthread_mutex_clocklock" is undefined
```
`pthread_cond_clockwait` / `pthread_mutex_clocklock` are **glibc ≥2.30** symbols.
nvcc's CUDA frontend (cudafe++) compiles host code against the **container's
glibc-2.28 system headers** while pulling g++-13's libstdc++ headers — the modern
libstdc++ references pthread functions the 2.28 headers don't declare. **No
compiler-wiring trick fixes this**: the JIT host compile fundamentally needs
glibc ≥2.30 *system headers + runtime*, which a Debian-10 container cannot provide.
The serve box must be a modern-glibc container, or ship a pre-warmed JIT cache.

## Fix

**The real fix is operational, in the serve container, one of:**
1. Install/mount a **g++ ≥10** that this glibc-2.28 container can exec (a static or
   glibc-2.28-compatible build), set `NVCC_CCBIN` to it. Then runtime JIT compiles.
2. **Pre-warm `DG_JIT_CACHE_DIR`** on the build host (which has the right
   compiler), ship the cubins, and ensure the serve binary's generated `code` hash
   matches (same arch + same template params). On a cache hit, `get_or_build_runtime`
   skips `compile_with_nvcc` and only needs `cuobjdump` (toolkit present is enough).
   The pod already had 113 warm cubins from a prior good build — but the current
   binary missed them (code/arch hash drift), so it tried to compile and hit gcc-8.
3. Run the serve in a **modern-glibc container** with a full CUDA toolkit + g++≥12
   (the intended deployment; the binary targets glibc 2.39).

Verified on pod: `CUDA_HOME`+`ARLE_DEEPGEMM_CUTLASS_INCLUDE` clear the preflight
`missing=` list; the remaining failure is purely the gcc-8 C++20 compile.

## Rule

- DeepGEMM is **runtime-JIT**, not build-static. "DeepGEMM enabled" in the build
  log ≠ DeepGEMM running. Confirm at runtime: no `DeepGEMM disabled` warn AND
  fresh cubins land in `DG_JIT_CACHE_DIR` AND the hot kernel in nsys is
  `sm90_fp8_gemm_1d2d_impl`, not `gemv_handwritten`/`dsv4_fp8_gemv`.
- The JIT host compiler must be **g++ ≥10** (C++20). gcc-8 silently drops
  `-std=c++20` → CUTLASS parse errors → `CUDA_ERROR_UNKNOWN` → silent BF16
  fallback. The `native_bridge=not_compiled` / `build_with_ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`
  message is a **stub-only** string and misleads toward a rebuild; the preflight
  `missing=` list and a direct `nvcc ... -std=c++20` repro are the real diagnostics.
- Running a glibc-2.39 binary in a glibc-2.28 container is possible via a foreign
  `ld-linux` interp patch (short `/opt/m/ld` symlink, fits the 27-byte `.interp`)
  + rootfs `LD_LIBRARY_PATH`, host driver `libcuda` winning — but the JIT toolchain
  gap remains.
