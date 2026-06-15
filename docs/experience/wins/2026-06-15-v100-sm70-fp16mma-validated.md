# V100 sm_70 TileLang attention fixed via fp16-MMA — validated on real Volta

## Context

v0.2.0 shipped with the V100 (sm_70) release binary **red**: build.rs's TileLang
AOT step failed to compile the paged attention kernels for sm_70 with
`Layout infer conflict between m_prev and scale_i in T.Parallel loop`. Root: Volta
has no bf16 tensor cores, so the online-softmax rescale's `scale_i` could not get a
fragment layout reconcilable with both the QK scores fragment and the PV `acc_o`
fragment — structural to every prefill/decode kernel with online-softmax + a PV
`T.gemm`. The earlier in-tree workarounds (the `scripts/sm70_tilelang.patch`
cuda.fma scalar fallback + an `SM70_FORCE_TWO_KV_TILES` band-aid in hd256 prefill)
cleared an *earlier* conflict but not this one. v0.2.1 needed the V100 lane green.

## What Worked

**fp16-MMA approach** — applied to all 4 attention kernels
(`batch_{prefill,decode}_paged_hd{128,256}.py`):

- I/O tensors stay **bf16** (runtime ABI unchanged). On `sm < 80` only, feed the
  GEMM **operands** as fp16 (`gemm_dtype = "float16" if sm_arch < 80 else
  "bfloat16"`, `sm_arch` from env `ARLE_TILELANG_CUDA_ARCH`, which
  `gen_tilelang_aot.py` sets from `--cuda-arch`). fp16 dispatch selects
  **kCudaMMA → stock `GemmMMASm70`** (`mma.sync`, Volta's native fp16 tensor
  cores), avoiding the scalar kCudaFMA path that produced the layout conflict.
- bf16→fp16 routes through f32 (`T.cast(T.cast(x, accum_dtype), gemm_dtype)`) —
  direct bf16→fp16 is an ambiguous CUTLASS conversion nvcc rejects.
- PV gemm operand A goes through SHARED memory (`p_shared`) on `sm < 80`; `sm ≥ 80`
  keeps the byte-identical bf16 fragment `p_bf16` else-branch.
- The obsolete `SM70_FORCE_TWO_KV_TILES` band-aid is removed (docstring note only).
- `gen_tilelang_aot.parse_target` hardened: from-source tilelang exposes TVM as
  top-level `tvm` (after `import tilelang`), not `tilelang.tvm` — try the submodule
  then fall back; and construct the Target via dict form
  `{"kind":"cuda","arch":"sm_70"}` (the CLI string form is rejected by from-source
  tilelang).

**sm ≥ 80 is byte-identical by construction** (the default sm_90/sm_89 production
path): `to_gemm(x)` returns `x` unchanged when `gemm_dtype == dtype`, shared tiles
alloc with the original dtype, and PV uses the `p_bf16` branch. The traced TIR is
provably identical — no A/B needed for the default path.

**Validation — real Tesla V100-SXM2-32GB (cap 7.0), CUDA 12.4, from-source
tilelang 0.1.11.** TileLang JIT numerical test (JIT-compile each kernel for
`arch=sm_70` with `ARLE_TILELANG_CUDA_ARCH=70`, feed random paged K/V + Q, compare
against a torch f32 attention reference):

| kernel | config | cosine | max_rel_err |
|---|---|---|---|
| prefill hd128 | q32/kv8 | 0.999999 | 1.37e-3 |
| decode  hd128 | q32/kv8 | 0.999999 | 2.69e-3 |
| prefill hd256 | q16/kv4 | 0.999999 | 1.38e-3 |
| decode  hd256 | q16/kv4 | 0.999999 | 2.66e-3 |

Errors are exactly the fp16-operand + f32-accum tolerance. The full `arle` sm_70
binary also builds green on the box. Committed bytes are md5-identical to the
validated artifact.

**Why a kernel-level numerical gate, not a model-level needle/guidellm bench:**
Qwen3.5-4B on V100 uses **native CUDA C** attention (`qwen35.rs:2190`), not these
TileLang kernels. No cached model both routes to the TileLang paged-attention path
(R6 / Qwen3.6) *and* fits in 32 GB. So the correct-inference gate (§0 SOLID) is met
at the kernel level — a serve-throughput guidellm run is not applicable here
because no V100-fitting model exercises these kernels. This is a **correctness**
fix (V100 build unblock), not a perf optimization; correctness is the relevant
gate, and it is met with runtime numerical evidence on real Volta. (`gen_tilelang_aot.py`
is build-time AOT tooling — bench-exempt.)

## Rule

For Volta (`sm < 80`) bf16 TileLang attention: feed GEMM **operands** as fp16 so
dispatch hits the stock `GemmMMASm70` tensor-core path; never depend on the scalar
fma fallback for online-softmax + PV-gemm kernels. Keep `sm ≥ 80` byte-identical by
gating only the operand dtype. When no available model routes to a kernel on the
target HW, the correct-inference gate is a **kernel-level numerical test on real
hardware** — compile-green is not the gate.
