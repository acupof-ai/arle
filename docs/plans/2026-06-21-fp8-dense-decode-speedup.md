# 2026-06-21 · Qwen3.6-27B-FP8 dense decode speedup

Target: the dense 27B-FP8 decode path (`infer-cuda` continuous-batching engine),
the bottleneck behind the rubric-OPD rollout + eval loop. Observed ~5-7 tok/s;
HBM roofline says ~100-150 tok/s should be reachable. Single-GPU, 8×H20
(sm_90a, CUDA 12.9).

**Status: investigation + spec. No kernel landed** — the #1 lever is a `.cu`
change that cannot be compiled or A/B'd locally (no nvcc on the Mac), so it is
written as a line-level spec for pod execution, not implemented blind. The one
safe non-GPU change (a routing/gate refactor) is also spec'd; see §4.

---

## 1. Root cause (file:line + quoted source)

The WARN in the symptom is **real but a red herring for decode**. There are two
separate facts; the decode bottleneck is the second.

### 1a. The WARN: native DeepGEMM is not compiled into this binary (build-flag stub)

`crates/infer-cuda/src/ops/quant_linear.rs:78-95` gates the Qwen FP8 dense
DeepGEMM path on a **runtime preflight**:

```rust
fn qwen_fp8_deepgemm_dense_enabled() -> bool {
    ...
    match cuda_moe::dsv4_deepgemm_native_preflight() {
        Ok(_) => true,
        Err(err) => {
            log::warn!("Qwen FP8 dense DeepGEMM disabled: native bridge unavailable ({err})");
            false
        }
    }
}
```

The preflight resolves to `dsv4_deepgemm_native_preflight_cuda`. The binary in
this run links the **stub** TU
(`crates/cuda-kernels/csrc/gemm/deepgemm_bridge_stub.cu:6-19`):

```cpp
#ifndef ARLE_ENABLE_DEEPGEMM_NATIVE
extern "C" CUresult dsv4_deepgemm_native_preflight_cuda(char* out, size_t out_len) {
  static constexpr const char* kMessage =
      "status=failed native_bridge=not_compiled "
      "reason=build_with_ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1";
  ...
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif
```

The `-DARLE_ENABLE_DEEPGEMM_NATIVE=1` define is only added when
`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` (or the `_TORCH` alias) is set at build time
(`crates/cuda-kernels/build.rs:2160-2161,2226-2228`). So this binary = stub →
`CUDA_ERROR_NOT_SUPPORTED` → the exact WARN string. Cause = **(a) build-flag
stub linked**, confirmed by the literal `native_bridge=not_compiled
reason=build_with_ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` in the message; not a real
arch-unsupported and not a preflight bug.

Even when the flag *is* set, the native path is a **runtime JIT** that needs the
full DeepGEMM source tree + nvcc + cuobjdump + cutlass headers present at run
time (`deepgemm_native.cu:353-408` checks `deepgemm_include`, `deepgemm_header`,
`cutlass_include`, `nvcc`, `cuobjdump`). `vendor/deepgemm/` **does not exist in
this checkout** (`vendor/` holds only `llama.cpp`, `mlx-sys`), so the native path
is a pod-only capability. This is *not* "just rebuild and it's fixed".

### 1b. The real decode bottleneck: DeepGEMM dense is gated OFF for decode regardless

The dense DeepGEMM path is gated by `fp8_deepgemm_dense_shape`
(`quant_linear.rs:164-172`):

```rust
fn fp8_deepgemm_dense_shape(weight: &DeviceMatrix, seq_len: usize) -> bool {
    weight.weight_format == WeightFormat::Fp8BlockScaled
        && seq_len >= QWEN_FP8_DEEPGEMM_DENSE_MIN_M   // = 1024
        ...
        && qwen_fp8_deepgemm_dense_enabled()
}
```

`QWEN_FP8_DEEPGEMM_DENSE_MIN_M = 1024` (`quant_linear.rs:12`). In the decode
dispatch, `seq_len` is the **number of decode rows in the continuous batch (B)**,
not a sequence axis (`ops.rs:175-176` → `quant_linear::gemm_batch`; the gemm
ensures `weight.cols == x.hidden_dim`, so `x.seq_len` = batch rows). For a single
rollout/eval stream B = 1 (a few at most). **So decode never satisfies
`seq_len >= 1024` — even on a pod with native DeepGEMM compiled and warmed,
decode FP8 dense GEMM never takes the WGMMA path.** DeepGEMM is the *prefill/warm*
lever (`warm_fp8_deepgemm_dense_prefill` warms at M = min(max_seq,2048) ≥ 1024,
`qwen35.rs:1141-1145`); decode is structurally excluded.

Decode therefore always lands on the fallback at `quant_linear.rs:372-408`
(`gemm_batch`, `Fp8BlockScaled | Fp8PerShard` arm) →
`ffi::gemv_fp8_block_scaled_batch_cuda` →
`fp8_f32_block_gemv_batch_kernel` (`csrc/gemm/quantized_gemv.cu:791-845`,
launched at `:3141-3156`).

That kernel is the wall. It is a **GEMV-shaped, CUDA-core, no-tensor-core**
kernel:

- grid `(N/GEMV_ROWS, B)` = `(N/4, B)`, block 256 threads
  (`quantized_gemv.cu:3151-3154`, `GEMV_ROWS=4`, `GEMV_THREADS=256` at `:46-47`).
- each thread strip-mines K, dequantizing FP8E4M3→f32 via `fp8_f32_dot16`
  (`:297-310`) and accumulating with **scalar FMA in f32** — tensor pipe = 0,
  weight reuse across the (tiny) B dimension = 0 (each batch row is a separate
  `blockIdx.y` that re-reads the full weight row).

This is the same kernel *family* ncu profiled on the DSv4 side at **<1% tensor
pipe, ~10% HBM BW** (`docs/experience/errors/2026-06-05-fp8-linear-per-projection-deepgemm-no-win.md`).
For a 27 GB dense model, the HBM-roofline B=1 floor is ~27 GB / ~4 TB/s ≈ 6.8
ms/token (~148 tok/s); the observed ~150-200 ms/token (~5-7 tok/s) is **~25-30×
off the roofline**, consistent with a kernel that touches HBM at ~10% efficiency
and burns the rest on scalar decode + poor occupancy at low B.

### 1c. The Qwen f32-block FP8 path has NO fast variant; DSv4 does

The DSv4 FP8 format (E8M0 scales) already has both a **batched tiled** kernel
(`dsv4_fp8_gemv_batch_tiled_kernel`, `DSV4_BATCH_TILE=32`,
`quantized_gemv.cu:482`) and a **tensor-core MMA** kernel
(`dsv4_fp8_gemv_batch_mma_kernel` / `dsv4_fp8_gemv_batch_mma_launch`,
`csrc/gemm/quantized_gemv_mma.cu`, `mma.m16n8k16` BF16×BF16→FP32 after FP8→BF16
dequant). The **Qwen `Fp8BlockScaled` (f32 block-scale) path has neither** — it
only routes to the naive `fp8_f32_block_gemv_batch_kernel`. The fast kernels were
written for the DSv4 scale layout and never ported to the Qwen f32-block layout.

### 1d. Training shares the same scalar pattern (point 4 — confirmed)

The autograd CE forward FP8 GEMM is `fp8_block_scaled_matmul_bt_f32`
(`crates/autograd/src/backend_cuda/kernels/fp8_block_scaled.cu:29-62`, dispatched
from `backend_cuda.rs:888-936` `matmul_bt_device_f32_fp8_block_scaled`). It is
**even worse than the inference fallback**: one CUDA block per *single output
element* `(m, n)` (`grid = (M, N)`), scalar dequant-to-f32, blockDim reduction
over K, no tensor cores, no vectorized load. The same f32-block FP8 layout. So a
single optimized f32-block FP8 dense GEMM (tensor-core, fused) would lift **both**
the inference decode wall and the training CE forward — they are the same
operator with two scalar implementations.

---

## 2. Ranked fix options (impact × tractability)

Ranking is for the **decode** wall (the actual bottleneck). DeepGEMM-native
rebuild is listed but is a prefill lever, not a decode lever.

### Option A (#1) — port the tensor-core MMA GEMV kernel to the Qwen f32-block FP8 layout

**Impact: high. Tractability: medium (well-defined port, not a new kernel).**

The DSv4 MMA kernel `dsv4_fp8_gemv_batch_mma_kernel`
(`quantized_gemv_mma.cu`) already does exactly the structural thing we need:
FP8→BF16 dequant in registers, then `mma.m16n8k16` BF16×BF16→FP32 tensor-core
mainloop, batched across B. The *only* DSv4-specific piece is the scale decode
(`gemv_mma_decode_fp8_e4m3` + E8M0 `dsv4_decode_e8m0`). Swap that for the Qwen
f32-block scale lookup (`fp8_f32_block_scale`, `quantized_gemv.cu:281-295`) and
the kernel applies to `Fp8BlockScaled`.

- **Change site:** new `fp8_f32_block_gemv_batch_mma_kernel` in
  `csrc/gemm/quantized_gemv_mma.cu` (clone of `dsv4_fp8_gemv_batch_mma_kernel`,
  scale decode = `fp8_f32_block_scale`); new extern
  `gemv_fp8_block_scaled_batch_mma_launch` next to the DSv4 launch at
  `quantized_gemv_mma.cu:299`; FFI decl in `crates/cuda-kernels/src/ffi/gemm.rs`
  (alongside `dsv4_fp8_gemv_batch_mma_launch:261`); dispatch in
  `quant_linear.rs:372-408` (`Fp8BlockScaled | Fp8PerShard` arm of `gemm_batch`)
  behind a new `qwen_fp8_mma_gemv_enabled()` env gate.
- **Gating / opt-in:** default OFF via env (`ARLE_QWEN35_FP8_MMA=1`); the existing
  `fp8_f32_block_gemv_batch_kernel` stays the default until the A/B licenses the
  flip. Byte-for-byte baseline preserved.
- **Mechanism:** moves the dot product from scalar CUDA-core FMA (<1% tensor) to
  WGMMA/MMA tensor pipe; weight bytes are still HBM-bound but the math overlaps
  and the kernel can be tuned to higher BW efficiency than the GEMV strip-mine.
  Expected to close most of the ~25-30× roofline gap that is *not* pure HBM.
- **Correctness gate:** `scripts/needle_gate.py` ×3 same-config vs the scalar
  baseline envelope + per-projection max_abs (the DSv4 MMA port hit cosine
  0.999999 / max_abs 0.07-0.13, the bf16/quant noise floor). NOT byte-identity
  (MoE/atomic non-determinism is N/A for dense but the correct-inference gate is
  the project standard).
- **License measurement:** `ncu` on the new kernel (tensor-pipe % up, BW % up)
  + decode tok/s same-binary A/B (`ARLE_QWEN35_FP8_MMA=0` vs `=1`) on the 27B-FP8
  rollout shape, per `docs/bench-and-trace-spec.md`. Must clear decode tok/s
  before any default flip.
- **Caveat to verify on the pod:** at B=1 the op is HBM-bound; a tensor-core
  kernel only wins if the scalar kernel was leaving BW on the table (ncu says
  ~10% BW — so yes, headroom exists). If ncu shows the MMA kernel also stuck at
  ~10% BW, the lever is occupancy/launch-shape, not the math pipe — re-decompose
  before claiming the win.

### Option B — lower the DeepGEMM dense `MIN_M` threshold so larger decode batches use WGMMA

**Impact: medium (only helps when B is large). Tractability: low (needs the pod
native bridge + the prior no-win caveat).**

`QWEN_FP8_DEEPGEMM_DENSE_MIN_M=1024` excludes all realistic decode B. Lowering it
would route batched decode to `dsv4_deepgemm_fp8_gemm_nt`. But:
(1) requires the native bridge compiled + DeepGEMM source on the box (§1a);
(2) the DSv4 per-projection DeepGEMM experiment shipped **no wall-clock win / a
regression** because per-call launch + per-call `pack_quantize` of the activation
dominated (`errors/2026-06-05-fp8-linear-per-projection-deepgemm-no-win.md`). The
dense Qwen path has the same call-count structure (5 projections × N layers). So
this is only viable *with* activation-quantize-once fusion (Option D) and a large
enough B to amortize, which the rollout/eval workload (B≈1) does not have.
**Not the decode lever.** Keep for the throughput/batched-eval regime, gated and
A/B'd separately.

### Option C — optimize the fallback `fp8_f32_block_gemv_batch_kernel` in place

**Impact: low-medium. Tractability: high.**

Tune the existing scalar kernel: borrow the `DSV4_BATCH_TILE` batched-tile pattern
(`quantized_gemv.cu:482`, accumulate multiple B rows per weight load → weight
reuse across batch) and revisit `GEMV_ROWS`/occupancy. This is the
"micro-kernel knob" class the kernel review repeatedly KILLs on tiny operators
(`cuda-kernels/AGENTS.md` distilled lessons; `errors/2026-05-16-p3-*`), but the
27B dense GEMM is **not** tiny (large N,K), so a batched-tile rewrite that adds
real weight reuse could be a genuine, not sub-noise, win — *if* B > 1. At B=1
there is no reuse to exploit and this collapses to the current kernel. Lower
ceiling than Option A (still no tensor cores). Worth it only as a fallback if the
MMA port stalls.

### Option D — fuse same-input projections + quantize activation once (call-form fix)

**Impact: high (this is SGLang's actual advantage). Tractability: high effort
(forward restructure, not a kernel swap).**

Per the 2026-06-05 errors entry, the upstream (SGLang 4.94 ms/token) win is the
*fused/batched call form*: fuse q/k/v (and gate/up) into one GEMM, BF16→FP8
quantize the shared activation **once**, and issue fewer/larger GEMMs. The Qwen
decode forward currently issues separate `gemm_batch` per projection
(`qwen35.rs:3376-3380` MLP gate/up/down; `:3438-3440` q/k/v). Fusing q+k+v and
gate+up into single weight matrices (concat at load) cuts launch + activation-
quant overhead and feeds a larger-M GEMM that a tensor-core kernel amortizes.
This is the highest-ceiling lever but it is an architecture change to the model
forward + the loader (concat projection weights), so it trails Option A as the
first move. It also *amplifies* Option A (a fused larger GEMM gives the MMA kernel
more to chew on). Sequence: A first (kernel pipe), then D (call form) if A alone
doesn't close the gap.

### Option E (not a decode fix) — rebuild with `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` + vendored DeepGEMM

Removes the WARN and enables the **prefill/warm** WGMMA path (M ≥ 1024). Helps
TTFT on long prompts, **does nothing for decode tok/s** (§1b). Requires
`vendor/deepgemm` present + nvcc/cuobjdump/cutlass at runtime (§1a). Do this on
the pod for prefill, but do not expect it to move the rollout/eval decode wall.

---

## 3. #1 recommended lever

**Port the existing DSv4 tensor-core MMA FP8 GEMV kernel
(`dsv4_fp8_gemv_batch_mma_kernel`) to the Qwen f32-block-scale layout and route
the `Fp8BlockScaled` decode path to it behind `ARLE_QWEN35_FP8_MMA=1`** — moving
the dense FP8 decode dot product off the <1%-tensor, ~10%-BW scalar
`fp8_f32_block_gemv_batch_kernel` and onto the WGMMA/MMA tensor pipe that the
DSv4 path already validated at cosine 0.999999, which is expected to close most of
the ~25-30× decode roofline gap that is not pure HBM bandwidth.

It is the highest impact × tractability because the kernel already exists and is
correctness-proven on a sibling FP8 format — the port is a scale-decode swap, not
a new kernel — and the same swap simultaneously fixes the training CE forward
(§1d), which shares the f32-block FP8 layout. (Option D fusion is the higher
ceiling but a forward+loader restructure; do it second if A alone falls short.)

---

## 4. What was / was not implemented here

**Not implemented as a kernel.** The #1 lever is a `.cu`/FFI change that requires
nvcc to compile and a GPU to A/B; nvcc is unavailable locally and the GPU is
owned by a training run. Writing the MMA kernel blind and committing it would
violate "no speculative kernel code that can't be checked" — it is left as the
line-level spec in §2 Option A for pod execution.

**Safe local refactor available but deferred (flagged, not landed):** the decode
gate could be made explicit so the WARN does not fire per layer on every decode
(it currently logs because `qwen_fp8_deepgemm_dense_enabled()` is a `OnceLock`, so
it actually only warns **once** — verified at `quant_linear.rs:79-80`; the
per-layer spam in the symptom is the *prefill warm* loop, not decode). No code
change is warranted for the WARN itself — it is correct and one-shot. The only
load-bearing change is Option A, which must be done on the pod. Therefore **no
code was modified and no typecheck was run** (nothing to check); this deliverable
is the spec.

---

## 5. Files referenced (all absolute under repo root)

- `crates/infer-cuda/src/ops/quant_linear.rs` — dispatch + gates (`:12,78-95,164-172,372-408`)
- `crates/infer-cuda/src/ops.rs` — `gemm_batch` → `quant_linear` (`:175-176`)
- `crates/cuda-kernels/csrc/gemm/deepgemm_bridge_stub.cu` — NOT_SUPPORTED stub (`:6-19`)
- `crates/cuda-kernels/csrc/gemm/deepgemm_native.cu` — JIT preflight (`:353-408,1755-1766`)
- `crates/cuda-kernels/build.rs` — `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE` define (`:2160-2161,2226-2228`)
- `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu` — scalar fallback (`:791-845,3141-3156`); DSv4 tiled (`:482`)
- `crates/cuda-kernels/csrc/gemm/quantized_gemv_mma.cu` — DSv4 tensor-core MMA GEMV (port source)
- `crates/cuda-kernels/src/ffi/gemm.rs` — FFI decls (`:261,301`)
- `crates/autograd/src/backend_cuda/kernels/fp8_block_scaled.cu` — training scalar FP8 GEMM (`:29-62`)
- `crates/infer-cuda/src/qwen35.rs` — warm-prefill + decode call sites (`:1141,3376-3440`)
- `docs/experience/errors/2026-06-05-fp8-linear-per-projection-deepgemm-no-win.md` — prior art (DeepGEMM-per-projection no-win)
