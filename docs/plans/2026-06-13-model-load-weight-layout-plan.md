# Model load and weight-layout optimization plan

Date: 2026-06-13

Status: plan only. No code change, benchmark, or default flip is licensed by
this document. Every performance claim below is either explicitly marked as
existing evidence or a hypothesis that needs a same-binary gate.

## Goal

Reduce two separate costs without mixing them:

1. **Cold load cost:** checkpoint discovery, shard I/O, host copies, H2D copies,
   and load-time repacking.
2. **Steady-state weight traffic:** bytes read per token, load width, scale
   layout, padding rows, and duplicate resident buffers.

The target architecture is: load weights once into the execution-native layout
for each kernel band, keep only the resident copies that are actually reachable,
and make every transformation measurable.

## Evidence state

### Solid evidence already in-tree

- DSv4 decode-band FP8 MoE improved B=1 from 36.9 to about 44.1 tok/s by using
  warp-per-row kernels with 16-byte `uint4` weight loads and fused SwiGLU. The
  previous compact scalar GEMV lost because 1-byte loads reached only about 25%
  HBM bandwidth. See
  [`../experience/wins/2026-06-13-dsv4-fp8-decode-moe-lane-44.md`](../experience/wins/2026-06-13-dsv4-fp8-decode-moe-lane-44.md)
  and
  [`../experience/errors/2026-06-13-dsv4-decode-gemv-lane-bandwidth-kill.md`](../experience/errors/2026-06-13-dsv4-decode-gemv-lane-bandwidth-kill.md).
- DSv4 decode alignment is shape-banded: 128-aligned packing was correct but
  too expensive for B=1 decode; 64-align recovered about 9% by halving pad rows.
  See
  [`../experience/wins/2026-06-13-dsv4-decode-band-64align-regression-fix.md`](../experience/wins/2026-06-13-dsv4-decode-band-64align-regression-fix.md).
- Metal already follows the mmap/lazy-load direction: `infer-metal` loads
  safetensors through MLX mmap, load-time transposes dense weights, and merges
  Qwen3.5/3.6 projections such as `qkvz`, `ba`, and `gate_up` where the layout
  permits it. Entry points:
  [`../../crates/infer-metal/src/loader.rs`](../../crates/infer-metal/src/loader.rs),
  [`../../crates/infer-metal/src/qwen35.rs`](../../crates/infer-metal/src/qwen35.rs),
  [`../../crates/infer-metal/src/weights.rs`](../../crates/infer-metal/src/weights.rs).
- CUDA DSv4 already builds resident DeepGEMM FP8 caches with kernel-native
  weight and scale layout. Entry points:
  [`../../crates/infer-cuda/src/loader.rs`](../../crates/infer-cuda/src/loader.rs),
  [`../../crates/cuda-kernels/src/tensor.rs`](../../crates/cuda-kernels/src/tensor.rs).
- CUDA Qwen3.5/3.6 real-checkpoint load has a rough observed signal: 67 GB loads
  in about 85 s. That run was not a correctness pass, so it is a load-speed clue
  only, not a runtime-quality baseline. See
  [`../experience/errors/2026-06-11-qwen35-cuda-rewrite-35b-degenerate-output.md`](../experience/errors/2026-06-11-qwen35-cuda-rewrite-35b-degenerate-output.md).

### Hypotheses to license or kill

- CUDA shard mmap will reduce cold load RSS and host memcpy. It is not proven
  until per-shard read/copy/H2D attribution exists.
- Dense/Qwen CUDA load-time `qkv` and `gate_up` fusion should reduce GEMM count
  and weight reads, mirroring the vLLM pattern already surveyed in
  [`../experience/wins/2026-05-07-m_pf-fuse-hypothesis-validated-vllm-source-confirms.md`](../experience/wins/2026-05-07-m_pf-fuse-hypothesis-validated-vllm-source-confirms.md).
  It still needs same-binary ARLE A/B because this repo's model mix and graph
  capture path differ from vLLM.
- Persistent prepacked sidecar artifacts should reduce repeat cold-load time for
  expensive DeepGEMM/quant layouts. This is a storage-format feature, not a
  default-worthy perf claim until invalidation and corruption gates pass.

## Current code map

### CUDA

- Production constructor:
  [`../../crates/infer-api/src/loaded.rs`](../../crates/infer-api/src/loaded.rs)
  `build_cuda_engine`.
- Safetensors loader:
  [`../../crates/infer-cuda/src/loader.rs`](../../crates/infer-cuda/src/loader.rs)
  `SafetensorLoader`.
  - It uses `model.safetensors.index.json` when available.
  - It caches each shard once as `Rc<Vec<u8>>`.
  - The generic `load_raw_from_shard` path copies tensor bytes with `to_vec()`.
  - The Qwen3.5 stacked MoE path already borrows from cached shard bytes to avoid
    about 1.5 GiB host memcpy per MoE layer.
- Dense BF16 upload:
  [`../../crates/cuda-kernels/src/tensor.rs`](../../crates/cuda-kernels/src/tensor.rs)
  `DeviceMatrix::from_safetensors` and `DeviceVec::from_safetensors`.
- DSv4 resident layout:
  `Dsv4Fp8DeepGemmWeightCache` stores row-major FP8 weights and FP32 128x128
  block scales in DeepGEMM's expected layout.

### Metal

- Production constructor:
  [`../../crates/infer-api/src/loaded.rs`](../../crates/infer-api/src/loaded.rs)
  `metal_serve_handle`.
- Resource plan runs before weights are constructed and can clamp scheduler
  capacity.
- MLX safetensors loading is mmap-backed; dense weights are transposed/evaluated
  at load time; quantized weights keep `{weight, scales, biases}` without
  dequantizing hot projections.

## Industry pattern distilled

Treat this as design input, not ARLE evidence:

- **Mmap/lazy checkpoint access:** keep file bytes file-backed when possible;
  switch to eager/prefetch only for hostile remote filesystems after measuring.
- **Load-time fusion:** concatenate HF checkpoint tensors into the projection the
  kernel will actually run, e.g. `qkv` and `gate_up`, rather than launching
  separate matmuls in the hot path.
- **Kernel-native resident layout:** prepack scales, tile order, expert grouping,
  and row alignment to the serving kernel's ABI. Do not transpose or concatenate
  in the token loop.
- **Band-specific layout:** prefill and decode can need different alignments or
  kernels. A layout that is correct and fast for prefill can be wrong for B=1
  decode.
- **Persistent build artifacts:** expensive deterministic conversions can be
  cached by checkpoint hash, layout version, GPU arch, dtype, and code version.

## Non-goals

- No PyTorch or Python on the hot path.
- No default flip based on source survey alone.
- No single change that mixes loader I/O, projection fusion, quant format, and
  kernel replacement.
- No deletion of fallback resident copies until the fallback is proven
  unreachable or explicitly made opt-in.
- No comparing cold first request against a warm SGLang/vLLM baseline.

## Execution DAG

```text
P0 load attribution
  |
  +-- P1 CUDA loader I/O and copy hygiene
  |      |
  |      +-- P4 persistent prepacked sidecar
  |
  +-- P2 load-time fused/resident layouts
         |
         +-- P3 band-specific vectorized decode layouts
```

P0 is the gate for every later phase. P1 and P2 may run in parallel only after
P0 names their independent bottlenecks. P4 must not start before at least one P1
or P2 conversion is measurable and deterministic.

## P0. Load attribution first

Question: where does cold load time and memory actually go?

Tasks:

1. Add a lightweight load trace around the CUDA and Metal constructors.
2. In CUDA `SafetensorLoader`, record:
   - shard open/read or mmap ms
   - shard bytes
   - header parse ms
   - owned tensor copy bytes/ms
   - borrowed tensor bytes
   - H2D bytes/ms
   - GPU repack/cache build ms
3. In Metal, record:
   - MLX safetensors load ms
   - load-time transpose/eval ms
   - projection merge ms
   - resource-plan result and effective scheduler clamp
4. Emit one rank-0 summary table at load end. Sums must reconcile with wall
   clock within a small, stated error bound.

Gate:

- Same binary, same checkpoint, two cold boots if practical.
- Log table includes total load ms, peak host RSS if available, and device
  resident bytes by category.
- No performance optimization is claimed from P0 itself.

## P1. CUDA loader I/O and copy hygiene

Question: can cold-load host I/O and host copies be reduced without changing
resident semantics?

Tasks:

1. Replace `Rc<Vec<u8>>` shard cache with an mmap-backed shard object. There is
   already workspace precedent in `crates/train/src/qwen35_loader.rs` and
   `crates/infer-gguf/src/gguf.rs`.
2. Split `OwnedTensor` and borrowed tensor consumers so paths that immediately
   H2D-copy do not first `to_vec()` the tensor bytes.
3. Keep an explicit eager-read mode if P0 shows mmap hurts a target filesystem.
4. Only consider pinned staging buffers after P0 shows H2D, not disk or host
   copy, is the load bottleneck.

Gate:

- Cold-load A/B: load ms, peak RSS, total H2D bytes, tensor-copy bytes.
- Correctness gate for the model family being loaded.
- No steady-state speed claim unless P1 also changes resident layout, which it
  should not.

## P2. Load-time fused and resident layouts

Question: are weights resident in the layout the hot kernel actually consumes?

Tasks:

1. Dense/Qwen CUDA:
   - build `qkv` and `gate_up` resident weights at load time where model config
     and TP sharding make the concat unambiguous
   - remove the corresponding separate matmul launches only after correctness and
     perf gates pass
2. Qwen3.5/3.6 MoE CUDA:
   - keep the load-time build-and-replace rule for `fused_sglang` and DeepGEMM
     caches
   - do not allow first-forward lazy restack to allocate a second 1.5 GiB/layer
     copy
3. DSv4 CUDA:
   - preserve DeepGEMM FP8 resident caches
   - audit raw source `DeviceMatrix` copies and free or gate them only when all
     fallback users are named
   - keep expert scales as true FP32 where the checkpoint class requires it

Gate:

- Needle or model-specific correct-inference gate before throughput.
- Same-binary A/B for TTFT, TPOT, tok/s, and load ms.
- Nsys/ncu attribution must show either fewer launches, fewer resident bytes, or
  higher achieved memory bandwidth for the target lane.

## P3. Band-specific vectorized decode layouts

Question: can decode read each needed weight exactly once at wide load width?

Tasks:

1. Use the DSv4 FP8 decode MoE result as the template: warp-per-row, vectorized
   `uint4` loads, fused epilogue, and no pad rows in the decode band.
2. Treat alignment as a per-band parameter, not a global constant.
3. For every dequant or quantized decode kernel, record:
   - bytes per route/token
   - achieved bandwidth
   - scale encoding and scale load width
   - pad rows versus real rows
4. Prefer adopting official/vendored kernels where an equivalent SGLang/vLLM or
   vendor implementation exists; hand kernels need an achieved-bandwidth win.

Gate:

- Decode B=1 and at least one multi-request shape.
- Correctness gate before perf.
- A losing compact path is killed or kept opt-in with an errors entry.

## P4. Persistent prepacked sidecar

Question: can repeat startup skip deterministic repack work safely?

Tasks:

1. Define a sidecar manifest keyed by:
   - checkpoint file list and hashes
   - ARLE commit or layout schema version
   - CUDA arch / Metal family
   - dtype and quantization config
   - kernel layout version
2. Cache only deterministic layout transforms, e.g. fused rows, transposed dense
   matrices, DeepGEMM FP8 weight/scale layout, or expert grouped buffers.
3. Make sidecar load optional and fail-closed: bad hash, version mismatch, or
   partial file means rebuild from checkpoint.
4. Add corruption and stale-version tests before enabling by default.

Gate:

- Cold-load A/B with and without sidecar.
- Resident bytes and correctness identical to the non-sidecar build.
- Sidecar invalidation tests pass.

## Verification protocol

- Correctness first: use the family-appropriate needle / same-config /
  self-consistency gate, not token-exact-vs-baseline when MoE nondeterminism is
  a known confound.
- Wall-clock first: every perf claim includes total request metrics, not only a
  narrow kernel-window win.
- Same-binary A/B: same shell, same checkpoint, same flags, only one variable
  changed.
- Every runtime diff under `crates/infer-*`, `crates/cuda-kernels`, or
  `crates/mlx-sys` needs a wins/errors entry per the benchmark contract.
- Load optimizations and serving optimizations are reported separately:
  `load_ms` does not license TTFT/TPOT, and TTFT/TPOT does not explain load.

## Review checklist

- [ ] Does the phase change only one variable?
- [ ] Are all mutated resident buffers named?
- [ ] Are source checkpoint bytes, transformed resident bytes, and fallback
      copies accounted separately?
- [ ] Is the scale encoding tensor-class-specific rather than assumed model-wide?
- [ ] Does the plan avoid deleted pre-rewrite paths?
- [ ] Is every claim labeled evidence or hypothesis?
- [ ] Is there a kill condition before implementation expands?

## Self-review result

- Stale path check passed: this plan does not use deleted `infer/src/**`,
  `-p infer`, or the old `metal_serve` binary. `metal_serve_handle` is the
  current function name in `infer-api`.
- Evidence/hypothesis split is explicit: existing wins/errors are treated as
  evidence; industry patterns and future optimizations are design inputs until
  same-binary gates run.
- Confounder control is explicit: P0 is observability-only; P1, P2, P3, and P4
  each change a different variable class and have separate gates.
- Load speed and serving speed are separated: `load_ms` cannot license TTFT/TPOT,
  and TTFT/TPOT cannot explain loader cost.
- Remaining uncertainty is intentional: exact P1/P2 ordering is deferred until
  P0 attribution names the real bottleneck.

## Immediate next step

Implement P0 only. If the load trace shows host copy or disk I/O dominates,
start P1. If GPU repack/cache build dominates, start P2 or P4. If load is not a
meaningful user-facing bottleneck, skip loader I/O work and spend the next window
on P3-style steady-state weight bandwidth.
