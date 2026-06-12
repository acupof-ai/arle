# Triton AOT Integration

Build-time AOT for CUDA kernels written as `@triton.jit`. `gen_triton_aot.py`
programmatically compiles a kernel to a cubin and emits a self-contained C
launch wrapper (cubin bytes embedded, returns `CUresult`). The C wrappers are
compiled into `libtriton_kernels_aot.a` and linked alongside the native CUDA C
and TileLang kernels.

## What this covers

- The three SGLang GDN decode kernels vendored under `kernels/`:
  `arle_gdn_fused_recurrent.py`, `arle_gdn_conv1d_update.py`,
  `arle_gdn_rms_norm_gated.py`. Each file header cites its SGLang source and
  lists every deviation; the kernel math is byte-equal to upstream.
- Opt-in behind `ARLE_QWEN35_SGL_GDN` at runtime (default OFF — the hand
  kernels remain the default decode arm).
- Hopper-only for now (sm_90). On any other SM target, on `KernelSet::Dsv4Flash`,
  or when no triton-capable Python is found, the build links
  `CUDA_ERROR_NOT_SUPPORTED` stubs and prints a `cargo:warning` — the lane is
  optional, so Mac / CI builds stay green without triton.

## Prerequisites

Triton must be importable by the build's Python. Point the build at an
interpreter that has triton installed:

```bash
export INFER_TRITON_PYTHON=/path/to/venv/bin/python   # must `import triton`
```

If `INFER_TRITON_PYTHON` is unset, the build probes `python3` then `python`.
If none import triton, the build falls back to stubs (non-fatal).

Pinned/verified on the pod with triton 3.5.1. The programmatic compile uses
`GPUTarget("cuda", 90, 32)` (int arch — triton's CLI `--target` path has a
str-vs-int bug), `ASTSource` + `make_backend` + `parse_options`, and asserts
zero global/profile scratch before emitting the launch wrapper.

## Build

```bash
CUDA_HOME=/usr/local/cuda INFER_TRITON_PYTHON=$VENV/bin/python \
  cargo build --release --features cuda
```

Artifacts land under
`target/release/build/cuda-kernels-*/out/triton_aot/<artifact>/`. The generated
C embeds the cubin via `cuModuleLoadData`, so the binary is self-contained and
survives `cargo clean` / relocation.

## Adding a kernel

1. Drop the `@triton.jit` kernel under `kernels/` (header: source + deviations).
2. Add a `TRITON_AOT_KERNELS` entry in `build.rs` (artifact dir, kernel path,
   jit fn name, out_name base, signature string, num_warps, num_stages, public
   C decl, call args).
3. Add the matching FFI extern (`<base>_cuda` + `<base>_load_cuda`) in
   `crates/cuda-kernels/src/ffi/triton.rs`.

The signature grammar mirrors triton's `tools/compile.py`: `*<dtype>[:N]` =
pointer (`:N` divisibility hint), a bare scalar dtype (`fp32`/`i32`) = runtime
scalar, a bare int/float literal or `"string"` = baked constexpr removed from
the C prototype.
