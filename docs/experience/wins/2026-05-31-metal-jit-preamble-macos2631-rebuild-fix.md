# Metal JIT crash after macOS 26.3.1 / Xcode 26.5 update — stale MLX preamble, fixed by forced clean rebuild

## Context

Freshly built `metal_serve` (Qwen3.6-35B-A3B-4bit) **loaded the model fine but
panicked on the first forward** (startup warmup) with:

```
mlx_async_eval failed: [metal::Device] Unable to build metal library from source
mlx/backend/metal/kernels/utils.h:64: error: unknown type name 'bfloat16_t'; did you mean 'float16_t'?
mlx/backend/metal/kernels/utils.h:144: error: no template named 'vec'; did you mean 'metal::vec'?
mlx/backend/metal/kernels/indexing/gather_front.h:14: error: use of undeclared identifier 'offset_neg_idx'
... gather_front<bfloat16_t, int32_t, int, 2>
```

Env: macOS **26.3.1**, Xcode **26.5**, Metal compiler **32023.883**, M4 Pro,
MLX runtime 0.31.1. The error chain is pure MLX/Metal — **not** the Rust
control-plane changes that prompted the rebuild.

## Root cause (evidence, not inference)

- `gather_front` (and gather/scatter generally) are **always runtime-JIT'd** in
  MLX — `indexing.cpp:84` assembles the kernel source from `metal::gather_front()`
  at runtime and compiles it with the live `xcrun metal`.
- That `metal::gather_front()` preamble string is **generated at build time** by
  `make_compiled_preamble.sh` (`vendor/mlx/mlx/backend/metal/CMakeLists.txt:15`)
  into `jit/gather_front.cpp`, `jit/utils.cpp`, … The on-disk generated files in
  the build dir were dated **May 19** (the prior toolchain).
- The macOS/Xcode update changed the Metal stdlib (`bfloat16_t`, `metal::vec`,
  `complex64_t` handling). The **precompiled `.metallib` still loads** (AIR is
  forward-compatible enough), so the model loads — but the **May-19-generated
  JIT preamble is now rejected by the newer compiler**, so the first JIT'd
  kernel crashes.
- Confirmed the fix is viable *before* spending the rebuild: MLX's own kernel
  source compiles cleanly with the current toolchain —
  `xcrun metal -std=metal3.2 -I vendor/mlx -c utils.h` → exit 0 (only
  `-std=metal3.0` fails, pre-native-bfloat). So this is **stale build output**,
  not an MLX-vs-toolchain incompatibility; no MLX version bump needed.

## What worked

Force a true clean `mlx-sys` rebuild so `make_compiled_preamble.sh` re-runs
under the current toolchain:

```bash
rm -rf target/release/build/mlx-sys-*     # cargo clean -p mlx-sys does NOT remove this
cargo clean -p mlx-sys
cargo build --release --no-default-features --features metal,no-cuda --bin arle
cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve
```

After: `jit/gather_front.cpp` + `mlx.metallib` re-dated today; `metal_serve`
**READY after ~19 s, past warmup**, served real generations and tool-calls.

**Critical gotcha:** the first attempt (`cargo clean -p mlx-sys` alone) was a
**no-op** — the cmake build dir `target/release/build/mlx-sys-<hash>/` survived
the clean, so the cmake custom commands saw their outputs as up-to-date and
never re-ran `make_compiled_preamble.sh` (45 s "rebuild", metallib unchanged at
May 19). You must `rm -rf` the build dir explicitly.

## Rule

After a macOS/Xcode update, an MLX Metal program that **loads but crashes on the
first forward with `Unable to build metal library from source` / `unknown type
bfloat16_t` / `metal::vec`** is a **stale build-time JIT preamble**, not a code
or model bug. Fix = `rm -rf target/<profile>/build/mlx-sys-*` then rebuild
(plain `cargo clean -p mlx-sys` leaves the cmake build dir and is a no-op).
Verify the toolchain can still compile MLX source (`xcrun metal -std=metal3.2 -I
vendor/mlx`) before assuming an MLX bump is needed.
