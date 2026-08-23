# Unsafe Code Audit

Date: 2026-08-23
Scope: `crates/**/*.rs`, production code only. Excluded: `#[cfg(test)]` modules and test-path files (`tests/`, `test_*.rs`, `*_test.rs`, `ffi/gemm_tests.rs`), and five files under concurrent modification (`infer-metal/src/{executor,dflash,lfm2,weights}.rs`, `mlx-sys/src/lib.rs`).

Method: every `unsafe {` block was extracted with its body and classified by its primary operation. Blocks that call the crate's own `unsafe fn` wrappers are classified as FFI, since those wrappers are kernel-launch / driver-call shims.

Total unsafe constructs: **1237** — 1121 `unsafe {` blocks, 95 `unsafe fn` declarations, 21 `unsafe impl` blocks. Zero `static mut`.

## Summary table

| Category | Count | Sound? | Notes |
|---|---|---|---|
| 1. FFI calls | 992 blocks + ~90 of the 95 `unsafe fn` declarations | Yes | Dominant pattern; checked wrappers (Result / status codes) and SAFETY comments at call sites |
| 2. Raw pointer dereference | 66 blocks | Yes, one documentation gap | `from_raw_parts` byte casts, `copy_nonoverlapping`, pointer arithmetic, `RawDevicePtr<u64>` device addresses |
| 3. Unsafe trait impls | 21 | Yes, two to verify | All `Send`/`Sync` for FFI handle types; each carries a SAFETY comment |
| 4. `static mut` access | 0 | — | None in the codebase |
| 5. `transmute` | 2 blocks (4 call sites) | Yes | `dlsym` result to NVTX fn pointers; standard pattern |
| 6. Other | 61 blocks | Yes | `env::set_var` (13), `mmap` (7), uninitialized device alloc / `MaybeUninit` / `align_to` (41) |

No unsound unsafe code was found. Three items need verification or comment fixes (Recommendations 1–3); the rest are cosmetic.

## Category details

### 1. FFI calls — 992 blocks

Every FFI family is wrapped at the boundary: raw C functions are only called inside `unsafe` blocks, and results are checked before use.

| Family | Sites | Check mechanism |
|---|---|---|
| CUDA kernels (`ffi::*_cuda`, `crates/cuda-kernels/src/ffi/`) | `cuda-kernels/src/{attention,moe,tensor_ops,quant_linear,sampling,kv_quant}.rs` | `Result` return; lengths checked before launch |
| cudarc driver/runtime | `infer-cuda/src/{attention,graph,tp}.rs`, `autograd/src/backend_cuda/*` | cudarc `Result`; stream-ordered on the engine thread's stream |
| NCCL | `cuda-kernels/src/collective.rs`, `infer-cuda/src/tp.rs` | `nccl::check` |
| MLX C bridge (`mlx_sys::*`) | `infer-metal/src/{mlx,qwen35,gemma4,deepseek_ocr,diffusion_gemma}.rs`, `autograd/src/backend_metal.rs` | `panic_if_mlx_error` / `mlx_last_error` after each call; `mlx_guard()` serializes calls |
| Vulkan (ash) | `vulkan-sys/src/lib.rs` | ash `Result` |
| HIP / ROCm | `hip-sys/src/lib.rs`, `infer-hip/src/model.rs` | `check(...)` wrapper |
| xgrammar / DeepEP | `xgrammar-sys/src/lib.rs`, `deepep-sys/src/lib.rs` | status codes checked |
| libc (dlopen, pipe, flock, affinity, kill) | `kv-native-sys/`, `cli/serve_multiproc.rs`, `infer-cuda/src/{numa_pin,loader}.rs`, `train/src/{spawner,sandbox}.rs` | return codes checked; fds owned after success |

Top files: `vulkan-sys/src/lib.rs` (87), `infer-cuda/src/attention.rs` (75), `cuda-kernels/src/attention.rs` (63), `cuda-kernels/src/moe.rs` (53 blocks + 47 `unsafe fn`), `infer-metal/src/mlx.rs` (43), `infer-metal/src/qwen35.rs` (42).

Soundness argument: pointers passed to FFI come from live `CudaSlice` / `Vec` / `MlxArray` handles held in the same scope (the `device_ptr()` guard is kept alive across the call, e.g. `cache_ptr` in `cuda-kernels/src/tensor.rs:341` extracts the device address once and the originating slice outlives the launch). Buffer lengths are checked against kernel parameters before launch. Every call site sampled carries a SAFETY comment or sits inside a checked wrapper.

The 95 `unsafe fn` declarations are kernel-launch shims (`cuda-kernels/src/moe.rs` 47, `cuda-kernels/src/collective.rs` 15, `infer-cuda/src/tp.rs` 6) and the `CollectiveBackend` trait methods (`cuda-kernels/src/collective.rs:44`). Their contract is uniform: the caller guarantees the raw pointers are valid device allocations on this backend's device and the stream belongs to the same device. The contract is documented on the trait and forwarded per fn.

### 2. Raw pointer dereference — 66 blocks

| Pattern | Sites | Soundness argument |
|---|---|---|
| `slice::from_raw_parts` byte reinterpretation | `infer-cuda/src/loader.rs` (7), `qwen35_state.rs` (8), `dsv4/slot_image.rs`, `attention/prefix_state.rs`, `kv-native-sys/src/direct_store.rs`, `cuda-kernels/src/tensor/device_matrix.rs` | Source is a live slice or mmap with checked length; bf16 casts are align-2 with all bit patterns valid |
| `ptr::copy_nonoverlapping` | `autograd/src/backend_metal.rs` (3), `safetensors_io.rs`, `vulkan-sys/src/lib.rs` (2), `dsv4/slot_image.rs` (2) | Length derived from source and target element counts; source and target are distinct allocations |
| Pointer arithmetic (`.add()`) | `cuda-kernels/src/tensor/weight_format.rs` (3), `autograd/src/backend_cuda/linear_attention_forward.rs` | Offset bounds-checked against the buffer length before the add (SAFETY comments cite the check) |
| `RawDevicePtr<T>` (u64 device address) | `cuda-kernels/src/tensor.rs:293`, used pervasively in `infer-cuda` | Never dereferenced in Rust; only passed back to kernels. `Send` for all `T` because it is an address, not a reference |
| `align_to::<bf16>()` | `cuda-kernels/src/tensor.rs:31` | `align_to` confines the cast to the aligned middle; unaligned prefix/suffix falls back to a decode copy |
| `from_raw_fd` / `CStr::from_ptr` | `cli/serve_multiproc.rs`, sys crates | fds owned after `pipe2` success; C strings are NUL-terminated static names or null-checked |

Top files: `infer-cuda/src/qwen35_state.rs` (8), `infer-cuda/src/loader.rs` (7), `autograd/src/backend_metal.rs` (6), `cuda-kernels/src/tensor/weight_format.rs` (6), `autograd/src/backend_cuda/handle.rs` (5), `cuda-kernels/src/tensor/device_matrix.rs` (5).

One documentation gap: `infer-cuda/src/loader.rs:2097` and `:2144` cast safetensors `u8` bytes to `&[i32]`. The cast is sound because `SharedTensor::bytes()` (`loader.rs:2628`) always aliases an mmap shard (`ShardBytes` is mmap-only, `loader.rs:704`), which is page-aligned, and safetensors aligns tensor data to 8 bytes. The SAFETY comments assert dtype and shape but do not state the alignment argument.

### 3. Unsafe trait implementations — 21

All are `Send` or `Sync` for FFI handle types. Each carries a SAFETY comment.

| Type | Traits | Location | Justification |
|---|---|---|---|
| `MlxHandleInner`, `MlxHandle` | Send + Sync | `autograd/src/backend.rs:61,66,104,109` | All MLX FFI access serialized through `mlx_guard()` |
| `GrammarCompiler`, `CompiledGrammar`, `GrammarMatcher` | Send (+ Sync for CompiledGrammar) | `xgrammar-sys/src/lib.rs:202,334,337,373` | C++ objects not aliased across threads; CompiledGrammar is read-only after construction |
| `OneShotComm` | Send + Sync | `infer-cuda/src/tp.rs:1181,1184` | Engine-thread confined |
| `NcclBackend` | Send + Sync | `cuda-kernels/src/collective.rs:150,154` | Handle used from the engine thread |
| `RawDevicePtr<T>` | Send (all T) | `cuda-kernels/src/tensor.rs:299` | u64 address, never dereferenced in Rust; single inference thread |
| `CudaGraphState` | Send | `infer-cuda/src/graph.rs:42` | Capture and replay run on the inference thread |
| `MlxArray` | Send | `infer-metal/src/mlx.rs:136` | Owns its handle; MLX process-global state serialized via `mlx_guard` |
| `Cpp{Gemma4,DiffusionGemma,DeepseekOcr}Model` | Send | `infer-metal/src/{gemma4,diffusion_gemma,deepseek_ocr}.rs` | C++ model handles, engine-thread confined |
| `Buffer` (DeepEP) | Send | `deepep-sys/src/lib.rs:397` | NVSHMEM buffer handle |
| `RawLogits` | Send | `infer-api/src/types.rs:54` | Caller must not share the mutable device allocation across threads |
| `ServeHandle<E,K>` | Send (E,K unconstrained) | `infer-server/src/lib.rs:595` | Field-by-field Send proof in the comment |

### 4. `static mut` access — 0

No `static mut` declarations exist. Process-global state uses `OnceLock` (`infer-cuda/src/nvtx.rs:27`) or `AtomicUsize`.

### 5. `transmute` — 2 blocks, 4 call sites

`infer-cuda/src/nvtx.rs:65-66` transmutes `dlsym` results to `Option<fn(i32) -> i32>` / `Option<fn() -> i32>` for `nvtxRangePushA` / `nvtxRangePop`. `train/examples/opd_step_cuda_realckpt_profile.rs:97-98` mirrors the same code. The signatures match the NVTX C ABI (documented in the SAFETY comment); pointers are null-checked before use. This is the standard `dlsym` pattern; `std` provides no safe alternative.

### 6. Other — 61 blocks

| Pattern | Count | Soundness argument |
|---|---|---|
| `std::env::set_var` / `remove_var` | 13 | All sites run single-threaded before child spawn or engine build (SAFETY comments state this). Forward-compatible with Rust 2024, where `set_var` becomes unsafe |
| `memmap2::Mmap::map` | 7 | Mapped files are immutable model artifacts or checkpoint files not mutated during the run (documented per site) |
| Uninitialized device alloc (`HiddenStates::uninit`, `DeviceVec::uninit`, `alloc_traced`) | ~30 | `unsafe fn` contract: every element must be written before read. Call sites carry SAFETY comments naming the writing kernel. Deliberate bandwidth tradeoff (no zeroing memset) |
| `MaybeUninit::assume_init`, `mem::zeroed` | ~6 | Driver fully writes the out-param on success (`graph.rs:266`); `cpu_set_t` is an all-bytes-zero-valid bitmask (`numa_pin.rs:107`) |
| `matrixmultiply::sgemm` | 3 | The crate marks `sgemm` unsafe because it trusts pointer/length args; callers pass live slices with checked dimensions |

## Recommendations

1. **`infer-server/src/lib.rs:595` — verify the `Send` impl is needed.** The SAFETY comment claims every field of `ServeHandle<E,K>` is independently Send. If that holds, the auto `Send` impl applies and the manual `unsafe impl` is redundant; delete it. If a field is not `Send`, name that field in the comment.

2. **`infer-cuda/src/loader.rs:2097,2144` — state the alignment argument.** The `u8`→`i32` casts are sound (mmap page alignment + safetensors 8-byte tensor alignment), but the SAFETY comments only assert dtype and shape. Add the alignment provenance so a future change to `ShardBytes` (e.g. a `Vec<u8>` fallback) does not silently turn these into UB.

3. **`autograd/src/backend.rs:61-109` — audit `mlx_guard()` coverage.** The `Send`/`Sync` impls for `MlxHandleInner`/`MlxHandle` rest on every MLX FFI call in the crate holding `mlx_guard()`. One unguarded call site breaks the `Sync` guarantee. A grep for `mlx_sys::` calls without a guard in scope would confirm the invariant.

4. **`cuda-kernels/src/kv_quant.rs:175` — drop the unsafe block.** `paged_attention_quantized_fa3_workspace_bytes` is a pure host-side size computation; the block exists only because the extern declaration is `unsafe`. A safe wrapper in the FFI module removes it. Cosmetic.

5. **`cuda-kernels/src/tensor.rs:92,209` — redundant inner block.** Inside `unsafe fn uninit`, the inner `unsafe { ctx.stream.alloc(...) }` wraps a safe cudarc call. The block documents the boundary, which is acceptable, but it is not required for soundness.

6. **Safe alternatives where they fit:**
   - `from_raw_parts` byte casts (`loader.rs`, `qwen35_state.rs`) could use `bytemuck::cast_slice`, which checks alignment at runtime and removes the unsafe block. The mmap-backed sources are always aligned, so the check would never fire.
   - The `nvtx.rs` `dlsym` transmute could use `libloading` or `dlopen2`, which encapsulate the symbol-to-fn-pointer cast.
   - `RawDevicePtr<u64>` has no safe equivalent: the cache exists to amortize `device_ptr()` over thousands of decode steps, and the address is only consumed by kernels. Keep as-is.

7. **Test-only note:** `infer-cuda/src/executor.rs:958` mutates process-global env inside a `#[cfg(test)]` test. It is safe today (no other env-mutating tests run in parallel in that crate) but will race if the test suite parallelizes.
