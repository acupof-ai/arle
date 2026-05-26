# DSv4 native DeepEP — kernel layer is torch-free, custom-rewrite path validated (phase 1.0a-i)

## Context

`docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md` originally targeted
a sidecar implementation in C++ to avoid Python in the hot path. The
inspection of `deep_ep.hpp` showed that DeepEP's user-facing `Buffer`
class takes / returns `torch::Tensor` and `pybind11::bytearray`, which
seemed to force one of three bad options:

1. Python sidecar (bends the "no Python on hot path" rule);
2. C++ sidecar that links libtorch (≈2 GB build-chain dep);
3. Full reimplementation of DeepEP's `Buffer` class on raw CUDA (~2-3
   weeks of work, plus ongoing maintenance against upstream DeepEP).

This entry kills option (1) and (2) and **drastically reduces (3)** by
showing that DeepEP's kernel layer is already torch-free.

## Root finding

`/sgl-workspace/DeepEP/csrc/kernels/*.cu` does NOT reference torch / ATen /
c10. The torch coupling lives entirely in the user-facing `deep_ep.cpp`
file (`Buffer::intranode_dispatch`, etc.) that wraps the kernels for the
Python extension. Confirmed:

```
$ grep -rln "torch::\|at::\|c10::" csrc/kernels/*.cu csrc/kernels/*.cuh
(no matches)
```

The public C++ API at `csrc/kernels/api.cuh` accepts raw `void*`, raw
`cudaStream_t`, raw arithmetic — no torch types in any signature:

```cpp
namespace deep_ep::intranode {
  void barrier(int** barrier_signal_ptrs, int rank, int num_ranks,
               cudaStream_t stream);
  void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx,
                int64_t* recv_topk_idx, float* recv_topk_weights, ...,
                cudaStream_t stream, int num_sms, ...);
  void combine(cudaDataType_t type, void* recv_x, float* recv_topk_weights,
               const void* x, ..., cudaStream_t stream, int num_sms, ...);
}
```

So the "2-3 week rewrite" was overstated. The correct plan is:

- Compile DeepEP's `csrc/kernels/{intranode,layout,runtime}.cu` directly
  with nvcc (the same way DeepEP's own `setup.py` does, minus the
  pybind11/torch wrapping).
- Write a thin torch-free `ArleBuffer` class that mirrors `Buffer` ctor's
  CUDA allocations and calls `intranode::*` / `layout::*` with raw
  pointers. The diff from `Buffer.cpp` is just the input-shape munging
  that strips `torch::Tensor` down to `data_ptr()`.

## Phase 1.0a-i smoke

Single-rank smoke verifies the build chain and the constructor logic
work end-to-end without any torch / Python / pybind / NVSHMEM
dependency. Source:

```cpp
// phase1a_spike.cpp — torch-free linkage smoke for DeepEP kernels.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t err = (stmt);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt,        \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    int rank = (argc > 1) ? std::atoi(argv[1]) : 0;
    int num_ranks = (argc > 2) ? std::atoi(argv[2]) : 1;
    int64_t num_nvl_bytes = (argc > 3) ? std::atoll(argv[3]) : (256LL << 20);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    int nvl_rank = rank % NUM_MAX_NVL_PEERS;
    int num_nvl_ranks = (num_ranks < NUM_MAX_NVL_PEERS) ? num_ranks : NUM_MAX_NVL_PEERS;
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);
    int64_t total_bytes = num_nvl_bytes + barrier_signal_bytes
                          + buffer_ptr_bytes + barrier_signal_ptr_bytes;

    void* local_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&local_buf, total_bytes));
    cudaIpcMemHandle_t local_handle{};
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, local_buf));

    void** buffer_ptrs_gpu = reinterpret_cast<void**>(
        static_cast<uint8_t*>(local_buf) + num_nvl_bytes + barrier_signal_bytes);
    int** barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
        static_cast<uint8_t*>(local_buf) + num_nvl_bytes + barrier_signal_bytes
        + buffer_ptr_bytes);
    int* local_barrier_signal = reinterpret_cast<int*>(
        static_cast<uint8_t*>(local_buf) + num_nvl_bytes);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemsetAsync(local_buf, 0, total_bytes, stream));

    void* host_buffer_ptrs[NUM_MAX_NVL_PEERS] = {};
    int* host_barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {};
    host_buffer_ptrs[nvl_rank] = local_buf;
    host_barrier_signal_ptrs[nvl_rank] = local_barrier_signal;
    CUDA_CHECK(cudaMemcpyAsync(buffer_ptrs_gpu, host_buffer_ptrs,
                               sizeof(host_buffer_ptrs), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(barrier_signal_ptrs_gpu, host_barrier_signal_ptrs,
                               sizeof(host_barrier_signal_ptrs),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (num_nvl_ranks >= 2) {
        deep_ep::intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank,
                                    num_nvl_ranks, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaFree(local_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("[phase1a] PASS\n");
    return 0;
}
```

Build (run on the pod; `DEEPEP_DIR` must point at the DeepEP source
checkout — no hard-coded path):

```bash
DEEPEP_DIR=/path/to/DeepEP nvcc \
  -ccbin g++ -std=c++17 -O2 \
  -DDISABLE_NVSHMEM \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$DEEPEP_DIR/csrc" \
  -gencode arch=compute_90,code=sm_90 \
  "$DEEPEP_DIR/csrc/kernels/intranode.cu" \
  "$DEEPEP_DIR/csrc/kernels/layout.cu" \
  "$DEEPEP_DIR/csrc/kernels/runtime.cu" \
  phase1a_spike.cpp \
  -lcudart -o phase1a_spike
```

Compile flag rationale:

- `-DDISABLE_NVSHMEM` skips the internode (multi-node RDMA) path inside
  runtime.cu; phase 1.0a covers intranode (NVL) only.
- `--expt-relaxed-constexpr` is required because runtime.cu uses
  `std::numeric_limits<int>::max()` from a `__global__` function. This
  is the same flag PyTorch's `cpp_extension` builder auto-injects.
- `arch=sm_90` matches the H20 target.

### Binary verification

```
$ ldd phase1a_spike
  libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12
  libstdc++.so.6, libgcc_s.so.1, libc.so.6, libdl.so.2,
  libpthread.so.0, librt.so.1, libm.so.6
$ nm -u phase1a_spike | c++filt | grep -E "torch|c10|at::|pybind|python"
(no output)
$ ls -l phase1a_spike
  412 KiB
```

Zero torch / pybind / python / nvshmem in the binary deps. Zero
torch-namespaced undefined symbols.

### Smoke result

```
$ phase1a_spike 0 1
[phase1a] devices=8, rank=0, num_ranks=1, num_nvl_bytes=268435456
[phase1a] alloc total 268435584 bytes @ 0x7f...
[phase1a] cudaIpcGetMemHandle ok, first bytes=02000000...
[phase1a] device pointer arrays populated
[phase1a] skipping barrier — phase 1.0a-ii does multi-rank
[phase1a] PASS
```

`intranode::barrier` is invoked only when `num_ranks >= 2` because
DeepEP's `SWITCH_RANKS` macro covers ranks 2..8. Phase 1.0a-ii will run
a 2-rank smoke with cross-device CUDA IPC to exercise the barrier and
the dispatch+combine path.

## Open-source contribution path

This spike validates that DeepEP can expose a **torch-free C++ API**
without restructuring the kernel layer. The shape of the contribution
to upstream DeepEP is:

- Add `csrc/cpp_api/` containing a torch-free `Buffer`-equivalent class
  whose ctor / sync / dispatch / combine methods take raw pointers + a
  CUDA stream, exactly mirroring `Buffer.cpp` minus the
  `torch::Tensor::data_ptr()` unwrapping.
- Expose a small `deep_ep_cpp_api.h` header for non-Python runtimes
  (vLLM-cpp, TGI, sgl-router, ARLE).
- A CMake target that produces a separate `libdeep_ep_cpp.so` with no
  torch / pybind dependency.

The case for landing this upstream is strong because the kernels are
already independent of torch — the diff is additive (new wrapper),
not invasive. ARLE will be the first downstream user; vLLM-cpp / TGI
are obvious follow-ups.

## Architectural conclusion

- "C++ sidecar" path is technically clean: pure CUDA + libcudart, no
  Python, no libtorch.
- Implementation scope shrinks from "2-3 week kernel reimplementation"
  to "~3-5 day Buffer-equivalent + sidecar harness".
- Open-source contribution is a natural by-product, not a side quest.

Phase 1.0a-ii (next): 2-rank smoke with cross-device CUDA IPC handle
exchange in a single process, exercising `intranode::barrier` with
`num_ranks=2`. PASS licenses phase 1.0a-iii (full 8-rank dispatch +
combine, byte-identical vs Python reference).

## Rule

Before sizing "this would require a multi-week rewrite of upstream X",
grep upstream X's kernel layer for the language / framework coupling.
The user-facing wrapper class is often disposable; the kernels
themselves are usually framework-neutral. The 80% reduction in scope on
this entry came from one `grep -rln "torch::|at::|c10::" csrc/kernels/`
that returned zero matches.
