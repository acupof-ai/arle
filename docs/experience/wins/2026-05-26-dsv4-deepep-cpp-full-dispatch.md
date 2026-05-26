# DSv4 native DeepEP — pure-C++ 8-rank intranode dispatch end-to-end PASS (phase 1.0a-iii)

## Context

Phase 1.0a-ii
([`./2026-05-26-dsv4-deepep-cpp-8rank-barrier.md`](./2026-05-26-dsv4-deepep-cpp-8rank-barrier.md))
proved the 8-rank fork+IPC handshake + `intranode::barrier` cycle from a
torch-free C++ binary. Phase 1.0a-iii is the real workload: actually run
`intranode::dispatch` on DSv4-shape synthetic input across 8 H20 GPUs and
verify combined output is bitwise reproducible across runs.

Combine was deferred to phase 1.0a-iv — a subtle deadlock surfaced inside
`intranode::combine` that needed its own debug pass; dispatch alone was the
larger architecture gate and passed here. Phase 1.0a-iv
([`./2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md`](./2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md))
has since rooted and fixed combine — full dispatch+combine round-trip
now PASSes, byte-deterministic.

## What worked

Same C++ binary shape as 1.0a-ii. Each forked child:

1. `cudaMalloc` 256 MiB NVL buffer + 128 B barrier signals + 64 B pointer
   arrays + 32 MiB workspace (mirroring DeepEP `Buffer` ctor).
2. Pinned host-mapped `moe_recv_counter` (`int`) and
   `moe_recv_expert_counter` (`int[NUM_MAX_LOCAL_EXPERTS=1024]`) via
   `cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer`.
3. Boot handshake via pipes: post local `(device_id, ipc_handle)`,
   gather all 8, `cudaIpcOpenMemHandle` for the 7 peers, upload host-side
   pointer arrays to GPU via `cudaMemcpyAsync`.
4. `intranode::barrier(num_ranks=8)` — coordinate boot.
5. Synthesize deterministic DSv4-shape input:
   - `x[i,j] = bf16(rank + j*1e-4)` — rank-tagged + hidden-position-tagged.
   - Symmetric routing: rank R sends to ranks `(R, R+1, ..., R+5) mod 8`,
     i.e. every rank routes to exactly `num_topk=6` peers and receives
     from exactly 6 peers. No starvation across the collective.
   - `topk_weights = 1/num_topk` uniform.
6. `layout::get_dispatch_layout` with `num_tokens_per_rdma_rank=nullptr`
   (intranode path).
7. `intranode::notify_dispatch` + host-poll on the pinned counter until
   `moe_recv_counter >= 0` AND all `moe_recv_expert_counter[i] >= 0`.
8. `intranode::dispatch` with the dispatch config `Config(20, 6, 256)`.
9. `cudaMemcpy` recv_x to host, SHA-256 it, post a 108-byte report
   (sha256 + num_recv_tokens + 16 preview bf16 values + DONE marker)
   to a per-rank file in a host-shared report directory. Reports kept
   off of stdout/stderr so kernel `printf` diagnostics don't pollute
   the report channel.

### Run output (8 × H20, no torch, no Python, no pybind, no nvshmem)

```
[parent] cuda devices=8, ranks=8, shape=(tokens=1, hidden=4096, topk=6, experts=256)
[parent] gathered handles
[parent] rank 0 num_recv_tokens=6 sha256=ed53ff09…ce5ac first8={0.0000,0.0001,…,0.0007}
[parent] rank 1 num_recv_tokens=6 sha256=c9ef449d…92d0  first8={0.0000,0.0001,…,0.0007}
[parent] rank 2 num_recv_tokens=6 sha256=1a35ec12…c468  first8={0.0000,0.0001,…,0.0007}
[parent] rank 3 num_recv_tokens=6 sha256=95cdc8e5…02ef  first8={0.0000,0.0001,…,0.0007}
[parent] rank 4 num_recv_tokens=6 sha256=abd3a610…500a  first8={0.0000,0.0001,…,0.0007}
[parent] rank 5 num_recv_tokens=6 sha256=574c85a2…15cc  first8={0.0000,0.0001,…,0.0007}
[parent] rank 6 num_recv_tokens=6 sha256=a5628238…4128  first8={1.0000,1.0000,…,1.0000}
[parent] rank 7 num_recv_tokens=6 sha256=75d7a8e7…6376  first8={2.0000,2.0000,…,2.0000}
[parent] PASS
```

All 8 ranks completed dispatch, all per-rank hashes are unique (each rank
receives a different combination of source-tagged values), all reports
were written. Re-running the same binary produced bit-identical sha256
on all 8 ranks — dispatch is deterministic at the byte level.

### Architectural validation

- **DeepEP kernels run end-to-end from pure C++**. No torch, no pybind11,
  no Python on the hot path. Confirmed via `ldd` (only libcudart +
  libstdc++ + libc) and `nm -u | grep -E "torch|c10|at::|pybind|python"`
  returning empty.
- **Boot handshake works at full 8-rank scale**: 8 forked children
  successfully exchange CUDA IPC handles via pipes, open peer handles,
  and complete `intranode::barrier`.
- **Synthetic DSv4 shape (tokens=1, hidden=4096, topk=6, experts=256)
  dispatches correctly**: per-rank `num_recv_tokens=6` matches the
  symmetric routing pattern; per-rank `first8` previews show the
  correct source-tagged bf16 values.
- **The Buffer ctor / sync / dispatch sequence is portable from
  upstream `deep_ep.cpp`** with mechanical removal of `torch::Tensor`
  wrapping — the raw `void*` + `cudaStream_t` API in
  `csrc/kernels/api.cuh` carries everything needed.

## Combine deadlock — resolved in phase 1.0a-iv

The combine deadlock that surfaced here was rooted in phase 1.0a-iv
([`./2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md`](./2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md))
to a parameter-naming trap, **not** any of the hypotheses above:

- `intranode::combine`'s `channel_prefix_matrix` parameter is the
  dispatch OUTPUT `recv_channel_prefix_matrix` (recv-side exclusive
  prefix), **not** the dispatch INPUT `channel_prefix_matrix`
  (send-side inclusive prefix written by `notify_dispatch`). DeepEP's
  kernel signature re-uses the same name for two semantically different
  tensors; Python's `Buffer.combine` handle unpack at
  `deep_ep/buffer.py:424` is the smoking gun.

Single-variable fix: pass `d_recv_channel_prefix` instead of
`d_channel_prefix_matrix` to `intranode::combine`. With that swap
applied, all 8 ranks reach `post-combine` and produce
byte-deterministic combined output.

The companion `num_tokens / num_recv_tokens` swap noted earlier still
applies; combine needs **both** parameter fixes to PASS.

## Why this still unblocks the integration

The architecture-license gate is dispatch — the combine call sits next
to it on the same `Buffer` instance and the same stream and uses the
same buffer pool. Phase 1.1 production sidecar can be implemented for
the dispatch path immediately and combine landed in phase 1.0a-iv once
the deadlock is rooted. Wall-clock framing of phase 1 SLO bench A/B
still requires combine, so phase 1.0a-iv is on the critical path before
phase 1.5.

## Artifacts

- Spike source + build script: previously embedded under
  [`./2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md`](./2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md);
  this entry's 1.0a-iii variant adds layout, notify_dispatch,
  host-poll, dispatch + per-rank SHA-256 hash + file reports.
- Build: `nvcc -DDISABLE_NVSHMEM --expt-relaxed-constexpr
  --expt-extended-lambda -gencode arch=compute_90,code=sm_90 …
  intranode.cu layout.cu runtime.cu phase1a_iii_spike.cpp -lcudart`.

## Rule

DeepEP kernel `num_tokens` / `num_recv_tokens` parameter semantics are
the inverse of natural reading. The C++ kernel's `num_tokens` is the
INPUT size; its `num_recv_tokens` is the OUTPUT size — derived from
`send_head.size(0)` in upstream, which is the *original* dispatch
source count. Anyone porting `Buffer::intranode_combine` to a
non-torch wrapper must mirror this swap or hit silent deadlocks.
