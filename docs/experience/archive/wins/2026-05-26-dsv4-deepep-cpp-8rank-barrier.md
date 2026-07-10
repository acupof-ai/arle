# DSv4 native DeepEP — 8-rank fork + IPC + barrier PASS (phase 1.0a-ii)

## Context

Phase 1.0a-i
([`./2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md`](./2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md))
proved DeepEP's kernels link torch-free into a pure C++ binary, but only
exercised constructor + single-rank smoke. The barrier kernel itself
was not yet tested because `intranode::barrier`'s `SWITCH_RANKS` macro
covers `num_ranks` in `[2, 8]` only.

Phase 1.0a-ii is the next architectural gate: 8 forked child processes,
each bound to one of 8 H20 GPUs, exchange `(device_id, ipc_handle)`
pairs via pipes through the parent, open peer IPC handles, populate the
buffer/barrier pointer arrays, and run `intranode::barrier(num_ranks=8)`
to convergence. This is the **complete sidecar handshake** the
production binary will use.

## What worked

Same single binary as 1.0a-i, extended with a parent-vs-child mode
selected by `argv[1] == "--child"`. Parent forks 8 children with
bidirectional pipes; each child:

1. `cudaSetDevice(rank)`;
2. `cudaMalloc` the 256 MiB + metadata layout (matches Buffer ctor);
3. `cudaIpcGetMemHandle` on the local buffer;
4. writes its `(device_id, ipc_handle)` to the parent;
5. reads back the gathered 8-entry table from the parent;
6. for each peer (i ≠ rank), calls
   `cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)`;
7. fills the host-side `buffer_ptrs[]` / `barrier_signal_ptrs[]`
   arrays and uploads them to the GPU pointer arrays;
8. calls `deep_ep::intranode::barrier(barrier_signal_ptrs_gpu, rank,
   8, stream)`, then `cudaStreamSynchronize`;
9. writes `OK\n` to the parent and exits 0.

Parent gathers handles → broadcasts → reads 8 OKs → `waitpid` each
child → reports PASS or FAIL.

### Run output (8xH20)

```
[parent] cuda devices=8, ranks=8
[parent] gathered 8 handles, broadcasting...
[parent] rank 0 OK
[parent] rank 1 OK
[parent] rank 2 OK
[parent] rank 3 OK
[parent] rank 4 OK
[parent] rank 5 OK
[parent] rank 6 OK
[parent] rank 7 OK
[parent] PASS
```

All 8 children completed the barrier without timeout, deadlock, or
CUDA error. The barrier kernel itself spins until all `num_ranks`
slots flip, so PASS proves both that (a) `cudaIpcOpenMemHandle`
succeeded on all peers and (b) the kernel actually sees the shared
barrier counters via the GPU-side pointer arrays.

### Architectural verification

This handshake is identical in shape to what the production sidecar
needs:

- One process per rank, one CUDA device bound, one DeepEP buffer.
- `(device_id, ipc_handle)` exchange via the parent (in production:
  via the ARLE scheduler instead of a parent C binary; same protocol).
- Peer buffer pointer + barrier signal pointer arrays uploaded once
  at boot; intranode kernels operate on the GPU-resident arrays.
- Clean shutdown via `cudaIpcCloseMemHandle` + `cudaFree`.

Once dispatch+combine work in 1.0a-iii, the only remaining piece is
the persistent command loop (read "dispatch X" / "combine Y" from a
pipe, post the operation, signal completion) — which is a thin layer
over what's already proven.

## Artifacts

- Spike source + build script: previously embedded in
  [`./2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md`](./2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md);
  this entry's 1.0a-ii variant adds the fork+pipe orchestration in
  `phase1a_ii_spike.cpp` (~250 LOC, single file).
- Build: `nvcc -DDISABLE_NVSHMEM --expt-relaxed-constexpr
  --expt-extended-lambda -gencode arch=compute_90,code=sm_90 ...
  intranode.cu layout.cu runtime.cu phase1a_ii_spike.cpp -lcudart`.

## Architectural conclusion

The sidecar process-and-IPC layer is fully validated end-to-end on
8xH20 hardware. Phase 1.0a-iii adds the real workload — synthetic
DSv4-shape input, `layout::get_dispatch_layout` →
`intranode::notify_dispatch` → `intranode::dispatch` → identity
"expert GEMM" → `intranode::cached_notify_combine` →
`intranode::combine`, with the combined output hashed and compared
against the determinism-check reference from phase 0.5
([`./2026-05-26-dsv4-deepep-child-process-spike.md`](./2026-05-26-dsv4-deepep-child-process-spike.md)).
PASS there licenses phase 1.1 (production sidecar binary +
ARLE Rust host integration).

## Rule

When a multi-rank kernel needs a properly-set-up shared buffer state
to converge (DeepEP's barrier needs `barrier_signal_ptrs[i]` valid
for every `i`), the only honest test is a multi-rank end-to-end run.
A single-rank smoke proves linkage but does NOT prove the
cross-process IPC handshake — that's a separate gate, even if it
looks like a trivial extension.
