# Phase B-2 + B-3.1 — torch-free DeepEP binding + native boot path

## Goal

After B-1 closed the multi-process scaffolding, B-2 and B-3 plug the
actual DeepEP kernels in. Both ARLE-side: no torch, no Python, no
pybind11. The phase 1.0a-iv spike proved DeepEP's `csrc/kernels/api.cuh`
is torch-free; B-2 turns that into a library callable from Rust, B-3
boots the library in the production scheduler boot path.

## What landed

| Commit | Subphase | What |
|---|---|---|
| `a308c93a` | B-2 | New workspace crate `crates/deepep-sys`. C++ wrapper `csrc/deepep_buffer.cpp` exposes `intranode::dispatch` + `intranode::combine` (with the correct `recv_channel_prefix` vs `channel_prefix_matrix` choice per the C.4.6 memory) behind a stable C ABI. build.rs nvcc-compiles into `libarle_deepep.a` when `ARLE_DEEPEP_DIR` is set; stub mode otherwise. src/lib.rs `Buffer` type with `new/local_ipc_handle/sync/dispatch/combine`. |
| `3dfdbea9` | B-3.1 | `NcclGroup::all_gather_bytes` helper (Int8 NCCL dtype, mirrors `all_gather_f32`). New `infer::native_deepep::NativeDeepEp::boot(rank, world_size, &NcclGroup)` constructs Buffer, all-gathers N×64-byte IPC handles + N×4-byte device ids over the EP NCCL group, calls Buffer::sync. Returns `Arc<NativeDeepEp>` with the `Buffer` behind `Arc<Mutex<>>` for shared use. |

## Build verification

| Build | Result |
|---|---|
| Mac `cargo check -p deepep-sys` (stub mode, env unset) | PASS |
| Mac `cargo check -p infer --features cuda,nccl,no-cuda` | PASS |
| 8×H20 pod `cargo build --release -p deepep-sys` with `ARLE_DEEPEP_DIR=/<DeepEP-src>` | PASS — `libarle_deepep.a` archived |
| 8×H20 pod `cargo build --release -p infer --features cuda,nccl --lib` with `ARLE_DEEPEP_DIR=…` | (in flight at write time; result appended below if needed) |

The pod-side `libarle_deepep.a` (release build, 9.18 s) confirms:
- nvcc accepts the wrapper's source paths.
- DeepEP's `intranode.cu` + `layout.cu` + `runtime.cu` link with our
  `deepep_buffer.cpp` cleanly under `-DDISABLE_NVSHMEM
  --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=
  compute_90,code=sm_90`.
- `ar rcs` packs them into the expected static archive.
- Rust links cudart + stdc++ alongside.

## What's NOT done — B-3.2 brief

NativeDeepEp boots, but `forward_deepep_routed_gpu`
(`infer/src/model/deepseek/mlp.rs:3094-3982`) still routes through the
NCCL DeepEP-style fallback. The actual switch from NCCL-emulated
dispatch/combine to deepep-sys's `Buffer::dispatch` / `Buffer::combine`
is the next step.

**Scope (~400 LOC across 4 files)**:

1. **Plumb NativeDeepEp through model construction**.
   `Deepseek::from_safetensors` accepts an optional pre-booted
   `Arc<NativeDeepEp>` (constructed in `async_main` /
   `run_worker_mode` once NCCL is up). DeepseekModel stores it
   alongside the existing `layer_communicator`.
2. **New helper `forward_native_deepep_routed_gpu`** alongside the
   existing `forward_deepep_routed_gpu`. Shape per SGLang's
   `deepep.py::_DeepEPDispatcherImplNormal::dispatch_b / combine_b`:
   - gate matmul → `topk_idx`, `topk_weights`
   - `buffer.dispatch(x, topk_idx, topk_weights, …)` → `recv_x`,
     `num_recv_tokens`, `send_head`, `rank_prefix`,
     `recv_channel_prefix` (all stored in scratch buffers passed
     into `deepep_sys::DispatchParams`).
   - local expert FFN on `recv_x` (reuse the existing
     `forward_local_routed_gpu` body, or call the same
     `dsv4_expert_backend` helper)
   - `buffer.combine(processed_x, send_head, rank_prefix,
     recv_channel_prefix, …)` → `combined_x`
   - return `DeepseekRoutedMoeOutput { hidden: combined_x, ready:
     None }`
3. **Route selection** in `weights.rs:2243-2310`: when
   `ARLE_DSV4_MOE_BACKEND=native-deepep` (currently bails per commit
   `cd780fc2`), route to `forward_native_deepep_routed_gpu`.
4. **Scratch buffer allocation**. `Buffer::dispatch` needs worst-case
   slot allocations for `recv_x` / `recv_src_idx` / `recv_topk_idx`
   / `recv_topk_weights` / `rank_prefix_matrix` /
   `recv_channel_prefix` / `send_head`. Reuse the
   `DeepseekMoeRuntimeCache` scratch pool that the existing path
   already maintains (`infer/src/model/deepseek/mlp.rs:3097`).

**Estimated split**:
- B-3.2.1 (~100 LOC): plumb NativeDeepEp through model construction +
  scheduler boot.
- B-3.2.2 (~200 LOC): `forward_native_deepep_routed_gpu` body, with
  scratch buffer reuse.
- B-3.2.3 (~50 LOC): route selection + drop the `bail!` in
  `dsv4_moe_deepep_enabled` for `native-deepep`.
- B-3.2.4 (~50 LOC): end-to-end serve smoke (2-rank on the pod, one
  greedy completion).

## What's NOT done — phase B-4 (SLO bench)

`scripts/bench_guidellm.sh dsv4-native-deepep` vs
`dsv4-nccl-deepep-fallback` per the pivot doc PASS gate (TTFT +5%,
TPOT +5%, p99 not regressed >3%, byte-identical greedy on 32-prompt
set). Hours of pod time, blocked on B-3.2 landing.

## Bench-exempt notes

B-2 + B-3.1 are pure library additions:
- `crates/deepep-sys` ships with stub mode default; native build is
  opt-in via `ARLE_DEEPEP_DIR`.
- `infer::native_deepep` is `#[cfg(cuda + nccl)]` and only invoked by
  B-3.2's scheduler boot path (not yet wired).

Default user (`ARLE_DSV4_MOE_BACKEND` unset or `deepep`) sees zero
behavior change.

## Rule

When binding a torch-dependent C++ library from Rust, look at whether
the *kernel* layer is torch-free before assuming you need a torch
shim. DeepEP's `csrc/kernels/api.cuh` proved torch-free at phase
1.0a-i; the resulting binding (B-2) is ~500 LOC of nvcc-compiled C +
~200 LOC of Rust + 0 LOC of torch. Same applies to other CUDA libs
that ship Python-friendly wrappers — the headless kernel layer is
usually what you actually want.
