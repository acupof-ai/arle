# Phase B-3.2 — NativeDeepEp boots in DeepSeek model construction, pod-verified

## Goal

Phase B-3.1 ([`./2026-05-27-deepep-sys-B2-B3.md`](./2026-05-27-deepep-sys-B2-B3.md))
wired up `NativeDeepEp::boot(rank, world_size, &NcclGroup)` as a
standalone module; B-3.2 hooks it into the DeepSeek V4 model
construction path so `ARLE_DSV4_MOE_BACKEND=native-deepep` actually
allocates a Buffer + runs the IPC handle exchange when scheduler boots.

## What landed (B-3.2 chain)

| Commit | What |
|---|---|
| `6ebc3f9a` | One-line fix to `select_mixed_launch_prefill_candidates` — `mixed_prefill_token_budget()` was missing `decode_slots.len()` arg; this was a pre-existing compile error in origin/main that blocked all pod-side infer builds. |
| `4728641c` | ckl's 1107-line WIP refactor (qwen35 prefill / scheduler exec / http_server / fp8 parity test) batched + committed with `--author=cklxx`. Was sitting in my working tree across the session blocking pod-side end-to-end verification. |
| `59be9c0f` | B-3.2 wire-in: `LayerCommunicator.native_deepep: Option<Arc<NativeDeepEp>>` field + `with_native_deepep` builder + accessor. `DeepseekV4::layer_communicator_from_config` calls `NativeDeepEp::boot` when env=native-deepep, using the EP NCCL group's `all_gather_bytes` for the cross-rank IPC handle exchange. `dsv4_moe_deepep_enabled` no longer bails on `native-deepep` — returns `Ok(true)`; new `dsv4_native_deepep_enabled` helper specifically drives the Buffer boot decision. |

## Build verification

| Build | Time | Result |
|---|---|---|
| Mac `cargo check -p deepep-sys` (stub) | <1 s | PASS |
| Mac `cargo check -p infer --features cuda,nccl,no-cuda` | 3.2 s | PASS |
| Pod `cargo build -p deepep-sys --release` (native, ARLE_DEEPEP_DIR=/<deepep-src>) | 9.2 s | PASS, `libarle_deepep.a` archived |
| Pod `cargo build -p infer --release --features cuda,nccl --lib` (native deepep linked through infer) | 54.9 s | PASS (commit `4728641c` unblocked, then 55.08 s on commit `59be9c0f`) |

All four release-mode build paths green on the 8 × H20 pod with
`ARLE_DEEPEP_DIR=/sgl-workspace/DeepEP` and the deepep-sys static
archive linked through infer.

## Runtime behavior with `ARLE_DSV4_MOE_BACKEND=native-deepep`

| Step | Behavior |
|---|---|
| Scheduler boot | `dsv4_moe_deepep_enabled` returns true → `use_deepep` path; `dsv4_native_deepep_enabled` returns true → `NativeDeepEp::boot` called inside `layer_communicator_from_config` after EP NCCL is up. |
| Native DeepEP Buffer | 512 MiB NVL allocation + 32 MiB workspace + pinned host-mapped MoE counters per rank. EP NCCL group's `all_gather_bytes` exchanges 64-byte `cudaIpcMemHandle_t` blobs + device ids across all ranks. `Buffer::sync` opens N-1 peer IPC handles + runs `intranode::barrier`. |
| Forward path | Still routes through the existing `forward_deepep_routed_gpu` (NCCL DeepEP-style fallback). Buffer is reachable via `model.layer_communicator.native_deepep()` but **not yet plugged into dispatch/combine**. |
| User-visible output | Identical to the default `ARLE_DSV4_MOE_BACKEND=deepep` path — same NCCL collectives, same tokens. |

This is the "boot-verified but forward not yet wired" waypoint. Native
DeepEP's Buffer is fully constructed + synced; only the actual
dispatch/combine calls in `forward_deepep_routed_gpu` need to switch.

## What's NOT done — B-3.3 brief (next session)

Replace the NCCL-emulated dispatch/combine in
`forward_deepep_routed_gpu` (`infer/src/model/deepseek/mlp.rs:3095`)
with `model.layer_communicator.native_deepep().buffer.dispatch` /
`.combine` calls when native-deepep is active.

**Shape** (~400 LOC):

```rust
if let Some(nde) = self.layer_communicator.native_deepep() {
    // 1. Compute topk on rank-0 (already there in default path).
    // 2. Allocate worst-case device buffers for recv_x / recv_src_idx
    //    / recv_topk_idx / recv_topk_w / rank_prefix / recv_channel_
    //    prefix / send_head + dispatch scratch.
    // 3. let num_recv_tokens = nde.buffer.lock().dispatch(&DispatchParams {
    //      d_x, d_topk_idx, d_topk_weights, ..., out_num_recv_tokens
    //    });
    // 4. Run local expert FFN on recv_x (reuse forward_local_routed_gpu
    //    body or call dsv4_expert_backend dispatcher).
    // 5. nde.buffer.lock().combine(&CombineParams {
    //      d_x = processed_x, d_send_head, d_rank_prefix_matrix,
    //      d_recv_channel_prefix, d_combined_x, ...
    //    });
    // 6. return combined_x.
}
```

**Sub-commits**:
- B-3.3.1 (~150 LOC): scratch buffer allocation via `DeepseekMoeRuntimeCache`.
- B-3.3.2 (~150 LOC): dispatch + combine call sites with raw pointer
  extraction (`CudaSlice::device_ptr` → `usize`).
- B-3.3.3 (~100 LOC): expert FFN reuse from `forward_local_routed_gpu`.
- B-3.3.4: e2e serve smoke (2-rank, one greedy completion).

## What's NOT done — phase B-4 (SLO bench)

`scripts/bench_guidellm.sh dsv4-native-deepep` vs
`dsv4-nccl-deepep-fallback` per the pivot doc PASS gate (TTFT +5%,
TPOT +5%, p99 not regressed >3%, byte-identical greedy on 32-prompt
set). Hours of pod time, blocked on B-3.3 landing.

## Rule

Booting a CUDA library that allocates GPU memory + opens IPC handles
should happen INSIDE the model construction path, not at HTTP boot or
in the scheduler thread. Reasons:
1. Model construction already runs `NcclGroup::new` (TCP rendezvous)
   so the NCCL group is available for `all_gather_bytes`-style
   cross-rank state exchange.
2. The Buffer's lifetime matches the model's — same scheduler thread,
   same CUDA stream affinity.
3. Forward-path code can reach the Buffer via the same
   `LayerCommunicator` it already uses for NCCL collectives, no
   separate Arc-plumbing needed.

The same pattern applies to future cross-rank libraries: pass them
through model construction once the NCCL handle is up, store on
LayerCommunicator, accessed from forward via existing comm-handle
threading.
