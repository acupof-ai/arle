# Multi-GPU — Full Parallelism Compatibility (SGLang/vLLM aligned)

Owner: ckl
Status: Proposal (2026-04-28, replaces prior MVP-scoped drafts)
Scope: CUDA backend. Metal stays single-card.

**Goal.** Full feature parity with SGLang/vLLM's parallelism stack, not a
TP-only MVP. The architecture must support — designed in from F0, even
when individual features are turned on later — every axis below:

1. **Tensor parallel (TP)** — within-layer GEMM split
2. **Pipeline parallel (PP)** — across-layer split, micro-batched
3. **Attention data parallel (Attn-DP)** — replicate KV per attn rank, dense FFN still TP'd
4. **Attention context parallel (Attn-CP)** — sequence-axis split for long context
5. **MoE expert parallel (EP)** — experts sharded across ranks
6. **MoE tensor parallel (MoE-TP)** — within-expert TP
7. **MoE data parallel (MoE-DP)** — replicated experts, per-rank tokens
8. **Speculative draft worker** — `is_draft_worker` branch
9. **CUDA graph + collectives** — full-capture path with comm inside
10. **Custom-AR fast paths** — pynccl/custom_all_reduce/mscclpp/quick_ar/symm_mem
11. **Multi-host rendezvous** — TCP store / `MASTER_ADDR:MASTER_PORT`

Single-host comes first to keep the F0–F8 cycle on one machine, but
multi-host (F9) is **the same code path** with a different rendezvous —
not a separate runtime.

Out of scope (orthogonal, separate plans):

- New model specs (DeepSeek-V3, Llama 3/4, Mistral, Mixtral). The
  parallel wiring lands in `qwen3-spec` / `qwen35-spec`; adding more
  models is a model-spec effort, not a parallelism effort.
- KV-tier multi-tier across nodes (already covered by `tiered-kv-*`
  plans).
- Disaggregated prefill/decode cluster (separate, much bigger plan).

Audit baselines (every link/path in this doc resolves against these):

- SGLang: `1a55646dcdf06f77441506be5c74afb045341636` (HEAD of `main` 2026-04-28, `/tmp/sglang`)
- vLLM: `fd74c90d9c3b5c35308f1f0ab308469235fa5277` (HEAD of `main` 2026-04-27, `/tmp/vllm`)
- TGI: `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` (2026-03-21, `/tmp/tgi`)

---

## 1. Survey: how four engines do this, and what they trade

| Concern | vLLM | SGLang | TensorRT-LLM | TGI (HF) |
|---|---|---|---|---|
| Process model | Executor zoo: `Ray` / `mp` / `uniproc` / `external_launcher` (`vllm/v1/executor/abstract.py:37,53–80`) | `mp.Process` per rank (`entrypoints/engine.py:577,603`); `run_scheduler_process` per rank (`scheduler.py:3764`) | Build-time engine + per-rank C++ runtime; MPI launch | Rust router/launcher spawns Python shard processes per GPU (`launcher/src/main.rs:601–612`); shards talk gRPC to router |
| Distributed init | `init_distributed_environment` + `initialize_model_parallel` (`vllm/distributed/parallel_state.py:1350,1486`) | Same names (`parallel_state.py:1646,1721`) — SGLang lifted from vLLM | NCCL via MPI bootstrap | `initialize_torch_distributed` (`server/.../utils/dist.py:48`) — torchrun-style env |
| Group coordinators | `GroupCoordinator` (`parallel_state.py:290`); WORLD/TP/PP/EP/DP groups | Same name, more groups (`parallel_state.py:1721–1900` initializes WORLD/TP/PP/EP/Attn-TP/Attn-CP/MoE-TP/MoE-EP/MoE-DP) | C++ `NcclCommunicator` per parallel mode | Single TP group only |
| Linear sharding | `ColumnParallelLinear` etc. (`vllm/.../linear.py:410,609,977,1394`) | Same names (`sglang/.../linear.py:289,485,889,1335`) — same lineage | Build-time tactic selection per shard | `TensorParallelColumnLinear` / `TensorParallelRowLinear` (`tensor_parallel.py:118,183`) |
| KV-head replication when `kv_heads < tp_size` | Yes | Yes (`models/qwen3.py:84–95`) | Build-time | Yes |
| Comm primitives | `device_communicators/`: pynccl, custom_all_reduce, mscclpp, quick_all_reduce, flashinfer_all_reduce, ray_communicator | Same set: pynccl, custom_all_reduce, custom_all_reduce_v2, pymscclpp, quick_all_reduce, shm_broadcast (`distributed/device_communicators/`) | Built-in NCCL only; user-replaceable plugin in 0.13+ | NCCL only via torch |
| PP scheduling | Virtual pipeline + 1F1B; `IntermediateTensors` proxy (`vllm/v1/worker/gpu_worker.py:73`) | 1F1B-ish: async send + sync recv (`scheduler_pp_mixin.py:47–112`); `PPProxyTensors` | Static graph PP | Not supported |
| DP attention | Yes (later addition, vLLM v1) | Yes (`dp_attention.py:240,257,274 initialize_dp_attention`) — SGLang's invention for DeepSeek-V3 | Limited | No |
| Speculative decode | Separate proposer/scorer hierarchy (`vllm/v1/spec_decode/{eagle,medusa,dflash,...}`) | `is_draft_worker: bool` flag on `TpModelWorker` (`tp_worker.py:231`); concrete workers in `speculative/{eagle_worker.py:169, standalone_worker.py:86, dflash_worker.py}` | Plugin, less mature | Medusa via plugin |
| CUDA graph + collectives | Piecewise capture (attention skipped, rest captured) — landed earlier | Full capture incl. AR via `graph_capture` context (`parallel_state.py:1529`) + `custom_all_reduce` inside | Always-graph by design | Eager only |
| Comm fusion | `flashinfer_all_reduce` (`device_communicators/flashinfer_all_reduce.py`) | `flashinfer_allreduce_residual_rmsnorm` (`layers/flashinfer_comm_fusion.py:415`) + `CommunicateWithAllReduceAndLayerNormFn` (`layers/communicator.py:865`) | Built-in fused | None |
| Multi-host | TCP store / Ray | TCP store (`parallel_state.py:1591`) + `init_method="env://"` | MPI | torchrun env |

### Tradeoff axes and what each engine optimizes for

- **Deploy flexibility vs. code size.** vLLM's executor zoo (4 backends) maximizes deploy options; SGLang collapses to one (mp). vLLM pays in code-paths-to-maintain; SGLang pays in deploy-shape lock-in. **For us:** one path is enough — we have one user (ckl) and one deploy shape (single binary, single host first, env-bootstrapped multi-host later). Add Ray when a real customer demands it.
- **Process vs. thread.** Every Python engine spawns processes (GIL escape; CUDA-context isolation; clean kill). TGI also processes, but only because the model code is Python. **For us:** Rust has no GIL, cudarc is thread-safe per-context, our hot path is one binary — thread-per-rank wins on startup time, IPC absence, and shared-state simplicity. NCCL doesn't care thread vs. process; one comm per rank is one comm per rank.
- **Build-time vs. runtime parallelism choice.** TensorRT-LLM compiles the parallelism decision into the engine — fastest runtime, slowest change cycle, can't A/B `tp_size` without rebuild. vLLM/SGLang/us: dynamic at startup. **For us:** dynamic, like SGLang. We're a research-friendly Rust runtime — runtime flexibility matters more than a 5% startup win.
- **Custom AR breadth.** vLLM and SGLang both ship 4–5 AR backends behind a thin selector (chosen per tensor size + topology). The cost is real: 5 paths × correctness × graph-capture × ROCm/CUDA = lots of code. The win is real too: small-tensor AR via custom IPC beats NCCL by 2–10×, which moves decoding tok/s materially. **For us:** trait-based `CollectiveBackend` from F0; NCCL impl in F0; the other backends land in F7 behind the same trait without changing call sites.
- **PP scheduling style.** vLLM virtual pipeline is more sophisticated (better bubble overlap) but more code. SGLang's 1F1B is simpler and good enough for the prompt-heavy workloads we care about. **For us:** copy SGLang's; revisit if the bubble shows up in benches.
- **DP-attention is no longer optional.** Originally a SGLang invention for DeepSeek-V3 (group-query attention with very few KV heads — TP can't shard them when `kv_heads < tp_size` without replication waste). Now Qwen3.5-MoE benefits too. vLLM caught up. **For us:** mandatory, not "future" — F3.
- **Spec abstraction shape.** vLLM splits proposer/scorer into separate worker types — generalizes to non-EAGLE proposers more cleanly. SGLang does `is_draft_worker: bool` on the same `TpModelWorker` — less code, harder to extend to wildly different draft schemes. **For us:** SGLang shape. We already have Metal DFlash in beta (per `support-matrix.md` §1) — mirror its shape for the CUDA path so the cross-backend story stays one-shape.
- **Graph capture under collectives.** vLLM piecewise (skip attention, capture the rest), SGLang full-capture with custom_ar inside graph. Full capture is faster end-to-end but only works when all collectives are graph-friendly. **For us:** SGLang full-capture; piecewise is a fallback only if F8 hits a wall.

### What we deliberately do *not* take from any engine

- vLLM's `RayDistributedExecutor` and `ExecutorWithExternalLauncher`. Single-binary, TCP-store rendezvous covers our needs.
- TensorRT-LLM's build-time engine. Conflicts with our "Rust dynamic loader, swap weights at runtime" model.
- TGI's gRPC-between-Rust-and-Python sidecars. We're pure-Rust on the hot path.
- vLLM's separate proposer/scorer hierarchy. SGLang's `is_draft_worker` is closer to what we already do on Metal.

---

## 2. Decisions (one line each, justified above)

| # | Decision | Why |
|---|---|---|
| D1 | Thread-per-rank, not process-per-rank | No Python GIL to escape; cudarc is thread-clean; faster spawn; no IPC |
| D2 | One executor path (SGLang `run_scheduler_thread` shape), not a vLLM-style zoo | One user, one deploy shape; Ray adds 4 backends of code we don't need |
| D3 | NCCL primary + `CollectiveBackend` trait designed in F0 | F7 plugs custom_ar/mscclpp/symm_mem/quick_ar without touching call sites |
| D4 | Full graph capture + comm-inside-graph (SGLang shape) | Faster end-to-end than piecewise once custom_ar lands |
| D5 | DP-attention as a first-class parallelism axis (F3, not "later") | MoE serving needs it; Qwen3.5-MoE already on the support matrix |
| D6 | `is_draft_worker: bool` on `TpModelWorker`, mirror Metal DFlash | One spec shape across backends; smaller surface than vLLM's hierarchy |
| D7 | Layer/scheduler/comm names ported verbatim from SGLang | Side-by-side diff against `parallel_state.py`/`tp_worker.py`/`linear.py` is the audit method |
| D8 | Multi-host = same code, env-var rendezvous (`MASTER_ADDR:MASTER_PORT`); F9 is wiring, not a new path | Falls out of the F0 design if `init_distributed_environment` is generic |
| D9 | Architecture-first per `feedback_architecture_before_features.md`: F0 lays every group/trait/struct shape; F1–F8 fill in features | Avoids retrofitting EP/DP-attn after TP ships and gets cemented |

---

## 3. Parallelism axis inventory (with anchors)

The 8-axis layout below comes verbatim from SGLang `parallel_state.py:1721–1900`'s `initialize_model_parallel` and `dp_attention.py:240–271`. We compute group ranks the same way; this is rule-derived, not opinion.

```
world_size = tp_size × pp_size                                 (parallel_state.py:1781)
attn_tp_size = tp_size // attn_dp_size // attn_cp_size          (parallel_state.py:1812)
moe_tp_size = tp_size // moe_ep_size // moe_dp_size             (parallel_state.py:1900)
attn_tp_rank = tp_rank % attn_tp_size                           (dp_attention.py:245)
attn_dp_rank = tp_rank // (attn_tp_size × attn_cp_size)         (dp_attention.py:252)
```

| Axis | Symbol | What it splits | Comm primitive | Where it inserts |
|---|---|---|---|---|
| TP | `tp_size` | per-layer GEMMs | AR after row-parallel | `LayerCommunicator` post-attn / post-MLP |
| PP | `pp_size` | layer groups | P2P send/recv of `IntermediateTensors` | inter-stage boundary |
| EP | `moe_ep_size` | MoE experts | All-to-all-v of dispatched tokens | MoE block |
| MoE-TP | `moe_tp_size` | within-expert GEMMs | AR within MoE expert | inside MoE block |
| MoE-DP | `moe_dp_size` | replicated experts, per-rank tokens | None inside expert; AR at output | MoE block |
| Attn-DP | `attn_dp_size` | KV per attn rank, FFN replicated | AR over attn-tp group only | attention block |
| Attn-CP | `attn_cp_size` | sequence axis | AR/AG over CP group | attention + comms surround |
| Spec | (orthogonal) | draft vs target model | broadcast accept mask | spec verify step |

---

## 4. Architecture (designed in F0, filled in F1–F9)

### 4.1 Distributed init + group coordinators

`infer/src/distributed/` (new module):

- `parallel_state.rs` — `init_distributed_environment(world_size, rank, init_method, local_rank)` and `initialize_model_parallel(tp, pp, ep, attn_dp, attn_cp, moe_dp)`. Direct port of SGLang's signatures (`parallel_state.py:1646,1721`).
- `group_coordinator.rs` — `GroupCoordinator { all_reduce, all_gather, reduce_scatter, broadcast, send, recv, group_start, group_end }`. Mirror of SGLang `parallel_state.py:197 GroupCoordinator`. Methods take a `&Stream` and an `Op`.
- `init_method.rs` — single-host TCP store on `127.0.0.1:<ephemeral>` *and* multi-host `tcp://MASTER_ADDR:MASTER_PORT`. Same code, different addresses. Same as SGLang `parallel_state.py:1707` `local_rank = int(os.environ.get("LOCAL_RANK", "0"))`.
- Group accessors: `get_world_group`, `get_tp_group`, `get_pp_group`, `get_ep_group`, `get_attn_tp_group`, `get_attn_dp_group`, `get_attn_cp_group`, `get_moe_tp_group`, `get_moe_ep_group`, `get_moe_dp_group`. Stored thread-local; one set per worker thread.

### 4.2 NCCL FFI + `CollectiveBackend` trait

`crates/cuda-kernels/src/ffi/nccl.rs` — minimal bindings: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast`, `ncclSend`, `ncclRecv`, `ncclGroupStart`, `ncclGroupEnd`, `ncclCommDestroy`. **All collectives needed by F0–F9 land here in F0**, even ones first used by F2 (send/recv for PP) or F5 (group_start/end for MoE all-to-all). One pass, one diff.

`crates/cuda-kernels/src/collective.rs` — `CollectiveBackend` trait:

```rust
pub trait CollectiveBackend: Send + Sync {
    fn all_reduce(&self, tensor: &mut DeviceMatrix, op: ReduceOp, stream: &Stream) -> Result<()>;
    fn all_gather(&self, input: &DeviceMatrix, output: &mut DeviceMatrix, stream: &Stream) -> Result<()>;
    fn reduce_scatter(&self, input: &DeviceMatrix, output: &mut DeviceMatrix, op: ReduceOp, stream: &Stream) -> Result<()>;
    fn broadcast(&self, tensor: &mut DeviceMatrix, src: usize, stream: &Stream) -> Result<()>;
    fn send(&self, tensor: &DeviceMatrix, peer: usize, stream: &Stream) -> Result<()>;
    fn recv(&self, tensor: &mut DeviceMatrix, peer: usize, stream: &Stream) -> Result<()>;
    fn supports_graph_capture(&self) -> bool;
}
```

F0 ships `NcclBackend`. F7 adds `CustomAllReduceBackend`, `MscclppBackend`, `SymmMemBackend`, `QuickAllReduceBackend`. The `GroupCoordinator` holds a `Box<dyn CollectiveBackend>` and a small selector that picks the backend per (tensor-size, topology, graph-capture-active). Selector logic ported from SGLang `parallel_state.py:1562,1567,1572` flags.

### 4.3 Per-rank worker

`infer/src/scheduler/cuda/tp_worker.rs` — `TpModelWorker` with the **full SGLang signature** even when fields are inert in early phases:

```rust
pub struct TpModelWorker {
    server_args: ServerArgs,
    gpu_id: u32,
    tp_rank: u32, tp_size: u32,
    pp_rank: u32, pp_size: u32,
    ep_rank: u32, ep_size: u32,
    attn_tp_rank: u32, attn_tp_size: u32,
    attn_dp_rank: u32, attn_dp_size: u32,
    attn_cp_rank: u32, attn_cp_size: u32,
    moe_tp_rank: u32, moe_ep_rank: u32, moe_dp_rank: u32,
    is_draft_worker: bool,
    nccl_port: u16,
    model_runner: ModelRuntime,
    world_group: Arc<GroupCoordinator>,
    pp_group: Arc<GroupCoordinator>,
    // ... matching SGLang tp_worker.py:237–290
}
```

**Reason** (the reason this doesn't violate `feedback_no_speculative_interface_shaping.md`): all of these fields are real callers in F0–F9; we are not pre-shaping for hypothetical consumers. Each field has a SGLang line that uses it. Putting them in F0 means F2 (PP) doesn't widen the constructor again.

`infer/src/scheduler/cuda/process.rs` — `run_scheduler_thread(server_args, gpu_id, tp_rank, pp_rank, …, nccl_port)`. One function per rank, mirror of SGLang `scheduler.py:3764 run_scheduler_process`. Spawned via `std::thread::spawn` from `bootstrap.rs` (D1).

### 4.4 ForwardBatch + `IntermediateTensors`

`infer/src/scheduler/forward_batch.rs` (new) — `ForwardBatch` struct with a `pp_proxy: Option<IntermediateTensors>` field present from F0. PP impl in F2 fills it; F0/F1 leaves it `None`. Mirror of vLLM `vllm/v1/worker/gpu_worker.py:73 AsyncIntermediateTensors` and SGLang `model_executor/forward_batch_info.py PPProxyTensors`.

### 4.5 Sharded layers

`infer/src/model/parallel_linear.rs` — port the SGLang/vLLM class hierarchy 1:1:

| Type | SGLang anchor | Notes |
|---|---|---|
| `ReplicatedLinear` | `linear.py:194` | for non-sharded layers (pre-norm, etc.) |
| `ColumnParallelLinear` | `linear.py:289` | output dim sharded |
| `MergedColumnParallelLinear` | `linear.py:485` | fused gate+up |
| `QkvParallelLinear` | `linear.py:889` | KV-head replication rule (`models/qwen3.py:84–95`) lives here |
| `RowParallelLinear` | `linear.py:1335` | input dim sharded; `reduce_results: bool` |
| `VocabParallelEmbedding` | `vocab_parallel_embedding.py:161` | masked lookup + AR |
| `ParallelLmHead` | `vocab_parallel_embedding.py:512` | column over vocab; AG in `LogitsProcessor` |

All weight-loader narrow math copied from `linear.py:404–405,1447–1448`. The `Shard` annotation lands in `crates/qwen3-spec` and `crates/qwen35-spec` per tensor.

### 4.6 Layer-level comm

`infer/src/model/layer_communicator.rs` — `LayerCommunicator` with the same method surface as SGLang `layers/communicator.py:424 LayerCommunicator`:

- `all_reduce_post_attention(hidden, residual)` — plain AR (F1)
- `all_reduce_post_mlp(hidden, residual)` — plain AR (F1)
- `fused_allreduce_residual_rmsnorm(...)` — fused (F8, ports `flashinfer_comm_fusion.py:415` + `communicator.py:865`)
- `dp_attention_gather/scatter(...)` — DP-attention buffer ops (F3, ports `dp_attention.py`)
- `cp_attention_split/gather(...)` — sequence-parallel ops (F4)

Method signatures are present from F0 (no-op or `unimplemented!` body until their phase). Call sites in `qwen3/decode.rs`/`prefill.rs` etc. land in F1 and don't move.

### 4.7 MoE expert dispatch

`infer/src/model/moe.rs` (new in F5) — port of SGLang `layers/moe/moe_runner` + `token_dispatcher`:

- `MoeRouter` — top-k selection (`layers/moe/topk.py`)
- `TokenDispatcher` — all-to-all-v of `(token_id, expert_id)` pairs across EP group (`layers/moe/token_dispatcher/`)
- `FusedMoeExpert` — per-rank expert GEMMs; supports MoE-TP within
- `MoeReduce` — combine outputs back to original token positions

vLLM `vllm/model_executor/layers/fused_moe/` is the comparable surface; SGLang's is more modular and we copy that structure.

### 4.8 Speculative draft

`infer/src/scheduler/cuda/draft_worker.rs` (new in F6) — instantiates `TpModelWorker { is_draft_worker: true, .. }`. Mirrors SGLang `speculative/eagle_worker.py:169` and `standalone_worker.py:86`. Multi-layer EAGLE per `tp_worker.py:362–388 _init_multi_layer_eagle_model_runners`.

### 4.9 CUDA graph + collectives

`infer/src/model/cuda_graph.rs` already does single-GPU capture. F8 extends:

- `GraphCaptureContext` — port of SGLang `parallel_state.py:1529 graph_capture` context manager. Pins comm to `CollectiveBackend::supports_graph_capture() == true` for the duration.
- `CollectiveBackend::supports_graph_capture` returns `true` for `CustomAllReduceBackend` / `SymmMemBackend`, `false` for `NcclBackend` (NCCL graphs are doable but non-trivial; we do them later if needed).

---

## 5. Phasing (10 phases; each independently mergeable)

Every phase exits with: `cargo test` green, `codex review --uncommitted` pass, wins/ entry per `feedback_bench_every_change.md`. F0 is the only phase that lays foundations without features — its acceptance is "the whole skeleton compiles and the single-GPU path is unchanged."

| Phase | Scope | Acceptance gate |
|---|---|---|
| **F0** | Foundation: `parallel_state.rs` + full NCCL FFI + `CollectiveBackend` trait + `GroupCoordinator` (all 9 groups) + `TpModelWorker` skeleton (full signature, F1+ fills bodies) + `ForwardBatch.pp_proxy` slot + `LayerCommunicator` skeleton (F1+ fills) + multi-host TCP rendezvous + `INFER_CUDA_DEVICE` | Single-GPU bench within ±2% of latest baseline; greedy tokens identical; `--features cuda,nccl` builds and a 2-thread `all_reduce(sum)` smoke passes |
| **F1** | TP dense: 7 ParallelLinear types + KV-head replication rule + sharded weight loader + sharded `PagedKVPool` + plain AR insertion in Qwen3/3.5 dense + logits all-gather + token broadcast | TP=2 greedy parity vs TP=1 for ≥256 tokens on Qwen3-4B and Qwen3.5-0.8B |
| **F2** | PP: `SchedulerPPMixin` port (`scheduler_pp_mixin.py:47`) + `IntermediateTensors` send/recv + `PPMissingLayer` + per-stage layer slicing in `qwen3/decode.rs`/`prefill.rs` + `pp_group.is_last_rank` sampling | PP=2 and TP=2×PP=2 greedy parity vs TP=1; PP bubble < 30% on Qwen3-32B at c=8 |
| **F3** | DP-attention: `dp_attention.rs` port (`dp_attention.py:240,257,274`) + `LayerCommunicator::dp_attention_gather/scatter` + per-rank request routing in scheduler + `DpPaddingMode` | TP=4+DP=2 greedy parity vs TP=4; tok/s improvement on Qwen3.5-MoE serving (KV-bound regime) |
| **F4** | Attn-CP: sequence-axis split + `cp_attention_split/gather` + long-context prefill path | Long-prompt (32K) prefill parity at CP=2 vs CP=1; CP=2 prefill ≥ 1.5× CP=1 on 32K |
| **F5** | MoE EP / MoE-TP / MoE-DP: `moe.rs` (router/dispatcher/expert/reduce) + per-MoE-block all-to-all + Qwen3.5-MoE wiring | Qwen3.5-MoE A3B greedy parity at EP=2 vs EP=1 (lifts support-matrix entry from "Beta CUDA stub" to "Supported"); tok/s parity within 10% of SGLang on same hardware |
| **F6** | Spec draft: `draft_worker.rs` + multi-layer EAGLE + verify step + accept-mask broadcast | Spec accept rate within ε of SGLang on Qwen3-4B + EAGLE-3 draft; end-to-end tok/s win > 1.5× over no-spec at c=1 |
| **F7** | Custom-AR fast paths: `CustomAllReduceBackend` (small-tensor IPC AR) + `MscclppBackend` + `QuickAllReduceBackend` + `SymmMemBackend` + selector | TP=2 small-tensor AR latency < NCCL by ≥30%; end-to-end tok/s improvement at c=1 |
| **F8** | Graph + comm fusion: `GraphCaptureContext` + `flashinfer_allreduce_residual_rmsnorm` port + `CommunicateWithAllReduceAndLayerNormFn` port + graph-capture under TP=2,4 | TP=2 graph-on tok/s ≥ graph-off + AR-overhead-recovered; correctness parity preserved |
| **F9** | Multi-host: env-var rendezvous (`MASTER_ADDR`/`PORT`/`WORLD_SIZE`/`RANK`/`LOCAL_RANK`) + cross-host NCCL + multi-host launcher script | 2-host × 4-GPU = 8-rank TP greedy parity vs 1-host × 8-rank; tok/s within 15% (NVLink-limited within node, IB across) |

---

## 6. Trip wires

Stop, surface to ckl, do not paper over:

- F1 token-parity at TP=2 misses by even one token → shard or loader narrow doesn't match SGLang. Diff line-by-line against `linear.py:404–405`.
- F2 `IntermediateTensors` shape mismatches between stages → PP slicing wrong; check `models/qwen3.py PPMissingLayer` placement.
- F3 DP-attention loses tokens between dispatch/gather → buffer-len math wrong; re-verify `dp_attention.py:240–271` math.
- F5 MoE all-to-all-v drops or duplicates tokens → router or dispatcher index wrong; the cheapest probe is a per-token tag round-trip.
- F7 small-tensor AR slower than NCCL → IPC handle mapping or stream sync wrong; bench against SGLang `custom_all_reduce.py` reference.
- F8 graph capture fails under TP → the chosen `CollectiveBackend.supports_graph_capture()` lied; gate the graph and triage.
- F9 cross-host NCCL hangs → NIC discovery / IB topology issue; `NCCL_DEBUG=INFO` and don't ship until clean.

Per-phase memory check: per-card peak at TP=N must be < single-GPU peak / N + 10%; if not, a buffer is duplicated (usually KV pool sizing or activation forgot to use `*_local`).

---

## 7. Files (concrete)

New (F0 unless noted):

- `infer/src/distributed/{parallel_state,group_coordinator,init_method}.rs`
- `crates/cuda-kernels/src/ffi/nccl.rs`
- `crates/cuda-kernels/src/collective.rs` (trait + `NcclBackend`; F7 adds 4 more impls)
- `infer/src/scheduler/cuda/{tp_worker,process,draft_worker}.rs` (`draft_worker` in F6)
- `infer/src/scheduler/forward_batch.rs`
- `infer/src/model/{parallel_linear,layer_communicator,logits_processor,moe,dp_attention,cp_attention}.rs` (`moe` in F5; `dp_attention` in F3; `cp_attention` in F4)
- `infer/src/scheduler/cuda/scheduler_pp.rs` (F2)

Modified:

- `crates/cuda-kernels/src/tensor.rs` (F0 — `on_device(ordinal)` + per-device sm_count)
- `crates/cuda-kernels/src/paged_kv.rs` (F1 — `num_kv_heads_local`)
- `crates/qwen3-spec/src/lib.rs`, `crates/qwen35-spec/src/lib.rs` (F1 — per-tensor `Shard` annotation; F5 — MoE expert annotations)
- `infer/src/model/qwen3/{weights,decode,prefill}.rs`, `infer/src/model/qwen35/{weights,decode,prefill,batch_decode}.rs` (F1 + F2 + F3 + F5 — TP wiring, PP missing-layer, DP-attn comms, MoE block)
- `infer/src/model/kv_cache.rs` (F1 — local-heads sizing)
- `infer/src/model/cuda_graph.rs` (F1 gate to `tp_size==1`; F8 lift gate)
- `infer/src/scheduler/cuda/{core,core/construction,execution}.rs` (F1 + F2 + F3)
- `infer/src/backend/cuda/bootstrap.rs` (F0 + F2 — worker pool spawn including PP ranks)
- `infer/src/tensor_parallel.rs` (F0 — drop `NCCLComm` `todo!()`, re-export `GroupCoordinator`)

Docs / env:

- `docs/environment.md` — F0: `INFER_CUDA_DEVICE`, `INFER_TP_SIZE`, `INFER_PP_SIZE`, `INFER_EP_SIZE`, `INFER_ATTN_DP_SIZE`, `INFER_ATTN_CP_SIZE`, `INFER_CUDA_DEVICES`, `INFER_NCCL_PORT`; F9: `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`
- `docs/support-matrix.md` — update §1, §2, §3 per phase landing
- `docs/experience/wins/` — one entry per phase × `(model, parallelism config)`

HTTP / `LoadedInferenceEngine`: **no API change** through F8. F9 adds nothing user-visible; multi-host is configuration. The whole point of keeping `server_engine.rs` device-agnostic is paid back here.

---

## 8. Bench requirements (per CLAUDE.md §Benchmarks)

- F0: regression-only bench; ±2% of latest single-GPU baseline.
- F1, F2, F3, F4, F6: correctness-first; commit body cites parity test result; wins entry only when a perf number is claimed (TP=2 in F1; PP+TP=4 in F2; DP=2 in F3; CP=2 in F4; spec accept rate + tok/s in F6).
- F5: full sweep at EP ∈ {1, 2, 4} on Qwen3.5-MoE; one wins entry per config.
- F7: per-backend AR latency micro-bench + end-to-end TP=2 tok/s.
- F8: TP=2,4 graph-on vs graph-off comparison.
- F9: 2-host × 4-GPU vs 1-host × 8-GPU comparison.

All sweeps use canonical `scripts/bench_guidellm.sh` params per `docs/plans/guidellm-integration.md` §3 — no flag flips.

---

## 9. Open questions (answer before F1 starts)

1. **NCCL vs cudarc-bundled.** Add NCCL as a system dep (`NCCL_HOME`) or vendor via cmake? SGLang/vLLM rely on system. **Lean:** system, with build error if absent under `--features nccl`. Decide and document in `docs/environment.md` before F0 ships.
2. **Loader CPU memory budget.** mmap + per-rank narrow keeps CPU peak at ~1× model size + 1 shard. Verify on Qwen3-32B safetensors before F1 weight loader lands.
3. **Tied embeddings under `ParallelLmHead`.** Both Qwen3 and Qwen3.5 tie `lm_head` to `embed_tokens` by default. Confirm the tied path goes through `VocabParallelEmbedding` weight only (no separate lm_head file expected), matching SGLang `vocab_parallel_embedding.py:512`.
4. **Per-rank scheduler vs central scheduler.** SGLang runs one full Scheduler per rank (each rank's scheduler ticks the same way; rank 0 owns the request queue, others mirror). vLLM v1 has a single scheduler driving an executor. We've been leaning vLLM-shape (one scheduler, N workers); SGLang-shape would be one Scheduler per rank with rank-0-as-leader. **Lean:** vLLM-shape — simpler request bookkeeping, less duplicated state. Decide before F2 (PP) since PP scheduling differs between the two shapes.
5. **Model spec coverage.** Plan delivers parallelism wiring for Qwen3 + Qwen3.5 + Qwen3.5-MoE. Adding DeepSeek-V3 / Llama / Mistral is a *separate* model-spec plan; the parallelism architecture won't need to change to absorb them, but the per-model TP wiring (which projection is column vs row, MoE topology) does. Out-of-scope here, in-scope for follow-up plans.
