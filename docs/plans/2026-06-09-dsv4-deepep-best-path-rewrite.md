# DSv4 DeepEP best-path rewrite — converge on the SGLang fast contract

**Date:** 2026-06-09. **Backend:** CUDA, DSv4-Flash FP8, 8×H20. **Owner:** Claude (arch) + general-purpose subagents (impl). **Status:** plan; T0 (compile-accel + feature-forward) landed, T1 next.

Reference contract: SGLang `/workspace/sglang@0d51db3`, extracted verbatim
2026-06-09 (signatures + file:line inline below). Research framing:
[`docs/research/2026-06-01-dsv4-sglang-path-research.md`](../research/2026-06-01-dsv4-sglang-path-research.md).
This is an executor-copies-verbatim spec, not a principle list.

---

## 0. Goal + success metric (read before any "win")

Make DeepEP the **best DSv4 serving path** = SGLang contract:
`TP4/DP4 + --enable-dp-attention + token-owned EP + DeepEP LL(decode)/normal(prefill) + DeepGEMM consuming the deepep packed buffer directly`.

**Metric lane:** DeepEP pays off at the **batched throughput lane** (SGLang max-tput:
`--cuda-graph-max-bs 128 --max-running-requests 256`), measured as aggregate output
tok/s across concurrent requests. **Not** a B=1 latency win (at B=1 the all-to-all
moves one token; DP4 idles 3/4 ranks). Our B=1 = **38.24 tok/s** (allreduce+DeepGEMM,
2026-06-09) is the low-latency lane (SGLang serves it TP4+EAGLE). **License gate =
batched A/B (c=8/16/32/64) vs allreduce baseline**, never B=1.

## 1. Why current DeepEP is the slow fallback (code-grounded)

`dsv4_moe_forward_deepep` (moe.rs:1471) is **normal-mode-only** and replicated-token:

| Tax | Exact site | SGLang LL removes it by |
|---|---|---|
| Replicated tokens → 4.46× fanout | gets full token rows (no ownership) | token-owned EP (T2) |
| Mid-MoE D2H ×43 | `clone_dtoh(&scratch.recv_topk_idx)` moe.rs:1596 + host i64→i32 loop + H2D | LL packed layout: `masked_m` is a **device** `[E_local] int32`, never D2H |
| ~12 per-call device allocs/layer | moe.rs:1520-1689 (`HiddenStates::zeros`, `alloc_zeros`, `clone_htod(vec![-1])`) | pre-alloc LL scratch once; packed recv buffer IS the GEMM input |
| BF16 recv + re-quant before GEMM | `recv_x: HiddenStates::zeros(hidden, capacity)` BF16 (deepep.rs:119) | LL dispatch emits **FP8 e4m3** packed `[E_local, max_tok×ranks, hidden]` directly |
| Forfeits decode fast-paths | `use_moe_decode_scratch/comm_overlap/decode_graph` each `!use_deepep_transport` (dsv4.rs:1567,1721,1733) | LL path is itself the graph-captured decode path (T5) |
| Post-FFN all-reduce | non-deepep path only; deepep combines, but over-transported | LL combine reduces owned rows with weights |

deepep-sys today exposes **only** `arle_deepep_buffer_dispatch`/`combine` (normal
intranode, lib.rs:185-189). **No `low_latency_*`, no `num_max_dispatch_tokens_per_rank`.**

## 2. Dependency DAG — long pole is topology, not the LL swap

```
T0 compile-accel ........... DONE
T1 fail-fast/observability   ← land NOW (low-risk, no perf claim, default path unchanged)
     │
T2 TOPOLOGY (long pole) ..... DP-attention + token-owned EP execution  ──blocks──┐
     │                                                                            │
T3 attention contract ....... one flash_mla_with_kvcache(SWA+C4+C128)             │
     │                                                                            ▼
T4 MoE LL contract .......... deepep-sys LL FFI + dsv4_moe_forward_deepep_ll ◄─ needs T2
     │                         (packed FP8 → masked DeepGEMM, delete glue)
T5 graph + spec ............. persistent decode buffers + DeepEP-mode replay + EAGLE topk=1
```
**T4 (the no-glue payoff) is blocked by T2.** Without token ownership the LL packed
buffer still carries replicated rows → over-transport. "Just swap to LL" fails.

---

## T1 — fail-fast + startup observability  (land first, ~2 files)

**Goal:** the DSv4 path declares its effective contract and refuses to *claim* the
SGLang lane when components are missing. No perf claim; default route byte-unchanged.

**Files:** `crates/infer-cuda/src/dsv4.rs` (engine build), `crates/infer-api/src/loaded.rs` (serve log).

**Add** `fn dsv4_effective_contract(cfg) -> Dsv4Contract` emitting one `log::info!`:
```
DSv4 contract: lane={low-latency-allreduce|sglang-max-tput} tp=N dp=N ep=N
  attn=dsv4 kv=fp8_e4m3 page=? graph={off|decode} deepep={off|normal|ll|auto}
  deepgemm={native|stub} spec={off|eagle} moe_backend={allreduce|deepep}
```
**Guard:** when `ARLE_DSV4_SGLANG_PATH=1`, `bail!` unless ALL: FP8 KV ∧ `deepgemm_native_preflight` ok ∧ `cfg!(feature="deepep")` ∧ (graph|`ARLE_DSV4_DEBUG_LANE=1`). Default (flag absent) keeps allreduce, logged `lane=low-latency-allreduce (not SGLang max-tput)`.

**Buffers mutated:** none (observability). **Gate:** default serve still 38.24 B=1; startup log present; `cargo test` green.

---

## T2 — topology execution: DP-attention + token-owned EP  (long pole, own sub-plan)

**Goal:** each rank owns a token-row shard; attention runs DP, MoE runs EP, with a
gather→EP→scatter handoff. `MultiAxisConfig` (topology.rs:13) has the axis math but
execution is global-TP/EP only (topology.rs:35).

**Files:** `crates/infer-topo/src/topology.rs` (subgroup comm builders), `crates/infer-cuda/src/tp.rs` + collective layer (per-axis NCCL comms), `crates/infer-cuda/src/dsv4.rs` (attention on owned shard + gather/scatter), scheduler (batch partition by dp rank).

**T2.1 rank math** — port `compute_dp_attention_world_info` (dp_attention.py:240-271):
```
attn_tp_size = tp_size / attn_dp_size / attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank / (attn_tp_size * attn_cp_size)   // layout (dp,cp,tp), tp fastest
```
Add `MultiAxisConfig::attn_world_info(tp_rank) -> {attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size}`.

**T2.2 subgroup NCCL comms** — port group construction order (parallel_state.py:1862-2030):
`_ATTN_TP` (ranks sharing an attn_dp_rank), `_MOE_EP` (`range(start, end, moe_tp_size)`).
When a sub-size == tp_size the comm **aliases the global TP comm** (parallel_state.py:1970).
New: `infer_topo::build_axis_comms(world, cfg) -> AxisComms{attn_tp: Comm, moe_ep: Comm}`
backed by `ncclCommSplit` of the global comm (collective.rs).

**T2.3 batch partition** — port `get_dp_local_info` (dp_attention.py:385-419):
scheduler computes `global_num_tokens_gpu [attn_dp_size]`, `cumsum`, each rank takes
`local_start = cumsum[dp_rank-1]`, `local_num = global_num_tokens_gpu[dp_rank]`.
Attention forward consumes only `hidden[:, local_start..local_start+local_num]`.

**T2.4 handoff** (communicator.py:992 / :1234) — **enumerate the two collectives**:
- before MoE: `dp_gather_partial(global_hidden, local_hidden)` = zero `global_hidden [hidden, total_tokens]` → `memcpy` local slice in → all-reduce over `attn_tp` comm (dp_attention.py:469-494). MoE EP dispatch consumes `global_hidden`.
- after MoE: `dp_scatter(local_hidden, global_hidden)` = `local_hidden.fill(0)` → `memcpy` `global_hidden[:, local_start..]` out (dp_attention.py:550-568).

**Buffers mutated (enumerate + disposition):** `global_hidden` (new slot scratch, pre-alloc `[hidden, max_total_tokens]`, zeroed each step before gather — required, not self-healing); `local_hidden` (existing attn out, sliced); `global_num_tokens_gpu` (new `[attn_dp_size] i32`, scheduler-written per step); the two axis comms (created once at boot, immutable).

**Gate:** c=8 emits correct text; per-rank dispatch `num_recv` shows owned (≈ tokens/dp) not replicated rows; DeepEP fanout 4.46×→~1×; allreduce default lane unchanged.

---

## T3 — attention contract: one flash_mla_with_kvcache  (after T2)

**Goal:** decode attention = a single `flash_mla_with_kvcache` consuming SWA+C4+C128 via persistent page indices; delete standalone CSA selector staging as steady-state.

**SGLang call** (deepseek_v4_backend.py:1036-1054):
```
flash_mla_with_kvcache(q[N,1,h_q,576], swa_k_cache[blocks,swa_win,1,656]fp8,
  head_dim_v=512, is_fp8_kvcache=True, softmax_scale=head_dim**-0.5,
  indices=swa_page_indices[N,1,topk] (topk%64==0), topk_length=seq_lens[N]i32,
  attn_sink[h_q]f32, extra_k_cache=C4/C128, extra_indices_in_kvcache=...,
  extra_topk_length=...) -> o[N,1,h_q,512]
```
FP8 sparse KV/token = 656 B (512 NoPE-e4m3 + 16 scales-f32 + 128 RoPE-bf16-unquant),
head_dim 576 / v 512 / num_heads_k 1 (flash_mla_interface.py:91-95).
Persistent: the 3 `c1/c4/c128_flashmla_metadata` (`get_mla_metadata()`, init-once).
Per-step copy: page_table, seq_lens, positions, swa/c4/c128 page_indices+lengths
(backend `DSV4AttnMetadata.copy_` :135-167). ARLE today loops attention per row
(`forward_decode_batch`) + separate `attn_csa_select_kernel`.

**Gate:** ≥2-prompt needle retrieval; request_trace shows one attn stage not selector+hybrid.

---

## T4 — MoE LL contract: packed FP8 → masked DeepGEMM, delete glue  (blocked by T2; ~1500-2000 LOC)

**Goal:** `dsv4_moe_forward_deepep_ll` for decode (`--deepep-mode auto` → LL when not
extend). LL packed receive buffer feeds masked grouped DeepGEMM directly; zero alloc,
zero D2H. Prefill keeps normal-mode dispatch.

**T4.1 deepep-sys LL FFI** (`crates/deepep-sys/`): new structs + extern, mirroring
SGLang `Buffer.low_latency_dispatch/combine` (buffer.py:530-645):
```rust
pub struct LowLatencyDispatchParams {
  num_tokens:u32, hidden:u32, num_topk:u32, num_experts:u32,
  num_max_dispatch_tokens_per_rank:u32,   // env SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, <=1024
  use_fp8:u32,
  d_x:usize, d_topk_idx:usize,            // in: x[num_tok,hidden]bf16, idx[num_tok,topk]i64
  d_recv_x_fp8:usize,                     // out: [E_local, max_tok*world, hidden] e4m3
  d_recv_x_scales:usize,                  // out: [E_local, max_tok*world, hidden/128] f32
  d_recv_count:usize,                     // out: masked_m [E_local] i32 (DEVICE, no D2H)
  d_handle:usize,                         // out: src_info+layout_range (opaque, kept for combine)
}
pub struct LowLatencyCombineParams {
  num_combined_tokens:u32, hidden:u32, num_topk:u32, num_experts:u32,
  d_x:usize,                              // in: [E_local, max_tok*world, hidden] bf16 (GEMM2 out)
  d_topk_idx:usize, d_topk_weights:usize, d_handle:usize,
  d_combined_x:usize,                     // out: [num_combined, hidden] bf16
  compute_stream:usize,
}
extern: arle_deepep_buffer_low_latency_dispatch/_combine
```
`Buffer::new` gains `low_latency_mode:bool` + `num_qps_per_rank = num_experts/world_size`
(deepep.py:209). Wraps `deep_ep` legacy `internode_ll` (wholesale-adopt §4: drop
`-DDISABLE_NVSHMEM`; verify NVSHMEM init under multiproc launcher first).

**T4.2 deepep.rs LL scratch + calls** — pre-alloc once per slot (NOT per call):
```rust
struct DeepEpLlScratch {                         // sized [num_local_experts, max_tok*world, hidden]
  recv_x_fp8:  CudaSlice<u8>,                     // E_local*max_tok*world*hidden
  recv_x_scales: CudaSlice<f32>,                  // E_local*max_tok*world*(hidden/128)
  masked_m:    CudaSlice<i32>,                    // E_local  ← device, feeds GEMM directly
  handle:      CudaSlice<u8>,                     // opaque src_info/layout_range
  gemm1_out:   HiddenStates,                      // [E_local*max_tok*world, 2*I] bf16
  act_fp8:     CudaSlice<u8>, act_scales:CudaSlice<f32>,  // silu_mul_quant out
  combine_in:  HiddenStates,                      // [E_local*max_tok*world, hidden] bf16 (GEMM2 out)
}
fn low_latency_dispatch(..) -> expected_m:usize   // expected_m=(num_tok*world*topk+E)/E (deepep.py:639)
fn low_latency_combine(..)
```

**T4.3 moe.rs `dsv4_moe_forward_deepep_ll`** — sequence (SGLang DeepGemmRunnerCore deep_gemm.py:362-494):
```
route on-device (reuse dsv4_route_device, topk_idx i64 already on device — NO host loop)
→ low_latency_dispatch → (recv_x_fp8, recv_x_scales, masked_m device, expected_m host)
→ grouped_gemm_nt_f8f8bf16_masked((recv_x_fp8,scales),(w13,w13_scale), gemm1_out, masked_m, expected_m)
→ silu_and_mul_masked_post_quant(gemm1_out, act_fp8, act_scales, group=128, masked_m, swiglu_limit)  // JIT, carries clamp
→ grouped_gemm_nt_f8f8bf16_masked((act_fp8,act_scales),(w2,w2_scale), combine_in, masked_m, expected_m)
→ low_latency_combine(combine_in, topk_idx, topk_weights, handle) → out
```
**Delete** moe.rs:1594-1647 (recv_topk_idx D2H + i64→i32 loop + per-call allocs).

**T4.4 masked DeepGEMM bridge** — VERIFY/ADD: ARLE has contiguous-pooled grouped GEMM;
the **masked** variant (`fp8_m_grouped_gemm_nt_masked`, takes `masked_m` device + `expected_m`)
must exist in `csrc/gemm/deepgemm_native.cu`. If absent, add the bridge (vendored deepgemm
supports it). `silu_and_mul_masked_post_quant` masked variant likewise (current
`dsv4_deepgemm_swiglu_quantize_w13` is dense, not masked — needs `masked_m` arg + the
`[E, tok, hidden]` 3-D layout).

**Buffers mutated (enumerate):** all of `DeepEpLlScratch` (pre-alloc per slot at boot,
overwritten each decode step — masked_m/recv_count written by dispatch, gemm1_out/act/
combine_in by their kernels; no snapshot needed, fully overwritten before read). Route
idx/weights reuse the existing on-device `dsv4_route_device` outputs (no new D2H).

**Gate:** ≥2-prompt needle retrieval; batched (c=8/16/32) tok/s vs allreduce; nsys shows
no `clone_dtoh` in the MoE window, no per-layer `cuMemAlloc`.

---

## T5 — CUDA-graph decode + spec  (after T4)

**Persistent decode buffers** (port `DecodeInputBuffers`, cuda_graph_runner.py:174-280):
pre-alloc stable device addrs for input_ids[max_tok]i64, req_pool_indices[max_bs]i64,
seq_lens[max_bs]i32, out_cache_loc[max_tok], positions[max_tok]i64,
next_token_logits[max_tok,vocab]f32, global_num_tokens_gpu[dp_size]i32; replay memcpys
live batch in. **DeepEP mode replay** (DeepEPCudaGraphRunnerAdapter :1422-1439): capture
resolved mode (decode→LOW_LATENCY), re-pin before replay. `SGLANG_PREP_IN_CUDA_GRAPH`
lazy metadata upgrade (backend:922-934) optional. DSv4 graph is `unsupported` today
(start_pos host param, per-step scratch, uncaptured NCCL — research L578); T4's pre-alloc
removes the scratch blocker. **Spec:** EAGLE topk=1 target-verify buckets, report
accepted-output-token TPOT separately from target-step TPOT.

---

## Per-tranche gate (every tranche)
```
1. startup contract log proves the active path   2. decode emits real text, not token-ids
3. ≥2-prompt greedy needle retrieval (NOT byte-identity vs baseline — MoE non-det)
4. request_trace names the expected stage set     5. tok/s in the right lane (batched for DeepEP)
6. regress → revert or mark debug-only
```

## T0 compile-accel — DONE
`/data01/build/arle_build_fast.sh`: `RUSTC_WRAPPER=sccache` (rust, ~1GB warm) +
`ARLE_NVCC_WRAPPER=sccache` (nvcc) + `DG_JIT_CACHE_DIR=/root/.cache/deep_gemm` (15M warm,
skips 10-20m DeepGEMM warmup) + `--features cuda,nccl,deepep` (reuses cached
`libarle_deepep.a`; verified `deepep-sys` seconds-recompile). Prebuilt-archive symbol
gate already in `cuda-kernels/build.rs:1502-1594`. Feature-forward landed `ae889cce`.

## Risks / kill-criteria
- **B=1 masquerade:** measure only batched; B=1 "loss" is by construction.
- **NVSHMEM boot (T4):** verify LL `internode_ll` inits under multiproc before code (wholesale-adopt §4 same-process-timeout).
- **Masked DeepGEMM/act-quant may be absent (T4.4):** verify bridges before T4.3, else add first.
- **T2 is weeks:** if post-T1 batched DeepEP-normal already shows over-transport dominating, that quantifies T2 ROI before committing.
