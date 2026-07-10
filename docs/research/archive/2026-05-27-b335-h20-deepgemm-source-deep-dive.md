# B-3.3.5 H20 DeepGEMM 崩溃 — source-level deep dive (no cuda-memcheck)

**Status**: research. Source-level evidence + hypothesis ranking. No pod
verification this session.

**Trigger**: 1164f35d errors entry left three suspected root causes for the
H20 runtime crash ("unspecified launch failure" after JIT compile finally
passed in 38bf157b). Today I read all relevant call sites + kernel sources
to see what can be SOLID'd from code alone before paying for cuda-memcheck
time.

**Scope**: only B-3.3.5 wire-in (`forward_native_deepep_routed_gpu`'s
DeepGEMM branch, `mlp.rs:5170-5196`) vs baseline (`forward_deepep_routed_gpu`'s
DeepGEMM path, `mlp.rs:3975-4054`). Intranode + DeepGEMM kernels only.

---

## TL;DR

1. **Confirmed correctness bug (source-level evidence)**:
   B-3.3.5's DeepGEMM branch feeds `dsv4_scatter_all_route_slots_cuda` with
   `packed_token` as `expert_route_slot`. The scatter kernel does
   **assignment** (`route_out[token_idx] = w * v`), not accumulation. With
   topk=8 and multiple local experts per recv-token, **7/8 of the
   contributions are silently overwritten** (race + last-writer-wins).
   Combine then aggregates wrong values → garbage output. Not a crash.

2. **NOT confirmed source-level**: the H20 "unspecified launch failure" most
   likely traces to one of the four DeepGEMM JIT'd kernels (TMA / cluster
   path on H20 vs H100). Requires cuda-memcheck on pod. The scatter bug
   above is *separately* real, but is not in itself a crash candidate.

3. **DeepEP LL mode integration scope** (the actual nsys-supported lever,
   ~+15-19 ms TPOT): ~1500-2500 LOC + NVSHMEM install + container rebuild.

---

## 1. Confirmed bug — scatter assignment vs accumulation

### Evidence — kernel source

`crates/cuda-kernels/csrc/moe/dsv4_route.cu:1092-1110`:

```cuda
__global__ void dsv4_scatter_all_route_slots_kernel(...) {
  ...
  int route_slot = expert_route_slot[route];
  if (route_slot < 0) return;
  float weight = expert_weight[route];
  float value = dsv4_route_bf16_to_f32(expert_out[idx]);
  route_out[route_slot * hidden_dim + col] =
      dsv4_route_f32_to_bf16_bits(weight * value);  // ASSIGNMENT
}
```

Compare to the native scatter `dsv4_scatter_packed_expert_kernel`
(`dsv4_route.cu:526`):

```cuda
float prev = dsv4_route_bf16_to_f32(routed_out[out_idx]);
float value = dsv4_route_bf16_to_f32(expert_out[idx]);
routed_out[out_idx] = dsv4_route_f32_to_bf16_bits(prev + weight * value);
// ACCUMULATION
```

### Evidence — write-destination semantics

The two pack kernels differ in what they store as the per-slot destination
index:

| Kernel | Stored index | Source layout |
|--------|---|---|
| `dsv4_pack_received_experts_kernel` (baseline) | `expert_route_slot[slot] = route` where `route ∈ [0, num_routes)` | recv buffer already pre-expanded per route — each (token, expert) is a unique row |
| `dsv4_pack_local_experts_kernel` (B-3.3.5) | `packed_token[slot] = token` where `token = route / topk ∈ [0, num_recv)` | recv buffer is unique-token packed — multiple topk experts per token share one row |

In the **baseline**, every `expert_route_slot[i]` is unique, so the
assignment-scatter writes each row exactly once (no race, no lost data).
Combine reads route_out as "per-route" and aggregates per-token internally.

In **B-3.3.5**, `packed_token[i]` is duplicated for topk>1 (every local
expert this token routes to creates a slot with the same `packed_token`
value). The assignment-scatter creates a data race: GPU threads writing
to the same row stomp each other. Best case: one expert's contribution
survives; topk-1 are lost.

### Combine side expects accumulated values

B-3.3.5 combine call at `mlp.rs:5263-5286`:

```rust
let combine_params = deepep_sys::CombineParams {
    num_input_tokens: num_recv as u32,        // expert_out indexed [0, num_recv)
    num_output_tokens: hidden.seq_len as u32,
    ...
    d_x: eo_ptr as usize,                     // scratch.expert_out
    ...
};
```

`num_input_tokens = num_recv` means combine reads `expert_out[i]` for
`i ∈ [0, num_recv)`. For each `i`, the value must be the **sum over all
local experts** this recv-token routed to. The native scatter
(`dsv4_scatter_packed_expert_cuda`) achieves this via accumulate. The
DeepGEMM scatter (`dsv4_scatter_all_route_slots_cuda`) does not — it
overwrites.

Confidence: **high**. Three independent code locations
(pack kernel, scatter kernel, combine wiring) cross-verify the contract
violation.

### Impact

- Silent garbage decode output (lost 7/8 of expert contributions for topk=8).
- Not a crash by itself — race + assignment is benign on GPU memory
  (no fault, just clobber).
- Means even if H20 GEMM launch were fixed, perf "wins" would not be
  measurable as quality regression masks any throughput gain.

### Fix candidates

| Option | Effort | Tradeoff |
|--------|--------|----------|
| (A) Pre-zero `route_out`, then build an `_accumulate_` variant of `dsv4_scatter_all_route_slots_cuda` with atomicAdd on bf16 (or f32 staging) | ~80 LOC | Real fix; atomic bf16 cost vs the lost-perf cost trades favorably given the GEMM speedup |
| (B) Reuse `dsv4_scatter_packed_expert_cuda` per-expert loop instead of one-shot scatter | ~30 LOC | Loses the single-launch advantage but matches native path semantics |
| (C) Insert an aggregator kernel (`packed_token` → `expert_out` with f32 staging buffer + final cast) | ~60 LOC | Clean separation; small extra mem cost |

Recommendation: (A). Atomic bf16 is awkward but the gather-shape is
predictable (each token row is touched ≤topk times), so a 32-bit-aligned
read-modify-write pattern works without true atomicity (since per-row
contributors are deterministic from indexing).

Cannot ship the fix without H20 access to verify it doesn't make things
worse — but the design is ready to bake when the pod is back.

---

## 2. H20 crash hypothesis — still source-level unresolved

### Ruled out by source reading

- **`expert_hidden.seq_len` staleness** (one of three candidates in
  1164f35d): the grouped DeepGEMM kernel does not read seq_len, only
  raw pointer + `active.offsets`/`counts` from
  `dsv4_prepare_deepgemm_all_expert_metadata_cuda`. So even though
  B-3.3.5 leaves `packed_x.seq_len = capacity_local_routes` (way larger
  than `total_local_routes`), no OOB downstream.

- **Pack kernel mechanics**: both pack kernels produce valid per-slot
  layouts. The pre-fill `dsv4_fill_i32_cuda(-1)` sentinel that the
  baseline uses is not strictly required because the consumer walks
  exactly `total_local_routes` entries (counted-up by atomicAdd
  cursors). Skipping it ≠ crash.

- **Scratch sizing**: `ensure_deepgemm_scratch(capacity_experts=32,
  capacity_m=total_local_routes, hidden=4096, intermediate=2048)` for
  640 local routes per rank gives ~440 MB per allocation, well within
  H20 (94 GB). Probably not OOM.

- **Alignment**: `hidden_dim=4096` and `intermediate_dim` ÷ 128 cleanly.
  `scale_stride_m = total_local_routes.div_ceil(4) * 4` is f32-aligned.
  No source-level mis-alignment.

### Still suspected (cuda-memcheck required)

- **DeepGEMM FP8 GEMM kernel: H100-only TMA/cluster path**. DeepGEMM's
  JIT'd kernel may use `cute::TmaCopy` with cluster sizes that work on
  H100 SM_90a but fail on H20 (same SM_90 ISA but reduced HBM B/W +
  some collectives behave differently). The error is "unspecified launch
  failure" with no kernel name — generic device-side trap.
- **Possible interaction with B-3.3.5's race-prone scatter** above: if
  the assignment-scatter produces NaN/Inf bf16 values that flow into a
  downstream check, could surface as a kernel param invalid in the next
  iteration. Low probability — the scatter writes a single layer's
  output; combine then immediately consumes it; subsequent layers run
  fresh GEMMs.

### Plan for next session (when pod is back)

1. Wrap **single-request prefill** with
   `compute-sanitizer --tool memcheck` → exact kernel name + line of
   illegal access. ~30 min.
2. If memcheck pinpoints DeepGEMM GEMM kernel: try lowering DeepGEMM's
   cluster size via env var (DeepGEMM has tuning knobs).
3. Concurrently: ship the scatter-accumulate fix locally so when GEMM
   unblocks, perf measurement is meaningful.

---

## 3. DeepEP LL mode — real scope estimate

### What's wired today

- `crates/deepep-sys/csrc/deepep_buffer.{hpp,cpp}`: thin C wrapper
  around `deep_ep::Buffer` for intranode dispatch + combine. ~600 LOC.
- `crates/deepep-sys/build.rs`: NVSHMEM **explicitly disabled** via
  `-DDISABLE_NVSHMEM` flag (line 90). LL kernels are skipped from
  source list. `internode_ll.cu` (1289 LOC in DeepEP legacy tree) is
  not compiled.

### What needs to change for LL

| Step | Effort |
|------|--------|
| 1. NVSHMEM container install (≈500 MB, ucx + ibgda deps) + image rebuild | external |
| 2. Drop `-DDISABLE_NVSHMEM` from build.rs; add `internode_ll.cu` to sources | ~10 LOC |
| 3. New C wrapper: `low_latency_dispatch` + `low_latency_combine` + persistent counter setup at Buffer init | ~400-600 LOC |
| 4. Rust binding layer in `deepep-sys/src/`: extend `Buffer` with LL methods, new `LowLatencyDispatchParams` / `LowLatencyCombineParams` structs | ~200 LOC |
| 5. New forward path in `mlp.rs` (`forward_native_deepep_ll_routed_gpu`) — different metadata layout (LL packs (rank, expert) pairs differently), FP8 input/output (LL requires FP8 hidden), persistent buffer recycling | ~400-600 LOC |
| 6. Scheduler integration: `--moe-backend=native-deepep-ll` flag + boot-time NVSHMEM init via `nvshmemx_init_attr` + IBGDA setup | ~200-300 LOC |
| 7. Test fixtures: parity vs native-deepep on smoke prompt, A/B per nsys | ~150 LOC |

**Total ARLE-side**: ~1500-2000 LOC. **External**: NVSHMEM install on
production pods.

### Expected payoff (from nsys trace)

cached_notify_combine 26% of GPU time + collective-wait in intranode
combine ≈ 30-40% of per-token wall time. LL replaces both with
NVSHMEM persistent counters — empirically (DeepSeek paper) ~10× faster
notify path. Translates to ~15-19 ms TPOT reduction on current 65 ms
per-token (≈ 23-30% throughput uplift).

### Risk

- NVSHMEM IBGDA path requires Mellanox/CX-6 or newer with GDR enabled.
  H20 pods should have it (DSv4-Flash recommends it) but needs
  verification.
- LL FP8 hidden requires reuse of input quant cache across dispatch +
  GEMM — currently we re-quantize twice (once for dispatch nominal,
  once for GEMM). Need a shared quant pass.

---

## Decision tree for next session

```
Pod access available?
├── YES
│   ├── compute-sanitizer single-request prefill (DeepGEMM enabled)
│   │   ├── pinpoints kernel → fix in DeepGEMM JIT or downgrade cluster
│   │   │   ├── fix lands → ship scatter-accumulate fix (option A) too
│   │   │   └── re-bench → expect +50-80% per-layer FFN, ~+8-12 ms TPOT
│   │   └── inconclusive → switch to LL track
│   └── NVSHMEM install feasible? (ask infra)
│       ├── YES → LL track (≈2 weeks ARLE-side, ~+15-19 ms TPOT)
│       └── NO → DP-attention track (no NVSHMEM dep, ~+10-13 ms TPOT,
│             ≈1.5 weeks)
└── NO (today)
    └── DOC + DESIGN only. Ship scatter-accumulate-fix design ready
        for when pod returns. (this entry)
```

---

## Refs

- 1164f35d errors entry — three H20 crash candidates
- 38bf157b — c++17 nvcc fix (real JIT root cause)
- 67ac6400 — B-3.3.5 wire-in commit (the buggy DeepGEMM branch)
- `mlp.rs:5170-5196` — B-3.3.5 DeepGEMM branch
- `mlp.rs:3975-4054` — baseline DeepGEMM device-counts path
- `mlp.rs:2279-2585` — `forward_deepgemm_grouped_dsv4_experts_gpu`
- `mlp.rs:2587-2655` — `forward_deepgemm_all_dsv4_experts_gpu`
- `dsv4_route.cu:461-524` — `dsv4_pack_local_experts_kernel`
- `dsv4_route.cu:994-1049` — `dsv4_pack_received_experts_kernel`
- `dsv4_route.cu:526-575` — `dsv4_scatter_packed_expert_kernel` (native, accumulate)
- `dsv4_route.cu:1092-1130` — `dsv4_scatter_all_route_slots_kernel` (DeepGEMM, assign)
- `state.rs:805-905` — `ensure_deepgemm_scratch`
- `state.rs:538-642` — `ensure_native_deepep_scratch`
- `/private/tmp/pod-stage/DeepEP/csrc/kernels/legacy/internode_ll.cu` (1289 LOC) — LL kernel reference
- `crates/deepep-sys/build.rs:90` — `-DDISABLE_NVSHMEM` flag (where LL is currently dead)
