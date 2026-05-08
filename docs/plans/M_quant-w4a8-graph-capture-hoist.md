# M_quant W4A8 graph capture hoist — unlock default-on flip

> Trigger: W4A8 Marlin substrate LANDED (`e61d26e`) with mixed bench:
> TTFT -36% vs W4A16 ✅ but ITL +63% (eager-mode penalty) ❌. Default-on
> production flip blocked on CUDA Graph capture for W4A8 decode path.
>
> Owner: codex (≥200 LOC substrate work — scratch buffer hoisting in
> linear dispatch + decode buffer plumbing).
> Master strategy §6.1 5-cap moat: this unlocks W4A8 as the production
> decode default once ITL recovers to ~12 ms (parity with W4A16).

## Current blocker (grounded in code)

`infer/src/model/qwen3/forward.rs:735-742`:

```rust
fn supports_cuda_graph_decode(&self) -> bool {
    // LoRA decode allocates per-call temp DeviceVecs inside
    // `apply_lora_{gemv,gemm}_add`; CUDA stream capture rejects those.
    // W4A8 Marlin currently quantizes activations and allocates scratch
    // inside linear dispatch, so it must also stay eager until that
    // scratch is hoisted into decode buffers.
    self.enable_cuda_graph && self.lora.is_none() && !self.uses_marlin_w4a8()
}
```

Routed via `infer/src/scheduler/cuda/core/warmup.rs:32` →
`graph_capture_enabled = self.model.supports_cuda_graph_decode()` →
when false, eager-mode decode at line 42-49 logs "Graph capture disabled
(eager decode, e.g. LoRA)" and skips graph capture for B=1..max_slots.

For Qwen3-4B at c=4 longctx (W4A8 bench), this means decode runs without
graph replay — every step pays full launch overhead instead of replayed
captured graph.

## Phase 1 — Target (skill v1.3.0)

| Field | Value |
|---|---|
| Metric | longctx 4k/c=4 ITL p50 (auto-FP8 KV) |
| Baseline | W4A8 eager-mode ITL 19.23 ms (`e61d26e`) |
| W4A16 reference | 11.76 ms (`f6f3af3`) |
| **License** | ITL ≤ 12 ms (parity with W4A16, recovers eager-mode penalty) |
| Soft win | ITL ≤ 14 ms (-27% vs eager-mode but still slower than W4A16 — partial recovery) |
| Kill | ITL > 18 ms (graph capture didn't help; deeper W4A8 issue) |
| Wall-clock budget | 1-3 days (codex implementation) + 5 min (Claude bench) |

## Phase 2 — Hardware constraints (W4A8 graph specifics)

CUDA stream capture restrictions (per CUDA docs + LoRA precedent in `forward.rs`):
- No `cudaMalloc` / `cudaFree` during capture
- No event-based ordering against host
- Synchronous semantics in capture region map to recorded launches

Implication: W4A8's per-call allocations (`alloc x_int8`, `alloc workspace`,
`alloc out_bf16`) must be **pre-allocated outside the capture region** and
reused per decode step.

## Phase 3 — Binding constraint (formula-grounded)

W4A8 eager-mode per-decode-step cost (formula):
```
36 layers × 7 GEMMs × ~5 launches per Marlin call = 1260 launches per token
× ~5-10 us cudaLaunchKernel overhead = 6-12 ms launch overhead per token

W4A8 ITL bench: 19.23 ms
Subtract 7-10 ms for launch overhead → kernel-time ~9-12 ms
Compare W4A16 with graph capture: ~5 ms launches replayed + ~7 ms kernel = 12 ms ITL
                                  (matches f6f3af3 11.76 ms within noise)
```

Eager-mode launch overhead is the binding constraint. Removing it via graph
capture should bring W4A8 ITL to ~12 ms — parity with W4A16 (since both share
~9-12 ms underlying kernel time, dominated by HBM bandwidth on weight read).

## Phase 4 — Formula prediction

```
W4A8 graph-capture-enabled ITL_lower = kernel_time + replay_overhead
                                     ≈ 11 ms + 0.5-1 ms
                                     = 11.5-12 ms

vs W4A16 baseline 11.76 ms: parity expected (same 2 GB W4 weight,
                            same FP8 KV, similar kernel architectures)

Δ vs current eager W4A8 19.23 ms: -38% to -40%
Δ vs W4A16 11.76 ms: ±2% (parity)
Δ vs BF16 19.27 ms: -38% to -40%
```

The win after this hoist:
- TTFT: still -36% vs W4A16 (FP8 mma compute kept)
- ITL: parity with W4A16 (recovers eager penalty)
- Net: W4A8 becomes strict pareto improvement over W4A16 → production-default flip licensed

## Phase 5 — Implementation (codex own, ~200-400 LOC)

Three sub-changes:

### 5.1 Hoist x_fp8 / workspace / output_int32 buffers to decode context

`infer/src/model/qwen3/forward.rs` Qwen3DecodeContext or similar — add scratch
fields for W4A8:

```rust
struct Qwen3DecodeContext {
    // ... existing
    w4a8_x_int8_scratch: Option<CudaSlice<i8>>,        // [max_batch * max_hidden]
    w4a8_workspace_scratch: Option<CudaSlice<i32>>,    // [marlin_workspace_size]
    w4a8_output_int32_scratch: Option<CudaSlice<i32>>, // [max_batch * max_n]
}
```

Pre-allocated at decode context creation; reused per step.

### 5.2 Modify `run_marlin_w4a8` to accept pre-allocated scratch

`infer/src/ops/linear.rs run_marlin_w4a8()`:

```rust
fn run_marlin_w4a8_into_scratch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    scratch: &mut W4A8Scratch,  // <-- NEW
) {
    // 1. Use scratch.x_int8 instead of alloc
    // 2. quantize_bf16_rows_to_int8 into scratch
    // 3. Call gemm_w4a8_marlin_cuda with scratch.workspace
    // 4. Use scratch.output_int32 for intermediate
    // 5. Convert int32 → BF16 into `out`
}
```

Equivalent dispatch arm in `Qwen3DecodeContext::forward_decode()` passes its
scratch to each linear call.

### 5.3 Flip `supports_cuda_graph_decode` for W4A8

`infer/src/model/qwen3/forward.rs:741`:

```rust
fn supports_cuda_graph_decode(&self) -> bool {
    self.enable_cuda_graph && self.lora.is_none()
    // W4A8 hoist landed (commit ID), no longer requires eager mode
}
```

Comment block in code documents the hoist commit + bench evidence.

## Phase 6 — Combinational A/B (post-implementation)

Bench identical 3-arm matrix as W4A8 prod bench `e61d26e` to confirm hoist
delivered:

| Arm | Expected ITL | Expected TTFT |
|---|---|---|
| A BF16 | 19.27 ms | 1976 ms |
| B W4A16 Marlin | 11.76 ms | 2565 ms |
| C W4A8 eager (current `e61d26e`) | 19.23 ms | 1633.6 ms |
| **C' W4A8 graph-captured (post-hoist)** | **~12 ms predicted** | **~1633 ms (TTFT unchanged)** |

If C' achieves ITL ≤ 12 ms AND TTFT preserved → **production-default flip
W4A8 wins over W4A16 on both metrics**.

## Phase 7 — Tradeoffs (skill v1.3.0)

| Axis | Status | Note |
|---|---|---|
| LOC | ⚠ ~200-400 (codex substrate) | scratch hoist + dispatch refactor |
| Numerical correctness | ⚠ greedy_consistency must verify scratch reuse doesn't break | mandatory test |
| Memory budget | ⚠ +scratch buffer (~10-50 MB depending on max_batch × max_n) | acceptable on 16 GB GPU |
| Maintainability | ⚠ minor | scratch lifetime ties to decode context (similar to LoRA pattern but inverted) |
| Generality | ⚠ multi-shape gate | high-conc + multi-tenant + longctx-8k all need re-bench post-hoist |
| Hardware specificity | ✅ none | scratch-hoist pattern is GPU-arch-agnostic |
| **CUDA graph capture compatibility** | ✅ design unblocks it | per cuda-stream-capture rules met |

## Phase 8 — License decision

| Δ vs current W4A8 eager (19.23 ms) | Action |
|---|---|
| ITL ≤ -35% (≤ 12.5 ms) | LAND HARD — flip W4A8 production default. W4A8 is strict pareto improvement vs W4A16. |
| ITL -25% to -35% (12.5-14.4 ms) | LAND incremental — W4A8 default for prefill-heavy; W4A16 retained for decode-heavy |
| ITL -10% to -25% (14.4-17.3 ms) | partial — debug remaining gap, do not flip default yet |
| ITL > -10% (≥ 17.3 ms) | KILL — graph capture not the dominant overhead; deeper investigation |

## Cross-references

- W4A8 substrate LAND (mixed outcome): [`docs/experience/wins/2026-05-08-w4a8-marlin-prod-bench-mixed-outcome.md`](../experience/wins/2026-05-08-w4a8-marlin-prod-bench-mixed-outcome.md) (`e61d26e`)
- W4A8 prod bench plan: [`M_quant-w4a8-prod-bench.md`](M_quant-w4a8-prod-bench.md) (`db573c5`)
- ARLE eager-mode classification: `infer/src/model/qwen3/forward.rs:735-742`
- ARLE warmup graph dispatch: `infer/src/scheduler/cuda/core/warmup.rs:32-49`
- W4A8 Marlin kernel + activation quant: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` + `w4a8_activation_quant.cu` (`a019a0e`)
- LoRA precedent for scratch hoist pattern: `infer/src/ops/linear.rs::apply_lora_gemv_add` (currently per-call, future hoist target too)
- Skill v1.3.0 anti-pattern #12: this plan validates that hybrid dispatch is wrong; the right fix is removing per-call overhead, not bypassing tensor-core path

## Rule

The W4A8 mixed outcome (TTFT win + ITL eager-mode penalty) is solvable
**without** introducing hybrid dispatch (R4 #6 KILL evidence). The right
move is hoisting the per-call allocations out of capture region — same
fundamental pattern that LoRA will eventually need. This plan keeps W4A8
on the tensor-core path (consistent with skill v1.3.0 anti-pattern #12)
and unlocks graph capture by removing the cudaMalloc-during-capture
restriction.
