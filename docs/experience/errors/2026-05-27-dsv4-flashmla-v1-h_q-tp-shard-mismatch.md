# FlashMLA SM90 sparse prefill kernel hard-asserts h_q % 64 == 0 — incompatible with ARLE's TP-sharded Q layout

## SLO-shape probed?  Y — 28899-token prefill, the same shape as the 2026-05-27 baseline + GEMM bench

## Roofline check

Not measurable — V1 dispatch aborted before any wall-clock could be attributed. Listed for §7.6 completeness:

| Op | Achieved | Peak | % | Verdict |
|---|---:|---:|---:|---|
| FlashMLA prefill (intended) | n/a (aborted) | ~640 TFlops/H20 SM90a | n/a | **KILL — preconditions not met** |

## Context

After the 2026-05-27 grouped-GEMM result (`2026-05-27-dsv4-grouped-gemm-marginal-prefill-kernel-not-blocker.md`) showed kernel shape was NOT the dominant blocker (variant A and B both = 282s), the next axis was the attention kernel. SGLang/DeepSeek's open-source FlashMLA was the obvious target — V4 production serving uses exactly its SM90 sparse prefill kernel.

Plan (Option B): vendor sgl-project/FlashMLA @ df022eb into ARLE, write `extern "C"` shim around `sm90::run_fwd_kernel(SparseAttnFwdParams)` skipping FlashMLA's PyTorch-bound public interface, dispatch CSA-mode prefill through it. V1 scope deliberately narrow — only mode==CSA, only token_count>1, sliding-window skipped, attn_sink nullptr — to A/B the kernel before broader integration.

Commits land cleanly (`bbd23a20` vendor, `2675f3a4` kerutils-include fix, `3b7808ee` dispatch wiring). Pod CUDA build PASS in 9m 15s, all 5 FlashMLA SM90 sparse prefill .o files (`fwd.cu` + 4 `phase1_k{512,576}{,_topklen}.cu`) compiled clean.

## Hypothesis

Expected V1 prefill at 28899 tokens to drop from 282s → 100-200s (only the 21 CSA layers reach FlashMLA; the 20 HCA + 2 SWA layers still take the per-token kernel, so partial speedup).

## What actually happened

V1 server boot OK, model load OK, request admitted. First chunked-prefill step calls `arle_flashmla_sm90_sparse_prefill_fwd` → FlashMLA SM90 kernel internally asserts:

```
Assertion `params.h_q % B_H == 0` failed
  (vendor/flashmla/csrc/sm90/prefill/sparse/instantiations/../phase1.cuh:579)
fatal runtime error: Rust cannot catch foreign exceptions, aborting
```

`B_H` is a compile-time constant in FlashMLA = 64 (csrc/sm90/prefill/sparse/config.h:26). DSv4-Flash has `n_heads = 64` (from config.json). ARLE runs with TP=8 → local `h_q = 64 / 8 = 8` per rank. **8 % 64 = 8 ≠ 0** → assertion fires at runtime. Because the assertion is C++ `throw`, propagating through the Rust FFI boundary aborts the entire process — no graceful fallback path possible from inside the kernel call.

The shim's `try { sm90::run_fwd_kernel(...); } catch (const std::runtime_error&) { ... }` doesn't catch this — the throw site is inside CUTLASS/FlashMLA internals and the exception type doesn't match the catch.

## Root cause

FlashMLA SM90's sparse-prefill kernel was designed for the **post-allgather** Q layout used by SGLang's V4 backend, where every TP rank holds the full 64-head Q (TP-AllGather happens before the attention call, ReduceScatter after). ARLE currently passes Q already TP-sharded (8 heads/rank for TP=8), which violates the kernel's tile-block assumption. There is no `B_H=8` instantiation in the upstream FlashMLA tree, and patching the tile size into a fork would defeat the "inherit upstream tuning" rationale of Option B.

Same constraint applies to DSv3 (128 heads, TP=8 → 16 heads/rank — also % 64 ≠ 0). All TP configurations of DS-series models hit this assertion unless h_q is post-allgather (full 64+).

## Fix path (V2)

FlashMLA integration requires ARLE to emit AllGather before attention and ReduceScatter (or All-to-All) after. Concretely for V4 CSA layers:

1. Pre-FlashMLA: `q_local [B, S_q, 8, D] → allgather across TP → q_full [B, S_q, 64, D]`
2. Call `arle_flashmla_sm90_sparse_prefill_fwd(q_full, ...)` with h_q=64
3. Post-FlashMLA: `out_full [B, S_q, 64, D_v] → reduce-scatter / take rank slice → out_local [B, S_q, 8, D_v]`

This is the same primitive SGLang and vLLM use; it's well-trodden TP attention pattern. ARLE has NCCL allreduce for MoE expert outputs already, so AllGather is an add — not a from-scratch primitive.

Estimated scope: ~1-2 days (NCCL AllGather wrap, output slice, plumbing through dispatch site, end-to-end validation).

Alternative (cheaper diagnosis): bench FlashMLA with TP=1 first to confirm the underlying kernel works at scale, before committing to the AllGather work. Requires either a much smaller DSv4 fixture or a fits-on-one-GPU model — neither is set up locally; deferred unless the AllGather path turns out to be blocker-rich.

## Rule

**Vendor-kernel integration must include a precondition audit before wiring.** I checked d_qk (∈ {512, 576}, ARLE has 512 — OK), checked head_dim_v (= d_qk for MLA), checked stride conventions — but did NOT check `h_q` divisibility nor TP-shard layout assumptions. Result: shim compiles, build passes, kernel aborts at runtime. The constraint was visible at `config.h:26` and `phase1.cuh:579`; I read those files during research but didn't run "what's h_q at the call site vs what does the kernel require?" specifically. Adds a SOLID-§0-style "check called-with shape vs documented contract" step to the next-vendor checklist.

The Rust-aborts-on-C++-throw symptom is a separate concern: any C++ throw across the `extern "C"` boundary becomes a process kill, not a recoverable error. Future shims should either: (a) wrap throw-prone paths with a status-code return + internal catch, OR (b) ensure the upstream code never throws on a known-bad input (validate inputs and return early at the shim layer). Caught here as `cudaErrorInvalidValue` only when the throw goes through `<stdexcept>` — CUTLASS internal aborts (via `cuteAssert` or similar) skip C++ exception handling entirely.

## Status

- V1 wire-up correct + builds clean. Cannot run in ARLE's TP-sharded path.
- V2 = AllGather/ReduceScatter integration. Not yet started. Tracked as `#43`.
- 282s baseline still in effect.

## Refs

- `3b7808ee` — V1 dispatch wiring (committed, reverts not needed; the env flag defaults OFF)
- `bbd23a20` — FlashMLA vendor import
- `crates/cuda-kernels/vendor/flashmla/csrc/sm90/prefill/sparse/config.h:26` — `static constexpr int B_H = 64;`
- `crates/cuda-kernels/vendor/flashmla/csrc/sm90/prefill/sparse/phase1.cuh:579` — the failing assertion
- `2026-05-27-dsv4-grouped-gemm-marginal-prefill-kernel-not-blocker.md` — why this axis was picked
- Server log: `/sgl-workspace/arle-fresh/docs/trace-artifacts/2026-05-27-dsv4-flashmla-v1/server.log`
