# DSv4 SGLang path mismatch invalidated operator-first optimization

## SLO-shape probed? -- N (path-contract reset only; no new SLO bench)

## Context

Goal: make ARLE's DSv4 serving path comparable with SGLang before claiming
`>20% vs SGLang` or spending more time on isolated operator rooflines.

The user flagged that optimizing one item at a time was repeating the earlier
mistake: the measured ARLE path was not the SGLang-equivalent path. Treating
`attn_csa_select_kernel`, FFN all-reduce, or native DeepEP combine as separate
local bottlenecks hid the larger mismatch in rank layout, token ownership,
attention batching, KV format, MoE transport, graph capture, and speculative
decode framing.

The reference target must also be framed correctly. If the SGLang reference is
`18 ms` TPOT for the same raw target-step metric, then `>20% vs SGLang` means
ARLE must reach `<=14.4 ms` raw target-step TPOT. The earlier `60-64 ms` target
framing was not acceptable for that reference. Raw target-step TPOT and
speculative or effective TPOT must be reported as separate metrics.

No new benchmark was run for this entry. This entry records the path-contract
RCA so later work does not convert non-comparable traces into SLO evidence.

## Root Cause

ARLE's current DSv4 path is a replicated-token TP/EP implementation:

- every rank owns the same token rows for the request;
- attention still runs through a per-row path instead of a fully batched
  FlashMLA paged-FP8-KV path;
- routed experts run locally on each rank's expert shard;
- FFN output is reconciled with a hidden-state all-reduce;
- native DeepEP is unsafe as a default because the caller does not provide
  distinct token rows per EP rank.

That path can be correct as a fallback, but it is not the path SGLang is using.
The SGLang DSv4 path is organized around a TP/DP-style rank layout, batched
FlashMLA over paged FP8 KV, native DeepEP or MegaMoE MoE transport, graph or
in-graph metadata preparation, and optional EAGLE/MTP for speculative decode.

This makes the whole ARLE breakdown structurally unreasonable:

- Native DeepEP combine/transport is unreasonable because DeepEP dispatch and
  combine assume token-sharded EP ownership. Passing the full replicated token
  matrix from every TP/EP rank makes DeepEP transport duplicate rows across
  ranks, then pay combine and synchronization costs on top of the duplicate
  fanout.
- FFN all-reduce is unreasonable as a performance target because it is the
  reconciliation mechanism for replicated-token local experts. It moves full
  hidden states after routed expert execution and scales with the fallback data
  contract. It is a correctness fallback, not the SGLang-equivalent MoE path.
- Attention timing is unreasonable because per-row attention, CSA selection,
  cache metadata preparation, and KV writes are not fused into the batched
  FlashMLA paged-FP8-KV contract. A local CSA selector win can lower one
  launch, but it does not change the rank layout, batching contract, or KV
  memory traffic that dominate comparability with SGLang.
- Operator-level deltas are therefore not additive to an SGLang comparison.
  They describe the fallback path's internal costs, not the target path's
  roofline.

The root cause was an optimization-order error: the work treated operator
roofline items as the main queue before first locking the SGLang-equivalent
path contract.

A later user-supplied vLLM/SGLang trace reference sharpens the priority after
that contract is fixed: MoE MLP, expert GEMM, EP dispatch/combine, and buffer
materialization are the first SGLang-path roofline targets. Attention/CSA stays
important for the current ARLE fallback route, but it is not accepted as the
path-aligned P0 without a fresh matched trace.

## Fix

Reset the DSv4 optimization order:

1. Define and verify the path contract first: rank layout, token ownership,
   attention batching, KV dtype/layout, MoE transport, graph metadata, and
   decode mode.
2. Keep the replicated-token all-reduce path as a correctness fallback and do
   not use it for SGLang-relative performance claims.
3. Do not default native DeepEP until the caller owns distinct token rows per
   EP rank. DeepEP is a transport/data-layout contract, not a faster all-reduce
   replacement.
4. Only after the path contract is matched, run operator rooflines on that
   matched path.
5. Report raw target-step TPOT separately from speculative or effective TPOT.
   If SGLang uses EAGLE/MTP, the acceptance rate and effective output-token
   metric must not be mixed with raw model-step TPOT.

This is not a claim that ARLE exceeds SGLang. It is a stop-rule entry that
prevents the old non-comparable path from being used as evidence.

## Rule

Path contract before operator roofline. For DSv4, no `>20% vs SGLang` claim is
valid until ARLE and SGLang are matched on rank layout, token ownership,
attention/KV path, MoE transport, graph metadata, workload shape, and TPOT
definition. If any of those differ, the result is a path-mismatch trace, not a
competitive benchmark.
