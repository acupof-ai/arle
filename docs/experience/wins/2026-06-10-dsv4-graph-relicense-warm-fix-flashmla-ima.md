# Graph re-license: universal warm-before-capture lands (+6.7% on !flashmla base); FlashMLA capture is BROKEN on today's tree (IMA) — the next engineering item

**Date:** 2026-06-10 (night). **Backend:** CUDA DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commits:** `4b835fa4` (universal warm fix), `0ba67106` (gate lift). Same binary
per A/B, env-flips, serial arms, `dsv4_ab_bench.py` B=1.

## Context

The nsys skew anatomy re-licensed the whole-step graph (launch-drizzle 29% +
host-pacer evidence). ck: 直接做好 graph,方案要全局通用.

## What landed (universal, per ck)

`CudaGraphState::run_or_capture` now runs the closure eagerly once before its
first capture (`warm_remaining`, default 1) — at the SHARED layer, so every
graph user (per-portion attn/moe, tail, whole-step, Qwen dense decode, future
models) is JIT-in-capture-safe by construction. Root cause it fixes: DeepGEMM's
lazy `cuModuleLoad` on first shape use is illegal during stream capture (CUDA
error 900, STREAM_CAPTURE_UNSUPPORTED) — the 6-08 captures only worked because
earlier eager traffic had pre-loaded modules (ordering luck, not a property).

## A/B results

| arm (all + GPU_ROUTER=1 ⇒ pooled MoE) | B=1 p50 tok/s | verdict |
|---|---|---|
| eager, FLASHMLA=0 | 25.30 | base |
| **whole-step graph, FLASHMLA=0** | **26.99 (+6.7%)** | works end-to-end, coherent, 8/8 |
| eager FlashMLA masked default (reference) | 38.99 | the bar |
| whole-step graph + FlashMLA | **IMA** (illegal memory access, all ranks) | broken |
| per-portion graph + FlashMLA | **IMA** (12 hits) | **broken too** |

- +6.7% on the slow base = graph removes the non-overlapped host slice there
  (~2.5 ms/step); the base is GPU-paced (~39.5 ms GPU), so this neither
  confirms nor caps the host-pacer prediction for the fast path.
- **Per-portion FlashMLA captured cleanly on 2026-06-08 but IMAs today** → the
  FlashMLA/DSA decode path lost capture-safety somewhere in the changes since
  (official DSA indexer adoption, KV arena/pool restructure, incremental KV,
  dynamic KV budget). Not a whole-step-specific cross-step-state issue.

## Next engineering item (the real "FlashMLA-in-graph")

§0.1 enumeration over TODAY's FlashMLA/DSA decode path: every per-step
HOST-computed value that reaches a kernel (pointer arithmetic or scalar arg)
must become device-derived from `start_pos_device` or a pre-replay H2D update.
Suspect list to walk first: SW sliding-window ring write offsets; DSA index-key
page offsets; FlashMLA arena double-buffer/parity selection; incremental-KV
host-side pointers; metadata buffers reallocated per step. Per-portion mode is
the cheaper debug vehicle (smaller capture, same IMA).

## Rules

- Lazy JIT belongs OUTSIDE capture — solved once at the shared graph layer,
  never per call site (方案要全局通用).
- "X captures cleanly" is a TREE-VERSIONED fact, not a property of X: re-verify
  capture-safety after path rewrites (FlashMLA: clean 6-08 → IMA 6-10).
- Default risk: none — graph remains env-gated off; the gate-lift only makes
  the FlashMLA+graph combination reachable for debugging.

## Refs

- License: [`wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md`](2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md)
- 6-08 evidence pair: whole-step CONCLUSIVE entry + decode-graph-wash entry
- Pod logs: `ab_graph_eager.log`, `ab_graph_wholestep.log`, `ab_fm_{base,graph,pp}.log`
