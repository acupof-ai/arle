# CUDA decode-graph: wire the dead `enable_cuda_graph` flag + default-on (single-GPU)

**Date:** 2026-06-04. **Backend:** CUDA. **Scope:** `crates/infer-cuda/src/executor.rs`,
`crates/infer-cuda/src/lib.rs`, `crates/infer-api/src/loaded.rs`.
**Status:** wiring landed + Mac cuda-rust typecheck/clippy green; **GPU A/B `pending-remote`**
(single H20, queued behind the DSv4 grouped-cache bench — one pod, no overlap).

## Goal

Make the B=1 decode CUDA graph actually reachable in normal serving. It was implemented
+ verified (16/16 eager==replay, task #12) but **never engaged**: the CLI
`--cuda-graph`/`--no-cuda-graph` flag was discarded (`loaded.rs:326` `let _ = enable_cuda_graph`)
and the sole real gate was `INFER_CUDA_DECODE_GRAPH=1`, **default off**. So the graph never
ran unless an operator knew the undocumented env var. This is the "cuda graph 得支持上" gap.

## What changed (wiring, no numeric claim yet)

- `executor.rs`: `decode_graph_enabled()` was env-only. Now `INFER_CUDA_DECODE_GRAPH` is an
  explicit **override** (`1/true/on`→on, `0/false/off`→off); when **unset** it falls back to
  a load-time default `AtomicBool` (init on), set via the new `pub fn set_decode_graph_default`.
- `loaded.rs`: `let _ = enable_cuda_graph;` → `infer_cuda::set_decode_graph_default(enable_cuda_graph)`.
  The CLI default is `!args.no_cuda_graph` = **on**, so single-GPU Qwen dense now captures by default;
  `--no-cuda-graph` turns it off; the env var still overrides for ops.
- `lib.rs`: re-export `set_decode_graph_default` (cuda-gated).

**Unchanged guards (correctness floor):** `warmup()` still hard-disables the graph under TP
(NCCL all-reduce is not graph-capturable → a captured graph would skip the collective = wrong
logits) and under MoE (host routing per step). So this only flips the default for **single-GPU
Qwen dense decode**; DSv4 (TP=8) and Qwen3.5/3.6-MoE are unaffected.

## Params / Env (A/B to license the default — pending-remote)

- SKU: single H20 (1 GPU). Model: Qwen3 dense (BF16). Single-user decode focus.
- A/B, same binary: `INFER_CUDA_DECODE_GRAPH=1` (graph) vs `=0` (eager). 100 req ×
  1000-prefill→128-decode, greedy (temp=0). Metric: decode tok/s + ms/token; correctness =
  identical greedy tokens vs eager (must hold — graph is replay of the same kernels).
- **SLO-shape gate** (distilled lesson): also re-run at a production prompt length, not only the
  smoke shape, before calling the default-on licensed.

## Results

`pending-remote` — fill after the single-H20 A/B. **License:** ≥5% decode tok/s, 0 token
divergence → default-on confirmed. **Kill:** wash/regression at the prod shape → flip the CLI
default to off (`args.rs`), keep the override env var. The graph's purpose is removing ~250–400
per-token `cuLaunchKernel` calls, so a decode win is expected; this is the confirmation, not the
hypothesis.

## Rule

A feature that is "implemented + verified" but gated behind an undocumented default-off env var,
with the user-facing flag discarded (`let _ =`), is **not shipped** — it is a half-state where the
flag lies. Wire the flag to the real gate, default to the documented intent, and license the
default with the SLO-shape A/B.
