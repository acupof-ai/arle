# Metal Backend "World #1" Gap Analysis — 2026-05-07

Two parallel research subagents surveyed Apple Silicon SOTA on
2026-05-07. This entry synthesizes their findings against the **current
ARLE Metal backend code state** and ranks the gaps by leverage, so the
next runtime commits can target the largest deltas first.

This is a **review + ranked backlog**, not the master roadmap. The
master roadmap remains
[`mlx-backend-roadmap.md`](mlx-backend-roadmap.md); this entry is the
input it should pull from when deciding the next P-tier item.

## Method

- Kernel-track subagent: mlx / mlx-lm / llama.cpp Metal / candle / mistral.rs
  Metal kernel advances (PagedAttention, simdgroup-MMA, TurboQuant 4-bit
  KV, MTPLX, tree spec-decode).
- Serving-track subagent: vllm-mlx, oMLX, SGLang RadixAttention, llama.cpp
  Metal slots, mistral.rs PagedAttention, chunked prefill, disaggregated
  prefill, multi-LoRA, structured outputs.
- Cross-checked against current code:
  - [`infer/src/backend/metal/AGENTS.md`](../../infer/src/backend/metal/AGENTS.md)
    invariants 1–8 and Active Priority section.
  - [`infer/src/scheduler/AGENTS.md`](../../infer/src/scheduler/AGENTS.md).
  - `metal/scheduler.rs`, `metal/runtime.rs::execute_prefill_chunk`,
    `metal/prefix_cache.rs`, `metal/kv_pool.rs`, `metal/dflash.rs`.

## What is already in tree

Verified by reading the module map + grepping the runtime:

- Decode-first continuous batching loop (`run_metal_scheduler_runtime`).
- **Chunked prefill is already wired** (`execute_prefill_chunk` +
  `prefill_chunk(budget)` in `runtime.rs`) — research-track item #2
  (chunked prefill / decode-priority interleave) is *partially*
  implemented, not absent.
- Variable-length decode via `Qwen35PackedDecodeBatch` (left-padding +
  additive mask + per-row RoPE offsets).
- DFlash speculative decode dispatched through the scheduler runtime.
- `metal/prefix_cache.rs` (always-on) and `metal/kv_pool.rs` (always-on,
  not yet on the hot path).

## What the research found that we do NOT have

Ranked by leverage / cost. "L-of-K" annotations are the kernel-track
report; "L-of-S" annotations are the serving-track report.

### Tier A — biggest leverage per unit effort

1. **Token-level radix prefix cache.** `metal/prefix_cache.rs` today is
   accounting only; SGLang RadixAttention + oMLX block-CoW-radix gives
   2–5× TTFT on shared-prefix agentic workloads. (L-of-S #1)
   - Effort: M (CPU-side radix tree + page handle integration).
   - Prereq for paged KV (Tier B #1) — token-level identity is needed
     before block tables can attach pages cross-batch.

2. **Q8 / FP8 KV cache + Metal-aware wire-down cap** copying
   mistral.rs's `cap = max_seq × max_batch` policy. KV today is BF16.
   (L-of-S #3, L-of-K #4)
   - Effort: M (`metal/ops.rs::extend_kv_cache` + qwen35 cache path).
   - Win: 2× context @ iso-RAM, +10–25% decode @ long ctx.
   - Stacks linearly with Tier B #1.

3. **Decode-priority interleave proof.** Chunked prefill exists, but no
   test asserts that prefill yields to decode under c≥4 mixed traffic.
   Without that gate we cannot claim parity with SwiftLM's
   `--prefill-step-size 512` HOL-blocking story. (L-of-S #2)
   - Effort: S (regression test + scheduler invariant doc).
   - Win: surfaces TTFT p99 regressions that today are silent.

### Tier B — large but high-cost

1. **Paged-attention block tables on Metal.** Replaces left-padding
   with block tables; EricLBuehler's reported numbers: +77% Qwen3-30B-A3B
   4-bit and +131% Llama-3.2-3B-8bit decode tok/s vs llama.cpp continuous
   batching. (L-of-K #1, L-of-S #1 paged-KV half)
   - Effort: L (allocator, page-aware mask, per-page RoPE offsets).
   - Win: +30–80% decode @ varlen batches ≥ 4; 2–4× max ctx @ iso-VRAM.
   - **Wire `kv_pool.rs` onto the hot path as the substrate.**
   - Stacks with Tier A #2 (Q8 KV) and Tier A #1 (radix).

2. **MTP / EAGLE speculative decoding integrated as default for
   Qwen3.5.** DFlash is in tree but experimental; Qwen3.5 ships native
   MTP heads. MTPLX shows ~2.24× decode tok/s on M5 Max. (L-of-K #2,
   L-of-S #5)
   - Effort: M (Qwen3.5 MTP head wiring + scheduler verifier slot).
   - Win: 1.8–2.3× decode tok/s @ temp ≤ 0.7, lossless under residual
     correction.

### Tier C — smaller but cheap follow-ups

1. **Custom simdgroup-MMA M=16 quantized matmul** (`mma2big`,
   `mma2big_pipe` patterns) for the verify/draft step and MoE
   token-level paths. MLX default `quantized_matmul` is M=1 tuned.
   (L-of-K #3)
   - Effort: M (in `crates/mlx-sys/src/mlx_bridge.cpp`).
   - Win: +10–40% decode under spec/MoE; stacks with B #2.

2. **Tree-attention spec-decode mask** (DDTree pattern) once B #2 lands.
   (L-of-K #5)
   - Effort: S.
   - Win: +10–15% on top of B #2 for code / structured outputs.

3. **Two-tier prefix cache (RAM + SSD persistent across restarts)** —
   oMLX-style. Critical for `arle` agent loops where TTFT 22 s → 0.2 s
   warm-start matters. (L-of-S #4)
   - Effort: M (file-backed page tier behind A #1 radix).
   - Best deferred until A #1 lands.

### Tier D — known frontier, not catch-up

- **Multi-LoRA Punica/S-LoRA on Metal.** No one ships it. Not a gap;
  if we build it, we lead.
- **Disaggregated ANE-prefill / GPU-decode** (Squeezebits Yetter). Not
  production-ready upstream; do not adopt.
- **M5 TensorOps via Metal 4.** Automatic via `mlx ≥ 0.31` dep bump;
  reported 3.3–4× TTFT on M5. Not a code change for us, but a release
  gate (verify the `mlx-sys` pin picks it up when M5 hardware lands).

## Recommended sequencing

The cheapest world-first arc is **A1 → A2 → A3 → B1 → B2 → C1**.

- A1 (radix) is the keystone — both B1 (paged KV) and C3 (two-tier)
  depend on it.
- A2 (Q8 KV) is independently shippable and unblocks long-context
  benchmarks the scheduler-track report calls out as our weakest axis
  (BF16 KV is no longer competitive vs Q8 default in mlx-lm /
  llama.cpp / mistral.rs).
- A3 (decode-priority test) is the smallest commit and locks in a
  regression we will otherwise hit during A1+B1.

Ship in atomic commits per the workflow protocol; each commit lands a
bench entry under `docs/experience/wins/` per CLAUDE.md §Benchmarks.

## Reference benchmark targets

From the serving-track survey — to claim "world #1" on Metal we need
to beat or match these on M-series:

| Metric | Reference | Source |
|---|---|---|
| Qwen3-0.6B-8bit single-stream | **417.9 tok/s** M4 Max | `vllm-mlx` |
| Llama-3.2-3B-4bit single-stream | **205.6 tok/s** M4 Max | `vllm-mlx` |
| Qwen3-30B-A3B-4bit single-stream | **127.7 tok/s** M4 Max | `vllm-mlx` |
| DeepSeek-V3 Q4 c=32 aggregate | **1,150 tok/s** M4 Pro | `vllm-mlx` |
| Cached-prompt TTFT | **0.08 s** Gemma-4-e2b | `SwiftLM` |
| Qwen-32B Q4 ctx=32K | **19.0 tok/s** M3 Ultra | mlx-lm reference |

Our current single-request Qwen3.5-0.8B MLX 4bit on M4 Pro 20c is
**305.5 tok/s** at `1024/256` step-driver
([`mlx-backend-roadmap.md`](mlx-backend-roadmap.md)).

## Sources

Kernel track: `ml-explore/mlx#2228`, `MTPLX`, `ddtree-mlx`,
`dflash-mlx`, `vllm-mlx`, mlx-lm releases, mlx releases, Apple ML M5
post, mlx-mfa, TurboQuant on MLX (Antonrozanov 2026-03), llama.cpp
flash-attention DeepWiki.

Serving track: `waybarrios/vllm-mlx`, `jundot/omlx`,
`macgpu.com/2026-mac-inference-framework-benchmark`,
`ggml-org/llama.cpp#20574`, `EricLBuehler/mistral.rs/PAGED_ATTENTION.md`,
`lmsys.org/2024-01-17-sglang`, `ml-explore/mlx-lm#630`,
`SharpAI/SwiftLM`, `raullenchai/Rapid-MLX`,
`ml-explore/mlx#3209`, `roborhythms.com/reduce-local-llm-ttft-mac-studio`,
`Aryagm/dflash-mlx`, `youssofal/MTPLX`, Punica `arxiv:2310.18547`,
S-LoRA `arxiv:2311.03285`, `AmesianX/TurboQuant`,
`ggml-org/llama.cpp#20969`, vLLM chunked-prefill docs.
