# Qwen3.5/3.6 CUDA operator optimality review — vs kernel-set routing knowledge

**Date:** 2026-06-11. **Hardware frame:** H20 sm_90a — 4.0 TB/s HBM3, ~148
TFLOPS BF16 (bandwidth-fat, compute-thin; AI break-even ≈ 37 FLOP/B).
**Reference:** `/path/to/code/kernel-set` per-(op, GPU, dtype) routing
tables + its measured H20 numbers (`benchmarks/results/h20_sm90_vs_sota.md`).
**Baseline measurements (fc5b48ed → 1e0f05e1):** decode TP=1 36.0 tok/s
(27.8 ms/token), TP=2 45.5; needle 3k prefill 9.85 s at chunk=64.

## Headline verdict

**No — the operators are not optimal, but the gap is NOT mainly in kernel
micro-tuning.** Cost model (lane-B inventory, code-fact):

- Per decode token: **1,114 kernel launches + 41 full-stream `ctx.sync` + 41
  D2H + ~82 H2D**, weight traffic ≈ 6.8 GB @ kv4096 → HBM floor ≈ 1.7 ms.
  Measured 27.8 ms ⇒ **94% of decode is orchestration, not bytes**.
- Per 2048-token prefill chunk: ~10.3 TFLOP compute (≈ 91 ms at peak), but
  GDR serial scan moves 516 GB of state traffic and the attention rescan
  moves 344 GB — both algorithmic-shape problems, not tile-tuning problems.

kernel-set's strategy frame applies cleanly: own the memory-bound elementwise
(ARLE's are adequate), adopt the industry kernel for compute/algorithm-bound
ops (ARLE is right on cuBLASLt for dense GEMM — kernel-set's own Hopper
rank-1 — and wrong/legacy on MoE grouped GEMM, GDR prefill, attention core).

## Ranked gaps (formula-predicted, license-or-kill each)

| # | Op | Today | kernel-set best-on-Hopper | Formula / predicted Δ | Cost |
|---|----|-------|---------------------------|----------------------|------|
| 1 | **MoE routing** | HOST route: `ctx.sync`+D2H+CPU top-8+2×H2D **per layer per step** (40/token) — the sync also kills launch-queue pipelining for all 1,114 launches | fused device `topk_softmax` (sgl) — and ARLE's OWN `dsv4_route_cuda`+`qwen36_renorm_topk_weights_cuda` is default-ON for DSv4, just not wired into `gpu::moe_forward_into` for Qwen35 | remove 40 syncs ≈ 2–4 ms direct + unblocks pipelining of the remaining ~1,074 launches → predicted decode 27.8 → **8–12 ms/token (2.3–3.5×)**; prefill loses 40×(1 MB D2H + CPU top-8 over 2048×256) per chunk | **S** — in-tree, proven on DSv4 |
| 2 | **MoE expert GEMMs** | hand CUDA-core grouped kernels, no tensor cores (kernel-set measured this kernel class at ~3.9 TFLOP/s) | DeepGEMM `m_grouped_bf16_gemm_nt_{contiguous,masked}` — **already vendored for DSv4** | prefill routed compute 4.1 TFLOP/chunk: ~1.0 s → **~40–80 ms** at DeepGEMM-class rates; decode (R=8 skinny) mostly unaffected | **M** |
| 3 | **GDR prefill** | `gated_delta_rule_prefill_recurrent`: 32 blocks (~20% of SMs), serial over tokens, ~14 syncs/token, state RW 8.4 MB×S×30 layers = **516 GB/chunk** | FLA `chunk_gated_delta_rule` (WY-representation chunk-parallel, MIT Triton, GVA 16QK/32V native); **FlashQLA (QwenLM, TileLang, sm90, claims 2–3× FLA)** — same TileLang AOT toolchain ARLE already runs; in-tree TileLang chunkwise variant hangs on sm_90 (errors/2026-05-30) | state traffic ÷ chunk_len: 516 GB → ~16 GB + full token parallelism → GDR leaves the prefill critical path | **M–L** |
| 4 | **Full attention core** | hand nonpaged kernel, one block per (q_head, token), serial 64-pos tiles: decode reads kv ×8 (GQA ratio, 67 MB vs 8.4 MB ideal per layer @4k) on 16 blocks (~20% SM); prefill rescans the prefix per token-block: **344 GB/chunk over 10 layers** | FA3/FlashInfer class (1575–1757 GB/s measured decode BW on H20); **in-tree no-python option: TileLang `batch_{prefill,decode}_paged_hd256` built for exactly (16,2) heads — zero callers today** | prefill attn bytes ÷ ~100 (tiled smem reuse); decode Δ grows with kv_len (32k: 5.4 GB→0.7 GB ≈ 1.2 ms/token); occupancy 16→~5k blocks | **M** |
| 5 | micro: fuse in_proj qkv/z/b/a 4→1 GEMM (b/a are 32-row skinny cuBLASLt calls); argmax per-token `alloc_zeros(1)` (ops.rs:294); ncu-verify lm_head GEMV BW (1.017 GB/token = 15% of traffic — if <80% peak, switch to cuBLASLt N=1) | | | each ≤ launch-count noise until #1 lands | S |

Adequate as-is (no action): dense projections on cuBLASLt (= kernel-set
Hopper rank-1, 132–142 TFLOP/s measured); elementwise silu/add/norm class
(kernel-set's own fallbacks sit at 0.5–1.0× SOTA and ARLE's are the same
class; rmsnorm could adopt sgl-style 128-bit loads later — P3); embedding;
sampling argmax (torch.argmax-class).

## Sequencing discipline (per kernel-optimization skill)

#1 first and alone — it changes the binding constraint for everything else;
re-run the roofline numbers on the post-#1 base before licensing #2–4
(single-variable A/B each, needle+same-config-twice numerics gate, n≥3
σ<5%). #2–4 formulas above are hypothesis-grade until the post-#1 nsys
re-profile confirms their share of the new wall-clock.

## kernel-set integration note

kernel-set's C ABI (torch-free, Rust-bindable) is attractive as a channel,
but for these four hot ops its in-library clean-room fallbacks measure
0.01–0.5× SOTA (its own honest numbers) — the value is its ROUTING knowledge,
not its fallback kernels. ARLE's adopt-official-first rule keeps pointing at
the same destinations kernel-set routes to: DeepGEMM (vendored ✓), FlashQLA /
FLA algorithm (to vendor), FA3-class attention (in-tree TileLang HD256 first).
Worth transcribing back upstream: kernel-set's sm90 optimal.json cells are
heuristic while its own H20 measurements exist (doc lags bench).
