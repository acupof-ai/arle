# ARLE Roadmap — Post fp8 KV Baseline (2026-04-30)

After 2026-04-29 fp8 KV ablation + 2026-04-30 fp8 readback fix.
Anchor for next-step work; refresh when a phase closes.

## Current Baseline (CUDA L4, c=16, 4096/256, 120s)

| Dimension | ARLE | SGLang | Verdict |
|---|---:|---:|---|
| ITL Qwen3 fp8 | 72.55 ms | 86.20 ms | ARLE leads −16% |
| ITL Qwen3.5 fp8 | 57.14 ms | 70.40 ms | ARLE leads −19% |
| TTFT Qwen3 fp8 | 13.24 s | 8.38 s | ARLE +58% slower |
| TTFT Qwen3.5 fp8 | 13.27 s | 6.57 s | ARLE +102% slower |
| fp8 KV numerical match | 39.08% | 77.54% | ARLE residual bug |

Decode kernel path is genuinely faster. The gap is **prefill performance + fp8 numerics**.

## Phase 1 — Close Correctness + TTFT (this week)

### P1.1 — fp8 KV 5th residual bug (correctness, highest priority)
- Symptom: 3/16 exact, 39.08% common-token match, divergence at generated token 1.
- Candidates: RoPE order vs quantize, scale dtype/shape, FlashInfer fp8 reduce ordering.
- Method: 1-prompt 5-token reproduction; dump K/V hidden bytes and compare with SGLang.
- Exit: ≥70% match, divergence at token ≥2.
- Land: `errors/2026-04-30-arle-fp8kv-residual-bug.md` + `fix(qwen35): close fp8 attention math residual`.

### P1.2 — Qwen3.5 prefill workspace pooling (TTFT)
- Root cause confirmed today: `max_concurrent_prefill_requests=Some(1)` because workspace estimator over-provisions for 4-prompt parallel scratch.
- Effect: 16×4096 prefill takes 32 sequential steps vs SGLang's 4 packed.
- Fix: pool prefill scratch (GdrChunkwiseScratch + FlashInfer HD256 plan + linear-attn recurrent) by time-share; remove the cap.
- Exit: c=16 TTFT 13.27 s → ≤ 8 s, tok/s 152 → ≥ 165.
- Land: `wins/2026-04-30-prefill-workspace-pooling-qwen35.md`.
- Drain mode (`4d23a12d`) was reverted — empirically TTFT-flat. Do NOT revisit.

### P1.3 — Qwen3 TTFT residual (profile-driven)
- Qwen3 has no cap but still TTFT +58%.
- Profile first: FlashInfer prefill plan recompute? cuBLAS algo cache miss? per-step overhead?
- Candidates: plan caching, autotune coverage, prefill shape warmup.
- Exit: c=16 TTFT 13.24 s → ≤ 10 s (within +15% of SGLang).

**Phase 1 done = ARLE TTFT within +15% of SGLang on all four ARLE rows AND fp8 KV match within SGLang quartile. Then fp8-default vote can be opened.**

## Phase 2 — Performance Sweep (next week)

- P2.1: Multi-concurrency sweep (c=1, 4, 16, 64) — find scaling cliffs.
- P2.2: Nsight Systems profiles on remote GPU box (nsys missing locally).
- P2.3: Hidden caps → CLI flags (`--prefill-max-concurrent`, `--workspace-pooling on/off`).

## Phase 3 — Feature Surface (2-4 weeks)

| Feature | Why | Priority |
|---|---|---|
| Speculative decoding (lookahead / EAGLE) | Multiplier on existing decode lead | P0 |
| Multi-LoRA same-model concurrent | Multi-tenant requirement | P0 |
| Structured output / JSON mode | OpenAI compat | P1 |
| Prefix cache eviction tuning | +20% on long chats | P1 |
| Pluggable continuous-batching policy | Schedulability research | P2 |

## Phase 4 — Hardware Coverage (4-8 weeks)

1. **H100 (sm_90)** — TileLang TMA/WGMMA target; remote box flow + HD256 decode retry.
2. A100 (sm_80) — production parity.
3. RTX 4090 (sm_89) — dev workstation.
4. Metal (Apple Silicon) — finish Qwen3.5 hybrid path; serving.

## Phase 5 — Ecosystem & DevX (8+ weeks)

- Multi-GPU TP (single-node).
- Multi-node KV sharing via `kv_tier` + `kv-native-sys` (EIC pattern).
- Train ↔ Infer pipeline (`crates/train`, `crates/autograd`).
- Bench dashboard (auto-rendered guidellm comparisons).
- Easy bench harness (one-command vLLM/SGLang/ARLE shootout).

## Decision Log

- 2026-04-30 — Cold-start prefill drain rejected: empirically TTFT-flat, drains hurt min-TTFT. Reverted `4d23a12d` and `ed6df900`.
- 2026-04-30 — fp8 KV default flip held until P1.1 closes residual bug.
- 2026-04-30 — Single-GPU scheduler invariant: 1 prefill + N decode mixed per step. NOT drain. NOT multiple-prefill-per-step until workspace pooling lands.

## Cross-References

- Today's fp8 ablation wins: `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-fp8kv-ablation.md`
- Today's headline summary: `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-headline-summary.md`
- TileLang on/off: `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-tilelang-on-vs-off.md`
- TileLang HD256 sm_89 build blocker: `docs/experience/errors/2026-04-29-tilelang-decode-hd256-sm89-build.md`
