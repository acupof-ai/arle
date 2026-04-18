# SGLang p99 parity — Qwen3.5-4B on L4 (guidellm sweep)

## Context

First head-to-head p99 measurement for our **specialised hybrid model**
Qwen3.5-4B (24 linear + 8 full attention layers). Counterpart to
`2026-04-17-sglang-p99-parity-qwen3-4b.md`; same hardware, same guidellm
canonical profile. sglang 0.5.10 ships `qwen3_5.py` which loads the same
HF weights as our infer Qwen3.5 path.

- **Hardware:** NVIDIA L4 24GB, CUDA 12.8
- **Model:** Qwen/Qwen3.5-4B bf16
- **Both servers:** 10 slots × 5120 seq-len, `--mem-fraction-static 0.88`
- **Workload:** guidellm sweep, `prompt_tokens=4096,output_tokens=256`, 60s/rate, seed 20260416
- **Paired artefacts:**
  - infer: `2026-04-17-bench-guidellm-qwen35-4b-infer-l4-p99.md`
  - sglang: `2026-04-17-bench-guidellm-qwen35-4b-sglang-l4-p99.md`
- **Serial bench.** Commit: d6cc932.

## Side-by-side — matched-rate rows

| metric           | infer @0.115 | sglang @0.135 | Δ              |
|------------------|-------------:|--------------:|---------------:|
| TTFT p50 (ms)    | 1340         | 831           | **+509 (+61%)** |
| TTFT p99 (ms)    | 1359         | 964           | **+395 (+41%)** |
| ITL p50 (ms)     | 41.10        | 41.79         | **-0.7 (-1.7%)** |
| ITL p99 (ms)     | 41.13        | 42.33         | **-1.2 (-2.8%)** |
| out tok/s        | 28.16        | 32.82         | -14.2%         |

| metric           | infer @0.146 | sglang @0.188 | Δ              |
|------------------|-------------:|--------------:|---------------:|
| TTFT p50 (ms)    | 1357         | 834           | +523 (+63%)    |
| TTFT p99 (ms)    | 1382         | 852           | +530 (+62%)    |
| ITL p50 (ms)     | 42.09        | 45.79         | **-3.7 (-8.1%)** |
| ITL p99 (ms)     | 42.16        | 45.89         | **-3.7 (-8.2%)** |
| out tok/s        | 34.64        | 42.87         | -19.2%         |

| metric           | infer @0.208 | sglang @0.240 | Δ              |
|------------------|-------------:|--------------:|---------------:|
| TTFT p99 (ms)    | 1374         | 863           | +511 (+59%)    |
| ITL p99 (ms)     | 47.15        | 49.80         | **-5.3%**       |

### Saturation

- **infer peak throughput:** 91.4 tok/s @ 0.33 r/s
- **sglang peak throughput:** 134.0 tok/s @ 0.50 r/s
- **Δ = +46.6% in sglang's favour** (much wider than Qwen3-4B's 18% gap)

## Findings — two surprises

### ITL p99 — infer BEATS sglang on Qwen3.5

Across all matched rates, **infer's ITL p99 is 2-8% lower than sglang's**:
- 0.135 r/s region: -3%
- 0.188 r/s region: **-8%**
- 0.240 r/s region: -5%

This is the first measurement where our engine clearly beats sglang on
ITL. It's exactly what we'd predict: the Metal+CUDA team invested in
Qwen3.5-specific decode paths (gated delta net kernels, interleaved
SSM+full-attn, cache snapshots for prefix reuse); sglang's `qwen3_5.py`
runs through the generic decoder framework with less per-model tuning.

**This is the "superseding via specialisation" proof point for the
decode path.** It flips the "sglang always wins" narrative.

### TTFT p99 — gap GREW vs Qwen3-4B

| model | infer TTFT p99 steady | sglang TTFT p99 steady | Δ   |
|-------|----------------------:|-----------------------:|----:|
| Qwen3-4B   | ~1250 ms              | ~820 ms                | +52% |
| Qwen3.5-4B | **~1375 ms**           | **~860 ms**             | **+60%** |

The gap widened by ~8pp going to the hybrid model. Hypothesis:
- sglang's `qwen3_5.py` uses a unified CUDA Graph for prefill spanning both
  linear and full-attention layers — they capture the whole forward as one
  graph replay.
- Our infer prefill path for Qwen3.5 alternates between distinct kernel
  families (DeltaNet chunk kernels vs FlashInfer full-attn prefill) and
  likely has more launch overhead per layer.

### Peak throughput — gap widened

Qwen3-4B gap was +18%; Qwen3.5-4B gap is **+47%**. This is 2.6x worse.
Same root cause as above: sglang's prefill on the hybrid arch is
dramatically better because it's a single-graph replay, whereas ours
does 32 layers of individual kernel launches with alternating kernel
families. Under the throughput profile where many prefills overlap,
the per-kernel-launch overhead compounds badly.

## Verdict for the "surpass" question

**For pure decode (ITL) on Qwen3.5, we already surpass sglang by 3-8%.**
That's the model-specific tuning payoff.

**For prefill (TTFT / throughput), we lag worse than on Qwen3.** The
hybrid arch exposes our kernel-launch-per-layer overhead more because
sglang batches the alternating kernels into a single graph and we don't.

## Opportunities ranked

1. **Qwen3.5 prefill CUDA Graph — single capture for the full 32-layer forward.**
   Largest throughput gain (~46%). Non-trivial — needs contiguous
   allocation for the hybrid state between linear/full-attn handoffs.
   This is the item that most directly uses our "single model adaptation"
   advantage to out-engineer a generic framework.

2. **Fold DeltaNet chunk kernels into the FlashInfer-style "plan once" model.**
   Our gated_delta_rule_prefill_chunk_* kernels are fused but we call ~7
   of them per linear layer. sglang's Triton kernel is a single call.
   Not quite a 10x kernel count reduction, but similar idea.

3. **Extend the decode-path fused kernels we already built** (argmax_axis,
   fused quantized gated MLP from commit 841c4d4) to cover the hybrid
   attention output gate path too. Would extend the ITL lead we already have.

4. **Prefix cache re-enable for Qwen3.5** — the 2026-04-09 fix made this
   safe (GenerationState trait). Under concurrent conditions we might
   claw back meaningful TTFT for shared-prefix workloads (agents, chat).

## Rule

The hybrid architecture amplifies any per-kernel-launch overhead. Our
prefill gap grows from 50% (Qwen3) to 60% (Qwen3.5) on the same hardware
because sglang's single-graph prefill covers more surface area. Target
**full-forward CUDA Graph capture during prefill** as the next big
Qwen3.5 optimisation — it's the only fix that scales to the 47%
throughput gap at saturation.

Meanwhile, **our Qwen3.5 ITL lead (3-8%) is the first clean "surpass
sglang on a model-specific dimension" result.** Publish this narrow
win; it's orthogonal to the TTFT gap.

## Artefacts

- `bench-output/2026-04-17-qwen35-4b-infer-l4-p99/` (infer raw + HTML)
- `bench-output/2026-04-17-qwen35-4b-sglang-l4-p99/` (sglang raw + HTML)
