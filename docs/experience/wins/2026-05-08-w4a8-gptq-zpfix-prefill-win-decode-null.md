# W4A8 GPTQ-zpfix production bench — prefill TTFT −36% vs W4A16, decode ITL NULL

> First production W4A8 bench post-`2a3a6f0` qzeros +1 fix。
> `Qwen3-4B-GPTQ-W4A8-zpfix` checkpoint(corrected GPTQ + Phase 1b
> re-pack)through ARLE Marlin W4A8 path,4k longctx c=4。
>
> **Headline:W4A8 wins prefill(−36% TTFT vs W4A16)but ties BF16 on
> decode ITL**。INT8 activation quant overhead at small batch dominates
> W4 weight bandwidth gain — W4A8 is the prefill path,W4A16 remains
> decode path。Decode-vs-prefill duality(skill anti-pattern #12)
> empirically confirmed on a quant-format basis。

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | TTFT + ITL on Qwen3-4B-GPTQ-W4A8-zpfix,4k longctx c=4 |
| Baseline | BF16(`786a20a`):ITL 19.27 ms,TTFT 1976 ms;W4A16(`f6f3af3`):ITL 11.76 ms,TTFT 2565 ms |
| License threshold | ITL ≥ 1.5× BF16 OR TTFT ≤ 90% W4A16(per master strategy §1.2.1) |
| Kill threshold | ITL > 1.0× BF16 AND TTFT > 1.0× W4A16 |

## Phase 5 — Single-variable A/B

Same workload spec(4k prompt,256 out,c=4,120s × 10s warmup),same model
class(Qwen3-4B),different quant format only。

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer \
  --model-path infer/models/Qwen3-4B-GPTQ-W4A8-zpfix \
  --port 8000 --num-slots 8 --max-seq-len 5120

PATH=.venv/bin:$PATH \
  scripts/bench_guidellm.sh m_quant-w4a8-gptq-zpfix-c4-4k \
  --model Qwen3-4B-GPTQ-W4A8-zpfix \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-W4A8-zpfix \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,...,output_tokens=256,...'
```

## Results

| Metric | BF16(`786a20a`) | W4A16(`f6f3af3`) | **W4A8 GPTQ-zpfix** | Δ vs BF16 | Δ vs W4A16 |
|---|---:|---:|---:|---:|---:|
| ITL p50 | 19.27 ms | **11.76 ms** | **19.18 ms** | −0.5%(NULL) | **+63% REG** |
| ITL std | n/a | n/a | **0.42 ms** | very tight σ | tight |
| TTFT p50 | 1976 ms | 2565 ms | **1632 ms** | **−17%** | **−36%** |
| TTFT std | n/a | n/a | 112 ms | tight | tight |
| out tok/s | 153.83 | 191 | 155.57 | +1% | −19% |
| TPOT mean | n/a | n/a | 26.07 ms | — | — |
| Peak KV util | n/a | n/a | 79.1% | — | — |
| Plan labels | n/a | n/a | idle=8736,decode=4597,prefill=67,split=0 | — | — |
| greedy_consistency | n/a | PASS | **PASS 32/32 0% diff** | — | — |

Bench artifacts:`bench-output/2026-05-08-m_quant-w4a8-gptq-zpfix-c4-4k/`。

## Phase 7 — Tradeoffs explicit

| Axis | Status | Rationale |
|---|---|---|
| **Prefill (TTFT)** | ✅ W4A8 wins -36% vs W4A16 | INT8 mma at large batch(2048-token chunks)beats Marlin's BF16↔FP16 round-trip |
| **Decode (ITL)** | ⚠ W4A8 ties BF16,loses 1.6× to W4A16 | INT8 activation quant overhead per Linear call at batch=4 dominates weight bandwidth savings |
| Numerical correctness | ✅ greedy_consistency PASS 32/32 0% diff | qzeros +1 fix verified |
| LOC complexity | ✅ no kernel changes | corrected converter only |
| Hardware specificity | ✅ sm_89 OK | Marlin sm_80+ |
| Memory budget | ✅ ~2.65 GB checkpoint(vs 3.6 GB W4A16) | smaller checkpoint |
| Workflow | ⚠ requires GPTQ + 2-step convert | `convert_gptq.py + convert_gptq_w4a16_to_w4a8_marlin.py` |

**No-tradeoff axes**:correctness,LOC,HW。**Real tradeoff**:prefill-vs-decode duality at c=4。

## Phase 8 — License decision

| Threshold | Result | Verdict |
|---|---|---|
| ITL ≥ 1.5× BF16 | 1.005× | ❌ |
| TTFT ≤ 90% W4A16 | 64% (=−36%)| ✅ |
| Numerical correctness | PASS 0% diff | ✅ |

**Mixed verdict**:LICENSE for **prefill path**(TTFT win solid + correctness PASS),
DEFER for **decode path**(ITL no gain at c=4)。

## Strategic implication

W4A8 + W4A16 are **complementary,not substitutable** at production c=4
shape:
- **W4A8**:prefill path(TTFT win) — chunked-prefill stage
- **W4A16**:decode path(ITL win) — autoregressive sampling stage
- **Hybrid dispatch**(switch quant format between phases)is the
  optimal stack — but adds substantial weight pool memory cost(both
  W4A8 + W4A16 weights resident)

Master §1.2.1.A weight-axis status update:
- W4A16 production decode default:**still LICENSED at 1.64× ITL**
- W4A8 production:**LICENSED for prefill TTFT,DEFER for decode**
- Hybrid dispatch evaluation:**OPEN**(not yet attempted)

## Skill v1.4.0 anti-pattern #12 confirmed empirically

Per skill v1.4.0 anti-pattern #12 "Single-kernel choice ≠ optimal at all
batch sizes (decode vs prefill duality)":this bench provides **first
empirical W4A8-vs-W4A16 dispatch evidence**:
- Decode batch=4(small):W4A16 wins(11.76 ms vs 19.18 ms)
- Prefill batch=2048(large):W4A8 wins(TTFT −36%)

The duality is by quant format(W4A8 has activation overhead at small
batch),not just by kernel choice within W4A16 path。Generalizes to all
mixed-precision quant schemes:**activation precision change has
batch-size-dependent ROI**。

## Cross-references

- Codex qzeros +1 fix: `2a3a6f0`(`fix(cuda): correct GPTQ zero-point decode in converter`)
- Codex qzeros analysis: `5593865`(`docs/research/2026-05-08-gptq-qzeros-zero-minus-1-convention-bug.md`)
- Phase 1b conversion script: `09869bc`(`scripts/convert_gptq_w4a16_to_w4a8_marlin.py`)
- Pack GPTQ-aware:`12a54da`(`scripts/quantize_qwen3_w4a8.py`)
- Skill v1.4.0 anti-pattern #14 upstream parser:`6c627c4`
- Master strategy §1.2.1 update: `5dc27a2`
- W4A16 LICENSED baseline: `f6f3af3`
- BF16 baseline: `786a20a`
- W4A16 1.06× implementation gap entry: `2026-05-08-marlin-w4a16-bench-implementation-gap.md`
- W3+W4 substrate fix: `b708e00`
- Bench artifacts:`bench-output/2026-05-08-m_quant-w4a8-gptq-zpfix-c4-4k/`

## Status

- ✅ W4A8 production correctness LICENSED(0% diff,first ever)
- ✅ W4A8 prefill TTFT LICENSED(−36% vs W4A16)
- ⚠ W4A8 decode ITL DEFERRED(ties BF16,loses to W4A16 at c=4)
- 🔧 Hybrid dispatch evaluation OPEN(not yet attempted on quant axis)
- 📊 Master strategy §1.2.1.A:W4A16 + W4A8 prefill both production-ready

## Rule

**When evaluating activation-precision quant scheme(W4A8 vs W4A16)at
production**,**bench prefill AND decode separately**:activation quant
overhead is batch-size-sensitive。Single-shape bench(e.g. c=4 4k longctx)
mixes prefill TTFT signal with decode ITL signal — disambiguate via
explicit prefill-only(`output_tokens=1`)and decode-only(measure ITL
post-warmup)benches。

For ARLE specifically:W4A8 = prefill path,W4A16 = decode path。Hybrid
dispatch evaluation requires substantial substrate work(per
`r4-hybrid-dispatch-killed-batch4-decode-regression.md` precedent on
W4A16BatchGemv hybrid)— not pursued in this entry。
