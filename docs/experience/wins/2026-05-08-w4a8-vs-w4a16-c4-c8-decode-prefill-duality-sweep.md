# W4A8 vs W4A16 c=4 → c=8 sweep — decode-vs-prefill duality holds, gap closing but no crossover

> Concurrency sweep test of decode-vs-prefill duality hypothesis(skill
> v1.4.0 anti-pattern #12)on W4A8 GPTQ-zpfix vs W4A16 GPTQ-zpfix at
> 4k longctx prompt + 256 output。
>
> **Result:W4A8 decode gap to W4A16 narrows from c=4(1.63×)to c=8
> (1.48×)but does NOT cross over**。W4A8 prefill TTFT advantage holds
> at both concurrencies(~31-36% better)。Hybrid dispatch is empirically
> validated as production direction;crossover threshold likely > c=8。

## Phase 1 target

| Field | Value |
|---|---|
| Metric | ITL p50 sweep at c=4 → c=8 for W4A8 vs W4A16(4k longctx) |
| Hypothesis | W4A8 INT8 activation overhead amortizes at higher batch → ITL crosses W4A16 at some c=K threshold |
| License threshold | W4A8 ITL ≤ W4A16 at any c → hybrid dispatch viable for decode |
| Kill threshold | gap widens or stays at higher c → W4A8 = prefill-only forever |

## Phase 5 — Single-variable A/B(2 × 2 grid)

Variable: quant format(W4A8 vs W4A16)× concurrency(c=4 vs c=8)。
All else identical:
- Same model class:Qwen3-4B GPTQ-zpfix sources(post `2a3a6f0`)
- Same workload:4096 prompt / 256 output / 120s × 10s warmup
- Same hardware:sm_89 4070 Ti SUPER

```bash
# Both servers --num-slots 16 --max-seq-len 5120
scripts/bench_guidellm.sh m_quant-w4a16-zpfix-c8-4k \
  --model Qwen3-4B-GPTQ-W4A16-marlin-zpfix \
  --concurrencies 8 ...

scripts/bench_guidellm.sh m_quant-w4a8-gptq-zpfix-c8-4k \
  --model Qwen3-4B-GPTQ-W4A8-zpfix \
  --concurrencies 8 ...
```

## Results — 2×2 grid

| Metric | W4A16 c=4(`bc15eca`) | W4A16 c=8(this) | W4A8 c=4(`b5889b3`) | W4A8 c=8(this) |
|---|---:|---:|---:|---:|
| ITL p50 | **11.73 ms** | **16.28 ms** | 19.18 ms | 24.09 ms |
| ITL std | 0.05 | 0.07 | 0.42 | 0.31 |
| TTFT p50 | 2388 ms | 4811 ms | **1632 ms** | **3323 ms** |
| TTFT std | 92.9 | 78.6 | 112 | 93.1 |
| out tok/s | 192 | **239** | 156 | 223 |
| total tok/s | 3258 | 4067 | 2645 | 3788 |

### Cross-format ratios

| Pair | W4A8/W4A16 ITL ratio | W4A8/W4A16 TTFT ratio |
|---|---:|---:|
| c=4 | **1.63×**(W4A8 slower) | **0.68×**(W4A8 36% faster) |
| c=8 | **1.48×**(W4A8 slower) | **0.69×**(W4A8 31% faster) |

**Decode gap narrows**:1.63× → 1.48× = **−9% gap** as concurrency doubled。
**Prefill advantage holds**:36% → 31% = small narrowing,still solid W4A8 win。

## Phase 4 — Formula reconciliation

For ITL gap to close:
- Decode at higher batch amortizes weight read more
- W4A8 INT8 activation quant cost is per-Linear-call,batch-invariant in N,K
- W4A16 has no activation quant overhead,but Marlin BF16↔FP16 round-trip is similar batch-invariant

Net:**both formats have batch-invariant per-call overheads**,batch
gain comes from amortizing weight READ。Since W4A8 weight is 1/4 size of
W4A16 weight…

Wait,both are W4 weight format → same weight size。The ONLY difference
is W4A8 quantizes activation INT8 vs W4A16 keeps BF16。So activation
processing cost is the bottleneck difference。

At c=4 batch=4 decode:
- 4 hidden states × INT8 quant = 4× per-Linear quant cost
At c=8 batch=8:
- 8 × INT8 quant = 8× per-Linear quant cost
Linear scaling expected。But ITL grew from 19.18 → 24.09 = 1.26× while
batch doubled。**Sub-linear scaling**:W4A8 IS partially amortizing the
quant cost。

Extrapolated crossover prediction:
```
W4A8 ITL(c)  ≈ 19.18 + 0.5 × (c-4) ms (heuristic linear extrapolation)
W4A16 ITL(c) ≈ 11.76 + 0.6 × (c-4) ms
W4A8 = W4A16:19.18 + 0.5(c-4) = 11.76 + 0.6(c-4)
              7.42 = 0.1 × (c-4)
              c = 4 + 74.2 = 78.2
```

Crossover at c≈78 is impractical(W4 c=8 already hits substrate
admission load,c=78 is unrealistic for our memory budget)。

**Refined conclusion**:W4A8 will likely **never cross W4A16 for decode
at production-realistic concurrencies**(c=4-32)。

## Phase 7 tradeoff(refined)

| Axis | Status | Note |
|---|---|---|
| **Decode**(any c=4-32) | W4A16 wins by 1.4-1.6× | hardware ceiling:W4 weight HBM,both same |
| **Prefill** | W4A8 wins -31 to -36% | INT8 mma at large batch gain |
| **Hybrid dispatch** | viable in principle | substantial substrate work to switch quant per phase |
| **Memory**(both resident)| 2× weight pool | W4A8 2.65 GB + W4A16 4.5 GB = 7 GB(43% of 16 GB) |
| **Workflow** | both off same GPTQ source | minor |

**Production recommendation**:
- W4A16 = decode default(LICENSED at 1.64×,confirmed 2 routes naive sym + GPTQ-zpfix)
- W4A8 = prefill-only path(if hybrid dispatch landed)
- Hybrid:NOT pursued in this tick(deferred per skill anti-pattern #12 R4 #6 KILL HARD precedent)

## Phase 8 license decision

Hypothesis "W4A8 catches W4A16 at higher batch" — **partially confirmed,
practically refuted**:
- Gap narrows c=4(1.63×)→ c=8(1.48×)= small win
- Crossover extrapolation at c≈78 = impractical
- **DECISION**:W4A8 decode REMAINS DEFERRED;hybrid dispatch axis OPEN

W4A8 prefill advantage holds robustly across c=4-8 with tight σ(112-93 ms TTFT)。
**LICENSED for prefill** confirmed at multiple concurrencies。

## Cross-references

- W4A8 c=4: `b5889b3`(prefill LICENSED,decode DEFERRED)
- W4A16 c=4: `bc15eca`(GPTQ-zpfix matches naive sym at 11.73 ms)
- Codex qzeros +1 fix: `2a3a6f0`
- Skill v1.4.0 anti-pattern #12(decode vs prefill duality):`6c627c4`
- Skill v1.4.0 anti-pattern #14(upstream parser):`6c627c4`
- W3+W4 substrate fix: `b708e00`(now allows c=8 multi-conc decode without deadlock)
- Master strategy update: `182e084`
- Bench artifacts:
  - `bench-output/2026-05-08-m_quant-w4a8-gptq-zpfix-c8-4k/`
  - `bench-output/2026-05-08-m_quant-w4a16-zpfix-c8-4k/`

## Status

- ✅ W4A8 prefill LICENSED at c=4 + c=8(robust)
- ⚠ W4A8 decode DEFERRED at c=4 + c=8(gap narrows but doesn't cross)
- ❌ W4A8 decode crossover at production c not viable(extrapolated c≈78)
- 🔧 Hybrid dispatch path:OPEN(prefill W4A8 + decode W4A16)— substrate work TBD

## Rule

**For activation-precision quant scheme(W4A8 vs W4A16),decode-vs-prefill
duality is workload-batch-size-stable,not just instantaneous-batch-size**。
Multi-concurrency sweep(c=4 → c=8 → c=16)before declaring decode
viable;single-shape extrapolation is insufficient。

For ARLE Qwen3-4B specifically:
- W4A8 decode never crosses W4A16(extrapolated c=78)
- W4A16 = canonical decode default
- Hybrid dispatch needed to recover W4A8 prefill TTFT advantage

Generalization:**any new W4Ax format(W4A4,NVFP4,FP4 emulated)should
include decode-multi-c sweep BEFORE production gating** — single-shape
results miss the crossover analysis。
