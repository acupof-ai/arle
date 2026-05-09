---
title: M_rope-yarn-scaling Phase 3b — PPL eval plan(quality validation)
date: 2026-05-10
type: plan
status: ready-for-execution
prereq: Phase 3a smoke PASS (`4efd30b`) + ARLE `arle train eval` + long-ctx eval data
---

# Phase 3b — PPL eval(quality validation)concrete plan

> Phase 3a smoke confirmed end-to-end M_rope-yarn-scaling Phase 1+2 wire
> works in production CUDA serving(Qwen3-4B + YARN factor=2.0)。Phase 3b
> validates **quality**:does YARN factor=2.0 maintain reasonable PPL at
> extended context vs vanilla 40k baseline?

## 1. Infrastructure ALREADY EXISTS

`arle train eval` is the canonical PPL evaluation surface:
- `crates/train/src/eval_lm.rs:34-36` `EvalSummary::ppl() = self.loss.exp()`
- Supports `--backend {auto,cpu,metal,cuda}` — CUDA-side viable
- Accepts tokenized JSONL or chat JSONL via `--data` flag
- Returns per-token cross-entropy loss → exp = perplexity

Usage:
```bash
arle train eval \
  --model infer/models/Qwen3-4B \
  --data eval-longctx.tokenized.jsonl \
  --seq-len 40960 \
  --backend cuda \
  --metrics-jsonl metrics-baseline-40k.jsonl
```

## 2. Long-context eval data requirement

For YARN factor=2.0(40k → 81920 max),need eval examples that **stress
beyond native 40k context**:
- 40k baseline:examples in 32k-40k range
- YARN factor=2.0:examples in 40k-80k range(test extension regime)

Candidate datasets:
| Dataset | Avg ctx | Rationale | Acquisition |
|---------|---------|-----------|-------------|
| **WikiText-103**(test split, concatenated)| 1-100M chars chunked | Baseline standard | HF Hub `wikitext` |
| **PG19**(books)| ~70k tokens / book | Naturally long | HF Hub `pg19` |
| **GovReport**(government reports)| 8k-20k tokens | Domain document | HF Hub `ccdv/govreport-summarization` |
| **LongBench**(long-task benchmark)| Various 4k-32k | Multi-task long ctx | HF Hub `THUDM/LongBench` |

**Issue**:HF Hub download blocked(#34 still pending,Path A/B/C/D 4-fix matrix per `aa00dbe`)。

**Workaround**:use **synthetic long-context** generated locally:
- Concatenate Qwen3-4B model documentation OR
- Use `arle` repo source code(this repo!)concatenated to 40k+ tokens
- 100% offline,no HF Hub blocker

**Recommended**:simple offline data generation script(20-30 LOC Python)
→ tokenize ARLE codebase(README + main docs + select source)to ≥80k
tokens → split into 40k / 64k / 80k examples → save as tokenized JSONL。

## 3. Comparison protocol

### Bench A — Vanilla 40k baseline(no scaling)

```bash
./target/release/arle train eval \
  --model infer/models/Qwen3-4B \
  --data eval-longctx-40k.tokenized.jsonl \
  --seq-len 40960 \
  --backend cuda \
  --metrics-jsonl metrics-A-baseline.jsonl
```

Expected:reasonable PPL on 40k context(Qwen3-4B trained on 40k native)。

### Bench B — YARN factor=2.0,40k context(should match A within 5%)

```bash
./target/release/arle train eval \
  --model infer/models/Qwen3-4B-yarn-f2.0 \
  --data eval-longctx-40k.tokenized.jsonl \
  --seq-len 40960 \
  --backend cuda \
  --metrics-jsonl metrics-B-yarn-40k.jsonl
```

→ Same data,same context size,YARN active。**License**:|PPL_B / PPL_A - 1| ≤ 0.05
(YARN attention_factor adjustment shouldn't degrade quality much at native ctx)。

### Bench C — YARN factor=2.0,64k extended context(measure extension quality)

```bash
./target/release/arle train eval \
  --model infer/models/Qwen3-4B-yarn-f2.0 \
  --data eval-longctx-64k.tokenized.jsonl \
  --seq-len 65536 \
  --backend cuda \
  --metrics-jsonl metrics-C-yarn-64k.jsonl
```

→ Extended context with YARN active。**License**:PPL_C / PPL_A ≤ 1.20
(quality degradation acceptable up to 20%)。

### Bench D(optional)— YARN factor=2.0,80k full extended

```bash
... --data eval-longctx-80k.tokenized.jsonl --seq-len 81920 ...
```

→ Full YARN-extended context。**License**:PPL_D / PPL_A ≤ 1.50。

## 4. License decision matrix

| Comparison | Δ | Decision |
|-----------|----|----------|
| B vs A(40k YARN vs 40k vanilla)| ≤ +5% | ✅ YARN attention_factor 不退化 native quality |
| C vs A(64k YARN vs 40k vanilla)| ≤ +20% | ✅ YARN factor=2 64k quality acceptable |
| D vs A(80k YARN vs 40k vanilla)| ≤ +50% | ⚠ informational(extension limit)|
| Any | > 100% | ❌ KILL — YARN math bug |
| Any | NaN / inf | ❌ KILL — degenerate inv_freq |

## 5. Wall-clock estimate

- Eval data prep:**10-15 min**(local tokenize via Qwen3 tokenizer)
- Bench A 40k:~2-5 min(40k context single forward)
- Bench B 40k YARN:~2-5 min
- Bench C 64k YARN:~5-10 min(longer ctx O(N²) attention)
- Bench D 80k YARN:~10-15 min(if attempted)
- **Total Phase 3b**:**30-50 min** wall-clock

## 6. Memory feasibility check

64k context Qwen3-4B (36 layers, 8 KV heads, 128 head_dim):
- KV @ 64k BF16:**9.4 GB**
- Weights BF16:**8 GB**
- Eval scratch + activations:**1-2 GB**
- **Total**:**~19 GB** — **EXCEEDS 16 GB GPU**

→ 64k eval needs:
- FP8 KV(`--kv-cache-dtype fp8`)→ KV 4.7 GB,total ~14 GB ✓ fits
- OR W4 weights(use `Qwen3-4B-W4-hybrid-zpfix` instead)→ weights 2.5 GB,total ~13 GB ✓ fits

For 80k:KV BF16 11.7 GB → **FP8 mandatory**。

`arle train eval` may not currently expose `--kv-cache-dtype` flag — would
need check or modify。

## 7. Phase 3b execution checklist

```
[ ] Generate eval data via offline script (40k / 64k / 80k examples)
[ ] Verify arle train eval supports --kv-cache-dtype OR add it
[ ] Run Bench A (40k vanilla baseline) → metrics-A.jsonl
[ ] Run Bench B (40k YARN) → metrics-B.jsonl
[ ] Run Bench C (64k YARN with FP8 KV) → metrics-C.jsonl
[ ] Optional Bench D (80k YARN with FP8 KV)
[ ] Compare PPL ratios per §4 license matrix
[ ] Write wins entry: docs/experience/wins/2026-05-10-phase3b-rope-yarn-ppl.md
[ ] Phase 3 全闭合:M_rope-yarn-scaling task #39 mark COMPLETED
```

## 8. Risks

- HF Hub blocker(#34)means need offline eval data — workaround in §2
- 64k context may exceed 16 GB if BF16 KV(forces FP8 KV path)
- Qwen3-4B native train ctx is 40k,YARN factor=2 extends to 80k — may
  degenerate if YARN math drift accumulates(not seen in unit tests but
  empirical risk)

## 9. Cross-references

- M_rope-yarn-scaling plan:`docs/plans/M_rope-yarn-scaling.md`
- Phase 3a smoke PASS:`docs/experience/wins/2026-05-10-phase3a-rope-yarn-server-smoke.md`
- Phase 1+2 wins:`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`
- HF Hub blocker:`docs/research/2026-05-09-34-hf-hub-blocker-audit-fix-paths.md`
- ARLE PPL infra:`crates/train/src/eval_lm.rs:34`(`EvalSummary::ppl()`)
- Setup script:`scripts/setup_qwen3_yarn_config.py`(with `--symlink` per 8cb1be3)

## 10. 状态

Phase 3b ready for execution。30-50 min wall-clock total。Substrate
Phase 1+2 + 3a smoke proven;Phase 3b is quality validation only。Eval
data must be offline-generated due to #34 HF Hub blocker(20-30 LOC
Python script)。
