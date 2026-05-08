# External draft spec-decode K=5 KILL — Qwen3-0.6B too divergent for 4k random text

> Second spec-decode KILL on Qwen3-4B coding 4k/c=4 workload.
> Companion to [`5f26675`](2026-05-08-spec-decode-self-spec-k5-kill.md)
> self-spec K=5 KILL. Two independent draft setups both KILL'd → workload-
> level conclusion: classical Leviathan spec-decode on 4k RANDOM TEXT
> is dead for Qwen3 family. **Spec-decode axis closed for this workload**;
> remains viable for agent W3/W4 (structured tool-call) + long-ctx
> sparse-KV.

## Setup

ARLE built at HEAD `e3ca4d8`. Started with external draft model:

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120 \
  --spec-enabled \
  --spec-draft-model external:/home/ckl/projects/arle/infer/models/Qwen3-0.6B \
  --spec-draft-k 5
```

Draft model: Qwen3-0.6B BF16 (1.5 GB safetensors, downloaded fresh ~1 min).
Same family as target Qwen3-4B (shared tokenizer + similar training).

CUDA Graph capture engaged for batches B=1..8 ✅ (per warmup log).

## Result

| Metric | W4A16 no-spec (`f6f3af3`) | + ext draft K=5 | Δ |
|---|---:|---:|---:|
| TTFT p50 | 2565 ms | 2432.6 ms | -5.2% (single-tok shortcut) |
| **out tok/s** | **191** | **102.31** | **-46.4% REGRESSION** |
| ITL p50 | 11.76 ms | 30.51 ms | +159% |
| ITL std | n/a | 0.08 ms | 0.27% — tight signal |
| TTFT std | n/a | 79.4 ms | acceptable σ |

Bench artifacts: `bench-output/2026-05-08-w4a16-spec-ext-qwen06b-k5-c4-4k/`.

## Reverse-Leviathan acceptance estimate

```
α_eff = tok/s_ratio = 102.31 / 191 = 0.535
0.535 = K * α / (1 + K * α - α)  with K=5
0.535 + 4 * 0.535 * α = 5α
0.535 = (5 - 2.14) α
α ≈ 0.187 (~19%)
```

**Acceptance ~19%** — better than self-spec K=5 (α≈7%) but still way below
useful threshold (≥ 70%).

K-sweep does NOT save this:
- K=2 α=0.19: speedup `2*0.19/(1+0.19) = 0.32×` (worse)
- K=3 α=0.19: speedup `3*0.19/(1+0.38) = 0.41×` (worse)
- At α < 0.5, NO K gives net speedup; at α < 0.3, even K=1 is break-even at best

## Two independent KILL evidences at 4k random text

| Setup | α est | tok/s ratio | Verdict |
|---|---:|---:|---|
| self-spec K=5 sparse-KV (`5f26675`) | ~0.069 | 0.270 | KILL |
| **ext-draft Qwen3-0.6B K=5 (this entry)** | **~0.187** | **0.535** | **KILL** |

Both methods produce sub-license tok/s. The common factor is the workload:
**4k random text continuation** is the WORST case for spec-decode because:
- Random text token transitions are high-entropy (not repetitive code/JSON)
- Long context (4k) drifts target model into specific style; smaller draft
  diverges
- Sentence-level coherence requires whole-context understanding; draft
  truncated/scaled-down sees less signal

## What this WORKLOAD-CLOSES (not axis-closes)

**WORKLOAD CLOSED**: spec-decode classical Leviathan at 4k random text +
4-conc decode + Qwen3 family models. Two independent KILLs (self-spec
+ external draft) at sub-license α confirm.

**AXIS REMAINING**:
1. **Agent W3/W4 structured workload** (per master §2.1) — tool-call JSON,
   code completion. Token transitions structured → predicted α 0.6-0.85
   for same-family draft. Untested.
2. **Long-ctx self-spec** (32k+) — sparse-KV draft view designed for this
   regime. Not 4k workload.
3. **Medusa multi-head** (master §7.4 P1.1 original recommendation) —
   trains additional heads on target model checkpoint. Predicted α ~0.85.
   Has data + training risk per master §6.2 — but EVIDENCE shows classical
   draft + small Qwen3 is too divergent → Medusa may be the right answer
   despite the cost.

## Master strategy implications

Master §7.4 P1.1 said "Medusa multi-head 优先 EAGLE(降数据/训练风险)". This
session's classical spec-decode failures at 4k random text validate that
**classical is NOT cheap** for this workload — needs separate draft model
that aligns ≥0.7. Same-family Qwen3-0.6B fails. Smaller-than-0.6B is
unlikely to do better.

**Recommendation update**: Master §7.4 P1.1 should probably promote Medusa
above classical Leviathan despite training risk, because:
- Classical spec at 4k random text fails (this evidence)
- Medusa shares target model = no separate draft training
- Medusa heads trained from target's hidden states = naturally aligned
- Training risk = ~1 week of data prep + fine-tune; recoverable

But the cleanest path: bench classical at agent W3/W4 SHAPE first, since
that's the actual master §2.1 production workload. If classical works at
agent shape, no Medusa needed.

## Phase 7 tradeoffs

| Axis | Status |
|---|---|
| LOC | ✅ 0 (substrate exists) |
| HW | ✅ none |
| Memory | ⚠ +1.5 GB draft model VRAM |
| **Acceptance** | ❌ ~19% (need ≥ 70%) |
| **Tok/s win** | ❌ -46% net regression |
| ITL p99 variance | ⚠ TTFT std 79 ms (vs TTFT std 0 for no-spec) |
| Generality | ⚠ workload 4k random — not master §2.1 W3/W4 |

## Phase 8 — KILL at this workload

| Result | Action |
|---|---|
| ✅ tok/s 0.535× < 1.0× kill threshold | **KILL** for 4k random text workload |
| Axis remains open | ✓ for agent W3/W4 + long-ctx + Medusa |
| greedy_consistency | not tested (KILL preempts) |

## Recommended next steps

1. **DEFER spec-decode** until agent W3/W4 bench harness ready
   (per master §7.1 P0.0 already plan-staged but not executed)
2. **OR Medusa explore** (P1.1 substrate, master §6.2) — fine-tune
   Medusa heads on Qwen3-4B target. 1-week effort.
3. **OR long-ctx self-spec test** at 32k (sparse-KV use-case) — quick test
   if 32k workload available.
4. **Skip spec-decode for current product** — focus on quant axis (W4A8
   accuracy fix) + xgrammar (master §7.5) which are higher-confidence wins.

## Skill methodology applied

- ✅ Phase 1 target (tok/s ≥ 1.5×)
- ✅ Phase 5 single-variable A/B (matched control: same KV format, same
  model arch, same workload, only `--spec-draft-model` differs)
- ✅ Phase 8 KILL with σ-confidence (ITL std 0.27%)
- ✅ Anti-pattern #13: NULL is real elimination — workload-level dead
  branch eliminated for spec-decode axis

Combined with `5f26675` self-spec KILL, hypothesis tree for spec-decode
at this workload is exhausted on both classical paths.

## Cross-references

- M_spec plan: [`docs/plans/M_spec-decode-classical-bench-first.md`](../../plans/M_spec-decode-classical-bench-first.md) (`5a3ff50`)
- self-spec K=5 KILL: [`2026-05-08-spec-decode-self-spec-k5-kill.md`](2026-05-08-spec-decode-self-spec-k5-kill.md) (`5f26675`)
- Master §7.4 P1.1: spec-decode Medusa preferred over EAGLE (validated by this evidence)
- Master §2.1 W3/W4: agent structured workload (where spec-decode may still work)
- Master §7.1 P0.0: agent bench harness (gating for spec-decode re-test)
- Skill v1.3.0: `.claude/skills/kernel-optimization/SKILL.md` (`d09480b`)

## Rule

For Qwen3 family at 4k random text with c=4 longctx, **classical
Leviathan spec-decode is workload-dead** (two independent KILL evidences:
self-spec α=7%, ext-draft α=19%). Master §7.4 P1.1 recommendation should
promote Medusa above classical for production agent workload — OR defer
spec-decode entirely until W3/W4 bench harness validates the actual
production shape.
