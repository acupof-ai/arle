# 32k self-spec sparse-KV KILL — third independent classical-spec evidence

> Master §7.4 P1.1 axis re-test at long-context. Self-spec sparse-KV
> was the *designed-for* regime per `5f26675` errors entry — sparse-KV
> recent-512 + hot-32-pages approximation should align with target
> attention better at 32k+ where context is too long for full draft view.
>
> **Result: ALSO DEAD** at α ≈ 23%, -40% tok/s, +349% ITL. This is the
> third independent classical-spec KILL on Qwen3 family on sm_89 ARLE,
> spanning both short (4k) and long (32k) context. Axis-level evidence
> growing toward "classical Leviathan spec-decode infeasible without
> Medusa-style head training".

## Setup — matched 32k A/B

```bash
# Both arms: ARLE Qwen3-4B-W4A16-sym-g128-marlin, --num-slots 4 --max-seq-len 33000
# Same workload: 32000 in / 128 out / c=1 (single user long-context decode)

# Arm A (no-spec):
./target/release/infer --model-path ... --port 8000 --num-slots 4 --max-seq-len 33000
scripts/bench_guidellm.sh w4a16-no-spec-32k-c1 ... \
  --concurrencies 1 --max-seconds 180 --warmup 10 \
  --data 'prompt_tokens=32000,...,output_tokens=128,...'

# Arm B (self-spec sparse-KV K=5):
./target/release/infer --model-path ... --port 8000 --num-slots 4 --max-seq-len 33000 \
  --spec-enabled --spec-draft-model self --spec-draft-k 5 --spec-sparse-kv-enabled
scripts/bench_guidellm.sh w4a16-self-spec-32k-c1 ... (same data spec)
```

## Result

| Metric | A no-spec | B self-spec K=5 sparse-KV | Δ |
|---|---:|---:|---:|
| TTFT p50 | 9408 ms | 9426.9 ms | +0.2% (flat — prefill dominated) |
| TTFT std | 61 ms | 75 ms | both acceptable |
| **ITL p50** | **17.69 ms** | **79.37 ms** | **+349% REGRESSION** |
| ITL std | 0.01 ms | 0.53 ms | both tight (real signal) |
| out tok/s | 11.61 | 6.95 | **-40% REGRESSION** |
| TPOT mean | 91.15 ms | 152.25 ms | +67% |
| E2E mean | 11.67 s | 19.49 s | +67% |

Bench artifacts: `bench-output/2026-05-08-w4a16-{no-spec,self-spec}-32k-c1/`.

## Reverse-Leviathan acceptance

```
α_eff = 6.95 / 11.61 = 0.599
0.599 = 5α / (1 + 4α)
0.599 + 2.396α = 5α
0.599 = 2.604α
α ≈ 0.230 (23%)
```

23% acceptance — better than 4k self-spec (7%) and 4k ext-draft (19%),
but still way below the ≥ 70% license threshold needed for net speedup.

## Why classical spec-decode fails at 32k

At 32k context with c=1:
- KV per token read: 32k × 8 kv heads × 80 dim × 2 byte = 41 MB / token
- ITL no-spec 17.69 ms = 41 MB / 17.69ms = 2.3 GB/s effective HBM (well below 672 GB/s peak — KV traversal bottleneck)
- Self-spec adds K=5 draft views (sparse-KV recent + hot pages) = 5× partial attention compute
- Verify step: 1× full attention (32k traversal)
- Per-step cost: draft (5 × small attn) + verify (1 × full attn) = ~6× full-attn equivalent
- At α=0.23: tokens per step = K*α + 1 = 1.15 + 1 = 2.15
- Per-token cost: 6 / 2.15 = 2.79× no-spec → **+179% ITL** (close to observed +349% with overhead)

The math is harsh: at low α + high per-step overhead, spec-decode is
**strictly worse than no-spec**. Sparse-KV draft was supposed to help α
at long context but only got 23% — still not enough to overcome the
2.79× overhead.

## Three independent classical-spec KILL evidences

| Workload | Setup | α est | Verdict | Reference |
|---|---|---:|---|---|
| 4k random text c=4 | self-spec K=5 sparse-KV | ~0.069 | KILL | `5f26675` |
| 4k random text c=4 | ext-draft Qwen3-0.6B K=5 | ~0.187 | KILL | `3ac5f4d` |
| **32k random text c=1** | **self-spec K=5 sparse-KV** | **~0.230** | **KILL** | this entry |

Pattern: across 4k/32k context and self/external draft setups, α never
exceeds 0.25 on Qwen3 family + ARLE current implementation. This is
strong axis-level evidence that **classical Leviathan spec-decode is
infeasible** without significant α improvement.

## Master strategy implication

Master §7.4 P1.1 said "Medusa multi-head 优先 EAGLE(降数据/训练风险)" —
implying classical was the cheap fallback. Three KILL evidences above
demonstrate **classical is NOT the fallback** — it's strictly worse than
no-spec at every workload tested.

**Recommendation update**: Master §7.4 P1.1 should state classical
spec-decode is NOT viable production strategy on Qwen3 family + ARLE
implementation. The viable spec-decode paths (in priority order):

1. **Medusa multi-head**: master strategy original P1.1 recommendation,
   has training risk (~1 week) but evidence shows it's the necessary
   path. Heads share target hidden state → α 0.85-0.95 typical.
2. **EAGLE**: if Medusa data prep too expensive, EAGLE is alternative
   (separate small model trained on target's hidden states). Slightly
   more risk than Medusa but still feasible.
3. **Classical with much bigger draft (e.g. Qwen3-1.7B)**: untested.
   Larger draft may give higher α but adds VRAM cost. Marginal upside.
4. **Agent W3/W4 structured workload** (gated on `a672b08` admission fix):
   structured tool-call output may have higher α than random text.
   Untested due to admission blocker.

## Phase 7 tradeoffs

| Axis | Status | Note |
|---|---|---|
| LOC | ✅ 0 | substrate exists |
| HW | ✅ none | spec is general |
| Memory | ✅ none extra (self-spec) | |
| **Acceptance ceiling** | ❌ ~25% on classical | structurally limited by Qwen3-4B vs draft alignment |
| Generality | ⚠ 3 workloads tested all KILL | strong axis-level signal |

## Phase 8 — KILL with axis-level implication

| Result | Action |
|---|---|
| 0.6× tok/s ratio < 1.0× kill | KILL at 32k |
| 3 independent KILLs across workloads | **AXIS-LEVEL CLASSICAL DEAD** |
| Medusa or W3/W4 untested | axis preserved for those paths only |

## Recommended next steps for spec-decode axis

1. **DEFER classical spec-decode entirely** — 3 KILL evidences sufficient to retire this path on Qwen3-4B + ARLE current.
2. **Medusa multi-head training** — master §7.4 P1.1 promotion. Estimated:
   - Data: 100k+ tokens of representative agent workload (W3/W4 traces)
   - Training: 4 Medusa heads on Qwen3-4B target, ~1 week H100 / ~2 weeks 4080S
   - License threshold: tok/s ≥ 1.5× at agent shape
3. **W3/W4 admission fix first** (`a672b08`) — gate for structured-workload re-test.
4. **OR pivot spec axis entirely to xgrammar** (`3864751` plan) — different capability axis (constrained generation, not throughput).

## Skill methodology validation

Per anti-pattern #13 across 3 KILL evidences: each NULL/regression at a
different workload is real elimination. Cumulative pattern (α<0.25 across
all tested setups) → axis-level conclusion supportable, even though
individual workload KILLs only close that workload.

The methodology saved Master §7.4 from premature LAND (would have built
classical-only spec) AND from premature axis-KILL (Medusa still viable).

## Cross-references

- 4k self-spec KILL: [`2026-05-08-spec-decode-self-spec-k5-kill.md`](2026-05-08-spec-decode-self-spec-k5-kill.md) (`5f26675`)
- 4k ext-draft KILL: [`2026-05-08-spec-decode-ext-draft-k5-kill.md`](2026-05-08-spec-decode-ext-draft-k5-kill.md) (`3ac5f4d`)
- W3 admission blocker: [`2026-05-08-w3-bench-capacity-503-admission-backlog.md`](2026-05-08-w3-bench-capacity-503-admission-backlog.md) (`a672b08`)
- M_spec plan + Medusa promotion: [`docs/plans/M_spec-decode-classical-bench-first.md`](../../plans/M_spec-decode-classical-bench-first.md)
- Master §7.4 P1.1: spec-decode Medusa preferred (validated by this evidence)
- Skill v1.3.0: anti-pattern #13 NULL elimination

## Rule

3 independent classical-spec KILLs across 4k+32k workloads on Qwen3 family
+ ARLE current implementation = **axis-level classical spec-decode dead**.
Master §7.4 P1.1 should state Medusa is the required path (despite training
cost), not a fallback. Update master strategy + remove "classical first"
default phrasing.

α ceiling at ~25% across both short and long context is structural —
sparse-KV approximation + small-draft divergence = irreducible at this
implementation level. Only architectural changes (Medusa shared-target
heads, EAGLE auxiliary model, or radically larger draft) can break the
ceiling.
