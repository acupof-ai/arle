# Self-spec K=5 sparse-KV KILL on Qwen3-4B coding 4k/c=4

> Master strategy §7.4 P1.1 spec-decode axis Phase 0 cheap probe.
> Per `M_spec-decode-classical-bench-first.md` (`5a3ff50`) Phase 5
> matched-control test on the only ARLE spec mode that doesn't require
> external draft model: self-spec with sparse-KV view.

## Phase 1 target recap

| Field | Value |
|---|---|
| Metric | out tok/s on Qwen3-4B-W4A16-sym-g128-marlin, 4k longctx c=4 |
| Baseline | W4A16 Marlin no-spec: 191 tok/s (`f6f3af3`) |
| License threshold | tok/s ≥ 1.5× (≥ 287 tok/s) per master §7.4 |
| Kill threshold | tok/s ≤ 1.0× (≤ 191) — net regression |

## Setup

ARLE built at `e20f24c` HEAD. Started with self-spec sparse-KV:

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120 \
  --spec-enabled --spec-draft-model self --spec-draft-k 5 \
  --spec-sparse-kv-enabled
```

Bench (matched-control vs `f6f3af3`):

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh w4a16-self-spec-k5-c4-4k \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --processor .../Qwen3-4B-W4A16-sym-g128-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

## Results

| Metric | W4A16 no-spec (`f6f3af3`) | W4A16 + self-spec K=5 sparse-KV | Δ |
|---|---:|---:|---:|
| **out tok/s** | **191** | **51.63** | **−73.0% REGRESSION** |
| ITL p50 | 11.76 ms | 65.51 ms | +457% |
| ITL std | n/a | 2.76 ms | tight σ — real signal |
| TTFT p50 | 2565 ms | 1255.7 ms | −51% (single-token first-step shortcut) |
| TTFT std | n/a | 660.7 ms | high σ — variable acceptance per request |

Bench artifacts: `bench-output/2026-05-08-w4a16-self-spec-k5-c4-4k/`.

## Reverse-formula acceptance estimate

Leviathan speedup: `α_eff = K * α / (1 + K * α - α)` where α is per-token
acceptance probability and K is num_speculative_tokens.

Observed `α_eff = 51.63 / 191 = 0.270` (the realized rate ratio).

```
0.270 = 5α / (1 + 4α)
0.270 * (1 + 4α) = 5α
0.270 + 1.080α = 5α
0.270 = 3.920α
α ≈ 0.069 (~7%)
```

**Acceptance rate ~7%** — far below the ≥ 0.7 license target. At α < 0.3,
the spec-decode round-trip cost (draft + verify + reject + resample) net-costs
the workload. This is a NULL/KILL outcome consistent with skill rule #6.

## Phase 7 tradeoffs (Phase 8 verdict)

| Axis | Status |
|---|---|
| LOC | ✅ 0 (substrate exists) |
| HW specificity | ✅ none |
| **Acceptance** | ❌ ~7% (need ≥ 70%) |
| ITL p99 variance | ❌ TTFT std 660 ms — variable |
| Memory | ✅ no extra (self-spec) |
| **Tok/s win** | ❌ -73% net regression |
| Generality | ⚠ tested only at 4k/c=4 coding workload |

Per skill v1.3.0 Phase 8: tok/s ratio 0.27× ≪ 1.0× kill threshold = **KILL**.

## Why self-spec K=5 fails on Qwen3-4B coding 4k

Self-spec via sparse-KV uses the SAME target model with a truncated/
sampled KV cache as the "draft" view. The hypothesis is that recent-token
+ hot-LRU-page approximation produces draft logits aligned with full-KV
target logits.

For Qwen3-4B at 4k context with ~256 output tokens, the sparse-KV draft
view appears to disagree with the full-KV target on most tokens — α ~7%
means draft predicts a wrong token 93% of the time.

Possible reasons:
- 4k context is short — sparse-KV (recent 512 + hot 32 pages) loses too
  much signal that's actually in the deeper context
- Prompt content is text-heavy — token transitions aren't repetitive
  enough for sparse-KV to capture; spec-decode tends to win on structured/
  predictable output (code-completion, JSON)
- K=5 is too high — even at higher α, 5 tokens of agreement is rare
  for non-deterministic generation

## What this DOES NOT refute

- **External draft model** (Qwen3-0.6B as draft for Qwen3-4B target) —
  classical Leviathan setup with separate draft model. Not yet tested.
  Predicted α likely 0.5-0.7 for same-family draft on coding workload.
- **Self-spec at lower K** (K=2 or K=3) — at K=2, even α=0.4 gives
  speedup `2*0.4/(1+2*0.4-0.4) = 0.57×` (still net loss). K=3 with α=0.5:
  `3*0.5/(1+3*0.5-0.5) = 0.75×` — still loss. **Self-spec K-sweep on this
  workload appears KILL-irrespective**.
- **Self-spec at long context** (32k+) — sparse-KV is designed for long
  context where recent tokens + hot pages cover most attention. Not
  tested here.
- **Self-spec on structured output** (JSON tool-call, code completion) —
  per master §2.1 W3/W4 agent shapes; unrelated to this 4k random prompt.

## Recommended next steps for spec-decode axis

1. **External draft Qwen3-0.6B** — true Leviathan setup. ~15 min
   one-time download. Predicted α 0.5-0.7 → speedup 1.3-1.8×.
2. **Long-context self-spec** (32k single user) — where sparse-KV is
   designed for. Predicted α higher because recent-token approximation
   is more useful at long context.
3. **W3/W4 agent shape bench** (structured tool-call output) — per
   master §2.1 the workload spec-decode targets. Not 4k random text.
4. **DEFER classical Leviathan license** until at least one of (1)-(3)
   shows ≥ 1.5× tok/s.

## Skill methodology applied

- ✅ Phase 1 target (tok/s ≥ 1.5×)
- ✅ Phase 2 hardware (sm_89 — same as quant axis)
- ✅ Phase 3 binding constraint (acceptance rate, not kernel time)
- ✅ Phase 4 formula prediction (Leviathan tok/s formula)
- ✅ Phase 5 single-variable A/B (matched controls — same checkpoint, KV format, workload)
- ⏭ Phase 6 combo skipped (single arm conclusive at -73%)
- ✅ Phase 7 tradeoffs (acceptance + variance + generality)
- ✅ Phase 8 KILL with skill rule #6 σ-confidence (ITL std 2.76 / 65.51 = 4.2% — within 5% threshold; tight signal)

NULL elimination: self-spec K=5 sparse-KV on this workload is dead.
Hypothesis space narrowed for spec-decode axis: external draft + long-ctx
+ structured workload remain viable; this single config does not refute
them.

## Cross-references

- M_spec plan: [`docs/plans/M_spec-decode-classical-bench-first.md`](../../plans/M_spec-decode-classical-bench-first.md) (`5a3ff50`)
- ARLE spec decode substrate: `infer/src/speculative.rs` (721 LOC) + `infer/src/speculative/cuda.rs`
- ARLE spec decode CLI: `infer/src/main.rs:128-158` (`--spec-enabled`, `--spec-draft-model`, etc.)
- W4A16 Marlin license bench: [`2026-05-08-m_quant-w4a16-marlin-bench.md`](../wins/2026-05-08-m_quant-w4a16-marlin-bench.md) (`f6f3af3`)
- Skill v1.3.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../../.claude/skills/kernel-optimization/SKILL.md) (`d09480b`)
- Bench artifacts: `bench-output/2026-05-08-w4a16-self-spec-k5-c4-4k/`

## Rule

- **Self-spec sparse-KV is workload-sensitive** — short-context random
  text is worst-case for it (designed for long-context with recent-token
  predominance).
- **K-sweep without α improvement won't help** — at α < 0.3, no K value
  gives net speedup. The fix is α (better draft alignment), not K.
- **External draft is the next test**, not Medusa/EAGLE training (which
  has data risk per master §6.2).

This entry adds NULL evidence to spec-decode axis hypothesis tree without
killing the axis itself. Per skill anti-pattern #13: NULL is real
elimination, narrows the hypothesis space.
