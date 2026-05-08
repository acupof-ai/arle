# M_spec — Classical spec decode bench first (Phase 0 cheap verify before Medusa)

> Master strategy §7.4 P1.1 says Medusa multi-head preferred over EAGLE.
> But ARLE *already has* classical Leviathan-style spec decode substrate
> at `infer/src/speculative.rs` (721 LOC). Before adding Medusa heads (a
> draft-model training task with data + accuracy risk), bench what
> classical spec decode delivers on Qwen3-4B today. Phase 0 is cheap
> evidence; Medusa investment licenses on it.
>
> Trigger: R4 #6 (Marlin hybrid dispatch) lands. Spec decode is the next
> lever per §0.1 axis 3 ("投机"). Plan today, run after R4 #6 push.
> Per kernel-optimization skill v1.2.0 — formula → matched A/B → tradeoff.

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | tok/s (out tokens/sec) on Qwen3-4B coding/agent shape |
| Baseline (no spec) | from `f6f3af3` Marlin: 191 tok/s out at c=4 longctx 4k |
| **License** | tok/s ≥ **1.5×** (≥ 287 tok/s) per master §7.4 acceptance ≥70% target |
| Soft win | 1.2-1.5× (229-287 tok/s) — proceed but flag low acceptance |
| Kill | < 1.2× — classical spec is sub-license; pivot Medusa or KILL spec axis on Qwen3-4B |
| Wall-clock budget | 30-60 min (3 acceptance rates × n=3 each) |

## Phase 2 — Hardware

sm_89 RTX 4070 Ti SUPER · same as M_quant. Spec decode shares Marlin/BF16 GEMM
paths so binding constraint is on the dispatch + acceptance side, not kernel.

## Phase 3 — Binding constraint (formula-grounded)

Classical spec decode formula (Leviathan 2023):

```
speedup = K * α / (1 + K * α - α)
where K = num_speculative_tokens (4-8 typical)
      α = mean acceptance rate (varies by task; 70-85% for code/structured outputs)
```

For Qwen3-4B coding agent (high-prefix-overlap workload, structured tool-call output):

| K | α=0.7 | α=0.8 | α=0.9 |
|---:|---:|---:|---:|
| 4 | 1.85× | 2.13× | 2.43× |
| 6 | 2.07× | 2.50× | 3.00× |
| 8 | 2.20× | 2.78× | 3.46× |

**Binding constraint** for spec decode is the acceptance rate α, which is
draft-target alignment + workload structure. Kernel time is unchanged
(target model decoder runs once per K+1 verification step).

## Phase 4 — Formula prediction (Qwen3-4B coding agent)

Plausible α range from literature on code/JSON generation:
- α ≥ 0.85 if draft model is Qwen3 family (same tokenizer + similar training)
- α ≈ 0.75 if generic small draft (e.g. distilled checkpoint)
- α ≈ 0.5 if mismatched draft architecture

ARLE substrate cites `DEFAULT_QWEN3_DRAFT_MODEL_ID` — same family, expected
α ≥ 0.80 baseline.

Predicted range: 1.85× to 2.78× tok/s vs no-spec baseline.

License threshold 1.5× → **almost certain pass** at K=4-6, α≥0.7.

## Phase 5 — Single-variable A/B (matched controls)

Variable: `--enable-speculative` flag (or equivalent ARLE env / CLI).
Controls:
- Same checkpoint: `Qwen3-4B-W4A16-sym-g128-marlin` (post-R4 #6 production)
- Same KV dtype: auto (FP8 KV per skill v1.2.0 isolation-motive callout)
- Same `--num-slots 8 --max-seq-len 5120`
- Same workload spec (4096 in / 256 out, c=4, max-seconds=120, warmup=10)
- Same draft model (default per `DEFAULT_QWEN3_DRAFT_MODEL_ID`)
- σ < 5% n=3 mandatory (acceptance rate has run-to-run variance from sampling)

3 arms (matched, single variable = K and acceptance):

| Arm | spec? | K | Note |
|---|---|---|---|
| A — baseline | OFF | n/a | Marlin all-batch (Arm B from R4 #6 if R4 lands) |
| B — spec K=4 | ON | 4 | classical Leviathan, default `num_speculative_tokens=4` |
| C — spec K=6 | ON | 6 | longer speculation, higher tok/s if α stays high |

### Bench command (TBD CLI surface — depends on ARLE flag wiring)

```bash
# Inspection step: find ARLE spec decode CLI/env
grep -nE "num_speculative|enable_spec|speculative" infer/src/main.rs

# Bench (placeholder — adjust per actual flag)
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh m_spec-classical-K4-c4-4k \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

If spec-decode requires a separate draft model checkpoint: download or quantize
Qwen3-0.6B (or similar small Qwen3) as draft model first. ~15 min one-time setup.

## Phase 6 — Combinational A/B (post-license)

If Phase 5 wins ≥ 1.5×, sweep:

| K \ shape | longctx 4k | longctx 8k | high-conc 1k/256 | multi-tenant |
|---|---|---|---|---|
| 4 | sweep cell | | | |
| 6 | sweep cell | | | |
| 8 | sweep cell | | | |

Hypothesis: K may saturate around 4-6 for general workload but 8+ for code
(higher α for structured output).

## Phase 7 — Tradeoffs (skill mandatory)

| Axis | Status | Note |
|---|---|---|
| LOC | ✅ 0 (substrate exists) | Just wiring + bench |
| HW specificity | ✅ none | Spec decode is GEMM-format-agnostic |
| **Latency variance** | ⚠ accepts variable per request | Some requests bottlenecked by α=0; introduces ITL p99 noise — measure |
| Numerical correctness | ✅ Leviathan math is bit-identical | Already in `verify_tokens_greedy` (per source comment) |
| Memory budget | ⚠ +draft model VRAM | Qwen3-0.6B BF16 ≈ 1.2 GB; or Qwen3-0.6B Marlin ≈ 0.3 GB. Acceptable on 16 GB GPU |
| Generality | ⚠ workload-dependent α | Code/agent good; long-form prose maybe worse — multi-shape gate |
| Scheduling impact | ⚠ verification step is K+1 batch | Increases per-step batch. Multi-tenant may benefit; high-conc may shift |
| **Acceptance rate floor** | ❌ if α < 0.5 net regression | Sampling tok/s lower than no-spec when α drops too low |

## Phase 8 — License-or-kill

| Result | Action |
|---|---|
| tok/s ≥ 1.5× AND acceptance ≥ 0.7 | LAND classical spec; Medusa is now incremental — defer P1.1.b |
| tok/s 1.2-1.5× AND acceptance 0.5-0.7 | LAND with note; Medusa P1.1.b worth pursuing for higher α |
| tok/s ≤ 1.2× OR acceptance < 0.5 | KILL classical; Medusa is required for P1.1 axis (data+training risk per master §6.2) |
| ITL p99 regression > +20% | KILL — variance unacceptable for production |
| greedy_consistency divergence | KILL — verifier broken |

## Pre-execution checklist

Before running this plan post-R4 #6:

- [ ] R4 #6 LANDED or KILLED (spec decode benches Marlin hybrid baseline,
      not Marlin-all-batch). If R4 #6 KILLED, baseline = `f6f3af3` Marlin.
- [ ] Draft model checkpoint prepared (ARLE `DEFAULT_QWEN3_DRAFT_MODEL_ID`
      or local Qwen3-0.6B-Marlin)
- [ ] Spec decode CLI surface understood (`grep speculative infer/src/main.rs`)
- [ ] σ < 5% confirmed on n=3 (acceptance rate has run variance; 3 runs may need
      bumping to n=5 if sampling-RNG variance high)

## Cross-references

- Existing substrate: `infer/src/speculative.rs` (721 LOC, classical Leviathan)
- Master strategy §7.4 P1.1: Medusa multi-head preferred over EAGLE — but classical bench first
- Master §6.2 moat capability 4: spec decode (✓ if classical works, ⏳ if needs Medusa)
- Skill v1.2.0: `.claude/skills/kernel-optimization/SKILL.md` — Phase 5 matched controls + isolation-motive
- Existing spec plans: `2026-05-01-longctx-spec-decode-phase2.md`, `M_c-hybrid-spec-rollback.md`, `M_d-tier-kv-spec-decode-coordination.md`, `longctx-spec-tilelang-combo.md`
- W3/W4 agent shapes (`2026-05-02-agent-load-bench-spec.md`) — code/tool-call structured output favors high α

## Rule (per `feedback_docs_priority_roi_evidence.md`)

- **Don't license Medusa investment on hand-wave**. Classical spec bench
  must show < 1.5× before Medusa LOC budget approves.
- **Multi-shape mandatory** (Phase 6) — agent W3 short multi-turn vs
  W4 tool-call resume have different α regimes.
- **Same KV format both arms** — repeat skill v1.2.0 isolation-motive
  rule. Don't force baseline KV vs spec-decode KV.
