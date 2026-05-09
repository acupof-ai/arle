---
title: #36 follow-up bench spec — warm/cold prefix-mix workload (gate-mechanism prove)
date: 2026-05-10
type: research
status: ready-pending-codex-arm-b-finish
---

# #36 follow-up bench spec — warm/cold prefix-mix workload (gate-mechanism prove)

> Codex's #36 arm B (PrefixAware) bench is in flight (2m 24s into 120s
> window) but codex flagged: "workload 无 session/prefix hits，仍未必能
> 证明多租户收益". Current bench can show whether
> `prefix_aware_admit_deferrals_total` increments (gate fires), but
> can't measure the actual win mechanism (warm session reuse) because
> all requests are unique random prompts.
>
> This spec defines the follow-up bench: a deterministic JSONL workload
> with controlled warm/cold mix that exercises both gate-firing AND
> the warm-session reuse path the gate is supposed to enable.

## Problem with current bench (anticipated in 5e902da Open Question)

guidellm 0.6.0 synthetic data spec
`prompt_tokens=2048,prompt_tokens_stdev=512` generates **fully random
unique prompts** per request. From a prefix-cache perspective:

- Every request looks "cold" (no shared prefix with prior requests)
- `prefix_aware_admission_signals.is_cold_request()` always returns true
- The gate may fire (depends on `cold_soft_cap` triggering under queue
  pressure) but there are no warm requests to be admitted bypass-style
- A/B comparison measures only "queue-bound vs cold-soft-cap-deferred"
  which is a **degenerate** PrefixAware test

This is the same family as kernel-optimization skill v1.9.0 anti-pattern
#26 ("Smoke-test small-shape success ≠ production-shape success") —
the bench scope must match the mechanism under license.

## Generator script — `scripts/gen_36_warm_prefix_mix.py`

Created this tick (commit pending). Produces a JSONL with controlled
warm/cold mix:

```bash
./scripts/gen_36_warm_prefix_mix.py \
    --tokenizer infer/models/Qwen3-4B/tokenizer.json \
    --out bench-output/36-warm-mix.jsonl \
    --num-requests 256 \
    --warm-fraction 0.6 \
    --num-sessions 4 \
    --shared-prefix-tokens 1024 \
    --tail-tokens 256 \
    --output-tokens 128
```

Output structure:
- 256 total requests
- ~154 warm: 4 sessions × ~38 requests each, sharing a 1024-token prefix
  per session (above 256 typical block size → multi-block reuse)
- ~102 cold: unique random 1280-token prompts

Workload size on disk: ~1.5MB JSONL (text-decoded prompts), each
~1280-token prompt × 256 requests × ~5 chars/token ≈ 1.6M chars.

Determinism: `--seed 0xA12E` → identical workload across runs (matched
control). Same JSONL drives both arms A and B — no per-arm RNG drift.

## Bench spec

```bash
# Step 1: generate workload (one-time, reusable)
./scripts/gen_36_warm_prefix_mix.py \
    --tokenizer infer/models/Qwen3-4B/tokenizer.json \
    --out bench-output/36-warm-mix.jsonl \
    --num-requests 256 --warm-fraction 0.6 --num-sessions 4 \
    --shared-prefix-tokens 1024 --tail-tokens 256 --output-tokens 128

# Step 2: arm A — QueueBound baseline
# Stop existing server, restart with:
#   --admission-policy queue-bound --max-waiting-requests 4 --num-slots 8 ...
# Then:
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh 36-warmmix-A-queuebound \
    --concurrencies 8 --max-seconds 120 --warmup 10 \
    --target http://127.0.0.1:8765 \
    --model Qwen3-4B-W4-hybrid-zpfix \
    --processor infer/models/Qwen3-4B \
    --data bench-output/36-warm-mix.jsonl

# Step 3: arm B — PrefixAware
# Stop server, restart with:
#   --admission-policy prefix-aware --max-waiting-requests 4 --num-slots 8 ...
# Then same bench command but:
  scripts/bench_guidellm.sh 36-warmmix-B-prefixaware \
    --concurrencies 8 --max-seconds 120 --warmup 10 \
    --target http://127.0.0.1:8765 \
    --model Qwen3-4B-W4-hybrid-zpfix \
    --processor infer/models/Qwen3-4B \
    --data bench-output/36-warm-mix.jsonl

# Step 4: capture /v1/stats post-bench for both arms
# Look for:
#   - prefix_aware_admit_deferrals_total > 0 (gate fired)
#   - matched_prefix_tokens distribution (warm reuse evidence)
#   - prefix hit rate per request
```

## Gate-license matrix

| Metric | Arm A (QB) expected | Arm B (PA) expected | License if Δ |
|--------|---------------------|---------------------|--------------|
| `prefix_aware_admit_deferrals_total` | 0 | > 0 | gate fires (sub-license) |
| Warm-session p50 TTFT | high (no admission preference) | low (warm bypass) | -20%+ → license PA |
| Cold-request p50 TTFT | high | high or higher | regression OK if warm wins |
| Aggregate throughput | baseline | ≥ baseline | -5% kill, ≥ +10% license |
| `matched_prefix_tokens` p50 | low (random distribution) | high on warm | proves prefix reuse |
| Warm-vs-cold p99 fairness | n/a | warm < 3× cold | starvation kill ≥ 3× |

If arm B's gate fires (counter > 0) AND warm-session p50 TTFT improves
≥ 20%: **license PrefixAware as the multi-tenant default**.

If gate fires but no warm-session perf delta: PrefixAware substrate
works but real-world workload doesn't benefit at this concurrency / mix
ratio. Errors entry: "PrefixAware works at substrate, no perf delta at
this workload" — does NOT kill the feature, may license at a different
op-point.

If gate does NOT fire (counter = 0) even with warm-mix workload:
substrate bug (gate not wired correctly to plan) — open follow-up
investigation. Unlikely given pre-build audit (60ffa41).

## Companion to current bench (codex's arm A + arm B)

Codex's current arm A + arm B are still useful evidence:
- Arm A: confirms baseline QueueBound behavior (cold-only stress)
- Arm B: confirms gate fires under cold-only pressure (proves
  counter wiring works even without warm reuse)

This warm-mix follow-up adds the **mechanism evidence** layer that the
synthetic-random workload lacks. Both layers needed for the wins
entry to be complete:
- Layer 1 (synthetic random): "gate substrate wired correctly"
- Layer 2 (warm-mix): "PrefixAware actually helps when prefix reuse
  exists"

## Pickup brief for codex (post-arm-B)

After codex's current arm B finishes + writes wins/errors entry for
synthetic-random A/B:

1. Read this spec
2. Run `scripts/gen_36_warm_prefix_mix.py` once to produce JSONL
3. Repeat A/B bench protocol with `--data bench-output/36-warm-mix.jsonl`
4. Capture /v1/stats for both arms
5. Add a §"Layer 2 — Warm-mix mechanism evidence" section to the
   wins/errors entry (or open a follow-up entry if the synthetic-random
   one is already shipped)
6. Decision: license-or-kill PrefixAware as multi-tenant default per
   gate matrix above

Wall time: 5min generator run + 10min bench × 2 arms + 0.5h doc =
~30min total.

## Cross-references

- Substrate survey: `docs/research/2026-05-10-36-prefix-aware-admission-substrate-complete-bench-pending.md`
  (5e902da, "Open question — does this workload actually exercise the gate")
- Counter audit: `docs/research/2026-05-10-36-codex-counter-audit-clean.md` (60ffa41)
- Brief gap research: `docs/research/2026-05-10-36-brief-gap-bench-server-restart-protocol.md` (0f4d0ae)
- Generator script: `scripts/gen_36_warm_prefix_mix.py` (this tick)
- Skill anti-pattern #26 (smoke-shape ≠ production-shape):
  `.claude/skills/kernel-optimization/SKILL.md`
- Counter implementation: `infer/src/scheduler/cuda/runtime/admission.rs:349`
  (codex 079639c)

## 状态

Follow-up bench spec ready, JSONL generator script ready, deterministic
workload defined. Awaits codex's arm B finish + synthetic-random wins/
errors entry land, then codex (or Claude) runs Layer 2 warm-mix bench.
The two-layer evidence stack (substrate works + mechanism helps) is
the complete #36 wins-entry shape.
