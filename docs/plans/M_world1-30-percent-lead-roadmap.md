# M_world1 — World #1 inference engine (lead #2 by 30%+)

> ⚠️ **Strategic master**: [`../projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md)
> 重定义产品 = **Rust-native coding/agent runtime**(非 generic LLM serving)。
> 本 M_world1 plan 的 4 canonical shapes 框架在 master §2.3 重排为 agent
> 相关度;**真正 mission threshold 用 W3/W4 1.30× margin**(per
> [`2026-05-02-agent-load-bench-spec.md`](2026-05-02-agent-load-bench-spec.md))。
> 本 plan 的 P0.1 baseline 完成(`12c4c86` + `4ae3b7b` SGLang/ARLE 4k/8k 跨引擎),
> 后续 P1/P2 行动以 master §7 为准。

> User directive 2026-05-07: target = world #1 inference engine,
> 领先第二 30%+. NOT just "beat vLLM". Reframes priority stack
> from competitive parity to commanding lead.

## Priority & ROI

**Priority**: **P0 strategic** — defines the bar for all future
plan acceptance. Any optimization that doesn't move ARLE toward
"30% past #2 competitor" gets demoted.

**ROI basis**:
- Current ARLE position by workload (vs vLLM only):
  - High-conc 1k/256/c=64: ARLE 843 vs vLLM 647 = +30.3% ✓ ALREADY at target
  - Long-ctx 4k/c=4: ARLE 153.9 vs vLLM 159.1 = -3.4% ✗ FAR FROM target
  - Multi-tenant shared-prefix: ARLE 318 vs vLLM 573 ms TTFT = +80% ✓ ALREADY past target
  - Long-ctx 8k/c=4: TBD (need re-bench post-fix)
- **Real #2 competitor unclear** without direct measurement:
  - SGLang v0.5 has zero-overhead scheduler (potentially > vLLM)
  - TensorRT-LLM claims 16× speedup vs BF16 (vendor vague baseline)
  - LMDeploy / TGI also candidates
- 30% lead = market-defining position. Worth significant
  engineering investment. Per memory rule, this is "research-y
  work" → Phase 0 license-or-kill required before committing.

**Negative case**:
- If #2 is TensorRT-LLM at 1.5× vLLM at same shape, ARLE 30% past
  TRT = 2× vLLM. May be unachievable on 4080S consumer hardware
  (TRT optimizes for H100-class).
- "30% lead at every shape" may force impossible trade-offs (high-
  conc vs long-ctx kernel choices conflict).
- Tail latency, correctness, and stability matter — not just peak
  throughput. "World #1" means production-grade, not benchmark
  cheating.

**Kill criteria**:
- Phase 0 baseline measurement shows current #2 already exceeds
  ARLE by > 50% at 2+ workload shapes → reframe to "competitive
  niche" not "world #1"
- Required engineering investment > 6 weeks of focused work →
  reassess scope (might pursue specific shape niche rather than
  full coverage)
- Specific shape's 30% target requires kernel work that risks
  correctness regression → defer that shape, ship 30% lead at
  others

## Phase 0 — Establish real #2 baseline (1-2 days)

Currently we benchmark only vs vLLM. **vLLM may not be #2**. To
target "30% past #2" we need actual #2 numbers on this hardware.

### P0.1 — Bench SGLang local

Setup SGLang v0.5+ on RTX 4080 SUPER. Run same canonical workloads:

```bash
# High-conc
sglang launch --model Qwen/Qwen3-4B --port 8001 \
  --max-running-requests 64 --tp 1 --enable-fp8-kv-cache
guidellm benchmark run --target http://localhost:8001 \
  --profile concurrent --rate 64 \
  --data 'prompt_tokens=1024,output_tokens=256'
```

For all 4 canonical workloads:
- High-conc 1k/256/c=64
- Long-ctx 4k/c=4
- Long-ctx 8k/c=4
- Multi-tenant shared-prefix (custom Python runner)

### P0.2 — Bench TensorRT-LLM local (if feasible)

TRT-LLM requires model compilation step (engine .plan files).
Setup is harder; ROI of including TRT-LLM as #2 reference depends
on whether we have time + tooling.

Defer unless P0.1 shows SGLang is clearly #1 — then TRT-LLM
becomes the relevant comparison.

### P0.3 — Decision: who is #2?

Build the table:

| Shape | ARLE | vLLM | SGLang | TRT-LLM | #2 (max non-ARLE) | 30% target |
|---|---:|---:|---:|---:|---:|---:|
| high-conc | 843 | 647 | TBD | TBD | TBD | TBD |
| long-ctx 4k | 153.9 | 159.1 | TBD | TBD | TBD | TBD |
| long-ctx 8k | TBD | 105.6 | TBD | TBD | TBD | TBD |
| multi-tenant | 318ms | 573ms | TBD | TBD | TBD | TBD |

Per-shape "30% target" = `(max non-ARLE bench) × 1.30`.

## Phase 1 — Per-shape gap analysis (1-2 days)

For each shape where ARLE < 30% target:
1. Identify dominant gap layer (kernel / scheduler / pipeline / memory)
2. Estimate engineering cost to close gap
3. Prioritize by ROI (gap size / cost)

Likely gap candidates per current state:
- Long-ctx 4k/c=4 at -3% vs vLLM: chunk admission policy +
  per-prefill prep loop (codex investigating now). Low cost, may
  reach +5-10% vs vLLM. To reach +30% need kernel speedup too.
- High-conc already +30% vs vLLM. May be #1 or #2 depending on
  SGLang. If SGLang ahead, need to close their gap.
- Multi-tenant already +80% vs vLLM. Likely #1 across all
  competitors at this shape (no other engine has this specific
  cascade pattern).

## Phase 2 — Targeted optimization (2-4 weeks per shape)

After Phase 1 identifies gaps, deploy specific optimizations:

| Probable target | Likely optimization | Cost | Risk |
|---|---|---|---|
| Long-ctx prefill TTFT | M_b.2 FP8 prefill TileLang OR scheduler chunk policy | 1-2 weeks | Medium (kernel correctness) |
| High-conc per-row decode | Already at 0.99 ms/row, likely near hardware limit | n/a | Diminishing returns |
| Long-ctx multi-chunk pipeline | Mixed-path workspace + per-prefill prep unify (codex M3.9 follow-up) | 1-2 weeks | High (Mixed regression precedent) |
| New: spec-decode + F4 compound | Apply F4-Small async pattern to spec verify | 2-3 weeks | Medium (acceptance rate dependency) |
| New: M_b.3 unified prefill prep loop | Per-prefill-row prep → varlen single launch | 1 week | Low |

## Phase 3 — Validate world #1 claim (1 week)

Per-shape:
1. Re-bench all engines (ARLE, vLLM, SGLang, TRT-LLM) at the
   final ARLE state
2. Confirm ARLE leads the 2nd-place engine by > 30% at every
   targeted shape
3. Wins entry: "world #1 by 30%+ lead" with full data table

Stretch: get an external 3rd party to reproduce the bench.

## What's already done vs what's left

Done (this session 2026-05-07):
- ARLE +30% at high-conc 1k/256/c=64 vs vLLM ✓
- ARLE +80% at multi-tenant shared-prefix vs vLLM ✓
- F4-Small (decode async sync) — substrate
- M_b.1 Phase B (TileLang HD128 decode kernel) — substrate
- M_d.1 (tokenizer fingerprint correctness) — substrate
- M_pf P0 (peek_prefix_classify substrate)
- M_nsys P0 (cuProfilerStart/Stop signal handler) — diagnostic infra
- Phase 1A v3 fix (multi-slot ring substrate kept, default Split)
- M_ibp ABANDONED (already winning at this shape)

Needed for world #1:
- **Phase 0 baseline measurement (SGLang + TRT-LLM) — IMMEDIATE**
- Per-shape gap analysis post-baseline
- Long-ctx 4k/c=4: +30% from current -3% = +33% engineering need
- M_nsys P1 (graph capture timing) — enables proper trace at long-ctx
- Targeted kernel/scheduler work per gap

## Tasks

| # | Task | Owner | LOC est. | Trigger |
|---|---|---|---|---|
| P0.1 | SGLang v0.5+ install + bench at 4 canonical workloads | Claude | 0 (config + bench) | Now |
| P0.2 | TRT-LLM bench (deferred unless SGLang clearly #1) | Claude | 0 | After P0.1 |
| P0.3 | Build #2 baseline table + 30% target column | Claude | 0 | After P0.1+2 |
| P1 | Per-shape gap analysis | Claude + codex review | 0 | After P0 |
| P2 | Targeted optimizations | Codex (impl) + Claude (review/bench) | varies | After P1 |
| P3 | Final validation bench | Claude | 0 | After P2 |

## Acceptance

- ARLE leads 2nd-place engine by **≥ 30%** at every canonical
  workload:
  - High-conc 1k/256/c=64
  - Long-ctx 4k/c=4
  - Long-ctx 8k/c=4
  - Multi-tenant shared-prefix
- Wins entry with full per-shape data table cross-referenced
  to bench artifacts and sha256
- All correctness tests pass (e2e, greedy_consistency, M_d.1
  isolation)
- No tail latency regression > 20% vs current state at any shape

## Cross-references

- M-final integration roadmap: `d16effe`(updated)
- Current canonical baselines:
  - F4-Small high-conc: `8f83c80`
  - F4-Small longctx 8k: `c63c31c`
  - vLLM longctx 8k control: `9afcd57`
  - vLLM longctx 4k control: `bench-output/2026-05-07-vllm-longctx-4k-c4/`
  - Phase 1A v3 fix bench: `adb2757`
- Industry SOTA references (web-research'd):
  - vLLM official benchmarks: https://docs.vllm.ai/
  - SGLang releases: https://github.com/sgl-project/sglang/releases
  - TensorRT-LLM Qwen3 docs:
    https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/

## Rule (per memory `feedback_docs_priority_roi_evidence.md`)

- **"World #1" is a measurable target, not aspirational**. Phase 0
  baseline measurement is mandatory; without it, every optimization
  is unanchored.
- **Per-shape lead matters more than aggregate**. ARLE leading by
  20% in 3 shapes and -10% in 1 is NOT "world #1" — it's
  "competitive at this shape mix".
- **Defending the lead matters**. After hitting 30% lead, must
  protect against regressions on every commit. Bench gauntlet at
  every PR.
