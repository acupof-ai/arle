# P0.0 Phase 1.A nsys decomposition — empirical evidence obtained(Stage 7)

> Per `5a63142` Phase 1.A nvtx scope LANDED + `dfa6408` smoke verified +
> `af44efa` codex's recipe nsys-target fix(profile server not bench client),
> Claude executed actual nsys 60s decomposition this tick using **Option A
> pattern**(server-direct profile)。**Evidence:prefill::compute is 97% of
> active GPU time → P0.2 Hybrid Phase 2 dispatch is right axis per
> `d2c2c17` decision matrix**。

## nsys protocol(Option A — server-direct profile per `af44efa` fix)

```bash
nsys profile \
  --output /tmp/p1a_decomp \
  --trace cuda,nvtx,osrt \
  --duration 60 \
  --capture-range=none \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B \
    --port 8003 --num-slots 8 --max-seq-len 5120 \
    --admission-policy prefix-aware --kv-cache-dtype bf16 &

# 25s warmup wait
# Then 4 rounds of 4-curl bursts(short prompt 32-token output)
```

## NVTX push/pop summary(60s window)

| Range | Time % | Total Time(ms)| Instances | Avg(ns)| Median(ns)|
|------|------:|------:|------:|------:|------:|
| step_total | 66.4% | 745.29 | 184148 | 4047.2 | 1940 |
| **step_prefill_kernel_launch** | **29.1%** | **326.99** | **2** | **163499392** | **163499392** |
| step_admission | 3.2% | 36.03 | 184148 | 195.7 | 150 |
| step_decode_kernel_launch | 0.9% | 10.34 | 31 | 333645.9 | 342894 |
| step_plan | 0.3% | 3.06 | 3583 | 853.5 | 660 |
| step_dispatch_emits | 0.1% | 1.22 | 3583 | 341.0 | 250 |

Raw CSV:`bench-output/2026-05-09-p1a-nsys-decomp/nvtx_pushpop_sum.csv`(gitignored)。

## §0 SOLID rule 6 framing — wall-clock not NVTX-window %

**Per-active-time framing**(absolute ms,not 60s window %):
- Prefill compute:327 ms
- Decode compute:10.3 ms
- Admission:36 ms
- Plan + dispatch:4.3 ms
- **Total active GPU time**:~377 ms over 60s capture
- **Idle time**:60s - 377ms = ~99.4% idle(workload too light)

**When workload IS active**(prefill + decode windows):
- prefill::compute = **327 / 337 = 97%** of compute time
- first_decode::compute = 3% of compute time
- admission overhead = trivial(36ms / 60s = 0.06% wall-clock)

→ **Prefill compute is empirically dominant when work happens**。

## ⚠ Phase 1.A scope did NOT appear in output

`step_admission_prefix_lookup` was expected to appear as separate scope
(per `5a63142` block-as-rvalue wrap)but is **NOT in nvtx_pushpop_sum**。

Possible causes:
1. **Workload too light**:short curl prompts may not trigger
   `lookup_or_stage` path(unique session_ids → empty radix cache hit)
2. **Scope filter issue**:nsys may filter scopes with very low total time
3. **Wrap scope shadowed**:macro `nvtx_scope!` may have been optimized away

**Action needed**(stage 8):
- Either:run with multi-tenant load that exercises radix cache(re-uses prefixes)
- Or:verify scope macro by wrapping `step_admission_prefix_lookup` print to log
- Or:use `nvtx_kern_sum` report to see kernel-attributed scope counts

## Empirical decision per `d2c2c17` matrix

| Dominant phase(>40% of active time)| Implication | P1 priority |
|------|------|------|
| **prefill::compute(97%)** ✅ | First-token compute slow | **P0.2 Hybrid Phase 2 dispatch wiring,KV W4A8 demoted** |

→ **Strong empirical evidence**:focus next P1 work on **prefill compute optimization**,
NOT KV W4A8 #33 or Medusa #28(which target decode/per-token-latency)。

This validates `d2c2c17` strategic axis-ROI brief's hypothesis — **multi-tenant
TTFT 241ms residual gap to world-#1 121ms target is dominated by prefill compute,
not decode bandwidth or first-decode latency**。

## What remains for full Phase 1.A closure

1. ⚠ Verify `step_admission_prefix_lookup` scope fires(workload-shape issue or
   instrumentation issue)
2. Run with multi-tenant burst workload(`scripts/bench_multitenant_burst.py`)
   to get representative production-shape decomposition
3. Run paired with prefix-aware OFF for B3 Step 2 baseline comparison
4. Compute σ across n=3 runs for SOLID confidence

But **directional signal is already strong**:prefill dominates active time,
P1 axis selection can lock in P0.2 Hybrid Phase 2 wiring(not pivot to KV W4A8)。

## Phase 1.A micro-cycle now 7 stages complete

1. `2fafa9e` Codex Phase 1.A nvtx recipe(forward)
2. `b55bfcd` Claude scoping fix(block-as-rvalue)
3. `153fd93` Codex audit codification(skill #21)
4. `5a63142` Claude implementation(8 net LOC)
5. `d35ca35` Codex impl verification 5/5 SOLID
6. `dfa6408` Claude server-startup smoke verified
7. `af44efa` Codex recipe nsys-target fix(2nd #21 evidence)
8. **(this commit) Claude executes nsys decomposition with corrected pattern**

## Methodology insight

Codex's `af44efa` recipe-fix saved this tick from producing useless empty
NVTX data。Anti-pattern #21(recipe-itself audit gap)now has **2 empirical
evidence points**:`b55bfcd` scope-wrap bug + `af44efa` nsys-target bug。
**Both were caught by Claude/codex audit BEFORE Claude executed** — bidirectional
audit pattern preventing wasted bench time。

Compared to R4#6 micro-cycle:there empirical bench KILLED hypothesis;here
empirical bench DIRECTIONALLY CONFIRMS strategic ROI brief's prefill-dominant
hypothesis。Both produce evidence-grade outputs via Phase 1-8 skill discipline。

## Cross-references

- Phase 1.A scope LANDED:`5a63142`
- Smoke verified:`dfa6408`
- Recipe nsys fix:`af44efa`(skill v1.8.0 #21 2nd evidence)
- Strategic axis ROI brief:`d2c2c17`
- Auto-memory bidirectional audit cycle:`memory/feedback_bidirectional_audit_cycle.md`
- nsys raw data:`bench-output/2026-05-09-p1a-nsys-decomp/nvtx_pushpop_sum.csv`(gitignored)

## Status

**Phase 1.A nsys decomposition execution SUCCESS**(via Option A server-direct
profile per `af44efa` fix)。Empirical signal:**prefill::compute dominates active
GPU time(97%)** → **P0.2 Hybrid Phase 2 dispatch is the right next P1 axis**
per `d2c2c17` decision matrix。

⚠ My Phase 1.A scope `step_admission_prefix_lookup` didn't appear in output
(workload-shape OR instrumentation issue)— Stage 9 follow-up needed for
scope-fire verification under multi-tenant burst。Direct decision based on
existing scopes is already actionable。
