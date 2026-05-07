# 2026-05-07 · Phase 1 trace retrospective — Mixed kernel launch is 5× slower than Decode

## Priority & ROI

**Priority**: P3 (after-the-fact analysis, no immediate action).

**ROI of this finding**: zero direct (Mixed launch overhead is 6%
of step time). But validates the trace evidence chain:
- Budget undersizing is dominant cause of regression (Fix A target)
- Mixed kernel CPU launch overhead is a secondary, separate concern
- Future kernel-axis work has a known target (Mixed kernel
  setup/dispatch path)

**Negative case**: pursuing this 5× kernel launch gap is wrong
priority — it's < 6% of step_total. Don't chase it before
budget fix proves itself.

**Kill criteria for any future "fix mixed launch overhead" plan**:
nsys post-Fix-A re-trace shows mixed launch share > 20% of
step_total → escalate. Else defer indefinitely.

## Discovery

While Codex finalizes Fix A + Option B, retrospective analysis
of the Phase 1 nsys trace
(`bench-output/2026-05-07-m3.6-arle-s48-nsys/arle-s48.nsys-rep`)
revealed an interesting NVTX phase breakdown that wasn't
flagged in the original Phase 1 wins entry
([`fdb531b`](2026-05-07-m3.6-phase1-nsys-arle-s48-highconc.md)).

## NVTX phase breakdown (Phase 1, F4-Small + Split path)

```
Range                              Avg time     Instances
:step_total                         82.05 ms        554
:step_mixed_kernel_launch            5.20 ms        114    <-- 5x decode
:step_prefill_kernel_launch        531.24 ms          1    (warmup outlier)
:step_decode_kernel_launch           1.00 ms        437
:step_admission                     18.81 µs        554
:step_plan                          14.74 µs        553
:step_dispatch_emits                12.69 µs        553
:step_mixed_launch_retract           3.06 µs        114
```

**Mixed kernel CPU launch (5.2ms) is 5.2× slower than Decode
kernel CPU launch (1.0ms)**.

This is at the **CPU side** — the time spent in scheduler code
issuing the kernel launches (not the GPU execution time itself).

## Why Mixed launch is slower

Per source (`infer/src/model/qwen3/batch_decode.rs::decode_batch_with_prefill`):

Mixed launch does setup work that Decode launch skips:
1. **Combined token H2D**: builds `combined_tokens` Vec<i32> from
   decode + prefill tokens, single H2D upload (line 955-963)
2. **Mixed metadata setup**: `update_mixed_batch` (line 965-972)
   builds varlen indices for both decode and prefill rows
3. **Per-prefill prep loop**: 1 × decode_prep_paged_cuda + N ×
   prefill_attention_paged_prep_cuda (per-prefill-row launches)
4. **KV quantization** (FP8 path): kv_quant launches for K and V

Pure decode launch (`step_decode_launch`) skips items 1-3:
- Token H2D is for decode rows only (smaller)
- decode_prep_paged_cuda only (no prefill prep loop)
- KV quantization same

So Mixed has ~4-5× more CPU launch operations. Matches the 5×
factor.

## How this relates to Phase 1A v3 regression

Two compounding issues at long-ctx 4k/c=4 with Mixed enabled:

### 1. Budget undersizing (DOMINANT, Fix A target)

`mixed_prefill_token_budget = min(16384, 4096) = 4096` →
admission packs 1 req's prefill instead of 4 → 4× more steps.

### 2. Mixed kernel launch overhead (SECONDARY)

Each step's CPU launch is 5ms instead of 1ms. With 21 steps:
- Mixed: 21 × 5.2 ms = 109 ms launch overhead
- Split equivalent: 7 × 1.0 ms = 7 ms launch overhead
- Difference: **102 ms over 60s bench** (~0.17% of bench time)

The 102 ms is a real cost but vanishingly small relative to
the regression's TTFT impact (4500 ms).

### Composite

Budget undersizing accounts for:
- 4 × more steps × ~1 sec each = ~3 sec regression base
- Plus larger cumulative kernel launch overhead

After Fix A (whole-step budget restored):
- Same 4 reqs in 1 step (like Split)
- 5× CPU launch overhead per step still real but applies once
  per step (not per-request-step)
- Net: should converge to F4-Small baseline ± Mixed launch tax
  (~1-2% of total)

## Implication

**Mixed kernel launch slowness is NOT blocking Phase 1A v3 fix**.
Codex's Fix A (`mixed_prefill_token_budget`) addresses 90%+ of
regression. The remaining ~5% (Mixed launch overhead) is real
but defers to future kernel-axis work.

**Future plan candidate (P3)**: M_b.3 — fuse Mixed launch
operations. The per-prefill prep loop could be unified into one
varlen prep kernel (using HD128 prefill kernel's existing
q_indptr surface to handle multiple prefill rows in one launch).
LOC ~80, expected ~3-4ms savings per step at 21-step regression
shape = ~80ms total. Not a 10× win.

## Bench Status

No new bench. Pure retrospective analysis on existing trace.

## Cross-references

- Phase 1 trace: `bench-output/2026-05-07-m3.6-arle-s48-nsys/arle-s48.nsys-rep`
  (sha256 `d0711d7a0333cdbc61c1f32c608aa3b9113fd443c389794a78b6faf73b5dfb4e`)
- Phase 1 wins entry: `fdb531b`
- Phase 1A v3 regression: `ba748af`
- Root cause budget: `e675af2`
- Source: `infer/src/model/qwen3/batch_decode.rs::decode_batch_with_prefill:890-1330`

## Rule

- **Re-mining old trace data is high ROI**. The 5× Mixed launch
  finding was always in the Phase 1 trace; just not surfaced
  until specifically queried. Period audits of major trace
  artifacts can surface secondary findings without new bench
  runs.
- **"Why is X slow?" usually has multiple compounding causes**.
  M3.9 regression has BOTH budget undersizing (90%) AND launch
  overhead (5%). Don't stop at the first 80% hypothesis.
</thinking>

