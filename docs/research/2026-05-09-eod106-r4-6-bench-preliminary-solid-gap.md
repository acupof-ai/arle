# R4 #6 W4A16BatchGemv override preliminary bench — SOLID gap discovered + refined hypothesis

> Per `02209f4` R4#6 implementation LANDED + `a377e57` wins stub。Claude
> ran first actual bench this session(60s c=4 4096-in/256-out)。**Result:
> ITL +37% regression vs Round 1 baseline,4/8 incomplete**。Bench shape
> inadequate for license decision but reveals **critical SOLID gap**:
> override fires in WRONG context(prefill not decode)。

## Preliminary bench data(inadequate-shape caveat)

Server config:
```bash
INFER_R4_W4A16_GEMV_OVERRIDE=1 ./target/release/infer \
  --model-path infer/models/Qwen3-4B-GPTQ-Int4-marlin \
  --port 8003 --num-slots 8 --max-seq-len 5120 --kv-cache-dtype bf16
```

Bench(`bench-output/2026-05-09-r4-w4a16-gemv-override-c4-4k/`):
| Metric | R4#6 override | Round 1 Marlin baseline | Δ |
|---|---:|---:|---:|
| ITL p50 | **24.85 ms** | 18.13 ms | **+37%(WORSE)** |
| TTFT p50 | **47.45 sec** | 1.976 sec | **+24×(severely WORSE)** |
| TPOT p50 | 210.1 ms | 18-25 ms est | +8-12× WORSE |
| Out tok/s p50 | 82.0 | 150 | -45% |
| Successful | 4/8(50%) | 100% | severe regression |
| Duration actual | 50s(60s window cut by warmup) | 120s full | inadequate |

**License decision deferred** — bench shape inadequate(60s vs needed 120s,4 incomplete vs needed 0)。

## §0 SOLID gap — hypothesis-context vs implementation-context mismatch

**Original Round 4 #6 hypothesis**(per `marlin-w4a16-bench-implementation-gap.md` §"Round 4 prep"):
> "**Decode (M ≤ 8) launch overhead** | W4A16BatchGemv expected to win |
> Compute trivial at M=4; per-launch fixed cost dominates"

→ Hypothesis target:**DECODE with M ≤ 8**(M = batch in decode = number of concurrent sequences,each emitting 1 token per step)。

**My implementation**(`02209f4`):
```rust
if batch > 1
    && marlin_prefill_aligned(weight).is_ok()
    && std::env::var("INFER_R4_W4A16_GEMV_OVERRIDE").as_deref().ok() != Some("1")
{
    return Self::MarlinW4Gemm;
}
```

→ Override fires when `batch > 1`。**But `batch` here = `x.seq_len`**(per
`linear.rs:1196` `LinearKernelPlan::batched(weight, x.seq_len)`)。

**For prefill**:`x.seq_len` = prompt_length(e.g. 4096 tokens)。`batch > 1`
is TRUE,override fires,W4A16BatchGemv selected for **prefill GEMM with
M=4096**(not M≤8)。

**For decode**:`x.seq_len` = 1 typically(autoregressive emission)。`batch > 1`
is FALSE,override does NOT fire,decode goes to `decode()` branch with
`W4A16Gemv`(GEMV not BatchGemv)。

→ **Override fires in EXACTLY the WRONG context**:prefill where Marlin
matrix-matrix tensor-core utilization wins,not decode where launch-overhead
dominates。

## Refined hypothesis(R4 #6.B candidate)

**Correct override should**:fire only when batch GEMM has M ≤ 8 in actual
shape sense(decode batched across multiple sequences),not M = seq_len
prefill。

**Implementation refinement**:identify "decode batched" path explicitly
(probably via different dispatch entry point,not via `batched()` with
batch=seq_len)。This requires deeper grep into how `batch` parameter
flows from caller。

**Or simplification**:add explicit batch-shape gate to `batched()`:
```rust
if batch > 1
    && batch <= 8                       // ← R4#6.B refinement: only DECODE batch
    && marlin_prefill_aligned(weight).is_ok()
    && std::env::var("INFER_R4_W4A16_GEMV_OVERRIDE").as_deref().ok() != Some("1")
{
    return Self::MarlinW4Gemm;
}
```

→ **Wait — this is wrong direction**。Current code returns
`Self::MarlinW4Gemm` when this condition is TRUE,so adding `batch <= 8`
narrows when MARLIN is selected(makes Marlin only used for batch=1
which is decode contradicted)。Reading current code more carefully:

Original code returns `MarlinW4Gemm` for batch > 1。My override adds
"unless env var set"。If override selected W4A16BatchGemv for ALL batch>1
(including prefill 4096),that explains regression。

To target ONLY decode batch=2-8:override should fire ONLY when `batch ∈ 2..=8`,
NOT `batch > 1` open-ended。**Refinement**:

```rust
if batch > 1
    && marlin_prefill_aligned(weight).is_ok()
    && (
        // For prefill (batch=seq_len > 8): always use Marlin
        batch > 8
        // OR for decode batch ∈ 2..=8 with override OFF: use Marlin
        || std::env::var("INFER_R4_W4A16_GEMV_OVERRIDE").as_deref().ok() != Some("1")
    )
{
    return Self::MarlinW4Gemm;
}
```

→ Override only flips for `batch ∈ 2..=8` (decode-batched),preserves Marlin
for prefill(batch > 8)+ batch=1 already in decode branch。

## Anti-pattern #25 candidate(skill v1.8.0)

**"Hypothesis-context vs implementation-context mismatch"**:When porting
a hypothesis(scoped to specific shape/range)into implementation,verify
the implementation **fires only in that shape/range**。Implementation
condition `batch > 1` is broader than hypothesis context "decode M ≤ 8";
override leaks into prefill where the cost-tradeoff inverts。

**Evidence**:`02209f4` override + this preliminary bench → +37% ITL
regression because override fired for prefill GEMM(M=4096)not just
decode batched GEMV(M=2..=8)。

**Cure**:add explicit shape-range gate matching the hypothesis context。
Don't use overly-broad condition that includes contexts the hypothesis
doesn't cover。

**Compound with**:
- #20 hypothesis-inheritance(`c076aae`)
- #21 recipe-itself audit(`b55bfcd`)
- #22 twin-commit attribution(`3fea979`)
- #23 truncated-output partial-view(`156d2c2`)
- #24 cell-collapse blindness(`1ccb448`)
- **#25 hypothesis-context vs implementation-context mismatch(this brief)**

→ Skill v1.8.0 batch now has **6 candidates** with coherent theme:
"audit-at-every-prescription-layer + verify-context-match"。

## Recommended actions

### Path A — refine + re-bench(preferred)

1. Edit override at `linear.rs:71-83` to add `batch ∈ 2..=8` gate
2. Rebuild release(~1-2 min incremental)
3. Re-bench with full 120s + same protocol
4. License-or-kill on refined override

**Effort**:~10 min Claude

### Path B — full revert(safer if uncertain)

1. `git revert 02209f4`(or restore line)
2. Mark R4#6 hypothesis as **REQUIRES SCOPING** before next attempt
3. Update wins stub with KILL via implementation-context-mismatch finding

**Effort**:~5 min Claude

### Path C — codex-pickup-verify

Hand to codex:"my preliminary bench showed +37% regression but bench
shape was inadequate (60s/4-incomplete);also override may be too broad
(fires for prefill not just decode);refine + re-bench OR revert"。

**Recommended**:Path A — refinement is small + we have build infrastructure
warm + bench script ready。

## Implementation status

`02209f4` env-gated override is LANDED on origin/main but **fires too broadly**。
Default OFF preserves Marlin path so production unaffected,但 the override
itself doesn't deliver predicted win when ON。

**Pickup queue update needed**:R4#6 status from "READY FOR PICKUP"(per
`6ade2d4`)→ "**REFINEMENT NEEDED — implementation-context mismatch
discovered**"(per this brief)。

## Cross-references

- Original Round 4 #6 hypothesis:`docs/experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md`§"Round 4 prep"
- Phase 0 audit:`6ade2d4`(audited but missed context-mismatch)
- Codex audit-of-audit:`5bb99d7`(verified 6/6 SOLID claims but didn't catch context-mismatch either!— reveals shared blindspot)
- Implementation:`02209f4`
- Wins stub(pending-remote → now PRELIMINARY-DATA-AVAILABLE):`a377e57`
- Bench data:`bench-output/2026-05-09-r4-w4a16-gemv-override-c4-4k/`
- Skill v1.7.0 #18 + #19 applied successfully but missed shape-context dimension
- Auto-memory natural-closure heuristic:`memory/feedback_bidirectional_audit_cycle.md`

## §0 first principle observation

**Both my Phase 0 audit AND codex's audit-of-audit missed the context-mismatch**。
This shows even bidirectional audit has limits:**both auditors examining the
SAME claim space miss the same dimensions if their shared mental model is
incomplete**。

Layer-9 SOLID gap on R4#6 axis(parallel to c20b1ce 8-layer chain on bimodal
axis):**bidirectional audit is necessary but not sufficient**。Empirical
bench is the truly orthogonal SOLID layer that catches what code-level audits
both miss。

Tonight's lesson:**run benches earlier in the audit chain when feasible**,not
defer to "next tick"。Empirical data eliminates entire classes of audit gaps
that pure source review can't reach。

## Status

**PRELIMINARY KILL/REFINE SIGNAL** — R4#6 implementation-context-mismatch
discovered via 60s bench(inadequate-shape but directionally clear)。Next
action:Path A refinement(~10 min)or Path B revert(~5 min)。Decision deferred
to next tick or codex pickup。

§0 first principle escalates to **9-level chain** for R4#6 axis:
implementation → recipe → strategic → LICENSE meta → fix-effect-target-env →
twin-commit-attribution → downstream-citation-pollution → multi-variable-confound →
**hypothesis-context vs implementation-context mismatch**。
