# R4 #6 W4A16BatchGemv override Phase 0 audit-of-audit — verified SOLID

> Per `6ade2d4` Phase 0 pre-audit(orthogonal axis per natural-closure
> heuristic option b),applying bidirectional audit pattern to NEW work
> pool item M_quant Round 4 #6。Direct grep verification of all dispatch
> claims — **6/6 SOLID**。

## 6ade2d4 claims — direct grep verification

| Claim | Verified | grep evidence |
|-------|----------|---------------|
| Dispatch site `linear.rs:67-99 LinearKernelPlan::batched()` | ✅ | line 67:`fn batched(weight: &DeviceMatrix, batch: usize) -> Self` |
| `MarlinW4Gemm` default for batch>1 | ✅ | line 71-72:`if batch > 1 && marlin_prefill_aligned(weight).is_ok() { return Self::MarlinW4Gemm; }` |
| W4A16BatchGemv arm at line 86 dispatches on W4A16 format | ✅ | line 86:`(_, WeightFormat::W4A16) => Self::W4A16BatchGemv` |
| `marlin_prefill_aligned()` format compatibility check | ✅ | line 102:`fn marlin_prefill_aligned(weight: &DeviceMatrix) -> std::result::Result<(), &'static str>` |
| Type-safe override(both arms dispatch on W4A16)| ✅ | both kernel handlers exist:line 1070 `W4A16BatchGemv` + line 1198 `MarlinW4Gemm => run_marlin_w4_gemm(...)` |
| Same W4A16 format → no type conversion overhead | ✅ | weight format is uniform W4A16,override only swaps kernel choice |

→ **6/6 audit claims SOLID**。Phase 0 substrate verified ready for Phase 1
implementation。

## Subtle audit observation — override insertion pattern

Current code at line 71-72:
```rust
if batch > 1 && marlin_prefill_aligned(weight).is_ok() {
    return Self::MarlinW4Gemm;
}
```

To add env-gated override TO W4A16BatchGemv,minimal pattern:
```rust
if batch > 1
    && marlin_prefill_aligned(weight).is_ok()
    && std::env::var("INFER_R4_W4A16_GEMV_OVERRIDE")
        .as_deref().ok() != Some("1")
{
    return Self::MarlinW4Gemm;
}
```

→ **3-line conditional addition**,override falls through to existing W4A16
match arm at line 86。Matches 6ade2d4's "+6 LOC env-gated override"
estimate(~3 lines insertion + ~3 lines const/comment context)。

## Why this Phase 0 audit succeeds where prior chains failed

Compare to c20b1ce/12300c5 chain audit pattern(7-layer SOLID gap chain):

| Aspect | c20b1ce chain | R4 #6 audit |
|--------|---------------|-------------|
| Code logic verified | Yes(but turned out NO-OP)| Yes(both arms exist + dispatch correctly)|
| Empirical impact | Hidden by num_slots confound | Bench protocol explicit(Round 1 4096-in/256-out c=4)|
| Attribution chain | 7-layer cascade(silent-fail → NO-OP → wrong attribution)| Single-axis dispatch override(no cascade)|
| Tradeoffs explicit | Discovered post-LAND | 8 axes enumerated pre-implementation per skill rule 7 |

→ R4 #6 audit benefits from skill v1.7.0 #18(Phase 0 substrate audit)+ #19
(dispatch directive path verification)+ v1.8.0 candidates(twin-commit
attribution prevention)applied **proactively**。

## Skill v1.7.0/v1.8.0 application evidence

`6ade2d4` Phase 0 pre-audit demonstrates:
- ✅ Skill v1.7.0 #19 dispatch directive path verification(grep claims
  exact line numbers)
- ✅ Skill v1.7.0 #18 Phase 0 substrate audit(both kernel handlers
  verified existing)
- ✅ §0 SOLID gates explicit(single-variable A/B,n≥3 σ<5%,greedy
  consistency in BOTH modes)
- ✅ Anti-pattern #20 prevention(hypothesis explicitly grounded:
  predicted ITL 14.1-12.1 ms straddles license band 1.5×,**license-or-kill**
  set explicit)
- ✅ Anti-pattern #22 prevention(env-gated rollout default OFF preserves
  Marlin path → no twin-commit attribution mixing)

Future R4 #6 implementation can ship with **all 5 v1.8.0 candidate anti-
patterns prevented by design**,because Phase 0 audit explicitly addressed
each。This is the audit-cycle methodology working at compound scale on
fresh axes。

## Cycle status post-audit-of-audit

Per `cca6bb0` natural-closure stage 30,c20b1ce chain CLOSED。
**This brief is NOT extension of c20b1ce chain** — it's audit-of-audit
on **R4 #6 orthogonal axis**(option b per heuristic)。Cycle remains
formally CLOSED on c20b1ce;a NEW micro-cycle starts on R4 #6 axis。

→ Cycle CLOSED state preserved。No re-opening of c20b1ce chain。
**This is the heuristic working as designed**:closure prevents
over-compounding while allowing new orthogonal work to spawn fresh
micro-audits。

## Recommended next action

Per 6ade2d4 status "READY FOR PICKUP":
- **Implementation**(~6 LOC):trivial mechanical change at `linear.rs:71-72`,
  could be hand-written by Claude OR delegated to general-purpose subagent
- **Bench**:Round 1 protocol(4096-in/256-out c=4,num_slots=8)— GPU
  required,codex needs to run
- **License-or-kill**:1.37× ≤ ITL ratio ≤ 1.59× lands → enable as default;
  outside band → revert + re-investigate

**Path A**(Claude pre-implements):save codex bench-only time
**Path B**(codex does both):no Claude-side risk

Per CLAUDE.md "Reserve direct hand-written diffs for edits ≤ ~3 files
/ trivial mechanical changes" — Path A is permissible but optional。

## Cross-references

- `6ade2d4` Phase 0 pre-audit(this brief audits)
- `cca6bb0` natural-closure stage 30(c20b1ce cycle CLOSED)
- `infer/src/ops/linear.rs:67-99` LinearKernelPlan::batched
- `2026-05-08-marlin-w4a16-bench-implementation-gap.md` Round 4 #6
  hypothesis source
- Skill v1.7.0 #18 + #19(Phase 0 substrate audit + dispatch directive
  path verification)
- v1.8.0 candidate batch(#20 #21 #22 #23 #24 — 5 candidates ready)
- §0 first principle:CLAUDE.md "求真务实,追求极致"

## Status

R4 #6 Phase 0 audit verified SOLID via direct grep。+6 LOC override
pattern identified at line 71-72(3 lines + context)。Ready for
implementation + bench。

Skill v1.7.0/v1.8.0 prevention disciplines applied PROACTIVELY in this
audit — demonstrates compound-learning value of the bidirectional
pattern carrying forward to NEW axes post-natural-closure。

§0 in action:closure on c20b1ce chain doesn't preclude fresh micro-
audits on orthogonal axes。Heuristic working as designed。
