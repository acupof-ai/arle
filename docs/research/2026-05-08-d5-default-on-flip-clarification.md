# D5 W4A8 default-on flip — clarified as docs/policy decision,not code change

> Code-grep finds W4A8 dispatch is **already automatic** when checkpoint
> has `quant_type=marlin_w4a8` config(per `5dc27a2` master strategy +
> `linear.rs:1186-1203` `LinearKernelPlan::batched` plan dispatch)。
> D5 was originally framed as "code flip" pending graph capture wiring。
> Reality:**no code change needed,policy/recommendation decision only**。

## Code path verification

`infer/src/ops/linear.rs:1186-1203`:
```rust
let plan = LinearKernelPlan::batched(weight, x.seq_len);
match plan {
    LinearKernelPlan::MarlinW4Gemm => run_marlin_w4_gemm(ctx, weight, x, out)?,
    LinearKernelPlan::MarlinW4A8Gemm => {
        run_marlin_w4a8_linear(ctx, weight, &x.data, x.seq_len, &mut out.data)?;
    }
    LinearKernelPlan::TurboQuantGemv | LinearKernelPlan::TurboQuantDequantCublasGemm => {...}
    LinearKernelPlan::Bf16CublasGemm if deterministic_gemm_enabled() => {...}
    LinearKernelPlan::Bf16GraphsafeGemm | LinearKernelPlan::Bf16CublasGemm => {...}
    _ => run_qweight_linear(ctx, weight, x, out, plan),
}
```

`LinearKernelPlan::batched(weight, x.seq_len)` selects `MarlinW4A8Gemm`
when:
- `weight.marlin_w4a8 == true` (loader detected `quant_type=marlin_w4a8`)
- `seq_len` qualifies(varies by plan logic)

So when user serves with `--model-path Qwen3-4B-GPTQ-W4A8-marlin`,
W4A8 path engages automatically。No env var or config flag needed。

## Original D5 framing(per `b04b5fb` and `c6bfa05`)

D5 was framed as:
> Default-on flip W4A8 — when?
> Per `62e75ee` plan,W4A8 default-on flip blocked on graph capture wiring。

This was based on the assumption that graph capture was needed for
W4A8 to be production-acceptable。But:
1. Graph capture(`M_pf-graph-prefill-capture`)plan was **KILLED** per
   master strategy(per multiple briefs in KILL log accumulation)
2. W4A8 prefill TTFT -36% LICENSED **without graph capture**(`b5889b3`)
3. W4A8 decode ITL +63% slower vs W4A16(`b5889b3`)— this is
   independent of graph capture,fundamental to INT8 vs FP16 activation
   path

So graph capture was never the binding constraint for D5。

## Updated D5 framing

D5 is actually a **deployment policy decision**:given W4A8 + W4A16
both work,which checkpoint does ARLE recommend as default for new
deployments?

### Decision branches

**A. Recommend W4A16 default**:
- Pros:better decode ITL(11.73 vs 19.18 ms,1.64×)at small batch
- Cons:slower prefill TTFT(2388 vs 1632 ms,+46%)
- Best for:short-output workloads(decode dominates total time)

**B. Recommend W4A8 default**:
- Pros:faster prefill TTFT(-36%)+ smaller weight memory
- Cons:slower decode ITL at small batch(+63%)
- Best for:long-prompt + short-output workloads

**C. Recommend hybrid(W4A8 prefill / W4A16 decode)**:
- Pros:best-of-both,-14% E2E latency vs W4A16-only
- Cons:2× weight memory(5.32GB Qwen3-4B,fits 16GB at c=4-8)
- Requires Phase 1b loader work(`6be30ce` directive,~0.5d)
- Recommendation per master §1.2.1.A,but not yet implemented

### Recommendation:**C. Hybrid pending Phase 1b**

Master strategy §1.2.1.A weight axis 全套 commitment:both W4A16 + W4A8
production paths。Hybrid is the optimal routing。

**Path forward**:
1. Codex picks up Phase 1b loader patch(`6be30ce`)— ~0.5d
2. Phase 2 Linear dispatch by `StepPlan::variant`(prefill→W4A8 / decode→W4A16)
3. Phase 3 E2E test
4. Phase 4 bench(target -14% E2E)
5. Update `docs/support-matrix.md` recommending hybrid checkpoint format
6. **D5 closes**:default deployment uses hybrid checkpoint

**Until Phase 1b lands**:document W4A16 as "default for general decode-
heavy workloads" + W4A8 as "opt-in for prefill-heavy"。

## D5 status update

| Context | Original framing | Reality | Action |
|---------|------------------|---------|--------|
| Code change needed | Yes,gated on graph capture | No,already automatic | None |
| Graph capture prerequisite | Yes | KILLED per master §7.7 | n/a |
| Production deployment | Pending flip | Auto-routes per quant_type | Doc update |
| Recommended format | TBD | Hybrid pending Phase 1b | C above |

## Cross-references

- D5 origin: `b04b5fb` synthesis EOD+43
- D5 ready-to-execute: `c6bfa05` (was misframed)
- W4A8 prefill LICENSED: `b5889b3`
- W4A16 LICENSED: `bc15eca`
- Hybrid plan: `9754aca`(refined `1959a21` Phase 0)
- Phase 1b directive: `6be30ce`
- Linear dispatch: `infer/src/ops/linear.rs:1186-1203`
- Loader detection: `infer/src/weight_loader.rs:514` `marlin_w4a8` quant_type
- Master strategy weight axis: `5dc27a2`(§1.2.1.A bench update)

## Methodology rule

When a "default-on flip" decision shows up after multiple bench LICENSEs,
**verify whether code change is actually needed**。Modern dispatch
patterns(plan selection by checkpoint metadata)often make "flip"
decisions automatic — the actual decision is documentation/recommendation
policy。

Don't queue code work for decisions that are really just "what to write
in support-matrix.md"。

## Status

**D5 closed** at code-decision level:no flip needed,dispatch auto-routes
per checkpoint metadata。Documentation update follows from hybrid Phase
1b landing。

Until then:
- W4A16 GPTQ-zpfix is ready for production short-output workloads
- W4A8 GPTQ-zpfix is ready for prefill-heavy workloads(opt-in)
- Both are LICENSED individually per `bc15eca` + `b5889b3`
- Hybrid is the future default,blocked only by Phase 1b loader work
