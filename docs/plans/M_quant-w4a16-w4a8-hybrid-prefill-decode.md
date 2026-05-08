# M_quant — W4A16 / W4A8 hybrid dispatch by prefill-vs-decode phase

**Status:** P1 plan — proposal opened by `b5889b3` empirical finding
**Owner:** TBD(codex impl,Claude planning)
**Effort:** estimate 1-2 days codex(~150-300 LOC)+ 0.5d bench
**Trigger:** `b5889b3` W4A8 LICENSED prefill / DEFERRED decode + W4A16 LICENSED decode finding
**Master strategy:** §1.2.1.A weight axis 全套 — production routing optimal

## §0 一句话

W4A8 wins prefill TTFT(-36%);W4A16 wins decode ITL(1.6×)。Static
phase-routing(prefill→W4A8 / decode→W4A16)gives **best-of-both
without runtime decision overhead**,**unlike** prior R4 #6 hybrid which
was dynamic batch-size dispatch within the same kernel(KILL HARD)。

## §1 Why now

`b5889b3` Bench post-`2a3a6f0` qzeros fix(2026-05-08):

| Workload | W4A16 GPTQ Marlin | W4A8 GPTQ-Marlin re-pack | Winner |
|----------|---:|---:|---|
| Decode ITL p50(c=4)| **11.73 ms** | 19.18 ms | **W4A16** by 1.64× |
| Prefill TTFT p50(4k batch)| 2388 ms | **1632 ms** | **W4A8** by -36% |
| greedy_consistency | PASS | PASS 32/32 | tie |
| out tok/s | 191.63 | 155.57 | W4A16 |

Static phase routing predicted gain:
- Prefill TTFT:adopt W4A8 → -36% TTFT(matches `b5889b3`)
- Decode ITL:adopt W4A16 → 1.64× ITL(matches `bc15eca`)
- Combined E2E:**W4A8 prefill + W4A16 decode beats either alone**

## §2 Difference from R4 #6 KILL precedent

R4 #6 hybrid dispatch was **dynamic batch-size routing within same
W4A16 kernel variant**:
- Inside Marlin GEMM,switch between variant A(small batch tile)and
  variant B(large batch tile)based on runtime `m` size
- KILL HARD per `4571082`:added complexity without throughput benefit
- Reason for KILL HARD:turned out qzeros bug masked any potential
  variant-A vs variant-B benefit;both arms produced corrupted output
  at different speeds

**THIS hybrid is fundamentally different**:
- Two **separate kernels**(W4A16 Marlin vs W4A8 Marlin)
- Routing happens at **phase boundary**(prefill / decode)not within
  kernel
- Phase is **statically determined** by scheduler step(prefill_rows >
  0 → use W4A8 path,else decode → use W4A16 path)
- No runtime dispatch overhead — kernel selection is per scheduler
  decision tick

So R4 #6 KILL precedent does NOT apply。This is "two specialized
kernels routed by phase" pattern,common in production systems
(SGLang/vLLM both do similar).

## §3 Implementation phases

### Phase 0 — Reconnaissance(0.25d Claude)
- [ ] Verify ARLE scheduler has clean phase boundary detection(it does
      per `infer/src/scheduler/cuda/execution.rs:plan_step` line 385
      `StepPlan::Decode` vs `StepPlan::Mixed`)
- [ ] Identify Linear weight loading path:can a single Linear hold
      both W4A16 and W4A8 packed bytes simultaneously?
  - Likely YES if loader stores both `marlin_qweight` + `marlin_w4a8_qweight`
    side-tensors(extra ~1× weight memory but saves runtime conversion)
- [ ] Memory cost analysis:Qwen3-4B model has ~252 Linear layers,
      each ~2.66GB packed → 5.32GB total,still fits in 16GB GPU

### Phase 1 — Loader storage augmentation(0.5d codex)
- [ ] `weight_loader.rs` reads BOTH `marlin_qweight` + `marlin_scales`
      AND `marlin_w4a8_qweight` + `marlin_w4a8_s_channel` + `marlin_w4a8_s_group`
- [ ] `DeviceMatrix` struct gains `w4a8_qweight: Option<...>` etc
- [ ] Loader prefers files containing BOTH(ideally a single hybrid
      checkpoint)or auto-converts on startup if only one available

### Phase 2 — Linear dispatch(0.5d codex)
- [ ] `infer/src/ops/linear.rs:run_marlin_w4a8_linear` already exists;
      add `run_marlin_w4a16_linear` companion(if not already present)
- [ ] Single `run_linear` entry that takes `phase: PhaseHint` enum
      `{Prefill, Decode}` and dispatches:
  ```rust
  match phase {
      PhaseHint::Prefill => run_marlin_w4a8_linear(...),
      PhaseHint::Decode  => run_marlin_w4a16_linear(...),
  }
  ```
- [ ] Caller(scheduler step plan)passes hint based on `StepPlan` variant

### Phase 3 — End-to-end test(0.25d codex)
- [ ] `cargo test --release -p infer --features cuda --test greedy_consistency`
      with hybrid weight checkpoint should PASS(both arms greedy-PASS,
      so combined should pass too)
- [ ] Add new test `test_hybrid_w4_dispatch_phase_routing` that asserts
      prefill rows used W4A8 path,decode steps used W4A16 path

### Phase 4 — Production bench(0.25d)
- [ ] `./scripts/bench_guidellm.sh m_quant-hybrid-w4a8-w4a16` at 4k
      longctx c=4 same workload as `bc15eca`/`b5889b3`
- [ ] Expected:
  - TTFT:matches W4A8 standalone(prefill path) ≈ 1632 ms
  - ITL:matches W4A16 standalone(decode path) ≈ 11.73 ms
  - tok/s:should beat both arms standalone(combined gains)
- [ ] Ship `wins/2026-05-08-w4-hybrid-prefill-decode.md`

## §4 Risk + KILL criteria

### Risks
1. **Weight memory overhead 2×**:If both packings stored side-by-side
   in GPU,memory usage doubles from 2.66GB → 5.32GB
   - Mitigation:still fits in 16GB on 4070Ti SUPER for Qwen3-4B
   - For larger models(Qwen3.6 35B-A3B):could exceed budget
   - **Decision point**:if Qwen3.6 35B doesn't fit,fall back to
     dynamic conversion at phase boundary(slow,~50ms overhead per phase
     switch)
2. **Activation re-quantization at phase boundary**:
   - W4A8 needs INT8 activation,W4A16 needs FP16/BF16 activation
   - Switching kernel mid-decode would require re-quantizing the residual
   - Mitigation:phase routing happens at scheduler granularity(not
     mid-kernel),so each step has consistent activation type
3. **Per-step overhead from dispatch logic**:
   - Phase classification costs ~10-100 ns per step(negligible vs
     1-20 ms kernel)

### KILL criteria
- **Phase 0**:if Qwen3-4B 2× weight storage doesn't fit local GPU →
  fall back to dynamic conversion path(then re-evaluate)
- **Phase 3**:if greedy gate fails on hybrid(should not — both arms
  pass independently)→ dispatch logic bug,debug
- **Phase 4**:if hybrid bench shows < 5% combined improvement vs best
  single arm → KILL,not worth weight memory cost

### Phase 0 reconnaissance results(`1959a21`)
- ✅ Scheduler boundary CLEAN:`StepPlan` enum at `execution.rs:411`
  discriminates Decode/Prefill/Mixed/Split → hybrid dispatch = trivial
  match
- ⚠ DeviceMatrix needs TWO matrices per Linear(`HybridLinear` ~50 LOC
  vs new WeightFormat variant ~150 LOC — Option A recommended)
- ⚠ Memory cost:**45%**(7.15GB / 16GB)
  - c=4:acceptable(76% with KV)
  - c=8:tight(87% with KV,may need shorter max-seq-len)
  - **c=16:NOT FEASIBLE** without KV quant(`#33` master §1.2.1.B paired axis)
- Mixed-step dispatch:Option A(prefill priority) — use W4A8 for ALL
  Linear in mixed steps,decode capped by prefill chunking anyway

### Concurrency sweep results(`8588f6a`)
| Concurrency | W4A16 ITL | W4A8 ITL | W4A16 TTFT | W4A8 TTFT |
|-------------|---:|---:|---:|---:|
| c=4 | 11.73 ms | 19.18 ms(+63%)| 2388 ms | 1632 ms(-32%)|
| c=8 | 16.28 ms | 24.09 ms(+48%)| 4811 ms | 3323 ms(-31%)|

- Decode gap **narrows -9% as concurrency doubles**(W4A8 catching up)
- Prefill advantage **holds robustly**(both -31% to -36%)
- **Crossover formula c ≈ 78** — impractical at production memory budget
- **Practical conclusion**:W4A8 decode never crosses W4A16 at production c=4-32

So decode-side decision **W4A16 dominates throughout production range**。
Hybrid case is **purely about prefill TTFT win** at c≤8 where memory
allows side-by-side storage。

## §5 Probability + ROI

P(hybrid landed within 1-2 days) = 75% — UPDATED post-Phase-0 to 80%
- High because both arms are LICENSED individually,routing is mechanical
- Risk concentrated in loader storage augmentation(Phase 1)
- Phase 0 reconnaissance(`1959a21`)confirmed scheduler clean + impl
  scope ~50 LOC `HybridLinear` simpler than expected
- BUT memory budget tight at c=8 + infeasible at c=16 → **scope limited
  to c≤8 single-tenant production**(c=16 hybrid blocked by KV W4A8 task #33)

ROI estimate(production E2E):
- W4A16-only baseline:TTFT 2388 ms + ITL 11.73 ms × 256 tok = 5391 ms
- W4A8-only:TTFT 1632 ms + ITL 19.18 ms × 256 tok = 6543 ms
- **Hybrid**:TTFT 1632 ms + ITL 11.73 ms × 256 tok = **4635 ms**
- Hybrid vs W4A16-only:**−14% E2E latency**
- Hybrid vs W4A8-only:**−29% E2E latency**

For agent workload(short prompt + short output),decode dominates →
W4A16-only might be enough。But for long-output(>=256 tok)workloads,
hybrid is genuinely better。

## §6 Decision points

D6.1 — **Pursue hybrid OR stay with W4A16 default**:
- A. Pursue per this plan(1-2 days,probability 75%)
- B. W4A16 default,W4A8 as opt-in for prefill-heavy workloads
- C. Defer until Qwen3.6 35B-A3B production deployment(may need
  hybrid for 35B due to compute scale)

Recommendation:**A** — concrete win(-14% E2E),small impl risk,
unblocks future production deployment without revisit。Codex picks up
when D2'-D5 settled。

D6.2 — **Side-by-side weights vs dynamic conversion**:
- A. Side-by-side(2× weight memory,zero conversion overhead)
- B. Dynamic conversion at phase switch(1× weight memory,~50ms switch)
- C. Pick A for ≤4B models,B for ≥7B models

Recommendation:**A for now**(Qwen3-4B fits 16GB easily)。Re-evaluate
when 35B production looms。

## §7 Cross-references

- W4A16 LICENSED: [`bc15eca`](../experience/wins/2026-05-08-w4a16-gptq-zpfix-round4-bench.md)
- W4A8 prefill LICENSED: [`b5889b3`](../experience/wins/2026-05-08-w4a8-gptq-zpfix-canonical-bench.md)
- qzeros fix: [`2a3a6f0`](../research/2026-05-08-gptq-qzeros-zero-minus-1-convention-bug.md)
- Master strategy update: [`182e084`](../projects/2026-05-07-arle-master-strategy.md)
- R4 #6 hybrid KILL precedent: `4571082`(W4A16 dynamic batch-size dispatch — different from this plan)
- M_quant umbrella: [`662cbbb`](M_quant-autogptq-marlin-integration.md)
- EOD+43 synthesis: [`b04b5fb`](../research/2026-05-08-eod43-w4a8-axis3-licensed-end-to-end.md)

## §8 Methodology validation

This plan exists because empirical bench(`b5889b3`)found
**complementary win pattern**(W4A8 prefill / W4A16 decode)— neither
arm strictly dominates。Static phase routing is the natural pattern
when each arm has distinct strengths。

Per skill v1.4.0 anti-pattern #14:always validate that NEW hybrid
proposals don't share the same upstream-data corruption issue that
killed previous attempts。Both W4A16 and W4A8 paths use the same
corrected `convert_gptq.py +1` qzeros decoding,so the foundation is
sound。

## §9 Status

**Plan ready for codex pickup** when D2'-D5 settled and codex has
bandwidth。Effort 1-2 days,impl risk Med。Strategic value:close axis
3 weight quant 全套 with production-optimal routing for Qwen3-4B。

If user prefers Medusa(D3)or TTFT tail(D4)first,this plan can wait。
The W4A8/W4A16 wins individually are already in production。Hybrid is
icing on cake(-14% E2E),not blocker。
