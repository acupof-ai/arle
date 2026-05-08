# 2026-05-08 EOD+43 — 🎉 AXIS 3 W4A8 LICENSED END-TO-END

> Codex `2a3a6f0` shipped 1-line fix(`+1` to qzeros decode in
> `convert_gptq.py`)。**Greedy gate `test_w4a8_vs_bf16_token_diff`
> PASSES with 32/32 tokens matched, 0.0% diff**。
>
> This concludes the 9-iteration W4A8 debugging chain(EOD+22 → EOD+43,
> ~22 hours of cron+codex collaboration)。Master strategy §1.2.1.A
> axis 3 weight 全套 unblocked simultaneously for W4A16 + W4A8。

## What landed

`2a3a6f0` `fix(cuda): correct GPTQ zero-point decode in converter`:
```python
# scripts/convert_gptq.py:53-56
z_expanded = (qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & mask
# GPTQ/AutoGPTQ-family checkpoints store zero-points as (zero - 1).
# Convert back to the real zero before applying the signed transform.
zeros_unpacked = z_expanded.reshape(num_groups, N) + 1  # ← THE FIX
```

Validation(per codex pane):
1. **W4A16 corrected source**:`test_greedy_solo_vs_concurrent` PASS,
   output coherent English
2. **W4A8 corrected source**:`test_w4a8_vs_bf16_token_diff` PASS,
   32/32 tokens matched, 0.0% diff
3. `py_compile` PASS
4. `codex review --uncommitted` 2 rounds:0 actionable findings
5. Working tree clean,GPU not occupied

Doc updates:
- New errors entry:`docs/experience/errors/2026-05-08-gptq-qzeros-off-by-one-broke-w4a8-source.md`
- 2 old research entries superseded with notes
- D2 ready-to-execute updated with correct 2-step conversion

## 9-iteration cumulative chain(EOD+22 → EOD+43)

| Iter | Fix | Outcome | Real role |
|------|-----|---------|-----------|
| H3 | row stride skip-8 vs 4-cons | partial | wrong-class identification |
| H3 revert | back to 4-cons | partial | restored W4A8Layer-correct |
| H3b | scale_perm_single applied to s_channel | partial | minor cleanup |
| H3c | scale_perm_single AFTER division | regressed initially | applied/reverted/reinstated |
| H4 | remove `s_pack = s.t()` | partial | broadcast index alignment |
| MAGIC_NUM | identify kernel IEEE-754 fast-path bound | important | safety constraint |
| Fix A | clamp s_group_stored ≤ 16 | safety patch | kernel range guard |
| GPTQ-aware | `pack_w4a8(gptq_scales=...)` | calibration mechanism | preserve GPTQ values |
| **+1 fix** | qzeros zero-1 convention decode | **REAL ROOT CAUSE** | **single source of all symptoms** |

**8 of 9 iterations were red herrings** — secondary bugs / safety patches
that didn't address the root cause。But each iteration **was real
progress**:
- H3-H4:fixed pack/perm contract bugs(real,but secondary)
- MAGIC_NUM:identified real kernel constraint(safety,still useful)
- Fix A:adds defense-in-depth even after root cause fix(keeps `163c8ee`)
- +1:1-line fix that unblocks both W4A16 and W4A8 simultaneously

## Strategic state — axis 3 LICENSED ✅

Master strategy §1.2.1.A weight quant 全套(updated `5dc27a2`):

| Format | Pre-fix | Post-fix |
|--------|---------|----------|
| BF16 | ✅ baseline | ✅ |
| FP8(weight + activation)| 🔴 cuBLASLt 1.88× KILL,cutlass v2 待验 | unchanged |
| **W4A16 GPTQ Marlin** | ⚠ "marginal accuracy"(actually +1 corrupted!)| **✅ greedy_solo_vs_concurrent PASS — coherent English** |
| **W4A8 GPTQ-Marlin re-pack** | ❌ all-`!` garbage | **✅ test_w4a8_vs_bf16_token_diff PASS 32/32 0.0% diff** |
| TurboQuant W2/W3/W4 | ✅ production | unchanged |
| W4A_INT8 | 📋 deferred | unchanged |
| NVFP4(sm_100 only)| 📋 substrate | unchanged |

**Both W4A16 and W4A8 production accuracy unblocked by 1-line `+1`
fix in `convert_gptq.py`**。No kernel changes,no FFI changes,no loader
changes(all those were already correct per audit `01ace86`)。

## Decision points NOW

### D2'(updated):W4A8 GPTQ guidellm bench — proceed?
Per CLAUDE.md mandatory bench rule:every runtime change → wins/ entry。
The `convert_gptq.py +1` fix counts as runtime-affecting(loader path
weight values changed)。

Steps:
1. Re-run `convert_gptq_w4a16_to_w4a8_marlin.py` with corrected source
   to produce `infer/models/Qwen3-4B-GPTQ-W4A8-marlin/`
2. `./scripts/bench_guidellm.sh m_quant-w4a8-gptq-canonical-bench`
3. Compare TTFT/ITL/throughput vs:
   - BF16 baseline
   - W4A16 Marlin(corrected,re-bench needed)
   - SGLang / vLLM W4A8 if available
4. Ship `wins/2026-05-08-w4a8-gptq-canonical-bench.md`

Recommendation:**proceed** — 30 min total。Unblocks default-on flip
decision。

### D3:Medusa axis 2 implementation — start now?
Plan `528844c` ready,4-KILL classical-spec evidence dead。Production-
shape baseline exists post-`b708e00` admission unblock。

Recommendation:**start after D2'** — serializes GPU,Medusa is 1-2 weeks。

### D4:TTFT p99 tail-latency plan — write now?
W4 c=8 license noted "TTFT p99 still very poor"(separate from liveness
deadlock fix)。Codex flagged as separate plan-needed item。

Recommendation:**write low-priority `M_pf-tail-latency`(0.5d Claude),
implement later** — don't block axes 2/3。

### D5(NEW):Default-on flip W4A8 — when?
Per `62e75ee` plan,W4A8 default-on flip blocked on graph capture wiring。
Now that accuracy is preserved,graph capture work can proceed。

Recommendation:**hold flip until graph capture wired**(separate
`M_pf-graph-prefill-capture` task #22 closed/killed per master strategy
KILL log,need to re-evaluate)。Use W4A16 Marlin as default until
graph wired。

## Cumulative loop value(22-hour cron run)

EOD+22 → EOD+43:
- **30+ commits across cron Claude + codex**
- **2 strategic axis milestones**:
  - Axis 1(agent workload):W4 c=8 100% / W3 c=16 98%(`b708e00` `f5cf829`)
  - Axis 3(weight quant):W4A8 GPTQ greedy gate PASS(`2a3a6f0`)
- **5 methodology rules captured**(adds to skill v1.3.0 → v1.4.0):
  1. Round-trip diagnostic FIRST when investigating quantization
  2. Identify EXACT class hierarchy(Layer vs W4A8Layer)
  3. Iteration scope matches budget accounting period
  4. Tensor shape ≠ byte layout(perm pattern matters)
  5. **NEW(skill v1.4.0)**:Audit upstream-data parsers BEFORE internal kernel logic — silent +1 corruption can hide for ~1 year
- **Multi-agent collaboration validated**:codex(GPU + substrate)+
  cron Claude(GPU-independent doc/audit)pattern produced 30-min
  fix-validation cycles when blocker localized

## Cross-references

- `2a3a6f0` THE FIX(qzeros +1 in convert_gptq.py)
- `5593865` Claude-side qzeros bug brief
- `6c627c4` codex skill v1.4.0 anti-pattern #14
- `5dc27a2` master strategy §1.2.1.A update
- `b708e00` axis 1 admission deadlock unblock
- `e753af7` Phase 1b script-level PASS(was misleading — corrupted source)
- `39237b9` "naive max-scale W4 too lossy"(was misleading — corrupted source)
- `01ace86` kernel + wiring audit clean(correct,but missed upstream parser)
- `36830bf` EOD+34 loop synthesis
- `fdb951f` EOD+37 milestone consolidation

## Methodology rule earned

The 8-iteration red-herring chain(H3 → H3b → H3c → H4 → MAGIC_NUM →
clamp → GPTQ-aware → naive-overlay)before the +1 root cause was
discovered. Each iteration was empirically driven and produced real
finding,but **all were downstream of the upstream parser bug**。

**Cost**:~22 hours human + ~3 GPU-hours compute + many commits and
research briefs。

**If round-trip diagnostic against AutoGPTQ source spec had been done
on day 1**:fix would have been ~1 hour total(spot the +1 missing,
patch,validate)。

**Methodology rule(distilled)**:When investigating
"checkpoint-shaped output corruption":
1. Dump raw upstream tensor values(qweight/qzeros/scales)
2. Compare to source format spec EXACTLY(AutoGPTQ source code,
   not docs which may lag)
3. Verify each "hidden contract"(zero-stored-as-zero-1,sym/asym,
   scale magnitude convention,g_idx interpretation)
4. ONLY then iterate on internal pack/kernel layers

Internal pack/kernel iteration without this upstream check produces
diminishing returns and accumulates "explanation patches" that are
red herrings until a fresh perspective(codex's `qzeros all = 7`
empirical dump)reveals the real bug。

## Status

**Codex idle**(EOD+43,after 47m 42s work session)。Awaiting next
direction:D2'(bench)/ D3(Medusa)/ D4(tail latency)/ D5(default-on)。

PushNotification sent for milestone。All blockers cleared for axis 3
production deployment(pending bench numbers + default-on decision)。
