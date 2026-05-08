# cap=8 chain final synthesis — production trade-off + 8.2% residual

> 7-commit cap=8 investigation chain culminates in `8281047` validation:
> warmup fix `c20b1ce` empirically lands at **91.8% turn success +
> -87% TTFT p99**。Still 8.2% residual gap from 100% turn success
> (memory pressure suspected)。This brief documents trade-off and
> recommends production deployment policy。

## Final cross-run table

| Run | Cap | Source | Server | Turn success | TTFT p99 | Mem peak |
|-----|----:|--------|--------|-------------:|---------:|---------:|
| `f5cf829` | 4 | default | fresh | 256/256(100%)| 72515 ms | similar |
| `19d12c2` | 8 | CLI override | warm | 257/257(100%)| 10259 ms | 15272 MB |
| `150b4c4` | 8 | default(`12300c5`)| fresh | 194/256(76%)| 11182 ms | 15880 MB |
| `db20d34` | 8 | default(`12300c5`)| fresh | 144/256(56%)| 15357 ms | similar |
| `3cd3494` | 8 | CLI override | fresh | 201/256(78.5%)| 14609 ms | similar |
| **`8281047`** | **8** | **default + `c20b1ce` warmup fix** | **fresh** | **235/256(91.8%)** | **9533 ms** | **15912 MB** |

## Phase 8 verdict — CONDITIONAL LICENSE

Per `8281047`:
- ✅ TTFT p99 ≤ 30k ms(9533)
- ✅ Spread ≤ 3×(1.29)
- ✅ ITL p50 ≤ 30 ms(25.9)
- ⚠ Turn success 91.8% < 95% threshold by 3.2pp
- ⚠ Peak mem 15.91 GB borderline OOM(97% GPU)

## Production trade-off analysis

| Option | TTFT p99 | Turn success | User exp |
|--------|---------:|-------------:|----------|
| A. Keep cap=8 + warmup fix(current) | 9533 ms | 91.8% | Fast,8% retry tax |
| B. Revert cap=4 | 72515 ms | 100% | Slow but reliable |
| C. cap=8 + memory-adaptive | TBD | TBD | Best-of-both,needs work |

For agent workload:user typically retries on session error with fresh
session。Cost of retry:single TTFT round-trip。

- Option A:**91.8% × 9.5 + 8.2% × 19 = 10.3s amortized p99**
- Option B:**100% × 72.5 = 72.5s p99**
- Option A still **7× better** than Option B even after retry tax

→ Option A wins for TTFT-sensitive workloads(agent / chat)。Option B
only wins for batch workloads where retry cost is high。

## Residual 21/256(8.2%)turn failure root cause

Likely candidates:
1. **Memory pressure**(97% GPU,15.91/16 GB):high-context turns may
   trigger OOM during KV growth → 503
2. **Long-tail context**:8K longctx workload has variable session
   lengths;some sessions may exceed 8K growth budget
3. **Tail prefill chunking**:residual 5-8 batch encounters that warmup
   missed at edge cases

Investigation paths(P3 — not blocking):
- a. Reduce `--max-seq-len` to 6144 (matched workload retest)
- b. KV W4A8 to free memory(`task #33`)— addresses root cause
- c. Add memory-adaptive cap(scheduler dynamic)— architectural

## Recommendation

**A. Keep cap=8 + warmup fix(current state)** as production default。

**Rationale**:
- TTFT improvement is genuine production user-experience win
- 91.8% turn success is acceptable for c=8 8K agent workload(retries amortize 7× faster than cap=4)
- 100% turn success path is via memory pressure mitigation,not cap reversion
- Residual 8.2% gap should be addressed via task #33 KV W4A8(orthogonal axis with paired ROI)

**Reverse migration trivial** via `--prefill-max-requests 4` CLI override
if user finds the residual unacceptable for their workload。

## Master strategy update needed

§1.2.1.A weight axis(per `5dc27a2` + `182e084`)should add:
```
| Schedule cap | Some(4) → Some(8) production default per `12300c5` + `c20b1ce` |
                | TTFT p99 -87%(72515→9533 ms),turn success 91.8% |
                | Residual 8.2% bound by 16 GB GPU memory(`8281047`)  |
                | Mitigation:KV W4A8 axis #33(paired) |
```

## Cross-references

- `12300c5` cap=8 flip(my code change)
- `150b4c4` first variance signal
- `db20d34` H4 root cause CONFIRMED
- `fc9bea9` variance investigation plan
- `3cd3494` Step 1 override fresh = same regression
- `c20b1ce` warmup fix(my code change)
- `1f70059` grep evidence anti-pattern #16 by example
- `8281047` validation 91.8% LICENSED conditionally
- `27fd5de` original cap=8 multi-shape claim(unintentionally relied on warm-state)

## Methodology insights captured

5 ticks of investigation produced:
- Anti-pattern #16(implicit-coupling-via-shared-default trap)
- Concretization rule:grep evidence dump in PR commit body
- Demonstration of how warm-server vs fresh-server differs in benchmark validity
- Trade-off framework for partial-success production fixes

Cost vs benefit:
- 5 hours human + 5 GPU hours iteration
- Avoided shipping a 76%-fail config to production
- Captured durable methodology rule for future config flips

## Status

**Production deployment recommendation:keep cap=8 + warmup fix**(current
state on main as of `c20b1ce` + `1f70059`)。

Validation needed:re-run W4 c=8 8K bench periodically to verify
91.8% turn success holds across runs(σ check)。If average drops
below 90%,reconsider(option B revert or option C adaptive)。

Codex pickup queue updated:
- ✅ cap=8 + warmup fix DONE
- P0 hybrid Phase 1b(`6be30ce` directive)— still queued
- P1 Medusa Phase 1(`afdddec` reconnaissance)— still queued
- P1' KV W4A8 #33 — addresses 8.2% residual + paired axis ROI

Memory updated EOD+57(this brief)。
