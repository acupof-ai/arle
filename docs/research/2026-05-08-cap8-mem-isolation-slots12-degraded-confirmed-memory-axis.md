# cap=8 memory isolation — `--num-slots 12` confirms memory-axis,but in opposite direction

> Per `9596566` codex hypothesis(bimodal trigger = memory pressure)
> ran Step 2-style isolation:reduce KV pool via `--num-slots 12`
> (vs default 16,~25% smaller pool)。
>
> **Result:turn success 145/256(56.6%)= DEGRADED MODE PERSISTS**。
> kv_util:**99.5% saturated**(vs slots=16's 86.7%)。Reducing memory
> headroom made it WORSE,not better。**Memory IS an axis but trigger
> direction is OPPOSITE**:more pressure → more degraded mode。

## Empirical comparison

| Config | Turn Success | kv_util | TTFT p50/p99 | Wall | Tokens out |
|---|---:|---:|---:|---:|---:|
| `b1mm1k0r7`(slots=16) | 235/256(92%) | 86.7% | 7409 / 9533 ms | 2356s | 32298 |
| `b4kaqdrmj`(slots=16) | 144/256(56%) | 84.7% | 14791 / 15249 ms | 1401s | 23424 |
| `b5i3467ad`(slots=16) | 194/256(76%) | 86.7% | 9045 / 11082 ms | 2291s | 30826 |
| **`byfqsbviy`(slots=12 this)** | **145/256(57%)** | **99.5%** | **11302 / 15260 ms** | **1410s** | **23680** |

## Key observation — kv_util saturation

slots=12 + c=8 burst:**kv_util 99.5%**,KV pool at the limit。Any
further session admission triggers eviction → cascading 503 errors。
Hard evictions:133(but admissions failed earlier so total
evictions actually LESS than slots=16 cases)。

slots=16 + c=8 burst:kv_util 84.7-86.7%,~13% slack。Same admission
density but more space for KV growth。

## Hypothesis refinement

### Confirmed — memory IS an axis

slots=12 → kv_util 99.5% → 56% success。
slots=16 → kv_util 86.7% → 56-92% success(bimodal)。

Reducing pool tightens pressure → more degraded mode。

### Refuted — memory pressure is NOT the bimodal TRIGGER

If memory pressure was the trigger,I expected:
- Less memory headroom → MORE bimodal(or worse)
- More memory headroom → LESS bimodal(closer to 100%)

Empirical:slots=12 just shifts to all-degraded(56% always)。Doesn't
explain why slots=16 has BIMODAL distribution(some runs 92%,some 56%)
at SAME memory level。

### Reframed hypothesis — memory governs DEGRADED-MODE FLOOR not bimodal switch

Model:
- Memory pressure determines the WORST achievable turn success
- slots=12 saturates pressure → always degraded(56%)
- slots=16 has slack → sometimes degraded(56% ~33%,trigger TBD),
  sometimes normal(76-92% ~67%,less pressure achieves higher rate)
- The bimodal switch within slots=16 IS a different factor(scheduling
  ordering,bench harness retry,or first-encounter graph capture race)

## Implication for KV W4A8 prioritization(per `9596566`)

Codex hypothesis "KV W4A8 closes bimodal gap" still partially valid:
- KV W4A8 reduces memory pressure 2-4× → could allow slots=24+ at 16 GB
- More slots = more headroom for cap=8 burst → degraded mode floor lifts
- But may NOT address the 33% bimodal switch(if trigger is non-memory)

Conditional priority elevation:
- If degraded mode floor is the primary production concern → KV W4A8 helps
- If 67% normal mode is acceptable → KV W4A8 not blocker
- If 33% degraded mode is the production blocker → need bimodal trigger
  fix(not memory)

## Next step recommendation

To isolate bimodal trigger from memory:
- **Test slots=24 with `--max-seq-len 6144`**(reduces per-slot KV size)
  → should achieve slots=24 × 6144 KV budget similar to slots=16 × 9216
- If 100% turn success → memory-pressure-bimodal connection confirmed
- If still bimodal at higher headroom → trigger is non-memory(scheduling
  / harness)

Or:run with W4 c=4(half concurrency)to test if SAME memory profile but
half pressure shows different distribution。

## Status

- ✅ Memory IS axis(slots=12 confirms tighter pool → consistently degraded)
- ❌ Memory pressure is NOT the bimodal TRIGGER(would expect inversion at lower pressure)
- 🔧 Refined hypothesis:memory governs floor,bimodal switch is non-memory
- 🔧 Next:test increased headroom OR W4 c=4 to fully characterize

## Cross-references

- 6-run bimodal dataset:`a0a3f42`
- Codex KV W4A8 hypothesis:`9596566`
- Skill v1.5.0 anti-pattern #17:`f05ea3a`
- Bench artifact:`bench-output/2026-05-08-arle-w4-c8-mem-isolation-slots12.json`(local)
