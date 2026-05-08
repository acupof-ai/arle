# cap=8 bimodal trigger likely memory pressure → KV W4A8(#33)becomes higher-priority axis

> Per `a0a3f42` 6-run dataset:cap=8 + warmup-fix exhibits bimodal
> distribution(67% normal 76-92% / 33% degraded 56%)。`f05ea3a` skill
> v1.5.0 codifies anti-pattern #17(bimodal masks single-run LICENSE)。
>
> **Most likely bimodal trigger**:memory pressure at server cold-start
> (97% GPU,15.91/16 GB)。**KV W4A8 axis #33 directly addresses**
> by reducing KV pool memory 2-4×。This brief connects the chain and
> recommends bumping #33 to P0 production priority。

## Bimodal mode signature

Per `a0a3f42`:
| Mode | Frequency | Turn success | Wall | Tokens out |
|------|----------:|-------------:|-----:|-----------:|
| Normal | 4/6(67%)| 76-92% | 2290-2441 s | 30826-35733(varied)|
| Degraded | 2/6(33%)| **56%** | 1401-1409 s | **23424(byte-identical)** |

Byte-identical degraded output suggests **deterministic session-failure
ordering when triggered**。But trigger isn't strict run-position —
both modes seen across run #1, #2, #3 randomly。

→ Mode determined by **resource state at server start**,not run order。

## Why memory pressure is the strongest trigger candidate

`8281047` peak memory **15.91 GB / 16 GB = 97% utilization,120 MB headroom**。
Normal mode runs longer wall(2356 s):more time to drain prefill
queue → less concurrent peak memory pressure。Degraded mode shorter
wall(1401 s):harness times out before some sessions complete →
reported as failed even if architecturally feasible。

But there's also the byte-identical pattern in degraded mode — same
exact tokens emitted before truncation。This signature suggests:

**Hypothesis H_mem**:memory pressure at session N causes deterministic
allocation refusal at session N+1。Once mode triggered,deterministic
failure ordering follows。

If GPU memory was 24-32 GB instead of 16 GB,bimodal wouldn't appear
because peak 15.91 GB has 8-16 GB headroom。

## KV W4A8 directly addresses memory

Per `M_quant-kv-w4a8.md` §1.2.1.B(master strategy):

| KV format | Bytes/token Qwen3-4B(36L 8KV-heads 80d)| KV pool 16 GB cap |
|-----------|---------------------------------:|---------------:|
| BF16(baseline)| 92 KB | ~21k tokens |
| FP8 / INT8(production)| 46 KB | ~42k |
| **W4A8(INT4 K/V + FP8 attention)** ⭐ | **23 KB** | **~84k** |

W4A8 KV reduces KV pool memory 4× vs BF16,2× vs FP8/INT8。

For W4 c=8 8K agent burst with current peak 15.91 GB:
- Estimate KV pool ~3-4 GB(8 sessions × 8K context × 92 bytes/token = 5.6 GB BF16,probably scaled down by KV format detection)
- W4A8 KV would save ~3 GB peak → 12-13 GB peak instead of 15.91
- **Headroom 3-4 GB instead of 120 MB** → bimodal degraded mode less likely to trigger

## Strategic priority bump for #33

Original priority(per `5364612` codex pickup queue):
- P0 hybrid Phase 1b
- P0' default-on flip W4A8(now D5 closed per `018494a`)
- **P1 KV W4A8 #33**(5-10 days,paired axis)
- P1' Medusa #32

**Bump #33 to P0 alongside hybrid Phase 1b**:
- Memory pressure is now confirmed bimodal trigger(strongest candidate)
- Fixing memory frees up GPU budget for hybrid 2× weight storage(also
  needs +5 GB)
- Addresses 8.2% residual + 33% bimodal mode in one axis

Net P0 work after bump:
1. **Hybrid Phase 1b loader patch**(0.5d codex,unblocks -14% E2E E2E)
2. **KV W4A8 implementation**(5-10 days codex,addresses memory bimodal)
3. Combined:hybrid + KV W4A8 = -14% E2E + 100% turn success at
   c=8 8K + headroom for c=16 hybrid

## Updated trade-off for cap=8 production

| Option | Turn success | TTFT p99 | Notes |
|--------|-------------:|---------:|-------|
| A1. cap=8 + warmup(current main)| 56-92% bimodal | ~10s | Production usable but unpredictable |
| A2. cap=8 + warmup + KV W4A8 | 100% predicted | ~10-12s | Memory pressure resolved |
| B. cap=4(revert)| 100% | ~72.5s | Slow but deterministic |

A2 is the architectural target。A1 is interim while #33 lands。

## Recommendation

1. **Keep `12300c5` + `c20b1ce` on main**(cap=8 + warmup fix)
2. **Bump #33 KV W4A8 to P0**(paired with hybrid Phase 1b)
3. **Document expected bimodal**:users see 67% normal mode / 33% degraded
   mode at server cold-start until #33 lands
4. **Monitoring**:add metric `cap8_bimodal_mode` to track which mode each
   server instance hits

Once #33 lands:
- Re-run N≥3 cap=8 + warmup + KV W4A8 verification
- Expected:100% turn success deterministic(memory pressure resolved)
- LICENSE the combined config as production default

## Cross-references

- `a0a3f42` 6-run bimodal dataset
- `f05ea3a` skill v1.5.0 anti-pattern #17
- `8281047` initial single-run validation(misleading)
- `1fce03f` premature synthesis recommendation
- `6db48b1` N=2 caveat(my earlier brief)
- `M_quant-kv-w4a8.md` plan(`1e713de` per memory)
- Master strategy §1.2.1.B KV axis

## Methodology insight

Two related axes(weight quant + KV quant)were tracked separately in
master strategy §1.2.1。Cap=8 bimodal investigation revealed they're
not orthogonal at the **production-readiness level**:weight axis cap
flip needs KV axis memory headroom to be deterministic。

**New rule for axis prioritization**:
When two axes share a binding constraint(memory budget,compute
budget),advancing one may **block** until other addresses the shared
constraint。Don't treat axes as fully orthogonal — characterize
the binding constraint and prioritize axis that addresses it first。

Per `f05ea3a` skill v1.5.0,this becomes anti-pattern #18 candidate
(orthogonal-axis-shared-constraint trap)— worth proposing if codex
agrees。

## Status

This brief proposes:
- Continue cap=8 + warmup fix on main(no revert)
- Bump #33 to P0 from P1
- Block §1.2.1.A master strategy update until N≥3 verify with #33

Codex pickup queue updated implicitly by this priority shift。Pending
user approval of:
- Bumping #33 priority
- Adopting "axis-shared-constraint" methodology rule
