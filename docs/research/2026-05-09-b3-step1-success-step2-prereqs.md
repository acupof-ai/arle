# B3 Step 1 SUCCESS + Step 2 prerequisite analysis(A1 RadixCache integration)

> Per `7c8fd61` admission_allows signature refactor + `c30e298` byte-identical
> production verify,B3 Step 1 LANDED successfully(my `e53e4d8` directive
> executed cleanly)。
>
> **Step 2 has implicit dependency on A1 RadixCache integration** —
> this brief audits A1 readiness before Step 2 codex pickup。

## B3 Step 1 success summary

`7c8fd61`:
- 14 insertions / 4 deletions across 1 file
- `admission_allows` signature changed to take `SchedulerSignals`
- `submit()` + `is_full()` callers pass `SchedulerSignals::queue_state(N, 0)`
- `QueueBoundAdmission` body passes through(was constructing internally)
- Behavior preserved:`QueueBoundAdmission` only reads `queued_requests`,no semantic change

`c30e298` production verify:
- W3 c=4 short multiturn fresh-build bench
- **Byte-identical** to `063da81` baseline:384/384,TTFT p50 208ms,ITL p50 8.5ms
- All metrics within σ noise band
- Step 1 production-safe empirically confirmed

**Effort actual**:1 cron tick of codex pickup,~30 LOC + tests + bench verify(within `e53e4d8` 0.5d estimate)。

## Step 2 prerequisite — A1 RadixCache integration

`types.rs:584-588` comment:
> When present, the scheduler will **(once A1's RadixCache integration lands)** prefer to route successive turns of the same session to the slot or radix subtree that already holds their KV prefix.

Per `docs/projects/agent-first-architecture.md::A1`:
- M1b `323aee0`:RadixCache wired as **shadow observer**(production-side reads but doesn't act)
- M2a `4402ab0`:RadixCache scheduler integration(extent unknown)

→ Status:**partially shipped**。

For B3 Step 2 to compute `prefix_hit_tokens` at `submit()`,need:
1. `SchedulerHandle` to have access to live RadixCache(currently no field)
2. `RadixCache::match_prefix(tokens)` to work as **production lookup**(not just shadow observer)
3. RadixCache must be **populated** with active KV prefixes(not empty at startup)

## Step 2 scope refinement

Original `e53e4d8` Step 2 estimate:100-150 LOC,0.5 day。

If A1 is production-ready:
- Scope holds(100-150 LOC,plumb RadixCache into SchedulerHandle + lookup at submit)

If A1 is shadow-only:
- A1 production wiring becomes prerequisite(unknown LOC,~200-500 LOC est)
- Step 2 timeline:1-2 days additional

## Codex action — A1 readiness audit needed first

**Recommended sequence**:
1. **A1 audit**(0.5d Claude side):code-grep `M1b 323aee0` + `M2a 4402ab0` to assess RadixCache integration depth
2. **If A1 ready**:proceed to B3 Step 2 per `e53e4d8` Step 2 outline(100-150 LOC)
3. **If A1 shadow-only**:defer B3 Step 2,unblock A1 production-wiring first

**A1 audit grep checklist**:
- `RadixCache::match_prefix` callers in production scheduler path
- `prefix_hit_tokens` field consumers(currently policy.rs only)
- session_id → slot routing in scheduler/cuda/

## Pickup queue update

| Priority | Task | Status |
|----------|------|--------|
| P0 | Hybrid Phase 1b loader | Queued(`6be30ce`)|
| P0' | M_warmup prefill pass | Queued(`56dbd1c`)|
| **DONE** | **B3 Step 1 admission_allows signature** | ✅ **Landed**(`7c8fd61` + `c30e298`)|
| **NEW P0** | **A1 readiness audit**(0.5d Claude)| Pending — blocks Step 2 |
| P1 | B3 Step 2(post-A1 audit) | 100-150 LOC,0.5d if A1 ready |
| P1 | KV W4A8 #33 | 5-10d |
| P1' | Medusa Phase 1.B | 10-14d |

## Methodology insight

Step 1 success demonstrates **smallest-verification-step pattern works**:
- 30-50 LOC change + tests + 1 bench verify
- Production-identical empirical confirmation
- Unblocks chained Steps 2-3 without breaking anything

Per `e53e4d8` pattern,each subsequent step continues this practice。
Risk concentrated in Step 2(RadixCache integration)— audit A1
readiness before committing to Step 2 effort estimate。

## Cross-references

- Step 1 directive:`e53e4d8`
- Step 1 implementation:`7c8fd61`
- Step 1 verify:`c30e298`
- A1 spec:`docs/projects/agent-first-architecture.md`
- A1 partial-shipped commits:`323aee0`(M1b shadow observer)+ `4402ab0`(M2a scheduler integration)
- B3 SGLang gap source:`a1965ab`
- PrefixAwareAdmission impl:`policy.rs:98-130`(verified existing per `c0ddd4f`)
- submit() landing site:`types.rs:740-764`(post-`7c8fd61`)

## Status

B3 Step 1 ✅ LANDED + production-verified。Step 2 unblocked at scope
level,requires A1 readiness audit before LOC estimate finalizes。

A1 audit is the actual P0 next step now — if A1 is shadow-only,Step 2
timeline grows from 0.5d → 1-2d。If A1 is production,Step 2 proceeds
per original estimate。

Codex pickup queue:A1 audit can be Claude-side(0.5d)or codex-side
(faster,direct code knowledge)。
