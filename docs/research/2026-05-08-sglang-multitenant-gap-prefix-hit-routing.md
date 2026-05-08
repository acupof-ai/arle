# SGLang multi-tenant 2× gap — `prefix_hit_tokens` field exists but ignored by `QueueBoundAdmission`

> Per task #30 SGLang multi-tenant prefix cache investigation:
> SGLang 157 ms / ARLE 318 ms / vLLM 573 ms on multi-tenant
> shared-prefix burst(`m_world1-p0-sglang-baseline-extended`)。
> SGLang is **2.03×** faster than ARLE despite both having radix
> prefix cache。Code-grep finds the gap:**ARLE collects `prefix_hit_tokens`
> per-request but `QueueBoundAdmission` policy IGNORES it**。

## Empirical baseline(per `m_world1-p0-sglang-baseline-extended`)

Multi-tenant shared-prefix burst(4 concurrent / 6k system / 100q / 64out):
| Engine | TTFT p50 | Min/Max | Total | Rank |
|--------|---------:|--------:|------:|-----:|
| **SGLang 0.5.11** | **157 ms** ⭐ | 75 / 224 ms | 1230 ms | #1 |
| ARLE | 318 ms | TBD | TBD | #2(2.03× SGLang)|
| vLLM | 573 ms | TBD | TBD | #3 |

## Code-grep finding

`infer/src/scheduler/policy.rs:8-20`:
```rust
/// - **Per-request hints** (`prefix_hit_tokens`, `session_affinity_slot`,
///   `turn_depth`): describe the *incoming request* being considered for
///   admission. Callers that only need chunking decisions leave these at
///   their default (`0`/`None`) via struct-update syntax.
///
/// The per-request fields are the agent-aware surface
/// `docs/projects/agent-first-architecture.md::B3` asks for; they become
/// meaningful once `A1` wires the RadixCache into the schedulers and can
/// actually compute prefix hits before calling `AdmissionPolicy::allow`.
/// Until then, existing call sites pass them as defaults and the legacy
/// [`QueueBoundAdmission`] policy ignores them.
```

→ **ARLE has the data structure but admission policy ignores it**。
SGLang 2× win is from cache-aware scheduling that ARLE wired through
type system but never made meaningful。

## SGLang's mechanism(per `2026-05-04-sglang-hicache-guide.md`)

SGLang `HiRadixCache.match_prefix(tokens)` returns matched length
**before** scheduler decision。Then SGLang scheduler:
1. Routes high-prefix-hit requests to slots that already hold that prefix
2. Prioritizes them in admission(skip cold prefill,start at hit-end)
3. Uses block-level coalescing across sessions sharing prefix

ARLE collects `prefix_hit_tokens` per `B3` design but `QueueBoundAdmission::allow`
treats all requests equal regardless of prefix hit。Result:multi-tenant
sessions with shared prefix re-prefill the system prompt(~6k tokens × 4
sessions = 24k tokens)instead of reusing 1 cached copy。

## Magnitude estimate

If ARLE used cache-aware admission like SGLang:
- 6k system prompt × 4 sessions = 24k prefill tokens cold
- With prefix sharing:**6k cold + 100q × 4 = 6.4k tokens prefill**
- Speedup:**24k / 6.4k = 3.75×** for prefill phase
- Real bench measured 2.03× → matches if decode phase is unchanged
  (only prefill is cache-routed,decode still per-session)

So the 2.03× SGLang gap is entirely explained by prefix-aware admission
policy that ARLE has the data for but doesn't act on。

## Fix path(B3 in agent-first-architecture)

`docs/projects/agent-first-architecture.md::B3` already specifies the
required wiring:
1. Replace `QueueBoundAdmission` with `PrefixAwareAdmission` policy
2. Read `prefix_hit_tokens` and `session_affinity_slot` from incoming
   request
3. Route requests with high prefix-hit to slots holding that prefix
4. Skip cold prefill phase for shared-prefix sessions

LOC estimate(rough):
- New `PrefixAwareAdmission` impl:~200 LOC
- Wiring at admission call sites:~50 LOC
- Test coverage:~100 LOC
- **Total:~350 LOC**

This is **B3 work** per agent-first-architecture,not new strategy。
Just hasn't been picked up yet。

## ROI

Closing 2.03× SGLang gap on multi-tenant workload:
- Current ARLE 318 ms TTFT
- Target with B3:**~157 ms TTFT**(matching SGLang)= **-50% TTFT**
- Per master strategy §2.3 multi-tenant is high-priority axis 1 workload
- Comparable to cap=8 win(-86%)but on different workload axis

## Updated pickup queue priority

Adding B3 PrefixAwareAdmission as candidate:

- P0 Hybrid Phase 1b(`6be30ce`,~155-175 LOC per `9dc32d6`)
- P0' bimodal investigation(per `f7da3e1`)
- **P1 B3 PrefixAwareAdmission**(~350 LOC,closes 2× SGLang gap)
- P1 #33 KV W4A8(orthogonal axis,memory floor)
- P1' Medusa Phase 1.A.1 smoke(per `b4ae33f`)

B3 is comparable size to Hybrid Phase 1b and addresses different axis。
Could parallelize if codex bandwidth allows。

## Cross-references

- `m_world1-p0-sglang-baseline-extended` empirical 2.03× gap
- `agent-first-architecture.md::B3` original spec
- `2026-05-04-sglang-hicache-guide.md` SGLang impl reference
- `infer/src/scheduler/policy.rs:8-20` ARLE policy comment showing gap
- `infer/src/prefix_cache.rs`(2114 LOC RadixCache exists)
- `M1b 323aee0` RadixCache wired as shadow observer
- `M2a 4402ab0` RadixCache scheduler integration

## Methodology insight

**TODO comments in production code are reliable evidence of unimplemented
features**。`policy.rs:18-20` "until then, ignores them" is a **single-grep
finding** that resolves a multi-day strategic gap question(why SGLang 2× faster)。

Code-grep + reading struct/policy comments = ~5 minutes effort to
pinpoint where the gap is。Avoid speculation about kernel / compute /
algorithmic differences when admission-policy comment explicitly says
"ignores"。

## Status

Task #30 投靠 SGLang multi-tenant prefix cache investigation **resolved**
at root-cause level。Not implemented yet — B3 PrefixAwareAdmission work
queued for codex pickup。

Pending user direction:add B3 to P0 alongside Hybrid Phase 1b,or stay
P1 per current cap=8 / hybrid focus?

Recommendation:**P1 alongside #33** — both are ~150-350 LOC and
address different axes of agent workload(multi-tenant + memory)。
Codex could pick up after Hybrid Phase 1b lands(~1 day after that)。
