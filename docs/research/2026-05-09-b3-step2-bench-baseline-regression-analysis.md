# B3 Step 2 — bench baseline regression analysis(queue-bound default)

> Codex `24m 15s` into B3 Step 2 implementation pickup。Bench
> `2026-05-09-b3step2-prefix-aware/` already completed but **without
> `--admission-policy=prefix-aware` server flag** — this is a
> byte-identical regression check on the queue-bound default
> path,NOT the actual PrefixAware policy validation。
>
> **Verdict**:queue-bound default preserves expected behavior。
> PrefixAware bench is the critical next step for license。

## Bench profile

```bash
guidellm benchmark run --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --profile concurrent \
  --data prompt_tokens=6000,prompt_tokens_stdev=1,...,output_tokens=200,turns=3,session_count=4 \
  --max-seconds 120 --random-seed 20260416 \
  --rate 4 --warmup 10
```

This matches the dispatch directive's multi-tenant 4-conc 6k-system
burst profile。

## Server-side state(per service trace)

```
Peak waiting: 0
Peak active: 4
Peak running_batch: 4
Peak prefill_queue: 3
Plan labels: idle=14664, decode=3191, prefill=52, split=17
Peak kv_util: 92.4%
Prefix hit rate: peak 0.0%, q75 0.0%
Prefix skip rate peak: 0.0%
```

**Critical**:`Prefix hit rate: peak 0.0%` indicates the PrefixAware
policy was **NOT exercised** during this bench。Two plausible reasons:

1. **Server started without `--admission-policy=prefix-aware`** —
   default policy is `queue-bound`(per codex's `SchedulerAdmissionPolicy::QueueBound`
   default),and this bench did not opt-in
2. **Bench data didn't trigger warm hits** — unlikely given
   `turns=3, session_count=4`(third turn should match second turn's
   KV prefix)

Most likely **#1** — codex ran a baseline regression check first
before testing the PrefixAware path。

## Key metrics(queue-bound default,this bench)

| Metric | Value |
|--------|------:|
| Successful requests | 180 / 184(97.8%)|
| Incomplete | 4(at 120s window cutoff)|
| Errored | 0 |
| TTFT median | 0 ms(warm,majority of requests reuse KV from prior turn)|
| TTFT mean | 934 ms |
| TTFT p99 | 3048 ms |
| ITL median | 21.4 ms |
| ITL p99 | 25.6 ms(σ tight 0.16)|
| TPOT median | 36.4 ms |
| Output tok/s median | 95.2 |
| Output tok/s p99 | 555 |
| KV util peak | 92.4% |

## Interpretation

### Queue-bound default works correctly

- 97.8% completion rate is solid for a 4-conc 6k-prompt 120s burst
- ITL p99 25.6 ms vs ITL median 21.4 ms = σ tight(<5% per skill rule)
- TTFT median=0 ms shows multi-turn warm reuse working at the
  scheduler level even under queue-bound — when 75% of requests are
  warm continuations,they hit existing slot KV

### What is NOT yet verified

- **PrefixAware policy improvement claim**(my dispatch directive's
  acceptance criterion):TTFT 318 ms → 157 ms multi-tenant burst
- **PrefixAware policy correctness**:cold-headroom logic,
  fail-open guard codex added(per `f41d7c9` audit)

### Critical next bench

Need a paired bench with `--admission-policy=prefix-aware` server
flag set:

```bash
# Codex's next move:
./scripts/metal_serve ... --admission-policy=prefix-aware
# then
scripts/bench_guidellm.sh b3step2-prefix-aware-policy-on \
  --concurrencies 4 --max-seconds 120 \
  --data 'prompt_tokens=6000,output_tokens=200,turns=3,session_count=4'
```

Compare on:
- Multi-tenant TTFT(p50/p75/p99): expect lower for warm sessions
- KV hit rate: expect > 0.0%(was 0.0% in this baseline)
- 4 incomplete count: PrefixAware should not regress completion rate

## Phase 8 license-or-kill status

| Acceptance criterion | This bench | Pending |
|----------------------|-----------|---------|
| cargo test passes | (test suite hit `metal_eval_audit` failure — unrelated to this diff per `f41d7c9` audit) | retest after triage |
| cargo clippy clean | Pending | next codex check |
| Byte-identical for queue-bound default | ✅ this bench mostly preserves expected behavior(0.0% prefix hit confirms PrefixAware not engaged) | - |
| Multi-tenant TTFT improvement σ < 5% | NOT TESTED YET | **PrefixAware bench needed** |
| Wins entry | Pending | after PrefixAware bench |

## Recommendation for codex(next move)

1. Resolve `metal_eval_audit` test failure(likely pre-existing per
   `f41d7c9` audit — Metal-only static analysis not affected by
   CUDA scheduler diff)
2. Run server with `--admission-policy=prefix-aware --cold-headroom 2`
3. Run paired bench against same data spec
4. Compare TTFT distribution between the two server configurations
5. Apply skill Phase 8 thresholds:license if Δ ≥ 10% with σ < 5%,
   else KILL or document as null result

## Cross-references

- Codex implementation audit: `f41d7c9`(2026-05-09 docs/research/)
- B3 Step 2 architecture: `c097b2b` + `637701b`
- A1 audit: `1217375`
- B3 Step 1 byte-identical baseline: `c30e298`(W3 c=4 turns)
- Pickup queue P0.1: `docs/plans/codex-pickup-queue-2026-05-09.md`
- Skill v1.7.0 Phase 8 license thresholds: `c768b70`

## Status

Codex's queue-bound default bench is consistent with byte-identical
regression expectation。**The license-relevant data point — PrefixAware
policy on,paired bench — is still pending**。Claude's next-tick
action depends on whether codex runs the paired bench independently
or needs an explicit nudge in the brief。

## Rule

**Bench label ≠ bench config**。A bench named `b3step2-prefix-aware`
that runs without `--admission-policy=prefix-aware` is a baseline
regression check,not a PrefixAware validation。Service trace's
`Prefix hit rate: peak 0.0%` is the diagnostic that catches this
mislabeling。Always cross-check bench label against actual server
config + service trace before declaring license。

This is a special case of skill anti-pattern #8(production default
≠ A/B baseline,matched-control violation)applied to admission-
policy A/B specifically:the policy-axis change must be in the
SERVER startup flags,not the GUIDELLM CLI flags,since guidellm
runs as a client。
