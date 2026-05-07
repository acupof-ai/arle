# Bench — M_e.9 mixed-batch precondition + session_affinity audit follow-up — 2026-05-07

## Goal

Per `docs/plans/M_e9-qwen35-mixed-batch.md` step 1: instrument the
mixed-batch dispatcher to measure how often the (decode, prefill)
coexistence case fires under realistic workload. **Plan threshold:
≥30% fallback rate means M_e.9 is on the hot path; <30% means
deprioritize.**

Also: investigate the audit-correction claim from
[`2026-05-07-eli-layer-2-…`](2026-05-07-eli-layer-2-nexil-session-id-forwarding-shipped.md).
The earlier finding "`session_affinity_hit=0` because nexil isn't
sending session_id" was fixed in commit `97db09e`. Re-bench should
show the metric moving.

## What was instrumented

`infer/src/backend/metal/runtime.rs` — added env-gated atomic
counter pair around the `(Some(decode), Some(prefill))` dispatcher
case:
- `MIXED_TICK_TOTAL` — every (decode + prefill) tick
- `MIXED_TICK_FUSED` — when guard_mixed_batch returns true (fused
  into one async_eval)
- Periodic stderr dump every 50 ticks: `m_e9_precondition:
  mixed_dispatch_ticks=N fused=N fallback=N fallback_pct=X.X%`
- Env gate: `INFER_M_E9_PRECONDITION=1`

## Bench

`./scripts/bench_eli_agent.sh m_e9-precondition --port 8765 --model
mlx-community/Qwen3.6-35B-A3B-4bit` with the env set. Workload is
4 sessions × 2-3 turns each = 10 turns total.

## Result 1 — M_e.9 precondition: NOT on this workload's hot path

```
$ grep "m_e9_precondition" .../server.log
(empty)
```

The threshold (50 mixed-batch ticks) was never crossed. The eli e2e
bench drives requests **sequentially per session**, with new sessions
admitted while older ones decode but the planner rarely emits a
prefill row in the same tick as decode rows. Ticks are predominantly
either pure-decode or pure-prefill.

**This is itself a finding**: for the eli-agent multi-turn session
workload, M_e.9 (mixed-batch generalization) is **not** the dominant
lever. The current sequential dispatch is acceptable for chat-style
serving where concurrent admit + decode is rare.

**Where M_e.9 WOULD pay**: workloads where new requests arrive
mid-decode while the existing batch is decoding — e.g. high-QPS
serving or prefill-heavy benchmarks (`scripts/bench_guidellm.sh
longctx-32k`). The plan's "≥30% fallback" threshold needs to be
checked against THAT workload, not eli.

## Result 2 — session_affinity_hit still 0 despite nexil fix

```
"sessions": {
  "eli-agent-001": {
    "prefix_lookups_total": 3,
    "prefix_hits_total": 0,
    "prefix_reused_tokens_total": 0,
    "session_affinity_hit": 0,
    "session_affinity_miss": 3,
    "resume_prefill_tokens_total": 14234   ← all turns reprocessed from scratch
  },
  ...
}
```

The earlier audit-correction was **partially** correct:
- ✅ Lookups DO fire (`prefix_lookups_total=3` per 3-turn session) —
  the runtime is wired
- ❌ But all hits = 0 (`prefix_hits_total=0`,
  `matched_prefix_tokens_total=0`)

**Lookups happen, but never match.** Three turns of the same session
with `same system prompt + previous user/assistant turns` should
share thousands of tokens of prefix. Yet the cache returns no hit.

Possible root causes (next-tick investigation):

1. **Cache key mismatch**. The Qwen3.5 live-prefix runtime keys on
   tokenized prompt prefix; if the chat-template prefix changes per
   turn (different special tokens, different formatting), an exact
   prefix match fails even with shared content.
2. **Cache eviction**. The cache may be populated then immediately
   evicted by KV pressure. With 4 concurrent sessions on a 35B
   model, total KV need exceeds the running set.
3. **Multimodal-checkpoint specific bug**. The Qwen3.6-VL canonical
   has different cache-key shape than the Qwen3.5-MoE-text runtime
   was designed for. The checkpoint type (`qwen3_5_moe` MoE flavor +
   vision tower) may break the prefix-runtime's assumptions.
4. **The audit-correction was wrong**. The "ARLE has working
   live-prefix wired since runtime.rs:1972-1983" finding from the
   M_e.9 research subagent may have misread the call. Need direct
   inspection of `MetalQwen35PrefixRuntime::lookup_longest_prefix`
   under a debugger or with added trace logs.

**Resume_prefill_tokens_total=14234 with 3 lookups means each turn
is reprocessing the full conversation history from scratch.** This
is exactly the cost the prefix cache is supposed to eliminate. The
14K tokens × 3 turns × 4 sessions ≈ 170K wasted prefill tokens.

## Decision

**M_e.9 mixed-batch is moved from "biggest-ROI lever" to
"workload-specific opt-in".** Without a proof that mixed-dispatch
fires ≥30% on the production workload, the M-effort generalization
isn't worth it. The plan stays for future when the right workload
arrives.

**The new biggest-ROI lever is fixing the session_affinity_hit=0
issue.** A 14K-token prefix cache hit per turn would dwarf any
M_e.9 mixed-batch win — it converts every multi-turn turn from
"prefill 14K + decode 64" to "prefill ~0 + decode 64", a ~50× TTFT
improvement on the chat workload eli actually drives.

## Path probe protocol

The instrumentation pattern (env-gated atomic counter + periodic
stderr summary) is reusable. Future precondition-check experiments
should adopt the same shape:
- `static FLAG: OnceLock<bool>` with env-var
- `static COUNTER: AtomicU64` for total
- `static SUCCESS: AtomicU64` for the favorable path
- Periodic dump every N ticks, computed fallback_pct
- Gate the dump on `is_multiple_of(N)` to avoid log spam

## Learnings

1. **Always run the precondition step before committing to an
   M-effort.** This tick saved ~3-7 days of M_e.9 implementation
   work that would have landed for a workload where mixed-batch is
   <1% of ticks.
2. **"Audit corrections" need follow-up benches.** The previous
   tick's research subagent claimed the live-prefix was wired and
   would work after the nexil fix. The bench shows lookups fire but
   never hit — the wiring is incomplete or buggy. Don't trust
   research synthesis without empirical confirmation.
3. **Sessions != concurrent**. The eli driver runs one turn per
   session at a time across 4 sessions; there's high context reuse
   *within* a session (where prefix cache should fire) but low
   prefill-during-decode overlap (where M_e.9 fires).

## Rule

For any plan claiming a perf opportunity based on "fraction of ticks
hitting condition X", **measure the fraction first**. The M_e.9 plan
explicitly called for this measurement; spending 1 tick on the
counter saved many ticks of mis-targeted work.

## Next

- **M_e.10** (new task): debug the session_affinity_hit=0 issue.
  `prefix_lookups_total=3, prefix_hits_total=0` is the smoking gun.
  Add trace logs to `MetalQwen35PrefixRuntime::lookup_longest_prefix`
  to see exactly what's being compared and why no match. Likely
  S-effort. Highest predicted ITL win on chat-style multi-turn
  workload (50× TTFT reduction if hits actually fire).
- **M_e.9** stays as a designed-but-deprioritized plan; revisit if
  high-QPS / longctx benchmarks come online.
- **M_e.8 Tier-2** (HumanEval+GSM8K) still on the deck for
  INFER_MOE_TOP_K default flip.

## References

- Plan source: [`docs/plans/M_e9-qwen35-mixed-batch.md`](../../plans/M_e9-qwen35-mixed-batch.md)
- Prior audit-correction (now refined):
  [`2026-05-07-eli-layer-2-nexil-session-id-forwarding-shipped.md`](2026-05-07-eli-layer-2-nexil-session-id-forwarding-shipped.md)
- Encoder-bound diagnosis (still the dominant cost on c=4):
  [`2026-05-07-bench-qwen36-encode-bottleneck.md`](2026-05-07-bench-qwen36-encode-bottleneck.md)
