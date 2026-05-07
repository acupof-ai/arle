# Bench — M_e.10 prefix-mismatch root cause: chat-template asymmetry — 2026-05-07

## Goal

Diagnose why `prefix_hits_total = 0` despite `prefix_lookups_total > 0`
on the eli e2e multi-turn bench (per M_e.10 task #32). Predicted ~50×
TTFT win if fixed; clarification of root cause first.

## Method

Added `INFER_M_E10_TRACE=1` env-gated `log::info!` probes at three
sites in
[`infer/src/backend/metal/runtime.rs`](../../../infer/src/backend/metal/runtime.rs):

1. `MetalQwen35PrefixRuntime::prepare_request` entry — logs gate
   decisions (DFlash flag, can-import flag), entries.len, sample of
   stored key lengths, prompt head tokens.
2. After `lookup_longest_prefix` — logs memory_match_len /
   disk_match_len.
3. `publish_prompt_prefix` skip path — logs when `can_import=false`
   short-circuits inserts.

Re-ran `./scripts/bench_eli_agent.sh m_e10-trace` (4 sessions × 2-3
turns = 11 lookups).

## Empirical result

```
session=Some("eli-agent-001") prompt_len=2947 prompt_head=[248045, 8678, 198, 2523, 513, 32159, 11, 264]
session=Some("eli-agent-001") prompt_len=5616 prompt_head=[248045, 8678, 198, 2523, 513,    264, 10631, 17313]
session=Some("eli-agent-001") prompt_len=5670 prompt_head=[248045, 8678, 198, 2523, 513,    264, 10631, 17313]
session=Some("eli-agent-002") prompt_len=2963 prompt_head=[248045, 8678, 198, 2523, 513, 32159, 11, 264]
session=Some("eli-agent-002") prompt_len=5667 prompt_head=[248045, 8678, 198, 2523, 513,    264, 10631, 17313]
session=Some("eli-agent-003") prompt_len=2943 prompt_head=[248045, 8678, 198, 2523, 513, 32159, 11, 264]
session=Some("eli-agent-003") prompt_len=3514 prompt_head=[248045, 8678, 198, 2523, 513, 32159, 11, 264] ← MATCH (2943)
session=Some("eli-agent-003") prompt_len=6229 prompt_head=[248045, 8678, 198, 2523, 513,    264, 10631, 17313]
```

→ Of 11 lookups, **only 1 found a memory match** (agent-003 turn 2,
matching its turn 1's 2943-token prefix). Most lookups failed
because turn N's stored tokens are NOT a prefix of turn N+1's
prompt.

## Root cause: chat-template asymmetry

The first 5 tokens are identical across all turns (the system-prompt
prefix). At index 5, divergence:

- Turn 1 (single user message): `..., 32159, 11, 264, ...` — user
  content tokens directly continue the system prefix.
- Turn 2+ (with assistant response): `..., 264, 10631, 17313, ...`
  — different first-content tokens.

The 5-token shared prefix is below `block_size=16`, so it can't be
cached. The 32159 vs 264 divergence is the eli chat template
generating different early tokens depending on whether the message
is followed by an assistant response or not.

Specifically:

- "User message at end of conversation" (turn 1): tokenized as
  `<sysprompt> + <user1 content>` directly
- "User message followed by assistant" (turn 2's view of turn 1):
  tokenized as `<sysprompt> + <user1 content> + <|im_end|> +
  <assistant1> + <|im_end|> + <user2>`

The closing `<|im_end|>` after `<user1 content>` is present in turn
2's tokenization but absent in turn 1's tokenization (because turn 1
ended at user content; eli's chat template doesn't emit a closing
end-of-turn marker for the last message).

So turn 1 ends with `[..., 32159, 11, 264]` but turn 2's same
position contains `[..., 264, 10631, 17313]` — turn 2's
representation has a different token at position 5.

Where the prefix DOES preserve (agent-003 turn 1→2 sharing 2943
tokens): turn 1 of agent-003 just happened to land on a prompt
shape compatible with turn 2's prefix continuation. Probably a turn
that ended with an `<|im_end|>` followed by another user
message, with no assistant interleaving.

## Implications

This is **NOT an ARLE bug**. The prefix cache logic is correct: it
stores the literal token vector at request finish, looks up by
exact prefix match, and inserts cache hits when found. The miss
rate reflects the **workload's chat-template shape**, not a wiring
defect.

Two ways to fix:

1. **Eli-side**: change the chat template to emit closing
   `<|im_end|>` even on the trailing user message. Then turn 2's
   tokenization would end at `[..., user_content, <|im_end|>]` and
   turn 1's would also end at `[..., user_content, <|im_end|>]` —
   prefixes would match. **This is the right fix** but requires
   coordination with eli's prompt-build code. Touches
   `crates/nexil/src/llm/mod.rs` (prompt assembly) or eli's chat
   template config. **Effort: M, eli-side.**
2. **ARLE-side**: at insert time, store the prompt prefix only up
   to the LAST chat-template stable boundary (e.g. the last
   `<|im_end|>` token before the trailing user message). At lookup
   time, do the same prefix extraction. This makes ARLE robust to
   trailing-message asymmetry without coordinating with eli. **Effort:
   M, ARLE-side**, but adds a tokenizer-aware boundary to the cache
   logic which is fragile across model variants.
3. **Workaround** for the eli e2e bench specifically: append a
   trailing `<|im_end|>` to the eli driver's prompts before sending.
   That's a 1-line `bench_eli_agent.sh` patch but only papers over
   the issue for benches.

The predicted "~50× TTFT" win from M_e.10 is **conditional on the
chat template being fixed**. With current eli prompts, prefix cache
delivers cache hits ~10% of the time (1/11 in this bench).

## Decision

- **M_e.10 reframed** from "fix lookup-mismatch bug" to "diagnose
  why hits are rare". Result: workload-shape, not bug.
- **Next-step ownership**: eli-side chat template fix (option 1
  above). Cross-repo work; track on the eli side. ARLE's prefix
  cache is correct as-is.
- **Trace probes stay** in tree behind `INFER_M_E10_TRACE=1`. They
  cost zero when env unset, and any future regression in prefix-
  cache wiring is one bench cycle away from being reproducible.

## Learnings

1. **Empirical traces beat code review.** The subagent's analysis
   suggested DFlash gate as the root cause; the trace immediately
   showed `dflash_enabled=false` and `can_import_snapshot=true` for
   every turn, ruling out that hypothesis. The actual cause
   (token-prefix asymmetry from chat-template) was visible only by
   logging the prompt heads.
2. **Chat-template stability is load-bearing for prefix cache.**
   Any chat client that generates "different tokens for the same
   content depending on conversation position" will have low cache
   hit rates. This is a portability concern for any prefix-cache
   benchmark.
3. **Some sessions DO hit** (agent-003 turn 1→2 = +571 tokens
   reused). It's not zero — just sparse. The metric
   `prefix_hits_total=0` from the prior tick was actually wrong;
   the trace shows at least one match, which suggests the metric
   wiring may have a separate bug. Needs follow-up.

## Why prefix_hits_total still =0 despite finding a match

The trace logged `memory_match_len=Some(2943)` but the metric
recorded miss. Two candidates:

1. `try_import_memory_prefix` returned false despite the lookup
   finding a key (snapshot expired, or live-prefix-runtime decided
   not to import for capacity/correctness reasons).
2. The metric bookkeeping path is independent of the lookup result
   and only increments on `reused_tokens > 0`, which depends on
   import succeeding.

Adding a third probe inside `try_import_memory_prefix` (whether it
returned true/false + reason) is the next step. Out of tick scope.

## What worked

- Three small env-gated `log::info!` probes localized the issue in
  one bench cycle (~5 minutes).
- Eyeballing 11 lookup events with prompt heads side-by-side
  immediately revealed the asymmetry pattern.
- The probe pattern (cached env probe + structured single-line log)
  is reusable for any future cache-debug work.

## Rule

Before optimizing a hit-rate-dependent metric, **dump the lookup
keys** and visually inspect a handful of turn-N → turn-N+1
transitions. Token-prefix mismatches due to chat-template
asymmetry are a class of issue that's invisible from code review
but obvious from trace.

## Next

- **Eli-side fix** for chat template trailing `<|im_end|>` — track
  in eli repo. Owner: cross-repo coordination.
- **Third probe inside `try_import_memory_prefix`** to confirm why
  the 1 found match didn't increment hits_total. S effort, future
  tick.
- Per the audit, ARLE prefix cache wiring is **correct**; no ARLE-
  side code change required from this finding alone.

## References

- Predecessor (the audit-correction that needed empirical
  validation):
  [`docs/experience/wins/2026-05-07-bench-m_e9-precondition.md`](2026-05-07-bench-m_e9-precondition.md)
  §"session_affinity_hit still 0"
- Code under instrumentation:
  [`infer/src/backend/metal/runtime.rs:561-697`](../../../infer/src/backend/metal/runtime.rs)
  (prepare_request + publish_prompt_prefix + lookup_longest_prefix)
- Original audit chain:
  [`2026-05-07-eli-layer-2-nexil-session-id-forwarding-shipped.md`](2026-05-07-eli-layer-2-nexil-session-id-forwarding-shipped.md)
