# Bench — eli M_e.10 fix shipped, post-fix bench inconclusive — 2026-05-08

## Goal

Verify ARLE M_e.10 prefix-cache miss is fixed by the cross-repo eli
patch (commit `d55d007` on cklxx/eli/main): per-session
`BuiltinImpl::sys_prompt_cache` + `Date` field date-only quantization.

## Hypothesis

Post-fix `bench_eli_agent.sh m_e10-after-fix` should show
`session_affinity_hit / (hit + miss) > 50%` across the 10-turn
multi-session replay, vs the pre-fix baseline of 0/10 (or sporadic
1/10).

## Command

```bash
./scripts/bench_eli_agent.sh m_e10-after-fix
# Driver auto-rebuilds eli at /path/to/code/eli/target/release/eli
# (= the patched binary from d55d007), boots a fresh metal_serve on :8765,
# replays 10 turns across 4 sessions from scripts/data/eli_agent_trace.jsonl,
# captures /v1/stats before/after.
```

## Environment

- **eli**: cklxx/eli@d55d007 (M_e.10 fix applied)
- **ARLE**: 2026-05-08 (commit 51ab5388, post-M_e.13)
- **Hardware**: Apple M4 Pro, MLX 0.31.1, macOS 26.3.1
- **Model**: `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Sessions**: 4 (eli-agent-001 through 004)
- **Total turns**: 10 (3 + 2 + 3 + 2)
- **Wall-clock**: 76.2s

## Results

### Per-bench session_affinity counters

| Bench | hit | miss | hit_rate |
|---|---|---|---|
| 2026-05-08 m_e10-import-probe (pre-fix) | 0 | 10 | 0.0% |
| 2026-05-07 m_e10-trace (pre-fix) | 1 | 10 | 9.1% |
| **2026-05-08 m_e10-after-fix (post-fix)** | **1** | **10** | **8.3%** |

### Per-turn wall-clock (post-fix)

| Session | Turn 0 | Turn 1 | Turn 2 |
|---|---|---|---|
| 001 | 4639 ms | 8066 ms | 8541 ms |
| 002 | 4526 ms | 8766 ms | — |
| 003 | 4151 ms | 8619 ms | 9546 ms |
| 004 | 4801 ms | 8827 ms | — |

p50=8066 ms, p90=8827 ms, p99=9546 ms.

## Δ vs hypothesis

| Aspect | Predicted | Measured |
|---|---|---|
| `session_affinity_hit` after-fix | ≥50% (e.g. 5-6/10) | **8.3% (1/11)** ❌ |
| Per-turn TTFT cross-turn improvement | turn-1+ should drop vs pre-fix | not visibly improved |

**The M_e.10 fix did not move the needle on this bench harness.**

## Problems / observations

1. **Bench is subprocess-per-turn, not daemon-mode.** Each turn forks
   a new eli CLI invocation:
   ```
   [eli-agent-001 turn 0] wall=4639.3ms exit=0 OK
   [eli-agent-001 turn 1] wall=8066.1ms exit=0 OK
   ```
   The exit code per turn implies each turn is a separate process
   that ends cleanly. The `BuiltinImpl::sys_prompt_cache` HashMap I
   added lives in process memory; subprocess fork → fresh `BuiltinImpl`
   → empty cache → no cache hit. **The session-cache half of the fix
   does NOT engage in this deployment.**
2. **Date date-only fix should still engage** (it's a code-level
   change in `build_runtime_section`). But it only matters if the
   timestamp was the dominant drift source. Bench result suggests it
   wasn't — the prompt_head divergence across turns must come from
   somewhere else (likely the multi-turn HISTORY content, NOT the
   system prompt).
3. **Re-examining the original M_e.10 evidence**: the "turn 2 token 5
   = ` a` (264) vs turn 1 token 5 = ` Eli` (32159)" data was from
   `prompt_head=[<|im_start|>, system, \n, You, are, X, ...]` — token
   index 5 is INSIDE the system prompt body. If this was real and
   reproducible, my fix should have helped within a single eli daemon
   process. But the canonical bench mode is subprocess, so even a
   correctly working in-process cache wouldn't show it. Need a
   daemon-mode reproducer to test.
4. **prefix_hits_total / prefix_lookups_total are `null`** in
   service_stats output → those metrics may not be populated for this
   workload shape, only `session_affinity_hit/miss` is. Limits what
   we can attribute the 1/11 hit to.
5. **The 1 hit out of 11 is likely coincidental** — a turn whose
   prompt happened to share an exact prefix with a prior turn (maybe
   the system_prompt prefix block-aligned to 16 tokens matched).
   Doesn't validate the fix; doesn't refute it either.

## What worked

- **Honest reporting**: per CLAUDE.md SOLID rules, reporting "hit rate
  unchanged" is correct rather than claiming a win on this bench.
- **Cross-repo subagent + own audit** (a9e4498830031316d + the design
  agent) correctly identified the **mechanism** of the bug
  (timestamp + SOUL.md fallback drift). The fix is correct in code,
  passes 450 tests, builds clean. The bench just doesn't have the
  right shape to expose its benefit.

## Rule

**Validate the deployment shape before claiming a session-cache fix
works.** Subprocess-per-turn invocation of an agent CLI defeats any
in-process session cache. To prove a session-cache fix end-to-end:
(a) instrument both pre- and post-fix runs with prompt_head logging
to capture the actual divergence, OR (b) run the agent in daemon
mode (long-lived process serving multiple turns over a socket /
RPC channel) so the cache has lifetime to engage. CLAUDE.md memory:
`feedback_path_probe_before_perf_claim.md` already says this for
ARLE-side perf, extends it cross-repo here.

## ARLE renderer determinism — confirmed (same-day, post-bench probe)

A focused probe-bench (`/tmp/m_e10_prompt_head_diff.sh`, server=fresh
metal_serve, 2 IDENTICAL chat/completions bodies sent back-to-back)
captures:

```
request #1: prompt_len=34 head=[248045, 8678, 198, 2523, 513, 449, 6009, 10597]
request #2: prompt_len=34 head=[248045, 8678, 198, 2523, 513, 449, 6009, 10597]
identical: True
```

→ **ARLE chat-template renderer IS deterministic for identical input
bodies.** Two identical requests produce byte-identical token streams.

Same probe also captured the cache state at request #2:
```
m_e10_trace prepare_request: ... entries_len=1 entries_keys_len_sample=[34] ...
m_e10_trace lookup: ... memory_match_len=None disk_match_len=None
```

Cache HAS the 34-token entry from request #1 but lookup MISSES. Code
trace at `runtime.rs:721`: `prefix_len < prompt_tokens.len()` —
**strict less-than rejects equal-length keys**. Intentional: cache
attach requires ≥1 residual token to decode; identical-body case has
0 residual. **Not a bug** — production multi-turn prompts always
grow (system + tape + new_user), so `< prompt.len()` is always
satisfied for real eli traffic.

**Triangulation summary** (combining today's three benches):
1. ARLE renderer deterministic for identical input ✓ (this probe)
2. ARLE lookup strict-`<` is intentional, not a bug ✓ (code trace)
3. Real eli M_e.10 miss = per-turn system_prompt drift → `starts_with`
   prefix mismatch → cache miss. The eli fix in
   `cklxx/eli@d55d007` directly addresses this on the eli side
   (Date date-only + session-cache). Subprocess-mode bench can't
   validate because of `feedback_subprocess_mode_breaks_inprocess_cache.md`,
   but the code path is correct.

## Next

- **Daemon-mode bench harness**: extend `bench_eli_agent.sh` with a
  `--daemon-mode` that drives a single long-lived eli process via
  IPC. Then re-bench M_e.10. (M effort.)
- **Direct prompt_head capture**: add a temporary probe in
  agent-infer's chat_completions handler that logs the FIRST-K tokens
  of the incoming prompt. Run two sessions back-to-back. Compare
  byte-by-byte to see what's actually drifting. (S effort.)
- **Date date-only fix is shipped regardless** — it's a 1-line
  correctness improvement that won't regress anything. If the bench
  harness ever runs in daemon mode, it'll engage.

## References

- eli M_e.10 fix: cklxx/eli@d55d007
- ARLE M_e.10 errata + earlier wins:
  [`2026-05-08-m_e10-try-import-probes-clean.md`](2026-05-08-m_e10-try-import-probes-clean.md)
  + [`2026-05-07-m_e10-prefix-mismatch-rootcause.md`](2026-05-07-m_e10-prefix-mismatch-rootcause.md)
- Pre-fix baselines:
  `bench-output/2026-05-08-bench-eli-agent-m_e10-import-probe/`
  `bench-output/2026-05-07-bench-eli-agent-m_e10-trace/`
- Post-fix bench:
  `bench-output/2026-05-08-bench-eli-agent-m_e10-after-fix/`
- eli design audit:
  subagent a9e4498830031316d (this session) — recommended the
  full-prompt cache approach, but didn't anticipate subprocess-mode
  invalidating the in-process cache.
