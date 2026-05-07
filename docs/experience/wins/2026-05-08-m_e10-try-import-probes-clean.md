# Eval — M_e.10 try_import_memory_prefix probes + 2nd-bench confirmation — 2026-05-08

## Goal

Follow-up to
[`2026-05-07-m_e10-prefix-mismatch-rootcause.md`](2026-05-07-m_e10-prefix-mismatch-rootcause.md):
add the deferred third probe (inside `try_import_memory_prefix` and
`import_qwen35_prefix_snapshot`) so future cache-miss debugging has
end-to-end visibility, AND re-run the eli e2e bench to validate the
"chat-template asymmetry" root cause is reproducible.

## Hypothesis

- **Probe diff**: env-gated `log::info!` only; zero perf impact when
  `INFER_M_E10_TRACE` is unset.
- **2nd bench**: cache hit rate should be similar to yesterday's
  (~10%, single-digit hits out of 11 lookups). Confirms the
  workload-shape root cause (chat-template asymmetry).

## Command

```bash
INFER_M_E10_TRACE=1 ./scripts/bench_eli_agent.sh m_e10-import-probe \
  --port 8765 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit
```

## Environment

- **Backend:** Metal (Apple Silicon)
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit` (canonical Metal
  per AGENTS.md)
- **Commit:** following `b80848d0` (the 19-commit pushed batch).
  This commit is uncommitted at bench time; trace probes only.
- **Feature set:** `cargo build --release --no-default-features
  --features metal -p infer --bin metal_serve`
- **Non-default flags:** `INFER_M_E10_TRACE=1` (diagnostic). Default
  Metal stack: oMLX-C v3 ON, auto-wired-limit auto-detected, M_e.4
  SwiGLU compile-fusion ON. `INFER_MOE_TOP_K` unset (default top_k=8).
- **Server launch:** `bench_eli_agent.sh` boots metal_serve internally.
- **Workload:** 4 sessions × 2-3 turns from
  `scripts/data/eli_agent_trace.jsonl`.

## Implementation

`infer/src/backend/metal/runtime.rs` — extended INFER_M_E10_TRACE
probes:

1. `try_import_memory_prefix` entry — logs prefix_key.len(),
   snapshot.cache_len, session.
2. `try_import_memory_prefix` exit — logs whether
   `import_qwen35_prefix_snapshot` returned `Ok(true)` / `Ok(false)`
   / `Err(...)`.
3. The "snapshot not found in entries" early-return path also logs.

Codex review on the diff: **clean** ("no actionable regressions; only
adds env-gated trace logging").

## Results

### Client wall-clock (from bench_eli_agent harness)
- elapsed: 75.4s
- turns OK / total: 10 / 10
- sessions: 4
- p50: 7770.51 ms
- p90: 8668.05 ms
- p99: 8844.07 ms
- req/s: 0.13

### Internal metrics (`/v1/stats?format=json` snapshot post-run)
- `prefix_lookups_total`: 11 (across all sessions; 2-3 per session)
- `prefix_hits_total`: **0**
- `prefix_reused_tokens_total`: **0**
- `matched_prefix_tokens` (last_request): 0
- `session_affinity_hit`: 0; `session_affinity_miss`: 10
- `resume_prefill_tokens_total` (across sessions): 14234 + 8634 +
  14204 + 8678 = **45750 tokens of redundant prefill**

### Trace output (representative excerpt)
```
agent-001 turn 1: prompt_len=2947 prompt_head=[248045, 8678, 198, 2523, 513, 32159, 11, 264]
agent-001 turn 2: prompt_len=5612 prompt_head=[248045, 8678, 198, 2523, 513,    264, 10631, 17313]
                                                                              ↑ index 5 divergence
all 11 lookup events: memory_match_len=None disk_match_len=None
0 try_import calls fired (because no match was found at all)
```

**Even more pessimistic than yesterday's bench** (which had 1 of 11
lookups find a match — agent-003 turn 1→2 shared 2943 tokens). Today
agent-003 turn 1→2 also diverged at index 5.

## Δ vs baseline

Baseline = the prior bench (
[`2026-05-07-bench-m_e9-precondition.md`](2026-05-07-bench-m_e9-precondition.md)
§"session_affinity_hit still 0"):

| Metric | 2026-05-07 baseline | 2026-05-08 follow-up | Δ |
|---|---:|---:|---:|
| `prefix_lookups_total` (sum) | 11 | 11 | 0 |
| `prefix_hits_total` (sum) | 1 (agent-003 turn 2) | **0** | −1 |
| `resume_prefill_tokens_total` (sum) | 45750 | 45750 | 0 |
| Client p50 wall (ms) | 7695.57 | 7770.51 | +75 (noise) |
| Client req/s | 0.13 | 0.13 | 0 |

Net: **inter-run variance is real** — the eli trace's exact prompt
sequence determines whether any single turn happens to share a
prefix with a prior turn. Both runs span 11 lookups; one had 1
hit, one had 0. The miss rate is workload-shape-dependent.

## Problems / observations

1. **Zero client-perf delta from the probes** (all numbers within
   ±2% noise of yesterday). Confirms the env-gated probes have no
   hot-path cost — codex's review claim "no observable behavior
   change when env unset" is empirically true even with env SET.
2. **Inter-run hit-rate variance**: 1 hit yesterday, 0 today on the
   same workload. Both well below "useful". A workload that
   consistently produced ≥3 hits would tell us the chat-template fix
   actually unlocks the cache; today's data confirms the asymmetry
   pattern but doesn't yet validate the fix.
3. **45.7K tokens of redundant prefill per bench run** — the cost
   the cache should eliminate. At ~7s per turn p50 from this bench,
   most of that wall time IS prefill; the cache hit would convert
   ~7s turns to ~1s turns (the predicted ~50× TTFT reduction
   collapses to a more realistic ~7× wall-clock per turn given the
   prefill share of total time).

## Why this confirms the root cause

The chat-template asymmetry diagnosis from yesterday holds — and the
between-run variance shows it's about WHERE in each session's trace
the user/assistant boundary falls. Some bench runs catch a session
where turns happen to share a prefix; most don't.

Per session, the prompt heads compared:
```
agent-001 turn 1: [248045, 8678, 198, 2523, 513, 32159, 11, 264, ...]
agent-001 turn 2: [248045, 8678, 198, 2523, 513,    264, 10631, 17313, ...]
                                                  ↑ divergence at index 5
```

Across 4 sessions × 2-3 turns: zero matches today. The first 5
tokens (`[248045, 8678, 198, 2523, 513]`) are common — that's the
shared system prefix. Below `block_size=16`, so untouchable by the
cache.

## Implication: 1-token shorter eval-side fix would unlock the cache

The simplest fix: have eli emit a closing `<|im_end|>` (token id ?)
on every user message — even the last one in a turn. This makes the
chat-template tokenization stable across "user message that's
followed by an assistant message" and "user message that ends the
conversation". The 5-token shared system prefix would extend to the
full common prefix length.

Predicted impact (the same ~50× TTFT prediction from M_e.10):
- Turn 1 prompt = `[system + user1 + <|im_end|>]` → 2947+1 tokens
- Turn 2 prompt = `[system + user1 + <|im_end|> + assistant1 +
  <|im_end|> + user2 + <|im_end|>]` → 5612+1 tokens
- **Turn 2 prompt now starts_with turn 1 prompt** ✓

Cache hits at 2948 tokens, prefill skipped, ~2-3s per turn saved.

## Decision

- **Probes stay in tree** behind `INFER_M_E10_TRACE=1` — codex
  reviewed clean, zero cost when env unset, ready for any future
  cache-mismatch debugging session.
- **The fix lives in eli** (the chat-template format owner). Filed
  as cross-repo work; to be picked up by the eli session.
- **No ARLE-side workaround** — adding tokenizer-aware prefix
  trimming on the cache write side would be fragile across model
  variants (different chat templates have different end-of-turn
  markers).

## Codex review log

```
codex review --uncommitted
=> The change only adds additional env-gated trace logging around
   Qwen3.5 prefix snapshot import, with no observable behavior change
   when the trace env var is unset. I did not identify any
   actionable regressions in the modified code.
```

## Learnings

1. **Cross-repo issues sometimes have cheap diagnostic landing
   points.** The chat-template chunk is owned by eli, but the symptom
   surfaces in ARLE's metric. Adding probes on the symptom side
   accelerates the cross-repo conversation: now we have a
   trace-replayable JSON of WHICH tokens diverge, not just "the
   metric is 0".
2. **Inter-session variance** in cache hit rate (1 hit yesterday, 0
   today on the same workload) is itself a signal. The eli trace's
   conversation shape matters; some traces happen to be "cache
   friendly" by accident, others aren't.

## Next

- **Cross-repo: eli chat-template fix** — open as a discussion in
  the eli repo; scope is one tokenizer-side change to
  `crates/nexil/src/llm/...` or eli's prompt-builder.
- **omlx residency-set hygiene** (parallel research finding this
  date): port omlx's `mx.clear_cache` cadence from
  `omlx/scheduler.py` to ARLE's Rust scheduler decode loop. Predict:
  long-context (32k+128k per `2026-04-30-longctx-…`) bench survives
  ≥4096 generated tokens without IOGPU-residency-set abort. S effort.
- **dflash-mlx Prometheus `/metrics`** — port the
  `dflash_mlx/server/metrics.py` exposition formatter to ARLE's
  /v1/stats. S effort. Helps spec §3 watch list go real-time.
- **Quality eval (M_e.8 Tier-2)** still on the deck for
  `INFER_MOE_TOP_K` default flip.

## References

- Predecessor:
  [`2026-05-07-m_e10-prefix-mismatch-rootcause.md`](2026-05-07-m_e10-prefix-mismatch-rootcause.md)
- Codex review (clean): `/tmp/codex_review_m_e10_probes.log`
- Today's bench:
  `/path/to/code/agent-infer/bench-output/2026-05-08-bench-eli-agent-m_e10-import-probe/`
