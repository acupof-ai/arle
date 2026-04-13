# 2026-04-13 · bench_agent_trace baseline — P1 scoreboard starting point

## Context
First real-server run of `scripts/bench_agent_trace.py` (commit `c531315`
feat D + `8adec3c` /v1/stats probe I). The replayer drives
`scripts/data/agent_trace_default.jsonl` (6 sessions, 14 user turns
total) against the infer HTTP server, threading `session_id` through
every request so the server can route turns of the same conversation to
the same slot/radix subtree.

This baseline is the **P1 scoreboard** — the exit gate for P1 is
"≥70 % cross-session prefix hit rate" (per
`docs/plans/tiered-kv-cache-tasks.md` §2). Tracking here will be
compared against the same file name (different body) after the P1 (b)
behavior PR lands and the scheduler actually consults RadixCache.

Paired artifact: `2026-04-13-bench-agent-trace-baseline.json` (raw
per-turn dump in the same directory).

## Environment
- GPU: NVIDIA L4 24GB (driver 580.82.07, CUDA 13.0)
- Model: Qwen3-4B BF16, HuggingFace `Qwen/Qwen3-4B` (Instruct variant)
- Commit: `876b986` (2026-04-13 local batch head)
- Server: `target/release/infer --model-path models/Qwen3-4B --num-slots 4 --port 8000`
  - `cuda_graph=true`, warm batch sizes {1,2,4}, `num_slots=4`
  - `kv_cache_dtype=bf16`
- Bench tool:
  ```
  python3 scripts/bench_agent_trace.py \
      --server http://localhost:8000 \
      --label baseline-main-2026-04-13 \
      --out docs/experience/wins/2026-04-13-bench-agent-trace-baseline.json
  ```
  - `--num-concurrent 4` (default, matches num_slots)
  - `--max-tokens 256` (default)
- Trace: `scripts/data/agent_trace_default.jsonl` (6 sessions, 14 user turns)

## Results

```
session          turn  msgs   wall(ms)  ttft(ms)   itl(ms)  tokens  finish
──────────────────────────────────────────────────────────────────────────
agent-001           0     2     5085.9      47.2      33.8     144  stop
agent-001           2     4     8113.3     158.7      34.1     229  stop
agent-001           4     6     8909.3     122.7      34.1     256  length
agent-002           0     2     8943.8     123.2      34.0     256  length
agent-002           2     4     2579.1     112.0      34.2      72  stop
agent-002           4     6     8909.8     207.0      34.1     256  length
agent-003           0     2     8942.4     164.4      34.0     256  length
agent-003           2     4     7848.0     111.8      34.1     224  stop
agent-004           0     2      307.6     205.5      33.9       3  stop
agent-004           2     4     8114.2     112.2      34.1     229  stop
agent-005           0     2     8900.4     109.7      33.9     256  length
agent-005           2     4      459.5     112.8      34.4      10  stop
agent-006           0     2     8932.7     106.1      34.1     256  length
agent-006           2     4     8876.0      49.2      34.1     256  length
──────────────────────────────────────────────────────────────────────────

turns OK:        14 / 14
tokens total:    2703
wall total (s):  94.92
TTFT p50/p99:    112.2 / 207.0 ms
ITL  p50/p99:    34.1 /  34.4 ms
```

## `/v1/stats` probe (commit I `8adec3c`)

```
before: requests=0 active=0 waiting=0 tokens_out=0 kv_util=0.0% ttft_p50=— ttft_p99=— tpot_p50=—
after:  requests=0 active=0 waiting=0 tokens_out=0 kv_util=0.0% ttft_p50=— ttft_p99=— tpot_p50=—
delta:  requests=+0 tokens_out=+0
note:   prefix_hit_rate not exposed by /v1/stats yet; server-side counter
        addition pending (see I1 research in docs/plans/tiered-kv-cache-tasks.md)
```

The `after` snapshot shows zeros even though 14 turns / 2703 tokens
streamed successfully — `/v1/stats` is wired to a counter bucket the
scheduler does not currently increment under the active runtime path.
Confirming the plan's note that `metrics.rs` needs a follow-up commit
to expose `prefix_hit_rate` and the other counters.

## Observations
- **All 14 turns succeeded.** 2703 total tokens across 6 sessions, no errors.
- **ITL is flat at 34 ms** — matches throughput sweep's C=1 ITL floor.
  Inter-token latency is decode-GEMV-bound and independent of prompt
  history length in this range.
- **TTFT p50 = 112 ms, p99 = 207 ms** — TTFT scales with prompt length
  but is dominated by the fresh prefill cost because the **scheduler
  does not yet reuse RadixCache state across turns of the same session**.
  Every turn re-prefills the full history from the KV pool's clean
  state. Evidence: `agent-001` turn 4 (6 messages of accumulated
  history) has 122 ms TTFT vs. turn 0's 47 ms with just 2 messages —
  cost scales ~linearly with history, exactly as cold prefill would.
- **P1 scoreboard starting point**: the `scheduler` line (not the
  `RadixCache` module) is what currently routes and evicts KV blocks.
  P1 (b) wires the prefix cache into the scheduler; once that lands,
  the same turns should hit the cache for the prefix span that
  matches the prior turn's conversation. Expected delta at that point:
  TTFT p50 drops to something close to ITL (~35–50 ms) for warm
  sessions.

## Rule
- Baseline for P1 exit gate ("≥70 % cross-session prefix hit rate"):
  14 / 14 turns succeed with **TTFT p50 = 112 ms** and
  **ITL p50 = 34 ms** at `--num-concurrent 4`.
- After P1 (b) lands, rerun with the same command, same trace,
  same `num_slots`; save to a new immutable file named
  `YYYY-MM-DD-bench-agent-trace-p1b.md`.
- `/v1/stats` currently reports zeros for everything during the bench
  run. Do not trust the probe for prefix-hit-rate measurements until
  the `metrics.rs` follow-up lands (tracked as I1 research).
