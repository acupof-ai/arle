# ARLE W4 Tool Resume - agent-load benchmark, agent-w4-tool-resume, arle-cuda, 2026-05-02

> Workload contract:
> [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../../plans/2026-05-02-agent-load-bench-spec.md).

## Goal

- Diagnosis: measure current `main` on the ARLE-only W4 tool-resume trace, with
  no A3 code and no competitor server, to decide whether W4 is already a
  winnable battlefield from A1/A2 alone.

## Hypothesis

- If current resident-session resume is healthy, W4 should show high
  matched-prefix depth, a high avoided-prefill ratio, and resume TTFT well below
  the 8k warmup/cold proxy. If it only reports shallow prefix hits, A3 or a
  deeper resume-prefix path is the blocker.

## Command

Trace generation:

```bash
rm -rf bench-output/2026-05-02-agent-load-arle-w4-tool-resume
mkdir -p bench-output/2026-05-02-agent-load-arle-w4-tool-resume
python3 scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --generate-only \
  --trace-out bench-output/2026-05-02-agent-load-arle-w4-tool-resume/trace.jsonl
```

Server:

```bash
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
./target/release/infer \
  --model-path models/default \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 12288 \
  --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85 \
  2>&1 | tee bench-output/2026-05-02-agent-load-arle-w4-tool-resume/server.log
```

Client:

```bash
TRACE_FILE=bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_trace.jsonl
: > "$TRACE_FILE"
(
  while true; do
    printf '{"ts":"%s","stats":' "$(date -Iseconds)" >> "$TRACE_FILE"
    curl -sS 'http://127.0.0.1:8000/v1/stats?format=json' >> "$TRACE_FILE" || printf 'null' >> "$TRACE_FILE"
    printf '}\n' >> "$TRACE_FILE"
    sleep 5
  done
) &
TRACE_PID=$!
START_MS=$(date +%s%3N)
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --server http://127.0.0.1:8000 \
  --label arle-w4-tool-resume \
  --out bench-output/2026-05-02-agent-load-arle-w4-tool-resume/results.json \
  --trace-out bench-output/2026-05-02-agent-load-arle-w4-tool-resume/trace.jsonl \
  2>&1 | tee bench-output/2026-05-02-agent-load-arle-w4-tool-resume/client.log
STATUS=${PIPESTATUS[0]}
END_MS=$(date +%s%3N)
kill "$TRACE_PID" 2>/dev/null || true
wait "$TRACE_PID" 2>/dev/null || true
echo "ELAPSED_MS=$((END_MS - START_MS))"
curl -sS http://127.0.0.1:8000/v1/stats \
  -o bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_after.txt
curl -sS 'http://127.0.0.1:8000/v1/stats?format=json' \
  -o bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_after.json
exit "$STATUS"
```

## Environment

- **Workload:** `agent-w4-tool-resume`
- **Backend / engine:** `arle-cuda`
- **Model:** Qwen3-4B Instruct, `models/default -> models/Qwen3-4B`
- **Tokenizer / processor:** `models/default -> models/Qwen3-4B`
- **Hardware:** NVIDIA L4, 23,034 MiB, driver 580.82.07, CUDA nvcc 12.8.93
- **Commit:** `c6a43717`, clean tree before run
- **Feature set:** existing `target/release/infer` CUDA release binary
- **KV dtype / cache mode:** FP8E4M3 paged KV, RadixCache on
- **Session / prefix flags:** OpenAI-compatible `session_id` on every request
- **Non-default flags / env vars:** `--num-slots 8`, `--max-seq-len 12288`,
  `--kv-cache-dtype fp8`, `--mem-fraction-static 0.85`,
  `PEGAINFER_CUDA_SM=89`, `INFER_TILELANG_PYTHON=/usr/bin/python3`

Server startup resolved:

```text
Scheduler ready: model=default, slots=8, max_seq_len=12288,
chunked_prefill_size=2048, max_num_batched_tokens=16384,
prefix_cache=on, short_prompt_bypass_tokens=256
TokenKVPool: 137328 max tokens (8583 pages @ page_size=16), 11.0 GB,
format=FP8E4M3
```

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `8` |
| sessions | `128` |
| scored turns | `128` resume turns |
| prompt shape | base `8192 +/- 64` tokens, then tool output `256 +/- 16` tokens |
| max output tokens | warmup `64`, resume `256` |
| warm/cold mix | n/a |
| tool output tokens | `256 +/- 16` |
| run cap | full trace completion |

## Results - Headline

`successful output tok/s` below uses the externally measured full-trace
wall-clock (`614.344s`). That is a conservative lower bound for resume tok/s
because this harness version does not record a separate scored-phase elapsed
time; its printed `wall total` is the sum of per-request latencies, not elapsed
wall-clock.

| metric | value |
|---|---:|
| successful scored turns | 78 / 128 |
| incomplete scored turns | 50 / 128 |
| successful output tok/s | 25.24 |
| TTFT p50 (ms) | 4340.3 |
| TTFT p99 (ms) | 26279.7 |
| ITL p50 (ms) | 79.9 |
| ITL p99 (ms) | 1601.2 |
| E2E p50 (ms) | 39262.1 |
| E2E p99 (ms) | 50476.7 |

Additional run accounting:

| metric | value |
|---|---:|
| all turns OK | 155 / 256 |
| warmup turns OK | 77 / 128 |
| scored resume output tokens | 15504 |
| server total output tokens | 20006 |
| external elapsed wall-clock | 614.344 s |
| summed scored resume wall | 2830.390 s |
| client-reported HTTP errors | 0 |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | n/a | n/a |
| TTFT p50 (ms) | n/a | n/a |
| TTFT p99 (ms) | n/a | n/a |
| output tok/s | n/a | n/a |

## Results - W4 Resume

The `cold 8k TTFT` row is not a canonical §4.2 cold-control run. It is the
successful warmup-request proxy from the same W4 trace (`max_tokens=64`, not the
resume cap of 256). No separate cold-control server run was executed for this
ARLE-only information-gathering tick.

| metric | value |
|---|---:|
| resume TTFT p50 (ms) | 4340.3 |
| resume TTFT p99 (ms) | 26279.7 |
| resume E2E p50 (ms) | 39262.1 |
| resume E2E p99 (ms) | 50476.7 |
| cold 8k TTFT p50 proxy (ms) | 5642.2 |
| cold 8k TTFT p99 proxy (ms) | 15321.4 |
| matched prefix tokens | 32 latest request, 32 peak in stats trace |
| avoided-prefill ratio | 0.37% latest request, 0.35% aggregate `prefix_skip_rate` |

Latest request accounting from `/v1/stats?format=json`:

```json
{
  "last_request": {
    "matched_prefix_tokens": 32,
    "prefix_skip_rate": 0.0037335200093338,
    "prompt_tokens": 8571,
    "resume_prefill_tokens": 8539,
    "session_id": "w4-session-119"
  }
}
```

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| peak active | 8 |
| peak waiting | 5 |
| peak prefill_queue | unavailable in JSON trace |
| peak kv_util | 99.98% |
| `prefix_hit_rate` | 95.31% final |
| `prefix_skip_rate` | 0.35% final |
| `session_affinity_hit` | 244 |
| `session_affinity_miss` | 12 |
| `tool_resume_count` | 128 scored resume requests, 78 successful |
| `tool_resume_prefill_tokens` | `resume_prefill_tokens=8539` latest request |
| `kv_fetch_q` | 0/16 final text stats |
| `kv_fetch_waiters` | 0 final text stats |
| `kv_store_q` | 0/16 final text stats |
| `kv_store` | `sub:0,done:0,fail:0,rej:0` final text stats |
| `kv_bp` | `fetch:0,store:0` final text stats |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

Aggregate session counters:

| metric | value |
|---|---:|
| prefix lookups | 256 |
| prefix hits | 244 |
| prefix reused tokens total | 7616 |
| prefix lookup prompt tokens total | 2145702 |
| average reused tokens per request | 29.75 |

Server log prefix-attach distribution:

| matched tokens | attach events |
|---|---:|
| 32 | 232 |
| 16 | 12 |

## Four-Engine Comparison

| engine | commit/tag | output tok/s | TTFT p99 (ms) | E2E p99 (ms) | cache report | raw artefact |
|---|---|---:|---:|---:|---|---|
| ARLE | `c6a43717` | 25.24 | 26279.7 | 50476.7 | `/v1/stats`, 32 matched tokens, 0.35% skip | `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/results.json` |
| SGLang | n/a | n/a | n/a | n/a | n/a | not run |
| vLLM | n/a | n/a | n/a | n/a | n/a | not run |
| TensorRT-LLM | n/a | n/a | n/a | n/a | n/a | not run |
| Mooncake | n/a | n/a | n/a | n/a | n/a | not run |

Mission margin:

```text
best_competitor = n/a for this ARLE-only run
W4_margin = n/a

single-engine entrance signal:
matched_prefix_tokens / expected_reused_tokens ~= 32 / 8571 = 0.004
resume_ttft_p99 / cold_8k_ttft_p99_proxy = 26279.7 / 15321.4 = 1.72
```

## Problems

- Current W4 resident-session reuse is shallow. The final request matched only
  32 / 8571 prompt tokens and still paid 8539 resume-prefill tokens.
- The high `prefix_hit_rate` is not enough as a W4 success signal. It mostly
  means the request matched a small shared prefix; it does not prove the 8k
  session transcript was reused.
- Only 78 / 128 scored resume requests emitted tokens. The other 50 completed
  without client-side HTTP errors but with zero output tokens and no TTFT.
- The scheduler reached KV pressure during the run: server logs include 251
  `prefix cache pressure fallback` warnings and 2 `TokenKVPool: out of pages`
  mixed-batch launch failures.
- The harness currently lacks per-turn start timestamps, so exact resume-phase
  elapsed wall-clock cannot be reconstructed after the run. The reported
  25.24 tok/s uses full-trace elapsed time and is conservative.
- No canonical cold-control W4 run was executed. The cold 8k row uses W4 warmup
  successful turns as a proxy only.

Representative server log:

```text
Request 255: paged prefix ATTACH 32/8571 tokens
Request 255: chunked prefill starting (8539 effective tokens, chunk_size=2048)
```

## Learnings

- A1/A2 alone do not make current `main` W4-winnable. The service sees a
  session-tagged prefix hit, but the reusable prefix depth stays at 16-32
  tokens instead of the expected 8k transcript.
- W4 strategy should treat avoided-prefill depth as the primary validity
  signal. Throughput and TTFT are secondary until matched-prefix tokens reach
  the §4.3 entrance scale.
- A3 or a deeper transcript/cache-publication fix is the likely next lever if
  the strategy moves to W4; this entry intentionally contains no A3 code.

## Delta Vs Baseline

- **Baseline:** first local ARLE W4 trace replay. The earlier
  [`2026-05-02-bench-agent-load-w4-harness-pending.md`](2026-05-02-bench-agent-load-w4-harness-pending.md)
  entry validated trace generation only.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| output tok/s | n/a | 25.24 | n/a |
| TTFT p99 | n/a | 26279.7 ms | n/a |
| E2E p99 | n/a | 50476.7 ms | n/a |
| matched prefix tokens | pending | 32 | n/a |
| avoided-prefill ratio | pending | 0.35% aggregate | n/a |

## Artefacts

- Raw turns: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/client_turns.jsonl`
- Client summary: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/client_summary.json`
- Combined harness result: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/results.json`
- Server launch: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/server_launch.txt`
- Engine metadata: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/engine_metadata.json`
- Server log: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/server.log`
- Service trace before: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_before.txt`
- Service trace during: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_trace.jsonl`
- Service trace after: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_after.txt`
- Service trace summary: `bench-output/2026-05-02-agent-load-arle-w4-tool-resume/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: none for this measurement. Runtime
  code remains A1 at `b1716819`; docs head for this run was `c6a43717`.
- Suspected cause of W4 weakness: only the small shared transcript prefix is
  resident/matched for resume; the long session prompt is re-prefilled.
- Refined root-cause (added 2026-05-02 by supervisor after A3 implementation
  exploration): the byte-level prefix between warmup and resume turns shares
  ~49.5 KB out of 51.4 KB (>96%), but token-level prefix matches only 32 tokens.
  The chat template re-encodes the assistant turn to splice in the
  `<tool_call>` markup after warmup completes, which forces the tokenizer to
  re-segment the boundary; subsequent tokens diverge byte-for-byte-equal but
  token-id-different. Fingerprint-based RadixCache lookup cannot bridge this
  divergence; the fix is session-id-keyed lookup (independent of
  token-fingerprint), implemented in commit `8aa5d7ab feat(scheduler): add
  session resume prefix admission lookup`.
- Follow-ups: wait for the strategy decision before starting A3, W3 deepening,
  or four-engine W4 comparison.
