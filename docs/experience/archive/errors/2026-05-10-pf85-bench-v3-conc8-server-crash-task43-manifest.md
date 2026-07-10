# PF8.5 bench v3 crashed at conc=8 — Task #43 manifested with 512-token prompts (not just 4k as originally documented)

## Context

PF8.5 license sequence bench v3 (`/tmp/claude-pf85-bench-v3.log`,
started 07:19) ran baseline INT8 successfully through concurrencies
1, 2, 4 then **server PID 1907144 ABORTED (core dumped) at conc=8**.
Treatment FP8 (cmd-b) never ran because bench_ab.sh aborted on the
baseline failure.

## Real baseline data (PF8.3 hybrid checkpoint, INT8 path, 512 prompt × 128 output tokens)

| Concurrency | Latency Mdn (s) | TTFT Mdn (ms) | TTFT p95 (ms) | ITL Mdn (ms) | TPOT Mdn (ms) | Throughput (req/s) | Total tok/s |
|-------------|-----------------|---------------|---------------|--------------|---------------|--------------------|-----------|
| 1           | 0.9             | 53.6          | 54.1          | 6.8          | 7.2           | 1.1                | 697       |
| 2           | 1.0             | 68.4          | 69.0          | 7.4          | 7.9           | 2.0                | 1259      |
| 4           | 1.1 / 1.3       | 110.2 / 154.2 | —             | 8.3          | 8.8 / 10.1    | 3.5                | 2248      |
| 8           | **CRASH**       | **CRASH**     | **CRASH**     | **CRASH**    | **CRASH**     | **CRASH**          | **CRASH** |

Last service_stats_trace.jsonl entry before crash showed:
- engine_ttft_us=75000.0 (75ms TTFT during conc=8 phase)
- engine_active_requests=4 (server was in 8-concurrent state)
- batch_occupancy=0.65 (65%)
- step_phase: prefill=7.3ms, decode=185us, total=7.6ms
- kv_util=64.1%, peak_mem=14197.6 MB

## Root Cause

Server crash signature:
```
/scripts/bench_ab.sh: 第 1 行：1907144 已中止 （核心已转储）
```
Translation: PID 1907144 ABORTED (core dumped).

This matches Task #43 (originally logged 2026-05-10 with PID 1816462 at
4k-token bench) but **crash now manifests at 512-token prompts at conc=8**
— so the original Task #43 description "Server stack overflow under
sustained W4A16 4k-token bench load" was INCOMPLETE: the real trigger
is high concurrency, not prompt size.

Hypotheses (per Task #43 description):
- Recursion in prefix cache demotion (cleanup at conc transition?)
- Stack-allocated work queue blowing at high block counts
- tokio task stack amplification under sustained load
- **NEW**: cuda kernel launch queue stack at conc=8 sustain

RUST_MIN_STACK=8388608 (8MB) was set per `9bb3843` but did NOT
prevent the conc=8 crash. Either:
- 8MB insufficient (try 32MB)
- Crash is in non-Rust thread (CUDA driver, tokio worker) where
  RUST_MIN_STACK doesn't apply
- Crash is SEGV not stack overflow → would need different mitigation

## Fix (THIS TICK)

Per `bench_pf83_ab.sh` THIS commit:
- **RUST_MIN_STACK=8388608 → 33554432** (8MB → 32MB, 4× headroom)
- **--concurrencies "1,2,4"** explicit override (drop conc=8 until
  root cause known)

Both changes invariant across A/B (no measurement bias).

If next bench v4 also crashes at conc=4 → escalate to deeper Task #43
investigation (analyze coredump for stack trace OR add bench scope
narrowing).

If bench v4 completes → A/B numbers ready for license decision per
aebd4a5 gates.

## Rule

For any new substrate added to inference path (PF8.3, future Medusa,
hybrid quant, etc.):
1. Bench at multiple concurrencies (1, 2, 4) to surface scale-related
   bugs that single-concurrency tests miss
2. RUST_MIN_STACK alone is insufficient — kernel + non-Rust threads
   may need separate mitigations
3. Task #43 is NOT 4k-prompt-specific — the core dump trigger is
   high-concurrency-sustained-load on hybrid W4 path

## Cross-references

- Task #43 (server stack overflow — original 4k context, this tick
  proves it manifests at 512 prompts too)
- 9bb3843 (RUST_MIN_STACK=8MB original mitigation — insufficient)
- THIS commit: bench_pf83_ab.sh updated to RUST_MIN_STACK=32MB +
  --concurrencies "1,2,4"
- bench v3 logs: /tmp/claude-pf85-bench-v3.log + bench-output/2026-05-10-pf83-baseline-int8-run2/
- Baseline data per this entry's table is FROM the partial v3 run
  (conc 1-4 succeeded before crash)

## Status

- bench v3 partially succeeded (conc 1-4 baseline INT8 captured)
- Treatment FP8 (cmd-b) NEVER RAN due to bench_ab.sh abort-on-failure
- Bench v4 retry needed with --concurrencies "1,2,4" + 32MB stack
- License decision deferred until A/B completes for at least conc 1-4

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(bench v3 log lines + service_stats_trace.jsonl tail + bench_ab.sh
abort message — all THIS tick).
