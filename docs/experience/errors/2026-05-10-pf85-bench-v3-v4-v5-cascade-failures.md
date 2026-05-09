# PF8.5 bench v3/v4/v5 cascade failures — 3 attempts blocked, baseline INT8 data captured but treatment FP8 never measured

## Context

After PF8.3 substrate landed (`11763ba`) + cargo build refresh,
attempted 3 bench A/B sequences via `scripts/pf83_license_sequence.sh
--skip-greedy --skip-ppl`. ALL 3 failed differently:

## Attempt summary

| Run | Args | Outcome | Data captured | Failure |
|-----|------|---------|---------------|---------|
| v3 | `--quick` | partial | baseline INT8 conc 1-4 | server core dump at conc=8 |
| v4 | `--quick` (after RUST_MIN_STACK 8→32MB + --concurrencies "1,2,4" attempt) | killed by Claude | none | --quick override hardcoded streams=[1,2,4,8] (bench_guidellm.sh:252); v4 would hit same conc=8 crash + orphan v3 server still alive interfering |
| v5 | `--concurrencies "1,2,4" --max-seconds 30` (no --quick) | crashed | none | bench_ab.sh stuck in pkill cleanup loop, eventually core dumped (`timeout: 被监视的命令已核心转储`) |

## v3 baseline INT8 numbers (captured before conc=8 crash)

| Concurrency | TTFT mdn (ms) | TTFT p95 (ms) | ITL mdn (ms) | TPOT mdn (ms) | Throughput (req/s) | Total tok/s |
|-------------|---------------|---------------|--------------|---------------|--------------------|-----------|
| 1           | 53.6          | 54.1          | 6.8          | 7.2           | 1.1                | 697       |
| 2           | 68.4          | 69.0          | 7.4          | 7.9           | 2.0                | 1259      |
| 4           | 110.2         | 154.2         | 8.3          | 8.8           | 3.5                | 2248      |
| 8           | **CRASH**     | —             | —            | —             | —                  | —         |

Source: `/tmp/claude-pf85-bench-v3.log` lines 168-178 + corresponding
guidellm output table.

## Root causes (per skill v1.11.0+ #28+#31 raw evidence)

### v3 conc=8 crash (Task #43 manifest)

Server `target/release/infer` PID 1907144 ABORTED (核心已转储) at
conc=8 phase. Per `dc0db7e` errors entry: Task #43 manifests at
512-token prompts (not just 4k as originally documented). 8MB
RUST_MIN_STACK insufficient.

### v4 --quick override blocker

Per `bench_guidellm.sh:252`:
```
--quick) RATE_OVERRIDE="1,2,4,8" ;;
```

`--quick` is hardcoded to include conc=8. My `--concurrencies "1,2,4"`
flag in bench_pf83_ab.sh is OVERRIDDEN by --quick. v4 would hit same
crash if let run.

Killed v4 to prevent wasted bench cycle.

### v5 bench_ab.sh cleanup loop crash

v5 started fresh (no --quick, explicit `--concurrencies 1,2,4
--max-seconds 30`) but bench_ab.sh got stuck in repeated pkill
cleanup:
```
已终止 pkill -f "metal_serve|cuda_serve|infer_serve" 2> /dev/null
[× 50+ iterations]
timeout: 被监视的命令已核心转储
```

Core dump cause unclear without coredump analysis. Likely:
- bench_ab.sh's launch-loop retry mechanism hit infinite retry on
  some condition
- OR bash error in script crashed after many iterations

bench_ab.sh wasn't designed for the explicit-args path; --quick was
the canonical invocation per its docstring.

## What we know about PF8.3 substrate from v3 partial

- ✅ Server starts cleanly with INFER_HYBRID_W4A8_PREFILL=1 +
  INFER_MARLIN_W4_FP8_PREFILL=0
- ✅ Conc=1 baseline: 53.6ms TTFT, 6.8ms ITL — REAL number
- ✅ Conc=2,4 also produced real numbers
- ✅ GPU 100% utilized during conc=4 phase
- ❌ Server crashes under conc=8 sustained load
- ❌ Treatment FP8 (INFER_MARLIN_W4_FP8_PREFILL=1) NEVER MEASURED

## Next path forward

**Option A — bypass bench_ab.sh + run guidellm directly** (recommended):

```bash
cd /home/ckl/projects/arle
PATH=$PWD/.venv/bin:$PATH

# Start server with PF8 enabled
RUST_MIN_STACK=33554432 INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_MARLIN_W4_FP8_PREFILL=1 \
  target/release/infer \
  --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix --port 8000 \
  > /tmp/pf83-treatment-fp8-direct.log 2>&1 &
sleep 30  # wait for warmup

# Run guidellm at conc 1,2,4 only (skip 8)
guidellm benchmark run --target http://127.0.0.1:8000 \
  --model infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --processor infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --profile concurrent \
  --data 'prompt_tokens=512,output_tokens=128' \
  --rate "1,2,4" --max-seconds 30 \
  --output-dir bench-output/2026-05-10-pf83-treatment-direct

# Kill server
pkill -f "target/release/infer.*--port 8000"
```

Compare manually to v3 baseline INT8 numbers for license decision.

**Option B — fix bench_ab.sh** (more invasive):
- Diagnose pkill cleanup loop bug
- Add `--no-conc8` flag OR fix --quick to be configurable
- Risk: significant scope creep

**Option C — defer PF8.5 license** (safest):
- Document substrate landed but bench infrastructure broken
- Pivot to #28 Medusa Phase 1.A (separate axis, unblocked per `8735361`)
- Revisit PF8.5 when Task #43 gets fixed (deeper investigation)

## Rule

For PF8.5 (and similar A/B benches):
1. NEVER use `--quick` for new substrate validation — it bundles
   conc=8 which can hit Task #43 stack overflow
2. bench_ab.sh has fragile cleanup logic — direct guidellm invocation
   is more reliable for one-off A/B
3. Always run with explicit `--rate "1,2,4"` to avoid the conc=8 trap
4. Include cleanup verification: `pgrep -f "infer.*--port 8000"`
   should return empty before next bench

## Cross-references

- 11763ba PF8.3 substrate landed
- dc0db7e v3 errors entry + RUST_MIN_STACK 8→32MB attempt
- 9bb3843 original RUST_MIN_STACK=8MB (insufficient)
- 172c311 PATH .venv/bin fix (worked for v3)
- 45579c0 INFER_HYBRID_W4A8_PREFILL=1 fix (worked for v3)
- bench_guidellm.sh:252 (--quick hardcodes streams 1,2,4,8)
- Task #43 (server stack overflow — manifests at conc=8 with 512 prompts too)

## Status

3 bench attempts blocked. v3 captured useful baseline INT8 numbers
but treatment FP8 NEVER measured. PF8.5 license decision DEFERRED
until either:
- Direct-guidellm bypass produces FP8 treatment numbers (Option A)
- bench_ab.sh fixed (Option B)
- OR axis pivots to #28 Medusa (Option C, accepts substrate-only PF8.3)

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(bench v3/v4/v5 logs + core dump messages + bench_guidellm.sh source
grep — all THIS tick).
