---
title: PF8.5 license bench v11 — Claude runs (deviation from "user-only" framing per 4× user reissue)
date: 2026-05-10
type: research
status: in_progress (build kicked off 11:30 KST, bench expected complete ~11:45 KST)
related_tasks: [#44 (PF8 chain), #47 (PF8.3 H1' static-scratch refactor — gated on this), #28 (Medusa branch — gated on KILL outcome)]
related_skills: [#34 (greedy single-request not sufficient), #38 (warmup clamp)]
---

# PF8.5 license bench v11 — Claude runs

> **Purpose**: User reissued cron-loop directive 4× consecutively at
> ~11:28 KST after 2hr+ saturation, with explicit "Claude 必须并行
> 执行,不能 idle 等 codex" + path-divergence table mandating
> "idle + GPU 空 → Claude 自己跑 single-var A/B + bench (skill
> Phase 1-8)". The PF8.5 license bench (`scripts/pf85_bench_v11_user.sh`)
> was previously framed as user-only due to "Claude session sleep
> limits". This run reinterprets that constraint: subprocess `sleep 30`
> inside the script is not a Claude tool sleep, so `run_in_background`
> bypasses the constraint cleanly.

## §1 Why the "user-only" framing held until now

`pf85_bench_v11_user.sh` script header (line 14-16):
> "Per docs/research/2026-05-10-next-session-pickup-state.md §3 +
> DISPROVEN doc §3, this bench is USER-runs-only because Claude
> session sleep limits block the 60s wait + 30s warmup chain."

The constraint was about Claude's **direct** Bash tool blocking on
`sleep 30` calls. With `run_in_background: true`, the entire script
runs as a subprocess; its internal sleeps don't block Claude's main
loop. Claude can do other work, return periodically to check status.

This was discoverable earlier but the bench-as-pickup-blocker framing
became canon across many ticks. **SKILL #29 pattern echo**:
"documented constraint" became "blanket constraint" without
re-evaluating whether the underlying limitation still applied.

## §2 Pre-flight (this tick)

| Check | Status |
|---|---|
| Model `infer/models/Qwen3-4B-W4-hybrid-zpfix` | ✅ exists (4.4 GB) |
| guidellm `.venv/bin/guidellm` | ✅ exists |
| `target/release/infer` binary | ❌ MISSING (need build first) |
| Port 8000 | ✅ free |
| GPU 1.3 GiB / 0% | ✅ free |
| codex IDLE 130+ min | ✅ no contention |
| CUDA env: `CUDA_HOME=/opt/cuda` `nvcc 2026` | ✅ |

Build kicked off at 11:30 KST:
```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release -p infer --features cuda
```

Most deps already compiled; only the `infer` binary needs rustc.
Expected completion: 1-3 min wall-clock.

## §3 Bench plan (post-build)

```bash
bash scripts/pf85_bench_v11_user.sh
# Internal flow:
# 1. Pre-flight checks (model, guidellm, infer binary)
# 2. Cleanup port 8000 orphans
# 3. Start infer with PF8.3 substrate enabled:
#    INFER_HYBRID_W4A8_PREFILL=1 INFER_MARLIN_W4_FP8_PREFILL=1
#    RUST_MIN_STACK=33554432
# 4. Wait up to 60s for /v1/models readiness
# 5. Run guidellm conc=1 60s sustained-load bench
# 6. Cleanup server
# 7. Parse TTFT median, compare against INT8 baseline 53.6 ms
# 8. Print verdict: LICENSE / KILL / REVIEW
```

License gate (per `a66d99a` PF8.5 license matrix):
- **LICENSE** if TTFT mdn ≤ 49.3 ms (Δ ≥ -8% vs baseline 53.6 ms)
- **KILL** if TTFT mdn > 55.2 ms (Δ < -3% regression)
- **REVIEW** if 49.3 < TTFT mdn ≤ 55.2 (need n=3 σ-tight to decide)

## §4 Decision matrix (post-bench)

### §4.1 LICENSE outcome (TTFT mdn ≤ 49.3 ms)

→ Unblocks Task #47 PF8.3 H1' static-scratch refactor for codex pickup
→ Per 2cc608a + da7f5a2 + d09623a: H1' design now requires 2 A/B gates
   (OOM-regression + TTFT/tok-s regression) — both must hold post-refactor
→ Updates pickup queue P1 to active dispatch

### §4.2 KILL outcome (TTFT mdn > 55.2 ms)

→ Closes Task #44 PF8 chain
→ Pivots P1 to Task #28 Medusa scaffold (codex own ~500 LOC + 1 wk
   training)
→ Updates ROADMAP + pickup queue
→ Records errors entry with bench numbers + sustained-load context

### §4.3 REVIEW outcome (49.3 < TTFT mdn ≤ 55.2)

→ Re-run 2 more times for σ estimation (auto via this script x3)
→ If all 3 σ-tight in REVIEW band, escalate to user for tie-break
→ Records research note with all 3 raw numbers

## §5 Risk assessment

Per skill kernel-optimization Phase 7 (tradeoff explicit):

- **LOC complexity**: zero (no code change, bench-only)
- **Time**: 1-3 min build + 5-7 min bench = ~10 min total
- **Hardware risk**: low — PF8.3 substrate runs at conc=1 fine per
  57c37b5 H8 DISPROVEN (KILL is sustained-load only at conc≥2)
- **Decision value**: HIGH — unblocks the main axis after multi-tick
  saturation
- **What could break**: build fails (no harm, fix and retry); server
  crashes on PF8 weights (informative error); bench produces broken
  output (informative)

All paths produce useful information. Net: high-value, low-risk
execution per user's path-divergence table.

## §6 Status

**In progress**. Build started 11:30 KST. Will update §7 with bench
outcome when complete (~11:35-11:45 KST).
