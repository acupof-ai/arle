# P0.0 Phase 1.A nvtx scope — server-startup smoke verified

> Per `5a63142` Phase 1.A nvtx scope LANDED + `d35ca35` codex audit-of-audit
> 5/5 SOLID,Claude ran server-startup smoke test to verify scope doesn't
> break production server。**Verdict:PASS**(no errors,normal warmup time,
> accepts requests)。

## Smoke test protocol(~3 min wall-clock)

```bash
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8003 \
  --num-slots 8 \
  --max-seq-len 5120 \
  --admission-policy prefix-aware \
  --kv-cache-dtype bf16 > /tmp/p1a_server.log 2>&1 &
# Wait 25s for warmup
curl -s http://localhost:8003/v1/models  # Server health
```

## Smoke verification results

✅ **Compiles with `--features cuda`**(prior `5a63142` commit cargo check 4.76s)
✅ **Server starts cleanly**:`Server listening on 0.0.0.0:8003`
✅ **Warmup time normal**:1250ms(vs prior ~1247ms = within noise <0.3%)
✅ **CUDA Graph captured for B=1..8**:`Re-captured 8 graphs with autotuned GEMM algorithms`
✅ **`/v1/models` endpoint healthy**:returns model list
✅ **No errors/panics in server log**:grep "panic|error|undefined|fatal" returns 0
✅ **Server shutdown clean**:GPU returns to 1293 MiB / 0% post-kill

## What's NOT yet verified

⚠ **NVTX scope actually fires during admission**:smoke test only verified
server up + serves。Did not capture an nsys trace to confirm
`step_admission_prefix_lookup` shows up as a distinct NVTX range。Next-tick
action:run nsys 30s during actual bench load。

## Next-tick nsys decomposition pickup(~10-15 min)

```bash
# Server with prefix-aware policy + Phase 1.A scope (current main)
./target/release/infer ... &
SERVER_PID=$!
sleep 25

# nsys 30s capture during multi-tenant burst
nsys profile --output /tmp/p0.0-phase1a-multitenant \
    --trace cuda,nvtx,osrt --duration 30 --capture-range=none \
    /home/ckl/projects/arle/.venv/bin/python \
    scripts/bench_multitenant_burst.py http://localhost:8003 Qwen/Qwen3-4B

kill $SERVER_PID

# Extract per-NVTX-range stats
nsys stats /tmp/p0.0-phase1a-multitenant.nsys-rep \
    --report nvtxsum --format csv > /tmp/p1a_nvtx.csv

# Filter for step_admission_prefix_lookup + adjacent scopes
cat /tmp/p1a_nvtx.csv | grep -E "step_total|step_admission|step_admission_prefix_lookup|step_prefill_kernel_launch|step_decode_kernel_launch"
```

Then 4-phase decomposition formula(per `2fafa9e` recipe):
- prefix::lookup ms = `step_admission_prefix_lookup` total / N_requests
- prefill::compute ms = `step_prefill_kernel_launch` total / N_requests
- first_decode::compute ms = `step_decode_kernel_launch` total / N_requests
- scheduling::overhead ms = `step_total - step_admission - prefill - decode`

§0 SOLID rule 6 reminder:**absolute ms not NVTX-window %**(2026-05-08 EOD+19 framing trap)。

## Decision matrix(per `d2c2c17` strategic axis-ROI brief)

After 4-phase breakdown,decide P1 axis priority:

| Dominant phase(>40%)| Implication | P1 priority |
|---|---|---|
| prefix::lookup | RadixCache lookup itself slow | **invest radix-cache opt,deprio KV W4A8/Medusa** |
| prefill::compute | First-token compute slow | **P0.2 Hybrid Phase 2 dispatch wiring,KV W4A8 demoted** |
| first_decode::compute | First decode slow | **KV W4A8 ROI valid**(decode memory bw)|
| scheduling::overhead | Scheduler/queue latency | **scheduler/queue refactor,not quant/spec** |

No dominant phase(<40% any)→ KILL P1 sequence,pivot architectural OR
Option D(re-target hw/model tier per ROADMAP)。

## Phase 1.A micro-sub-cycle complete(stages 1-5)

| # | Stage | Commit | Layer |
|---|-------|--------|-------|
| 1 | Codex Phase 1.A nvtx recipe | `2fafa9e` | Recipe forward |
| 2 | Claude scoping fix(block-as-rvalue)| `b55bfcd` | Recipe-itself audit |
| 3 | Codex audit codification(skill #21) | `153fd93` | Methodology |
| 4 | Claude implementation | `5a63142` | Substrate(8 net LOC)|
| 5 | Codex impl verification 5/5 SOLID | `d35ca35` | Audit-of-audit |
| 6 | Claude server-startup smoke verify(this brief) | `(next commit)` | Empirical smoke |
| 7 | (next tick) actual nsys 30s + 4-phase decomposition | (pending) | Phase 8 evidence |

Compares to R4#6 micro-cycle:both prove bidirectional audit + natural-closure
heuristic + bench-driven evidence layer generalize to fresh axes。

## Status

**Phase 1.A scope LANDED + smoke verified**。Production server starts/serves
with scope active,no regressions。**Ready for nsys decomposition pickup
next tick**(~10-15 min wall-clock once started)。
