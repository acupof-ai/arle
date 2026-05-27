# DSv4 TP/allreduce KILL at SLO workload — 29K prefill = 325s (67× off TTFT target)

## Context

After today's TP/allreduce route switch (`a98c3dde` wins entry — c=1
short decode 94.85 ms TPOT, 2.21× faster than EP/deepep), the next
SOLID step was a true SLO bench at the canonical workload from the
project doc: input 32K / output 1.5K / c=8 / qps=8. Target TTFT
≤ 5000 ms (current target 4800 ms).

I expected the SLO bench to surface real per-rank-range cost
breakdowns that would inform the next axis. What actually happened
is the **prefill phase didn't reach SLO range at all** — a single
29K-token prefill took **325 seconds wall-clock**, killing both the
guidellm bench harness and the server's recoverability.

## Hypothesis going in

Given c=1 short-decode was 2.21× faster on TP/allreduce vs EP/deepep:

- Expected SLO TPOT ~80-120 ms (somewhat worse than c=1 isolated due
  to NCCL congestion at c=8).
- Expected TTFT: 5-30 s for 32K prefill (would be off SLO target but
  in same order of magnitude).
- Expected to surface either (a) NCCL allreduce as the SLO bottleneck
  in c=8 mode, or (b) prefill compute as the actual blocker — which
  would tell us whether to invest in DP-attention vs A3 in-graph
  metadata vs DeepEP-LL.

## Params

| Item | Value |
|---|---|
| Hardware | 8× NVIDIA H20, driver 535.161.08 |
| Toolchain | a98c3dde (allreduce + native default) |
| Server | `--num-slots 4 --max-seq-len 34816 --mem-fraction-static 0.6 --kv-cache-dtype fp8 --deepseek-distributed-layers 43` |
| MoE backend | ARLE_DSV4_MOE_BACKEND=allreduce |
| Expert backend | ARLE_DSV4_EXPERT_BACKEND=native |
| NCCL | 2.21.5 (cu12.2-compatible) |
| Workload (target) | prompt=32768, output=1500, c=8, qps=8, guidellm |
| Workload (actual measurable) | single direct curl, prompt≈29795 tokens (after tokenizer chat template), max_tokens=128 |

The full guidellm c=8 run stalled — server received 8 requests, only
4 entered prefill, 0 tokens emitted after 12 minutes. Switched to a
single-request direct curl probe to isolate per-request prefill cost
without c=8 concurrency confounding.

## Results — wall-clock SOLID

### Single 29795-token prefill (TP/allreduce, no other concurrency)

```
2026-05-27T18:28:56  Request 0 admitted, prompt=29795 tokens
2026-05-27T18:28:56  chunked prefill starting, chunk_size=16384
2026-05-27T18:34:21  step breakdown: plan=prefill admission=0us decode=0us
                     emit=0us prefill=324763013us total=324763013us batch=1
2026-05-27T18:34:21  Request 0 done: 0 tokens (cancelled by client)
```

**prefill = 324.76 seconds for ~29K tokens, batch=1, TP=8.**

- SLO TTFT target: 4800 ms.
- Actual: 324,763 ms.
- **67.7× off target.**
- Per-token prefill rate: 91 tokens/sec across the full 8-rank cluster
  (essentially single-rank prefill rate).

### Server fragility under cancellation

When the client TCP-closed (curl killed), the server cancelled
Request 0 but **did not recover NCCL collective state**. Symptoms:

- GPU 0 stayed at 100% utilization indefinitely.
- GPUs 1-7 dropped to 0% (waiting for the next collective).
- Subsequent 7-token tiny request `Request 1` got admitted but never
  progressed past `chunked prefill starting` — server effectively
  dead.
- Required SIGKILL of all 8 worker ranks to clear.

This is a **separate bug**: in-process multi-thread NCCL TP path is
not robust to mid-prefill cancellation. Not the SLO blocker, but
discovered while debugging the SLO blocker. Worth logging.

## What this means

### The c=1 short-decode win does NOT generalize to long-context

Earlier today's `2026-05-27-dsv4-tp-allreduce-route-switch.md` wins
entry (94.85 ms TPOT, 2.21× over EP/deepep) was measured at:

- prompt = 17 tokens (smoke prompt)
- max_tokens = 32 (short decode)
- c = 1 (single in-flight request)

At long prompt (29K tokens, c=1 even), the **prefill stage is
catastrophic**. The MoE allreduce path that's fast for 1-token
decode steps is pathologically slow when expanded to a 16K-token
prefill chunk:

| Workload | TPOT/step | Why |
|---|---|---|
| c=1 short decode (M=1) | 94.85 ms | NCCL allreduce of 1 token × hidden_dim is bandwidth-trivial |
| c=1 long prefill (M=16384) | ~5 min/chunk | NCCL allreduce of 16384 tokens × hidden_dim per layer × 43 layers, plus M=16384 MoE expert routing per layer with per-token (token, expert) all over the 32 local experts |

Hypothesis why prefill is so slow (not yet verified):

1. **Per-token MoE expert dispatch in long prefill**: for M=16384 tokens
   each routed to topk=8 experts, the `forward_local_routed_gpu` walks
   experts per-batch via `forward_local_routes` (mlp.rs:1507). If this
   walks one (token, expert) at a time instead of grouped batched, M=16384
   is brutal vs M=1.
2. **Allreduce per layer at M=16384**: 16384 × 4096 hidden = 64 MB per
   allreduce × 43 layers = 2.8 GB per chunk. At NCCL ~50 GB/s would be
   55 ms — bandwidth-fine. **Not the wall-clock blocker by itself.**
3. **Compute side: M=16384 × topk=8 × 32 local experts = 4M (token,
   expert) GEMV invocations** if the expert kernel isn't grouped/batched.
   Each call ~ μs of launch overhead alone could explain 5 min wall-clock.

The c=1 nsys earlier today saw `cudaLaunchKernel 424k calls / 20.2ms`
— that was for 31 decode tokens. Extrapolated to a 16K prefill chunk
the launch count could be ~220M, which would be ~10 seconds of
launch overhead alone in optimal case. The fact that it's 5 min not
10s means there's *additional* serial bottleneck — likely the
per-expert kernel walk inside `forward_local_routes`.

### KILL framing — wall-clock, not narrow window

Per CLAUDE.md §0:
- nsys per-rank-range numbers today (TP allreduce c=1): 32% NCCL,
  29% DtoH, 21% launch, 12% expert FFN GEMV.
- **Wall-clock** 32K prefill (TP allreduce c=1): 325 s vs SLO 4.8 s
  → **67.7× off**.

If I had only looked at the nsys per-rank-range breakdown from this
morning, I'd have concluded "the levers are A3 + multi-stream
overlap" and missed the fact that the prefill code path itself is
on a different scaling curve. This is the M_pf-graph v2 framing
trap re-encountered with the roles inverted: **nsys at c=1 small
prompt does not predict wall-clock at c=8 large prompt.** SLO must
be measured at the SLO workload, not extrapolated from synthetic
nsys microbenchmarks.

### Route switch verdict — partial PASS, partial KILL

| Workload | TP/allreduce vs EP/deepep |
|---|---|
| c=1 short decode (M=1) | **2.21× better** (PASS, `a98c3dde`) |
| c=1 long prefill (M=29K) | **67.7× off SLO target** (KILL, this entry) |
| c=8 SLO 32K/1.5K | not measurable — server dies after first prefill |

The TP/allreduce default switch in `a6c910b2` is still correct as
default **for current workloads** (short prompts, single-stream
decode). But for the SLO workload it doesn't unblock the SLO. Both
EP/deepep and TP/allreduce have separate failure modes at the SLO
shape today.

## Problems

- guidellm c=8 32K bench cannot complete in max-seconds=180 because
  even a single 29K prefill takes 325s — bench window is 6× too
  short for any one request to finish prefill.
- guidellm doesn't drop in-flight requests at max-seconds, so the
  harness hangs waiting for server to respond.
- ARLE server doesn't gracefully cancel in-prefill requests; TCP
  close from client desyncs NCCL collectives.

## Next axes (revised against this KILL)

The SLO blocker is **prefill scaling**, not decode kernel mix.
Earlier axis ranking (A3 in-graph metadata, A4 multi-stream overlap)
was based on c=1 decode nsys and is **wrong** for SLO. New axes:

1. **Investigate `forward_local_routes` MoE prefill scaling** — is
   it doing per-token-per-expert iteration vs grouped? This is the
   most likely 10-100× lever. ~1d code inspection + nsys trace of
   ONE long prefill chunk to confirm.
2. **Investigate chunked prefill chunk_size** — current 16384 may
   force one giant chunk; smaller chunks (1024, 2048) might trade
   throughput for incrementality and unblock c>1 scheduling.
3. **Replicated-MoE c=1 long prefill** (cheap experiment from earlier
   discussion) — if `forward_local_routes` MoE scaling is the blocker,
   replicated-MoE removes the allreduce-after-experts loop entirely;
   might be the cheapest way to surface whether allreduce or compute
   is the bottleneck. ~100-200 LOC change.
4. **Compare against EP/deepep at same long prompt** — does it have
   the same scaling pathology, or is this allreduce-path-specific?
   Would isolate "MoE forward scaling" from "MoE backend choice".

The previous A1-A4 ranking still applies for **decode-bound**
workloads. For the **SLO prefill-heavy** workload, the blocker is
upstream.

## Refs

- `a98c3dde` — c=1 short-decode TP/allreduce wins (now known to be
  workload-specific, not SLO-applicable)
- `704cc09f` — B-3.3.5 source-level deep dive (DeepGEMM scatter race)
- `a6c910b2` — toolchain default switch to allreduce
- `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` — SLO
  frame and axis backlog
- `infer/src/model/deepseek/mlp.rs:1507` — `forward_local_routes`
  (suspected prefill scaling bottleneck)
- Server log: `docs/trace-artifacts/2026-05-27-allreduce-slo-bench/
  server.log.run2` — contains the `prefill=324763013us` step
  breakdown line that anchors this entry.

## Rule

**SLO verdict must be measured at SLO workload.** A c=1 short-prompt
nsys breakdown and a 2× improvement on that workload do not predict
SLO performance. The first thing to do after switching a critical
path (here: MoE backend) is run a single-request probe at the **SLO
prompt length**, not at a smoke prompt — because the cost curve as
M scales is path-specific and can flip the verdict completely.

Today's failure mode was the inverse of M_pf-graph v2: I had c=1
narrow-window data showing PASS, projected to SLO, and missed that
the path I picked has a different scaling curve in prefill than in
decode. Saved 1 day's misdirected optimization work by running the
true wall-clock check before committing more.
