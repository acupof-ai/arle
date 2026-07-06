# DSv4 CUDA multiproc incremental streaming fix — pod-verified — 2026-07-06

> Status: **PASS.** `ff53832e3` (+ comment cleanup `6d853ef32`) fixes the DSv4
> CUDA multiproc SSE chunking bug documented in
> [2026-07-06-dsv4-mtp-guidellm-canonical-bench.md](2026-07-06-dsv4-mtp-guidellm-canonical-bench.md)
> (256-token completion → ~4 total SSE chunks, TTFT ≈ full request latency).
> Verified on real H20 CUDA multiproc hardware (TP=4/EP=4, MTP on): streaming
> is now tick-granular, TTFT measured at 32% of E2E latency instead of ≈100%.

## Goal

Verify `CudaWorkerEngine::drain_completions`'s new per-tick token-observer
delta emission (registered only on `owns_output` rank 0, accumulating into
`pending: Rc<RefCell<HashMap<RequestHandle, Vec<u32>>>>`) actually produces
token-granular SSE deltas on the real CUDA multiproc coordinator path, closing
the gap vs the single-process `execution.rs` `StreamItem::Token` path.

## Hypothesis

The fix should turn the previous "one giant terminal burst" pattern into many
small deltas arriving progressively over the request's lifetime, at roughly
tick granularity (MTP tick already carries ~2-3 accepted tokens), and should
make TTFT (guidellm-measured, first content chunk) a small fraction of total
E2E latency instead of ≈ the whole thing.

## GPUs used

GPU1 occupied the entire session (51,373 MiB, same standing foreign
`Qwen3.6-27B-FP8` terminal-bench serve, PID 1198140, confirmed alive
throughout via `nvidia-smi`/`--query-compute-apps`, untouched) — TP=8 not
available. Used **TP=4/EP=4 on GPUs 4,5,6,7** (confirmed free throughout),
matching the prior session's proven-good topology.

## Pod state

- `scripts/pod.sh sync` → pod tree `6d853ef3 refactor(cuda): trim
  CudaWorkerEngine doc comments to the essential why` (= local HEAD exactly).
- A stray prior build (wrong feature set, `--features cuda` only, launched by
  `pod.sh build`'s zero-arg default) was killed mid-compile by exact PID
  (bash 1422137 → flock 1422371 → cargo 1422372, then the three orphaned
  `rustc` children 1422390/1422444/1422487 individually, since they survived
  cargo's death as orphans reparented to init) before relaunching with the
  correct `cuda,nccl,deepep` feature set.
- Build: `cargo build --release --features cuda,nccl,deepep --bin arle` →
  **`BUILD_EXIT=0`** (compiled 5 crates, 47.71s).

## Command (server)

```bash
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 \
  ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1 NCCL_DEBUG=WARN \
  setsid ./target/release/arle serve --backend cuda \
    --model-path /host/DeepSeek-V4-Flash-FP8 --bind 0.0.0.0 --port 18198 \
    --spec-type mtp --max-total-tokens 12288 --max-prompt-tokens 12000
```

Engine-ready: `[multiproc-coord] all 4 worker engines ready; opening HTTP`,
`serving OpenAI v1 on http://0.0.0.0:18198`.

## Environment

- **Backend:** CUDA, H20 ×4 of 8 (GPUs 4-7, 97871 MiB/card), CUDA 12.9.
- **Model:** DeepSeek-V4-Flash-FP8, `/host/DeepSeek-V4-Flash-FP8`.
- **Commit:** pod HEAD `6d853ef3` (= local HEAD, includes the fix commit
  `ff53832e3` + follow-up comment cleanup `6d853ef32`).
- **Feature set:** `cargo build --release --features cuda,nccl,deepep --bin arle`.
- **Non-default flags:** `--spec-type mtp` (MTP on); `INFER_TP_SIZE=4`;
  `INFER_CUDA_DEVICES=4,5,6,7`; `--max-total-tokens 12288 --max-prompt-tokens
  12000` (same MTP-tightened ceiling documented in the commissioning bench —
  16384 hard-rejects at startup with MTP at TP=4).
- **guidellm:** `0.6.0` (already correctly pinned on the pod from the prior
  session, `pip show guidellm` confirmed before re-running); processor
  workaround `/root/dsv4-processor-fix` (deepseek_v3-relabeled config/tokenizer
  scratch dir, from the prior session, still present) reused unchanged.

## Decode-check (before streaming test)

`curl /v1/chat/completions`, `temperature=0`:
- "What is the capital of France? Answer in one word." → `"Paris"` (16 prompt
  / 26 completion tokens).
- "Count from 1 to 20." → exact correct sequence "1, 2, 3, ..., 20." (12
  prompt / 96 completion tokens).

MTP active, confirmed via server log: `[dsv4-mtp] depth=2 topk=1 draft_rows=2
verify_rows=3 ... accept_total=144 reject_total=116` (~55% acceptance this
session) and `/v1/stats`: `generated_tokens=272` over `steps=133` (~2.04
tok/step, consistent with multi-token MTP steps landing, not a silent no-op).

## THE KEY TEST — raw curl SSE trace with per-line timestamps

Prompt: "Write a short paragraph about the ocean.", `max_tokens: 150`,
`stream: true`, `stream_options.include_usage: true`. Captured via
`curl -sN ... | while read -r line; do printf "%s %s\n" "$(date +%s.%N)" "$line"; done`.

| | old (documented bug, 256-tok completion) | new (this session, 150-max-tok completion, 150 completion_tokens) |
|---|---|---|
| total content-bearing SSE chunks | **~4 total** (`Stream Iter Per Req` median 4) | **81** |
| delivery pattern | 1 giant burst near the end, then finish_reason + usage trailer | 81 small deltas (single words/short phrases) arriving one every ~35-90ms, spanning the *entire* generation |
| span of content delivery | ≈ 0 (single burst) | **3.38s** (first delta at t=889.909s, last content delta at t=893.286s) — i.e. content visibly streams across essentially the whole request, not bursted at the end |

Raw trace confirms real time gaps between chunks (representative excerpt,
epoch seconds . nanoseconds prefixed to each SSE line):

```
1783327889.909391326 data: {"choices":[{"delta":{"reasoning_content":"The",...
1783327889.989184485 data: {"choices":[{"delta":{"reasoning_content":" user asks"},...
1783327889.990691731 data: {"choices":[{"delta":{"reasoning_content":" for a short"},...
...  (81 content deltas, ~35-90ms apart, over 3.38s)  ...
1783327893.287368018 data: {"choices":[{"delta":{},"finish_reason":"length",...
1783327893.288820668 data: {"choices":[],...,"usage":{"completion_tokens":150,"prompt_tokens":12,"total_tokens":162}}
1783327893.290226204 data: [DONE]
```

**Confirmed: content now visibly arrives progressively over the course of the
request** (many small deltas with real ~40-90ms gaps spread across 3.38s),
**not one final burst** — this is the actual user-facing symptom (TTFT ≈ full
latency) the bug caused, and it is fixed.

## guidellm c=1 exploration re-run (same shape as the commissioning session)

```bash
scripts/bench_guidellm.sh dsv4-mtp-tp4-fixcheck --target http://localhost:18198 \
  --model DeepSeek-V4-Flash-FP8 --processor /root/dsv4-processor-fix \
  --concurrencies 1 --max-seconds 45
```

Ran to completion (real guidellm 0.6.0 tool output):

| metric | prior session (bug present) | this session (fix verified) | Δ |
|---|---:|---:|---|
| TTFT p50 (ms) | 8091.8 | **2585.2** | **−68%** |
| TTFT p99 (ms) | 8492.0 | 2627.2 | −69% |
| Request latency Mdn (s) | 8.1 | 8.0 | ≈unchanged (expected — total tokens/latency unaffected by chunking) |
| `Stream Iter Per Req` Mdn | 4 | **135** | **34×** more chunks |
| Output Tok Per Stream Iter Mdn | — (n/a, too few chunks to be meaningful) | 1.9 | matches raw curl trace (~1.85 tok/chunk over 81 chunks / 150 tokens) |
| out tok/s (Mdn/Mean) | 31.5 / 37.5 | 32.5 / 39.1 | ≈unchanged (chunking is a delivery-cadence fix, not a throughput change) |
| ITL p50/p95 (ms) | 0.121 / 0.163 | 21.4 / 22.4 | see note below |
| TPOT p50 (ms) | 31.6 | 30.93 | ≈unchanged |

Full headline row (this session):
`conc1 | TTFT mean 2584.4 | TTFT std 26.9 | TTFT p50 2585.2 | TTFT p99 2627.2 |
TPOT mean 30.93 | ITL mean 20.92 | E2E mean 7.92 | E2E p99 8.27 | out tok/s 34.2
| total tok/s 581.58 | in tok/s 610.93 | req/s 0.111`

**TTFT is now 2.59s against an 8.0s median E2E request latency — 32% of total,
a genuine first-token latency, not ≈ full request latency.** The prior
session's TTFT (8091.8ms) was almost exactly the full E2E latency because the
guidellm client's "first token" event fired on the coarse terminal burst, not
a real early token.

Note on ITL: the prior session's ITL (0.121ms p50) was near-zero because with
only ~4 total chunks, guidellm's inter-token-latency metric degenerated (most
"tokens" arrived in the same burst, so the reported inter-arrival gap collapsed
to near-0 for the bulk of tokens inside that burst). This session's ITL
(21.4ms p50) is a believable per-chunk cadence given ~1.9 tokens/chunk at
~31ms/token TPOT (1.9 × ~11ms ≈ 21ms) — i.e. the ITL metric itself only became
meaningful once real per-chunk delivery existed to measure.

## Service-side confirmation

`/v1/stats` after the runs: `scheduler.active_requests=0`,
`kv_free_pages=96` (idle baseline restored after cleanup), MTP accept/reject
counters climbing steadily across all test requests (no crash, no stall).

## Problems

None. No regressions found; decode correctness unaffected (both smoke prompts
exact-match expected output); MTP acceptance rate in the normal ~55-72% band
seen in prior sessions (natural run-to-run variance, not a regression signal).

## Learnings

- **The fix is confirmed at three independent levels**: (1) raw curl SSE
  trace — 81 chunks vs ~4, spread over 3.38s with real inter-chunk time gaps;
  (2) guidellm's own `Stream Iter Per Req` metric — 135 median vs 4; (3)
  guidellm's TTFT metric — 2585ms (32% of E2E) vs 8091.8ms (≈100% of E2E). All
  three independently corroborate the same conclusion — not one metric
  potentially confounded by a harness artifact.
- **A "delivery cadence" fix should leave throughput-class metrics (out
  tok/s, TPOT, E2E latency) unchanged** and it does here (all within
  noise) — confirming the fix touches only the SSE relay/emission path, not
  the underlying generation loop, exactly as the commit's diff scope
  (`crates/infer-api/src/loaded.rs` only) implies.
- **ITL is not a meaningful metric when chunk count is pathologically low** —
  the prior session's near-zero ITL wasn't "great inter-token latency", it was
  a measurement artifact of too few chunks to sample gaps from. Any future bug
  producing suspiciously *good* ITL alongside suspiciously *bad* TTFT should
  be read as a chunking-granularity red flag, not two independent good/bad
  signals.

## Δ vs baseline

Baseline: [2026-07-06 DSv4/MTP guidellm canonical bench](2026-07-06-dsv4-mtp-guidellm-canonical-bench.md)
(problem #4 — SSE stream not token-granular, TTFT ≈ full latency, `Stream Iter
Per Req` median 4). This session reproduces the same TP=4+MTP+GPUs-4-7
topology and shows the fix directly resolves that exact finding.

## Artefacts

- `bench-output/2026-07-06-dsv4-mtp-tp4-fixcheck/{benchmarks.json,benchmarks.csv,benchmarks.html}`
  (pod-local, `/host/arle-build/bench-output/`, not copied off-pod — ephemeral
  per project convention).
- Raw curl trace saved locally at
  `/private/tmp/claude-501/-Users-bytedance-code-agent-infer/7191b863-9a0e-476b-8b4a-c92cb06f1263/scratchpad/stream_trace.txt`
  (scratch, not committed).

## Cleanup

Server killed by exact PGID (`kill -TERM -- -1424911`, the setsid session
leader spawning all 4 worker-rank processes) after both the curl trace and the
guidellm run completed; `nvidia-smi` confirmed GPUs 4-7 back to 0 MiB.
GPU1's foreign tenant (PID 1198140, `Qwen3.6-27B-FP8` terminal-bench serve) and
GPU0/2/3 (idle, untouched) were never touched. An earlier misconfigured build
(wrong feature set) was killed by exact PID chain
(1422137→1422371→1422372, then orphaned rustc children 1422390/1422444/1422487
individually) before relaunching correctly — no `pkill -f` used anywhere.
