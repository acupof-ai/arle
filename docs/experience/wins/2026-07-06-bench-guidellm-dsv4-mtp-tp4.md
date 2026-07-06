# DSv4 TP=4/EP=4 MTP speculative-decode bench — guidellm BLOCKED (2 structural gaps found), 2026-07-06

> Status: **BLOCKED — canonical guidellm table not produced.** Serve itself is
> verified correct (TP=4/EP=4, MTP checkpoint-native spec decode, clean decoded
> output). The blocker is NOT DSv4/MTP/TP=4-specific: `scripts/bench_guidellm.sh`
> cannot complete against the current `infer-server` rewrite on ANY backend
> because streaming responses never populate `usage` — a rewrite regression.
> A second, unrelated dependency-drift bug (installed guidellm CLI mismatch)
> was found and fixed en route. A third, TP=4-specific finding: the server
> **crashes** (does not gracefully reject) under c=4 concurrency at the
> canonical 4096-token prompt shape. Every number below is measured; nothing
> is fabricated to fill the table.

## SLO-shape probed?  N

Canonical sweep never ran (blocked at the streaming preflight). One
supplementary non-canonical c=1 measurement is included (see Results),
explicitly NOT the SLO-shape gate (no batch≥4, no sweep).

## Roofline check

Not run — no canonical sweep executed. Deferred, not a KILL.

## Goal

Measure TTFT / TPOT(ITL) / output tok/s for DeepSeek-V4-Flash-FP8 served at
**TP=4/EP=4** (4 of 8 physical GPUs — a deliberate substitute for the canonical
TP=8/EP=8 shape, since GPU1 was held by a different concurrent session's
standing terminal-bench serve for the entire prior session) with MTP
(checkpoint-native NextN speculative decode, `--spec-type mtp`) enabled, via
the canonical `scripts/bench_guidellm.sh` sweep.

## Hypothesis

MTP draft-verify (`mtp_draft_tokens=2`, `mtp_draft_topk=1`, exact-greedy
acceptance) lowers per-token decode latency whenever the draft head's greedy
token matches the trunk's verify token. Directionally a win, per the same
NextN design already measured correct on Qwen3.6
([2026-06-06 EAGLE/MTP phase2 wins](2026-06-06-dsv4-eagle-mtp-phase2-verify-correct.md)).

## GPUs used

**Confirmed free via `nvidia-smi` immediately before launch** (0 MiB / 0%
util, no `nvidia-smi --query-compute-apps` entry): GPUs **0, 2, 3, 4, 5, 6, 7**.
GPU1 (51243 MiB, 95-96% util) was the only occupied GPU — same standing
`Qwen3.6-27B-FP8` terminal-bench serve (PID 2094648) documented in the
[prior TP=8 attempt](2026-07-06-bench-guidellm-dsv4-mtp.md). **Used GPUs 4,5,6,7**
(contiguous subset of the confirmed-free set) for this TP=4/EP=4 run.
Re-verified free immediately at launch time (second `nvidia-smi` check, same
result) — never assumed stale.

## Command

Server (TP=4/EP=4, MTP on, GPUs 4/5/6/7 via `INFER_CUDA_DEVICES`, NOT
`CUDA_VISIBLE_DEVICES` — DSv4 multi-rank serve reads physical ordinals from
`INFER_CUDA_DEVICES`/`INFER_TP_SIZE`, per `crates/cli/src/serve_multiproc.rs:56-84`):

```bash
export INFER_CUDA_DEVICES=4,5,6,7
export INFER_TP_SIZE=4
export INFER_DSV4_MAX_SEQ_LEN=8192   # see Problems — 16384 fails startup at TP=4
export ARLE_DSV4_MOE_BACKEND=allreduce
export ARLE_DSV4_INCREMENTAL_KV=1
export NCCL_DEBUG=WARN
/host/arle-build/target/release/arle serve --backend cuda \
  --model-path /host/DeepSeek-V4-Flash-FP8 --bind 0.0.0.0 --port 18196 \
  --spec-type mtp
```

Bench (canonical, unmodified wrapper — locked params, `docs/plans/guidellm-integration.md` §3):

```bash
scripts/bench_guidellm.sh dsv4-mtp-tp4 \
  --target http://localhost:18196 \
  --model DeepSeek-V4-Flash-FP8 \
  --processor /host/DeepSeek-V4-Flash-FP8
```

## Environment

- **Backend:** CUDA, H20 ×4 of 8 (97871 MiB/card), CUDA 12.9.
- **Model:** DeepSeek-V4-Flash-FP8, `/host/DeepSeek-V4-Flash-FP8`
  (`num_nextn_predict_layers=1` — ships an MTP draft head).
- **Commit:** `f22ad1ff0` (pod HEAD, matches local; verified via
  `git -C /host/arle-build log --oneline -1` and
  `strings target/release/arle | grep -c 'cancel req#'` = 1).
- **Feature set:** `cargo build --release --features cuda,nccl,deepep --bin arle`
  (binary already built from the prior session, mtime Jul 6 03:40, reused —
  no rebuild needed).
- **Non-default flags:** `--spec-type mtp` (defaults: `mtp_draft_tokens=2`,
  `mtp_draft_topk=1`); `INFER_TP_SIZE=4`; `INFER_CUDA_DEVICES=4,5,6,7`;
  `INFER_DSV4_MAX_SEQ_LEN=8192` (lowered from the usual 16384 — see Problems).
- **Profiling state:** OFF (no `ARLE_DSV4_DECODE_PHASE_TIME` /
  `ARLE_DSV4_LINEAR_PROFILE`).
- **Server launch:** verified engine-ready — coordinator log:
  `[multiproc-coord] all 4 worker engines ready; opening HTTP`,
  `serving OpenAI v1 on http://0.0.0.0:18196`. VRAM ledger: weights 76377MB +
  adapter 522MB + Σ49 slots 19915MB = 96815MB predicted vs 95687MB measured
  (residual -1129MB, ledger consistent).

## Canonical params (locked, blocked before use)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256` (+ stdev/min/max clamps)
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| — | **NOT RUN — guidellm preflight blocked** | — | — | — | — | — |

## Results — supplementary non-canonical measurement (NOT the SLO gate)

Server verified live and correct (decode-checked, not assumed) before this
measurement. Ran `/host/bench_nonstream.py` (the same non-streaming
concurrent-HTTP substitute tool used successfully post-rewrite in
[2026-06-29-cuda-throughput-ceiling-three-models.md](2026-06-29-cuda-throughput-ceiling-three-models.md)),
at the canonical prompt/output shape (4096 in / 256 out):

| c | reqs | out tok/s (aggregate) | "TTFT" p50 (ms)¹ | "TTFT" p95 (ms)¹ | outcome |
|---|---|---|---|---|---|
| 1 | 12 | 42.5 | 5820 | 6645 | OK |
| 4 | — | — | — | — | **server crashed mid-run** (see Problems) |

¹ Non-streaming: this tool's "TTFT" is total request latency (first byte ==
last byte), NOT a true streaming time-to-first-token. Not comparable to a real
guidellm TTFT column — reported only as a directional throughput/latency
sanity check.

Decode-checked (case-as-fact, not assumed from aggregate): two chat-completion
smoke requests before the sweep both correct — "What is the capital of
France?" → "Paris" (16 prompt / 26 completion tokens, 13 steps ⇒ ~2
tok/step avg, consistent with MTP draft_tokens=2 actively accepting drafts,
not a silent no-op); "Count from 1 to 20" → exact correct sequence (17 prompt
/ 74 completion tokens).

## Results — service-side KV / scheduler metrics

Not collected via guidellm (blocked). `/v1/stats` after the c=1 smoke:
`kv_free_pages: 49`, `steps: 13`, `generated_tokens: 26`,
`requests_completed: 1`.

## Results — request accounting

Not collected — no guidellm run.

## Problems

1. **BLOCKER — streaming responses never populate `usage`, on any backend
   (rewrite regression).** `scripts/bench_guidellm.sh`'s preflight probe
   (`probe_streaming_completions`, added
   [2026-04-21](../errors/2026-04-21-guidellm-streaming-metrics-invalid-zero.md)
   against the old `infer/` monolith) requires at least one SSE chunk with a
   populated `usage` object. Verified directly:
   `curl -sN .../v1/completions -d '{"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},...}'`
   returns real, non-empty text chunks but **`"usage":null` on every chunk,
   including the terminal one**. Root cause, read at the source (not
   inferred): `crates/infer-server/src/coordinator.rs:583,608,622` calls
   `completion_stream_chunk(..., usage: None)` unconditionally — the function
   signature (`sse_util.rs:7-23`) accepts an `Option<Value>` but no call site
   ever passes `Some(..)`; `chat_stream_chunk` (`sse_util.rs:25-40`) doesn't
   even have a usage parameter — hardcodes `"usage": null` as a JSON literal.
   `stream_options`/`include_usage`/`continuous_usage_stats` have **zero**
   references anywhere in `crates/infer-server/src` or `crates/infer-api/src`
   — the feature is entirely unimplemented in the rewrite, though the
   2026-04-21 errors doc confirms the pre-rewrite `infer/` monolith had it
   (that doc's fix references `infer/src/scheduler/cuda/request.rs`, deleted
   2026-06-04). **Not DSv4/TP4-specific**: this blocks every canonical
   `scripts/bench_guidellm.sh` run against `infer-server` regardless of
   backend/model. Cross-checked against the wins/ corpus: every
   `bench_guidellm.sh`-tagged entry dated after 2026-06-04 (the monolith
   deletion) is a `pending-remote` stub, none show an executed sweep table —
   this bug has silently blocked the canonical bench pipeline for a month.
2. **Found + fixed en route — guidellm dependency drift.** `pip3 install
   guidellm` (no pin) resolves the current PyPI `0.7.1`/`0.7.0`, whose CLI
   (`guidellm run --backend kind=openai_http,...`) is a **completely
   different, incompatible shape** from what `scripts/bench_guidellm.sh`
   assumes (`guidellm benchmark run --target ... --data "spec-string"`).
   `requirements-bench.txt` pins `guidellm[recommended]==0.7.0` with a stale
   comment ("pin to 0.6.0" — the pin value and the comment disagree); neither
   the comment's 0.7.0 pin nor plain 0.7.1 match the wrapper's assumed CLI.
   Installing the **exact `0.6.0`** (the version the comment actually
   describes) restored the expected `guidellm benchmark run` CLI shape
   end-to-end (confirmed via `guidellm benchmark run --help`). This is
   real and reproducible — not a guess — but is now a landmine for the next
   session unless `requirements-bench.txt` is corrected to `==0.6.0` or the
   wrapper is ported to the current CLI.
3. **TP=4-specific — DSv4 crashes (not gracefully rejects) under c=4 at
   canonical prompt length.** `INFER_DSV4_MAX_SEQ_LEN=16384` (the usual
   default) failed engine build entirely at TP=4: `DSv4 FlashMLA pool page
   mismatch: page_size=64 pages=42 need page_size=64 pages>=66`
   (`crates/infer-cuda/src/attention/kv_layout.rs:1020`) — halving GPU count
   vs the canonical TP=8 roughly doubles per-GPU weight footprint (76377 MB
   of 97871 MB here), starving the post-weights KV budget so badly that the
   budget-clamped slot floor (42) fell below the FlashMLA fixed-band minimum
   for one slot (66) at that context length. Lowering
   `INFER_DSV4_MAX_SEQ_LEN` to 8192 (still 2× the canonical 4352-token need)
   fixed startup (49 slots, clean HTTP open, correct decode). But even at
   8192, **c=4 concurrency crashed the whole coordinator**: all 4 workers hit
   `HostPagedKvPool out of fixed-band pages: slot 1 needs 34, free 15` and
   exited `Some(1)`, tearing down the serve (`coordinator.rs:264`) — this is
   a hard crash, not an admission-control rejection/queue-back. Since
   guidellm's `sweep` profile auto-ramps concurrency, it would hit this exact
   wall even if blocker #1 were fixed — TP=4/EP=4 cannot currently sustain
   the canonical sweep's concurrency range at the canonical 4096-token
   prompt shape; this is a genuine capacity ceiling of the halved-GPU
   config, not a config mistake.

## Learnings

- **A structural/global bug can hide behind a backend-specific bench task.**
  This DSv4-TP4-labeled task surfaced a month-old, all-backend regression in
  the canonical bench pipeline (`usage` never populated in SSE streams) that
  no post-rewrite wins entry had actually hit, because every one either used
  a non-canonical substitute tool (`bench_nonstream.py`) or never got run.
  Worth a fleet-level flag, not just a DSv4 note.
- **"pending-remote" wins entries are not proof the pipeline works** — audit
  showed every post-2026-06-04 `bench_guidellm.sh`-tagged entry was a stub;
  cross-referencing the wins corpus by grep was cheap and load-bearing here.
- **TP=N is not a free linear substitute for TP=8 on DSv4.** Halving GPU
  count doesn't just halve compute — it roughly doubles per-GPU weight
  footprint, which can push post-weights KV budget below hard minimums
  (FlashMLA fixed-band size) that don't exist at the canonical shape. Always
  re-derive `INFER_DSV4_MAX_SEQ_LEN` down when substituting a smaller TP, and
  expect a materially lower safe concurrency ceiling, not just lower peak
  throughput.
- **KV-exhaustion should degrade to admission rejection, not process crash.**
  `HostPagedKvPool out of fixed-band pages` propagating to a worker `exit(1)`
  and tearing down the whole multi-rank coordinator (rather than the
  scheduler queuing/rejecting the offending request) is a robustness gap
  worth a follow-up issue independent of this bench.

## Δ vs baseline

- **Baseline:** none — first DSv4 TP=4 MTP guidellm attempt; the TP=8 attempt
  ([2026-07-06](2026-07-06-bench-guidellm-dsv4-mtp.md)) also produced no
  numbers (GPU1 contention). No prior snapshot to diff against.

## Artefacts

- None from guidellm (blocked before any benchmarks.json was written).
- Server log: `/root/run-dsv4mtptp4.log` (pod, ephemeral).
- Supplementary probe output: inline above (not a persisted artefact).

## Notes

- What changed in code since the commissioning push: nothing DSv4/MTP-related
  touched this session; no code was modified (this is a devops/bench
  execution task — the bugs above are reported, not patched, pending the
  calling session's review and prioritization).
- Follow-ups (not filed as issues by this task — flagging for the calling
  session): (a) implement `stream_options.include_usage` /
  `continuous_usage_stats` in `crates/infer-server/src/coordinator.rs` +
  `sse_util.rs` — unblocks the entire canonical guidellm pipeline for every
  backend; (b) fix `requirements-bench.txt`'s guidellm pin (comment says
  0.6.0, value says 0.7.0, PyPI's actual 0.7.0/0.7.1 CLI is incompatible with
  the wrapper — pin `==0.6.0` explicitly or port the wrapper to the current
  CLI); (c) DSv4 `HostPagedKvPool` exhaustion should reject/queue, not crash
  all multi-rank workers.
- Server was cleanly torn down by its own crash (all 4 GPUs confirmed back to
  0 MiB via `nvidia-smi`); no manual kill was needed. GPU1 (foreign PID
  2094648) was never touched.
