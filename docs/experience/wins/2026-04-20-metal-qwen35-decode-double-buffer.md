# Metal Qwen3.5 — cross-step double-buffering in Qwen35StepDriver

**Date**: 2026-04-20
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA), macOS 26.3.1
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` (no DFlash for this bench)
**Commit**: this commit

## Goal

Close the last /loop-reachable lever on c=1 decode latency. Earlier analysis
(`2026-04-19-metal-qwen35-final-state.md`) ruled out the `gated_delta_step`
6.1 ms/row wall without an Xcode capture. The step-driver FFI path at c=1
was still synchronous per-token: `.item_i32()` on each step's sampled token
blocked the CPU before the next step's graph could be built, leaving the
GPU command queue idle between steps. mlx_lm's canonical trick keeps the
queue one step deep by dispatching step N+1 using step N's (not-yet-
materialized) sampled tensor, and only materializing step N's scalar after
N+1 is enqueued.

## Hypothesis

`Qwen35StepDriver` already had partial infrastructure (`pending_sampled:
Option<MlxArray>` pre-queue) but line 3475 materialized the current step's
token synchronously **before** building the next step, collapsing the
overlap. Moving the materialization **after** the next async_eval should
keep the GPU busy continuously and reduce per-step CPU idle time.

## Setup

Built `metal_bench --release --no-default-features --features metal` with
the patch applied. Matched A/B via git stash on the same build session:
- A = pre-patch (stash applied)
- B = post-patch (stash popped, includes the off-by-one fix for the
  hot-path prequeue guard flagged by `codex review`)

Bench command:
```
./target/release/metal_bench \
  --model $QWEN35_PATH \
  --prompt-tokens 32 --generation-tokens 256 \
  --warmup 3 --runs 5 --use-step-driver
```

Same process stays alive across all 5 timed runs; no re-load.

## Results

### Step-driver (tight-loop FFI path, c=1)

| Metric              |      A |      B |   Δ   |
|---------------------|-------:|-------:|------:|
| Gen tok/s mean      |  75.8  |  85.4  | +12.7% |
| Gen tok/s p50       |  76.4  |  85.3  | +11.6% |
| Gen tok/s p99       |  77.5  |  85.7  | +10.6% |
| Repo E2E tok/s mean |  74.6  |  83.8  | +12.3% |
| TTFT mean           |  54 ms |  54 ms |   0   |

Implied TPOT: ~13.2 ms → ~11.7 ms. The "5.2 ms TPOT c=1" figure from the
2026-04-19 HTTP baseline refers to the HTTP path, not the step-driver
bench — see below for the HTTP carry-over.

### HTTP concurrent (guidellm, `prompt_tokens=1024,output_tokens=256`, 30s each)

| c | A TPOT mdn | B TPOT mdn |  Δ   | A out tok/s agg | B out tok/s agg |
|---|-----------:|-----------:|-----:|----------------:|----------------:|
| 1 |    5.24 ms |    5.10 ms |  -3% |            63.8 |            66.2 |
| 2 |   22.98 ms |   23.20 ms |  +1% |            58.5 |            58.1 |
| 4 |   59.74 ms |   60.30 ms |  +1% |            56.5 |            56.7 |

c=1 carries a small but consistent improvement (+3-4% per-stream gen). c=2
and c=4 are flat within noise — expected, because those concurrency levels
go through `execute_qwen35_packed_decode_batch` (runtime.rs:1138), a
different code path that does not use `Qwen35StepDriver::decode_token`.

Source data: `/tmp/bench_regress_B/benchmarks.json` (A, this morning's
regression check), `/tmp/bench_dbuf_C/benchmarks.json` (B, post-patch).

## Interpretation

* Step-driver +12% is the real number — it's the tight FFI loop with no
  HTTP overhead, and the optimization's target audience.
* HTTP c=1 only gets +3-4% because per-token HTTP streaming (tokenizer
  decode, SSE frame construction, tokio scheduler yields) adds overhead
  that is no longer hidden behind GPU execution. The `.item_i32()` wait
  was ~1.5 ms at c=1; saving it helps the step-driver proportionally
  more than HTTP.
* c=2 and c=4 unchanged is architecturally correct, not a dud. To carry
  double-buffering into the packed batched path would be a separate,
  larger change (the batched path already overlaps across requests, so
  the marginal benefit is unclear without measurement).
* Patch is Rust-only (`Qwen35StepDriver::decode_token`, ~47 lines);
  C++ FFI, scheduler, and DFlash paths untouched.

## Problems

* The bench environment's c=1 HTTP TPOT baseline drifts 4-5% session-to-
  session purely from thermal/cache state. The +3% HTTP delta is below
  that noise floor as a *single* measurement — held across consecutive
  A/B runs only because they were back-to-back in a warm session. Future
  regression checks need to compare against a fresh matched A/B, not the
  published wins-doc numbers alone.
* `cargo clippy --features metal` fails to rebuild `mlx-sys` in the
  clippy profile (env-specific cmake config drift). `cargo build
  --release --features metal` is clean. Pre-existing environmental
  issue, not introduced by this patch.
* No Metal-gated integration test covers step-driver output-token
  equality — the 326 lib tests + 5 consecutive 256-token bench runs
  with stable repo-e2e are the proxy. Worth a follow-up to add a
  deterministic (greedy-only) Metal-feature-gated smoke test.

## Rule

**Before assuming a kernel / hardware ceiling, audit whether the host side
is sync-blocking on GPU work that could be pipelined.** The "6.1 ms GDR
kernel" terminal state was real for aggregate throughput but masked a
separable CPU-idle tax in the single-request decode path. `.item_i32()`
is the telltale — if it runs between two graph-build calls, they're
serializing when they don't need to.

The mlx_lm pattern generalizes: for any session-API decode (single token
per step, model fits in one command buffer), keep the sampled output of
step N as a live MlxArray, feed it as the input token of step N+1 via
reshape (no CPU round-trip), async_eval step N+1, and only THEN
materialize step N's scalar. The command queue stays one step deep.

## Cross-refs

- `infer/src/backend/metal/request_state.rs:3459-3521` — patched
  `Qwen35StepDriver::decode_token` with cold-path and hot-path variants.
- `docs/experience/wins/2026-04-19-metal-qwen35-final-state.md` — the
  "terminal state" entry this result amends (step-driver path was not
  exhausted; HTTP-level ceiling still binds at c≥2).
- `docs/plans/metal-gdr-kernel-xcode-capture.md` — next lever in sequence;
  this patch reduces the c=1 baseline the GDR capture will compare against.
- Codex review flagged P2 off-by-one in the hot-path guard (fixed in this
  commit before merge).
