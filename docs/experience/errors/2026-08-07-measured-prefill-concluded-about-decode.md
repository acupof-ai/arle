# Measured the prefill phase, concluded about decode — killed the right lever — CUDA, 2026-08-07

## Context

Chasing why aggregate decode throughput is flat in batch size on the DSpark
long-agent anchor: `B / TPOT` gives 116.7 / 107.2 / 112.4 / 123.8 / 136.0 tok/s
at B = 1 / 2 / 4 / 8 / 16. Sixteen times the batch buys 1.17x the throughput.

Four hypotheses were tested and all four were reported dead:

| hypothesis | evidence used | verdict I gave |
|---|---|---|
| host-orchestration / launch-bound | nsys: GPU busy >= 76% | dead |
| prefill x decode fusion | `--chunked-prefill-size` 2048 -> 512 moved TPOT 0.1% | dead |
| GEMM kernel efficiency | ncu: SM throughput 92.7% | dead |
| prefix cache ineffective | 74K tokens forwarded vs 58K ideal | dead |

Then a capture aimed at the decode phase returned:

```
PURE DECODE: window 30.75 s | kernel 8.78 s | GPU busy <= 28.5% | kernels 589071
```

**71.5% GPU idle, 19157 kernel launches per second.** The first hypothesis was
correct. I killed it with a measurement of the wrong phase.

## Root cause

Every nsys window I captured landed in **prefill**. The long-agent dataset is
32K prompts x 8 turns, so prefill dominates wall clock, and a capture that
opens at an arbitrary point lands there by default. Three separate captures
all did.

The two phases have opposite characteristics, and no amount of care within one
of them reveals the other:

| | prefill | decode |
|---|---:|---:|
| GPU busy | >= 76% | <= 28.5% |
| dominant kernel | compute-saturated (ncu: 92.7% SM throughput) | 16.1 us `fq_fwd`, barely above launch cost |
| bound by | arithmetic | host launch rate |

"GPU busy >= 76%, therefore not host-bound" is true of the window I measured
and false of the phase I was reasoning about.

## The second error, in the experiment that "confirmed" it

`--chunked-prefill-size` 2048 -> 512 moved TPOT by 0.1% and p99 by -0.6%, which
I read as "decode does not wait behind prefill". That is not what the knob
tests. From `infer-core/src/planner.rs:55-100`:

- `prefill_step_budget()` = 16384 tokens per tick, across all prefill rows
- `chunked_prefill_size` caps ONE row's chunk
- `max_concurrent_prefill()` = `running_cap()` = 16 rows

So chunk 2048 gives 8 rows x 2048 = 16384 (budget binds), and chunk 512 gives
16 rows x 512 = 8192 (row cap binds). Per-tick tokens halve while the forward
count doubles — two effects of opposite sign on step latency. A null is what a
confounded design produces, not a result.

The knob that isolates step latency is the per-tick token budget
(`max_num_batched_tokens`, Sarathi-Serve's "token budget"), which had no CLI
path at all: a hardcoded 16384 in `SchedulerConfig::default()` that nobody
chose and nobody could test. Exposed in `ed92c6d8c`.

## Fix

Aim the capture at the phase being reasoned about, and prove it landed there
before reading the ledger. Short prompts (~1000 tok) and long generation (512)
push prefill below the noise.

## Rule

**A phase is not a workload.** Before drawing a conclusion from a profile,
state which phase the window covers and show the evidence it covers that phase.
A bimodal workload gives opposite answers in its two modes, so a measurement
that does not name its phase cannot support a claim about the system.

Corollary that cost the most here: **the hypothesis a measurement kills must be
one that measurement could have confirmed.** GPU-busy in a prefill window
cannot confirm or deny launch-boundedness in decode, so it was never evidence
either way.

## Other traps from the same investigation

**`ARLE_QWEN35_PROFILE` parent ranges are inflated.** Each leaf range ends in
`stop.synchronize()`, and a parent range wraps its children, so it absorbs every
child's sync bubble. Only leaves are real measurements. Leaves:
`dense_ffn`, `linear/{in_proj,out_proj,gdr_fq,gdr_recurrent,conv1d,norm,allreduce}`,
`full_paged/{qkv_gemm,o_proj,attention,prep,gate,allreduce}`, `input_norm`,
`post_attn_norm`, `ffn_residual`, `ffn_allreduce`, `embedding`. The same
profiler is untrustworthy at small `seq` for a different reason: 705 sync points
on microsecond kernels. It is trustworthy at large `seq` — the 560-token
forward's 165 TFLOP/s matches the bench's own cold TTFT (160 TFLOP/s)
independently. Count forwards as `input_norm` instances / 64.

**Kernel instance counts are not comparable across lanes with different
denominators.** `fq_fwd` fires once per row per layer; the batched varlen GDR
fires once per layer. Reading 46744 against 4896 as "the batched lane fires 10%
of the time" is wrong by the row count — the two have different units. Neither
number was actionable without a lane counter, which does not exist.

**nsys CLI, three failed cycles:** `nsys profile --attach <pid>` is not an
option; `nsys start` does not accept `--trace` (tracing is fixed at `nsys
launch`); and a `profile --delay` cannot be aimed at a server whose ready time
swings 15 s to 1173 s with page-cache state. The working shape is
`nsys launch --session-new=S --trace=... <cmd>` then `nsys start --session=S -o
FILE` / `nsys stop --session=S`. Assert the `.nsys-rep` exists after `stop`;
the failure is otherwise silent until `nsys stats` reports a missing file.

## What the decode ledger found — a default flip that rerouted a batched path

`dspark_rollback_batch` (`qwen35/dspark.rs:1822`) opened with:

```rust
// The varlen replay is the recurrent kernel; leave the opt-in chunked
// path on its own route.
if super::qwen35_gdr_chunked_enabled() {
    for r in rolls.iter_mut() {
        self.replay_linear_only(r.slot, ws, &r.spec.capture, r.k)?;
        ...
    }
    return Ok(());
}
// batched varlen replay below — unreachable
```

`--qwen35-gdr-chunked` was opt-in when that branch was written. It defaulted on
in `c2eb5de9e` (08-02, 33K prefill −27%), and from that day the per-slot loop
was the only route: `replay_linear_only_batched` became dead code in every
shipped configuration. Partial accept fires on most ticks at `accept_rate`
0.30, so the cost is rows × 48 layers × ~6 launches — about 4608 a tick at 16
rows, against 144 for the batched path.

**This is the second instance of the same shape today.** The morning's fix was
`LinearCore::Rows` running a per-row loop that only started being paid when
`0ac780495` made FlashQLA real in the pod binary. Both are: a flag flip
silently re-routes a batched path to a per-row one, and nothing fails.

Rule: **a default flip is a routing change.** When a flag defaults on, every
`if flag { ... }` that was written as the minority branch becomes the majority
branch — grep the flag's call sites at the flip, not just the feature it names.
Neither of these had a test or a counter that could notice, which is why both
survived until a kernel-instance ledger made the per-row shape visible.

## What sizing-before-building saved

Two changes were priced and dropped before any code was written:

- **Prefill-prefill fusion**: 16 concurrent prompts produced 16 separate
  forwards, which looked like a 16x redundancy. Prefill is compute-bound
  (19.2 TFLOP against 6.7 ms of weight traffic at 560 tokens), so fusing saves
  only the 15 redundant weight reads — ~3% — and the FLOPs are identical.
- **The dominant GEMM**: ncu put `sm90_fp8_gemm_1d2d_impl` at 92.7% SM
  throughput. There is nothing in it to win.

Both would have been substantial edits to delicate paths.
