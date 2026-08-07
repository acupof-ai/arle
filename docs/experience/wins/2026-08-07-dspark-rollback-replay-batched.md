# DSpark rollback replay batched: c=16 TPOT −11.4%, p99 −16.9% — CUDA, 2026-08-07

> Status: **accepted.** Counterbalanced A/B, two sweeps per arm in both halves,
> single-commit delta. With
> [`2026-08-07-dspark-verify-linear-core-batched.md`](2026-08-07-dspark-verify-linear-core-batched.md)
> this closes the c≥4 regression opened by
> [`2026-08-06-dspark-anchor-remeasure-c1-plus-40-percent.md`](2026-08-06-dspark-anchor-remeasure-c1-plus-40-percent.md):
> c=16 is now 0.9% ahead of the 07-30 champion it had been 21.5% behind.

## Problem

`dspark_rollback_batch` (`qwen35/dspark.rs:1822`) opened with:

```rust
// The varlen replay is the recurrent kernel; leave the opt-in chunked
// path on its own route.
if super::qwen35_gdr_chunked_enabled() {
    for r in rolls.iter_mut() {
        self.replay_linear_only(r.slot, ws, &r.spec.capture, r.k)?;
        r.slot.set_seq_len(r.start_pos + r.k + 1);
    }
    return Ok(());
}
// batched varlen replay below
```

`--qwen35-gdr-chunked` was opt-in when that was written, so the per-slot loop
was the minority branch. The flag defaulted on in `c2eb5de9e` (08-02, 33K
prefill −27%) and from that day it was the **only** branch:
`replay_linear_only_batched` has been dead code in every shipped configuration
since, and its own doc comment states the cost it was written to avoid — "one
conv1d and one gated-delta launch per layer instead of two per slot per layer.
The launches are sub-100 µs, so the win is their count."

Partial accept fires on most ticks at `accept_rate` 0.30, so the per-slot loop
runs rows × 48 layers × ~6 launches — about 4608 a tick at 16 rows, against 144
for the batched path.

## What worked

Delete the branch. The condition was never a correctness requirement: the
replay restores a pre-verify snapshot and recomputes the accepted prefix, so it
never had to match the trunk's discarded state. And since `4933e1bf4` the
trunk's uniform short rows take this same recurrent varlen kernel, so matching
it is now the more consistent choice as well.

## Parameters

```bash
python3 scripts/gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8

arle serve --backend cuda --model-path ThinkingCap-Qwen3.6-27B-FP8 \
  --spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash \
  --dspark-block-size 6 --max-running-requests 16 --port 8321

python3 scripts/bench_throughput.py --url http://127.0.0.1:8321 \
  --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,2,4,8,16 --requests-per-concurrency 128 \
  --max-tokens 214 --temperature 0 --seed 20260416 --timeout-seconds 900
```

- 1× H20 GPU 6, TP=1, eager, 16 slots.
- Arm B `4933e1bf4`, arm C `70760bc09`. Only `qwen35/dspark.rs` differs.
- **Four sweeps in the order B, C, C, B** so each arm runs once in each half.
  Position was worth +3.4% to +6.3% within an arm here, consistent with the
  earlier finding that the arm loading second measures slower.
- 128/128 complete at every point, all four sweeps, 0 errors.

## Results

Each cell is the mean of that arm's two sweeps.

| c | B TPOT ms | C TPOT ms | Δ TPOT | B p99 ms | C p99 ms | Δ p99 | Δ total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.601 | 8.464 | −1.6% | 31.7 | 31.5 | −0.7% | +0.3% |
| 2 | 19.166 | 18.703 | −2.4% | 226.2 | 205.9 | −9.0% | +8.1% |
| 4 | 35.797 | 33.774 | −5.7% | 531.1 | 519.2 | −2.2% | +5.3% |
| 8 | 66.393 | 62.492 | −5.9% | 582.9 | 571.6 | −1.9% | +3.4% |
| 16 | 124.731 | **110.522** | **−11.4%** | 912.9 | **758.8** | **−16.9%** | **+7.7%** |

The gain grows monotonically with concurrency, which is the shape a
launches-saved-per-row mechanism has to produce. The c=16 tail moves further
than the mean, consistent with what is removed being per-tick orchestration
stall rather than kernel time.

Cumulative for the day at c=16 TPOT:

| commit | TPOT ms | vs morning |
|---|---:|---:|
| `010af0ede` (this morning's baseline) | 124.99 | — |
| `4933e1bf4` (verify core batched) | 117.68 | −5.8% |
| `70760bc09` (rollback replay batched) | **110.52** | **−11.6%** |

The 07-30 champion is 111.53 ms.

## How it was found

Not by reading code. Three nsys captures had all landed in the prefill phase,
where the GPU is ≥76% busy and the dominant GEMM is at 92.7% SM throughput —
nothing to win, and the ledger said so. A capture aimed at pure decode instead
(short prompts, long generation) returned **71.5% GPU idle at 19157 launches/s**,
and the kernel instance counts carried the signature directly: `fq_fwd`,
`fq_kkt`, `fq_cumsum` and `gdr_fq_prep` at 46744 each — four kernels per (row,
layer) is only producible by a per-row loop.

The phase mistake is written up in
[`errors/2026-08-07-measured-prefill-concluded-about-decode.md`](../errors/2026-08-07-measured-prefill-concluded-about-decode.md).

## Learnings

**A default flip is a routing change.** Both of today's wins are the same bug:
a flag defaulting on turned an `if flag { per-row }` minority branch into the
only path. The morning's was `LinearCore::Rows` after `0ac780495` made FlashQLA
real in the pod binary; this one is the rollback replay after `c2eb5de9e`.
Neither failed, neither had a test that could notice, and both survived until a
kernel-instance ledger made the per-row shape visible. When a flag defaults on,
grep its call sites — not just the feature it names.

**Dead code that a comment says is the fast path is worth grepping for.**
`replay_linear_only_batched` carried an accurate description of its own value
and had exactly one caller, behind a condition that could no longer be true.

**Counterbalance, again.** Position was worth up to 6.3% within an arm — more
than the c=4 and c=8 effects. A single-order pair would have been unreadable at
those points.
