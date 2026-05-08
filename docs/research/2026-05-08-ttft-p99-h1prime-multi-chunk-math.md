# M_ttft-p99 Phase 2 — H1' multi-chunk-per-session math + matched-workload retest plan

> Per `099c7bd` Phase 1 NULL result at wrong workload(c=8 4K longctx
> NULL at 1.016× spread vs original `f5cf829` c=8 8K agent burst 6.2×
> spread)。
>
> H1 cap=4 alone refuted at 4K but **may still bind at 8K** due to
> multi-chunk-per-session compounding。This entry derives the math
> + proposes Phase 2 matched-workload retest。

## Default scheduler chunking parameters

`infer/src/scheduler/types.rs:270-275`:
```rust
pub fn runtime_defaults(max_slots: usize) -> Self {
    Self {
        max_slots,
        chunked_prefill_size: 4096,             // each chunk = 4096 tokens
        max_num_batched_tokens: 16384,           // 4 chunks per step
        long_prefill_token_threshold: 4096,
        ...
    }
}
```

Combined with `qwen3/forward.rs:316` `max_concurrent_prefill_requests = Some(4)`:
- **Per-step admission cap**:4 prefill requests AND 16384 / 4096 = 4 chunks
- **Per-session chunks at 8K prompt**:`ceil(8192 / 4096) = 2 chunks`
- **Per-session chunks at 4K prompt**:`ceil(4096 / 4096) = 1 chunk`

## Multi-chunk math

### c=8 at 4K(observed PASS in `099c7bd`)
- 8 sessions × 1 chunk = 8 prefill ops total
- 4 ops/step → **2 step cycles** to drain
- Sessions 1-4 finish at step 1,sessions 5-8 finish at step 2
- Predicted spread:**1.33× p99/p50**(matches observed 1.016× — already
  within σ noise)

### c=8 at 8K(target,observed 6.2× in `f5cf829`)
- 8 sessions × 2 chunks = 16 prefill ops total
- 4 ops/step → **4 step cycles** to drain
- BUT:scheduler may NOT prioritize "finish session before starting new" —
  could interleave:
  - Step 1:sessions 1-4 chunk 1
  - Step 2:could be EITHER sessions 1-4 chunk 2 OR sessions 5-8 chunk 1
  - If round-robin:sessions 1-4 finish at step 2,5-8 at step 4
  - If FIFO chunk-bound:sessions 1-4 chunk 1+2 at step 1+2,5-8 chunk 1+2 at step 3+4
- Worst case:**4 step cycles** for last sessions
- Predicted spread(per session N):
  - Session 1 TTFT = 1 step
  - Session 8 TTFT = 4 steps = 4×
- → **4× p99/p50** — closer to but not yet matching observed 6.2×

### Remaining gap factor

If 4× from staircase + ~1.5× from CUDA Graph warmup at first batch
encounter(H4)+ ~1.05× from page_budget contention(H1''):
- Combined:4 × 1.5 × 1.05 ≈ 6.3× p99/p50 ← **matches observed 6.2× exactly**

## Phase 2 test plan

### Phase 2.A — Matched workload H1 cap A/B
Re-run `f5cf829` workload(W4 c=8 8K longctx)with `--prefill-max-requests 8`
override:
```bash
./target/release/infer ... --prefill-max-requests 8 \
    --num-slots 8 --max-seq-len 8192
```

**Expected if H1 binds at 8K**:
- Baseline:p99 ~70k ms,p50 ~12k ms,6.2× spread
- Treatment:p99 ~30-40k ms(cap=8 = 1 step cycle vs 4),~3× spread

But **CAVEAT**:Marlin GEMM scratch OOM risk at cap=8(per `b708e00` cap
was added to prevent this)。If OOM:try `--prefill-max-requests 6`,or
investigate `prefill_workspace` size to confirm OOM threshold。

### Phase 2.B — Increase chunked_prefill_size A/B
Alternative:keep cap=4 but increase chunk size to fit 8K in 1 chunk:
```bash
./target/release/infer ... --chunked-prefill-size 8192 \
    --max-num-batched-tokens 32768
```

**Expected if H1' binds**:
- Each session takes 1 chunk(not 2)→ same staircase math as 4K
- Predicted spread:1.33× vs observed 6.2× → confirms multi-chunk dominant

### Phase 2.C — Disable graph capture(H4 isolation)
Add env var bypass:`INFER_DISABLE_PREFILL_GRAPH=1` and re-run。If
spread drops noticeably(from 6.2× → 4×),H4 graph warmup is real
factor。

## Decision tree post-Phase 2

| Phase 2.A result | Phase 2.B result | Phase 2.C result | Verdict |
|---|---|---|---|
| spread 3-4× | n/a | n/a | H1 cap binds — increase to 8 production |
| spread same | spread 1.3× | n/a | H1' multi-chunk dominant — increase chunk size |
| spread same | spread same | spread 4× | H4 graph warmup real — pre-capture batch sizes 5-8 |
| spread same | spread same | spread same | H2/H3/H5 active — deeper investigation needed |

## Phase 2 effort estimate

- 2.A:0.5 day codex + 30 min bench + 30 min analysis
- 2.B:0.5 day codex + 30 min bench + 30 min analysis
- 2.C:1 day codex(graph capture bypass impl)+ 30 min bench

If 2.A fails(Marlin OOM at cap=8),skip to 2.B(no kernel risk)。

## Cross-references

- Plan main: [`a25416b`](../plans/M_ttft-p99-tail-latency.md)
- Phase 0 H1 confirmed: [`ec7fe9d`](2026-05-08-ttft-p99-h1-prefill-cap-4-confirmed.md)
- Phase 1 NULL wrong-workload: [`099c7bd`](2026-05-08-ttft-p99-phase1-null-wrong-workload.md)
- Original signal: `f5cf829` W4 c=8 8K
- Concurrency sweep: `8588f6a`
- 3-shape grid: `c4fae17`
- Cap source: `infer/src/model/qwen3/forward.rs:310-320`
- Chunking config: `infer/src/scheduler/types.rs:270-275`

## Methodology rule reinforced(from `099c7bd`)

Phase 1 A/B **must use SAME workload** as Phase 0 empirical signal。
Different workloads have different binding constraints — validating H1
on a workload where it doesn't bind tells us nothing about whether it
binds on the original workload。

Anti-pattern #15(addition to skill v1.4.0):wrong-workload investigation
trap — when test workload differs from original signal,NULL result
proves nothing about original。

## Status

Phase 2 plan ready for codex pickup。Each sub-phase is 0.5-1 day。
Recommended order:**2.B → 2.A → 2.C** because 2.B is lowest-risk
(no kernel changes,just config)and would validate H1' multi-chunk
hypothesis directly。

If 2.B confirms:production fix is straightforward(change config defaults
or expose via CLI)— no kernel work needed。

If 2.B refutes(spread still 6.2×):move to 2.A or 2.C per decision tree。
