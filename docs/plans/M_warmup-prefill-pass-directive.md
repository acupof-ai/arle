# M_warmup prefill pass directive — concrete codex pickup

> Per `641e9bf` cap=8 bimodal trigger CONFIRMED:`c20b1ce` covers DECODE
> only,PREFILL paths cold-start in first burst → 76-92%/56% bimodal。
>
> This directive provides concrete implementation path for codex pickup,
> ~100-150 LOC,1 day。Predicted:**100% turn success deterministic**。

## Problem(per `641e9bf`)

`warmup_cuda_graphs`(`infer/src/scheduler/cuda/core/warmup.rs:26+`):
- Lazy-inits `decode_bufs` only
- Pass 1 calls `warmup_graphs_pass` for decode batch sizes
- **No prefill code path warming**

Result:first 8-10 sessions in cold-start burst hit:
- Marlin GEMM JIT compilation
- cublasLt heuristic search
- FP16 GEMM scratch allocator slop
- Attention prefill kernel first-encounter
→ 1-5s admission cascade → 503 retry exhaustion

## Implementation outline

### Step 1 — Existing infrastructure to reuse

`ModelForward::forward_prefill_batch_with_pool`(per code-grep):
- Already exists in qwen3/qwen35/deepseek
- Takes `&[PrefillBatchRequest]`,`&mut [Self::State]`,`&mut PagedKVPool`
- Returns Result<bool>

### Step 2 — Add prefill warmup pass to `warmup.rs`

After existing decode warmup pass,add Pass 2:

```rust
// Pass 2 (per 641e9bf): prefill warmup. Drive forward_prefill_batch
// for each batch size up to prefill_cap to pre-populate:
// - Marlin GEMM kernel JIT compilation
// - cublasLt heuristic for prefill GEMM shapes
// - FP16 GEMM scratch allocator
// - Attention prefill kernel
// Eliminates first-burst admission tax (1-5s) at cap=8 cold-start.
let prefill_cap = self
    .model
    .max_concurrent_prefill_requests()
    .unwrap_or(num_slots)
    .min(num_slots);

if prefill_cap > 0 {
    info!("Warming up prefill code paths (1..={} batch sizes)...", prefill_cap);
    let dummy_prompt: Vec<u32> = vec![0u32; 64]; // ~64 tokens, realistic short prefill
    let prefill_t0 = std::time::Instant::now();

    for bs in 1..=prefill_cap {
        let requests: Vec<PrefillBatchRequest<'_>> = (0..bs)
            .map(|slot| PrefillBatchRequest {
                slot_idx: slot,
                tokens: &dummy_prompt,
                // ... other fields per actual struct
            })
            .collect();
        let states_slice: &mut [_] = &mut self.states[..bs];
        if let Err(e) = self.model.forward_prefill_batch_with_pool(
            &requests,
            states_slice,
            &mut self.paged_kv_pool,
        ) {
            warn!("Warmup prefill bs={} failed: {} (skipping larger sizes)", bs, e);
            break;
        }
    }
    info!("Prefill warmup done in {}ms", prefill_t0.elapsed().as_millis());
}
```

### Step 3 — Reset KV pool after prefill warmup

Prefill warmup writes dummy tokens to KV pool。Must reset before
serving real requests:

```rust
// Reset KV pool (prefill warmup wrote dummy tokens)
self.paged_kv_pool.reset_all_slots();
for state in &mut self.states {
    state.reset_for_warmup_clear();
}
```

May need `reset_for_warmup_clear` method added to State trait if not
present。Per existing decode warmup,similar reset already exists post
decode warmup — check current pattern + replicate。

### Step 4 — Validation

After landing:
```bash
cargo build --release -p infer --features cuda

# Re-run W4 c=8 8K bench fresh build N≥3
INFER_LOG_LEVEL=info ./target/release/infer ... # cold start
# Expected log:
#   "Warming up CUDA Graphs for 16 batch sizes (max 16)..."
#   "Prefill warmup done in <expected ~1500ms> ms"

# Run bench × 3
for i in 1 2 3; do
    python scripts/bench_agent_trace.py --workload w4-c8-8k --label warmup-pf-fix-run$i
done

# Expected: 100% turn success across all 3 runs
```

If 100% across N=3 → LICENSE the fix + close cap=8 bimodal investigation。

### Step 5 — Cost analysis

Cold-start time addition:
- Decode warmup currently:~10 sec(per existing `8281047` log)
- Prefill warmup added:~1-3 sec(8 batch sizes × 100-300 ms each)
- **Total cold-start growth:~10-30%**(11-13 sec total vs 10 sec before)

Pays back via:
- 100% turn success at c=8 8K(vs 56-92% bimodal currently)
- No 1-5s first-burst admission cascade
- Saves 30-60s of bench wall time per run that was wasted on retries

## Edge cases / risks

1. **`prefill_cap = num_slots` corner**:if model returns None,clamp to
   `num_slots`。Current code does this correctly。
2. **Marlin scratch OOM at cap=8**:per `b708e00` original concern。
   Warmup has all slots allocated already → if prefill at cap=8 OOMs
   during warmup,we fail SAFELY at startup(better than crash mid-bench)。
3. **State reset bug**:if reset_all_slots fails,subsequent serving
   leaks dummy tokens into real requests → catastrophic。Test
   thoroughly。Could add a `health_check` post-reset。
4. **Per-batch JIT compilation lag**:first prefill at bs=1 may take
   500ms+,bs=2 onwards faster。Total estimate 1-3s but worst-case 5s。

## KILL criteria

- **Build failure**:if forward_prefill_batch_with_pool signature different
  than expected → drop in placeholder,investigate per-impl
- **Validation N=3 still bimodal**:after fix,if turn success not 100%
  across all 3 → there's another factor beyond prefill warmup;continue
  investigation per `f7da3e1` H_sched/H_harness/H_alloc hypotheses
- **Marlin scratch OOM at cap=8 prefill**:revert,reduce cap to 6 or 4

## Cross-references

- `641e9bf` prefill warmup gap analysis(this directive's source)
- `61ebf45` first-burst session 0-9 failure pattern empirical
- `c20b1ce` decode-only warmup(complement)
- `12300c5` cap=8 flip
- `19d12c2` warm-server reference(100% baseline)
- `qwen3/forward.rs:363` `forward_prefill_batch_with_pool` reference
- `warmup.rs:26+` `warmup_cuda_graphs` function

## Status

Concrete codex pickup directive ready。Effort estimate:
- Implementation:1 day(150-200 LOC including reset logic + tests)
- Validation:0.25 day(N=3 fresh-server bench)
- Wins entry:0.25 day(if PASS)

Total:1.5 days codex。Closes cap=8 bimodal investigation。

If codex picks this up next:start with Step 2(implementation)→ Step 4
(validation)。Step 5(cost analysis)goes in commit body。

## Methodology rule

Per `641e9bf` skill candidate addition:**when bimodal pattern persists,
go granular per-failure-event** instead of aggregate metrics。Per-session
failure ID parsing(`61ebf45`)isolated trigger after 5 ticks of variance
hypotheses。

This rule could become anti-pattern #18 in skill v1.5.0 → v1.6.0 if
codex agrees。
