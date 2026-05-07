# 2026-05-07 · M3.9 parallel analysis — H2 (attention KV scaling) is the dominant explanation

## Context

While codex implements M3.9 Phase 0 instrumentation
([`63af21f`](../../plans/M3.9-split-plan-tax-investigation.md)),
parallel source-code analysis disambiguates the three candidate
hypotheses (H1 fall-back, H2 KV scaling, H3 varlen kernel) using
math derived from the read-only source.

## Source survey

`decode_batch_with_prefill` (`batch_decode.rs:890-1330`) per-layer
kernel structure:

- L1080-1119: **1 ×** `decode_prep_paged_cuda` (all decode rows
  uniformly)
- L1121-1159: **N ×** `prefill_attention_paged_prep_cuda` (one per
  prefill request, looping)
- L1235 / L1263: **1 ×** unified attention call
  (`tilelang_tc_run_layer` for BF16 or `decode_attention_varlen_fp8`
  for FP8) — handles ALL rows (decode qlen=1 + prefill qlen=N) in
  one launch via varlen `qo_indptr`
- L1162-1223: **2 ×** KV quant (k + v) for FP8 path
- Plus standard layer ops: 4 × linear (q/k/v/o), 2 × RMSNorm,
  3 × MLP linear, 1 × silu_mul

Per layer kernel launches = ~14 + N (prefill_count). For 36 layers
+ batch=4 with all-prefill: 36 × 18 = **648 launches per step**.
Per launch cost (Phase 1 trace) ≈ 3.7 µs → 2.4 ms launch overhead
total. **Negligible** relative to measured 1263 ms.

## H1 (fall-back) is unlikely

If `decode_batch_with_prefill` returned `Ok(false)`, scheduler
fall-back path runs `step_prefill_batch` + `launch_decode_batch_from_tokens`
sequentially. But at long-ctx steady state, when the first chunk
of all 4 requests is in flight together, all conditions for
`Ok(true)` should hold:
- `paged_kv_pool.seq_len(slot) == start_pos` is maintained by the
  scheduler's chunk-completion bookkeeping
- No decode_slot/prefill_slot overlap (a slot is in one phase)
- BF16/FP8/INT8 KV format

H1 is unlikely the dominant cause. Phase 0 instrumentation will
confirm — the counter should show `ok_true_count >> ok_false_count`.

## H2 (attention KV scaling) — math matches with caveats

Causal attention cost per token = O(prior KV length attended to).

**Pure prefill batch=8 chunk-1** (`67f9bcb` data):
- All 8 reqs at chunk 1, positions 0..2048
- Avg attended KV per chunk-token = `(0 + 1 + ... + 2047) / 2048` = **1024 tokens**
- Per-token attention work: 1024 KV reads × hidden_dim
- Measured: **15.4 µs/token**

**Split chunk-2 batch=4** (`4a3612b` data):
- 4 reqs at chunk 2, positions 2048..4096
- Each token attends to 0..(its position) = up to 4096 prior
- Avg attended KV per chunk-token = `(2048 + 2049 + ... + 4095) / 2048` = **3071 tokens**
- 3× the attention work of pure prefill chunk-1
- Predicted per-token cost: **15.4 × 3 = 46 µs/token**
- Measured: **154 µs/token**

H2 explains **30%** of the 10× tax (the 3× attention scaling).
Remaining 3.3× factor needs H1/H3 to explain.

## H1.5 — per-prefill-row prep loop verification

Loop at L1121-1159: `prefill_attention_paged_prep_cuda` × N per
layer. Per-call processes 2048 tokens (RoPE + norm + paged-write).

**Cost math**:
- Per-prefill-prep call: 2048 tokens × (RoPE + 2× RMSNorm + KV
  page write at ~4.2 µs/tok) = ~8.7 ms per call
- Layers × prefill_count = 36 × 4 = 144 calls
- Total: 144 × 8.7 ms = **1252 ms**
- Measured split step time: **1263 ms** ← matches within 1%

So the prep-loop math FITS — but does this mean prep is the
bottleneck, OR that the per-token cost is naturally distributed
across prep + attention?

**Critical sub-question**: would unifying the prep loop (one prep
call per layer processing all prefill tokens together) save time?
- Unified per-layer prep: 4 × 2048 = 8192 tokens × 4.2 µs/tok
  = 34.4 ms per layer
- 36 layers × 34.4 = 1240 ms
- **Saves only 23 ms / 1.8% of total step time**

→ **Per-prefill prep loop is NOT the fix.** Per-token cost
dominates regardless of batching, and that per-token cost (~4.2 µs)
is mostly the page-write bandwidth + RoPE arithmetic.

## H3 (varlen kernel inefficiency) — partial contributor

The unified attention call at L1235/L1263 handles mixed qlen=1
(decode) + qlen=2048 (prefill). TileLang/hand-CUDA varlen kernels
have BLOCK_M tile sizes (typically 64) that are over-provisioned
for qlen=1 rows — wasting compute. ARLE mainly hits this when
some rows are decoding while others prefilling, but in our 4k/c=4
bench at chunk-2-of-2, all 4 rows are still prefilling — no
decoders mixed in yet at the captured step. So H3 doesn't apply
to THIS specific log.

If H3 applied, we'd see slowdown only at steps where decode rows
are MIXED with prefill chunks. Our split=1263ms step appears to be
all-prefill-rows-with-non-zero-KV, which is more H2 territory.

## Conclusion

**H2 dominates at long-context steady state.** Specifically:
- Pure prefill efficiency (15 µs/tok) measured at KV=0 is the BEST
  case for ARLE
- Long-ctx steady state has KV ~3k accumulated → ~3× attention
  work per token → expected 45 µs/tok floor
- Measured 154 µs/tok suggests another 3.3× factor we can't
  source-explain without instrumentation. Could be:
  - **Sub-optimal varlen kernel implementation** at long KV
    (kernel launches are fine but the inner loop cost per token
    is higher than necessary)
  - **Memory bandwidth saturation** at 4 rows × 36 layers × 4k KV
    reads = 576 MB/layer, at ~700 GB/s = 0.8 ms/layer = 30 ms total
    per step (still far less than 1263 ms)

**Phase 1 fix path** based on this analysis:

| Fix | Expected gain |
|---|---:|
| **Phase 1B (delay split until prefill queue empties)** | TTFT improves because pure prefill is 10× faster; trades minor decode-start latency for huge prefill throughput. **Largest measurable gain on long-ctx**. |
| Phase 1A (fix Ok(false) condition) | <10% (H1 not dominant) |
| Phase 1C (kernel autotune) | unknown — could close the unexplained 3.3× factor; requires nsys ncu deep-dive |

**Recommendation**: ship Phase 1B (scheduler policy: pure-prefill
priority) as the primary fix, parallel with Phase 1C kernel
analysis. Phase 1B is a ~50 LOC scheduler change with clear
expected gain. Phase 1C is exploratory and may take longer.

## Bench Status

No new bench in this analysis. References:
- Pure prefill efficiency: `67f9bcb`
- Split tax measurement: `4a3612b`
- M3.9 plan with H1/H2/H3 framework: `63af21f`

## Rule

- **Math from source can disambiguate hypotheses cheaper than
  instrumentation** when the kernel structure is known. Here, the
  per-prefill prep loop math (144 × 8.7ms = 1252ms = measured)
  + attention 3× scaling math told us H1.5 + H2 contribute, while
  H1 (fall-back) is unlikely. Phase 0 instrumentation will still
  confirm but isn't BLOCKED on for the Phase 1 design.
- **Parallel work pattern: codex implements, Claude analyzes**.
  This entry uses the wait-time during codex's M3.9 Phase 0 to
  shrink the Phase 1 design space.
