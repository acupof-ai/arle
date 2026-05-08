# E1 — TileLang decode HD128 BLOCK_M sm_89 re-tune

> **Status**: SCAFFOLD pending [`E5 ncu baseline`](../research/2026-05-08-tilelang-hd128-ncu-baseline.md) data.
> Acceptance numbers, expected Δ%, and final BLOCK_M choice get filled
> in once §5 of E5 returns numbers.
>
> **Trigger**: E5 §5 must be populated and §6 row "Decode `sm__warps_active` < 20%"
> must fire (decode kernel actually thread-idle bound). If E5 shows decode is
> already HBM-bound or compute-bound, **E1 is KILLED, not landed**.

## Priority & ROI

**Priority**: P1 inference (master §7) — kernel-time win on tool-call short
output. Decode is the dominant phase for agent tool-call response (50-500
output tokens), so per-row decode efficiency directly moves agent ITL.

**ROI basis** (predicted, validate against E5 §5):

- Current decode HD128: `BLOCK_M=64` padded for `qo_len=1` → only 1/64
  rows is real work, others are masked. **Predicted utilization 1.5-3%**.
- Reducing `BLOCK_M` to 16 (= PAGE_SIZE) gives 4× rows per block of useful
  work. To 8 gives 8× nominally, but launch grid grows so wall-clock gain
  caps at ~3-5×.
- Decode kernel time at 4k/c=4 is **TBD ms/step (E5 §5)**. If decode is
  ~10-30% of agent ITL p50, this is +2-10% ITL improvement on tool-call workload.

**Negative case**:

- If E5 shows decode kernel is HBM-bound (KV reads dominate), reducing
  BLOCK_M doesn't help — we're already saturating HBM. Kill E1.
- If decode kernel is < 5% of total step time, even a 3× kernel speedup
  only moves end-to-end < 2%. Defer or bundle with E2.
- TileLang 0.1.9 codegen at smaller BLOCK_M may regress correctness
  (precision-pad rules baked into FlashAttention scaffold). Numeric
  divergence > 1e-2 vs current → kill.

**Kill criteria** (any → KILL E1):

1. E5 occupancy ≥ 50% on current BLOCK_M=64 (not actually idle-bound)
2. E5 HBM > 70% peak on decode (memory-bound, not utilization-bound)
3. A/B numerical divergence > 1e-2 on greedy_consistency
4. A/B wall-clock improvement < 5% (within bench noise)

## Phase plan

### Phase 0 — codegen sweep (~1 hr, no GPU bench)

For each candidate `BLOCK_M ∈ {32, 16, 8}`:

- Edit `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py`
  changing only `BLOCK_M = N` (per candidate).
- `cargo build --release --features cuda` to confirm cubin emit
  succeeds for all SUPPORTED_HEADS × SM target combinations.
- Sister kernel `batch_decode_paged_hd128_fp8.py`: keep BLOCK_M
  in lockstep — apply same change, build, confirm.
- HD256 decode (`batch_decode_paged_hd256.py`): **out of scope for E1**.
  HD256 only used by Qwen3.5; defer to a sibling task if data warrants.

If any candidate fails codegen → record in §Problems and proceed with
the survivors.

### Phase 1 — numerical correctness (~30 min, GPU brief)

For each surviving candidate:

1. `cargo test --release --features cuda --test e2e -- --test-threads=1`
2. `cargo test --release --features cuda --test greedy_consistency`
3. If both green, candidate proceeds to Phase 2. Otherwise record
   divergence → kill candidate.

### Phase 2 — bench A/B (~30 min, GPU)

Per-candidate matched A/B against current `BLOCK_M=64` baseline. Use
exact same workload as E5 (4k longctx c=4) for direct attribution.

```bash
# Baseline (current BLOCK_M=64)
git checkout origin/main -- crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py
cargo build --release --features cuda
scripts/bench_guidellm.sh decode-blockm-64 --workload longctx-4k --concurrencies 4 --max-seconds 120

# Candidate (BLOCK_M=N)
# (re-edit file, rebuild)
scripts/bench_guidellm.sh decode-blockm-N --workload longctx-4k --concurrencies 4 --max-seconds 120
```

Report: TTFT p50/p99, ITL p50/p99, tok/s — Δ% per row.

### Phase 3 — pick winner + commit (~15 min)

1. Best candidate by ITL p99 with Δ tok/s ≥ +5% threshold.
2. Update `BLOCK_M` in both BF16 + FP8 decode kernels in lockstep.
3. Wins entry: `docs/experience/wins/2026-05-08-e1-decode-hd128-blockm-retune.md`
   citing E5 baseline + Phase 2 numbers.
4. Cross-link wins in this plan + master §10 (mark
   "FlashInfer vs TileLang HD128 kernel-time" partial-resolution path).

## Acceptance

- ITL p99 improves by ≥ 10% on agent-shape decode (4k longctx c=4)
- No e2e/greedy_consistency regression
- σ < 5% across n=3 reruns
- Wins entry committed with full A/B table, σ, env, raw artifacts
- TileLang BLOCK_M change is `git diff` ≤ 5 lines per kernel file (just
  the constant)

## Cross-references

- E5 baseline (must read for tunable choice): [`../research/2026-05-08-tilelang-hd128-ncu-baseline.md`](../research/2026-05-08-tilelang-hd128-ncu-baseline.md)
- Master strategy §3.3 + §10: [`../projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md)
- Bench spec §6 §7: [`../bench-and-trace-spec.md`](../bench-and-trace-spec.md)
- Sibling plan: [`E2-tilelang-prefill-hd128-sm89-retune.md`](E2-tilelang-prefill-hd128-sm89-retune.md)
- Decode kernel: `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py`
- FP8 sister: `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128_fp8.py`

## Rule (per `feedback_docs_priority_roi_evidence.md`)

- **Don't pre-commit BLOCK_M choice**. The §Phase 0-3 sweep must produce
  the data; choosing a value before E5 + bench is unfounded.
- **Lockstep BF16 + FP8**. Skipping FP8 kernel = inconsistent code.
- **Kill is allowed**. If E5 §6 doesn't fire the decode-thread-idle row,
  this plan ends as KILLED.
