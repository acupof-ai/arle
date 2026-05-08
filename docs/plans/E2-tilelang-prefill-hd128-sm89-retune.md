# E2 — TileLang prefill HD128 sm_89 (Ada) re-tune

> **Status**: SCAFFOLD pending [`E5 ncu baseline`](../research/2026-05-08-tilelang-hd128-ncu-baseline.md) data.
> Tunable A/B grid, expected winner, and Δ% target get nailed once E5 §5
> returns numbers + §6 interpretation rules fire.
>
> **Trigger**: E5 §5 populated + at least one of these E5 §6 rows fires:
> - Prefill smem/block ≥ 96 KB (Hopper budget overflows Ada)
> - Prefill occupancy ≤ 1 block/SM (register or smem bound)
> - Prefill tensor % < 40% (tensor cores underused)
>
> If E5 shows prefill is HBM-bound (≥ 80% peak), **E2 is KILLED** — kernel
> time is already at hardware floor for this kernel shape.

## Priority & ROI

**Priority**: P1 inference (master §7). Prefill TTFT is the long-ctx
bottleneck (master §3.1 — 4k ARLE 1976 ms vs SGLang 973 ms = -51%).
Per master §3.3, graph capture alone closes < 1% of the gap; **kernel
impl is the second of three required closure paths**.

**ROI basis** (predicted, validate against E5 §5):

- Current prefill HD128: `BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2,
  NUM_THREADS=128`. Source comment: "Hopper defaults; tuned during the
  H100 spike". Never re-tuned for sm_89 Ada.
- Hopper smem/SM = 228 KB; Ada = 100 KB. Smem usage at current config
  ≈ `(BM*HD + BN*HD + BN*HD) * 2B * STAGES = 96 KB` — exhausts Ada
  budget, leaving 0 headroom for warps and pushing occupancy to 1
  block/SM at most.
- Plausible Ada-tuned configs to test:
  - `BM=64 BN=32 ST=2 NT=128` — halve smem to 48 KB, occupancy 2 blocks/SM
  - `BM=128 BN=64 ST=2 NT=128` — more matmul work per launch (if compute-bound)
  - `BM=64 BN=64 ST=1 NT=128` — single-stage to reclaim smem
  - `BM=64 BN=64 ST=2 NT=64` — smaller block, higher concurrency
- Expected best ROI: 5-15% on prefill kernel time alone. **Not enough
  to close the 51% gap** — E2 is one of three required moves
  (graph capture P0.1 + kernel re-tune E2 + FP8 paged KV M_b.2 A1).

**Negative case**:

- If E5 shows occupancy already > 50% on current config, smem isn't
  binding; expected gain < 5% → defer.
- If prefill is HBM-bound (PV matmul saturating HBM), kernel is at floor
  → kill E2.
- TileLang 0.1.9 may not codegen all candidates (e.g. STAGES=3 + BM=128
  may exceed register pressure) — partial sweep is OK.
- Larger BLOCK sizes shift NaN-safety windows in causal masking; higher
  numerical regression risk (per `2026-04-28-tilelang-prefill-short-qlen-nan.md`).

**Kill criteria** (any → KILL E2):

1. E5 prefill occupancy ≥ 50% on current config (not smem/reg bound)
2. E5 prefill HBM > 80% peak (already at hardware floor)
3. All A/B candidates fail codegen
4. All A/B candidates fail greedy_consistency
5. Best A/B candidate < 5% wall-clock improvement (within noise)

## Phase plan

### Phase 0 — A/B grid codegen (~2 hr, no GPU bench)

Edit `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`
once per candidate, attempt build:

| ID | BM | BN | ST | NT | Hypothesis |
|---|---:|---:|---:|---:|---|
| P0 | 64 | 64 | 2 | 128 | baseline (current) |
| P1 | 64 | 32 | 2 | 128 | smem halved → occupancy 2× |
| P2 | 128 | 64 | 2 | 128 | larger fragment for tensor cores |
| P3 | 64 | 64 | 1 | 128 | single stage, full smem reclaim |
| P4 | 64 | 64 | 2 | 64 | smaller block, more concurrent blocks |
| P5 | 128 | 128 | 2 | 256 | aggressive — likely codegen fail, useful negative result |

Each: `cargo build --release --features cuda`. Record cubin sizes and
codegen success per (config, SM target). Failures → record + drop.

Sister `batch_prefill_paged_hd256.py` (Qwen3.5 14B/30B): **out of E2
scope**. Same template re-tune deferred.

### Phase 1 — numerical correctness per surviving config (~1 hr, GPU)

Per candidate that codegens:

1. `cargo test --release --features cuda --test e2e -- --test-threads=1`
2. `cargo test --release --features cuda --test greedy_consistency`
3. Targeted: short-qlen NaN regression test (per
   `2026-04-28-tilelang-prefill-short-qlen-nan.md`)

Failures → drop candidate.

### Phase 2 — bench sweep on agent shape (~1-2 hr, GPU)

Per surviving candidate, matched A/B vs P0 baseline at 4k longctx c=4
+ multi-tenant prefix-cache shape. **Same workload as E5**:

```bash
scripts/bench_guidellm.sh prefill-PNCFG-4k --workload longctx-4k --concurrencies 4
scripts/bench_guidellm.sh prefill-PNCFG-multi --workload multi-tenant
```

Goal: TTFT p99 best at agent-relevant shapes. Don't optimize for
high-conc 1k/256/c=64 (already +30%, defending only).

### Phase 3 — winner pick + commit

1. Best by composite: `min(TTFT_p99_4k_Δ%, TTFT_p99_multi_Δ%)` ≥ +10%
2. Wins entry: `docs/experience/wins/2026-05-08-e2-prefill-hd128-sm89-retune.md`
3. Cross-link from master §10 + E5

## Acceptance

- 4k longctx TTFT p99 improves by ≥ 10% (kernel-only contribution; full
  master goal of 30% comes from E2 + P0.1 + M_b.2 A1 together)
- Multi-tenant TTFT p99 improves by ≥ 5% (defending +80% lead)
- No high-conc regression > 5%
- e2e + greedy_consistency green
- σ < 5% across n=3
- Wins entry with full grid table, σ, env, raw artifacts

## Cross-references

- E5 baseline: [`../research/2026-05-08-tilelang-hd128-ncu-baseline.md`](../research/2026-05-08-tilelang-hd128-ncu-baseline.md)
- Master strategy §3.3 + §7.3 + §10: [`../projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md)
- M_pf-graph plan: [`M_pf-graph-prefill-capture.md`](M_pf-graph-prefill-capture.md)
- Sibling: [`E1-tilelang-decode-hd128-blockm-retune.md`](E1-tilelang-decode-hd128-blockm-retune.md)
- Prefill kernel: `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`
- NaN regression history: `docs/experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md`

## Rule

- **Don't tune blind**. E5 §5 must be populated before Phase 0 starts.
  ncu data narrows candidate grid (e.g. if HBM > 80%, drop P2/P5
  big-fragment candidates).
- **Defending +80% multi-tenant lead is mandatory**. Phase 2 must
  include multi-tenant; a candidate that wins 4k but regresses
  multi-tenant > 5% loses.
- **Lockstep with M_pf-graph**. If P0.1 graph capture lands, E2 A/B
  must rebench under both `INFER_PREFILL_GRAPH=0` (eager) and `=1`
  (graph) — graph + smaller smem may unlock different optima.
