# #37 W4-hybrid prefill graph — Throughput License Bench (TEMPLATE)

> **Template** — fill in after `./scripts/post_p24_commit_pipeline.sh` runs
> bench A (graph OFF) + bench B (graph ON) per
> `docs/research/2026-05-10-37-rescope-post-codex-multikey-impl.md`.

## Goal

Validate codex's #24 W4A8 prefill graph capture hoist (commit `<HASH>`)
delivers throughput improvement on the matched-control 4k/c=4 prefill-
dominant workload. License threshold: TTFT p50 Δ ≥ +10% with σ < 5% n=3.

## Hypothesis

The 8-dim multi-key graph cache (per codex's #24 implementation) reduces
launch overhead enough to close part of the +76.6% SGLang gap on 4k/c=4,
**given** the prior Phase 0 KILL anti-pattern (single-key + tail-eager) is
absent.

## Environment

| Field | Value |
|---|---|
| Host GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB |
| Driver / CUDA | 595.71.05 / CUDA 13.2.78 |
| ARLE bench commit | `<HASH>` |
| ARLE model | `infer/models/Qwen3-4B-W4-hybrid-zpfix` |
| Bench A flags | `INFER_HYBRID_W4A8_PREFILL=1` (graph OFF baseline) |
| Bench B flags | `INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1` (treatment) |
| Server flags | `--num-slots 8 --max-seq-len 8192 --admission-policy prefix-aware` |

## Commands

ARLE server (per bench, A then B):

```bash
env INFER_HYBRID_W4A8_PREFILL=1 [INFER_PREFILL_GRAPH=1 for B] \
  CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  RUST_LOG=info \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --port 8000 --num-slots 8 --max-seq-len 8192 \
    --admission-policy prefix-aware
```

guidellm shape:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh p37-w4hybrid-prefill-graph-{off,on} \
  --concurrencies 4 --max-seconds 90 --warmup 15 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

(Or single-shot via `./scripts/post_p24_commit_pipeline.sh full`)

## Results

n=3 median per bench. σ across n=3 must be < 5% for license.

| Bench | TTFT p50 (ms) | TTFT σ% | ITL p50 (ms) | out tok/s | conc p50 |
|-------|--------------:|--------:|-------------:|----------:|---------:|
| A (graph OFF baseline) | TBD | TBD | TBD | TBD | TBD |
| B (graph ON treatment) | TBD | TBD | TBD | TBD | TBD |
| **Δ B vs A** | TBD% | -- | TBD% | TBD% | -- |
| SGLang reference (codex baseline) | 928.4 | < 1% | 9.41 | 272.67 | 4 |

Raw artifacts:
- `bench-output/2026-05-10-p37-w4hybrid-prefill-graph-off-r{1,2,3}/`
- `bench-output/2026-05-10-p37-w4hybrid-prefill-graph-on-r{1,2,3}/`

## License decision

| Δ TTFT p50 | σ Stability | License |
|------------|------------|---------|
| ≥ +25% | σ < 5% | ✅ **strong proceed** — multi-key cache big win, queue Path B device-mem opt as next stretch |
| +10% to +25% | σ < 5% | ✅ **proceed** — wins entry, brief next axis |
| +5% to +10% | σ < 5% | ⚠ marginal — may indicate cache miss pattern; check `cudaGraphLaunch` count vs `cudaGraphInstantiate` for reuse rate |
| < +5% OR σ ≥ 5% | -- | ❌ **KILL** — errors entry. Anti-pattern check: capture key churn? tail-1-token eager fallback? scheduler envelope clamp regression? |

## Anti-pattern verification(per skill v1.7.0 #6)

Required even on PASS:
- `cudaGraphInstantiate` count vs `cudaGraphLaunch` count(via nsys cuda_api_sum):
  - Healthy:launch ≫ instantiate(reuse working)
  - Pathologic:launch ≈ instantiate(re-capture pattern,Phase 0 KILL病重演)
- `prefill graph capture key: ...` count in server log vs request count:
  - Healthy:< 30 keys for c=4 4k/256 sustained workload
  - Pathologic:keys > requests(per-request capture = no reuse)
- Stress-test request distribution:fixed 4k vs varying 1k-8k seq → if TTFT
  variance > 10%,multi-key cache may be evicting too aggressively

## Problems(fill on bench)

(empty until bench runs)

## Learnings(fill on bench)

(empty until bench runs)

## Delta vs Baseline

- **Codex #24 baseline:** `docs/experience/wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`(ARLE 4k/c=4 TTFT p50 1639.3ms)
- **SGLang reference:** same 4k/c=4 = TTFT p50 928.4ms,Δ +76.6%(closing target)

| metric | codex #24 baseline | now (B graph ON) | Δ vs codex | Δ vs SGLang |
|--------|-------------------:|-----------------:|-----------:|------------:|
| 4k/c=4 TTFT p50 | 1639 ms | TBD | TBD% | TBD% |
| 4k/c=4 ITL p50 | 11.47 ms | TBD | TBD% | TBD% |
| 4k/c=4 out tok/s | 223.45 | TBD | TBD% | TBD% |

## Cross-references

- #24 implementation:`docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`
- #37 re-scope:`docs/research/2026-05-10-37-rescope-post-codex-multikey-impl.md`
- Phase 0 KILL precedent:`docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- Pipeline runner:`scripts/post_p24_commit_pipeline.sh`
- Validate runner:`scripts/validate_p24_phase0v3.sh`
- Codex baseline:`docs/experience/wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`

## Rule(fill on bench)

(empty until bench provides evidence of which rule applies)

---

**Template instructions**:rename to `2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`(remove `TEMPLATE-` prefix)post bench run + populate TBD cells from `bench-output/p37-*/headline_table.md` per-run + median across n=3。
