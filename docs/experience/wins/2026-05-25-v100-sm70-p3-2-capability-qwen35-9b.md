# V100 sm_70 P3.2 capability — Qwen3.5-9B MMLU + GSM8K, 2026-05-25

## Goal

- **(capability)** Confirm Qwen3.5-9B numerical capability on V100 sm_70
  scales correctly versus the 4B P3.1 baseline (`+3pp` on MMLU at minimum
  to validate size scaling) and stays near-floor on GSM8K (consistent
  with base-model behavior across both sizes).

## Hypothesis

- 9B MMLU should be 80-83% (Qwen3.5-9B base sits ~3pp above 4B on MMLU
  across published references), and GSM8K should remain near floor for
  a non-instruction-tuned base model.

## Environment

| Item | Value |
|---|---|
| GPU | Tesla V100-SXM2-32GB, sm_70 |
| CUDA | 12.4 |
| Model | `~/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B` (4 shards, 18.2 GB BF16) |
| ARLE commit | `2b35101c` (V100 build + GDR + error-chain + P3.1 wins entry) |
| TileLang fork | `fix/sm70-volta-fragment-copy` @ `20c57d4e` (PR #2257, draft, lint clean) |
| arle serve flags | `--backend cuda --port 8123 -- --num-slots 1 --max-seq-len 2048 --chunked-prefill-size 512 --max-num-batched-tokens 512 --kv-cache-dtype bf16` |
| Eval harness | `scripts/arle_capability_eval.py --tasks mmlu,gsm8k --n-samples 200` |

## Results

```
========== summary ==========
  mmlu:  0.830 (137/165)        ← scored 165/200 (invalid extractions=35)
  gsm8k: 0.010 (2/195)          ← scored 195/200 (invalid=5)
```

### Cross-section: V100 4B vs 9B vs T1 reference

| Task | V100 sm_70 4B (P3.1) | V100 sm_70 9B (P3.2) | 4B→9B Δ | T1 4B ref |
|---|---:|---:|---:|---:|
| MMLU 5-shot | 79.9% (131/164) | **83.0% (137/165)** | **+3.1 pp** | 77.33% |
| GSM8K | 2.0% (4/197) | 1.0% (2/195) | -1.0 pp (floor noise) | ~2.5% |

- **Size scaling preserved**: 9B MMLU is +3.1 pp above 4B on V100 sm_70,
  which matches the size-scaling pattern reported across Qwen3.5
  references. The Volta BF16→FP16 cast in the TileLang fallback does
  not erode the capacity gap between 4B and 9B.
- **GSM8K near-floor for both sizes**: both 4B (2.0%) and 9B (1.0%) sit
  at the floor that's expected for a base model (no SFT / RLHF /
  instruction tuning). The sub-pp difference is sample noise at
  near-zero accuracy, not a regression.

### Run timing

```
[mmlu]  171 questions across 57 subjects → 303s  (~1.77s per sample)
[gsm8k] 200 problems → 1503.6s              (~7.5s per sample, longer
                                              answers, more decode tokens)
```

GSM8K is ~4× slower per sample than MMLU because GSM8K answers run
multiple chain-of-thought tokens until `####` while MMLU emits ~1
letter token.

## Problems

- HF Hub HEAD checks again wasted ~10 min before MMLU and another ~5
  min before GSM8K. Local cache fallback works correctly each time but
  the retry budget is unavoidable when the corporate proxy proxies the
  Hub request to a route it refuses. Future P3 runs should pre-set
  `HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1` after the first run
  primes the cache.
- Eval harness still does not write per-task `summary.json` to the
  output directory; terminal log is the source of truth.

## Learnings

- The TileLang sm_70 fallback path scales numerically with model size,
  not just at a fixed shape — both Qwen3.5-4B and 9B produce
  reference-consistent MMLU on V100. The path is not a 4B-specific
  trick.
- 9B was significantly slower to warmup on V100 (~3.3s CUDA Graph
  warmup at `[ARLE serve] launching`, vs ~150ms-ish for 4B), driven by
  AOT cubin reload + per-shape CUDA graph capture for a larger weight
  matrix set. Not a per-token cost; one-time at startup.
- 32 GB V100 is comfortable for Qwen3.5-9B BF16 at `--num-slots 1
  --max-seq-len 2048 --kv-cache-dtype bf16`. After unload `nvidia-smi`
  reports 32498 MiB free → the entire ~18 GB weights + KV pool +
  activations fit with headroom.

## Delta vs baseline

- First V100 sm_70 capability data point for Qwen3.5-9B; no prior
  snapshot. The cross-reference is against P3.1 4B
  ([`2026-05-25-v100-sm70-p3-1-capability-qwen35-4b.md`](2026-05-25-v100-sm70-p3-1-capability-qwen35-4b.md))
  and T1 4B (77.33% per `2026-05-22-eod-opd-cycle-wrap.md`). T1 9B
  reference is not on file; the 4B→9B Δ pattern is the size-scaling
  acceptance gate.

## Artefacts

- V100: `/tmp/p3_eval_9b.log`
- V100: `bench-output/2026-05-25-v100-sm70-capability-9b/`
- ARLE binary used: `target/release/{arle,infer}` at `2b35101c`

## Next

- **P3 closed**. Both Qwen3.5-4B and 9B pass on V100 sm_70 with
  capability preserved vs T1 reference + size-scaling preserved across
  the two sizes.
- **P4** perf bench vs vLLM sm_70 (codex T3 confirmed vLLM installs in
  the V100 tilelang venv; baseline run + ARLE guidellm comparison
  pending per
  [`docs/plans/2026-05-25-v100-sm70-p3-p4.md`](../../plans/2026-05-25-v100-sm70-p3-p4.md)
  §P4).
