# Metal c=1 Batching Optimization — guidellm pending, 2026-05-02

## Goal

- Measure performance improvement from enabling batched decode for c=1 workloads on Metal backend.

## Hypothesis  

- c=1 requests will now use `metal_decode=batch` instead of scalar decode, improving tok/s by 15-20% to exceed SGLang baseline (11.57 tok/s).

## Context

- **Root Cause**: Metal backend forced c=1 requests to use scalar kernels due to `len() >= 2` thresholds in batching decision logic.
- **Fix**: Modified thresholds from 2→1 in `execute_qwen35_packed_decode_batch()`, `decode_batch()`, and related functions.
- **Preserved**: DFlash >=2 threshold (speculative batching requires multiple requests).

## Command

```bash
scripts/bench_guidellm.sh metal-c1-batching \
  --target http://localhost:8000 \
  --model Qwen3.5 \
  --data prompt_tokens=128,output_tokens=128 \
  --max-seconds 60 \
  --outputs json --outputs csv --outputs html
```

Status: `pending-remote` (requires Metal hardware).

## Environment

- **Backend:** metal  
- **Model:** Qwen3.5-4B
- **Hardware:** Apple Silicon (M4 Pro recommended)
- **Commit:** de47312
- **Feature set:** `--no-default-features --features metal`
- **Server launch:** `arle serve --backend metal <model-path>`

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=128,output_tokens=128`  
- `--max-seconds 60`
- `--random-seed 20260502`
- `--outputs json --outputs csv --outputs html`
- **Critical:** Test with c=1 (single concurrent request) to verify batching improvement.

## Expected Results

**Before (baseline)**:
- c=1 requests: `metal_decode=scalar` in traces
- tok/s: ~10.42 (15% below SGLang's 11.57)

**After (optimized)**:
- c=1 requests: `metal_decode=batch` in traces  
- tok/s: ~12-13 (15-20% improvement, exceeding SGLang)

## Problems

- Remote bench required due to Metal hardware dependency.

## Learnings

- Batching thresholds should consider single-request efficiency, not just multi-request scenarios.
- Metal unified memory makes single-request batching viable unlike traditional GPU architectures.
- Always check metrics/tracing to verify intended code paths are used.