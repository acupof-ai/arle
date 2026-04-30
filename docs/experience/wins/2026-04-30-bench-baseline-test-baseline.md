# Metal Baseline Benchmark: test-baseline

**Date:** 2026-04-30 06:19:41 UTC
**Model:** models/Qwen3.5-0.8B
**Concurrency:** 4
**Duration:** 60s

## Goal

Establish single-threaded Metal runtime performance baseline for comparison with multi-threaded implementation.

## Hypothesis

Single-threaded Metal runtime provides baseline performance that multi-threaded implementation should improve upon by 1.5-2x at high concurrency.

## Environment

```
Darwin L4L2Y39H4F 25.3.0 Darwin Kernel Version 25.3.0: Wed Jan 28 20:51:28 PST 2026; root:xnu-12377.91.3~2/RELEASE_ARM64_T6041 arm64
rustc 1.95.0 (59807616e 2026-04-14)
      Chip: Apple M4 Pro
      Memory: 48 GB
```

## Parameters

- **Model:** models/Qwen3.5-0.8B
- **Backend:** Metal (single-threaded)
- **Concurrency:** 4 requests
- **Duration:** 60s
- **Warmup:** 10s

## Results

