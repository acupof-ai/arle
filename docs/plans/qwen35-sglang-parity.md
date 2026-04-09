# Plan: Qwen3.5 SGLang 0.5.9 Parity (Qwen3.5-4B, A100-80GB)

> Status: **Near-complete** — prefix cache fixed (2026-04-09), batched prefill remaining
> Created: 2026-04-01
> Updated: 2026-04-09
> Goal: Match SGLang 0.5.9 throughput on Qwen3.5-4B

---

## Current Results (2026-04-02, 128 auto-slots, prefix cache disabled)

| Config | agent-infer | SGLang 0.5.9 | Gap |
|--------|------------|--------------|-----|
| C=1 throughput | 123 tok/s | 110 tok/s | **+12%** |
| C=4 throughput | 428 tok/s | 376 tok/s | **+14%** |
| C=8 throughput | 816 tok/s | 707 tok/s | **+15%** |
| C=16 throughput | 1320 tok/s | 1261 tok/s | **+5%** |
| C=32 throughput | 2021 tok/s | 2189 tok/s | -8% |
| C=64 throughput | 2709 tok/s | — | new |
| C=1 ITL p50 | **8.0ms** | 8.8ms | **+10% faster** |
| C=8 ITL p50 | **9.0ms** | 10.3ms | **+14% faster** |
| C=32 ITL p50 | **12.4ms** | 12.9ms | **+4% faster** |
| C=1 TTFT p50 | **21ms** | 71ms | **3.4x faster** |

## Completed Steps

### Step 1: Auto slots + batched kernels (2026-04-01)
- Auto-computed slots from GPU memory
- Batched conv1d + GDR decode kernels

### Step 2: Fix prefix cache crash (2026-04-02)
- Disabled prefix cache for Qwen3.5 (recurrent state contamination)

### Step 3: Fix auto-slots OOM (2026-04-02)
- RESERVED_BYTES 6 GB, MAX_SLOTS capped

### Step 4: 128 slots + high concurrency fixes (2026-04-02)
- MAX_SLOTS 32→128, kv_pool_headroom 2→4 GB
- Shared waiting counter bug fix (handle/scheduler separate counters)
- Prefill rate limiting: 1/step when decode active

### Step 5: Per-layer pointer array pre-upload (2026-04-02)
- Moved all 48 H2D pointer uploads before decode body
- C=1 ITL consistent 8.6ms (was 8.7–10.3ms)

### Step 6: Piecewise CUDA Graph (2026-04-02)
- Capture per-group graphs for 8 groups of 3 linear layers
- Full attention layers run eagerly between groups
- All ITL -6%

### Step 7: O(1) emit_delta (2026-04-02)
- Fixed O(N) tokenizer re-decode in emit_delta (re-decoded ALL prefix tokens)
- Cached prefix byte length between calls
- C=32 ITL: 13.5ms → 12.4ms (now beats SGLang 12.9ms)

## Remaining Work

| Step | Priority | Impact | Description |
|------|----------|--------|-------------|
| ~~Fix prefix cache for Qwen3.5~~ | ~~High~~ | ~~C=32 TTFT -135ms~~ | ✅ Done (2026-04-09) — recurrent state snapshot/restore via `GenerationState` trait |
| ~~Overlap scheduling~~ | ~~Medium~~ | ~~ITL~~ | ✅ Done (2026-04-09) — dual-stream + decode-first phase reordering |
| Batched prefill (multi-request) | Medium | TTFT | Prefill multiple requests in one forward pass |
| Increase prefill rate during ramp-up | Low | TTFT | More than 1 prefill/step when few decodes active |
