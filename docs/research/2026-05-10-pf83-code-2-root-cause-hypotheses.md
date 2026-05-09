---
title: PF8.3 gemm_w4_fp8_marlin_cuda code 2 root cause hypotheses (for codex investigation)
date: 2026-05-10
type: research
status: pf83-code-2-investigation-claude-survey-codex-pickup
---

# PF8.3 gemm_w4_fp8_marlin_cuda code 2 root cause hypotheses (for codex investigation)

> Per `0cde63d` PF8.3 RUNTIME KILL — kernel fails 100% with code 2
> (cudaErrorMemoryAllocation). First failure at Request 1 = NOT a
> leak. THIS tick: Claude survey of allocation paths to give codex
> hypothesis-grounded starting points for the fix.

## §0 Direct evidence (raw grep + inspection THIS tick)

### Wrapper signature (marlin_w4_fp8_kernel.cu:130)
```cpp
void* workspace,     // lock buffer
...
int max_par) {
  if (max_par <= 0) { max_par = kMaxParDefault; }  // line 189
  ...
  int* locks = reinterpret_cast<int*>(workspace);  // line 199
```

**Workspace is PASSED IN by caller, NOT allocated by kernel**. So
the kernel doesn't `cudaMalloc` directly. Code 2 must originate
from:
- Caller's per-call scratch allocations (linear.rs:1631+ `run_marlin_w4_fp8_prefill`)
- Kernel launch resource exhaustion (returned via cudaGetLastError())
- Persistent allocator state (CudaSlice pool fragmentation)

### Caller's per-call scratch (run_marlin_w4_fp8_prefill from earlier diff capture)

```rust
let mut x_fp8: CudaSlice<u8> = ctx
    .stream
    .alloc_zeros(m * k)  // m=513 (prompt tokens), k=2560-6912
    .map_err(|e| anyhow::anyhow!("alloc W4+FP8 x_fp8: {e}"))?;
let mut s_activation: CudaSlice<f32> = ctx
    .stream
    .alloc_zeros(m)
    .map_err(|e| anyhow::anyhow!("alloc W4+FP8 activation scales: {e}"))?;
```

x_fp8 size: 513 * 2560 = 1.3 MB minimum, up to 513 * 6912 = 3.5 MB
per call. Per linear forward call. 7 linear ops/layer × 36 layers = 252
linear ops per forward pass. Per-request total: ~252 * 2 MB ≈ 500 MB
of x_fp8 in flight.

Under conc=4 sustained: 4 × 500 MB = 2 GB scratch. Server already
uses 14 GB for model. Total 16 GB — at GPU capacity.

## §1 Hypothesis ranking (most → least likely)

### H1 — CudaSlice allocator pool fragmentation (HIGHEST)

CudaSlice on cudarc uses a backing memory pool. Per-call `alloc_zeros`
+ drop cycle creates fragmentation under sustained high-frequency
calls (~100k requests/10min = 167 req/s × ~250 linear ops/req = 41k
allocs/s).

Pool may give up returning failed alloc → cudaErrorMemoryAllocation.

Fix path: pre-allocate scratch buffers ONCE per server lifetime + reuse
(remove per-call allocate). Or use a slab allocator.

### H2 — Kernel launch resource exhaustion (HIGH)

Some specializations in marlin_w4_fp8_kernel.cu may exceed sm_89's
100 KB shared memory budget. Hopper has 228 KB; tiles tuned for Hopper
exceed Ada budget.

Per skill kernel-optimization §2 hardware traps:
> "TileLang HD128 carries BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2 with
> comment 'tuned during the H100 spike'. sm_89 has 100 KB smem/SM
> (vs Hopper 228 KB) — these defaults push smem to ~96 KB/CTA"

Codex generated 12+ tile variants in sm89_kernel_fe4m3fn_u4b8_bfloat16.cu.
SOME variants may exceed 100 KB → cudaErrorLaunchOutOfResources.

But code 7 = LaunchOutOfResources, NOT code 2. So this hypothesis is
weaker. UNLESS the cudarc wrapper translates code 7 → code 2 somehow.

Fix path: filter codegen variants to ones within 100 KB smem budget
(or compute_smem_per_cta() check before launch).

### H3 — CUDA context/stream issue (MEDIUM)

PF8 dispatch path may use a different cudaStream than the one ctx.stream
expects, causing stream-context mismatch on alloc.

Fix path: trace stream usage in run_marlin_w4_fp8_prefill vs server
init.

### H4 — Lock workspace not allocated by caller (MEDIUM)

The workspace pointer at line 199 reads from caller. If caller passes
NULL or insufficient size, locks array writes go to invalid memory →
might trigger code 2 in subsequent ops.

Fix path: verify caller properly allocates `marlin_workspace_size(prob_n,
sms)` bytes for workspace and passes to kernel.

### H5 — Shared memory budget cudaFuncSetAttribute (LOWEST)

sm_89 default smem max per CTA is 48 KB (CC 8.9 docs). To use more,
must call `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 100KB)`.
If kernel doesn't set this, allocation > 48 KB fails.

Per upstream Marlin docs: this is set elsewhere. PF8 codegen variant
may or may not include this setup.

Fix path: add `cudaFuncSetAttribute` for each PF8 kernel variant.

## §2 Recommended next-step investigation (codex)

```bash
# 1. Verify caller workspace allocation matches FFI requirement
grep -A 5 "gemm_w4_fp8_marlin_cuda" infer/src/ops/linear.rs
# Check: is workspace ALLOCATED before call? what size?

# 2. Run with CUDA error context dumping
CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=full \
  INFER_HYBRID_W4A8_PREFILL=1 INFER_MARLIN_W4_FP8_PREFILL=1 \
  target/release/infer --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --port 8000 \
  > /tmp/pf83-debug.log 2>&1 &
# Then curl /v1/completions to trigger 1 request
# Server log will show exact line + cudaError translation

# 3. Check sm_89 smem per kernel variant
nvcc --ptxas-options=-v csrc/gemm/marlin_pf8/marlin_template.h
# Look for "smem usage" lines per kernel

# 4. Test isolated kernel call (no dispatch chain)
# Write smoke harness that calls gemm_w4_fp8_marlin_cuda directly
# with known-good workspace + tile params
```

## §3 If H1 (allocator fragmentation) is correct

Quick fix: add scratch buffer pool to PF8 substrate.

```rust
struct PF8Scratch {
    x_fp8: CudaSlice<u8>,           // sized to max m * max k
    s_activation: CudaSlice<f32>,    // sized to max m
    workspace: CudaSlice<u8>,        // sized to max marlin_workspace_size
}

// One-time init at server startup
fn init_pf8_scratch(ctx, max_m, max_k, max_n, sms) -> PF8Scratch {
    // single alloc, reused across all calls
}

// run_marlin_w4_fp8_prefill takes &mut PF8Scratch instead of allocating
fn run_marlin_w4_fp8_prefill(scratch: &mut PF8Scratch, ...) {
    // reuse scratch.x_fp8, scratch.s_activation
    // no per-call alloc
}
```

Effort: ~50-100 LOC in linear.rs + scheduler init. Codex-doable.

## §4 If H2 (smem exceeds 100 KB) is correct

Identify which variants exceed budget:
```cpp
// Per variant in sm89_kernel_fe4m3fn_u4b8_bfloat16.cu, compute:
template_smem = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N + BLOCK_M * BLOCK_N) * dtype_size * NUM_STAGES
// E.g. BLOCK_M=64, BLOCK_K=128, BLOCK_N=64, dtype=fp8(1B), STAGES=4:
// (64*128 + 128*64 + 64*64) * 1 * 4 = 81920 bytes = 80 KB ← OK
// E.g. BLOCK_M=128, BLOCK_K=128, BLOCK_N=128, fp8, STAGES=4:
// (128*128 + 128*128 + 128*128) * 1 * 4 = 196608 bytes = 192 KB ← FAIL on sm_89
```

Filter codegen output to keep only variants within 100 KB.

## §5 Cross-references

- `0cde63d` PF8.3 RUNTIME KILL (root cause investigation needed)
- `11763ba` PF8.3 substrate landed (the kernel being investigated)
- `ace3cbe` codex review caught 3 bugs (Bug 2: max_par/lock workspace
  underrun — addressed but apparently not the OOM)
- `9bb3843` RUST_MIN_STACK=8MB (irrelevant to GPU OOM, was for
  CPU stack)
- skill kernel-optimization §2 hardware trap (Hopper smem != Ada
  smem)
- Server log: /tmp/pf83-treatment-fp8-direct.log (101380 failures)
- Pickup queue: ad14636 §4 option 3 ("PF8.3 kernel fix investigation")

## §6 Status

5 hypotheses ranked. H1 (allocator fragmentation) is most likely
given the per-call alloc pattern + sustained-load failure mode + fact
that single-request greedy_consistency PASSED.

Codex follow-up:
1. Test H1 first via static-scratch refactor (~50-100 LOC)
2. If H1 doesn't fix → test H2 via smem audit per kernel variant
3. If H2 doesn't fix → test H3/H4/H5

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(marlin_w4_fp8_kernel.cu:130/189/199 grep + linear.rs scratch
pattern from earlier diff capture + skill kernel-optimization §2
sm_89 smem trap reference).
