---
title: M_pf83 H1' static-scratch refactor — eliminate cudarc allocator fragmentation under sustained PF8 load
date: 2026-05-10
type: plan
status: BLOCKED-pending-redesign (PF8.5 bench v11 returned KILL 11:34 KST per 0be278f; original H1' design empirically broken per Arm B/C/D 4-arm A/B)
depends_on:
  - ~~bench v11 conc=1 PF8.5 license decision~~ — DONE 2026-05-10 11:34 KST: KILL (5878 kernel failures, see 0be278f)
  - Arm B refute (warmup-INDEPENDENT per 7ed8160)
  - Arm C+D controls (W4A16 + W4A8 HEALTHY at conc=1, isolates PF8.3 substrate per 06b7437 + d8b2870)
  - Twin-control SKILL candidate (430a4be)
gates:
  - ~~LICENSE: PF8 conc=1 TTFT Δ ≥ -8% vs INT8 baseline~~ — N/A, KILL outcome reached
  - ~~KILL: PF8 conc=1 TTFT regression > -3%~~ — REACHED via different mechanism (kernel failures, not regression)
  - **REDESIGN required**: original H1' (make MarlinScratch default-on) is empirically broken per Arm B (warmup-OFF still fails 5959 times). Per-call cudaMalloc cannot work for PF8 path; needs ground-up workspace allocation redesign.
new_acceptance_criteria_for_h1prime_v2:
  - OOM-regression A/B gate at conc=4 4k W4A16 sustained (per da7f5a2)
  - TTFT/tok-s regression A/B gate at same workload (per d09623a)
  - PF8 conc=1 sustained-load HEALTHY (zero kernel failures over 60s) — must beat current Arm A 5878 / Arm B 5959 failures
  - Match or improve W4A8 perf bar (54.2ms TTFT / 11.9ms ITL / 409 tok/s) at conc=1 — established by Arm D (d8b2870)
---

# M_pf83 H1' static-scratch refactor — eliminate cudarc allocator fragmentation under sustained PF8 load

> **Gating contract**: this plan is **codex-pickup-ready but conditionally
> blocked** on user-run bench v11 license outcome. If PF8 licenses at
> conc=1 → codex implements H1' to enable production conc≥2. If PF8 KILLs
> at conc=1 → this plan is shelved as reference for future GEMM kernel
> substrate; Task #44 closes; pivot to Task #28 Medusa.

## §0 Goal

Eliminate the per-call `CudaSlice::alloc_zeros` pattern in
`run_marlin_w4_fp8_prefill` so PF8.3 substrate runs sustained
concurrent load (conc≥2) without `cudaErrorMemoryAllocation` (code 2)
failures. Current pattern allocates 5 buffers per call (~10 MB total)
× 252 linear ops/forward × N concurrent prefills × 60s sustained =
~60k allocations triggering cudarc allocator pool fragmentation.

**Acceptance**: PF8 path completes 60s `--concurrencies "1,2,4"`
guidellm bench with 0 `gemm_w4_fp8_marlin_cuda failed with code 2`
log lines AND TTFT/ITL within license threshold per a66d99a §2.

## §1 Root cause (re-confirmed)

Per `0cde63d` PF8.3 RUNTIME KILL: 101380/101380 failures with code 2
under sustained conc=4 60s bench.

Per `57c37b5` H8 verify: kernel WORKS at conc=1 single-request
greedy_consistency (4.33s, valid Chinese+English completions).

Per `cd7732a` §7 H1' refined hypothesis: per-call alloc fragmentation
under sustained load. Mechanism:

```
per call: 5 × CudaSlice::alloc_zeros (x_fp8 + s_activation + reduce + workspace + y_fp16)
        ≈ 1.3 + 0.002 + 5.0 + 0.001 + 2.6 MB ≈ ~9 MB total/call

per forward: 252 linear ops × ~9 MB = ~2.3 GB churn
per conc=4 sustained 60s: ~60480 allocs, ~580 GB cumulative alloc/free

→ cudarc CachingAllocator pool fragments → small-block requests fail
  with cudaErrorMemoryAllocation despite > 14 GB free VRAM
```

Why conc=1 works: single request = ~252 allocs total → allocator pool
has not fragmented yet → all calls succeed.

## §2 Buffer audit (current per-call allocs, infer/src/ops/linear.rs:1637-1693)

| Buffer | Type | Size formula | Worst-case (Qwen3-4B, max_m=2048, max_k=2560, max_n=11008) |
|---|---|---|---|
| `x_fp8` | `CudaSlice<u8>` | `m * k` | 2048 × 2560 = 5.0 MB |
| `s_activation` | `CudaSlice<f32>` | `m` | 2048 × 4 = 8 KB |
| `reduce` | `CudaSlice<f32>` | `sm_count() * tmp_m * 256` (tmp_m clamped to 64) | 60 × 64 × 256 × 4 = 3.75 MB |
| `workspace` | `CudaSlice<i32>` | `(n / 128) * max_par` | (11008 / 128) × 16 × 4 = 5.5 KB |
| `y_fp16` | `CudaSlice<Half>` | `m * n` | 2048 × 11008 × 2 = 43 MB |

**Worst-case total per call: ~52 MB.** Realistic Qwen3-4B prefill
(m=512 tokens, n=2560/4096 across attention/MLP layers) averages
~9 MB/call as documented in §1.

**Note** `tmp_m.min(64)` per linear.rs:1680 caps reduce buffer at
`sm_count * 64 * 256`. The dimensioning constraint must hold in
static-scratch refactor: `tmp_m` is a kernel constant (4 thread_m_blocks
* 16) not workload-dependent.

## §3 Proposed `PF8Scratch` struct

```rust
// crates/cuda-kernels/src/scratch.rs (new file, or inline in linear.rs)
pub struct PF8Scratch {
    pub x_fp8: CudaSlice<u8>,         // sized: max_m * max_k
    pub s_activation: CudaSlice<f32>, // sized: max_m
    pub reduce: CudaSlice<f32>,       // sized: sm_count * 64 * 256
    pub workspace: CudaSlice<i32>,    // sized: max_n / 128 * MARLIN_MAX_PAR
    pub y_fp16: CudaSlice<Half>,      // sized: max_m * max_n
    capacity_m: usize,
    capacity_n: usize,
    capacity_k: usize,
    sm_count: usize,
}

impl PF8Scratch {
    /// Allocate once at server startup (or first PF8 forward) for the
    /// max dimensions the model can produce.
    pub fn new(
        ctx: &DeviceContext,
        capacity_m: usize,
        capacity_n: usize,
        capacity_k: usize,
    ) -> Result<Self> {
        let sm_count = ctx.sm_count();
        Ok(Self {
            x_fp8: ctx.stream.alloc_zeros(capacity_m * capacity_k)?,
            s_activation: ctx.stream.alloc_zeros(capacity_m)?,
            reduce: ctx.stream.alloc_zeros(sm_count * 64 * 256)?,
            workspace: ctx.stream.alloc_zeros((capacity_n / 128).max(1) * MARLIN_MAX_PAR)?,
            y_fp16: ctx.stream.alloc_zeros(capacity_m * capacity_n)?,
            capacity_m,
            capacity_n,
            capacity_k,
            sm_count,
        })
    }

    /// Verify a request's (m, n, k) fits within scratch capacity.
    pub fn check(&self, m: usize, n: usize, k: usize) -> Result<()> {
        anyhow::ensure!(m <= self.capacity_m, "PF8Scratch m={m} > capacity_m={}", self.capacity_m);
        anyhow::ensure!(n <= self.capacity_n, "PF8Scratch n={n} > capacity_n={}", self.capacity_n);
        anyhow::ensure!(k <= self.capacity_k, "PF8Scratch k={k} > capacity_k={}", self.capacity_k);
        Ok(())
    }

    /// Reset the lock workspace before each kernel launch. Marlin
    /// workspace is read+written by kernel; must be zeroed between calls.
    /// reduce buffer is also kernel-internal — kernel writes its own
    /// partial reductions; no caller reset required (verify with kernel author
    /// in marlin_pf8/kernel.h).
    pub fn reset_workspace(&mut self, stream: &CudaStream) -> Result<()> {
        stream.memset_zeros(&mut self.workspace)?;
        Ok(())
    }
}
```

## §4 `run_marlin_w4_fp8_prefill` refactor (~30 LOC delta)

```rust
fn run_marlin_w4_fp8_prefill(
    ctx: &DeviceContext,
    scratch: &mut PF8Scratch,           // NEW: pass mut scratch
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    rows: usize,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    hybrid_w4_fp8_aligned(weight)?;
    let qweight = weight.hybrid_w4_fp8_qweight.as_ref().unwrap();
    let scales = weight.marlin_scales.as_ref().unwrap();
    let m = rows;
    let n = weight.rows;
    let k = weight.cols;
    let max_par = MARLIN_MAX_PAR;

    scratch.check(m, n, k)?;             // NEW: capacity guard
    scratch.reset_workspace(&ctx.stream)?; // NEW: zero locks

    // REMOVED 5× alloc_zeros calls (saves ~9 MB churn/call)
    // Use scratch.x_fp8, scratch.s_activation, scratch.reduce,
    //     scratch.workspace, scratch.y_fp16 directly via slice
    //     into the first m*k / m / sm*64*256 / etc. elements.

    // Pass scratch.x_fp8 into quantize_bf16_rows_to_fp8_e4m3_cuda
    // Pass scratch.{x_fp8, reduce, workspace, y_fp16, s_activation, scales}
    //   into ffi::gemm_w4_fp8_marlin_cuda
    // Existing fp16→bf16 conversion at end uses scratch.y_fp16 (no change)
    // ...
}
```

**Total delta**: ~30 LOC removed (per-call allocs) + ~10 LOC added
(scratch.check + reset_workspace) + plumbing ~30 LOC (`PF8Scratch` plumbed
through call chain to ops/linear.rs:2094 caller).

## §5 State integration

PF8Scratch is **per-stream** (ties to ctx.stream lifetime), so:

- **Where to store**: per `State` instance. Add `pf8_scratch: Option<PF8Scratch>`
  to qwen3 / qwen35 / deepseek `State` impls (only models that may
  hit PF8 path). Initialize lazily on first `run_marlin_w4_fp8_prefill`
  call OR eagerly at warmup if INFER_MARLIN_W4_FP8_PREFILL=1 env on.
- **Capacity sizing**: read max_m from `Config::max_seq_len` (typically
  2048 or 4096), max_n from `Config::intermediate_size` (largest of
  attention QKV / MLP gate/up/down output dim), max_k from same.
- **Lazy init thread-safety**: per-state, single-threaded. No mutex needed.
- **Eager init at warmup**: cleaner, matches Pass 3 cap=8 warmup pattern
  (Task #35) — both pre-allocate worst-case scratch at startup.
  Recommend eager.

## §6 Test plan (per SKILL v1.12.0 #33 + #34)

1. **Build + clippy**: `cargo build --release -p infer --features cuda`
   + `cargo clippy --release -p infer --features cuda --lib -- -D warnings`
2. **Greedy consistency at conc=1**: must still pass (regression baseline)
3. **Greedy consistency at conc=2 AND conc=4**: NEW gate per #34 —
   greedy_consistency single-thread is not sufficient for sustained-load
   substrate. Run with multiple concurrent in-flight requests.
4. **Sustained-load bench at conc=1,2,4 60s**: critical gate. 0 `code 2`
   log lines required. TTFT/ITL within license threshold.
5. **codex review --uncommitted BEFORE commit** per #33: cross-State
   trait change + FFI buffer ownership change + scratch lifetime
   reasoning warrant codex review pass. Skip only if total diff <100 LOC
   (likely not — plumbing alone is ~30 LOC, scratch struct ~50 LOC,
   3× State impls ~30 LOC = ~110 LOC total).
6. **VRAM peak measurement**: nvidia-smi during bench. PF8Scratch
   adds ~52 MB peak (max_m × max_n × 2). Must stay within budget.

## §7 Tradeoff explicit (per SKILL v1.12.0 mantra rule 5)

| Axis | Sacrifice |
|---|---|
| **VRAM headroom** | +~52 MB peak per State for max-shape scratch (cheap on 16 GB consumer card; verify on smaller cards) |
| **Server startup time** | +~1 ms eager scratch init (negligible vs 30s+ model load) |
| **Code complexity** | +~110 LOC, +1 struct, +3 State impl changes; plumbing through call chain via `&mut PF8Scratch` parameter |
| **Lifetime tracking** | scratch lives with State (per-request), not per-call; existing `&mut` discipline holds |
| **Capacity rigidity** | scratch sized to max_m at init; requests within max are free, requests OVER max fail with capacity check (graceful) |
| **Generality** | PF8-specific; if other GEMM kernels adopt similar pattern they'd need their own scratch (or generalize) |

**No free lunch caught**: the alternative — keep per-call allocs but
warm up the cudarc allocator pool at startup — was considered but
rejected because cudarc CachingAllocator doesn't expose pre-warm
hooks, and the underlying fragmentation problem is allocation
*pattern* (many small blocks of varying size) not pool-emptiness.

## §8 Cross-references

- `0cde63d` PF8.3 RUNTIME KILL evidence (101380 failures sustained load)
- `57c37b5` H8 verify (kernel works at conc=1)
- `cd7732a` §7 H1' refined hypothesis (this plan implements §7 substrate)
- `81672c3` H8 diagnostic patch (kept in tree as defensive instrumentation)
- `0be7220` SKILL v1.12.0 #33 (codex review load-bearing) + #34 (greedy not sufficient)
- `a66d99a` PF8.5 license matrix (TTFT Δ ≥ -8% LICENSE)
- `infer/src/ops/linear.rs:1637-1693` (current per-call alloc pattern)
- `infer/src/ops/linear.rs:2094` (sole caller — hybrid W4A8 prefill dispatch)

## §9 Codex pickup checklist (when bench v11 LICENSES PF8)

1. Read this plan + cross-referenced commits in §8
2. Implement `PF8Scratch` struct (own crate or inline; recommend inline
   in `infer/src/ops/linear.rs` to keep scope narrow first iteration)
3. Add `pf8_scratch: Option<PF8Scratch>` to qwen3/qwen35/deepseek State
4. Refactor `run_marlin_w4_fp8_prefill` to take `&mut PF8Scratch`
5. Update sole caller at linear.rs:2094 to plumb scratch from State
6. Update warmup.rs (post Task #35 land) to eagerly init pf8_scratch
   when INFER_MARLIN_W4_FP8_PREFILL=1
7. Run §6 test plan
8. `codex review --uncommitted` per §6.5
9. Commit + push + wins entry with sustained-load bench Δ% numbers
10. Update Task #44 PF8 chain status; mark Task #46 deferred work closed

## §10 Estimated effort

- Codex implementation: ~2-3 hours (110 LOC + plumbing + tests)
- Bench validation (codex-runnable, doesn't hit Claude session sleep limits
  if codex initiates): ~30 min
- codex review pass: ~15-30 min
- Wins entry write: ~15 min

**Total: ~3-4 hours codex bandwidth, single tick if user available to
respond to clarifications.**
