# 2026-05-20 — Axis F: rayon N-shard for `matmul_bt` backward (`grad_b` slab)

> **Status:** plan / implementation-ready. Snippet pre-written so codex
> can pick up after the current robustness sprint without re-deriving.
> Acceptance criterion pre-licensed. Replaces the placeholder Axis F
> entry in the cycle-wrap doc.

## Problem

After `0b593e1` (matmul_bt) + `e0bfbb0` (LoRA matmul_bt extension) +
`506f02b` (AdamW host-zip-loop), the per-step matmul cost concentrated
in **`MatmulBT` backward at 56 % of `backward` = 16 % of step**. The
largest single backward call is `lm_head` bwd where the `grad_b` output
is `[N=vocab, K=hidden]`:

- At moderate shape (vocab=32 768, hidden=512): grad_b = `[32_768, 512]` = 67 MB
- At Qwen3-0.6B production shape (vocab=151_936, hidden=1024): grad_b = `[151_936, 1024]` = 622 MB

The current kernel (`crates/autograd/src/backend.rs:1918-1955`
`matmul_at_b_into`) is a single-threaded `matrixmultiply::sgemm` call.
Codex's earlier session confirmed `matrixmultiply::threading` regresses
22 % at OPD M=4 shapes — the per-tile coordination overhead exceeds the
parallel speedup at this M. The correct shape is **explicit per-thread
`sgemm` calls** that each compute a disjoint slab of the output.

## Shard axis

The `grad_b` output's row axis (`N_vocab`) is the natural shard axis:

- Output: `[N_vocab, K_hidden]` row-major
- Each thread t computes rows `[t*N_chunk .. (t+1)*N_chunk]` of `grad_b`
- Output regions are disjoint → no write conflicts
- Reads of `a` (= grad_out, `[M_seq, N_vocab]`) need each thread's
  column slice — addressable via `matrixmultiply::sgemm`'s stride args
  with no physical re-layout
- Reads of `b` (= forward a, `[M_seq, K_hidden]`) shared across threads
  — read-only, no contention

## Snippet (CPU backend addition)

In `crates/autograd/src/backend.rs`, alongside the existing
`matmul_at_b_into`:

```rust
/// Parallel variant of `matmul_at_b_into` that shards the K-output-rows
/// axis across threads via per-thread `sgemm` calls. Each thread writes
/// to a disjoint K-row slab of `out`, so there are no atomic/lock
/// requirements. Falls back to the single-threaded path when the K axis
/// is too small to amortise thread overhead.
///
/// Shapes (caller-enforced):
/// - `a`: `[M, K]` (row-major contiguous, len `M * K`) — logical pre-transpose
/// - `b`: `[M, N]` (row-major contiguous, len `M * N`)
/// - `out`: `[K, N]` (row-major contiguous, len `K * N`, pre-zeroed)
///
/// Threshold: `K >= 4 * threads` is the minimum chunk size for the
/// parallel path to amortise sgemm pack/unpack overhead. At `K < 4096`
/// the single-threaded kernel wins.
#[inline]
fn matmul_at_b_into_parallel(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) {
    let (m_a, k) = a_shape;
    let (m_b, n) = b_shape;
    debug_assert_eq!(m_a, m_b, "a and b must share the M dim");
    let m = m_a;
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Get logical thread count; std-only, no rayon dep.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    const K_PER_THREAD_MIN: usize = 4096;
    if threads == 1 || k < threads * K_PER_THREAD_MIN {
        matmul_at_b_into(a, a_shape, b, b_shape, out);
        return;
    }

    let chunk_k = (k + threads - 1) / threads;
    std::thread::scope(|scope| {
        for (t, out_slab) in out.chunks_mut(chunk_k * n).enumerate() {
            let k_start = t * chunk_k;
            let k_local = out_slab.len() / n;
            // Each thread reads the full a/b and writes its own out_slab.
            let a_ref = a;
            let b_ref = b;
            scope.spawn(move || {
                // Safety contract identical to matmul_at_b_into:
                // - a viewed with rsa=1, csa=k addresses (k_log, m_log)
                //   = (k_start + i, m) at &a[a_offset + m*k + i]
                // - We offset a by k_start so logical (0, m) → physical (m, k_start).
                let a_offset = k_start;
                unsafe {
                    matrixmultiply::sgemm(
                        k_local,
                        m,
                        n,
                        1.0,
                        a_ref.as_ptr().add(a_offset),
                        1,            // rsa
                        k as isize,   // csa
                        b_ref.as_ptr(),
                        n as isize,   // rsb
                        1,            // csb
                        0.0,
                        out_slab.as_mut_ptr(),
                        n as isize,   // rsc
                        1,            // csc
                    );
                }
            });
        }
    });
}
```

Note: `std::thread::scope` is stable since Rust 1.63; no new deps. The
workspace `rustc 1.95.0` (per the wins entries) supports it.

## Dispatch in `cpu_matmul_bt_backward`

Replace the single-thread `matmul_at_b_into` call in
`cpu_matmul_bt_backward` (`backend.rs:1838`):

```rust
let grad_b = if need_grad_b {
    let mut out = vec![0.0f32; n * k];
    matmul_at_b_into_parallel(grad_out, a_shape_2d(m, n), a, b_shape_2d(m, k), &mut out);
    out
} else {
    Vec::new()
};
```

Only the `grad_b` call changes — `grad_a` is already on the fast
`cpu_matmul_forward` path which has its own dispatch.

## Acceptance criterion (pre-licensed)

| Gate | Threshold |
|---|---|
| `cargo test -p autograd --release` | All passing |
| `cargo test -p train --test test_opd_determinism --release` | Bit-identical |
| `cargo test -p train --test test_opd_grad_check --release` | Max relerr ≤ 0.5 % |
| `cargo clippy -p autograd --all-targets --release -- -D warnings` | Clean |
| Isolated `cpu_matmul_bt_backward` bench at lm_head shape (moderate) | ≥ 1.5 × wall-clock vs single-thread |
| End-to-end OPD step (moderate, 5-sample A/B with matched controls) | ≥ 1.10 × step median, σ ≤ 5 % |

**KILL** if either of the last two fails. Revert the dispatch change
and remove `matmul_at_b_into_parallel`. The wins stub is empty until
both gates pass.

## Risk + safety net

- `std::thread::scope` ergonomics may force a closure structure that
  doesn't compose cleanly with the existing `cpu_matmul_bt_backward`
  signature. If so, codex's call is whether to pull in rayon
  (workspace dep change) or use explicit handle joining.
- The shard chunk size constant (`K_PER_THREAD_MIN = 4096`) is a guess
  based on `lm_head`'s N_vocab vs other backward shapes. A 2-shape
  sweep (lm_head vs an MLP projection bwd) would calibrate; codex's
  call whether to make this configurable or hard-code.
- If `matrixmultiply::sgemm`'s internal packing is per-call expensive
  for small `k_local`, the threshold above may need bumping. The
  `K_PER_THREAD_MIN` floor protects against thrashing this.
- Determinism: `std::thread::scope` doesn't guarantee deterministic
  thread interleaving, but each thread writes to a disjoint output
  slab and reads from immutable inputs, so the result is bit-identical
  modulo thread scheduling order. `test_opd_determinism` will catch any
  violation immediately.

## Why this is the right next axis

Per the cycle-wrap state, `MatmulBT` backward is 56 % of `backward`
which is 29 % of step → 16 % of step total. A 1.5 × kernel speedup on
this single axis projects to ~5 % step saving. A clean 2 × (which 8C
parallelism makes plausible at the lm_head shape's bandwidth budget)
projects to ~8 % step. Either clears the 5 % minimum for a "land it"
license.

The merge_grad axis has been exhausted; the remaining backward time is
dominated by this single kernel call. Sharding it is the
highest-confidence path to further backward reduction.

## Cross-links

- Cycle wrap: [`../projects/2026-05-20-opd-cpu-perf-cycle-wrap.md`](../projects/2026-05-20-opd-cpu-perf-cycle-wrap.md)
- Hand-offs index: [`./2026-05-20-opd-cpu-perf-codex-handoffs.md`](2026-05-20-opd-cpu-perf-codex-handoffs.md)
- Existing kernel: `crates/autograd/src/backend.rs::matmul_at_b_into` (1918-1955)
- matmul_bt backward: `crates/autograd/src/backend.rs::cpu_matmul_bt_backward` (1808-1844)
- Backward sub-phase data: [`../research/2026-05-20-opd-backward-sub-phase-attribution.md`](../research/2026-05-20-opd-backward-sub-phase-attribution.md)
