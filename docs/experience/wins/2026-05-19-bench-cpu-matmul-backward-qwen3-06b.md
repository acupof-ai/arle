# CPU matmul **backward** at Qwen3-0.6B production shapes — gap after `499bfc0` row-major forward — AMD Ryzen 7 3700X, 2026-05-19

## Goal

- **(diagnosis)** Verify whether codex's `499bfc0 perf(autograd): improve CPU matmul locality` row-major rewrite of `cpu_matmul_forward` flows through to `cpu_matmul_backward`. Codex's `499bfc0` commit body explicitly flagged this as the next licensing question.
- Provide independent SOLID evidence at production shape (Qwen3-0.6B, seq=4) so the next CPU OPD perf tranche is licensed against measurement, not source-grep inference.

## Hypothesis

- `cpu_matmul_backward` (`crates/autograd/src/backend.rs:1537-1571`) is implemented as `transpose_last_two_ref` + 2× `cpu_matmul_forward`. The forward path is now ~20 GFLOPs/s at production shape, but `transpose_last_two_ref` (`crates/autograd/src/backend.rs:1577-1608`) is still a naive scalar double loop with **column-strided writes** (`out[col * rows + row] = data[row * cols + col]`). For `lm_head`-shape transpose of `B: [1024, 151936]`, that is a 622 MB host buffer churn at scalar speed — easily 0.3–0.6 s per call, plus a second transpose for the other sgemm.
- If confirmed, backward will run at **a small fraction of forward GFLOPs/s** at production shape and dominate the OPD step wall-clock, even with codex's forward rewrite already landed.

## Command

```bash
mkdir -p bench-output/2026-05-19-cpu-matmul-backward-row-inner-col-qwen3-06b && \
  cargo run -p autograd --example cpu_matmul_backward_microbench --release \
  | tee bench-output/2026-05-19-cpu-matmul-backward-row-inner-col-qwen3-06b/cpu_matmul_backward.txt
```

Benchmark constants (`crates/autograd/examples/cpu_matmul_backward_microbench.rs`):

```text
shape catalog = same Qwen3-0.6B per-layer projection set + lm_head as cpu_matmul_microbench
warmup_runs = 1
measured_runs = 5
target_fmas_per_run = 1_000_000_000  (inner-iter autoscaling like codex's forward bench)
need_grad_a = need_grad_b = true     (both sgemms in the backward path)
```

## Environment

| Item | Value |
|---|---|
| Backend | CPU `cpu_matmul_backward` (forward inherits `499bfc0` row-major rewrite) |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T, Zen 2 |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Runtime substrate commit | `499bfc0` (forward row-major rewrite landed; backward + transpose untouched) |
| Feature set | `cargo run -p autograd --example cpu_matmul_backward_microbench --release` |
| Non-default flags | none |

## Results

### Per-shape median (5 runs, warmup=1, auto inner-iters)

| shape | FMAs (fwd) | median (s) | mean (s) | σ/μ | GFLOPs/s (eff) | ×/fwd | bwd cost (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q_proj    [4,1024] @ [1024,2048]`   | 8.389e6 | 0.024331 | 0.024363 | 0.245% | **1.379** | 28 | 0.681256 |
| `k_proj    [4,1024] @ [1024,1024]`   | 4.194e6 | 0.008718 | 0.008713 | 0.163% | **1.924** | 28 | 0.244116 |
| `v_proj    [4,1024] @ [1024,1024]`   | 4.194e6 | 0.008695 | 0.008680 | 0.644% | **1.929** | 28 | 0.243467 |
| `o_proj    [4,2048] @ [2048,1024]`   | 8.389e6 | 0.024230 | 0.024346 | 0.637% | **1.385** | 28 | 0.678447 |
| `gate_proj [4,1024] @ [1024,3072]`   | 1.258e7 | 0.028902 | 0.028944 | 0.560% | **1.741** | 28 | 0.809248 |
| `up_proj   [4,1024] @ [1024,3072]`   | 1.258e7 | 0.029214 | 0.029193 | 0.310% | **1.723** | 28 | 0.818001 |
| `down_proj [4,3072] @ [3072,1024]`   | 1.258e7 | 0.038099 | 0.038115 | 0.418% | **1.321** | 28 | 1.066776 |
| `lm_head   [4,1024] @ [1024,151936]` | 6.223e8 | 2.097973 | 2.096225 | 0.624% | **1.187** |  1 | 2.097973 |

GFLOPs/s = `2 × (2 × FMAs_fwd) / median` — the `2× FMAs_fwd` accounts for backward = two sgemms; the outer `2×` is the standard "ops = 2 × FMA" convention.

### Aggregate cost per OPD step

| Quantity | Value |
|---|---:|
| Backward matmul cost per single student backward pass (28 layers + lm_head) | **6.639 s** |
| Forward matmul cost per single forward (from `499bfc0` wins) | **0.348 s** |
| Backward / forward ratio | **~19.1×** slower |

### Forward vs backward GFLOPs/s contrast (post-`499bfc0`)

| Shape | Forward (`499bfc0`) | Backward | Backward gap |
|---|---:|---:|---:|
| `q_proj`    | 20.480 | 1.379 | **14.9×** slower |
| `k_proj`    | 20.733 | 1.924 | **10.8×** slower |
| `v_proj`    | 20.513 | 1.929 | **10.6×** slower |
| `o_proj`    | 20.100 | 1.385 | **14.5×** slower |
| `gate_proj` | 18.211 | 1.741 | **10.5×** slower |
| `up_proj`   | 12.139 | 1.723 |  **7.0×** slower |
| `down_proj` | 15.226 | 1.321 | **11.5×** slower |
| `lm_head`   |  8.144 | 1.187 |  **6.9×** slower |

## Problems

- **Backward did NOT inherit the forward win.** Forward is now ~20 GFLOPs/s post-`499bfc0`; backward is stuck at 1.2–1.9 GFLOPs/s. The single largest single-op cost in the entire step is now **`lm_head` backward at 2.10 s** — 2.4× the prior forward total.
- **Single backward pass = 6.64 s.** OPD step has 1 backward pass through the student. With forward at 0.35 s × 3 forwards = 1.05 s and backward at 6.64 s, the projected step is now **dominated by the backward path (~85% of matmul wall-clock)**. Forward optimisation gave us 27× per-forward but only ~3× per-step.
- **Root cause is the host transpose**, not the inner sgemm. `cpu_matmul_backward` calls `transpose_last_two_ref` twice per backward (one for `A^T`, one for `B^T`) and then forwards into `cpu_matmul_forward`. `transpose_last_two_ref` is a naive double loop with **column-strided writes** at `crates/autograd/src/backend.rs:1577-1608`. For the `lm_head` shape, transposing `B: [1024, 151936]` writes 622 MB of f32 with stride-1024 access — easily 0.5–1.0 s of pure memory churn before any sgemm runs.

## Learnings

1. **Inheritance must be measured, not assumed.** Codex's `499bfc0` commit body anticipated backward would auto-inherit through the shared `cpu_matmul_forward`. That was a reasonable inference from the source, but at production shape the inheritance is **broken** because the host transpose is on the wall-clock critical path *before* the now-fast forward runs. SOLID §0 frame held: hypothesis ≠ evidence; the bench is the licence.
2. **The next perf lever is transpose elimination, not SIMD.** Even raising the inner sgemm to 30 GFLOPs/s (matrixmultiply ceiling) only buys ~1.5× on the parts of backward that *aren't* transpose. The transpose itself is the dominant share — eliminate it and the inner sgemm already meets the wall-clock budget.
3. **Two viable transpose-elimination paths:**
   - **(a) Transpose-aware sgemm variants.** Add `cpu_matmul_forward_a_bt` (computes `C = A @ B^T` without materialising `B^T`) and `cpu_matmul_forward_at_b` (`C = A^T @ B`). For `grad_a = grad_out @ B^T` the kernel iterates `B` in row-major order naturally:
     ```
     for m in 0..M:
         for k in 0..K:
             acc = 0
             for n in 0..N:
                 acc += grad_out[m, n] * B[k, n]
             grad_a[m, k] = acc
     ```
     Both inner-loop operand reads are sequential — same locality story as codex's `499bfc0` row-major forward. Should land 15–20 GFLOPs/s like the forward.
   - **(b) Loop-reorder `cpu_matmul_backward` to interleave M-outer, write-into-output.** For `grad_b = A^T @ grad_out` (which is `[K, N] = A^T @ grad_out`):
     ```
     for m in 0..M:
         for k in 0..K:
             a_val = A[m, k]
             for n in 0..N:
                 grad_b[k, n] += a_val * grad_out[m, n]
     ```
     Same cache-friendly pattern; no physical transpose; reuses the existing forward primitive shape but flips A's stride interpretation. This is essentially the variant (a) inlined.
4. **Wall-clock per-step framing:**
   - **Pre-`8e8effd`** (naive triple-loop forward): forward 9.86 s, backward likely 2× ≈ 20 s → step ≈ **30 s**.
   - **Post-`499bfc0`** (this entry): forward 0.35 s × 3 = 1.05 s, backward 6.64 s → step ≈ **7.7 s** (4× from pre, but still 22× from 0.35 s ceiling).
   - **Post-transpose-elimination** (projected, backward also ~20 GFLOPs/s): forward 1.05 s, backward ≈ 0.7 s → step ≈ **1.8 s**. That's **another ~4× per step** beyond what `499bfc0` delivered.

## Rule

When a kernel rewrite is licensed on a forward-only benchmark, the **next gate** is a measurement of the backward path at the same production shape. Inheritance through shared primitives is not free — auxiliary host transforms (transpose, layout copies, padding) can silently re-introduce sub-GFLOPs/s scalar work right next to a now-fast kernel, and they bypass any "I improved the matmul" framing. The pattern generalises: any time a backward op is implemented as `host_transform + forward_kernel`, bench the **wrapper**, not the kernel.

## Artefacts

- Bench source: `crates/autograd/examples/cpu_matmul_backward_microbench.rs`
- Raw: `bench-output/2026-05-19-cpu-matmul-backward-row-inner-col-qwen3-06b/cpu_matmul_backward.txt`
- Raw sha256: `d8146ff737bfd2fe34fe75331f740e205a4ec7436412b71eae09d8ff6db2352d`
- Companion entries:
  - [`2026-05-19-bench-cpu-matmul-naive-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-naive-qwen3-06b.md) — pre-fix baseline (naive triple loop)
  - `2026-05-19-bench-cpu-matmul-row-inner-col-qwen3-06b.md` — codex's `499bfc0` forward-rewrite win
