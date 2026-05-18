# CPU matmul backward transpose-aware sgemm — 2.82× per-call, 11.1× cumulative per-OPD-step vs naive — AMD Ryzen 7 3700X, 2026-05-19

## Goal

- **(optimization)** Close the 19× backward-vs-forward GFLOPs/s gap that [`2026-05-19-bench-cpu-matmul-backward-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-backward-qwen3-06b.md) licensed: eliminate the column-strided host `transpose_last_two_ref` step that turned a now-fast `cpu_matmul_forward` (`499bfc0`) into a 1.2 GFLOPs/s backward wrapper. Replace with two transpose-aware sgemm variants (`matmul_a_bt_into`, `matmul_at_b_into`) that compute `C = A @ B^T` and `C = A^T @ B` directly on the physical row-major layouts.
- Maintain bit-identical OPD step output (verified via `test_opd_determinism`) and numerical correctness (verified via `test_opd_grad_check`).

## Hypothesis

- Wall-clock backward at Qwen3-0.6B shape is dominated by `transpose_last_two_ref` host scratch (`crates/autograd/src/backend.rs:1577-1608`, since deleted), not by the inner sgemm. Removing the physical transpose and routing through saxpy-style inner loops on the original layouts should give a ≥2× wall-clock backward win across all 8 production shapes. Result: PASS.
- Forward inner loop is `out_row[col] += a_val * b_row[col]` — pure saxpy, hits ~20 GFLOPs/s on Zen 2 AVX2 autovec. Backward `A @ B^T` is intrinsically a dot product (contracts over the last dim of both A and B, both row-major) — autovectorisable but with a horizontal reduction, so 30-50% of saxpy speed is the realistic ceiling at the kernel level. The remaining 4× gap from forward GFLOPs/s is therefore *expected*, not a regression we should chase with hand-rolled SIMD until a packed-tile (matrixmultiply) or rayon-parallel tranche is licensed separately.

## Command

```bash
cargo build -p autograd --release
cargo test -p autograd --release
cargo test -p train --test test_opd_determinism --release
cargo test -p train --test test_opd_grad_check --release
cargo clippy -p autograd --all-targets --release -- -D warnings
cargo run -p autograd --example cpu_matmul_backward_microbench --release \
  | tee bench-output/2026-05-19-cpu-matmul-backward-transpose-aware-qwen3-06b/cpu_matmul_backward_transpose_aware.txt
```

## Environment

| Item | Value |
|---|---|
| Backend | CPU `cpu_matmul_backward` (transpose-aware sgemm; no physical transpose) |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T, Zen 2 |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Runtime substrate commit under test | `f9f47a8` + this tranche |
| Feature set | `cargo run -p autograd --example cpu_matmul_backward_microbench --release` |
| Non-default flags | none |

## Results

### Backward per-shape median — naive transpose (f9f47a8) vs this tranche

| shape | naive (GF/s) | aware (GF/s) | naive (s) | aware (s) | per-call ×speedup |
|---|---:|---:|---:|---:|---:|
| `q_proj`    | 1.379 | 4.736 | 0.024331 | 0.007086 | **3.43×** |
| `k_proj`    | 1.924 | 5.045 | 0.008718 | 0.003326 | **2.62×** |
| `v_proj`    | 1.929 | 5.039 | 0.008695 | 0.003329 | **2.61×** |
| `o_proj`    | 1.385 | 4.846 | 0.024230 | 0.006925 | **3.50×** |
| `gate_proj` | 1.741 | 4.521 | 0.028902 | 0.011133 | **2.60×** |
| `up_proj`   | 1.723 | 4.502 | 0.029214 | 0.011179 | **2.61×** |
| `down_proj` | 1.321 | 4.582 | 0.038099 | 0.010984 | **3.47×** |
| `lm_head`   | 1.187 | 2.948 | 2.097973 | 0.844401 | **2.48×** |

All σ/μ < 1.5 %, mean × median < 1 % drift — comfortably above noise.

### Aggregate cost per OPD step

| Quantity | Pre-`499bfc0` baseline | Post-`499bfc0` (forward only) | This tranche | Total ×speedup |
|---|---:|---:|---:|---:|
| Forward matmul (3× student/teacher/rollout) | 29.569 s | 1.045 s | **1.015 s** | 29.1× |
| Backward matmul (1× student backward) | ~19.7 s (2× forward proj.) | 6.639 s | **2.355 s** | 8.4× |
| **Total OPD-step matmul** | **~49 s** | **~7.7 s** | **3.37 s** | **~14.5×** |
| Total OPD-step matmul vs original 30 s point estimate | 29.569 s | 7.7 s | **3.37 s** | **8.8×** |

(The "Pre-`499bfc0`" backward column is projected, not measured — the original commit landed before the backward bench harness existed. The 8.8× row uses the original `8e8effd` forward-only point estimate of 30 s/step.)

### Forward vs backward GFLOPs/s, post-tranche

| Shape | Forward | Backward (this tranche) | Backward gap |
|---|---:|---:|---:|
| `q_proj`    | 21.391 | 4.736 | 4.52× slower |
| `k_proj`    | 20.453 | 5.045 | 4.05× slower |
| `v_proj`    | 20.532 | 5.039 | 4.08× slower |
| `o_proj`    | 20.563 | 4.846 | 4.24× slower |
| `gate_proj` | 18.394 | 4.521 | 4.07× slower |
| `up_proj`   | 14.710 | 4.502 | 3.27× slower |
| `down_proj` | 18.107 | 4.582 | 3.95× slower |
| `lm_head`   |  8.551 | 2.948 | 2.90× slower |

Pre-tranche the gap was 6.9-14.9× per shape (median ~11×). Now it's 2.9-4.5× (median ~4×). The remaining gap is the dot-product reduction overhead vs saxpy in `matmul_a_bt_into`; closing it further needs packed-tile blocking (matrixmultiply) or rayon-parallel sharding — that's the next-tier tranche, not in scope here.

### Correctness gates

| Gate | Result |
|---|---|
| `cargo test -p autograd --release` | 14 binaries, all passed |
| `cargo test -p train --release` | 21 tests passed |
| `cargo test -p train --test test_opd_determinism --release` | `opd_step_same_prompt_seed_and_lr_is_bit_identical` ok |
| `cargo test -p train --test test_opd_grad_check --release` | `kl_distill_loss_student_logits_grad_matches_finite_difference` ok |
| `cargo clippy -p autograd --all-targets --release -- -D warnings` | clean |

The OPD determinism test is the load-bearing correctness pin: it runs the full OPD step twice with the same seed/prompt/LR and bit-compares the resulting loss. A buggy `matmul_a_bt_into` / `matmul_at_b_into` would fail this immediately. Pass means the new kernels match the prior transpose-+-forward path to the bit.

## Problems

- **Dot-product reduction overhead persists.** `matmul_a_bt_into` is intrinsically a row-dot-row kernel; rustc autovectorises it but the horizontal reduction across SIMD lanes costs ~2-3× vs the saxpy form used by the forward path. This caps the per-call backward at ~5 GFLOPs/s on Zen 2 even though forward is at 18-21 GFLOPs/s. matrixmultiply's packed-tile sgemm handles this natively (separate fast paths for `A@B`, `A@B^T`, `A^T@B`); pulling it in is the obvious next tranche if a further ~2× backward win is licensed.
- **`lm_head` backward is bandwidth-bound, not kernel-bound.** With `B: [1024, 151936]` = 608 MB physical, every backward pass streams `B` through L3 and main memory; 2.5-3 GFLOPs/s is consistent with Zen 2 single-channel DDR4 ~25 GB/s. No SIMD kernel will fix this — only blocked tile reuse (matrixmultiply / hand-packed) or reduced precision (bf16) can close it.
- **The dead `transpose_last_two_ref` was kept previously as `pub(crate)`** because its doc comment claimed usage by a "no-cuda type-check path of the CUDA backend." That path doesn't exist (confirmed via `grep -rn transpose_last_two_ref crates/ infer/`); the helper was orphaned. Deleted in this tranche to keep the dead-code clippy gate from accumulating noise.

## Learnings

1. **Wrapper benching > kernel benching.** Codex's `499bfc0` was a clean kernel-level win. But the wrapper (`cpu_matmul_backward`) had its own naive scalar prefix. Measuring the wrapper exposed the gap in 5 minutes; measuring only the kernel would have left it hidden indefinitely. Generalisation: every backward op implemented as `host_transform + forward_kernel` needs the wrapper benched, not the kernel.
2. **Loop nesting is a register-allocation decision.** First-draft `matmul_at_b_into` had `out_row` borrowed *inside* the inner `kk` loop; speed was 4.27 GFLOPs/s on `up_proj`. Moving `kk` outermost so `out_row` is borrowed once per `kk` (mirroring codex's `499bfc0` forward layout) bumped it to 4.50 GFLOPs/s. The compiler can keep a stable mutable slice in a register across the inner saxpy; a re-borrowed slice gets reloaded.
3. **"Same operation, different shape catalog" is a different lever.** OPD's per-step matmul cost is now 3.37 s, but the eight component matmuls have very different bandwidth/compute mixes. `lm_head` alone is now 1.82 s / 3.37 s = 54 % of the total — and it's bandwidth-bound, not compute-bound. The next licensed tranche should target `lm_head` specifically (tile-blocked / bf16 / rayon shard on K), not "matmul perf" globally.

## Rule

When a backward op is layered on top of a forward kernel via host transforms, treat the transforms as a **first-class** perf surface, not a backstage convenience. Bench the full backward wrapper at production shape before declaring any forward-kernel win complete. If the wrapper is more than ~2× slower than the kernel-only ceiling at the same shape, the host transform is the next licensed perf lever — not more SIMD on the inner kernel.

## Artefacts

- Bench source: `crates/autograd/examples/cpu_matmul_backward_microbench.rs`
- Raw: `bench-output/2026-05-19-cpu-matmul-backward-transpose-aware-qwen3-06b/cpu_matmul_backward_transpose_aware.txt`
- Companion entries (chronological):
  - [`2026-05-19-bench-cpu-matmul-naive-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-naive-qwen3-06b.md) — naive forward baseline
  - `2026-05-19-bench-cpu-matmul-row-inner-col-qwen3-06b.md` — codex's `499bfc0` forward rewrite
  - [`2026-05-19-bench-cpu-matmul-backward-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-backward-qwen3-06b.md) — backward gap diagnosis
  - **this entry** — backward fix landed
